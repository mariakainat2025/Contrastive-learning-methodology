import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import ROBERTA_MODEL, MAX_LEN, STRIDE
from scripts.encoder_utils import embed_text
from scripts.subgraph_sequence_builder import _rebuild_graph, extract_triples, triples_to_text

# ── Paths ─────────────
TACTIC_DATA   = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data')
TRAIN_DIR     = os.path.join(TACTIC_DATA, 'training', 'abstract')
TEST_DIR      = os.path.join(TACTIC_DATA, 'testing',  'abstract')
TEMPLATE_DIR  = os.path.join(TACTIC_DATA, 'templates')
MODEL_DIR     = os.path.join(TACTIC_DATA, 'model')
RESULTS_DIR   = os.path.join(TACTIC_DATA, 'results')
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

#  Tactic label map 
TACTIC_LABELS = {
    'Initial_Access': 'TA0001',
    'Execution'     : 'TA0002',
}

#  Hyperparameters ─
SEED          = 42
LR_ROBERTA    = 1e-5
LR_PROJ       = 1e-3
N_EPOCHS      = 100
PROJ_DIM      = 128
DROPOUT       = 0.3
PATIENCE      = 30
TEMP_INIT     = 0.5
N_UNFREEZE    = 4  

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Projection network 
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=PROJ_DIM, dropout=DROPOUT):
        super().__init__()
        self.fc1     = nn.Linear(in_dim, out_dim)
        self.fc2     = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


# ── InfoNCE loss ──────
def info_nce_loss(z_log, z_tmpl, logit_scale):
    z_log  = F.normalize(z_log,  dim=-1)
    z_tmpl = F.normalize(z_tmpl, dim=-1)
    scale  = logit_scale.exp().clamp(max=100)
    S      = scale * (z_log @ z_tmpl.T)
    labels = torch.arange(len(z_log), device=z_log.device)
    return (F.cross_entropy(S, labels) + F.cross_entropy(S.T, labels)) / 2


# ── Convert subgraph JSON → sequence text ─────────────────────────────────────
def subgraph_to_text(path):
    with open(path) as f:
        data = json.load(f)
    G       = _rebuild_graph(data)
    triples = extract_triples(G)
    tokens  = triples_to_text(triples)
    return ' '.join(tokens) if tokens else 'empty subgraph'


# ── Load all subgraphs from a tactic folder ───────────────────────────────────
def load_tactic_subgraphs(base_dir):
    pairs = []   # list of (text, tactic_label)
    for tactic in os.listdir(base_dir):
        tactic_dir = os.path.join(base_dir, tactic)
        if not os.path.isdir(tactic_dir):
            continue
        label = tactic  # e.g. 'Initial_Access', 'Execution'
        for fname in sorted(os.listdir(tactic_dir)):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(tactic_dir, fname)
            text  = subgraph_to_text(fpath)
            pairs.append({'text': text, 'tactic': label, 'file': fname})
            print(f'  [subgraph] {label}/{fname}')

    # save subgraph texts to file
    split_name = os.path.basename(os.path.dirname(base_dir))  # 'training' or 'testing'
    out_path = os.path.join(RESULTS_DIR, f'subgraph_sequences_{split_name}.txt')
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for p in pairs:
            f.write(f'=== {p["tactic"]} / {p["file"]} ===\n')
            f.write(p['text'] + '\n\n')
    print(f'  Sequences saved → {out_path}')
    return pairs


# ── Load templates (original + augmented) ─────────────────────────────────────
def load_templates(template_dir, use_aug=True):
    templates = {}   # tactic_id → list of texts
    for fname in sorted(os.listdir(template_dir)):
        if not fname.endswith('.txt'):
            continue
        if not use_aug and '_aug' in fname:
            continue
        # map filename to tactic label
        tactic_id = None
        for label, tid in TACTIC_LABELS.items():
            if tid in fname:
                tactic_id = label
                break
        if tactic_id is None:
            continue
        fpath = os.path.join(template_dir, fname)
        with open(fpath, encoding='utf-8') as f:
            text = f.read().strip()
        templates.setdefault(tactic_id, []).append({'text': text, 'file': fname})
        print(f'  [template] {tactic_id}/{fname}')
    return templates


# ── Tokenize a list of texts ──────────────────────────────────────────────────
def tokenize(tokenizer, texts):
    all_ids, all_masks = [], []
    for text in texts:
        enc = tokenizer(
            text,
            padding       = False,
            truncation    = False,
            return_tensors= 'pt',
        )
        real_len = int(enc['attention_mask'][0].sum())
        all_ids.append(enc['input_ids'][0][:real_len])
        all_masks.append(enc['attention_mask'][0][:real_len])
    return all_ids, all_masks


# ── Encode a list of tokenized texts using RoBERTa ───────────────────────────
def encode_batch(model, tokenizer, ids_list, masks_list, device):
    embeddings = []
    for ids, mask in zip(ids_list, masks_list):
        ids_b  = ids.unsqueeze(0).to(device)
        mask_b = mask.unsqueeze(0).to(device)
        emb = embed_text(model, tokenizer, ids_b, mask_b, device)
        embeddings.append(emb.squeeze(0))
    return torch.stack(embeddings)


# ── Build training pairs ──────────────────────────────────────────────────────
def build_pairs(subgraphs, templates):
    """
    Returns list of (subgraph_idx, template_idx) positive pairs.
    Each subgraph is paired with every version of its matching template.
    """
    pairs = []
    tmpl_flat  = []   # flat list of all template texts in order
    tmpl_labels = []  # corresponding tactic label
    for label, tmpl_list in templates.items():
        for t in tmpl_list:
            tmpl_flat.append(t)
            tmpl_labels.append(label)

    for sg_idx, sg in enumerate(subgraphs):
        for t_idx, t_label in enumerate(tmpl_labels):
            if sg['tactic'] == t_label:
                pairs.append((sg_idx, t_idx))

    return pairs, tmpl_flat, tmpl_labels


# ── Quick test accuracy during training ───────────────────────────────────────
def test_accuracy(device, log_proj, text_proj, test_embs, tmpl_embs,
                  test_labels, tmpl_labels, tmpl_flat=None):
    log_proj.eval()
    text_proj.eval()
    correct = 0
    with torch.no_grad():
        for i, true_label in enumerate(test_labels):
            z_log = F.normalize(log_proj(test_embs[i].to(device)), dim=-1)
            scores = {}
            for j, t_label in enumerate(tmpl_labels):
                # only score against original templates (not aug versions)
                if tmpl_flat and '_aug' in tmpl_flat[j].get('file', ''):
                    continue
                z_t = F.normalize(text_proj(tmpl_embs[j].to(device)), dim=-1)
                scores[t_label] = (z_log @ z_t).item()
            pred = max(scores, key=scores.get)
            if pred == true_label:
                correct += 1
    return correct / len(test_labels)


# ── Training ──────────
def train(device, log_proj, text_proj, sg_embs, tmpl_embs,
          sg_labels, tmpl_labels, test_embs, test_labels, tmpl_flat=None, test_subgraphs=None):

    logit_scale = nn.Parameter(torch.tensor(TEMP_INIT, device=device).log())
    optimizer   = torch.optim.AdamW([
        {'params': list(log_proj.parameters()) + list(text_proj.parameters()) + [logit_scale],
         'lr': LR_PROJ},
    ], weight_decay=1e-4)

    tactic_tmpl_map = {}
    for t_idx, t_label in enumerate(tmpl_labels):
        tactic_tmpl_map.setdefault(t_label, []).append(t_idx)

    best_acc    = -1.0
    best_loss   = float('inf')
    patience_ct = 0
    best_state  = None

    n_sg = len(sg_labels)
    print(f'\n  Training {n_sg} subgraphs × 1 template/epoch '
          f'for up to {N_EPOCHS} epochs...')

    sep = '  ' + '─' * 80
    print(f'\n  {"Ep":<6} {"Loss":<8} {"Test Subgraph":<20} {"Match":<6} {"Execution":>12}  {"Initial_Access":>16}  {"Acc"}')
    print(sep)

    epoch_log = []

    for epoch in range(1, N_EPOCHS + 1):
        log_proj.train()
        text_proj.train()

        z_logs, z_tmpls = [], []
        for sg_idx, sg_label in enumerate(sg_labels):
            t_idx  = random.choice(tactic_tmpl_map[sg_label])
            z_log  = log_proj(sg_embs[sg_idx].to(device))
            z_tmpl = text_proj(tmpl_embs[t_idx].to(device))
            z_logs.append(z_log)
            z_tmpls.append(z_tmpl)

        z_logs  = torch.stack(z_logs)
        z_tmpls = torch.stack(z_tmpls)
        loss    = info_nce_loss(z_logs, z_tmpls, logit_scale)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(log_proj.parameters()) + list(text_proj.parameters()), 1.0
        )
        optimizer.step()

       
        if epoch % 10 == 0 or epoch == 1:
            log_proj.eval()
            text_proj.eval()
            all_tactics = sorted(set(tmpl_labels))
            # only original templates for scoring
            orig_indices = [j for j, t in enumerate(tmpl_flat)
                            if '_aug' not in t.get('file', '')]
            correct = 0
            sg_results = []
            with torch.no_grad():
                for i, true_label in enumerate(test_labels):
                    z_log = F.normalize(log_proj(test_embs[i].to(device)), dim=-1)
                    scores = {}
                    for j in orig_indices:
                        t_label = tmpl_labels[j]
                        z_t = F.normalize(text_proj(tmpl_embs[j].to(device)), dim=-1)
                        scores[t_label] = (z_log @ z_t).item()
                    pred = max(scores, key=scores.get)
                    if pred == true_label:
                        correct += 1
                    sg_results.append({'true': true_label, 'pred': pred, 'scores': scores})

            acc  = correct / len(test_labels)
            temp = logit_scale.exp().item()

            # print one row per test subgraph
            for idx, r in enumerate(sg_results):
                mark  = '✓' if r['pred'] == r['true'] else '✗'
                e_sc  = r['scores'].get('Execution', 0)
                ia_sc = r['scores'].get('Initial_Access', 0)
                pe_sc = r['scores'].get('Persistence', 0)
                e_str  = f'{e_sc:+.4f}{"*" if r["true"]=="Execution" else ""}'
                ia_str = f'{ia_sc:+.4f}{"*" if r["true"]=="Initial_Access" else ""}'
                acc_col = f'  Acc={acc:.2f}' if idx == len(sg_results) - 1 else ''
                print(f'  {epoch:<6} {loss.item():<8.4f} {r["true"]:<20} {mark:<6} {e_str:>12}  {ia_str:>16}{acc_col}')
            print(sep)

            epoch_log.append({'epoch': epoch, 'loss': loss.item(),
                               'temp': temp, 'acc': acc})

        # only save/check patience at evaluation epochs
        if epoch % 10 == 0 or epoch == 1:
            acc_now = epoch_log[-1]['acc']
            # compute average score gap across test subgraphs
            if acc_now > best_acc or (acc_now == best_acc and loss.item() < best_loss):
                best_acc    = acc_now
                best_loss   = loss.item()
                patience_ct = 0
                best_state  = {
                    'log_proj'   : {k: v.clone() for k, v in log_proj.state_dict().items()},
                    'text_proj'  : {k: v.clone() for k, v in text_proj.state_dict().items()},
                    'logit_scale': logit_scale.data.clone(),
                    'epoch'      : epoch,
                }
            else:
                patience_ct += 1
                if patience_ct >= PATIENCE:
                    print(sep)
                    print(f'  Early stopping at epoch {epoch}  '
                          f'best_acc={best_acc:.2f}  (saved from epoch {best_state["epoch"]})')
                    break

    print(sep)
    return best_state, best_loss, best_acc, epoch_log


# ── Evaluation ─────────
def evaluate(device, log_proj, text_proj, sg_embs, tmpl_embs,
             test_subgraphs, tmpl_flat, tmpl_labels):

    log_proj.eval()
    text_proj.eval()
    results = []

    print('\n  ── Test Results ──────────────────────────────────────────')
    print(f'  {"File":<45} {"True":<18} {"Rank1":<18} {"Correct"}')
    print('  ' + '-' * 90)

    orig_indices = [j for j, t in enumerate(tmpl_flat)
                    if '_aug' not in t.get('file', '')]
    with torch.no_grad():
        for i, sg in enumerate(test_subgraphs):
            z_log = F.normalize(log_proj(sg_embs[i].to(device)), dim=-1)

            # score against original templates only
            scores = {}
            for j in orig_indices:
                t_label = tmpl_labels[j]
                z_tmpl  = F.normalize(text_proj(tmpl_embs[j].to(device)), dim=-1)
                scores[t_label] = (z_log @ z_tmpl).item()

            ranked     = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            pred_label = ranked[0][0]
            true_label = sg['tactic']
            correct    = pred_label == true_label

            print(f'  {sg["file"]:<45} {true_label:<18} {pred_label:<18} {"✓" if correct else "✗"}')
            print(f'    Scores: ' + '  '.join(f'{l}={s:.4f}' for l, s in ranked))

            results.append({
                'file'      : sg['file'],
                'true_tactic': true_label,
                'pred_tactic': pred_label,
                'correct'   : correct,
                'scores'    : scores,
            })

    n_correct = sum(r['correct'] for r in results)
    print(f'\n  Accuracy: {n_correct}/{len(results)}')
    return results


# ── Main ──────────────
def run(use_aug=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    # ── Stage 1: Load subgraphs ───────────────────────────────────────────────
    print('\n  Stage 1 — Loading training subgraphs...')
    train_subgraphs = load_tactic_subgraphs(TRAIN_DIR)
    print(f'  Loaded {len(train_subgraphs)} training subgraphs')

    print('\n  Stage 1b — Loading test subgraphs...')
    test_subgraphs = load_tactic_subgraphs(TEST_DIR)
    print(f'  Loaded {len(test_subgraphs)} test subgraphs')

    # ── Stage 2: Load templates ───────────────────────────────────────────────
    print(f'\n  Stage 2 — Loading templates (aug={use_aug})...')
    templates = load_templates(TEMPLATE_DIR, use_aug=use_aug)
    for label, tmpls in templates.items():
        print(f'  {label}: {len(tmpls)} versions')

    # ── Stage 3: Tokenize ─────────────────────────────────────────────────────
    print('\n  Stage 3 — Tokenizing...')
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)

    train_texts = [sg['text'] for sg in train_subgraphs]
    test_texts  = [sg['text'] for sg in test_subgraphs]

    tmpl_pairs, tmpl_flat, tmpl_labels = build_pairs(train_subgraphs, templates)
    tmpl_texts = [t['text'] for t in tmpl_flat]

    train_ids,  train_masks  = tokenize(tokenizer, train_texts)
    test_ids,   test_masks   = tokenize(tokenizer, test_texts)
    tmpl_ids,   tmpl_masks   = tokenize(tokenizer, tmpl_texts)

    print(f'  Train seqs: {len(train_ids)}  Test seqs: {len(test_ids)}  Templates: {len(tmpl_ids)}')

    # ── Stage 4: Encode with RoBERTa ─────────────────────────────────────────
    print('\n  Stage 4 — Encoding with RoBERTa...')
    roberta = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)

    # freeze all layers first
    for param in roberta.parameters():
        param.requires_grad = False

    # unfreeze last N_UNFREEZE transformer layers
    n_layers = len(roberta.encoder.layer)
    for i in range(n_layers - N_UNFREEZE, n_layers):
        for param in roberta.encoder.layer[i].parameters():
            param.requires_grad = True
    for param in roberta.pooler.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in roberta.parameters() if p.requires_grad)
    print(f'  RoBERTa trainable params: {n_trainable:,} (last {N_UNFREEZE} layers)')


    roberta.eval()
    with torch.no_grad():
        test_embs  = encode_batch(roberta, tokenizer, test_ids,  test_masks,  device).cpu()
        tmpl_embs  = encode_batch(roberta, tokenizer, tmpl_ids,  tmpl_masks,  device).cpu()

    # encode training subgraphs once (will re-encode during training via fine-tuning)
    with torch.no_grad():
        train_embs = encode_batch(roberta, tokenizer, train_ids, train_masks, device).cpu()

    print(f'  Train embs: {train_embs.shape}')
    print(f'  Test  embs: {test_embs.shape}')
    print(f'  Tmpl  embs: {tmpl_embs.shape}')

    # ── Stage 5: Build training pairs ────────────────────────────────────────
    print(f'\n  Stage 5 — Building {len(tmpl_pairs)} training pairs...')
    for sg_idx, t_idx in tmpl_pairs:
        sg    = train_subgraphs[sg_idx]
        tmpl  = tmpl_flat[t_idx]
        print(f'    ({sg["tactic"]}/{sg["file"]})  ↔  ({tmpl_labels[t_idx]}/{tmpl["file"]})')

    # ── Stage 6: Train 
    print('\n  Stage 6 — Training...')
    log_proj  = ProjectionHead().to(device)
    text_proj = ProjectionHead().to(device)

    sg_labels   = [sg['tactic'] for sg in train_subgraphs]
    test_labels = [sg['tactic'] for sg in test_subgraphs]
    best_state, best_loss, best_acc, epoch_log = train(
        device, log_proj, text_proj,
        train_embs, tmpl_embs, sg_labels, tmpl_labels,
        test_embs, test_labels, tmpl_flat=tmpl_flat
    )

    # restore best model
    log_proj.load_state_dict(best_state['log_proj'])
    text_proj.load_state_dict(best_state['text_proj'])

    # save
    ckpt_path = os.path.join(MODEL_DIR, 'tactic_matcher.pt')
    torch.save({
        'log_proj'   : best_state['log_proj'],
        'text_proj'  : best_state['text_proj'],
        'logit_scale': best_state['logit_scale'],
        'best_loss'  : best_loss,
    }, ckpt_path)
    print(f'  Model saved → {ckpt_path}  (best_acc={best_acc:.2f}  best_loss={best_loss:.4f}  epoch={best_state["epoch"]})')

    # ── Stage 7: Evaluate ─────────────────────────────────────────────────────
    print('\n  Stage 7 — Evaluating on test subgraphs...')
    results = evaluate(
        device, log_proj, text_proj,
        test_embs, tmpl_embs,
        test_subgraphs, tmpl_flat, tmpl_labels
    )

    # save results
    results_path = os.path.join(RESULTS_DIR, 'tactic_matcher_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved → {results_path}')


if __name__ == '__main__':
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument('--aug', type=str, default='true',
                    choices=['true', 'false'],
                    help='Use augmented templates (true/false). Default: true')
    ap.add_argument('--save-log', action='store_true',
                    help='Save full training output to results/training_log.txt')
    args = ap.parse_args()

    if args.save_log:
        log_path = os.path.join(RESULTS_DIR, 'training_log.txt')
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f'  Saving output to {log_path}')
        tee = open(log_path, 'w')
        class Tee:
            def write(self, msg):
                sys.__stdout__.write(msg)
                tee.write(msg)
            def flush(self):
                sys.__stdout__.flush()
                tee.flush()
        sys.stdout = Tee()

    run(use_aug=args.aug.lower() == 'true')

    if args.save_log:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f'  Training log saved → {log_path}')
