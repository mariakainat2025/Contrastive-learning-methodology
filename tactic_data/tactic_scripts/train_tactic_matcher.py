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

SEQUENCE_DIR          = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'sequnces')
ABSTRACT_SEQUENCE_DIR = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'abstract_sequnce')
TEMPLATE_DIR     = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'templates')
AUG_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'templates_augmented')
MODEL_DIR     = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'model')
RESULTS_DIR   = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'results')
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TACTIC_LABELS = {
    'Initial_Access'      : 'TA0001',
    'Execution'           : 'TA0002',
    'Persistence'         : 'TA0003',
    'Privilege_Escalation': 'TA0004',
    'Stealth'             : 'TA0005',
    'Defense_Impairment'  : 'TA0112',
    'Credential_Access'   : 'TA0006',
    'Discovery'           : 'TA0007',
    'Lateral_Movement'    : 'TA0008',
    'Collection'          : 'TA0009',
    'Command_and_Control' : 'TA0011',
    'Exfiltration'        : 'TA0010',
    'Impact'              : 'TA0040',
    'Reconnaissance'      : 'TA0043',
    'Resource_Development': 'TA0042',
}

MITRE_TO_LABEL = {v: k for k, v in TACTIC_LABELS.items()}

def tactic_to_label(tactic_str):
    name = tactic_str.split('—')[0].strip()
    name = name.replace('&', 'and').strip()
    return '_'.join(name.split())

SEED       = 42
LR_PROJ    = 1e-3
N_EPOCHS   = 100
PROJ_DIM   = 128
DROPOUT    = 0.3
PATIENCE   = 30
TEMP_INIT  = 0.5
N_UNFREEZE = 4

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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


def info_nce_loss(z_sg, z_tmpl, logit_scale):
    z_sg   = F.normalize(z_sg,   dim=-1)
    z_tmpl = F.normalize(z_tmpl, dim=-1)
    scale  = logit_scale.exp().clamp(max=100)
    S      = scale * (z_sg @ z_tmpl.T)
    labels = torch.arange(len(z_sg), device=z_sg.device)
    return (F.cross_entropy(S, labels) + F.cross_entropy(S.T, labels)) / 2


def load_sequences(sequence_dir, test_file):
    train, test = [], []
    for fname in sorted(os.listdir(sequence_dir)):
        if not fname.endswith('.json') or fname == 'sequences_tactics_all.json':
            continue
        fpath = os.path.join(sequence_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        label = tactic_to_label(data.get('tactic', ''))
        if not label:
            print(f'  [skip] no tactic field in {fname}')
            continue
        text  = ' '.join(data.get('sequence', []))
        entry = {'text': text, 'tactic': label, 'file': fname}
        if fname == test_file:
            test.append(entry)
        else:
            train.append(entry)
    return train, test


def extract_linux_section(full_text):
    marker = 'LINUX DETECTION STRATEGIES'
    idx = full_text.find(marker)
    if idx == -1:
        return full_text.strip()
    after = full_text[idx + len(marker):]
    lines = after.splitlines()
    body = []
    for line in lines:
        if line.strip().startswith('===') or line.strip().startswith('---'):
            if body:
                break
            continue
        body.append(line)
    return '\n'.join(body).strip()


def load_templates(template_dir, linux_only=False, use_aug=False, n_aug=3):
    templates = {}
    for fname in sorted(os.listdir(template_dir)):
        if not fname.endswith('.txt'):
            continue
        label = None
        for lbl, tid in TACTIC_LABELS.items():
            if tid in fname:
                label = lbl
                break
        if label is None:
            continue
        with open(os.path.join(template_dir, fname), encoding='utf-8') as f:
            text = f.read().strip()
        if linux_only:
            text = extract_linux_section(text)
        templates.setdefault(label, []).append({'text': text, 'file': fname})
        print(f'  [template] {label}/{fname}')

        if use_aug:
            tid = TACTIC_LABELS.get(label, '')
            for i in range(1, n_aug + 1):
                aug_txt = os.path.join(AUG_TEMPLATE_DIR, f'{tid}_{fname.split("_", 1)[1].replace(".txt", "")}_aug{i}.txt')
                if os.path.exists(aug_txt):
                    with open(aug_txt, encoding='utf-8') as f:
                        aug_text = f.read().strip()
                    if linux_only:
                        aug_text = extract_linux_section(aug_text)
                    templates[label].append({'text': aug_text, 'file': os.path.basename(aug_txt)})
                    print(f'  [aug]      {label}/{os.path.basename(aug_txt)}')

    return templates


def tokenize(tokenizer, texts):
    all_ids, all_masks = [], []
    for text in texts:
        enc = tokenizer(text, padding=False, truncation=False, return_tensors='pt')
        real_len = int(enc['attention_mask'][0].sum())
        all_ids.append(enc['input_ids'][0][:real_len])
        all_masks.append(enc['attention_mask'][0][:real_len])
    return all_ids, all_masks


def encode_batch(model, tokenizer, ids_list, masks_list, device):
    embeddings = []
    for ids, mask in zip(ids_list, masks_list):
        emb = embed_text(model, tokenizer, ids.unsqueeze(0).to(device),
                         mask.unsqueeze(0).to(device), device)
        embeddings.append(emb.squeeze(0))
    return torch.stack(embeddings)


def build_pairs(train_subgraphs, templates):
    tmpl_flat, tmpl_labels = [], []
    for label, tmpl_list in templates.items():
        for t in tmpl_list:
            tmpl_flat.append(t)
            tmpl_labels.append(label)
    pairs = []
    for sg_idx, sg in enumerate(train_subgraphs):
        for t_idx, t_label in enumerate(tmpl_labels):
            if sg['tactic'] == t_label:
                pairs.append((sg_idx, t_idx))
    return pairs, tmpl_flat, tmpl_labels


def train(device, log_proj, text_proj, sg_embs, tmpl_embs,
          sg_labels, tmpl_labels, test_embs, test_labels, tmpl_flat,
          eval_tmpl_embs=None, eval_tmpl_labels=None, sg_names=None):
    if eval_tmpl_embs is None:
        eval_tmpl_embs, eval_tmpl_labels = tmpl_embs, tmpl_labels

    logit_scale = nn.Parameter(torch.tensor(TEMP_INIT, device=device).log())
    optimizer   = torch.optim.AdamW(
        list(log_proj.parameters()) + list(text_proj.parameters()) + [logit_scale],
        lr=LR_PROJ, weight_decay=1e-4
    )

    best_acc, best_loss, patience_ct, best_state = -1.0, float('inf'), 0, None
    best_matrix_info = None

    print(f'\n  Training {len(sg_labels)} sequences for up to {N_EPOCHS} epochs...')
    print(f'  Train templates: {len(tmpl_labels)}  |  Eval templates (originals): {len(eval_tmpl_labels)}')
    sep = '  ' + '─' * 90
    print(sep)

    def print_score_matrix(epoch_tag, z_sg_all, z_tmpl_all):
        print(f'\n  ── Score Matrix at {epoch_tag} (cosine similarity, train subgraphs × train templates) ──')
        unique_tmpls = list(dict.fromkeys(tmpl_labels))
        hdr = f"  {'Subgraph':<42}"
        for t in unique_tmpls:
            hdr += f" {t[:6]:>7}"
        print(hdr)
        print("  " + "-" * (42 + 8 * len(unique_tmpls)))
        for i, sl in enumerate(sg_labels):
            row = f"  {train_seqs_names[i][:41]:<42}" if hasattr(train_seqs_names, '__len__') else f"  sg{i:<39}"
            for t in unique_tmpls:
                j = tmpl_labels.index(t)
                s = (z_sg_all[i] @ z_tmpl_all[j]).item()
                mark = '*' if sl == t else ' '
                row += f" {s:+.3f}{mark}"
            row += f"  [{sl}]"
            print(row)
        print()

    epoch_log = []
    train_seqs_names = sg_names if sg_names else sg_labels

    tactic_tmpl_map = {}
    for t_idx, t_label in enumerate(tmpl_labels):
        tactic_tmpl_map.setdefault(t_label, []).append(t_idx)

    for epoch in range(1, N_EPOCHS + 1):
        log_proj.train()
        text_proj.train()

        z_logs  = torch.stack([log_proj(sg_embs[i].to(device))    for i in range(len(sg_labels))])
        z_tmpls = torch.stack([text_proj(tmpl_embs[j].to(device)) for j in range(len(tmpl_labels))])

        sg_z, tmpl_z = [], []
        for i, sl in enumerate(sg_labels):
            t_idx = random.choice(tactic_tmpl_map[sl])
            sg_z.append(z_logs[i])
            tmpl_z.append(z_tmpls[t_idx])
        loss = info_nce_loss(torch.stack(sg_z), torch.stack(tmpl_z), logit_scale)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(log_proj.parameters()) + list(text_proj.parameters()), 1.0)
        optimizer.step()

        if epoch == 1:
            log_proj.eval(); text_proj.eval()
            with torch.no_grad():
                zs = F.normalize(torch.stack([log_proj(sg_embs[i].to(device))    for i in range(len(sg_labels))]), dim=-1)
                zt = F.normalize(torch.stack([text_proj(tmpl_embs[j].to(device)) for j in range(len(tmpl_labels))]), dim=-1)
            print_score_matrix('Epoch 1', zs, zt)
            log_proj.train(); text_proj.train()

        if epoch % 10 == 0 or epoch == 1:
            log_proj.eval()
            text_proj.eval()
            correct, sg_results = 0, []
            with torch.no_grad():
                for i, true_label in enumerate(test_labels):
                    z_log  = F.normalize(log_proj(test_embs[i].to(device)), dim=-1)
                    scores = {}
                    for j, t_label in enumerate(eval_tmpl_labels):
                        z_t = F.normalize(text_proj(eval_tmpl_embs[j].to(device)), dim=-1)
                        s = (z_log @ z_t).item()
                        scores[t_label] = max(scores.get(t_label, float('-inf')), s)
                    pred = max(scores, key=scores.get)
                    correct += int(pred == true_label)
                    sg_results.append({'true': true_label, 'pred': pred, 'scores': scores})

            acc = correct / len(test_labels)
            for idx, r in enumerate(sg_results):
                mark    = '✓' if r['pred'] == r['true'] else '✗'
                true_sc = r['scores'].get(r['true'], 0)
                ranked  = sorted(r['scores'].items(), key=lambda x: x[1], reverse=True)
                top1_l, top1_s = ranked[0]
                top2_l, top2_s = ranked[1] if len(ranked) > 1 else ('—', 0.0)
                acc_col = f'  Acc={acc:.2f}' if idx == len(sg_results) - 1 else ''
                print(f'  Ep={epoch:<4} Loss={loss.item():.4f}  {mark}  '
                      f'True={r["true"]}({true_sc:+.4f})  '
                      f'Top1={top1_l}({top1_s:+.4f})  '
                      f'Top2={top2_l}({top2_s:+.4f}){acc_col}')
            print(sep)

            epoch_log.append({'epoch': epoch, 'loss': loss.item(), 'acc': acc})

            if acc > best_acc or (acc == best_acc and loss.item() < best_loss):
                best_acc, best_loss, patience_ct = acc, loss.item(), 0
                best_state = {
                    'log_proj'   : {k: v.clone() for k, v in log_proj.state_dict().items()},
                    'text_proj'  : {k: v.clone() for k, v in text_proj.state_dict().items()},
                    'logit_scale': logit_scale.data.clone(),
                    'epoch'      : epoch,
                }
                with torch.no_grad():
                    zs = F.normalize(torch.stack([log_proj(sg_embs[i].to(device))    for i in range(len(sg_labels))]), dim=-1)
                    zt = F.normalize(torch.stack([text_proj(tmpl_embs[j].to(device)) for j in range(len(tmpl_labels))]), dim=-1)
                best_matrix_info = (epoch, zs.cpu(), zt.cpu())
            else:
                patience_ct += 1
                if patience_ct >= PATIENCE:
                    print(f'  Early stopping at epoch {epoch}  best_acc={best_acc:.2f}')
                    break

    if best_matrix_info is not None:
        best_ep, zs, zt = best_matrix_info
        log_proj.eval(); text_proj.eval()
        print_score_matrix(f'Best Epoch ({best_ep})', zs.to(device), zt.to(device))

    return best_state, best_loss, best_acc, epoch_log


def evaluate(device, log_proj, text_proj, test_embs, eval_tmpl_embs,
             test_subgraphs, eval_tmpl_flat, eval_tmpl_labels):
    log_proj.eval()
    text_proj.eval()
    results = []

    print('\n  ── Test Results ──────────────────────────────────────────')
    print(f'  {"File":<50} {"True":<22} {"True Score":>10}  {"Predicted":<22} {"Pred Score":>10}  Match')
    print('  ' + '-' * 120)

    with torch.no_grad():
        for i, sg in enumerate(test_subgraphs):
            z_log  = F.normalize(log_proj(test_embs[i].to(device)), dim=-1)
            scores = {}
            for j, t_label in enumerate(eval_tmpl_labels):
                z_t = F.normalize(text_proj(eval_tmpl_embs[j].to(device)), dim=-1)
                s = (z_log @ z_t).item()
                scores[t_label] = max(scores.get(t_label, float('-inf')), s)
            ranked     = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            pred_label = ranked[0][0]
            true_label = sg['tactic']
            true_score = scores.get(true_label, 0)
            pred_score = scores[pred_label]
            correct    = pred_label == true_label
            print(f'  {sg["file"]:<50} {true_label:<22} {true_score:>+10.4f}  {pred_label:<22} {pred_score:>+10.4f}  {"✓" if correct else "✗"}')
            print(f'    All scores: ' + '  '.join(f'{l}={s:+.4f}' for l, s in ranked))
            results.append({'file': sg['file'], 'true_tactic': true_label,
                            'pred_tactic': pred_label, 'correct': correct, 'scores': scores})

    n_correct = sum(r['correct'] for r in results)
    print(f'\n  Accuracy: {n_correct}/{len(results)}')
    return results


def run(test_file, abstract=False, linux_only=False, use_aug=False):
    seq_dir = ABSTRACT_SEQUENCE_DIR if abstract else SEQUENCE_DIR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*72}')
    print(f'  TACTIC MATCHER TRAINING')
    print(f'{"="*72}')
    print(f'  Device   : {device}')
    tmpl_mode = "linux-only" if linux_only else "full"
    tmpl_mode += " + aug(x3)" if use_aug else ""
    print(f'  Mode     : {"abstract" if abstract else "raw"} | templates: {tmpl_mode}')
    print(f'  Sequences: {seq_dir}')
    print(f'  Templates: {TEMPLATE_DIR}')

    print('\n  Stage 1 — Loading sequences...')
    train_seqs, test_seqs = load_sequences(seq_dir, test_file)
    print(f'  Train: {len(train_seqs)}  Test: {len(test_seqs)}')
    for s in train_seqs:
        print(f'    [train] {s["tactic"]} / {s["file"]}')
    for s in test_seqs:
        print(f'    [test]  {s["tactic"]} / {s["file"]}')

    if not test_seqs:
        print('\n  ERROR: no test file found.')
        return

    print('\n  Stage 2 — Loading templates...')
    templates = load_templates(TEMPLATE_DIR, linux_only=linux_only, use_aug=use_aug)
    print(f'  Train templates: {sum(len(v) for v in templates.values())} across {len(templates)} tactics')
    orig_templates = load_templates(TEMPLATE_DIR, linux_only=linux_only, use_aug=False)
    print(f'  Eval  templates: {sum(len(v) for v in orig_templates.values())} (originals only)')

    print('\n  Stage 3 — Tokenizing...')
    tokenizer  = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    pairs, tmpl_flat, tmpl_labels = build_pairs(train_seqs, templates)
    _, orig_tmpl_flat, orig_tmpl_labels = build_pairs(train_seqs, orig_templates)

    train_ids,     train_masks     = tokenize(tokenizer, [s['text'] for s in train_seqs])
    test_ids,      test_masks      = tokenize(tokenizer, [s['text'] for s in test_seqs])
    tmpl_ids,      tmpl_masks      = tokenize(tokenizer, [t['text'] for t in tmpl_flat])
    orig_tmpl_ids, orig_tmpl_masks = tokenize(tokenizer, [t['text'] for t in orig_tmpl_flat])
    print(f'  Train: {len(train_ids)}  Test: {len(test_ids)}  Train-tmpls: {len(tmpl_ids)}  Eval-tmpls: {len(orig_tmpl_ids)}')

    print('\n  Stage 4 — Encoding with RoBERTa...')
    roberta = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    for param in roberta.parameters():
        param.requires_grad = False
    n_layers = len(roberta.encoder.layer)
    for i in range(n_layers - N_UNFREEZE, n_layers):
        for param in roberta.encoder.layer[i].parameters():
            param.requires_grad = True
    for param in roberta.pooler.parameters():
        param.requires_grad = True

    roberta.eval()
    with torch.no_grad():
        train_embs     = encode_batch(roberta, tokenizer, train_ids,     train_masks,     device).cpu()
        test_embs      = encode_batch(roberta, tokenizer, test_ids,      test_masks,      device).cpu()
        tmpl_embs      = encode_batch(roberta, tokenizer, tmpl_ids,      tmpl_masks,      device).cpu()
        orig_tmpl_embs = encode_batch(roberta, tokenizer, orig_tmpl_ids, orig_tmpl_masks, device).cpu()
    print(f'  Train embs: {train_embs.shape}  Test embs: {test_embs.shape}  Tmpl embs: {tmpl_embs.shape}  Orig-tmpl embs: {orig_tmpl_embs.shape}')

    print(f'\n  Stage 5 — {len(pairs)} training pairs:')
    for sg_idx, t_idx in pairs:
        print(f'    {train_seqs[sg_idx]["tactic"]}/{train_seqs[sg_idx]["file"]}  ↔  {tmpl_labels[t_idx]}/{tmpl_flat[t_idx]["file"]}')

    print('\n  Stage 6 — Training...')
    log_proj  = ProjectionHead().to(device)
    text_proj = ProjectionHead().to(device)
    sg_labels   = [s['tactic'] for s in train_seqs]
    sg_names    = [s['file'].replace('abstract_','').replace('.json','').replace('_sequence','') for s in train_seqs]
    test_labels = [s['tactic'] for s in test_seqs]
    best_state, best_loss, best_acc, epoch_log = train(
        device, log_proj, text_proj,
        train_embs, tmpl_embs, sg_labels, tmpl_labels,
        test_embs, test_labels, tmpl_flat,
        eval_tmpl_embs=orig_tmpl_embs, eval_tmpl_labels=orig_tmpl_labels,
        sg_names=sg_names
    )

    log_proj.load_state_dict(best_state['log_proj'])
    text_proj.load_state_dict(best_state['text_proj'])

    ckpt_path = os.path.join(MODEL_DIR, 'tactic_matcher.pt')
    torch.save({'log_proj': best_state['log_proj'], 'text_proj': best_state['text_proj'],
                'logit_scale': best_state['logit_scale'], 'best_loss': best_loss}, ckpt_path)
    print(f'  Model saved → {ckpt_path}  (best_acc={best_acc:.2f}  epoch={best_state["epoch"]})')

    print('\n  Stage 7 — Evaluating...')
    results = evaluate(device, log_proj, text_proj,
                       test_embs, orig_tmpl_embs, test_seqs, orig_tmpl_flat, orig_tmpl_labels)

    results_path = os.path.join(RESULTS_DIR, 'tactic_matcher_results.json')
    with open(results_path, 'w') as f:
        json.dump({'epoch_log': epoch_log, 'results': results}, f, indent=2)
    print(f'  Results saved → {results_path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('test', help='filename of the sequence to use as test')
    ap.add_argument('--abstract',    action='store_true')
    ap.add_argument('--linux-only',  action='store_true')
    ap.add_argument('--aug',         action='store_true')
    args = ap.parse_args()
    run(test_file=args.test, abstract=args.abstract, linux_only=args.linux_only, use_aug=args.aug)
