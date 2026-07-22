import os
import re
import sys
import json
import glob
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

PROJECT_ROOT = '/csse/research/contructive-learning'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import ROBERTA_MODEL
from scripts.encoder_utils import embed_text
from test_split_presets import resolve_test_file_arg

CAM_LDS_DIR   = '/csse/research/contructive-learning/CAM-LDS'
SEQUENCES_DIR = os.path.join(CAM_LDS_DIR, 'sequences')
TEMPLATE_DIR  = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'templates_dc')
OUTPUT_TRAINING = os.path.join(CAM_LDS_DIR, 'checkpoints')
RESULTS_DIR     = os.path.join(CAM_LDS_DIR, 'results')
os.makedirs(OUTPUT_TRAINING, exist_ok=True)
os.makedirs(RESULTS_DIR,     exist_ok=True)

TACTIC_FOLDER_TO_LABEL = {
    'command_and_control': 'Command_and_Control',
    'initial_access':      'Initial_Access',
}


TACTIC_IDS = {
    'Command_and_Control': 'TA0011',
    'Initial_Access':      'TA0001',
}
ALL_TACTICS = sorted(TACTIC_FOLDER_TO_LABEL.values())

SEED       = 42
LR         = 1e-5
LR_PROJ    = 1e-3
DROPOUT    = 0.5
N_EPOCHS   = 100
EMB_DIM    = 768
PROJ_DIM   = 128
PATIENCE   = 20
N_FREEZE   = 9
K_PER_TACTIC = 1


random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


class ProjectionNetwork(nn.Module):
    def __init__(self, in_dim=EMB_DIM, out_dim=PROJ_DIM, dropout=DROPOUT):
        super().__init__()
        self.fc1     = nn.Linear(in_dim, out_dim)
        self.fc2     = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        return h1 + h2


def two_tactic_directional_loss(z_pos_anchors, u_pos_tmpl, u_neg_tmpl, z_neg_all, logit_scale):
    candidates = torch.cat([u_pos_tmpl.unsqueeze(0), u_neg_tmpl.unsqueeze(0), z_neg_all], dim=0)
    candidates = F.normalize(candidates, dim=-1)
    z_pos_anchors = F.normalize(z_pos_anchors, dim=-1)
    scale  = logit_scale.exp().clamp(max=100)
    logits = scale * (z_pos_anchors @ candidates.T)
    targets = torch.zeros(z_pos_anchors.size(0), dtype=torch.long, device=z_pos_anchors.device)
    return F.cross_entropy(logits, targets)


def freeze_lower_layers(model, n_freeze=N_FREEZE):
    for param in model.embeddings.parameters():
        param.requires_grad = False
    for i in range(n_freeze):
        for param in model.encoder.layer[i].parameters():
            param.requires_grad = False


def load_sequences():
    by_tactic = {}
    for folder, label in TACTIC_FOLDER_TO_LABEL.items():
        entries = []
        for fpath in sorted(glob.glob(os.path.join(SEQUENCES_DIR, folder, '*', 'sequence_*.json'))):
            with open(fpath) as f:
                seq = json.load(f)
            entries.append({
                'text'    : ' '.join(seq['sequence']),
                'tactic'  : label,
                'file'    : f"{seq['technique']}/{os.path.basename(fpath)}",
                'basename': os.path.basename(fpath),
            })
        by_tactic[label] = entries

    basenames_by_tactic = {t: set(e['basename'] for e in es) for t, es in by_tactic.items()}
    shared = set.intersection(*basenames_by_tactic.values()) if len(basenames_by_tactic) > 1 else set()

    entries = []
    n_dropped = 0
    for label, es in by_tactic.items():
        for e in es:
            if e['basename'] in shared:
                n_dropped += 1
                continue
            entries.append(e)

    print('  Cross-tactic files dropped (tagged under both tactics): {} ({} entries removed)'.format(
        len(shared), n_dropped))
    for b in sorted(shared):
        print('    - {}'.format(b))

    return entries


def leave_out_split(entries, match_substrs, seed=SEED):
    if isinstance(match_substrs, str):
        match_substrs = [match_substrs]
    rng = random.Random(seed)
    train, test = [], []
    for e in entries:
        (test if any(m in e['file'] for m in match_substrs) else train).append(e)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def stratified_split(entries, test_size=0.2, seed=SEED):
    rng = random.Random(seed)
    by_label = {}
    for e in entries:
        by_label.setdefault(e['tactic'], []).append(e)
    train, test = [], []
    for label, items in by_label.items():
        items = items[:]
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_size))
        test  += items[:n_test]
        train += items[n_test:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


TECHNIQUES_USE_RE = re.compile(r'TECHNIQUES USE\s*\n-{10,}\s*\n+(.*?)\n+-{10,}', re.DOTALL)


def extract_techniques_use(text):
    m = TECHNIQUES_USE_RE.search(text)
    return ' '.join(m.group(1).split()) if m else text


def load_templates():
    templates = {}
    for fname in sorted(os.listdir(TEMPLATE_DIR)):
        if not fname.endswith('.txt'):
            continue
        label = None
        for lbl, tid in TACTIC_IDS.items():
            if tid in fname:
                label = lbl
                break
        if label is None:
            continue
        with open(os.path.join(TEMPLATE_DIR, fname), encoding='utf-8') as f:
            templates[label] = extract_techniques_use(f.read())
    return templates


def encode_one(tokenizer, encoder, text, device):
    enc  = tokenizer(text, padding=False, truncation=False, return_tensors='pt')
    rlen = int(enc['attention_mask'][0].sum())
    ids  = enc['input_ids'][0][:rlen].unsqueeze(0).to(device)
    mask = enc['attention_mask'][0][:rlen].unsqueeze(0).to(device)
    return embed_text(encoder, tokenizer, ids, mask, device, truncate=False).squeeze(0)


def run_contrastive_train(n_epochs=N_EPOCHS, test_file_match=None, patience=PATIENCE, k_per_tactic=K_PER_TACTIC):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('  Device   : {}'.format(device))
    print('  Tactics  : {}'.format(', '.join(ALL_TACTICS)))
    print()

    entries = load_sequences()
    if test_file_match:
        train_entries, test_entries = leave_out_split(entries, test_file_match)
        print('  Split mode: leave-out — test = every file matching "{}"'.format(test_file_match))
    else:
        train_entries, test_entries = stratified_split(entries)
        print('  Split mode: stratified 80/20 (seed={})'.format(SEED))

    tactic_to_idxs = {}
    for i, e in enumerate(train_entries):
        tactic_to_idxs.setdefault(e['tactic'], []).append(i)
    print('  Train sequences: {}  Test sequences: {}'.format(len(train_entries), len(test_entries)))
    for e in test_entries:
        print('    [test] {} / {}'.format(e['tactic'], e['file']))
    for lbl, idxs in tactic_to_idxs.items():
        print('    {} : {} train'.format(lbl, len(idxs)))
    if len(tactic_to_idxs) != 2:
        print('  ERROR: this script requires exactly 2 tactics with training data, found {}'.format(
            len(tactic_to_idxs)))
        return
    tactic_a, tactic_b = sorted(tactic_to_idxs.keys())

    templates = load_templates()
    print('  Templates: {} (real templates only — the other 13 MITRE tactics have zero CAM-LDS '
          'graphs and are excluded from training).'.format(', '.join(sorted(templates.keys()))))
    print('  Each step: {} anchor scored against its own template vs [{} template + ALL {} training '
          'graphs], and vice versa — every negative is a real, correctly-labeled graph/template pair.'
          .format(tactic_a, tactic_b, tactic_b))
    print()

    print('  Loading RoBERTa encoders...')
    tokenizer    = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    seq_encoder  = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    tmpl_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)

    freeze_lower_layers(seq_encoder,  n_freeze=N_FREEZE)
    freeze_lower_layers(tmpl_encoder, n_freeze=N_FREEZE)

    if device.type == 'cuda':
        seq_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        tmpl_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    seq_encoder.train()
    tmpl_encoder.train()
    n_layers = len(seq_encoder.encoder.layer)
    print('  Encoders loaded — layers 0-{} frozen, layers {}-{} trainable, gradient checkpointing {}.'.format(
        N_FREEZE - 1, N_FREEZE, n_layers - 1, 'ON' if device.type == 'cuda' else 'OFF (cpu)'))
    print()

    log_proj  = ProjectionNetwork(EMB_DIM, PROJ_DIM, DROPOUT).to(device)
    text_proj = ProjectionNetwork(EMB_DIM, PROJ_DIM, DROPOUT).to(device)

    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    optimizer = torch.optim.Adam(
        [p for p in seq_encoder.parameters()  if p.requires_grad] +
        [p for p in tmpl_encoder.parameters() if p.requires_grad] +
        list(log_proj.parameters())  +
        list(text_proj.parameters()) +
        [logit_scale],
        lr=LR,
    )

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_loss  = float('inf')
    no_improve = 0
    best_state = None
    history    = []

    train_start = time.time()
    print('  Training  epochs={} lr={} dropout={} amp={} patience={} k_per_tactic={}'.format(
        n_epochs, LR, DROPOUT, use_amp, patience, k_per_tactic))
    print('  Start time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print()

    idxs_a = tactic_to_idxs[tactic_a]
    idxs_b = tactic_to_idxs[tactic_b]

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        seq_encoder.train(); tmpl_encoder.train(); log_proj.train(); text_proj.train()

        k_a = min(k_per_tactic, len(idxs_a))
        k_b = min(k_per_tactic, len(idxs_b))
        anchor_a_idxs = random.sample(idxs_a, k_a)
        anchor_b_idxs = random.sample(idxs_b, k_b)

        texts_a = [train_entries[i]['text'] for i in idxs_a]
        texts_b = [train_entries[i]['text'] for i in idxs_b]
        pos_a_local = [idxs_a.index(i) for i in anchor_a_idxs]
        pos_b_local = [idxs_b.index(i) for i in anchor_b_idxs]

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                z_a = log_proj(torch.stack([encode_one(tokenizer, seq_encoder, t, device) for t in texts_a]))
                z_b = log_proj(torch.stack([encode_one(tokenizer, seq_encoder, t, device) for t in texts_b]))
                u_a = text_proj(encode_one(tokenizer, tmpl_encoder, templates[tactic_a], device))
                u_b = text_proj(encode_one(tokenizer, tmpl_encoder, templates[tactic_b], device))

                loss_a = two_tactic_directional_loss(z_a[pos_a_local], u_a, u_b, z_b, logit_scale)
                loss_b = two_tactic_directional_loss(z_b[pos_b_local], u_b, u_a, z_a, logit_scale)
                loss = (loss_a + loss_b) / 2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            z_a = log_proj(torch.stack([encode_one(tokenizer, seq_encoder, t, device) for t in texts_a]))
            z_b = log_proj(torch.stack([encode_one(tokenizer, seq_encoder, t, device) for t in texts_b]))
            u_a = text_proj(encode_one(tokenizer, tmpl_encoder, templates[tactic_a], device))
            u_b = text_proj(encode_one(tokenizer, tmpl_encoder, templates[tactic_b], device))

            loss_a = two_tactic_directional_loss(z_a[pos_a_local], u_a, u_b, z_b, logit_scale)
            loss_b = two_tactic_directional_loss(z_b[pos_b_local], u_b, u_a, z_a, logit_scale)
            loss = (loss_a + loss_b) / 2
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logit_scale.clamp_(0, math.log(100))

        history.append({
            'epoch'      : epoch,
            'loss'       : round(loss.item(), 6),
            'logit_scale': round(logit_scale.exp().item(), 4),
        })

        epoch_time = (time.time() - epoch_start) / 60
        elapsed    = (time.time() - train_start) / 60
        if epoch % 5 == 0 or epoch == 1:
            print('  epoch {:>4d}  loss={:.6f}  logit_scale={:.4f}  epoch_time={:.2f}m  total_time={:.2f}m'.format(
                      epoch, loss.item(), logit_scale.exp().item(), epoch_time, elapsed))

        if loss.item() < best_loss - 1e-6:
            best_loss  = loss.item()
            no_improve = 0
            best_state = {
                'seq_encoder' : {k: v.clone() for k, v in seq_encoder.state_dict().items()},
                'tmpl_encoder': {k: v.clone() for k, v in tmpl_encoder.state_dict().items()},
                'log_proj'    : {k: v.clone() for k, v in log_proj.state_dict().items()},
                'text_proj'   : {k: v.clone() for k, v in text_proj.state_dict().items()},
                'logit_scale' : logit_scale.detach().clone(),
                'epoch'       : epoch,
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print('  Early stopping at epoch {} (no improvement for {} epochs)'.format(epoch, patience))
                break

        if epoch % 10 == 0 and best_state:
            ckpt_path = os.path.join(OUTPUT_TRAINING, 'camlds_epoch{}.pt'.format(epoch))
            torch.save({
                'seq_encoder' : best_state['seq_encoder'],
                'tmpl_encoder': best_state['tmpl_encoder'],
                'log_proj'    : best_state['log_proj'],
                'text_proj'   : best_state['text_proj'],
                'logit_scale' : best_state['logit_scale'],
                'proj_dims'   : (EMB_DIM, PROJ_DIM),
                'tactics'     : ALL_TACTICS,
                'best_loss'   : best_loss,
                'history'     : history[:epoch],
                'seed'        : SEED,
            }, ckpt_path)
            print('  [checkpoint] camlds_epoch{}.pt saved (best_loss={:.6f})'.format(epoch, best_loss))

    if best_state:
        seq_encoder.load_state_dict(best_state['seq_encoder'])
        tmpl_encoder.load_state_dict(best_state['tmpl_encoder'])
        log_proj.load_state_dict(best_state['log_proj'])
        text_proj.load_state_dict(best_state['text_proj'])
        with torch.no_grad():
            logit_scale.copy_(best_state['logit_scale'])

    total_time = (time.time() - train_start) / 60
    print()
    print('  Best train loss : {:.6f}'.format(best_loss))
    print('  logit_scale     : {:.4f}'.format(logit_scale.exp().item()))
    print('  End time        : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('  Total time      : {:.2f}m'.format(total_time))

    model_path = os.path.join(OUTPUT_TRAINING, 'camlds_matcher.pt')
    torch.save({
        'seq_encoder' : seq_encoder.state_dict(),
        'tmpl_encoder': tmpl_encoder.state_dict(),
        'log_proj'    : log_proj.state_dict(),
        'text_proj'   : text_proj.state_dict(),
        'logit_scale' : logit_scale,
        'proj_dims'   : (EMB_DIM, PROJ_DIM),
        'tactics'     : ALL_TACTICS,
        'best_loss'   : best_loss,
        'best_epoch'  : best_state['epoch'] if best_state else None,
        'history'     : history,
        'seed'        : SEED,
    }, model_path)

    hist_path = os.path.join(RESULTS_DIR, 'train_history_camlds.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print('  camlds_matcher.pt         → {}'.format(model_path))
    print('  train_history_camlds.json → {}'.format(hist_path))
    print()
    print('  Run test_camlds_matcher.py for the detailed per-file test report.')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=N_EPOCHS)
    ap.add_argument('--patience', type=int, default=PATIENCE,
                    help='epochs with no accuracy/loss improvement before early stopping '
                         '(default {}). Training here is noisy, so a short patience can cut '
                         'it off before a later spike — raise this to let it run longer.'.format(PATIENCE))
    ap.add_argument('--k-per-tactic', type=int, default=K_PER_TACTIC,
                    help='anchors sampled per tactic per step (default {}), each step\'s loss '
                         'is the mean over these K real examples per direction instead of just '
                         '1 — smooths out the per-epoch loss noise. Clamped to the training pool '
                         'size if K is larger than the number of available sequences.'.format(K_PER_TACTIC))
    ap.add_argument('--test-file', type=str, default=None,
                    help='either a named preset (aug1/aug2/aug3 — see test_split_presets.py) or '
                         'a comma-separated list of substrings — hold out every sequence whose '
                         'file path contains ANY of them as the test set. Use full filenames '
                         '(not short substrings like "-4") to avoid accidental prefix matches, '
                         'e.g. "-41_", "-43_" etc. would all match a bare "-4" substring. '
                         'Train on everything else — overrides the default stratified 80/20 split')
    args = ap.parse_args()
    test_files = resolve_test_file_arg(args.test_file)
    run_contrastive_train(n_epochs=args.epochs, test_file_match=test_files, patience=args.patience,
                           k_per_tactic=args.k_per_tactic)
