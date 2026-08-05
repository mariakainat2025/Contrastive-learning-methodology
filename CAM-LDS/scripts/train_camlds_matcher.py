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
from prototype_multilabel_loss import multilabel_prototype_loss

CAM_LDS_DIR       = '/csse/research/contructive-learning/CAM-LDS'
SEQUENCES_DIR     = os.path.join(CAM_LDS_DIR, 'sequences')
TEMPLATE_DIR      = os.path.join(CAM_LDS_DIR, 'templates_dc')
STEP_TACTICS_PATH = os.path.join(CAM_LDS_DIR, 'scripts', 'step_tactics.json')
OUTPUT_TRAINING = os.path.join(CAM_LDS_DIR, 'checkpoints')
RESULTS_DIR     = os.path.join(CAM_LDS_DIR, 'results')
os.makedirs(OUTPUT_TRAINING, exist_ok=True)
os.makedirs(RESULTS_DIR,     exist_ok=True)

TACTIC_FOLDERS = ['command_and_control', 'credential_access', 'execution', 'initial_access',
                   'lateral_movement', 'persistence']

TACTIC_IDS = {
    'collection':            'TA0009',
    'command_and_control':   'TA0011',
    'credential_access':     'TA0006',
    'defense_impairment':    'TA0112',
    'discovery':             'TA0007',
    'execution':             'TA0002',
    'exfiltration':          'TA0010',
    'impact':                'TA0040',
    'initial_access':        'TA0001',
    'lateral_movement':      'TA0008',
    'persistence':           'TA0003',
    'privilege_escalation':  'TA0004',
    'reconnaissance':        'TA0043',
    'stealth':               'TA0005',
}

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


def freeze_lower_layers(model, n_freeze=N_FREEZE):
    for param in model.embeddings.parameters():
        param.requires_grad = False
    for i in range(n_freeze):
        for param in model.encoder.layer[i].parameters():
            param.requires_grad = False


def load_step_tactics():
    with open(STEP_TACTICS_PATH) as f:
        report = json.load(f)
    step_tactics = {}
    for row in report:
        tactics = set()
        for occ in row['occurrences']:
            tactics.update(occ['tactics'])
        step_tactics[row['step']] = sorted(t for t in tactics if t in TACTIC_IDS)
    return step_tactics


def load_sequences(min_events=None):
    step_tactics = load_step_tactics()

    by_step = {}
    for folder in TACTIC_FOLDERS:
        for fpath in sorted(glob.glob(os.path.join(SEQUENCES_DIR, folder, '*', 'sequence_*.json'))):
            with open(fpath) as f:
                seq = json.load(f)
            step = seq['step']
            if step in by_step:
                continue
            labels = step_tactics.get(step, [])
            by_step[step] = {
                'sequence': ' '.join(seq['sequence']),
                'tactics' : labels,
                'file'    : step,
                'n_events': len(seq['sequence']),
            }

    entries = list(by_step.values())

    if min_events is not None:
        n_before = len(entries)
        dropped = [e['file'] for e in entries if e['n_events'] < min_events]
        entries = [e for e in entries if e['n_events'] >= min_events]
        print('  min_events={}: dropped {}/{} sequences with too few events: {}'.format(
            min_events, len(dropped), n_before, dropped))

    multi = [e for e in entries if len(e['tactics']) > 1]
    print('  Loaded {} unique sequences ({} multi-tactic — kept as multi-label, not dropped).'.format(
        len(entries), len(multi)))
    for e in multi:
        print('    - {} : {}'.format(e['file'], e['tactics']))

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


def random_split(entries, test_size=0.2, seed=SEED):
    rng = random.Random(seed)
    items = entries[:]
    rng.shuffle(items)
    n_test = max(1, int(len(items) * test_size))
    test  = items[:n_test]
    train = items[n_test:]
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


def build_pos_mask(entries, all_tactics, tactic_to_col):
    mask = torch.zeros(len(entries), len(all_tactics))
    for i, e in enumerate(entries):
        for t in e['tactics']:
            mask[i, tactic_to_col[t]] = 1
    return mask


def build_tactic_pools(entries, all_tactics):
    return {t: [i for i, e in enumerate(entries) if t in e['tactics']] for t in all_tactics}


def stratified_batch_indices(tactic_pools, all_tactics, k_per_tactic, rng):
    chosen = set()
    for t in all_tactics:
        pool = tactic_pools[t]
        if not pool:
            continue
        if len(pool) >= k_per_tactic:
            chosen.update(rng.sample(pool, k_per_tactic))
        else:
            chosen.update(rng.choices(pool, k=k_per_tactic))
    return sorted(chosen)


def run_contrastive_train(n_epochs=N_EPOCHS, test_file_match=None, patience=PATIENCE, k_per_tactic=K_PER_TACTIC,
                           test_size=0.2, split_seed=SEED, run_tag=None, min_events=None):
    device = torch.device('cuda')
    print('  Device   : {}'.format(device))
    print()

    entries = load_sequences(min_events=min_events)

    all_tactics = sorted({t for e in entries for t in e['tactics']})
    tactic_to_col = {t: i for i, t in enumerate(all_tactics)}
    print('  Tactics (from full true multi-label membership of our {} steps): {}'.format(
        len(entries), ', '.join(all_tactics)))

    if run_tag is None:
        run_tag = test_file_match if test_file_match else 'seed{}'.format(split_seed)

    if test_file_match:
        train_entries, test_entries = leave_out_split(entries, test_file_match)
        print('  Split mode: leave-out — test = every file matching "{}"'.format(test_file_match))
    else:
        train_entries, test_entries = random_split(entries, test_size=test_size, seed=split_seed)
        print('  Split mode: random {:.0f}/{:.0f} (split_seed={})'.format(
            (1 - test_size) * 100, test_size * 100, split_seed))

    tactic_counts = {t: 0 for t in all_tactics}
    for e in train_entries:
        for t in e['tactics']:
            tactic_counts[t] += 1
    print('  Train sequences: {}  Test sequences: {}'.format(len(train_entries), len(test_entries)))
    for e in test_entries:
        print('    [test] {} / {}'.format(e['tactics'], e['file']))
    for t in all_tactics:
        print('    {} : {} train'.format(t, tactic_counts[t]))
    if not train_entries:
        print('  ERROR: no training sequences found.')
        return
    missing = [t for t in all_tactics if tactic_counts[t] == 0]
    if missing:
        print('  WARNING: these tactics have zero training examples: {} '
              '(their prototype will only ever be learned as a negative).'.format(missing))

    templates = load_templates()
    print('  Templates: {}'.format(', '.join(sorted(templates.keys()))))
    print()

    print('  Loading RoBERTa encoders...')
    tokenizer    = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    seq_encoder  = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    freeze_lower_layers(seq_encoder, n_freeze=N_FREEZE)
    seq_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    seq_encoder.train()

    tmpl_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    freeze_lower_layers(tmpl_encoder, n_freeze=N_FREEZE)
    tmpl_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    tmpl_encoder.train()
    text_proj = ProjectionNetwork(EMB_DIM, PROJ_DIM, DROPOUT).to(device)

    n_layers = len(seq_encoder.encoder.layer)
    print('  Encoders loaded — layers 0-{} frozen, layers {}-{} trainable, gradient checkpointing ON.'.format(
        N_FREEZE - 1, N_FREEZE, n_layers - 1))
    print()

    log_proj = ProjectionNetwork(EMB_DIM, PROJ_DIM, DROPOUT).to(device)

    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    trainable_params = ([p for p in seq_encoder.parameters() if p.requires_grad] + list(log_proj.parameters()) +
                         [p for p in tmpl_encoder.parameters() if p.requires_grad] + list(text_proj.parameters()) +
                         [logit_scale])

    optimizer = torch.optim.Adam(trainable_params, lr=LR)

    scaler = torch.amp.GradScaler('cuda')

    best_loss  = float('inf')
    no_improve = 0
    best_state = None
    history    = []

    train_start = time.time()
    batch_size = min(len(train_entries), k_per_tactic * len(all_tactics))
    print('  Training  epochs={} lr={} dropout={} patience={} batch_size={}'.format(
        n_epochs, LR, DROPOUT, patience, batch_size))
    print('  Start time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print()

    train_pos_mask = build_pos_mask(train_entries, all_tactics, tactic_to_col).to(device)
    tactic_pools = build_tactic_pools(train_entries, all_tactics)
    steps_per_epoch = max(1, -(-len(train_entries) // batch_size))

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        seq_encoder.train(); tmpl_encoder.train(); log_proj.train(); text_proj.train()

        epoch_losses = []

        for _ in range(steps_per_epoch):
            batch_idxs = stratified_batch_indices(tactic_pools, all_tactics, k_per_tactic, random)
            texts = [train_entries[i]['sequence'] for i in batch_idxs]
            pos_mask_batch = train_pos_mask[batch_idxs]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                z = log_proj(torch.stack([encode_one(tokenizer, seq_encoder, t, device) for t in texts]))
                prototypes = text_proj(torch.stack(
                    [encode_one(tokenizer, tmpl_encoder, templates[t], device) for t in all_tactics]))
                loss = multilabel_prototype_loss(z, prototypes, pos_mask_batch, logit_scale)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                logit_scale.clamp_(0, math.log(100))

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)

        history.append({
            'epoch'      : epoch,
            'loss'       : round(epoch_loss, 6),
            'n_steps'    : len(epoch_losses),
            'logit_scale': round(logit_scale.exp().item(), 4),
        })

        epoch_time = (time.time() - epoch_start) / 60
        elapsed    = (time.time() - train_start) / 60
        if epoch % 5 == 0 or epoch == 1:
            print('  epoch {:>4d}  loss={:.6f}  steps={}  logit_scale={:.4f}  epoch_time={:.2f}m  total_time={:.2f}m'.format(
                      epoch, epoch_loss, len(epoch_losses), logit_scale.exp().item(), epoch_time, elapsed))

        if epoch_loss < best_loss - 1e-6:
            best_loss  = epoch_loss
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
            ckpt_name = 'camlds_{}_epoch{}.pt'.format(run_tag, epoch)
            ckpt_path = os.path.join(OUTPUT_TRAINING, ckpt_name)
            torch.save({
                'seq_encoder' : best_state['seq_encoder'],
                'tmpl_encoder': best_state['tmpl_encoder'],
                'log_proj'    : best_state['log_proj'],
                'text_proj'   : best_state['text_proj'],
                'logit_scale' : best_state['logit_scale'],
                'proj_dims'   : (EMB_DIM, PROJ_DIM),
                'tactics'     : all_tactics,
                'best_loss'   : best_loss,
                'history'     : history[:epoch],
                'seed'        : SEED,
                'split_seed'  : split_seed,
                'test_size'   : test_size,
                'run_tag'     : run_tag,
                'test_file_match': test_file_match,
            }, ckpt_path)
            print('  [checkpoint] {} saved (best_loss={:.6f})'.format(ckpt_name, best_loss))

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

    model_path = os.path.join(OUTPUT_TRAINING, 'camlds_matcher_{}.pt'.format(run_tag))
    torch.save({
        'seq_encoder' : seq_encoder.state_dict(),
        'tmpl_encoder': tmpl_encoder.state_dict(),
        'log_proj'    : log_proj.state_dict(),
        'text_proj'   : text_proj.state_dict(),
        'logit_scale' : logit_scale,
        'proj_dims'   : (EMB_DIM, PROJ_DIM),
        'tactics'     : all_tactics,
        'best_loss'   : best_loss,
        'best_epoch'  : best_state['epoch'] if best_state else None,
        'history'     : history,
        'seed'        : SEED,
        'split_seed'  : split_seed,
        'test_size'   : test_size,
        'run_tag'     : run_tag,
        'test_file_match': test_file_match,
    }, model_path)

    hist_path = os.path.join(RESULTS_DIR, 'train_history_camlds_{}.json'.format(run_tag))
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print('  Checkpoint    → {}'.format(model_path))
    print('  Train history → {}'.format(hist_path))
    print()
    print('  Run test_camlds_matcher.py --run-tag {} for the detailed per-file test report.'.format(run_tag))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-per-tactic', type=int, default=K_PER_TACTIC)
    ap.add_argument('--test-file', type=str, default=None)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=SEED)
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--min-events', type=int, default=None,
                     help='Drop sequences with fewer than this many events before splitting.')
    args = ap.parse_args()
    test_files = resolve_test_file_arg(args.test_file)

    run_tag = args.run_tag
    if run_tag is None and args.test_file:
        run_tag = args.test_file

    run_contrastive_train(test_file_match=test_files, k_per_tactic=args.k_per_tactic,
                           test_size=args.test_size, split_seed=args.split_seed, run_tag=run_tag,
                           min_events=args.min_events)
