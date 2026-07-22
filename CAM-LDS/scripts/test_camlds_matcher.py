
import os
import re
import sys
import math
import json
import glob
import random

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

CAM_LDS_DIR     = '/csse/research/contructive-learning/CAM-LDS'
SEQUENCES_DIR   = os.path.join(CAM_LDS_DIR, 'sequences')
TEMPLATE_DIR    = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'templates_dc')
MODEL_DIR       = os.path.join(CAM_LDS_DIR, 'checkpoints')
RESULTS_DIR     = os.path.join(CAM_LDS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TACTIC_FOLDER_TO_LABEL = {
    'command_and_control': 'Command_and_Control',
    'initial_access':      'Initial_Access',
}


TACTIC_IDS = {
    'Command_and_Control': 'TA0011',
    'Initial_Access':      'TA0001',
}


ALL_TACTIC_IDS = {
    'Initial_Access':       'TA0001',
    'Execution':            'TA0002',
    'Persistence':          'TA0003',
    'Privilege_Escalation': 'TA0004',
    'Stealth':               'TA0005',
    'Defense_Impairment':   'TA0112',
    'Credential_Access':    'TA0006',
    'Discovery':             'TA0007',
    'Lateral_Movement':     'TA0008',
    'Collection':             'TA0009',
    'Command_and_Control':  'TA0011',
    'Exfiltration':           'TA0010',
    'Impact':                 'TA0040',
    'Reconnaissance':        'TA0043',
    'Resource_Development': 'TA0042',
}

SEED     = 42

PROJ_DIM = 128
DROPOUT  = 0.5
EMB_DIM  = 768


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
    for label, es in by_tactic.items():
        for e in es:
            if e['basename'] not in shared:
                entries.append(e)
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


def load_templates(all_templates=False):
    id_map = ALL_TACTIC_IDS if all_templates else TACTIC_IDS
    templates = {}
    for fname in sorted(os.listdir(TEMPLATE_DIR)):
        if not fname.endswith('.txt'):
            continue
        label = None
        for lbl, tid in id_map.items():
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
    with torch.no_grad():
        return embed_text(encoder, tokenizer, ids, mask, device, truncate=False).squeeze(0).cpu()


def short(label, n=20):
    return label[:n]


def score_table(test_entries, seq_embs, tokenizer, tmpl_encoder, text_proj,
                 display_scale, all_templates):
    templates = load_templates(all_templates=all_templates)
    labels = list(templates.keys())
    texts  = [templates[l] for l in labels]

    tmpl_embs = torch.stack([encode_one(tokenizer, tmpl_encoder, t, tmpl_encoder.device) for t in texts])
    zt_all = F.normalize(torch.stack([text_proj(e.to(tmpl_encoder.device)) for e in tmpl_embs]), dim=-1)

    results = []
    with torch.no_grad():
        for entry, z_s in zip(test_entries, seq_embs):
            logits = display_scale * (z_s.to(tmpl_encoder.device) @ zt_all.T)
            probs = F.softmax(logits, dim=-1).cpu().tolist()
            scores = dict(zip(labels, probs))
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            pred_label, pred_score = ranked[0]
            true_label = entry['tactic']
            true_score = scores[true_label]
            true_rank  = [l for l, s in ranked].index(true_label) + 1
            top1_exact = pred_label == true_label
            if all_templates:
                correct = true_label in [r[0] for r in ranked[:3]]
            else:
                correct = top1_exact

            results.append({
                'file': entry['file'], 'true_tactic': true_label, 'pred_tactic': pred_label,
                'correct': correct, 'top1_exact': top1_exact, 'true_score': true_score,
                'true_rank': true_rank, 'scores': scores,
                'ranked': [{'tactic': l, 'score': s} for l, s in ranked],
            })
    return results, labels


def print_table(results, labels, all_templates):
    col_file = 40
    print('\n  ── Test Results (scored against {} templates: {}) ──'.format(
        len(labels), ', '.join(labels)))
    if all_templates:
        header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<14} {:<20} {:<20} {:<20} {:>10} {:>5} {:<6}'
        print(header_fmt.format('#', 'File', 'True', '#1 (score)', '#2 (score)', '#3 (score)', 'TrueScore', 'Rank', 'Result'))
        print('  ' + '-' * (4 + col_file + 14 + 20 + 20 + 20 + 10 + 5 + 6 + 8))
        row_fmt = '  {:<4} {:<' + str(col_file) + '} {:<14} {:<20} {:<20} {:<20} {:>10.4f} {:>5} {:<6}'
        for i, r in enumerate(results, 1):
            ranked = r['ranked']
            p1 = f"{short(ranked[0]['tactic'], 12)} {ranked[0]['score']:.4f}"
            p2 = f"{short(ranked[1]['tactic'], 12)} {ranked[1]['score']:.4f}"
            p3 = f"{short(ranked[2]['tactic'], 12)} {ranked[2]['score']:.4f}"
            result = 'MATCH' if r['correct'] else 'MISS'
            if r['top1_exact']:
                result += '*'
            fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
            print(row_fmt.format(
                i, fname, short(r['true_tactic'], 12), p1, p2, p3, r['true_score'], f"#{r['true_rank']}", result
            ))
        print('  (* = exact top-1 match, not just top-3, out of {} templates)'.format(len(labels)))
    else:
        header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<22} {:<26} {:<26} {:<6}'
        print(header_fmt.format('#', 'File', 'True', '#1 (score)', '#2 (score)', 'Result'))
        print('  ' + '-' * (4 + col_file + 22 + 26 + 26 + 6 + 6))
        row_fmt = '  {:<4} {:<' + str(col_file) + '} {:<22} {:<26} {:<26} {:<6}'
        for i, r in enumerate(results, 1):
            ranked = r['ranked']
            p1 = f"{short(ranked[0]['tactic'])} {ranked[0]['score']:.4f}"
            p2 = f"{short(ranked[1]['tactic'])} {ranked[1]['score']:.4f}"
            result = 'MATCH' if r['correct'] else 'MISS'
            fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
            print(row_fmt.format(i, fname, short(r['true_tactic']), p1, p2, result))

    n_correct    = sum(r['correct'] for r in results)
    n_top1_exact = sum(r['top1_exact'] for r in results)
    accuracy      = n_correct / len(results) if results else 0.0
    top1_accuracy = n_top1_exact / len(results) if results else 0.0

    from collections import Counter
    pred_counts = Counter(r['pred_tactic'] for r in results)
    true_counts = Counter(r['true_tactic'] for r in results)

    print()
    print('  ─────────────────────────────────────')
    print('  Accuracy (top-1) : {}/{} ({:.1f}%)'.format(n_top1_exact, len(results), top1_accuracy * 100))
    if all_templates:
        print('  Accuracy (top-3) : {}/{} ({:.1f}%)'.format(n_correct, len(results), accuracy * 100))
    print()
    print('  Prediction breakdown (how many times each tactic was picked as #1):')
    for tactic, n in pred_counts.most_common():
        print('    {:<22} predicted {} times  (actually true {} times in this test set)'.format(
            tactic, n, true_counts.get(tactic, 0)))

    return {
        'accuracy'     : accuracy,
        'n_correct'    : n_correct,
        'top1_accuracy': top1_accuracy,
        'n_top1_exact' : n_top1_exact,
        'n_total'      : len(results),
        'results'      : results,
    }


def run(test_file_match=None, temp=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    ckpt_path = os.path.join(MODEL_DIR, 'camlds_matcher.pt')
    if not os.path.exists(ckpt_path):
        print(f'  ERROR: model not found at {ckpt_path}')
        print('  Run train_camlds_matcher.py first.')
        return

    print(f'\n  Loading model from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f'  Best train loss: {ckpt.get("best_loss", 0):.4f}  Best epoch: {ckpt.get("best_epoch", "?")}')


    trained_logit_scale = ckpt.get('logit_scale')
    trained_scale = trained_logit_scale.exp().item() if trained_logit_scale is not None else 1.0
    display_scale = math.exp(math.log(1.0 / temp)) if temp else trained_scale
    print(f'  Trained scale: {trained_scale:.2f} (temp={1/trained_scale:.4f})   '
          f'Display scale: {display_scale:.2f}' + (f' (temp={temp} override)' if temp else ' (using trained value)'))

    log_proj  = ProjectionNetwork().to(device)
    text_proj = ProjectionNetwork().to(device)
    log_proj.load_state_dict(ckpt['log_proj'])
    text_proj.load_state_dict(ckpt['text_proj'])
    log_proj.eval()
    text_proj.eval()

    print('\n  Loading RoBERTa encoders (fine-tuned weights from checkpoint)...')
    tokenizer    = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    seq_encoder  = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    tmpl_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    seq_encoder.load_state_dict(ckpt['seq_encoder'])
    tmpl_encoder.load_state_dict(ckpt['tmpl_encoder'])
    seq_encoder.eval()
    tmpl_encoder.eval()
    for p in seq_encoder.parameters():
        p.requires_grad = False
    for p in tmpl_encoder.parameters():
        p.requires_grad = False

    entries = load_sequences()
    if test_file_match:
        print('\n  Loading test sequences (leave-out mode — test = files matching "{}")...'.format(test_file_match))
        _, test_entries = leave_out_split(entries, test_file_match)
    else:
        print('\n  Loading test sequences (same stratified split as training, seed={})...'.format(SEED))
        _, test_entries = stratified_split(entries)
    print(f'  Test sequences: {len(test_entries)}')
    for e in test_entries:
        print(f'    [test] {e["tactic"]} / {e["file"]}')


    print('\n  Encoding test sequences...')
    with torch.no_grad():
        seq_embs = [F.normalize(log_proj(encode_one(tokenizer, seq_encoder, e['text'], device).to(device)), dim=-1).cpu()
                    for e in test_entries]

    out_by_mode = {}
    for all_templates in (True, False):
        print('\n  Loading templates ({})...'.format('all 15' if all_templates else '2 real'))
        results, labels = score_table(test_entries, seq_embs, tokenizer, tmpl_encoder, text_proj,
                                       display_scale, all_templates)
        out = print_table(results, labels, all_templates)
        out_by_mode['all_templates' if all_templates else 'two_templates'] = out

    results_path = os.path.join(RESULTS_DIR, 'camlds_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(out_by_mode, f, indent=2)
    print(f'\n  Results saved → {results_path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-file', type=str, default=None,
                    help='either a named preset (aug1/aug2/aug3 — see test_split_presets.py) or '
                         'a comma-separated list — must exactly match whatever --test-file was '
                         'passed to train_camlds_matcher.py, otherwise the reconstructed test '
                         'split won\'t match what the model was actually trained/held-out on')
    ap.add_argument('--temp', type=float, default=0.01,
                    help='display-only temperature for the printed scores (default 0.01, scale=100 '
                         '— the max the model is clamped to). Lower = bigger visual gap between '
                         'match/non-match. Does not change which template wins, only how spread '
                         'out the printed numbers look.')
    args = ap.parse_args()
    test_files = resolve_test_file_arg(args.test_file)
    run(test_file_match=test_files, temp=args.temp)
