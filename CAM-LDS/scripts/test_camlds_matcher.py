
import os
import re
import sys
import math
import json

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score

PROJECT_ROOT = '/csse/research/contructive-learning'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import ROBERTA_MODEL
from scripts.encoder_utils import embed_text
from test_split_presets import resolve_test_file_arg
from train_camlds_matcher import (
    load_sequences, leave_out_split, leave_out_scenario_split, random_split, stratified_split, load_templates,
    extract_techniques_use, ProjectionNetwork, SEED, TACTIC_IDS, MAX_CHUNKS,
)

CAM_LDS_DIR = '/csse/research/contructive-learning/CAM-LDS'
MODEL_DIR   = os.path.join(CAM_LDS_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(CAM_LDS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def step_sort_key(step_name):
    m = re.search(r'(\d+)$', step_name)
    return (int(m.group(1)) if m else 0, step_name)


RUN_GROUP_RE = re.compile(r'^(.+)-(\d+)$')


def run_group_of(step_name):
    m = RUN_GROUP_RE.match(step_name)
    return m.group(1) if m else step_name


def encode_one(tokenizer, encoder, text, device):
    enc  = tokenizer(text, padding=False, truncation=False, return_tensors='pt')
    rlen = int(enc['attention_mask'][0].sum())
    ids  = enc['input_ids'][0][:rlen].unsqueeze(0).to(device)
    mask = enc['attention_mask'][0][:rlen].unsqueeze(0).to(device)
    with torch.no_grad():
        return embed_text(encoder, tokenizer, ids, mask, device, truncate=False, max_chunks=MAX_CHUNKS).squeeze(0).cpu()


def score_table(test_entries, seq_embs, all_tactics, tmpl_embs_by_tactic, display_scale):
    zt_all = F.normalize(torch.stack([tmpl_embs_by_tactic[t] for t in all_tactics]), dim=-1)

    results = []
    with torch.no_grad():
        for entry, z_s in zip(test_entries, seq_embs):
            logits = display_scale * (z_s @ zt_all.T)
            probs = [round(p, 4) for p in F.softmax(logits, dim=-1).tolist()]
            scores = dict(zip(all_tactics, probs))
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranked_tactics = [t for t, s in ranked]

            true_tactics = entry['tactics']
            true_ranks = {t: ranked_tactics.index(t) + 1 for t in true_tactics}

            results.append({
                'file': entry['file'], 'true_tactics': true_tactics,
                'true_ranks': true_ranks, 'scores': scores,
                'ranked': [{'tactic': t, 'score': s} for t, s in ranked],
            })
    return results


def tid(tactic):
    return TACTIC_IDS.get(tactic, tactic)


def build_label_matrices(results, all_tactics):
    y_true = np.array([[1 if t in r['true_tactics'] else 0 for t in all_tactics] for r in results])
    y_score = np.array([[r['scores'][t] for t in all_tactics] for r in results])
    return y_true, y_score


def compute_lrap(results, all_tactics):
    y_true, y_score = build_label_matrices(results, all_tactics)
    return label_ranking_average_precision_score(y_true, y_score)


def compute_aupr(results, all_tactics):
    y_true, y_score = build_label_matrices(results, all_tactics)
    valid_cols = y_true.sum(axis=0) > 0
    if not valid_cols.any():
        return 0.0
    return average_precision_score(y_true[:, valid_cols], y_score[:, valid_cols], average='macro')


def print_legend(all_tactics):
    print('\n  Tactic ID legend:')
    for t in all_tactics:
        print('    {:<6} = {}'.format(tid(t), t))


def print_table(results, all_tactics, top_n=4):
    print_legend(all_tactics)
    col_file = 26
    col_true = 36
    print('\n  ── Test Results (scored against {} tactic prototypes) ──'.format(len(all_tactics)))
    header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} {:<14} {:<14} {:<14} {:<14} {:<7} {:<8} {}'
    print(header_fmt.format('#', 'File', 'True tactics', '#1 (score)', '#2 (score)', '#3 (score)', '#4 (score)', 'Found', 'Top4', 'Missing in top 4 tactic'))
    print('  ' + '-' * (4 + col_file + col_true + 14 + 14 + 14 + 14 + 7 + 8 + 14))
    row_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} {:<14} {:<14} {:<14} {:<14} {:<7} {:<8} {}'
    n_wrong_top4 = 0
    for i, r in enumerate(results, 1):
        ranked = r['ranked']
        cols = []
        for j in range(top_n):
            if j < len(ranked):
                cols.append('{} {:.4f}'.format(tid(ranked[j]['tactic']), ranked[j]['score']))
            else:
                cols.append('')
        true_str = ','.join(tid(t) for t in r['true_tactics'])
        fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
        top4_tactics = {ranked[j]['tactic'] for j in range(min(4, len(ranked)))}
        true_set = set(r['true_tactics'])
        n_found = len(top4_tactics & true_set)
        n_true = len(true_set)
        found_str = '{}/{}'.format(n_found, n_true) if n_true else '-'
        missing_str = ','.join(tid(t) for t in true_set - top4_tactics)
        top4_wrong = bool(ranked) and not any(t in top4_tactics for t in r['true_tactics'])
        if top4_wrong:
            n_wrong_top4 += 1
        flag = 'WRONG' if top4_wrong else ''
        print(row_fmt.format(i, fname, true_str, cols[0], cols[1], cols[2], cols[3], found_str, flag, missing_str))

    n = len(results)
    lrap = compute_lrap(results, all_tactics) if n else 0.0
    aupr = compute_aupr(results, all_tactics) if n else 0.0

    print()
    print('  ─────────────────────────────────────')
    print('  LRAP (Label Ranking Average Precision)          : {:.1f}%'.format(lrap * 100))
    print('  AUPR (Area Under Precision-Recall Curve, macro) : {:.1f}%'.format(aupr * 100))
    print('  Wrong (true label not in top-4)                 : {}/{} samples'.format(n_wrong_top4, n))

    return {
        'lrap'          : lrap,
        'aupr'          : aupr,
        'n_total'       : n,
        'n_wrong_top4'  : n_wrong_top4,
        'results'       : results,
    }


def run(test_file_match=None, test_scenario=None, exclude_match=None, temp=0.07, test_size=0.2, split_seed=SEED,
        run_tag=None, min_events=None, template_dir=None, sequences_dir=None, run_filter=None):
    device = torch.device('cuda')
    print('  Device: {}'.format(device))

    if run_tag is None:
        if test_scenario:
            run_tag = 'scenario{}'.format(test_scenario)
        elif test_file_match:
            run_tag = test_file_match
        else:
            run_tag = 'seed{}'.format(split_seed)

    ckpt_path = os.path.join(MODEL_DIR, 'camlds_matcher_{}.pt'.format(run_tag))
    if not os.path.exists(ckpt_path):
        print('  ERROR: model not found at {}'.format(ckpt_path))
        print('  Run train_camlds_matcher.py --run-tag {} first (or matching --test-file/--split-seed).'.format(run_tag))
        return

    print('\n  Loading model from {}'.format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device)
    all_tactics = ckpt['tactics']
    print('  Best train loss: {:.4f}  Best epoch: {}'.format(ckpt.get('best_loss', 0), ckpt.get('best_epoch', '?')))
    print('  Tactics (prototypes) this model was trained on: {}'.format(', '.join(all_tactics)))

    trained_logit_scale = ckpt.get('logit_scale')
    trained_scale = trained_logit_scale.exp().item() if trained_logit_scale is not None else 1.0
    display_scale = math.exp(math.log(1.0 / temp)) if temp else trained_scale
    print('  Trained scale: {:.2f} (temp={:.4f})   Display scale: {:.2f}{}'.format(
        trained_scale, 1 / trained_scale, display_scale,
        ' (temp={} override)'.format(temp) if temp else ' (using trained value)'))

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

    ckpt_test_size      = ckpt.get('test_size', test_size)
    ckpt_split_seed     = ckpt.get('split_seed', split_seed)
    ckpt_test_scenario  = test_scenario if test_scenario else ckpt.get('test_scenario')
    ckpt_test_file_match = test_file_match if test_file_match else ckpt.get('test_file_match')
    ckpt_exclude_match  = exclude_match if exclude_match else ckpt.get('exclude_match')

    ckpt_sequences_dir = sequences_dir or ckpt.get('sequences_dir')
    entries = load_sequences(min_events=min_events, sequences_dir=ckpt_sequences_dir)
    excluded_steps = []
    excluded_substrs = None
    if ckpt_exclude_match:
        excluded_substrs = [ckpt_exclude_match] if isinstance(ckpt_exclude_match, str) else ckpt_exclude_match
        excluded_steps = [e['file'] for e in entries if any(m in e['file'] for m in excluded_substrs)]
        entries = [e for e in entries if e['file'] not in excluded_steps]
    ckpt_stratified = ckpt.get('stratified', False)
    if ckpt_test_scenario:
        print('\n  Loading test sequences (leave-one-scenario-out — test = ALL steps of scenario {})...'.format(
            ckpt_test_scenario))
        _, test_entries = leave_out_scenario_split(entries, ckpt_test_scenario, seed=ckpt_split_seed)
    elif ckpt_test_file_match:
        print('\n  Loading test sequences (leave-out mode — test = files matching "{}")...'.format(ckpt_test_file_match))
        _, test_entries = leave_out_split(entries, ckpt_test_file_match)
    elif ckpt_stratified:
        print('\n  Loading test sequences (same stratified split as training, test_size={} split_seed={})...'.format(
            ckpt_test_size, ckpt_split_seed))
        _, test_entries = stratified_split(entries, test_size=ckpt_test_size, seed=ckpt_split_seed)
    else:
        print('\n  Loading test sequences (same random split as training, test_size={} split_seed={})...'.format(
            ckpt_test_size, ckpt_split_seed))
        _, test_entries = random_split(entries, test_size=ckpt_test_size, seed=ckpt_split_seed)
    print('  Test sequences: {}'.format(len(test_entries)))

    print('\n  Encoding test sequences...')
    with torch.no_grad():
        seq_embs = [F.normalize(log_proj(encode_one(tokenizer, seq_encoder, e['sequence'], device).to(device)), dim=-1).cpu()
                    for e in test_entries]

    ckpt_template_dir = template_dir or ckpt.get('template_dir')
    print('\n  Encoding {} tactic templates... (dir={})'.format(len(all_tactics), ckpt_template_dir or 'templates_dc (default)'))
    templates = load_templates(template_dir=ckpt_template_dir)
    with torch.no_grad():
        tmpl_embs_by_tactic = {
            t: F.normalize(text_proj(encode_one(tokenizer, tmpl_encoder, templates[t], device).to(device)), dim=-1).cpu()
            for t in all_tactics
        }

    results = score_table(test_entries, seq_embs, all_tactics, tmpl_embs_by_tactic, display_scale)
    results = sorted(results, key=lambda r: step_sort_key(r['file']))

    display_results = results
    if run_filter:
        display_results = [r for r in results if run_group_of(r['file']) == run_filter]
        print('\n  (showing only run group: {})'.format(run_filter))

    if excluded_steps:
        print('\n  Excluded from BOTH train and test (leakage guard, matching {}): {} steps -- {}'.format(
            excluded_substrs, len(excluded_steps), excluded_steps))

    out = print_table(display_results, all_tactics)
    out['results'] = results

    results_path = os.path.join(RESULTS_DIR, 'camlds_test_results_{}.json'.format(run_tag))
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n  Results saved → {}'.format(results_path))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-file', type=str, default=None)
    ap.add_argument('--test-scenario', type=str, default=None,
                     help='Evaluate on an ENTIRE held-out scenario (e.g. "7"). Normally not needed -- '
                          'this is read from the checkpoint automatically if it was trained with '
                          '--test-scenario. Pass it here only to override.')
    ap.add_argument('--exclude', type=str, default=None,
                     help='Comma-separated substrings to drop from both train and test (normally read '
                          'automatically from the checkpoint). Pass it here only to override.')
    ap.add_argument('--temp', type=float, default=0.07)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=SEED)
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--template-dir', type=str, default=None,
                     help='Override the template dir (default: reads it from the checkpoint, same one used to train).')
    ap.add_argument('--sequences-dir', type=str, default=None,
                     help='Override the sequences dir (default: reads it from the checkpoint, same one used to train).')
    ap.add_argument('--run', type=str, default=None,
                     help='Only print/score this one run group (exact match), e.g. --run 2_cron. '
                          'The saved JSON still has everything.')
    args = ap.parse_args()
    test_files = resolve_test_file_arg(args.test_file)
    exclude_match = [s.strip() for s in args.exclude.split(',')] if args.exclude else None

    run_tag = args.run_tag
    if run_tag is None:
        if args.test_scenario:
            run_tag = 'scenario{}'.format(args.test_scenario)
        elif args.test_file:
            run_tag = args.test_file

    run(test_file_match=test_files, test_scenario=args.test_scenario, exclude_match=exclude_match,
        temp=args.temp, test_size=args.test_size,
        split_seed=args.split_seed, run_tag=run_tag, run_filter=args.run,
        template_dir=args.template_dir, sequences_dir=args.sequences_dir)
