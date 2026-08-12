
import os
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
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scripts.config import ROBERTA_MODEL
from test_split_presets import resolve_test_file_arg
from train_camlds_matcher import (
    load_sequences, leave_out_split, random_split, load_templates,
    ProjectionNetwork, encode_one, SEED, TACTIC_IDS,
)

CAM_LDS_DIR = '/csse/research/contructive-learning/CAM-LDS'
MODEL_DIR   = os.path.join(CAM_LDS_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(CAM_LDS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def tid(tactic):
    return TACTIC_IDS.get(tactic, tactic)


def score_table(test_entries, seq_embs, all_tactics, proto_embs, display_scale):
    zt_all = F.normalize(proto_embs, dim=-1)  

    results = []
    with torch.no_grad():
        for entry, z_s in zip(test_entries, seq_embs):
            logits = display_scale * (z_s @ zt_all.T)
            probs = [round(p, 6) for p in F.softmax(logits, dim=-1).tolist()]
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


def print_table(results, all_tactics, top_n=3):
    print_legend(all_tactics)
    col_file = 26
    col_true = 36
    print('\n  ── Test Results (scored against {} LEARNED class prototypes) ──'.format(len(all_tactics)))
    header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} {:<14} {:<14} {:<14} {:<8}'
    print(header_fmt.format('#', 'File', 'True tactics', '#1 (score)', '#2 (score)', '#3 (score)', 'Top1'))
    print('  ' + '-' * (4 + col_file + col_true + 14 + 14 + 14 + 8 + 14))
    row_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} {:<14} {:<14} {:<14} {:<8}'
    n_wrong_top1 = 0
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
        top1_wrong = bool(ranked) and ranked[0]['tactic'] not in r['true_tactics']
        if top1_wrong:
            n_wrong_top1 += 1
        flag = 'WRONG' if top1_wrong else ''
        print(row_fmt.format(i, fname, true_str, cols[0], cols[1], cols[2], flag))

    n = len(results)
    lrap = compute_lrap(results, all_tactics) if n else 0.0
    aupr = compute_aupr(results, all_tactics) if n else 0.0

    print()
    print('  ─────────────────────────────────────')
    print('  LRAP (Label Ranking Average Precision)          : {:.1f}%'.format(lrap * 100))
    print('  AUPR (Area Under Precision-Recall Curve, macro) : {:.1f}%'.format(aupr * 100))
    print('  Totally wrong top-1 prediction                  : {}/{} samples'.format(n_wrong_top1, n))

    return {'lrap': lrap, 'aupr': aupr, 'n_total': n, 'n_wrong_top1': n_wrong_top1, 'results': results}


def run(proto_mode='class', test_file_match=None, temp=0.07, test_size=0.2, split_seed=SEED,
        run_tag=None, min_events=None):
    device = torch.device('cuda')
    print('  Device: {}'.format(device))

    if run_tag is None:
        run_tag = test_file_match if test_file_match else 'seed{}'.format(split_seed)
    full_run_tag = '{}_{}'.format(run_tag, proto_mode)

    ckpt_path = os.path.join(MODEL_DIR, 'camlds_classproto_matcher_{}.pt'.format(full_run_tag))
    if not os.path.exists(ckpt_path):
        print('  ERROR: model not found at {}'.format(ckpt_path))
        print('  Run train_camlds_class_prototype.py --proto-mode {} --run-tag {} first.'.format(proto_mode, run_tag))
        return

    print('\n  Loading model from {}'.format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device)
    all_tactics = ckpt['tactics']
    print('  Proto mode: {}'.format(ckpt.get('proto_mode', proto_mode)))
    print('  Best train loss: {:.4f}  Best epoch: {}'.format(ckpt.get('best_loss', 0), ckpt.get('best_epoch', '?')))
    print('  Tactics (prototypes) this model was trained on: {}'.format(', '.join(all_tactics)))

    trained_logit_scale = ckpt.get('logit_scale')
    trained_scale = trained_logit_scale.exp().item() if trained_logit_scale is not None else 1.0
    display_scale = math.exp(math.log(1.0 / temp)) if temp else trained_scale
    print('  Trained scale: {:.2f} (temp={:.4f})   Display scale: {:.2f}{}'.format(
        trained_scale, 1 / trained_scale, display_scale,
        ' (temp={} override)'.format(temp) if temp else ' (using trained value)'))

    log_proj = ProjectionNetwork().to(device)
    log_proj.load_state_dict(ckpt['log_proj'])
    log_proj.eval()

    print('\n  Loading RoBERTa sequence encoder (fine-tuned weights from checkpoint)...')
    tokenizer   = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    seq_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    seq_encoder.load_state_dict(ckpt['seq_encoder'])
    seq_encoder.eval()
    for p in seq_encoder.parameters():
        p.requires_grad = False

    ckpt_test_size       = ckpt.get('test_size', test_size)
    ckpt_split_seed      = ckpt.get('split_seed', split_seed)
    ckpt_test_file_match = test_file_match if test_file_match else ckpt.get('test_file_match')

    entries = load_sequences(min_events=min_events)
    if ckpt_test_file_match:
        print('\n  Loading test sequences (leave-out mode — test = files matching "{}")...'.format(ckpt_test_file_match))
        _, test_entries = leave_out_split(entries, ckpt_test_file_match)
    else:
        print('\n  Loading test sequences (same random split as training, test_size={} split_seed={})...'.format(
            ckpt_test_size, ckpt_split_seed))
        _, test_entries = random_split(entries, test_size=ckpt_test_size, seed=ckpt_split_seed)
    print('  Test sequences: {}'.format(len(test_entries)))

    print('\n  Encoding test sequences...')
    with torch.no_grad():
        seq_embs = [F.normalize(log_proj(encode_one(tokenizer, seq_encoder, e['sequence'], device).to(device)), dim=-1).cpu()
                    for e in test_entries]

    if proto_mode == 'class':
        print('\n  Loading {} LEARNED class prototypes from checkpoint (no template encoding)...'.format(len(all_tactics)))
        proto_embs = ckpt['class_prototypes'].cpu()   
    else:
        print('\n  Encoding {} tactic templates...'.format(len(all_tactics)))
        text_proj = ProjectionNetwork().to(device)
        text_proj.load_state_dict(ckpt['text_proj'])
        text_proj.eval()
        tmpl_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
        tmpl_encoder.load_state_dict(ckpt['tmpl_encoder'])
        tmpl_encoder.eval()
        templates = load_templates()
        with torch.no_grad():
            proto_embs = torch.stack([
                F.normalize(text_proj(encode_one(tokenizer, tmpl_encoder, templates[t], device).to(device)), dim=-1).cpu()
                for t in all_tactics
            ])

    results = score_table(test_entries, seq_embs, all_tactics, proto_embs, display_scale)
    out = print_table(results, all_tactics)

    results_path = os.path.join(RESULTS_DIR, 'camlds_classproto_test_results_{}.json'.format(full_run_tag))
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n  Results saved → {}'.format(results_path))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto-mode', type=str, choices=['class', 'template'], default='class')
    ap.add_argument('--test-file', type=str, default=None)
    ap.add_argument('--temp', type=float, default=0.07)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=SEED)
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--min-events', type=int, default=None)
    args = ap.parse_args()
    test_files = resolve_test_file_arg(args.test_file)

    run_tag = args.run_tag
    if run_tag is None and args.test_file:
        run_tag = args.test_file

    run(proto_mode=args.proto_mode, test_file_match=test_files, temp=args.temp, test_size=args.test_size,
        split_seed=args.split_seed, run_tag=run_tag, min_events=args.min_events)

