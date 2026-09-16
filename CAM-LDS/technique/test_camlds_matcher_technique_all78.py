import os
import sys
import math
import json

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score

TECHNIQUE_DIR = os.path.dirname(os.path.abspath(__file__))
if TECHNIQUE_DIR not in sys.path:
    sys.path.insert(0, TECHNIQUE_DIR)
PROJECT_ROOT = '/csse/research/contructive-learning'
CAM_LDS_SCRIPTS = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'scripts')
if CAM_LDS_SCRIPTS not in sys.path:
    sys.path.insert(0, CAM_LDS_SCRIPTS)

from config import ROBERTA_MODEL
from train_camlds_matcher_technique_all78 import (
    load_sequences, stratified_split, random_split, load_templates,
    encode_one, ProjectionNetwork, WideModel, SEED, TECHNIQUE_IDS, MAX_CHUNKS, WIDE_DIM_OUT,
)
from wide_feature_bridge import build_step_to_graph_path, fit_node_edge_bins_and_masks, node_edge_vector_for_step

CAM_LDS_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS')
MODEL_DIR   = os.path.join(TECHNIQUE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(TECHNIQUE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def score_table(test_entries, seq_embs, all_techniques, tmpl_embs_by_technique, display_scale):
    zt_all = F.normalize(torch.stack([tmpl_embs_by_technique[t] for t in all_techniques]), dim=-1)

    results = []
    with torch.no_grad():
        for entry, z_s in zip(test_entries, seq_embs):
            logits = display_scale * (z_s @ zt_all.T)
            probs = [round(p, 6) for p in F.softmax(logits, dim=-1).tolist()]
            scores = dict(zip(all_techniques, probs))
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranked_techniques = [t for t, s in ranked]

            true_techniques = entry['techniques']
            true_ranks = {t: ranked_techniques.index(t) + 1 for t in true_techniques}

            results.append({
                'file': entry['file'], 'true_techniques': true_techniques,
                'true_ranks': true_ranks, 'scores': scores,
                'ranked': [{'technique': t, 'score': s} for t, s in ranked],
            })
    return results


def build_label_matrices(results, all_techniques):
    y_true = np.array([[1 if t in r['true_techniques'] else 0 for t in all_techniques] for r in results])
    y_score = np.array([[r['scores'][t] for t in all_techniques] for r in results])
    return y_true, y_score


def compute_lrap(results, all_techniques):
    y_true, y_score = build_label_matrices(results, all_techniques)
    return label_ranking_average_precision_score(y_true, y_score)


def compute_aupr(results, all_techniques):
    y_true, y_score = build_label_matrices(results, all_techniques)
    valid_cols = y_true.sum(axis=0) > 0
    if not valid_cols.any():
        return 0.0
    return average_precision_score(y_true[:, valid_cols], y_score[:, valid_cols], average='macro')


def print_table(results, all_techniques, top_n=3):
    col_file = 26
    col_true = 40
    print('\n  ── Test Results (scored against {} technique prototypes) ──'.format(len(all_techniques)))
    header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} {:<16} {:<16} {:<16} {:<8}'
    print(header_fmt.format('#', 'File', 'True techniques', '#1 (score)', '#2 (score)', '#3 (score)', 'Top3'))
    row_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} {:<16} {:<16} {:<16} {:<8}'
    n_wrong_top3 = 0
    for i, r in enumerate(results, 1):
        ranked = r['ranked']
        cols = []
        for j in range(top_n):
            if j < len(ranked):
                cols.append('{} {:.4f}'.format(ranked[j]['technique'], ranked[j]['score']))
            else:
                cols.append('')
        true_str = ','.join(r['true_techniques'])
        fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
        top3_techniques = {ranked[j]['technique'] for j in range(min(3, len(ranked)))}
        top3_wrong = bool(ranked) and not any(t in top3_techniques for t in r['true_techniques'])
        if top3_wrong:
            n_wrong_top3 += 1
        flag = 'WRONG' if top3_wrong else ''
        print(row_fmt.format(i, fname, true_str, cols[0], cols[1], cols[2], flag))

    n = len(results)
    lrap = compute_lrap(results, all_techniques) if n else 0.0
    aupr = compute_aupr(results, all_techniques) if n else 0.0

    print()
    print('  LRAP (Label Ranking Average Precision)          : {:.1f}%'.format(lrap * 100))
    print('  AUPR (Area Under Precision-Recall Curve, macro) : {:.1f}%'.format(aupr * 100))
    print('  Wrong (true label not in top-3)                 : {}/{} samples'.format(n_wrong_top3, n))
    return lrap, aupr, n_wrong_top3


def run(run_tag, test_file_match=None, split_seed=SEED, min_events=None, template_dir=None,
        stratified=True, test_size=0.2, sequences_dir=None):
    device = torch.device('cuda')
    ckpt_path = os.path.join(MODEL_DIR, 'camlds_wide_matcher_{}.pt'.format(run_tag))
    ckpt = torch.load(ckpt_path, map_location=device)
    all_techniques = ckpt['techniques']
    (emb_dim, proj_dim, wide_dim_out, wide_dim_raw) = ckpt['proj_dims']
    ckpt_split_seed = ckpt.get('split_seed', split_seed)

    entries = load_sequences(min_events=min_events, sequences_dir=sequences_dir)

    if test_file_match:
        test_entries = [e for e in entries if e['file'] in test_file_match]
        train_entries = [e for e in entries if e['file'] not in test_file_match]
    elif 'test_files' in ckpt:
        test_files = set(ckpt['test_files'])
        test_entries = [e for e in entries if e['file'] in test_files]
        train_entries = [e for e in entries if e['file'] not in test_files]
    elif stratified:
        train_entries, test_entries = stratified_split(entries, test_size=test_size, seed=ckpt_split_seed)
    else:
        train_entries, test_entries = random_split(entries, test_size=test_size, seed=ckpt_split_seed)

    print('  Test sequences: {}'.format(len(test_entries)))

    step_to_path = build_step_to_graph_path()
    train_steps = [e['file'] for e in train_entries]
    bins, masks, wide_dim_check = fit_node_edge_bins_and_masks(train_steps, step_to_path, seed=ckpt_split_seed)

    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    seq_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    seq_encoder.load_state_dict(ckpt['seq_encoder'])
    seq_encoder.eval()

    tmpl_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    tmpl_encoder.load_state_dict(ckpt['tmpl_encoder'])
    tmpl_encoder.eval()

    log_proj = ProjectionNetwork(emb_dim, proj_dim).to(device)
    log_proj.load_state_dict(ckpt['log_proj'])
    log_proj.eval()

    text_proj = ProjectionNetwork(emb_dim, proj_dim + wide_dim_out).to(device)
    text_proj.load_state_dict(ckpt['text_proj'])
    text_proj.eval()

    wide_model = WideModel(wide_dim_raw, wide_dim_out).to(device)
    wide_model.load_state_dict(ckpt['wide_model'])
    wide_model.eval()

    display_scale = ckpt['logit_scale'].exp().clamp(max=100)

    templates = load_templates(template_dir=template_dir or ckpt.get('template_dir'))

    with torch.no_grad():
        seq_embs = []
        for e in test_entries:
            text_embed = log_proj(encode_one(tokenizer, seq_encoder, e['sequence'], device).unsqueeze(0)).squeeze(0)
            wide_vec = node_edge_vector_for_step(e['file'], step_to_path, bins, masks)
            wide_t = torch.tensor(wide_vec, dtype=torch.float32, device=device)
            wide_embed = wide_model(wide_t)
            z = F.normalize(torch.cat([text_embed, wide_embed], dim=-1), dim=-1)
            seq_embs.append(z)

        tmpl_embs_by_technique = {}
        for t in all_techniques:
            emb = text_proj(encode_one(tokenizer, tmpl_encoder, templates[t], device).unsqueeze(0)).squeeze(0)
            tmpl_embs_by_technique[t] = emb

    results = score_table(test_entries, seq_embs, all_techniques, tmpl_embs_by_technique, display_scale)
    lrap, aupr, n_wrong_top3 = print_table(results, all_techniques)

    out = {
        'run_tag': run_tag, 'n_total': len(test_entries), 'n_techniques': len(all_techniques),
        'lrap': lrap, 'aupr': aupr, 'n_wrong_top3': n_wrong_top3,
        'results': results,
    }
    out_path = os.path.join(RESULTS_DIR, 'camlds_wide_test_results_{}.json'.format(run_tag))
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n  Results saved → {}'.format(out_path))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-tag', type=str, required=True)
    ap.add_argument('--split-seed', type=int, default=SEED)
    ap.add_argument('--test-size', type=float, default=0.2)
    args = ap.parse_args()
    run(args.run_tag, split_seed=args.split_seed, test_size=args.test_size)
