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
SCRIPTS_DIR = os.path.join(os.path.dirname(TECHNIQUE_DIR), 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from config import ROBERTA_MODEL
from train_camlds_matcher_technique_wide import (
    load_sequences, leave_out_split, random_split, stratified_split, load_templates,
    encode_one, ProjectionNetwork, WideModel, SEED, MAX_CHUNKS, WIDE_DIM_OUT,
)
from wide_feature_bridge import build_step_to_graph_path, fit_wide_bins_and_masks, wide_vector_for_step

CAM_LDS_DIR = '/csse/research/contructive-learning/CAM-LDS'
MODEL_DIR   = os.path.join(CAM_LDS_DIR, 'technique', 'checkpoints')
RESULTS_DIR = os.path.join(CAM_LDS_DIR, 'technique', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def encode_sequence_combined(tokenizer, seq_encoder, log_proj, wide_model, entry, step_to_path, bins, masks, device):
    with torch.no_grad():
        text_embed = log_proj(encode_one(tokenizer, seq_encoder, entry['sequence'], device).unsqueeze(0)).squeeze(0)
        wide_vec = wide_vector_for_step(entry['file'], step_to_path, bins, masks)
        wide_t = torch.tensor(wide_vec, dtype=torch.float32, device=device)
        wide_embed = wide_model(wide_t)
        return torch.cat([text_embed, wide_embed], dim=-1).cpu()


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
    col_true = 36
    print('\n  ── Test Results (scored against {} technique prototypes) ──'.format(len(all_techniques)))
    header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} {:<16} {:<16} {:<16} {:<8}'
    print(header_fmt.format('#', 'File', 'True techniques', '#1 (score)', '#2 (score)', '#3 (score)', 'Top3'))
    print('  ' + '-' * (4 + col_file + col_true + 16 + 16 + 16 + 8 + 14))
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
    print('  ─────────────────────────────────────')
    print('  LRAP (Label Ranking Average Precision)          : {:.1f}%'.format(lrap * 100))
    print('  AUPR (Area Under Precision-Recall Curve, macro) : {:.1f}%'.format(aupr * 100))
    print('  Wrong (true label not in top-3)                 : {}/{} samples'.format(n_wrong_top3, n))

    return {
        'lrap'          : lrap,
        'aupr'          : aupr,
        'n_total'       : n,
        'n_wrong_top3'  : n_wrong_top3,
        'results'       : results,
    }


def run(test_file_match=None, temp=0.07, test_size=0.2, split_seed=SEED, run_tag=None, min_events=None,
        template_dir=None, sequences_dir=None):
    device = torch.device('cuda')
    print('  Device: {}'.format(device))

    if run_tag is None:
        run_tag = test_file_match if test_file_match else 'seed{}'.format(split_seed)

    ckpt_path = os.path.join(MODEL_DIR, 'camlds_technique_wide_matcher_{}.pt'.format(run_tag))
    if not os.path.exists(ckpt_path):
        print('  ERROR: model not found at {}'.format(ckpt_path))
        print('  Run train_camlds_matcher_technique_wide.py --run-tag {} first.'.format(run_tag))
        return

    print('\n  Loading model from {}'.format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device)
    all_techniques = ckpt['techniques']
    print('  Best train loss: {:.4f}  Best epoch: {}'.format(ckpt.get('best_loss', 0), ckpt.get('best_epoch', '?')))
    print('  Techniques (prototypes) this model was trained on: {}'.format(', '.join(all_techniques)))

    trained_logit_scale = ckpt.get('logit_scale')
    trained_scale = trained_logit_scale.exp().item() if trained_logit_scale is not None else 1.0
    display_scale = math.exp(math.log(1.0 / temp)) if temp else trained_scale
    print('  Trained scale: {:.2f} (temp={:.4f})   Display scale: {:.2f}{}'.format(
        trained_scale, 1 / trained_scale, display_scale,
        ' (temp={} override)'.format(temp) if temp else ' (using trained value)'))

    proj_dims = ckpt['proj_dims']
    proj_dim = proj_dims[1]
    combined_dim = proj_dim + proj_dims[2]
    wide_dim_raw = proj_dims[3]

    log_proj   = ProjectionNetwork(out_dim=proj_dim).to(device)
    text_proj  = ProjectionNetwork(out_dim=combined_dim).to(device)
    wide_model = WideModel(wide_dim_raw, proj_dims[2]).to(device)
    log_proj.load_state_dict(ckpt['log_proj'])
    text_proj.load_state_dict(ckpt['text_proj'])
    wide_model.load_state_dict(ckpt['wide_model'])
    log_proj.eval()
    text_proj.eval()
    wide_model.eval()

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
    ckpt_test_file_match = test_file_match if test_file_match else ckpt.get('test_file_match')

    ckpt_sequences_dir = sequences_dir or ckpt.get('sequences_dir')
    entries = load_sequences(min_events=min_events, sequences_dir=ckpt_sequences_dir)
    ckpt_stratified = ckpt.get('stratified', False)
    if ckpt_test_file_match:
        print('\n  Loading test sequences (leave-out mode — test = files matching "{}")...'.format(ckpt_test_file_match))
        train_entries, test_entries = leave_out_split(entries, ckpt_test_file_match)
    elif ckpt_stratified:
        print('\n  Loading test sequences (same stratified split as training, test_size={} split_seed={})...'.format(
            ckpt_test_size, ckpt_split_seed))
        train_entries, test_entries = stratified_split(entries, test_size=ckpt_test_size, seed=ckpt_split_seed)
    else:
        print('\n  Loading test sequences (same random split as training, test_size={} split_seed={})...'.format(
            ckpt_test_size, ckpt_split_seed))
        train_entries, test_entries = random_split(entries, test_size=ckpt_test_size, seed=ckpt_split_seed)
    print('  Test sequences: {}'.format(len(test_entries)))

    print('\n  Refitting wide-feature bins/masks on this seed\'s training instances (same recipe as training)...')
    step_to_path = build_step_to_graph_path()
    train_steps = [e['file'] for e in train_entries]
    bins, masks, wide_dim_check = fit_wide_bins_and_masks(train_steps, step_to_path, seed=ckpt_split_seed)
    if wide_dim_check != wide_dim_raw:
        print('  WARNING: refit wide_dim={} does not match checkpoint wide_dim={} -- '
              'sequences_dir or split may not match training exactly.'.format(wide_dim_check, wide_dim_raw))

    print('\n  Encoding test sequences (text + wide, combined)...')
    seq_embs = [encode_sequence_combined(tokenizer, seq_encoder, log_proj, wide_model, e,
                                          step_to_path, bins, masks, device)
                for e in test_entries]
    seq_embs = [F.normalize(z, dim=-1) for z in seq_embs]

    ckpt_template_dir = template_dir or ckpt.get('template_dir')
    print('\n  Encoding {} technique templates... (dir={})'.format(len(all_techniques), ckpt_template_dir or 'technique/templates (default)'))
    templates = load_templates(template_dir=ckpt_template_dir)
    with torch.no_grad():
        tmpl_embs_by_technique = {
            t: F.normalize(text_proj(encode_one(tokenizer, tmpl_encoder, templates[t], device).to(device)), dim=-1).cpu()
            for t in all_techniques
        }

    results = score_table(test_entries, seq_embs, all_techniques, tmpl_embs_by_technique, display_scale)
    results = sorted(results, key=lambda r: r['file'])
    out = print_table(results, all_techniques)

    results_path = os.path.join(RESULTS_DIR, 'camlds_technique_wide_test_results_{}.json'.format(run_tag))
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n  Results saved → {}'.format(results_path))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-file', type=str, default=None)
    ap.add_argument('--temp', type=float, default=0.07)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=SEED)
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--template-dir', type=str, default=None)
    ap.add_argument('--sequences-dir', type=str, default=None)
    args = ap.parse_args()

    run_tag = args.run_tag
    if run_tag is None and args.test_file:
        run_tag = args.test_file

    run(test_file_match=args.test_file, temp=args.temp, test_size=args.test_size,
        split_seed=args.split_seed, run_tag=run_tag,
        template_dir=args.template_dir, sequences_dir=args.sequences_dir)
