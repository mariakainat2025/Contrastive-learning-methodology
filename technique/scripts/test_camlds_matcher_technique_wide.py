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

TECHNIQUE_DIR = os.path.dirname(os.path.abspath(__file__))
if TECHNIQUE_DIR not in sys.path:
    sys.path.insert(0, TECHNIQUE_DIR)

from config import ROBERTA_MODEL
from train_camlds_matcher_technique_wide import (
    load_sequences, leave_out_split, leave_out_scenario_split, random_split, stratified_split,
    grouped_stratified_split, load_templates,
    encode_one, ProjectionNetwork, WideModel, SEED, MAX_CHUNKS, WIDE_DIM_OUT,
    TECHNIQUE_MERGE, ZOOMER_TECHNIQUES, STEP_TACTICS_PATH, base_technique, TEMPLATE_DIR,
)
from wide_feature_bridge import build_step_to_graph_path, fit_wide_bins_and_masks, wide_vector_for_step
from tactic_to_stage import STAGE_ORDER, TACTIC_TO_STAGE, tactics_to_stages
from stage_accuracy import (
    stage_top_k_accuracy, stage_accuracy_at_n_true,
    tactic_top_k_accuracy, tactic_accuracy_at_n_true,
    technique_top_k_accuracy, technique_accuracy_at_n_true,
    is_correct, classify,
)

CAM_LDS_DIR = '/csse/research/contructive-learning/CAM-LDS'
MODEL_DIR   = os.path.join(CAM_LDS_DIR, 'technique', 'checkpoints')
RESULTS_DIR = os.path.join(CAM_LDS_DIR, 'technique', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

ZOOMER_SCOPE_MIN4 = [
    'T1003', 'T1018', 'T1021', 'T1033', 'T1036', 'T1053', 'T1056', 'T1057',
    'T1059', 'T1070', 'T1072', 'T1078', 'T1082', 'T1083', 'T1105', 'T1110',
    'T1115', 'T1176', 'T1204', 'T1219', 'T1222', 'T1543', 'T1546', 'T1548',
    'T1564', 'T1574',
]


def load_technique_to_tactics(restrict_to_scope=True):
    with open(STEP_TACTICS_PATH) as f:
        report = json.load(f)
    technique_to_tactics = {}
    for row in report:
        for occ in row['occurrences']:
            t = TECHNIQUE_MERGE.get(occ['technique'], base_technique(occ['technique']))
            if (not restrict_to_scope) or t in ZOOMER_TECHNIQUES:
                technique_to_tactics.setdefault(t, set()).update(occ['tactics'])
    return {t: sorted(tactics) for t, tactics in technique_to_tactics.items()}


def expand_ranked(ranked_tuples, key_to_labels):
    seen = set()
    out = []
    for label, score in ranked_tuples:
        for target in key_to_labels.get(label, []):
            if target not in seen:
                seen.add(target)
                out.append((target, score))
    return out


def build_tactic_and_stage_rows(results, technique_to_tactics, technique_to_tactics_all):
    tactic_to_stage_list = {t: [s] for t, s in TACTIC_TO_STAGE.items()}
    technique_rows, tactic_rows, stage_rows = [], [], []
    for r in results:
        technique_ranked = [(d['technique'], d['score']) for d in r['ranked']]
        technique_rows.append({'true_techniques': set(r['true_techniques']), 'ranked': technique_ranked})

        true_tactics = set()
        for t in r['true_techniques']:
            true_tactics.update(technique_to_tactics_all.get(t, []))
        tactic_ranked = expand_ranked(technique_ranked, technique_to_tactics)
        tactic_rows.append({'true_tactics': true_tactics, 'ranked': tactic_ranked})

        true_stages = set(tactics_to_stages(true_tactics))
        stage_ranked = expand_ranked(tactic_ranked, tactic_to_stage_list)
        stage_rows.append({'true_stages': true_stages, 'ranked': stage_ranked})
    return technique_rows, tactic_rows, stage_rows


def step_sort_key(step_name):
    m = re.search(r'(\d+)$', step_name)
    return (int(m.group(1)) if m else 0, step_name)


STAGE_ABBREV = {
    'Initial Compromise':      'IC',
    'Establish Foothold':      'EF',
    'Escalate Privilege':      'EP',
    'Internal Reconnaissance': 'IR',
    'Move Laterally':          'ML',
    'Maintain Persistence':    'MP',
    'Complete Mission':        'CM',
}


def sid(stage):
    return STAGE_ABBREV.get(stage, stage)


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


def tid(tactic):
    return TACTIC_IDS.get(tactic, tactic)


def print_tactic_results_table(tactic_rows, files, n_tactics, top_n=3):
    col_file = 28
    col_true = 26
    col_score = 22
    print()
    print()
    print()
    print('  ── Tactic Results (derived from technique predictions, {} reachable tactics) ──'.format(n_tactics))
    header_fmt = ('  {:<4}   {:<' + str(col_file) + '}   {:<' + str(col_true) + '}   ' +
                  '   '.join(['{:<' + str(col_score) + '}'] * top_n) + '   {:<6}')
    print(header_fmt.format('#', 'File', 'True tactics', *['#{} (score)'.format(j + 1) for j in range(top_n)], 'Top{}'.format(top_n)))
    print('  ' + '-' * (4 + col_file + col_true + col_score * top_n + 6 + 3 * (top_n + 3)))
    n_wrong = 0
    for i, (r, fname_full) in enumerate(zip(tactic_rows, files), 1):
        ranked = r['ranked']
        top_tactics = {t for (t, _) in ranked[:top_n]}
        cols = []
        for j in range(top_n):
            cols.append('{} {:.4f}'.format(tid(ranked[j][0]), ranked[j][1]) if j < len(ranked) else '')
        true_str = ','.join(tid(t) for t in sorted(r['true_tactics']))
        if len(true_str) > col_true:
            true_str = true_str[:col_true - 3] + '...'
        fname = fname_full if len(fname_full) <= col_file else fname_full[:col_file - 3] + '...'
        verdict = classify(r['true_tactics'], top_tactics, top_n)
        if verdict == 'wrong':
            n_wrong += 1
        print(header_fmt.format(i, fname, true_str, *cols, verdict.upper() if verdict and verdict != 'correct' else ''))
    return n_wrong


def print_stage_results_table(stage_rows, files, max_stages=3):
    col_file = 28
    col_true = 24
    col_score = 18
    col_found = 8
    print()
    print()
    print('  ── Stage Results (KnowHow 7-stage lifecycle, derived from technique predictions) ──')
    print('  IC=Initial Compromise  EF=Establish Foothold  EP=Escalate Privilege  IR=Internal Reconnaissance '
          'ML=Move Laterally  MP=Maintain Persistence  CM=Complete Mission')
    header_fmt = ('  {:<4}   {:<' + str(col_file) + '}   {:<' + str(col_true) + '}   ' +
                  '   '.join(['{:<' + str(col_score) + '}'] * max_stages) + '   {:<' + str(col_found) + '}   {:<6}   {}')
    print(header_fmt.format('#', 'File', 'True stage(s)', *['#{}'.format(j + 1) for j in range(max_stages)], 'Found', 'Wrong', 'Missing'))
    print('  ' + '-' * (4 + col_file + col_true + col_score * max_stages + col_found + 6 + 3 * (max_stages + 5)))
    n_wrong = 0
    for i, (r, fname_full) in enumerate(zip(stage_rows, files), 1):
        kept = r['ranked'][:max_stages]
        cols = []
        for j in range(max_stages):
            cols.append('{} ({:.4f})'.format(sid(kept[j][0]), kept[j][1]) if j < len(kept) else '')
        true_str = ', '.join(sid(s) for s in sorted(r['true_stages'], key=STAGE_ORDER.index)) or '(none)'
        kept_names = {s for (s, _) in kept}
        n_found = len(kept_names & r['true_stages'])
        n_true = len(r['true_stages'])
        found_str = '{}/{}'.format(n_found, n_true) if n_true else '-'
        missing = sorted(r['true_stages'] - kept_names, key=STAGE_ORDER.index)
        missing_str = ','.join(sid(s) for s in missing)
        verdict = classify(r['true_stages'], kept_names, max_stages)
        if verdict == 'wrong':
            n_wrong += 1
        fname = fname_full if len(fname_full) <= col_file else fname_full[:col_file - 3] + '...'
        print(header_fmt.format(i, fname, true_str, *cols, found_str, verdict.upper() if verdict and verdict != 'correct' else '', missing_str))
    return n_wrong


def encode_sequence_combined(tokenizer, seq_encoder, log_proj, wide_model, entry, step_to_path, bins, masks, device):
    with torch.no_grad():
        text_embed = log_proj(encode_one(tokenizer, seq_encoder, entry['sequence'], device).unsqueeze(0)).squeeze(0)
        if wide_model is None:
            return text_embed.cpu()
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

            true_techniques = entry['techniques_all']
            true_ranks = {t: ranked_techniques.index(t) + 1 for t in true_techniques if t in ranked_techniques}

            results.append({
                'file': entry['file'], 'true_techniques': true_techniques,
                'true_ranks': true_ranks, 'scores': scores,
                'ranked': [{'technique': t, 'score': s} for t, s in ranked],
                'n_events': entry.get('n_events'),
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


def topn_any_all_match(rows, true_key, top_n=3):
    if not rows:
        return 0.0, 0.0
    n_any, n_all = 0, 0
    for r in rows:
        true_set = set(r[true_key])
        top_set = {lbl for lbl, _ in r['ranked'][:top_n]}
        if true_set & top_set:
            n_any += 1
        if true_set <= top_set:
            n_all += 1
    return n_any / len(rows), n_all / len(rows)


def row_is_seen(true_techniques, train_tech_counts):
    return all(train_tech_counts.get(t, 0) >= 1 for t in true_techniques)


def filter_rows_by_mask(rows, mask):
    return [r for r, m in zip(rows, mask) if m]


def compute_metrics_block(rows, true_key, top_k_fn, at_n_fn):
    if not rows:
        return 0.0, 0.0, 0.0, 0.0
    top1 = top_k_fn(rows)[1]
    at_n = at_n_fn(rows)
    any_match, all_match = topn_any_all_match(rows, true_key, 3)
    return top1, at_n, any_match, all_match


def build_label_matrices_generic(rows, true_key, all_labels):
    y_true = np.array([[1 if lbl in r[true_key] else 0 for lbl in all_labels] for r in rows])
    y_score = np.array([[dict(r['ranked']).get(lbl, 0.0) for lbl in all_labels] for r in rows])
    return y_true, y_score


def compute_lrap_generic(rows, true_key, all_labels):
    if not rows:
        return 0.0
    y_true, y_score = build_label_matrices_generic(rows, true_key, all_labels)
    return label_ranking_average_precision_score(y_true, y_score)


def compute_aupr_generic(rows, true_key, all_labels):
    if not rows:
        return 0.0
    y_true, y_score = build_label_matrices_generic(rows, true_key, all_labels)
    valid_cols = y_true.sum(axis=0) > 0
    if not valid_cols.any():
        return 0.0
    return average_precision_score(y_true[:, valid_cols], y_score[:, valid_cols], average='macro')


def print_table(results, all_techniques, top_n=3, train_tech_counts=None):
    col_file = 28
    col_true = 38
    col_score = 18
    col_nsamp = 10
    col_nevt = 8
    print('\n\n  ── Test Results (scored against {} technique prototypes) ──'.format(len(all_techniques)))
    header_fmt = ('  {:<4}   {:<' + str(col_file) + '}   {:<' + str(col_true) + '}   {:<' + str(col_nsamp) +
                  '}   {:<' + str(col_nevt) + '}   {:<' + str(col_score) + '}   {:<' + str(col_score) +
                  '}   {:<' + str(col_score) + '}   {:<8}')
    print(header_fmt.format('#', 'File', 'True techniques', '#train', '#events', '#1 (score)', '#2 (score)', '#3 (score)', 'Top3'))
    print('  ' + '-' * (4 + col_file + col_true + col_nsamp + col_nevt + col_score * 3 + 8 + 4 * 7))
    row_fmt = header_fmt
    n_wrong_top3 = 0
    n_any_top3 = 0
    n_all_top3 = 0
    for i, r in enumerate(results, 1):
        ranked = r['ranked']
        cols = []
        for j in range(top_n):
            if j < len(ranked):
                cols.append('{} {:.4f}'.format(ranked[j]['technique'], ranked[j]['score']))
            else:
                cols.append('')
        true_str = ','.join(r['true_techniques'])
        if train_tech_counts is not None:
            nsamp_str = ','.join(str(train_tech_counts.get(t, 0)) for t in r['true_techniques'])
        else:
            nsamp_str = ''
        nevt_str = str(r['n_events']) if r.get('n_events') is not None else ''
        fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
        true_set = set(r['true_techniques'])
        top_set = {ranked[j]['technique'] for j in range(min(top_n, len(ranked)))}
        if true_set & top_set:
            n_any_top3 += 1
        if true_set <= top_set:
            n_all_top3 += 1
        verdict = classify(true_set, top_set, top_n)
        if verdict == 'wrong':
            n_wrong_top3 += 1
        flag = verdict.upper() if verdict and verdict != 'correct' else ''
        print(row_fmt.format(i, fname, true_str, nsamp_str, nevt_str, cols[0], cols[1], cols[2], flag))

    n = len(results)
    lrap = compute_lrap(results, all_techniques) if n else 0.0
    aupr = compute_aupr(results, all_techniques) if n else 0.0
    top3_any_pct = (n_any_top3 / n * 100) if n else 0.0
    top3_all_pct = (n_all_top3 / n * 100) if n else 0.0

    print()
    print('  ─────────────────────────────────────')
    print('  LRAP (Label Ranking Average Precision)          : {:.1f}%'.format(lrap * 100))
    print('  AUPR (Area Under Precision-Recall Curve, macro) : {:.1f}%'.format(aupr * 100))
    print('  Wrong (true label not in top-3)                 : {}/{} samples'.format(n_wrong_top3, n))
    print('  Top-3 ANY-match (>=1 true label in top-3)       : {:.1f}%  ({}/{})'.format(top3_any_pct, n_any_top3, n))
    print('  Top-3 ALL-match (every true label in top-3)     : {:.1f}%  ({}/{})'.format(top3_all_pct, n_all_top3, n))

    return {
        'lrap'          : lrap,
        'top3_any_match': top3_any_pct / 100,
        'top3_all_match': top3_all_pct / 100,
        'aupr'          : aupr,
        'n_total'       : n,
        'n_wrong_top3'  : n_wrong_top3,
        'results'       : results,
    }


def run(test_file_match=None, test_scenario=None, temp=0.07, test_size=0.2, split_seed=SEED, run_tag=None,
        template_dir=None, sequences_dir=None, run_filter=None, zoomer_scope=False,
        strict_native=True, candidate_cap=None):
    device = torch.device('cuda')
    print('  Device: {}'.format(device))

    if run_tag is None:
        if test_scenario:
            run_tag = 'scenario{}'.format(test_scenario)
        elif test_file_match:
            run_tag = test_file_match
        else:
            run_tag = 'seed{}'.format(split_seed)

    ckpt_path = os.path.join(MODEL_DIR, 'camlds_technique_wide_matcher_{}.pt'.format(run_tag))
    if not os.path.exists(ckpt_path):
        print('  ERROR: model not found at {}'.format(ckpt_path))
        print('  Run train_camlds_matcher_technique_wide.py --run-tag {} first.'.format(run_tag))
        return

    print('\n  Loading model from {}'.format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device)
    trained_techniques = ckpt['techniques']
    ckpt_zoomer_scope = ckpt.get('zoomer_scope', False)
    print('  Best train loss: {:.4f}  Best epoch: {}'.format(ckpt.get('best_loss', 0), ckpt.get('best_epoch', '?')))
    print('  Techniques this model was fine-tuned on ({}): {}{}'.format(
        len(trained_techniques), ', '.join(trained_techniques),
        '  [trained with --zoomer-scope]' if ckpt_zoomer_scope else ''))

    trained_logit_scale = ckpt.get('logit_scale')
    trained_scale = trained_logit_scale.exp().item() if trained_logit_scale is not None else 1.0
    display_scale = math.exp(math.log(1.0 / temp)) if temp else trained_scale
    print('  Trained scale: {:.2f} (temp={:.4f})   Display scale: {:.2f}{}'.format(
        trained_scale, 1 / trained_scale, display_scale,
        ' (temp={} override)'.format(temp) if temp else ' (using trained value)'))

    ckpt_template_dir = template_dir or ckpt.get('template_dir')
    templates = load_templates(template_dir=ckpt_template_dir)
    print('  Template library available ({} techniques have template text, dir={})'.format(
        len(templates), ckpt_template_dir or TEMPLATE_DIR))

    proj_dims = ckpt['proj_dims']
    proj_dim = proj_dims[1]
    combined_dim = proj_dim + proj_dims[2]
    wide_dim_raw = proj_dims[3]
    ckpt_use_wide = ckpt.get('use_wide', True)

    log_proj   = ProjectionNetwork(out_dim=proj_dim).to(device)
    text_proj  = ProjectionNetwork(out_dim=combined_dim).to(device)
    log_proj.load_state_dict(ckpt['log_proj'])
    text_proj.load_state_dict(ckpt['text_proj'])
    log_proj.eval()
    text_proj.eval()
    if ckpt_use_wide:
        wide_model = WideModel(wide_dim_raw, proj_dims[2]).to(device)
        wide_model.load_state_dict(ckpt['wide_model'])
        wide_model.eval()
    else:
        wide_model = None
        print('  --no-wide checkpoint: text-only embedding, no wide features used.')

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

    ckpt_sequences_dir = sequences_dir or ckpt.get('sequences_dir')
    entries = load_sequences(sequences_dir=ckpt_sequences_dir)

    native_techniques = sorted({t for e in entries for t in e['techniques']})

    ckpt_stratified = ckpt.get('stratified', False)
    ckpt_grouped_split = ckpt.get('grouped_split', False)
    if ckpt_test_scenario:
        print('\n  Loading test sequences (leave-one-scenario-out — test = ALL steps of scenario {})...'.format(
            ckpt_test_scenario))
        train_entries, test_entries = leave_out_scenario_split(entries, ckpt_test_scenario, seed=ckpt_split_seed)
    elif ckpt_test_file_match:
        print('\n  Loading test sequences (leave-out mode — test = files matching "{}")...'.format(ckpt_test_file_match))
        train_entries, test_entries = leave_out_split(entries, ckpt_test_file_match)
    elif ckpt_stratified and ckpt_grouped_split:
        print('\n  Loading test sequences (same grouped stratified split as training, test_size={} '
              'split_seed={})...'.format(ckpt_test_size, ckpt_split_seed))
        train_entries, test_entries = grouped_stratified_split(entries, test_size=ckpt_test_size, seed=ckpt_split_seed)
    elif ckpt_stratified:
        print('\n  Loading test sequences (same stratified split as training, test_size={} split_seed={})...'.format(
            ckpt_test_size, ckpt_split_seed))
        train_entries, test_entries = stratified_split(entries, test_size=ckpt_test_size, seed=ckpt_split_seed)
    else:
        print('\n  Loading test sequences (same random split as training, test_size={} split_seed={})...'.format(
            ckpt_test_size, ckpt_split_seed))
        train_entries, test_entries = random_split(entries, test_size=ckpt_test_size, seed=ckpt_split_seed)
    if run_filter:
        before_filter = len(test_entries)
        test_entries = [e for e in test_entries if run_filter in e['file']]
        print('  Run filter "{}": kept {}/{} test sequences'.format(run_filter, len(test_entries), before_filter))
    print('  Test sequences: {}'.format(len(test_entries)))
    if ckpt_zoomer_scope:
        min4_set = set(ZOOMER_SCOPE_MIN4)
        for e in train_entries:
            e['techniques'] = [t for t in e['techniques'] if t in min4_set]
    train_techs_present = {t for e in train_entries for t in e['techniques']}
    test_techs_present = {t for e in test_entries for t in e['techniques_all']}
    train_tech_counts = {}
    for e in train_entries:
        for t in e['techniques']:
            train_tech_counts[t] = train_tech_counts.get(t, 0) + 1

    rare_test_only = [] if strict_native else sorted(test_techs_present - set(native_techniques))
    if rare_test_only:
        templates.update(load_templates(template_dir=ckpt_template_dir, extra_techniques=rare_test_only))
    if strict_native:
        print('  Candidate set capped at the native {}-technique scope, no rare test-only extension '
              '(pass --allow-extra-techniques to include rare test-only techniques via zero-shot template match).'.format(
                  len(native_techniques)))
        all_techniques = list(native_techniques)
    else:
        all_techniques = sorted(set(native_techniques) | test_techs_present)
    missing_templates = sorted(set(all_techniques) - set(templates.keys()))
    if missing_templates:
        print('  WARNING: {} technique(s) have no template text at all, dropping from candidate '
              'set (truly unscoreable): {}'.format(len(missing_templates), missing_templates))
        all_techniques = [t for t in all_techniques if t not in missing_templates]
        rare_test_only = [t for t in rare_test_only if t not in missing_templates]
    print('  Native candidate set ({} techniques): {}'.format(len(native_techniques), ', '.join(native_techniques)))
    if rare_test_only:
        print('  + test-only, below training-sufficiency bar but template available (zero-shot): '
              '{} -> {}'.format(len(rare_test_only), rare_test_only))
    if candidate_cap is not None:
        before_cap = len(all_techniques)
        all_techniques = [t for t in all_techniques if t in set(candidate_cap)]
        print('  --candidate-cap: candidate set restricted from {} to {} techniques: {}'.format(
            before_cap, len(all_techniques), all_techniques))
    print('  Final candidate set ({} techniques)'.format(len(all_techniques)))
    unseen_at_train = sorted(set(all_techniques) - set(trained_techniques))
    if unseen_at_train:
        print('  Of those, NOT fine-tuned on during training (zero-shot candidates): {} -> {}'.format(
            len(unseen_at_train), unseen_at_train))
    if zoomer_scope:
        before = all_techniques
        all_techniques = [t for t in all_techniques if t in ZOOMER_SCOPE_MIN4]
        dropped = sorted(set(before) - set(all_techniques))
        print('  --zoomer-scope: narrowed candidate set from {} to {} techniques (dropped: {})'.format(
            len(before), len(all_techniques), ', '.join(dropped)))

    techs_zero_train = sorted(set(all_techniques) - train_techs_present)
    techs_test_not_train = sorted(test_techs_present - train_techs_present)
    techs_test_out_of_scope = sorted(test_techs_present - set(all_techniques))
    print()
    print('  ── Coverage summary (technique / tactic / stage) ──')
    print('  Techniques: total={}  in-test={}  zero-training(removed)={}{}  in-test-not-in-training={}{}'.format(
        len(all_techniques), len(test_techs_present),
        len(techs_zero_train), ' ' + str(techs_zero_train) if techs_zero_train else '',
        len(techs_test_not_train), ' ' + str(techs_test_not_train) if techs_test_not_train else ''))
    if techs_test_out_of_scope:
        print('  Of those, entirely outside the {}-technique candidate set (no template, ever): {} -> {}'.format(
            len(all_techniques), len(techs_test_out_of_scope), techs_test_out_of_scope))

    technique_to_tactics = load_technique_to_tactics()
    technique_to_tactics_all = load_technique_to_tactics(restrict_to_scope=False)
    all_reachable_tactics = sorted({tac for tacs in technique_to_tactics.values() for tac in tacs})
    train_tactics_present = {tac for t in train_techs_present for tac in technique_to_tactics.get(t, [])}
    test_tactics_present = {tac for t in test_techs_present for tac in technique_to_tactics_all.get(t, [])}
    tactics_test_not_train = sorted(test_tactics_present - train_tactics_present)
    print('  Tactics:    total={}  in-test={}  in-test-not-in-training={}{}'.format(
        len(all_reachable_tactics), len(test_tactics_present),
        len(tactics_test_not_train), ' ' + str(tactics_test_not_train) if tactics_test_not_train else ''))

    all_reachable_stages = tactics_to_stages(all_reachable_tactics)
    train_stages_present = set(tactics_to_stages(train_tactics_present))
    test_stages_present = set(tactics_to_stages(test_tactics_present))
    stages_test_not_train = sorted(test_stages_present - train_stages_present, key=STAGE_ORDER.index)
    print('  Stages:     total={}  in-test={}  in-test-not-in-training={}{}'.format(
        len(all_reachable_stages), len(test_stages_present),
        len(stages_test_not_train), ' ' + str(stages_test_not_train) if stages_test_not_train else ''))

    if ckpt_use_wide:
        print('\n  Refitting wide-feature bins/masks on this seed\'s training instances (same recipe as training)...')
        step_to_path = build_step_to_graph_path()
        train_steps = [e['file'] for e in train_entries if e['techniques']]
        bins, masks, wide_dim_check = fit_wide_bins_and_masks(train_steps, step_to_path, seed=ckpt_split_seed)
        if wide_dim_check != wide_dim_raw:
            print('  WARNING: refit wide_dim={} does not match checkpoint wide_dim={} -- '
                  'sequences_dir or split may not match training exactly.'.format(wide_dim_check, wide_dim_raw))
    else:
        step_to_path, bins, masks = None, None, None

    print('\n  Encoding test sequences (text{}, combined)...'.format(' + wide' if ckpt_use_wide else '-only'))
    seq_embs = [encode_sequence_combined(tokenizer, seq_encoder, log_proj, wide_model, e,
                                          step_to_path, bins, masks, device)
                for e in test_entries]
    seq_embs = [F.normalize(z, dim=-1) for z in seq_embs]

    print('\n  Encoding {} technique templates...'.format(len(all_techniques)))
    with torch.no_grad():
        tmpl_embs_by_technique = {
            t: F.normalize(text_proj(encode_one(tokenizer, tmpl_encoder, templates[t], device).to(device)), dim=-1).cpu()
            for t in all_techniques
        }

    results = score_table(test_entries, seq_embs, all_techniques, tmpl_embs_by_technique, display_scale)
    results = sorted(results, key=lambda r: step_sort_key(r['file']))
    out = print_table(results, all_techniques, train_tech_counts=train_tech_counts)

    technique_rows, tactic_rows, stage_rows = build_tactic_and_stage_rows(results, technique_to_tactics, technique_to_tactics_all)
    files = [r['file'] for r in results]

    n_reachable_tactics = len(all_reachable_tactics)
    print_tactic_results_table(tactic_rows, files, n_reachable_tactics)
    print_stage_results_table(stage_rows, files)

    tac_lrap = compute_lrap_generic(tactic_rows, 'true_tactics', all_reachable_tactics)
    tac_aupr = compute_aupr_generic(tactic_rows, 'true_tactics', all_reachable_tactics)
    stg_lrap = compute_lrap_generic(stage_rows, 'true_stages', all_reachable_stages)
    stg_aupr = compute_aupr_generic(stage_rows, 'true_stages', all_reachable_stages)

    tech_acc_k = technique_top_k_accuracy(technique_rows)
    tech_acc_top1, tech_acc_top3 = tech_acc_k[1], tech_acc_k[3]
    tech_acc_at_n = technique_accuracy_at_n_true(technique_rows)

    tac_acc_k = tactic_top_k_accuracy(tactic_rows)
    tac_acc_top1, tac_acc_top3 = tac_acc_k[1], tac_acc_k[3]
    tac_acc_at_n = tactic_accuracy_at_n_true(tactic_rows)

    stg_acc_k = stage_top_k_accuracy(stage_rows)
    stg_acc_top1, stg_acc_top3 = stg_acc_k[1], stg_acc_k[3]
    stg_acc_at_n = stage_accuracy_at_n_true(stage_rows)

    tech_top3_any, tech_top3_all = topn_any_all_match(technique_rows, 'true_techniques', 3)
    tac_top3_any, tac_top3_all = topn_any_all_match(tactic_rows, 'true_tactics', 3)
    stg_top3_any, stg_top3_all = topn_any_all_match(stage_rows, 'true_stages', 3)

    print()
    print()
    print('  ── Accuracy derived from technique-level predictions (technique -> tactic -> stage) ──')
    print('  Technique Accuracy     top-1: {:.1f}%   at-n: {:.1f}%   top-3-any: {:.1f}%   top-3-all: {:.1f}%'.format(
        tech_acc_top1 * 100, tech_acc_at_n * 100, tech_top3_any * 100, tech_top3_all * 100))
    print('  Tactic Accuracy        top-1: {:.1f}%   at-n: {:.1f}%   top-3-any: {:.1f}%   top-3-all: {:.1f}%'.format(
        tac_acc_top1 * 100, tac_acc_at_n * 100, tac_top3_any * 100, tac_top3_all * 100))
    print('  Stage Accuracy         top-1: {:.1f}%   at-n: {:.1f}%   top-3-any: {:.1f}%   top-3-all: {:.1f}%'.format(
        stg_acc_top1 * 100, stg_acc_at_n * 100, stg_top3_any * 100, stg_top3_all * 100))
    print('  (top-3-any = >=1 true label in top-3 counts as correct; top-3-all = every true label must be in top-3)')
    print()

    seen_mask = [row_is_seen(r['true_techniques'], train_tech_counts) for r in results]
    n_seen = sum(seen_mask)
    seen_technique_rows = filter_rows_by_mask(technique_rows, seen_mask)
    seen_tactic_rows = filter_rows_by_mask(tactic_rows, seen_mask)
    seen_stage_rows = filter_rows_by_mask(stage_rows, seen_mask)

    seen_tech = compute_metrics_block(seen_technique_rows, 'true_techniques', technique_top_k_accuracy, technique_accuracy_at_n_true)
    seen_tac = compute_metrics_block(seen_tactic_rows, 'true_tactics', tactic_top_k_accuracy, tactic_accuracy_at_n_true)
    seen_stg = compute_metrics_block(seen_stage_rows, 'true_stages', stage_top_k_accuracy, stage_accuracy_at_n_true)

    overall_tech = (tech_acc_top1, tech_acc_at_n, tech_top3_any, tech_top3_all)
    overall_tac = (tac_acc_top1, tac_acc_at_n, tac_top3_any, tac_top3_all)
    overall_stg = (stg_acc_top1, stg_acc_at_n, stg_top3_any, stg_top3_all)

    print('  ── Seen (trained on every true label) vs Overall ──')
    print('  "Seen" = every true label for that step has >=1 training example (n={}/{} steps).'.format(
        n_seen, len(results)))
    print()
    row_fmt = '  {:<22}{:>9}{:>9}{:>13}{:>13}'
    print(row_fmt.format('', 'top-1', 'at-n', 'top-3-any', 'top-3-all'))
    print('  Technique')
    print(row_fmt.format('    Seen ({})'.format(n_seen), *['{:.1f}%'.format(v * 100) for v in seen_tech]))
    print(row_fmt.format('    Overall ({})'.format(len(results)), *['{:.1f}%'.format(v * 100) for v in overall_tech]))
    print('  Tactic')
    print(row_fmt.format('    Seen ({})'.format(n_seen), *['{:.1f}%'.format(v * 100) for v in seen_tac]))
    print(row_fmt.format('    Overall ({})'.format(len(results)), *['{:.1f}%'.format(v * 100) for v in overall_tac]))
    print('  Stage')
    print(row_fmt.format('    Seen ({})'.format(n_seen), *['{:.1f}%'.format(v * 100) for v in seen_stg]))
    print(row_fmt.format('    Overall ({})'.format(len(results)), *['{:.1f}%'.format(v * 100) for v in overall_stg]))
    print()

    seen_results = filter_rows_by_mask(results, seen_mask)
    seen_lrap_tech = compute_lrap(seen_results, all_techniques) if seen_results else 0.0
    seen_aupr_tech = compute_aupr(seen_results, all_techniques) if seen_results else 0.0
    seen_lrap_tac = compute_lrap_generic(seen_tactic_rows, 'true_tactics', all_reachable_tactics)
    seen_aupr_tac = compute_aupr_generic(seen_tactic_rows, 'true_tactics', all_reachable_tactics)
    seen_lrap_stg = compute_lrap_generic(seen_stage_rows, 'true_stages', all_reachable_stages)
    seen_aupr_stg = compute_aupr_generic(seen_stage_rows, 'true_stages', all_reachable_stages)

    lrap_fmt = '  {:<22}{:>9}{:>9}'
    print('  ── LRAP / AUPR: Seen vs Overall ──')
    print(lrap_fmt.format('', 'LRAP', 'AUPR'))
    print('  Technique')
    print(lrap_fmt.format('    Seen ({})'.format(n_seen), '{:.1f}%'.format(seen_lrap_tech * 100), '{:.1f}%'.format(seen_aupr_tech * 100)))
    print(lrap_fmt.format('    Overall ({})'.format(len(results)), '{:.1f}%'.format(out['lrap'] * 100), '{:.1f}%'.format(out['aupr'] * 100)))
    print('  Tactic')
    print(lrap_fmt.format('    Seen ({})'.format(n_seen), '{:.1f}%'.format(seen_lrap_tac * 100), '{:.1f}%'.format(seen_aupr_tac * 100)))
    print(lrap_fmt.format('    Overall ({})'.format(len(results)), '{:.1f}%'.format(tac_lrap * 100), '{:.1f}%'.format(tac_aupr * 100)))
    print('  Stage')
    print(lrap_fmt.format('    Seen ({})'.format(n_seen), '{:.1f}%'.format(seen_lrap_stg * 100), '{:.1f}%'.format(seen_aupr_stg * 100)))
    print(lrap_fmt.format('    Overall ({})'.format(len(results)), '{:.1f}%'.format(stg_lrap * 100), '{:.1f}%'.format(stg_aupr * 100)))

    print()
    print('  ── Technique / tactic coverage detail ──')
    print('  Techniques: total={}  in-training={}  in-test={}'.format(
        len(all_techniques), len(train_techs_present), len(test_techs_present)))
    if techs_test_not_train:
        print('  Techniques in TEST but NOT in TRAINING (no prototype signal):')
        for t in techs_test_not_train:
            step_names = sorted(e['file'] for e in test_entries if t in e['techniques_all'])
            print('    {} -> steps: {}'.format(t, ', '.join(step_names)))
    else:
        print('  Techniques in TEST but NOT in TRAINING: none')
    print('  Tactics:    total={}  in-training={}  in-test={}'.format(
        len(all_reachable_tactics), len(train_tactics_present), len(test_tactics_present)))

    out['tech_acc_top1']  = tech_acc_top1
    out['tech_acc_top3']  = tech_acc_top3
    out['tech_acc_at_n']  = tech_acc_at_n
    out['tac_acc_top1']   = tac_acc_top1
    out['tac_acc_top3']   = tac_acc_top3
    out['tac_acc_at_n']   = tac_acc_at_n
    out['stage_acc_top1'] = stg_acc_top1
    out['stage_acc_top3'] = stg_acc_top3
    out['stage_acc_at_n'] = stg_acc_at_n
    out['tac_lrap']   = tac_lrap
    out['tac_aupr']   = tac_aupr
    out['stage_lrap'] = stg_lrap
    out['stage_aupr'] = stg_aupr
    out['tech_top3_any']  = tech_top3_any
    out['tech_top3_all']  = tech_top3_all
    out['tac_top3_any']   = tac_top3_any
    out['tac_top3_all']   = tac_top3_all
    out['stage_top3_any'] = stg_top3_any
    out['stage_top3_all'] = stg_top3_all
    out['n_seen'] = n_seen
    out['seen_vs_overall'] = {
        'technique': {'seen': seen_tech, 'overall': overall_tech},
        'tactic':    {'seen': seen_tac,  'overall': overall_tac},
        'stage':     {'seen': seen_stg,  'overall': overall_stg},
    }
    out['lrap_aupr_seen_vs_overall'] = {
        'technique': {'seen': [seen_lrap_tech, seen_aupr_tech], 'overall': [out['lrap'], out['aupr']]},
        'tactic':    {'seen': [seen_lrap_tac, seen_aupr_tac],   'overall': [tac_lrap, tac_aupr]},
        'stage':     {'seen': [seen_lrap_stg, seen_aupr_stg],   'overall': [stg_lrap, stg_aupr]},
    }

    results_tag = run_tag
    if zoomer_scope:
        results_tag += '_zoomerscope'
    if not strict_native:
        results_tag += '_withextra'
    if candidate_cap is not None:
        results_tag += '_candcap'
    results_path = os.path.join(RESULTS_DIR, 'camlds_technique_wide_test_results_{}.json'.format(results_tag))
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n  Results saved → {}'.format(results_path))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-file', type=str, default=None)
    ap.add_argument('--test-scenario', type=str, default=None,
                     help='Hold out an ENTIRE scenario as test (e.g. "4"). Overrides --test-file when set.')
    ap.add_argument('--temp', type=float, default=0.07)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=SEED)
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--template-dir', type=str, default=None)
    ap.add_argument('--sequences-dir', type=str, default=None)
    ap.add_argument('--run', type=str, default=None, dest='run_filter',
                     help='Only show test steps whose filename contains this substring, e.g. "ssh_apt".')
    ap.add_argument('--zoomer-scope', action='store_true',
                     help="Narrow the 36-technique candidate set down to ZOOMER's 26-technique min4 "
                          '(>=4 dataset-wide samples) scope, for a true apples-to-apples comparison. '
                          'No retraining needed -- the checkpoint already has all 36 templates, this '
                          'just restricts which ones get scored against at test time.')
    ap.add_argument('--allow-extra-techniques', dest='strict_native', action='store_false',
                     help='Default is to cap the candidate set at exactly the trained native scope. Pass '
                          'this flag to also include rare test-only techniques (below the '
                          'training-sufficiency bar, e.g. T1599) as extra zero-shot candidates whenever a '
                          'template exists for them, even though they were never trained on.')
    args = ap.parse_args()

    run_tag = args.run_tag
    if run_tag is None:
        if args.test_scenario:
            run_tag = 'scenario{}'.format(args.test_scenario)
        elif args.test_file:
            run_tag = args.test_file

    run(test_file_match=args.test_file, test_scenario=args.test_scenario, temp=args.temp, test_size=args.test_size,
        split_seed=args.split_seed, run_tag=run_tag,
        template_dir=args.template_dir, sequences_dir=args.sequences_dir, run_filter=args.run_filter,
        zoomer_scope=args.zoomer_scope, strict_native=args.strict_native)
