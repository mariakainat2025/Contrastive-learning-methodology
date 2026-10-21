import json
import os
import sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
from Deep_Wide_Model import TSGModel
from data_utils import GraphTensorCache, instance_from_filename, load_tactic_map, instance_tactics
from create_data_split import get_zoomer_split, get_zoomer_split_scenario
from discretize_features import ALL_DIMS, fit_bins
from cross_product import generate_masks, DEFAULT_K as CROSS_PRODUCT_K
from Train_TTP_Recognition import IN_DIM, checkpoint_path
from tactic_ids import TACTIC_IDS
from tactic_to_stage import STAGE_ORDER, TACTIC_TO_STAGE, tactics_to_stages
from stage_accuracy import (stage_top_k_accuracy, stage_accuracy_at_n_true,
                             tactic_top_k_accuracy, tactic_accuracy_at_n_true,
                             technique_top_k_accuracy, technique_accuracy_at_n_true)

RESULTS_DIR = '/csse/research/contructive-learning/CAM-LDS/zoomer/results'

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

def stage_scores_from_tactic_scores_max(tactic_scores):
    """ZOOMER-only stage scoring: a stage's score is the BEST (max) of its tactics'
    scores, not the sum. tactic_to_stage.py's own stage_scores_from_tactic_scores()
    sums instead -- fine for CAM-LDS's softmax-probability scores, but these are
    negative distances, so summing 2-3 tactics into one stage (every stage except
    "Move Laterally", which has exactly one contributing tactic) systematically
    drags that stage's score down regardless of true relevance, letting "Move
    Laterally" win almost every row by default. Same fix already applied on the
    multilabel side (see stage_scores_from_tactic_scores_max in
    Test_TTP_Recognition_Multilabel.py)."""
    stage_scores = {}
    for (tactic, score) in tactic_scores.items():
        stage = TACTIC_TO_STAGE.get(tactic)
        if stage is None:
            continue
        if stage not in stage_scores or score > stage_scores[stage]:
            stage_scores[stage] = score
    return stage_scores

def topn_any_all_match(rows, true_key, top_n=3):
    """ANY-match = row counts as correct if >=1 true label is in the top-n.
    ALL-match = correct only if every true label is in the top-n. Same rule
    CAM-LDS's test script uses, for a direct side-by-side comparison."""
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
    """A row counts as 'seen' only if EVERY true technique it has was present
    at least once in training -- only counting techniques actually in the
    trainable scope (train_tech_counts is pre-filtered to `classes`), so an
    out-of-scope technique with stray raw samples elsewhere never counts."""
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

def embed_graph(model, cache, path, device):
    (h, adjacency, wide_x) = cache.get(path)
    with torch.no_grad():
        return model(h.to(device), adjacency.to(device), wide_x.to(device))

def build_final_prototypes(model, cache, split, classes, device):
    prototypes = []
    used_classes = []
    for technique in classes:
        paths = [path for (_, path) in split[technique]['train']]
        if not paths:
            continue
        embeds = torch.stack([embed_graph(model, cache, p, device) for p in paths])
        prototypes.append(embeds.mean(dim=0))
        used_classes.append(technique)
    return torch.stack(prototypes), used_classes

def print_results_table(rows, n_classes, top_n=3):
    col_file = 32
    col_true = 24
    col_score = 18
    col_found = 7
    print()
    print('  -- Test Results (scored against {} technique prototypes) --'.format(n_classes))
    header_fmt = ('  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} ' +
                  ' '.join(['{:<' + str(col_score) + '}'] * top_n) + ' {:<' + str(col_found) + '} {:<6} {}')
    print(header_fmt.format('#', 'File', 'True technique(s)', *['#{} (score)'.format(j + 1) for j in range(top_n)], 'Found', 'Wrong', 'Missing'))
    print('  ' + '-' * (4 + col_file + col_true + col_score * top_n + col_found + 6 + top_n + 4))
    n_wrong = 0
    for (i, r) in enumerate(rows, 1):
        ranked = r['ranked']
        kept = ranked[:top_n]
        cols = []
        for j in range(top_n):
            cols.append('{} ({:.4f})'.format(*kept[j]) if j < len(kept) else '')
        fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
        true_str = ', '.join(sorted(r['true_techniques'])) or '(none)'
        if len(true_str) > col_true:
            true_str = true_str[:col_true - 3] + '...'
        kept_names = {t for (t, _) in kept}
        n_found = len(kept_names & r['true_techniques'])
        n_true = len(r['true_techniques'])
        found_str = '{}/{}'.format(n_found, n_true) if n_true else '-'
        missing = sorted(r['true_techniques'] - kept_names)
        missing_str = ','.join(missing)
        top1_wrong = not (kept_names & r['true_techniques'])
        if top1_wrong:
            n_wrong += 1
        print(header_fmt.format(i, fname, true_str, *cols, found_str, 'WRONG' if top1_wrong else '', missing_str))
    print()
    print('  Totally wrong (no true technique in top-{}) : {}/{} samples'.format(top_n, n_wrong, len(rows)))
    return n_wrong

def tid(tactic):
    return TACTIC_IDS.get(tactic, tactic)

def print_tactic_results_table(tactic_rows, n_tactics, top_n=3):
    col_file = 26
    col_true = 24
    col_score = 16
    print()
    print('-- Test Results (scored against {} tactic prototypes) --'.format(n_tactics))
    header_fmt = '{:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} ' + ' '.join(['{:<' + str(col_score) + '}'] * top_n) + ' {:<6}'
    print(header_fmt.format('#', 'File', 'True tactics', *['#{} (score)'.format(j + 1) for j in range(top_n)], 'Top{}'.format(top_n)))
    print('-' * (4 + col_file + col_true + col_score * top_n + 6 + top_n + 3))
    n_wrong = 0
    for (i, r) in enumerate(tactic_rows, 1):
        ranked = r['ranked']
        top_tactics = {t for (t, _) in ranked[:top_n]}
        cols = []
        for j in range(top_n):
            cols.append('{} {:.4f}'.format(tid(ranked[j][0]), ranked[j][1]) if j < len(ranked) else '')
        true_str = ','.join((tid(t) for t in sorted(r['true_tactics'])))
        if len(true_str) > col_true:
            true_str = true_str[:col_true - 3] + '...'
        fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
        wrong = not r['true_tactics'] & top_tactics
        if wrong:
            n_wrong += 1
        print(header_fmt.format(i, fname, true_str, *cols, 'WRONG' if wrong else ''))
    return n_wrong

def print_stage_results_table(stage_rows, max_stages=4):
    col_file = 26
    col_true = 22
    col_score = 12
    col_found = 7
    print()
    print('-- Stage Results (KnowHow 7-stage lifecycle) --')
    print('IC=Initial Compromise  EF=Establish Foothold  EP=Escalate Privilege  IR=Internal Reconnaissance '
          'ML=Move Laterally  MP=Maintain Persistence  CM=Complete Mission')
    header_fmt = ('{:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} ' +
                  ' '.join(['{:<' + str(col_score) + '}'] * max_stages) + ' {:<' + str(col_found) + '} {:<6} {}')
    print(header_fmt.format('#', 'File', 'True stage(s)', *['#{}'.format(j + 1) for j in range(max_stages)], 'Found', 'Wrong', 'Missing'))
    print('-' * (4 + col_file + col_true + col_score * max_stages + col_found + 6 + max_stages + 4))
    n_wrong = 0
    for (i, r) in enumerate(stage_rows, 1):
        kept = r['ranked'][:max_stages]
        cols = []
        for j in range(max_stages):
            cols.append('{} ({:.2f})'.format(sid(kept[j][0]), kept[j][1]) if j < len(kept) else '')
        true_str = ', '.join(sid(s) for s in sorted(r['true_stages'], key=STAGE_ORDER.index)) or '(none)'
        kept_names = {s for (s, _) in kept}
        n_found = len(kept_names & r['true_stages'])
        n_true = len(r['true_stages'])
        found_str = '{}/{}'.format(n_found, n_true) if n_true else '-'
        missing = sorted(r['true_stages'] - kept_names, key=STAGE_ORDER.index)
        missing_str = ','.join(sid(s) for s in missing)
        wrong = not (kept_names & r['true_stages'])
        if wrong:
            n_wrong += 1
        fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
        print(header_fmt.format(i, fname, true_str, *cols, found_str, 'WRONG' if wrong else '', missing_str))
    return n_wrong

def main(seed, scenario=None, test_file=None, run_tag=None, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    held_out = test_file or scenario
    if run_tag is None:
        run_tag = test_file if test_file else ('scenario{}'.format(scenario) if scenario else 'seed{}'.format(seed))
    if verbose:
        if held_out:
            print('Held out: {}  Device: {}'.format(held_out, device))
        else:
            print('Seed: {}  Device: {}'.format(seed, device))
    ckpt = torch.load(checkpoint_path(run_tag), map_location=device)
    classes = ckpt['classes']
    model = TSGModel(deep_in_dim=IN_DIM, wide_in_dim=ckpt['wide_in_dim']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    split = get_zoomer_split_scenario(scenario, single_label=True, test_file=test_file) if held_out else get_zoomer_split(seed)
    train_paths = sorted({path for t in classes for (_, path) in split[t]['train']})
    bins = fit_bins(train_paths)
    h_cat_dim = sum(len(bins[dim]) for dim in ALL_DIMS)
    masks = generate_masks(h_cat_dim, seed=seed)
    cache = GraphTensorCache(bins, masks)

    tactic_map = load_tactic_map()
    prototypes, classes = build_final_prototypes(model, cache, split, classes, device)
    class_tactic = {t: split[t]['tactic'] for t in classes}
    all_tactics = sorted(set(class_tactic.values()))
    unique_test = {}
    for technique in split:
        for (filename, path) in split[technique]['test']:
            inst = instance_from_filename(filename)
            entry = unique_test.setdefault(inst, {'filename': filename, 'path': path, 'true_techniques': set()})
            entry['true_techniques'].add(technique)
    if held_out and verbose:
        classes_set = set(classes)
        orphaned = sorted(inst for (inst, e) in unique_test.items() if not (e['true_techniques'] & classes_set))
        print()
        print('  Test steps with NO trained technique (scored anyway, will show WRONG): {}/{}'.format(
            len(orphaned), len(unique_test)))
        for inst in orphaned:
            print('    {:20s} true technique(s): {}'.format(inst, sorted(unique_test[inst]['true_techniques'])))
    n_total = 0
    n_correct_tech = 0
    n_correct_tac = 0
    y_true_tac = []
    y_score_tac = []
    y_true_tech = []
    y_score_tech = []
    y_true_stage = []
    y_score_stage = []
    rows = []
    tactic_rows = []
    stage_rows = []
    for inst in sorted(unique_test):
        filename = unique_test[inst]['filename']
        path = unique_test[inst]['path']
        true_techniques = unique_test[inst]['true_techniques']
        embed = embed_graph(model, cache, path, device)
        dists = torch.norm(prototypes - embed, dim=1)
        pred_idx = int(torch.argmin(dists))
        pred_technique = classes[pred_idx]
        true_tactics = set()
        for technique in true_techniques:
            true_tactics |= instance_tactics(tactic_map, technique, filename)
        n_total += 1
        if pred_technique in true_techniques:
            n_correct_tech += 1
        if class_tactic[pred_technique] in true_tactics:
            n_correct_tac += 1
        scores_by_class = (-dists).tolist()
        tactic_scores = {t: -1000000000.0 for t in all_tactics}
        for (c, s) in zip(classes, scores_by_class):
            t = class_tactic[c]
            if s > tactic_scores[t]:
                tactic_scores[t] = s
        y_true_tac.append([1 if t in all_tactics and t in true_tactics else 0 for t in all_tactics])
        y_score_tac.append([tactic_scores[t] for t in all_tactics])
        y_true_tech.append([1 if c in true_techniques else 0 for c in classes])
        y_score_tech.append(scores_by_class)
        ranked = sorted(zip(classes, scores_by_class), key=lambda x: x[1], reverse=True)
        rows.append({'file': filename, 'true_technique': '/'.join(sorted(true_techniques)), 'true_techniques': true_techniques, 'ranked': ranked})
        tactic_ranked = sorted(tactic_scores.items(), key=lambda x: x[1], reverse=True)
        tactic_rows.append({'file': inst, 'true_tactics': true_tactics, 'ranked': tactic_ranked})
        stage_scores = stage_scores_from_tactic_scores_max(tactic_scores)
        stage_ranked = sorted(stage_scores.items(), key=lambda x: x[1], reverse=True)
        true_stages = set(tactics_to_stages(true_tactics))
        stage_rows.append({'file': inst, 'true_stages': true_stages, 'ranked': stage_ranked})
        y_true_stage.append([1 if s in true_stages else 0 for s in STAGE_ORDER])
        y_score_stage.append([stage_scores.get(s, -1e9) for s in STAGE_ORDER])
    if verbose:
        n_wrong_tech = print_results_table(rows, len(classes))
        n_wrong_top3_tac = print_tactic_results_table(tactic_rows, len(all_tactics))
        n_wrong_stage = print_stage_results_table(stage_rows)
    else:
        top_techniques_by_row = [{t for (t, _) in r['ranked'][:3]} for r in rows]
        n_wrong_tech = sum(1 for (r, top) in zip(rows, top_techniques_by_row) if not r['true_techniques'] & top)
        top_tactics_by_row = [{t for (t, _) in r['ranked'][:3]} for r in tactic_rows]
        n_wrong_top3_tac = sum(1 for (r, top) in zip(tactic_rows, top_tactics_by_row) if not r['true_tactics'] & top)
        top_stages_by_row = [{s for (s, _) in r['ranked'][:4]} for r in stage_rows]
        n_wrong_stage = sum(1 for (r, top) in zip(stage_rows, top_stages_by_row) if not r['true_stages'] & top)
    stage_acc = stage_top_k_accuracy(stage_rows)
    stage_acc_top1, stage_acc_top3 = stage_acc[1], stage_acc[3]
    stage_acc_at_n = stage_accuracy_at_n_true(stage_rows)
    tac_acc_k = tactic_top_k_accuracy(tactic_rows)
    tac_acc_top1, tac_acc_top3 = tac_acc_k[1], tac_acc_k[3]
    tac_acc_at_n = tactic_accuracy_at_n_true(tactic_rows)
    tech_acc_k = technique_top_k_accuracy(rows)
    tech_acc_top1, tech_acc_top3 = tech_acc_k[1], tech_acc_k[3]
    tech_acc_at_n = technique_accuracy_at_n_true(rows)
    tech_acc = n_correct_tech / n_total
    tac_acc = n_correct_tac / n_total

    def lrap_aupr(y_true_list, y_score_list):
        if not y_true_list:
            return 0.0, 0.0
        yt = np.array(y_true_list)
        ys = np.array(y_score_list)
        lrap_v = label_ranking_average_precision_score(yt, ys)
        valid_cols = yt.sum(axis=0) > 0
        aupr_v = average_precision_score(yt[:, valid_cols], ys[:, valid_cols], average='macro') if valid_cols.any() else 0.0
        return float(lrap_v), float(aupr_v)

    lrap, aupr = lrap_aupr(y_true_tech, y_score_tech)
    tac_lrap, tac_aupr = lrap_aupr(y_true_tac, y_score_tac)
    stage_lrap, stage_aupr = lrap_aupr(y_true_stage, y_score_stage)

    tech_top3_any, tech_top3_all = topn_any_all_match(rows, 'true_techniques', 3)
    tac_top3_any, tac_top3_all = topn_any_all_match(tactic_rows, 'true_tactics', 3)
    stage_top3_any, stage_top3_all = topn_any_all_match(stage_rows, 'true_stages', 3)

    classes_set = set(classes)
    train_tech_counts = {t: len(split[t]['train']) for t in split if t in classes_set}
    seen_mask = [row_is_seen(r['true_techniques'], train_tech_counts) for r in rows]
    n_seen = sum(seen_mask)
    seen_rows = filter_rows_by_mask(rows, seen_mask)
    seen_tactic_rows = filter_rows_by_mask(tactic_rows, seen_mask)
    seen_stage_rows = filter_rows_by_mask(stage_rows, seen_mask)
    seen_y_true_tech = filter_rows_by_mask(y_true_tech, seen_mask)
    seen_y_score_tech = filter_rows_by_mask(y_score_tech, seen_mask)
    seen_y_true_tac = filter_rows_by_mask(y_true_tac, seen_mask)
    seen_y_score_tac = filter_rows_by_mask(y_score_tac, seen_mask)
    seen_y_true_stage = filter_rows_by_mask(y_true_stage, seen_mask)
    seen_y_score_stage = filter_rows_by_mask(y_score_stage, seen_mask)

    seen_tech = compute_metrics_block(seen_rows, 'true_techniques', technique_top_k_accuracy, technique_accuracy_at_n_true)
    seen_tac = compute_metrics_block(seen_tactic_rows, 'true_tactics', tactic_top_k_accuracy, tactic_accuracy_at_n_true)
    seen_stg = compute_metrics_block(seen_stage_rows, 'true_stages', stage_top_k_accuracy, stage_accuracy_at_n_true)
    overall_tech = (tech_acc_top1, tech_acc_at_n, tech_top3_any, tech_top3_all)
    overall_tac = (tac_acc_top1, tac_acc_at_n, tac_top3_any, tac_top3_all)
    overall_stg = (stage_acc_top1, stage_acc_at_n, stage_top3_any, stage_top3_all)

    seen_lrap_tech, seen_aupr_tech = lrap_aupr(seen_y_true_tech, seen_y_score_tech)
    seen_lrap_tac, seen_aupr_tac = lrap_aupr(seen_y_true_tac, seen_y_score_tac)
    seen_lrap_stg, seen_aupr_stg = lrap_aupr(seen_y_true_stage, seen_y_score_stage)

    technique_results = [{'file': r['file'], 'true_techniques': sorted(r['true_techniques']),
                           'ranked': [{'technique': t, 'score': s} for (t, s) in r['ranked']]}
                          for r in rows]
    tactic_results = [{'file': r['file'], 'true_tactics': sorted(r['true_tactics']),
                        'ranked': [{'tactic': t, 'score': s} for (t, s) in r['ranked']]}
                       for r in tactic_rows]
    stage_results = [{'file': r['file'], 'true_stages': sorted(r['true_stages'], key=STAGE_ORDER.index),
                       'ranked': [{'stage': s, 'score': sc} for (s, sc) in r['ranked']]}
                      for r in stage_rows]
    metrics = {'seed': seed, 'scenario': scenario, 'run_tag': run_tag, 'n_total': n_total,
               'n_classes': len(classes), 'n_tactics': len(all_tactics),
               'tech_acc': tech_acc, 'tac_acc': tac_acc, 'lrap': lrap, 'aupr': aupr,
               'tac_lrap': tac_lrap, 'tac_aupr': tac_aupr, 'stage_lrap': stage_lrap, 'stage_aupr': stage_aupr,
               'stage_acc_top3': float(stage_acc_top3), 'stage_acc_top1': float(stage_acc_top1),
               'stage_acc_at_n': float(stage_acc_at_n),
               'tac_acc_top3': float(tac_acc_top3), 'tac_acc_top1': float(tac_acc_top1),
               'tac_acc_at_n': float(tac_acc_at_n),
               'tech_acc_top3': float(tech_acc_top3), 'tech_acc_top1': float(tech_acc_top1),
               'tech_acc_at_n': float(tech_acc_at_n),
               'n_wrong_tech': n_wrong_tech, 'n_wrong_top3_tac': n_wrong_top3_tac, 'n_wrong_stage': n_wrong_stage,
               'tech_top3_any': tech_top3_any, 'tech_top3_all': tech_top3_all,
               'tac_top3_any': tac_top3_any, 'tac_top3_all': tac_top3_all,
               'stage_top3_any': stage_top3_any, 'stage_top3_all': stage_top3_all,
               'n_seen': n_seen,
               'seen_vs_overall': {
                   'technique': {'seen': list(seen_tech), 'overall': list(overall_tech)},
                   'tactic': {'seen': list(seen_tac), 'overall': list(overall_tac)},
                   'stage': {'seen': list(seen_stg), 'overall': list(overall_stg)},
               },
               'lrap_aupr_seen_vs_overall': {
                   'technique': {'seen': [seen_lrap_tech, seen_aupr_tech], 'overall': [lrap, aupr]},
                   'tactic': {'seen': [seen_lrap_tac, seen_aupr_tac], 'overall': [tac_lrap, tac_aupr]},
                   'stage': {'seen': [seen_lrap_stg, seen_aupr_stg], 'overall': [stage_lrap, stage_aupr]},
               },
               'technique_results': technique_results, 'tactic_results': tactic_results,
               'stage_results': stage_results}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'ttp_recognition_{}.json'.format(run_tag))
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print()
        print('Technique LRAP: {:.1f}%   AUPR: {:.1f}%'.format(lrap * 100, aupr * 100))
        print('Tactic    LRAP: {:.1f}%   AUPR: {:.1f}%'.format(tac_lrap * 100, tac_aupr * 100))
        print('Stage     LRAP: {:.1f}%   AUPR: {:.1f}%'.format(stage_lrap * 100, stage_aupr * 100))
        print('Technique level wrong (no true technique in top-3) : {}/{} samples'.format(n_wrong_tech, n_total))
        print('Tactic level wrong (no true tactic in top-3)       : {}/{} samples'.format(n_wrong_top3_tac, n_total))
        print('Stage level wrong (no true stage in top-4)         : {}/{} samples'.format(n_wrong_stage, n_total))
        print()
        print('Test samples          : {}'.format(n_total))
        print('Technique classes     : {}'.format(len(classes)))
        print('Tactics represented   : {}'.format(len(all_tactics)))
        print()
        print('Technique Accuracy     top-1: {:.1f}%   top-3: {:.1f}%   at-n: {:.1f}%'.format(
            tech_acc_top1 * 100, tech_acc_top3 * 100, tech_acc_at_n * 100))
        print('Tactic Accuracy        top-1: {:.1f}%   top-3: {:.1f}%   at-n: {:.1f}%'.format(
            tac_acc_top1 * 100, tac_acc_top3 * 100, tac_acc_at_n * 100))
        print('Stage Accuracy        top-1: {:.1f}%   top-3: {:.1f}%   at-n: {:.1f}%'.format(
            stage_acc_top1 * 100, stage_acc_top3 * 100, stage_acc_at_n * 100))
        print()
        print('  ── Seen (trained on every true label) vs Overall ──')
        print('  "Seen" = every true label for that step has >=1 training example (n={}/{} steps).'.format(
            n_seen, n_total))
        print()
        row_fmt = '  {:<22}{:>9}{:>9}{:>13}{:>13}'
        print(row_fmt.format('', 'top-1', 'at-n', 'top-3-any', 'top-3-all'))
        print('  Technique')
        print(row_fmt.format('    Seen ({})'.format(n_seen), *['{:.1f}%'.format(v * 100) for v in seen_tech]))
        print(row_fmt.format('    Overall ({})'.format(n_total), *['{:.1f}%'.format(v * 100) for v in overall_tech]))
        print('  Tactic')
        print(row_fmt.format('    Seen ({})'.format(n_seen), *['{:.1f}%'.format(v * 100) for v in seen_tac]))
        print(row_fmt.format('    Overall ({})'.format(n_total), *['{:.1f}%'.format(v * 100) for v in overall_tac]))
        print('  Stage')
        print(row_fmt.format('    Seen ({})'.format(n_seen), *['{:.1f}%'.format(v * 100) for v in seen_stg]))
        print(row_fmt.format('    Overall ({})'.format(n_total), *['{:.1f}%'.format(v * 100) for v in overall_stg]))
        print()
        print('  ── LRAP / AUPR: Seen vs Overall ──')
        lrap_fmt = '  {:<22}{:>9}{:>9}'
        print(lrap_fmt.format('', 'LRAP', 'AUPR'))
        print('  Technique')
        print(lrap_fmt.format('    Seen ({})'.format(n_seen), '{:.1f}%'.format(seen_lrap_tech * 100), '{:.1f}%'.format(seen_aupr_tech * 100)))
        print(lrap_fmt.format('    Overall ({})'.format(n_total), '{:.1f}%'.format(lrap * 100), '{:.1f}%'.format(aupr * 100)))
        print('  Tactic')
        print(lrap_fmt.format('    Seen ({})'.format(n_seen), '{:.1f}%'.format(seen_lrap_tac * 100), '{:.1f}%'.format(seen_aupr_tac * 100)))
        print(lrap_fmt.format('    Overall ({})'.format(n_total), '{:.1f}%'.format(tac_lrap * 100), '{:.1f}%'.format(tac_aupr * 100)))
        print('  Stage')
        print(lrap_fmt.format('    Seen ({})'.format(n_seen), '{:.1f}%'.format(seen_lrap_stg * 100), '{:.1f}%'.format(seen_aupr_stg * 100)))
        print(lrap_fmt.format('    Overall ({})'.format(n_total), '{:.1f}%'.format(stage_lrap * 100), '{:.1f}%'.format(stage_aupr * 100)))
        print()
        print('Results saved -> {}'.format(out_path))
    return metrics
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scenario', type=str, default=None,
                     help='Test against the checkpoint trained with this scenario held out.')
    ap.add_argument('--test-file', type=str, default=None,
                     help='Test against the checkpoint trained with --test-file (e.g. "2_rootkit"). '
                          'Takes priority over --scenario if both are passed.')
    ap.add_argument('--run-tag', type=str, default=None)
    args = ap.parse_args()
    main(args.seed, scenario=args.scenario, test_file=args.test_file, run_tag=args.run_tag)
