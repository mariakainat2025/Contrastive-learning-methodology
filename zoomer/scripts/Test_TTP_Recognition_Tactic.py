import json
import os
import re
import sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
from Deep_Wide_Model import TSGModel
from data_utils import GraphTensorCache, instance_from_filename
from create_data_split import get_zoomer_split_tactic, get_zoomer_split_scenario_tactic
from discretize_features import ALL_DIMS, fit_bins
from cross_product import generate_masks, DEFAULT_K as CROSS_PRODUCT_K
from Train_TTP_Recognition_Tactic import IN_DIM, checkpoint_path

RESULTS_DIR = '/csse/research/contructive-learning/CAM-LDS/zoomer/results'
_CAM_LDS_SCRIPTS = '/csse/research/contructive-learning/CAM-LDS/scripts'
if _CAM_LDS_SCRIPTS not in sys.path:
    sys.path.insert(0, _CAM_LDS_SCRIPTS)
from train_camlds_matcher import TACTIC_IDS
from tactic_to_stage import STAGE_ORDER, TACTIC_TO_STAGE, tactics_to_stages

def stage_scores_from_tactic_scores_max(tactic_scores):
    """ZOOMER-only stage scoring: a stage's score is the BEST (max) of its tactics'
    scores, not the sum. This guarantees the stage holding the single best-scoring
    tactic overall always wins at the stage level too -- sum (CAM-LDS's method,
    left untouched in tactic_to_stage.py) can lose that race just by having to add
    a weaker second or third tactic on top."""
    stage_scores = {}
    for (tactic, score) in tactic_scores.items():
        stage = TACTIC_TO_STAGE.get(tactic)
        if stage is None:
            continue
        if stage not in stage_scores or score > stage_scores[stage]:
            stage_scores[stage] = score
    return stage_scores

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

def step_sort_key(step_name):
    m = re.search(r'(\d+)$', step_name)
    return (int(m.group(1)) if m else 0, step_name)

RUN_GROUP_RE = re.compile(r'^(.+)-(\d+)$')

def run_group_of(step_name):
    m = RUN_GROUP_RE.match(step_name)
    return m.group(1) if m else step_name

def embed_graph(model, cache, path, device):
    (h, adjacency, wide_x) = cache.get(path)
    with torch.no_grad():
        return model(h.to(device), adjacency.to(device), wide_x.to(device))

def build_final_prototypes(model, cache, split, classes, device):
    prototypes = []
    for tactic in classes:
        paths = [path for (_, path) in split[tactic]['train']]
        embeds = torch.stack([embed_graph(model, cache, p, device) for p in paths])
        prototypes.append(embeds.mean(dim=0))
    return torch.stack(prototypes)

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

def print_tactic_training_summary(split, classes):
    classes_set = set(classes)
    all_tacs = sorted(split.keys())
    skip_tacs = sorted(t for t in all_tacs if t not in classes_set)
    tested_tacs = sorted(t for t in all_tacs if len(split[t]['test']) > 0)
    training_present = [t for t in tested_tacs if t in classes_set]
    training_absent = [t for t in tested_tacs if t not in classes_set]

    step_tactics = {}
    for (tactic, pools) in split.items():
        for (filename, _) in pools['test']:
            inst = instance_from_filename(filename)
            step_tactics.setdefault(inst, set()).add(tactic)
    take_steps = sorted(inst for (inst, tacs) in step_tactics.items() if tacs & classes_set)
    skip_steps = sorted(inst for (inst, tacs) in step_tactics.items() if not (tacs & classes_set))

    train_step_tactics = {}
    for (tactic, pools) in split.items():
        for (filename, _) in pools['train']:
            inst = instance_from_filename(filename)
            train_step_tactics.setdefault(inst, set()).add(tactic)
    skip_train_steps = sorted(inst for (inst, tacs) in train_step_tactics.items() if not (tacs & classes_set))

    print()
    print('-- Tactic Training Summary --')
    print('Total tactics             : {}'.format(len(all_tacs)))
    print('Include tactics (trained) : {}'.format(len(classes)))
    print('Skip tactics (not trained): {}'.format(len(skip_tacs)))
    print()
    print('Training present ({} tactics): {}'.format(len(training_present), ', '.join(training_present)))
    print('Training NOT present ({} tactics): {}'.format(len(training_absent), ', '.join(training_absent)))
    print()
    print('Total training steps : {}'.format(len(train_step_tactics)))
    print('Total skip steps (training): {}'.format(len(skip_train_steps)))
    print()
    print('Total test steps : {}'.format(len(step_tactics)))
    print('Take steps ({}): {}'.format(len(take_steps), ', '.join(take_steps)))
    print('Skip steps ({}): {}'.format(len(skip_steps), ', '.join(skip_steps)))
    print()

def main(seed, scenario=None, run_tag=None, verbose=True, run_filter=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if run_tag is None:
        run_tag = 'scenario{}'.format(scenario) if scenario else 'seed{}'.format(seed)
    if verbose:
        if scenario:
            print('[tactic] Scenario held out: {}  Device: {}'.format(scenario, device))
        else:
            print('[tactic] Seed: {}  Device: {}'.format(seed, device))
        if run_filter:
            print('  (reporting headline metrics only for run group: {} -- full scenario still trained/scored)'.format(run_filter))
    ckpt = torch.load(checkpoint_path(run_tag), map_location=device)
    classes = ckpt['classes']
    model = TSGModel(deep_in_dim=IN_DIM, wide_in_dim=ckpt['wide_in_dim']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    split = get_zoomer_split_scenario_tactic(scenario) if scenario else get_zoomer_split_tactic(seed)
    train_paths = sorted({path for t in classes for (_, path) in split[t]['train']})
    bins = fit_bins(train_paths)
    h_cat_dim = sum(len(bins[dim]) for dim in ALL_DIMS)
    masks = generate_masks(h_cat_dim, seed=seed)
    cache = GraphTensorCache(bins, masks)

    prototypes = build_final_prototypes(model, cache, split, classes, device)
    unique_test = {}
    for tactic in split:
        for (filename, path) in split[tactic]['test']:
            inst = instance_from_filename(filename)
            entry = unique_test.setdefault(inst, {'filename': filename, 'path': path, 'true_tactics': set()})
            entry['true_tactics'].add(tactic)
    if scenario and verbose:
        print_tactic_training_summary(split, classes)
    all_rows = []
    for inst in sorted(unique_test, key=step_sort_key):
        path = unique_test[inst]['path']
        true_tactics = unique_test[inst]['true_tactics']
        embed = embed_graph(model, cache, path, device)
        dists = torch.norm(prototypes - embed, dim=1)
        scores_by_class = (-dists).tolist()
        tactic_scores = dict(zip(classes, scores_by_class))
        tactic_ranked = sorted(tactic_scores.items(), key=lambda x: x[1], reverse=True)
        stage_scores = stage_scores_from_tactic_scores_max(tactic_scores)
        stage_ranked = sorted(stage_scores.items(), key=lambda x: x[1], reverse=True)
        true_stages = set(tactics_to_stages(true_tactics))
        all_rows.append({
            'file': inst, 'true_tactics': true_tactics, 'tactic_ranked': tactic_ranked,
            'true_stages': true_stages, 'stage_ranked': stage_ranked,
            'y_true': [1 if t in true_tactics else 0 for t in classes], 'y_score': scores_by_class,
        })

    # full scenario is always trained and scored (nothing hidden) -- run_filter only narrows
    # which steps the HEADLINE metrics below are computed from, same as CAM-LDS's --run flag
    display_rows = [r for r in all_rows if run_group_of(r['file']) == run_filter] if run_filter else all_rows
    if run_filter and verbose:
        print('\n  (showing only run group: {})'.format(run_filter))

    tactic_rows = [{'file': r['file'], 'true_tactics': r['true_tactics'], 'ranked': r['tactic_ranked']} for r in display_rows]
    stage_rows = [{'file': r['file'], 'true_stages': r['true_stages'], 'ranked': r['stage_ranked']} for r in display_rows]
    n_total = len(display_rows)
    n_correct_tac = sum(1 for r in display_rows if r['tactic_ranked'][0][0] in r['true_tactics'])
    if verbose:
        n_wrong_top3_tac = print_tactic_results_table(tactic_rows, len(classes), top_n=4)
        n_wrong_stage = print_stage_results_table(stage_rows)
    else:
        top_tactics_by_row = [{t for (t, _) in r['ranked'][:4]} for r in tactic_rows]
        n_wrong_top3_tac = sum(1 for (r, top) in zip(tactic_rows, top_tactics_by_row) if not r['true_tactics'] & top)
        top_stages_by_row = [{s for (s, _) in r['ranked'][:4]} for r in stage_rows]
        n_wrong_stage = sum(1 for (r, top) in zip(stage_rows, top_stages_by_row) if not r['true_stages'] & top)
    tac_acc = n_correct_tac / n_total
    y_true_tac = np.array([r['y_true'] for r in display_rows])
    y_score_tac = np.array([r['y_score'] for r in display_rows])
    lrap = label_ranking_average_precision_score(y_true_tac, y_score_tac)
    valid_cols = y_true_tac.sum(axis=0) > 0
    aupr = average_precision_score(y_true_tac[:, valid_cols], y_score_tac[:, valid_cols], average='macro') if valid_cols.any() else 0.0
    # full (unfiltered) per-step breakdown -- saved in full even when run_filter narrows the headline numbers above
    tactic_results = [{'file': r['file'], 'true_tactics': sorted(r['true_tactics']),
                        'ranked': [{'tactic': t, 'score': s} for (t, s) in r['tactic_ranked']]}
                       for r in all_rows]
    stage_results = [{'file': r['file'], 'true_stages': sorted(r['true_stages'], key=STAGE_ORDER.index),
                       'ranked': [{'stage': s, 'score': sc} for (s, sc) in r['stage_ranked']]}
                      for r in all_rows]
    metrics = {'seed': seed, 'scenario': scenario, 'run_tag': run_tag, 'run_filter': run_filter, 'n_total': n_total,
               'n_classes': len(classes), 'tac_acc': float(tac_acc), 'lrap': float(lrap), 'aupr': float(aupr),
               'n_wrong_top3_tac': n_wrong_top3_tac, 'n_wrong_stage': n_wrong_stage,
               'tactic_results': tactic_results, 'stage_results': stage_results}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'ttp_recognition_tactic_{}.json'.format(run_tag))
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print()
        print('Wrong (true label not in top-4)                 : {}/{} samples'.format(n_wrong_top3_tac, n_total))
        print('Wrong at stage level (no true stage in top-4)   : {}/{} samples'.format(n_wrong_stage, n_total))
        print()
        print('Test samples          : {}'.format(n_total))
        print('Tactic classes        : {}'.format(len(classes)))
        print()
        print('Tactic Accuracy       : {:.1f}%'.format(tac_acc * 100))
        print()
        print('LRAP (Label Ranking Average Precision)          : {:.1f}%'.format(lrap * 100))
        print('AUPR (Area Under Precision-Recall Curve, macro) : {:.1f}%'.format(aupr * 100))
        print()
        print('Results saved -> {}'.format(out_path))
    return metrics
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scenario', type=str, default=None,
                     help='Test against the checkpoint trained with this scenario held out.')
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--run', type=str, default=None,
                     help='Only report headline metrics for this one run group (exact match), e.g. --run 6_macro_binary. '
                          'The full scenario is still trained and scored -- this only narrows what gets reported.')
    args = ap.parse_args()
    main(args.seed, scenario=args.scenario, run_tag=args.run_tag, run_filter=args.run)
