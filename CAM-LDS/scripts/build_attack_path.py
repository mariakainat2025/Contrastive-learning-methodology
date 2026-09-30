import argparse
import json
import os
import re

from tactic_to_stage import STAGE_ORDER, stage_scores_from_tactic_scores, tactics_to_stages

RESULTS_DIR = '/csse/research/contructive-learning/CAM-LDS/results'
REQUIRED_ONE_OF = {'Escalate Privilege', 'Internal Reconnaissance', 'Move Laterally', 'Maintain Persistence'}
MAX_STAGES_PER_STEP = 4

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

RUN_GROUP_RE = re.compile(r'^(.+)-(\d+)$')


def run_group_of(step_name):
    m = RUN_GROUP_RE.match(step_name)
    return m.group(1) if m else step_name


def step_sort_key(step_name):
    m = re.search(r'(\d+)$', step_name)
    return (int(m.group(1)) if m else 0, step_name)


def predicted_stages_for_step(scores):
    stage_scores = stage_scores_from_tactic_scores(scores)
    ranked = sorted(stage_scores.items(), key=lambda x: -x[1])
    return {'kept': ranked[:MAX_STAGES_PER_STEP], 'all_stages': ranked}


def apply_reasoning(raw_predicted):


    cleaned = []
    for step_name, pred in raw_predicted:
        cleaned.append({'step': step_name, 'kept': list(pred['kept']),
                         'all_stages': list(pred['all_stages']), 'dropped': []})
    return cleaned


def check_completeness(cleaned):
    all_stages = {s for row in cleaned for s, _ in row['kept']}
    complete = 'Initial Compromise' in all_stages and 'Establish Foothold' in all_stages\
        and bool(all_stages & REQUIRED_ONE_OF)
    return complete, all_stages


def build_true_path(results):
    return [{'step': r['file'], 'stages': tactics_to_stages(r['true_tactics'])} for r in results]


def print_stage_table(group_name, cleaned, true_path, max_stages):
    col_file  = 16
    col_true  = 22
    col_score = 12
    col_found = 7
    print()
    print('  ── Reconstructed Stage Path: {}  ({} steps, max {} stages/step) ──'.format(
        group_name, len(cleaned), max_stages))
    print('  IC=Initial Compromise  EF=Establish Foothold  EP=Escalate Privilege  IR=Internal Reconnaissance '
          'ML=Move Laterally  MP=Maintain Persistence  CM=Complete Mission')
    header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} ' +\
                 ' '.join(['{:<' + str(col_score) + '}'] * max_stages) + ' {:<' + str(col_found) + '} {:<6} {}'
    print(header_fmt.format('#', 'File', 'True stage(s)',
                             *['#{}'.format(j + 1) for j in range(max_stages)], 'Found', 'Wrong', 'Missing in top {} stages'.format(max_stages)))
    print('  ' + '-' * (4 + col_file + col_true + col_score * max_stages + col_found + 6 + max_stages + 4))

    n_wrong = 0
    for i, (row, true_row) in enumerate(zip(cleaned, true_path), 1):
        kept = row['kept']
        cols = []
        for j in range(max_stages):
            cols.append('{} ({:.2f})'.format(sid(kept[j][0]), kept[j][1]) if j < len(kept) else '')
        true_str = ', '.join(sid(s) for s in true_row['stages']) or '(none)'
        kept_names = {s for s, _ in kept}
        true_set = set(true_row['stages'])
        n_found = len(kept_names & true_set)
        n_true = len(true_set)
        found_str = '{}/{}'.format(n_found, n_true) if n_true else '-'
        missing = sorted(true_set - kept_names, key=STAGE_ORDER.index)
        missing_str = ','.join(sid(s) for s in missing)
        wrong = not (kept_names & true_set)
        if wrong:
            n_wrong += 1
        fname = row['step'] if len(row['step']) <= col_file else row['step'][:col_file - 3] + '...'
        print(header_fmt.format(i, fname, true_str, *cols, found_str, 'WRONG' if wrong else '', missing_str))

    return n_wrong


def process_run(group_name, group_results):
    results = sorted(group_results, key=lambda r: step_sort_key(r['file']))
    raw_predicted = [(r['file'], predicted_stages_for_step(r['scores'])) for r in results]
    cleaned = apply_reasoning(raw_predicted)
    complete, stage_set = check_completeness(cleaned)
    true_path = build_true_path(results)

    n_wrong = print_stage_table(group_name, cleaned, true_path, MAX_STAGES_PER_STEP)

    return {
        'run'           : group_name,
        'predicted_path': cleaned,
        'true_path'     : true_path,
        'stages_reached': sorted(stage_set, key=STAGE_ORDER.index),
        'complete'      : complete,
        'n_wrong_steps' : n_wrong,
        'n_steps'       : len(cleaned),
    }


def main(results_path, run_filter=None):
    with open(results_path) as f:
        data = json.load(f)

    groups = {}
    for r in data['results']:
        groups.setdefault(run_group_of(r['file']), []).append(r)

    names = sorted(groups)
    if run_filter:
        names = [n for n in names if n == run_filter]

    run_reports = [process_run(name, groups[name]) for name in names]

    n = len(run_reports)
    n_complete = sum(1 for r in run_reports if r['complete'])
    total_wrong = sum(r['n_wrong_steps'] for r in run_reports)
    total_steps = sum(r['n_steps'] for r in run_reports)

    if n > 1:
        print()
        print('  ' + '=' * 70)
        print('  Summary across {} run(s)'.format(n))
        print('  Runs passing completeness check : {}/{}'.format(n_complete, n))
        print('  Wrong steps (per-step, no true stage kept) : {}/{}'.format(total_wrong, total_steps))

    out = {
        'results_path'  : results_path,
        'runs'          : run_reports,
        'n_complete'    : n_complete,
        'n_runs'        : n,
    }
    out_path = os.path.join(RESULTS_DIR, 'attack_path_' + os.path.basename(results_path))
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print()
    print('  Saved -> {}'.format(out_path))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('results_file', type=str,
                     help='Path to a saved camlds test-results JSON, e.g. '
                          'results/camlds_wide_test_results_scenario7.json')
    ap.add_argument('--run', type=str, default=None,
                     help='Only show this one run group (exact match), e.g. --run 2_cron')
    args = ap.parse_args()
    main(args.results_file, args.run)
