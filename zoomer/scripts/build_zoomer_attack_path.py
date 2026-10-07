"""
ZOOMER equivalent of CAM-LDS's scripts/build_attack_path.py -- same table
format, same columns, reading from an already-saved ZOOMER results JSON
(Test_TTP_Recognition_Tactic.py), which already has per-file stage scores.
"""
import argparse
import json
import os
import re

RESULTS_DIR = '/csse/research/contructive-learning/CAM-LDS/zoomer/results'
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
STAGE_ORDER = list(STAGE_ABBREV.keys())


def sid(stage):
    return STAGE_ABBREV.get(stage, stage)


RUN_GROUP_RE = re.compile(r'^(.+)-(\d+)$')


def run_group_of(step_name):
    m = RUN_GROUP_RE.match(step_name)
    return m.group(1) if m else step_name


def step_sort_key(step_name):
    m = re.search(r'(\d+)$', step_name)
    return (int(m.group(1)) if m else 0, step_name)


def print_stage_table(group_name, rows, max_stages):
    col_file, col_true, col_score, col_found = 16, 22, 12, 7
    print()
    print('  -- ZOOMER -- Reconstructed Stage Path: {}  ({} steps, max {} stages/step) --'.format(
        group_name, len(rows), max_stages))
    print('  IC=Initial Compromise  EF=Establish Foothold  EP=Escalate Privilege  IR=Internal Reconnaissance '
          'ML=Move Laterally  MP=Maintain Persistence  CM=Complete Mission')
    header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} ' + \
                 ' '.join(['{:<' + str(col_score) + '}'] * max_stages) + ' {:<' + str(col_found) + '} {:<6} {}'
    print(header_fmt.format('#', 'File', 'True stage(s)',
                             *['#{}'.format(j + 1) for j in range(max_stages)], 'Found', 'Wrong', 'Missing in top {} stages'.format(max_stages)))
    print('  ' + '-' * (4 + col_file + col_true + col_score * max_stages + col_found + 6 + max_stages + 4))

    n_wrong = 0
    for i, row in enumerate(rows, 1):
        true_stages = row['true_stages']
        ranked = row['ranked'][:max_stages]
        cols = ['{} ({:.2f})'.format(sid(s), sc) for s, sc in ranked]
        true_str = ', '.join(sid(s) for s in true_stages) or '(none)'
        kept_names = {s for s, _ in ranked}
        true_set = set(true_stages)
        n_found = len(kept_names & true_set)
        n_true = len(true_set)
        found_str = '{}/{}'.format(n_found, n_true) if n_true else '-'
        missing = sorted(true_set - kept_names, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)
        missing_str = ','.join(sid(s) for s in missing)
        wrong = not (kept_names & true_set)
        if wrong:
            n_wrong += 1
        fname = row['file'] if len(row['file']) <= col_file else row['file'][:col_file - 3] + '...'
        print(header_fmt.format(i, fname, true_str, *cols, found_str, 'WRONG' if wrong else '', missing_str))

    return n_wrong


def main(results_path, run_filter=None):
    with open(results_path) as f:
        data = json.load(f)

    groups = {}
    for r in data['stage_results']:
        groups.setdefault(run_group_of(r['file']), []).append(r)

    names = sorted(groups)
    if run_filter:
        names = [n for n in names if n == run_filter]

    total_wrong = 0
    total_steps = 0
    for name in names:
        rows = sorted(groups[name], key=lambda r: step_sort_key(r['file']))
        rows = [
            {'file': r['file'], 'true_stages': r['true_stages'],
             'ranked': sorted([(x['stage'], x['score']) for x in r['ranked']], key=lambda t: -t[1])}
            for r in rows
        ]
        n_wrong = print_stage_table(name, rows, MAX_STAGES_PER_STEP)
        total_wrong += n_wrong
        total_steps += len(rows)

    if len(names) > 1:
        print()
        print('  ' + '=' * 70)
        print('  Summary across {} run(s)'.format(len(names)))
        print('  Wrong steps (per-step, no true stage kept) : {}/{}'.format(total_wrong, total_steps))
    else:
        print()
        print('  Wrong steps (per-step, no true stage kept) : {}/{}'.format(total_wrong, total_steps))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('results_file', type=str,
                     help='Path to a saved ZOOMER test-results JSON, e.g. '
                          'zoomer/results/ttp_recognition_tactic_scenario2.json')
    ap.add_argument('--run', type=str, default=None,
                     help='Only show this one run group (exact match), e.g. --run 2_cron')
    args = ap.parse_args()
    main(args.results_file, args.run)
