
import os
import sys
import json
import argparse
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from score_sequence_against_ckd import load_tagger
from score_scenario_against_ckd import find_scenario_sequence_files, sequence_instance_from_filename
from extract_step_techniques import load_gioc_points, closest_gioc_events, techniques_to_top_tactics, tactics_to_stage_scores
from mitre_tactic_to_stage import load_tech2tac

DEFAULT_SEQUENCES_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'sequences')
DEFAULT_CLUSTERS_PATH = os.path.join(
    PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402', 'meanshift_clustered_phrases.json')
DEFAULT_TECH2TAC_PATH = os.path.join(
    PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402', 'tech2tac.txt')
STEP_TACTICS_PATH = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'technique', 'step_tactics.json')


def base_technique(t):
    return t.split('.')[0].split('-')[0]


def group_files_by_step(files):
    by_step = {}
    for fpath in files:
        step = sequence_instance_from_filename(os.path.basename(fpath))
        by_step.setdefault(step, fpath)
    return by_step


def load_scenario_ground_truth(scenario, step_names):
    with open(STEP_TACTICS_PATH) as f:
        report = json.load(f)
    step_names = set(step_names)
    true_techniques = set()
    per_step_true = {}
    for row in report:
        if row['step'] not in step_names:
            continue
        techs = {base_technique(occ['technique']) for occ in row['occurrences']}
        per_step_true[row['step']] = techs
        true_techniques.update(techs)
    return true_techniques, per_step_true


def union_techniques_for_step(tagger, gioc_points, fpath, top_keys):
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    step_techs = defaultdict(int)
    n_events = 0
    for text in data.get('sequence', []):
        emb = tagger._encode_string(text)
        if emb is None:
            continue
        n_events += 1
        for technique, _phrase, _score in closest_gioc_events(emb, gioc_points, top_n=top_keys):
            step_techs[base_technique(technique)] += 1
    return step_techs, n_events


def run(scenario, sequences_dir=None, top_keys=5, top_tactics=4, top_techniques_out=10,
        model_path=None, clusters_path=None, cluster_keys_path=None, patterns_path=None,
        names_path=None, tech2tac_path=None, score_with_clustering=False):

    sequences_dir = sequences_dir or DEFAULT_SEQUENCES_DIR
    files = find_scenario_sequence_files(sequences_dir, scenario)
    by_step = group_files_by_step(files)
    print(f'Scenario {scenario}: {len(by_step):,} distinct steps ({len(files):,} sequence files before dedup)')
    if not by_step:
        print('  Nothing matched -- check --sequences-dir, or that this scenario has sequence files.')
        return None

    tagger = load_tagger(model_path, clusters_path, cluster_keys_path, patterns_path, names_path,
                          score_with_clustering)
    gioc_points = load_gioc_points(clusters_path or DEFAULT_CLUSTERS_PATH, tagger.device)
    tech2tac = load_tech2tac(tech2tac_path or DEFAULT_TECH2TAC_PATH)
    print(f'Loaded {len(gioc_points):,} gIoC events from the CKD database')

    scenario_step_count = defaultdict(int)
    scenario_event_count = defaultdict(int)
    per_step_predicted = {}

    for i, (step, fpath) in enumerate(sorted(by_step.items()), 1):
        step_techs, n_events = union_techniques_for_step(tagger, gioc_points, fpath, top_keys)
        per_step_predicted[step] = set(step_techs)
        for tech, n_hits in step_techs.items():
            scenario_step_count[tech] += 1
            scenario_event_count[tech] += n_hits
        if i % 10 == 0 or i == len(by_step):
            print(f'  scored {i:,}/{len(by_step):,} steps')

    ranked_techniques = sorted(
        scenario_event_count.items(),
        key=lambda kv: (kv[1], scenario_step_count[kv[0]]),
        reverse=True,
    )[:top_techniques_out]

    top_tacs = techniques_to_top_tactics(
        [(t, '', float(n_events)) for t, n_events in ranked_techniques], tech2tac, top_tactics)
    stage_scores = tactics_to_stage_scores(top_tacs)

    true_techniques, per_step_true = load_scenario_ground_truth(scenario, by_step.keys())

    predicted_set = {t for t, _ in ranked_techniques}
    correct = sorted(predicted_set & true_techniques)
    missing = sorted(true_techniques - predicted_set)
    extra = sorted(predicted_set - true_techniques)

    print()
    print(f'── Scenario {scenario}: top-{top_techniques_out} techniques, flat union across {len(by_step)} steps ──')
    print(f'{"Technique":<12}{"Events":>8}{"Steps":>7}  (out of {len(by_step)} steps)')
    for tech, n_events in ranked_techniques:
        flag = '  (true positive)' if tech in true_techniques else ''
        print(f'{tech:<12}{n_events:>8}{scenario_step_count[tech]:>7}{flag}')

    print()
    print(f'Top tactics -> ' + '; '.join(f'{t}:{s:.0f}' for t, s in top_tacs))
    print(f'Stage(s)    -> ' + '; '.join(f'{t}:{s:.0f}' for t, s in stage_scores))

    print()
    print(f'── vs ground truth (base techniques true anywhere in these {len(by_step)} steps) ──')
    print(f'True techniques total : {len(true_techniques)}  -> {sorted(true_techniques)}')
    print(f'Correct (predicted & true)  : {len(correct)}/{len(true_techniques)}  -> {correct}')
    print(f'Missing (true, not predicted): {len(missing)}  -> {missing}')
    print(f'Extra (predicted, not true)  : {len(extra)}  -> {extra}')

    return {
        'scenario': scenario, 'n_steps': len(by_step),
        'ranked_techniques': ranked_techniques, 'top_tactics': top_tacs, 'stages': stage_scores,
        'true_techniques': sorted(true_techniques), 'correct': correct, 'missing': missing, 'extra': extra,
        'per_step_predicted': {k: sorted(v) for k, v in per_step_predicted.items()},
        'per_step_true': {k: sorted(v) for k, v in per_step_true.items()},
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, help='scenario number ("4") or run-group ("2_cron")')
    ap.add_argument('--sequences-dir', default=None)
    ap.add_argument('--top-keys', type=int, default=5, help='top-N gIoC matches kept per event (union input)')
    ap.add_argument('--top-tactics', type=int, default=4)
    ap.add_argument('--top-techniques-out', type=int, default=7, help='how many top-ranked techniques to report for the scenario (e.g. 3 or 7)')
    ap.add_argument('--model', default=None)
    ap.add_argument('--clusters', default=None)
    ap.add_argument('--cluster-keys', default=None)
    ap.add_argument('--patterns', default=None)
    ap.add_argument('--names', default=None)
    ap.add_argument('--tech2tac', default=None)
    ap.add_argument('-c', '--score-with-clustering', action='store_true')
    ap.add_argument('--output', default=None, help='optional path to also write the result as JSON')
    args = ap.parse_args()

    result = run(args.scenario, args.sequences_dir, args.top_keys, args.top_tactics, args.top_techniques_out,
                 args.model, args.clusters, args.cluster_keys, args.patterns, args.names, args.tech2tac,
                 args.score_with_clustering)

    if result and args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f'\nWritten -> {args.output}')
