
import os
import re
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KNOWN_HOSTS = ('attacker', 'inetfw', 'videoserver', 'wazuh', 'reposerver', 'client',
               'corpdns', 'linuxshare', 'docker-log', 'docker', 'adminpc')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from score_sequence_against_ckd import load_tagger, score_sequence_file, write_scored_file

DEFAULT_SEQUENCES_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'sequences')


def sequence_instance_from_filename(filename):
    stem = filename[len('sequence_'):-len('.json')]
    for host in KNOWN_HOSTS:
        suffix = '_' + host
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def is_scenario_test_file(filename, scenario):
    scenario = str(scenario)
    instance = sequence_instance_from_filename(filename)
    return instance.startswith(scenario + '_') or instance.startswith(scenario + '-')


def find_scenario_sequence_files(sequences_dir, scenario):
    matches = []
    for root, _, files in os.walk(sequences_dir):
        for fname in files:
            if not fname.startswith('sequence_') or not fname.endswith('.json'):
                continue
            if fname.endswith('_ckd_scored.json'):
                continue
            if is_scenario_test_file(fname, scenario):
                matches.append(os.path.join(root, fname))
    return sorted(matches)


def run(scenario, sequences_dir=None, output_dir=None, top_keys=5, score_with_clustering=False,
        model_path=None, clusters_path=None, cluster_keys_path=None, patterns_path=None, names_path=None):

    sequences_dir = sequences_dir or DEFAULT_SEQUENCES_DIR
    files = find_scenario_sequence_files(sequences_dir, scenario)
    print(f'Scenario {scenario}: found {len(files):,} matching sequence files under {sequences_dir}')
    if not files:
        print('  Nothing matched -- check --sequences-dir, or that this scenario has sequence files.')
        return None

    tagger = load_tagger(model_path, clusters_path, cluster_keys_path, patterns_path, names_path,
                          score_with_clustering)

    output_dir = output_dir or THIS_DIR
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f'ckd_scenario_{scenario}_summary.tsv')

    with open(summary_path, 'w', encoding='utf-8') as fsum:
        fsum.write('\t'.join([
            'tactic', 'technique', 'step', 'host', 'n_triples',
            'max_anomaly_score', 'mean_anomaly_score', 'file',
        ]) + '\n')

        for i, fpath in enumerate(files, 1):
            data, results = score_sequence_file(tagger, fpath, top_keys, score_with_clustering)
            write_scored_file(fpath, data, results)

            scores = [r['anomaly_score'] for r in results]
            max_s = max(scores) if scores else 0.0
            mean_s = (sum(scores) / len(scores)) if scores else 0.0

            fsum.write('\t'.join([
                str(data.get('tactic', '')), str(data.get('technique', '')),
                str(data.get('step', '')), str(data.get('host', '')),
                str(data.get('n_triples', '')),
                f'{max_s:.3f}', f'{mean_s:.3f}', fpath,
            ]) + '\n')

            if i % 25 == 0 or i == len(files):
                print(f'  scored {i:,}/{len(files):,}')

    print(f'Summary written -> {summary_path}')
    return summary_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, help='scenario number, e.g. 4')
    ap.add_argument('--sequences-dir', default=None, help=f'default: {DEFAULT_SEQUENCES_DIR}')
    ap.add_argument('--output-dir', default=None, help='default: this script\'s own folder')
    ap.add_argument('--top-keys', type=int, default=5)
    ap.add_argument('-c', '--score-with-clustering', action='store_true')
    ap.add_argument('--model', default=None)
    ap.add_argument('--clusters', default=None)
    ap.add_argument('--cluster-keys', default=None)
    ap.add_argument('--patterns', default=None)
    ap.add_argument('--names', default=None)
    args = ap.parse_args()
    run(args.scenario, args.sequences_dir, args.output_dir, args.top_keys, args.score_with_clustering,
        args.model, args.clusters, args.cluster_keys, args.patterns, args.names)
