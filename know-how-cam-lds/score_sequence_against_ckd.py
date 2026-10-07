"""score_sequence_against_ckd.py

Scores CAM-LDS sequence file(s) against the same gIoC cluster database
(meanshift_clustered_phrases.json / meanshift_cluster_key_vectors.json) used
by benigntag_paral_gpu.py to score the benign dataset -- same model, same
clusters, same similarity scoring, just applied to CAM-LDS's own sequence
sentences directly instead of raw DARPA log lines (these sentences are
already clean subject-verb-object text, e.g.
"/usr/bin/zmcontroller connect internal network address.", so no log parsing
step is needed first).

Input: a sequence_*.json file like
  CAM-LDS/CAM-LDS/sequences/collection/T1005-000/sequence_2_cron-15_videoserver.json
  { "tactic": ..., "technique": ..., "step": ..., "host": ...,
    "n_triples": N, "sequence": ["sentence 1", "sentence 2", ...] }

Two modes:
  --file <one sequence file>          score just that file
  --scan-dir <dir> --scenarios 2,3,4,6,7   score every sequence file under
                                            that dir whose filename matches
                                            sequence_<scenario>_* or
                                            sequence_<scenario>-* (scenarios
                                            4 and 7 use a hyphen, 2/3/6 use
                                            an underscore)

Output: a sequence_*_ckd_scored.json next to each input file (original
fields + a "scored" list with index/text/tech_scores/anomaly_score per
sentence), and in --scan-dir mode also one combined
ckd_scenario_summary.tsv with one row per file: tactic, technique, step,
host, n_triples, max_anomaly_score, mean_anomaly_score.
"""

import os
import re
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWHOW_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402')
if KNOWHOW_DIR not in sys.path:
    sys.path.insert(0, KNOWHOW_DIR)

from knowhow.benigntag_paral_gpu import TaggerGPU  # noqa: E402


def load_tagger(model_path=None, clusters_path=None, cluster_keys_path=None,
                 patterns_path=None, names_path=None, score_with_clustering=False):
    import torch

    model_path        = model_path        or os.path.join(KNOWHOW_DIR, 'technique-embedding-128.model')
    clusters_path     = clusters_path     or os.path.join(KNOWHOW_DIR, 'meanshift_clustered_phrases.json')
    cluster_keys_path = cluster_keys_path or os.path.join(KNOWHOW_DIR, 'meanshift_cluster_key_vectors.json')
    patterns_path     = patterns_path     or os.path.join(KNOWHOW_DIR, 'patterns.ini')
    names_path        = names_path        or os.path.join(
        PROJECT_ROOT, 'output', 'theia', 'parsed', 'names.json')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    tagger = TaggerGPU(
        model_path, clusters_path, cluster_keys_path, patterns_path, names_path,
        score_with_clustering, device=device,
    )
    tagger.load_resources()
    return tagger


def score_sequence_file(tagger, input_file, top_keys=5, score_with_clustering=False):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for i, text in enumerate(data.get('sequence', [])):
        emb = tagger._encode_string(text)
        if emb is None:
            results.append({'index': i, 'text': text, 'tech_scores': {}, 'anomaly_score': 0.0})
            continue

        if score_with_clustering:
            closest = tagger._find_closest_clusters(emb, top_n=3)
            scores = tagger._calculate_similarities(emb, closest)
        else:
            scores = tagger._calculate_similarities(emb, tagger.cluster_centers.keys())

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_keys]
        tech_scores = {k: float(v) for k, v in top}
        anomaly_score = max(tech_scores.values()) if tech_scores else 0.0
        results.append({'index': i, 'text': text, 'tech_scores': tech_scores, 'anomaly_score': anomaly_score})

    return data, results


def write_scored_file(input_file, data, results, output_file=None):
    if output_file is None:
        base = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(os.path.dirname(os.path.abspath(input_file)), f'{base}_ckd_scored.json')
    out = dict(data)
    out['scored'] = results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return output_file


def run_single(input_file, output_file, tagger, top_keys, score_with_clustering):
    data, results = score_sequence_file(tagger, input_file, top_keys, score_with_clustering)
    out_path = write_scored_file(input_file, data, results, output_file)

    print(f'Technique : {data.get("technique")}  Step: {data.get("step")}  Host: {data.get("host")}')
    for r in results:
        top_str = '; '.join(f'{k}:{v:.2f}' for k, v in r['tech_scores'].items())
        print(f'  [{r["index"]}] anomaly_score={r["anomaly_score"]:.3f}  {r["text"]!r}  -> {top_str}')
    print(f'Written -> {out_path}')
    return out_path


def find_scenario_files(scan_dir, scenarios):
    patterns = []
    for s in scenarios:
        patterns.append(re.compile(rf'^sequence_{s}_'))
        patterns.append(re.compile(rf'^sequence_{s}-'))

    matches = []
    for root, _, files in os.walk(scan_dir):
        for fname in files:
            if not fname.endswith('.json') or fname.endswith('_ckd_scored.json'):
                continue
            if any(p.match(fname) for p in patterns):
                matches.append(os.path.join(root, fname))
    return sorted(matches)


def run_batch(scan_dir, scenarios, tagger, top_keys, score_with_clustering, summary_path):
    files = find_scenario_files(scan_dir, scenarios)
    print(f'Found {len(files):,} sequence files for scenarios {scenarios}')

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
    ap.add_argument('--file', default=None, help='a single sequence_*.json file to score')
    ap.add_argument('--scan-dir', default=None, help='score every matching sequence file under this dir')
    ap.add_argument('--scenarios', default='2,3,4,6,7', help='comma-separated scenario numbers (used with --scan-dir)')
    ap.add_argument('--summary', default=None, help='combined summary tsv path (default: <scan-dir>/ckd_scenario_summary.tsv)')
    ap.add_argument('--output', default=None, help='output path (only valid with --file)')
    ap.add_argument('--model', default=None)
    ap.add_argument('--clusters', default=None)
    ap.add_argument('--cluster-keys', default=None)
    ap.add_argument('--patterns', default=None)
    ap.add_argument('--names', default=None, help='only needed by TaggerGPU.load_resources(), unused for sequence text')
    ap.add_argument('--top-keys', type=int, default=5)
    ap.add_argument('-c', '--score-with-clustering', action='store_true')
    args = ap.parse_args()

    if not args.file and not args.scan_dir:
        ap.error('pass either --file or --scan-dir')

    tagger = load_tagger(args.model, args.clusters, args.cluster_keys,
                          args.patterns, args.names, args.score_with_clustering)

    if args.file:
        run_single(args.file, args.output, tagger, args.top_keys, args.score_with_clustering)
    else:
        scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]
        summary_path = args.summary or os.path.join(args.scan_dir, 'ckd_scenario_summary.tsv')
        run_batch(args.scan_dir, scenarios, tagger, args.top_keys, args.score_with_clustering, summary_path)
