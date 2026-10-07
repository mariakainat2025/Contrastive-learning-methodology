"""extract_step_techniques.py

Give it one step id (e.g. "2_cron-15" or "4-16") and it finds every matching
sequence_*.json file under CAM-LDS/CAM-LDS/sequences/ (a step can live under
more than one technique/tactic folder if it has multiple true labels), and
for every event (sentence) in it, finds the single closest gIoC entry in the
CKD cluster database (meanshift_clustered_phrases.json -- the raw
"TECHID*****(subject, verb, object)" points, not just the per-cluster
technique key vectors) via direct cosine similarity, and prints that matched
gIoC event + its technique + its score. Per event the flow is then exactly:

  1. extract techniques -- top-N closest gIoC matches (--top-keys, default 5)
  2. techniques -> tactics -- each matched technique's score is added to every
     tactic it belongs to (tech2tac.txt), then the top-4 tactics by summed
     score are kept (--top-tactics, default 4)
  3. tactics -> stage -- those top-4 tactics' scores are summed per stage
     using ZOOMER's own tactic_to_stage.py vocabulary (mitre_tactic_to_stage.py
     in this folder), so the stage names match exactly what ZOOMER reports.

Usage:
  python3 extract_step_techniques.py --step 2_cron-15
  python3 extract_step_techniques.py --step 4-16 --top-keys 3
"""

import os
import sys
import json
import argparse
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from score_sequence_against_ckd import load_tagger  # noqa: E402
from score_scenario_against_ckd import sequence_instance_from_filename  # noqa: E402
from mitre_tactic_to_stage import load_tech2tac, mitre_tactic_to_stage  # noqa: E402

DEFAULT_SEQUENCES_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'CAM-LDS', 'sequences')
DEFAULT_CLUSTERS_PATH = os.path.join(
    PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402', 'meanshift_clustered_phrases.json')
DEFAULT_TECH2TAC_PATH = os.path.join(
    PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402', 'tech2tac.txt')


def load_gioc_points(clusters_path, device):
    import torch
    with open(clusters_path, 'r', encoding='utf-8') as f:
        clustered = json.load(f)

    points = []  # list of (technique, phrase_text, vector)
    for cluster in clustered.values():
        for key, vec in cluster['points']:
            technique, _, phrase = key.partition('*****')
            points.append((technique, phrase, torch.tensor(vec, dtype=torch.float32, device=device)))
    return points


def closest_gioc_events(emb, gioc_points, top_n=3):
    import torch
    from torch.nn.functional import cosine_similarity
    scored = []
    for technique, phrase, vec in gioc_points:
        sim = cosine_similarity(emb.unsqueeze(0), vec.unsqueeze(0)).item()
        scored.append((technique, phrase, sim))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


def find_step_files(sequences_dir, step):
    matches = []
    for root, _, files in os.walk(sequences_dir):
        for fname in files:
            if not fname.startswith('sequence_') or not fname.endswith('.json'):
                continue
            if fname.endswith('_ckd_scored.json'):
                continue
            if sequence_instance_from_filename(fname) == step:
                matches.append(os.path.join(root, fname))
    return sorted(matches)


def techniques_to_top_tactics(tech_matches, tech2tac, top_n=4):
    """Step 2: sum each matched technique's score into every tactic it
    belongs to, keep the top-N tactics by summed score."""
    tac_scores = defaultdict(float)
    for technique, _phrase, score in tech_matches:
        for tac in tech2tac.get(technique, []):
            tac_scores[tac] += score
    return sorted(tac_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def tactics_to_stage_scores(tac_scores):
    """Step 3: a stage's score is the BEST (max) of its tactics' scores, not
    the sum -- if two kept tactics land on the same stage, only the higher
    one counts."""
    stage_scores = defaultdict(float)
    for tac, score in tac_scores:
        stage = mitre_tactic_to_stage(tac)
        if stage and score > stage_scores[stage]:
            stage_scores[stage] = score
    return sorted(stage_scores.items(), key=lambda x: x[1], reverse=True)


def _fit(text, width):
    text = str(text)
    if len(text) <= width:
        return text.ljust(width)
    return text[:width - 1] + '…'


def _pair_list(pairs, fmt='{}:{:.2f}'):
    return '; '.join(fmt.format(k, v) for k, v in pairs)


def print_event_table(rows, top_keys):
    """Two lines per event, so the long technique match text gets the full
    terminal width instead of being squeezed into a narrow column:
      [#] Event: <event text>
          <technique match line>
    """
    idx_width = max((len(str(r[0])) for r in rows), default=1)
    header = f'[{"#".rjust(idx_width)}] Event  (then: top {top_keys} matched gIoC events, technique:score)'
    print(header)
    print('-' * max(len(header), 60))
    for idx, event, tech_str in rows:
        print(f'[{str(idx).rjust(idx_width)}] Event: {event}')
        print(f'{" " * (idx_width + 3)}{tech_str}')


def run(step, sequences_dir=None, top_keys=5, top_tactics=4, score_with_clustering=False,
        model_path=None, clusters_path=None, cluster_keys_path=None, patterns_path=None, names_path=None,
        tech2tac_path=None):

    sequences_dir = sequences_dir or DEFAULT_SEQUENCES_DIR
    files = find_step_files(sequences_dir, step)
    if not files:
        print(f'No sequence files found for step "{step}" under {sequences_dir}')
        return {}

    tagger = load_tagger(model_path, clusters_path, cluster_keys_path, patterns_path, names_path,
                          score_with_clustering)
    gioc_points = load_gioc_points(clusters_path or DEFAULT_CLUSTERS_PATH, tagger.device)
    tech2tac = load_tech2tac(tech2tac_path or DEFAULT_TECH2TAC_PATH)
    print(f'Loaded {len(gioc_points):,} gIoC events from the CKD database')

    file_results = []

    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f'\n{os.path.relpath(fpath, sequences_dir)}  (tactic={data.get("tactic")}, technique={data.get("technique")})')

        rows = []
        best_score_per_tech = defaultdict(float)

        for i, text in enumerate(data.get('sequence', [])):
            emb = tagger._encode_string(text)
            if emb is None:
                rows.append((i, text, '(no embeddable words)'))
                continue

            tech_matches = closest_gioc_events(emb, gioc_points, top_n=top_keys)  # 1. top-4 techniques

            tech_str = '; '.join(f'{t}:{s:.2f} {p}' for t, p, s in tech_matches)
            rows.append((i, text, tech_str))

            for technique, _phrase, score in tech_matches:
                if score > best_score_per_tech[technique]:
                    best_score_per_tech[technique] = score

        print_event_table(rows, top_keys)

        # bottom of the file: technique -> tactic -> stage, from only the
        # top-4 techniques (by best score) found across this file's events --
        # not every distinct technique that ever appeared in any event's top-4
        ranked_tech = sorted(best_score_per_tech.items(), key=lambda x: x[1], reverse=True)[:top_keys]
        top_tacs = techniques_to_top_tactics(
            [(t, '', s) for t, s in ranked_tech], tech2tac, top_tactics)          # 2. top-4 tactics
        stage_scores = tactics_to_stage_scores(top_tacs)                         # 3. stages

        print(f'Techniques -> {_pair_list(ranked_tech)}')
        print(f'Tactics    -> {_pair_list(top_tacs)}')
        print(f'Stage(s)   -> {_pair_list(stage_scores)}')

        file_results.append({'file': fpath, 'techniques': dict(ranked_tech),
                              'tactics': dict(top_tacs), 'stages': dict(stage_scores)})

    return file_results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, help='step id, e.g. 2_cron-15 or 4-16')
    ap.add_argument('--sequences-dir', default=None)
    ap.add_argument('--top-keys', type=int, default=5, help='how many techniques to extract per event (step 1)')
    ap.add_argument('--top-tactics', type=int, default=4, help='how many top tactics to keep per event (step 2)')
    ap.add_argument('-c', '--score-with-clustering', action='store_true')
    ap.add_argument('--model', default=None)
    ap.add_argument('--clusters', default=None)
    ap.add_argument('--cluster-keys', default=None)
    ap.add_argument('--patterns', default=None)
    ap.add_argument('--names', default=None)
    ap.add_argument('--tech2tac', default=None)
    args = ap.parse_args()
    run(args.step, args.sequences_dir, args.top_keys, args.top_tactics, args.score_with_clustering,
        args.model, args.clusters, args.cluster_keys, args.patterns, args.names, args.tech2tac)
