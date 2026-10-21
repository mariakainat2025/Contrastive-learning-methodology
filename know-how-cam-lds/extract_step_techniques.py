
import ast
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
from score_scenario_against_ckd import sequence_instance_from_filename
from mitre_tactic_to_stage import load_tech2tac, mitre_tactic_to_stage

DEFAULT_SEQUENCES_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'sequences')
DEFAULT_CLUSTERS_PATH = os.path.join(
    PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402', 'meanshift_clustered_phrases.json')
DEFAULT_TECH2TAC_PATH = os.path.join(
    PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402', 'tech2tac.txt')

SCOPE_TECHNIQUES = {
    'T1003', 'T1016', 'T1018', 'T1021', 'T1033', 'T1036', 'T1041', 'T1046',
    'T1053', 'T1056', 'T1057', 'T1059', 'T1070', 'T1071', 'T1072', 'T1078',
    'T1082', 'T1083', 'T1098', 'T1105', 'T1110', 'T1115', 'T1176', 'T1201',
    'T1205', 'T1218', 'T1219', 'T1222', 'T1543', 'T1546', 'T1547', 'T1548',
    'T1556', 'T1564', 'T1566', 'T1574', 'T1039', 'T1040', 'T1120', 'T1133',
    'T1136', 'T1190', 'T1213', 'T1486', 'T1497', 'T1204', 'T1595',
}


def load_gioc_points(clusters_path, device, scope=SCOPE_TECHNIQUES):
    import torch
    with open(clusters_path, 'r', encoding='utf-8') as f:
        clustered = json.load(f)

    points = []
    for cluster in clustered.values():
        for key, vec in cluster['points']:
            technique, _, phrase = key.partition('*****')
            base = technique.split('.')[0].split('-')[0]
            if scope is not None and base not in scope:
                continue
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


def closest_gioc_technique_scores(emb, gioc_points):
    import torch
    from torch.nn.functional import cosine_similarity
    best = {}
    for technique, phrase, vec in gioc_points:
        sim = cosine_similarity(emb.unsqueeze(0), vec.unsqueeze(0)).item()
        if technique not in best or sim > best[technique][0]:
            best[technique] = (sim, phrase)
    return best


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
    seen = set()
    tac_scores = []
    for technique, _phrase, score in tech_matches:
        for tac in tech2tac.get(technique, []):
            if tac not in seen:
                seen.add(tac)
                tac_scores.append((tac, score))
    return sorted(tac_scores, key=lambda x: x[1], reverse=True)[:top_n]


def tactics_to_stage_scores(tac_scores):
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


def _readable_phrase(phrase):
    try:
        parts = ast.literal_eval(phrase)
    except (ValueError, SyntaxError):
        return phrase
    if not isinstance(parts, tuple):
        return phrase
    non_empty = [str(p) for p in parts if p]
    return ' -> '.join(non_empty) if non_empty else phrase


def print_event_table(rows, top_keys):
    idx_width = max((len(str(r[0])) for r in rows), default=1)
    header = f'[{"#".rjust(idx_width)}] Event  (then top {top_keys} matched gIoC points, one per line, most similar first)'
    print(header)
    print('-' * max(len(header), 60))
    indent = ' ' * (idx_width + 3)
    for idx, event, tech_matches in rows:
        print(f'[{str(idx).rjust(idx_width)}] Event: {event}')
        if tech_matches is None:
            print(f'{indent}(no embeddable words)')
            continue
        for rank, (technique, phrase, score) in enumerate(tech_matches, 1):
            readable = _readable_phrase(phrase)
            print(f'{indent}{rank:>2}. {technique:<9} {score:.4f}  {readable}')


def run(step, sequences_dir=None, top_keys=5, top_techniques=6, top_tactics=4, score_with_clustering=False,
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
        best_match_per_tech = {}
        all_events_per_tech = defaultdict(dict)
        n_events = 0

        for i, text in enumerate(data.get('sequence', [])):
            emb = tagger._encode_string(text)
            if emb is None:
                rows.append((i, text, None))
                continue
            n_events += 1

            tech_matches = closest_gioc_events(emb, gioc_points, top_n=top_keys)
            rows.append((i, text, tech_matches))

            for technique, (score, phrase) in closest_gioc_technique_scores(emb, gioc_points).items():
                all_events_per_tech[technique][i] = (score, phrase)
                if score > best_score_per_tech[technique]:
                    best_score_per_tech[technique] = score
                    best_match_per_tech[technique] = (i, text, phrase)

        print_event_table(rows, top_keys)

        ranked_tech = sorted(best_score_per_tech.items(), key=lambda x: x[1], reverse=True)[:top_techniques]
        top_tacs = techniques_to_top_tactics(
            [(t, '', s) for t, s in ranked_tech], tech2tac, top_tactics)
        stage_scores = tactics_to_stage_scores(top_tacs)

        print(f'Techniques -> {_pair_list(ranked_tech)}')
        print(f'  (proof: this technique\'s own score in EVERY event of this step -- {n_events} events -- '
              f'union just keeps the highest of these)')
        for technique, score in ranked_tech:
            per_event_str = '; '.join(
                f'event[{i}]={all_events_per_tech[technique][i][0]:.4f}'
                for i in sorted(all_events_per_tech[technique]))
            (best_ev_idx, _ev_text, best_phrase) = best_match_per_tech[technique]
            print(f'    {technique}: {per_event_str}  -> best=event[{best_ev_idx}] ({score:.2f}) matched gIoC {best_phrase!r}')
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
    ap.add_argument('--top-techniques', type=int, default=6,
                     help='how many techniques survive (by best score) into the tactic/stage rollup -- '
                          'kept wider than top-tactics so a handful of techniques sharing few tactics '
                          'still leaves enough distinct tactics to reach top-tactics')
    ap.add_argument('--top-tactics', type=int, default=4, help='how many top tactics to keep per event (step 2)')
    ap.add_argument('-c', '--score-with-clustering', action='store_true')
    ap.add_argument('--model', default=None)
    ap.add_argument('--clusters', default=None)
    ap.add_argument('--cluster-keys', default=None)
    ap.add_argument('--patterns', default=None)
    ap.add_argument('--names', default=None)
    ap.add_argument('--tech2tac', default=None)
    args = ap.parse_args()
    run(args.step, args.sequences_dir, args.top_keys, args.top_techniques, args.top_tactics,
        args.score_with_clustering, args.model, args.clusters, args.cluster_keys, args.patterns,
        args.names, args.tech2tac)
