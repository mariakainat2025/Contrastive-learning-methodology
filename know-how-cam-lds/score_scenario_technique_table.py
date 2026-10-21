
import os
import re
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from score_sequence_against_ckd import load_tagger
from score_scenario_against_ckd import find_scenario_sequence_files
from score_scenario_union_techniques import group_files_by_step, base_technique, STEP_TACTICS_PATH
from extract_step_techniques import load_gioc_points, closest_gioc_technique_scores
from mitre_tactic_to_stage import load_tech2tac, mitre_tactic_to_stage, TACTIC_TO_STAGE

DEFAULT_SEQUENCES_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'sequences')
DEFAULT_CLUSTERS_PATH = os.path.join(
    PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402', 'meanshift_clustered_phrases.json')
DEFAULT_TECH2TAC_PATH = os.path.join(
    PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402', 'tech2tac.txt')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')

DEFENSE_EVASION_SPLIT = ('stealth', 'defense_impairment')

STAGE_ABBREV = {
    'Initial Compromise':      'IC',
    'Establish Foothold':      'EF',
    'Escalate Privilege':      'EP',
    'Internal Reconnaissance': 'IR',
    'Move Laterally':          'ML',
    'Maintain Persistence':    'MP',
    'Complete Mission':        'CM',
}

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


def sid(stage):
    return STAGE_ABBREV.get(stage, stage)


def tid(tactic):
    return TACTIC_IDS.get(tactic, tactic)


def mitre_tactic_to_camlds(tactic_name):
    if tactic_name == 'Defense Evasion':
        return list(DEFENSE_EVASION_SPLIT)
    if tactic_name == 'Resource Development':
        return []
    return [tactic_name.lower().replace(' ', '_')]


def load_step_ground_truth(step_names):
    with open(STEP_TACTICS_PATH) as f:
        report = json.load(f)
    step_names = set(step_names)
    true_techniques, true_tactics = {}, {}
    for row in report:
        if row['step'] not in step_names:
            continue
        techs, tacs = set(), set()
        for occ in row['occurrences']:
            techs.add(base_technique(occ['technique']))
            tacs.update(occ['tactics'])
        true_techniques[row['step']] = techs
        true_tactics[row['step']] = tacs
    return true_techniques, true_tactics


def predict_step(tagger, gioc_points, fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    best_score = defaultdict(float)
    for text in data.get('sequence', []):
        emb = tagger._encode_string(text)
        if emb is None:
            continue
        for technique, (score, _phrase) in closest_gioc_technique_scores(emb, gioc_points).items():
            tech = base_technique(technique)
            if score > best_score[tech]:
                best_score[tech] = score
    return sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)


def step_sort_key(step_name):
    m = re.search(r'(\d+)$', step_name)
    return (int(m.group(1)) if m else 0, step_name)


def expand_ranked(ranked_tuples, key_to_labels):
    seen, out = set(), []
    for label, score in ranked_tuples:
        for target in key_to_labels.get(label, []):
            if target not in seen:
                seen.add(target)
                out.append((target, score))
    return out


def classify(true_set, top_set, top_n):
    total = len(true_set)
    if total == 0:
        return None
    found = len(true_set & top_set)
    if total <= top_n:
        if found == total:
            return 'correct'
        if found == 0:
            return 'wrong'
        return 'partial'
    wrong = len(top_set - true_set)
    if found == 0:
        return 'wrong'
    if wrong == 0:
        return 'correct'
    return 'partial'


def top_k_accuracy(rows, true_key, ks=(1, 3)):
    if not rows:
        return {k: 0.0 for k in ks}
    return {k: sum(1 for r in rows if r[true_key] <= {lbl for lbl, _ in r['ranked'][:k]}) / len(rows) for k in ks}


def accuracy_at_n_true(rows, true_key):
    if not rows:
        return 0.0
    n_correct = sum(1 for r in rows if r[true_key] <= {lbl for lbl, _ in r['ranked'][:len(r[true_key])]})
    return n_correct / len(rows)


def lrap_aupr(rows, true_key, all_labels):
    if not rows or not all_labels:
        return 0.0, 0.0
    y_true = np.array([[1 if lbl in r[true_key] else 0 for lbl in all_labels] for r in rows])
    y_score = np.array([[dict(r['ranked']).get(lbl, 0.0) for lbl in all_labels] for r in rows])
    lrap = label_ranking_average_precision_score(y_true, y_score)
    valid_cols = y_true.sum(axis=0) > 0
    aupr = average_precision_score(y_true[:, valid_cols], y_score[:, valid_cols], average='macro') if valid_cols.any() else 0.0
    return lrap, aupr


def topn_any_all_match(rows, top_n):
    if not rows:
        return 0.0, 0.0
    n_any, n_all = 0, 0
    for r in rows:
        top_set = {lbl for lbl, _ in r['ranked'][:top_n]}
        if r['true'] & top_set:
            n_any += 1
        if r['true'] <= top_set:
            n_all += 1
    return n_any / len(rows), n_all / len(rows)


def row_is_seen(true_set, train_counts):
    return all(train_counts.get(t, 0) >= 1 for t in true_set)


def filter_rows_by_mask(rows, mask):
    return [r for r, m in zip(rows, mask) if m]


def compute_metrics_block(rows, top_n):
    if not rows:
        return 0.0, 0.0, 0.0, 0.0
    top1 = top_k_accuracy(rows, 'true', ks=(1,))[1]
    at_n = accuracy_at_n_true(rows, 'true')
    any_match, all_match = topn_any_all_match(rows, top_n)
    return top1, at_n, any_match, all_match


def print_level_table(level_name, rows, top_n, label_fn=None):
    label_fn = label_fn or (lambda x: x)
    col_file, col_true, col_score = 22, 30, 16
    print()
    print(f'── {level_name} Results (top-{top_n}) ──')
    header = f'{"#":<4}{"File":<{col_file}}{"True":<{col_true}}' + ''.join(f'{("#%d" % (i+1)):<{col_score}}' for i in range(top_n)) + 'Found   Verdict'
    print(header)
    print('-' * len(header))
    counts = defaultdict(int)
    for i, r in enumerate(rows, 1):
        kept = r['ranked'][:top_n]
        true_str = ','.join(sorted(label_fn(t) for t in r['true']))[:col_true - 1]
        cols = ''.join(f'{(f"{label_fn(lbl)} {score:.2f}"):<{col_score}}' for lbl, score in kept)
        cols += ' ' * (col_score * (top_n - len(kept)))
        found = len({lbl for lbl, _ in kept} & r['true'])
        verdict = r['verdict'] or '-'
        counts[verdict] += 1
        print(f'{i:<4}{r["file"]:<{col_file}}{true_str:<{col_true}}{cols}{found}/{len(r["true"]):<4}  {verdict}')
    n = len(rows)
    print(f'Correct: {counts["correct"]}/{n}   Partial: {counts["partial"]}/{n}   Wrong: {counts["wrong"]}/{n}')
    return counts


def run(scenario, sequences_dir=None, model_path=None, clusters_path=None,
        cluster_keys_path=None, patterns_path=None, names_path=None, tech2tac_path=None,
        score_with_clustering=False, run_tag=None):

    sequences_dir = sequences_dir or DEFAULT_SEQUENCES_DIR
    files = find_scenario_sequence_files(sequences_dir, scenario)
    by_step = group_files_by_step(files)
    print(f'Scenario {scenario}: {len(by_step):,} distinct steps ({len(files):,} sequence files before dedup)')
    if not by_step:
        return None

    tagger = load_tagger(model_path, clusters_path, cluster_keys_path, patterns_path, names_path,
                          score_with_clustering)
    gioc_points = load_gioc_points(clusters_path or DEFAULT_CLUSTERS_PATH, tagger.device)
    train_tech_counts = defaultdict(int)
    for technique, _phrase, _vec in gioc_points:
        train_tech_counts[base_technique(technique)] += 1
    tech2tac_raw = load_tech2tac(tech2tac_path or DEFAULT_TECH2TAC_PATH)
    tech2tac_camlds = {t: [camlds_tac for mitre_tac in tacs for camlds_tac in mitre_tactic_to_camlds(mitre_tac)]
                        for t, tacs in tech2tac_raw.items()}
    tactic_to_stage_list = {t: [s] for t, s in TACTIC_TO_STAGE.items()}

    true_techniques, true_tactics_gt = load_step_ground_truth(by_step.keys())

    technique_rows, tactic_rows, stage_rows = [], [], []
    for i, (step, fpath) in enumerate(sorted(by_step.items(), key=lambda kv: step_sort_key(kv[0])), 1):
        technique_ranked = predict_step(tagger, gioc_points, fpath)
        true_tech = true_techniques.get(step, set())
        top_tech = {t for t, _ in technique_ranked[:3]}
        technique_rows.append({'file': step, 'true': true_tech, 'ranked': technique_ranked,
                                'verdict': classify(true_tech, top_tech, 3)})

        tactic_ranked = expand_ranked(technique_ranked, tech2tac_camlds)
        true_tac = true_tactics_gt.get(step, set())
        top_tac = {t for t, _ in tactic_ranked[:3]}
        tactic_rows.append({'file': step, 'true': true_tac, 'ranked': tactic_ranked,
                             'verdict': classify(true_tac, top_tac, 3)})

        stage_ranked = expand_ranked(tactic_ranked, tactic_to_stage_list)
        true_stg = {TACTIC_TO_STAGE[t] for t in true_tac if t in TACTIC_TO_STAGE}
        top_stg = {s for s, _ in stage_ranked[:4]}
        stage_rows.append({'file': step, 'true': true_stg, 'ranked': stage_ranked,
                            'verdict': classify(true_stg, top_stg, 4)})

        if i % 10 == 0 or i == len(by_step):
            print(f'  scored {i:,}/{len(by_step):,} steps')

    print_level_table('Technique', technique_rows, 3)
    print_level_table('Tactic', tactic_rows, 3, label_fn=tid)
    print_level_table('Stage', stage_rows, 4, label_fn=sid)

    all_techs = sorted({t for r in technique_rows for t in (r['true'] | {x for x, _ in r['ranked']})})
    all_tacs = sorted({t for r in tactic_rows for t in (r['true'] | {x for x, _ in r['ranked']})})
    all_stages = sorted({t for r in stage_rows for t in (r['true'] | {x for x, _ in r['ranked']})})

    tech_acc = top_k_accuracy(technique_rows, 'true')
    tac_acc = top_k_accuracy(tactic_rows, 'true')
    stg_acc = top_k_accuracy(stage_rows, 'true')
    tech_at_n = accuracy_at_n_true(technique_rows, 'true')
    tac_at_n = accuracy_at_n_true(tactic_rows, 'true')
    stg_at_n = accuracy_at_n_true(stage_rows, 'true')
    tech_lrap, tech_aupr = lrap_aupr(technique_rows, 'true', all_techs)
    tac_lrap, tac_aupr = lrap_aupr(tactic_rows, 'true', all_tacs)
    stg_lrap, stg_aupr = lrap_aupr(stage_rows, 'true', all_stages)

    tech_top3_any, tech_top3_all = topn_any_all_match(technique_rows, 3)
    tac_top3_any, tac_top3_all = topn_any_all_match(tactic_rows, 3)
    stg_top3_any, stg_top3_all = topn_any_all_match(stage_rows, 4)

    print()
    print('── Accuracy (CKD, technique -> tactic -> stage) ──')
    print(f'Technique Accuracy   top-1: {tech_acc[1]*100:.1f}%   top-3: {tech_acc[3]*100:.1f}%   at-n: {tech_at_n*100:.1f}%')
    print(f'Tactic Accuracy      top-1: {tac_acc[1]*100:.1f}%   top-3: {tac_acc[3]*100:.1f}%   at-n: {tac_at_n*100:.1f}%')
    print(f'Stage Accuracy       top-1: {stg_acc[1]*100:.1f}%   top-3: {stg_acc[3]*100:.1f}%   at-n: {stg_at_n*100:.1f}%')
    print()
    print(f'Technique LRAP: {tech_lrap*100:.1f}%   AUPR: {tech_aupr*100:.1f}%')
    print(f'Tactic    LRAP: {tac_lrap*100:.1f}%   AUPR: {tac_aupr*100:.1f}%')
    print(f'Stage     LRAP: {stg_lrap*100:.1f}%   AUPR: {stg_aupr*100:.1f}%')

    seen_mask = [row_is_seen(r['true'], train_tech_counts) for r in technique_rows]
    n_seen = sum(seen_mask)
    seen_technique_rows = filter_rows_by_mask(technique_rows, seen_mask)
    seen_tactic_rows = filter_rows_by_mask(tactic_rows, seen_mask)
    seen_stage_rows = filter_rows_by_mask(stage_rows, seen_mask)

    seen_tech = compute_metrics_block(seen_technique_rows, 3)
    seen_tac = compute_metrics_block(seen_tactic_rows, 3)
    seen_stg = compute_metrics_block(seen_stage_rows, 4)

    overall_tech = (tech_acc[1], tech_at_n, tech_top3_any, tech_top3_all)
    overall_tac = (tac_acc[1], tac_at_n, tac_top3_any, tac_top3_all)
    overall_stg = (stg_acc[1], stg_at_n, stg_top3_any, stg_top3_all)

    print()
    print('── Seen (trained on every true label) vs Overall ──')
    print(f'"Seen" = every true label for that step has >=1 gIoC point in the database (n={n_seen}/{len(by_step)} steps).')
    print()
    row_fmt = '{:<22}{:>9}{:>9}{:>13}{:>13}'
    print(row_fmt.format('', 'top-1', 'at-n', 'top-3-any', 'top-3-all'))
    print('Technique')
    print(row_fmt.format(f'    Seen ({n_seen})', *[f'{v*100:.1f}%' for v in seen_tech]))
    print(row_fmt.format(f'    Overall ({len(by_step)})', *[f'{v*100:.1f}%' for v in overall_tech]))
    print('Tactic')
    print(row_fmt.format(f'    Seen ({n_seen})', *[f'{v*100:.1f}%' for v in seen_tac]))
    print(row_fmt.format(f'    Overall ({len(by_step)})', *[f'{v*100:.1f}%' for v in overall_tac]))
    print('Stage')
    print(row_fmt.format(f'    Seen ({n_seen})', *[f'{v*100:.1f}%' for v in seen_stg]))
    print(row_fmt.format(f'    Overall ({len(by_step)})', *[f'{v*100:.1f}%' for v in overall_stg]))

    seen_lrap_tech, seen_aupr_tech = lrap_aupr(seen_technique_rows, 'true', all_techs)
    seen_lrap_tac, seen_aupr_tac = lrap_aupr(seen_tactic_rows, 'true', all_tacs)
    seen_lrap_stg, seen_aupr_stg = lrap_aupr(seen_stage_rows, 'true', all_stages)

    lrap_fmt = '{:<22}{:>9}{:>9}'
    print()
    print('── LRAP / AUPR: Seen vs Overall ──')
    print(lrap_fmt.format('', 'LRAP', 'AUPR'))
    print('Technique')
    print(lrap_fmt.format(f'    Seen ({n_seen})', f'{seen_lrap_tech*100:.1f}%', f'{seen_aupr_tech*100:.1f}%'))
    print(lrap_fmt.format(f'    Overall ({len(by_step)})', f'{tech_lrap*100:.1f}%', f'{tech_aupr*100:.1f}%'))
    print('Tactic')
    print(lrap_fmt.format(f'    Seen ({n_seen})', f'{seen_lrap_tac*100:.1f}%', f'{seen_aupr_tac*100:.1f}%'))
    print(lrap_fmt.format(f'    Overall ({len(by_step)})', f'{tac_lrap*100:.1f}%', f'{tac_aupr*100:.1f}%'))
    print('Stage')
    print(lrap_fmt.format(f'    Seen ({n_seen})', f'{seen_lrap_stg*100:.1f}%', f'{seen_aupr_stg*100:.1f}%'))
    print(lrap_fmt.format(f'    Overall ({len(by_step)})', f'{stg_lrap*100:.1f}%', f'{stg_aupr*100:.1f}%'))

    def ranked_to_json(rows, label_key):
        return [{'file': r['file'], f'true_{label_key}s': sorted(r['true']),
                  'ranked': [{label_key: lbl, 'score': score} for lbl, score in r['ranked']]} for r in rows]

    out = {
        'scenario': scenario, 'n_total': len(by_step),
        'tech_acc_top1': tech_acc[1], 'tech_acc_top3': tech_acc[3], 'tech_acc_at_n': tech_at_n,
        'tac_acc_top1': tac_acc[1], 'tac_acc_top3': tac_acc[3], 'tac_acc_at_n': tac_at_n,
        'stage_acc_top1': stg_acc[1], 'stage_acc_top3': stg_acc[3], 'stage_acc_at_n': stg_at_n,
        'lrap': tac_lrap, 'aupr': tac_aupr,
        'tech_top3_any': tech_top3_any, 'tech_top3_all': tech_top3_all,
        'tac_top3_any': tac_top3_any, 'tac_top3_all': tac_top3_all,
        'stage_top3_any': stg_top3_any, 'stage_top3_all': stg_top3_all,
        'n_seen': n_seen,
        'seen_vs_overall': {
            'technique': {'seen': seen_tech, 'overall': overall_tech},
            'tactic': {'seen': seen_tac, 'overall': overall_tac},
            'stage': {'seen': seen_stg, 'overall': overall_stg},
        },
        'lrap_aupr_seen_vs_overall': {
            'technique': {'seen': [seen_lrap_tech, seen_aupr_tech], 'overall': [tech_lrap, tech_aupr]},
            'tactic': {'seen': [seen_lrap_tac, seen_aupr_tac], 'overall': [tac_lrap, tac_aupr]},
            'stage': {'seen': [seen_lrap_stg, seen_aupr_stg], 'overall': [stg_lrap, stg_aupr]},
        },
        'technique_results': ranked_to_json(technique_rows, 'technique'),
        'tactic_results': ranked_to_json(tactic_rows, 'tactic'),
        'stage_results': ranked_to_json(stage_rows, 'stage'),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f'ckd_test_results_{run_tag or scenario}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved -> {out_path}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, help='scenario number ("4") or run-group ("2_cron")')
    ap.add_argument('--sequences-dir', default=None)
    ap.add_argument('--run-tag', default=None, help='output filename tag (default: --scenario)')
    ap.add_argument('--model', default=None)
    ap.add_argument('--clusters', default=None)
    ap.add_argument('--cluster-keys', default=None)
    ap.add_argument('--patterns', default=None)
    ap.add_argument('--names', default=None)
    ap.add_argument('--tech2tac', default=None)
    ap.add_argument('-c', '--score-with-clustering', action='store_true')
    args = ap.parse_args()
    run(args.scenario, args.sequences_dir, args.model, args.clusters, args.cluster_keys,
        args.patterns, args.names, args.tech2tac, args.score_with_clustering, args.run_tag)
