"""ckd_tag_edges.py

Ports the CKD (keyword-dictionary) technique/tactic/stage mapping from
sean_1's alert_generate.py onto our own edges files (the output of
parse_attack_events.py / simplify_names.py), instead of requiring a
pre-built .dot graph with sysdig-style edge attributes.

Same algorithm as alert_generate.py:
  1. For each edge, build one lowercase text blob from edgeType + src name
     + dst name (using the simplified name when available, since that's
     already cleaner than sean's ad hoc path/IP string replacements).
  2. For every technique in tech_dic (CKD keyword dictionary), count how
     many of its keywords appear as substrings in that text. Keep the top 5
     techniques by hit count.
  3. Map each kept technique to its tactic(s) via tech2tac.txt, summing
     hit-count scores per tactic. Keep the top 3 tactics.
  4. Map each kept tactic to its stage via tac2stage.txt, summing scores
     per stage. Keep the top 4 stages -- these become the edge's
     stage_name / stage_score columns.
  5. For every node (by srcId), aggregate the stage scores of its outgoing
     edges and take the highest-scoring stage as that node's dominant
     stage -- same rule alert_generate.py uses to label graph nodes.

Input edges file columns (either form produced by this project):
  9 cols  (edges_<tag>_readable.txt):
      srcId srcName srcType dstId dstName dstType edgeType timestamp label
  11 cols (edges_<tag>_simplified.txt):
      srcId srcName srcNameSimplified srcType
      dstId dstName dstNameSimplified dstType
      edgeType timestamp label
"""

import os
import re
import json
import argparse
import ast
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402')
TOP_TECH   = 5
TOP_TAC    = 3
TOP_STAGE  = 4


def load_tech_dic(path):
    with open(path, 'r', encoding='utf-8') as f:
        return ast.literal_eval(f.readline())


def load_tech2tac(path):
    tech2tac = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            tech, tac = line.split('\t')
            tech2tac[tech] = tac
    return tech2tac


def load_tac2stage(path):
    tac2stage = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            tac_list, stage = line.split('...')
            for tac in tac_list.split(', '):
                tac2stage[tac] = stage
    return tac2stage


def parse_edge_line(fields):
    if len(fields) == 11:
        srcId, srcName, srcSimplified, srcType, \
            dstId, dstName, dstSimplified, dstType, \
            edgeType, ts, label = fields
    elif len(fields) == 9:
        srcId, srcName, srcType, dstId, dstName, dstType, edgeType, ts, label = fields
        srcSimplified = srcName
        dstSimplified = dstName
    else:
        return None
    return {
        'srcId': srcId, 'srcName': srcName, 'srcSimplified': srcSimplified, 'srcType': srcType,
        'dstId': dstId, 'dstName': dstName, 'dstSimplified': dstSimplified, 'dstType': dstType,
        'edgeType': edgeType, 'ts': ts, 'label': label,
    }


def top_techniques(text, tech_dic, top_n=TOP_TECH):
    scored = []
    for tech, keywords in tech_dic.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            scored.append((tech, hits))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def techs_to_tactics(tech_scores, tech2tac):
    tac_scores = defaultdict(int)
    for tech, score in tech_scores:
        tac_list = tech2tac.get(tech)
        if not tac_list:
            continue
        for tac in tac_list.split(', '):
            tac_scores[tac] += score
    return sorted(tac_scores.items(), key=lambda x: x[1], reverse=True)


def tactics_to_stages(tac_scores, tac2stage, top_n=TOP_STAGE):
    stage_scores = defaultdict(float)
    for tac, score in tac_scores[:TOP_TAC]:
        stage = tac2stage.get(tac)
        if not stage:
            continue
        stage_scores[stage] += score
    return sorted(stage_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def run(input_file, output_dir=None, tech_dic_path=None, tech2tac_path=None, tac2stage_path=None):
    tech_dic_path  = tech_dic_path  or os.path.join(DEFAULT_DIR, 'tech_dic.txt')
    tech2tac_path  = tech2tac_path  or os.path.join(DEFAULT_DIR, 'tech2tac.txt')
    tac2stage_path = tac2stage_path or os.path.join(DEFAULT_DIR, 'tac2stage.txt')

    print(f'Loading CKD dictionary : {tech_dic_path}')
    tech_dic = load_tech_dic(tech_dic_path)
    print(f'  {len(tech_dic):,} techniques')
    tech2tac  = load_tech2tac(tech2tac_path)
    tac2stage = load_tac2stage(tac2stage_path)

    output_dir = output_dir or os.path.dirname(os.path.abspath(input_file))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(input_file)
    base = re.sub(r'\.txt$', '', base)
    base = re.sub(r'_(readable|simplified)$', '', base)
    edges_out = os.path.join(output_dir, f'{base}_ckd_edges.txt')
    nodes_out = os.path.join(output_dir, f'{base}_ckd_nodes.txt')

    node_names = {}
    node_stage_scores = defaultdict(lambda: defaultdict(float))

    n_rows = n_scored = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(edges_out, 'w', encoding='utf-8') as fout:

        for line in fin:
            fields = line.rstrip('\n').split('\t')
            edge = parse_edge_line(fields)
            if edge is None:
                continue
            n_rows += 1

            node_names[edge['srcId']] = edge['srcName']
            node_names[edge['dstId']] = edge['dstName']

            text = ' '.join([
                edge['edgeType'], edge['srcSimplified'], edge['dstSimplified'],
            ]).lower()

            tech_scores  = top_techniques(text, tech_dic)
            tac_scores   = techs_to_tactics(tech_scores, tech2tac)
            stage_scores = tactics_to_stages(tac_scores, tac2stage)

            if stage_scores:
                n_scored += 1
                for stage, score in stage_scores:
                    node_stage_scores[edge['srcId']][stage] += score

            tech_str  = ';'.join(f'{t}:{s}' for t, s in tech_scores)
            tac_str   = ';'.join(f'{t}:{s}' for t, s in tac_scores[:TOP_TAC])
            stage_str = ';'.join(f'{t}:{s:.1f}' for t, s in stage_scores)

            fout.write('\t'.join([
                edge['srcId'], edge['srcName'], edge['srcType'],
                edge['dstId'], edge['dstName'], edge['dstType'],
                edge['edgeType'], edge['ts'], edge['label'],
                tech_str, tac_str, stage_str,
            ]) + '\n')

    with open(nodes_out, 'w', encoding='utf-8') as fnodes:
        for node_id, stage_scores in node_stage_scores.items():
            dominant_stage, score = max(stage_scores.items(), key=lambda x: x[1])
            fnodes.write('\t'.join([
                node_id, node_names.get(node_id, ''), dominant_stage, f'{score:.1f}',
            ]) + '\n')

    print(f'  Edges read            : {n_rows:,}')
    print(f'  Edges w/ a stage match : {n_scored:,}')
    print(f'  Edges written          : {edges_out}')
    print(f'  Nodes written          : {len(node_stage_scores):,}  -> {nodes_out}')
    return edges_out, nodes_out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='edges_*_readable.txt or edges_*_simplified.txt')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--tech-dic', default=None, help='default: CAM-LDS/knowhow/KNOWHOW-seng402/tech_dic.txt')
    ap.add_argument('--tech2tac', default=None)
    ap.add_argument('--tac2stage', default=None)
    args = ap.parse_args()
    run(args.file, args.output_dir, args.tech_dic, args.tech2tac, args.tac2stage)
