import os
import json
import csv
import sys
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import OUTPUT_GRAPHS, OUTPUT_ATTACK

SUBGRAPHS_JSON        = os.path.join(OUTPUT_GRAPHS,  'subgraphs_all.json')
ATTACK_SUBGRAPHS_JSON = os.path.join(OUTPUT_ATTACK,  'attack_subgraphs_{}.json')

RUNS = [
    {
        'attack_name': 'Browser_Extension_Drakon_Dropper',
        'csv'        : os.path.join(PROJECT_ROOT, 'input', 'malicious_nodes', 'node_Browser_Extension_Drakon_Dropper.csv'),
        'output_json': os.path.join(OUTPUT_ATTACK, 'subgraph_labels_Browser_Extension_Drakon_Dropper.json'),
    },
    {
        'attack_name': 'Firefox_Backdoor_Drakon_In_Memory',
        'csv'        : os.path.join(PROJECT_ROOT, 'input', 'malicious_nodes', 'node_Firefox_Backdoor_Drakon_In_Memory.csv'),
        'output_json': os.path.join(OUTPUT_ATTACK, 'subgraph_labels_Firefox_Backdoor_Drakon_In_Memory.json'),
    },
]

def ns_to_est(ns):
    utc = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    est = utc - timedelta(hours=4)   
    return est.strftime('%Y-%m-%d %H:%M:%S')

def load_attack_uuids(csv_path, attack_name):
    uuids = {}
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            uuid = parts[0].strip()
            info = parts[1].strip() if len(parts) > 1 else ''
            uuids[uuid] = {'attack': attack_name, 'info': info}
    return uuids

def run_labeling(subgraphs, attack_uuids, attack_name, output_json):
    results = []

    for sg in subgraphs:
        malicious_nodes_found = []

        for node_id, node_data in sg['nodes']:
            node_uuid = node_data.get('uuid', '')
            if node_uuid in attack_uuids:
                malicious_nodes_found.append({
                    'uuid'  : node_uuid,
                    'name'  : node_data.get('name', 'UNKNOWN'),
                    'type'  : node_data.get('type', 'UNKNOWN'),
                    'attack': attack_uuids[node_uuid]['attack'],
                })

        if len(malicious_nodes_found) < 2:
            continue

        start_ts = sg.get('start_ts')
        end_ts   = sg.get('end_ts')
        results.append({
            'dep_id'              : sg['dep_id'],
            'part_idx'            : sg['part_idx'],
            'total_parts'         : sg['total_parts'],
            'seed_uuid'           : sg['seed_uuid'],
            'seed_name'           : sg['seed_name'],
            'n_nodes'             : sg.get('n_nodes', len(sg.get('nodes', []))),
            'n_edges'             : sg.get('n_edges', len(sg.get('edges', []))),
            'start_ts'            : start_ts,
            'end_ts'              : end_ts,
            'start_time'          : ns_to_est(start_ts) if start_ts else None,
            'end_time'            : ns_to_est(end_ts)   if end_ts   else None,
            'malicious_nodes'     : malicious_nodes_found,
            'num_malicious_nodes' : len(malicious_nodes_found),
        })

    results.sort(key=lambda r: r['num_malicious_nodes'], reverse=True)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print('  Malicious subgraphs found : {:,}'.format(len(results)))
    print('  Saved to                  : {}'.format(output_json))

    filtered_times = {}
    _atk_json = ATTACK_SUBGRAPHS_JSON.format(attack_name)
    if os.path.exists(_atk_json):
        with open(_atk_json) as f:
            atk_data = json.load(f)
        for sg in atk_data.get('subgraphs', []):
            filtered_times[sg['dep_id']] = (
                ns_to_est(sg['start_ts']),
                ns_to_est(sg['end_ts']),
            )

    top5 = results[:5]
    print('\n  --- Top 5 Malicious Subgraphs ---')
    print('  {:<8} {:<12} {:<10} {:<10} {:<22} {:<22} {}'.format(
        'dep_id', 'part', 'nodes', 'edges', 'start_time', 'end_time', 'mal_nodes'))
    for r in top5:
        start, end = filtered_times.get(r['dep_id'], (r['start_time'], r['end_time']))
        print('  {:<8} {}/{:<9} {:<10} {:<10} {:<22} {:<22} {}'.format(
            r['dep_id'],
            r['part_idx'] + 1, r['total_parts'],
            r.get('n_nodes', 'N/A'),
            r.get('n_edges', 'N/A'),
            start or 'N/A',
            end   or 'N/A',
            r['num_malicious_nodes']))

    found_uuids = {
        node_data.get('uuid', '')
        for sg in subgraphs
        for node_id, node_data in sg['nodes']
        if node_data.get('uuid', '') in attack_uuids
    }
    unfound = set(attack_uuids.keys()) - found_uuids
    print('\n  --- UUID Coverage ---')
    print('  Total attack UUIDs : {:,}'.format(len(attack_uuids)))
    print('  Found in subgraphs : {:,}'.format(len(found_uuids)))
    print('  NOT found          : {:,}'.format(len(unfound)))

def main():
    print('Loading subgraphs from: {}'.format(SUBGRAPHS_JSON))
    with open(SUBGRAPHS_JSON, 'r') as f:
        subgraphs = json.load(f)
    if isinstance(subgraphs, dict):
        subgraphs = subgraphs['subgraphs']
    print('Subgraphs loaded : {:,}\n'.format(len(subgraphs)))

    for run in RUNS:
        attack_name = run['attack_name']
        print('Attack : {}'.format(attack_name))
        attack_uuids = load_attack_uuids(run['csv'], attack_name)
        print('Attack UUIDs loaded : {:,}'.format(len(attack_uuids)))
        run_labeling(subgraphs, attack_uuids, attack_name, run['output_json'])
        print()

if __name__ == '__main__':
    main()
