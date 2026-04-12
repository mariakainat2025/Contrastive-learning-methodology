import os
import sys
import json
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import OUTPUT_GRAPHS, OUTPUT_ATTACK
from scripts.config import (
    _ATTACK_311_START, _ATTACK_311_END,
    _ATTACK_33_START,  _ATTACK_33_END,
)
from scripts.label_subgraphs import load_attack_uuids, run_labeling

ATTACKS = [
    {
        'name'      : 'Browser_Extension_Drakon_Dropper',
        'start_ns'  : _ATTACK_311_START,
        'end_ns'    : _ATTACK_311_END,
    },
    {
        'name'      : 'Firefox_Backdoor_Drakon_In_Memory',
        'start_ns'  : _ATTACK_33_START,
        'end_ns'    : _ATTACK_33_END,
    },
]

def ns_to_est(ns):
    utc = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return (utc - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')

def filter_attack_subgraphs(attack_name, attack_start_ns, attack_end_ns):

    subgraphs_path = os.path.join(OUTPUT_GRAPHS,  'subgraphs_all.json')
    labels_paths   = [
        os.path.join(OUTPUT_ATTACK, f'subgraph_labels_{attack_name}.json'),
    ]
    out_path       = os.path.join(OUTPUT_ATTACK, f'attack_subgraphs_{attack_name}.json')
    os.makedirs(OUTPUT_ATTACK, exist_ok=True)

    ATTACK_NAME     = attack_name
    ATTACK_START_NS = attack_start_ns
    ATTACK_END_NS   = attack_end_ns

    with open(subgraphs_path, 'r') as f:
        data = json.load(f)
    subgraphs = data['subgraphs'] if isinstance(data, dict) else data

    attack_keys  = {}
    mal_uuid_map = {}
    for labels_path in labels_paths:
        if not os.path.exists(labels_path):
            continue
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        for r in labels:
            attack_keys[(r['dep_id'], r['part_idx'])] = True
            dep = r['dep_id']
            if dep not in mal_uuid_map:
                mal_uuid_map[dep] = set()
            mal_uuid_map[dep] |= {n['uuid'] for n in r.get('malicious_nodes', [])}


    results = []
    for sg in subgraphs:
        key = (sg['dep_id'], sg['part_idx'])
        if key not in attack_keys:
            continue

        mal_uuids = mal_uuid_map.get(sg['dep_id'], set())

        filtered_edges = [
            e for e in sg['edges']
            if ATTACK_START_NS <= e[3].get('ts', 0) <= ATTACK_END_NS
        ]

        node_uuids = set()
        for e in filtered_edges:
            node_uuids.add(e[0])
            node_uuids.add(e[1])

        filtered_nodes = [
            n for n in sg['nodes']
            if isinstance(n, (list, tuple)) and (
                n[1].get('uuid', '') in node_uuids or
                n[1].get('uuid', '') in mal_uuids
            )
        ]

        # ── Fluxbox UUID filter ───────────────────────────────────────────────
        if sg.get('seed_name', '') == 'fluxbox':
            csv_path_fluxbox = os.path.join(PROJECT_ROOT, 'input', 'malicious_nodes', f'node_{attack_name}.csv')
            if os.path.exists(csv_path_fluxbox):
                attack_specific_uuids = set()
                with open(csv_path_fluxbox) as f:
                    for line in f:
                        uuid = line.split(',')[0].strip()
                        if uuid:
                            attack_specific_uuids.add(uuid)

                # Keep edges where source OR destination is an attack-specific UUID
                filtered_edges = [
                    e for e in filtered_edges
                    if e[0] in attack_specific_uuids and e[1] in attack_specific_uuids
                ]

                # Keep nodes referenced in kept edges AND in malicious UUIDs
                keep_uuids = set()
                for e in filtered_edges:
                    keep_uuids.add(e[0])
                    keep_uuids.add(e[1])

                filtered_nodes = [
                    n for n in filtered_nodes
                    if n[1].get('uuid', '') in keep_uuids and n[1].get('uuid', '') in mal_uuids
                ]
        # ─────────────────────────────────────────────────────────────────────

        edge_ts   = [e[3].get('ts', 0) for e in filtered_edges if e[3].get('ts', 0) > 0]
        start_ts  = min(edge_ts) if edge_ts else sg['start_ts']
        end_ts    = max(edge_ts) if edge_ts else sg['end_ts']

        if not filtered_edges:
            continue

        # ── Ensure all malicious UUIDs from CSV are included ─────────────────
        csv_path = os.path.join(PROJECT_ROOT, 'input', 'malicious_nodes', f'node_{attack_name}.csv')
        csv_uuids = set()
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for line in f:
                    uuid = line.split(',')[0].strip()
                    if uuid:
                        csv_uuids.add(uuid)

        filtered_node_uuids = {n[1].get('uuid', '') for n in filtered_nodes}
        missing_csv = csv_uuids - filtered_node_uuids
        if missing_csv:
            added = 0
            for n in sg['nodes']:
                if isinstance(n, (list, tuple)) and n[1].get('uuid', '') in missing_csv:
                    filtered_nodes.append(n)
                    added += 1
            # ─────────────────────────────────────────────────────────────────────

        filtered_node_uuids = {n[1].get('uuid', '') for n in filtered_nodes}
        n_malicious = sum(1 for u in mal_uuids if u in filtered_node_uuids)

        results.append({
            'dep_id'      : sg['dep_id'],
            'part_idx'    : sg['part_idx'],
            'total_parts' : sg['total_parts'],
            'seed_uuid'   : sg.get('seed_uuid', ''),
            'seed_name'   : sg.get('seed_name', ''),
            'start_ts'    : start_ts,
            'end_ts'      : end_ts,
            'n_nodes'     : len(filtered_nodes),
            'n_edges'     : len(filtered_edges),
            'n_malicious' : n_malicious,
            'nodes'       : filtered_nodes,
            'edges'       : filtered_edges,
        })

    output = {
        'total_attack_subgraphs': len(results),
        'attack_start_ns'       : attack_start_ns,
        'attack_end_ns'         : attack_end_ns,
        'subgraphs'             : results,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'Attack subgraphs (saved)    : {len(results)}')
    print(f'Saved → {out_path}')
    print()
    print(f'  {"dep_id":<8}  {"part":<6}  {"n_nodes":>9}  {"n_edges":>9}  {"mal_nodes":>9}  {"start_time":<22}  {"end_time":<22}  {"seed_name"}')
    print('  ' + '-' * 110)
    for r in results:
        print(f'  {r["dep_id"]:<8}  {r["part_idx"]:<6}  {r["n_nodes"]:>9,}  {r["n_edges"]:>9,}  {r["n_malicious"]:>9,}  {ns_to_est(r["start_ts"]):<22}  {ns_to_est(r["end_ts"]):<22}  {r["seed_name"]}')

    # ── Run labeling on filtered subgraphs ────────────────────────────────────
    csv_path = os.path.join(PROJECT_ROOT, 'input', 'malicious_nodes', f'node_{attack_name}.csv')
    label_out = os.path.join(OUTPUT_ATTACK, f'subgraph_labels_{attack_name}.json')
    if os.path.exists(csv_path):
        print(f'\n  --- Labeling filtered subgraphs ---')
        attack_uuids = load_attack_uuids(csv_path, attack_name)
        print(f'  Attack UUIDs loaded : {len(attack_uuids):,}')
        run_labeling(results, attack_uuids, attack_name, label_out)
    else:
        print(f'\n  [WARNING] No CSV found for labeling: {csv_path}')
    # ─────────────────────────────────────────────────────────────────────────

def run_all():
    for atk in ATTACKS:
        print(f'\n=== Attack : {atk["name"]} ===')
        filter_attack_subgraphs(atk['name'], atk['start_ns'], atk['end_ns'])

if __name__ == '__main__':
    run_all()
