
import os
import json
import csv
import sys
from datetime import datetime, timezone, timedelta

PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACK_SG_PATH    = os.path.join(PROJECT_ROOT, 'output', 'theia', 'graphs', 'attack_subgraphs.json')
CSV_BROWSER_EXT   = os.path.join(PROJECT_ROOT, 'input', 'malicious_nodes', 'node_Browser_Extension_Drakon_Dropper.csv')

def ns_to_est(ns):
    utc = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return (utc - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')

def load_attack_uuids(csv_path):
    uuids = {}
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            uuid  = parts[0].strip()
            info  = parts[1].strip() if len(parts) > 1 else ''
            uuids[uuid] = info
    return uuids

def main():
    attack_uuids = load_attack_uuids(CSV_BROWSER_EXT)
    print(f'Malicious UUIDs loaded : {len(attack_uuids)}')

    with open(ATTACK_SG_PATH) as f:
        data = json.load(f)
    subgraphs = data['subgraphs']
    print(f'Attack subgraphs       : {len(subgraphs)}')
    print(f'Attack window          : {ns_to_est(data["attack_start_ns"])} → {ns_to_est(data["attack_end_ns"])} ET')

    all_found   = set()
    all_unfound = set(attack_uuids.keys())

    for i, sg in enumerate(subgraphs):
        found_in_sg = []

        for n_entry in sg['nodes']:
            if isinstance(n_entry, (list, tuple)):
                node_data = n_entry[1] if len(n_entry) > 1 else {}
            else:
                node_data = {}
            uuid = node_data.get('uuid', '')
            if uuid in attack_uuids:
                found_in_sg.append({
                    'uuid' : uuid,
                    'name' : node_data.get('name', 'UNKNOWN'),
                    'type' : node_data.get('type', 'UNKNOWN'),
                    'info' : attack_uuids[uuid],
                })
                all_found.add(uuid)
                all_unfound.discard(uuid)

        print(f'\n{"="*70}')
        print(f'  Subgraph [{i}]  dep#{sg["dep_id"]}  '
              f'{ns_to_est(sg["start_ts"])} → {ns_to_est(sg["end_ts"])} ET')
        print(f'  seed      : {sg["seed_name"]}')
        print(f'  nodes     : {sg["n_nodes"]:,}    edges : {sg["n_edges"]:,}')
        print(f'  malicious : {len(found_in_sg)} / {len(attack_uuids)} UUIDs found')
        print(f'{"─"*70}')

        if found_in_sg:
            print(f'  {"UUID":<40}  {"name":<35}  type')
            print(f'  {"─"*40}  {"─"*35}  {"─"*20}')
            for n in found_in_sg:
                print(f'  {n["uuid"]:<40}  {n["name"][:35]:<35}  {n["type"]}')
        else:
            print('  (no malicious nodes found)')

    print(f'\n{"="*70}')
    print(f'  OVERALL UUID COVERAGE')
    print(f'{"─"*70}')
    print(f'  Total malicious UUIDs : {len(attack_uuids)}')
    print(f'  Found in subgraphs    : {len(all_found)}')
    print(f'  NOT found             : {len(all_unfound)}')

    if all_unfound:
        print(f'\n  UUIDs NOT found in any attack subgraph:')
        for uuid in sorted(all_unfound):
            print(f'    {uuid}  |  {attack_uuids[uuid]}')

if __name__ == '__main__':
    main()
