
import os
import sys
import json
import random
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import OUTPUT_GRAPHS, OUTPUT_BENIGN, INPUT_TEST
from scripts.subgraph_sequence_builder import _rebuild_graph, extract_triples, triples_to_text

_APR2_START_NS  = int(datetime(2018, 4,  2,  4, 0, 0, tzinfo=timezone.utc).timestamp() * 1e9)
_APR10_START_NS = int(datetime(2018, 4, 10,  4, 0, 0, tzinfo=timezone.utc).timestamp() * 1e9)

INPUT_JSON   = os.path.join(OUTPUT_GRAPHS,  'subgraphs_all.json')
BENIGN_JSON  = os.path.join(OUTPUT_BENIGN,  'benign_subgraphs.json')
TRAIN_JSON   = os.path.join(OUTPUT_BENIGN,  'benign_training.json')
TEST_JSON    = os.path.join(OUTPUT_BENIGN,  'benign_testing.json')

MAX_NODES    = None  
TRAIN_SIZE   = 10000
TEST_SIZE    = 2000

def ns_to_edt(ns):
    utc = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    edt = utc - timedelta(hours=4)
    return edt.strftime('%Y-%m-%d %H:%M:%S EDT')

def main():
    print(f'Reading : {INPUT_JSON}')
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    all_subgraphs = data['subgraphs'] if isinstance(data, dict) else data
    print(f'Total subgraphs in file : {len(all_subgraphs):,}')

    benign = []
    skipped_large = 0
    for sg in tqdm(all_subgraphs, desc='Filtering', unit='sg'):
        if (sg.get('start_ts') is not None
                and sg.get('end_ts') is not None
                and _APR2_START_NS <= sg['start_ts'] < _APR10_START_NS
                and _APR2_START_NS <= sg['end_ts']   < _APR10_START_NS):
            if MAX_NODES is not None and sg.get('n_nodes', 0) > MAX_NODES:
                skipped_large += 1
            else:
                benign.append(sg)

    print(f'\nSubgraphs from Apr 2 – Apr 9 : {len(benign):,}')
    print(f'  Skipped (n_nodes > {MAX_NODES})  : {skipped_large:,}')
    if benign:
        earliest = min(sg['start_ts'] for sg in benign)
        latest   = max(sg['end_ts']   for sg in benign)
        print(f'  Earliest start : {ns_to_edt(earliest)}')
        print(f'  Latest end     : {ns_to_edt(latest)}')

    os.makedirs(OUTPUT_BENIGN, exist_ok=True)
    with open(BENIGN_JSON, 'w') as f:
        json.dump({'total_subgraphs': len(benign), 'subgraphs': benign}, f, indent=2)
    print(f'\n  Saved -> {BENIGN_JSON}')

    random.seed(42)
    shuffled  = benign[:]
    random.shuffle(shuffled)
    train_sgs = shuffled[:TRAIN_SIZE]
    test_sgs  = shuffled[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

    print(f'\n--- Train / Test Split ---')
    print(f'  Total benign subgraphs : {len(benign):,}')
    print(f'  Training subgraphs     : {len(train_sgs):,}')
    print(f'  Testing  subgraphs     : {len(test_sgs):,}')

    with open(TRAIN_JSON, 'w') as f:
        json.dump({'total_subgraphs': len(train_sgs), 'subgraphs': train_sgs}, f, indent=2)
    print(f'  Saved training -> {TRAIN_JSON}')

    with open(TEST_JSON, 'w') as f:
        json.dump({'total_subgraphs': len(test_sgs), 'subgraphs': test_sgs}, f, indent=2)
    print(f'  Saved testing  -> {TEST_JSON}')

    # ── Save raw test subgraphs to input/test/ for evaluation ─────────────────
    os.makedirs(INPUT_TEST, exist_ok=True)
    test_sg_path = os.path.join(INPUT_TEST, 'benign_subgraphs.json')
    with open(test_sg_path, 'w') as f:
        json.dump({'total_subgraphs': len(test_sgs), 'subgraphs': test_sgs}, f, indent=2)
    print(f'  Saved test subgraphs -> {test_sg_path}')

    # ── Extract sequences from test subgraphs and save to input/test/ ────────
    print(f'\n--- Extracting sequences from {len(test_sgs):,} test subgraphs ---')
    os.makedirs(INPUT_TEST, exist_ok=True)
    test_seq_path = os.path.join(INPUT_TEST, 'sequences_benign.json')
    sequences = []
    for idx, sg in enumerate(tqdm(test_sgs, desc='Building sequences', unit='sg')):
        G       = _rebuild_graph(sg)
        triples = extract_triples(G)
        text    = triples_to_text(triples)
        sequences.append({
            'idx'        : idx,
            'dep_id'     : sg.get('dep_id'),
            'part_idx'   : sg.get('part_idx', 0),
            'seed_name'  : sg.get('seed_name', ''),
            'n_triples'  : len(triples),
            'sequence'   : text,
        })
    with open(test_seq_path, 'w') as f:
        json.dump(sequences, f, indent=2)
    print(f'  Saved test sequences -> {test_seq_path}')

if __name__ == '__main__':
    main()
