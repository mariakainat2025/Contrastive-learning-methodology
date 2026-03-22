
import os
import sys
import json
import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import OUTPUT_GRAPHS, OUTPUT_SEQUENCES

TRIPLE_SEP = ' '


def extract_triples(G):

    triples = []
    for src, dst, key, attr in G.edges(keys=True, data=True):
        src_name = G.nodes[src].get('name', str(src))
        dst_name = G.nodes[dst].get('name', str(dst))
        edge_op  = attr.get('edge_type', 'unknown')
        ts       = attr.get('ts', 0)
        triples.append((src_name, edge_op, dst_name, ts))

    return sorted(triples, key=lambda x: x[3])


# triples → text sentence

def triples_to_text(triples):

    return [
        '{} {} {}'.format(src_n, op, dst_n)
        for src_n, op, dst_n, _ in triples
    ]


# reconstruct nx.MultiDiGraph

def _rebuild_graph(sg_dict):
    G = nx.MultiDiGraph()

    for n_entry in sg_dict['nodes']:
        if isinstance(n_entry, (list, tuple)):
            nid   = n_entry[0]
            attrs = n_entry[1] if len(n_entry) > 1 else {}
        else:
            nid   = n_entry
            attrs = {}
        G.add_node(nid, **attrs)

    for e_entry in sg_dict['edges']:
        src   = e_entry[0]
        dst   = e_entry[1]
        attrs = e_entry[3]
        G.add_edge(src, dst, **attrs)

    return G


# main

def build_log_sequences(tag):

    in_path  = os.path.join(OUTPUT_GRAPHS,    'subgraphs_{}.json'.format(tag))
    out_path = os.path.join(OUTPUT_SEQUENCES, 'log_sequences_{}.json'.format(tag))

    with open(in_path, 'r', encoding='utf-8') as f:
        subgraphs = json.load(f)

    print('  loaded {} subgraphs from {}'.format(len(subgraphs), os.path.basename(in_path)))

    sequences = []
    for i, sg in enumerate(subgraphs):
        G       = _rebuild_graph(sg)
        triples = extract_triples(G)
        text    = triples_to_text(triples)

        sequences.append({
            'idx'        : i,
            'dep_id'     : sg['dep_id'],
            'part_idx'   : sg['part_idx'],
            'total_parts': sg['total_parts'],
            'seed_uuid'  : sg.get('seed_uuid', ''),
            'seed_name'  : sg.get('seed_name', ''),
            'n_triples'  : len(triples),
            'sequence'   : text,
        })

    os.makedirs(OUTPUT_SEQUENCES, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, indent=2)

    print('  log_sequences_{}.json saved : {}'.format(tag, out_path))
    print()
    print('  {:<6}  {:<8}  {:<24}  {}'.format('Idx', 'Triples', 'Partition', 'Text preview'))
    print('  ' + '-' * 110)
    for s in sequences[:5]:
        part = 'dep#{} part {}/{}'.format(s['dep_id'], s['part_idx'] + 1, s['total_parts'])
        preview = ' '.join(s['sequence'])[:80]
        print('  {:<6}  {:<8}  {:<24}  {}'.format(
            s['idx'], s['n_triples'], part, preview))

    return sequences


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True )
    args = parser.parse_args()
    build_log_sequences(args.tag)
