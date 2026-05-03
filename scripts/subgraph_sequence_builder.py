
import os
import sys
import json
import time
import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import OUTPUT_GRAPHS, OUTPUT_SEQUENCES, OUTPUT_BENIGN, OUTPUT_ATTACK
from scripts.node_abstraction import abstract_node_name

TRIPLE_SEP = ' '

MAX_NAME_TOKENS = 20

import re
_IP_RE = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')

EDGE_TO_TEXT = {
    'EVENT_BOOT'                  : 'boots',
    'EVENT_READ'                  : 'reads',
    'EVENT_CONNECT'               : 'connects to',
    'EVENT_SENDTO'                : 'sends to',
    'EVENT_RECVMSG'               : 'receives message from',
    'EVENT_READ_SOCKET_PARAMS'    : 'reads socket parameters from',
    'EVENT_SENDMSG'               : 'sends message to',
    'EVENT_CLONE'                 : 'clones',
    'EVENT_EXECUTE'               : 'executes',
    'EVENT_RECVFROM'              : 'receives from',
    'EVENT_WRITE'                 : 'writes to',
    'EVENT_WRITE_SOCKET_PARAMS'   : 'writes socket parameters to',
    'EVENT_UNLINK'                : 'unlinks',
    'EVENT_MODIFY_FILE_ATTRIBUTES': 'modifies attributes of',
    'EVENT_OPEN'                  : 'opens',
    'EVENT_ACCEPT'                : 'accepts',
    'EVENT_BIND'                  : 'binds to',
    'CONNECT_SENDMSG'             : 'connects and sends message to',
    'CONNECT_SENDTO'              : 'connects and sends to',
    'SENDMSG_SENDTO'              : 'sends message and data to',
    'RECVFROM_RECVMSG'            : 'receives data and message from',
    'ACCEPT_BIND'                 : 'accepts and binds',
    'CONNECT_RECVFROM'            : 'connects and receives from',
    'CONNECT_RECVMSG'             : 'connects and receives message from',
    'BIND_CONNECT'                : 'binds and connects to',
    'RECVFROM_SENDTO'             : 'receives and sends to',
    'RECVMSG_SENDMSG'             : 'receives and sends message to',
}

def _normalize_name(name):
    if not name:
        return name
    parts = name.split('_')
    if len(parts) <= 4 or not _IP_RE.match(parts[0]):
        return name
    src_ip = parts[0]
    dst_ip = None
    for p in reversed(parts):
        if _IP_RE.match(p):
            dst_ip = p
            break
    if dst_ip is None:
        dst_ip = parts[-1]
    n_ports = len(parts) - 2
    return '{}_[{}]_{}'.format(src_ip, n_ports, dst_ip)

def extract_triples(G):
    triples = []
    for src, dst, key, attr in G.edges(keys=True, data=True):
        src_name = G.nodes[src].get('name', str(src))
        dst_name = G.nodes[dst].get('name', str(dst))
        edge_op  = attr.get('edge_type', 'unknown')
        ts       = attr.get('ts', 0)
        triples.append((src_name, edge_op, dst_name, ts))
    return sorted(triples, key=lambda x: x[3])

def _clean_edge(op):
    return op.replace('EVENT_', '').lower()

def triples_to_text(triples):
    return [
        '{} {} {}.'.format(src_n, _clean_edge(op), dst_n)
        for src_n, op, dst_n, _ in triples
    ]
def _rebuild_graph(sg_dict):
    G = nx.MultiDiGraph()

    for n_entry in sg_dict['nodes']:
        if isinstance(n_entry, (list, tuple)):
            nid   = n_entry[0]
            attrs = n_entry[1] if len(n_entry) > 1 and isinstance(n_entry[1], dict) else {}
        elif isinstance(n_entry, dict):
            nid   = n_entry.get('id', '')
            attrs = n_entry
        else:
            nid   = n_entry
            attrs = {}

        G.add_node(nid, **attrs)

        uuid = attrs.get('uuid', '')
        if uuid and uuid != nid:
            G.add_node(uuid, **attrs)

    for e_entry in sg_dict['edges']:
        if isinstance(e_entry, (list, tuple)):
            src   = e_entry[0]
            dst   = e_entry[1]
            attrs = e_entry[3] if len(e_entry) >= 4 and isinstance(e_entry[3], dict) else (
                    e_entry[2] if len(e_entry) == 3 and isinstance(e_entry[2], dict) else {})
        elif isinstance(e_entry, dict):
            src   = e_entry.get('src', e_entry.get('source', ''))
            dst   = e_entry.get('dst', e_entry.get('dest', e_entry.get('target', '')))
            attrs = e_entry
        else:
            continue
        G.add_edge(src, dst, **attrs)

    return G

def _iter_subgraphs(in_path, max_subgraphs=None):
    try:
        import ijson
        with open(in_path, 'rb') as f:
            count = 0
            for sg in ijson.items(f, 'subgraphs.item'):
                yield sg
                count += 1
                if max_subgraphs and count >= max_subgraphs:
                    return
            if count == 0:
                f.seek(0)
                for sg in ijson.items(f, 'item'):
                    yield sg
                    count += 1
                    if max_subgraphs and count >= max_subgraphs:
                        return
    except ImportError:
        with open(in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        subgraphs = data.get('subgraphs', data) if isinstance(data, dict) else data
        if max_subgraphs:
            subgraphs = subgraphs[:max_subgraphs]
        for sg in subgraphs:
            yield sg

def build_log_sequences(in_path, out_path, label='unknown', max_subgraphs=None):
    print('\n[{}]'.format(label))
    print('  input    : {}'.format(in_path))
    print('  output   : {}'.format(out_path))
    if max_subgraphs:
        print('  max      : {}'.format(max_subgraphs))
    t0 = time.time()

    sequences = []
    for i, sg in enumerate(_iter_subgraphs(in_path, max_subgraphs)):
        G       = _rebuild_graph(sg)
        triples = extract_triples(G)
        text    = triples_to_text(triples)

        sequences.append({
            'idx'        : i,
            'dep_id'     : sg.get('dep_id',      i),
            'part_idx'   : sg.get('part_idx',    0),
            'total_parts': sg.get('total_parts', 1),
            'seed_uuid'  : sg.get('seed_uuid',   ''),
            'seed_name'  : sg.get('seed_name',   ''),
            'n_triples'  : len(triples),
            'sequence'   : text,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, indent=2)

    elapsed = time.time() - t0
    print('  saved    : {}'.format(out_path))
    print('  elapsed  : {:.1f}s'.format(elapsed))

    return sequences

def _prefer_abstract(path):
    """Use abstract_ prefixed file if it exists, otherwise use original."""
    from scripts.node_abstraction import abstract_path
    abs_path = abstract_path(path)
    if os.path.exists(abs_path):
        print(f'  [sequence input] abstract file → {os.path.basename(abs_path)}')
        return abs_path
    print(f'  [sequence input] raw file     → {os.path.basename(path)}')
    return path


def run_sequences():
    from scripts.filter_attack_subgraphs import ATTACKS
    os.makedirs(OUTPUT_SEQUENCES, exist_ok=True)
    inputs = [
        (_prefer_abstract(os.path.join(OUTPUT_BENIGN, 'benign_training.json')),
         os.path.join(OUTPUT_SEQUENCES, 'sequences_benign.json'),
         'benign'),
    ]
    for atk in ATTACKS:
        inputs.append((
            _prefer_abstract(os.path.join(OUTPUT_ATTACK, f'attack_subgraphs_{atk["name"]}.json')),
            os.path.join(OUTPUT_SEQUENCES, f'sequences_{atk["name"]}.json'),
            'attack',
        ))
    for in_path, out_path, label in inputs:
        build_log_sequences(in_path=in_path, out_path=out_path, label=label)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True,  help='path to subgraph JSON file')
    parser.add_argument('--output', required=True,  help='path to output sequences JSON')
    parser.add_argument('--label',  default='unknown')
    parser.add_argument('--max',    type=int, default=None, help='max subgraphs to process')
    args = parser.parse_args()
    build_log_sequences(args.input, args.output, args.label, args.max)
