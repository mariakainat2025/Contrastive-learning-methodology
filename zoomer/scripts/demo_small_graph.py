import json

from abstract_node import (
    node_abstract_counts, node_abstract_vector, HNODE_DIMS,
    FILE_ABSTRACT_FUNCS, PROCESS_ABSTRACT_FUNCS, ACCOUNT_ABSTRACT_FUNCS, SOCKET_ABSTRACT_FUNCS,
)
from edge_feature import edge_type_counts, edge_type_vector, EDGE_TYPE_DIMS
from ioc_feature import load_knowledge_base, ioc_feature_counts, ioc_feature_vector, IOC_DIMS, ATOMIC_IOCS_PATH
from Feature_Initialization import load_graph, _normalize

GRAPH_PATH = '/csse/research/contructive-learning/CAM-LDS/graphs/discovery/T1082-000/graph_6_macro_binary-3_client.json'

FUNCS_BY_TYPE = {
    'FILE': FILE_ABSTRACT_FUNCS,
    'SUBJECT_PROCESS': PROCESS_ABSTRACT_FUNCS,
    'PRINCIPAL_LOCAL': ACCOUNT_ABSTRACT_FUNCS,
    'NetFlowObject': SOCKET_ABSTRACT_FUNCS,
}


def main():
    with open(GRAPH_PATH) as f:
        raw = json.load(f)

    id_to_name = {n['id']: n['name'] for n in raw['nodes']}

    print('=' * 70)
    print('GRAPH:', GRAPH_PATH)
    print('=' * 70)

    print('\n-- Nodes ({}) --'.format(len(raw['nodes'])))
    for n in raw['nodes']:
        print('  [{}] {}'.format(n['type'], n['name']))

    print('\n-- Events / links ({}) --'.format(len(raw['links'])))
    for link in raw['links']:
        src = id_to_name[link['source']]
        dst = id_to_name[link['target']]
        print('  {:<18s} {} -> {}'.format(link['edge_type'], src, dst))

    print('\n-- Per-node classification (what node_hnode-style logic sees) --')
    for n in raw['nodes']:
        funcs = FUNCS_BY_TYPE.get(n['type'], {})
        matches = [label for (label, fn) in funcs.items() if fn(n['name'])]
        print('  [{}] {!r} -> {}'.format(n['type'], n['name'], matches or '(no match)'))

    G = load_graph(raw)
    counts = node_abstract_counts(G)
    vector = node_abstract_vector(G)

    print('\n-- Final node_abstract_counts() for the WHOLE graph --')
    for dim in HNODE_DIMS:
        if counts[dim]:
            print('  {:<14s}: {}'.format(dim, counts[dim]))

    print('\n-- Same thing as the raw 18-dim vector (node_abstract_vector) --')
    print(' ', list(zip(HNODE_DIMS, vector)))

    print('\n-- Per-edge classification (what edge_type_counts sees) --')
    for link in raw['links']:
        norm = _normalize(link['edge_type'])
        in_vocab = norm in EDGE_TYPE_DIMS
        print('  {:<18s} -> normalized={!r}  {}'.format(
            link['edge_type'], norm, 'counted' if in_vocab else 'NOT in vocab, ignored'))

    edge_counts = edge_type_counts(G)
    edge_vector = edge_type_vector(G)

    print('\n-- Final edge_type_counts() for the WHOLE graph --')
    for dim in EDGE_TYPE_DIMS:
        if edge_counts[dim]:
            print('  {:<14s}: {}'.format(dim, edge_counts[dim]))

    print('\n-- Same thing as the raw {}-dim vector (edge_type_vector) --'.format(len(EDGE_TYPE_DIMS)))
    print(' ', list(zip(EDGE_TYPE_DIMS, edge_vector)))

    kb = load_knowledge_base()
    print('\n-- Knowledge base source: {} --'.format(ATOMIC_IOCS_PATH))
    print('   (extracted from Atomic Red Team CTI reports -- file paths are Windows-style,')
    print('    since that\'s where Atomic Red Team\'s own test scripts/reports came from)')
    print('\n-- IoC check: is this graph\'s file/IP in the known-bad knowledge base? --')
    file_names = [n['name'] for n in raw['nodes'] if n['type'] == 'FILE']
    for tactic, entry in kb.items():
        for fname in file_names:
            if fname in entry['file_paths']:
                print('  {!r} IS a known IoC for tactic {!r}'.format(fname, tactic))
    ioc_counts = ioc_feature_counts(G, kb)
    ioc_vector = ioc_feature_vector(G, kb)

    print('\n-- Final ioc_feature_counts() for the WHOLE graph --')
    for dim in IOC_DIMS:
        if ioc_counts[dim]:
            print('  {:<24s}: {}'.format(dim, ioc_counts[dim]))

    print('\n-- Same thing as the raw {}-dim vector (ioc_feature_vector) --'.format(len(IOC_DIMS)))
    print(' ', list(zip(IOC_DIMS, ioc_vector)))

    print('\n-- How all three get added together (same pattern as raw_vector_for_graph()) --')
    merged = {}
    merged.update(counts)
    merged.update(edge_counts)
    merged.update(ioc_counts)
    print('  node dims: {}  + edge dims: {}  + IoC dims: {}  = merged dims: {}'.format(
        len(counts), len(edge_counts), len(ioc_counts), len(merged)))
    print('  (dict.update() just stacks all three -- their dim NAMES never collide,')
    print('   so nothing gets overwritten; node/edge/IoC counts all sit side by side')
    print('   in the same combined dict, which is what later gets discretized and')
    print('   cross-producted into the wide model\'s actual input.)')
    nonzero_merged = {k: v for (k, v) in merged.items() if v}
    print('  non-zero entries in the merged dict:', nonzero_merged)


if __name__ == '__main__':
    main()
