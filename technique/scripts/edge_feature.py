from Feature_Initialization import MASTER_EDGE_TYPES, _normalize

EDGE_TYPE_DIMS = tuple(_normalize(t) for t in MASTER_EDGE_TYPES)
assert len(EDGE_TYPE_DIMS) == len(set(EDGE_TYPE_DIMS))

def edge_type_counts(G):
    counts = {dim: 0 for dim in EDGE_TYPE_DIMS}
    for (_, _, data) in G.edges(data=True):
        norm = _normalize(data.get('edge_type'))
        if norm in counts:
            counts[norm] += 1
    return counts

def edge_type_vector(G):
    counts = edge_type_counts(G)
    return [counts[dim] for dim in EDGE_TYPE_DIMS]

if __name__ == '__main__':
    import json
    from Feature_Initialization import load_graph

    fp = '/csse/research/contructive-learning/CAM-LDS/graphs/collection/T1056-001/graph_6_plugin-26_client.json'
    with open(fp) as f:
        G = load_graph(json.load(f))

    counts = edge_type_counts(G)
    total = sum(counts.values())
    nonzero = sorted(((k, v) for (k, v) in counts.items() if v), key=lambda kv: -kv[1])

    print('Graph:', fp)
    print('Total edges:', total, '| edge types with vocab match:', sum(counts.values()))
    print('Nonzero dims:', len(nonzero), '/', len(EDGE_TYPE_DIMS))
    print()
    print(f'{"edge_type":20s} {"count":6s} {"raw%":6s}')
    for (k, v) in nonzero:
        print(f'{k:20s} {v:<6d} {100 * v / total:5.1f}%')
