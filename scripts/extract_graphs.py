import os
import json
from tqdm import tqdm
import networkx as nx
from networkx.readwrite import json_graph

from scripts.config import (
    OUTPUT_PARSED, OUTPUT_GRAPHS, EDGES_FILE,
)

def read_single_graph(dataset, path, tag=None):

    with open(os.path.join(OUTPUT_PARSED, 'node_type_map.json'), 'r') as f:
        node_type_dict = json.load(f)
    with open(os.path.join(OUTPUT_PARSED, 'edge_type_map.json'), 'r') as f:
        edge_type_dict = json.load(f)
    with open(os.path.join(OUTPUT_PARSED, 'names.json'), 'r') as f:
        uuid2name = json.load(f)

    g = nx.MultiDiGraph()
    print('converting {} ...'.format(path))
    skipped = 0
    lines = []
    with open(path, 'r', errors='replace') as f:
        for l in f.readlines():
            if '\x00' in l:
                skipped += 1
                continue
            split_line = l.rstrip('\n').split('\t')
            if len(split_line) != 6:
                skipped += 1
                continue
            src, src_type, dst, dst_type, edge_type, ts = split_line
            try:
                ts = int(ts)
            except ValueError:
                skipped += 1
                continue
            if 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type:
                lines.append([dst, src, dst_type, src_type, edge_type, ts])
            else:
                lines.append([src, dst, src_type, dst_type, edge_type, ts])
    if skipped:
        print('  Skipped {:,} malformed lines (null bytes / wrong fields)'.format(skipped))
    lines.sort(key=lambda l: l[5])

    node_map = {}
    node_cnt = 0

    for l in tqdm(lines):
        src, dst, src_type, dst_type, edge_type, ts = l

        if src_type not in node_type_dict or dst_type not in node_type_dict:
            continue

        src_name = uuid2name.get(src, "UNKNOWN")
        dst_name = uuid2name.get(dst, "UNKNOWN")

        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, type=src_type, ts=ts, name=src_name, uuid=src)
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type=dst_type, ts=ts, name=dst_name, uuid=dst)
            node_cnt += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], key=0,
                       edge_type=edge_type, ts=ts,
                       src_uuid=src, dst_uuid=dst,
                       src_name=src_name, dst_name=dst_name)
        elif edge_type not in g[node_map[src]][node_map[dst]][0]['edge_type']:
            key = list(g.get_edge_data(node_map[src], node_map[dst]).keys())[-1]
            g.add_edge(node_map[src], node_map[dst], key=key+1,
                       edge_type=edge_type, ts=ts,
                       src_uuid=src, dst_uuid=dst,
                       src_name=src_name, dst_name=dst_name)

    # Graph summary 
    print("\n--- Graph Summary ---")
    print("  Nodes : {}".format(g.number_of_nodes()))
    print("  Edges : {}".format(g.number_of_edges()))

    # ── Save outputs
    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)

    suffix = '_{}'.format(tag) if tag else ''
    node_map_path = os.path.join(OUTPUT_GRAPHS, 'node_map_output{}.json'.format(suffix))
    with open(node_map_path, 'w', encoding='utf-8') as f:
        json.dump(node_map, f, indent=2)
    print("  node_map saved : {}".format(node_map_path))

    graph_path = os.path.join(OUTPUT_GRAPHS, 'graph_output{}.json'.format(suffix))
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(json_graph.node_link_data(g), f, indent=2)
    print("  graph saved    : {}".format(graph_path))

    return node_map, g

THETA_MAX_NS  = 10 * 60 * int(1e9)    
MIN_NODES     = 1
MIN_EDGES     = 1
MIN_TYPES     = 2
# MAX_NODES     = 300                  

def temporal_split(E_dep, theta_max):
   
    if not E_dep:
        return []

    sorted_edges = sorted(E_dep, key=lambda e: e['timestamp'])
    segments     = []
    start        = 0

    for i in range(1, len(sorted_edges)):
        delta_t = sorted_edges[i]['timestamp'] - sorted_edges[i - 1]['timestamp']
        if delta_t >= theta_max:
            segments.append(sorted_edges[start:i])
            start = i

    segments.append(sorted_edges[start:])
    return segments


def dfs(node, G_prov, visited, V_dep):
    stack = [node]
    while stack:
        n = stack.pop()
        if n in visited:
            V_dep.add(n)
            continue
        visited.add(n)
        V_dep.add(n)
        for neighbor in G_prov.successors(n):
            if neighbor not in visited:
                stack.append(neighbor)
            else:
                V_dep.add(neighbor)


def subgraph_partition(G_prov, theta_max=THETA_MAX_NS):
   

    if G_prov.number_of_nodes() == 0:
        return []

    V = list(G_prov.nodes())
    visited              = set()
    seen                 = set()
    S                    = []
    dep_id               = 0

    for node in V:
        if node in visited:
            continue

        # DFS
        V_dep = set()
        dfs(node, G_prov, visited, V_dep)
        V_dep = frozenset(V_dep)

        E_dep = [
            {'src': src, 'dst': dst, 'timestamp': data['ts'], 'edge_type': data['edge_type']}
            for src, dst, key, data in G_prov.edges(V_dep, keys=True, data=True)
            if dst in V_dep
        ]
        if not E_dep:
            continue

        # temporal split → system behaviour subgraphs
        segs = temporal_split(E_dep, theta_max)
        valid_segs = []
        for E_seg in segs:
            V_seg      = {e['src'] for e in E_seg} | {e['dst'] for e in E_seg}
            node_types = {G_prov.nodes[n]['type'] for n in V_seg}
            if len(V_seg) < MIN_NODES:
                continue
            if len(E_seg) < MIN_EDGES:
                continue
            if len(node_types) < MIN_TYPES:
                continue
            sig = frozenset(V_seg)
            if sig in seen:
                continue
            seen.add(sig)
            valid_segs.append((V_seg, E_seg))

        total_parts = len(valid_segs)
        for part_idx, (V_seg, E_seg) in enumerate(valid_segs):
            S.append((V_seg, E_seg, node, dep_id, part_idx, total_parts))

        if valid_segs:
            dep_id += 1

    S.sort(key=lambda x: (x[3], x[4]))
    return S


def run_extract_windows(g, tag='subgraphs'):
    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)

    print('  Running subgraph partition (Algorithm 1) ...')
    subgraphs = subgraph_partition(g)
    print('  Subgraphs found : {:,}'.format(len(subgraphs)))


    serializable = []
    graphs       = []
    for V_seg, E_seg, seed, dep_id, part_idx, total_parts in subgraphs:

        G = nx.MultiDiGraph()
        for e in E_seg:
            if e['src'] not in G:
                G.add_node(e['src'], **g.nodes[e['src']])
            if e['dst'] not in G:
                G.add_node(e['dst'], **g.nodes[e['dst']])
            G.add_edge(e['src'], e['dst'], edge_type=e['edge_type'])

        graphs.append(G)
        serializable.append({
            'seed'       : seed,
            'seed_name'  : g.nodes[seed].get('name', str(seed)),
            'seed_uuid'  : g.nodes[seed].get('uuid', 'N/A'),
            'dep_id'     : dep_id,
            'part_idx'   : part_idx,
            'total_parts': total_parts,
            'nodes'      : list(G.nodes(data=True)),
            'edges'      : list(G.edges(keys=True, data=True)),
        })

    out_path = os.path.join(OUTPUT_GRAPHS, '{}.json'.format(tag))
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print('{}.json saved : {}'.format(tag, out_path))

    # print summary
    print()
    print('  {:<6}  {:<8}  {:<8}  {:<38}  {:<20}  {:<24}  {}'.format(
          'Idx', 'Nodes', 'Edges', 'Seed UUID', 'Seed name', 'Partition', 'Node types'))
    print('  ' + '-' * 130)
    for i, (sg, G) in enumerate(zip(serializable[:5], graphs[:5])):
        types = set(data.get('type', '?') for _, data in G.nodes(data=True))
        part_info = 'dep#{} part {}/{}'.format(
            sg['dep_id'], sg['part_idx'] + 1, sg['total_parts'])

      
        print('  {:<6}  {:<8}  {:<8}  {:<38}  {:<20}  {:<24}  {}'.format(
              i,
              len(sg['nodes']),
              len(sg['edges']),
              sg['seed_uuid'],
              sg['seed_name'][:20],
              part_info,
              ', '.join(sorted(types))))

    return subgraphs



   



