import os
import json
import pickle as pkl
import networkx as nx
from scripts.config import (
    show,
    OUTPUT_IOC, OUTPUT_GRAPHS,
)

node_type_dict = {}   
edge_type_dict = {}   
node_type_cnt  = 0
edge_type_cnt  = 0


def _get_node_type_id(ntype):
    global node_type_cnt
    if ntype not in node_type_dict:
        node_type_dict[ntype] = node_type_cnt
        node_type_cnt += 1
    return node_type_dict[ntype]


def _get_edge_type_id(etype):
    global edge_type_cnt
    if etype not in edge_type_dict:
        edge_type_dict[etype] = edge_type_cnt
        edge_type_cnt += 1
    return edge_type_dict[etype]



REVERSE_EDGE_KEYWORDS = ('READ', 'RECV', 'LOAD')

def _should_reverse(edge_type):
    return any(kw in edge_type for kw in REVERSE_EDGE_KEYWORDS)




def build_graph_from_edges(edge_rows):

    rows = sorted(edge_rows, key=lambda r: int(r['timestamp']))

    g = nx.MultiDiGraph()          
    node_map = {}                 
    node_cnt = 0

    for row in rows:
        src       = row['src']
        dst       = row['dst']
        src_type  = row['src_type']
        dst_type  = row['dst_type']
        edge_type = row['edge_type']

        # reverse causal direction for READ / RECV / LOAD
        if _should_reverse(edge_type):
            src, dst           = dst, src
            src_type, dst_type = dst_type, src_type

        src_type_id  = _get_node_type_id(src_type)
        dst_type_id  = _get_node_type_id(dst_type)
        edge_type_id = _get_edge_type_id(edge_type)

        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, type=src_type_id)
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type=dst_type_id)
            node_cnt += 1

        g.add_edge(node_map[src], node_map[dst], type=edge_type_id)

    return node_map, g



def build_window_graph(window_data):

    edge_rows  = window_data.get('subgraph_edges', [])
    seed_nodes = window_data.get('seed_nodes', [])
    seed_uuids = [s['uuid'] for s in seed_nodes]

    node_map, g = build_graph_from_edges(edge_rows)
    return node_map, g, seed_uuids



def build_seed_subgraphs(window_data):

    seed_nodes = window_data.get('seed_nodes', [])
    all_edges  = window_data.get('subgraph_edges', [])

    neighbours = {}
    for row in all_edges:
        s, d = row['src'], row['dst']
        neighbours.setdefault(s, set()).add(d)
        neighbours.setdefault(d, set()).add(s)

    seed_graphs = []
    for seed in seed_nodes:
        seed_uuid   = seed['uuid']
        seed_reason = seed.get('seed_reason', '')

        hop1           = neighbours.get(seed_uuid, set())
        hop2           = set()
        for h in hop1:
            hop2 |= neighbours.get(h, set())
        subgraph_uuids = {seed_uuid} | hop1 | hop2

        local_edges = [
            row for row in all_edges
            if row['src'] in subgraph_uuids and row['dst'] in subgraph_uuids
        ]

        node_map, g = build_graph_from_edges(local_edges)


        if seed_uuid not in node_map:
            ntype_id = _get_node_type_id(seed.get('node_type', 'SUBJECT_PROCESS'))
            idx = len(node_map)
            node_map[seed_uuid] = idx
            g.add_node(idx, type=ntype_id)

        seed_graphs.append({
            'seed_uuid'  : seed_uuid,
            'seed_reason': seed_reason,
            'node_map'   : node_map,
            'graph'      : g,
        })

    return seed_graphs



def build_graphs(maps):

    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)

    malicious_uuids = maps['malicious_uuids']
    id_nodename_map = maps['id_nodename_map']
    id_nodetype_map = maps['id_nodetype_map']

    ws_path = OUTPUT_IOC + 'window_subgraphs.json'
    if not os.path.exists(ws_path):
        print('  window_subgraphs.json not found — run create_subgraph first.')
        return

    with open(ws_path, 'r', encoding='utf-8') as f:
        all_windows = json.load(f)

    window_ids = sorted(all_windows.keys())
    print('  Windows found : {}'.format(len(window_ids)))

    print()
    show('  Building window graphs ...')
    window_graphs    = {}   
    window_node_maps = {}
    for wkey in window_ids:
        wdata = all_windows[wkey]
        node_map, g, seed_uuids = build_window_graph(wdata)
        window_graphs[wkey]    = {'node_map': node_map, 'graph': g, 'seed_uuids': seed_uuids}
        window_node_maps[wkey] = node_map
        print('    {} : nodes={}, edges={}, seeds={}'.format(
            wkey, g.number_of_nodes(), g.number_of_edges(), len(seed_uuids)))

    
    print()
    show('  Building per-seed 2-hop subgraphs ...')
    all_seed_graphs = {}
    for wkey in window_ids:
        wdata   = all_windows[wkey]
        seed_gs = build_seed_subgraphs(wdata)
        all_seed_graphs[wkey] = [
            {
                'seed_uuid'  : sg['seed_uuid'],
                'seed_reason': sg['seed_reason'],
                'node_map'   : sg['node_map'],
                'graph'      : sg['graph'],   
            }
            for sg in seed_gs
        ]
        print('    {} : {} seed subgraphs'.format(wkey, len(seed_gs)))

    # ── save PKL files (raw NetworkX DiGraph objects) ─────────────────────
    pkl.dump(window_graphs,
             open(OUTPUT_GRAPHS + 'window_graphs.pkl', 'wb'))
    pkl.dump(all_seed_graphs,
             open(OUTPUT_GRAPHS + 'seed_graphs.pkl', 'wb'))

    # ── save human-readable JSON (src uuid → dst uuid, edge_type name) ────
    inv_edge_type = {v: k for k, v in edge_type_dict.items()}

    def _graph_to_json(g, node_map):
        inv_node_map = {v: k for k, v in node_map.items()}   # int → uuid
        return {
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges(),
            'edges': [
                {
                    'src'      : inv_node_map[u],
                    'dst'      : inv_node_map[v],
                    'edge_type': inv_edge_type.get(data['type'], str(data['type'])),
                }
                for u, v, _key, data in g.edges(keys=True, data=True)
            ],
        }

    # ── helper: numeric JSON (integer node IDs + integer edge type IDs) ──────
    def _graph_to_numeric_json(g):
        return {
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges(),
            'edges': [
                [u, v, data['type']]          # [src_int, dst_int, edge_type_int]
                for u, v, _key, data in g.edges(keys=True, data=True)
            ],
        }

    # ── human-readable JSON (UUID strings + edge type names) ─────────────
    wg_json     = {}
    wg_num_json = {}
    for wkey, v in window_graphs.items():
        wg_json[wkey]     = _graph_to_json(v['graph'], v['node_map'])
        wg_num_json[wkey] = _graph_to_numeric_json(v['graph'])

    with open(OUTPUT_GRAPHS + 'window_graphs.json', 'w', encoding='utf-8') as f:
        json.dump(wg_json, f, indent=2)
    with open(OUTPUT_GRAPHS + 'window_graphs_numeric.json', 'w', encoding='utf-8') as f:
        json.dump(wg_num_json, f, indent=2)

    sg_json     = {}
    sg_num_json = {}
    for wkey, entries in all_seed_graphs.items():
        sg_json[wkey] = [
            {
                'seed_uuid'  : e['seed_uuid'],
                'seed_reason': e['seed_reason'],
                'node_map'   : e['node_map'],
                **_graph_to_json(e['graph'], e['node_map']),
            }
            for e in entries
        ]
        sg_num_json[wkey] = [
            {
                'seed_uuid'  : e['seed_uuid'],
                'seed_reason': e['seed_reason'],
                **_graph_to_numeric_json(e['graph']),
            }
            for e in entries
        ]

    with open(OUTPUT_GRAPHS + 'seed_graphs.json', 'w', encoding='utf-8') as f:
        json.dump(sg_json, f, indent=2)
    with open(OUTPUT_GRAPHS + 'seed_graphs_numeric.json', 'w', encoding='utf-8') as f:
        json.dump(sg_num_json, f, indent=2)

    # ── save type dicts ───────────────────────────────────────────────────
    with open(OUTPUT_GRAPHS + 'node_type_dict.json', 'w', encoding='utf-8') as f:
        json.dump(node_type_dict, f, indent=2)
    with open(OUTPUT_GRAPHS + 'edge_type_dict.json', 'w', encoding='utf-8') as f:
        json.dump(edge_type_dict, f, indent=2)

    # ── save node maps as JSON (uuid → integer index) ────────────────────
    # window_node_maps.json  : {wkey: {uuid: int}}
    # seed_node_maps.json    : {wkey: [{seed_uuid, seed_reason, node_map: {uuid: int}}]}

    with open(OUTPUT_GRAPHS + 'window_node_maps.json', 'w', encoding='utf-8') as f:
        json.dump(window_node_maps, f, indent=2)
    print('  window_node_maps.json saved')

    seed_node_maps_json = {
        wkey: [
            {
                'seed_uuid'  : e['seed_uuid'],
                'seed_reason': e['seed_reason'],
                'node_map'   : e['node_map'],   # {uuid: int}
            }
            for e in entries
        ]
        for wkey, entries in all_seed_graphs.items()
    }
    with open(OUTPUT_GRAPHS + 'seed_node_maps.json', 'w', encoding='utf-8') as f:
        json.dump(seed_node_maps_json, f, indent=2)
    print('  seed_node_maps.json saved')

   