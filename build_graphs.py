import os
import json
import pickle as pkl
import networkx as nx

from scripts.config import (
    show,
    OUTPUT_WINDOWS, OUTPUT_GRAPHS,
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
    """True if this event type's causal direction should be reversed."""
    return any(kw in edge_type for kw in REVERSE_EDGE_KEYWORDS)


def build_graph_from_edges(edge_rows):
    rows = sorted(edge_rows, key=lambda r: int(r['timestamp']))

    g        = nx.MultiDiGraph()
    node_map = {}
    node_cnt = 0

    for row in rows:
        src       = row['src']
        dst       = row['dst']
        src_type  = row['src_type']
        dst_type  = row['dst_type']
        edge_type = row['edge_type']

       
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
    """Build a NetworkX MultiDiGraph from a window's subgraph edges."""
    edge_rows = window_data.get('subgraph_edges', [])
    node_map, g = build_graph_from_edges(edge_rows)
    return node_map, g


def run_graph_builder(maps):
    """Build NetworkX window graphs for malicious and benign windows and save to disk.

    Parameters
    ----------
    maps : dict  (from parse_provenance + extract_windows; used for metadata only)
    """
    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)

    ws_path = OUTPUT_WINDOWS + 'window_subgraphs.json'
    if not os.path.exists(ws_path):
        print('  ERROR: window_subgraphs.json not found — run extract_windows first.')
        return

    with open(ws_path, 'r', encoding='utf-8') as f:
        all_windows = json.load(f)

    window_ids = sorted(all_windows.keys())
    print('  Windows loaded : {:,}'.format(len(window_ids)))
    print()

    
    show('  Building full window graphs ...')
    window_graphs    = {}
    window_node_maps = {}
    for wkey in window_ids:
        wdata = all_windows[wkey]
        node_map, g = build_window_graph(wdata)
        window_graphs[wkey]    = {'node_map': node_map, 'graph': g}
        window_node_maps[wkey] = node_map
        print('    {}  nodes={:,}  edges={:,}'.format(
            wkey, g.number_of_nodes(), g.number_of_edges()))

  
    pkl.dump(window_graphs,
             open(OUTPUT_GRAPHS + 'window_graphs.pkl', 'wb'))
    print()
    print(' window_graphs.pkl')

   
    inv_edge_type = {v: k for k, v in edge_type_dict.items()}

    def _graph_to_json(g, node_map):
        inv = {v: k for k, v in node_map.items()}
        return {
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges(),
            'edges': [
                {'src': inv[u], 'dst': inv[v],
                 'edge_type': inv_edge_type.get(data['type'], str(data['type']))}
                for u, v, _k, data in g.edges(keys=True, data=True)
            ],
        }

    def _graph_to_numeric_json(g):
        return {
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges(),
            'edges': [
                [u, v, data['type']]
                for u, v, _k, data in g.edges(keys=True, data=True)
            ],
        }

    wg_json     = {wk: _graph_to_json(v['graph'], v['node_map'])
                   for wk, v in window_graphs.items()}
    wg_num_json = {wk: _graph_to_numeric_json(v['graph'])
                   for wk, v in window_graphs.items()}

    with open(OUTPUT_GRAPHS + 'window_graphs.json', 'w', encoding='utf-8') as f:
        json.dump(wg_json, f, indent=2)
    with open(OUTPUT_GRAPHS + 'window_graphs_numeric.json', 'w', encoding='utf-8') as f:
        json.dump(wg_num_json, f, indent=2)

 
    with open(OUTPUT_GRAPHS + 'node_type_dict.json', 'w', encoding='utf-8') as f:
        json.dump(node_type_dict, f, indent=2)
    with open(OUTPUT_GRAPHS + 'edge_type_dict.json', 'w', encoding='utf-8') as f:
        json.dump(edge_type_dict, f, indent=2)

 
    with open(OUTPUT_GRAPHS + 'window_node_maps.json', 'w', encoding='utf-8') as f:
        json.dump(window_node_maps, f, indent=2)

    print('window_graphs.json / _numeric.json')
    print('node_type_dict.json  edge_type_dict.json')
    print('window_node_maps.json')


    bs_path = OUTPUT_WINDOWS + 'benign_window_subgraphs.json'
    if os.path.exists(bs_path):
        print()
        show('  Building benign window graphs ...')
        with open(bs_path, 'r', encoding='utf-8') as f:
            all_benign = json.load(f)

        benign_graphs    = {}
        benign_node_maps = {}
        for bkey in sorted(all_benign.keys()):
            bdata = all_benign[bkey]
            node_map, g = build_window_graph(bdata)
            benign_graphs[bkey]    = {'node_map': node_map, 'graph': g}
            benign_node_maps[bkey] = node_map
            print('    {}  nodes={:,}  edges={:,}'.format(
                bkey, g.number_of_nodes(), g.number_of_edges()))

        
        pkl.dump(benign_graphs,
                 open(OUTPUT_GRAPHS + 'benign_graphs.pkl', 'wb'))

       
        inv_edge_type = {v: k for k, v in edge_type_dict.items()}
        bg_json     = {bk: _graph_to_json(v['graph'], v['node_map'])
                       for bk, v in benign_graphs.items()}
        bg_num_json = {bk: _graph_to_numeric_json(v['graph'])
                       for bk, v in benign_graphs.items()}

        with open(OUTPUT_GRAPHS + 'benign_graphs.json', 'w', encoding='utf-8') as f:
            json.dump(bg_json, f, indent=2)
        with open(OUTPUT_GRAPHS + 'benign_graphs_numeric.json', 'w', encoding='utf-8') as f:
            json.dump(bg_num_json, f, indent=2)
        with open(OUTPUT_GRAPHS + 'benign_node_maps.json', 'w', encoding='utf-8') as f:
            json.dump(benign_node_maps, f, indent=2)

        
        with open(OUTPUT_GRAPHS + 'node_type_dict.json', 'w', encoding='utf-8') as f:
            json.dump(node_type_dict, f, indent=2)
        with open(OUTPUT_GRAPHS + 'edge_type_dict.json', 'w', encoding='utf-8') as f:
            json.dump(edge_type_dict, f, indent=2)

        print()
        print('  benign_graphs.pkl')
        print('  benign_graphs.json / _numeric.json')
        print('  benign_node_maps.json')
    else:
        print('  benign_window_subgraphs.json not found — skipping benign graphs')

    print()
    print('  Node types registered : {:,}'.format(len(node_type_dict)))
    print('  Edge types registered : {:,}'.format(len(edge_type_dict)))
    print()
    show('build_graphs.py — DONE')
