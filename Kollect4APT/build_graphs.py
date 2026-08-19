import glob
import json
import os

import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm

PARSED_EDGES_DIR = '/csse/research/contructive-learning/Kollect4APT/parsed_edges'
GRAPH_DIR = '/csse/research/contructive-learning/Kollect4APT/graph'


def read_single_kellect_graph(edges_path, node_details):
    g = nx.MultiDiGraph()
    lines = []
    with open(edges_path, 'r', errors='replace') as f:
        for l in f:
            split_line = l.rstrip('\n').split('\t')
            if len(split_line) != 6:
                continue
            src, src_type, dst, dst_type, edge_type, ts = split_line
            try:
                ts = int(ts)
            except ValueError:
                continue
            lines.append([src, dst, src_type, dst_type, edge_type, ts])
    lines.sort(key=lambda l: l[5])

    node_map = {}
    node_cnt = 0

    for src, dst, src_type, dst_type, edge_type, ts in lines:
        src_name = node_details.get(src, {}).get('name', 'UNKNOWN')
        dst_name = node_details.get(dst, {}).get('name', 'UNKNOWN')

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
            g.add_edge(node_map[src], node_map[dst], key=key + 1,
                       edge_type=edge_type, ts=ts,
                       src_uuid=src, dst_uuid=dst,
                       src_name=src_name, dst_name=dst_name)

    return node_map, g


def save_graph(node_map, g, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    node_map_path = os.path.join(out_dir, 'node_map.json')
    with open(node_map_path, 'w', encoding='utf-8') as f:
        json.dump(node_map, f, indent=2)

    graph_path = os.path.join(out_dir, 'graph.json')
    graph_data = json_graph.node_link_data(g)
    for edge in graph_data.get('links', []):
        edge.pop('source', None)
        edge.pop('target', None)
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)


def main(parsed_edges_dir=PARSED_EDGES_DIR, graph_dir=GRAPH_DIR):
    with open(os.path.join(parsed_edges_dir, 'node_details.json'), 'r', encoding='utf-8') as f:
        node_details = json.load(f)

    edges_files = sorted(glob.glob(f'{parsed_edges_dir}/*/edges.txt'))
    print(f'Found {len(edges_files)} sample edge files to build graphs from.')

    total_nodes = 0
    total_edges = 0

    for edges_path in tqdm(edges_files, desc='Building graphs'):
        tag = os.path.basename(os.path.dirname(edges_path))
        node_map, g = read_single_kellect_graph(edges_path, node_details)
        total_nodes += g.number_of_nodes()
        total_edges += g.number_of_edges()
        save_graph(node_map, g, os.path.join(graph_dir, tag))

    print(f'\nBuilt {len(edges_files)} graphs.')
    print(f'  Total nodes : {total_nodes:,}')
    print(f'  Total edges : {total_edges:,}')
    print(f'Saved to: {graph_dir}/<sample_tag>/node_map.json + graph.json')


if __name__ == '__main__':
    main()
