
import os
import sys
import json
from pathlib import Path
from networkx.readwrite import json_graph

PROJECT_ROOT = '/csse/research/contructive-learning'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.reduce_graph import reduce_directory_cascade

CAM_LDS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR   = os.path.join(CAM_LDS_DIR, 'graphs')
OUT_DIR      = os.path.join(CAM_LDS_DIR, 'reduce_graph')
SUMMARY_PATH = os.path.join(OUT_DIR, 'reduce_summary.json')


def _edge_key(u, v, attrs):
    return (u, v, attrs.get('edge_type'), attrs.get('event_id'))


def _strip_cascade_edges(G, removed_nodes):
    for item in removed_nodes:
        parent = item['node']
        for p in item['processes']:
            proc = p['node']
            for u, v, k in list(G.edges(keys=True)):
                if (u == proc and v == parent) or (u == parent and v == proc):
                    G.remove_edge(u, v, k)
        if parent in G and G.degree(parent) == 0:
            G.remove_node(parent)


def batch_main(graphs_dir=GRAPHS_DIR, out_dir=OUT_DIR, summary_path=SUMMARY_PATH):
    files = sorted(Path(graphs_dir).glob('*/*/graph_*.json'))
    print(f'Found {len(files)} graph JSON files under {graphs_dir}')

    all_merges = []
    total_merged_graphs = 0
    total_cascades = 0

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        G = json_graph.node_link_graph(data)

        edges_before = {_edge_key(u, v, a): (u, v, a) for u, v, a in G.edges(data=True)}

        G, combine, removed_nodes = reduce_directory_cascade(G)
        node_name = {n: d.get('name', n) for n, d in G.nodes(data=True)}  
        _strip_cascade_edges(G, removed_nodes)

        edges_after = {_edge_key(u, v, a) for u, v, a in G.edges(data=True)}
        removed_edges_all = [edges_before[k] for k in edges_before if k not in edges_after]

        rel = fpath.relative_to(graphs_dir)
        out_subdir = Path(out_dir) / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        reduced_name = fpath.name.replace('graph_', 'reduced_', 1)
        with open(out_subdir / reduced_name, 'w') as f:
            json.dump(json_graph.node_link_data(G), f, indent=2)

        if combine > 0:
            total_merged_graphs += 1
        total_cascades += combine

        for item in removed_nodes:
            parent = item['node']
            child  = item['kept_node']
            edges_removed = [
                {'src': node_name.get(u, u), 'dst': node_name.get(v, v),
                 'edge_type': a.get('edge_type'), 'event_id': a.get('event_id')}
                for u, v, a in removed_edges_all if u == parent or v == parent
            ]
            all_merges.append({
                'tactic'         : G.graph.get('tactic'),
                'technique'      : G.graph.get('technique'),
                'step'           : G.graph.get('step'),
                'host'           : G.graph.get('host'),
                'removed_path'   : item.get('name', parent),
                'kept_path'      : item.get('kept_name', child),
                'processes'      : [p['name'] for p in item['processes']],
                'edges_removed'  : edges_removed,
            })

    summary = {
        'total_graphs'        : len(files),
        'total_merged_graphs' : total_merged_graphs,
        'total_cascades'      : total_cascades,
        'merges'              : all_merges,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f'Total graphs        : {len(files)}')
    print(f'Graphs with merges  : {total_merged_graphs}')
    print(f'Total cascades      : {total_cascades}')
    print(f'Summary saved -> {summary_path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs-dir', default=GRAPHS_DIR)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--summary-path', default=SUMMARY_PATH)
    args = ap.parse_args()
    batch_main(args.graphs_dir, args.out_dir, args.summary_path)

