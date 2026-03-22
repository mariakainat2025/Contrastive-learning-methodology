# python scripts/visualize_subgraphs.py --idx 0 --save

import os
import sys
import json
import argparse
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

SCRIPTS_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.dirname(SCRIPTS_DIR)
SUBGRAPHS_PATH = os.path.join(BASE_DIR, 'output', 'theia', 'graphs', 'subgraphs.json')
GRAPH_PATH     = os.path.join(BASE_DIR, 'output', 'theia', 'graphs', 'graph_output.json')
OUTPUT_VIZ_DIR = os.path.join(BASE_DIR, 'output', 'theia', 'graphs', 'viz')

TYPE_COLOURS = {
    'SUBJECT_PROCESS'   : '#e74c3c',
    'NetFlowObject'     : '#3498db',
    'FILE_OBJECT_BLOCK' : '#2ecc71',
    'MemoryObject'      : '#f39c12',
}
DEFAULT_COLOUR = '#95a5a6'


def _colour(node_type):
    for key, col in TYPE_COLOURS.items():
        if key in node_type:
            return col
    return DEFAULT_COLOUR


def _short_name(name, uuid='', maxlen=12):
    if not name or name == 'UNKNOWN':
        return uuid[-8:] if uuid else '?'
    parts = name.replace('\\', '/').split('/')
    short = parts[-1] if parts[-1] else (parts[-2] if len(parts) > 1 else name)
    return short[:maxlen]


def load_full_graph():
    if not os.path.exists(GRAPH_PATH):
        print('  ERROR: {} not found'.format(GRAPH_PATH))
        return {}
    with open(GRAPH_PATH, 'r') as f:
        data = json.load(f)
    if 'links' not in data and 'edges' in data:
        data['links'] = data['edges']
    G = json_graph.node_link_graph(data)
    node_attrs = {}
    for n, attrs in G.nodes(data=True):
        node_attrs[n] = {
            'type': attrs.get('type', ''),
            'name': attrs.get('name', ''),
            'uuid': attrs.get('uuid', ''),
        }
    return node_attrs


def plot_subgraph(idx, subgraph_data, node_attrs, save=False, tag='subgraph'):
    nodes     = subgraph_data['nodes']
    edges     = subgraph_data['edges']
    seed      = subgraph_data.get('seed')
    seed_name = subgraph_data.get('seed_name', str(seed))

    G = nx.MultiDiGraph()

    for n_entry in nodes:
        if isinstance(n_entry, list):
            n      = n_entry[0]
            stored = n_entry[1] if len(n_entry) > 1 else {}
        else:
            n      = n_entry
            stored = {}
        attrs = node_attrs.get(n, stored)
        G.add_node(n,
                   ntype=attrs.get('type', stored.get('type', '')),
                   name=attrs.get('name',  stored.get('name',  '')),
                   uuid=attrs.get('uuid',  stored.get('uuid',  '')))

    for e in edges:
        if isinstance(e, list):
            src   = e[0]
            dst   = e[1]
            edata = e[2] if len(e) > 2 else {}
            G.add_edge(src, dst, edge_type=edata.get('edge_type', ''))
        else:
            G.add_edge(e['src'], e['dst'], edge_type=e.get('edge_type', ''))

    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=2)

    G_undir = nx.Graph(G)
    art_pts = set(nx.articulation_points(G_undir))

    seed_uuid    = subgraph_data.get('seed_uuid', '')
    node_colours = [_colour(G.nodes[n].get('ntype', '')) for n in G.nodes()]
    node_sizes   = []
    node_shapes_star = []
    for n in G.nodes():
        is_seed = G.nodes[n].get('uuid', '') == seed_uuid
        is_art  = n in art_pts
        if is_seed:
            node_sizes.append(1200)
        elif is_art:
            node_sizes.append(700)
        else:
            node_sizes.append(300)
        node_shapes_star.append(is_art)
    node_labels = {n: _short_name(G.nodes[n].get('name', ''),
                                   G.nodes[n].get('uuid', '')) for n in G.nodes()}

    edge_label_map = defaultdict(set)
    for u, v, data in G.edges(data=True):
        etype = data.get('edge_type', '')
        if etype:
            short_e = etype.split('_')[-1]
            edge_label_map[(u, v)].add(short_e)
    edge_labels = {k: '/'.join(sorted(v)) for k, v in edge_label_map.items()}

    fig, ax = plt.subplots(figsize=(16, 10))

    normal_nodes = [n for n in G.nodes() if n not in art_pts]
    normal_cols  = [_colour(G.nodes[n].get('ntype', '')) for n in normal_nodes]
    normal_sizes = [1200 if G.nodes[n].get('uuid','') == seed_uuid else 300
                    for n in normal_nodes]
    if normal_nodes:
        nx.draw_networkx_nodes(G, pos,
                               nodelist=normal_nodes,
                               node_color=normal_cols,
                               node_size=normal_sizes,
                               ax=ax, alpha=0.92)

    art_list  = list(art_pts)
    art_cols  = [_colour(G.nodes[n].get('ntype', '')) for n in art_list]
    art_sizes = [1200 if G.nodes[n].get('uuid','') == seed_uuid else 700
                 for n in art_list]
    if art_list:
        nx.draw_networkx_nodes(G, pos,
                               nodelist=art_list,
                               node_color=art_cols,
                               node_size=art_sizes,
                               node_shape='D',
                               edgecolors='black',
                               linewidths=2.0,
                               ax=ax, alpha=0.95)

    G_simple = nx.DiGraph()
    G_simple.add_nodes_from(G.nodes())
    for u, v in set((u, v) for u, v in G.edges()):
        G_simple.add_edge(u, v)

    nx.draw_networkx_edges(G_simple, pos,
                           ax=ax,
                           arrows=True,
                           arrowsize=15,
                           edge_color='#555555',
                           width=1.2,
                           alpha=0.7,
                           connectionstyle='arc3,rad=0.08')

    nx.draw_networkx_labels(G, pos,
                            labels=node_labels,
                            ax=ax,
                            font_size=6,
                            font_color='black')

    nx.draw_networkx_edge_labels(G_simple, pos,
                                 edge_labels=edge_labels,
                                 ax=ax,
                                 font_size=5,
                                 font_color='#c0392b',
                                 bbox=dict(boxstyle='round,pad=0.1',
                                           fc='white', alpha=0.5))

    legend_handles = [
        mpatches.Patch(color=col, label=typ)
        for typ, col in TYPE_COLOURS.items()
    ] + [
        mpatches.Patch(color=DEFAULT_COLOUR, label='other'),
        mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2,
                       label='◆ articulation point ({})'.format(len(art_pts))),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    ax.set_title(
        'Subgraph {}  |  nodes={}  edges={}  art_pts={}  seed={}'.format(
            idx, len(nodes), len(edges), len(art_pts), seed_name[:50]),
        fontsize=10)
    ax.axis('off')
    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_VIZ_DIR, '{}_{:03d}.png'.format(tag, idx))
    else:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        out_path = tmp.name
        tmp.close()

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print('  saved : {}'.format(out_path))

    if not save:
        os.system('open "{}"'.format(out_path))

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx',  type=int, default=None,
                        help='Index of subgraph to plot (default: all)')
    parser.add_argument('--save', action='store_true',
                        help='Save PNGs to output/theia/graphs/viz/')
    parser.add_argument('--file', type=str, default=None,
                        help='Path to subgraphs JSON file (default: subgraphs.json)')
    args = parser.parse_args()

    if args.file:
        subgraphs_path = args.file
    else:
        graphs_dir = os.path.dirname(SUBGRAPHS_PATH)
        candidates = [
            os.path.join(graphs_dir, f)
            for f in os.listdir(graphs_dir)
            if f.endswith('.json') and f.startswith('subgraph')
        ]
        if not candidates:
            print('  ERROR: no subgraphs*.json found in {}'.format(graphs_dir))
            sys.exit(1)
        subgraphs_path = max(candidates, key=os.path.getmtime)
        print('  Auto-selected: {}'.format(os.path.basename(subgraphs_path)))

    for path in [subgraphs_path, GRAPH_PATH]:
        if not os.path.exists(path):
            print('  ERROR: {} not found'.format(path))
            sys.exit(1)

    tag = os.path.splitext(os.path.basename(subgraphs_path))[0]

    with open(subgraphs_path, 'r') as f:
        subgraphs = json.load(f)
    print('  Loaded {:,} subgraphs'.format(len(subgraphs)))

    node_attrs = load_full_graph()
    print('  Loaded {:,} node attributes from graph_output.json'.format(len(node_attrs)))

    indices = [args.idx] if args.idx is not None else range(len(subgraphs))
    for idx in indices:
        if idx >= len(subgraphs):
            print('  ERROR: index {} out of range (max {})'.format(
                idx, len(subgraphs) - 1))
            continue
        print('  Plotting subgraph {} ...'.format(idx))
        plot_subgraph(idx, subgraphs[idx], node_attrs, save=args.save, tag=tag)

    if args.save:
        print('  Done. PNGs saved to {}'.format(OUTPUT_VIZ_DIR))


if __name__ == '__main__':
    main()
