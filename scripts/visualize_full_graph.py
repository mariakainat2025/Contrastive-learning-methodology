
import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

SCRIPTS_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.dirname(SCRIPTS_DIR)
import sys
PROJECT_ROOT = BASE_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from scripts.config import OUTPUT_GRAPHS, OUTPUT_VIZ
GRAPH_PATH     = os.path.join(OUTPUT_GRAPHS, 'graph_output_all.json')
OUTPUT_VIZ_DIR = OUTPUT_VIZ.rstrip(os.sep)

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

def _short_name(name, uuid='', maxlen=20):
    if not name or name == 'UNKNOWN':
        return uuid[-8:] if uuid else '?'
    parts = name.replace('\\', '/').split('/')
    short = parts[-1] if parts[-1] else (parts[-2] if len(parts) > 1 else name)
    return short[:maxlen]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true',
                        help='Save PNG to output/theia/graphs/viz/')
    parser.add_argument('--uuid', type=str, default=None,
                        help='Show only the subgraph around this UUID')
    args = parser.parse_args()

    if not os.path.exists(GRAPH_PATH):
        print('  ERROR: {} not found. Run main.py first.'.format(GRAPH_PATH))
        return

    print('  Loading full graph from: {}'.format(GRAPH_PATH))
    with open(GRAPH_PATH, 'r') as f:
        data = json.load(f)

    G = nx.MultiDiGraph()
    uuid_to_id = {}
    for n in data.get('nodes', []):
        nid  = n['id']
        uuid = n.get('uuid', str(nid))
        uuid_to_id[uuid] = uuid
        G.add_node(uuid,
                   type=n.get('type', ''),
                   name=n.get('name', ''),
                   uuid=uuid)
    for e in data.get('links', []):
        src = e.get('src_uuid', e.get('source'))
        dst = e.get('dst_uuid', e.get('target'))
        if src is None or dst is None:
            continue
        G.add_edge(src, dst,
                   edge_type=e.get('edge_type', ''),
                   ts=e.get('ts', 0))

    if args.uuid:
        target_uuid = args.uuid.strip()
        if target_uuid not in uuid_to_id:
            print('  ERROR: UUID {} not found in graph.'.format(target_uuid))
            return
        seed_node = target_uuid
        successors   = nx.descendants(G, seed_node)
        predecessors = nx.ancestors(G, seed_node)
        sub_nodes    = successors | predecessors | {seed_node}
        G = G.subgraph(sub_nodes).copy()
        print('  Filtering to UUID: {}'.format(target_uuid))
        print('  Subgraph — Nodes : {}'.format(G.number_of_nodes()))
        print('  Subgraph — Edges : {}'.format(G.number_of_edges()))
    else:
        print('  Nodes : {}'.format(G.number_of_nodes()))
        print('  Edges : {}'.format(G.number_of_edges()))

    if G.number_of_nodes() > 200:
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=30)
    else:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2)

    G_undir  = nx.Graph(G)
    art_pts  = set(nx.articulation_points(G_undir)) if G_undir.number_of_nodes() > 1 else set()

    normal_nodes = [n for n in G.nodes() if n not in art_pts]
    art_list     = list(art_pts)

    normal_cols  = [_colour(G.nodes[n].get('type', '')) for n in normal_nodes]
    normal_sizes = [500 for _ in normal_nodes]

    art_cols  = [_colour(G.nodes[n].get('type', '')) for n in art_list]
    art_sizes = [900 for _ in art_list]

    node_labels = {
        n: _short_name(G.nodes[n].get('name', ''), G.nodes[n].get('uuid', ''))
        for n in G.nodes()
    }

    edge_label_map = defaultdict(set)
    for u, v, edata in G.edges(data=True):
        etype = edata.get('edge_type', '')
        if etype:
            edge_label_map[(u, v)].add(etype.replace('EVENT_', ''))
    edge_labels = {k: '/'.join(sorted(v)) for k, v in edge_label_map.items()}

    G_simple = nx.DiGraph()
    G_simple.add_nodes_from(G.nodes())
    for u, v in set((u, v) for u, v in G.edges()):
        G_simple.add_edge(u, v)

    fig, ax = plt.subplots(figsize=(20, 14))

    if normal_nodes:
        nx.draw_networkx_nodes(G, pos,
                               nodelist=normal_nodes,
                               node_color=normal_cols,
                               node_size=normal_sizes,
                               ax=ax, alpha=0.92)

    if art_list:
        nx.draw_networkx_nodes(G, pos,
                               nodelist=art_list,
                               node_color=art_cols,
                               node_size=art_sizes,
                               node_shape='D',
                               edgecolors='black',
                               linewidths=2.0,
                               ax=ax, alpha=0.95)

    nx.draw_networkx_edges(G_simple, pos,
                           ax=ax,
                           arrows=True,
                           arrowsize=15,
                           edge_color='#555555',
                           width=1.5,
                           alpha=0.7,
                           connectionstyle='arc3,rad=0.08')

    if G.number_of_nodes() <= 200:
        nx.draw_networkx_labels(G, pos,
                                labels=node_labels,
                                ax=ax,
                                font_size=7,
                                font_color='black')

        nx.draw_networkx_edge_labels(G_simple, pos,
                                     edge_labels=edge_labels,
                                     ax=ax,
                                     font_size=6,
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
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9)
    title = 'UUID: {}  |  nodes={}  edges={}  art_pts={}'.format(
        args.uuid, G.number_of_nodes(), G.number_of_edges(), len(art_pts)
    ) if args.uuid else 'Full Provenance Graph  |  nodes={}  edges={}  art_pts={}'.format(
        G.number_of_nodes(), G.number_of_edges(), len(art_pts))
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    plt.tight_layout()

    os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
    if args.save:
        fname = 'uuid_{}.png'.format(args.uuid[:8]) if args.uuid else 'full_graph.png'
        out_path = os.path.join(OUTPUT_VIZ_DIR, fname)
    else:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        out_path = tmp.name
        tmp.close()

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved : {}'.format(out_path))
    os.system('open "{}"'.format(out_path))

if __name__ == '__main__':
    main()
