
import argparse
import json
import os
import sys
import glob as _glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

import networkx as nx
from networkx.readwrite import json_graph

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.config import OUTPUT_GRAPHS

NODE_STYLE = {
    'SUBJECT_PROCESS'  : {'color': '#E74C3C', 'size': 800},
    'FILE_OBJECT_BLOCK': {'color': '#3498DB', 'size': 400},
    'NetFlowObject'    : {'color': '#2ECC71', 'size': 400},
    'SUBJECT_UNIT'     : {'color': '#F39C12', 'size': 400},
}
DEFAULT_STYLE    = {'color': '#95A5A6', 'size': 300}
REMOVED_COLOR    = '#FF4444'   # bright red for removed nodes
HIGHLIGHT_COLOR  = '#F1C40F'   # gold for queried node

EDGE_COLOR = {
    'EVENT_OPEN'   : '#5DADE2',
    'EVENT_READ'   : '#58D68D',
    'EVENT_WRITE'  : '#EC7063',
    'EVENT_MMAP'   : '#BB8FCE',
    'EVENT_EXECUTE': '#F0B27A',
}
DEFAULT_EDGE_COLOR = '#BDC3C7'



def load_full_graph() -> nx.MultiDiGraph:
    path = os.path.join(OUTPUT_GRAPHS, 'graph_output.json')
    if not os.path.exists(path):
        sys.exit(f'ERROR: full graph not found at {path}\n'
                 f'       Run main.py first.')
    print(f'Loading full graph from {path} ...')
    with open(path, 'r') as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data, directed=True, multigraph=True)
    print(f'  Nodes: {G.number_of_nodes():,}   Edges: {G.number_of_edges():,}')
    return G


def find_node_by_uuid(G: nx.MultiDiGraph, uuid: str):
    uuid_lower = uuid.strip().lower()
    for n, d in G.nodes(data=True):
        if d.get('uuid', '').lower() == uuid_lower:
            return n
    return None


def extract_neighbourhood(G: nx.MultiDiGraph, root: int) -> nx.MultiDiGraph:
    visited, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        stack.extend(G.predecessors(n))
        stack.extend(G.successors(n))
    return G.subgraph(visited).copy()


def load_removed_nodes(tag: str) -> list:
    """
    Load the R2 removed-nodes JSON saved by reduce_graph.
    Searches for any reduction_report_*_removed_nodes.json matching the tag.
    Returns list of dicts with keys: node, uuid, name, type, kept_node, ...
    """
    pattern = os.path.join(OUTPUT_GRAPHS, f'reduction_report_{tag}*_removed_nodes.json')
    files   = _glob.glob(pattern)
    if not files:
        # fall back: any removed_nodes file
        files = _glob.glob(os.path.join(OUTPUT_GRAPHS, '*_removed_nodes.json'))
    if not files:
        return []
    with open(files[0], 'r') as f:
        return json.load(f)


def print_graph_summary(G: nx.MultiDiGraph, title: str = ''):
    if title:
        print(f'\n{"="*60}\n  {title}\n{"="*60}')
    print(f'  Nodes : {G.number_of_nodes()}   Edges : {G.number_of_edges()}')
    print(f'\n  {"Node":>6}  {"Type":<28}  {"UUID":<38}  Name')
    print(f'  {"-"*110}')
    for n, d in sorted(G.nodes(data=True), key=lambda x: x[0]):
        print(f'  {n:>6}  {d.get("type",""):<28}  {d.get("uuid","N/A"):<38}  {d.get("name","")}')
    print(f'\n  {"Src":>6}  {"Dst":>6}  {"EdgeType":<28}  TS')
    print(f'  {"-"*80}')
    for src, dst, k, d in sorted(G.edges(keys=True, data=True),
                                  key=lambda x: x[3].get('ts', 0)):
        print(f'  {src:>6}  {dst:>6}  {d.get("edge_type",""):<28}  {d.get("ts","")}')


def save_graph(G: nx.MultiDiGraph, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(json_graph.node_link_data(G), f, indent=2)
    print(f'  JSON → {out_path}')



def save_png(G: nx.MultiDiGraph, title: str, out_path: str,
             highlight_uuid: str = None,
             removed_nodes: list = None):
    """
    Render G as a PNG.
    - highlight_uuid  : queried node shown in gold
    - removed_nodes   : list of dicts from R2 report; drawn as ghost nodes
                        (dashed red circle) next to their kept node
    """
    removed_nodes = removed_nodes or []
    highlight_uuid_lower = (highlight_uuid or '').strip().lower()

    # Layout on the real graph only (no ghost nodes in layout)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=2.0)

    # Compute a scale unit = ~5% of the plot extent, used for ghost circle radius
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    x_span = max(all_x) - min(all_x) if len(all_x) > 1 else 1
    y_span = max(all_y) - min(all_y) if len(all_y) > 1 else 1
    ghost_r = max(x_span, y_span) * 0.04   # radius for ghost circle

    # Place ghost nodes manually: offset above their kept node
    nodes_in_graph_uuids = {d.get('uuid', '').lower() for _, d in G.nodes(data=True)}
    ghost_entries = []   # list of (ghost_pos, rm_dict)
    for rm in removed_nodes:
        if rm.get('uuid', '').lower() in nodes_in_graph_uuids:
            continue
        kept_n = rm.get('kept_node')
        if kept_n is not None and G.has_node(kept_n):
            kx, ky = pos[kept_n]
            # Place ghost slightly above and to the left of kept node
            gx = kx - ghost_r * 2.5
            gy = ky + ghost_r * 2.5
            ghost_entries.append(((gx, gy), rm))

    fig, ax = plt.subplots(figsize=(24, 18))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_title(title, color='white', fontsize=13, pad=14)

    colors, sizes = [], []
    for n in G.nodes():
        d     = G.nodes[n]
        style = NODE_STYLE.get(d.get('type', ''), DEFAULT_STYLE)
        colors.append(HIGHLIGHT_COLOR if d.get('uuid', '').lower() == highlight_uuid_lower
                      else style['color'])
        sizes.append(style['size'])

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors, node_size=sizes, alpha=0.92)

    import numpy as np
    for (gx, gy), rm in ghost_entries:
        circle = plt.Circle((gx, gy), radius=ghost_r, color=REMOVED_COLOR,
                             fill=True, alpha=0.20, zorder=2)
        border = plt.Circle((gx, gy), radius=ghost_r, color=REMOVED_COLOR,
                             fill=False, linestyle='--', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.add_patch(border)
        # dashed arrow from ghost to kept node
        kept_n = rm['kept_node']
        kx, ky = pos[kept_n]
        ax.annotate('', xy=(kx, ky), xytext=(gx, gy),
                    arrowprops=dict(arrowstyle='->', color=REMOVED_COLOR,
                                    lw=1.5, linestyle='dashed', alpha=0.8),
                    zorder=4)
        short = rm['name'].split('/')[-1][:22] or rm['name'][:22]
        ax.text(gx, gy, f'REMOVED\n{short}',
                ha='center', va='center', fontsize=7,
                color='white', style='italic', zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground=REMOVED_COLOR)])

    edge_colors = [EDGE_COLOR.get(d.get('edge_type', ''), DEFAULT_EDGE_COLOR)
                   for _, _, d in G.edges(data=True)]

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors,
                           arrows=True, arrowsize=14,
                           connectionstyle='arc3,rad=0.08',
                           alpha=0.75, width=1.2)

    labels = {n: (G.nodes[n].get('name', str(n)).split('/')[-1]
                  or G.nodes[n].get('name', str(n)))[:22]
              for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=7, font_color='white')

    legend_patches = [mpatches.Patch(color=v['color'], label=k)
                      for k, v in NODE_STYLE.items()]
    legend_patches += [
        mpatches.Patch(color=HIGHLIGHT_COLOR, label='Queried node'),
        mpatches.Patch(color=REMOVED_COLOR,   label='Removed by R2 (ghost)',
                       alpha=0.5, hatch='//'),
    ]
    ax.legend(handles=legend_patches, loc='upper left',
              facecolor='#2c2c54', labelcolor='white', fontsize=8,
              framealpha=0.85)

    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  PNG  → {out_path}')



def query_full(uuid: str, tag: str = 'theia'):
    G    = load_full_graph()
    root = find_node_by_uuid(G, uuid)
    if root is None:
        sys.exit(f'ERROR: UUID "{uuid}" not found in the full graph.')

    rd = G.nodes[root]
    print(f'\nFound: node={root}  type={rd.get("type")}  name={rd.get("name")}')

    sub   = extract_neighbourhood(G, root)
    title = f'Full graph — neighbourhood of "{rd.get("name")}"'
    print_graph_summary(sub, title=title)

    safe  = uuid.replace(':', '_').replace('/', '_')
    base  = os.path.join(OUTPUT_GRAPHS, f'query_full_{safe}')
    print()
    save_graph(sub, base + '.json')
    save_png(sub, title, base + '.png', highlight_uuid=uuid)



def query_subgraphs(uuid: str, tag: str = 'theia'):
    path = os.path.join(OUTPUT_GRAPHS, f'subgraphs_{tag}.json')
    if not os.path.exists(path):
        sys.exit(f'ERROR: subgraph file not found at {path}')
    print(f'Loading subgraphs from {path} ...')
    with open(path, 'r') as f:
        subgraphs = json.load(f)
    print(f'  Total subgraphs: {len(subgraphs)}')

    removed = load_removed_nodes(tag)
    print(f'  R2 removed nodes loaded: {len(removed)}')

    uuid_lower = uuid.strip().lower()
    matches = []
    for i, sg in enumerate(subgraphs):
        for node_id, node_data in sg['nodes']:
            if isinstance(node_data, dict) and \
               node_data.get('uuid', '').lower() == uuid_lower:
                matches.append((i, sg, node_id, node_data))
                break

    if not matches:
        sys.exit(f'ERROR: UUID "{uuid}" not found in any subgraph.')

    print(f'\nFound in {len(matches)} subgraph(s):')
    for i, sg, node_id, node_data in matches:
        print(f'\n  idx={i}  dep_id={sg["dep_id"]}  part={sg["part_idx"]+1}/{sg["total_parts"]}')
        print(f'  Seed: {sg["seed_name"]}  ({sg["seed_uuid"]})')
        print(f'  Matched: node={node_id}  name={node_data.get("name")}')

        G = nx.MultiDiGraph()
        for nid, ndata in sg['nodes']:
            G.add_node(nid, **ndata)
        for edge in sg['edges']:
            src, dst, key, edata = edge
            G.add_edge(src, dst, key=key, **edata)

        # Find R2 removed nodes whose kept_node is in this subgraph
        sg_node_ids = set(G.nodes())
        relevant_removed = [rm for rm in removed
                            if rm.get('kept_node') in sg_node_ids]
        if relevant_removed:
            print(f'  R2 removed nodes shown as ghost: {len(relevant_removed)}')
            for rm in relevant_removed:
                print(f'    REMOVED node={rm["node"]}  name={rm["name"]}')

        title = (f'Subgraph {i} — dep#{sg["dep_id"]} '
                 f'part {sg["part_idx"]+1}/{sg["total_parts"]}')
        print_graph_summary(G, title=title)

        safe = uuid.replace(':', '_').replace('/', '_')
        base = os.path.join(OUTPUT_GRAPHS, f'query_sub_{safe}_sg{i}')
        print()
        save_graph(G, base + '.json')
        save_png(G, title, base + '.png',
                 highlight_uuid=uuid,
                 removed_nodes=relevant_removed)



def main():
    parser = argparse.ArgumentParser(
        description='Query provenance graph by UUID — outputs PNG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/query_graph.py --uuid 0100D00F-011E-1C00-0000-0000C1AB2000
  python scripts/query_graph.py --uuid 0100D00F-011E-1C00-0000-0000C1AB2000 --sub --tag benign.jon_benign_1k_0001
        """
    )
    parser.add_argument('--uuid', required=True, help='UUID to search for')
    parser.add_argument('--sub',  action='store_true',
                        help='Search subgraph partition instead of full graph')
    parser.add_argument('--tag',  default='theia',
                        help='Dataset tag in file names (default: theia)')
    args = parser.parse_args()

    if args.sub:
        query_subgraphs(args.uuid, tag=args.tag)
    else:
        query_full(args.uuid, tag=args.tag)


if __name__ == '__main__':
    main()
