# python scripts/show_node.py --uuid <UUID>
# python scripts/show_node.py --uuid <UUID> --sub

import argparse
import glob as _glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_GRAPHS = os.path.join(PROJECT_ROOT, 'output', 'theia', 'graphs')

TYPE_COLOR = {
    'SUBJECT_PROCESS'  : '#E67E22',
    'FILE_OBJECT_BLOCK': '#2980B9',
    'NetFlowObject'    : '#27AE60',
    'SUBJECT_UNIT'     : '#8E44AD',
}
DEFAULT_COLOR = '#7F8C8D'

EDGE_COLOR = {
    'EVENT_OPEN'   : '#5DADE2',
    'EVENT_READ'   : '#58D68D',
    'EVENT_WRITE'  : '#EC7063',
    'EVENT_MMAP'   : '#BB8FCE',
    'EVENT_EXECUTE': '#F0B27A',
    'EVENT_CLONE'  : '#F7DC6F',
    'EVENT_UNLINK' : '#E74C3C',
}
DEFAULT_EDGE = '#888888'


def load_full_graph(tag=None):
    suffix = '_{}'.format(tag) if tag else ''
    path = os.path.join(OUTPUT_GRAPHS, 'graph_output{}.json'.format(suffix))
    if not os.path.exists(path):
        sys.exit(f'ERROR: {path} not found — run main.py first.')
    with open(path) as f:
        data = json.load(f)
    nodes = {n['id']: n for n in data['nodes']}
    return nodes, data['links']


def load_subgraphs(tag=None):
    if tag:
        path = os.path.join(OUTPUT_GRAPHS, f'subgraphs_{tag}.json')
        if not os.path.exists(path):
            sys.exit(f'ERROR: {path} not found.')
    else:
        files = _glob.glob(os.path.join(OUTPUT_GRAPHS, 'subgraphs_*.json'))
        if not files:
            sys.exit('ERROR: No subgraphs_*.json found — run main.py first.')
        path = sorted(files)[-1]
    print(f'Subgraph file: {os.path.basename(path)}')
    with open(path) as f:
        return json.load(f)


def load_removed_nodes():
    files = _glob.glob(os.path.join(OUTPUT_GRAPHS, '*_removed_nodes.json'))
    if not files:
        return {}
    with open(sorted(files)[-1]) as f:
        data = json.load(f)
    index = {}
    for rm in data:
        index.setdefault(rm['kept_node'], []).append(rm)
        for p in rm.get('processes', []):
            index.setdefault(p['node'], []).append(rm)
    return index


def find_node_by_uuid(nodes, uuid):
    u = uuid.upper()
    for nid, nd in nodes.items():
        if nd.get('uuid', '').upper() == u:
            return nid, nd
    return None, None


def get_neighbours(target_id, nodes, links):
    incoming = [(l, nodes.get(l['source'], {})) for l in links if l['target'] == target_id]
    outgoing = [(l, nodes.get(l['target'], {})) for l in links if l['source'] == target_id]
    return incoming, outgoing


def get_direct_neighbourhood(target_id, nodes, links):
    direct_links = [l for l in links
                    if l['source'] == target_id or l['target'] == target_id]
    neighbour_ids = {l['source'] for l in direct_links} | {l['target'] for l in direct_links}
    view_nodes = {nid: nodes[nid] for nid in neighbour_ids if nid in nodes}
    return view_nodes, direct_links


def print_edges(target_id, nodes, links, label):
    nd = nodes.get(target_id, {})
    incoming, outgoing = get_neighbours(target_id, nodes, links)

    print(f'\n{"═"*70}\n  {label}\n{"═"*70}')
    print(f'\n  TARGET  node={target_id}  uuid={nd.get("uuid","")}')
    print(f'          type={nd.get("type","")}  name={nd.get("name","")}')

    print(f'\n  INCOMING ({len(incoming)})')
    print(f'  {"─"*66}')
    for l, src in incoming:
        print(f'  [{l.get("edge_type","?")}]  ts={l.get("ts","")}')
        print(f'    node={l["source"]}  uuid={src.get("uuid","")}  name={src.get("name","")}')

    print(f'\n  OUTGOING ({len(outgoing)})')
    print(f'  {"─"*66}')
    for l, dst in outgoing:
        print(f'  [{l.get("edge_type","?")}]  ts={l.get("ts","")}')
        print(f'    node={l["target"]}  uuid={dst.get("uuid","")}  name={dst.get("name","")}')
    print()


def save_png(target_id, nodes, links, out_path, title, removed_by_kept=None):
    G = nx.MultiDiGraph()
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs, _removed=False)
    for l in links:
        G.add_edge(l['source'], l['target'],
                   edge_type=l.get('edge_type', ''),
                   ts=l.get('ts', ''), _removed=False)

    removed_by_kept = removed_by_kept or {}
    removed_overlays = removed_by_kept.get(target_id, [])
    for rm in removed_overlays:
        rm_id = rm['node']
        if not G.has_node(rm_id):
            G.add_node(rm_id, name=rm['name'], uuid=rm.get('uuid', ''),
                       type=rm.get('type', 'FILE_OBJECT_BLOCK'), _removed=True)
        for proc in rm.get('processes', []):
            pid = proc['node']
            if G.has_node(pid):
                G.add_edge(pid, rm_id, edge_type='REMOVED', _removed=True)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='twopi', args='-Granksep=3')
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=3.0)

    all_edges          = list(G.edges(data=True, keys=True))
    normal_edges       = [(s, d, k, dt) for s, d, k, dt in all_edges if not dt.get('_removed')]
    removed_edges      = [(s, d, k, dt) for s, d, k, dt in all_edges if dt.get('_removed')]
    normal_nodes       = [n for n in G.nodes() if not G.nodes[n].get('_removed')]
    removed_nodes_list = [n for n in G.nodes() if G.nodes[n].get('_removed')]

    def nc(n):
        if n == target_id:             return '#FFD700'
        if G.nodes[n].get('_removed'): return '#C0392B'
        return TYPE_COLOR.get(G.nodes[n].get('type', ''), DEFAULT_COLOR)

    def ns(n):
        return 2200 if n == target_id else 900

    fig, ax = plt.subplots(figsize=(28, 22))
    ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_title(title, color='white', fontsize=12, pad=14)

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=normal_nodes,
                           node_color=[nc(n) for n in normal_nodes],
                           node_size=[ns(n) for n in normal_nodes],
                           edgecolors='#333355', linewidths=1.5, alpha=0.95)
    if removed_nodes_list:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=removed_nodes_list,
                               node_color='#C0392B', node_size=900,
                               edgecolors='#FF4444', linewidths=3.0, alpha=0.85)

    if normal_edges:
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=[(s, d, k) for s, d, k, _ in normal_edges],
                               edge_color=[EDGE_COLOR.get(dt.get('edge_type', ''),
                                           DEFAULT_EDGE) for _, _, _, dt in normal_edges],
                               arrows=True, arrowsize=20,
                               connectionstyle='arc3,rad=0.12',
                               width=1.8, alpha=0.8,
                               min_source_margin=22, min_target_margin=22)
    if removed_edges:
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=[(s, d, k) for s, d, k, _ in removed_edges],
                               edge_color='#FF4444', arrows=True, arrowsize=18,
                               connectionstyle='arc3,rad=0.25',
                               width=2.0, alpha=0.85, style='dashed',
                               min_source_margin=22, min_target_margin=22)

    labels = {}
    for n, d in G.nodes(data=True):
        name  = d.get('name', str(n))
        short = name.rstrip('/').split('/')[-1] or name
        if len(short) > 16 and short.replace('-', '').isalnum():
            short = short[:8] + '…'
        else:
            short = short[:20]
        uuid_short = d.get('uuid', '')[:8]
        if n == target_id:
            labels[n] = f'★ {short}\n{uuid_short}'
        elif d.get('_removed'):
            labels[n] = f'[REMOVED]\n{short}\n{uuid_short}'
        else:
            labels[n] = f'{short}\n{uuid_short}'

    nx.draw_networkx_labels(G, pos,
                            labels={n: l for n, l in labels.items() if not G.nodes[n].get('_removed')},
                            ax=ax, font_size=8, font_color='white', font_weight='bold')
    if removed_nodes_list:
        nx.draw_networkx_labels(G, pos,
                                labels={n: labels[n] for n in removed_nodes_list},
                                ax=ax, font_size=7, font_color='#FF9999', font_weight='bold')

    pair_types = {}
    for s, d, k, data in all_edges:
        et = data.get('edge_type', '').replace('EVENT_', '')
        if et and et not in pair_types.get((s, d), []):
            pair_types.setdefault((s, d), []).append(et)

    edge_labels = {(s, d): ', '.join(types) for (s, d), types in pair_types.items()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=7, font_color='#FFD700',
        label_pos=0.5,
        bbox=dict(boxstyle='round,pad=0.1', facecolor='#0d0d1a',
                  alpha=0.7, edgecolor='none'))

    legend = [
        mpatches.Patch(color='#FFD700', label='★ Target node'),
        mpatches.Patch(color='#E67E22', label='Process'),
        mpatches.Patch(color='#2980B9', label='File'),
        mpatches.Patch(color='#27AE60', label='NetFlow'),
        mpatches.Patch(color='#5DADE2', label='OPEN'),
        mpatches.Patch(color='#58D68D', label='READ'),
        mpatches.Patch(color='#EC7063', label='WRITE'),
        mpatches.Patch(color='#E74C3C', label='UNLINK'),
        mpatches.Patch(color='#F7DC6F', label='CLONE'),
    ]
    ax.legend(handles=legend, loc='upper left', facecolor='#1a1a2e',
              labelcolor='white', fontsize=8, framealpha=0.9)

    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  PNG saved → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uuid', required=True, help='UUID of the node')
    parser.add_argument('--sub',  action='store_true',
                        help='use subgraph instead of full graph')
    parser.add_argument('--tag',  default=None,
                        help='file tag e.g. ta1-theia-e3-official-6r0')
    args = parser.parse_args()

    uuid       = args.uuid.strip()
    short_uuid = uuid[:8]
    tag        = args.tag
    removed_by_kept = load_removed_nodes()

    if args.sub:
        subs = load_subgraphs(tag=tag)
        found = False
        for i, sg in enumerate(subs):
            sg_nodes = {n[0]: n[1] for n in sg['nodes']}
            sg_links = [{'source': e[0], 'target': e[1],
                         'edge_type': e[3].get('edge_type', ''),
                         'ts':        e[3].get('ts', '')}
                        for e in sg['edges']]
            sid, snd = find_node_by_uuid(sg_nodes, uuid)
            if sid is None:
                continue
            found = True
            seed = sg.get('seed_name', f'subgraph#{i}')
            view_nodes, view_links = get_direct_neighbourhood(sid, sg_nodes, sg_links)
            print_edges(sid, sg_nodes, sg_links, f'SUBGRAPH #{i}  seed={seed}')
            print(f'  direct neighbourhood: {len(view_nodes)} nodes, {len(view_links)} edges')
            out   = os.path.join(OUTPUT_GRAPHS, f'node_{short_uuid}_sub{i}.png')
            title = (f'Subgraph #{i}  |  direct neighbours  (red = R2 removed parent)\n'
                     f'uuid={uuid}   name={snd.get("name","")}')
            save_png(sid, view_nodes, view_links, out, title, removed_by_kept=removed_by_kept)
        if not found:
            print(f'UUID {uuid} NOT FOUND in any subgraph.')

    else:
        nodes, links = load_full_graph(tag=tag)
        print(f'Full graph: {len(nodes)} nodes, {len(links)} edges')
        nid, nd = find_node_by_uuid(nodes, uuid)
        if nid is None:
            print(f'UUID {uuid} NOT FOUND in full graph.')
            return
        view_nodes, view_links = get_direct_neighbourhood(nid, nodes, links)
        print(f'  direct neighbourhood: {len(view_nodes)} nodes, {len(view_links)} edges')
        print_edges(nid, nodes, links, 'FULL GRAPH')
        out   = os.path.join(OUTPUT_GRAPHS, f'node_{short_uuid}_full.png')
        title = (f'Full Graph  |  direct neighbours\n'
                 f'uuid={uuid}   name={nd.get("name","")}')
        save_png(nid, view_nodes, view_links, out, title)


if __name__ == '__main__':
    main()
