
import os, sys, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from scripts.config import OUTPUT_GRAPHS, OUTPUT_VIZ, OUTPUT_ATTACK

ATTACK_NAMES = {
    'browser_extension': 'Browser_Extension_Drakon_Dropper',
    'firefox_backdoor' : 'Firefox_Backdoor_Drakon_In_Memory',
    'phishing'         : 'Phishing_Email_Credential_Harvest',
}
DEFAULT_ATTACK = 'browser_extension'
VIZ_DIR        = OUTPUT_VIZ.rstrip(os.sep)
MAX_FULL     = 999999

TYPE_COLOUR = {
    'SUBJECT_PROCESS'  : '#4A90D9',
    'FILE_OBJECT_BLOCK': '#27AE60',
    'NetFlowObject'    : '#E67E22',
    'MemoryObject'     : '#9B59B6',
    'UNKNOWN'          : '#95A5A6',
}
MALICIOUS_COLOUR = '#E74C3C'
SEED_COLOUR      = '#F1C40F'

def edge_colour(edge_type):
    e = edge_type.upper()
    if any(k in e for k in ('CLONE', 'FORK', 'EXECUTE')):  return '#C0392B'
    if any(k in e for k in ('READ', 'RECV', 'LOAD')):      return '#2980B9'
    if any(k in e for k in ('WRITE', 'SEND')):              return '#16A085'
    if any(k in e for k in ('OPEN', 'CLOSE', 'CREATE')):   return '#F39C12'
    if any(k in e for k in ('CONNECT','ACCEPT','BIND')):    return '#8E44AD'
    return '#7F8C8D'

def short_name(name, max_len=24):
    name = str(name or '?')
    return ('...' + name[-(max_len-3):]) if len(name) > max_len else name

def neighbourhood(G, mal_nodes, hops=2):
    keep = set(mal_nodes)
    frontier = set(mal_nodes)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt |= set(G.predecessors(n)) | set(G.successors(n))
        frontier = nxt - keep
        keep |= frontier
    return keep

def limit_netflow(G, int_to_data, keep_n=10):
    netflow_nodes = [n for n in G.nodes()
                     if int_to_data.get(n, {}).get('type') == 'NetFlowObject']
    if len(netflow_nodes) <= keep_n:
        return G
    remove = set(netflow_nodes[keep_n:])
    G = G.copy()
    G.remove_nodes_from(remove)
    return G

def build_png(sg, mal_uuids, out_path, focus_uuid=None, hops=3, max_netflow=None, focus_uuids=None):
    dep_id    = sg['dep_id']
    seed_name = sg.get('seed_name', '?')
    seed_uuid = sg.get('seed_uuid', '')

    uuid_to_int = {nd['uuid']: iid for iid, nd in sg['nodes']}
    int_to_data = {iid: nd   for iid, nd in sg['nodes']}
    mal_ints    = {uuid_to_int[u] for u in mal_uuids if u in uuid_to_int}

    # Support multiple focus UUIDs
    if focus_uuids is None and focus_uuid is not None:
        focus_uuids = [focus_uuid]

    focus_int    = None
    focus_ints   = set()
    title_suffix = ''
    keep_nodes   = None

    if focus_uuids:
        for fu in focus_uuids:
            fi = uuid_to_int.get(fu.upper(), uuid_to_int.get(fu))
            if fi is not None:
                focus_ints.add(fi)
        focus_int = next(iter(focus_ints), None)

        if focus_ints:
            def is_netflow(nid):
                nd = int_to_data.get(nid, {})
                if nd.get('type') == 'NetFlowObject':
                    return True
                name = nd.get('name', '')
                return isinstance(name, str) and name.count(',') >= 2

            adj     = {}
            nf_seen = {}
            cap     = max_netflow if max_netflow is not None else 999999
            for edge in sg['edges']:
                src_uuid, dst_uuid, _, _ = edge
                s = uuid_to_int.get(src_uuid)
                d = uuid_to_int.get(dst_uuid)
                if s is None or d is None:
                    continue
                if is_netflow(d):
                    seen = nf_seen.setdefault(s, set())
                    if d not in seen:
                        if len(seen) >= cap:
                            continue
                        seen.add(d)
                if is_netflow(s):
                    seen = nf_seen.setdefault(d, set())
                    if s not in seen:
                        if len(seen) >= cap:
                            continue
                        seen.add(s)
                adj.setdefault(s, set()).add(d)
                adj.setdefault(d, set()).add(s)

            visited  = set(focus_ints)
            frontier = set(focus_ints)
            for _ in range(hops):
                nxt = set()
                for n in frontier:
                    nxt |= adj.get(n, set())
                frontier = nxt - visited
                visited |= frontier
            keep_nodes   = visited
            title_suffix = f'({hops}-hop neighbourhood around {len(focus_ints)} focus nodes)'

    G = nx.MultiDiGraph()
    for iid, nd in sg['nodes']:
        if keep_nodes is None or iid in keep_nodes:
            G.add_node(iid, **nd)

    edge_counts = {}
    for edge in sg['edges']:
        src_uuid, dst_uuid, _, edata = edge
        etype   = edata.get('edge_type', '?')
        src_int = uuid_to_int.get(src_uuid)
        dst_int = uuid_to_int.get(dst_uuid)
        if src_int is None or dst_int is None:
            continue
        if keep_nodes is not None and (src_int not in keep_nodes or dst_int not in keep_nodes):
            continue
        key = (src_int, dst_int, etype)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    for (src_int, dst_int, etype), count in edge_counts.items():
        G.add_edge(src_int, dst_int, etype=etype, count=count)

    n_total = G.number_of_nodes()

    if focus_uuid is None:
        if n_total > MAX_FULL and mal_ints:
            keep   = neighbourhood(G, mal_ints, hops=2)
            G      = G.subgraph(keep).copy()
            title_suffix = f'(2-hop neighbourhood around {len(mal_ints)} malicious nodes)'

    if max_netflow is not None:
        G = limit_netflow(G, int_to_data, keep_n=max_netflow)

    nodes   = list(G.nodes())
    n_nodes = len(nodes)
    n_edges = G.number_of_edges()

    if n_nodes <= 20:
        pos = nx.spring_layout(G, seed=42, k=3.0)
    elif n_nodes <= 200:
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=60)
    else:
        try:    pos = nx.kamada_kawai_layout(G)
        except: pos = nx.spring_layout(G, seed=42, k=1.5)

    node_colours, node_sizes, node_shapes = [], [], []
    node_labels = {}

    for n in nodes:
        nd    = int_to_data.get(n, {})
        ntype = nd.get('type', 'UNKNOWN')
        name  = nd.get('name', nd.get('uuid', '?'))
        if n in focus_ints:
            colour = SEED_COLOUR
            size   = 900
        elif n in mal_ints:
            colour = MALICIOUS_COLOUR
            size   = 900
        elif nd.get('uuid','') == seed_uuid:
            colour = SEED_COLOUR
            size   = 700
        else:
            colour = TYPE_COLOUR.get(ntype, '#95A5A6')
            size   = 300
        node_colours.append(colour)
        node_sizes.append(size)
        node_labels[n] = short_name(name)

    edge_colours = []
    for u, v, data in G.edges(data=True):
        edge_colours.append(edge_colour(data.get('etype', '')))

    fig_w = max(14, min(n_nodes * 0.35, 38))
    fig_h = max(10, min(n_nodes * 0.28, 28))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colours,
                           node_size=node_sizes,
                           alpha=0.93)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels=node_labels,
                            font_size=7, font_color='white')
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colours,
                           arrows=True, arrowsize=12,
                           width=1.0, alpha=0.65,
                           connectionstyle='arc3,rad=0.08')
    elabels_raw = {}
    for u, v, data in G.edges(data=True):
        count = data.get('count', 1)
        base  = data.get('etype', '').replace('EVENT_', '')
        label = f'{base}({count})' if count > 1 else base
        if (u, v) in elabels_raw:
            if label not in elabels_raw[(u, v)].split('\n'):
                elabels_raw[(u, v)] += '\n' + label
        else:
            elabels_raw[(u, v)] = label
    elabels = elabels_raw
    nx.draw_networkx_edge_labels(G, pos, ax=ax,
                                 edge_labels=elabels,
                                 font_size=6,
                                 font_color='#F0E68C',
                                 bbox=dict(alpha=0))

    legend_items = [
        mpatches.Patch(color=MALICIOUS_COLOUR, label='Malicious'),
        mpatches.Patch(color=SEED_COLOUR,      label='Seed'),
    ] + [
        mpatches.Patch(color=c, label=t) for t, c in TYPE_COLOUR.items()
    ]
    ax.legend(handles=legend_items, loc='upper left',
              facecolor='#0f0f23', edgecolor='white',
              labelcolor='white', fontsize=8)

    title = (f'dep_id={dep_id}  |  seed: {seed_name}  |  '
             f'nodes={n_nodes}/{n_total}  edges={n_edges}  '
             f'malicious={len(mal_ints)}\n{title_suffix}')
    ax.set_title(title, color='white', fontsize=10, pad=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  saved -> {out_path}  ({n_nodes} nodes shown)')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None,
                        help='path to a test subgraph JSON file (overrides --attack)')
    parser.add_argument('--attack', type=str, default=DEFAULT_ATTACK,
                        choices=list(ATTACK_NAMES.keys()),
                        help='which attack to visualize (default: browser_extension)')
    parser.add_argument('--dep',  type=int, default=None,
                        help='dep_id of the single subgraph to visualize')
    parser.add_argument('--uuid', type=str, action='append', default=None,
                        help='node UUID — can be specified multiple times to show union of neighbourhoods')
    parser.add_argument('--hops', type=int, default=3,
                        help='neighbourhood hops when --uuid is used (default 3)')
    parser.add_argument('--max-netflow', type=int, default=None,
                        help='cap number of NetFlow/IP nodes shown (e.g. 10)')
    args = parser.parse_args()

    os.makedirs(VIZ_DIR, exist_ok=True)

    if args.input is not None:
        # Load from a custom input file (e.g. input/test/)
        attack_json = args.input
        labels_json = None
        with open(attack_json) as f:
            data = json.load(f)
        subgraphs = data['subgraphs'] if isinstance(data, dict) else data
        mal_map = {}
    else:
        tag         = ATTACK_NAMES[args.attack]
        attack_json = os.path.join(OUTPUT_ATTACK, f'attack_subgraphs_{tag}.json')
        labels_json = os.path.join(OUTPUT_ATTACK, f'subgraph_labels_{tag}.json')
        with open(attack_json) as f:
            data = json.load(f)
        subgraphs = data['subgraphs'] if isinstance(data, dict) else data
        with open(labels_json) as f:
            labels = json.load(f)
        mal_map = {}
        for r in labels:
            mal_map[(r['dep_id'], r['part_idx'])] = {n['uuid'] for n in r.get('malicious_nodes', [])}

    if args.uuid is not None:
        targets = {u.upper() for u in args.uuid}
        matched = []
        for sg in subgraphs:
            found = False
            for _, nd in sg['nodes']:
                if nd.get('uuid', '').upper() in targets:
                    if not found:
                        matched.append(sg)
                        found = True
                    print(f'  Found UUID {nd["uuid"]} in dep_id={sg["dep_id"]}  '
                          f'node_name={nd.get("name")}  type={nd.get("type")}')
        if not matched:
            print(f'No UUIDs found in any subgraph')
            return
        subgraphs = matched

    elif args.dep is not None:
        subgraphs = [sg for sg in subgraphs if sg['dep_id'] == args.dep]
        if not subgraphs:
            print(f'dep_id={args.dep} not found')
            return

    print(f'Subgraphs to render: {len(subgraphs)}')
    for sg in subgraphs:
        dep_id    = sg['dep_id']
        mal_uuids = mal_map.get((dep_id, sg['part_idx']), set())
        if args.uuid is not None:
            fname = f'subgraph_{dep_id}_focus.png'
        else:
            fname = f'subgraph_{dep_id}.png'
        out = os.path.join(VIZ_DIR, fname)
        print(f'  dep_id={dep_id}  seed={sg["seed_name"]}  '
              f'nodes={sg["n_nodes"]}  malicious={len(mal_uuids)} ...')
        build_png(sg, mal_uuids, out,
                  focus_uuids=args.uuid,
                  hops=args.hops,
                  max_netflow=args.max_netflow)

    print(f'\nDone. PNGs saved to: {VIZ_DIR}')

if __name__ == '__main__':
    main()
