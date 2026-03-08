import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import defaultdict

WINDOW_JSON = "output/theia/windows/window_1_subgraph.json"
OUT_PNG     = "window_1_graph.png"


with open(WINDOW_JSON, 'r') as f:
    raw = json.load(f)

key  = list(raw.keys())[0]
data = raw[key]

nodes = data['subgraph_nodes']
edges = data['subgraph_edges']

print(f"Window  : {key}")
print(f"Period  : {data['start_et']}  →  {data['end_et']}")
print(f"Nodes   : {len(nodes)}")
print(f"Edges   : {len(edges)}")
print()


TYPE_COLOR = {
    'SUBJECT_PROCESS'   : '#4C9BE8', 
    'FILE_OBJECT_BLOCK' : '#5DBB63',  
    'FILE_OBJECT_CHAR'  : '#5DBB63',
    'FILE_OBJECT'       : '#5DBB63',
    'NetFlowObject'     : '#F4A261',  
    'MemoryObject'      : '#9B59B6',  
    'UnnamedPipeObject' : '#E74C3C',  
    'PRINCIPAL_LOCAL'   : '#F1C40F',  
}
DEFAULT_COLOR = '#AAAAAA'


def short_label(nd):
    name = nd.get('name', '')
    if name:
        base = os.path.basename(name.rstrip('/'))
        return base[:20] if base else name[:20]
    ip = nd.get('remote_ip', '')
    if ip:
        return ip
    return nd['uuid'][:8]


G = nx.DiGraph()

uuid_to_node = {nd['uuid']: nd for nd in nodes}

for nd in nodes:
    G.add_node(nd['uuid'],
               ntype=nd.get('node_type', 'UNKNOWN'),
               label=short_label(nd),
               malicious=nd.get('in_theia_txt', 0))

for e in edges:
    G.add_edge(e['src'], e['dst'], etype=e.get('edge_type', ''))


pos = nx.spring_layout(G, seed=42, k=2.5)


fig, ax = plt.subplots(figsize=(18, 13))
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#F8F9FA')

node_colors  = [TYPE_COLOR.get(G.nodes[n]['ntype'], DEFAULT_COLOR) for n in G.nodes()]
node_sizes   = [2800 if G.nodes[n]['malicious'] else 2000 for n in G.nodes()]
node_borders = ['#E74C3C' if G.nodes[n]['malicious'] else '#555555' for n in G.nodes()]


nx.draw_networkx_nodes(G, pos, ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    linewidths=2.5,
)

mal_nodes = [n for n in G.nodes() if G.nodes[n]['malicious']]
if mal_nodes:
    nx.draw_networkx_nodes(G, pos, nodelist=mal_nodes, ax=ax,
        node_color='none',
        node_size=2900,
        linewidths=3,
    )


nx.draw_networkx_edges(G, pos, ax=ax,
    edge_color='#555555',
    width=1.2,
    alpha=0.7,
    arrows=True,
    arrowsize=15,
    connectionstyle='arc3,rad=0.1',
    min_source_margin=25,
    min_target_margin=25,
)

# Node labels
labels = {n: G.nodes[n]['label'] for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7.5, font_weight='bold')


edge_labels = {}
seen = set()
for u, v, d in G.edges(data=True):
    key_e = (u, v)
    if key_e not in seen:
        etype = d.get('etype', '').replace('EVENT_', '')
        edge_labels[key_e] = etype
        seen.add(key_e)

nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
    font_size=6,
    font_color='#333333',
    label_pos=0.4,
    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6),
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color='#4C9BE8', label='Process (SUBJECT_PROCESS)'),
    mpatches.Patch(color='#5DBB63', label='File (FILE_OBJECT)'),
    mpatches.Patch(color='#F4A261', label='Network (NetFlowObject)'),
    mpatches.Patch(color='#9B59B6', label='Memory (MemoryObject)'),
    mpatches.Patch(color='#E74C3C', label='Pipe (UnnamedPipeObject)'),
    mpatches.Patch(color='#AAAAAA', label='Other'),
]
if mal_nodes:
    legend_handles.append(
        mpatches.Patch(facecolor='white', edgecolor='#E74C3C',
                       linewidth=2.5, label='Malicious node (in theia.txt)')
    )
ax.legend(handles=legend_handles, loc='lower left', fontsize=9,
          framealpha=0.9, edgecolor='#CCCCCC')


type_counts = defaultdict(int)
for nd in nodes:
    type_counts[nd.get('node_type', 'UNKNOWN')] += 1
print("Node types:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:<30} {c}")
print()

mal_count = sum(1 for nd in nodes if nd.get('in_theia_txt', 0))
print(f"Malicious nodes (in_theia_txt=1): {mal_count}")
print()


print(f"{'#':<4} {'UUID':<40} {'Type':<22} {'first_seen':<22} {'Mal':<6} Full Name")
print("-" * 140)
for i, nd in enumerate(nodes, 1):
    uuid       = nd['uuid']
    ntype      = nd.get('node_type', 'UNKNOWN')
    name       = nd.get('name', nd.get('remote_ip', ''))   # full name, not truncated
    first_seen = nd.get('first_seen_et', '')
    mal        = '⚠ MAL' if nd.get('in_theia_txt', 0) else ''
    print(f"{i:<4} {uuid:<40} {ntype:<22} {first_seen:<22} {mal:<6} {name}")
print()

ax.set_title(
    f'{key}  |  {data["start_et"]}  →  {data["end_et"]}\n'
    f'{len(nodes)} nodes  |  {len(edges)} edges  |  {mal_count} malicious nodes (red border)',
    fontsize=12, fontweight='bold', pad=15
)
ax.axis('off')
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"Saved → {OUT_PNG}")


print()
print("LABEL → FULL NAME mapping")
print("-" * 80)
uuid_label   = {nd['uuid']: short_label(nd) for nd in nodes}
uuid_fullname = {nd['uuid']: nd.get('name', nd.get('remote_ip', nd['uuid'])) for nd in nodes}
seen_labels  = {}
for nd in nodes:
    label = short_label(nd)
    full  = uuid_fullname[nd['uuid']]
    if label not in seen_labels:
        seen_labels[label] = full
        print(f"  {label:<22}  →  {full}")

print()
print("ADJACENCY (directed: src → dst)")
print("-" * 60)
for e in edges:
    src_name = uuid_label.get(e['src'], e['src'][:8])
    dst_name = uuid_label.get(e['dst'], e['dst'][:8])
    etype    = e.get('edge_type', '').replace('EVENT_', '')
    print(f"  {src_name:<22} ──{etype}──▶  {dst_name}")


import sys
if len(sys.argv) > 1:
    trace_uuid = sys.argv[1]
    uuid_info  = {nd['uuid']: nd for nd in nodes}


    dir_adj = defaultdict(list)
    edge_map = defaultdict(list)
    for e in edges:
        dir_adj[e['src']].append(e['dst'])
        edge_map[(e['src'], e['dst'])].append(e.get('edge_type','').replace('EVENT_',''))

    def dfs_trace(node, depth, visited, max_depth=3, prefix=''):
        nd      = uuid_info.get(node, {})
        label   = uuid_label.get(node, node[:8])
        mal     = '  ⚠ MAL' if nd.get('in_theia_txt', 0) else ''
        ntype   = nd.get('node_type', '')
        print(f"{prefix}[depth={depth}]  {label}  ({ntype}){mal}")
        if depth >= max_depth:
            return
        children = sorted(set(dir_adj.get(node, [])))
        for i, child in enumerate(children):
            if child in visited:
                continue
            visited.add(child)
            etypes = ', '.join(set(edge_map.get((node, child), [])))
            connector = '└──' if i == len(children)-1 else '├──'
            spacer    = '    ' if i == len(children)-1 else '│   '
            print(f"{prefix}{connector}──{etypes}──▶")
            dfs_trace(child, depth+1, visited, max_depth, prefix + spacer)

    print()
    print(f"DFS TRACE from: {trace_uuid}")
    print("=" * 60)
    if trace_uuid not in uuid_info:
        print(f"  UUID not found in this subgraph.")
    else:
        dfs_trace(trace_uuid, 0, {trace_uuid})
