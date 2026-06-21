"""
Visualize the Browser Extension Drakon Dropper attack subgraph.
Uses AND filter — only edges where BOTH src AND dst are malicious.

Usage:
    cd /csse/research/contructive-learning
    source .venv/bin/activate
    python output/theia/tactic_data/browerextension/visualize_window.py
"""

import os, json
from datetime import datetime, timezone, timedelta
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

SRC_JSON = os.path.join(HERE, 'stage3_malicious_edges_12_38_to_13_27.json')
MAL_CSV  = os.path.join(PROJECT_ROOT, 'input', 'malicious_nodes',
                         'node_Browser_Extension_Drakon_Dropper.csv')
OUT_PNG  = os.path.join(HERE, 'full_attack_and.png')

# ── Load malicious UUIDs ──────────────────────────────────────────────────────
mal_info = {}
with open(MAL_CSV) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split(',', 1)
        uuid  = parts[0].strip()
        label = parts[1].strip() if len(parts) > 1 else uuid
        mal_info[uuid] = label
mal_uuids = set(mal_info.keys())

# ── Load edges and apply AND filter ──────────────────────────────────────────
with open(SRC_JSON) as f:
    data = json.load(f)

edges = [e for e in data['edges']
         if e['src'] in mal_uuids and e['dst'] in mal_uuids]

used_nodes = set()
for e in edges:
    used_nodes.add(e['src']); used_nodes.add(e['dst'])

mal_in_graph = {u: mal_info[u] for u in used_nodes if u in mal_uuids}

print(f'\n{"="*55}')
print(f'  Browser Extension Drakon Dropper — Full Attack')
print(f'  {data["window_start"]}  ->  {data["window_end"]}')
print(f'{"="*55}')
print(f'  Edges (AND filter) : {len(edges):,}')
print(f'  Nodes              : {len(used_nodes):,}')
print(f'  Malicious nodes    : {len(mal_in_graph)} / {len(mal_uuids)}')
print(f'\n--- Malicious nodes in graph ---')
for uuid, label in sorted(mal_in_graph.items(), key=lambda x: x[1]):
    print(f'  {uuid}  |  {label}')

if not edges:
    print('\nNo AND-filtered edges found.')
    exit()

# ── Build graph ───────────────────────────────────────────────────────────────
TYPE_COLOUR = {
    'SUBJECT_PROCESS'  : '#4A90D9',
    'FILE_OBJECT_BLOCK': '#F39C12',
    'NetFlowObject'    : '#8E44AD',
}
DEFAULT_COLOUR = '#7F8C8D'

def edge_colour(etype):
    e = etype.upper()
    if any(k in e for k in ('CLONE', 'FORK', 'EXECUTE')): return '#E74C3C'
    if any(k in e for k in ('READ', 'RECV', 'LOAD')):     return '#2980B9'
    if any(k in e for k in ('WRITE', 'SEND')):             return '#27AE60'
    if any(k in e for k in ('CONNECT', 'ACCEPT', 'BIND')): return '#9B59B6'
    return '#95A5A6'

G = nx.MultiDiGraph()
node_meta = {}
for e in edges:
    for uid, ntype, nname in [(e['src'], e['src_type'], e['src_name']),
                               (e['dst'], e['dst_type'], e['dst_name'])]:
        if uid not in node_meta:
            node_meta[uid] = {'type': ntype, 'name': nname}
    G.add_edge(e['src'], e['dst'], etype=e['etype'])

n   = G.number_of_nodes()
pos = nx.spring_layout(G, seed=42, k=3.0) if n <= 20 else \
      nx.spring_layout(G, seed=42, k=2.0, iterations=80)

mal_nodes    = [u for u in G.nodes() if u in mal_uuids]
normal_nodes = [u for u in G.nodes() if u not in mal_uuids]
labels       = {u: (node_meta.get(u, {}).get('name') or u)[-28:] for u in G.nodes()}

fig_w = max(18, min(n * 0.6, 44))
fig_h = max(14, min(n * 0.45, 32))
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')

if normal_nodes:
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=normal_nodes,
                           node_color=[TYPE_COLOUR.get(node_meta.get(u, {}).get('type', ''), DEFAULT_COLOUR) for u in normal_nodes],
                           node_size=400, alpha=0.88)
if mal_nodes:
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=mal_nodes,
                           node_color=[TYPE_COLOUR.get(node_meta.get(u, {}).get('type', ''), DEFAULT_COLOUR) for u in mal_nodes],
                           node_size=900, edgecolors='#E74C3C', linewidths=3.5, alpha=0.95)

nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=7, font_color='white')

e_colours = [edge_colour(d.get('etype', '')) for _, _, d in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=e_colours,
                       arrows=True, arrowsize=15, width=1.3,
                       alpha=0.75, connectionstyle='arc3,rad=0.08')

elabels = {}
for u, v, d in G.edges(data=True):
    elabels[(u, v)] = d.get('etype', '').replace('EVENT_', '')
nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=elabels,
                             font_size=6, font_color='#F0E68C', bbox=dict(alpha=0))

legend = [
    mpatches.Patch(facecolor='#4A90D9', edgecolor='#E74C3C', linewidth=3, label='Malicious (red border)'),
    mpatches.Patch(color='#4A90D9', label='Process'),
    mpatches.Patch(color='#F39C12', label='File'),
    mpatches.Patch(color='#8E44AD', label='Network'),
]
ax.legend(handles=legend, loc='upper left',
          facecolor='#0f0f23', edgecolor='white', labelcolor='white', fontsize=9)
ax.set_title(
    f'Browser Extension Drakon Dropper — Full Attack\n'
    f'{data["window_start"]} — {data["window_end"]}  |  '
    f'nodes={n}  edges={G.number_of_edges()}  malicious={len(mal_nodes)}',
    color='white', fontsize=12, pad=10)
ax.axis('off')
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f'\nSaved -> {OUT_PNG}')
