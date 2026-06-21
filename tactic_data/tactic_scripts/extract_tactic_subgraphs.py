"""
Full pipeline:
  1. Load malicious UUIDs from CSV
  2. Read raw CDM18 JSON — keep events where src OR dst is malicious
  3. Build NetworkX MultiDiGraph
  4. Apply reduce_graph (netflow merge + directory cascade + similar files)
  5. Split into tactic time windows
  6. For each window: keep edges where at least one endpoint is malicious
  7. Save JSON + PNG for each tactic

Usage:
    cd /csse/research/contructive-learning
    source .venv/bin/activate
    python tactic_scripts/extract_tactic_subgraphs.py
"""

import os, sys, json
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scripts.reduce_graph import reduce_netflow_edges, reduce_directory_cascade, reduce_similar_files
import scripts.reduce_graph as rg

ATTACK_NAME = 'Firefox_Backdoor_Drakon_In_Memory'
RAW_LOG  = os.path.join(PROJECT_ROOT, 'input', 'theia', 'ta1-theia-e3-official-6r0.json')
MAL_CSV  = os.path.join(PROJECT_ROOT, 'input', 'malicious_nodes', f'node_{ATTACK_NAME}.csv')
OUT_DIR  = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data',
                         'extracted_windows', 'firefox_backdoor')
os.makedirs(OUT_DIR, exist_ok=True)

SKIP_EVENTS = {'EVENT_MPROTECT', 'EVENT_MMAP', 'EVENT_SHM'}

def to_ns(h, m, s):
    dt = datetime(2018, 4, 10, h, m, s, tzinfo=timezone(timedelta(hours=-4)))
    return int(dt.timestamp() * 1e9)

def ns_to_et(ns):
    try:
        utc = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
        return (utc - timedelta(hours=4)).strftime('%H:%M:%S EDT')
    except Exception:
        return str(ns)

TACTICS = [
    {'name': 'Initial_Access',        'technique': 'T1189',
     'start': to_ns(14, 30, 31),      'end': to_ns(14, 35, 16)},
    {'name': 'Execution',             'technique': 'T1203',
     'start': to_ns(14, 35, 17),      'end': to_ns(14, 44, 27)},
    {'name': 'Execution_ReExploit',   'technique': 'T1203',
     'start': to_ns(14, 54, 16),      'end': to_ns(14, 56, 38)},
    {'name': 'Persistence',           'technique': 'T1546',
     'start': to_ns(14, 56, 39),      'end': to_ns(14, 58, 54)},
]

# ── 1. Load malicious UUIDs ───────────────────────────────────────────────────
print('Loading malicious UUIDs...')
mal_uuids = set()
with open(MAL_CSV) as f:
    for line in f:
        u = line.strip().split(',')[0].strip()
        if u: mal_uuids.add(u)
print(f'  {len(mal_uuids)} malicious UUIDs')

# ── 2. Pass 1: collect node metadata ─────────────────────────────────────────
print('\nPass 1 — node metadata...')
nodes_meta = {}
principals = {}

with open(RAW_LOG, errors='replace') as f:
    for line in f:
        if 'Subject' not in line and 'FileObject' not in line \
           and 'NetFlowObject' not in line and 'Principal' not in line:
            continue
        try:
            obj  = json.loads(line.strip())
            data = obj['datum']
        except Exception:
            continue

        if 'com.bbn.tc.schema.avro.cdm18.Principal' in data:
            p = data['com.bbn.tc.schema.avro.cdm18.Principal']
            principals[p['uuid']] = p.get('userId', '?')

        elif 'com.bbn.tc.schema.avro.cdm18.Subject' in data:
            s = data['com.bbn.tc.schema.avro.cdm18.Subject']
            cmd = s.get('cmdLine', '')
            if isinstance(cmd, dict): cmd = cmd.get('string', '')
            props = s.get('properties', {}).get('map', {})
            nodes_meta[s['uuid']] = {
                'uuid': s['uuid'], 'type': 'SUBJECT_PROCESS',
                'name': cmd or props.get('path', ''),
                'localPrincipal': s.get('localPrincipal', ''),
                'ts': s.get('startTimestampNanos', 0),
            }

        elif 'com.bbn.tc.schema.avro.cdm18.FileObject' in data:
            fo = data['com.bbn.tc.schema.avro.cdm18.FileObject']
            base = fo.get('baseObject', {})
            props = base.get('properties', {}).get('map', {})
            name = (base.get('filename') or base.get('path')
                    or props.get('filename') or props.get('path') or '')
            nodes_meta[fo['uuid']] = {
                'uuid': fo['uuid'], 'type': 'FILE_OBJECT_BLOCK', 'name': name, 'ts': 0,
            }

        elif 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in data:
            nf = data['com.bbn.tc.schema.avro.cdm18.NetFlowObject']
            def _s(v): return v.get('string', str(v)) if isinstance(v, dict) else str(v or 'null')
            def _p(v): return str(v.get('int', v))    if isinstance(v, dict) else str(v or 'null')
            name = f"{_s(nf.get('localAddress','null'))}_{_p(nf.get('localPort','null'))}_{_s(nf.get('remoteAddress','null'))}_{_p(nf.get('remotePort','null'))}"
            nodes_meta[nf['uuid']] = {
                'uuid': nf['uuid'], 'type': 'NetFlowObject', 'name': name, 'ts': 0,
            }

for uid, nd in nodes_meta.items():
    if nd['type'] == 'SUBJECT_PROCESS':
        nd['userId'] = principals.get(nd.get('localPrincipal', ''), '?')

print(f'  {len(nodes_meta):,} nodes collected')

# ── 3. Pass 2: extract edges where src OR dst is malicious ────────────────────
print('\nPass 2 — extracting edges (src OR dst malicious)...')
raw_edges = {}   # (src, dst, etype) → ts

with open(RAW_LOG, errors='replace') as f:
    for line in f:
        if 'Event' not in line:
            continue
        try:
            obj  = json.loads(line.strip())
            data = obj['datum']
        except Exception:
            continue
        if 'com.bbn.tc.schema.avro.cdm18.Event' not in data:
            continue

        ev    = data['com.bbn.tc.schema.avro.cdm18.Event']
        etype = ev.get('type', '')
        if etype in SKIP_EVENTS:
            continue

        ts  = ev.get('timestampNanos', 0)
        src = (ev.get('subject',         {}) or {}).get('com.bbn.tc.schema.avro.cdm18.UUID', '')
        dst = (ev.get('predicateObject', {}) or {}).get('com.bbn.tc.schema.avro.cdm18.UUID', '')

        if not src or not dst or dst == '00000000-0000-0000-0000-000000000000':
            continue
        if src not in mal_uuids and dst not in mal_uuids:
            continue

        key = (src, dst, etype)
        if key not in raw_edges or ts < raw_edges[key]:
            raw_edges[key] = ts

print(f'  {len(raw_edges):,} unique edges extracted')

# ── 4. Build NetworkX MultiDiGraph ────────────────────────────────────────────
print('\nBuilding graph...')
G = nx.MultiDiGraph()

node_uuids = set()
for (src, dst, etype) in raw_edges:
    node_uuids.add(src); node_uuids.add(dst)

for uid in node_uuids:
    nd = nodes_meta.get(uid, {'uuid': uid, 'type': 'UNKNOWN', 'name': '', 'ts': 0})
    G.add_node(uid, uuid=uid, type=nd['type'], name=nd['name'],
               ts=nd.get('ts', 0), is_malicious=(uid in mal_uuids),
               userId=nd.get('userId', '?'))

for (src, dst, etype), ts in raw_edges.items():
    G.add_edge(src, dst, edge_type=etype, ts=ts,
               src_uuid=src, dst_uuid=dst,
               src_name=G.nodes[src]['name'], dst_name=G.nodes[dst]['name'])

print(f'  Graph: {G.number_of_nodes():,} nodes  {G.number_of_edges():,} edges')

# ── 5. Apply reduce_graph ─────────────────────────────────────────────────────
print('\nApplying graph reduction...')
n0, e0 = G.number_of_nodes(), G.number_of_edges()

rg.NETFLOW_WINDOW_NS = 300_000_000_000   # 5-minute merge window
G, _, _ = reduce_directory_cascade(G)
n1, e1 = G.number_of_nodes(), G.number_of_edges()
print(f'  Directory cascade : {n0}→{n1} nodes  {e0}→{e1} edges')

G, _, _, _, _, _ = reduce_netflow_edges(G)
n2, e2 = G.number_of_nodes(), G.number_of_edges()
print(f'  Netflow merge     : {n1}→{n2} nodes  {e1}→{e2} edges')

G, _, _, _, _, _ = reduce_similar_files(G)
n3, e3 = G.number_of_nodes(), G.number_of_edges()
print(f'  Similar files     : {n2}→{n3} nodes  {e2}→{e3} edges')

print(f'\n  Final: {n3:,} nodes  {e3:,} edges  (from {n0:,} nodes  {e0:,} edges)')

# ── 6. Extract tactic windows + visualize ─────────────────────────────────────
TYPE_COLOR = {
    'SUBJECT_PROCESS':   '#4A90D9',
    'FILE_OBJECT_BLOCK': '#F39C12',
    'NetFlowObject':     '#8E44AD',
    'UNKNOWN':           '#7F8C8D',
}
EDGE_COLOR = {
    'EVENT_CLONE':    '#E74C3C',
    'EVENT_EXECUTE':  '#C0392B',
    'EVENT_WRITE':    '#27AE60',
    'EVENT_READ':     '#2ECC71',
    'EVENT_CONNECT':  '#9B59B6',
    'EVENT_SENDTO':   '#8E44AD',
    'EVENT_RECVFROM': '#8E44AD',
    'EVENT_SENDMSG':  '#8E44AD',
    'EVENT_RECVMSG':  '#8E44AD',
    'EVENT_READ_SOCKET_PARAMS': '#8E44AD',
}

def short_label(name, ntype):
    if not name: return '?'
    name = name.strip()
    if ntype == 'NetFlowObject':
        parts = name.split('_')
        if len(parts) >= 4:
            return f'{parts[2]}:{parts[3]}'
        return name[:20]
    return name.split('/')[-1][:22] or name[:22]

print()
for t in TACTICS:
    tname  = t['name']
    tstart = t['start']
    tend   = t['end']

    window_edges = [
        (u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)
        if tstart <= d.get('ts', 0) <= tend
    ]

    if not window_edges:
        print(f'  {tname}: no edges in window')
        continue

    mal_edges = [
        (u, v, k, d) for u, v, k, d in window_edges
        if G.nodes[u].get('is_malicious') or G.nodes[v].get('is_malicious')
    ]

    viz_edges = [
        (u, v, k, d) for u, v, k, d in mal_edges
        if G.nodes[u].get('is_malicious') and G.nodes[v].get('is_malicious')
    ]

    if not mal_edges:
        print(f'  {tname}: no malicious edges in window')
        continue

    Gs = nx.MultiDiGraph()
    node_set = set()
    for u, v, k, d in mal_edges:
        node_set.add(u); node_set.add(v)
    for uid in node_set:
        Gs.add_node(uid, **G.nodes[uid])
    for u, v, k, d in mal_edges:
        Gs.add_edge(u, v, **d)

    Gv = nx.MultiDiGraph()
    viz_node_set = set()
    for u, v, k, d in viz_edges:
        viz_node_set.add(u); viz_node_set.add(v)
    for uid in viz_node_set:
        Gv.add_node(uid, **G.nodes[uid])
    for u, v, k, d in viz_edges:
        Gv.add_edge(u, v, **d)

    print(f'  {tname} ({t["technique"]}): json={Gs.number_of_nodes()}n/{Gs.number_of_edges()}e  viz={Gv.number_of_nodes()}n/{Gv.number_of_edges()}e  [{ns_to_et(tstart)} → {ns_to_et(tend)}]')

    nodes_out = [dict(Gs.nodes[n]) for n in Gs.nodes]
    edges_out = []
    for u, v, k, d in Gs.edges(keys=True, data=True):
        edges_out.append({
            'src': u, 'src_name': Gs.nodes[u]['name'],
            'dst': v, 'dst_name': Gs.nodes[v]['name'],
            'etype': d.get('edge_type', ''), 'ts': d.get('ts', 0),
            'time_edt': ns_to_et(d.get('ts', 0)),
        })
    edges_out.sort(key=lambda x: x['ts'])

    json_path = os.path.join(OUT_DIR, f'tactic_{tname}_{ATTACK_NAME}.json')
    with open(json_path, 'w') as f:
        json.dump({'tactic': tname, 'technique': t['technique'],
                   'attack': ATTACK_NAME, 'n_nodes': Gs.number_of_nodes(),
                   'n_edges': Gs.number_of_edges(), 'nodes': nodes_out,
                   'edges': edges_out}, f, indent=2)

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    try:
        pos = nx.nx_agraph.graphviz_layout(Gv, prog='dot',
              args='-Grankdir=TB -Gnodesep=1.0 -Granksep=1.8')
    except Exception:
        pos = nx.kamada_kawai_layout(Gv)

    node_colors = [TYPE_COLOR.get(Gv.nodes[n]['type'], '#7F8C8D') for n in Gv.nodes]
    node_sizes  = [3200 if Gv.nodes[n]['type']=='SUBJECT_PROCESS' else 2400 for n in Gv.nodes]

    nx.draw_networkx_nodes(Gv, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        alpha=0.97, edgecolors='white', linewidths=2.5)
    nx.draw_networkx_labels(Gv, pos, ax=ax,
        labels={n: short_label(Gv.nodes[n]['name'], Gv.nodes[n]['type']) for n in Gv.nodes},
        font_size=8, font_color='white', font_weight='bold')

    for u, v, data in Gv.edges(data=True):
        etype = data.get('edge_type', '')
        color = EDGE_COLOR.get(etype, '#AAAAAA')
        nx.draw_networkx_edges(Gv, pos, ax=ax, edgelist=[(u, v)],
            edge_color=color, arrows=True, arrowsize=20,
            width=2.0, connectionstyle='arc3,rad=0.12', alpha=0.9)

    elabels = {}
    for u, v, data in Gv.edges(data=True):
        k = (u, v)
        lbl = data.get('edge_type', '').replace('EVENT_', '')
        elabels[k] = elabels.get(k, '') + ('\n' if k in elabels else '') + lbl
    if elabels:
        nx.draw_networkx_edge_labels(Gv, pos, ax=ax, edge_labels=elabels,
            font_size=6.5, font_color='#FFD700',
            bbox=dict(boxstyle='round,pad=0.2', fc='#0d0d1a', alpha=0.75))

    legend = [
        mpatches.Patch(color='#4A90D9', label='Process'),
        mpatches.Patch(color='#F39C12', label='File'),
        mpatches.Patch(color='#8E44AD', label='Network'),
        mpatches.Patch(color='#E74C3C', label='CLONE/EXECUTE'),
        mpatches.Patch(color='#27AE60', label='WRITE/READ'),
        mpatches.Patch(color='#9B59B6', label='CONNECT/SEND/RECV'),
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=9,
        facecolor='#1a1a2e', labelcolor='white', framealpha=0.9)
    ax.set_title(
        f'{tname}  ({t["technique"]})  —  {ATTACK_NAME}\n'
        f'{ns_to_et(tstart)} → {ns_to_et(tend)}  |  '
        f'viz: {Gv.number_of_nodes()} nodes  {Gv.number_of_edges()} edges  '
        f'(full json: {Gs.number_of_nodes()}n/{Gs.number_of_edges()}e)',
        color='white', fontsize=13, fontweight='bold', pad=14)
    ax.axis('off')
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, f'tactic_{tname}_{ATTACK_NAME}.png')
    plt.savefig(png_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'    saved → {png_path}')

print('\nDone.')
