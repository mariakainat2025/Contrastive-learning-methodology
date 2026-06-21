import os, sys, json, pickle
from datetime import datetime, timezone, timedelta
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PKL_PATH     = os.path.join(PROJECT_ROOT, 'cache', 'stage3_graph_all.pkl')
EDT          = timezone(timedelta(hours=-4))

ATTACKS = {

    'browser_extension': {
        'title'    : 'Browser Extension Drakon Dropper',
        'csv'      : 'input/malicious_nodes/node_Browser_Extension_Drakon_Dropper.csv',
        'date'     : (2018, 4, 12),
        'out_dir'  : 'output/theia/tactic_data/browerextension',
        'win_start': (12, 40, 0),
        'win_end'  : (13, 27, 0),
        'tactics': [
            dict(
                name   = 'initial_access',
                label  = 'Initial Access',
                mitre  = 'TA0001',
                filter = 'AND',
                start  = (12, 40, 0),
                end    = (12, 51, 19),
            ),
            dict(
                name   = 'persistence',
                label  = 'Persistence',
                mitre  = 'TA0003',
                filter = 'AND',
                start  = (12, 51, 19),
                end    = (13, 16, 48),
            ),
            dict(
                name   = 'discovery',
                label  = 'Discovery',
                mitre  = 'TA0007',
                filter = 'OR',
                start  = (13, 16, 49),
                end    = (13, 27, 0),
            ),
        ],
    },

    'backdoor': {
        'title'    : 'Firefox Backdoor Drakon In-Memory',
        'csv'      : 'input/malicious_nodes/node_Firefox_Backdoor_Drakon_In_Memory.csv',
        'date'     : (2018, 4, 10),
        'out_dir'  : 'output/theia/tactic_data/backdoor',
        'win_start': (14, 25, 0),
        'win_end'  : (15,  0, 0),
        'tactics': [
            dict(
                name   = 'initial_access',
                label  = 'Initial Access',
                mitre  = 'TA0001',
                filter = 'AND',
                start  = (14, 31, 22),
                end    = (14, 35, 17),
            ),
            dict(
                name        = 'privilege_escalation',
                label       = 'Privilege Escalation',
                mitre       = 'TA0004',
                filter      = 'OR',
                start       = (14, 35, 17),
                end         = (14, 50,  0),
                include_src = ['/home/admin/clean', '/usr/bin/firefox'],
            ),
            dict(
                name        = 'command_and_control',
                label       = 'Command & Control — root shell operator channel',
                mitre       = 'TA0011',
                filter      = 'AND',
                start       = (14, 35, 30),
                end         = (14, 51,  0),
                include_src = ['/home/admin/clean'],
            ),
            dict(
                name   = 'initial_access_2',
                label  = 'Initial Access — re-exploit gatech.edu',
                mitre  = 'TA0001',
                filter = 'AND',
                start  = (14, 51,  0),
                end    = (14, 56, 32),
            ),
            dict(
                name   = 'persistence',
                label  = 'Persistence — putfile profile + execute + open OC2',
                mitre  = 'TA0003',
                filter = 'AND',
                start  = (14, 56, 32),
                end    = (14, 58, 55),
            ),
        ],
    },

    'phishing_attachment': {
        'title'    : 'Phishing Email With Executable Attachment',
        'csv'      : 'input/malicious_nodes/node_Phishing_Email_With_Executable_Attachment.csv',
        'date'     : (2018, 4, 13),
        'out_dir'  : 'output/theia/tactic_data/phishing_attachment',
        'win_start': (14, 4, 0),
        'win_end'  : (14, 7, 0),
        'tactics': [
            dict(
                name   = 'initial_access',
                label  = 'Initial Access — phishing email with executable attachment',
                mitre  = 'TA0001',
                filter = 'AND',
                start  = (14, 4, 45),
                end    = (14, 5, 55),
            ),
            dict(
                name   = 'execution',
                label  = 'Execution — user runs malicious attachment (fails: missing QT)',
                mitre  = 'TA0002',
                filter = 'AND',
                start  = (14, 5, 56),
                end    = (14, 6, 30),
            ),
        ],
    },

    # 'phishing': {
    #     'title'    : 'Phishing Email With Link',
    #     'csv'      : 'input/malicious_nodes/node_Phishing_Email_With_Link.csv',
    #     'date'     : (2018, 4, 10),
    #     'out_dir'  : 'output/theia/tactic_data/phishing',
    #     'win_start': (13, 30, 0),
    #     'win_end'  : (14, 10, 0),
    #     'tactics': [
    #         dict(
    #             name    = 'initial_access',
    #             label   = 'Initial Access — phishing email link clicked',
    #             mitre   = 'TA0001 / T1566.002',
    #             filter  = 'AND',
    #             start   = (13, 42, 29),
    #             end     = (13, 42, 55),
    #             exclude = [
    #                 '/home/admin/.thunderbird/pu7u3e84.default/ImapMail/128.55.12.73/INBOX-1.msf',
    #                 '128.55.12.110_36196_128.55.12.73_143',
    #                 '128.55.12.110_36197_128.55.12.73_143',
    #                 '128.55.12.110_36198_128.55.12.73_143',
    #                 '/dev/glx_alsa_675',
    #                 '/home/admin/Downloads/firefox/firefox',
    #                 '/home/admin/.thunderbird/pu7u3e84.default/logins.json',
    #                 '/home/admin/.mozilla/firefox/pe11scpa.default/logins.json',
    #             ],
    #         ),
    #         dict(
    #             name    = 'credential_access',
    #             label   = 'Credential Access — credentials harvested via fake form',
    #             mitre   = 'TA0006 / T1056.003',
    #             filter  = 'AND',
    #             start   = (13, 42, 53),
    #             end     = (13, 44, 56),
    #             exclude = [
    #                 '128.55.12.110_36196_128.55.12.73_143',
    #                 '128.55.12.110_36197_128.55.12.73_143',
    #                 '128.55.12.110_36198_128.55.12.73_143',
    #                 '/home/admin/.mozilla/firefox/pe11scpa.default/cookies.sqlite',
    #                 '/home/admin/.mozilla/firefox/pe11scpa.default/cookies.sqlite-wal',
    #                 '/home/admin/.mozilla/firefox/pe11scpa.default/formhistory.sqlite',
    #                 '/home/admin/.mozilla/firefox/pe11scpa.default/formhistory.sqlite-journal',
    #             ],
    #         ),
    #     ],
    # },

}

ACTIVE = [
    'browser_extension',
    'backdoor',
    # 'phishing',
    'phishing_attachment',
]

FORCE = True

def to_ns(h, m, s, date):
    dt = datetime(date[0], date[1], date[2], h, m, s, tzinfo=EDT)
    return int(dt.timestamp() * 1e9)

def ns_to_edt(ns):
    try:
        utc = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
        return (utc - timedelta(hours=4)).strftime('%H:%M:%S EDT')
    except Exception:
        return str(ns)

TYPE_COLOUR = {
    'SUBJECT_PROCESS'  : '#4A90D9',
    'FILE_OBJECT_BLOCK': '#F39C12',
    'NetFlowObject'    : '#8E44AD',
}
DEFAULT_COLOUR = '#7F8C8D'

def _edge_colour(etype):
    e = etype.upper()
    if any(k in e for k in ('CLONE', 'FORK', 'EXECUTE')): return '#E74C3C'
    if any(k in e for k in ('READ',  'RECV', 'LOAD')):    return '#2980B9'
    if any(k in e for k in ('WRITE', 'SEND')):             return '#27AE60'
    if any(k in e for k in ('CONNECT','ACCEPT','BIND')):   return '#9B59B6'
    if 'OPEN' in e:                                         return '#F39C12'
    return '#95A5A6'

_LEGEND = [
    mpatches.Patch(facecolor='#4A90D9', edgecolor='#E74C3C', linewidth=3,
                   label='Malicious node (red border)'),
    mpatches.Patch(color='#4A90D9', label='Process'),
    mpatches.Patch(color='#F39C12', label='File / OPEN edge'),
    mpatches.Patch(color='#8E44AD', label='Network flow'),
    mpatches.Patch(color='#E74C3C', label='Clone / Execute'),
    mpatches.Patch(color='#9B59B6', label='Connect'),
    mpatches.Patch(color='#2980B9', label='Read / Recv'),
    mpatches.Patch(color='#27AE60', label='Write / Send'),
]

def draw_tactic(window_edges, mal_uuids, out_png, title):
    node_meta = {}
    G = nx.MultiDiGraph()
    for e in window_edges:
        for uid, ntype, nname in [
            (e['src'], e['src_type'], e['src_name']),
            (e['dst'], e['dst_type'], e['dst_name']),
        ]:
            if uid not in node_meta:
                node_meta[uid] = {'type': ntype, 'name': nname or uid}
        G.add_edge(e['src'], e['dst'], etype=e['etype'])

    n   = G.number_of_nodes()
    pos = nx.spring_layout(G, seed=42, k=3.0) if n <= 20 else \
          nx.spring_layout(G, seed=42, k=2.0, iterations=80)

    mal_nodes  = [u for u in G.nodes() if u in mal_uuids]
    norm_nodes = [u for u in G.nodes() if u not in mal_uuids]
    labels     = {u: (node_meta.get(u, {}).get('name') or u)[-30:] for u in G.nodes()}

    fig, ax = plt.subplots(figsize=(max(20, n * 1.5), max(14, n * 1.2)))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    if norm_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=norm_nodes,
            node_color=[TYPE_COLOUR.get(node_meta.get(u, {}).get('type', ''),
                        DEFAULT_COLOUR) for u in norm_nodes],
            node_size=400, alpha=0.88)
    if mal_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=mal_nodes,
            node_color=[TYPE_COLOUR.get(node_meta.get(u, {}).get('type', ''),
                        DEFAULT_COLOUR) for u in mal_nodes],
            node_size=900, edgecolors='#E74C3C', linewidths=3.5, alpha=0.95)

    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
                            font_size=8, font_color='white')

    e_colours = [_edge_colour(d.get('etype', '')) for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=e_colours,
                           arrows=True, arrowsize=15, width=1.3, alpha=0.75)

    from collections import defaultdict
    _emap = defaultdict(list)
    for u, v, d in G.edges(data=True):
        lbl = d.get('etype', '').replace('EVENT_', '')
        if lbl not in _emap[(u, v)]:
            _emap[(u, v)].append(lbl)
    elabels = {k: '\n'.join(v) for k, v in _emap.items()}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=elabels,
        font_size=8, font_color='#F0E68C',
        bbox=dict(boxstyle='round,pad=0.1', facecolor='#0f0f23', alpha=0.7))

    ax.legend(handles=_LEGEND, loc='upper left',
              facecolor='#0f0f23', edgecolor='white', labelcolor='white', fontsize=9)
    ax.set_title(f'{title}\nnodes={n}  edges={G.number_of_edges()}',
                 color='white', fontsize=12, pad=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'    PNG  -> {out_png}')

def _cache_names(cfg):
    h1, m1, _ = cfg['win_start']
    h2, m2, _ = cfg['win_end']
    tag = f'{h1:02d}_{m1:02d}_to_{h2:02d}_{m2:02d}'
    return f'stage3_edges_{tag}.json', f'stage3_malicious_edges_{tag}.json'

def load_cache(cfg):
    out_dir  = os.path.join(PROJECT_ROOT, cfg['out_dir'])
    os.makedirs(out_dir, exist_ok=True)

    cache_all, cache_mal = _cache_names(cfg)
    path_all = os.path.join(out_dir, cache_all)
    path_mal = os.path.join(out_dir, cache_mal)

    mal_uuids = set()
    csv_path  = os.path.join(PROJECT_ROOT, cfg['csv'])
    with open(csv_path) as f:
        for line in f:
            uid = line.strip().split(',')[0].strip()
            if uid:
                mal_uuids.add(uid)
    print(f'  {len(mal_uuids)} malicious UUIDs  ({os.path.basename(csv_path)})')

    if os.path.exists(path_mal) and os.path.exists(path_all):
        print(f'  [cache] loading edges from JSON ...')
        with open(path_mal) as f:
            mal_edges = json.load(f)['edges']
        with open(path_all) as f:
            all_edges = json.load(f)['edges']
        print(f'  AND-filtered: {len(mal_edges):,}  |  all edges: {len(all_edges):,}')
        return mal_uuids, mal_edges, all_edges

    print(f'  Loading Stage 3 graph')
    with open(PKL_PATH, 'rb') as f:
        g = pickle.load(f)
    print(f'  Graph: {g.number_of_nodes():,} nodes  {g.number_of_edges():,} edges')

    date = cfg['date']
    ws   = to_ns(*cfg['win_start'], date=date)
    we   = to_ns(*cfg['win_end'],   date=date)
    print(f'  Filtering {ns_to_edt(ws)} → {ns_to_edt(we)} ...')

    all_edges = []
    for u, v, k, data in g.edges(keys=True, data=True):
        ts = data.get('ts', 0)
        if ts < ws or ts > we:
            continue
        all_edges.append({
            'src'     : data.get('src_uuid', ''),
            'src_name': data.get('src_name', ''),
            'src_type': g.nodes[u].get('type', '') if u in g.nodes else '',
            'dst'     : data.get('dst_uuid', ''),
            'dst_name': data.get('dst_name', ''),
            'dst_type': g.nodes[v].get('type', '') if v in g.nodes else '',
            'etype'   : data.get('edge_type', ''),
            'ts'      : ts,
            'time_edt': ns_to_edt(ts),
        })
    all_edges.sort(key=lambda x: x['ts'])
    mal_edges = [e for e in all_edges
                 if e['src'] in mal_uuids and e['dst'] in mal_uuids]

    used     = set(n for e in all_edges for n in (e['src'], e['dst']))
    found    = mal_uuids & used
    missing  = mal_uuids - used
    print(f'  Total edges: {len(all_edges):,}  |  AND-filtered: {len(mal_edges):,}')
    print(f'  Malicious nodes found: {len(found)}/{len(mal_uuids)}'
          + (f'  ({len(missing)} missing)' if missing else ''))

    with open(path_all, 'w') as f:
        json.dump({'edges': all_edges}, f, indent=2)
    with open(path_mal, 'w') as f:
        json.dump({'edges': mal_edges}, f, indent=2)
    print(f'  Saved: {cache_all}, {cache_mal}')

    return mal_uuids, mal_edges, all_edges

def extract_tactic(tac, cfg, attack_key, mal_uuids, mal_edges, all_edges):
    date    = cfg['date']
    out_dir = os.path.join(PROJECT_ROOT, cfg['out_dir'])

    t_start  = to_ns(*tac['start'], date=date)
    t_end    = to_ns(*tac['end'],   date=date)
    out_json = os.path.join(out_dir, f"{tac['name']}_{attack_key}.json")
    out_png  = os.path.join(out_dir, f"{tac['name']}_{attack_key}.png")

    if not FORCE and os.path.exists(out_json) and os.path.exists(out_png):
        with open(out_json) as f:
            _cached = json.load(f)
        _nodes = set(e['src'] for e in _cached.get('edges', [])) | set(e['dst'] for e in _cached.get('edges', []))
        covered = _nodes & mal_uuids
        print(f'  [cache] {tac["name"]} already exists — skipping  mal_uuids_covered={len(covered)}/{len(mal_uuids)}')
        return covered

    if tac['filter'] == 'AND':
        edges = [e for e in mal_edges  if t_start <= e['ts'] <= t_end]
    else:
        edges = [e for e in all_edges
                 if t_start <= e['ts'] <= t_end
                 and (e['src'] in mal_uuids or e['dst'] in mal_uuids)]

    exclude = set(tac.get('exclude', []))
    if exclude:
        edges = [e for e in edges
                 if (e['src_name'] or '') not in exclude
                 and (e['dst_name'] or '') not in exclude]

    include_src = set(tac.get('include_src', []))
    if include_src:
        edges = [e for e in edges if (e['src_name'] or '') in include_src]

    nodes        = set(n for e in edges for n in (e['src'], e['dst']))
    covered_uuids = nodes & mal_uuids
    print(f'  {tac["name"]:30s}  filter={tac["filter"]}  '
          f'edges={len(edges):4d}  nodes={len(nodes):3d}  '
          f'mal_uuids_covered={len(covered_uuids)}/{len(mal_uuids)}  '
          f'({ns_to_edt(t_start)} → {ns_to_edt(t_end)})')

    with open(out_json, 'w') as f:
        json.dump({
            'attack'      : cfg['title'],
            'tactic'      : tac['label'],
            'mitre'       : tac['mitre'],
            'date'        : f"{date[0]}-{date[1]:02d}-{date[2]:02d}",
            'filter'      : tac['filter'],
            'window_start': ns_to_edt(t_start),
            'window_end'  : ns_to_edt(t_end),
            'total_edges' : len(edges),
            'total_nodes' : len(nodes),
            'edges'       : edges,
        }, f, indent=2)

    if edges:
        title = (f'{cfg["title"]}\n'
                 f'{tac["label"]}  |  {tac["mitre"]}  |  '
                 f'{ns_to_edt(t_start)[:-4]} → {ns_to_edt(t_end)[:-4]}')
        draw_tactic(edges, mal_uuids, out_png, title)
    else:
        print(f'    (no edges — skipping PNG)')

    return covered_uuids

for attack_key in ACTIVE:
    cfg = ATTACKS[attack_key]
    print(f'\n{"="*65}')
    print(f'  ATTACK: {cfg["title"]}')
    print(f'  OUTPUT: {cfg["out_dir"]}')
    print(f'{"="*65}')

    mal_uuids, mal_edges, all_edges = load_cache(cfg)

    print()
    all_covered = set()
    for tac in cfg['tactics']:
        covered = extract_tactic(tac, cfg, attack_key, mal_uuids, mal_edges, all_edges)
        if covered:
            all_covered |= covered

    missing = mal_uuids - all_covered
    print(f'\n  --- UUID COVERAGE SUMMARY ---')
    print(f'  Total malicious UUIDs : {len(mal_uuids)}')
    print(f'  Covered (all tactics) : {len(all_covered)}')
    print(f'  Missing (not in any)  : {len(missing)}')
    if missing:
        print(f'  Missing UUIDs:')
        for uid in sorted(missing):
            print(f'    {uid}')

print('\nDone.')
