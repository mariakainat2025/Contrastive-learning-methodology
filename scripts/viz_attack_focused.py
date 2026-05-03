"""
Clean, focused visualization of the phishing-email-executable-attachment attack.
Shows only the key nodes: fluxbox, thunderbird chain, IMAP server, INBOX file,
gnome-terminal, and bash — with clear colour coding and labels.
"""
import os, sys, json, collections
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Key node UUIDs ──────────────────────────────────────────────────────────
FOCUS_NODES = {
    # Attack chain
    '8C081700-0000-0000-0000-000000000020': ('fluxbox\n(window mgr)', 'process',  True),
    'A414A414-0400-0000-0000-000000000020': ('fluxbox\n(session)',    'process',  True),
    'A514A414-0400-0000-0000-000000000020': ('thunderbird\n(launcher)','process', True),
    'BC14A414-0400-0000-0000-000000000020': ('thunderbird\n(INBOX thread)','process', True),
    '0100D00F-BB0E-2600-0000-0000BF8A1F28': ('INBOX-1\n(phishing email)', 'file',  True),
    '0100D00F-890E-2600-0000-00008D17591E': ('INBOX-1.msf\n(mail store)',  'file',  True),
    '80370C6E-09A5-8037-0C49-8F0000000040': ('128.55.12.73:143\n(IMAP server)', 'net', True),
    # Background noise
    '17173C19-0400-0000-0000-000000000020': ('gnome-terminal\n(background)', 'process', False),
    '1E173C19-0400-0000-0000-000000000020': ('bash\n(background)',           'process', False),
}

COLOURS = {
    ('process', True):  '#E74C3C',   # red  = attack process
    ('process', False): '#7F8C8D',   # grey = background process
    ('file',    True):  '#F39C12',   # orange = attack file
    ('net',     True):  '#8E44AD',   # purple = network
}

INPUT  = 'input/test/attack_subgraphs_phishing_email_executable_attachment.json'
OUTPUT = 'output/theia/viz/phishing_attack_focused.png'

def short(name):
    return name  # already short labels

def main():
    with open(INPUT) as f:
        data = json.load(f)
    sg = data['subgraphs'][0]
    uuid_to_int = {nd['uuid']: iid for iid, nd in sg['nodes']}
    int_to_data = {iid: nd         for iid, nd in sg['nodes']}

    focus_uuids = set(FOCUS_NODES.keys())
    focus_ints  = {uuid_to_int[u] for u in focus_uuids if u in uuid_to_int}

    # ── Build graph with only focus nodes ──────────────────────────────────
    G = nx.MultiDiGraph()
    for uid, (label, ntype, is_attack) in FOCUS_NODES.items():
        iid = uuid_to_int.get(uid)
        if iid is not None:
            G.add_node(iid, label=label, ntype=ntype, is_attack=is_attack)

    # Aggregate edges between focus nodes
    edge_agg = collections.defaultdict(set)  # (src, dst) -> set of etypes
    for edge in sg['edges']:
        src_uuid, dst_uuid, _, edata = edge
        si = uuid_to_int.get(src_uuid)
        di = uuid_to_int.get(dst_uuid)
        if si in focus_ints and di in focus_ints and si != di:
            etype = edata.get('edge_type','?').replace('EVENT_','')
            edge_agg[(si, di)].add(etype)

    for (si, di), etypes in edge_agg.items():
        G.add_edge(si, di, label='\n'.join(sorted(etypes)))

    # ── Layout ─────────────────────────────────────────────────────────────
    # Manual positions for clarity (attack chain left-to-right, noise on side)
    pos_map = {
        '8C081700-0000-0000-0000-000000000020': (0.0,  0.0),   # fluxbox seed
        'A414A414-0400-0000-0000-000000000020': (1.5,  0.0),   # fluxbox session
        'A514A414-0400-0000-0000-000000000020': (3.0,  0.0),   # tb launcher
        'BC14A414-0400-0000-0000-000000000020': (4.5,  0.0),   # tb thread
        '0100D00F-BB0E-2600-0000-0000BF8A1F28': (6.0,  0.8),   # INBOX
        '0100D00F-890E-2600-0000-00008D17591E': (6.0, -0.8),   # INBOX.msf
        '80370C6E-09A5-8037-0C49-8F0000000040': (7.5,  0.0),   # IMAP server
        # Background — below the main chain
        '17173C19-0400-0000-0000-000000000020': (3.0, -2.0),   # gnome-terminal
        '1E173C19-0400-0000-0000-000000000020': (4.5, -2.0),   # bash
    }
    pos = {}
    for uid, xy in pos_map.items():
        iid = uuid_to_int.get(uid)
        if iid in G.nodes():
            pos[iid] = xy

    # ── Styling ────────────────────────────────────────────────────────────
    node_colours, node_sizes, labels = [], [], {}
    for n in G.nodes():
        nd = G.nodes[n]
        key = (nd['ntype'], nd['is_attack'])
        node_colours.append(COLOURS.get(key, '#95A5A6'))
        node_sizes.append(2200 if nd['is_attack'] else 1600)
        labels[n] = nd['label']

    fig, ax = plt.subplots(figsize=(22, 9))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colours,
                           node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
                            font_size=8, font_color='white', font_weight='bold')

    edge_labels = {}
    for u, v, data_ in G.edges(data=True):
        lbl = data_.get('label','')
        edge_labels[(u, v)] = lbl

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color='#BDC3C7', arrows=True,
                           arrowsize=20, width=1.8, alpha=0.75,
                           connectionstyle='arc3,rad=0.12',
                           min_source_margin=30, min_target_margin=30)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                 font_size=7, font_color='#F0E68C',
                                 bbox=dict(alpha=0))

    # Annotation boxes
    ax.text(3.75, 0.55, 'ATTACK CHAIN', color='#E74C3C',
            fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#E74C3C'))
    ax.text(3.75, -1.45, 'BACKGROUND NOISE  (NOT part of attack)', color='#7F8C8D',
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#7F8C8D'))

    legend_items = [
        mpatches.Patch(color='#E74C3C', label='Attack process'),
        mpatches.Patch(color='#F39C12', label='Email file (attack artifact)'),
        mpatches.Patch(color='#8E44AD', label='Network (IMAP server)'),
        mpatches.Patch(color='#7F8C8D', label='Background process (noise)'),
    ]
    ax.legend(handles=legend_items, loc='upper left',
              facecolor='#0f0f23', edgecolor='white',
              labelcolor='white', fontsize=9)

    ax.set_title(
        'Phishing Email w/ Executable Attachment — Focused Attack Graph\n'
        'thunderbird reads INBOX (phishing email) → connects to IMAP server 128.55.12.73\n'
        'gnome-terminal & bash are background noise: they share system libs with thunderbird but are NOT part of the attack',
        color='white', fontsize=10, pad=12)
    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved -> {OUTPUT}')

if __name__ == '__main__':
    main()
