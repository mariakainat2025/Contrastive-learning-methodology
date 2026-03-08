import os
import json
from collections import defaultdict
from tqdm import tqdm

from scripts.config import (
    show, ns_to_et, make_node_dict,
    OUTPUT_WINDOWS, OUTPUT_BENIGN, EDGES_FILE, MALICIOUS_FILE,
    ATTACK_START_NS, ATTACK_END_NS, WINDOW_SIZE_NS,
)

MAX_NODES      = 20
MAX_DEPTH      = 3    
MIN_NODES      = 5    
MIN_TYPES      = 2    
TOP_K          = 1   



def _dedup_netflow(node_set, edges, id_nodetype_map, id_nodename_map):
    """Collapse NetFlowObjects with the same remote IP (CLIProv §5.2.1 #3)."""
    ip_to_repr = {}
    dedup_map  = {}
    for uid in list(node_set):
        if id_nodetype_map.get(uid) != 'NetFlowObject':
            continue
        ip = id_nodename_map.get(uid, '')
        if not ip:
            continue
        if ip not in ip_to_repr:
            ip_to_repr[ip] = uid
        else:
            dedup_map[uid] = ip_to_repr[ip]

    if not dedup_map:
        return node_set, edges, {}

    new_nodes = node_set - set(dedup_map.keys())
    seen_keys = set()
    new_edges = []
    for e in edges:
        s   = dedup_map.get(e['src'], e['src'])
        d   = dedup_map.get(e['dst'], e['dst'])
        key = (s, d, e['edge_type'])
        if key not in seen_keys and s != d:
            new_edges.append({**e, 'src': s, 'dst': d})
            seen_keys.add(key)
    return new_nodes, new_edges, dedup_map


def _dedup_files(node_set, edges, id_nodetype_map, id_nodename_map):
    file_paths = {
        uid: id_nodename_map.get(uid, uid)
        for uid in node_set
        if id_nodetype_map.get(uid) == 'FILE_OBJECT_BLOCK'
    }
    all_paths = set(file_paths.values())

    cascade_remove = set()
    for uid, path in file_paths.items():
        for other in all_paths:
            if other != path and other.startswith(path.rstrip('/') + '/'):
                cascade_remove.add(uid)
                break

    path_to_repr = {}
    file_dedup   = {}
    for uid in list(node_set):
        if id_nodetype_map.get(uid) != 'FILE_OBJECT_BLOCK':
            continue
        if uid in cascade_remove:
            file_dedup[uid] = None
            continue
        path = id_nodename_map.get(uid, uid)
        if path not in path_to_repr:
            path_to_repr[path] = uid
        else:
            file_dedup[uid] = path_to_repr[path]

    if not file_dedup:
        return node_set, edges, {}

    seen_keys = set()
    new_edges = []
    for e in edges:
        s = file_dedup.get(e['src'], e['src'])
        d = file_dedup.get(e['dst'], e['dst'])
        if s is None or d is None:
            continue
        if s == d:
            continue
        key = (s, d, e['edge_type'])
        if key not in seen_keys:
            new_edges.append({**e, 'src': s, 'dst': d})
            seen_keys.add(key)

    new_nodes = set()
    for e in new_edges:
        new_nodes.add(e['src'])
        new_nodes.add(e['dst'])
    return new_nodes, new_edges, file_dedup


# DFS Sampler

def _dfs_from_seed(seed, adj, node_pool, max_nodes, max_depth):
    """DFS from a process seed node.
    """
    visited  = set()
    node_set = set()
    stack    = [(seed, 0)]          # (node_uuid, current_depth)

    while stack and len(node_set) < max_nodes:
        node, depth = stack.pop()
        if node in visited or node not in node_pool:
            continue
        visited.add(node)
        node_set.add(node)

        if depth < max_depth:
            for nb in sorted(adj.get(node, ())):  
                if nb not in visited and nb in node_pool:
                    stack.append((nb, depth + 1))

    return frozenset(node_set)


def _diversity_score(node_set, id_nodetype_map):
    types      = {id_nodetype_map.get(n, 'UNKNOWN') for n in node_set}
    n_types    = len(types)
    size_ratio = len(node_set) / MAX_NODES
    return n_types + size_ratio        


def _extract_subgraphs_for_window(period_edges, id_nodetype_map, id_nodename_map,
                                   top_k=TOP_K):
    if not period_edges:
        return []

    adj        = defaultdict(set)
    node_pool  = set()
    proc_seeds = set()

    for e in period_edges:
        adj[e['src']].add(e['dst'])
        adj[e['dst']].add(e['src'])
        node_pool.add(e['src'])
        node_pool.add(e['dst'])
        if e['src_type'] == 'SUBJECT_PROCESS':
            proc_seeds.add(e['src'])
        if e['dst_type'] == 'SUBJECT_PROCESS':
            proc_seeds.add(e['dst'])

    if not proc_seeds:
        return []

   
    # first_seen_ts for each process node — from the earliest edge timestamp
    first_seen = {}
    for e in period_edges:
        ts = e.get('timestamp', 0)
        for uid in (e['src'], e['dst']):
            if uid in proc_seeds:
                if uid not in first_seen or ts < first_seen[uid]:
                    first_seen[uid] = ts

    # Dedup by exe_path — keep the EARLIEST process (by first_seen_ts)
    exe_seen    = {}
    dedup_seeds = []
    for proc in sorted(proc_seeds, key=lambda u: first_seen.get(u, 0)):
        exe = id_nodename_map.get(proc, proc)
        if exe not in exe_seen:
            exe_seen[exe] = proc
            dedup_seeds.append(proc)


    seen_sigs  = set()
    candidates = []          

    for seed in dedup_seeds:
        # Step 3: DFS
        node_fs = _dfs_from_seed(seed, adj, node_pool, MAX_NODES, MAX_DEPTH)


        if len(node_fs) < MIN_NODES:
            continue


        types = {id_nodetype_map.get(n, 'UNKNOWN') for n in node_fs}
        if len(types) < MIN_TYPES:
            continue


        if node_fs in seen_sigs:
            continue
        seen_sigs.add(node_fs)


        node_set  = set(node_fs)
        edge_list = [e for e in period_edges
                     if e['src'] in node_set and e['dst'] in node_set]

        if not edge_list:
            continue


        score = _diversity_score(node_set, id_nodetype_map)
        candidates.append((score, node_set, edge_list))

    if not candidates:
        return []

    # Sort by diversity score descending → keep top_k
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:top_k]

    return [(ns, el) for (_, ns, el) in top]


#  Standard JSON output dict ──

def _build_window_dict(window_id, w_start, w_end, node_set, edge_list,
                       maps, parent_window=None):
    node_list = []
    for uid in node_set:
        nd            = make_node_dict(uid, maps)
        nd['uuid']    = uid
        nd['is_seed'] = 0
        node_list.append(nd)

    # Build directed adjacency list: {src_uuid: [dst_uuid, ...]}
    adj = defaultdict(list)
    for e in edge_list:
        adj[e['src']].append(e['dst'])
    adjacency = {k: sorted(v) for k, v in adj.items()}

    return {
        'window_id'          : window_id,
        'parent_window'      : parent_window if parent_window else str(window_id),
        'start_ns'           : w_start,
        'end_ns'             : w_end,
        'start_et'           : ns_to_et(w_start),
        'end_et'             : ns_to_et(w_end),
        'window_node_count'  : len(node_set),
        'seed_count'         : 0,
        'subgraph_node_count': len(node_list),
        'subgraph_edge_count': len(edge_list),
        'seed_nodes'         : [],
        'subgraph_nodes'     : node_list,
        'subgraph_edges'     : edge_list,
        'adjacency'          : adjacency,
    }

#  Collect edges for multiple time windows in one file pass 

def _collect_edges_for_windows(windows, id_nodetype_map,
                                exclude_uuids=None, key_field='id'):
    if exclude_uuids is None:
        exclude_uuids = set()

    win_edges = {w[key_field]: [] for w in windows}
    win_seen  = {w[key_field]: set() for w in windows}

    with open(EDGES_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='  edges'):
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            src, src_type, dst, dst_type, etype, ts_str = (
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
            ts = int(ts_str)

            if src not in id_nodetype_map or dst not in id_nodetype_map:
                continue
            if src_type == 'MemoryObject' or dst_type == 'MemoryObject':
                continue
            if src in exclude_uuids or dst in exclude_uuids:
                continue

            for w in windows:
                if w['start'] <= ts < w['end']:
                    wid = w[key_field]
                    key = (src, dst, etype)
                    if key not in win_seen[wid]:
                        win_edges[wid].append({
                            'src': src, 'src_type': src_type,
                            'dst': dst, 'dst_type': dst_type,
                            'edge_type': etype, 'timestamp': ts,
                        })
                        win_seen[wid].add(key)
                    break
    return win_edges




def _process_window(edges, parent_wk, w, id_nodetype_map, id_nodename_map, maps):
    all_nodes = {e['src'] for e in edges} | {e['dst'] for e in edges}
    all_nodes, edges, _ = _dedup_netflow(all_nodes, edges, id_nodetype_map, id_nodename_map)
    all_nodes, edges, _ = _dedup_files(all_nodes, edges, id_nodetype_map, id_nodename_map)

    subgraphs = _extract_subgraphs_for_window(
        edges, id_nodetype_map, id_nodename_map, top_k=TOP_K)

    output = {}
    for sg_idx, (node_set, edge_list) in enumerate(subgraphs):

        if len(node_set) < MIN_NODES:
            continue

        wkey = parent_wk

        print('  {}  nodes={:,}  edges={:,}'.format(wkey, len(node_set), len(edge_list)))

        ts_vals = [e['timestamp'] for e in edge_list]
        w_start = min(ts_vals) if ts_vals else w['start']
        w_end_  = max(ts_vals) if ts_vals else w['end']

        output[wkey] = _build_window_dict(
            wkey, w_start, w_end_, node_set, edge_list, maps,
            parent_window=parent_wk)

    return output


#  Main entry point

def run_extraction(maps):
    os.makedirs(OUTPUT_WINDOWS, exist_ok=True)

    id_nodetype_map = maps['id_nodetype_map']
    id_nodename_map = maps['id_nodename_map']

    # ── Load malicious UUIDs (excluded from benign) 
    malicious_uuids = set()
    if os.path.exists(MALICIOUS_FILE):
        with open(MALICIOUS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                uid = line.strip()
                if uid:
                    malicious_uuids.add(uid)
    else:
        print('theia.txt not found — benign may contain malicious nodes!')

    show('  Step 1/2 — Malicious windows (15-min + process-seeded DFS, top-{}) ...'.format(TOP_K))

    atk_windows = []
    t = ATTACK_START_NS
    wid = 1
    while t < ATTACK_END_NS:
        w_end = min(t + WINDOW_SIZE_NS, ATTACK_END_NS)
        atk_windows.append({'id': wid, 'start': t, 'end': w_end})
        t   = w_end
        wid += 1

    show('  Collecting attack-period edges ...')
    atk_win_edges = _collect_edges_for_windows(
        atk_windows, id_nodetype_map, exclude_uuids=set())

    all_output = {}
    for w in atk_windows:
        wid       = w['id']
        parent_wk = 'window_{}'.format(wid)
        edges     = atk_win_edges[wid]

        sg_output = _process_window(
            edges, parent_wk, w, id_nodetype_map, id_nodename_map, maps)

        all_output.update(sg_output)


        out_w = OUTPUT_WINDOWS + '{}_subgraph.json'.format(parent_wk)
        with open(out_w, 'w', encoding='utf-8') as f:
            json.dump(sg_output, f, indent=2)

    out_all = OUTPUT_WINDOWS + 'window_subgraphs.json'
    with open(out_all, 'w', encoding='utf-8') as f:
        json.dump(all_output, f, indent=2)

    print(' window_subgraphs.json  ({} subgraphs)'.format(len(all_output)))
    print()


    # BENIGN windows
    show('  Step 2/2 — Benign windows (15-min + process-seeded DFS, top-{}, pre-attack) ...'.format(TOP_K))
    os.makedirs(OUTPUT_WINDOWS, exist_ok=True)

    # Find earliest pre-attack timestamp
    benign_min_ts = None
    with open(EDGES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            ts = int(parts[5])
            if ts < ATTACK_START_NS:
                if benign_min_ts is None or ts < benign_min_ts:
                    benign_min_ts = ts

    if benign_min_ts is None:
        print('No pre-attack edges found — skipping benign windows.')
        return

    # Fixed 15-min tumbling windows over benign period
    ben_windows = []
    t   = benign_min_ts
    wid = 1
    while t < ATTACK_START_NS:
        w_end = min(t + WINDOW_SIZE_NS, ATTACK_START_NS)
        ben_windows.append({'id': wid, 'start': t, 'end': w_end})
        t   = w_end
        wid += 1

    show('  Collecting benign-period edges ...')
    ben_win_edges = _collect_edges_for_windows(
        ben_windows, id_nodetype_map, exclude_uuids=malicious_uuids)

    benign_all_output = {}
    slice_idx = 1   
    for w in ben_windows:
        wid   = w['id']
        edges = ben_win_edges[wid]

        if not edges:
            continue

        parent_bk = 'benign_{}'.format(slice_idx)
        sg_output = _process_window(
            edges, parent_bk, w, id_nodetype_map, id_nodename_map, maps)

        if not sg_output:
            continue

        benign_all_output.update(sg_output)

        # Save per-slice json
        out_w = OUTPUT_WINDOWS + '{}_subgraph.json'.format(parent_bk)
        with open(out_w, 'w', encoding='utf-8') as f:
            json.dump(sg_output, f, indent=2)

        slice_idx += 1

    # Save combined benign json
    out_all = OUTPUT_WINDOWS + 'benign_window_subgraphs.json'
    with open(out_all, 'w', encoding='utf-8') as f:
        json.dump(benign_all_output, f, indent=2)

    print('benign_window_subgraphs.json  ({} subgraphs, {} slices)'.format(
        len(benign_all_output), slice_idx - 1))
    print()
    show('extract_windows.py — DONE')