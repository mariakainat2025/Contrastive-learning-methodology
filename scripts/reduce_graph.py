import os
import json
import time
from collections import defaultdict

import networkx as nx
from networkx.readwrite import json_graph
from Levenshtein import ratio as lev_ratio
from tqdm import tqdm

from scripts.config import OUTPUT_GRAPHS

SIMILAR_FILE_THRESHOLD = 0.7
SIMILAR_FILE_WINDOW_NS = 30_000_000_000

import re

_TEMP_LOG_CACHE_DIRS = ('/tmp/', '/var/tmp/', '/var/log/', '/var/cache/')
_TEMP_LOG_CACHE_EXTS = {'.log', '.tmp', '.temp', '.cache', '.lock', '.pid'}

def _is_temp_log_cache(fullpath):
    for d in _TEMP_LOG_CACHE_DIRS:
        if d in fullpath:
            return True
    _, ext = os.path.splitext(fullpath)
    return ext.lower() in _TEMP_LOG_CACHE_EXTS

NETFLOW_WINDOW_NS = 1_000_000_000

MERGE_EVENTS = {
    'EVENT_CONNECT', 'EVENT_SENDTO', 'EVENT_RECVFROM',
    'EVENT_SENDMSG', 'EVENT_RECVMSG', 'EVENT_ACCEPT', 'EVENT_BIND',
}

def _parse_netflow_name(name):
    parts = name.split('_')
    local_ip    = parts[0].strip() if len(parts) > 0 else 'null'
    local_port  = parts[1].strip() if len(parts) > 1 else 'null'
    remote_ip   = parts[2].strip() if len(parts) > 2 else 'null'
    remote_port = parts[3].strip() if len(parts) > 3 else 'null'
    return local_ip, local_port, remote_ip, remote_port

def _process_neighbours(G, n):
    return {
        nb for nb in list(G.predecessors(n)) + list(G.successors(n))
        if G.nodes[nb].get('type') == 'SUBJECT_PROCESS'
    }

def _redirect_and_remove(G, old_n, new_n, shared_procs=None):
    if not shared_procs:
        return

    for pred, _, k, d in list(G.in_edges(old_n, keys=True, data=True)):
        if pred in shared_procs and d.get('edge_type') == 'EVENT_OPEN':
            G.remove_edge(pred, old_n, k)

def reduce_directory_cascade(G: nx.MultiDiGraph):
    combine       = 0
    removed_nodes = []

    file_nodes = [n for n, d in G.nodes(data=True)
                  if 'FILE' in d.get('type', '') and d.get('name', '').strip('/')]

    path_node = {}
    for n in file_nodes:
        p = G.nodes[n]['name'].rstrip('/')
        if p:
            path_node[p] = n

    proc_file_order = {}
    for n, d in G.nodes(data=True):
        if d.get('type') != 'SUBJECT_PROCESS':
            continue
        neighbours = list(G.predecessors(n)) + list(G.successors(n))
        files = [nb for nb in neighbours
                 if 'FILE' in G.nodes[nb].get('type', '')]
        files.sort(key=lambda x: (G.nodes[x].get('ts', 0), x))
        seen, ordered = set(), []
        for x in files:
            if x not in seen:
                seen.add(x)
                ordered.append(x)
        if ordered:
            proc_file_order[n] = ordered

    all_paths = sorted(path_node.keys(), key=lambda p: p.count('/'), reverse=True)

    pbar = tqdm(all_paths, desc='  dir cascade', unit='path', leave=False)
    for child_path in pbar:

        child_n = path_node.get(child_path)
        if child_n is None or not G.has_node(child_n):
            continue

        parent_path = os.path.dirname(child_path)
        if not parent_path or parent_path == child_path:
            continue

        parent_n = path_node.get(parent_path)
        if parent_n is None or not G.has_node(parent_n):
            continue

        if G.nodes[parent_n].get('name', '').rstrip('/') != parent_path:
            continue

        parent_ts = G.nodes[parent_n].get('ts', 0)
        child_ts  = G.nodes[child_n].get('ts', 0)
        if parent_ts > child_ts:
            continue

        parent_procs = _process_neighbours(G, parent_n)
        child_procs  = _process_neighbours(G, child_n)

        shared_procs = parent_procs & child_procs
        if not shared_procs:
            continue

        consecutive = False
        for proc in shared_procs:
            ordered = proc_file_order.get(proc, [])
            for idx in range(len(ordered) - 1):
                if ordered[idx] == parent_n and ordered[idx + 1] == child_n:
                    consecutive = True
                    break
            if consecutive:
                break
        if not consecutive:
            continue

        combine          += 1
        parent_uuid       = G.nodes[parent_n].get('uuid', 'N/A')
        child_uuid        = G.nodes[child_n].get('uuid',  'N/A')
        child_actual_name = G.nodes[child_n].get('name', child_path)
        proc_info = []
        for p in shared_procs:
            proc_info.append({
                'node': p,
                'uuid': G.nodes[p].get('uuid', 'N/A'),
                'name': G.nodes[p].get('name', str(p)),
            })
        removed_nodes.append({
            'node'      : parent_n,
            'uuid'      : parent_uuid,
            'name'      : parent_path,
            'type'      : G.nodes[parent_n].get('type', ''),
            'kept_node' : child_n,
            'kept_uuid' : child_uuid,
            'kept_name' : child_actual_name,
            'processes' : proc_info,
        })
        _redirect_and_remove(G, parent_n, child_n, shared_procs=shared_procs)
        path_node[parent_path] = child_n

    return G, combine, removed_nodes

def reduce_netflow_edges(G: nx.MultiDiGraph):
    nodes_before = G.number_of_nodes()
    edges_before = G.number_of_edges()

    groups = defaultdict(list)

    for proc, nf, key, data in list(G.edges(keys=True, data=True)):
        etype = data.get('edge_type', '')
        if etype not in MERGE_EVENTS:
            continue
        nf_data = G.nodes.get(nf, {})
        if nf_data.get('type') != 'NetFlowObject':
            continue
        nf_name = nf_data.get('name', '')
        local_ip, local_port, remote_ip, remote_port = _parse_netflow_name(nf_name)
        groups[(proc, remote_ip)].append((data.get('ts', 0), nf, key, etype))

    edges_to_remove  = []
    edges_to_add     = []
    node_name_update = {}
    nodes_to_check   = set()

    total_before   = 0
    total_after    = 0
    windows_merged = 0

    def _apply_window(proc, events, window_ns=None):
        nonlocal total_before, total_after, windows_merged
        win = window_ns if window_ns is not None else NETFLOW_WINDOW_NS
        total_before += len(events)
        i = 0
        while i < len(events):
            t_start = events[i][0]
            window  = [events[i]]
            j       = i + 1
            while j < len(events) and events[j][0] - t_start <= win:
                window.append(events[j])
                j += 1

            if len(window) > 1:
                windows_merged += 1
                rep_nf   = window[0][1]
                rep_ts   = window[0][0]
                rep_name = G.nodes[rep_nf].get('name', '')
                local_ip, local_port, remote_ip, remote_port = _parse_netflow_name(rep_name)

                event_types = sorted(set(e[3].replace('EVENT_', '') for e in window))
                merged_etype = '_'.join(event_types)

                remote_ports = sorted(set(
                    _parse_netflow_name(G.nodes[e[1]].get('name', ''))[3]
                    for e in window
                    if _parse_netflow_name(G.nodes[e[1]].get('name', ''))[3]
                    not in ('null', '0', '', 'NA')
                ))
                merged_remote_ports = '_'.join(remote_ports) if remote_ports else 'null'

                local_ports = sorted(set(
                    _parse_netflow_name(G.nodes[e[1]].get('name', ''))[1]
                    for e in window
                    if _parse_netflow_name(G.nodes[e[1]].get('name', ''))[1]
                    not in ('null', '0', '', 'NA')
                ))
                merged_local_ports = '_'.join(local_ports) if local_ports else 'null'

                merged_name = f'{local_ip}_{local_port}_{remote_ip}_{merged_remote_ports}'

                for ts, nf, key, etype in window:
                    edges_to_remove.append((proc, nf, key))

                edges_to_add.append((proc, rep_nf, {
                    'edge_type': merged_etype,
                    'ts'       : rep_ts,
                    'src_uuid' : G.nodes[proc].get('uuid', ''),
                    'dst_uuid' : G.nodes[rep_nf].get('uuid', ''),
                    'src_name' : G.nodes[proc].get('name', ''),
                    'dst_name' : merged_name,
                }))

                node_name_update[rep_nf] = merged_name

                unique_nf = {e[1] for e in window}
                unique_nf.discard(rep_nf)
                nodes_to_check.update(unique_nf)

                total_after += 1
            else:
                total_after += 1

            i = j

    pbar = tqdm(groups.items(), desc='  netflow merge', unit='group', leave=False,
                total=len(groups))
    for (proc, remote_ip), events in pbar:
        events.sort(key=lambda x: x[0])
        _apply_window(proc, events)

    for proc, nf, key in edges_to_remove:
        if G.has_edge(proc, nf, key):
            G.remove_edge(proc, nf, key)

    for proc, nf, data in edges_to_add:
        G.add_edge(proc, nf, **data)

    for nf, new_name in node_name_update.items():
        if nf in G.nodes:
            G.nodes[nf]['name'] = new_name

    nf_nodes_removed = 0
    for nf in nodes_to_check:
        if nf in G.nodes and G.degree(nf) == 0:
            G.remove_node(nf)
            nf_nodes_removed += 1

    nodes_after = G.number_of_nodes()
    edges_after = G.number_of_edges()

    return G, windows_merged, nodes_before, nodes_after, edges_before, edges_after

def reduce_similar_files(G: nx.MultiDiGraph,
                         threshold=SIMILAR_FILE_THRESHOLD,
                         window_ns=SIMILAR_FILE_WINDOW_NS):
    nodes_before = G.number_of_nodes()
    edges_before = G.number_of_edges()
    merged_count = 0

    file_procs = {}
    for n, d in G.nodes(data=True):
        if 'FILE' not in d.get('type', '') or not d.get('name'):
            continue
        procs = frozenset(
            nb for nb in list(G.predecessors(n)) + list(G.successors(n))
            if G.nodes[nb].get('type') == 'SUBJECT_PROCESS'
        )
        if procs:
            file_procs[n] = procs

    dir_groups = defaultdict(list)
    for n, procs in file_procs.items():
        name   = G.nodes[n].get('name', '')
        parent = os.path.dirname(name)
        if parent:
            dir_groups[parent].append(
                (n, os.path.basename(name), G.nodes[n].get('ts', 0), procs)
            )

    pbar = tqdm(dir_groups.items(), desc='  similar files', unit='dir', leave=False,
                total=len(dir_groups))
    for parent_dir, files in pbar:
        if len(files) < 2:
            continue

        files.sort(key=lambda x: x[2])

        representative = {}
        merge_sim      = {}

        for i in range(len(files)):
            nid_i, bname_i, ts_i, procs_i = files[i]
            if not G.has_node(nid_i):
                continue
            fullpath_i = G.nodes[nid_i].get('name', '')
            if not _is_temp_log_cache(fullpath_i):
                continue
            rep_i = representative.get(nid_i, nid_i)

            for j in range(i + 1, len(files)):
                nid_j, bname_j, ts_j, procs_j = files[j]
                fullpath_j = G.nodes[nid_j].get('name', '')
                if not _is_temp_log_cache(fullpath_j):
                    continue
                if nid_j in representative or not G.has_node(nid_j):
                    continue

                if ts_j - ts_i > window_ns:
                    break

                if not (procs_i & procs_j):
                    continue

                sim = lev_ratio(bname_i, bname_j)
                if sim > threshold:
                    representative[nid_j] = rep_i
                    merge_sim[nid_j]      = sim

        if not representative:
            continue

        for old_node, rep_node in representative.items():
            if not G.has_node(old_node) or not G.has_node(rep_node):
                continue
            existing_in  = {(p, d.get('edge_type'))
                            for p, _, d in G.in_edges(rep_node, data=True)}
            existing_out = {(s, d.get('edge_type'))
                            for _, s, d in G.out_edges(rep_node, data=True)}
            for pred, _, k, d in list(G.in_edges(old_node, keys=True, data=True)):
                if pred != rep_node and (pred, d.get('edge_type')) not in existing_in:
                    G.add_edge(pred, rep_node, **d)
                    existing_in.add((pred, d.get('edge_type')))
            for _, succ, k, d in list(G.out_edges(old_node, keys=True, data=True)):
                if succ != rep_node and (succ, d.get('edge_type')) not in existing_out:
                    G.add_edge(rep_node, succ, **d)
                    existing_out.add((succ, d.get('edge_type')))
            G.remove_node(old_node)
            merged_count += 1

    nodes_after = G.number_of_nodes()
    edges_after = G.number_of_edges()

    return G, merged_count, nodes_before, nodes_after, edges_before, edges_after


def reduce_graph(G: nx.MultiDiGraph):
    n0, e0 = G.number_of_nodes(), G.number_of_edges()

    G, r2_count, r2_removed = reduce_directory_cascade(G)
    n1, e1 = G.number_of_nodes(), G.number_of_edges()
    tqdm.write(f'  [1/3] Directory Cascade')
    tqdm.write(f'        nodes  : {n0:,} → {n1:,}  removed: {n0-n1:,}')
    tqdm.write(f'        edges  : {e0:,} → {e1:,}  removed: {e0-e1:,}')
    tqdm.write(f'        merged : {r2_count:,} parent-child path pairs')

    G, r3_windows, r3_nb, r3_na, r3_eb, r3_ea = reduce_netflow_edges(G)
    n2, e2 = G.number_of_nodes(), G.number_of_edges()
    tqdm.write(f'  [2/3] NetFlow Merge')
    tqdm.write(f'        nodes  : {n1:,} → {n2:,}  removed: {n1-n2:,}')
    tqdm.write(f'        edges  : {e1:,} → {e2:,}  removed: {e1-e2:,}')
    tqdm.write(f'        windows: {r3_windows:,} merged')

    G, r4_merged, r4_nb, r4_na, r4_eb, r4_ea = reduce_similar_files(G)
    n3, e3 = G.number_of_nodes(), G.number_of_edges()
    tqdm.write(f'  [3/3] Similar File Merge')
    tqdm.write(f'        nodes  : {n2:,} → {n3:,}  removed: {n2-n3:,}')
    tqdm.write(f'        edges  : {e2:,} → {e3:,}  removed: {e2-e3:,}')
    tqdm.write(f'        merged : {r4_merged:,} similar file nodes')

    print()
    print(f'  ── Total ──────────────────────────────')
    print(f'  nodes  : {n0:,} → {n3:,}  removed: {n0-n3:,}')
    print(f'  edges  : {e0:,} → {e3:,}  removed: {e0-e3:,}')

    out_dir = os.path.join(OUTPUT_GRAPHS, 'reduced_graph')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'reduced_graph.json')
    graph_data = json_graph.node_link_data(G)
    for edge in graph_data.get('links', []):
        edge.pop('source', None)
        edge.pop('target', None)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)
    print(f'  Saved  → {out_path}')

    return G
