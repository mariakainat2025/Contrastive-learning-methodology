import json
import os
import time
from datetime import datetime

import networkx as nx

DIR_CASCADE_WINDOW_NS = 5 * 1_000_000_000


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
    t0            = time.time()
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

    for child_path in all_paths:

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

    print(f'  [R2] Directory cascade — {combine:,} merges, cascade EVENT_OPEN edges removed  ({time.time()-t0:.1f}s)')
    return G, combine, removed_nodes


def _build_r2_section(removed_nodes: list) -> list:
    lines = []
    lines.append('\n── R2  Directory Cascade ────────────────────────────────────────────────────────────\n')
    lines.append('  Rule : parent directory absorbed into child when same process accessed both.\n')
    lines.append('         Only the deepest (final) directory level is kept.\n')
    lines.append('  Example: /usr/home/user  →  absorbed into  /usr/home/user/mail\n\n')
    for i, rm in enumerate(removed_nodes, 1):
        lines.append(f'\nMerge #{i}\n')
        lines.append(f'  REMOVED (parent) : node={rm["node"]}  uuid={rm["uuid"]}  name={rm["name"]}\n')
        lines.append(f'  KEPT    (child)  : node={rm["kept_node"]}  uuid={rm["kept_uuid"]}  name={rm["kept_name"]}\n')
        lines.append(f'  Connected process:\n')
        for p in rm['processes']:
            lines.append(f'    node={p["node"]}  uuid={p["uuid"]}  name={p["name"]}\n')
        lines.append(f'  {"─"*88}\n')
    return lines


def _write_report(report_path: str, n0: int, e0: int, n1: int, e1: int,
                  r2_count: int, r2_removed: list):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        W = 92
        f.write('=' * W + '\n')
        f.write('  CLIProv §5.2.1 Graph Reduction Report\n')
        f.write(f'  Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * W + '\n\n')

        f.write('── SUMMARY ──────────────────────────────────────────────────────────────────────────\n')
        f.write(f'  Nodes before : {n0:,}\n')
        f.write(f'  Nodes after  : {n1:,}\n')
        f.write(f'  Nodes removed: {n0-n1:,}\n')
        f.write(f'  Edges before : {e0:,}\n')
        f.write(f'  Edges after  : {e1:,}\n')
        f.write(f'  Edges removed: {e0-e1:,}\n')
        f.write(f'  R2 — Directory cascade    : {r2_count:,} merges, cascade EVENT_OPEN edges removed\n')
        f.write('\n' + '─' * W + '\n')

        f.writelines(_build_r2_section(r2_removed))

        f.write('\n' + '=' * W + '\n')
        f.write('  END OF REPORT\n')
        f.write('=' * W + '\n')

    print(f'  Reduction report saved : {report_path}')


def reduce_graph(G: nx.MultiDiGraph, report_path: str = None):
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    print(f'  Before: nodes {n0:,}  edges {e0:,}')

    G, r2_count, r2_removed = reduce_directory_cascade(G)

    n1, e1 = G.number_of_nodes(), G.number_of_edges()
    print(f'  After : nodes {n1:,}  edges {e1:,}')

    if report_path:
        _write_report(report_path, n0, e0, n1, e1,
                      r2_count=r2_count, r2_removed=r2_removed)
        removed_json_path = report_path.replace('.txt', '_removed_nodes.json')
        with open(removed_json_path, 'w', encoding='utf-8') as f:
            json.dump(r2_removed, f, indent=2)
        print(f'  Removed nodes JSON     : {removed_json_path}')

    return G
