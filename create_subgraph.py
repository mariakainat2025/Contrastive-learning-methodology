import os
import json
from collections import defaultdict
from tqdm import tqdm

from scripts.config import (
    show, ns_to_et, make_node_dict,
    OUTPUT_IOC, EDGES_FILE,
    ATTACK_START_NS, ATTACK_END_NS, WINDOW_SIZE_NS,
    SCAN_TARGET_IPS,
    IOC_IPS, IOC_FILES, IOC_KEYWORDS,
)

def matches_ioc(data):

    # 1. IP exact match
    for field in ('remote_ip', 'local_ip'):
        val = data.get(field, '')
        if val and val in IOC_IPS:
            return True, val

    # 2. File path substring match
    fp = data.get('file_path', '')
    if fp:
        fp_lower = fp.lower()
        for ioc in IOC_FILES:
            if ioc.lower() in fp_lower:
                return True, ioc

    # 3. Process fields
    for field in ('exe_path', 'cmdline'):
        val = data.get(field, '')
        if not val:
            continue
        v = val.lower()
        for ioc in IOC_FILES:
            if ioc.lower() in v:
                return True, ioc


    # 4. Generic name fallback
    name = data.get('name', '')
    if name:
        n = name.lower()
        for ioc in IOC_FILES:
            if ioc.lower() in n:
                return True, ioc
        for kw in IOC_KEYWORDS:
            if kw in n:
                return True, kw

    return False, None

# seed selection
def get_seed_reason(uid, malicious_uuids, node_dict):
    
    is_malicious = uid in malicious_uuids
    hit, ioc_val = matches_ioc(node_dict)

    if is_malicious and hit:
        return 'malicious_uuid+ioc:{}'.format(ioc_val)
    elif is_malicious:
        return 'malicious_uuid'
    else:
        return None

# Main matching + subgraph pipeline

def run_matching(maps):

    os.makedirs(OUTPUT_IOC, exist_ok=True)

    id_nodetype_map   = maps['id_nodetype_map']
    id_nodename_map   = maps['id_nodename_map']
    id_ts_map         = maps['id_ts_map']
    malicious_uuids   = maps['malicious_uuids']


    print('  Attack window : {} → {}'.format(
        ns_to_et(ATTACK_START_NS), ns_to_et(ATTACK_END_NS)))
    print()

    time_ioc_matched = {}
    no_ts_ioc_count  = 0
    outside_skip     = 0

    for uid, ntype in id_nodetype_map.items():
        ts    = id_ts_map.get(uid, {})
        first = ts.get('first', '')
        last  = ts.get('last',  '')

        has_ts    = (first != '' and last != '')
        in_window = has_ts and (int(first) <= ATTACK_END_NS and int(last) >= ATTACK_START_NS)

        if has_ts and not in_window:
            outside_skip += 1
            continue

        node = make_node_dict(uid, maps)
        hit, ioc_val = matches_ioc(node)
        node['matched_ioc'] = ioc_val if ioc_val else ''

        if hit:
            time_ioc_matched[uid] = node
            if not has_ts:
                no_ts_ioc_count += 1

    out_matched = OUTPUT_IOC + 'time_ioc_matched.json'
    with open(out_matched, 'w', encoding='utf-8') as f:
        json.dump(time_ioc_matched, f, indent=2)

    print('  Nodes outside attack window (skipped)  : {}'.format(outside_skip))
    print('  Nodes in window  + IoC matched          : {}'.format(
        len(time_ioc_matched) - no_ts_ioc_count))
    print('  Nodes no timestamp + IoC matched        : {}'.format(no_ts_ioc_count))
    print()
    print('time_ioc_matched.json : {} entries'.format(len(time_ioc_matched)))

    # IoC value breakdown
    matched_ioc_vals = set()
    ioc_counts = {}
    for d in time_ioc_matched.values():
        k = d['matched_ioc']
        matched_ioc_vals.add(k)
        ioc_counts[k] = ioc_counts.get(k, 0) + 1
    if ioc_counts:
        print()
        print('  IoC value breakdown:')
        for k, c in sorted(ioc_counts.items(), key=lambda x: x[1], reverse=True):
            print('    {:45} : {}'.format(k, c))

    type_counts = {}
    for d in time_ioc_matched.values():
        t = d['node_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print()
    print('  Node type breakdown:')
    for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print('    {:40} : {}'.format(t, c))

    also_in_theia = sum(1 for d in time_ioc_matched.values() if d['in_theia_txt'] == 1)
    print()
    print('  theia.txt cross-check:')
    print('    Confirmed in theia.txt                 : {}'.format(also_in_theia))
    print('    NEW (not in theia.txt)                 : {}'.format(
        len(time_ioc_matched) - also_in_theia))

    # IoCs not matched
    all_iocs = (
        [('IP',      v) for v in sorted(IOC_IPS)]     +
        [('FILE',    v) for v in sorted(IOC_FILES)]    +
        [('KEYWORD', v) for v in sorted(IOC_KEYWORDS)]
    )
    unmatched_iocs = [(cat, val) for cat, val in all_iocs if val not in matched_ioc_vals]

    out_not_found = OUTPUT_IOC + 'iocs_not_found.txt'
    with open(out_not_found, 'w', encoding='utf-8') as f:
        f.write('IoCs from DARPA ground truth NOT matched in provenance graph\n')
        f.write('=' * 60 + '\n')
        f.write('Total IoCs defined  : {}\n'.format(len(all_iocs)))
        f.write('IoCs matched        : {}\n'.format(len(all_iocs) - len(unmatched_iocs)))
        f.write('IoCs NOT matched    : {}\n'.format(len(unmatched_iocs)))
        f.write('=' * 60 + '\n\n')
        current_cat = None
        for cat, val in unmatched_iocs:
            if cat != current_cat:
                f.write('\n[{}]\n'.format(cat))
                current_cat = cat
            f.write('  {}\n'.format(val))

    print()
    print('  IoCs NOT matched : {} / {}  →  iocs_not_found.txt'.format(
        len(unmatched_iocs), len(all_iocs)))


  
    print('fixed-window slicing + seed extraction')

    # 6a. Build undirected adjacency from edge list 
    show('  6a. Building adjacency ...')
    adj = defaultdict(set)
    with open(EDGES_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='  adj'):
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            src, dst = parts[0], parts[2]
            adj[src].add(dst)
            adj[dst].add(src)
    print('  Adjacency : {} nodes'.format(len(adj)))

    # Save adjacency to JSON
    adj_path = OUTPUT_IOC + 'adjacency.json'
    with open(adj_path, 'w', encoding='utf-8') as f:
        json.dump({k: list(v) for k, v in adj.items()}, f, indent=2)
    print('  adjacency.json saved : {}'.format(adj_path))

    # Generate fixed-size windows 
    windows = []
    t = ATTACK_START_NS
    win_id = 1
    while t < ATTACK_END_NS:
        w_end = min(t + WINDOW_SIZE_NS, ATTACK_END_NS)
        windows.append({'id': win_id, 'start': t, 'end': w_end})
        t      = w_end
        win_id += 1

    print()
    print('  Windows generated : {}'.format(len(windows)))
    for w in windows:
        dur_min = (w['end'] - w['start']) / 1e9 / 60
        print('    Window {} : {} → {}  ({:.0f} min)'.format(
            w['id'], ns_to_et(w['start']), ns_to_et(w['end']), dur_min))

    # Per-window: nodes → seeds → 2-hop BFS 
    window_results = {}
    all_output     = {}

    print()
    for w in windows:
        w_id    = w['id']
        w_start = w['start']
        w_end   = w['end']

        # nodes active in this window
        window_nodes = {
            uid for uid, ts_ in id_ts_map.items()
            if int(ts_['first']) <= w_end and int(ts_['last']) >= w_start
        }

        # seed selection
        seeds = {}
        for uid in window_nodes:
            nd     = make_node_dict(uid, maps)
            reason = get_seed_reason(uid, malicious_uuids, nd)
            if reason is not None:
                seeds[uid] = reason

        def _is_scan_target(uid):
            return (id_nodetype_map.get(uid) == 'NetFlowObject'
                    and id_nodename_map.get(uid, '') in SCAN_TARGET_IPS)


        subgraph_nodes  = set(seeds.keys())
        blocked_seeds   = 0
        blocked_hop1    = 0
        for seed_uid in list(seeds.keys()):
            if _is_scan_target(seed_uid):      
                blocked_seeds += 1
                continue
            hop1 = adj.get(seed_uid, set()) & window_nodes
            subgraph_nodes |= hop1
            for h1 in hop1:
                if _is_scan_target(h1):       
                    blocked_hop1 += 1
                    continue
                subgraph_nodes |= (adj.get(h1, set()) & window_nodes)

        dedup_map = {}   

        ip_to_repr = {}
        dedup_map  = {}

        for uid in list(subgraph_nodes):
            if id_nodetype_map.get(uid) != 'NetFlowObject':
                continue
            ip = id_nodename_map.get(uid, '')
            if not ip:
                continue
            if ip not in ip_to_repr:
                ip_to_repr[ip] = uid
            else:
                dedup_map[uid] = ip_to_repr[ip]
        for uid in dedup_map:
            subgraph_nodes.discard(uid)
       
        for uid in list(seeds.keys()):
            if uid in dedup_map:
                repr_uid = dedup_map[uid]
                if repr_uid not in seeds:
                    seeds[repr_uid] = seeds[uid]
                del seeds[uid]

        window_results[w_id] = {
            'window_nodes'  : window_nodes,
            'subgraph_nodes': subgraph_nodes,
            'seeds'         : seeds,
            'dedup_map'     : dedup_map,
        }

        # per-window summary
        seed_cats = {}
        for r in seeds.values():
            cat = 'malicious_uuid+ioc' if '+' in r else r.split(':')[0]
            seed_cats[cat] = seed_cats.get(cat, 0) + 1

        print('  Window {} ({} → {}):'.format(w_id, ns_to_et(w_start), ns_to_et(w_end)))
        print('    window_nodes      : {}'.format(len(window_nodes)))
        print('    seeds             : {}'.format(len(seeds)))
        for cat, cnt in sorted(seed_cats.items()):
            print('      {:30} : {}'.format(cat, cnt))
        print('    dedup removed     : {}  (NetFlowObj same IP collapsed)'.format(len(dedup_map)))
        print('    subgraph_nodes    : {}'.format(len(subgraph_nodes)))
        print()

    # Collect subgraph edges 
    show('Collecting subgraph edges ...')
    win_edge_lists = {w['id']: [] for w in windows}
    win_edge_seen  = {w['id']: set() for w in windows}

    with open(EDGES_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='  edges'):
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            src, src_type, dst, dst_type, etype, ts_str = (
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
            ts = int(ts_str)

            for w in windows:
                w_id = w['id']
                if ts < w['start'] or ts > w['end']:
                    continue
                sg   = window_results[w_id]['subgraph_nodes']
                dmap = window_results[w_id]['dedup_map']
                eff_src = dmap.get(src, src)
                eff_dst = dmap.get(dst, dst)
                if eff_src in sg and eff_dst in sg:
                    key = (eff_src, eff_dst, etype)
                    if key not in win_edge_seen[w_id]:
                        win_edge_lists[w_id].append({
                            'src'      : eff_src,
                            'src_type' : src_type,
                            'dst'      : eff_dst,
                            'dst_type' : dst_type,
                            'edge_type': etype,
                            'timestamp': ts,
                        })
                        win_edge_seen[w_id].add(key)

    # Build final output dict and save window_subgraphs.json
    for w in windows:
        w_id    = w['id']
        sg      = window_results[w_id]['subgraph_nodes']
        wn      = window_results[w_id]['window_nodes']
        seeds_w = window_results[w_id]['seeds']
        edges_w = win_edge_lists[w_id]

        seed_list = []
        for uid, reason in seeds_w.items():
            nd = make_node_dict(uid, maps)
            nd['uuid']        = uid
            nd['seed_reason'] = reason

            seed_list.append(nd)

        subgraph_node_list = []
        for uid in sg:
            nd = make_node_dict(uid, maps)
            nd['uuid']    = uid
            nd['is_seed'] = 1 if uid in seeds_w else 0
            subgraph_node_list.append(nd)

        all_output['window_{}'.format(w_id)] = {
            'window_id'          : w_id,
            'start_ns'           : w['start'],
            'end_ns'             : w['end'],
            'start_et'           : ns_to_et(w['start']),
            'end_et'             : ns_to_et(w['end']),
            'window_node_count'  : len(wn),
            'seed_count'         : len(seeds_w),
            'subgraph_node_count': len(sg),
            'subgraph_edge_count': len(edges_w),
            'seed_nodes'         : seed_list,
            'subgraph_nodes'     : subgraph_node_list,
            'subgraph_edges'     : edges_w,
        }

    out_windows = OUTPUT_IOC + 'window_subgraphs.json'
    with open(out_windows, 'w', encoding='utf-8') as f:
        json.dump(all_output, f, indent=2)
    print()
    print('window_subgraphs.json saved : {}'.format(out_windows))

    # ── save one JSON file per window 
    for wkey, wdata in all_output.items():
        out_w = OUTPUT_IOC + '{}_subgraph.json'.format(wkey)
        with open(out_w, 'w', encoding='utf-8') as f:
            json.dump(wdata, f, indent=2)
        print('{}_subgraph.json saved'.format(wkey))

    # summary table
    print()
    print('  {:<10} {:<25} {:<25} {:>12} {:>8} {:>14} {:>14}'.format(
        'Window', 'Start ET', 'End ET',
        'Win.Nodes', 'Seeds', 'Subgr.Nodes', 'Subgr.Edges'))
    print('  ' + '-' * 110)
    for w in windows:
        w_id  = w['id']
        winfo = all_output['window_{}'.format(w_id)]
        print('  {:<10} {:<25} {:<25} {:>12} {:>8} {:>14} {:>14}'.format(
            w_id, winfo['start_et'], winfo['end_et'],
            winfo['window_node_count'], winfo['seed_count'],
            winfo['subgraph_node_count'], winfo['subgraph_edge_count']))

    # seed category breakdown
    print()
    print('  Seed category breakdown:')
    print('  {:<10} {:>16} {:>21} {:>14}'.format(
        'Window', 'malicious_uuid', 'malicious_uuid+ioc', 'TOTAL_mal'))
    print('  ' + '-' * 65)
    for w in windows:
        w_id     = w['id']
        seeds_w  = window_results[w_id]['seeds']
        cnt_mal  = sum(1 for r in seeds_w.values() if r == 'malicious_uuid')
        cnt_both = sum(1 for r in seeds_w.values() if '+' in r)
        print('  {:<10} {:>16} {:>21} {:>14}'.format(
            w_id, cnt_mal, cnt_both, cnt_mal + cnt_both))

    # Save window_both_matched_seeds.txt 
    out_both = OUTPUT_IOC + 'window_both_matched_seeds.txt'
    with open(out_both, 'w', encoding='utf-8') as f:
        f.write('Seeds confirmed by BOTH malicious UUID AND IoC\n')
        f.write('=' * 70 + '\n')
        f.write('Attack window : {} → {}\n\n'.format(
            ns_to_et(ATTACK_START_NS), ns_to_et(ATTACK_END_NS)))

        grand_total = 0
        for w in windows:
            w_id    = w['id']
            seeds_w = window_results[w_id]['seeds']
            both_seeds = {
                uid: reason for uid, reason in seeds_w.items()
                if '+' in reason
            }
            grand_total += len(both_seeds)

            f.write('=' * 70 + '\n')
            f.write('Window {}  :  {} → {}\n'.format(
                w_id, ns_to_et(w['start']), ns_to_et(w['end'])))
            f.write('  Both-matched seeds : {}\n\n'.format(len(both_seeds)))

            if not both_seeds:
                f.write('  (none)\n\n')
                continue

            by_type = {}
            for uid, reason in both_seeds.items():
                ntype = id_nodetype_map.get(uid, 'UNKNOWN')
                by_type.setdefault(ntype, []).append((uid, reason))

            for ntype in sorted(by_type.keys()):
                f.write('  [{} — {} nodes]\n'.format(ntype, len(by_type[ntype])))
                for uid, reason in sorted(
                        by_type[ntype],
                        key=lambda x: id_nodename_map.get(x[0], '')):
                    name     = id_nodename_map.get(uid, '')
                    matched  = reason.split('ioc:', 1)[1] if 'ioc:' in reason else '-'
                    ts_info  = maps['id_ts_map'].get(uid, {})
                    first_ns = ts_info.get('first', '')
                    last_ns  = ts_info.get('last',  '')
                    f.write('    UUID        : {}\n'.format(uid))
                    f.write('    node_type   : {}\n'.format(ntype))
                    f.write('    name        : {}\n'.format(name or '(no name)'))
                    f.write('    matched_ioc : {}\n'.format(matched))
                    f.write('    first_seen  : {}\n'.format(
                        ns_to_et(first_ns) if first_ns else '(unknown)'))
                    f.write('    last_seen   : {}\n\n'.format(
                        ns_to_et(last_ns)  if last_ns  else '(unknown)'))

        f.write('=' * 70 + '\n')
        f.write('GRAND TOTAL both-matched seeds : {}\n'.format(grand_total))

    print('window_both_matched_seeds.txt saved : {}'.format(out_both))
    print()
    show('create_subgraph.py — DONE')
