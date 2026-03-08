import os
import json

from scripts.config import (
    show, make_node_dict, matches_ioc, ns_to_et,
    OUTPUT_IOC_MATCH, MALICIOUS_FILE,
    ATTACK_START_NS, ATTACK_END_NS,
    IOC_IPS, IOC_FILES, IOC_KEYWORDS,
)


def run_ioc_matching(maps):
    os.makedirs(OUTPUT_IOC_MATCH, exist_ok=True)

    id_nodetype_map = maps['id_nodetype_map']
    id_ts_map       = maps['id_ts_map']

    malicious_uuids = set()
    if os.path.exists(MALICIOUS_FILE):
        with open(MALICIOUS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                uid = line.strip()
                if uid:
                    malicious_uuids.add(uid)
        print('  Ground-truth malicious UUIDs : {:,}'.format(len(malicious_uuids)))
    else:
        print('  WARNING: theia.txt not found at {}'.format(MALICIOUS_FILE))
    maps['malicious_uuids'] = malicious_uuids

    # Step 2: IoC + UUID node classification 
    show('  Step 2/3 — Classifying nodes by IoC + malicious UUID ...')
    node_category = {}  
    ioc_matched   = {}  

    for uid in id_nodetype_map:
        nd         = make_node_dict(uid, maps)
        hit, ioc_v = matches_ioc(nd)
        is_mal     = uid in malicious_uuids
        if is_mal and hit:
            node_category[uid] = 'malicious_uuid+ioc'
            ioc_matched[uid]   = ioc_v
        elif is_mal:
            node_category[uid] = 'malicious_uuid'
        elif hit:
            node_category[uid] = 'ioc_only'
            ioc_matched[uid]   = ioc_v
        else:
            node_category[uid] = 'clean'

    maps['node_category'] = node_category
    maps['ioc_matched']   = ioc_matched

    _cats = ('malicious_uuid+ioc', 'malicious_uuid', 'ioc_only', 'clean')
    _cnt  = {c: sum(1 for v in node_category.values() if v == c) for c in _cats}
    print('  Node classification results:')
    print('    malicious_uuid+ioc : {:>6,}'.format(_cnt['malicious_uuid+ioc']))
    print('    malicious_uuid     : {:>6,}'.format(_cnt['malicious_uuid']))
    print('    ioc_only           : {:>6,}'.format(_cnt['ioc_only']))
    print('    clean              : {:>6,}'.format(_cnt['clean']))
    print()

    # Step 3: Time-window IoC filtering 
    show('  Step 3/3 — Filtering IoC nodes by attack time window ...')
    time_ioc_matched = {}
    no_ts_ioc_count  = 0
    outside_skip     = 0

    for uid, ioc_val in ioc_matched.items():
        ts    = id_ts_map.get(uid, {})
        first = ts.get('first', '')
        last  = ts.get('last',  '')
        has_ts    = (first != '' and last != '')
        in_window = has_ts and (int(first) <= ATTACK_END_NS and int(last) >= ATTACK_START_NS)

        if has_ts and not in_window:
            outside_skip += 1
            continue

        node = make_node_dict(uid, maps)
        node['matched_ioc'] = ioc_val
        time_ioc_matched[uid] = node
        if not has_ts:
            no_ts_ioc_count += 1

    out_matched = OUTPUT_IOC_MATCH + 'time_ioc_matched.json'
    with open(out_matched, 'w', encoding='utf-8') as f:
        json.dump(time_ioc_matched, f, indent=2)

    print('  Nodes outside attack window (skipped)  : {:,}'.format(outside_skip))
    print('  Nodes in window  + IoC matched          : {:,}'.format(
        len(time_ioc_matched) - no_ts_ioc_count))
    print('  Nodes no timestamp + IoC matched        : {:,}'.format(no_ts_ioc_count))
    print('  time_ioc_matched.json  : {:,} entries'.format(len(time_ioc_matched)))

    # IoC value breakdown
    ioc_counts       = {}
    matched_ioc_vals = set()
    for d in time_ioc_matched.values():
        k = d['matched_ioc']
        matched_ioc_vals.add(k)
        ioc_counts[k] = ioc_counts.get(k, 0) + 1
    if ioc_counts:
        print()
        print('  IoC value breakdown:')
        for k, c in sorted(ioc_counts.items(), key=lambda x: x[1], reverse=True):
            print('    {:45} : {:,}'.format(k, c))

    type_counts = {}
    for d in time_ioc_matched.values():
        t = d['node_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print()
    print('  Node type breakdown:')
    for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print('    {:40} : {:,}'.format(t, c))

    also_in_theia = sum(1 for d in time_ioc_matched.values() if d['in_theia_txt'] == 1)
    print()
    print('  Ground-truth cross-check (theia.txt):')
    print('    Confirmed in theia.txt : {:,}'.format(also_in_theia))
    print('    NEW (not in theia.txt) : {:,}'.format(len(time_ioc_matched) - also_in_theia))

    # Log unmatched IoCs
    all_iocs = (
        [('IP',      v) for v in sorted(IOC_IPS)]     +
        [('FILE',    v) for v in sorted(IOC_FILES)]    +
        [('KEYWORD', v) for v in sorted(IOC_KEYWORDS)]
    )
    unmatched_iocs = [(cat, val) for cat, val in all_iocs if val not in matched_ioc_vals]
    out_not_found  = OUTPUT_IOC_MATCH + 'iocs_not_found.txt'
    with open(out_not_found, 'w', encoding='utf-8') as f:
        f.write('IoCs from DARPA ground truth NOT matched in provenance graph\n')
        f.write('=' * 60 + '\n')
        f.write('Total IoCs defined : {}\n'.format(len(all_iocs)))
        f.write('IoCs matched       : {}\n'.format(len(all_iocs) - len(unmatched_iocs)))
        f.write('IoCs NOT matched   : {}\n'.format(len(unmatched_iocs)))
        f.write('=' * 60 + '\n\n')
        current_cat = None
        for cat, val in unmatched_iocs:
            if cat != current_cat:
                f.write('\n[{}]\n'.format(cat))
                current_cat = cat
            f.write('  {}\n'.format(val))
    print()
    print('  Unmatched IoCs : {:,} / {:,}  →  iocs_not_found.txt'.format(
        len(unmatched_iocs), len(all_iocs)))

    print()
    show('match_iocs.py — DONE')
