"""

Paper attribution:
   
    TRAP, ThreaTrace
"""

import os
import json
from tqdm import tqdm

from scripts.config import (
    show, update_ts, ns_to_et,
    DATA_PATH, OUTPUT_PARSED, JSON_FILE, EDGES_FILE,
    pattern_uuid, pattern_type, pattern_time,
    pattern_src, pattern_dst1, pattern_dst2,
    pattern_file_path, pattern_file_name,
    pattern_cmdline, pattern_exe_path, pattern_netflow_name,
    pattern_pid, pattern_ppid, pattern_start_ts,
    pattern_parent_uuid, pattern_remote_port,
    pattern_local_addr, pattern_local_port,
    pattern_mem_addr,
)


_SKIP_TYPES = {
    'com.bbn.tc.schema.avro.cdm18.Event',
    'com.bbn.tc.schema.avro.cdm18.Host',
    'com.bbn.tc.schema.avro.cdm18.TimeMarker',
    'com.bbn.tc.schema.avro.cdm18.StartMarker',
    'com.bbn.tc.schema.avro.cdm18.UnitDependency',
    'com.bbn.tc.schema.avro.cdm18.EndMarker',
}


def run_parser():
    os.makedirs(OUTPUT_PARSED, exist_ok=True)
    filepath = DATA_PATH + JSON_FILE

    # Per-UUID attribute maps 
    id_nodetype_map   = {}
    id_nodename_map   = {}
    id_cmdline_map    = {}
    id_exepath_map    = {}
    id_pid_map        = {}
    id_ppid_map       = {}
    id_parent_map     = {}
    id_startts_map    = {}
    id_remoteport_map = {}
    id_localaddr_map  = {}
    id_localport_map  = {}
    id_memaddr_map    = {}
    id_ts_map         = {}

    # Pass 1: Entity extraction
    print()
    print('  ── Pass 1 / 2 : entity extraction ──')
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='  PASS 1'):
            # skip non-entity record types
            if any(t in line for t in _SKIP_TYPES):
                continue

            uuid_match = pattern_uuid.findall(line)
            if not uuid_match:
                continue
            uuid = uuid_match[0]
            if uuid == '00000000-0000-0000-0000-000000000000':
                continue  # null UUID — skip

          
            subject_type = pattern_type.findall(line)
            if len(subject_type) < 1:
                if   'com.bbn.tc.schema.avro.cdm18.MemoryObject'      in line:
                    subject_type = 'MemoryObject'
                elif 'com.bbn.tc.schema.avro.cdm18.NetFlowObject'     in line:
                    subject_type = 'NetFlowObject'
                elif 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                    subject_type = 'UnnamedPipeObject'
                else:
                    continue
            else:
                subject_type = subject_type[0]

            if subject_type == 'SUBJECT_UNIT':
                continue  
            id_nodetype_map[uuid] = subject_type

            #  Type-specific attribute extraction 
            if 'FILE' in subject_type:
                fpath = pattern_file_path.findall(line)
                if fpath:
                    id_nodename_map[uuid] = fpath[0]
                else:
                    fname = pattern_file_name.findall(line)
                    if fname:
                        id_nodename_map[uuid] = fname[0]

            elif subject_type == 'SUBJECT_PROCESS':
                cmd = pattern_cmdline.findall(line)
                if cmd:
                    id_cmdline_map[uuid] = cmd[0]
                exe = pattern_exe_path.findall(line)
                if exe:
                    id_exepath_map[uuid] = exe[0]
                if exe:
                    id_nodename_map[uuid] = exe[0]
                elif cmd:
                    id_nodename_map[uuid] = cmd[0]
                pid = pattern_pid.findall(line)
                if pid:
                    id_pid_map[uuid] = pid[0]
                ppid = pattern_ppid.findall(line)
                if ppid:
                    id_ppid_map[uuid] = ppid[0]
                sts = pattern_start_ts.findall(line)
                if sts:
                    id_startts_map[uuid] = sts[0]
                par = pattern_parent_uuid.findall(line)
                if par and par[0] != '00000000-0000-0000-0000-000000000000':
                    id_parent_map[uuid] = par[0]

            elif subject_type == 'MemoryObject':
                maddr = pattern_mem_addr.findall(line)
                if maddr:
                    id_memaddr_map[uuid] = hex(int(maddr[0]))

            elif subject_type == 'NetFlowObject':
                raddr = pattern_netflow_name.findall(line)
                if raddr:
                    id_nodename_map[uuid] = raddr[0]
                rport = pattern_remote_port.findall(line)
                if rport:
                    id_remoteport_map[uuid] = rport[0]
                laddr = pattern_local_addr.findall(line)
                if laddr:
                    id_localaddr_map[uuid] = laddr[0]
                lport = pattern_local_port.findall(line)
                if lport:
                    id_localport_map[uuid] = lport[0]

    print('  Entities extracted : {:,}'.format(len(id_nodetype_map)))

    # ── Pass 2: Event (edge) extraction ───────────────────────────────────────
    edge_count  = 0
    write_edges = not os.path.exists(EDGES_FILE)
    fw = open(EDGES_FILE, 'w', encoding='utf-8') if write_edges else None

    print()
    print('  ── Pass 2 / 2 : event (edge) extraction ──')
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='  PASS 2'):
                if 'com.bbn.tc.schema.avro.cdm18.Event' not in line:
                    continue

                etype_match = pattern_type.findall(line)
                ts_match    = pattern_time.findall(line)
                if not etype_match or not ts_match:
                    continue

                edgeType  = etype_match[0]
                timestamp = int(ts_match[0].strip())

                srcId_match = pattern_src.findall(line)
                if not srcId_match:
                    continue
                srcId = srcId_match[0]
                if srcId not in id_nodetype_map:
                    continue
                srcType = id_nodetype_map[srcId]
                update_ts(id_ts_map, srcId, timestamp)

                for dst_pattern in (pattern_dst1, pattern_dst2):
                    dstId_match = dst_pattern.findall(line)
                    if dstId_match and dstId_match[0] != 'null':
                        d = dstId_match[0]
                        if d in id_nodetype_map:
                            update_ts(id_ts_map, d, timestamp)
                            if fw:
                                fw.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                                    srcId, srcType,
                                    d, id_nodetype_map[d],
                                    edgeType, timestamp))
                                edge_count += 1
    finally:
        if fw:
            fw.close()

    if write_edges:
        print('  Edges written         : {:,}'.format(edge_count))
        print('  Edge list saved       : {}'.format(EDGES_FILE))
    else:
        print('  Edge list already exists — skipped : {}'.format(EDGES_FILE))

    ts_covered = sum(1 for uid in id_nodetype_map if uid in id_ts_map)
    print('  Nodes with timestamps : {:,} / {:,}  ({:.1f}%)'.format(
        ts_covered, len(id_nodetype_map),
        100.0 * ts_covered / len(id_nodetype_map) if id_nodetype_map else 0))

    # ── Save human-readable lookup tables ─────────────────────────────────────
    with open(OUTPUT_PARSED + 'names.json', 'w', encoding='utf-8') as f:
        json.dump(id_nodename_map, f, indent=2)
    print('  names.json saved      : {:,} entries'.format(len(id_nodename_map)))

    with open(OUTPUT_PARSED + 'types.json', 'w', encoding='utf-8') as f:
        json.dump(id_nodetype_map, f, indent=2)
    print('  types.json saved      : {:,} entries'.format(len(id_nodetype_map)))

    maps = {
        'id_nodetype_map'   : id_nodetype_map,
        'id_nodename_map'   : id_nodename_map,
        'id_cmdline_map'    : id_cmdline_map,
        'id_exepath_map'    : id_exepath_map,
        'id_pid_map'        : id_pid_map,
        'id_ppid_map'       : id_ppid_map,
        'id_parent_map'     : id_parent_map,
        'id_startts_map'    : id_startts_map,
        'id_remoteport_map' : id_remoteport_map,
        'id_localaddr_map'  : id_localaddr_map,
        'id_localport_map'  : id_localport_map,
        'id_memaddr_map'    : id_memaddr_map,
        'id_ts_map'         : id_ts_map,
    }

    print()
    show('parse_provenance.py — DONE')
    return maps
