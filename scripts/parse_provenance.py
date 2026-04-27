import os
import re
import string
import json
from tqdm import tqdm

from scripts.config import (
    show, update_ts, ns_to_et,
    DATA_PATH, OUTPUT_PARSED, EDGES_FILE,
    ATTACK_START_NS, ATTACK_END_NS,
    pattern_uuid, pattern_type, pattern_time,
    pattern_src, pattern_dst1, pattern_dst2,
    pattern_mem_addr,
)

try:
    from nostril import nonsense as _nonsense
    _HAS_NOSTRIL = True
except ImportError:
    _HAS_NOSTRIL = False

_SKIP_TYPES = {
    'com.bbn.tc.schema.avro.cdm18.Event',
    'com.bbn.tc.schema.avro.cdm18.Host',
    'com.bbn.tc.schema.avro.cdm18.TimeMarker',
    'com.bbn.tc.schema.avro.cdm18.StartMarker',
    'com.bbn.tc.schema.avro.cdm18.UnitDependency',
    'com.bbn.tc.schema.avro.cdm18.EndMarker',
}

def run_parser(file=None):
    os.makedirs(OUTPUT_PARSED, exist_ok=True)

    import re as _re
    def _is_cdm_file(fname):
        if fname.startswith('._'):
            return False
        if fname.endswith('.json'):
            return True
        if _re.search(r'\.json\.\d+$', fname):
            return True
        return False

    def _natural_key(path):
        import re
        name = os.path.splitext(os.path.basename(path))[0]
        parts = re.split(r'(\d+)', name)
        return [int(p) if p.isdigit() else p for p in parts]

    if file is not None:
        fpath = file if os.path.isabs(file) else os.path.join(DATA_PATH, file)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f'File not found: {fpath}')
        json_files = [fpath]
    else:
        json_files = sorted(
            (os.path.join(DATA_PATH, f)
             for f in os.listdir(DATA_PATH)
             if _is_cdm_file(f)),
            key=_natural_key
        )
    if not json_files:
        raise FileNotFoundError(
            'No CDM18 .json files found in {}\n'
            'Place your DARPA provenance log(s) there and re-run.'.format(DATA_PATH)
        )
    print('  Input file(s) found : {}'.format(len(json_files)))
    for p in json_files:
        print('    {}'.format(p))

   
    node_type_map_path = os.path.join(OUTPUT_PARSED, 'node_type_map.json')
    edge_type_map_path = os.path.join(OUTPUT_PARSED, 'edge_type_map.json')

    if os.path.exists(node_type_map_path):
        with open(node_type_map_path, 'r') as f:
            node_type_dict = json.load(f)
    else:
        node_type_dict = {}

    if os.path.exists(edge_type_map_path):
        with open(edge_type_map_path, 'r') as f:
            edge_type_dict = json.load(f)
    else:
        edge_type_dict = {}

   

    id_nodetype_map   = {}
    id_nodename_map   = {}
    id_cmdline_map    = {}
    id_exepath_map    = {}
    id_remoteport_map = {}
    id_localaddr_map  = {}
    id_localport_map  = {}
    id_memaddr_map    = {}
    id_ts_map         = {}

    print()
    print('  ── Pass 1 / 2 : entity extraction ──')
    for filepath in json_files:
        print('  {}'.format(os.path.basename(filepath)))
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='  PASS 1'):
                if any(t in line for t in _SKIP_TYPES):
                    continue

                uuid_match = pattern_uuid.findall(line)
                if not uuid_match:
                    continue
                uuid = uuid_match[0]
                if uuid == '00000000-0000-0000-0000-000000000000':
                    continue

                subject_type = pattern_type.findall(line)
                if len(subject_type) < 1:
                    if   'com.bbn.tc.schema.avro.cdm18.MemoryObject'      in line:
                         continue
                    elif 'com.bbn.tc.schema.avro.cdm18.NetFlowObject'     in line:
                        subject_type = 'NetFlowObject'
                    elif 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        continue
                    else:
                        continue
                else:
                    subject_type = subject_type[0]

                if subject_type == 'SUBJECT_UNIT':
                    continue
                id_nodetype_map[uuid] = subject_type
                if subject_type not in node_type_dict:
                    node_type_dict[subject_type] = len(node_type_dict)

                if 'FILE' in subject_type:
                    filename = None
                    try:
                        obj = json.loads(line)
                        fileobj = obj["datum"]["com.bbn.tc.schema.avro.cdm18.FileObject"]
                        base  = fileobj.get("baseObject", {})
                        props = base.get("properties", {}).get("map", {})
                        if "filename" in base:
                            filename = base["filename"]
                        elif "path" in base:
                            filename = base["path"]
                        if "filename" in props:
                            filename = props["filename"]
                        elif "path" in props:
                            filename = props["path"]
                    except Exception:
                        pass
                    if filename and filename != "null":
                        id_nodename_map[uuid] = filename

                elif subject_type == 'SUBJECT_PROCESS':
                    cmd = None
                    exe = None
                    try:
                        obj = json.loads(line)
                        subj_obj = obj["datum"]["com.bbn.tc.schema.avro.cdm18.Subject"]
                      
                        cmd_raw = subj_obj.get("cmdLine")
                        if isinstance(cmd_raw, str):
                            cmd = cmd_raw
                        elif isinstance(cmd_raw, dict):
                            cmd = cmd_raw.get("string")
                        props = subj_obj.get("properties", {}).get("map", {})
                        if "path" in props:
                            exe = props["path"]
                    except Exception:
                        pass
                    if cmd:
                        id_cmdline_map[uuid] = cmd
                    if exe:
                        id_exepath_map[uuid] = exe
                    if cmd:
                        id_nodename_map[uuid] = cmd
                    elif exe:
                        id_nodename_map[uuid] = exe
                
                elif subject_type == 'NetFlowObject':
                    srcaddr = "null"
                    srcport = "null"
                    dstaddr = "null"
                    dstport = "null"
                    try:
                        obj = json.loads(line)
                        netobj = obj["datum"]["com.bbn.tc.schema.avro.cdm18.NetFlowObject"]
                        srcaddr = netobj.get("localAddress",  "null")
                        srcport = netobj.get("localPort",     "null")
                        dstaddr = netobj.get("remoteAddress", "null")
                        dstport = netobj.get("remotePort",    "null")
                        if isinstance(srcaddr, dict):
                            srcaddr = srcaddr.get("string", "null")
                        if isinstance(dstaddr, dict):
                            dstaddr = dstaddr.get("string", "null")
                        if isinstance(srcport, dict):
                            srcport = str(srcport.get("int", "null"))
                        if isinstance(dstport, dict):
                            dstport = str(dstport.get("int", "null"))
                    except Exception:
                        pass
                    nodeproperty = f"{str(srcaddr)}_{str(srcport)}_{str(dstaddr)}_{str(dstport)}"
                    id_nodename_map[uuid] = nodeproperty
                    if dstport and dstport != "null":
                        id_remoteport_map[uuid] = dstport
                    if srcaddr and srcaddr != "null":
                        id_localaddr_map[uuid] = srcaddr
                    if srcport and srcport != "null":
                        id_localport_map[uuid] = srcport

    print('  Entities extracted : {:,}'.format(len(id_nodetype_map)))

    print()
    print('  ── Pass 2 / 2 : event (edge) extraction ──')
    total_events_skipped = 0
    skip_type_counts     = {}
    edges_files          = []   

    for filepath in json_files:
        basename   = os.path.basename(filepath)
        import re as _re
        tag        = _re.sub(r'\.json(\.\d+)?$', '', basename)
        tag        = tag.replace(' ', '_')
        skip_count = 0
        file_edges = []

        print('  {}'.format(basename))
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in tqdm(f, desc='  PASS 2 {}'.format(tag[:30])):
                if '\x00' in line:
                    skip_count += 1
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.Event' not in line:
                    continue

                etype_match = pattern_type.findall(line)
                ts_match    = pattern_time.findall(line)
                if not etype_match or not ts_match:
                    continue

                edgeType  = etype_match[0]
                try:
                    timestamp = int(ts_match[0].strip())
                except ValueError:
                    skip_count += 1
                    continue

                if edgeType in {'EVENT_MPROTECT', 'EVENT_MMAP', 'EVENT_SHM'}:
                    skip_type_counts[edgeType] = skip_type_counts.get(edgeType, 0) + 1
                    skip_count += 1
                    continue

                if edgeType not in edge_type_dict:
                    edge_type_dict[edgeType] = len(edge_type_dict)

                srcId_match = pattern_src.findall(line)
                if not srcId_match:
                    continue
                srcId = srcId_match[0]
                if srcId not in id_nodetype_map:
                    src_name = id_nodename_map.get(srcId, None)
                    if src_name and src_name not in ('<unknown>', 'null', 'unknown'):
                        id_nodetype_map[srcId] = 'SUBJECT_PROCESS'
                    else:
                        continue
                srcType = id_nodetype_map[srcId]
                update_ts(id_ts_map, srcId, timestamp)

                for dst_pattern in (pattern_dst1, pattern_dst2):
                    dstId_match = dst_pattern.findall(line)
                    if dstId_match and dstId_match[0] != 'null':
                        d = dstId_match[0]
                        if d == '00000000-0000-0000-0000-000000000000':
                            continue
                        dType = id_nodetype_map.get(d, 'UNKNOWN')
                        update_ts(id_ts_map, d, timestamp)
                        file_edges.append((timestamp, srcId, srcType,
                                           d, dType, edgeType))

        total_events_skipped += skip_count

        
        file_edges.sort(key=lambda x: x[0])
        edges_out = os.path.join(OUTPUT_PARSED, 'edges_{}.txt'.format(tag))
        with open(edges_out, 'w', encoding='utf-8') as fw:
            for ts, srcId, srcType, d, dType, edgeType in file_edges:
                fw.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    srcId, srcType, d, dType, edgeType, ts))
        edges_files.append((tag, edges_out, len(file_edges)))
        print('  Total edges written : {:,}  →  {}'.format(len(file_edges), edges_out))

    total_edges_written = sum(n for _, _, n in edges_files)
    print()
    print('  ── Pass 2 totals ──')
    print('  Files processed     : {:,}'.format(len(edges_files)))
    print('  Total edges written : {:,}'.format(total_edges_written))

    ts_covered = sum(1 for uid in id_nodetype_map if uid in id_ts_map)
    print('  Nodes with timestamps : {:,} / {:,}  ({:.1f}%)'.format(
        ts_covered, len(id_nodetype_map),
        100.0 * ts_covered / len(id_nodetype_map) if id_nodetype_map else 0))

    id_names_full_map = {
        uid: name
        for uid, name in id_nodename_map.items()
        if name
    }
    with open(OUTPUT_PARSED + 'names.json', 'w', encoding='utf-8') as f:
        json.dump(id_names_full_map, f, indent=2)
    print('  names.json saved         : {:,} entries'.format(len(id_names_full_map)))

    with open(OUTPUT_PARSED + 'types.json', 'w', encoding='utf-8') as f:
        json.dump(id_nodetype_map, f, indent=2)
    print('  types.json saved         : {:,} entries'.format(len(id_nodetype_map)))

    with open(OUTPUT_PARSED + 'node_type_map.json', 'w', encoding='utf-8') as f:
        json.dump(node_type_dict, f, indent=2)
    print('  node_type_map.json saved : {:,} types'.format(len(node_type_dict)))

    with open(OUTPUT_PARSED + 'edge_type_map.json', 'w', encoding='utf-8') as f:
        json.dump(edge_type_dict, f, indent=2)
    print('  edge_type_map.json saved : {:,} types'.format(len(edge_type_dict)))

   
    combined_edges_path = os.path.join(OUTPUT_PARSED, 'edges_all.txt')
    total_combined = 0
    with open(combined_edges_path, 'w', encoding='utf-8') as fw:
        for _, ef_path, _ in edges_files:
            with open(ef_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    fw.write(line)
                    total_combined += 1
    print('  Combined edges file : {:,} edges  →  {}'.format(total_combined, combined_edges_path))

    maps = {
        'id_nodetype_map'   : id_nodetype_map,
        'id_nodename_map'   : id_nodename_map,
        'id_cmdline_map'    : id_cmdline_map,
        'id_exepath_map'    : id_exepath_map,
        'id_remoteport_map' : id_remoteport_map,
        'id_localaddr_map'  : id_localaddr_map,
        'id_localport_map'  : id_localport_map,
        'id_memaddr_map'    : id_memaddr_map,
        'id_ts_map'         : id_ts_map,
        'edges_files'       : edges_files,          
        'edges_all'         : combined_edges_path,  
    }

    print()
    show('parse_provenance.py — DONE')
    return maps

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', type=str, default=None,
                    help='Parse only this file (filename or full path). '
                         'If omitted, parses all JSON files in input/theia/')
    args = ap.parse_args()
    run_parser(file=args.file)