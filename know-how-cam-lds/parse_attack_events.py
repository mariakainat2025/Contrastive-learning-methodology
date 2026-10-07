"""parse_attack_events.py

Converts a raw, event-only CDM18 attack log (no Subject/FileObject/NetFlowObject
entity records of its own -- just Event lines, each carrying a ground-truth
"label":"benign"/"malicious" field) into two tab-separated files:

  edges_<tag>.txt            srcId  srcType  dstId  dstType  edgeType  timestamp
                             (lean format, feeds straight into
                              knowhow's benigntag_paral_gpu.py)

  edges_<tag>_readable.txt   srcId  srcName  srcType  dstId  dstName  dstType
                             edgeType  timestamp  label
                             (human-readable, for manual inspection)

Because the attack file has no entity records, srcType/dstType/srcName/dstName
can't be discovered from the file itself -- they're looked up from the
existing types.json / names.json built from the full ta1-theia-e3-official-1r1
parse (the attack window's UUIDs come from the same host/dataset, so they're
already in there).

Also writes a sidecar labels file (same row order as the edges files) with
just the per-event ground-truth label, since benigntag_paral_gpu.py's scored
output doesn't carry it through -- needed later to evaluate detections
against the threshold.
"""

import os
import re
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import (
    OUTPUT_PARSED,
    pattern_type, pattern_time,
    pattern_src, pattern_dst1, pattern_dst2,
)

pattern_label = re.compile(r'"label":"(.*?)"')


def run(input_file, output_dir, types_path=None, names_path=None, tag=None):
    types_path = types_path or os.path.join(OUTPUT_PARSED, 'types.json')
    names_path = names_path or os.path.join(OUTPUT_PARSED, 'names.json')

    print('Loading existing node-type map: {}'.format(types_path))
    with open(types_path, 'r', encoding='utf-8') as f:
        id_nodetype_map = json.load(f)
    print('  {:,} known node types'.format(len(id_nodetype_map)))

    print('Loading existing name map: {}'.format(names_path))
    with open(names_path, 'r', encoding='utf-8') as f:
        id_nodename_map = json.load(f)
    print('  {:,} known node names'.format(len(id_nodename_map)))

    if tag is None:
        tag = re.sub(r'\.json(\.\d+)?$', '', os.path.basename(input_file))

    os.makedirs(output_dir, exist_ok=True)
    edges_out    = os.path.join(output_dir, 'edges_{}.txt'.format(tag))
    readable_out = os.path.join(output_dir, 'edges_{}_readable.txt'.format(tag))
    labels_out   = os.path.join(output_dir, 'edges_{}_labels.txt'.format(tag))

    n_lines = n_events = n_edges = n_unknown_type = n_unknown_name = 0

    with open(input_file, 'r', encoding='utf-8', errors='replace') as fin, \
         open(edges_out, 'w', encoding='utf-8') as fedges, \
         open(readable_out, 'w', encoding='utf-8') as freadable, \
         open(labels_out, 'w', encoding='utf-8') as flabels:

        for line in fin:
            n_lines += 1
            if 'com.bbn.tc.schema.avro.cdm18.Event' not in line:
                continue

            etype_match = pattern_type.findall(line)
            ts_match    = pattern_time.findall(line)
            if not etype_match or not ts_match:
                continue
            edgeType = etype_match[0]
            try:
                timestamp = int(ts_match[0].strip())
            except ValueError:
                continue

            if edgeType in {'EVENT_MPROTECT', 'EVENT_MMAP', 'EVENT_SHM'}:
                continue

            n_events += 1

            label_match = pattern_label.findall(line)
            label = label_match[0] if label_match else 'unknown'

            srcId_match = pattern_src.findall(line)
            if not srcId_match:
                continue
            srcId = srcId_match[0]
            srcType = id_nodetype_map.get(srcId)
            if srcType is None:
                n_unknown_type += 1
                srcType = 'UNKNOWN'
            srcName = id_nodename_map.get(srcId)
            if srcName is None:
                n_unknown_name += 1
                srcName = 'N/A'

            for dst_pattern in (pattern_dst1, pattern_dst2):
                dstId_match = dst_pattern.findall(line)
                if dstId_match and dstId_match[0] != 'null':
                    d = dstId_match[0]
                    if d == '00000000-0000-0000-0000-000000000000':
                        continue
                    dType = id_nodetype_map.get(d, 'UNKNOWN')
                    dName = id_nodename_map.get(d, 'N/A')
                    fedges.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        srcId, srcType, d, dType, edgeType, timestamp))
                    freadable.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        srcId, srcName, srcType, d, dName, dType, edgeType, timestamp, label))
                    flabels.write(label + '\n')
                    n_edges += 1

    print()
    print('  Lines read               : {:,}'.format(n_lines))
    print('  Events matched           : {:,}'.format(n_events))
    print('  Edges written            : {:,}  -> {}'.format(n_edges, edges_out))
    print('  Readable edges written   : {:,}  -> {}'.format(n_edges, readable_out))
    print('  Labels written           : {:,}  -> {}'.format(n_edges, labels_out))
    print('  Src UUIDs w/o known type : {:,}'.format(n_unknown_type))
    print('  Src UUIDs w/o known name : {:,}'.format(n_unknown_name))
    return edges_out, readable_out, labels_out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True,
                     help='Raw attack CDM18 json file (absolute or relative path)')
    ap.add_argument('--output-dir', default=None,
                     help='Where to write edges_<tag>.txt / edges_<tag>_labels.txt '
                          '(default: same directory as --file)')
    ap.add_argument('--types', default=None,
                     help='Path to existing types.json (default: output/theia/parsed/types.json)')
    ap.add_argument('--names', default=None,
                     help='Path to existing names.json (default: output/theia/parsed/names.json)')
    ap.add_argument('--tag', default=None,
                     help='Override the output tag (default: input filename without .json)')
    args = ap.parse_args()

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.file))
    run(args.file, out_dir, types_path=args.types, names_path=args.names, tag=args.tag)
