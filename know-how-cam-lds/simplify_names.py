"""simplify_names.py

Takes a *_readable.txt edges file (produced by parse_attack_events.py:
srcId  srcName  srcType  dstId  dstName  dstType  edgeType  timestamp  label)
and adds a simplified version of each FILE and NetFlowObject name, reusing the
knowhow pipeline's own normalization rules instead of re-deriving them:

  - FILE_OBJECT_* names  -> sub_filepath() from CAM-LDS/knowhow (Table II of
    the knowhow paper: collapses a path like
    /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.22 into "x86_64-linux-gnu
    library file").
  - NetFlowObject names  -> _is_internal_ip(), ported from sean_1's legacy
    benigntag_paral_gpu.py. Our NetFlow node names are built by
    parse_provenance.py as "<localAddr>_<localPort>_<remoteAddr>_<remotePort>",
    so each address half is classified separately as an internal (RFC1918) or
    external network address.
  - SUBJECT_PROCESS names are left as-is -- they're executable paths/cmdlines,
    not something either rule applies to.

Output adds two columns (srcNameSimplified, dstNameSimplified) rather than
replacing the originals, so the raw and simplified forms can be compared side
by side.
"""

import os
import sys
import argparse
from pathlib import PurePosixPath

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWHOW_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'knowhow', 'KNOWHOW-seng402')
if KNOWHOW_DIR not in sys.path:
    sys.path.insert(0, KNOWHOW_DIR)

from knowhow.tools import sub_filepath  # noqa: E402


def _is_internal_ip(ip):
    if ip in ("0.0.0.0",) or ip.startswith("10.") or ip.startswith("127.") \
            or ip.startswith("169.254.") or ip.startswith("192.168."):
        return True
    if ip.startswith("172."):
        parts = ip.split(".")
        if len(parts) > 1 and parts[1].isdigit() and 16 <= int(parts[1]) <= 31:
            return True
    return False


def simplify_netflow_name(name):
    parts = name.split("_")
    if len(parts) != 4:
        return name
    local_addr, _local_port, remote_addr, _remote_port = parts
    local_kind  = "internal" if _is_internal_ip(local_addr)  else "external"
    remote_kind = "internal" if _is_internal_ip(remote_addr) else "external"
    return f"{local_kind} network address to {remote_kind} network address"


def simplify_name(name, node_type):
    if not name or name == 'N/A':
        return name
    if 'FILE' in node_type:
        try:
            return sub_filepath(PurePosixPath(name))
        except Exception:
            return name
    if node_type == 'NetFlowObject':
        return simplify_netflow_name(name)
    return name


def run(input_file, output_file=None):
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        base = base[:-len('_readable')] if base.endswith('_readable') else base
        output_file = base + '_simplified' + ext

    n = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            fields = line.rstrip('\n').split('\t')
            srcId, srcName, srcType, dstId, dstName, dstType, edgeType, ts, label = fields
            src_simplified = simplify_name(srcName, srcType)
            dst_simplified = simplify_name(dstName, dstType)
            fout.write('\t'.join([
                srcId, srcName, src_simplified, srcType,
                dstId, dstName, dst_simplified, dstType,
                edgeType, ts, label,
            ]) + '\n')
            n += 1

    print(f'  Rows written: {n:,}  -> {output_file}')
    return output_file


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='*_readable.txt file to simplify')
    ap.add_argument('--output', default=None, help='Output path (default: <file>_simplified.txt)')
    args = ap.parse_args()
    run(args.file, args.output)
