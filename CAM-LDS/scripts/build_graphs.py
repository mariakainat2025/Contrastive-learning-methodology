
import os
import sys
import json
import glob
import uuid as uuid_lib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_audit import build_graph

CAM_LDS_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED_EVENTS_DIR  = os.path.join(CAM_LDS_DIR, "parser", "parsed_events")
GRAPH_OUT_DIR       = os.path.join(CAM_LDS_DIR, "graph")


def infer_node_type(name, roles):
    if name.startswith(("pam:", "acct:")):
        return "PRINCIPAL_LOCAL"
    if name.startswith(("unit:", "bpf:", "pid:")):
        return "SUBJECT_PROCESS"
    if any(side == "src" for side, sc in roles):
        return "SUBJECT_PROCESS"
    if "/" in name:
        return "FileObject"
    if ":" in name and name.rsplit(":", 1)[-1].isdigit():
        return "NetFlowObject"
    return "SUBJECT_PROCESS"


def graph_to_theia_format(G, events, tactic, technique, step, host):
    eid_to_ts = {}
    for e in events:
        if e.get("timestamp"):
            eid_to_ts[e["event_id"]] = int(float(e["timestamp"]) * 1e9)

    def edge_ts(eids_str):
        first_ids = [x.split("+")[0] for x in eids_str.split(",") if x]
        tss = [eid_to_ts[i] for i in first_ids if i in eid_to_ts]
        return min(tss) if tss else 0

    node_uuid  = {}
    node_ts    = {}
    node_roles = {}

    for u, v, k, d in G.edges(keys=True, data=True):
        ts = edge_ts(d.get("eids", ""))
        sc = d.get("syscall", "")
        node_roles.setdefault(u, set()).add(("src", sc))
        node_roles.setdefault(v, set()).add(("dst", sc))
        if ts:
            node_ts[u] = min(node_ts.get(u, ts), ts)
            node_ts[v] = min(node_ts.get(v, ts), ts)

    nodes = []
    for nid, name in enumerate(G.nodes()):
        nuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, f"{tactic}/{technique}/{step}/{host}/{name}"))
        node_uuid[name] = nuid
        ntype = infer_node_type(name, node_roles.get(name, set()))
        nodes.append([nid, {
            "type": ntype,
            "ts"  : node_ts.get(name, 0),
            "name": name,
            "uuid": nuid,
        }])

    raw_edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        raw_edges.append((edge_ts(d.get("eids", "")), u, v, k, d))
    raw_edges.sort(key=lambda x: x[0])

    edges    = []
    triplets = []
    for ts, u, v, k, d in raw_edges:
        edge_type = d.get("syscall", "").upper()
        edges.append([node_uuid[u], node_uuid[v], k, {
            "edge_type": edge_type,
            "ts"       : ts,
            "count"    : d.get("count", 1),
            "event_ids": d.get("eids", ""),
        }])
        triplets.append([u, edge_type, v])

    ts_values = [n[1]["ts"] for n in nodes if n[1]["ts"]]
    return {
        "tactic"    : tactic,
        "technique" : technique,
        "step"      : step,
        "host"      : host,
        "start_ts"  : min(ts_values) if ts_values else 0,
        "end_ts"    : max(ts_values) if ts_values else 0,
        "n_nodes"   : len(nodes),
        "n_edges"   : len(edges),
        "nodes"     : nodes,
        "edges"     : edges,
        "triplets"  : triplets,
    }


def main():
    files = sorted(glob.glob(os.path.join(PARSED_EVENTS_DIR, "*", "*", "audit_*.json")))
    print(f"Found {len(files)} parsed-event JSON files under {PARSED_EVENTS_DIR}")

    n_ok, n_empty = 0, 0
    for fpath in files:
        technique = os.path.basename(os.path.dirname(fpath))
        tactic    = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
        base      = os.path.basename(fpath)[len("audit_"):-len(".json")]
        step, host = base.rsplit("_", 1)
        fname     = f"graph_{base}.json"

        with open(fpath) as f:
            events = json.load(f)

        G = build_graph(events)
        subgraph = graph_to_theia_format(G, events, tactic, technique, step, host)

        out_dir = os.path.join(GRAPH_OUT_DIR, tactic, technique)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(subgraph, f, indent=2)

        if subgraph["n_nodes"] == 0:
            n_empty += 1
        else:
            n_ok += 1
        print(f"[{tactic:20s}] {technique:12s} {base:45s} "
              f"nodes={subgraph['n_nodes']:4d} edges={subgraph['n_edges']:4d} -> {out_path}")

    print()
    print(f"Built {n_ok} graphs ({n_empty} empty) -> {GRAPH_OUT_DIR}")


if __name__ == "__main__":
    main()

