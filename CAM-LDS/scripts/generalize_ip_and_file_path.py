import os
import re
import sys
import json
from pathlib import Path
from collections import Counter
from networkx.readwrite import json_graph

from ip_classifier import classify_ip
from abstract_file_paths import (
    lift_etc_var, lift_etc_var_root, lift_bin, lift_home, lift_root, lift_lib, lift_other,
)
import build_sequences

PROJECT_ROOT = '/csse/research/contructive-learning'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from scripts.reduce_graph import reduce_directory_cascade

CAM_LDS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(CAM_LDS_DIR, "graphs")
OUT_DIR = os.path.join(CAM_LDS_DIR, "generalize_graph")
SUMMARY_PATH = os.path.join(OUT_DIR, "reduce_summary.json")
UNCHANGED_PATHS_PATH = os.path.join(OUT_DIR, "unchanged_file_paths.json")

IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")

EVENT_PREFIX = "EVENT_"

def _is_ipv4(s):
    m = IPV4_RE.match(s)
    return bool(m) and all(0 <= int(o) <= 255 for o in m.groups())


def _generalize_name(name):
    if _is_ipv4(name):
        return classify_ip(name), name

    if ":" in name:
        ip_part, _, _port_part = name.rpartition(":")
        if _is_ipv4(ip_part):
            return classify_ip(ip_part), ip_part

    return name, None


def _lift_file_path(raw_path):
    try:
        return (lift_etc_var(raw_path) or lift_etc_var_root(raw_path) or lift_bin(raw_path)
                or lift_home(raw_path) or lift_root(raw_path) or lift_lib(raw_path)
                or lift_other(raw_path))
    except Exception:
        return None


def _edge_key(u, v, attrs):
    return (u, v, attrs.get("edge_type"), attrs.get("event_id"))


def _strip_cascade_edges(G, removed_nodes):
    for item in removed_nodes:
        parent = item["node"]
        for p in item["processes"]:
            proc = p["node"]
            for u, v, k in list(G.edges(keys=True)):
                if (u == proc and v == parent) or (u == parent and v == proc):
                    G.remove_edge(u, v, k)
        if parent in G and G.degree(parent) == 0:
            G.remove_node(parent)


def batch_main(graphs_dir=GRAPHS_DIR, out_dir=OUT_DIR, summary_path=SUMMARY_PATH,
                unchanged_paths_path=UNCHANGED_PATHS_PATH):
    files = sorted(Path(graphs_dir).glob("*/*/graph_*.json"))
    print(f"Found {len(files)} graph JSON files under {graphs_dir}")

    total_generalized_nodes = 0
    total_internal = 0
    total_external = 0
    unique_ips = set()

    total_lifted_paths = 0
    unique_paths = set()
    unchanged_paths = Counter()

    total_merged_graphs = 0
    total_cascades = 0
    all_merges = []

    total_events_generalized = 0
    event_verb_counts = Counter()

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        G = json_graph.node_link_graph(data)

        for _, attrs in G.nodes(data=True):
            new_name, ip_found = _generalize_name(attrs["name"])
            if ip_found is not None:
                attrs["name"] = new_name
                unique_ips.add(ip_found)
                total_generalized_nodes += 1
                if "internal" in new_name:
                    total_internal += 1
                else:
                    total_external += 1

        edges_before = {_edge_key(u, v, a): (u, v, a) for u, v, a in G.edges(data=True)}
        G, combine, removed_nodes = reduce_directory_cascade(G)
        node_name = {n: d.get("name", n) for n, d in G.nodes(data=True)}
        _strip_cascade_edges(G, removed_nodes)
        edges_after = {_edge_key(u, v, a) for u, v, a in G.edges(data=True)}
        removed_edges_all = [edges_before[k] for k in edges_before if k not in edges_after]

        if combine > 0:
            total_merged_graphs += 1
        total_cascades += combine

        for item in removed_nodes:
            parent = item["node"]
            child = item["kept_node"]
            edges_removed = [
                {"src": node_name.get(u, u), "dst": node_name.get(v, v),
                 "edge_type": a.get("edge_type"), "event_id": a.get("event_id")}
                for u, v, a in removed_edges_all if u == parent or v == parent
            ]
            all_merges.append({
                "tactic": G.graph.get("tactic"),
                "technique": G.graph.get("technique"),
                "step": G.graph.get("step"),
                "host": G.graph.get("host"),
                "removed_path": item.get("name", parent),
                "kept_path": item.get("kept_name", child),
                "processes": [p["name"] for p in item["processes"]],
                "edges_removed": edges_removed,
            })

        for _, attrs in G.nodes(data=True):
            if attrs.get("type") == "FILE":
                raw_name = attrs["name"]
                lifted = _lift_file_path(raw_name)
                if lifted is not None and lifted != raw_name:
                    unique_paths.add(raw_name)
                    total_lifted_paths += 1
                    attrs["name"] = lifted
                else:
                    unchanged_paths[raw_name] += 1

        for _, _, attrs in G.edges(data=True):
            raw_event = attrs.get("edge_type")
            if raw_event and raw_event.startswith(EVENT_PREFIX):
                attrs["edge_type"] = raw_event[len(EVENT_PREFIX):]
                event_verb_counts[raw_event] += 1
                total_events_generalized += 1

        rel = fpath.relative_to(graphs_dir)
        out_subdir = Path(out_dir) / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_name = fpath.name.replace("graph_", "generalize_", 1)
        with open(out_subdir / out_name, "w") as f:
            json.dump(json_graph.node_link_data(G), f, indent=2)

    summary = {
        "total_graphs": len(files),
        "total_merged_graphs": total_merged_graphs,
        "total_cascades": total_cascades,
        "merges": all_merges,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    unchanged_report = {
        "total_unchanged_occurrences": sum(unchanged_paths.values()),
        "total_unique_unchanged_paths": len(unchanged_paths),
        "paths": [p for p, _ in sorted(unchanged_paths.items(), key=lambda kv: -kv[1])],
    }
    with open(unchanged_paths_path, "w") as f:
        json.dump(unchanged_report, f, indent=2)

    print()
    print(f"Graphs processed        : {len(files)}")
    print(f"Nodes generalized (IP)  : {total_generalized_nodes}")
    print(f"  -> internal           : {total_internal}")
    print(f"  -> external           : {total_external}")
    print(f"Unique IPs generalized  : {len(unique_ips)}")
    print()
    print(f"Graphs with cascades    : {total_merged_graphs}")
    print(f"Total cascades          : {total_cascades}")
    print()
    print(f"Nodes lifted (paths)    : {total_lifted_paths}")
    print(f"Unique paths lifted     : {len(unique_paths)}")
    print()
    print(f"Unchanged path nodes    : {sum(unchanged_paths.values())}")
    print(f"Unique unchanged paths  : {len(unchanged_paths)}")
    print()
    print(f"Events generalized      : {total_events_generalized}")
    for raw_event, n in event_verb_counts.most_common():
        print(f"  {raw_event:22s} -> {raw_event[len(EVENT_PREFIX):]:28s} ({n})")
    print()
    print(f"Saved graphs -> {out_dir}")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved unchanged paths -> {unchanged_paths_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-dir", default=GRAPHS_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--summary-path", default=SUMMARY_PATH)
    args = ap.parse_args()
    batch_main(args.graphs_dir, args.out_dir, args.summary_path)

    print()
    print("Building sequences from the generalized graphs just written...")
    build_sequences.main(graph_dir=args.out_dir)

