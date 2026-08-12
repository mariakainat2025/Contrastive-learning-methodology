
import os
import sys
import json
import glob
from collections import Counter
from networkx.readwrite import json_graph

from abstract_file_paths import (
    lift_etc_var, lift_etc_var_root, lift_bin, lift_home, lift_root, lift_lib, lift_other,
)

PROJECT_ROOT = '/csse/research/contructive-learning'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from scripts.reduce_graph import reduce_directory_cascade

CAM_LDS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR  = os.path.join(CAM_LDS_DIR, "graphs")
OUT_PATH    = os.path.join(CAM_LDS_DIR, "results", "unresolved_file_paths.json")


def _lift(raw_path):
    try:
        return (lift_etc_var(raw_path) or lift_etc_var_root(raw_path) or lift_bin(raw_path)
                or lift_home(raw_path) or lift_root(raw_path) or lift_lib(raw_path)
                or lift_other(raw_path))
    except Exception:
        return None


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


def main():
    files = sorted(glob.glob(os.path.join(GRAPHS_DIR, "*", "*", "graph_*.json")))
    print(f"Found {len(files)} graph JSON files under {GRAPHS_DIR}")

    total_file_nodes = 0
    unresolved = Counter()
    total_cascades = 0

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        G = json_graph.node_link_graph(data)
        G, combine, removed_nodes = reduce_directory_cascade(G)
        _strip_cascade_edges(G, removed_nodes)
        total_cascades += combine

        for n, attrs in G.nodes(data=True):
            if attrs.get("type") == "FILE":
                total_file_nodes += 1
                name = attrs["name"]
                lifted = _lift(name)
                if lifted is None or lifted == name:
                    unresolved[name] += 1

    report = {
        "total_file_nodes": total_file_nodes,
        "total_unresolved_occurrences": sum(unresolved.values()),
        "total_unique_unresolved_paths": len(unresolved),
        "paths": [p for p, _ in sorted(unresolved.items(), key=lambda kv: -kv[1])],
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"Total cascades collapsed     : {total_cascades}")
    print(f"Total FILE nodes (post-cascade) : {total_file_nodes}")
    print(f"Unresolved occurrences       : {report['total_unresolved_occurrences']}")
    print(f"Unique unresolved paths      : {report['total_unique_unresolved_paths']}")
    print()
    print("Top 30 by frequency:")
    for p, c in unresolved.most_common(30):
        print(f"  {c:6d}  {p}")
    print()
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

