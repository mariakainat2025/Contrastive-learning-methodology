import os
import re
import sys
import json
import networkx as nx
from pathlib import Path
from networkx.readwrite import json_graph

DESC_RE = re.compile(r"step=(\S+)\s+host=(\S+)")


def subgraph_json_to_nx(g):
    description = g.get("description", "")
    m = DESC_RE.search(description)
    step, host = (m.group(1), m.group(2)) if m else (None, None)

    G = nx.MultiDiGraph(
        tactic=g.get("tactic"),
        attack=g.get("attack"),
        technique=g.get("attack"),
        step=step,
        host=host,
        description=description,
    )
    for nid, attrs in g["nodes"]:
        ntype = "FILE" if attrs["type"] == "FileObject" else attrs["type"]
        G.add_node(attrs["uuid"], type=ntype, name=attrs["name"], ts=attrs["ts"])
    for src, dst, w, attrs in g["edges"]:
        G.add_edge(src, dst, **attrs)
    return G


def batch_main(subgraphs_dir, out_dir):
    files = sorted(Path(subgraphs_dir).glob("*/*/subgraph_*.json"))
    print(f"Found {len(files)} subgraph JSON files under {subgraphs_dir}")

    n_ok = 0
    for fpath in files:
        with open(fpath) as f:
            g = json.load(f)
        G = subgraph_json_to_nx(g)

        rel = fpath.relative_to(subgraphs_dir)
        out_subdir = Path(out_dir) / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_name = fpath.name.replace("subgraph_", "graph_", 1)

        data = json_graph.node_link_data(G)
        with open(out_subdir / out_name, "w") as f:
            json.dump(data, f, indent=2)

        n_ok += 1

    print()
    print(f"Built {n_ok} graphs -> {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraphs-dir", default="/csse/research/contructive-learning/CAM-LDS/parser/subgraphs")
    ap.add_argument("--out-dir", default="/csse/research/contructive-learning/CAM-LDS/graphs")
    args = ap.parse_args()
    batch_main(args.subgraphs_dir, args.out_dir)

