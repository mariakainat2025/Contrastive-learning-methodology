
import os
import sys
import json
import glob
import collections

CAM_LDS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_DIR   = os.path.join(CAM_LDS_DIR, "parser", "subgraphs")
OUT_PATH    = os.path.join(CAM_LDS_DIR, "parser", "file_paths.json")


def main():
    files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*", "*", "subgraph_*.json")))
    print(f"Found {len(files)} subgraph JSON files under {GRAPH_DIR}")

    path_counts = collections.Counter()

    for fpath in files:
        with open(fpath) as f:
            g = json.load(f)

        node_type = {n[1]["uuid"]: n[1]["type"] for n in g.get("nodes", [])}
        node_name = {n[1]["uuid"]: n[1]["name"] for n in g.get("nodes", [])}

        for src, dst, _, attrs in g.get("edges", []):
            if node_type.get(dst) == "FileObject":
                name = node_name.get(dst)
                if name:
                    path_counts[name] += 1

    out = {
        "total_file_paths": sum(path_counts.values()),
        "unique_file_paths": len(path_counts),
        "paths": sorted(path_counts.keys()),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Total file paths  : {out['total_file_paths']}")
    print(f"Unique file paths : {out['unique_file_paths']}")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

