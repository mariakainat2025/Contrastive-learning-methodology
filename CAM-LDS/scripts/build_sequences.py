import os
import re
import json
import glob
from networkx.readwrite import json_graph

from parse_audit import RARE_EDGE_TYPES

CAM_LDS_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_DIR     = os.path.join(CAM_LDS_DIR, "generalize_graph")
SEQUENCES_DIR = os.path.join(CAM_LDS_DIR, "sequences")

# RARE_EDGE_TYPES (parse_audit.py) is the same list zoomer/scripts/
# Feature_Initialization.py excludes from its edge-type vocabulary
# (MASTER_EDGE_TYPES). It's defined there with the raw "EVENT_" prefix
# (e.g. "EVENT_ADD_USER"), but generalize_graph/ -- what this script reads
# -- already has that prefix stripped (see generalize_ip_and_file_path.py),
# so the comparison set here is stripped the same way.
_EVENT_PREFIX = "EVENT_"
RARE_EDGE_TYPES_STRIPPED = {
    t[len(_EVENT_PREFIX):] if t.startswith(_EVENT_PREFIX) else t
    for t in RARE_EDGE_TYPES
}

TACTIC_FOLDER_TO_LABEL = {
    "collection":            "Collection",
    "command_and_control":   "Command_and_Control",
    "credential_access":     "Credential_Access",
    "defense_impairment":    "Defense_Impairment",
    "discovery":             "Discovery",
    "execution":             "Execution",
    "exfiltration":          "Exfiltration",
    "impact":                "Impact",
    "initial_access":        "Initial_Access",
    "lateral_movement":      "Lateral_Movement",
    "persistence":           "Persistence",
    "privilege_escalation":  "Privilege_Escalation",
    "reconnaissance":        "Reconnaissance",
    "stealth":               "Stealth",
}

DESC_RE = re.compile(r"step=(\S+)\s+host=(\S+)")


def triplet_to_sentence(src, edge_type, dst):
    return f"{src} {edge_type.lower()} {dst}."


def build_one(graph_path):
    with open(graph_path) as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data)

    node_name = {n: d["name"] for n, d in G.nodes(data=True)}
    sentences = [
        triplet_to_sentence(node_name[src], attrs["edge_type"], node_name[dst])
        for src, dst, attrs in G.edges(data=True)
        if attrs["edge_type"] not in RARE_EDGE_TYPES_STRIPPED
    ]

    m = DESC_RE.search(G.graph.get("description", ""))
    step, host = (m.group(1), m.group(2)) if m else ("?", "?")

    return {
        "tactic"   : TACTIC_FOLDER_TO_LABEL[G.graph["tactic"]],
        "technique": G.graph["attack"],
        "step"     : step,
        "host"     : host,
        "n_triples": len(sentences),
        "sequence" : sentences,
    }


def main(graph_dir=GRAPH_DIR, prefix="generalize_", sequences_dir=SEQUENCES_DIR):
    files = sorted(glob.glob(os.path.join(graph_dir, "*", "*", f"{prefix}*.json")))
    print(f"Found {len(files)} graph JSON files under {graph_dir}")

    n_ok, n_empty = 0, 0
    for fpath in files:
        technique = os.path.basename(os.path.dirname(fpath))
        tactic    = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
        fname     = os.path.basename(fpath).replace(prefix, "sequence_", 1)

        seq = build_one(fpath)

        out_dir = os.path.join(sequences_dir, tactic, technique)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(seq, f, indent=2)

        if seq["n_triples"] == 0:
            n_empty += 1
        else:
            n_ok += 1
    print()
    print(f"Built {n_ok} sequences ({n_empty} empty) -> {sequences_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default=GRAPH_DIR)
    ap.add_argument("--prefix", default="generalize_",
                     help='File prefix to glob for, e.g. "generalize_" (default) or "graph_" for raw graphs/.')
    ap.add_argument("--sequences-dir", default=SEQUENCES_DIR,
                     help='Output directory for built sequences (default: sequences/). Use a different '
                          'directory (e.g. sequences_raw/) to avoid overwriting an existing sequences/ '
                          'folder built from different graphs (e.g. generalized vs raw).')
    args = ap.parse_args()
    main(graph_dir=args.graph_dir, prefix=args.prefix, sequences_dir=args.sequences_dir)

