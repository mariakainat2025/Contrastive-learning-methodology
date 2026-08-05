import os
import re
import json
import glob

CAM_LDS_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_DIR     = os.path.join(CAM_LDS_DIR, "parser", "subgraphs")
SEQUENCES_DIR = os.path.join(CAM_LDS_DIR, "sequences")

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
        g = json.load(f)

    node_name = {n[1]["uuid"]: n[1]["name"] for n in g["nodes"]}
    sentences = [
        triplet_to_sentence(node_name[src], attrs["edge_type"], node_name[dst])
        for src, dst, _, attrs in g["edges"]
    ]

    m = DESC_RE.search(g.get("description", ""))
    step, host = (m.group(1), m.group(2)) if m else ("?", "?")

    return {
        "tactic"   : TACTIC_FOLDER_TO_LABEL[g["tactic"]],
        "technique": g["attack"],
        "step"     : step,
        "host"     : host,
        "n_triples": len(sentences),
        "sequence" : sentences,
    }


def main():
    files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*", "*", "subgraph_*.json")))
    print(f"Found {len(files)} graph JSON files under {GRAPH_DIR}")

    n_ok, n_empty = 0, 0
    for fpath in files:
        technique = os.path.basename(os.path.dirname(fpath))
        tactic    = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
        fname     = os.path.basename(fpath).replace("subgraph_", "sequence_", 1)

        seq = build_one(fpath)

        out_dir = os.path.join(SEQUENCES_DIR, tactic, technique)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(seq, f, indent=2)

        if seq["n_triples"] == 0:
            n_empty += 1
        else:
            n_ok += 1
        print(f"[{tactic:20s}] {technique:12s} {seq['step']:35s} {seq['host']:12s} "
              f"n_triples={seq['n_triples']:4d} -> {out_path}")

    print()
    print(f"Built {n_ok} sequences ({n_empty} empty) -> {SEQUENCES_DIR}")


if __name__ == "__main__":
    main()
