
import os
import json
import glob

CAM_LDS_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_DIR     = os.path.join(CAM_LDS_DIR, "graph")
SEQUENCES_DIR = os.path.join(CAM_LDS_DIR, "sequences")

TACTIC_FOLDER_TO_LABEL = {
    "command_and_control": "Command_and_Control",
    "initial_access":      "Initial_Access",
}


def triplet_to_sentence(t):
    src, edge_type, dst = t
    return f"{src} {edge_type.lower()} {dst}."


def build_one(graph_path):
    with open(graph_path) as f:
        g = json.load(f)

    sentences = [triplet_to_sentence(t) for t in g.get("triplets", [])]

    return {
        "tactic"   : TACTIC_FOLDER_TO_LABEL[g["tactic"]],
        "technique": g["technique"],
        "step"     : g["step"],
        "host"     : g["host"],
        "n_triples": len(sentences),
        "sequence" : sentences,
    }


def main():
    files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*", "*", "graph_*.json")))
    print(f"Found {len(files)} graph JSON files under {GRAPH_DIR}")

    n_ok, n_empty = 0, 0
    for fpath in files:
        technique = os.path.basename(os.path.dirname(fpath))
        tactic    = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
        fname     = os.path.basename(fpath).replace("graph_", "sequence_", 1)

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
