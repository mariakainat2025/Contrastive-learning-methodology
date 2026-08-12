import json
from pathlib import Path
from collections import defaultdict

GROUPED_DIR = Path("/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic")
OUT_PATH = Path(__file__).parent / "folder_tactic_map.json"

TACTICS = ["collection", "command_and_control", "credential_access", "defense_impairment",
           "discovery", "execution", "exfiltration", "impact", "initial_access",
           "lateral_movement", "persistence", "privilege_escalation", "reconnaissance", "stealth"]


def has_audit_log(step_dir):
    return next(step_dir.rglob("audit.log"), None) is not None


def build_map():
    step_tactics = defaultdict(set)

    for tactic in TACTICS:
        tactic_dir = GROUPED_DIR / tactic
        if not tactic_dir.is_dir():
            continue
        for technique_dir in tactic_dir.iterdir():
            if not technique_dir.is_dir():
                continue
            for step_dir in technique_dir.iterdir():
                if not step_dir.is_dir():
                    continue
                key = f"{technique_dir.name}/{step_dir.name}"
                step_tactics[key].add(tactic)

    step_map = {}
    for key, tactics in step_tactics.items():
        technique, step = key.split("/", 1)
        step_dir = GROUPED_DIR / next(iter(tactics)) / technique / step
        if not has_audit_log(step_dir):
            continue
        step_map[key] = sorted(tactics)

    return step_map


def main():
    step_map = build_map()
    multi = {k: v for k, v in step_map.items() if len(v) > 1}
    single = {k: v for k, v in step_map.items() if len(v) == 1}

    print(f"Total steps (with audit.log)  : {len(step_map)}")
    print(f"  single-tactic                : {len(single)}")
    print(f"  multi-tactic                 : {len(multi)}")
    print()
    print(f"Sample multi-tactic steps (first 15 of {len(multi)}):")
    for k, v in list(multi.items())[:15]:
        print(f"  {k:45s} tactics={v}")

    with open(OUT_PATH, "w") as f:
        json.dump(step_map, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

