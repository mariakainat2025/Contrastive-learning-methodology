import json
import argparse
from pathlib import Path
from collections import defaultdict

TECHNIQUES_DIR = Path("/csse/research/contructive-learning/aaaa/manifestations_filtered/manifestations_filtered/techniques")
GROUPED_DIR = Path("/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic")
OUT_PATH = Path(__file__).parent / "step_tactics.json"

TACTICS = ["collection", "command_and_control", "credential_access", "defense_impairment",
           "discovery", "execution", "exfiltration", "impact", "initial_access",
           "lateral_movement", "persistence", "privilege_escalation", "reconnaissance", "stealth"]

OUR_TACTICS = ["privilege_escalation", "command_and_control", "credential_access", "execution", "initial_access", "lateral_movement", "persistence", "stealth", "defense_impairment", "reconnaissance", "impact", "discovery", "collection", "exfiltration"]


def build_technique_to_tactics():
    technique_tactics = defaultdict(set)
    for tactic in TACTICS:
        tactic_dir = GROUPED_DIR / tactic
        if not tactic_dir.is_dir():
            continue
        for technique_dir in tactic_dir.iterdir():
            if not technique_dir.is_dir():
                continue
            technique_tactics[technique_dir.name].add(tactic)
    return technique_tactics


def get_technique_tactics(technique_tactics, technique):
    return sorted(technique_tactics.get(technique, set()))


def build_name_index():
    name_index = defaultdict(set)
    for tactic in OUR_TACTICS:
        tactic_dir = GROUPED_DIR / tactic
        if not tactic_dir.is_dir():
            continue
        for technique_dir in tactic_dir.iterdir():
            if not technique_dir.is_dir():
                continue
            technique = technique_dir.name
            for step_dir in technique_dir.iterdir():
                if not step_dir.is_dir():
                    continue
                step = step_dir.name
                has_audit = next(step_dir.rglob("audit.log"), None) is not None
                if not has_audit:
                    continue
                name_index[step].add(technique)
    return {step: sorted(techs) for step, techs in name_index.items()}


def lookup_step(step_name):
    technique_tactics = build_technique_to_tactics()
    name_index = build_name_index()

    techniques = sorted(set(name_index.get(step_name, [])))
    if not techniques:
        print(f"No step named '{step_name}' found with an audit.log.")
        return

    all_tactics = set()
    print(f"step={step_name}")
    for technique in techniques:
        tactics = get_technique_tactics(technique_tactics, technique)
        all_tactics.update(tactics)
        print(f"  {technique:12s} tactics={tactics}")

    print(f"\nTotal tactics for '{step_name}': {len(all_tactics)}")


def main():
    technique_tactics = build_technique_to_tactics()
    name_index = build_name_index()

    cross_technique = {
        step: techniques for step, techniques in name_index.items()
        if len(set(techniques)) > 1
    }
    single_technique = {
        step: techniques for step, techniques in name_index.items()
        if len(set(techniques)) == 1
    }

    print(f"Total distinct step names (with audit.log) : {len(name_index)}")
    print(f"  step names appearing under 1 technique    : {len(single_technique)}")
    print(f"  step names appearing under 2+ techniques  : {len(cross_technique)}")
    print()

    report = []
    for step, techniques in name_index.items():
        row = {"step": step, "occurrences": []}
        for technique in sorted(set(techniques)):
            row["occurrences"].append({
                "technique": technique,
                "tactics": get_technique_tactics(technique_tactics, technique),
            })
        report.append(row)

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=str, default=None)
    args = ap.parse_args()

    if args.step:
        lookup_step(args.step)
    else:
        main()

