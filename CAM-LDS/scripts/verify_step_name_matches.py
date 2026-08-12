import sys
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from parse_audit import parse_audit, print_events

TECHNIQUES_DIR = Path("/csse/research/contructive-learning/aaaa/manifestations_filtered/manifestations_filtered/techniques")


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def build_name_index():
    name_index = defaultdict(list)
    for technique_dir in sorted(TECHNIQUES_DIR.iterdir()):
        if not technique_dir.is_dir():
            continue
        technique = technique_dir.name
        for step_dir in sorted(technique_dir.iterdir()):
            if not step_dir.is_dir():
                continue
            step = step_dir.name
            audit_logs = list(step_dir.rglob("audit.log"))
            if not audit_logs:
                continue
            name_index[step].append((technique, step_dir, audit_logs))
    return name_index


def show_events_for_step(step_name):
    name_index = build_name_index()
    entries = name_index.get(step_name)
    if not entries:
        print(f"No step named '{step_name}' found with an audit.log.")
        return

    techniques = sorted(set(e[0] for e in entries))
    if len(techniques) < 2:
        print(f"'{step_name}' only appears under 1 technique ({techniques[0]}) -- nothing to compare.")

    for technique, step_dir, audit_logs in entries:
        for audit_path in audit_logs:
            host = audit_path.relative_to(step_dir).parts[0]
            print(f"\n{'=' * 80}")
            print(f"technique={technique}  step={step_name}  host={host}")
            print(f"{'=' * 80}")
            events = parse_audit(audit_path)
            print_events(events)


def main():
    name_index = build_name_index()
    cross_technique = {
        step: entries for step, entries in name_index.items()
        if len(set(e[0] for e in entries)) > 1
    }

    identical = 0
    different = 0
    partial = 0
    mismatches = []

    for step, entries in cross_technique.items():
        per_technique_hashes = {}
        for technique, step_dir, audit_logs in entries:
            hashes = set()
            for audit_path in audit_logs:
                host = audit_path.relative_to(step_dir).parts[0]
                hashes.add((host, hash_file(audit_path)))
            per_technique_hashes[technique] = hashes

        all_hash_sets = list(per_technique_hashes.values())
        common = set.intersection(*all_hash_sets) if all_hash_sets else set()
        union = set.union(*all_hash_sets) if all_hash_sets else set()

        if common == union and len(union) > 0:
            identical += 1
        elif common:
            partial += 1
            mismatches.append((step, per_technique_hashes))
        else:
            different += 1
            mismatches.append((step, per_technique_hashes))

    total = len(cross_technique)
    print(f"Total cross-technique name-matched steps : {total}")
    print(f"  100% identical content (all hosts match) : {identical}")
    print(f"  partial overlap (some hosts match, some don't) : {partial}")
    print(f"  no overlap at all (name coincidence only) : {different}")
    print()

    if mismatches:
        print(f"Steps with partial/no content overlap (first 10 of {len(mismatches)}):")
        for step, per_technique_hashes in mismatches[:10]:
            print(f"\n  step={step}")
            for technique, hashes in per_technique_hashes.items():
                hosts = sorted(h for h, _ in hashes)
                print(f"    {technique:12s} hosts={hosts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=str, default=None)
    args = ap.parse_args()

    if args.step:
        show_events_for_step(args.step)
    else:
        main()

