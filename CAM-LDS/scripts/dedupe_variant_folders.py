import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

GROUPED_DIR = Path("/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic")
ARCHIVE_DIR = Path("/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic_duplicates")

SUFFIX_RE = re.compile(r'^(.*)-(\d+)$')


def group_by_suffix(technique_dir):
    """Groups variant folders in a technique dir by their trailing -<number> suffix.
    Same suffix, different method-name prefix = duplicate content (verified by hand
    on execution/T1059-000, persistence/T1053-000, initial_access/T1078-003,
    command_and_control -- only timestamp/pid differ, not the actual commands)."""
    groups = defaultdict(list)
    for entry in sorted(technique_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = SUFFIX_RE.match(entry.name)
        if not m:
            groups[entry.name].append(entry)  # no recognizable suffix -- keep as its own group
            continue
        suffix = m.group(2)
        groups[suffix].append(entry)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true',
                     help='actually move duplicates to grouped_by_tactic_duplicates/ '
                          '(default is dry-run: report only, no files touched)')
    args = ap.parse_args()

    total_before = 0
    total_kept = 0
    total_moved = 0
    per_tactic = defaultdict(lambda: [0, 0, 0])  # before, kept, moved

    for tactic_dir in sorted(GROUPED_DIR.iterdir()):
        if not tactic_dir.is_dir():
            continue
        tactic = tactic_dir.name
        for technique_dir in sorted(tactic_dir.iterdir()):
            if not technique_dir.is_dir():
                continue
            groups = group_by_suffix(technique_dir)
            for suffix, variants in groups.items():
                total_before += len(variants)
                per_tactic[tactic][0] += len(variants)
                keep = variants[0]
                drop = variants[1:]
                total_kept += 1
                per_tactic[tactic][1] += 1
                for d in drop:
                    total_moved += 1
                    per_tactic[tactic][2] += 1
                    if args.apply:
                        dest = ARCHIVE_DIR / tactic / technique_dir.name / d.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(d), str(dest))

    print(f"{'Tactic':22s} {'before':>8s} {'kept':>8s} {'moved':>8s}")
    for tactic, (before, kept, moved) in sorted(per_tactic.items()):
        print(f"{tactic:22s} {before:>8d} {kept:>8d} {moved:>8d}")
    print(f"{'TOTAL':22s} {total_before:>8d} {total_kept:>8d} {total_moved:>8d}")
    print()
    if args.apply:
        print(f"Applied — duplicates moved to {ARCHIVE_DIR}")
    else:
        print("Dry run only -- no files touched. Re-run with --apply to actually move duplicates.")


if __name__ == '__main__':
    main()
