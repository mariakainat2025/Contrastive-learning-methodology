"""
Runs the existing CAM-LDS matcher pipeline (train_camlds_matcher.py /
test_camlds_matcher.py) restricted to the EXACT same instances used by the
ZOOMER prototypical-network pipeline (zoomer/scripts/data_utils.py) --
so results from both systems are comparable on identical data.

Reads its split from data_split_output.json (CAM-LDS/create_data_split.py's
output) instead of recomputing anything itself -- run create_data_split.py
first. This keeps ZOOMER and this pipeline reading the exact same,
already-decided split off disk, rather than each independently calling
data_utils.build_split() and risking the two calls drifting apart.

Two small patches are applied at runtime (not to the original files):

1. load_sequences() -- restricted to only the instances in our ZOOMER
   split, instead of the full ~195-step corpus.
2. leave_out_split() -- switched from substring matching to EXACT matching.
   The original does `any(m in e['file'] for m in match_substrs)`, which
   has real false positives on these instance names: "6_macro_binary-1"
   is a literal prefix of "6_macro_binary-11/-12/-13/-15/-17", so using it
   as a substring test marker would wrongly pull 5 unrelated training
   instances into the test set too (verified against our actual instances
   before writing this).

Both train_camlds_matcher and test_camlds_matcher get patched separately,
because test_camlds_matcher imported its own copies of these two names via
`from train_camlds_matcher import (...)` -- patching the source module
alone would not reach test_camlds_matcher's already-bound references.
"""
import json
import os
import random
import sys

PROJECT_ROOT = "/csse/research/contructive-learning"
CAM_LDS_SCRIPTS = os.path.join(PROJECT_ROOT, "CAM-LDS", "scripts")
ZOOMER_SCRIPTS = os.path.join(PROJECT_ROOT, "CAM-LDS", "zoomer", "scripts")
# Remove-then-reinsert (not "insert only if absent"): running this file
# directly makes Python auto-add its own directory (CAM_LDS_SCRIPTS) to
# sys.path before this code runs. An "if not already present" guard would
# then skip re-inserting CAM_LDS_SCRIPTS while still freshly inserting
# PROJECT_ROOT ahead of it -- letting a stale root-level parse_audit.py
# shadow the real one in CAM_LDS_SCRIPTS/. Removing first guarantees each
# path lands in the intended position every time, regardless of what
# Python already put on sys.path.
for _p in (PROJECT_ROOT, CAM_LDS_SCRIPTS, ZOOMER_SCRIPTS):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from data_utils import instance_from_filename  # noqa: E402

import train_camlds_matcher  # noqa: E402
import test_camlds_matcher  # noqa: E402

_original_load_sequences = train_camlds_matcher.load_sequences

DATA_SPLIT_PATH = os.path.join(PROJECT_ROOT, "CAM-LDS", "zoomer", "scripts", "data_split_output.json")


def _load_data_split():
    """Reads the split create_data_split.py already computed and saved --
    run that script first if this raises FileNotFoundError. Nothing in
    this file calls build_split() directly, so this pipeline and ZOOMER's
    own scripts are guaranteed to see the exact same split, read from the
    same file, instead of two independent live calls that could drift
    apart."""
    with open(DATA_SPLIT_PATH) as f:
        return json.load(f)


def get_zoomer_instances(match_zoomer_test=False):
    """Instances used by ZOOMER's graph pipeline. Reads the "multi_label_
    techniques" split (an instance can repeat across several techniques --
    this pipeline handles multi-label natively, so it should see the full
    multi-technique instance set, not ZOOMER's single-label simplification).

    match_zoomer_test=True: swap in ZOOMER's own saved test instances as
    TEST_INSTANCES here too, instead of this pipeline's own -- for a true
    apples-to-apples comparison, both pipelines scored on the exact same
    held-out samples. Only 5 of ZOOMER's 24 test instances fall in this
    pipeline's own 30-test pool (verified 2026-08-27); the other 19 sit in
    this pipeline's 113-train pool, so this requires RETRAINING with those
    19 moved out, not just re-scoring an existing checkpoint -- otherwise
    the model would be tested on data it already trained on."""
    data = _load_data_split()
    train_instances, test_instances = set(), set()
    for pools in data["multi_label_techniques"].values():
        for row in pools["train"]:
            train_instances.add(instance_from_filename(row["filename"]))
        for row in pools["test"]:
            test_instances.add(instance_from_filename(row["filename"]))

    if match_zoomer_test:
        all_instances = train_instances | test_instances
        test_instances = set(data["zoomer_test_instances"])
        train_instances = all_instances - test_instances

    return train_instances, test_instances


def get_zoomer_tactics():
    """The exact tactics ZOOMER's own single-label techniques map to (see
    Test_TTP_Recognition.py's "one technique implies one tactic" rule), as already
    decided and saved by create_data_split.py. This dataset natively has
    13-14 tactics; without this restriction the sequence pipeline trains/
    evaluates against tactics ZOOMER's surviving techniques never touch
    (e.g. defense_impairment, exfiltration, impact, lateral_movement),
    which isn't a like-for-like comparison."""
    return set(_load_data_split()["zoomer_tactics"])


MATCH_ZOOMER_TEST = os.environ.get("MATCH_ZOOMER_TEST") == "1"
TRAIN_INSTANCES, TEST_INSTANCES = get_zoomer_instances(match_zoomer_test=MATCH_ZOOMER_TEST)
ALLOWED_INSTANCES = TRAIN_INSTANCES | TEST_INSTANCES
ZOOMER_TACTICS = get_zoomer_tactics() if MATCH_ZOOMER_TEST else None


def _restricted_load_sequences(*args, **kwargs):
    """Same as the original load_sequences(), but only keeps steps that
    are part of our ZOOMER split (see module docstring). When
    MATCH_ZOOMER_TEST is on, also drops any tactic label outside ZOOMER_
    TACTICS from each entry (a multi-label step keeps its other, still-
    valid labels), and drops any step left with zero labels afterward --
    it can no longer be a positive example for anything in this
    restricted 9-tactic space."""
    return_total = kwargs.get("return_total", False)
    result = _original_load_sequences(*args, **kwargs)
    entries, n_total_unfiltered = result if return_total else (result, None)

    filtered = [e for e in entries if e["file"] in ALLOWED_INSTANCES]

    if ZOOMER_TACTICS is not None:
        n_before = len(filtered)
        tactic_narrowed = []
        for e in filtered:
            kept = [t for t in e["tactics"] if t in ZOOMER_TACTICS]
            if not kept:
                continue
            tactic_narrowed.append(dict(e, tactics=kept))
        filtered = tactic_narrowed
        print("  MATCH_ZOOMER_TEST tactic filter: kept {}/{} steps with >=1 label in "
              "ZOOMER's 9 tactics: {}".format(len(filtered), n_before, ", ".join(sorted(ZOOMER_TACTICS))))

    if return_total:
        return filtered, n_total_unfiltered
    return filtered


def _exact_leave_out_split(entries, match_set, seed=train_camlds_matcher.SEED):
    """Same as the original leave_out_split(), but exact-match against
    e['file'] instead of substring search (see module docstring)."""
    if isinstance(match_set, str):
        match_set = {match_set}
    else:
        match_set = set(match_set)
    rng = random.Random(seed)
    train, test = [], []
    for e in entries:
        (test if e["file"] in match_set else train).append(e)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


for _module in (train_camlds_matcher, test_camlds_matcher):
    _module.load_sequences = _restricted_load_sequences
    _module.leave_out_split = _exact_leave_out_split


def main(run_tag=None):
    if run_tag is None:
        run_tag = "zoomer_test_match" if MATCH_ZOOMER_TEST else "zoomer_split"
    print("ZOOMER-matched split (match_zoomer_test={}): {} train instances, {} test instances, {} total".format(
        MATCH_ZOOMER_TEST, len(TRAIN_INSTANCES), len(TEST_INSTANCES), len(ALLOWED_INSTANCES)))

    # min_events=None disables their own separate "too few events" filter --
    # our ZOOMER split (MIN_TRAIN_SAMPLES) is the only quality gate we want
    # applied here, not two independent, disagreeing filters.
    train_camlds_matcher.run_contrastive_train(
        test_file_match=TEST_INSTANCES,
        run_tag=run_tag,
        min_events=None,
    )
    test_camlds_matcher.run(
        test_file_match=TEST_INSTANCES,
        run_tag=run_tag,
        min_events=None,
    )


if __name__ == "__main__":
    main()
