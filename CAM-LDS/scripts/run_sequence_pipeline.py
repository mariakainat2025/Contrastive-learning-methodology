import json
import os
import random
import sys

PROJECT_ROOT = "/csse/research/contructive-learning"
CAM_LDS_SCRIPTS = os.path.join(PROJECT_ROOT, "CAM-LDS", "scripts")
ZOOMER_SCRIPTS = os.path.join(PROJECT_ROOT, "CAM-LDS", "zoomer", "scripts")


for _p in (PROJECT_ROOT, CAM_LDS_SCRIPTS, ZOOMER_SCRIPTS):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from data_utils import instance_from_filename

import train_camlds_matcher
import test_camlds_matcher

_original_load_sequences = train_camlds_matcher.load_sequences

DATA_SPLIT_PATH = os.path.join(PROJECT_ROOT, "CAM-LDS", "zoomer", "scripts", "data_split_output.json")


def _load_data_split():
    with open(DATA_SPLIT_PATH) as f:
        return json.load(f)


def get_zoomer_instances(match_zoomer_test=False):
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
    return set(_load_data_split()["zoomer_tactics"])


MATCH_ZOOMER_TEST = os.environ.get("MATCH_ZOOMER_TEST") == "1"
TRAIN_INSTANCES, TEST_INSTANCES = get_zoomer_instances(match_zoomer_test=MATCH_ZOOMER_TEST)
ALLOWED_INSTANCES = TRAIN_INSTANCES | TEST_INSTANCES
ZOOMER_TACTICS = get_zoomer_tactics() if MATCH_ZOOMER_TEST else None


def _restricted_load_sequences(*args, **kwargs):
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
