"""
Runs the CAM-LDS matcher (train_camlds_matcher.py / test_camlds_matcher.py)
3 times, once per seed (0, 1, 2), each time restricted to the EXACT same
train/test instances ZOOMER's multi-label pipeline used for that seed
(zoomer/scripts/create_data_split.py's get_zoomer_split_multilabel(seed)) --
so both systems are compared on identical data, seed for seed.

Unlike run_sequence_pipeline.py (which reads a single, pre-saved, fixed-seed
split from data_split_output.json), this calls get_zoomer_split_multilabel
LIVE for each of the 3 seeds, since that function itself builds a fresh
random split per seed rather than reading one fixed file.

Two runtime patches, same as run_sequence_pipeline.py and for the same
reasons (see that file's docstring for the full explanation):
1. load_sequences() -- restricted to only this seed's ZOOMER instances, and
   (matching ZOOMER's own 9-tactic scope) each entry's tactic labels are
   narrowed to ZOOMER's 9 tactics, dropping labels outside it. Without this,
   the matcher would be scored on tactics ZOOMER can never predict at all
   (it has no output class for them), which isn't a fair comparison.
2. leave_out_split() -- exact match instead of substring match. The
   original does `any(m in e['file'] for m in match_substrs)`, which has
   real false positives on these instance ids (e.g. "6_macro_binary-1" is a
   literal prefix of "6_macro_binary-11/-12/-13/-15/-17").

Uses sequences/ (IP/path-generalized), the same generalized text your main
CAM-LDS pipeline (CAM-LDS/main.py) trains on by default -- so this compares
your real, as-used matcher configuration against ZOOMER, not a stripped-down
raw-text variant built just for this comparison. (sequences_raw/, built
straight from the raw graphs with no generalization step, is the other
option if a same-raw-source comparison is ever wanted instead -- both
directories have full coverage of every ZOOMER-scoped instance, verified
2026-09-10.)

Both train_camlds_matcher and test_camlds_matcher get patched, because
test_camlds_matcher imported its own copies of load_sequences/leave_out_split
via `from train_camlds_matcher import (...)` -- patching the source module
alone would not reach test_camlds_matcher's already-bound references.
"""
import json
import os
import random
import statistics
import sys

PROJECT_ROOT = '/csse/research/contructive-learning'
CAM_LDS_SCRIPTS = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'scripts')
ZOOMER_SCRIPTS = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'zoomer', 'scripts')
for _p in (PROJECT_ROOT, CAM_LDS_SCRIPTS, ZOOMER_SCRIPTS):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import torch

from data_utils import instance_from_filename
from create_data_split import get_zoomer_split_multilabel

import train_camlds_matcher
import test_camlds_matcher

_original_load_sequences = train_camlds_matcher.load_sequences

SEEDS = (0, 1, 2)
CAM_LDS_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS')
RESULTS_DIR = os.path.join(CAM_LDS_DIR, 'results')
SEQUENCES_DIR = os.path.join(CAM_LDS_DIR, 'sequences')


def zoomer_instances_and_tactics(seed):
    """Same multi-label train/test split ZOOMER trained/tested on for this
    seed, read LIVE (not from a saved file) so this always matches whatever
    ZOOMER's own split logic currently does."""
    split = get_zoomer_split_multilabel(seed)
    train_instances = set()
    test_instances = set()
    for pools in split.values():
        for (fn, _path) in pools['train']:
            train_instances.add(instance_from_filename(fn))
        for (fn, _path) in pools['test']:
            test_instances.add(instance_from_filename(fn))
    zoomer_tactics = {pools['tactic'] for pools in split.values()}
    return (train_instances, test_instances, zoomer_tactics)


def make_restricted_load_sequences(allowed_instances, zoomer_tactics):
    def _restricted_load_sequences(*args, **kwargs):
        return_total = kwargs.get('return_total', False)
        result = _original_load_sequences(*args, **kwargs)
        (entries, n_total_unfiltered) = result if return_total else (result, None)

        filtered = [e for e in entries if e['file'] in allowed_instances]

        n_before = len(filtered)
        narrowed = []
        for e in filtered:
            kept = [t for t in e['tactics'] if t in zoomer_tactics]
            if not kept:
                continue
            narrowed.append(dict(e, tactics=kept))
        filtered = narrowed
        print('  [zoomer-match] kept {}/{} steps with >=1 label in ZOOMER\'s 9 tactics'.format(
            len(filtered), n_before))

        if return_total:
            return (filtered, n_total_unfiltered)
        return filtered
    return _restricted_load_sequences


def make_exact_leave_out_split(test_instances):
    def _exact_leave_out_split(entries, match_set, seed=train_camlds_matcher.SEED):
        rng = random.Random(seed)
        (train, test) = ([], [])
        for e in entries:
            (test if e['file'] in test_instances else train).append(e)
        rng.shuffle(train)
        rng.shuffle(test)
        return (train, test)
    return _exact_leave_out_split


def run_one_seed(seed, class_reweight=True, tag_suffix=''):
    (train_instances, test_instances, zoomer_tactics) = zoomer_instances_and_tactics(seed)
    allowed = train_instances | test_instances
    print('Seed {}: {} train instances, {} test instances, {} total, tactics={}'.format(
        seed, len(train_instances), len(test_instances), len(allowed), sorted(zoomer_tactics)))

    for module in (train_camlds_matcher, test_camlds_matcher):
        module.load_sequences = make_restricted_load_sequences(allowed, zoomer_tactics)
        module.leave_out_split = make_exact_leave_out_split(test_instances)

    torch.manual_seed(seed)
    run_tag = 'zoomer_multilabel_seed{}'.format(seed)
    if not class_reweight:
        run_tag += '_noreweight'
    run_tag += tag_suffix

    train_camlds_matcher.run_contrastive_train(
        test_file_match=test_instances,
        split_seed=seed,
        run_tag=run_tag,
        min_events=None,
        sequences_dir=SEQUENCES_DIR,
        class_reweight=class_reweight,
    )
    test_camlds_matcher.run(
        test_file_match=test_instances,
        split_seed=seed,
        run_tag=run_tag,
        min_events=None,
        sequences_dir=SEQUENCES_DIR,
    )

    results_path = os.path.join(RESULTS_DIR, 'camlds_test_results_{}.json'.format(run_tag))
    with open(results_path) as f:
        out = json.load(f)
    return {'seed': seed, 'lrap': out['lrap'], 'aupr': out['aupr'],
            'n_total': out['n_total'], 'n_wrong_top3': out['n_wrong_top3']}


def main(seeds=SEEDS):
    all_metrics = []
    for seed in seeds:
        print()
        print('=' * 70)
        print('CAM-LDS matcher -- SEED {} (same split as ZOOMER multilabel)'.format(seed))
        print('=' * 70)
        all_metrics.append(run_one_seed(seed))

    print()
    print('=' * 70)
    print('Summary across {} seeds: {}'.format(len(seeds), list(seeds)))
    print('=' * 70)
    summary = {'seeds': list(seeds), 'per_seed': all_metrics}
    for key in ('lrap', 'aupr'):
        values = [m[key] * 100 for m in all_metrics]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        per_seed = ', '.join('seed{}={:.1f}%'.format(m['seed'], v) for (m, v) in zip(all_metrics, values))
        print('{:22s}: {:5.1f}% +/- {:4.1f}%   ({})'.format(key.upper(), mean, std, per_seed))
        summary[key] = {'mean': mean, 'std': std}

    summary_path = os.path.join(RESULTS_DIR, 'summary_camlds_matcher_zoomer_multilabel.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print('Saved -> {}'.format(summary_path))
    return all_metrics


if __name__ == '__main__':
    main()
