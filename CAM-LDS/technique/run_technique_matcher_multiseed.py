"""
Runs the CAM-LDS matcher at TECHNIQUE level, restricted to the EXACT same train/test
instances ZOOMER used for a given seed (create_data_split.get_zoomer_split_multilabel),
so this is directly comparable to ZOOMER and to the tactic-level matcher -- same seeds,
same data, same pattern as run_camlds_matcher_multiseed.py (CAM-LDS/scripts/), just at
technique granularity instead of tactic granularity.

get_zoomer_split_multilabel(seed) is already keyed by ZOOMER's 16 technique classes, so
no tactic-collapsing step is needed here -- we just pull train/test instances straight
from it. The one adjustment: T1059-004 (Unix Shell) is merged into T1059-000 (Command
and Scripting Interpreter) to match TECHNIQUE_MERGE in train_camlds_matcher_technique.py,
since both share one base-technique template and would otherwise be indistinguishable.
"""
import json
import os
import random
import sys

TECHNIQUE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = '/csse/research/contructive-learning'
ZOOMER_SCRIPTS = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'zoomer', 'scripts')
for _p in (TECHNIQUE_DIR, PROJECT_ROOT, ZOOMER_SCRIPTS):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import torch

from data_utils import instance_from_filename
from create_data_split import get_zoomer_split_multilabel

import train_camlds_matcher_technique as train_mod
import test_camlds_matcher_technique as test_mod

_original_load_sequences = train_mod.load_sequences

SEEDS = (0, 1, 2)
RESULTS_DIR = os.path.join(TECHNIQUE_DIR, 'results')


def zoomer_instances_and_techniques(seed):
    split = get_zoomer_split_multilabel(seed)
    train_instances = set()
    test_instances = set()
    for pools in split.values():
        for (fn, _path) in pools['train']:
            train_instances.add(instance_from_filename(fn))
        for (fn, _path) in pools['test']:
            test_instances.add(instance_from_filename(fn))
    zoomer_techniques = {train_mod.TECHNIQUE_MERGE.get(t, t) for t in split.keys()}
    return (train_instances, test_instances, zoomer_techniques)


def make_restricted_load_sequences(allowed_instances, zoomer_techniques):
    def _restricted_load_sequences(*args, **kwargs):
        return_total = kwargs.get('return_total', False)
        result = _original_load_sequences(*args, **kwargs)
        (entries, n_total_unfiltered) = result if return_total else (result, None)

        filtered = [e for e in entries if e['file'] in allowed_instances]

        n_before = len(filtered)
        narrowed = []
        for e in filtered:
            kept = [t for t in e['techniques'] if t in zoomer_techniques]
            if not kept:
                continue
            narrowed.append(dict(e, techniques=kept))
        filtered = narrowed
        print('  [zoomer-match] kept {}/{} steps with >=1 label in ZOOMER\'s {} techniques'.format(
            len(filtered), n_before, len(zoomer_techniques)))

        if return_total:
            return (filtered, n_total_unfiltered)
        return filtered
    return _restricted_load_sequences


def make_exact_leave_out_split(test_instances):
    def _exact_leave_out_split(entries, match_set, seed=train_mod.SEED):
        rng = random.Random(seed)
        (train, test) = ([], [])
        for e in entries:
            (test if e['file'] in test_instances else train).append(e)
        rng.shuffle(train)
        rng.shuffle(test)
        return (train, test)
    return _exact_leave_out_split


def run_one_seed(seed, class_reweight=False, tag_suffix=''):
    (train_instances, test_instances, zoomer_techniques) = zoomer_instances_and_techniques(seed)
    allowed = train_instances | test_instances
    print('Seed {}: {} train instances, {} test instances, {} total, techniques={}'.format(
        seed, len(train_instances), len(test_instances), len(allowed), sorted(zoomer_techniques)))

    for module in (train_mod, test_mod):
        module.load_sequences = make_restricted_load_sequences(allowed, zoomer_techniques)
        module.leave_out_split = make_exact_leave_out_split(test_instances)

    torch.manual_seed(seed)
    run_tag = 'zoomer_technique_seed{}'.format(seed)
    if not class_reweight:
        run_tag += '_noreweight'
    run_tag += tag_suffix

    train_mod.run_contrastive_train(
        test_file_match=test_instances, split_seed=seed, run_tag=run_tag,
        min_events=None, sequences_dir=train_mod.SEQUENCES_DIR, class_reweight=class_reweight,
    )
    test_mod.run(
        test_file_match=test_instances, split_seed=seed, run_tag=run_tag,
        min_events=None, sequences_dir=train_mod.SEQUENCES_DIR,
    )

    results_path = os.path.join(RESULTS_DIR, 'camlds_technique_test_results_{}.json'.format(run_tag))
    with open(results_path) as f:
        out = json.load(f)
    return {'seed': seed, 'lrap': out['lrap'], 'aupr': out['aupr'],
            'n_total': out['n_total'], 'n_wrong_top3': out['n_wrong_top3']}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--class-reweight', action='store_true')
    ap.add_argument('--tag-suffix', type=str, default='')
    args = ap.parse_args()
    result = run_one_seed(args.seed, class_reweight=args.class_reweight, tag_suffix=args.tag_suffix)
    print()
    print('Seed {} -> {}'.format(args.seed, result))
