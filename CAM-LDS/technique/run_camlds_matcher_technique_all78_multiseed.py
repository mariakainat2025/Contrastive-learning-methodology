"""
Trains + tests the 78-technique wide CAM-LDS matcher across 3 seeds, plain stratified
80/20 split each time -- no ZOOMER instance matching (this pipeline covers the full
CAM-LDS technique scope, independent of ZOOMER's 15/16-technique subset).
"""
import json
import os

import train_camlds_matcher_technique_all78 as train_mod
import test_camlds_matcher_technique_all78 as test_mod

TECHNIQUE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(TECHNIQUE_DIR, 'results')

SEEDS = (0, 1, 2)


def run_one_seed(seed):
    run_tag = 'all78_wide_seed{}'.format(seed)
    print()
    print('=' * 70)
    print('[all78] SEED {}  (stratified 80/20, no ZOOMER matching)'.format(seed))
    print('=' * 70)
    train_mod.run_contrastive_train(split_seed=seed, run_tag=run_tag, test_size=0.2, stratified=True)
    out = test_mod.run(run_tag, split_seed=seed, test_size=0.2)
    return {'seed': seed, 'lrap': out['lrap'], 'aupr': out['aupr'],
            'n_total': out['n_total'], 'n_wrong_top3': out['n_wrong_top3']}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None, help='run just this one seed; omit to run all 3')
    args = ap.parse_args()

    seeds = (args.seed,) if args.seed is not None else SEEDS
    all_results = []
    for seed in seeds:
        result = run_one_seed(seed)
        all_results.append(result)
        print()
        print('Seed {} -> {}'.format(seed, result))

    if len(all_results) > 1:
        summary_path = os.path.join(RESULTS_DIR, 'summary_all78_wide.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print()
        print('Summary saved -> {}'.format(summary_path))
        for r in all_results:
            print('  seed{} lrap={:.1f}% aupr={:.1f}% wrong={}/{}'.format(
                r['seed'], r['lrap'] * 100, r['aupr'] * 100, r['n_wrong_top3'], r['n_total']))
