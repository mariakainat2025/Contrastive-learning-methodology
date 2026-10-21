import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import Train_TTP_Recognition_Multilabel as train_mod
import Test_TTP_Recognition_Multilabel as test_mod


def header(title):
    print()
    print('=' * 70)
    print('  {}'.format(title))
    print('=' * 70)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for k-means bins/masks (and the random split when --scenario is not given).')
    ap.add_argument('--scenario', type=str, default=None,
                     help='Hold out this scenario/run-group (e.g. "3" for the whole scenario, '
                          '"3_ssh_apt" for only that run-group) instead of a random split.')
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--min-train-samples', type=int, default=None,
                     help='Override MIN_TRAIN_SAMPLES_SCENARIO (default 2) for scenario mode. '
                          'Ignored when --fixed-scope is set.')
    ap.add_argument('--fixed-scope', action='store_true',
                     help="Use CAM-LDS's fixed 36 base-technique scope on both train and test, "
                          'so this run is directly comparable to the CAM-LDS text+wide side.')
    ap.add_argument('--run', type=str, default=None,
                     help='Only report headline test metrics for this one run group (exact match), '
                          'e.g. --run 3_ssh_apt. Training/scoring still cover the full scenario.')
    ap.add_argument('--skip-train', action='store_true', help='Skip training, test an existing checkpoint.')
    ap.add_argument('--skip-test', action='store_true', help='Skip testing (train only).')
    args = ap.parse_args()

    run_tag = args.run_tag
    if run_tag is None and args.scenario:
        if args.fixed_scope:
            run_tag = 'scenario{}_camldsscope'.format(args.scenario)
        elif args.min_train_samples is not None:
            run_tag = 'scenario{}_min{}'.format(args.scenario, args.min_train_samples)

    if not args.skip_train:
        header('Step 1/2: Training (ZOOMER graph + wide features, technique prototypes)')
        train_mod.main(args.seed, scenario=args.scenario, run_tag=run_tag,
                        min_train_samples=args.min_train_samples, fixed_scope=args.fixed_scope)

    if not args.skip_test:
        header('Step 2/2: Testing (technique / tactic / stage accuracy)')
        test_mod.main(args.seed, scenario=args.scenario, run_tag=run_tag,
                       run_filter=args.run, fixed_scope=args.fixed_scope)


if __name__ == '__main__':
    main()
