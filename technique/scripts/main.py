
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import step_tactic_lookup
import generate_technique_templates
import train_camlds_matcher_technique_wide as train_mod
import test_camlds_matcher_technique_wide as test_mod


def header(title):
    print()
    print('=' * 70)
    print('  {}'.format(title))
    print('=' * 70)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test-file', type=str, default=None)
    ap.add_argument('--test-scenario', type=str, default=None,
                     help='Hold out an ENTIRE scenario as test (e.g. "4"). Overrides --test-file when set.')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=train_mod.SEED)
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--k-per-technique', type=int, default=train_mod.K_PER_TACTIC)
    ap.add_argument('--template-dir', type=str, default=None)
    ap.add_argument('--no-stratified', dest='stratified', action='store_false')
    ap.add_argument('--grouped-split', action='store_true',
                     help='Split at the near-duplicate-sibling-group level instead of the individual-step '
                          'level (e.g. 3_ssh_apt-8 / 3_ssh_healthcheck-8 / 3_ssh_puppet-8 always land on the '
                          'same side) -- eliminates leakage between near-identical scenario variants, so '
                          'test-set performance reflects genuine generalization, not just recognizing a '
                          'near-copy of a training example. Only affects the default stratified 80/20 split '
                          '(ignored when --test-scenario or --test-file is set).')
    ap.add_argument('--lambda-uniform', type=float, default=train_mod.LAMBDA_UNIFORM)
    ap.add_argument('--class-reweight', action='store_true')
    ap.add_argument('--reweight-cap', type=float, default=3.0)
    ap.add_argument('--sequences-dir', type=str, default=None)
    ap.add_argument('--lr', type=float, default=train_mod.LR)
    ap.add_argument('--full-batch', action='store_true',
                     help='Use ALL training examples every step instead of a small stratified '
                          'subsample -- much more gradient signal per step on small datasets.')
    ap.add_argument('--max-chunks', type=int, default=None,
                     help='Cap how many 512-token chunks any single long sequence contributes '
                          '(a few sequences run to 100k+ tokens and will OOM a full-batch step '
                          'if left uncapped). None = no cap (existing behavior).')
    ap.add_argument('--patience', type=int, default=train_mod.PATIENCE,
                     help='Epochs to wait for improvement before early-stopping.')
    ap.add_argument('--n-freeze', type=int, default=train_mod.N_FREEZE,
                     help='How many of RoBERTa\'s 12 layers to freeze (lower = more trainable capacity).')
    ap.add_argument('--dropout', type=float, default=train_mod.DROPOUT)
    ap.add_argument('--zoomer-scope', action='store_true',
                     help="Train on ZOOMER's 26-technique min4 scope instead of CAM-LDS's native "
                          '36-technique one, for a 100%% identical training set between the two models. '
                          'Testing still scores against ALL 36 templates by default regardless of this '
                          "flag (CAM-LDS can zero-shot-match a technique via its template text even "
                          'without having fine-tuned on it) -- pass --test-zoomer-scope as well to also '
                          'narrow the test-time candidate set down to 26, mirroring ZOOMER exactly.')
    ap.add_argument('--test-zoomer-scope', action='store_true',
                     help='Narrow the TEST-time candidate set down to the 26-technique min4 scope too '
                          '(independent of --zoomer-scope, which only affects training).')
    ap.add_argument('--allow-extra-techniques', dest='strict_native', action='store_false',
                     help='Default is to cap test-time scoring at exactly the trained native scope. Pass '
                          'this flag to also include rare test-only techniques (below the '
                          'training-sufficiency bar) as extra zero-shot candidates whenever a template '
                          'exists for them, even though they were never trained on.')
    ap.add_argument('--no-wide', dest='use_wide', action='store_false',
                     help='Text-only embedding, no ZOOMER wide (node/edge/IoC count) features '
                          'concatenated in at all -- an ablation to check whether the wide features '
                          'are helping or hurting, independent of the text side.')
    ap.add_argument('--loss', dest='loss_fn', choices=['contrastive', 'asl'], default='contrastive',
                     help="'contrastive' (default) = the existing prototype/logsumexp loss. "
                          "'asl' = Asymmetric Loss for Multi-Label Classification -- independent "
                          'per-class sigmoid with asymmetric focal weighting, built for severe '
                          'per-sample positive/negative imbalance.')
    ap.add_argument('--asl-gamma-pos', type=float, default=1.0, help='ASL positive focusing exponent.')
    ap.add_argument('--asl-gamma-neg', type=float, default=4.0, help='ASL negative focusing exponent.')
    ap.add_argument('--asl-clip', type=float, default=0.05, help='ASL negative probability-shift margin.')
    ap.add_argument('--run', type=str, default=None, dest='run_filter',
                     help='Only report test steps whose filename contains this substring, e.g. "ssh_apt" -- '
                          'narrows what gets PRINTED/scored in the final table, not what the model trains '
                          'on (training still holds out the full --test-scenario as usual).')
    ap.add_argument('--temp', type=float, default=0.07, help='Test-time logit-scale override.')
    ap.add_argument('--skip-labels', action='store_true', help='Skip step 1 (step_tactics.json refresh).')
    ap.add_argument('--skip-templates', action='store_true', help='Skip step 2 (template generation).')
    ap.add_argument('--skip-train', action='store_true', help='Skip step 3 (training).')
    ap.add_argument('--skip-test', action='store_true', help='Skip step 4 (testing).')
    args = ap.parse_args()

    if args.max_chunks is not None:
        train_mod.MAX_CHUNKS = args.max_chunks

    run_tag = args.run_tag
    if run_tag is None:
        if args.test_scenario:
            run_tag = 'scenario{}'.format(args.test_scenario)
        elif args.test_file:
            run_tag = args.test_file

    if not args.skip_labels:
        header('Step 1/4: Refreshing technique/tactic ground-truth labels (step_tactics.json)')
        step_tactic_lookup.main()

    if not args.skip_templates:
        header('Step 2/4: Generating technique templates')
        generate_technique_templates.main()

    if not args.skip_train:
        header('Step 3/4: Training (text + wide features, technique prototypes)')
        train_mod.run_contrastive_train(
            test_file_match=args.test_file, test_scenario=args.test_scenario,
            k_per_technique=args.k_per_technique,
            test_size=args.test_size, split_seed=args.split_seed, run_tag=run_tag,
            template_dir=args.template_dir,
            stratified=args.stratified, lambda_uniform=args.lambda_uniform,
            class_reweight=args.class_reweight, reweight_cap=args.reweight_cap,
            sequences_dir=args.sequences_dir,
            lr=args.lr, full_batch=args.full_batch, patience=args.patience,
            n_freeze=args.n_freeze, dropout=args.dropout,
            zoomer_scope=args.zoomer_scope, use_wide=args.use_wide,
            loss_fn=args.loss_fn, asl_gamma_pos=args.asl_gamma_pos,
            asl_gamma_neg=args.asl_gamma_neg, asl_clip=args.asl_clip,
            grouped_split=args.grouped_split,
        )

    if not args.skip_test:
        header('Step 4/4: Testing (technique / tactic / stage accuracy)')
        test_mod.run(
            test_file_match=args.test_file, test_scenario=args.test_scenario, temp=args.temp,
            test_size=args.test_size, split_seed=args.split_seed, run_tag=run_tag,
            template_dir=args.template_dir, sequences_dir=args.sequences_dir,
            zoomer_scope=args.test_zoomer_scope, strict_native=args.strict_native,
            run_filter=args.run_filter,
        )


if __name__ == '__main__':
    main()
