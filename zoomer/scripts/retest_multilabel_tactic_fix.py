"""
Re-runs ONLY the test side of Test_TTP_Recognition_Multilabel.py for seeds 0/1/2, using
the already-trained checkpoints -- no retraining needed, since the tactic-derivation fix
(technique score now counts toward ALL its real tactics, not just the alphabetically-first
one) is entirely in test-time scoring, and training never touched tactics at all.
"""
import json
import os

from Test_TTP_Recognition_Multilabel import main as test_main

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZOOMER_DIR = os.path.dirname(SCRIPTS_DIR)
RESULTS_DIR = os.path.join(ZOOMER_DIR, 'results')

SEEDS = (0, 1, 2)


def result_path(seed):
    return os.path.join(RESULTS_DIR, 'test_results_multilabel_seed{}.json'.format(seed))


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for seed in SEEDS:
        print()
        print('=' * 70)
        print('[multilabel retest] SEED {}  (existing checkpoint, corrected tactic scoring)'.format(seed))
        print('=' * 70)
        metrics = test_main(seed)
        path = result_path(seed)
        old_lrap = old_aupr = None
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
            old_lrap, old_aupr = old.get('lrap'), old.get('aupr')
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        if old_lrap is not None:
            print('  lrap {:.4f} -> {:.4f}   aupr {:.4f} -> {:.4f}'.format(
                old_lrap, metrics['lrap'], old_aupr, metrics['aupr']))
        print('Saved -> {}'.format(path))
