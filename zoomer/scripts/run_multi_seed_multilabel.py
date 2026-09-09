import json
import os
import statistics

from Train_TTP_Recognition_Multilabel import main as train_main
from Test_TTP_Recognition_Multilabel import main as test_main

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZOOMER_DIR = os.path.dirname(SCRIPTS_DIR)
RESULTS_DIR = os.path.join(ZOOMER_DIR, 'results')

SEEDS = (0, 1, 2)

def result_path(seed):
    return os.path.join(RESULTS_DIR, 'test_results_multilabel_seed{}.json'.format(seed))

def summary_path():
    return os.path.join(RESULTS_DIR, 'summary_multilabel.json')

def main(seeds=SEEDS):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_metrics = []
    for seed in seeds:
        print()
        print('=' * 70)
        print('[multilabel] SEED {}  (independent random split + refit bins/masks + retrain)'.format(seed))
        print('=' * 70)
        train_main(seed)
        metrics = test_main(seed)
        with open(result_path(seed), 'w') as f:
            json.dump(metrics, f, indent=2)
        print('Saved -> {}'.format(result_path(seed)))
        all_metrics.append(metrics)

    print()
    print('=' * 70)
    print('[multilabel] Summary across {} seeds: {}'.format(len(seeds), list(seeds)))
    print('=' * 70)
    keys = ('tech_acc', 'tac_acc', 'lrap', 'aupr')
    labels = {'tech_acc': 'Technique Accuracy', 'tac_acc': 'Tactic Accuracy',
              'lrap': 'LRAP', 'aupr': 'AUPR (macro)'}
    summary = {'seeds': list(seeds)}
    for key in keys:
        values = [m[key] * 100 for m in all_metrics]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        per_seed = ', '.join('seed{}={:.1f}%'.format(m['seed'], v) for (m, v) in zip(all_metrics, values))
        print('{:22s}: {:5.1f}% +/- {:4.1f}%   ({})'.format(labels[key], mean, std, per_seed))
        summary[key] = {'mean': mean, 'std': std, 'per_seed': {m['seed']: v for (m, v) in zip(all_metrics, values)}}
    with open(summary_path(), 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print('Saved -> {}'.format(summary_path()))
    return all_metrics

if __name__ == '__main__':
    main()
