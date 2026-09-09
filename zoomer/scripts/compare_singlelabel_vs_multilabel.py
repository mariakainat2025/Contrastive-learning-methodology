import json
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZOOMER_DIR = os.path.dirname(SCRIPTS_DIR)
RESULTS_DIR = os.path.join(ZOOMER_DIR, 'results')

SINGLELABEL_MATCHED_SUMMARY = os.path.join(RESULTS_DIR, 'summary_singlelabel_matched.json')
MULTILABEL_SUMMARY = os.path.join(RESULTS_DIR, 'summary_multilabel.json')

KEYS = ('tech_acc', 'tac_acc', 'lrap', 'aupr')
LABELS = {'tech_acc': 'Technique Accuracy', 'tac_acc': 'Tactic Accuracy', 'lrap': 'LRAP', 'aupr': 'AUPR (macro)'}


def main():
    with open(SINGLELABEL_MATCHED_SUMMARY) as f:
        single = json.load(f)
    with open(MULTILABEL_SUMMARY) as f:
        multi = json.load(f)

    if single['seeds'] != multi['seeds']:
        print('WARNING: seed lists differ ({} vs {}) -- results below are not apples-to-apples.'.format(
            single['seeds'], multi['seeds']))

    print('Single-label vs multi-label, SAME test instances per seed (single-label-matched to multi-label\'s split):')
    print()
    header = '{:22s} {:>22s} {:>22s} {:>12s}'.format('Metric', 'Single-label-matched', 'Multi-label', 'Delta')
    print(header)
    print('-' * len(header))
    for key in KEYS:
        s_mean = single[key]['mean']
        m_mean = multi[key]['mean']
        s_std = single[key]['std']
        m_std = multi[key]['std']
        delta = m_mean - s_mean
        print('{:22s} {:>14.1f}% +/- {:<4.1f} {:>14.1f}% +/- {:<4.1f} {:>+11.1f}%'.format(
            LABELS[key], s_mean, s_std, m_mean, m_std, delta))


if __name__ == '__main__':
    main()
