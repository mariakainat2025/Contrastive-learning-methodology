import os
import subprocess

from scoring_common import build_rows, accuracy_block, RESULTS_DIR

OUT_BASENAME = 'scenario_report'

RUNS = [
    ('CAM-LDS -- technique + tactic text matcher (multi-label only)', [
        ('2_cron_only_basetech', 'scenario 2 / cron', [
            '6 of this run\'s wrong technique predictions (`2\\_cron-6/8/10/13/16/17`) are not a '
            'model failure: those steps\' sequence text is byte-identical background noise '
            '(`zmcontroller connect ... network address`) pulled from the wrong host, with the '
            'real attacker command never making it into the sequence -- a known data-pipeline bug.',
            '2 more wrong (`2\\_cron-20/21`) are explained by T1046 having zero training examples '
            'under this specific holdout (every T1046 occurrence fell into the held-out set), so no '
            'prototype exists for it at all -- a coverage gap, not a model weakness.',
            'So 8 of this run\'s 14 technique-level misses are structural; only 6 are genuine confusions.',
        ]),
        ('ssh_apt_only_basetech', 'scenario 3 / ssh\\_apt', []),
        ('scenario4_only_basetech', 'scenario 4', [
            'Widened test set (19 steps, was 17): technique top-1 36.8%, top-3 47.4%, LRAP 51.7%, AUPR 38.7%.',
            'T1205 has zero training examples under this holdout (every T1205 occurrence -- '
            '`4-3, 4-9, 4-16` -- fell into the held-out scenario), so those 3 steps have no '
            'prototype signal for one of their true labels -- a coverage gap, not a model weakness.',
            'T1599 (`4-17, 4-18`) is entirely outside the 36-technique scope -- no prototype on '
            'either side, ever -- structurally unwinnable, same category as the out-of-scope steps '
            'found in `ssh_apt` and `2_cron`.',
        ]),
        ('6_macro_binary_only_basetech', 'scenario 6, macro\\_binary', [
            'Widened test set (18 steps, was 16): technique top-1 16.7%, top-3 27.8%, LRAP 39.8%, '
            'AUPR 29.4% (13/18 wrong in top-3) -- still the weakest technique-level run. Tactic '
            'level is similarly weak (top-1 22.2%), though stage level still holds up (top-3 83.3%).',
            'T1010 and T1204 (`6\\_macro\\_binary-19` and `-1/-2`) are entirely outside the '
            '36-technique scope -- no prototype on either side, ever.',
        ]),
    ]),
]


def counts_of(rows):
    return {
        'correct': sum(1 for r in rows if r['verdict'] == 'correct'),
        'partial': sum(1 for r in rows if r['verdict'] == 'partial'),
        'wrong':   sum(1 for r in rows if r['verdict'] == 'wrong'),
    }


def md_row(label, n, counts, out):
    def trip(lvl):
        c = counts[lvl]
        return '{} | {} | {}'.format(c['correct'], c['partial'], c['wrong'])
    tech_top1 = '{:.1f}%'.format(out.get('tech_acc_top1', 0) * 100)
    return '| {} | {} | {} | {} | {} | {} |'.format(label, n, trip('technique'), trip('tactic'), trip('stage'), tech_top1)


def build_section(title, runs):
    lines = []
    lines.append('## {}'.format(title))
    lines.append('')
    lines.append('(Technique/Tactic/Stage top-3 throughout.)')
    lines.append('')
    lines.append('| Scenario | N | Tech C | Tech P | Tech W | Tactic C | Tactic P | Tactic W | Stage C | Stage P | Stage W | Tech top-1 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    scored = []
    for tag, label, notes in runs:
        data = build_rows(tag)
        n = len(data['technique'])
        counts = {lvl: counts_of(data[lvl]) for lvl in ('technique', 'tactic', 'stage')}
        lines.append(md_row(label, n, counts, data['out']))
        scored.append((tag, label, notes, data['out']))
    lines.append('')

    for tag, label, notes, out in scored:
        lines.append('### {}'.format(label))
        lines.append('')
        lines.append('```')
        lines.append(accuracy_block(out))
        lines.append('```')
        lines.append('')
        if notes:
            for note in notes:
                lines.append('- {}'.format(note))
            lines.append('')
    return '\n'.join(lines)


def build_markdown():
    lines = []
    lines.append('# CAM-LDS / ZOOMER scenario-by-scenario results')
    lines.append('')
    lines.append(
        'Base-technique level (not sub-technique) throughout, 36-class fixed scope, '
        'scenario/run-group holdout (everything else stays in training). '
        '**Correct**/**Partial**/**Wrong** per level: when a step has no more true labels than '
        'prediction slots (N), judged on recall (correct = every true label was found). When a step '
        'has MORE true labels than slots (e.g. 5 true tactics, top-3 shown), full recall is '
        'structurally impossible, so judged on precision instead (correct = every one of the N '
        'guesses was right, even though some true labels were necessarily left out). Wrong = none '
        'of the N guesses were right, either way. '
        'Technique -> tactic -> stage derived by walking the technique ranking '
        '(same expansion the test script itself uses, first-occurrence-wins for '
        'techniques/tactics that map to more than one tactic/stage).'
    )
    lines.append('')
    for title, runs in RUNS:
        lines.append(build_section(title, runs))
    return '\n'.join(lines)


def main():
    md = build_markdown()
    md_path = os.path.join(RESULTS_DIR, OUT_BASENAME + '.md')
    with open(md_path, 'w') as f:
        f.write(md)
    print('Wrote {}'.format(md_path))

    pdf_path = os.path.join(RESULTS_DIR, OUT_BASENAME + '.pdf')
    subprocess.run(['pandoc', md_path, '-o', pdf_path, '-V', 'geometry:margin=0.7in'], check=True)
    print('Rendered {}'.format(pdf_path))


if __name__ == '__main__':
    main()
