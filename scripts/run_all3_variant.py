import sys
import traceback

import run_camlds_matcher_multiseed as R

tag_suffix = sys.argv[1] if len(sys.argv) > 1 else ''
class_reweight = sys.argv[2].lower() != 'false' if len(sys.argv) > 2 else False

results = {}
for seed in (0, 1, 2):
    print()
    print('=' * 70)
    print('SEED {} -- tag_suffix={!r}, class_reweight={}'.format(seed, tag_suffix, class_reweight))
    print('=' * 70)
    try:
        results[seed] = R.run_one_seed(seed, class_reweight=class_reweight, tag_suffix=tag_suffix)
        print('Seed {} -> {}'.format(seed, results[seed]))
    except Exception as e:
        print('Seed {} CRASHED: {}'.format(seed, e))
        traceback.print_exc()
        results[seed] = None

print()
print('=' * 70)
print('Summary (tag_suffix={!r}, class_reweight={})'.format(tag_suffix, class_reweight))
print('=' * 70)
for seed in (0, 1, 2):
    r = results[seed]
    if r is None:
        print('  Seed {}: CRASHED'.format(seed))
    else:
        print('  Seed {}: lrap={:.1f}%  aupr={:.1f}%  wrong={}/{}'.format(
            seed, r['lrap'] * 100, r['aupr'] * 100, r['n_wrong_top3'], r['n_total']))
