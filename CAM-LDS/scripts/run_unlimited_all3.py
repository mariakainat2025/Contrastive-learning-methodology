import traceback

import run_camlds_matcher_multiseed as R

results = {}
for seed in (0, 1, 2):
    print()
    print('=' * 70)
    print('SEED {} -- MAX_CHUNKS=None (unlimited), class_reweight=False'.format(seed))
    print('=' * 70)
    try:
        results[seed] = R.run_one_seed(seed, class_reweight=False, tag_suffix='_unlimited')
        print('Seed {} -> {}'.format(seed, results[seed]))
    except Exception as e:
        print('Seed {} CRASHED: {}'.format(seed, e))
        traceback.print_exc()
        results[seed] = None

print()
print('=' * 70)
print('Summary')
print('=' * 70)
for seed in (0, 1, 2):
    r = results[seed]
    if r is None:
        print('  Seed {}: CRASHED'.format(seed))
    else:
        print('  Seed {}: lrap={:.1f}%  aupr={:.1f}%  wrong={}/{}'.format(
            seed, r['lrap'] * 100, r['aupr'] * 100, r['n_wrong_top3'], r['n_total']))
