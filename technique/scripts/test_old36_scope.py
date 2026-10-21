import argparse

import test_camlds_matcher_technique_wide as test_mod

ORIGINAL_36 = [
    'T1003', 'T1016', 'T1018', 'T1021', 'T1033', 'T1036',
    'T1041', 'T1046', 'T1053', 'T1056', 'T1057',
    'T1059', 'T1070', 'T1071', 'T1072', 'T1078',
    'T1082', 'T1083', 'T1098', 'T1105', 'T1110',
    'T1115', 'T1176', 'T1201', 'T1205', 'T1218', 'T1219',
    'T1222', 'T1543', 'T1546', 'T1547', 'T1548', 'T1556',
    'T1564', 'T1566', 'T1574',
]
assert len(ORIGINAL_36) == 36

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-scenario', type=str, default=None)
    ap.add_argument('--run', type=str, default=None, dest='run_filter')
    ap.add_argument('--run-tag', type=str, required=True)
    args = ap.parse_args()

    test_mod.run(
        test_scenario=args.test_scenario,
        run_tag=args.run_tag,
        run_filter=args.run_filter,
        candidate_cap=ORIGINAL_36,
    )
