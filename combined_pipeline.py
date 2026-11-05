import argparse
import os
import subprocess
import sys

CAM_LDS_DIR = '/csse/research/contructive-learning/CAM-LDS'
KNOWHOW_DIR = os.path.join(CAM_LDS_DIR, 'know-how-cam-lds')
KNOWHOW_GIOCS_DIR = os.path.join(KNOWHOW_DIR, 'Giocs')
CAMLDS_DIR = os.path.join(CAM_LDS_DIR, 'technique', 'scripts')
ZOOMER_DIR = os.path.join(CAM_LDS_DIR, 'zoomer', 'scripts')

SEQUENCES_RAW_DIR = os.path.join(CAM_LDS_DIR, 'sequences_raw')
DEMO_SEQ_FILE = os.path.join(SEQUENCES_RAW_DIR, 'collection', 'T1005-000',
                              'sequence_2_cron-15_videoserver.json')
DEMO_OUT_DIR = '/tmp'

CTI_REPORTS_DIR = os.path.join(CAM_LDS_DIR, 'knowhow', 'CTI Reports')
ATOMIC_DIR = os.path.join(CAM_LDS_DIR, 'zoomer', 'atomic-red-team', 'atomics')
KNOWHOW_CKD_FILE = os.path.join(CAM_LDS_DIR, 'knowhow', 'KNOWHOW-seng402',
                                 'meanshift_clustered_phrases.json')

KNOWHOW_RESULTS_GLOB = os.path.join(KNOWHOW_DIR, 'results', 'ckd_test_results_*.json')
ZOOMER_RESULTS_GLOB = os.path.join(CAM_LDS_DIR, 'zoomer', 'results', '*.json')
CAMLDS_RESULTS_GLOB = os.path.join(CAM_LDS_DIR, 'technique', 'results',
                                    'camlds_technique_wide_test_results_*.json')

STAGE_NAMES = [
    'Log Construction', 'Log Simplification', 'Logs Embedding', '2nd Source',
    '2nd Source Processing and Embedding', 'Training', 'Testing', 'Scoring',
]


def header(stage_num, title):
    print()
    print('=' * 78)
    print('  Stage {}/8: {}'.format(stage_num, title))
    print('=' * 78)


def method_header(name):
    print()
    print('-- {} --'.format(name))


def run(args_list, cwd):
    print('  $ (cd {} && {})'.format(cwd, ' '.join(args_list)))
    subprocess.run(args_list, cwd=cwd, check=True)


def py(code, cwd):
    run([sys.executable, '-c', code], cwd=cwd)


def none(reason):
    print('  none -- {}'.format(reason))


class Args:
    def __init__(self, ns):
        self.skip_knowhow = ns.skip_knowhow
        self.skip_zoomer = ns.skip_zoomer
        self.skip_camlds = ns.skip_camlds
        self.only_stage = ns.only_stage
        self.knowhow_scenario = ns.knowhow_scenario
        self.knowhow_run_tag = ns.knowhow_run_tag
        self.zoomer_test_file = ns.zoomer_test_file
        self.zoomer_fixed_scope = ns.zoomer_fixed_scope
        self.zoomer_run_tag = ns.zoomer_run_tag
        self.zoomer_wide_features = ns.zoomer_wide_features
        self.camlds_test_file = ns.camlds_test_file
        self.camlds_run_tag = ns.camlds_run_tag
        self.camlds_wide_features = ns.camlds_wide_features
        self.camlds_max_chunks = ns.camlds_max_chunks

    def wants(self, stage_name):
        return self.only_stage is None or self.only_stage == stage_name


def stage_log_construction(a):
    header(1, 'Log Construction')
    if not a.wants('Log Construction'):
        return
    method_header('KnowHow')
    if a.skip_knowhow:
        none('--skip-knowhow')
    else:
        py("import glob; print('event-level sequence files on disk:', "
           "len(glob.glob('{}/*/*/sequence_*.json')))".format(SEQUENCES_RAW_DIR), cwd=KNOWHOW_DIR)
    method_header('ZOOMER')
    if a.skip_zoomer:
        none('--skip-zoomer')
    else:
        py("import glob; print('provenance graph files on disk:', "
           "len(glob.glob('{}/graphs/*/*/*.json')))".format(CAM_LDS_DIR), cwd=ZOOMER_DIR)
    method_header('CAM-LDS')
    if a.skip_camlds:
        none('--skip-camlds')
    else:
        py("import glob; print('provenance graph files on disk:', "
           "len(glob.glob('{}/graphs/*/*/*.json')))".format(CAM_LDS_DIR), cwd=CAMLDS_DIR)


def stage_log_simplification(a):
    header(2, 'Log Simplification')
    if not a.wants('Log Simplification'):
        return
    method_header('KnowHow')
    if a.skip_knowhow:
        none('--skip-knowhow')
    else:
        run([sys.executable, 'show_simplification_example.py'], cwd=KNOWHOW_GIOCS_DIR)
    method_header('ZOOMER')
    if a.skip_zoomer:
        none('--skip-zoomer')
    else:
        py("import parse_audit as pa; from Feature_Initialization import MASTER_EDGE_TYPES; "
           "print('Raw edge-type vocabulary includes', len(pa.RARE_EDGE_TYPES), "
           "'rare/noisy types, filtered out at parse time:'); "
           "print(' ', sorted(pa.RARE_EDGE_TYPES)); "
           "print('Kept ->', len(MASTER_EDGE_TYPES), "
           "'edge types feed into the real edge-type wide feature.')", cwd=ZOOMER_DIR)
    method_header('CAM-LDS')
    if a.skip_camlds:
        none('--skip-camlds')
    else:
        run([sys.executable, os.path.join(KNOWHOW_DIR, 'generalize_raw_sequence.py'),
             '--file', DEMO_SEQ_FILE,
             '--output', os.path.join(DEMO_OUT_DIR, 'combined_pipeline_demo_generalized.json')],
            cwd=CAMLDS_DIR)


def stage_logs_embedding(a):
    header(3, 'Logs Embedding')
    if not a.wants('Logs Embedding'):
        return
    method_header('KnowHow')
    if a.skip_knowhow:
        none('--skip-knowhow')
    else:
        run([sys.executable, 'score_sequence_against_ckd.py', '--file', DEMO_SEQ_FILE,
             '--output', os.path.join(DEMO_OUT_DIR, 'combined_pipeline_demo_ckd_scored.json')],
            cwd=KNOWHOW_DIR)
    method_header('ZOOMER')
    if a.skip_zoomer:
        none('--skip-zoomer')
    else:
        run([sys.executable, 'Deep_Model.py'], cwd=ZOOMER_DIR)
    method_header('CAM-LDS')
    if a.skip_camlds:
        none('--skip-camlds')
    else:
        run([sys.executable, 'encoder_utils.py', '--file', DEMO_SEQ_FILE], cwd=CAMLDS_DIR)


def stage_2nd_source(a):
    header(4, '2nd Source')
    if not a.wants('2nd Source'):
        return
    method_header('KnowHow')
    if a.skip_knowhow:
        none('--skip-knowhow')
    else:
        py("import os; print('CTI reports on disk:', len(os.listdir('{}')))".format(CTI_REPORTS_DIR),
           cwd=KNOWHOW_DIR)
    method_header('ZOOMER')
    if a.skip_zoomer:
        none('--skip-zoomer')
    else:
        py("import os; print('Atomic Red Team technique folders on disk:', "
           "len([x for x in os.listdir('{}') if x.startswith('T')]))".format(ATOMIC_DIR), cwd=ZOOMER_DIR)
    method_header('CAM-LDS')
    if a.skip_camlds:
        none('--skip-camlds')
    else:
        py("import generate_technique_templates as g; "
           "print('MITRE technique descriptions in scope:', len(g.OUR_TECHNIQUES))", cwd=CAMLDS_DIR)


def stage_2nd_source_processing(a):
    header(5, '2nd Source Processing and Embedding')
    if not a.wants('2nd Source Processing and Embedding'):
        return
    method_header('KnowHow')
    if a.skip_knowhow:
        none('--skip-knowhow')
    else:
        run([sys.executable, 'extract_gioc_from_eset_reports.py', '--only', 'new-eset'],
            cwd=KNOWHOW_GIOCS_DIR)
        run([sys.executable, 'extract_gioc_from_procedure_examples.py'], cwd=KNOWHOW_GIOCS_DIR)
        run([sys.executable, '-m', 'knowhow.cal_tech_dic', '-m'], cwd=KNOWHOW_GIOCS_DIR)
    method_header('ZOOMER')
    if a.skip_zoomer:
        none('--skip-zoomer')
    else:
        run([sys.executable, 'extract_atomic_iocs.py'], cwd=ZOOMER_DIR)
    method_header('CAM-LDS')
    if a.skip_camlds:
        none('--skip-camlds')
    else:
        run([sys.executable, 'main.py', '--skip-labels', '--skip-train', '--skip-test'] +
            (['--test-file', a.camlds_test_file] if a.camlds_test_file else []),
            cwd=CAMLDS_DIR)
        print('  (template TEXT written; actual RoBERTa embedding of it happens inside Training, stage 6)')


def stage_training(a):
    header(6, 'Training')
    if not a.wants('Training'):
        return
    method_header('KnowHow')
    if a.skip_knowhow:
        none('--skip-knowhow')
    else:
        py("import json; d = json.load(open('{}')); "
           "print('No training step -- reading the real CKD built in stage 5 instead:'); "
           "print(' ', len(d), 'clusters,', sum(len(v['points']) for v in d.values()), "
           "'gIoC phrases total')".format(KNOWHOW_CKD_FILE), cwd=KNOWHOW_DIR)
    method_header('ZOOMER')
    if a.skip_zoomer:
        none('--skip-zoomer')
    else:
        cmd = [sys.executable, 'main.py', '--skip-test', '--wide-features', a.zoomer_wide_features]
        if a.zoomer_test_file:
            cmd += ['--scenario', a.zoomer_test_file]
        if a.zoomer_fixed_scope:
            cmd.append('--fixed-scope')
        if a.zoomer_run_tag:
            cmd += ['--run-tag', a.zoomer_run_tag]
        run(cmd, cwd=ZOOMER_DIR)
    method_header('CAM-LDS')
    if a.skip_camlds:
        none('--skip-camlds')
    else:
        cmd = [sys.executable, 'main.py', '--skip-labels', '--skip-templates', '--skip-test',
               '--wide-features', a.camlds_wide_features]
        if a.camlds_test_file:
            cmd += ['--test-file', a.camlds_test_file]
        if a.camlds_run_tag:
            cmd += ['--run-tag', a.camlds_run_tag]
        if a.camlds_max_chunks:
            cmd += ['--max-chunks', str(a.camlds_max_chunks)]
        run(cmd, cwd=CAMLDS_DIR)


def stage_testing(a):
    header(7, 'Testing')
    if not a.wants('Testing'):
        return
    method_header('KnowHow')
    if a.skip_knowhow:
        none('--skip-knowhow')
    else:
        cmd = [sys.executable, 'score_scenario_technique_table.py',
               '--scenario', a.knowhow_scenario]
        if a.knowhow_run_tag:
            cmd += ['--run-tag', a.knowhow_run_tag]
        run(cmd, cwd=KNOWHOW_DIR)
        print('  (this same call also computes Scoring -- stage 8 below reads its output)')
    method_header('ZOOMER')
    if a.skip_zoomer:
        none('--skip-zoomer')
    else:
        cmd = [sys.executable, 'main.py', '--skip-train']
        if a.zoomer_test_file:
            cmd += ['--scenario', a.zoomer_test_file]
        if a.zoomer_fixed_scope:
            cmd.append('--fixed-scope')
        if a.zoomer_run_tag:
            cmd += ['--run-tag', a.zoomer_run_tag]
        run(cmd, cwd=ZOOMER_DIR)
        print('  (this same call also computes Scoring -- stage 8 below reads its output)')
    method_header('CAM-LDS')
    if a.skip_camlds:
        none('--skip-camlds')
    else:
        cmd = [sys.executable, 'main.py', '--skip-labels', '--skip-templates', '--skip-train']
        if a.camlds_test_file:
            cmd += ['--test-file', a.camlds_test_file]
        if a.camlds_run_tag:
            cmd += ['--run-tag', a.camlds_run_tag]
        run(cmd, cwd=CAMLDS_DIR)
        print('  (this same call also computes Scoring -- stage 8 below reads its output)')


def _read_latest_json(glob_pattern):
    return ("import glob, json, os; "
            "files = sorted(glob.glob('{}'), key=os.path.getmtime); ".format(glob_pattern) +
            "assert files, 'no results file found yet -- run stage 7 (Testing) first'; "
            "d = json.load(open(files[-1])); "
            "print('Reading real scores written by stage 7:', os.path.basename(files[-1])); ")


def stage_scoring(a):
    header(8, 'Scoring')
    if not a.wants('Scoring'):
        return
    method_header('KnowHow')
    if a.skip_knowhow:
        none('--skip-knowhow')
    else:
        py(_read_latest_json(KNOWHOW_RESULTS_GLOB) +
           "print(' tech_acc_top1={:.3f} tac_acc_top1={:.3f} stage_acc_top1={:.3f} "
           "lrap={:.3f} aupr={:.3f}'.format(d['tech_acc_top1'], d['tac_acc_top1'], "
           "d['stage_acc_top1'], d['lrap'], d['aupr']))", cwd=KNOWHOW_DIR)
    method_header('ZOOMER')
    if a.skip_zoomer:
        none('--skip-zoomer')
    else:
        py(_read_latest_json(ZOOMER_RESULTS_GLOB) +
           "print(' tech_acc={:.3f} tac_acc={:.3f} lrap={:.3f} aupr={:.3f}'.format("
           "d['tech_acc'], d['tac_acc'], d['lrap'], d['aupr']))", cwd=ZOOMER_DIR)
    method_header('CAM-LDS')
    if a.skip_camlds:
        none('--skip-camlds')
    else:
        py(_read_latest_json(CAMLDS_RESULTS_GLOB) +
           "print(' tech_acc_top1={:.3f} tac_acc_top1={:.3f} stage_acc_top1={:.3f} "
           "lrap={:.3f} aupr={:.3f}'.format(d['tech_acc_top1'], d['tac_acc_top1'], "
           "d['stage_acc_top1'], d['lrap'], d['aupr']))", cwd=CAMLDS_DIR)


STAGE_FNS = [
    stage_log_construction, stage_log_simplification, stage_logs_embedding,
    stage_2nd_source, stage_2nd_source_processing, stage_training,
    stage_testing, stage_scoring,
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-knowhow', action='store_true')
    ap.add_argument('--skip-zoomer', action='store_true')
    ap.add_argument('--skip-camlds', action='store_true')
    ap.add_argument('--only-stage', type=str, default=None, choices=STAGE_NAMES,
                     help='Run just this one stage (for all non-skipped methods) instead of all 8.')

    ap.add_argument('--knowhow-scenario', type=str, default='4',
                     help='KnowHow scenario number for stage 7/8 (default: 4).')
    ap.add_argument('--knowhow-run-tag', type=str, default=None)

    ap.add_argument('--zoomer-test-file', type=str, default=None,
                     help='e.g. "6_macro_binary" -- held-out scenario for ZOOMER training+testing.')
    ap.add_argument('--zoomer-fixed-scope', action='store_true')
    ap.add_argument('--zoomer-run-tag', type=str, default=None)
    ap.add_argument('--zoomer-wide-features', type=str, default='all', choices=['all', 'node_edge'])

    ap.add_argument('--camlds-test-file', type=str, default=None,
                     help='e.g. "6_macro_binary" -- held-out scenario for CAM-LDS training+testing.')
    ap.add_argument('--camlds-run-tag', type=str, default=None)
    ap.add_argument('--camlds-wide-features', type=str, default='all', choices=['all', 'node_edge'])
    ap.add_argument('--camlds-max-chunks', type=int, default=200,
                     help='Caps long-sequence token chunks to avoid CUDA OOM (default: 200).')

    args = Args(ap.parse_args())

    for fn in STAGE_FNS:
        fn(args)

    print()
    print('Done.')


if __name__ == '__main__':
    main()
