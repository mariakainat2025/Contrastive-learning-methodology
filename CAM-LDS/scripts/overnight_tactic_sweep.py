
import os
import re
import sys
import json
import time
import subprocess
import traceback

CAM_LDS_DIR   = '/csse/research/contructive-learning/CAM-LDS'
SCRIPTS_DIR   = os.path.join(CAM_LDS_DIR, 'scripts')
RESULTS_DIR   = os.path.join(CAM_LDS_DIR, 'results')
VENV_PYTHON   = '/csse/research/contructive-learning/.venv/bin/python3'
TRAIN_MATCHER_PATH = os.path.join(SCRIPTS_DIR, 'train_camlds_matcher.py')
STEP_LOOKUP_PATH   = os.path.join(SCRIPTS_DIR, 'step_tactic_lookup.py')
MAIN_PY_PATH       = os.path.join(CAM_LDS_DIR, 'main.py')
SUMMARY_JSON        = os.path.join(CAM_LDS_DIR, 'overnight_sweep_summary.json')
COMPARISON_MD        = os.path.join(RESULTS_DIR, 'tactic_expansion_comparison.md')

BASE_9 = ['privilege_escalation', 'command_and_control', 'credential_access', 'execution', 'initial_access',
          'lateral_movement', 'persistence', 'stealth', 'defense_impairment']

ADD_ORDER = ['reconnaissance', 'impact', 'discovery', 'collection', 'exfiltration']

SPLIT_SEED = 1
TEST_SIZE  = 0.1


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print('[{}] {}'.format(ts, msg), flush=True)


def rewrite_list_literal(path, var_name, new_values):
    with open(path) as f:
        src = f.read()
    pattern = re.compile(r'{}\s*=\s*\[[^\]]*\]'.format(re.escape(var_name)), re.DOTALL)
    quote = "'" if var_name == 'TACTIC_FOLDERS' else '"'
    items = ', '.join('{q}{v}{q}'.format(q=quote, v=v) for v in new_values)
    replacement = '{} = [{}]'.format(var_name, items)
    new_src, n = pattern.subn(replacement, src, count=1)
    if n != 1:
        raise RuntimeError('Failed to find/replace {} in {}'.format(var_name, path))
    with open(path, 'w') as f:
        f.write(new_src)


def wipe_cache():
    for sub in ['parser/parsed_events', 'parser/subgraphs', 'sequences']:
        d = os.path.join(CAM_LDS_DIR, sub)
        for fn in os.listdir(d) if os.path.isdir(d) else []:
            fp = os.path.join(d, fn)
            if os.path.isdir(fp):
                subprocess.run(['rm', '-rf', fp])
            else:
                os.remove(fp)
    cache_dir = os.path.join(CAM_LDS_DIR, 'cache')
    if os.path.isdir(cache_dir):
        for fn in os.listdir(cache_dir):
            if fn.endswith('.pkl'):
                os.remove(os.path.join(cache_dir, fn))


def run_cmd(cmd, cwd, log_prefix):
    log('{} :: RUNNING: {}'.format(log_prefix, ' '.join(cmd)))
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        line = line.rstrip('\n')
        lines.append(line)
        print('{} | {}'.format(log_prefix, line), flush=True)
    proc.wait()
    log('{} :: exit={}'.format(log_prefix, proc.returncode))
    if proc.returncode != 0:
        raise RuntimeError('{} failed (exit {})'.format(log_prefix, proc.returncode))
    return '\n'.join(lines)


def load_json_safe(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def append_comparison_row(n_tactics, tactic_added, method, loss_fix, lrap, aupr, wrong_str, collapsed):
    row = '| {} | {} | {} | {} | {} | {} | {} | {} |\n'.format(
        n_tactics, tactic_added, method, loss_fix,
        '{:.1f}%'.format(lrap * 100) if lrap is not None else 'ERROR',
        '{:.1f}%'.format(aupr * 100) if aupr is not None else 'ERROR',
        wrong_str, collapsed)
    with open(COMPARISON_MD, 'a') as f:
        f.write(row)


def append_summary(entry):
    data = []
    if os.path.exists(SUMMARY_JSON):
        with open(SUMMARY_JSON) as f:
            data = json.load(f)
    data.append(entry)
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(data, f, indent=2)


def detect_collapse(results):
    if not results or len(results) < 3:
        return 'unknown'
    from collections import Counter
    top1s = [r['ranked'][0]['tactic'] for r in results if r.get('ranked')]
    if not top1s:
        return 'unknown'
    most_common, count = Counter(top1s).most_common(1)[0]
    frac = count / len(top1s)
    return 'YES' if frac >= 0.9 else 'no'


def run_stage(n_tactics, tactics, tactic_added):
    run_tag_base = 'seed{}_{}tactic'.format(SPLIT_SEED, n_tactics)
    log('=' * 80)
    log('STAGE: {} tactics -> adding "{}"  tactics={}'.format(n_tactics, tactic_added, tactics))
    log('=' * 80)

    rewrite_list_literal(TRAIN_MATCHER_PATH, 'TACTIC_FOLDERS', tactics)
    rewrite_list_literal(STEP_LOOKUP_PATH, 'OUR_TACTICS', tactics)
    wipe_cache()

    try:
        run_cmd([VENV_PYTHON, 'main.py',
                 '--split-seed', str(SPLIT_SEED), '--test-size', str(TEST_SIZE),
                 '--run-tag', run_tag_base, '--force'],
                cwd=CAM_LDS_DIR, log_prefix='[{}t][template][main.py]'.format(n_tactics))
        tmpl_results = load_json_safe(os.path.join(RESULTS_DIR, 'camlds_test_results_{}.json'.format(run_tag_base)))
        if tmpl_results:
            collapsed = detect_collapse(tmpl_results['results'])
            append_comparison_row(n_tactics, tactic_added, 'template', 'none',
                                   tmpl_results['lrap'], tmpl_results['aupr'],
                                   '{}/{}'.format(tmpl_results['n_wrong_top1'], tmpl_results['n_total']),
                                   collapsed)
            append_summary({'n_tactics': n_tactics, 'tactic_added': tactic_added, 'method': 'template',
                             'lrap': tmpl_results['lrap'], 'aupr': tmpl_results['aupr'],
                             'n_wrong_top1': tmpl_results['n_wrong_top1'], 'n_total': tmpl_results['n_total'],
                             'collapsed': collapsed})
        else:
            append_comparison_row(n_tactics, tactic_added, 'template', 'none', None, None, 'ERROR', 'ERROR')
    except Exception:
        log('[{}t][template] FAILED:\n{}'.format(n_tactics, traceback.format_exc()))
        append_comparison_row(n_tactics, tactic_added, 'template', 'none', None, None, 'CRASHED', 'ERROR')

    try:
        run_cmd([VENV_PYTHON, 'train_camlds_class_prototype.py',
                 '--proto-mode', 'class', '--split-seed', str(SPLIT_SEED), '--test-size', str(TEST_SIZE),
                 '--run-tag', run_tag_base],
                cwd=SCRIPTS_DIR, log_prefix='[{}t][class][train]'.format(n_tactics))
        run_cmd([VENV_PYTHON, 'test_camlds_class_prototype.py',
                 '--proto-mode', 'class', '--run-tag', run_tag_base,
                 '--split-seed', str(SPLIT_SEED), '--test-size', str(TEST_SIZE)],
                cwd=SCRIPTS_DIR, log_prefix='[{}t][class][test]'.format(n_tactics))
        class_results = load_json_safe(os.path.join(
            RESULTS_DIR, 'camlds_classproto_test_results_{}_class.json'.format(run_tag_base)))
        if class_results:
            collapsed = detect_collapse(class_results['results'])
            append_comparison_row(n_tactics, tactic_added, 'class', 'none',
                                   class_results['lrap'], class_results['aupr'],
                                   '{}/{}'.format(class_results['n_wrong_top1'], class_results['n_total']),
                                   collapsed)
            append_summary({'n_tactics': n_tactics, 'tactic_added': tactic_added, 'method': 'class',
                             'lrap': class_results['lrap'], 'aupr': class_results['aupr'],
                             'n_wrong_top1': class_results['n_wrong_top1'], 'n_total': class_results['n_total'],
                             'collapsed': collapsed})
        else:
            append_comparison_row(n_tactics, tactic_added, 'class', 'none', None, None, 'ERROR', 'ERROR')
    except Exception:
        log('[{}t][class] FAILED:\n{}'.format(n_tactics, traceback.format_exc()))
        append_comparison_row(n_tactics, tactic_added, 'class', 'none', None, None, 'CRASHED', 'ERROR')

    log('STAGE {} tactics COMPLETE.\n'.format(n_tactics))


def main():
    log('Overnight tactic sweep starting. Base 9 tactics: {}'.format(BASE_9))
    log('Add order for 10-14: {}'.format(ADD_ORDER))

    existing = ''
    if os.path.exists(COMPARISON_MD):
        with open(COMPARISON_MD) as f:
            existing = f.read()
    if '## Overnight sweep results' not in existing:
        with open(COMPARISON_MD, 'a') as f:
            f.write('\n\n## Overnight sweep results ({})\n\n'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
            f.write('| # tactics | added tactic | method | loss fix | LRAP | AUPR | wrong top-1 | collapsed? |\n')
            f.write('|---|---|---|---|---|---|---|---|\n')
    else:
        log('Comparison table header already exists -- appending rows without a new header (resumed run).')

    tactics = list(BASE_9)
    run_stage(len(tactics), tactics, 'defense_impairment (base)')

    for next_tactic in ADD_ORDER:
        tactics = tactics + [next_tactic]
        run_stage(len(tactics), tactics, next_tactic)

    log('OVERNIGHT SWEEP COMPLETE. See {} and {}'.format(COMPARISON_MD, SUMMARY_JSON))


if __name__ == '__main__':
    main()

