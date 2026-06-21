
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TACTIC_DATA_ROOT = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data')
OUTPUT_DIR       = os.path.join(TACTIC_DATA_ROOT, 'abstract_sequnce')

ATTACK_FOLDERS = [
    ('browerextension',     'Browser_Extension_Drakon_Dropper'),
    ('backdoor',            'Firefox_Backdoor_Drakon_In_Memory'),
    ('phishing',            'Phishing_Email_With_Link'),
    ('phishing_attachment', 'Phishing_Email_With_Executable_Attachment'),
]

SKIP_PREFIXES = ('stage3_',)

_WEB_PROCS   = {'firefox'}
_MAIL_PROCS  = {'thunderbird'}
_SERV_PROCS  = {'sshd', 'cron'}
_SHELL_PROCS = {'bash', 'sh'}
_UI_PROCS    = {'gnome-terminal'}
_WM_PROCS    = {'fluxbox'}

_USER_PROCESS_PATHS = ['/home/', '/root/', '/tmp/', '/temp/', '/var/tmp/']
_SYSTEM_BIN_PATHS = [
    '/usr/bin/', '/usr/sbin/', '/bin/', '/sbin/',
    '/usr/local/bin/', '/usr/local/sbin/', '/snap/bin/',
]


def _is_internal(ip):
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    if a == 10: return True
    if a == 172 and 16 <= b <= 31: return True
    if a == 192 and b == 168: return True
    if a == 127: return True
    if a == 128 and b == 55: return True
    return False


def abstract_netflow(name):
    parts = name.split('_')

    if parts[0] == 'LOCAL':
        src_port = parts[1] if len(parts) > 1 else '0'
        return 'unreachable_host local:{}'.format(src_port)

    if len(parts) < 4:
        return name

    src_port = parts[1]
    dst_ip   = parts[2]
    dst_port = parts[-1]

    if dst_ip == 'NA' or dst_port in ('null', '0'):
        return 'unreachable_host NA:{}'.format(src_port)

    if _is_internal(dst_ip):
        return 'internal_host {}:{}'.format(dst_ip, dst_port)
    return 'external_host {}:{}'.format(dst_ip, dst_port)


def abstract_process(name):
    if not name:
        return name

    full_name   = name.strip()
    first_token = full_name.split()[0]

    if first_token.startswith('./'):
        return 'user_process {}'.format(full_name)

    stem     = os.path.basename(first_token).lower()
    raw_path = first_token.lower()

    if any(p in raw_path for p in _USER_PROCESS_PATHS):
        return 'user_process {}'.format(full_name)

    if stem in _WEB_PROCS:
        return 'browser_process {}'.format(full_name)
    if stem in _MAIL_PROCS:
        return 'mail_process {}'.format(full_name)
    if stem in _SERV_PROCS:
        return 'service_process {}'.format(full_name)
    if stem in _SHELL_PROCS:
        return 'shell_process {}'.format(full_name)
    if stem in _UI_PROCS:
        return 'ui_process {}'.format(full_name)
    if stem in _WM_PROCS:
        return 'window_manager {}'.format(full_name)

    if any(p in raw_path for p in _SYSTEM_BIN_PATHS):
        return 'utility_process {}'.format(full_name)

    return 'process_{}'.format(full_name)


def abstract_file(path):
    if not path:
        return path
    if path in ('/', '/etc', '/root', '/usr'):
        return 'directory {}'.format(path)

    n        = path.lower()
    basename = os.path.basename(path)

    if '.mozilla' in n and '/home/' in n:
        return 'browser_data {}'.format(path)

    if '.thunderbird' in n:
        return 'mail_data {}'.format(path)

    if '/tmp/' in n or '/var/tmp/' in n:
        return 'temporary_file {}'.format(path)

    if basename.lower().endswith('.cache'):
        return 'cache_file {}'.format(path)

    if '.so' in basename.lower() or '/lib/' in n:
        return 'library_file {}'.format(path)

    if '/var/log/' in n:
        return 'log_file {}'.format(path)

    if n.startswith('/etc/'):
        return 'configuration_file {}'.format(path)

    if n.startswith('/dev/'):
        return 'device_file {}'.format(path)

    if n.startswith('/sys/'):
        return 'system_file {}'.format(path)

    if (n.startswith('/bin/') or n.startswith('/sbin/')
            or basename.lower().endswith('.bin')
            or '/downloads/' in n):
        return 'bin_file {}'.format(path)

    if '/home/' in n:
        return 'user_file {}'.format(path)

    return 'directory {}'.format(path)


def abstract_node(name, node_type):
    if 'SUBJECT' in node_type:
        return abstract_process(name)
    if 'NetFlow' in node_type:
        return abstract_netflow(name)
    if 'FILE' in node_type or 'BLOCK' in node_type:
        return abstract_file(name)
    return name


def build_abstract_sequence(tactic_json_path):
    with open(tactic_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    edges = data.get('edges', [])
    lines = []
    for e in edges:
        src    = abstract_node(e['src_name'], e['src_type'])
        dst    = abstract_node(e['dst_name'], e['dst_type'])
        etype  = e['etype'].replace('EVENT_', '').replace('RECVFROM', 'receivefrom').lower()
        triple = '{} {} {}.'.format(src, etype, dst)
        if 'unreachable_host local:0' in triple:
            continue
        lines.append(triple)

    return {
        'attack'       : data.get('attack',       ''),
        'tactic'       : data.get('tactic',        ''),
        'mitre'        : data.get('mitre',          ''),
        'date'         : data.get('date',            ''),
        'filter'       : data.get('filter',          ''),
        'window_start' : data.get('window_start',    ''),
        'window_end'   : data.get('window_end',      ''),
        'abstraction'  : 'category_process_file',
        'n_triples'    : len(lines),
        'sequence'     : lines,
    }


def run():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for folder, attack_label in ATTACK_FOLDERS:
        folder_path = os.path.join(TACTIC_DATA_ROOT, folder)
        if not os.path.isdir(folder_path):
            print('  [skip] not found: {}'.format(folder_path))
            continue

        files = sorted(
            f for f in os.listdir(folder_path)
            if f.endswith('.json') and not any(f.startswith(p) for p in SKIP_PREFIXES)
        )

        print('\n[{}]'.format(attack_label))
        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                entry     = build_abstract_sequence(fpath)
                out_fname = 'abstract_' + fname
                out_path  = os.path.join(OUTPUT_DIR, out_fname)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, indent=2)
                print('  {:<55}  {:>4} triples  → {}'.format(
                    fname, entry['n_triples'], out_fname))
            except Exception as ex:
                print('  [error] {}: {}'.format(fname, ex))

    elapsed = time.time() - t0
    print('\nSaved to: {}'.format(OUTPUT_DIR))
    print('Elapsed : {:.1f}s'.format(elapsed))


if __name__ == '__main__':
    run()
