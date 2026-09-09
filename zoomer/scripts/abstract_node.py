import ipaddress

def classify_ip(name):
    if name.startswith('/') or name.startswith('iface:'):
        return 'internal'

    if '.' in name:
        ip_str = name.rsplit(':', 1)[0] if ':' in name else name
    elif name.count(':') >= 2:
        ip_str = name.rsplit(':', 1)[0]
    else:
        ip_str = name

    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return 'internal'

    return 'internal' if ip.is_private else 'external'

def is_internal_socket(name):
    return classify_ip(name) == 'internal'

def is_external_socket(name):
    return classify_ip(name) == 'external'

SOCKET_ABSTRACT_FUNCS = {
    'internalIP': is_internal_socket,
    'externalIP': is_external_socket,
}

def socket_abstract_counts(G):
    counts = {label: 0 for label in SOCKET_ABSTRACT_FUNCS}
    for (_, data) in G.nodes(data=True):
        if data.get('type') != 'NetFlowObject':
            continue
        name = data.get('name')
        if not name:
            continue
        for (label, fn) in SOCKET_ABSTRACT_FUNCS.items():
            if fn(name):
                counts[label] += 1
    return counts

def is_tmp_file(path):
    p = path.lower()
    if p.startswith('/var/lib/dpkg/'):
        return False
    if '/run/user/' in p:
        return True
    if p.endswith(('.swp', '.swpx', '~')):
        return True
    segments = path.rstrip('/').split('/')
    for seg in segments:
        base = seg.lower()
        if base in ('tmp', 'temp'):
            return True
        if '.' in base:
            (name, ext) = base.rsplit('.', 1)
            if name in ('tmp', 'temp') or ext in ('tmp', 'temp'):
                return True
    return False

def is_dll_file(path):
    base = path.rsplit('/', 1)[-1].lower()
    parts = base.split('.')
    if 'so' not in parts:
        return False
    idx = parts.index('so')
    return all(p.isdigit() for p in parts[idx + 1:])

BIN_DIRS = ('/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/', '/usr/local/bin/', '/usr/local/sbin/', '/snap/bin/')

def is_exe_file(path):
    if path.endswith('[not found]') or path.endswith('/'):
        return False
    if path.endswith(('.dpkg-new', '.dpkg-tmp')):
        return False
    p = path.lower()
    return any(d in p for d in BIN_DIRS)

SCRIPT_EXTENSIONS = ('sh', 'py')

def is_script_file(path):
    base = path.rsplit('/', 1)[-1].lower()
    if '.' not in base:
        return False
    ext = base.rsplit('.', 1)[-1]
    return ext in SCRIPT_EXTENSIONS

def is_dir_file(path):
    return path.endswith('/')

def is_log_file(path):
    if path.lower().startswith('/var/log/'):
        return True
    base = path.rsplit('/', 1)[-1].lower()
    if '.' in base and base.rsplit('.', 1)[-1] == 'log':
        return True
    return False

def is_system_file(path):
    if is_log_file(path) or is_tmp_file(path):
        return False
    if path.endswith('[not found]'):
        return False
    p = path.lower()
    return p.startswith(('/etc/', '/root', '/usr/', '/var/', '/run/', '/sys/', '/dev/', '/sbin/', '/proc/', '/bin/', '/lib/'))

def is_user_file(path):
    return path.lower().startswith('/home/')

def is_other_file(path):
    return not (
        is_tmp_file(path)
        or is_dll_file(path)
        or is_exe_file(path)
        or is_script_file(path)
        or is_dir_file(path)
        or is_log_file(path)
        or is_system_file(path)
        or is_user_file(path)
    )

FILE_ABSTRACT_FUNCS = {
    'tmpFile': is_tmp_file,
    'dllFile': is_dll_file,
    'exeFile': is_exe_file,
    'scriptFile': is_script_file,
    'dirFile': is_dir_file,
    'logFile': is_log_file,
    'systemFile': is_system_file,
    'userFile': is_user_file,
    'otherFile': is_other_file,
}

def file_abstract_counts(G):
    counts = {label: 0 for label in FILE_ABSTRACT_FUNCS}
    for (_, data) in G.nodes(data=True):
        if data.get('type') != 'FILE':
            continue
        name = data.get('name')
        if not name:
            continue
        for (label, fn) in FILE_ABSTRACT_FUNCS.items():
            if fn(name):
                counts[label] += 1
    return counts

ROOT_PROCESS_NAMES = ('kernel', '/usr/lib/systemd/systemd')

SYSTEM_PROCESS_DIRS = ('/bin/', '/sbin/', '/usr/', '/lib/')

SYSTEM_PROCESS_BARE_NAMES = {
    'bash', '-bash', 'sh', 'curl', 'vim', 'crontab', 'true', 'dircolors',
    'dirname', 'groups', 'iptables', 'ln', 'python3', 'stty', 'basename',
    'lesspipe', 'chmod', 'tar', 'dmesg', 'systemctl',
}

def _first_token(name):
    parts = name.split()
    return parts[0] if parts else name

def is_root_process(name):
    return name in ROOT_PROCESS_NAMES

def is_web_process(name):
    if is_root_process(name):
        return False
    return 'firefox' in _first_token(name).lower()

def is_system_process(name):
    if is_root_process(name) or is_web_process(name):
        return False
    token = _first_token(name)
    if token.startswith(SYSTEM_PROCESS_DIRS):
        return True
    if token.startswith(('bpf-prog:', 'unit:')):
        return True
    return token in SYSTEM_PROCESS_BARE_NAMES

def is_user_process(name):
    return not (is_root_process(name) or is_web_process(name) or is_system_process(name))

PROCESS_ABSTRACT_FUNCS = {
    'rootProcess': is_root_process,
    'systemProcess': is_system_process,
    'webProcess': is_web_process,
    'userProcess': is_user_process,
}

def process_abstract_counts(G):
    counts = {label: 0 for label in PROCESS_ABSTRACT_FUNCS}
    for (_, data) in G.nodes(data=True):
        if data.get('type') != 'SUBJECT_PROCESS':
            continue
        name = data.get('name')
        if not name:
            continue
        for (label, fn) in PROCESS_ABSTRACT_FUNCS.items():
            if fn(name):
                counts[label] += 1
    return counts

ROOT_ACCOUNT_NAMES = ('acct:root', 'uid:0')
SYSTEM_ACCOUNT_NAMES = ('acct:www-data',)

def is_root_account(name):
    return name in ROOT_ACCOUNT_NAMES

def is_system_account(name):
    return name in SYSTEM_ACCOUNT_NAMES

def is_user_account(name):
    return not is_root_account(name) and not is_system_account(name)

ACCOUNT_ABSTRACT_FUNCS = {
    'rootAccount': is_root_account,
    'systemAccount': is_system_account,
    'userAccount': is_user_account,
}

def account_abstract_counts(G):
    counts = {label: 0 for label in ACCOUNT_ABSTRACT_FUNCS}
    for (_, data) in G.nodes(data=True):
        if data.get('type') != 'PRINCIPAL_LOCAL':
            continue
        name = data.get('name')
        if not name:
            continue
        for (label, fn) in ACCOUNT_ABSTRACT_FUNCS.items():
            if fn(name):
                counts[label] += 1
    return counts

def node_abstract_counts(G):
    counts = file_abstract_counts(G)
    counts.update(process_abstract_counts(G))
    counts.update(account_abstract_counts(G))
    counts.update(socket_abstract_counts(G))
    return counts

HNODE_DIMS = (
    'tmpFile', 'dllFile', 'exeFile', 'scriptFile', 'dirFile', 'logFile', 'systemFile', 'userFile', 'otherFile',
    'rootProcess', 'systemProcess', 'webProcess', 'userProcess',
    'rootAccount', 'systemAccount', 'userAccount',
    'internalIP', 'externalIP',
)
assert len(HNODE_DIMS) == 18

def node_abstract_vector(G):
    counts = node_abstract_counts(G)
    return [counts[dim] for dim in HNODE_DIMS]

FUNCS_BY_NODE_TYPE = {
    'FILE': FILE_ABSTRACT_FUNCS,
    'SUBJECT_PROCESS': PROCESS_ABSTRACT_FUNCS,
    'PRINCIPAL_LOCAL': ACCOUNT_ABSTRACT_FUNCS,
    'NetFlowObject': SOCKET_ABSTRACT_FUNCS,
}

def node_hnode(node_type, name):
    funcs = FUNCS_BY_NODE_TYPE.get(node_type, {})
    matches = {label for (label, fn) in funcs.items() if name and fn(name)}
    return [1 if dim in matches else 0 for dim in HNODE_DIMS]

def _report_sockets():
    import glob
    import json
    from collections import Counter

    files = glob.glob('/csse/research/contructive-learning/CAM-LDS/graphs/*/*/*.json')
    names = set()
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        for n in d.get('nodes', []):
            if n.get('type') == 'NetFlowObject':
                names.add(n.get('name', ''))

    counts = Counter(classify_ip(n) for n in names)
    print('Total unique socket names:', len(names))
    for bucket, c in counts.most_common():
        print(' ', bucket, ':', c)

def _report_files():
    import json
    import os

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(scripts_dir, 'file_paths.json')
    with open(in_path) as f:
        paths = json.load(f)

    print('Total file paths:', len(paths))
    for (label, fn) in FILE_ABSTRACT_FUNCS.items():
        matches = [p for p in paths if fn(p)]
        out_path = os.path.join(scripts_dir, '{}_paths.json'.format(label.lower()))
        with open(out_path, 'w') as out_f:
            json.dump(matches, out_f, indent=2)
        print('  {:12s}: {:5d} -> {}'.format(label, len(matches), out_path))

def _report_processes():
    import json
    import os

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(scripts_dir, 'process_names.json')
    with open(in_path) as f:
        names = json.load(f)

    print('Total process names:', len(names))
    for (label, fn) in PROCESS_ABSTRACT_FUNCS.items():
        matches = [n for n in names if fn(n)]
        out_path = os.path.join(scripts_dir, '{}_names.json'.format(label.lower()))
        with open(out_path, 'w') as out_f:
            json.dump(matches, out_f, indent=2)
        print('  {:14s}: {:5d} -> {}'.format(label, len(matches), out_path))

if __name__ == '__main__':
    _report_sockets()
    print()
    _report_files()
    print()
    _report_processes()
