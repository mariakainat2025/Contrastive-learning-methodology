import os
import json
import re

_IP_RE = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')


_EMAIL_DIRS = {'thunderbird', 'mozilla-thunderbird', 'outlook', 'evolution', 'mutt', '.mail'}

def _get_role(n):
    if any(e in n for e in _EMAIL_DIRS):
        return 'email_file'
    if any(ext in n for ext in ['.dll', '.exe', '.lib', '.so', '.a']) or '/lib' in n:
        return 'library_file'
    if '/proc' in n:
        return 'process_file'
    if any(p in n for p in ['/dev/', '/sys/', '/boot', '/bin', '/sbin']):
        return 'system_file'
    if any(p in n for p in ['/etc/', '/var/', '/root', '/run']):
        return 'system_file'
    if '/tmp' in n or '/temp' in n:
        return 'temp_file'
    if any(p in n for p in ['/home/', '/usr/', '/opt/', '/.cache']):
        return 'user_file'
    return None


def _abstract_file(name):
    n = name.lower()
    if (n.startswith('hkey') or n.startswith('hkcu') or n.startswith('hkcr')
            or n.startswith('hklm') or n.startswith('hku') or n.startswith('hkcc')):
        return 'registry run key'
    role = _get_role(n)
    if role is None:
        return name
    stem = os.path.splitext(os.path.basename(n))[0].strip('._- ')
    if stem:
        return f'{role} {stem}'
    return role


def _abstract_network(name):
    n = name.strip()
    if 'NA_null' in n or 'NA_0' in n:
        return 'network_failure'
    if n.startswith('NETLINK'):
        return 'kernel_socket'
    parts = n.split('_')
    ip_indices = [i for i, p in enumerate(parts) if _IP_RE.match(p)]
    if len(ip_indices) < 2:
        return 'local_connection'
    dst_ip = parts[ip_indices[-1]]
    octets = dst_ip.split('.')
    if len(octets) != 4:
        return 'public_netflow'
    try:
        o = [int(x) for x in octets]
    except ValueError:
        return 'public_netflow'
    if o[0] == 127:
        return 'local_connection'
    if (o[0] == 10 or
            (o[0] == 172 and 16 <= o[1] <= 31) or
            (o[0] == 192 and o[1] == 168)):
        return 'private_netflow'
    return 'public_netflow'


_WEB_PROCS      = {'firefox', 'chrome'}
_EMAIL_KEYWORDS = {'mail', 'thunderbird', 'outlook', 'evolution', 'mutt'}
_SERV_PROCS     = {'cron', 'crond', 'sshd', 'rsyslogd', 'ntpd', 'httpd',
                   'mysqld', 'postgres', 'postgresql', 'apache', 'apache2',
                   'nginx', 'salt-minion', 'dhclient', 'dhclient3',
                   'xvnc4', 'xvnc', 'x11vnc', 'vncserver', 'apt-config'}
_USR_PROCS      = {'metacity', 'nautilus', 'unity', 'fluxbox'}
_SYS_CMDS       = {'bash', 'sh', 'dash', 'zsh', 'ksh', 'sed', 'awk', 'grep',
                   'sort', 'date', 'ls', 'cat', 'mv', 'rm', 'uname', 'lesspipe',
                   'sudo', 'su', 'mount', 'groups', 'chmod'}


def _abstract_process(name):
    n = name.strip()
    if not n:
        return name
    cmd = n.split()[0]
    if cmd.startswith('-') and not cmd.startswith('./'):
        cmd = cmd[1:]
    stem = os.path.basename(cmd).lower().rstrip(':')
    if any(w in stem for w in _WEB_PROCS):
        return f'web_process {stem}'
    if any(k in stem for k in _EMAIL_KEYWORDS):
        return f'email_process {stem}'
    if any(s in stem for s in _SERV_PROCS):
        return f'service_process {stem}'
    if any(u in stem for u in _USR_PROCS):
        return f'user_process {stem}'
    if stem in _SYS_CMDS:
        return f'system_process {stem}'
    cl = n.lower()
    if any(p in cl for p in ['/bin/', '/usr/bin/', '/usr/sbin/',
                               '/usr/local/bin/', '/usr/local/sbin/', '/snap/bin/']):
        return f'util_process {stem}'
    if any(p in cl for p in ['/bin', '/sbin', '/etc/', '/var', '/sys',
                               '/run', '/lib/systemd']):
        return f'system_process {stem}'
    if '/tmp/' in cl or '/temp/' in cl:
        return f'temp_process {stem}'
    if '/home/' in cl or '/usr/' in cl or '/opt/' in cl:
        return f'user_process {stem}'
    return name


def _get_dst_ip(name):
    """Return destination IP string from a netflow name, or None."""
    n = name.strip()
    parts = n.split('_')
    ip_indices = [i for i, p in enumerate(parts) if _IP_RE.match(p)]
    if len(ip_indices) < 2:
        return None
    dst_ip = parts[ip_indices[-1]]
    octets = dst_ip.split('.')
    if len(octets) != 4:
        return None
    try:
        o = [int(x) for x in octets]
    except ValueError:
        return None
    if o[0] == 127:
        return None
    if (o[0] == 10 or
            (o[0] == 172 and 16 <= o[1] <= 31) or
            (o[0] == 192 and o[1] == 168)):
        return None
    return dst_ip


def build_netflow_map(sg):
    """
    Scan all netflow nodes in a subgraph and return a dict mapping
    raw node name → abstract label, using public_netflow_1/2/... when
    multiple distinct public IPs exist, plain public_netflow when only one.
    """
    ip_order = []
    for entry in sg.get('nodes', []):
        node = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else entry
        if not isinstance(node, dict):
            continue
        t = node.get('type', '').lower()
        if 'flow' not in t and 'netflow' not in t:
            continue
        raw = node.get('name', '')
        ip = _get_dst_ip(raw)
        if ip and ip not in ip_order:
            ip_order.append(ip)

    ip_to_label = {}
    if len(ip_order) <= 1:
        for ip in ip_order:
            ip_to_label[ip] = 'public_netflow'
    else:
        for idx, ip in enumerate(ip_order, start=1):
            ip_to_label[ip] = f'public_netflow_{idx}'
    return ip_to_label


def abstract_node_name(name, node_type, netflow_map=None):
    if not name or not node_type:
        return name
    t = node_type.lower()
    if 'file' in t:
        return _abstract_file(name)
    if 'flow' in t or 'netflow' in t:
        if netflow_map:
            ip = _get_dst_ip(name)
            if ip and ip in netflow_map:
                return netflow_map[ip]
        return _abstract_network(name)
    if 'subject' in t:
        return _abstract_process(name)
    return name


def abstract_subgraph_file(in_path, out_path=None):
    if out_path is None:
        out_path = in_path
    print(f'  abstracting: {in_path}')
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    subgraphs = data['subgraphs'] if 'subgraphs' in data else (data if isinstance(data, list) else [data])
    for sg in subgraphs:
        netflow_map = build_netflow_map(sg)
        for entry in sg.get('nodes', []):
            node = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else entry
            if isinstance(node, dict) and 'name' in node and 'type' in node:
                node['name'] = abstract_node_name(node['name'], node['type'], netflow_map)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f'  saved: {out_path}')
    return out_path


def abstract_path(path):
    d, f = os.path.split(path)
    return os.path.join(d, 'abstract_' + f)


def run_node_abstraction():
    from scripts.config import OUTPUT_BENIGN, OUTPUT_ATTACK
    from scripts.filter_attack_subgraphs import ATTACKS
    files = [
        os.path.join(OUTPUT_BENIGN, 'benign_training.json'),
        os.path.join(OUTPUT_BENIGN, 'benign_testing.json'),
    ]
    for atk in ATTACKS:
        files.append(os.path.join(OUTPUT_ATTACK, f'attack_subgraphs_{atk["name"]}.json'))
    for path in files:
        if os.path.exists(path):
            abstract_subgraph_file(path, out_path=abstract_path(path))
        else:
            print(f'  skipping (not found): {path}')
