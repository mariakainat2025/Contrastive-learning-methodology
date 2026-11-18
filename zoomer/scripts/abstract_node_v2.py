import ipaddress
from pathlib import PurePosixPath

from abstract_node import (
    is_script_file, is_log_file, is_system_file,
    ACCOUNT_ABSTRACT_FUNCS, account_abstract_counts,
    ROOT_PROCESS_NAMES, _first_token,
)

# Replaced ZOOMER's original directory-based check (any BIN_DIRS substring, no
# extension needed) with a pure extension check -- a ".exe" file is exeFile no
# matter where it sits, and nothing else (bin-dir location, etc.) qualifies on
# its own anymore.
def is_exe_file(path):
    return path.lower().endswith('.exe')

# Ported from scripts/generalize_ip_and_file_path_dev.py's lift_lib matching
# condition -- directory-location based (any ancestor dir literally named lib/
# lib32/lib64, anywhere in the path, regardless of prefix) instead of ZOOMER's
# original filename-pattern check (must end in ".so" + optional numeric version).
_LIB_NAMES = {"lib", "lib32", "lib64"}

def is_library_file(path):
    if not path.startswith('/') or path.endswith('/'):
        return False
    p = PurePosixPath(path)
    if len(p.suffix) <= 1:
        return False
    parents = list(p.parents)
    for i, parent in enumerate(parents):
        if parent.name in _LIB_NAMES:
            return i != 0
    return False

# Simplified to match scripts/generalize_ip_and_file_path_dev.py's _lift_temp_file --
# plain substring check for "/tmp" or "/temp" anywhere in the path (case-insensitive).
# Narrower than it looks in one way (no more .swp/.swpx/~ backup-file or /run/user/
# session-dir special cases, no /var/lib/dpkg/ exclusion) but looser in another (no
# extension requirement, and dev's own substring check can false-positive on a real
# folder name that merely starts with "temp", e.g. "/opt/temporary_cache/").
def is_tmp_file(path):
    p = path.lower()
    return '/tmp' in p or '/temp' in p

# Ported from scripts/generalize_ip_and_file_path_dev.py's _lift_directory matching
# condition -- narrower than ZOOMER's original dirFile (plain path.endswith('/')):
# also rejects the bare root "/" and any relative (non-"/"-rooted) dir-like string.
def is_directory_file(path):
    if not path.endswith('/'):
        return False
    stripped = path.rstrip('/')
    return stripped != '' and stripped.startswith('/')

# Ported from scripts/generalize_ip_and_file_path_dev.py's lift_home matching
# condition (not its string output) -- narrower than ZOOMER's original userFile
# (plain "/home/" prefix check): now also requires a real extension and at least
# 4 path levels under /home (i.e. /home/<user>/<subdir>/<file>, not a bare
# /home/<user>/<file>).
_HOME_ROOT = PurePosixPath("/home")

def is_user_file(path):
    if not path.startswith('/') or path.endswith('/'):
        return False
    p = PurePosixPath(path)
    if len(p.suffix) <= 1:
        return False
    if _HOME_ROOT not in p.parents:
        return False
    return len(list(p.parents)) > 3

def is_other_file(path):
    return not (
        is_tmp_file(path)
        or is_library_file(path)
        or is_exe_file(path)
        or is_script_file(path)
        or is_directory_file(path)
        or is_log_file(path)
        or is_system_file(path)
        or is_user_file(path)
    )

FILE_ABSTRACT_FUNCS = {
    'tmpFile': is_tmp_file,
    'library_file': is_library_file,
    'exeFile': is_exe_file,
    'scriptFile': is_script_file,
    'directory_file': is_directory_file,
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

# Process classification rewritten to match scripts/generalize_ip_and_file_path_dev.py's
# own chain (system -> web -> service -> util -> user), in that priority order, plus a
# 6th "unknownProcess" catch-all for whatever matches none of the five (dev's own chain
# returns None in that case -- e.g. "kernel" alone, or "crontab" which is explicitly
# excepted from service_process and matches nothing else). Root is folded into system
# as two exact-match names (ZOOMER's original rootProcess names), checked first, since
# dev's own chain has no equivalent rule and would otherwise leave them unresolved too.

def is_web_process(name):
    if name in ROOT_PROCESS_NAMES:
        return False
    return 'firefox' in _first_token(name).lower()

# Dev's own system prefix list -- notably does NOT include "/usr/" (unlike ZOOMER's
# original SYSTEM_PROCESS_DIRS), so paths under /usr/ fall through to service/util/user
# instead, matching dev's actual behavior.
_SYSTEM_PROCESS_PREFIXES = ("/etc/", "/var", "/sys", "/bin", "/run/", "/sbin", "/lib/systemd/system/")

def is_system_process(name):
    if name in ROOT_PROCESS_NAMES:
        return True
    if is_web_process(name):
        return False
    return name.startswith(_SYSTEM_PROCESS_PREFIXES)

# Ported from scripts/generalize_ip_and_file_path_dev.py's _is_service_process --
# substring-anywhere keyword match on the full name, with "crontab" explicitly
# excepted (it contains "cron" but is the user-facing CLI tool, not the daemon).
_SERVICE_PROCESS_KEYWORDS = ("cron", "sshd", "ssh", "rsyslogd", "ntpd", "httpd", "mysqld")
_SERVICE_PROCESS_NAME_EXCEPTIONS = ("crontab",)

def is_service_process(name):
    if is_system_process(name) or is_web_process(name):
        return False
    token = _first_token(name)
    basename = token.rsplit('/', 1)[-1]
    if basename in _SERVICE_PROCESS_NAME_EXCEPTIONS:
        return False
    return any(k in name for k in _SERVICE_PROCESS_KEYWORDS)

_UTIL_PROCESS_SUBSTRINGS = ("/bin/", "/usr/bin/", "/usr/sbin/", "/usr/local/bin/",
                            "/usr/local/sbin/", "/snap/bin/")

def is_util_process(name):
    if is_system_process(name) or is_web_process(name) or is_service_process(name):
        return False
    return any(s in name for s in _UTIL_PROCESS_SUBSTRINGS)

_USER_PROCESS_PREFIXES = ("/usr/", "/home/", "/opt/")

def is_user_process(name):
    if (is_system_process(name) or is_web_process(name)
            or is_service_process(name) or is_util_process(name)):
        return False
    return name.startswith(_USER_PROCESS_PREFIXES)

def is_unknown_process(name):
    return not (is_system_process(name) or is_web_process(name) or is_service_process(name)
                or is_util_process(name) or is_user_process(name))

PROCESS_ABSTRACT_FUNCS = {
    'systemProcess': is_system_process,
    'webProcess': is_web_process,
    'serviceProcess': is_service_process,
    'utilProcess': is_util_process,
    'userProcess': is_user_process,
    'unknownProcess': is_unknown_process,
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

def classify_ip(name):
    if name.startswith('/') or name.startswith('iface:'):
        return 'internal network address'

    if '.' in name:
        ip_str = name.rsplit(':', 1)[0] if ':' in name else name
    elif name.count(':') >= 2:
        ip_str = name.rsplit(':', 1)[0]
    else:
        ip_str = name

    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return 'internal network address'

    return 'internal network address' if ip.is_private else 'external network address'

def is_internal_socket(name):
    return classify_ip(name) == 'internal network address'

def is_external_socket(name):
    return classify_ip(name) == 'external network address'

SOCKET_ABSTRACT_FUNCS = {
    'internal network address': is_internal_socket,
    'external network address': is_external_socket,
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

def node_abstract_counts(G):
    counts = file_abstract_counts(G)
    counts.update(process_abstract_counts(G))
    counts.update(account_abstract_counts(G))
    counts.update(socket_abstract_counts(G))
    return counts

HNODE_DIMS = (
    'tmpFile', 'library_file', 'exeFile', 'scriptFile', 'directory_file', 'logFile', 'systemFile', 'userFile', 'otherFile',
    'systemProcess', 'webProcess', 'serviceProcess', 'utilProcess', 'userProcess', 'unknownProcess',
    'rootAccount', 'systemAccount', 'userAccount',
    'internal network address', 'external network address',
)
assert len(HNODE_DIMS) == 20

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
