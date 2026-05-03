import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_IP_PORT_RE = re.compile(
    r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(:\d+)?\b'
)

_URL_RE = re.compile(
    r'https?://[^\s,;)>"\']+'
)

_DOMAIN_RE = re.compile(
    r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?::\d+)?\b'
)

_FILE_PATH_RE = re.compile(
    r'(?<![:/])/(?:[^\s,;)>"\'\n]+)'
)

_EMAIL_DIRS = {'thunderbird', 'mozilla-thunderbird', 'outlook', 'evolution', 'mutt', '.mail'}


def _get_role(n):
    if any(e in n for e in _EMAIL_DIRS):
        return 'email_file'
    if any(ext in n for ext in ['.dll', '.lib', '.so', '.a']) or '/lib' in n:
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


def _abstract_file_path(path):
    n = path.lower()
    role = _get_role(n)
    stem = os.path.splitext(os.path.basename(n))[0].strip('._- ')
    if role is None:
        return f'file {stem}' if stem else 'file'
    return f'{role} {stem}' if stem else role


def _classify_ip(o1, o2, o3, o4):
    if o1 == 127:
        return 'local_connection'
    if (o1 == 10 or
            (o1 == 172 and 16 <= o2 <= 31) or
            (o1 == 192 and o2 == 168)):
        return 'private_netflow'
    return 'public_netflow'


def abstract_cti_text(text: str) -> str:
    text = _URL_RE.sub('public_netflow', text)
    text = _FILE_PATH_RE.sub(lambda m: _abstract_file_path(m.group(0)), text)

    def _replace_ip(m):
        try:
            o = [int(m.group(i)) for i in range(1, 5)]
        except ValueError:
            return m.group(0)
        return _classify_ip(*o)

    text = _IP_PORT_RE.sub(_replace_ip, text)

    def _replace_domain(m):
        token = m.group(0)
        if token in ('public_netflow', 'private_netflow', 'local_connection'):
            return token
        if re.match(r'^\d+(\.\d+)+$', token):
            return token
        return 'public_netflow'

    text = _DOMAIN_RE.sub(_replace_domain, text)
    return text


def abstract_cti_file(in_path: str, out_path: str = None) -> str:
    with open(in_path, 'r', encoding='utf-8') as f:
        text = f.read()
    abstracted = abstract_cti_text(text)
    if out_path is None:
        out_path = in_path
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(abstracted)
    return abstracted


def run_cti_abstraction(dirs: list, inplace: bool = False):
    total_files   = 0
    total_changed = 0

    for d in dirs:
        if not os.path.isdir(d):
            print(f'  skipping (not found): {d}')
            continue

        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.txt') or fname.startswith('.'):
                continue
            if fname.endswith('_abstracted.txt'):
                continue

            in_path  = os.path.join(d, fname)
            out_path = in_path if inplace else \
                       os.path.join(d, fname.replace('.txt', '_abstracted.txt'))

            with open(in_path, 'r', encoding='utf-8') as f:
                original = f.read()

            abstracted = abstract_cti_text(original)
            total_files += 1

            if abstracted != original:
                total_changed += 1
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(abstracted)
                print(f'  changed : {fname}')
                for line_o, line_a in zip(original.splitlines(), abstracted.splitlines()):
                    if line_o != line_a:
                        print(f'    BEFORE: {line_o.strip()}')
                        print(f'    AFTER : {line_a.strip()}')
            else:
                print(f'  unchanged: {fname}')

    print()
    print(f'  Files processed : {total_files}')
    print(f'  Files changed   : {total_changed}')


if __name__ == '__main__':
    from scripts.config import INPUT_TEST
    CTI_REPORTS_DIR = os.path.join(PROJECT_ROOT, 'input', 'cti_reports')
    run_cti_abstraction([INPUT_TEST, CTI_REPORTS_DIR], inplace=False)