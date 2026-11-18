import os
import re
import json
from pathlib import Path, PurePosixPath
from collections import Counter

from ip_classifier import classify_ip
from abstract_file_paths import (
    lift_etc_var, lift_bin, lift_home, lift_root, lift_lib, lift_other,
)
import build_sequences

CAM_LDS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(CAM_LDS_DIR, "graphs")

GRAPH_DEV_FILEIP_DIR = os.path.join(CAM_LDS_DIR, "generalize_graph_dev_fileip")
SEQUENCES_DEV_FILEIP_DIR = os.path.join(CAM_LDS_DIR, "sequences_dev_fileip")

GRAPH_DEV_ALL_DIR = os.path.join(CAM_LDS_DIR, "generalize_graph_dev_all")
SEQUENCES_DEV_ALL_DIR = os.path.join(CAM_LDS_DIR, "sequences_dev_all")

EVENT_PREFIX = "EVENT_"

_SYSTEM_PROCESS_PREFIXES = ("/etc/", "/var", "/sys", "/bin", "/run/", "/sbin", "/lib/systemd/system/")


def _is_system_process(name):
    return name.startswith(_SYSTEM_PROCESS_PREFIXES)


def _lift_system_process(name):
    token = name.split(None, 1)[0] if name.split() else name
    basename = PurePosixPath(token).name
    return f"system_process {basename}"

_SERVICE_PROCESS_KEYWORDS = ("cron", "sshd", "ssh", "rsyslogd", "ntpd", "httpd", "mysqld")
_SERVICE_PROCESS_NAME_EXCEPTIONS = ("crontab",)


def _is_service_process(name):
    token = name.split(None, 1)[0] if name.split() else name
    basename = PurePosixPath(token).name
    if basename in _SERVICE_PROCESS_NAME_EXCEPTIONS:
        return False
    return any(k in name for k in _SERVICE_PROCESS_KEYWORDS)


def _lift_service_process(name):
    token = name.split(None, 1)[0] if name.split() else name
    basename = PurePosixPath(token).name
    return f"service_process {basename}"

_WEB_PROCESS_KEYWORDS = ("firefox", "chrome", "ie", "edge")


def _is_web_process(name):
    return any(k in name for k in _WEB_PROCESS_KEYWORDS)


def _lift_web_process(name):
    token = name.split(None, 1)[0] if name.split() else name
    basename = PurePosixPath(token).name
    return f"web_process {basename}"

_UTIL_PROCESS_SUBSTRINGS = ("/bin/", "/usr/bin/", "/usr/sbin/", "/usr/local/bin/",
                            "/usr/local/sbin/", "/snap/bin/")


def _is_util_process(name):
    return any(s in name for s in _UTIL_PROCESS_SUBSTRINGS)


def _lift_util_process(name):
    token = name.split(None, 1)[0] if name.split() else name
    basename = PurePosixPath(token).name
    return f"util_process {basename}"

_USER_PROCESS_PREFIXES = ("/usr/", "/home/", "/opt/")


def _is_user_process(name):
    return name.startswith(_USER_PROCESS_PREFIXES)


def _lift_user_process(name):
    token = name.split(None, 1)[0] if name.split() else name
    basename = PurePosixPath(token).name
    return f"user_process {basename}"


def _classify_process(name):
    if _is_system_process(name):
        return _lift_system_process(name)
    if _is_web_process(name):
        return _lift_web_process(name)
    if _is_service_process(name):
        return _lift_service_process(name)
    if _is_util_process(name):
        return _lift_util_process(name)
    if _is_user_process(name):
        return _lift_user_process(name)
    return None


_PPG_SYS_FILE_LOG_PATHS = (
    "/var/log/boot.log", "/var/log/kern.log", "/var/log/syslog", "/var/log/messages",
    "/var/log/auth.log", "/var/log/secure", "/var/log/dmesg", "/var/log/utx.log",
)


def _matches_ppg_sys_file(raw_path):
    lower_path = raw_path.lower()
    if "/etc/" in lower_path or "/root" in lower_path or "/var/run" in lower_path:
        return True
    return any(log_path in lower_path for log_path in _PPG_SYS_FILE_LOG_PATHS)

IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")


def _is_ipv4(s):
    m = IPV4_RE.match(s)
    return bool(m) and all(0 <= int(o) <= 255 for o in m.groups())


def _generalize_name(name):
    if _is_ipv4(name):
        return classify_ip(name), name

    if ":" in name:
        ip_part, _, port_part = name.rpartition(":")
        if _is_ipv4(ip_part):
            return classify_ip(ip_part), ip_part
        if _is_ipv4(port_part):
            return classify_ip(port_part), port_part

    return name, None


def _lift_temp_name_prefix(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None
    basename = raw_path.rsplit('/', 1)[-1]
    lower_basename = basename.lower()
    if lower_basename.startswith("tmp.") or lower_basename.startswith("temp."):
        return f"{basename} temp file"
    return None


def _lift_temp_file(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None
    lower_path = raw_path.lower()
    if "/tmp" not in lower_path and "/temp" not in lower_path:
        return None
    basename = raw_path.rsplit('/', 1)[-1]
    if len(PurePosixPath(basename).suffix) <= 1:
        return None
    return f"{basename} temp file"


_CONF_ROOTS = (PurePosixPath("/etc"), PurePosixPath("/var"))


def _lift_system_file(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None
    basename = raw_path.rsplit('/', 1)[-1]
    if len(PurePosixPath(basename).suffix) <= 1:
        return None
    if _matches_ppg_sys_file(raw_path):
        return f"{basename} system file"
    path = PurePosixPath(raw_path)
    if path.parent in _CONF_ROOTS:
        return f"{basename} system file"
    return None


_BARE_NUM_PREFIX_RE = re.compile(r"^\d+_")
_BARE_NUM_SUFFIX_RE = re.compile(r"_\d+$")


def _lift_bare_file(raw_path):
    if '/' in raw_path:
        return None
    path = PurePosixPath(raw_path)
    if len(path.suffix) <= 1:
        return None
    ext = path.suffix[1:]
    shape = _BARE_NUM_SUFFIX_RE.sub('', _BARE_NUM_PREFIX_RE.sub('', path.stem)).rstrip('_')
    if not shape:
        shape = path.stem
    return f"{shape} {ext} file"


_DONOTCRY_RE = re.compile(r"(\.donotcry)+$", re.IGNORECASE)


def _strip_donotcry(raw_path):
    stripped = _DONOTCRY_RE.sub('', raw_path)
    return stripped if stripped != raw_path else None


_MEDIA_IMAGES_ROOT = PurePosixPath("/media/data/Images")


def _lift_media_images(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None
    path = PurePosixPath(raw_path)
    if _MEDIA_IMAGES_ROOT not in path.parents:
        return None
    name = path.name
    donotcry_stripped = _DONOTCRY_RE.sub('', name)
    if donotcry_stripped == name:
        return None
    real_path = PurePosixPath(donotcry_stripped)
    if len(real_path.suffix) <= 1:
        return None
    e1 = real_path.suffix[1:]
    shape = _BARE_NUM_SUFFIX_RE.sub('', _BARE_NUM_PREFIX_RE.sub('', real_path.stem)).rstrip('_')
    if not shape:
        shape = real_path.stem
    return f"media {shape} {e1} donotcry file"


def _lift_firefox_idb_blob(raw_path):
    if not raw_path.startswith('/home/'):
        return None
    path = PurePosixPath(raw_path)
    if not path.name.isdigit():
        return None
    if '.mozilla' not in path.parts or 'firefox' not in path.parts or 'idb' not in path.parts:
        return None
    parent = path.parent
    if parent.name == 'journals' and parent.parent.name.endswith('.files'):
        return f"user .mozilla firefox idb journals {path.name}"
    if parent.name.endswith('.files'):
        return f"user .mozilla firefox idb journals {path.name}"
    return None


_AUTOSTART_SERVICE_RE = re.compile(r'^(app-.+@autostart)\.service$')


def _lift_autostart_service(raw_path):
    m = _AUTOSTART_SERVICE_RE.match(raw_path)
    if not m:
        return None
    name = m.group(1).replace(r'\x2d', '-')
    return f"{name} service file"


def _lift_bin_usr(raw_path):
    result = lift_bin(raw_path)
    if result is not None and raw_path.startswith('/usr/'):
        return f"usr {result}"
    return result


def _lift_lib_usr(raw_path):
    result = lift_lib(raw_path)
    if result is not None and raw_path.startswith('/usr/'):
        return f"usr {result}"
    return result


def _lift_generic_stem_ext(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None
    path = PurePosixPath(raw_path)
    if len(path.suffix) <= 1:
        return None
    ext = path.suffix[1:]
    result = f"{path.stem} {ext} file"
    if raw_path.startswith('/usr/'):
        return f"usr {result}"
    return result


_FIREFOX_PROFILE_DIR_PARENT = 'firefox'
_APT_KEY_GPGHOME_RE = re.compile(r'^apt-key-gpghome\.')
_CRONTAB_DIR_RE = re.compile(r'^crontab\.')
_ANSIBLE_TMP_DIR_RE = re.compile(r'^ansible-tmp-')
_USER_SERVICE_DIR_RE = re.compile(r'^user@\d+\.service$')


def _lift_directory(raw_path):
    stripped = raw_path.rstrip('/')
    if stripped == '' or not stripped.startswith('/'):
        return None
    parts = [p for p in stripped.split('/') if p]
    if not parts:
        return None
    root = parts[0]
    last = parts[-1]

    if len(parts) == 1:
        return f"{root} directory"
    if root == 'home' and len(parts) == 2:
        return "home directory"

    if len(parts) >= 2 and parts[-2] == _FIREFOX_PROFILE_DIR_PARENT:
        name = 'firefox'
    elif _APT_KEY_GPGHOME_RE.match(last):
        name = 'apt-key-gpghome'
    elif _CRONTAB_DIR_RE.match(last):
        name = 'crontab'
    elif _ANSIBLE_TMP_DIR_RE.match(last):
        name = 'ansible-tmp'
    elif _USER_SERVICE_DIR_RE.match(last):
        name = 'user.service'
    else:
        name = last

    return f"{root} {name} directory"


_FILE_PATH_RULES = (
    ('firefox_idb_blob', _lift_firefox_idb_blob),
    ('autostart_service', _lift_autostart_service),
    ('temp_name_prefix', _lift_temp_name_prefix),
    ('etc_var', lift_etc_var),
    ('system_file', _lift_system_file),
    ('bin_usr', _lift_bin_usr),
    ('home', lift_home),
    ('root', lift_root),
    ('lib_usr', _lift_lib_usr),
    ('temp_file', _lift_temp_file),
    ('generic_stem_ext', _lift_generic_stem_ext),
    ('other', lift_other),
    ('bare_file', _lift_bare_file),
)


def _lift_file_path_rule(raw_path):
    """Same resolution chain as _lift_file_path, but also returns which named rule
    matched (or None if unresolved) -- used for the per-category count report."""
    if raw_path.endswith(','):
        return None, None
    if raw_path.endswith('/'):
        result = _lift_directory(raw_path)
        return result, ('directory_file' if result is not None else None)
    try:
        media_result = _lift_media_images(raw_path)
        if media_result is not None:
            return media_result, 'media_images'
        stripped = _strip_donotcry(raw_path)
        if stripped is not None:
            inner, rule = _lift_file_path_rule(stripped)
            if inner is None:
                return None, None
            if inner.endswith(' file'):
                return inner[:-len(' file')] + ' donotcry file', rule
            return inner + ' donotcry', rule
        for (rule, fn) in _FILE_PATH_RULES:
            result = fn(raw_path)
            if result is not None:
                return result, rule
        return None, None
    except Exception:
        return None, None


def _lift_file_path(raw_path):
    return _lift_file_path_rule(raw_path)[0]


def batch_main(graphs_dir, out_graphs_dir, apply_process):
    """Reads graphs/*/*/graph_*.json, generalizes IPs + file paths always, and process
    names only when apply_process=True. Writes the mutated graphs to out_graphs_dir and
    three report files (ip/file/process) alongside them."""
    files = sorted(Path(graphs_dir).glob("*/*/graph_*.json"))
    print(f"Found {len(files)} graph JSON files under {graphs_dir}")

    unique_ips = set()
    total_internal = 0
    total_external = 0
    ip_report = {}

    file_report = {}
    file_rule_counts = Counter()

    process_report = {}
    category_counts = Counter()

    total_events_generalized = 0

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        for attrs in data.get("nodes", []):
            ntype = attrs.get("type")
            raw_name = attrs.get("name")
            if raw_name is None:
                continue

            if ntype == "NetFlowObject":
                new_name, ip_found = _generalize_name(raw_name)
                if ip_found is not None:
                    attrs["name"] = new_name
                    unique_ips.add(ip_found)
                    if "internal" in new_name:
                        total_internal += 1
                    else:
                        total_external += 1
                ip_report.setdefault(raw_name, new_name if ip_found is not None else None)

            elif ntype == "FILE":
                if raw_name not in file_report:
                    lifted, rule = _lift_file_path_rule(raw_name)
                    if lifted is not None and lifted != raw_name:
                        file_report[raw_name] = lifted
                        file_rule_counts[rule] += 1
                    else:
                        file_report[raw_name] = None
                        file_rule_counts['not_generalized'] += 1
                lifted = file_report[raw_name]
                if lifted is not None:
                    attrs["name"] = lifted

            elif ntype == "SUBJECT_PROCESS" and apply_process:
                if raw_name not in process_report:
                    label = _classify_process(raw_name)
                    process_report[raw_name] = label
                    if label is not None:
                        category_counts[label.split(" ", 1)[0]] += 1
                label = process_report[raw_name]
                if label is not None:
                    attrs["name"] = label

        for link in data.get("links", []):
            raw_event = link.get("edge_type")
            if raw_event and raw_event.startswith(EVENT_PREFIX):
                link["edge_type"] = raw_event[len(EVENT_PREFIX):]
                total_events_generalized += 1

        rel = fpath.relative_to(graphs_dir)
        out_subdir = Path(out_graphs_dir) / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_name = fpath.name.replace("graph_", "generalize_", 1)
        with open(out_subdir / out_name, "w") as f:
            json.dump(data, f, indent=2)

    os.makedirs(out_graphs_dir, exist_ok=True)
    ip_report_path = os.path.join(out_graphs_dir, "ip_generalization_report.txt")
    file_report_path = os.path.join(out_graphs_dir, "file_generalization_report.txt")
    process_report_path = os.path.join(out_graphs_dir, "process_generalization_report.txt")

    total_addresses = len(ip_report)
    generalized_addresses = sum(1 for v in ip_report.values() if v is not None)

    with open(ip_report_path, "w") as f:
        f.write("original_ip,generalized_ip\n")
        for raw_name in sorted(ip_report):
            generalized = ip_report[raw_name] if ip_report[raw_name] is not None else "(not generalized)"
            f.write(f"{raw_name},{generalized}\n")

    total_file_paths = len(file_report)
    changed_file_paths = sum(1 for v in file_report.values() if v is not None)

    with open(file_report_path, "w") as f:
        f.write("original_path,generalized_path\n")
        for raw_name in sorted(file_report):
            generalized = file_report[raw_name] if file_report[raw_name] is not None else "(not generalized)"
            f.write(f"{raw_name},{generalized}\n")

    print()
    print(f"Graphs processed           : {len(files)}")
    print(f"Events generalized (edges) : {total_events_generalized}")
    print()
    print(f"-- IP addresses --")
    print(f"Total addresses (unique)   : {total_addresses}")
    print(f"  -> generalized           : {generalized_addresses}")
    print(f"  -> not generalized       : {total_addresses - generalized_addresses}")
    print(f"Unique IPs generalized     : {len(unique_ips)}")
    print(f"  -> internal              : {total_internal}")
    print(f"  -> external              : {total_external}")
    print()
    print(f"-- File paths --")
    print(f"Total file paths (unique)  : {total_file_paths}")
    print(f"  -> changed               : {changed_file_paths}")
    print(f"  -> not generalized       : {total_file_paths - changed_file_paths}")
    print()
    print(f"-- File paths by rule (unique paths, first-match-wins order, count > 5 only) --")
    shown = [(r, c) for (r, c) in file_rule_counts.most_common() if c > 5]
    hidden = len(file_rule_counts) - len(shown)
    for (rule, count) in shown:
        print(f"  {rule:20s}: {count}")
    if hidden:
        print(f"  ... ({hidden} more rule(s) with count <= 5 not shown)")
    print()

    ext_counts = Counter()
    donotcry_count = 0
    for lifted in file_report.values():
        if lifted is None:
            continue
        label = lifted
        if label.endswith(' donotcry file'):
            donotcry_count += 1
            label = label[:-len(' donotcry file')] + ' file'
        elif label.endswith(' donotcry'):
            donotcry_count += 1
            label = label[:-len(' donotcry')]
        tokens = label.split()
        if len(tokens) >= 2 and tokens[-1] == 'file':
            ext_counts[tokens[-2]] += 1
        elif label.endswith('directory'):
            ext_counts['(directory)'] += 1
        else:
            ext_counts['(other, no file/directory suffix)'] += 1

    print(f"-- File paths by final label's extension/tag (unique paths, count > 5 only) --")
    print(f"  (of which donotcry-tagged: {donotcry_count} -- extension/tag counted underneath the marker)")
    shown_ext = [(e, c) for (e, c) in ext_counts.most_common() if c > 5]
    hidden_ext = len(ext_counts) - len(shown_ext)
    for (ext, count) in shown_ext:
        print(f"  {ext:20s}: {count}")
    if hidden_ext:
        print(f"  ... ({hidden_ext} more extension/tag(s) with count <= 5 not shown)")
    print()

    if apply_process:
        total_process_names = len(process_report)
        resolved_process_names = sum(1 for v in process_report.values() if v is not None)

        with open(process_report_path, "w") as f:
            f.write("original_name,generalized_name\n")
            for raw_name in sorted(process_report):
                generalized = process_report[raw_name] if process_report[raw_name] is not None else "(not generalized)"
                f.write(f"{raw_name},{generalized}\n")

        print(f"-- Process names --")
        print(f"Total process names (unique): {total_process_names}")
        print(f"  -> user_process          : {category_counts['user_process']}")
        print(f"  -> system_process        : {category_counts['system_process']}")
        print(f"  -> service_process       : {category_counts['service_process']}")
        print(f"  -> web_process           : {category_counts['web_process']}")
        print(f"  -> util_process          : {category_counts['util_process']}")
        print(f"  -> not generalized       : {total_process_names - resolved_process_names}")
        print()
        print(f"Saved process report -> {process_report_path}")
    else:
        print(f"Process names              : left untouched (apply_process=False)")
        print()

    print(f"Saved IP report -> {ip_report_path}")
    print(f"Saved file report -> {file_report_path}")
    print(f"Saved graphs -> {out_graphs_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-dir", default=GRAPHS_DIR)
    args = ap.parse_args()

    header = lambda title: print(f"\n{'=' * 10} {title} {'=' * 10}")

    header("Pass 1/2: file paths + IPs only (process names untouched)")
    batch_main(args.graphs_dir, GRAPH_DEV_FILEIP_DIR, apply_process=False)
    print("\nBuilding sequences from pass 1 graphs...")
    build_sequences.main(graph_dir=GRAPH_DEV_FILEIP_DIR, sequences_dir=SEQUENCES_DEV_FILEIP_DIR)

    header("Pass 2/2: file paths + IPs + process names")
    batch_main(args.graphs_dir, GRAPH_DEV_ALL_DIR, apply_process=True)
    print("\nBuilding sequences from pass 2 graphs...")
    build_sequences.main(graph_dir=GRAPH_DEV_ALL_DIR, sequences_dir=SEQUENCES_DEV_ALL_DIR)

    print("\nDone.")
