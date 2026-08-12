import sys
import re
import json
import uuid
import argparse
from pathlib import Path


PAM_TYPES = {"USER_ACCT", "CRED_ACQ", "USER_START", "USER_END", "CRED_REFR", "CRED_DISP"}


AUTH_TYPES = {"USER_AUTH", "USER_LOGIN", "LOGIN"}


ACCOUNT_TYPES = {"ADD_USER", "ADD_GROUP"}


SERVICE_TYPES = {"SERVICE_START", "SERVICE_STOP"}


CMD_TYPES = {"USER_CMD"}


BPF_TYPES = {"BPF"}


AVC_TYPES = {"AVC"}


ERR_TYPES = {"USER_ERR"}


ANOM_ABEND_TYPES = {"ANOM_ABEND"}


ANOM_PROMISC_TYPES = {"ANOM_PROMISCUOUS"}


NETFILTER_TYPES = {"NETFILTER_CFG"}


SHUTDOWN_TYPES = {"SYSTEM_SHUTDOWN"}


NON_SYSCALL_TYPES = (PAM_TYPES | AUTH_TYPES | ACCOUNT_TYPES | SERVICE_TYPES
                      | CMD_TYPES | BPF_TYPES | AVC_TYPES | ERR_TYPES
                      | ANOM_ABEND_TYPES | ANOM_PROMISC_TYPES | NETFILTER_TYPES | SHUTDOWN_TYPES)


HOST_PRIORITY = ["videoserver", "client", "linuxshare", "docker", "reposerver", "inetfw"]
FALLBACK_MIN_EVENTS = 5


def pick_primary_host(host_counts):
    host_tags = set(host_counts)
    priority_pick = None
    for p in HOST_PRIORITY:
        matches = sorted(h for h in host_tags if h == p or h.startswith(p + "-"))
        if matches:
            priority_pick = matches[0]
            break
    if priority_pick is None:
        return None
    if host_counts[priority_pick] < FALLBACK_MIN_EVENTS:
        richest = max(host_counts, key=host_counts.get)
        if host_counts[richest] > host_counts[priority_pick]:
            return richest
    return priority_pick


def decode_arg(val):
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    try:
        return bytes.fromhex(val).decode('utf-8', errors='replace')
    except Exception:
        return val


def _get(pattern, line, flags=0):
    m = re.search(pattern, line, flags)
    return m.group(1) if m else None


def _get_addr_field(pattern, line):
    v = _get(pattern, line)
    return None if v == "?" else v


def _drop_nulls(events):
    return [{k: v for k, v in e.items() if v is not None and v != []} for e in events]


def _select_path(paths, path_types):
    non_parent = [p for p, t in zip(paths, path_types) if t != "PARENT"]
    if non_parent:
        return non_parent[-1]
    return paths[-1] if paths else None


def parse_audit(path):
    events = {}
    order  = []

    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.search(r'msg=audit\([\d.]+:(\d+)\)', line)
            if not m:
                continue
            eid = m.group(1)
            if eid not in events:
                events[eid] = []
                order.append(eid)
            events[eid].append(line)

    results = []

    for eid in order:
        lines = events[eid]

        event = {
            "event_id"  : eid,
            "timestamp" : None,
            "type"      : "SYSCALL",
            "syscall"   : None,
            "success"   : None,
            "exit"      : None,
            "a0"        : None,
            "ppid"      : None,
            "pid"       : None,
            "comm"      : None,
            "exe"       : None,
            "subj"      : None,
            "key"       : None,
            "command"   : None,
            "src"       : None,
            "dest"      : None,
            "paths"     : [],
            "path_types": [],
            "pam_op"    : None,
            "pam_acct"  : None,
            "pam_res"   : None,

            "acct"      : None,
            "op"        : None,
            "res"       : None,
            "hostname"  : None,
            "addr"      : None,
            "unit"      : None,
            "cmd"       : None,
            "avc_result": None,
            "avc_op"    : None,
            "avc_name"  : None,
            "avc_profile": None,
            "bpf_op"    : None,
            "bpf_progid": None,

            "sig"       : None,
            "dev"       : None,
            "prom"      : None,
            "nf_table"  : None,
            "nf_op"     : None,
            "mmap_flags": None,
            "fcaps"     : None,
        }

        has_syscall = False

        for line in lines:
            if not line.startswith("type="):
                continue
            rec_type = line.split()[0][5:]


            if rec_type == "SYSCALL":
                has_syscall = True
                m = re.search(r'msg=audit\(([\d.]+):\d+\)', line)
                if m: event["timestamp"] = m.group(1)
                m = re.search(r'success=(\w+)', line)
                if m: event["success"] = m.group(1)
                m = re.search(r'\bexit=(-?\d+)', line)
                if m: event["exit"] = m.group(1)
                m = re.search(r'\ba0=(\S+)', line)
                if m: event["a0"] = m.group(1)
                m = re.search(r'ppid=(\d+)', line)
                if m: event["ppid"] = m.group(1)
                m = re.search(r'\bpid=(\d+)', line)
                if m: event["pid"] = m.group(1)
                m = re.search(r'comm="([^"]+)"', line)
                if m: event["comm"] = m.group(1)
                m = re.search(r'exe="([^"]+)"', line)
                if m: event["exe"] = m.group(1)
                m = re.search(r'\bsubj=(\S+)', line)
                if m: event["subj"] = m.group(1)
                m = re.search(r'key="([^"]+)"', line)
                if m: event["key"] = m.group(1)
                after_arch = line.split("ARCH=")[-1]
                m = re.search(r'SYSCALL=(\w+)', after_arch)
                if m: event["syscall"] = m.group(1)


            elif rec_type == "EXECVE":
                args = re.findall(r'a\d+=((?:"[^"]*")|(?:[0-9A-Fa-f]+))', line)
                decoded = [decode_arg(a) for a in args]
                event["command"] = " ".join(decoded)


            elif rec_type == "SOCKADDR":
                m = re.search(r'laddr=([\d.]+)', line)
                laddr = m.group(1) if m else None
                m = re.search(r'lport=(\d+)', line)
                lport = m.group(1) if m else None
                m = re.search(r'path=([^\s}]+)', line)
                lpath = m.group(1) if m else None
                if laddr and lport:
                    event["dest"] = f"{laddr}:{lport}"
                elif lpath:
                    event["dest"] = lpath


            elif rec_type == "PATH":
                m = re.search(r'name="([^"]+)"', line)
                name = m.group(1) if m else None
                if name:
                    event["paths"].append(name)
                    m2 = re.search(r'\bnametype=(\w+)', line)
                    event["path_types"].append(m2.group(1) if m2 else None)


            elif rec_type in PAM_TYPES:
                event["type"] = rec_type
                m = re.search(r'msg=audit\(([\d.]+):\d+\)', line)
                if m: event["timestamp"] = m.group(1)
                m = re.search(r'\bpid=(\d+)', line)
                if m: event["pid"] = m.group(1)
                m = re.search(r'\bsubj=(\S+)', line)
                if m: event["subj"] = m.group(1)
                m = re.search(r"op=([^\s']+)", line)
                if m: event["pam_op"] = m.group(1)
                m = re.search(r'acct="([^"]+)"', line)
                if m: event["pam_acct"] = m.group(1)
                m = re.search(r'exe="([^"]+)"', line)
                if m: event["exe"] = m.group(1)
                m = re.search(r'res=(\w+)', line)
                if m: event["pam_res"] = m.group(1)


            elif rec_type in AUTH_TYPES:
                event["type"] = rec_type
                event["timestamp"] = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["pid"]       = _get(r'\bpid=(\d+)', line) or event["pid"]
                event["subj"]      = _get(r'\bsubj=(\S+)', line) or event["subj"]
                event["op"]        = _get(r"op=([^\s']+)", line) or event["op"]
                event["exe"]       = _get(r'exe="([^"]+)"', line) or event["exe"]
                event["hostname"]  = _get_addr_field(r'hostname=(\S+)', line) or event["hostname"]
                event["addr"]      = _get_addr_field(r'\baddr=(\S+)', line) or event["addr"]
                event["res"]       = _get(r'res=(\w+)', line) or event["res"]
                acct = _get(r'acct="([^"]+)"', line)
                if not acct:
                    hexacct = _get(r'acct=([0-9A-Fa-f]{6,})\b', line)
                    if hexacct: acct = decode_arg(hexacct)
                if not acct:
                    acct = _get(r'\bUID="([^"]+)"', line)
                if acct: event["acct"] = acct


            elif rec_type in ACCOUNT_TYPES:
                event["type"] = rec_type
                event["timestamp"] = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["pid"]       = _get(r'\bpid=(\d+)', line) or event["pid"]
                event["op"]        = _get(r"op=([^\s']+ ?[^\s']*)", line) or event["op"]
                event["exe"]       = _get(r'exe="([^"]+)"', line) or event["exe"]
                event["hostname"]  = _get_addr_field(r'hostname=(\S+)', line) or event["hostname"]
                event["res"]       = _get(r'res=(\w+)', line) or event["res"]
                acct = _get(r'acct="([^"]+)"', line)
                if not acct:
                    uidnum = _get(r'\bid=(\d+)', line)
                    if uidnum: acct = f"uid:{uidnum}"
                if not acct:
                    acct = _get(r'\bID="([^"]+)"', line)
                if acct: event["acct"] = acct


            elif rec_type in SERVICE_TYPES:
                event["type"] = rec_type
                event["timestamp"] = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["pid"]       = _get(r'\bpid=(\d+)', line) or event["pid"]
                event["unit"]      = _get(r'unit=(\S+)', line) or event["unit"]
                event["comm"]      = _get(r'comm="([^"]+)"', line) or event["comm"]
                event["exe"]       = _get(r'exe="([^"]+)"', line) or event["exe"]
                event["res"]       = _get(r'res=(\w+)', line) or event["res"]


            elif rec_type in CMD_TYPES:
                event["type"] = rec_type
                event["timestamp"] = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["pid"]       = _get(r'\bpid=(\d+)', line) or event["pid"]
                cwd = _get(r'cwd="([^"]+)"', line)
                if cwd: event["op"] = f"cwd={cwd}"
                cmd_raw = _get(r'cmd=((?:"[^"]*")|(?:[0-9A-Fa-f]+))', line)
                if cmd_raw: event["cmd"] = decode_arg(cmd_raw)
                event["exe"] = _get(r'exe="([^"]+)"', line) or event["exe"]
                event["res"] = _get(r'res=(\w+)', line) or event["res"]
                acct = _get(r'\bUID="([^"]+)"', line)
                if acct: event["acct"] = acct


            elif rec_type in BPF_TYPES:
                event["type"] = rec_type
                event["timestamp"]  = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["bpf_progid"] = _get(r'prog-id=(\d+)', line) or event["bpf_progid"]
                event["bpf_op"]     = _get(r'\bop=(\w+)', line) or event["bpf_op"]


            elif rec_type in AVC_TYPES:
                event["type"] = rec_type
                event["timestamp"]   = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["avc_result"]  = _get(r'apparmor="(\w+)"', line) or event["avc_result"]
                event["avc_op"]      = _get(r'operation="(\w+)"', line) or event["avc_op"]
                event["avc_profile"] = _get(r'profile="([^"]+)"', line) or event["avc_profile"]
                event["avc_name"]    = _get(r'name="([^"]+)"', line) or event["avc_name"]
                event["pid"]         = _get(r'\bpid=(\d+)', line) or event["pid"]
                event["comm"]        = _get(r'comm="([^"]+)"', line) or event["comm"]


            elif rec_type in ERR_TYPES:
                event["type"] = rec_type
                event["timestamp"] = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["op"]        = _get(r"op=([^\s']+)", line) or event["op"]
                event["exe"]       = _get(r'exe="([^"]+)"', line) or event["exe"]
                event["res"]       = _get(r'res=(\w+)', line) or event["res"]
                acct = _get(r'acct="([^"]+)"', line)
                if acct: event["acct"] = acct


            elif rec_type in ANOM_ABEND_TYPES:
                event["type"] = rec_type
                event["timestamp"] = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["pid"]  = _get(r'\bpid=(\d+)', line) or event["pid"]
                event["comm"] = _get(r'comm="([^"]+)"', line) or event["comm"]
                event["exe"]  = _get(r'exe="([^"]+)"', line) or event["exe"]
                event["sig"]  = _get(r'\bsig=(\d+)', line) or event["sig"]
                event["res"]  = _get(r'res=(\w+)', line) or event["res"]


            elif rec_type in ANOM_PROMISC_TYPES:
                event["type"] = rec_type
                event["timestamp"] = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["dev"]  = _get(r'\bdev=(\S+)', line) or event["dev"]
                event["prom"] = _get(r'\bprom=(\d+)', line) or event["prom"]
                uidnum = _get(r'\buid=(\d+)', line)
                if uidnum: event["acct"] = f"uid:{uidnum}"


            elif rec_type in NETFILTER_TYPES:
                event["type"] = rec_type
                event["timestamp"] = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["nf_table"] = _get(r'\btable=(\S+)', line) or event["nf_table"]
                event["nf_op"]    = _get(r'\bop=(\S+)', line) or event["nf_op"]
                event["pid"]      = _get(r'\bpid=(\d+)', line) or event["pid"]
                event["comm"]     = _get(r'comm="([^"]+)"', line) or event["comm"]


            elif rec_type in SHUTDOWN_TYPES:
                event["type"] = rec_type
                event["timestamp"] = _get(r'msg=audit\(([\d.]+):\d+\)', line) or event["timestamp"]
                event["pid"]  = _get(r'\bpid=(\d+)', line) or event["pid"]
                event["comm"] = _get(r'comm="([^"]+)"', line) or event["comm"]
                event["exe"]  = _get(r'exe="([^"]+)"', line) or event["exe"]
                event["res"]  = _get(r'res=(\w+)', line) or event["res"]


            elif rec_type == "BPRM_FCAPS":
                event["fcaps"] = "yes"


            elif rec_type == "MMAP":
                event["mmap_flags"] = _get(r'\bflags=(\S+)', line) or event["mmap_flags"]


        if has_syscall:
            if not event["syscall"]:
                continue
            sc = event["syscall"]
            if sc == "execve" and event["success"] == "yes":
                event["dest"] = event["exe"]
            elif sc == "execve" and event["success"] == "no":
                if event["paths"]:
                    event["dest"] = _select_path(event["paths"], event["path_types"]) + " [not found]"
            elif sc not in ("connect",) and event["paths"]:
                p = _select_path(event["paths"], event["path_types"])
                if p and p != "/lib64/ld-linux-x86-64.so.2":
                    event["dest"] = p
            results.append(event)


        elif event["type"] in PAM_TYPES:
            if event["exe"] and event["pam_acct"]:
                results.append(event)


        elif event["type"] in ANOM_ABEND_TYPES:
            if event["pid"]:
                results.append(event)


        elif event["type"] in ANOM_PROMISC_TYPES:
            if event["dev"]:
                results.append(event)


        elif event["type"] in NETFILTER_TYPES:
            if event["nf_op"]:
                results.append(event)


        elif event["type"] in SHUTDOWN_TYPES:
            if event["pid"]:
                results.append(event)


        elif event["type"] in NON_SYSCALL_TYPES:
            if any([event["acct"], event["op"], event["unit"], event["cmd"],
                    event["avc_result"], event["bpf_op"]]):
                results.append(event)


    pid_to_exe = {e["pid"]: e["exe"] for e in results if e["pid"] and e["exe"]}
    for e in results:
        if e["type"] in PAM_TYPES:
            e["src"] = e["exe"] or f"pid:{e['pid']}"
            e["dest"] = f"pam:{e['pam_acct']}:{e['type']}"
            continue
        if e["type"] in AUTH_TYPES:
            e["src"] = e["addr"] or e["hostname"] or e["exe"] or e["comm"] or f"pid:{e['pid']}"
            e["dest"] = f"acct:{e['acct']}:{e['type']}"
            continue
        if e["type"] in ACCOUNT_TYPES:
            e["src"] = e["exe"] or f"pid:{e['pid']}"
            e["dest"] = f"acct:{e['acct']}:{e['type']}"
            continue
        if e["type"] in SERVICE_TYPES:
            e["src"] = e["exe"] or e["comm"] or f"pid:{e['pid']}"
            e["dest"] = f"unit:{e['unit']}"
            continue
        if e["type"] in CMD_TYPES:
            e["src"] = f"acct:{e['acct']}" if e["acct"] else f"pid:{e['pid']}"
            e["dest"] = e["cmd"] or e["exe"]
            continue
        if e["type"] in BPF_TYPES:
            e["src"] = f"bpf:{e['bpf_progid']}"
            e["dest"] = e["bpf_op"]
            continue
        if e["type"] in AVC_TYPES:
            e["src"] = e["comm"] or f"pid:{e['pid']}"
            e["dest"] = e["avc_name"] or e["avc_profile"]
            continue
        if e["type"] in ERR_TYPES:
            e["src"] = e["exe"] or f"pid:{e['pid']}"
            e["dest"] = f"acct:{e['acct']}:{e['type']}"
            continue
        if e["type"] in ANOM_ABEND_TYPES:
            e["src"] = e["exe"] or e["comm"] or f"pid:{e['pid']}"
            e["dest"] = f"crash:sig={e['sig']}"
            continue
        if e["type"] in ANOM_PROMISC_TYPES:
            e["src"] = e["acct"] or "unknown"
            e["dest"] = f"iface:{e['dev']}"
            continue
        if e["type"] in NETFILTER_TYPES:
            e["src"] = e["comm"] or f"pid:{e['pid']}"
            e["dest"] = f"netfilter:{e['nf_table']}"
            continue
        if e["type"] in SHUTDOWN_TYPES:
            e["src"] = e["exe"] or e["comm"] or f"pid:{e['pid']}"
            e["dest"] = "system:shutdown"
            continue

        sc = e["syscall"]
        if sc == "execve" and e["success"] == "yes":
            e["src"] = pid_to_exe.get(e["ppid"], f"pid:{e['ppid']}")
        else:
            e["src"] = e["exe"] or e["comm"] or f"pid:{e['pid']}"

    return results


def print_events(events):
    for e in events:
        if e["type"] in PAM_TYPES:
            print(f"[{e['event_id']}]  {e['type']}  @ {e['timestamp']}")
            print(f"  exe     : {e['exe']}")
            print(f"  acct    : {e['pam_acct']}")
            print(f"  op      : {e['pam_op']}")
            print(f"  res     : {e['pam_res']}")
            print(f"  subj    : {e['subj']}")
            print()
            continue

        if e["type"] in AUTH_TYPES:
            print(f"[{e['event_id']}]  {e['type']}  @ {e['timestamp']}")
            print(f"  exe     : {e['exe']}")
            print(f"  acct    : {e['acct']}")
            print(f"  op      : {e['op']}")
            print(f"  from    : {e['addr'] or e['hostname']}")
            print(f"  res     : {e['res']}")
            print()
            continue

        if e["type"] in ACCOUNT_TYPES:
            print(f"[{e['event_id']}]  {e['type']}  @ {e['timestamp']}")
            print(f"  exe     : {e['exe']}")
            print(f"  acct    : {e['acct']}")
            print(f"  op      : {e['op']}")
            print(f"  res     : {e['res']}")
            print()
            continue

        if e["type"] in SERVICE_TYPES:
            print(f"[{e['event_id']}]  {e['type']}  @ {e['timestamp']}")
            print(f"  unit    : {e['unit']}")
            print(f"  comm    : {e['comm']}")
            print(f"  res     : {e['res']}")
            print()
            continue

        if e["type"] in CMD_TYPES:
            print(f"[{e['event_id']}]  {e['type']}  @ {e['timestamp']}")
            print(f"  acct    : {e['acct']}")
            print(f"  cmd     : {e['cmd']}")
            print(f"  op      : {e['op']}")
            print(f"  res     : {e['res']}")
            print()
            continue

        if e["type"] in BPF_TYPES:
            print(f"[{e['event_id']}]  {e['type']}  @ {e['timestamp']}")
            print(f"  prog-id : {e['bpf_progid']}")
            print(f"  op      : {e['bpf_op']}")
            print()
            continue

        if e["type"] in AVC_TYPES:
            print(f"[{e['event_id']}]  {e['type']}  @ {e['timestamp']}")
            print(f"  result  : {e['avc_result']}")
            print(f"  op      : {e['avc_op']}")
            print(f"  name    : {e['avc_name']}")
            print(f"  profile : {e['avc_profile']}")
            print()
            continue

        if e["type"] in ERR_TYPES:
            print(f"[{e['event_id']}]  {e['type']}  @ {e['timestamp']}")
            print(f"  exe     : {e['exe']}")
            print(f"  acct    : {e['acct']}")
            print(f"  op      : {e['op']}")
            print(f"  res     : {e['res']}")
            print()
            continue

        status = "" if e["success"] == "yes" else " [FAILED]"
        print(f"[{e['event_id']}]  {e['syscall']}{status}  @ {e['timestamp']}")
        print(f"  src     : {e['src']}")
        print(f"  edge    : {e['syscall']}")
        print(f"  dst     : {e['dest']}")
        print(f"  pid     : ppid={e['ppid']} → pid={e['pid']}")
        if e["subj"]:
            print(f"  subj    : {e['subj']}")
        if e["key"]:
            print(f"  key     : {e['key']}")
        if e["exit"] and e["exit"] != "0":
            print(f"  exit    : {e['exit']}")
        if e["command"]:
            print(f"  command : {e['command']}")
        for p in e["paths"]:
            print(f"  path    : {p}")
        print()


def to_subgraph(events, tactic, attack, description=""):
    node_map  = {}
    node_list = []
    edges     = []
    counter   = [0]

    def get_node(key, name, ntype, ts):
        if key not in node_map:
            nid  = counter[0]; counter[0] += 1
            nuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
            node_map[key] = {"id": nid, "uuid": nuid}
            node_list.append([nid, {"type": ntype, "ts": ts, "name": name, "uuid": nuid}])
        return node_map[key]["uuid"]

    def proc_node(pid, exe_or_comm, ts):
        name = exe_or_comm or (f"pid:{pid}" if pid else "unknown")
        key  = f"pid:{pid}:{name}" if pid else name
        return get_node(key, name, "SUBJECT_PROCESS", ts)

    pid_to_exe = {e["pid"]: e["exe"] for e in events if e["pid"] and e["exe"]}

    for e in events:
        ts = int(float(e["timestamp"]) * 1e9) if e["timestamp"] else 0

        if e["type"] in PAM_TYPES:
            src = proc_node(e["pid"], e["exe"], ts)
            dst = get_node(f"acct:{e['pam_acct']}", f"acct:{e['pam_acct']}", "PRINCIPAL_LOCAL", ts)
            edges.append([src, dst, 0, {"edge_type": f"EVENT_{e['type']}", "pam_op": e["pam_op"],
                                        "result": e["pam_res"], "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in AUTH_TYPES:
            if e["addr"] or e["hostname"]:
                name = e["addr"] or e["hostname"]
                src = get_node(name, name, "NetFlowObject", ts)
            else:
                src = proc_node(e["pid"], e["exe"], ts)
            dst = get_node(f"acct:{e['acct']}", f"acct:{e['acct']}", "PRINCIPAL_LOCAL", ts)
            edges.append([src, dst, 0, {"edge_type": f"EVENT_{e['type']}",
                                        "result": e["res"], "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in ACCOUNT_TYPES:
            src = proc_node(e["pid"], e["exe"], ts)
            dst = get_node(f"acct:{e['acct']}", f"acct:{e['acct']}", "PRINCIPAL_LOCAL", ts)
            edges.append([src, dst, 0, {"edge_type": f"EVENT_{e['type']}",
                                        "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in SERVICE_TYPES:
            src = proc_node(e["pid"], e["exe"] or e["comm"], ts)
            dst = get_node(f"unit:{e['unit']}", f"unit:{e['unit']}", "SUBJECT_PROCESS", ts)
            edges.append([src, dst, 0, {"edge_type": f"EVENT_{e['type']}",
                                        "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in CMD_TYPES:
            if e["acct"]:
                src = get_node(f"acct:{e['acct']}", f"acct:{e['acct']}", "PRINCIPAL_LOCAL", ts)
            else:
                src = proc_node(e["pid"], None, ts)
            dst = proc_node(e["pid"], e["cmd"] or e["exe"] or "unknown", ts)
            edges.append([src, dst, 0, {"edge_type": "EVENT_SUDO_CMD",
                                        "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in BPF_TYPES:
            src = get_node("kernel", "kernel", "SUBJECT_PROCESS", ts)
            dst = get_node(f"bpf-prog:{e['bpf_progid']}", f"bpf-prog:{e['bpf_progid']}", "SUBJECT_PROCESS", ts)
            edges.append([src, dst, 0, {"edge_type": f"EVENT_BPF_{e['bpf_op']}",
                                        "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in AVC_TYPES:
            src = proc_node(e["pid"], e["comm"], ts)
            target = e["avc_name"] or e["avc_profile"] or "unknown"
            dst = get_node(target, target, "FileObject", ts)
            edges.append([src, dst, 0, {"edge_type": f"EVENT_AVC_{e['avc_result']}",
                                        "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in ERR_TYPES:
            src = proc_node(e["pid"], e["exe"], ts)
            if e["acct"]:
                dst = get_node(f"acct:{e['acct']}", f"acct:{e['acct']}", "PRINCIPAL_LOCAL", ts)
            else:
                label = f"error:{e['op']}" if e["op"] else "error:unknown"
                dst = get_node(label, label, "FileObject", ts)
            edges.append([src, dst, 0, {"edge_type": f"EVENT_{e['type']}", "op": e["op"],
                                        "result": e["res"], "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in ANOM_ABEND_TYPES:
            src = proc_node(e["pid"], e["exe"] or e["comm"], ts)
            label = f"crash:sig={e['sig']}"
            dst = get_node(label, label, "FileObject", ts)
            edges.append([src, dst, 0, {"edge_type": "EVENT_ANOM_ABEND", "signal": e["sig"],
                                        "result": e["res"], "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in ANOM_PROMISC_TYPES:
            src = get_node(e["acct"] or "unknown", e["acct"] or "unknown", "PRINCIPAL_LOCAL", ts)
            label = f"iface:{e['dev']}"
            dst = get_node(label, label, "NetFlowObject", ts)
            edges.append([src, dst, 0, {"edge_type": "EVENT_ANOM_PROMISCUOUS", "prom": e["prom"],
                                        "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in NETFILTER_TYPES:
            src = proc_node(e["pid"], e["comm"], ts)
            label = f"netfilter:{e['nf_table']}"
            dst = get_node(label, label, "FileObject", ts)
            edges.append([src, dst, 0, {"edge_type": f"EVENT_NETFILTER_{e['nf_op']}",
                                        "event_id": e["event_id"], "ts": ts}])
            continue

        if e["type"] in SHUTDOWN_TYPES:
            src = proc_node(e["pid"], e["exe"] or e["comm"], ts)
            dst = get_node("system:shutdown", "system:shutdown", "FileObject", ts)
            edges.append([src, dst, 0, {"edge_type": "EVENT_SYSTEM_SHUTDOWN",
                                        "result": e["res"], "event_id": e["event_id"], "ts": ts}])
            continue

        sc = e["syscall"]
        if not sc:
            continue

        extra_attrs = {}
        if e["fcaps"]:
            extra_attrs["fcaps"] = e["fcaps"]
        if e["mmap_flags"]:
            extra_attrs["mmap_flags"] = e["mmap_flags"]

        if sc == "execve":
            if e["success"] == "yes":
                parent_exe = pid_to_exe.get(e["ppid"])
                src = proc_node(e["ppid"], parent_exe, ts)
                dst = proc_node(e["pid"], e["exe"] or e["comm"], ts)
                edges.append([src, dst, 0, {"edge_type": "EVENT_EXECVE",
                                            "event_id": e["event_id"], "ts": ts, **extra_attrs}])
            else:
                src = proc_node(e["pid"], e["exe"], ts)
                target = (e["paths"][0] + " [not found]") if e["paths"] else "unknown"
                dst = get_node(target, target, "FileObject", ts)
                edges.append([src, dst, 0, {"edge_type": "EVENT_EXECVE_FAIL",
                                            "event_id": e["event_id"], "ts": ts, **extra_attrs}])

        elif sc == "connect" and e["dest"]:
            src = proc_node(e["pid"], e["exe"], ts)
            ntype = "NetFlowObject" if ":" in e["dest"] else "FileObject"
            dst = get_node(e["dest"], e["dest"], ntype, ts)
            edges.append([src, dst, 0, {"edge_type": "EVENT_CONNECT",
                                        "event_id": e["event_id"], "ts": ts, **extra_attrs}])

        elif e["dest"]:
            src = proc_node(e["pid"], e["exe"], ts)
            dst = get_node(e["dest"], e["dest"], "FileObject", ts)
            edges.append([src, dst, 0, {"edge_type": f"EVENT_{sc.upper()}",
                                        "event_id": e["event_id"], "ts": ts, **extra_attrs}])

        else:
            src = proc_node(e["pid"], e["exe"], ts)
            label = f"syscall:{sc}"
            dst = get_node(label, label, "SUBJECT_PROCESS", ts)
            status = "EVENT_" + sc.upper() + ("" if e["success"] == "yes" else "_FAIL")
            edges.append([src, dst, 0, {"edge_type": status,
                                        "event_id": e["event_id"], "ts": ts, **extra_attrs}])

    return {
        "tactic"     : tactic,
        "attack"     : attack,
        "description": description,
        "n_nodes"    : len(node_list),
        "n_edges"    : len(edges),
        "nodes"      : node_list,
        "edges"      : edges,
    }


def iter_audit_logs(base_dir, tactics):
    infos = []
    for tactic in tactics:
        tactic_dir = Path(base_dir) / tactic
        if not tactic_dir.is_dir():
            continue
        for log_path in sorted(tactic_dir.rglob("audit.log")):
            rel = log_path.relative_to(tactic_dir)
            technique = rel.parts[0]
            step      = rel.parts[1] if len(rel.parts) > 1 else "?"
            host      = rel.parts[2] if len(rel.parts) > 2 else "?"
            logs_idx  = rel.parts.index("logs") if "logs" in rel.parts else None
            logdir    = rel.parts[logs_idx + 1] if logs_idx is not None and len(rel.parts) > logs_idx + 1 else "log"
            infos.append({
                "path": log_path,
                "tactic": tactic,
                "technique": technique,
                "step": step,
                "host": host,
                "logdir": logdir,
            })


    key_counts = {}
    for info in infos:
        key = (info["tactic"], info["technique"], info["step"], info["host"])
        key_counts[key] = key_counts.get(key, 0) + 1
    for info in infos:
        key = (info["tactic"], info["technique"], info["step"], info["host"])
        info["host_tag"] = f"{info['host']}-{info['logdir']}" if key_counts[key] > 1 else info["host"]


    step_hosts = {}
    for info in infos:
        skey = (info["tactic"], info["technique"], info["step"])
        step_hosts.setdefault(skey, set()).add(info["host_tag"])

    multi_host_steps = {skey for skey, hosts in step_hosts.items() if len(hosts) > 1}
    step_host_counts = {}
    for info in infos:
        skey = (info["tactic"], info["technique"], info["step"])
        if skey not in multi_host_steps:
            continue
        counts = step_host_counts.setdefault(skey, {})
        if info["host_tag"] not in counts:
            counts[info["host_tag"]] = len(parse_audit(info["path"]))

    chosen_host = {
        skey: pick_primary_host(step_host_counts[skey]) if skey in multi_host_steps
              else next(iter(hosts))
        for skey, hosts in step_hosts.items()
    }
    for info in infos:
        skey = (info["tactic"], info["technique"], info["step"])
        if len(step_hosts[skey]) == 1:
            info["is_primary"] = True
        else:
            info["is_primary"] = (info["host_tag"] == chosen_host[skey])

    yield from infos


def batch_main(base_dir, tactics, out_dir, events_out_dir, write_json=True, write_subgraphs=True):
    type_counts = {}
    n_files = 0
    n_events = 0

    n_skipped = 0
    for info in iter_audit_logs(base_dir, tactics):
        if info["is_primary"] is False:
            n_skipped += 1
            continue

        n_files += 1
        events = parse_audit(info["path"])
        n_events += len(events)
        for e in events:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1

        if write_json:
            events_dir = Path(events_out_dir) / info["tactic"] / info["technique"]
            events_dir.mkdir(parents=True, exist_ok=True)
            json_name = f"audit_{info['step']}_{info['host_tag']}.json"
            with open(events_dir / json_name, "w") as f:
                json.dump(_drop_nulls(events), f, indent=2)

        if write_subgraphs:
            sg = to_subgraph(
                events,
                tactic=info["tactic"],
                attack=info["technique"],
                description=f"{info['technique']} step={info['step']} host={info['host']} logdir={info['logdir']}",
            )
            sg_dir = Path(out_dir) / info["tactic"] / info["technique"]
            sg_dir.mkdir(parents=True, exist_ok=True)
            sg_name = f"subgraph_{info['step']}_{info['host_tag']}.json"
            with open(sg_dir / sg_name, "w") as f:
                json.dump(sg, f, indent=2)

    print()
    print(f"Parsed {n_files} audit.log files, {n_events} total events "
          f"({n_skipped} non-primary host logs skipped).")
    print("Event-type breakdown:")
    for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {t:15s} {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse CAM-LDS audit.log files.")
    ap.add_argument("log", nargs="?", default=None,
                     help="parse a single audit.log file (old single-file mode)")
    ap.add_argument("--batch", action="store_true",
                     help="parse every audit.log under --base/<tactic> for all --tactics")
    ap.add_argument("--base", default="/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic",
                     help="grouped_by_tactic directory")
    ap.add_argument("--tactics", nargs="+",
                     default=["command_and_control", "credential_access", "execution", "initial_access",
                              "lateral_movement", "persistence"],
                     help="tactic subdirectories to walk in batch mode")
    ap.add_argument("--out", default="/csse/research/contructive-learning/CAM-LDS/parser/subgraphs",
                     help="output directory for subgraph JSON in batch mode")
    ap.add_argument("--events-out", default="/csse/research/contructive-learning/CAM-LDS/parser/parsed_events",
                     help="output directory for parsed-event JSON in batch mode")
    ap.add_argument("--json", action="store_true", help="print raw JSON (single-file mode)")
    args = ap.parse_args()

    if args.batch or args.log is None:
        batch_main(args.base, args.tactics, args.out, args.events_out)
        sys.exit(0)

    log = args.log
    events = parse_audit(log)

    out_path = Path(log).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(_drop_nulls(events), f, indent=2)
    print(f"Saved {len(events)} events → {out_path}")

    subgraph = to_subgraph(events,
                           tactic="Initial_Access",
                           attack="ExploitPublicFacingApplication",
                           description="ZoneMinder exploit — python3 reverse shell to 192.42.1.174:4444")
    sg_dir  = Path("/csse/research/contructive-learning/output/theia/intial access")
    sg_dir.mkdir(parents=True, exist_ok=True)
    sg_path = sg_dir / "subgraph_ExploitPublicFacingApplication.json"
    with open(sg_path, "w") as f:
        json.dump(subgraph, f, indent=2)
    print(f"Saved subgraph    → {sg_path}")

    if args.json:
        print(json.dumps(events, indent=2))
    else:
        print_events(events)

