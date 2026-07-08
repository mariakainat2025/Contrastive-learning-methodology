import sys
import re
import json
import uuid
from pathlib import Path


PAM_TYPES = {"USER_ACCT", "CRED_ACQ", "USER_START", "USER_END", "CRED_REFR", "CRED_DISP"}


def decode_arg(val):
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    try:
        return bytes.fromhex(val).decode('utf-8', errors='replace')
    except Exception:
        return val


def parse_audit(path):
    events = {}
    order  = []

    with open(path) as f:
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
            "type"      : "SYSCALL",   # SYSCALL or PAM type
            "syscall"   : None,
            "success"   : None,
            "exit"      : None,        # numeric return value e.g. -115 EINPROGRESS
            "ppid"      : None,
            "pid"       : None,
            "comm"      : None,
            "exe"       : None,
            "subj"      : None,        # docker-default / unconfined (AppArmor context)
            "key"       : None,        # MITRE tag from auditd rule e.g. T1166_Seuid_and_Setgid
            "command"   : None,
            "src"       : None,
            "dest"      : None,
            "paths"     : [],
            "pam_op"    : None,        # PAM:accounting / PAM:session_open etc.
            "pam_acct"  : None,        # account name (root, alice, ...)
            "pam_res"   : None,        # success / failed
        }

        has_syscall = False

        for line in lines:
            if not line.startswith("type="):
                continue
            rec_type = line.split()[0][5:]   # strip "type="

            # ── SYSCALL ──────────────────────────────────────────────
            if rec_type == "SYSCALL":
                has_syscall = True
                m = re.search(r'msg=audit\(([\d.]+):\d+\)', line)
                if m: event["timestamp"] = m.group(1)
                m = re.search(r'success=(\w+)', line)
                if m: event["success"] = m.group(1)
                m = re.search(r'\bexit=(-?\d+)', line)
                if m: event["exit"] = m.group(1)
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

            # ── EXECVE ───────────────────────────────────────────────
            elif rec_type == "EXECVE":
                args = re.findall(r'a\d+=((?:"[^"]*")|(?:[0-9A-Fa-f]+))', line)
                decoded = [decode_arg(a) for a in args]
                event["command"] = " ".join(decoded)

            # ── SOCKADDR ─────────────────────────────────────────────
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

            # ── PATH ─────────────────────────────────────────────────
            elif rec_type == "PATH":
                m = re.search(r'name="([^"]+)"', line)
                name = m.group(1) if m else None
                if name:
                    event["paths"].append(name)

            # ── PAM events ───────────────────────────────────────────
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

        # ── include SYSCALL events ────────────────────────────────────
        if has_syscall:
            if not event["syscall"]:
                continue
            sc = event["syscall"]
            if sc == "execve" and event["success"] == "yes":
                event["dest"] = event["exe"]
            elif sc == "execve" and event["success"] == "no":
                if event["paths"]:
                    event["dest"] = event["paths"][0] + " [not found]"
            elif sc not in ("connect",) and event["paths"]:
                p = event["paths"][0]
                if p and p != "/lib64/ld-linux-x86-64.so.2":
                    event["dest"] = p
            results.append(event)

        # ── include PAM events ────────────────────────────────────────
        elif event["type"] in PAM_TYPES:
            if event["exe"] and event["pam_acct"]:
                results.append(event)

    # ── second pass: fill src ─────────────────────────────────────────
    pid_to_exe = {e["pid"]: e["exe"] for e in results if e["pid"] and e["exe"]}
    for e in results:
        if e["type"] in PAM_TYPES:
            e["src"] = e["exe"] or f"pid:{e['pid']}"
            e["dest"] = f"pam:{e['pam_acct']}:{e['type']}"
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
    """Convert parsed events to the subgraph JSON format."""
    node_map  = {}
    node_list = []
    edges     = []
    counter   = [0]

    def get_node(name, ntype, ts):
        if name not in node_map:
            nid  = counter[0]; counter[0] += 1
            nuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, name))
            node_map[name] = {"id": nid, "uuid": nuid}
            node_list.append([nid, {"type": ntype, "ts": ts, "name": name, "uuid": nuid}])
        return node_map[name]["uuid"]

    pid_to_exe = {e["pid"]: e["exe"] for e in events if e["pid"] and e["exe"]}

    for e in events:
        if e["type"] in PAM_TYPES:
            continue

        sc = e["syscall"]
        if not sc:
            continue
        ts = int(float(e["timestamp"]) * 1e9) if e["timestamp"] else 0

        if sc == "execve":
            if e["success"] == "yes":
                parent_exe = pid_to_exe.get(e["ppid"], f"pid:{e['ppid']}")
                child_exe  = e["exe"] or e["comm"]
                src = get_node(parent_exe, "SUBJECT_PROCESS", ts)
                dst = get_node(child_exe,  "SUBJECT_PROCESS", ts)
                edges.append([src, dst, 0, {"edge_type": "EVENT_EXECVE",
                                            "event_id": e["event_id"], "ts": ts}])
            else:
                src = get_node(e["exe"] or f"pid:{e['pid']}", "SUBJECT_PROCESS", ts)
                target = (e["paths"][0] + " [not found]") if e["paths"] else "unknown"
                dst = get_node(target, "FileObject", ts)
                edges.append([src, dst, 0, {"edge_type": "EVENT_EXECVE_FAIL",
                                            "event_id": e["event_id"], "ts": ts}])

        elif sc == "connect" and e["dest"]:
            src = get_node(e["exe"] or f"pid:{e['pid']}", "SUBJECT_PROCESS", ts)
            ntype = "NetFlowObject" if ":" in e["dest"] else "FileObject"
            dst = get_node(e["dest"], ntype, ts)
            edges.append([src, dst, 0, {"edge_type": "EVENT_CONNECT",
                                        "event_id": e["event_id"], "ts": ts}])

    return {
        "tactic"     : tactic,
        "attack"     : attack,
        "description": description,
        "n_nodes"    : len(node_list),
        "n_edges"    : len(edges),
        "nodes"      : node_list,
        "edges"      : edges,
    }


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else \
        "/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic/initial_access/T1190-000/1_autostart_localaccount-5/videoserver/logs/log/audit/audit.log"

    events = parse_audit(log)

    out_path = Path(log).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(events, f, indent=2)
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

    if "--json" in sys.argv:
        print(json.dumps(events, indent=2))
    else:
        print_events(events)
