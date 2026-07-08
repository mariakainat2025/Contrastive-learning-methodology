import sys
import re

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

    for eid in order:
        lines = events[eid]

        timestamp = syscall_name = success = ppid = pid = comm = exe = uid = None
        command   = []
        dest      = None
        paths     = []

        for line in lines:

            # ── SYSCALL ──────────────────────────────────────────────
            if line.startswith("type=SYSCALL"):
                m = re.search(r'msg=audit\(([\d.]+):\d+\)', line)
                if m:
                    timestamp = m.group(1)

                m = re.search(r'success=(\w+)', line)
                if m:
                    success = m.group(1)

                m = re.search(r'ppid=(\d+)', line)
                if m:
                    ppid = m.group(1)

                m = re.search(r'\bpid=(\d+)', line)
                if m:
                    pid = m.group(1)

                m = re.search(r'comm="([^"]+)"', line)
                if m:
                    comm = m.group(1)

                m = re.search(r'exe="([^"]+)"', line)
                if m:
                    exe = m.group(1)

                m = re.search(r'UID="([^"]+)"', line)
                if m:
                    uid = m.group(1)

                after_arch = line.split("ARCH=")[-1]
                m = re.search(r'SYSCALL=(\w+)', after_arch)
                if m:
                    syscall_name = m.group(1)

            # ── EXECVE ───────────────────────────────────────────────
            elif line.startswith("type=EXECVE"):
                args = re.findall(r'a\d+=((?:"[^"]*")|(?:[0-9A-Fa-f]+))', line)
                command = [decode_arg(a) for a in args]

            # ── SOCKADDR ─────────────────────────────────────────────
            elif line.startswith("type=SOCKADDR"):
                m = re.search(r'laddr=([\d.]+)', line)
                laddr = m.group(1) if m else None
                m = re.search(r'lport=(\d+)', line)
                lport = m.group(1) if m else None
                m = re.search(r'path=([^\s}]+)', line)
                lpath = m.group(1) if m else None

                if laddr and lport:
                    dest = f"{laddr}:{lport}"
                elif lpath:
                    dest = lpath

            # ── PATH ─────────────────────────────────────────────────
            elif line.startswith("type=PATH"):
                m = re.search(r'name="([^"]+)"', line)
                name = m.group(1) if m else None
                m = re.search(r'OUID="([^"]+)"', line)
                ouid = m.group(1) if m else None
                if name:
                    paths.append({"name": name, "ouid": ouid})

        if not syscall_name:
            continue

        # ── PRINT CLEAN OUTPUT ────────────────────────────────────────
        status = "" if success == "yes" else " [FAILED]"
        print(f"[{eid}]  {syscall_name}{status}  @ {timestamp}")
        print(f"  process : {comm}  →  {exe}")
        print(f"  pid     : ppid={ppid} → pid={pid}")
        print(f"  user    : {uid}")
        if command:
            print(f"  command : {' '.join(command)}")
        if dest:
            print(f"  dest    : {dest}")
        for p in paths:
            print(f"  path    : {p['name']}  (owner={p['ouid']})")
        print()


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else \
        "/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic/initial_access/T1190-000/1_autostart_localaccount-5/videoserver/logs/log/audit/audit.log"
    parse_audit(log)
