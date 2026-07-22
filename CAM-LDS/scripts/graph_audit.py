import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import defaultdict
from parse_audit import (
    parse_audit, PAM_TYPES, AUTH_TYPES, ACCOUNT_TYPES, SERVICE_TYPES,
    CMD_TYPES, BPF_TYPES, AVC_TYPES, ERR_TYPES, NON_SYSCALL_TYPES,
)

FILE_SYSCALLS = {"open","openat","openat2","creat","read","write",
                 "unlink","unlinkat","rename","renameat","chmod",
                 "chown","stat","lstat","access","faccessat",
                 "readlink","symlink","link","mkdir","rmdir","setxattr"}


OPEN_SYSCALLS = {"open", "openat", "openat2", "creat"}
FD_SYSCALLS   = {"fchmod", "fchown", "fsetxattr", "fgetxattr", "fremovexattr",
                 "fstat", "fstatfs", "fsync", "fdatasync", "ftruncate",
                 "flock", "fallocate", "fchdir"}
PROC_DIRS = ("/usr/bin/","/usr/sbin/","/bin/","/sbin/",
             "/usr/local/bin/","/usr/local/sbin/")

PAM_TYPES_LOWER         = {t.lower() for t in PAM_TYPES}
NON_SYSCALL_TYPES_LOWER = {t.lower() for t in NON_SYSCALL_TYPES}

AUTH_TYPES_LOWER    = {t.lower() for t in AUTH_TYPES}
ACCOUNT_TYPES_LOWER = {t.lower() for t in ACCOUNT_TYPES}
SERVICE_TYPES_LOWER = {t.lower() for t in SERVICE_TYPES}
CMD_TYPES_LOWER      = {t.lower() for t in CMD_TYPES}
BPF_TYPES_LOWER      = {t.lower() for t in BPF_TYPES}
AVC_TYPES_LOWER      = {t.lower() for t in AVC_TYPES}
ERR_TYPES_LOWER      = {t.lower() for t in ERR_TYPES}

SHELLS = {"/bin/bash", "/bin/sh", "/bin/dash", "/usr/bin/bash",
          "/usr/bin/sh", "/usr/bin/dash"}


def _resolve_binary(name):
    if name.startswith("/"):
        return name
    for d in PROC_DIRS:
        return d + name
    return name


def _shell_c_child(command):
    import shlex
    try:
        parts = shlex.split(command)
    except Exception:
        return None

    if len(parts) < 3 or parts[1] != "-c":
        return None
    inner = parts[2]

    first = inner.split()[0] if inner.strip() else None
    if not first:
        return None

    first = first.lstrip("(").rstrip(")")
    return _resolve_binary(first)


def build_graph(events):
    pid_to_exe = {e["pid"]: e["exe"] for e in events if e["pid"] and e["exe"]}
    edge_data  = defaultdict(list)


    fd_table = {}

    def resolve_fd(pid, a0_hex):
        if not pid or not a0_hex:
            return None
        try:
            fd = int(a0_hex, 16)
        except ValueError:
            return None
        return fd_table.get(pid, {}).get(fd)

    for e in events:
        if e.get("type") in NON_SYSCALL_TYPES:
            if e.get("src") and e.get("dest"):
                edge_data[(e["src"], e["dest"], e["type"].lower())].append(e["event_id"])
            continue

        sc = e["syscall"]
        if not sc:
            continue
        pid = e.get("pid")
        src = e["exe"] or e["comm"] or f"pid:{pid}"


        if sc in OPEN_SYSCALLS and e["success"] == "yes" and e["paths"]:
            try:
                fd_num = int(e["exit"])
            except (TypeError, ValueError):
                fd_num = None
            if fd_num is not None and fd_num >= 0:
                fd_table.setdefault(pid, {})[fd_num] = e["paths"][-1]

        handled = False

        if sc == "execve":
            if e["success"] == "yes":
                src = pid_to_exe.get(e["ppid"], f"pid:{e['ppid']}")
                dst = e["exe"] or e["comm"]
                key = "execve"
                if src == dst:
                    if e["paths"] and e["paths"][0] != dst:
                        dst = e["paths"][0]
                    else:
                        dst = f"{dst} [pid:{pid}]"
            else:
                src = e["exe"] or e["comm"] or f"pid:{pid}"
                dst = (e["paths"][0].rsplit("/",1)[-1] + " [not found]")\
                      if e["paths"] else "unknown [not found]"
                key = "execve_fail"
            edge_data[(src, dst, key)].append(e["event_id"])


            if e["success"] == "yes" and e["exe"] in SHELLS and e.get("command"):
                child = _shell_c_child(e["command"])
                if child and child != e["exe"]:
                    edge_data[(e["exe"], child, "execve_inferred")].append(e["event_id"])
            handled = True

        elif sc == "connect":
            dst = e["dest"] or "connect [no target]"
            edge_data[(src, dst, "connect")].append(e["event_id"])
            handled = True

        elif sc in FILE_SYSCALLS and e["paths"]:
            paths  = e["paths"]
            ptypes = e.get("path_types") or [None] * len(paths)

            if sc in ("rename", "renameat", "renameat2"):
                old_name = next((f for f, t in zip(paths, ptypes) if t == "DELETE"), None)
                new_name = next((f for f, t in zip(paths, ptypes) if t == "CREATE"), None)
                if old_name and new_name:
                    edge_data[(src, f"{old_name} -> {new_name}", sc)].append(e["event_id"])
                elif old_name or new_name:
                    edge_data[(src, old_name or new_name, sc)].append(e["event_id"])
                else:
                    candidates = [f for f, t in zip(paths, ptypes)
                                 if t != "PARENT" and f not in ("/lib64/ld-linux-x86-64.so.2", ".")]
                    if candidates:
                        dst = " -> ".join(dict.fromkeys(candidates))
                        edge_data[(src, dst, sc)].append(e["event_id"])
                    else:
                        edge_data[(src, f"{sc} [filtered target]", sc)].append(e["event_id"])
            else:
                added_any = False
                for fname, ptype in zip(paths, ptypes):
                    if fname in ("/lib64/ld-linux-x86-64.so.2", "."):
                        continue
                    if ptype == "PARENT":
                        continue
                    edge_data[(src, fname, sc)].append(e["event_id"])
                    added_any = True
                if not added_any:
                    edge_data[(src, f"{sc} [filtered target]", sc)].append(e["event_id"])
            handled = True

        elif sc in FD_SYSCALLS:
            resolved = resolve_fd(pid, e.get("a0"))
            dst = resolved or f"fd:{e.get('a0')} [unresolved]"
            edge_data[(src, dst, sc)].append(e["event_id"])
            handled = True

        if not handled:
            dst = (e["paths"][0] if e["paths"] else None) or e.get("dest")\
                  or f"{sc} [no target]"
            edge_data[(src, dst, sc)].append(e["event_id"])

    G = nx.MultiDiGraph()
    for (src, dst, sc), ids in edge_data.items():
        n       = len(ids)
        ids_str = ",".join(ids[:6]) + (f"+{n-6}" if n > 6 else "")

        if sc == "execve_fail":
            ec    = "#e64553"
            style = "dashed"
            lbl   = f"EXECVE FAIL ×{n} [{ids_str}]" if n > 1\
                    else f"EXECVE FAIL [{ids[0]}]"
        elif sc == "execve_inferred":
            ec    = "#f38ba8"
            style = "dashed"
            lbl   = f"EXECVE [inferred] [{ids_str}]"
        elif sc in PAM_TYPES_LOWER:
            ec    = "#a6e3a1"
            style = "dashed"
            lbl   = f"{sc.upper()} [{ids[0]}]"
        elif sc in AUTH_TYPES_LOWER:
            ec    = "#fab387"
            style = "dashed"
            lbl   = f"{sc.upper()} [{ids[0]}]"
        elif sc in ACCOUNT_TYPES_LOWER:
            ec    = "#f9e2af"
            style = "dashed"
            lbl   = f"{sc.upper()} [{ids[0]}]"
        elif sc in SERVICE_TYPES_LOWER:
            ec    = "#94e2d5"
            style = "dashed"
            lbl   = f"{sc.upper()} [{ids[0]}]"
        elif sc in CMD_TYPES_LOWER:
            ec    = "#cba6f7"
            style = "dashed"
            lbl   = f"{sc.upper()} [{ids[0]}]"
        elif sc in BPF_TYPES_LOWER:
            ec    = "#eba0ac"
            style = "dashed"
            lbl   = f"{sc.upper()} [{ids[0]}]"
        elif sc in AVC_TYPES_LOWER:
            ec    = "#f2cdcd"
            style = "dashed"
            lbl   = f"{sc.upper()} [{ids[0]}]"
        elif sc in ERR_TYPES_LOWER:
            ec    = "#6c7086"
            style = "dashed"
            lbl   = f"{sc.upper()} [{ids[0]}]"
        else:
            ec    = "#f38ba8" if sc == "execve" else\
                    "#89dceb" if sc == "connect" else "#a6e3a1"
            style = "solid"
            lbl   = f"{sc.upper()} ×{n} [{ids_str}]" if n > 1\
                    else f"{sc.upper()} [{ids[0]}]"

        G.add_edge(src, dst, key=sc, label=lbl, color=ec, style=style,
                   syscall=sc, count=n, eids=ids_str)
    return G


def node_color(name):
    if name.startswith("pam:"):
        return "#1b5e20"
    if name.startswith("pid:"):
        return "#1a237e"
    if ":" in name and name.split(":")[-1].isdigit():
        return "#c0392b"
    if any(name.startswith(d) for d in PROC_DIRS):
        return "#1a237e"
    return "#6a0dad"

def node_edge_color(name):
    if name.startswith("pam:"):
        return "#4caf50"
    if name.startswith("pid:"):
        return "#3949ab"
    if ":" in name and name.split(":")[-1].isdigit():
        return "#e74c3c"
    if any(name.startswith(d) for d in PROC_DIRS):
        return "#3949ab"
    return "#8e24aa"


def draw_graph(G, title, out_path,
               border_color_fn=None, border_width_fn=None):
    if border_color_fn is None:
        border_color_fn = node_edge_color
    if border_width_fn is None:
        border_width_fn = lambda n: 3.0

    fig, ax = plt.subplots(figsize=(28, 19))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    pos = nx.spring_layout(G, seed=7, k=3.8)

    fill_colors   = [node_color(n)       for n in G.nodes()]
    border_colors = [border_color_fn(n)  for n in G.nodes()]
    border_widths = [border_width_fn(n)  for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos,
                           node_color=fill_colors,
                           edgecolors=border_colors,
                           linewidths=border_widths,
                           node_size=3800, ax=ax, alpha=0.95)

    node_labels = {}
    for n in G.nodes():
        if "/" in n and not n.startswith("pam:"):
            parent, base = n.rsplit("/", 1)
            node_labels[n] = f"{parent}/\n{base}"
        else:
            node_labels[n] = n

    nx.draw_networkx_labels(G, pos, labels=node_labels,
                            font_size=7.5, font_color="#cdd6f4",
                            font_weight="bold", ax=ax)

    solid_edges  = [(u,v) for u,v in G.edges() if G[u][v].get("style","solid")=="solid"]
    dashed_edges = [(u,v) for u,v in G.edges() if G[u][v].get("style")=="dashed"]

    nx.draw_networkx_edges(G, pos, edgelist=solid_edges,
                           edge_color=[G[u][v]["color"] for u,v in solid_edges],
                           arrows=True, arrowsize=20, width=1.8,
                           ax=ax, alpha=0.85,
                           min_source_margin=30, min_target_margin=30)
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges,
                           edge_color=[G[u][v]["color"] for u,v in dashed_edges],
                           arrows=True, arrowsize=20, width=1.5,
                           style="dashed", ax=ax, alpha=0.85,
                           min_source_margin=30, min_target_margin=30)

    edge_labels = {(u,v): G[u][v]["label"] for u,v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=edge_labels,
                                 font_size=7,
                                 font_color="#f9e2af",
                                 rotate=True,
                                 label_pos=0.45,
                                 bbox=dict(boxstyle="round,pad=0.25",
                                           fc="#11111b", ec="none", alpha=0.85),
                                 ax=ax)

    legend = [
        mpatches.Patch(color="#1a237e", label="Process"),
        mpatches.Patch(color="#c0392b", label="Network"),
        mpatches.Patch(color="#6a0dad", label="File / Socket"),
        mpatches.Patch(color="#1b5e20", label="PAM session"),
    ]
    ax.legend(handles=legend, loc="upper left",
              facecolor="#181825", edgecolor="#45475a",
              labelcolor="#cdd6f4", fontsize=10,
              framealpha=0.95, borderpad=1.0)

    ax.set_title(title, color="#cdd6f4", fontsize=12, fontweight="bold", pad=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved → {out_path}")


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else\
        "/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic/initial_access/T1190-000/1_autostart_localaccount-5/videoserver/logs/log/audit/audit.log"
    out = sys.argv[2] if len(sys.argv) > 2 else\
        "/csse/research/contructive-learning/CAM-LDS/provenance_graph.png"

    events = parse_audit(log)
    G      = build_graph(events)
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    title = f"Provenance Graph — nodes={G.number_of_nodes()}  edges={G.number_of_edges()}"
    draw_graph(G, title, out)
