import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import defaultdict
from parse_audit import parse_audit, PAM_TYPES

FILE_SYSCALLS = {"open","openat","openat2","creat","read","write",
                 "unlink","unlinkat","rename","renameat","chmod",
                 "chown","stat","lstat","access","faccessat",
                 "readlink","symlink","link","mkdir","rmdir"}
PROC_DIRS = ("/usr/bin/","/usr/sbin/","/bin/","/sbin/",
             "/usr/local/bin/","/usr/local/sbin/")

PAM_TYPES_LOWER = {t.lower() for t in PAM_TYPES}

SHELLS = {"/bin/bash", "/bin/sh", "/bin/dash", "/usr/bin/bash",
          "/usr/bin/sh", "/usr/bin/dash"}

# Processes automatically started by Linux on every SSH login (session setup noise)
SESSION_NOISE = {
    "/usr/lib/systemd/systemd-user-runtime-dir",
    "/usr/lib/systemd/systemd",
    "/usr/lib/systemd/systemd-logind",
    "/usr/lib/systemd/user-environment-generators/30-systemd-environment-d-generator",
    "/usr/lib/systemd/user-environment-generators/90gpg-agent",
    "/usr/lib/systemd/user-generators/systemd-xdg-autostart-generator",
    "/usr/bin/systemctl",
    "/usr/bin/gpgconf",
    "/usr/bin/gpg",
    "/usr/bin/gpg-agent",
    "/usr/bin/gpgsm",
    "/usr/bin/dirmngr",
    "/usr/bin/gawk",
    "/usr/bin/pinentry-curses",
}


def _resolve_binary(name):
    """Try to resolve a bare binary name to a full path via PROC_DIRS."""
    if name.startswith("/"):
        return name
    for d in PROC_DIRS:
        return d + name   # return first candidate — will be labelled [inferred]
    return name


def _shell_c_child(command):
    """Extract the first binary from a 'shell -c <cmd>' command string.

    Returns full path or bare name, or None if not a -c invocation.
    """
    import shlex
    try:
        parts = shlex.split(command)
    except Exception:
        return None
    # expect:  bash  -c  <inner_cmd>
    if len(parts) < 3 or parts[1] != "-c":
        return None
    inner = parts[2]
    # strip redirects and take first token
    first = inner.split()[0] if inner.strip() else None
    if not first:
        return None
    # strip shell meta chars that might be left over
    first = first.lstrip("(").rstrip(")")
    return _resolve_binary(first)


def build_graph(events):
    pid_to_exe = {e["pid"]: e["exe"] for e in events if e["pid"] and e["exe"]}
    edge_data  = defaultdict(list)

    for e in events:
        if e.get("type") in PAM_TYPES:
            if e.get("src") and e.get("dest"):
                # dest is already unique per type: pam:root:USER_START etc.
                edge_data[(e["src"], e["dest"], e["type"].lower())].append(e["event_id"])
            continue

        sc = e["syscall"]
        if not sc or sc in ("setuid","setgid","setreuid","setregid",
                            "setresuid","setresgid"):
            continue
        src = e["exe"] or e["comm"] or f"pid:{e['pid']}"
        if src in SESSION_NOISE:
            continue

        if sc == "execve":
            if e["success"] == "yes":
                src = pid_to_exe.get(e["ppid"], f"pid:{e['ppid']}")
                dst = e["exe"] or e["comm"]
                key = "execve"
            else:
                src = e["exe"] or e["comm"] or f"pid:{e['pid']}"
                dst = (e["paths"][0].rsplit("/",1)[-1] + " [not found]") \
                      if e["paths"] else "unknown [not found]"
                key = "execve_fail"
            if src in SESSION_NOISE or dst in SESSION_NOISE:
                continue
            if src != dst:
                edge_data[(src, dst, key)].append(e["event_id"])

            # inferred edge: shell -c "binary ..." → add shell→binary edge
            if e["success"] == "yes" and e["exe"] in SHELLS and e.get("command"):
                child = _shell_c_child(e["command"])
                if child and child != e["exe"]:
                    edge_data[(e["exe"], child, "execve_inferred")].append(e["event_id"])

        elif sc == "connect" and e["dest"]:
            edge_data[(src, e["dest"], "connect")].append(e["event_id"])

        elif sc in FILE_SYSCALLS and e["paths"]:
            for fname in e["paths"]:
                if fname in ("/lib64/ld-linux-x86-64.so.2", ".") or \
                   not fname.startswith("/"):
                    continue
                edge_data[(src, fname, sc)].append(e["event_id"])

    G = nx.DiGraph()
    for (src, dst, sc), ids in edge_data.items():
        n       = len(ids)
        ids_str = ",".join(ids[:6]) + (f"+{n-6}" if n > 6 else "")

        if sc == "execve_fail":
            ec    = "#e64553"
            style = "dashed"
            lbl   = f"EXECVE FAIL ×{n} [{ids_str}]" if n > 1 \
                    else f"EXECVE FAIL [{ids[0]}]"
        elif sc == "execve_inferred":
            ec    = "#f38ba8"
            style = "dashed"
            lbl   = f"EXECVE [inferred] [{ids_str}]"
        elif sc in PAM_TYPES_LOWER:
            ec    = "#a6e3a1"
            style = "dashed"
            lbl   = f"{sc.upper()} [{ids[0]}]"
        else:
            ec    = "#f38ba8" if sc == "execve" else \
                    "#89dceb" if sc == "connect" else "#a6e3a1"
            style = "solid"
            lbl   = f"{sc.upper()} ×{n} [{ids_str}]" if n > 1 \
                    else f"{sc.upper()} [{ids[0]}]"

        G.add_edge(src, dst, label=lbl, color=ec, style=style,
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
    log = sys.argv[1] if len(sys.argv) > 1 else \
        "/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic/initial_access/T1190-000/1_autostart_localaccount-5/videoserver/logs/log/audit/audit.log"
    out = sys.argv[2] if len(sys.argv) > 2 else \
        "/csse/research/contructive-learning/CAM-LDS/provenance_graph.png"

    events = parse_audit(log)
    G      = build_graph(events)
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    title = f"Provenance Graph — nodes={G.number_of_nodes()}  edges={G.number_of_edges()}"
    draw_graph(G, title, out)
