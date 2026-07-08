import sys
sys.path.insert(0, "/csse/research/contructive-learning/CAM-LDS")

from pathlib import Path
from parse_audit import parse_audit
from graph_audit import build_graph, PROC_DIRS, draw_graph as _draw_graph_base

# Core attack chain nodes — anything outside this set is noise (yellow border)
BASELINE = {
    "/usr/sbin/apache2", "/usr/bin/dash", "/usr/bin/perl",
    "/usr/bin/zmu", "/usr/bin/sleep", "/usr/bin/python3.9",
    "/usr/sbin/ldconfig", "/usr/bin/uname",
    "192.42.1.174:4444", "/var/run/mysqld/mysqld.sock",
    # graph_audit uses basename for failed-execve nodes
    "uname [not found]",
}

def is_noise(name):
    if name.startswith("pid:"):
        return False   # meterpreter/unknown attacker pids are attack nodes
    return name not in BASELINE

def border_color(name):
    if is_noise(name):
        return "#f9e2af"   # yellow — noise / extra event
    if name.startswith("pid:"):
        return "#3949ab"
    if ":" in name and name.split(":")[-1].isdigit():
        return "#e74c3c"
    if any(name.startswith(d) for d in PROC_DIRS):
        return "#3949ab"
    return "#8e24aa"

def border_width(name):
    return 4.5 if is_noise(name) else 3.0

def draw_graph(G, title, out_path):
    """Wrapper over graph_audit.draw_graph that adds yellow-border noise marking."""
    _draw_graph_base(G, title, out_path,
                     border_color_fn=border_color,
                     border_width_fn=border_width)


if __name__ == "__main__":
    base    = Path("/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic/initial_access/T1190-000")
    out_dir = base / "graphs"
    out_dir.mkdir(exist_ok=True)

    runs = sorted(d for d in base.iterdir()
                  if d.is_dir() and d.name != "graphs")

    for run in runs:
        log = run / "videoserver/logs/log/audit/audit.log"
        if not log.exists():
            print(f"  skip {run.name} — no audit.log")
            continue

        print(f"processing {run.name} ...")
        events = parse_audit(str(log))
        G      = build_graph(events)

        noise_count = sum(1 for n in G.nodes() if is_noise(n))
        title = (
            f"T1190-000 — {run.name}\n"
            f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}"
            + (f"  |  noise nodes={noise_count} (yellow border)" if noise_count else "")
        )

        out_path = out_dir / f"{run.name}.png"
        draw_graph(G, title, out_path)

    print(f"\nDone — {len(runs)} graphs saved to {out_dir}")
