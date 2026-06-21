import os, sys, subprocess, re, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABSTRACT_DIR  = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'abstract_sequnce')
RAW_DIR       = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'sequnces')
RESULTS_DIR   = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_output(text):
    """Extract metrics from one fold's full output."""
    result = {}

    # Loss at Ep=1
    m = re.search(r'Ep=1\s+Loss=([\d.]+)', text)
    result['loss_ep1'] = float(m.group(1)) if m else None

    # All epoch losses → best (minimum)
    losses = [float(x) for x in re.findall(r'Ep=\d+\s+Loss=([\d.]+)', text)]
    result['loss_best'] = min(losses) if losses else None

    # Parse from Stage 7 final evaluation (after "Stage 7 — Evaluating")
    # Line format:  <file>    <TrueTactic>    <TrueScore>    <PredTactic>    <PredScore>    ✓/✗
    stage7 = text.split('Stage 7')[1] if 'Stage 7' in text else text
    m = re.search(
        r'([\w_]+\.json)\s+([\w_]+)\s+([\+\-][\d.]+)\s+([\w_]+)\s+([\+\-][\d.]+)',
        stage7)
    if m:
        result['true_tactic'] = m.group(2)
        result['true_score']  = float(m.group(3))
        result['top1']        = m.group(4)
        result['top1_score']  = float(m.group(5))

    # Top2 and Top3: parse All scores line
    m2 = re.search(r'All scores: (.+)', stage7)
    if m2:
        score_pairs = re.findall(r'([\w_]+)=([\+\-][\d.]+)', m2.group(1))
        sorted_scores = sorted(score_pairs, key=lambda x: float(x[1]), reverse=True)
        top1_tactic = result.get('top1', '')
        others = [(t, s) for t, s in sorted_scores if t != top1_tactic]
        if len(others) >= 1:
            result['top2']       = others[0][0]
            result['top2_score'] = float(others[0][1])
        if len(others) >= 2:
            result['top3']       = others[1][0]
            result['top3_score'] = float(others[1][1])

    # Correct?
    result['correct'] = 'Accuracy: 1/1' in text

    return result


def make_pdf(rows, output_path, title):
    n = len(rows)
    fig_h = 1.2 + n * 0.55
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis('off')

    cols = ['#', 'Test File', 'True', 'True\nScore', 'Top1', 'Top1\nScore', 'Top2', 'Top2\nScore', 'Top3', 'Top3\nScore', 'Loss\nEp1', 'Loss\nBest', '✓/✗']
    col_w = [0.03, 0.15, 0.09, 0.055, 0.09, 0.055, 0.09, 0.055, 0.09, 0.055, 0.055, 0.055, 0.04]

    header_color = '#2c3e50'
    correct_color = '#d5f5e3'
    wrong_color   = '#fdfefe'
    alt_color     = '#eaf4fb'

    x0, y0 = 0.01, 0.97
    row_h  = 0.80 / (n + 1)

    # Header
    x = x0
    for ci, (col, cw) in enumerate(zip(cols, col_w)):
        ax.add_patch(FancyBboxPatch((x, y0 - row_h), cw - 0.002, row_h,
                                    boxstyle='round,pad=0.002', linewidth=0,
                                    facecolor=header_color, transform=ax.transAxes))
        ax.text(x + cw/2, y0 - row_h/2, col, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white', transform=ax.transAxes)
        x += cw

    # Rows
    for ri, row in enumerate(rows):
        y = y0 - row_h * (ri + 2)
        bg = correct_color if row['correct'] else (alt_color if ri % 2 == 0 else wrong_color)
        ax.add_patch(FancyBboxPatch((x0, y), sum(col_w) - 0.002, row_h,
                                    boxstyle='round,pad=0.002', linewidth=0.3,
                                    edgecolor='#bdc3c7', facecolor=bg, transform=ax.transAxes))
        vals = [
            str(ri + 1),
            row['file'].replace('abstract_','').replace('.json','').replace('_sequence',''),
            row.get('true_tactic', '?'),
            f"{row.get('true_score') or 0:+.4f}",
            row.get('top1', '?'),
            f"{row.get('top1_score') or 0:+.4f}",
            row.get('top2', '?'),
            f"{row.get('top2_score') or 0:+.4f}",
            row.get('top3', '?'),
            f"{row.get('top3_score') or 0:+.4f}",
            f"{row['loss_ep1']:.4f}" if row.get('loss_ep1') else '?',
            f"{row['loss_best']:.4f}" if row.get('loss_best') else '?',
            '✓' if row['correct'] else '✗',
        ]
        x = x0
        for ci, (val, cw) in enumerate(zip(vals, col_w)):
            color = '#27ae60' if val == '✓' else ('#e74c3c' if val == '✗' else '#2c3e50')
            fw = 'bold' if val in ('✓','✗') else 'normal'
            ax.text(x + cw/2, y + row_h/2, val, ha='center', va='center',
                    fontsize=7, color=color, fontweight=fw, transform=ax.transAxes)
            x += cw

    correct_n = sum(1 for r in rows if r['correct'])
    ax.text(0.5, 0.01, f'{title}    |    Accuracy: {correct_n}/{n}',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='#2c3e50', transform=ax.transAxes)

    plt.tight_layout(pad=0.2)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'\n  Table saved → {output_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abstract',   action='store_true')
    ap.add_argument('--linux-only', action='store_true')
    ap.add_argument('--aug',        action='store_true')
    args = ap.parse_args()

    seq_dir = ABSTRACT_DIR if args.abstract else RAW_DIR
    files   = sorted([f for f in os.listdir(seq_dir) if f.endswith('.json') and f != 'sequences_tactics_all.json'])

    seq_part  = 'abstract' if args.abstract else 'raw'
    tmpl_part = 'linux' if args.linux_only else 'full'
    aug_part  = '_aug' if args.aug else ''
    label     = f'{seq_part}_{tmpl_part}{aug_part}'

    cmd_flags = ['python', 'tactic_scripts/train_tactic_matcher.py']
    if args.abstract:   cmd_flags.append('--abstract')
    if args.linux_only: cmd_flags.append('--linux-only')
    if args.aug:        cmd_flags.append('--aug')

    rows = []
    correct = 0
    for i, fname in enumerate(files):
        print(f'\n[{i+1}/{len(files)}] {fname}')
        out = subprocess.run(cmd_flags + [fname], capture_output=True, text=True, cwd=PROJECT_ROOT)
        text = out.stdout + out.stderr
        metrics = parse_output(text)
        metrics['file'] = fname
        rows.append(metrics)
        if metrics['correct']:
            correct += 1
        print(f"  True={metrics.get('true_tactic','?')}({metrics.get('true_score') or 0:+.4f})  "
              f"Top1={metrics.get('top1','?')}({metrics.get('top1_score') or 0:+.4f})  "
              f"Top2={metrics.get('top2','?')}({metrics.get('top2_score') or 0:+.4f})  "
              f"Top3={metrics.get('top3','?')}({metrics.get('top3_score') or 0:+.4f})  "
              f"LossEp1={metrics.get('loss_ep1') or 0:.4f}  LossBest={metrics.get('loss_best') or 0:.4f}  "
              f"{'✓' if metrics['correct'] else '✗'}")

    print(f'\nFINAL: {correct}/{len(files)}')
    pdf_path = os.path.join(RESULTS_DIR, f'results_{label}.pdf')
    title = f"Loss: InfoNCE  |  Templates: {'linux-only' if args.linux_only else 'full'}  |  Aug: {args.aug}  |  Sequences: {'abstract' if args.abstract else 'raw'}"
    make_pdf(rows, pdf_path, title)


if __name__ == '__main__':
    main()
