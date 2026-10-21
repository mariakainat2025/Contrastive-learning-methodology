import html
import os
import re
import subprocess

from scoring_common import build_rows, build_zoomer_rows, accuracy_block, STAGE_ABBREV, TACTIC_IDS, ZOOMER_TECHNIQUES, RESULTS_DIR

OUT_BASENAME = 'scenario_detail_tables'

RUNS = [
    ('2_cron_only_basetech', 'Scenario 2, 2_cron'),
    ('ssh_apt_only_basetech', 'Scenario 3, ssh_apt'),
    ('scenario4_only_basetech', 'Scenario 4'),
    ('6_macro_binary_only_basetech', 'Scenario 6, macro_binary'),
]

SELECTED_TAGS = ['2_cron_only_basetech', 'ssh_apt_only_basetech', 'scenario4_only_basetech', '6_macro_binary_only_basetech']

COVERAGE = {
    'ssh_apt_only_basetech': """Techniques: total=36  in-training=36  in-test=22
Techniques in TEST but NOT in TRAINING (no prototype signal):
  T1039, T1040, T1087, T1133, T1213, T1485, T1486, T1489, T1490, T1565
Tactics:    total=12  in-training=12  in-test=11""",
    '2_cron_only_basetech': """Techniques: total=36  in-training=35  in-test=19
Techniques in TEST but NOT in TRAINING (no prototype signal):
  T1005, T1046, T1049, T1074, T1560, T1569, T1573
Tactics:    total=12  in-training=12  in-test=10""",
    'scenario4_only_basetech': """Techniques: total=36  in-training=35  in-test=10
Techniques in TEST but NOT in TRAINING (no prototype signal):
  T1205 -> steps: 4-16, 4-3, 4-9
  T1599 -> steps: 4-17, 4-18
Tactics:    total=12  in-training=12  in-test=7""",
    '6_macro_binary_only_basetech': """Techniques: total=36  in-training=36  in-test=12
Techniques in TEST but NOT in TRAINING (no prototype signal):
  T1010, T1204
Tactics:    total=12  in-training=12  in-test=8""",
}

ZOOMER_RESULTS = {
    '2_cron_only_basetech': '/csse/research/contructive-learning/CAM-LDS/zoomer/results/ttp_recognition_multilabel_scenario2_cron_camldsscope_min4_multi.json',
    'ssh_apt_only_basetech': '/csse/research/contructive-learning/CAM-LDS/zoomer/results/ttp_recognition_multilabel_scenario3_ssh_apt_camldsscope_min4_multi.json',
    'scenario4_only_basetech': '/csse/research/contructive-learning/CAM-LDS/zoomer/results/ttp_recognition_multilabel_scenario4_camldsscope_min4_multi.json',
    '6_macro_binary_only_basetech': '/csse/research/contructive-learning/CAM-LDS/zoomer/results/ttp_recognition_multilabel_scenario6_macro_binary_camldsscope_min4_multi.json',
}

ZOOMER_SINGLE_RESULTS = {
    '2_cron_only_basetech': '/csse/research/contructive-learning/CAM-LDS/zoomer/results/ttp_recognition_scenario2_cron_camldsscope_min4_single.json',
    'ssh_apt_only_basetech': '/csse/research/contructive-learning/CAM-LDS/zoomer/results/ttp_recognition_scenario3_ssh_apt_camldsscope_min4_single.json',
    'scenario4_only_basetech': '/csse/research/contructive-learning/CAM-LDS/zoomer/results/ttp_recognition_scenario4_camldsscope_min4_single.json',
    '6_macro_binary_only_basetech': '/csse/research/contructive-learning/CAM-LDS/zoomer/results/ttp_recognition_scenario6_macro_binary_camldsscope_min4_single.json',
}

_SCOPE_NOTE_TEXT = (
    'ZOOMER (both multi- and single-label) is scored here with a narrower 26-technique scope '
    '(>=4 dataset-wide samples) than CAM-LDS\'s 36-technique scope (>=3 samples) shown on this page. '
    'Techniques in CAM-LDS\'s scope but not ZOOMER\'s (T1016, T1041, T1046, T1071, T1098, T1201, T1205, '
    'T1218, T1547, T1556, T1566) were never trained on the ZOOMER side for this run -- any step naming '
    'one will show as wrong there regardless of prediction quality. ZOOMER also picks up T1204, which '
    'CAM-LDS\'s 36-list does not include.'
)
ZOOMER_SCOPE_NOTE = {
    '2_cron_only_basetech': _SCOPE_NOTE_TEXT,
    'ssh_apt_only_basetech': _SCOPE_NOTE_TEXT,
    'scenario4_only_basetech': _SCOPE_NOTE_TEXT,
    '6_macro_binary_only_basetech': _SCOPE_NOTE_TEXT,
}

ZOOMER_SCOPE_GAP = {'T1016', 'T1041', 'T1046', 'T1071', 'T1098', 'T1201', 'T1205', 'T1218', 'T1547', 'T1556', 'T1566'}

STAGE_LEGEND = ('IC=Initial Compromise&nbsp;&nbsp;&nbsp;EF=Establish Foothold&nbsp;&nbsp;&nbsp;'
                'EP=Escalate Privilege&nbsp;&nbsp;&nbsp;IR=Internal Reconnaissance&nbsp;&nbsp;&nbsp;'
                'ML=Move Laterally&nbsp;&nbsp;&nbsp;MP=Maintain Persistence&nbsp;&nbsp;&nbsp;CM=Complete Mission')
TACTIC_LEGEND = '&nbsp;&nbsp;&nbsp;'.join('{}={}'.format(v, k.replace('_', ' ').title()) for k, v in TACTIC_IDS.items())


def esc(s):
    return html.escape(str(s))


def score_grid(ranked, top_n, fmt):
    cells = []
    for label, score in ranked[:top_n]:
        cells.append('<div class="pred"><span class="lbl">{}</span> <span class="scr">({:.2f})</span></div>'.format(
            esc(fmt(label)), score))
    return '<div class="predgrid">{}</div>'.format(''.join(cells))


def pct(x):
    return '{:.1f}%'.format(x * 100)


LEVEL_ACC_KEYS = {
    'Technique': ('tech_acc_top1', 'tech_acc_top3', 'tech_acc_at_n'),
    'Tactic':    ('tac_acc_top1', 'tac_acc_top3', 'tac_acc_at_n'),
    'Stage':     ('stage_acc_top1', 'stage_acc_top3', 'stage_acc_at_n'),
}
CAM_LRAP_AUPR_KEYS = {'Technique': ('lrap', 'aupr'), 'Tactic': ('tac_lrap', 'tac_aupr'), 'Stage': ('stage_lrap', 'stage_aupr')}
ZOOMER_LRAP_AUPR_KEYS = {'Technique': ('lrap', 'aupr'), 'Tactic': ('tac_lrap', 'tac_aupr'), 'Stage': ('stage_lrap', 'stage_aupr')}


def lrap_aupr_value(o, key):
    return o[key] if key and key in o else None


def best_mark(text, value, max_value):
    if value is not None and max_value is not None and value > 0 and value == max_value:
        return '<strong class="best">{}</strong>'.format(text)
    return text


def build_scenario_summary_table(tag, label):
    """One table for ONE scenario, every model as a row, Technique/Tactic/Stage
    side by side as column-groups -- the full picture for that scenario alone."""
    out = ['<h3>{}</h3>'.format(esc(label))]
    out.append('<div class="tablewrap"><table class="summarytable">')
    out.append('<thead><tr><th rowspan="2" class="idcol">Model</th><th rowspan="2" class="idcol">N</th>'
                + ''.join('<th colspan="5" class="grouphead">{}</th>'.format(esc(lvl)) for lvl in LEVEL_ACC_KEYS)
                + '</tr><tr>'
                + '<th>Top-1 (s/l)</th><th>Top-3 (s/l)</th><th>At-n (s/l)</th><th>LRAP</th><th>AUPR</th>' * len(LEVEL_ACC_KEYS)
                + '</tr></thead><tbody>')

    group = []
    cam = build_rows(tag)
    group.append(('CAM-LDS', cam['out'], len(cam['technique'])))
    zoomer_path = ZOOMER_RESULTS.get(tag)
    if zoomer_path:
        zoomer = build_zoomer_rows(zoomer_path)
        group.append(('ZOOMER (multi)', zoomer['out'], len(zoomer['technique'])))
    single_path = ZOOMER_SINGLE_RESULTS.get(tag)
    if single_path:
        single = build_zoomer_rows(single_path)
        group.append(('ZOOMER (single)', single['out'], len(single['technique'])))

    level_acc_max, level_lrap_max, level_aupr_max = {}, {}, {}
    for level_title, acc_keys in LEVEL_ACC_KEYS.items():
        level_acc_max[level_title] = {k: max((o.get(k, 0) for _, o, _ in group), default=0) for k in acc_keys}
        lrap_vals, aupr_vals = [], []
        for model_name, o, _ in group:
            lk, ak = (CAM_LRAP_AUPR_KEYS if model_name == 'CAM-LDS' else ZOOMER_LRAP_AUPR_KEYS)[level_title]
            lrap_vals.append(lrap_aupr_value(o, lk))
            aupr_vals.append(lrap_aupr_value(o, ak))
        level_lrap_max[level_title] = max([v for v in lrap_vals if v is not None], default=None)
        level_aupr_max[level_title] = max([v for v in aupr_vals if v is not None], default=None)

    for model_name, o, n in group:
        row_cells = []
        for level_title, acc_keys in LEVEL_ACC_KEYS.items():
            for k in acc_keys:
                strict_v, loose_v = o.get(k, 0), o.get(k + '_loose', 0)
                row_cells.append('<td>{} / {}</td>'.format(
                    best_mark(pct(strict_v), strict_v, level_acc_max[level_title][k]), pct(loose_v)))
            lk, ak = (CAM_LRAP_AUPR_KEYS if model_name == 'CAM-LDS' else ZOOMER_LRAP_AUPR_KEYS)[level_title]
            lrap_v, aupr_v = lrap_aupr_value(o, lk), lrap_aupr_value(o, ak)
            lrap_txt = best_mark(pct(lrap_v), lrap_v, level_lrap_max[level_title]) if lrap_v is not None else '&mdash;'
            aupr_txt = best_mark(pct(aupr_v), aupr_v, level_aupr_max[level_title]) if aupr_v is not None else '&mdash;'
            row_cells.append('<td class="firstcol">{}</td><td>{}</td>'.format(lrap_txt, aupr_txt))
        out.append('<tr><td class="idcol">{}</td><td class="idcol">{}</td>{}</tr>'.format(
            esc(model_name), n, ''.join(row_cells)))
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def build_summary_page(runs):
    out = ['<section class="summarypage">', '<h1 class="pagehead">Scenario summaries</h1>']
    out.append('<p class="abbrev">Each accuracy cell shows <strong>strict / loose</strong>: strict = every true '
                'label must be in the top-k (subset match); loose = at least one true label in the top-k counts '
                'as correct (any-overlap). Both computed from the same saved predictions, no re-run needed. '
                '<strong class="best">Red bold</strong> = best model for that metric within each scenario.</p>')
    for tag, label in runs:
        out.append(build_scenario_summary_table(tag, label))
    out.append('</section>')
    return '\n'.join(out)


def zero_training_techniques(tag):
    """Pulls the technique IDs out of COVERAGE's 'Techniques in TEST but NOT in
    TRAINING' block for this tag -- these have no prototype at all, so any step
    naming one is structurally unwinnable, not a genuine model miss."""
    text = COVERAGE.get(tag, '')
    section = text.split('Techniques in TEST but NOT in TRAINING', 1)
    if len(section) < 2 or section[1].lstrip().startswith(': none'):
        return set()
    body = section[1].split('Tactics:', 1)[0]
    return set(re.findall(r'\b(T\d{4})\b', body))


def cpw_counts(rows):
    return (sum(1 for r in rows if r['verdict'] == 'correct'),
            sum(1 for r in rows if r['verdict'] == 'partial'),
            sum(1 for r in rows if r['verdict'] == 'wrong'))


def file_sort_key(f):
    m = re.search(r'(\d+)$', f)
    return (int(m.group(1)) if m else 0, f)


def by_file(rows):
    return {r['file']: r for r in rows}


def summary_line(model_name, tech_rows, tac_rows, stg_rows):
    tc, tp, tw = cpw_counts(tech_rows)
    ac, ap, aw = cpw_counts(tac_rows)
    sc, sp, sw = cpw_counts(stg_rows)
    n = len(tech_rows)
    return ('<strong>{m}</strong> (N={n}) &mdash; '
            'Technique {tc}/{n} correct&middot;{tp} partial&middot;{tw} wrong &nbsp; '
            'Tactic {ac}/{n} correct&middot;{ap} partial&middot;{aw} wrong &nbsp; '
            'Stage {sc}/{n} correct&middot;{sp} partial&middot;{sw} wrong').format(
        m=esc(model_name), n=n, tc=tc, tp=tp, tw=tw, ac=ac, ap=ap, aw=aw, sc=sc, sp=sp, sw=sw)


def build_scenario_table(tag, label):
    cam = build_rows(tag)
    cam_tech, cam_tac, cam_stg = cam['technique'], cam['tactic'], cam['stage']
    no_proto = zero_training_techniques(tag)

    zoomer_path = ZOOMER_RESULTS.get(tag)
    zoomer = build_zoomer_rows(zoomer_path) if zoomer_path else None
    single_path = ZOOMER_SINGLE_RESULTS.get(tag)
    single = build_zoomer_rows(single_path) if single_path else None

    extra_models = [m for m in (('ZOOMER (multi)', zoomer), ('ZOOMER (single)', single)) if m[1]]

    out_of_scope = set()
    for r in cam_tech:
        out_of_scope.update(t for t in r['true'] if t not in ZOOMER_TECHNIQUES)
    for (_, model) in extra_models:
        for r in model['technique']:
            out_of_scope.update(t for t in r['true'] if t not in ZOOMER_TECHNIQUES)

    out = []
    out.append('<section class="scenario">')
    out.append('<h2>{}</h2>'.format(esc(label)))
    if no_proto:
        out.append('<p class="abbrev">&dagger; = in 36-technique scope but zero training examples under this '
                    'holdout (no prototype -- structurally unwinnable, not a model miss): {}</p>'.format(
                        ', '.join(sorted(no_proto))))
    if out_of_scope:
        out.append('<p class="abbrev">&Dagger; = entirely outside the 36-technique scope (no prototype on either '
                    'side, ever): {}</p>'.format(', '.join(sorted(out_of_scope))))
    if extra_models:
        out.append('<p class="abbrev">CAM-LDS tests only in-scope-technique steps (N={cn}); ZOOMER tests every step '
                    'of the held-out run-group (N={zn}) -- rows CAM-LDS never saw show "(not tested)".</p>'.format(
                        cn=len(cam_tech), zn=len(extra_models[0][1]['technique'])))
    cam_by_file = {lvl: by_file(cam[lvl]) for lvl in ('technique', 'tactic', 'stage')}
    extra_by_file = [(name, {lvl: by_file(model[lvl]) for lvl in ('technique', 'tactic', 'stage')})
                      for (name, model) in extra_models]
    all_files = sorted(
        {r['file'] for r in cam_tech}.union(*[{r['file'] for r in model['technique']} for (_, model) in extra_models]),
        key=file_sort_key)

    out.append('<div class="tablewrap"><table>')
    if extra_models:
        out.append('<thead><tr><th rowspan="2" class="idcol">#</th><th rowspan="2" class="idcol">File</th>'
                    '<th rowspan="2" class="idcol">True technique</th><th rowspan="2" class="idcol">True tactic</th>'
                    '<th rowspan="2" class="idcol">True stage</th>'
                    '<th colspan="3" class="grouphead">CAM-LDS</th>'
                    + ''.join('<th colspan="3" class="grouphead">{}</th>'.format(esc(name)) for (name, _) in extra_models)
                    + '</tr><tr>'
                    + '<th>Technique (top-3)</th><th>Tactic (top-3)</th><th>Stage (top-3)</th>' * (1 + len(extra_models))
                    + '</tr></thead><tbody>')
    else:
        out.append('<thead><tr>'
                    '<th>#</th><th>File</th>'
                    '<th>True technique</th><th>True tactic</th><th>True stage</th>'
                    '<th>Technique (top-3, score)</th><th>Tactic (top-3, score)</th><th>Stage (top-3, score)</th>'
                    '</tr></thead><tbody>')

    unavailable = no_proto | out_of_scope

    def cell(verdict, grid, firstcol=False):
        cls = verdict + (' firstcol' if firstcol else '')
        return '<td class="{v}">{g}</td>'.format(v=cls, g=grid)

    for i, f in enumerate(all_files, 1):
        tr, ar, sr = cam_by_file['technique'].get(f), cam_by_file['tactic'].get(f), cam_by_file['stage'].get(f)
        true_src_t, true_src_a, true_src_s = tr, ar, sr
        if true_src_t is None:
            for (_, by_lvl) in extra_by_file:
                if by_lvl['technique'].get(f) is not None:
                    true_src_t = by_lvl['technique'].get(f)
                    true_src_a = by_lvl['tactic'].get(f)
                    true_src_s = by_lvl['stage'].get(f)
                    break
        def tech_html(t):
            if t in out_of_scope:
                return '<span class="notrain">{}<sup>&Dagger;</sup></span>'.format(esc(t))
            if t in no_proto:
                return '<span class="notrain">{}<sup>&dagger;</sup></span>'.format(esc(t))
            return esc(t)
        true_tech = ', '.join(tech_html(t) for t in sorted(true_src_t['true']))
        true_tac = ', '.join(esc(TACTIC_IDS.get(t, t)) for t in sorted(true_src_a['true']))
        true_stg = ', '.join(esc(STAGE_ABBREV.get(t, t)) for t in sorted(true_src_s['true']))

        if tr is not None:
            cam_cells = (
                cell(tr['verdict'], score_grid(tr['ranked'], tr['top_n'], lambda t: t))
                + cell(ar['verdict'], score_grid(ar['ranked'], ar['top_n'], lambda t: TACTIC_IDS.get(t, t)))
                + cell(sr['verdict'], score_grid(sr['ranked'], sr['top_n'], lambda t: STAGE_ABBREV.get(t, t)))
            )
        else:
            cam_cells = '<td colspan="3" class="notested">(not tested)</td>'

        model_cells = ''
        for (_, by_lvl) in extra_by_file:
            mtr, mar, msr = by_lvl['technique'].get(f), by_lvl['tactic'].get(f), by_lvl['stage'].get(f)
            if mtr is not None:
                model_cells += (
                    cell(mtr['verdict'], score_grid(mtr['ranked'], mtr['top_n'], lambda t: t), firstcol=True)
                    + cell(mar['verdict'], score_grid(mar['ranked'], mar['top_n'], lambda t: TACTIC_IDS.get(t, t)))
                    + cell(msr['verdict'], score_grid(msr['ranked'], msr['top_n'], lambda t: STAGE_ABBREV.get(t, t)))
                )
            else:
                model_cells += '<td colspan="3" class="notested firstcol">(not tested)</td>'

        out.append(
            '<tr><td class="idcol">{i}</td><td class="idcol">{f}</td>'
            '<td>{tt}</td><td>{ta}</td><td>{ts}</td>{cc}{mc}</tr>'.format(
                i=i, f=esc(f), tt=true_tech, ta=true_tac, ts=true_stg, cc=cam_cells, mc=model_cells))
    out.append('</tbody></table></div>')

    out.append('<p class="summary">{}</p>'.format(summary_line('CAM-LDS', cam_tech, cam_tac, cam_stg)))
    for (name, model) in extra_models:
        out.append('<p class="summary">{}</p>'.format(summary_line(name, model['technique'], model['tactic'], model['stage'])))
    out.append('</section>')
    return '\n'.join(out)


PAGE_CSS = """
:root {
  --bg: #fbfaf7; --surface: #ffffff; --ink: #211e1a; --ink-soft: #6b6358; --rule: #c9c2b2;
  --correct-bg: #e4f3e3; --correct-ink: #2f6b2a;
  --partial-bg: #fbf1cf; --partial-ink: #8a6a12;
  --wrong-bg: #fbe3e1; --wrong-ink: #a1352b;
  --accent: #6b4a2b;
  --best: #c0152f;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1b1812; --surface: #242019; --ink: #ece6da; --ink-soft: #b3a995; --rule: #41392c;
    --correct-bg: #23381f; --correct-ink: #9fe09a;
    --partial-bg: #3a3014; --partial-ink: #e8c766;
    --wrong-bg: #3c231f; --wrong-ink: #ef9a90;
    --accent: #d2a876;
    --best: #ff6b5e;
  }
}
:root[data-theme="dark"] {
  --bg: #1b1812; --surface: #242019; --ink: #ece6da; --ink-soft: #b3a995; --rule: #41392c;
  --correct-bg: #23381f; --correct-ink: #9fe09a;
  --partial-bg: #3a3014; --partial-ink: #e8c766;
  --wrong-bg: #3c231f; --wrong-ink: #ef9a90;
  --accent: #d2a876;
  --best: #ff6b5e;
}
@page { size: A3 landscape; margin: 4mm 5mm; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: "Nimbus Roman", "Times New Roman", Georgia, serif;
  font-size: 9px; line-height: 1.15; margin: 0; padding: 2px 2px; }
h1 { font-size: 14px; margin: 0 0 2px; }
.subtitle { color: var(--ink-soft); font-size: 9px; margin: 0 0 4px; max-width: 1200px; }
section.summarypage { page-break-after: always; }
h1.pagehead { font-size: 16px; margin: 0 0 8px; }
section.summarypage h3 { font-size: 12px; margin: 12px 0 3px; color: var(--accent); }
section.summarypage h3:first-of-type { margin-top: 2px; }
table.summarytable tbody td { font-size: 9.5px; }
section.scenario { padding-top: 1px; page-break-before: always; }
h2 { font-size: 11.5px; border-bottom: 1.2px solid var(--accent); padding-bottom: 1px; margin: 3px 0 2px; color: var(--accent); }
p.abbrev { font-size: 8px; color: var(--ink-soft); margin: 0.5px 0; }
.best { color: var(--best); font-weight: 800; }
table.summarytable tbody td { font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 14px; align-items: center; margin: 2px 0 3px; font-size: 8.5px; color: var(--ink-soft); }
.swatch { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.sw-correct { background: var(--correct-bg); border: 1px solid var(--correct-ink); }
.sw-partial { background: var(--partial-bg); border: 1px solid var(--partial-ink); }
.sw-wrong   { background: var(--wrong-bg); border: 1px solid var(--wrong-ink); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: auto; font-size: 8.5px; font-variant-numeric: tabular-nums; }
thead th { text-align: left; font-weight: 600; font-size: 8px; letter-spacing: 0.01em;
  color: var(--ink-soft); border-bottom: 1.2px solid var(--rule); padding: 1px 8px 1px 5px; white-space: nowrap; }
thead th.grouphead { text-align: center; font-size: 9.5px; color: var(--accent);
  border-bottom: 1.5px solid var(--accent); border-left: 1.5px solid var(--rule); }
thead tr:first-child th.idcol { border-bottom: none; }
tbody td { padding: 0.3px 8px 0.3px 5px; border-bottom: 1px solid var(--rule); vertical-align: top; white-space: nowrap; }
td.idcol { font-weight: 600; }
td.correct { background: var(--correct-bg); }
td.partial { background: var(--partial-bg); }
td.wrong   { background: var(--wrong-bg); }
td.notested { color: var(--ink-soft); font-style: italic; }
.notrain { color: var(--wrong-ink); font-weight: 700; }
td.firstcol { border-left: 1.5px solid var(--rule); }
.predgrid { display: flex; flex-direction: column; line-height: 1.02; }
.pred { white-space: nowrap; }
.pred .lbl { font-weight: 600; }
.pred .scr { color: var(--ink-soft); }
p.summary { margin: 3px 0 0; font-size: 9px; }
"""


def build_html(body_html):
    return """<!doctype html>
<title>CAM-LDS Per-Step Detail Tables</title>
<style>{css}</style>
<h1>Per-step detail: technique, tactic and stage together</h1>
<p class="subtitle">Base-technique level (36-class fixed scope), scenario/run-group holdout. One row per
test step; each level's prediction cell is colored by that level's own verdict (all true labels recovered
in its top-N predictions, some, or none).</p>
{body}
""".format(css=PAGE_CSS, body=body_html)


def main():
    runs = [(tag, label) for tag, label in RUNS if tag in SELECTED_TAGS]
    available = []
    for tag, label in runs:
        json_path = os.path.join(RESULTS_DIR, 'camlds_technique_wide_test_results_{}.json'.format(tag))
        if os.path.exists(json_path):
            available.append((tag, label))
        else:
            print('SKIPPING {} ({}): no result JSON at {} -- rerun its test step to restore it.'.format(
                tag, label, json_path))
    runs = available
    body = build_summary_page(runs) + '\n' + '\n'.join(build_scenario_table(tag, label) for tag, label in runs)
    html_doc = build_html(body)
    html_path = os.path.join(RESULTS_DIR, OUT_BASENAME + '.html')
    with open(html_path, 'w') as f:
        f.write(html_doc)
    print('Wrote {}'.format(html_path))

    pdf_path = os.path.join(RESULTS_DIR, OUT_BASENAME + '.pdf')
    subprocess.run([
        'chromium', '--headless', '--disable-gpu', '--no-sandbox',
        '--print-to-pdf=' + pdf_path, '--no-pdf-header-footer',
        '--print-to-pdf-no-header',
        'file://' + html_path,
    ], check=True, capture_output=True)
    print('Rendered {}'.format(pdf_path))


if __name__ == '__main__':
    main()
