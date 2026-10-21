import html
import json
import os
import subprocess
import sys

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
TECHNIQUE_DIR = os.path.dirname(RESULTS_DIR)
STEP_TACTICS_PATH = os.path.join(TECHNIQUE_DIR, 'step_tactics.json')
SCRIPTS_DIR = os.path.join(TECHNIQUE_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from tactic_to_stage import TACTIC_TO_STAGE

STAGE_ABBREV = {
    'Initial Compromise':      'IC',
    'Establish Foothold':      'EF',
    'Escalate Privilege':      'EP',
    'Internal Reconnaissance': 'IR',
    'Move Laterally':          'ML',
    'Maintain Persistence':    'MP',
    'Complete Mission':        'CM',
}

TACTIC_IDS = {
    'collection': 'TA0009', 'command_and_control': 'TA0011', 'credential_access': 'TA0006',
    'defense_impairment': 'TA0112', 'discovery': 'TA0007', 'execution': 'TA0002',
    'exfiltration': 'TA0010', 'impact': 'TA0040', 'initial_access': 'TA0001',
    'lateral_movement': 'TA0008', 'persistence': 'TA0003', 'privilege_escalation': 'TA0004',
    'reconnaissance': 'TA0043', 'stealth': 'TA0005',
}

ZOOMER_TECHNIQUES = [
    'T1003', 'T1016', 'T1018', 'T1021', 'T1033', 'T1036',
    'T1041', 'T1046', 'T1053', 'T1056', 'T1057',
    'T1059', 'T1070', 'T1071', 'T1072', 'T1078',
    'T1082', 'T1083', 'T1098', 'T1105', 'T1110',
    'T1115', 'T1176', 'T1201', 'T1205', 'T1218', 'T1219',
    'T1222', 'T1543', 'T1546', 'T1547', 'T1548', 'T1556',
    'T1564', 'T1566', 'T1574',
    'T1039', 'T1040', 'T1120', 'T1133', 'T1136', 'T1190', 'T1213', 'T1486', 'T1497',
    'T1204', 'T1595',
    'T1590', 'T1591',
]


def base_technique(t):
    return t.split('-')[0]


def load_technique_to_tactics():
    with open(STEP_TACTICS_PATH) as f:
        report = json.load(f)
    technique_train_count = {}
    technique_to_tactics = {}
    for row in report:
        for occ in row['occurrences']:
            t = base_technique(occ['technique'])
            technique_to_tactics.setdefault(t, set()).update(occ['tactics'])
    return {t: sorted(tacs) for t, tacs in technique_to_tactics.items()}


def expand_ranked(ranked, key_to_labels):
    seen, out = set(), []
    for label, score in ranked:
        for target in key_to_labels.get(label, []):
            if target not in seen:
                seen.add(target)
                out.append((target, score))
    return out


def classify(true_set, top_set, top_n):
    total = len(true_set)
    if total == 0:
        return None
    found = len(true_set & top_set)
    if total <= top_n:
        if found == total:
            return 'correct'
        if found == 0:
            return 'wrong'
        return 'partial'
    extra_wrong = len(top_set - true_set)
    if found == 0:
        return 'wrong'
    if extra_wrong == 0:
        return 'correct'
    return 'partial'


def esc(s):
    return html.escape(str(s))


KNOWN_HOSTS = ('attacker', 'inetfw', 'videoserver', 'wazuh', 'reposerver', 'client',
               'corpdns', 'linuxshare', 'docker-log', 'docker', 'adminpc')
STAGE_NAME_TO_ABBREV = {full: short for full, short in STAGE_ABBREV.items()}
TACTIC_NAME_TO_ID = TACTIC_IDS


def zoomer_instance_from_filename(filename):
    stem = filename[len('graph_'):-len('.json')] if filename.startswith('graph_') else filename
    for host in KNOWN_HOSTS:
        suffix = '_' + host
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def load_zoomer_rows(zoomer_path):
    with open(zoomer_path) as f:
        d = json.load(f)
    out = {}
    tech_by_inst = {zoomer_instance_from_filename(r['file']): r for r in d['technique_results']}
    tac_by_inst = {r['file']: r for r in d['tactic_results']}
    stage_by_inst = {r['file']: r for r in d['stage_results']}
    for inst in set(tech_by_inst) | set(tac_by_inst) | set(stage_by_inst):
        tech_top3 = [(e['technique'], e['score']) for e in tech_by_inst[inst]['ranked'][:3]] if inst in tech_by_inst else []
        tac_top3 = [(TACTIC_NAME_TO_ID.get(e['tactic'], e['tactic']), e['score']) for e in tac_by_inst[inst]['ranked'][:3]] if inst in tac_by_inst else []
        stage_top3 = [(STAGE_NAME_TO_ABBREV.get(e['stage'], e['stage']), e['score']) for e in stage_by_inst[inst]['ranked'][:3]] if inst in stage_by_inst else []
        out[inst] = {'tech_top3': tech_top3, 'tac_top3': tac_top3, 'stage_top3': stage_top3}
    return out


def _zoomer_fields(prefix, zoomer_by_inst, file_key, true_techs, true_tacs, true_stage_abbrevs):
    z = zoomer_by_inst.get(file_key)
    tech_top3 = z['tech_top3'] if z else []
    tac_top3 = z['tac_top3'] if z else []
    stage_top3 = z['stage_top3'] if z else []
    return {
        '{}has'.format(prefix): z is not None,
        '{}tech_top3'.format(prefix): tech_top3,
        '{}tac_top3'.format(prefix): tac_top3,
        '{}stage_top3'.format(prefix): stage_top3,
        '{}tech_verdict'.format(prefix): classify(true_techs, {t for t, _ in tech_top3}, 3) if z else None,
        '{}tac_verdict'.format(prefix): classify({TACTIC_IDS.get(t, t) for t in true_tacs}, {t for t, _ in tac_top3}, 3) if z else None,
        '{}stage_verdict'.format(prefix): classify(true_stage_abbrevs, {t for t, _ in stage_top3}, 3) if z else None,
    }


def build_rows(tag, train_counts, zoomer_path=None, zoomer_single_path=None):
    with open(os.path.join(RESULTS_DIR, 'camlds_technique_wide_test_results_{}.json'.format(tag))) as f:
        d = json.load(f)
    technique_to_tactics = load_technique_to_tactics()
    tactic_to_stage_list = {t: [s] for t, s in TACTIC_TO_STAGE.items()}
    zoomer_by_inst = load_zoomer_rows(zoomer_path) if zoomer_path else {}
    zoomer_single_by_inst = load_zoomer_rows(zoomer_single_path) if zoomer_single_path else {}

    rows = []
    for r in d['results']:
        true_techs = set(r['true_techniques'])
        tech_ranked = sorted(r['scores'].items(), key=lambda x: -x[1])
        tac_ranked = expand_ranked(tech_ranked, technique_to_tactics)
        stage_ranked = expand_ranked(tac_ranked, tactic_to_stage_list)

        true_tacs = set()
        for t in true_techs:
            true_tacs.update(technique_to_tactics.get(t, []))
        true_stages = {TACTIC_TO_STAGE[tac] for tac in true_tacs if tac in TACTIC_TO_STAGE}

        top3_tech = {t for t, _ in tech_ranked[:3]}
        top3_tac = {t for t, _ in tac_ranked[:3]}
        top3_stage = {t for t, _ in stage_ranked[:3]}

        true_stage_abbrevs = {STAGE_ABBREV.get(s, s) for s in true_stages}

        row = {
            'file': r['file'],
            'true_techs': sorted(true_techs),
            'true_tacs': sorted(true_tacs),
            'true_stages': sorted(true_stages, key=lambda s: list(STAGE_ABBREV).index(s) if s in STAGE_ABBREV else 99),
            'train': [train_counts.get(t, 0) for t in sorted(true_techs)],
            'n_events': r.get('n_events', r.get('events')),
            'tech_top3': tech_ranked[:3],
            'tac_top3': tac_ranked[:3],
            'stage_top3': stage_ranked[:3],
            'tech_verdict': classify(true_techs, top3_tech, 3),
            'tac_verdict': classify(true_tacs, top3_tac, 3),
            'stage_verdict': classify(set(true_stages), top3_stage, 3),
        }
        row.update(_zoomer_fields('z_', zoomer_by_inst, r['file'], true_techs, true_tacs, true_stage_abbrevs))
        row.update(_zoomer_fields('zs_', zoomer_single_by_inst, r['file'], true_techs, true_tacs, true_stage_abbrevs))
        rows.append(row)
    return rows, d


def compute_train_counts(tag_test_file=None, tag_scenario=None):
    with open(STEP_TACTICS_PATH) as f:
        report = json.load(f)
    counts = {}
    for row in report:
        step = row['step']
        is_test = False
        if tag_test_file:
            is_test = tag_test_file in step
        elif tag_scenario:
            s = str(tag_scenario)
            is_test = step.startswith(s + '_') or step.startswith(s + '-')
        if is_test:
            continue
        for occ in row['occurrences']:
            t = base_technique(occ['technique'])
            counts[t] = counts.get(t, 0) + 1
    return counts


def score_cell(verdict, ranked3, fmt_label):
    lines = ''.join(
        '<div class="pred"><span class="lbl">{}</span> <span class="scr">({:.3f})</span></div>'.format(
            esc(fmt_label(lbl)), score)
        for lbl, score in ranked3)
    return '<td class="{v}"><div class="predgrid">{lines}</div></td>'.format(v=verdict or 'correct', lines=lines)


STAGE_LEGEND = '&nbsp;&nbsp;&nbsp;'.join('{}={}'.format(v, k) for k, v in STAGE_ABBREV.items())
TACTIC_LEGEND = '&nbsp;&nbsp;&nbsp;'.join(
    '{}={}'.format(v, k.replace('_', ' ').title()) for k, v in TACTIC_IDS.items())


def build_html(tag, label, rows, zoomer_models=None):
    zoomer_models = zoomer_models or []
    out = []
    out.append('<h2>{}</h2>'.format(esc(label)))
    out.append('<p class="legend"><strong>Stage:</strong> {}</p>'.format(STAGE_LEGEND))
    out.append('<p class="legend"><strong>Tactic:</strong> {}</p>'.format(TACTIC_LEGEND))
    out.append('<table><thead>')
    if zoomer_models:
        out.append('<tr><th rowspan="2" class="idcol">#</th><th rowspan="2" class="idcol">Step</th>'
                    '<th rowspan="2">True technique</th><th rowspan="2">True tactic</th><th rowspan="2">True stage</th>'
                    '<th rowspan="2">Train</th><th rowspan="2">Events</th>'
                    '<th colspan="3" class="grouphead">CAM-LDS</th>'
                    + ''.join('<th colspan="3" class="grouphead">{}</th>'.format(esc(name)) for name, _ in zoomer_models)
                    + '</tr><tr><th>Technique</th><th>Tactic</th><th>Stage</th>'
                    + '<th class="firstcol">Technique</th><th>Tactic</th><th>Stage</th>' * len(zoomer_models)
                    + '</tr>')
    else:
        out.append('<tr>' + ''.join('<th>{}</th>'.format(esc(h)) for h in
                    ['#', 'Step', 'True technique', 'True tactic', 'True stage', 'Train', 'Events',
                     'Technique score', 'Tactic score', 'Stage score']) + '</tr>')
    out.append('</thead><tbody>')
    for i, r in enumerate(rows, 1):
        tech_cell = score_cell(r['tech_verdict'], r['tech_top3'], lambda t: t)
        tac_cell = score_cell(r['tac_verdict'], r['tac_top3'], lambda t: TACTIC_IDS.get(t, t))
        stage_cell = score_cell(r['stage_verdict'], r['stage_top3'], lambda t: STAGE_ABBREV.get(t, t))
        zoomer_cells = ''
        for _name, prefix in zoomer_models:
            if r['{}has'.format(prefix)]:
                zt = score_cell(r['{}tech_verdict'.format(prefix)], r['{}tech_top3'.format(prefix)], lambda t: t)
                za = score_cell(r['{}tac_verdict'.format(prefix)], r['{}tac_top3'.format(prefix)], lambda t: t)
                zs = score_cell(r['{}stage_verdict'.format(prefix)], r['{}stage_top3'.format(prefix)], lambda t: t)
                zoomer_cells += zt.replace('<td class="', '<td class="firstcol ', 1) + za + zs
            else:
                zoomer_cells += '<td colspan="3" class="notested firstcol">(not tested)</td>'
        out.append(
            '<tr><td class="idcol">{i}</td><td class="idcol">{f}</td>'
            '<td>{tt}</td><td>{ta}</td><td>{ts}</td><td>{tr}</td><td>{ev}</td>{tc}{ac}{sc}{zc}</tr>'.format(
                i=i, f=esc(r['file']),
                tt=esc(', '.join(r['true_techs'])),
                ta=esc(', '.join(TACTIC_IDS.get(t, t) for t in r['true_tacs'])),
                ts=esc(', '.join(STAGE_ABBREV.get(s, s) for s in r['true_stages'])),
                tr=esc(', '.join(str(c) for c in r['train'])),
                ev=esc(r['n_events']),
                tc=tech_cell, ac=tac_cell, sc=stage_cell, zc=zoomer_cells))
    out.append('</tbody></table>')
    return '\n'.join(out)


PAGE_CSS = """
:root {
  --bg: #fbfaf7; --ink: #211e1a; --ink-soft: #6b6358; --rule: #c9c2b2;
  --correct-bg: #e4f3e3; --correct-ink: #2f6b2a;
  --partial-bg: #fbf1cf; --partial-ink: #8a6a12;
  --wrong-bg: #fbe3e1; --wrong-ink: #a1352b;
  --accent: #6b4a2b;
}
@page { size: A3 landscape; margin: 6mm 7mm; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: "Nimbus Roman", "Times New Roman", Georgia, serif;
  font-size: 10px; line-height: 1.2; margin: 0; padding: 4px; }
h2 { font-size: 13px; border-bottom: 1.2px solid var(--accent); padding-bottom: 2px; margin: 2px 0 6px; color: var(--accent); }
p.legend { font-size: 8.5px; color: var(--ink-soft); margin: 1px 0; }
p.legend strong { color: var(--accent); }
table { border-collapse: collapse; width: 100%; font-size: 9.5px; font-variant-numeric: tabular-nums; }
thead th { text-align: left; font-weight: 600; font-size: 8.5px; color: var(--ink-soft);
  border-bottom: 1.2px solid var(--rule); padding: 2px 8px 2px 4px; white-space: nowrap; }
tbody td { padding: 2px 8px 2px 4px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
td.idcol { font-weight: 600; }
thead th.grouphead { text-align: center; font-size: 9.5px; color: var(--accent);
  border-bottom: 1.5px solid var(--accent); border-left: 1.5px solid var(--rule); }
thead tr:first-child th.idcol { border-bottom: none; }
td.firstcol, th.firstcol { border-left: 1.5px solid var(--rule); }
td.notested { color: var(--ink-soft); font-style: italic; }
td.correct { background: var(--correct-bg); }
td.partial { background: var(--partial-bg); }
td.wrong   { background: var(--wrong-bg); }
.predgrid { display: flex; flex-direction: column; line-height: 1.1; }
.pred { white-space: nowrap; }
.pred .lbl { font-weight: 600; }
.pred .scr { color: var(--ink-soft); }
section.summarypage { page-break-after: always; }
h1.pagehead { font-size: 16px; margin: 0 0 8px; }
p.scopeline { font-size: 9px; color: var(--ink-soft); margin: 2px 0; }
p.scopeline strong { color: var(--accent); }
p.scopeline sup { color: var(--wrong-ink); }
p.scopelist { font-size: 8.5px; color: var(--ink); margin: 2px 0 8px; line-height: 1.5; }
p.scopelist sup { color: var(--wrong-ink); font-weight: 700; }
section.summarypage h3 { font-size: 12px; margin: 10px 0 3px; color: var(--accent); }
section.summarypage h3:first-of-type { margin-top: 2px; }
table.summarytable { width: auto; min-width: 420px; }
table.summarytable tbody td { font-variant-numeric: tabular-nums; }
td.rowlabel { white-space: nowrap; padding-left: 12px; }
td.modelcol { white-space: nowrap; font-weight: 600; color: var(--accent); font-size: 8.5px; }
strong.best { color: var(--wrong-ink); font-weight: 800; }
td.levelhead { font-weight: 700; color: var(--accent); border-bottom: 1px solid var(--rule); padding-top: 6px; }
section.scenario { page-break-before: always; }
"""


def pct(x):
    return '{:.1f}%'.format(x * 100)


LEVELS = [('Technique', 'technique'), ('Tactic', 'tactic'), ('Stage', 'stage')]


def _level_values(sv_level, la_level, acc_key, lrap_key):
    top1, at_n, top3_any, top3_all = sv_level[acc_key]
    lrap, aupr = la_level[lrap_key]
    return (top1, at_n, top3_any, top3_all, lrap, aupr)


def _fmt_cell(value, others, group_divider):
    text = pct(value)
    if others and value > max(others):
        text = '<strong class="best">{}</strong>'.format(text)
    cls = ' class="firstcol"' if group_divider else ''
    return '<td{cls}>{text}</td>'.format(cls=cls, text=text)


def _model_row(model_name, row_label, values_by_level, other_values_by_level_list=None):
    others_list = other_values_by_level_list or []
    cells = []
    for i, (_, level_key) in enumerate(LEVELS):
        own = values_by_level[level_key]
        for j, v in enumerate(own):
            others = [ov[level_key][j] for ov in others_list if ov is not None]
            group_divider = (j == 0 and i > 0)
            cells.append(_fmt_cell(v, others, group_divider))
    return '<tr><td class="modelcol">{mn}</td><td class="rowlabel">{rl}</td>{cells}</tr>'.format(
        mn=esc(model_name), rl=esc(row_label), cells=''.join(cells))


def build_summary_table_html(tag, label, out_json, zoomer_jsons=None):
    zoomer_jsons = [(n, j) for n, j in (zoomer_jsons or []) if j is not None]

    out = ['<h3>{}</h3>'.format(esc(label)), '<div class="tablewrap"><table class="summarytable"><thead><tr>',
           '<th rowspan="2" class="idcol">Model</th><th rowspan="2" class="idcol">N</th>']
    for level_title, _ in LEVELS:
        out.append('<th colspan="6" class="grouphead">{}</th>'.format(esc(level_title)))
    out.append('</tr><tr>')
    for i in range(len(LEVELS)):
        th_fc = '' if i == 0 else ' class="firstcol"'
        out.append('<th{fc}>top-1</th><th>at-n</th><th>top-3-any</th><th>top-3-all</th><th>LRAP</th><th>AUPR</th>'.format(fc=th_fc))
    out.append('</tr></thead><tbody>')

    def model_values(j):
        sv, la = j['seen_vs_overall'], j['lrap_aupr_seen_vs_overall']
        n_seen, n_total = j['n_seen'], len(j.get('results', j.get('technique_results', [])))
        seen = {lk: _level_values(sv[lk], la[lk], 'seen', 'seen') for _, lk in LEVELS}
        overall = {lk: _level_values(sv[lk], la[lk], 'overall', 'overall') for _, lk in LEVELS}
        return n_seen, n_total, seen, overall

    models = [('CAM-LDS', out_json)] + zoomer_jsons
    parsed = [(name, *model_values(j)) for name, j in models]

    for name, n_seen, n_total, seen, overall in parsed:
        other_seen = [s for (n2, _, _, s, _) in parsed if n2 != name]
        other_overall = [o for (n2, _, _, _, o) in parsed if n2 != name]
        out.append(_model_row(name, 'Seen ({})'.format(n_seen), seen, other_seen))
        out.append(_model_row(name, 'Overall ({})'.format(n_total), overall, other_overall))
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def dataset_wide_sample_counts():
    with open(STEP_TACTICS_PATH) as f:
        report = json.load(f)
    counts = {}
    for row in report:
        for occ in row['occurrences']:
            t = base_technique(occ['technique'])
            counts[t] = counts.get(t, 0) + 1
    return counts


def build_scope_block_html():
    counts = dataset_wide_sample_counts()
    zero_sample = [t for t in ZOOMER_TECHNIQUES if counts.get(t, 0) == 0]
    n_realized = len(ZOOMER_TECHNIQUES) - len(zero_sample)
    return (
        '<p class="scopeline"><strong>Technique scope:</strong> {} trainable</p>\n'
        '<p class="scopeline"><strong>Tactic scope:</strong> {}</p>\n'
        '<p class="scopeline"><strong>Stage scope:</strong> {}</p>'
    ).format(n_realized, len(TACTIC_IDS), len(STAGE_ABBREV))


ZOOMER_RESULTS_DIR = os.path.join(os.path.dirname(TECHNIQUE_DIR), 'zoomer', 'results')


def _load_json_if_exists(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_summary_page(scenarios):
    out = ['<section class="summarypage">', '<h1 class="pagehead">Scenario summaries</h1>']
    out.append(build_scope_block_html())
    for tag, label, _kw, zoomer_tag, zoomer_single_tag in scenarios:
        with open(os.path.join(RESULTS_DIR, 'camlds_technique_wide_test_results_{}.json'.format(tag))) as f:
            out_json = json.load(f)
        zoomer_path = os.path.join(ZOOMER_RESULTS_DIR, 'ttp_recognition_multilabel_{}.json'.format(zoomer_tag)) if zoomer_tag else None
        zoomer_single_path = os.path.join(ZOOMER_RESULTS_DIR, 'ttp_recognition_{}.json'.format(zoomer_single_tag)) if zoomer_single_tag else None
        zoomer_jsons = [
            ('ZOOMER (multi)', _load_json_if_exists(zoomer_path)),
            ('ZOOMER (single)', _load_json_if_exists(zoomer_single_path)),
        ]
        out.append(build_summary_table_html(tag, label, out_json, zoomer_jsons=zoomer_jsons))
    out.append('</section>')
    return '\n'.join(out)


def build_detail_page(tag, label, test_file=None, scenario=None, zoomer_tag=None, zoomer_single_tag=None):
    train_counts = compute_train_counts(tag_test_file=test_file, tag_scenario=scenario)
    zoomer_path = os.path.join(ZOOMER_RESULTS_DIR, 'ttp_recognition_multilabel_{}.json'.format(zoomer_tag)) if zoomer_tag else None
    zoomer_single_path = os.path.join(ZOOMER_RESULTS_DIR, 'ttp_recognition_{}.json'.format(zoomer_single_tag)) if zoomer_single_tag else None
    rows, _ = build_rows(tag, train_counts, zoomer_path=zoomer_path, zoomer_single_path=zoomer_single_path)
    zoomer_models = []
    if zoomer_tag:
        zoomer_models.append(('ZOOMER (multi)', 'z_'))
    if zoomer_single_tag:
        zoomer_models.append(('ZOOMER (single)', 'zs_'))
    return '<section class="scenario">' + build_html(tag, label, rows, zoomer_models=zoomer_models) + '</section>'


SCENARIOS = [
    ('scenario2_rootkit_bigdataset_argfix', 'Scenario 2, rootkit', {'test_file': '2_rootkit'},
     'scenario2_rootkit_camldsscope49', 'scenario2_rootkit_camldsscope49_single'),
    ('scenario3_sshapt', 'Scenario 3, ssh_apt', {'test_file': '3_ssh_apt'},
     'scenario3_sshapt_camldsscope49', 'scenario3_sshapt_camldsscope49_single'),
    ('scenario4_bigdataset_argfix', 'Scenario 4', {'scenario': 4}, 'scenario4_camldsscope49',
     'scenario4_camldsscope49_single'),
    ('scenario6_macrobinary_best_argfix', 'Scenario 6, macro_binary', {'test_file': '6_macro_binary'},
     'scenario6_macrobinary_camldsscope49', 'scenario6_macrobinary_camldsscope49_single'),
]


def main():
    summary_html = build_summary_page(SCENARIOS)
    detail_html = '\n'.join(
        build_detail_page(tag, label, zoomer_tag=z, zoomer_single_tag=zs, **kw)
        for tag, label, kw, z, zs in SCENARIOS)
    body = summary_html + '\n' + detail_html
    html_doc = '<!doctype html><title>CAM-LDS scenario report</title><style>{css}</style>{body}'.format(
        css=PAGE_CSS, body=body)

    html_path = os.path.join(RESULTS_DIR, 'scenario_combined_report.html')
    with open(html_path, 'w') as f:
        f.write(html_doc)
    print('Wrote {}'.format(html_path))

    pdf_path = os.path.join(RESULTS_DIR, 'scenario_combined_report.pdf')
    subprocess.run([
        'chromium', '--headless', '--disable-gpu', '--no-sandbox',
        '--print-to-pdf=' + pdf_path, '--no-pdf-header-footer', '--print-to-pdf-no-header',
        'file://' + html_path,
    ], check=True, capture_output=True)
    print('Rendered {}'.format(pdf_path))


if __name__ == '__main__':
    main()
