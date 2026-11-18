import glob
import html
import json
import os
import re
import subprocess
import sys

COMBINE_RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_LDS_DIR = os.path.dirname(COMBINE_RESULTS_DIR)
TECHNIQUE_RESULTS_DIR = os.path.join(CAM_LDS_DIR, 'technique', 'results')
ZOOMER_RESULTS_DIR = os.path.join(CAM_LDS_DIR, 'zoomer', 'results')
if TECHNIQUE_RESULTS_DIR not in sys.path:
    sys.path.insert(0, TECHNIQUE_RESULTS_DIR)

from build_combined_page import (
    classify, expand_ranked, score_cell, esc, TACTIC_IDS, STAGE_ABBREV,
    TACTIC_TO_STAGE, load_technique_to_tactics,
)

RUN_TAG_RE = re.compile(
    r'^graph_encoder_(mymodel|knowhow|zoomer)_template_encoder_(mymodel|knowhow|zoomer|none)_'
    r'training_(mymodel|zoomer|none)_(?:(\d+)epoch_)?(.+?)(_graphseq_generalized)?(_wide)?(_k(\d+))?$')

ZOOMER_OWN_RE = re.compile(r'^ttp_recognition_multilabel_(.+?)_camldsscope(?:\d+)?(?:_v2|_nocron)?$')

ZOOMER_MULTILABEL_TEST_FILES = {
    '6_macro_binary': '6_macro_binary',
    'scenario2_rootkit': '2_rootkit',
    'scenario3_sshapt': '3_ssh_apt',
    'scenario4': '4',
    'scenario6_macrobinary': '6_macro_binary',
}

ALSO_IN_TABLE2 = {
    # scenario-4 table 3 row 2, also shown as a copy in table 2 for comparison
    'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_all_scenario4',
    # scenario-6 table 3 row 1, also shown as a copy in table 2 for comparison
    'knowhow_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_all',
}
HIDDEN_FROM_REPORT = {
    'knowhow_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_fileip_scenario4',
    # scenario-6 table 2: drop the cron-excluded (_v2) ZOOMER-own multi-label baseline
    'ttp_recognition_multilabel_6_macro_binary_camldsscope_v2',
    # scenario-6 table 2: drop stale pre-determinism-fix ZOOMER-own multi-label reruns,
    # keep only today's fresh rerun (ttp_recognition_multilabel_6_macro_binary_camldsscope)
    'ttp_recognition_multilabel_scenario6_macrobinary_camldsscope49',
    'ttp_recognition_multilabel_scenario6_macrobinary_camldsscope30',
    # scenario-6 table 3: drop all template=RoBERTa (mymodel) rows and all fileip rows
    'knowhow_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_fileip',
    'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_fileip',
    'knowhow_mymodel_zoomer_k1_wide_from_sequences_dev_fileip',
    'mymodel_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_all',
    'mymodel_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_fileip',
    'mymodel_mymodel_mymodel_k1_wide_from_sequences_dev_all',
    'mymodel_mymodel_mymodel_k1_wide_from_sequences_dev_fileip',
    'mymodel_mymodel_zoomer_k1_wide_from_sequences_dev_fileip',
    'mymodel_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_all_scenario4',
    # scenario-6 table 1: drop the cron-excluded (_nocron) variant of row 6
    'knowhow_mymodel_mymodel_k1_raw_wide_nocron',
    # scenario-4 table 3: wide-v2 experiment rows, pulled from the report (not working
    # as expected) -- underlying code (abstract_node_v2.py, discretize_features_v2.py,
    # the wiring) is left in place in case it's revisited later.
    'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_all_scenario4_widev2',
    'knowhow_mymodel_mymodel_k1_raw_wide_scenario4_widev2',
    # scenario-6: drop the cron-excluded rows that were rendered blue (the two table-1
    # duplicate rows stay visible in table 1 -- see HIDDEN_FROM_TABLE3 instead, so
    # "table 1 similar" still has a real row number to point to)
    'knowhow_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_all_v2',
    'knowhow_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_fileip_v2',
    'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_all_v2',
    'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_fileip_v2',
    'mymodel_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_all_v2',
}

TABLE3_PAIRS = {
    'knowhow_mymodel_mymodel_k1_generalized_nowide': 'knowhow_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_all',
    'knowhow_mymodel_mymodel_k1_generalized_wide': 'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_all',
    'knowhow_mymodel_mymodel_k1_generalized_nowide_scenario4': 'knowhow_mymodel_mymodel_k1_simplified_nowide_from_sequences_dev_all_scenario4',
    'knowhow_mymodel_mymodel_k1_generalized_wide_scenario4': 'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_all_scenario4',
    # chains off the row above: original -> dev_all -> dev_all-with-wide-features-v2
    'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_all_scenario4': 'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_all_scenario4_widev2',
}

# This key only exists in TABLE3_PAIRS to chain the ordering (so its widev2 row lands
# right after it); unlike the other keys it's not itself a "paired with a dev-variant"
# row in the blue-highlight sense, so it's excluded from the blue styling.
NOT_BLUE_IN_TABLE3 = {
    'knowhow_mymodel_mymodel_k1_wide_from_sequences_dev_all_scenario4',
}
ALSO_IN_TABLE3 = set(TABLE3_PAIRS.keys())

# Rows with no dev_all/dev_fileip suffix (so row_category() would put them in table 1
# by default) that we want shown ONLY in table 3 instead, for a direct side-by-side
# comparison -- paired with HIDDEN_FROM_TABLE1 below to suppress their natural home.
FORCE_INTO_TABLE3 = {
    'knowhow_mymodel_mymodel_k1_raw_wide_scenario4_widev2',
}

# Manual ancestor overrides for FORCE_INTO_TABLE3 rows, used only by the "table 1
# similar" column -- these aren't real dev-variant pairs, so they can't go through
# the TABLE3_PAIRS chain without side effects (that would also drag the ancestor
# itself into table 3).
TABLE3_MANUAL_ANCESTOR = {
    'knowhow_mymodel_mymodel_k1_raw_wide_scenario4_widev2': 'knowhow_mymodel_mymodel_k1_raw_wide_scenario4',
}


def reorder_table3_pairs(rows_for_group):
    # Walk the group once, and the first time we see either half of a known
    # (original, dev-variant) pair, emit both together right there -- so the
    # original always sits immediately above its dev-variant match instead of
    # all originals clustering before all dev rows (which plain alphabetical
    # discovery order would otherwise produce).
    by_tag = {r['run_tag']: r for r in rows_for_group}
    placed = set()
    out = []
    for r in rows_for_group:
        tag = r['run_tag']
        if tag in placed:
            continue
        out.append(r)
        placed.add(tag)
        # Follow the chain (original -> dev-variant -> widev2-of-that-dev-variant -> ...)
        # instead of stopping after one hop, so any further pairing off the dev-variant
        # (e.g. a wide-v2 ablation of it) still lands immediately after it too.
        next_tag = TABLE3_PAIRS.get(tag)
        while next_tag and next_tag in by_tag and next_tag not in placed:
            out.append(by_tag[next_tag])
            placed.add(next_tag)
            next_tag = TABLE3_PAIRS.get(next_tag)
    return out

PAGE_CSS = """
:root {
  --bg: #fbfaf7; --ink: #211e1a; --ink-soft: #6b6358; --rule: #c9c2b2;
  --accent: #6b4a2b; --best: #a1352b; --nocron: #2b5a8f;
  --correct-bg: #e4f3e3; --partial-bg: #fbf1cf; --wrong-bg: #fbe3e1;
}
@page { size: A3 landscape; margin: 8mm 10mm; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: "Nimbus Roman", "Times New Roman", Georgia, serif;
  font-size: 10.5px; line-height: 1.2; margin: 0; padding: 4px; }
h1.pagehead { font-size: 17px; margin: 0 0 6px; }
p.subtitle { font-size: 9.5px; color: var(--ink-soft); margin: 2px 0 10px; }
h3 { font-size: 12.5px; margin: 14px 0 4px; color: var(--accent); }
h3:first-of-type { margin-top: 4px; }
table.summarytable { border-collapse: collapse; width: auto; min-width: 900px; font-variant-numeric: tabular-nums; }
table.summarytable th, table.summarytable td { padding: 3px 9px 3px 5px; font-size: 9.5px; white-space: nowrap; }
table.summarytable thead th { text-align: left; font-weight: 600; color: var(--ink-soft);
  border-bottom: 1.2px solid var(--rule); }
table.summarytable thead th.grouphead { text-align: center; font-size: 10px; color: var(--accent);
  border-bottom: 1.5px solid var(--accent); border-left: 1.5px solid var(--rule); }
table.summarytable thead tr:first-child th.idcol { border-bottom: none; }
table.summarytable tbody td { border-bottom: 1px solid var(--rule); }
tr.model-end td { border-bottom: 1.5px solid var(--accent); }
td.firstcol, th.firstcol { border-left: 1.5px solid var(--rule); }
td.modelcol { font-weight: 600; color: var(--accent); font-size: 9px; vertical-align: middle;
  border-left: none; border-bottom: 1.5px solid var(--accent); }
td.modelcol.paired { color: var(--nocron); border-bottom: 1.5px solid var(--nocron); }
td.rowlabel { padding-left: 12px; }
strong.best { color: var(--best); font-weight: 800; }
.detail-page { page-break-before: always; padding-top: 4px; }
h2.combo-title { font-size: 15px; margin: 0 0 3px; color: var(--accent); }
p.combo-sub { font-size: 9.5px; color: var(--ink-soft); margin: 0 0 14px; }
table.metrics { border-collapse: collapse; margin-bottom: 14px; }
table.metrics th, table.metrics td { border: 1px solid var(--rule); padding: 3px 8px; font-size: 10px; text-align: right; }
table.metrics th { text-align: center; }
table.metrics td.label { text-align: left; font-weight: 700; }
table.fulldetail { border-collapse: collapse; width: 100%; font-size: 9px; font-variant-numeric: tabular-nums; }
table.fulldetail thead th { text-align: left; font-weight: 600; font-size: 8.5px; color: var(--ink-soft);
  border-bottom: 1.2px solid var(--rule); padding: 2px 8px 2px 4px; white-space: nowrap; }
table.fulldetail tbody td { padding: 2px 8px 2px 4px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
table.fulldetail td.idcol { font-weight: 600; }
table.fulldetail td.correct { background: var(--correct-bg); }
table.fulldetail td.partial { background: var(--partial-bg); }
table.fulldetail td.wrong   { background: var(--wrong-bg); }
.predgrid { display: flex; flex-direction: column; line-height: 1.1; }
.pred { white-space: nowrap; }
.pred .lbl { font-weight: 600; }
.pred .scr { color: var(--ink-soft); }
"""

LEVELS = [('Technique', 'technique'), ('Tactic', 'tactic'), ('Stage', 'stage')]


# Custom --run-tag files whose name doesn't match RUN_TAG_RE at all -- hand-entered metadata
# (template, graph, training, test_file, k_per_technique), grouped into the ZOOMER-comparison table.
CUSTOM_TAG_METADATA = {
    'kmm_k1_chunkall': ('knowhow', 'mymodel', 'mymodel', '6_macro_binary', 1),
}


VALID_PARTS = {'mymodel', 'knowhow', 'zoomer'}


def parse_human_tag(run_tag):
    """Parses this redo's human-readable tags: {template}_{graph}_{training}_k{N}[_raw|_generalized|_simplified][_nowide|_wide]"""
    parts = run_tag.split('_')
    if len(parts) < 3 or parts[0] not in VALID_PARTS or parts[1] not in VALID_PARTS or parts[2] not in VALID_PARTS:
        return None
    template, graph, training = parts[0], parts[1], parts[2]
    rest = '_'.join(parts[3:])
    if 'nowide' in rest:
        use_wide = False
    elif 'wide' in rest:
        use_wide = True
    else:
        use_wide = False
    sequence_style = 'generalized' if ('generalized' in rest or 'simplified' in rest) else 'raw'
    k_match = re.search(r'k(\d+)', rest)
    k_value = int(k_match.group(1)) if k_match else None
    return template, graph, training, sequence_style, use_wide, k_value


def test_file_for_human_tag(run_tag):
    if run_tag.endswith('_scenario4') or '_scenario4_' in run_tag:
        return '4'
    return '6_macro_binary'


def discover_results():
    rows = []
    for path in sorted(glob.glob(os.path.join(COMBINE_RESULTS_DIR, '*.json'))):
        fname = os.path.basename(path)
        if fname.endswith('_train_history.json'):
            continue
        run_tag = fname[:-len('.json')]
        if run_tag in CUSTOM_TAG_METADATA:
            template, graph, training, test_file, k_value = CUSTOM_TAG_METADATA[run_tag]
            with open(path) as f:
                d = json.load(f)
            rows.append({
                'template': template, 'graph': graph, 'training': training, 'test_file': test_file,
                'graph_sequences': 'zoomer_own', 'sequence_style': 'raw', 'use_wide': False,
                'k_per_technique': k_value, 'n_epochs': None, 'run_tag': run_tag, 'data': d,
            })
            continue
        parsed = parse_human_tag(run_tag)
        if parsed is not None:
            template, graph, training, sequence_style, use_wide, k_value = parsed
            graph_sequences = 'generalized' if graph == 'knowhow' else sequence_style
            if graph == 'mymodel' and strip_v2(run_tag).endswith(DEV_VARIANT_SUFFIXES):
                # dev_all/dev_fileip are a simplified sequence source too, same wording
                # as the plain generalized folder -- just a different simplification --
                # even when "generalized"/"simplified" isn't literally in the tag text.
                graph_sequences = 'generalized'
            with open(path) as f:
                d = json.load(f)
            rows.append({
                'template': template, 'graph': graph, 'training': training,
                'test_file': test_file_for_human_tag(run_tag),
                'graph_sequences': graph_sequences, 'sequence_style': sequence_style, 'use_wide': use_wide,
                'k_per_technique': k_value, 'n_epochs': None, 'run_tag': run_tag, 'data': d,
            })
            continue
        m = RUN_TAG_RE.match(run_tag)
        if not m:
            continue
        graph, template, training, n_epochs, test_file, graphseq_suffix, wide_suffix, _k_full, k_value = m.groups()
        sequence_style = 'generalized' if (graphseq_suffix or graph == 'knowhow') else 'raw'
        if graph == 'knowhow':
            graph_sequences = 'generalized'
        else:
            graph_sequences = sequence_style
        with open(path) as f:
            d = json.load(f)
        rows.append({
            'template': template, 'graph': graph, 'training': training, 'test_file': test_file,
            'graph_sequences': graph_sequences, 'sequence_style': sequence_style, 'use_wide': bool(wide_suffix),
            'k_per_technique': int(k_value) if k_value else None,
            'n_epochs': int(n_epochs) if n_epochs else None,
            'run_tag': fname[:-len('.json')], 'data': d,
        })
    return rows


def discover_zoomer_own_baseline():
    rows = []
    for path in sorted(glob.glob(os.path.join(ZOOMER_RESULTS_DIR, 'ttp_recognition_multilabel_*.json'))):
        fname = os.path.basename(path)
        run_tag = fname[:-len('.json')]
        m = ZOOMER_OWN_RE.match(run_tag)
        if not m:
            continue
        test_file = ZOOMER_MULTILABEL_TEST_FILES.get(m.group(1))
        if test_file is None:
            continue
        with open(path) as f:
            d = json.load(f)
        rows.append({
            'template': 'zoomer_own', 'graph': 'zoomer_own', 'training': 'zoomer_own',
            'test_file': test_file, 'graph_sequences': 'zoomer_own', 'k_per_technique': None,
            'n_epochs': None, 'run_tag': run_tag, 'data': d,
        })
    return rows


ZOOMER_SINGLE_TAGS = {
    'ttp_recognition_scenario6_macrobinary_camldsscope49_single': '6_macro_binary',
    'ttp_recognition_scenario4_camldsscope49_single': '4',
    'ttp_recognition_scenario2_rootkit_camldsscope49_single': '2_rootkit',
    'ttp_recognition_scenario3_sshapt_camldsscope49_single': '3_ssh_apt',
}


def discover_zoomer_singlelabel_baseline():
    rows = []
    for tag, test_file in ZOOMER_SINGLE_TAGS.items():
        path = os.path.join(ZOOMER_RESULTS_DIR, tag + '.json')
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            rows.append({
                'template': 'zoomer_own_single', 'graph': 'zoomer_own_single', 'training': 'zoomer_own_single',
                'test_file': test_file, 'graph_sequences': 'zoomer_own', 'k_per_technique': None,
                'n_epochs': None, 'run_tag': tag, 'data': d,
            })
    return rows


def metric(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def fmt_pct(v):
    return '{:.1f}%'.format(v * 100) if isinstance(v, (int, float)) else '-'


TEMPLATE_DESC = {'mymodel': 'template from RoBERTa', 'knowhow': 'template from KnowHow', 'zoomer': 'template from zoomer'}
TRAINING_DESC = {'mymodel': 'training my model', 'zoomer': "training zoomer's loss"}
TRAINING_DESC_NONE = {'mymodel': 'training on my loss function', 'zoomer': "training on zoomer's loss function"}


def graph_desc(r):
    if r['graph'] == 'mymodel':
        if r['graph_sequences'] == 'generalized':
            return 'simplified graph from knowhow, encoded by my model'
        return 'graph from mymodel'
    if r['graph'] == 'knowhow':
        return 'graph from knowhow'
    return 'graph from zoomer'


CHUNK_LABEL = {'kmm_k1_chunkall': 'all chunks (uncapped)'}


def model_name(r):
    if r['template'] == 'zoomer_own':
        suffix = ' [cron excluded]' if is_cron_excluded(r['run_tag']) else ''
        return 'ZOOMER own pipeline, multi-label, dataset graph' + suffix
    if r['template'] == 'zoomer_own_single':
        return 'ZOOMER single-label recognition, fixed 49-technique scope'
    if r['run_tag'] in CHUNK_LABEL:
        return 'graph from mymodel, K=1, {}\ntemplate from KnowHow, training my model'.format(
            CHUNK_LABEL[r['run_tag']])
    if r['template'] == 'none':
        if r['graph'] == 'mymodel':
            base = 'graph from my model'
            if r.get('use_wide'):
                base += ' + wide from zoomer'
            if r.get('sequence_style') == 'generalized':
                base += ', simplification from knowhow'
            return '{}, {}'.format(base, TRAINING_DESC_NONE.get(r['training'], r['training']))
        if r['graph'] == 'zoomer':
            return 'graph from zoomer (wide built in), {}'.format(
                TRAINING_DESC_NONE.get(r['training'], r['training']))
        return 'no template (avg of examples), graph from {}'.format(r['graph'])
    line2 = '{}, {}'.format(TEMPLATE_DESC[r['template']], TRAINING_DESC[r['training']])
    if r.get('use_wide'):
        line2 = '+ wide view from zoomer, ' + line2
    base_tag = strip_v2(r['run_tag'])
    if base_tag.endswith('_sequencedev'):
        line2 += ' (sequences_dev)'
    elif base_tag.endswith('_from_sequences_dev_all'):
        line2 += ' (process + file + ip generalized)'
    elif base_tag.endswith('_from_sequences_dev_fileip'):
        line2 += ' (sequences_dev_fileip)'
    if is_cron_excluded(r['run_tag']):
        line2 += ' [cron excluded]'
    if r['run_tag'].endswith('_widev2'):
        line2 += ' [wide v2]'
    return '{}\n{}'.format(graph_desc(r), line2)


def _level_values(sv_level, la_level, key):
    top1, at_n, top3_any, top3_all = sv_level[key]
    lrap, aupr = la_level[key]
    return (top1, at_n, top3_any, top3_all, lrap, aupr)


def _fmt_cell(value, others, group_divider):
    text = fmt_pct(value)
    if value is not None and others and value >= max(others):
        text = '<strong class="best">{}</strong>'.format(text)
    cls = ' class="firstcol"' if group_divider else ''
    return '<td{cls}>{text}</td>'.format(cls=cls, text=text)


def _model_row(row_label, values_by_level, other_values_by_level_list, model_cell='', row_class=''):
    cells = []
    for i, (_, level_key) in enumerate(LEVELS):
        own = values_by_level[level_key]
        for j, v in enumerate(own):
            others = [ov[level_key][j] for ov in other_values_by_level_list if ov is not None]
            group_divider = (j == 0 and i > 0)
            cells.append(_fmt_cell(v, others, group_divider))
    return '<tr{rc}>{mc}<td class="rowlabel">{rl}</td>{cells}</tr>'.format(
        rc=row_class, mc=model_cell, rl=html.escape(row_label), cells=''.join(cells))


def combo_values(d):
    sv = d.get('seen_vs_overall')
    la = d.get('lrap_aupr_seen_vs_overall')
    if not sv or not la:
        return None
    n_seen = metric(d, 'n_seen')
    n_total = metric(d, 'n_total')
    if n_total is None:
        n_total = len(d.get('results', []))
    seen = {lk: _level_values(sv[lk], la[lk], 'seen') for _, lk in LEVELS}
    overall = {lk: _level_values(sv[lk], la[lk], 'overall') for _, lk in LEVELS}
    return n_seen, n_total, seen, overall


def _table3_ancestor(run_tag):
    # Walk a table-3 row's tag back through the pairing chain (widev2 -> dev_all ->
    # generalized original, etc.) to whichever ancestor tag is the one that actually
    # lives in table 1, so its row number there can be looked up.
    if run_tag in TABLE3_MANUAL_ANCESTOR:
        return TABLE3_MANUAL_ANCESTOR[run_tag]
    dev_to_original = {v: k for (k, v) in TABLE3_PAIRS.items()}
    current = run_tag
    while current in dev_to_original:
        current = dev_to_original[current]
    return current


def build_summary_table_html(heading, rows_for_group, category=None, table1_lookup=None):
    table1_lookup = table1_lookup if table1_lookup is not None else {}
    parsed = []
    for r in rows_for_group:
        vals = combo_values(r['data'])
        if vals is None:
            continue
        n_seen, n_total, seen, overall = vals
        parsed.append((model_name(r), n_seen, n_total, seen, overall, r['run_tag']))
    if not parsed:
        return ''

    show_table1_col = category == 2

    out = ['<h3>{}</h3>'.format(html.escape(heading)),
           '<table class="summarytable"><thead><tr>',
           '<th rowspan="2" class="idcol">#</th>',
           '<th rowspan="2" class="idcol">Model</th>']
    if show_table1_col:
        out.append('<th rowspan="2" class="idcol">table 1 similar</th>')
    out.append('<th rowspan="2" class="idcol">N</th>')
    for level_title, _ in LEVELS:
        out.append('<th colspan="6" class="grouphead">{}</th>'.format(html.escape(level_title)))
    out.append('</tr><tr>')
    for i in range(len(LEVELS)):
        th_fc = '' if i == 0 else ' class="firstcol"'
        out.append('<th{fc}>top-1</th><th>at-n</th><th>top-3-any</th><th>top-3-all</th><th>LRAP</th><th>AUPR</th>'.format(fc=th_fc))
    out.append('</tr></thead><tbody>')

    for i, (name, n_seen, n_total, seen, overall, run_tag) in enumerate(parsed, 1):
        if category == 0:
            table1_lookup[run_tag] = i
        other_seen = [s for (n2, _, _, s, _, _) in parsed if n2 != name]
        other_overall = [o for (n2, _, _, _, o, _) in parsed if n2 != name]
        is_blue = is_cron_excluded(run_tag) or (
            category == 2 and run_tag in ALSO_IN_TABLE3 and run_tag not in NOT_BLUE_IN_TABLE3)
        modelcol_class = 'modelcol paired' if is_blue else 'modelcol'
        model_cell = '<td class="idcol" rowspan="2">{}</td><td class="{mc}" rowspan="2">{}</td>'.format(
            i, html.escape(name).replace('\n', '<br>'), mc=modelcol_class)
        if show_table1_col:
            table1_num = table1_lookup.get(_table3_ancestor(run_tag))
            model_cell += '<td class="idcol" rowspan="2">{}</td>'.format(table1_num if table1_num is not None else '-')
        out.append(_model_row('Seen ({})'.format(n_seen), seen, other_seen, model_cell=model_cell))
        out.append(_model_row('Overall ({})'.format(n_total), overall, other_overall, row_class=' class="model-end"'))
    out.append('</tbody></table>')
    return '\n'.join(out)


SEQUENCES_LABEL = {
    'raw': 'raw graph (sequences_rawfull/)',
    'generalized': 'simplified graph (sequences/, generalized)',
    'zoomer_own': "no template -- single source embeds both the prototype and the query (ZOOMER-style)",
}


ROW_MOVE = {'knowhow_mymodel_mymodel_k1_simplified_nowide_sequencedev': 3}


def apply_row_moves(rows_for_group):
    rows_for_group = list(rows_for_group)
    for run_tag, target_index in ROW_MOVE.items():
        for i, r in enumerate(rows_for_group):
            if r['run_tag'] == run_tag:
                rows_for_group.insert(target_index, rows_for_group.pop(i))
                break
    return rows_for_group


DEV_VARIANT_SUFFIXES = ('_from_sequences_dev_all', '_from_sequences_dev_fileip')


CRON_EXCLUDED_SUFFIXES = ('_v2', '_nocron')
# _widev2 marks a wide-features-v2 (abstract_node_v2/discretize_features_v2) experiment --
# unrelated to the cron-exclusion "_v2" suffix above, but still needs stripping here so
# row_category()/model_name() see through it to the underlying combo (e.g. DEV_VARIANT_SUFFIXES
# matching for table 3 membership).
TRAILING_SUFFIX_RE = re.compile(r'(?:_v2|_nocron|_widev2|_scenario\d+)+$')


def is_cron_excluded(run_tag):
    return bool(re.search(r'(?:_v2|_nocron)(?:_scenario\d+)?$', run_tag))


def strip_v2(run_tag):
    return TRAILING_SUFFIX_RE.sub('', run_tag)


def row_category(r):
    if r['graph_sequences'] == 'zoomer_own':
        return 1
    if strip_v2(r['run_tag']).endswith(DEV_VARIANT_SUFFIXES):
        return 2
    return 0


HIDDEN_FROM_TABLE1 = {
    'knowhow_mymodel_mymodel_k1_simplified_nowide_sequencedev',
    'knowhow_mymodel_mymodel_k1_raw_wide_sequencedev',
    # scenario-4 table 1: graph from knowhow, template from RoBERTa, training my model
    'mymodel_knowhow_mymodel_k1_scenario4',
    # shown only in table 3 instead (see FORCE_INTO_TABLE3)
    'knowhow_mymodel_mymodel_k1_raw_wide_scenario4_widev2',
}

# The wide-pair "original" row is still a real table-1 row (unaffected), it's just not
# wanted as the leading duplicate in table 3 -- only its dev_all/widev2 descendants there.
HIDDEN_FROM_TABLE3 = {
    'knowhow_mymodel_mymodel_k1_generalized_wide_scenario4',
    'knowhow_mymodel_mymodel_k1_generalized_nowide_scenario4',
    'knowhow_mymodel_mymodel_k1_generalized_wide',
    'knowhow_mymodel_mymodel_k1_generalized_nowide',
}


def build_summary_page(rows):
    by_group = {}
    for r in rows:
        key = (row_category(r), r['test_file'])
        if not (key[0] == 0 and r['run_tag'] in HIDDEN_FROM_TABLE1):
            by_group.setdefault(key, []).append(r)
        if r['run_tag'] in ALSO_IN_TABLE2:
            by_group.setdefault((1, r['test_file']), []).append(r)
        if r['run_tag'] in (ALSO_IN_TABLE3 | FORCE_INTO_TABLE3) and r['run_tag'] not in HIDDEN_FROM_TABLE3:
            by_group.setdefault((2, r['test_file']), []).append(r)

    for key in by_group:
        by_group[key] = apply_row_moves(by_group[key])
        if key[0] == 2:
            by_group[key] = reorder_table3_pairs(by_group[key])

    out = ['<h1 class="pagehead">Combination pipeline: results summary</h1>']
    table1_lookup = {}
    for category, test_file in sorted(by_group):
        if category == 1:
            heading = 'Scenario: {}  --  {}'.format(test_file, SEQUENCES_LABEL['zoomer_own'])
        elif category == 2:
            heading = 'Scenario: {}  --  process + file + ip simplification'.format(test_file)
        else:
            heading = 'Scenario: {}'.format(test_file)
        out.append(build_summary_table_html(heading, by_group[(category, test_file)], category=category,
                                             table1_lookup=table1_lookup))
    return '\n'.join(out)


def build_full_detail_rows(d):
    technique_to_tactics = load_technique_to_tactics()
    tactic_to_stage_list = {t: [s] for t, s in TACTIC_TO_STAGE.items()}

    rows = []
    for r in (d.get('results') or d.get('technique_results') or []):
        true_techs = set(r['true_techniques'])
        tech_ranked = [(e['technique'], e['score']) for e in r['ranked']]
        tac_ranked = expand_ranked(tech_ranked, technique_to_tactics)
        stage_ranked = expand_ranked(tac_ranked, tactic_to_stage_list)

        true_tacs = set()
        for t in true_techs:
            true_tacs.update(technique_to_tactics.get(t, []))
        true_stages = {TACTIC_TO_STAGE[tac] for tac in true_tacs if tac in TACTIC_TO_STAGE}

        top3_tech = {t for t, _ in tech_ranked[:3]}
        top3_tac = {t for t, _ in tac_ranked[:3]}
        top3_stage = {t for t, _ in stage_ranked[:3]}

        rows.append({
            'file': r['file'],
            'true_techs': sorted(true_techs),
            'true_tacs': sorted(true_tacs),
            'true_stages': sorted(true_stages, key=lambda s: list(STAGE_ABBREV).index(s) if s in STAGE_ABBREV else 99),
            'n_events': r.get('n_events'),
            'tech_top3': tech_ranked[:3],
            'tac_top3': tac_ranked[:3],
            'stage_top3': stage_ranked[:3],
            'tech_verdict': classify(true_techs, top3_tech, 3),
            'tac_verdict': classify(true_tacs, top3_tac, 3),
            'stage_verdict': classify(set(true_stages), top3_stage, 3),
        })
    return rows


def render_full_detail_table(rows):
    out = ['<table class="fulldetail"><thead><tr>']
    out.append(''.join('<th>{}</th>'.format(esc(h)) for h in
               ['#', 'Step', 'True technique', 'True tactic', 'True stage', 'Events',
                'Technique score', 'Tactic score', 'Stage score']))
    out.append('</tr></thead><tbody>')
    for i, r in enumerate(rows, 1):
        tech_cell = score_cell(r['tech_verdict'], r['tech_top3'], lambda t: t)
        tac_cell = score_cell(r['tac_verdict'], r['tac_top3'], lambda t: TACTIC_IDS.get(t, t))
        stage_cell = score_cell(r['stage_verdict'], r['stage_top3'], lambda t: STAGE_ABBREV.get(t, t))
        out.append(
            '<tr><td class="idcol">{i}</td><td class="idcol">{f}</td>'
            '<td>{tt}</td><td>{ta}</td><td>{ts}</td><td>{ev}</td>{tc}{ac}{sc}</tr>'.format(
                i=i, f=esc(r['file']),
                tt=esc(', '.join(r['true_techs'])),
                ta=esc(', '.join(TACTIC_IDS.get(t, t) for t in r['true_tacs'])),
                ts=esc(', '.join(STAGE_ABBREV.get(s, s) for s in r['true_stages'])),
                ev=esc(r['n_events']),
                tc=tech_cell, ac=tac_cell, sc=stage_cell))
    out.append('</tbody></table>')
    return '\n'.join(out)


def render_detail_page(r):
    d = r['data']
    detail_rows = build_full_detail_rows(d)
    return """
<div class="detail-page">
<h2 class="combo-title">{name}</h2>
<p class="combo-sub">run_tag: {run_tag}  &middot;  test_file: {test_file}</p>
{table}
</div>
""".format(
        name=html.escape(model_name(r)).replace('\n', '<br>'),
        run_tag=html.escape(r['run_tag']), test_file=html.escape(r['test_file']),
        table=render_full_detail_table(detail_rows),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=str, default=None,
                     help='Read *.json results from this folder instead of the default combination_results/, '
                          'and write the report there too (e.g. scenario4_run/combination_results). ZOOMER-own '
                          'baselines (which live in the shared zoomer/results/ folder) are still included, '
                          'filtered to whichever scenario(s) this folder'"'"'s own results cover.')
    ap.add_argument('--exclude-cron-excluded', action='store_true',
                     help='Drop every row whose run_tag is a cron-excluded rerun (_v2/_nocron) -- i.e. show '
                          'only the original, pre-nocron-fix results.')
    args = ap.parse_args()

    if args.results_dir:
        global COMBINE_RESULTS_DIR
        COMBINE_RESULTS_DIR = os.path.abspath(args.results_dir)

    rows = [r for r in discover_results() if r['template'] != 'none']
    present_test_files = {r['test_file'] for r in rows}
    rows += [r for r in discover_zoomer_own_baseline() if r['test_file'] in present_test_files]
    rows += [r for r in discover_zoomer_singlelabel_baseline() if r['test_file'] in present_test_files]
    rows = [r for r in rows if r['run_tag'] not in HIDDEN_FROM_REPORT]
    if args.exclude_cron_excluded:
        rows = [r for r in rows if not is_cron_excluded(r['run_tag'])]

    if not rows:
        print('No completed combination results found in {}'.format(COMBINE_RESULTS_DIR))
        return

    body = build_summary_page(rows) + '\n' + '\n'.join(render_detail_page(r) for r in rows)

    html_doc = '<!doctype html><title>Combination Results Report</title><style>{css}</style>{body}'.format(
        css=PAGE_CSS, body=body)

    html_path = os.path.join(COMBINE_RESULTS_DIR, 'combination_report.html')
    with open(html_path, 'w') as f:
        f.write(html_doc)
    print('Wrote {}'.format(html_path))

    pdf_path = os.path.join(COMBINE_RESULTS_DIR, 'combination_report.pdf')
    subprocess.run([
        'chromium', '--headless', '--disable-gpu', '--no-sandbox',
        '--print-to-pdf=' + pdf_path, '--no-pdf-header-footer', '--print-to-pdf-no-header',
        'file://' + html_path,
    ], check=True, capture_output=True)
    print('Rendered {}'.format(pdf_path))


if __name__ == '__main__':
    main()
