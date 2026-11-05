import html
import os
import subprocess
import sys

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_LDS_DIR_FOR_IMPORTS = os.path.dirname(RESULTS_DIR)
TECHNIQUE_SCRIPTS_DIR = os.path.join(CAM_LDS_DIR_FOR_IMPORTS, 'technique', 'scripts')
if TECHNIQUE_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, TECHNIQUE_SCRIPTS_DIR)

METHOD_ORDER = ["KnowHow", "ZOOMER", "CAM-LDS"]

STAGES = [
    {
        "title": "Log Construction",
        "methods": {
            "KnowHow": [
                "event level logs: (subject, object, event, commandline)",
                "e.g. /usr/bin/zmcontroller event_connect 172.17.100.254:53.",
            ],
            "ZOOMER": [
                "provenance subgraph per step (e.g. 198 nodes, 851 edges)",
            ],
            "CAM-LDS": [
                "provenance subgraph per step",
                "convert subgraph to sequences (flattened to sentence text)",
                "e.g. /usr/bin/zmcontroller event_connect 172.17.100.254:53.",
            ],
        },
    },
    {
        "title": "Log Simplification",
        "methods": {
            "KnowHow": [
                "simplify_text() / sub_filepath() (same function as CTI-side, not re-run on logs)",
                "none applied at match time -- reuses the already-generalized sequences/ text",
                "e.g. /usr/bin/zmcontroller connect internal network address.",
            ],
            "ZOOMER": [
                "none -- no separate simplification stage",
            ],
            "CAM-LDS": [
                "simplify file paths & IPs",
            ],
        },
    },
    {
        "title": "Logs Embedding",
        "methods": {
            "KnowHow": [
                "FastText embedding of the events",
            ],
            "ZOOMER": [
                "Deep model: GAT for raw subgraph embedding",
            ],
            "CAM-LDS": [
                "RoBERTa, step subgraph sequence level",
            ],
        },
    },
    {
        "title": "2nd Source",
        "methods": {
            "KnowHow": [
                "CTI reports: welivesecurity.com, 80 reports",
                "MITRE procedure examples: 47 techniques",
            ],
            "ZOOMER": [
                "none -- logs only",
            ],
            "CAM-LDS": [
                "MITRE ATT&CK technique descriptions: 47 techniques",
            ],
        },
    },
    {
        "title": "2nd Source Processing and Embedding",
        "methods": {
            "KnowHow": [
                ("Processing:", "simplify file paths & IPs -> create gIoCs (SVO) "
                                 "-> embed with FastText -> cluster into the CKD (MeanShift)"),
            ],
            "ZOOMER": [
                ("Source:", "only ONE source (the provenance graph itself) -- no "
                             "separate 2nd source; Wide model attaches onto the Deep "
                             "model, both built from that same graph, so Wide can "
                             "MEMORIZE known patterns while Deep learns general ones"),
                ("Abstract Node (18-dim):", "counts of node TYPES in the graph "
                                              "(e.g. system file, user process)"),
                ("Edge (47-dim):", "counts of EVENT TYPES on the graph's edges "
                                     "(e.g. CONNECT, EXECVE)"),
                ("IoC (28-dim):", "counts of common IoCs (file paths/IPs) associated "
                                    "with a specific tactic (14 tactics), sourced from "
                                    "Atomic Red Team's technique test YAML files"),
                ("Combine:", "all 3 families merged together FIRST -> 93 raw numbers"),
                ("Discretize:", "a feature discretization mechanism is employed to "
                                  "transform continuous values into categorical "
                                  "formats (hcat)"),
                ("Cross-Product:", "30 random feature combinations (by masking) "
                                     "added onto hcat"),
                ("Wide output:", "hcat + combinations -> linear layer -> 64-dim hwide"),
            ],
            "CAM-LDS": [
                ("Processing:", "no processing"),
                ("Embedding:", "RoBERTa (separate template encoder), "
                                "one vector per full technique description"),
                ("Wide features:", "reuses ZOOMER's own wide model (Abstract Node "
                                     "+ Edge) -- concatenated onto the step's own "
                                     "sequence embedding, not the template side"),
            ],
        },
    },
    {
        "title": "Training",
        "methods": {
            "KnowHow": [
                "none",
            ],
            "ZOOMER": [
                ("Loss:", "prototypical network"),
                ("1. Prototype:", "create the embedding (prototype) of the technique by "
                                    "averaging 3 training samples of the same technique"),
                ("2. Distance:", "use 1-2 more samples as query to check the distance "
                                   "between prototype and query"),
                ("3. Loss:", "pull the query closer to its true technique's prototype, "
                              "push it away from every other prototype"),
            ],
            "CAM-LDS": [
                ("Loss:", "multi-label prototype contrastive loss"),
                ("Eq.:", "L = -1/|P| Σₖ∈ₖᵖ log[ exp(sim(z,cpₖ)/τ) / "
                         "Σⱼ∉ᵖ exp(sim(z,cpⱼ)/τ) ]"),
                ("step embedding (z):", "this step's embedding (text + wide)"),
                ("technique embedding (cp):", "technique prototype = fixed description embedding (2nd source)"),
                ("P =", "this step's true technique(s); τ = learned temperature"),
            ],
        },
    },
    {
        "title": "Testing",
        "methods": {
            "KnowHow": [
                "cosine similarity between CKD (gIoCs) and log event",
            ],
            "ZOOMER": [
                "step embedding compared against all TTP prototypes; predicted class = "
                "nearest prototype under Euclidean distance",
                ("Eq.:", "d(x,cₖ) = ||z - cₖ||₂"),
            ],
            "CAM-LDS": [
                "cosine similarity between technique embedding and query step",
            ],
        },
    },
    {
        "title": "Scoring",
        "methods": {
            "KnowHow": [
                ("Technique:", "average each technique's score across all events in the"),
                ("Technique to Tactic:", "MITRE official technique->tactic tags"),
                ("Tactic to Stage:", "use KnowHow's own tactic-to-stage convert table"),
            ],
            "ZOOMER": [
                ("Technique:", "test step embedding -- check its distance from the "
                                "technique prototypes -- scoring for every technique"),
                ("Technique to Tactic:", "MITRE official technique->tactic tags"),
                ("Tactic to Stage:", "use KnowHow's own tactic-to-stage convert table"),
            ],
            "CAM-LDS": [
                ("Technique:", "test step embedding -- check its cosine similarity to the "
                                "technique templates -- scoring for every technique"),
                ("Technique to Tactic:", "MITRE official technique->tactic tags"),
                ("Tactic to Stage:", "use KnowHow's own tactic-to-stage convert table"),
            ],
        },
    },
]

WORKED_EXAMPLE_STEP = '2_cron-14'
WORKED_EXAMPLE_SEQ_PATH = os.path.join(
    os.path.dirname(RESULTS_DIR), 'sequences_raw', 'stealth', 'T1070-004',
    'sequence_2_cron-14_videoserver.json')
WORKED_EXAMPLE_GENERALIZED_SEQ_PATH = os.path.join(
    os.path.dirname(RESULTS_DIR), 'sequences', 'stealth', 'T1070-004',
    'sequence_2_cron-14_videoserver.json')

WORKED_EXAMPLE = {
    "tactic": "stealth",
    "technique": "T1070-004",
}


def load_worked_example_events():
    import json
    with open(WORKED_EXAMPLE_SEQ_PATH) as f:
        seq = json.load(f)
    return seq['step'], seq['sequence']


def load_worked_example_generalized_events():
    import json
    with open(WORKED_EXAMPLE_GENERALIZED_SEQ_PATH) as f:
        seq = json.load(f)
    return seq['sequence']


def load_worked_example_graph_size():
    import json
    from wide_feature_bridge import build_step_to_graph_path

    step_to_path = build_step_to_graph_path()
    with open(step_to_path[WORKED_EXAMPLE_STEP]) as f:
        g = json.load(f)
    return len(g['nodes']), len(g['links'])


def load_worked_example_wide_features():
    from wide_feature_bridge import (
        ALL_DIMS, HNODE_DIMS, NODE_EDGE_DIMS, build_step_to_graph_path, load_raw_vector,
    )
    from discretize_features import load_bins, _bucket_index

    step_to_path = build_step_to_graph_path()
    raw_vector = load_raw_vector(step_to_path[WORKED_EXAMPLE_STEP])
    raw_by_dim = dict(zip(ALL_DIMS, raw_vector))
    edge_dims = [d for d in NODE_EDGE_DIMS if d not in HNODE_DIMS]
    ioc_dims = [d for d in ALL_DIMS if d not in NODE_EDGE_DIMS]
    all_node = [(dim, raw_by_dim[dim]) for dim in HNODE_DIMS]
    all_edge = [(dim, raw_by_dim[dim]) for dim in edge_dims]
    all_ioc = [(dim, raw_by_dim[dim]) for dim in ioc_dims]

    bins = load_bins()
    centers = bins['tmpFile']
    idx = _bucket_index(raw_by_dim['tmpFile'], centers)
    discretize_example = (raw_by_dim['tmpFile'], centers, [1 if i == idx else 0 for i in range(len(centers))])

    return {
        'all_node': all_node,
        'all_edge': all_edge,
        'all_ioc': all_ioc,
        'discretize_example': discretize_example,
    }

PAGE_CSS = """
:root {
  --bg: #ffffff; --ink: #000000; --ink-soft: #444444; --rule: #000000;
}
@page { size: 600mm 280mm; margin: 10mm 14mm; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink);
  font-family: "Nimbus Roman", "Times New Roman", Georgia, serif;
  font-size: 12px; line-height: 1.4; margin: 0; }
h1.title { font-size: 20px; color: var(--ink); margin: 0 0 4px; }
p.subtitle { color: var(--ink-soft); font-size: 10.5px; margin: 0 0 20px; max-width: 900px; }
.diagram { display: flex; flex-wrap: nowrap; align-items: stretch; gap: 0; margin-top: 22px; }
.stagebox { border: 1.3px solid var(--ink); width: 250px; flex: 1 1 250px; min-width: 0; display: flex; flex-direction: column; }
.stagehead { font-weight: 700; font-size: 11.5px; text-align: center; padding: 6px 4px;
  border-bottom: 1.2px solid var(--ink); }
.methodblock { padding: 6px 8px; }
.methodblock + .methodblock { border-top: 1px solid var(--rule); }
.methodname { font-weight: 700; font-size: 9.5px; text-transform: uppercase;
  letter-spacing: 0.04em; margin: 0 0 3px; }
.methodblock ul { margin: 0; padding-left: 13px; }
.methodblock li { font-size: 9px; margin: 0 0 2px; line-height: 1.3; }
.arrow { font-size: 18px; padding: 0 5px; color: var(--ink); align-self: center; }
p.legend { font-size: 9px; color: var(--ink-soft); margin-top: 16px; }
.example-page { page-break-before: always; padding-top: 4px; }
h1.example-title { font-size: 18px; margin: 0 0 4px; }
p.example-meta { font-size: 10.5px; color: var(--ink-soft); margin: 0 0 16px; }
h2.example-heading { font-size: 12.5px; margin: 18px 0 6px; border-bottom: 1px solid var(--ink); padding-bottom: 2px; }
ol.example-events { font-size: 10px; margin: 0; padding-left: 20px; line-height: 1.6; }
.example-method-note { font-size: 9.5px; color: var(--ink-soft); margin: 0 0 8px; max-width: 700px; }
ol.example-abstraction { font-size: 10px; margin: 0; padding-left: 20px; line-height: 1.6; }
ol.example-abstraction .why, ol.example-events .why { color: var(--ink-soft); font-size: 9px; }
"""


def render_bullet(b):
    if isinstance(b, tuple):
        if len(b) == 3:
            label, text, extra_style = b
        else:
            label, text = b
            extra_style = ''
        style_attr = ' style="{}"'.format(extra_style) if extra_style else ''
        return '<li{}><strong>{}</strong> {}</li>'.format(style_attr, html.escape(label), html.escape(text))
    return '<li>{}</li>'.format(html.escape(b))


def render_diagram():
    out = ['<div class="diagram">']
    for i, stage in enumerate(STAGES):
        out.append('<div class="stagebox">')
        out.append('<div class="stagehead">{}</div>'.format(html.escape(stage["title"])))
        for method in METHOD_ORDER:
            bullets = stage["methods"].get(method, [])
            li = ''.join(render_bullet(b) for b in bullets)
            out.append(
                '<div class="methodblock"><p class="methodname">{}</p><ul>{}</ul></div>'.format(
                    html.escape(method), li))
        out.append('</div>')
        if i < len(STAGES) - 1:
            out.append('<span class="arrow">&#8594;</span>')
    out.append('</div>')
    return '\n'.join(out)


def render_worked_example():
    ex = WORKED_EXAMPLE
    step, raw_events = load_worked_example_events()
    events_html = ''.join(
        '<li>{}</li>'.format(html.escape(e)) for e in raw_events)

    generalized_events = load_worked_example_generalized_events()
    generalized_html = ''.join(
        '<li>{}</li>'.format(html.escape(e)) for e in generalized_events)

    n_nodes, n_edges = load_worked_example_graph_size()

    wf = load_worked_example_wide_features()
    node_html = ', '.join(
        '{}: {}'.format(html.escape(dim), v) for (dim, v) in wf['all_node'])
    edge_html = ', '.join(
        '{}: {}'.format(html.escape(dim), v) for (dim, v) in wf['all_edge'])
    ioc_html = ', '.join(
        '{}: {}'.format(html.escape(dim), v) for (dim, v) in wf['all_ioc'])
    ioc_hit = any(v for (_, v) in wf['all_ioc'])
    ioc_note = 'real hit' if ioc_hit else 'all zero'

    (tmp_value, tmp_centers, tmp_onehot) = wf['discretize_example']
    discretize_line = 'Discretize example: tmpFile = {} -&gt; centers {} -&gt; one-hot {}'.format(
        tmp_value, tmp_centers, tmp_onehot)

    return """
<div class="example-page">
<h1 class="example-title">Example</h1>
<p class="example-meta">step {step}  (tactic: {tactic}, technique: {technique})</p>

<h2 class="example-heading">Raw events</h2>
<ol class="example-events">{events}</ol>

<h2 class="example-heading">KnowHow</h2>
<ol class="example-abstraction">{generalized}</ol>

<h2 class="example-heading">ZOOMER</h2>
<p class="example-method-note">Deep -- nodes: {n_nodes}, edges: {n_edges}</p>
<p class="example-method-note">Wide -- raw counts, before discretize + cross-product (below are the inputs to that step, not the final wide vector):</p>
<p class="example-method-note">Abstract Node: {node_counts}</p>
<p class="example-method-note">Edges: {edge_counts}</p>
<p class="example-method-note">IoC ({ioc_note}): {ioc_counts}</p>
<p class="example-method-note">Discretize: continuous counts -&gt; categorical one-hot (h_cat)</p>
<p class="example-method-note">{discretize_line}</p>
<p class="example-method-note">Cross-Product: 30 random feature combinations (by masking) added onto h_cat</p>
<p class="example-method-note">Wide output: h_cat + cross-product -&gt; linear layer -&gt; 64-dim h_wide</p>
<p class="example-method-note">Final TSG embedding: h_wide + h_deep, concatenated</p>

<h2 class="example-heading">CAM-LDS</h2>
<ol class="example-abstraction">{generalized}</ol>
<p class="example-method-note">Wide model: reuses ZOOMER's wide model output above -- concatenated onto this sequence's own text embedding</p>
</div>
""".format(
        step=html.escape(step), tactic=html.escape(ex["tactic"]),
        technique=html.escape(ex["technique"]), events=events_html,
        generalized=generalized_html, n_nodes=n_nodes, n_edges=n_edges,
        node_counts=node_html, edge_counts=edge_html, ioc_counts=ioc_html,
        ioc_note=ioc_note, discretize_line=discretize_line,
    )


def main():
    body = """
<h1 class="title">Pipeline Comparison: KnowHow vs. ZOOMER vs. CAM-LDS</h1>
{diagram}
{example}
""".format(diagram=render_diagram(), example=render_worked_example())

    html_doc = '<!doctype html><title>Pipeline Comparison</title><style>{css}</style>{body}'.format(
        css=PAGE_CSS, body=body)

    html_path = os.path.join(RESULTS_DIR, 'pipeline_report.html')
    with open(html_path, 'w') as f:
        f.write(html_doc)
    print('Wrote {}'.format(html_path))

    pdf_path = os.path.join(RESULTS_DIR, 'pipeline_report.pdf')
    subprocess.run([
        'chromium', '--headless', '--disable-gpu', '--no-sandbox',
        '--print-to-pdf=' + pdf_path, '--no-pdf-header-footer', '--print-to-pdf-no-header',
        'file://' + html_path,
    ], check=True, capture_output=True)
    print('Rendered {}'.format(pdf_path))


if __name__ == '__main__':
    main()
