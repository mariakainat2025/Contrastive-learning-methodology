import os
import sys
import json
import math

import torch
import torch.nn.functional as F
from networkx.readwrite import json_graph
from transformers import RobertaTokenizer, RobertaModel

TECHNIQUE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = '/csse/research/contructive-learning/CAM-LDS/scripts'
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
if TECHNIQUE_DIR not in sys.path:
    sys.path.insert(0, TECHNIQUE_DIR)

from parse_audit import parse_audit, to_subgraph
from graph_audit import subgraph_json_to_nx
from generalize_ip_and_file_path import _generalize_name, _lift_file_path, EVENT_PREFIX
import build_sequences

from config import ROBERTA_MODEL
from train_camlds_matcher_technique_wide import (
    load_sequences, leave_out_scenario_split, leave_out_split, load_templates, encode_one,
    ProjectionNetwork, WideModel, SEED, STEP_TACTICS_PATH, TECHNIQUE_MERGE, base_technique,
    TEMPLATE_DIR,
)
from wide_feature_bridge import build_step_to_graph_path, fit_wide_bins_and_masks, load_raw_vector
from discretize_features import discretize_vector
from cross_product import hwide_pre_projection
from test_camlds_matcher_technique_wide import load_technique_to_tactics, expand_ranked
from tactic_to_stage import TACTIC_TO_STAGE, tactics_to_stages

CAM_LDS_DIR = '/csse/research/contructive-learning/CAM-LDS'
MODEL_DIR = os.path.join(CAM_LDS_DIR, 'technique', 'checkpoints')
STEPS_BASE = '/csse/research/contructive-learning/aaaa/manifestations_filtered/manifestations_filtered/steps'
DIAG_GRAPH_DIR = os.path.join(CAM_LDS_DIR, 'graphs_multihost_diag')
os.makedirs(DIAG_GRAPH_DIR, exist_ok=True)


def find_hosts_for_step(step):
    """Every host directory under this step that has a Linux-audit-format
    audit.log (skips 'attacker' and 'wazuh', which log in different formats
    and were never candidates for pick_primary_host() in the first place)."""
    step_dir = os.path.join(STEPS_BASE, step)
    hosts = []
    if not os.path.isdir(step_dir):
        return hosts
    for h in sorted(os.listdir(step_dir)):
        audit_path = os.path.join(step_dir, h, 'logs', 'log', 'audit', 'audit.log')
        if os.path.isfile(audit_path) and os.path.getsize(audit_path) > 0:
            hosts.append((h, audit_path))
    return hosts


def build_sequence_and_graph_file(audit_path, tactic, technique, step, host):
    """Parse one host's raw audit.log -> subgraph -> (sentence text, path to a
    written RAW graph JSON for wide-feature lookup). The written graph is the
    RAW (pre-generalization) one -- matching what the official pipeline's
    wide_feature_bridge.py reads from CAM-LDS/graphs/, where IP/path
    generalization has NOT been applied. Generalization (for the text sentences
    only) is done on a separate copy so it never contaminates the node names the
    wide-feature extractor sees -- IP strings replaced with placeholder text like
    'external network address' fail classify_ip()'s ipaddress.ip_address() parse
    and silently fall back to 'internal', which previously corrupted internalIP/
    externalIP (and, via lifted file paths losing their '/run/' etc. prefix,
    systemFile/otherFile) counts for every multihost diagnostic check."""
    events = parse_audit(audit_path)
    sg = to_subgraph(events, tactic=tactic, attack=technique,
                      description='{} step={} host={}'.format(technique, step, host))
    G_raw = subgraph_json_to_nx(sg)

    graph_path = os.path.join(DIAG_GRAPH_DIR, '{}_{}.json'.format(step, host))
    with open(graph_path, 'w') as f:
        json.dump(json_graph.node_link_data(G_raw), f)

    G = G_raw.copy()
    for _, attrs in G.nodes(data=True):
        new_name, ip_found = _generalize_name(attrs['name'])
        if ip_found is not None:
            attrs['name'] = new_name
    for _, attrs in G.nodes(data=True):
        if attrs.get('type') == 'FILE':
            lifted = _lift_file_path(attrs['name'])
            if lifted is not None and lifted != attrs['name']:
                attrs['name'] = lifted
    for _, _, attrs in G.edges(data=True):
        et = attrs.get('edge_type')
        if et and et.startswith(EVENT_PREFIX):
            attrs['edge_type'] = et[len(EVENT_PREFIX):]

    node_name = {n: d['name'] for n, d in G.nodes(data=True)}
    sentences = [
        build_sequences.triplet_to_sentence(node_name[src], attrs['edge_type'], node_name[dst])
        for src, dst, attrs in G.edges(data=True)
        if attrs['edge_type'] not in build_sequences.RARE_EDGE_TYPES_STRIPPED
    ]
    sequence_text = ' '.join(sentences)

    return sequence_text, graph_path, len(sentences)


def main(run_tag='scenario4_basetech_redo', test_scenario='4', top_n=3, run_filter=None):
    device = torch.device('cuda')

    ckpt_path = os.path.join(MODEL_DIR, 'camlds_technique_wide_matcher_{}.pt'.format(run_tag))
    print('Loading checkpoint ->', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    split_seed = ckpt.get('split_seed', SEED)
    sequences_dir = ckpt.get('sequences_dir')
    ckpt_test_scenario = ckpt.get('test_scenario')
    ckpt_test_file_match = ckpt.get('test_file_match')

    entries = load_sequences(sequences_dir=sequences_dir)
    if ckpt_test_scenario:
        train_entries, test_entries = leave_out_scenario_split(entries, ckpt_test_scenario, seed=split_seed)
    elif ckpt_test_file_match:
        train_entries, test_entries = leave_out_split(entries, ckpt_test_file_match)
    else:
        train_entries, test_entries = leave_out_scenario_split(entries, test_scenario, seed=split_seed)
    train_steps_native = [e['file'] for e in train_entries if e['techniques']]

    templates = load_templates(template_dir=ckpt.get('template_dir'))
    trained_techniques = ckpt['techniques']
    test_techs_present = {t for e in test_entries for t in e['techniques_all']}
    extra = sorted(test_techs_present - set(trained_techniques))
    if extra:
        templates.update(load_templates(template_dir=ckpt.get('template_dir'), extra_techniques=extra))
    all_techniques = sorted(set(trained_techniques) | test_techs_present)
    all_techniques = [t for t in all_techniques if t in templates]

    proj_dims = ckpt['proj_dims']
    proj_dim = proj_dims[1]
    combined_dim = proj_dim + proj_dims[2]
    wide_dim_raw = proj_dims[3]

    log_proj = ProjectionNetwork(out_dim=proj_dim).to(device)
    text_proj = ProjectionNetwork(out_dim=combined_dim).to(device)
    wide_model = WideModel(wide_dim_raw, proj_dims[2]).to(device)
    log_proj.load_state_dict(ckpt['log_proj'])
    text_proj.load_state_dict(ckpt['text_proj'])
    wide_model.load_state_dict(ckpt['wide_model'])
    log_proj.eval(); text_proj.eval(); wide_model.eval()

    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    seq_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    tmpl_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    seq_encoder.load_state_dict(ckpt['seq_encoder'])
    tmpl_encoder.load_state_dict(ckpt['tmpl_encoder'])
    seq_encoder.eval(); tmpl_encoder.eval()
    for p in seq_encoder.parameters():
        p.requires_grad = False
    for p in tmpl_encoder.parameters():
        p.requires_grad = False

    trained_logit_scale = ckpt.get('logit_scale')
    display_scale = trained_logit_scale.exp().item() if trained_logit_scale is not None else 1.0

    print('Fitting wide-feature bins (same training split as the official run)...')
    native_step_to_path = build_step_to_graph_path()
    bins, masks, wide_dim = fit_wide_bins_and_masks(train_steps_native, native_step_to_path, seed=split_seed)

    with torch.no_grad():
        tmpl_embs = {
            t: F.normalize(text_proj(encode_one(tokenizer, tmpl_encoder, templates[t], device).to(device)), dim=-1).cpu()
            for t in all_techniques
        }
        zt_all = F.normalize(torch.stack([tmpl_embs[t] for t in all_techniques]), dim=-1)

    def score_text_and_wide(sequence_text, graph_path):
        with torch.no_grad():
            text_embed = log_proj(encode_one(tokenizer, seq_encoder, sequence_text, device).unsqueeze(0)).squeeze(0)
            raw = load_raw_vector(graph_path)
            h_cat = discretize_vector(raw, bins)
            wide_vec = hwide_pre_projection(h_cat, masks)
            wide_t = torch.tensor(wide_vec, dtype=torch.float32, device=device)
            wide_embed = wide_model(wide_t)
            combined = torch.cat([text_embed, wide_embed], dim=-1)
            combined = F.normalize(combined, dim=-1).cpu()
            logits = display_scale * (combined @ zt_all.T)
            probs = F.softmax(logits, dim=-1).tolist()
            ranked = sorted(zip(all_techniques, probs), key=lambda x: -x[1])
            return ranked

    technique_to_tactics = load_technique_to_tactics()
    technique_to_tactics_all = load_technique_to_tactics(restrict_to_scope=False)
    tactic_to_stage_list = {t: [s] for t, s in TACTIC_TO_STAGE.items()}

    official_host = {}
    for folder in sorted(os.listdir(os.path.join(CAM_LDS_DIR, 'sequences'))):
        tdir = os.path.join(CAM_LDS_DIR, 'sequences', folder)
        if not os.path.isdir(tdir):
            continue
        for tech_dir in os.listdir(tdir):
            for fname in glob_safe(os.path.join(tdir, tech_dir)):
                if fname.startswith('sequence_'):
                    rest = fname[len('sequence_'):-len('.json')]
                    step, _, host = rest.rpartition('_')
                    official_host.setdefault(step, host)

    scored_entries = test_entries
    if run_filter:
        before = len(scored_entries)
        scored_entries = [e for e in scored_entries if run_filter in e['file']]
        print('  Run filter "{}": kept {}/{} test steps'.format(run_filter, len(scored_entries), before))

    print()
    print('=' * 110)
    n_help = {'technique': 0, 'tactic': 0, 'stage': 0}
    for e in sorted(scored_entries, key=lambda e: e['file']):
        step = e['file']
        true_techs = set(e['techniques_all'])
        true_tactics = set()
        for t in true_techs:
            true_tactics.update(technique_to_tactics_all.get(t, []))
        true_stages = set(tactics_to_stages(true_tactics))

        hosts = find_hosts_for_step(step)
        if not hosts:
            continue
        print('\nStep {}  (true technique: {} | true tactic: {} | true stage: {})'.format(
            step, ', '.join(sorted(true_techs)), ', '.join(sorted(true_tactics)), ', '.join(sorted(true_stages))))
        tech_dir_guess, tactic_guess = guess_tactic_technique(step, TECHNIQUE_MERGE, base_technique)
        any_correct = {'technique': False, 'tactic': False, 'stage': False}
        for host, audit_path in hosts:
            try:
                seq_text, graph_path, n_sent = build_sequence_and_graph_file(
                    audit_path, tactic_guess, tech_dir_guess, step, host)
            except Exception as ex:
                print('  {:<14} ERROR building graph: {}'.format(host, ex))
                continue
            if n_sent == 0:
                print('  {:<14} (empty -- 0 usable edges after filtering)'.format(host))
                continue
            ranked = score_text_and_wide(seq_text, graph_path)
            tactic_ranked = expand_ranked(ranked, technique_to_tactics)
            stage_ranked = expand_ranked(tactic_ranked, tactic_to_stage_list)

            tech_top = {t for t, _ in ranked[:top_n]}
            tac_top = {t for t, _ in tactic_ranked[:top_n]}
            stg_top = {t for t, _ in stage_ranked[:top_n]}

            tech_ok = bool(true_techs & tech_top)
            tac_ok = bool(true_tactics & tac_top)
            stg_ok = bool(true_stages & stg_top)
            for level, ok in (('technique', tech_ok), ('tactic', tac_ok), ('stage', stg_ok)):
                if ok:
                    any_correct[level] = True

            tag = '<- OFFICIAL' if host == official_host.get(step) else ''
            flags = '{}  {}  {}'.format(
                'tech:OK ' if tech_ok else 'tech:no ',
                'tac:OK ' if tac_ok else 'tac:no ',
                'stg:OK' if stg_ok else 'stg:no')
            print('  {:<14} n_events={:<4} top3-tech=[{:<30}] {}  {}'.format(
                host, n_sent, ', '.join('{} {:.2f}'.format(t, s) for t, s in ranked[:top_n]), flags, tag))
        for level in ('technique', 'tactic', 'stage'):
            if any_correct[level]:
                n_help[level] += 1
        print('-' * 110)

    print('\nOut of {} test steps, at least one host gives a correct top-{} match:'.format(len(scored_entries), top_n))
    print('  Technique : {}/{}'.format(n_help['technique'], len(scored_entries)))
    print('  Tactic    : {}/{}'.format(n_help['tactic'], len(scored_entries)))
    print('  Stage     : {}/{}'.format(n_help['stage'], len(scored_entries)))


def glob_safe(d):
    try:
        return os.listdir(d)
    except OSError:
        return []


def guess_tactic_technique(step, technique_merge, base_technique_fn):
    """step_tactics.json gives techniques+tactics per step but not the folder
    naming to_subgraph() wants -- any real tactic/technique string works here
    since it's only used for human-readable description text, never for
    scoring (scoring uses the trained prototypes, not this label)."""
    with open(STEP_TACTICS_PATH) as f:
        report = json.load(f)
    for row in report:
        if row['step'] == step:
            occ = row['occurrences'][0]
            return occ['technique'], occ['tactics'][0].lower().replace(' ', '_')
    return 'Unknown', 'discovery'


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-tag', default='scenario4_basetech_redo')
    ap.add_argument('--test-scenario', default='4')
    ap.add_argument('--top-n', type=int, default=3)
    ap.add_argument('--run', type=str, default=None, dest='run_filter',
                     help='Only check/report test steps whose filename contains this substring, e.g. "ssh_apt".')
    args = ap.parse_args()
    main(run_tag=args.run_tag, test_scenario=args.test_scenario, top_n=args.top_n, run_filter=args.run_filter)
