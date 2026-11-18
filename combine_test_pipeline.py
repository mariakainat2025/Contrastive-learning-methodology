import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

CAM_LDS_DIR = os.path.dirname(os.path.abspath(__file__))
TECHNIQUE_SCRIPTS_DIR = os.path.join(CAM_LDS_DIR, 'technique', 'scripts')
if TECHNIQUE_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, TECHNIQUE_SCRIPTS_DIR)

import train_camlds_matcher_technique_wide as camlds_train
import test_camlds_matcher_technique_wide as camlds_test

SEQUENCES_RAW_DIR = os.path.join(CAM_LDS_DIR, 'sequences_raw')
SEQUENCES_RAWFULL_DIR = os.path.join(CAM_LDS_DIR, 'sequences_rawfull')
CHECKPOINTS_DIR = os.path.join(CAM_LDS_DIR, 'checkpoint_combinations')
TECHNIQUE_CHECKPOINTS_DIR = os.path.join(CAM_LDS_DIR, 'technique', 'checkpoints')
RESULTS_DIR = os.path.join(CAM_LDS_DIR, 'technique', 'results')
COMBINE_RESULTS_DIR = os.path.join(CAM_LDS_DIR, 'combination_results')

DEFAULT_TEST_FILE = '6_macro_binary'
DEFAULT_MAX_CHUNKS = 200

def header(title):
    print()
    print('=' * 70)
    print('  {}'.format(title))
    print('=' * 70)

COMBINATIONS = {}

def combination(template, graph, training):
    def register(fn):
        COMBINATIONS[(template, graph, training)] = fn
        return fn
    return register

def clean_up_checkpoints(run_tag):
    epoch_ckpts = glob.glob(os.path.join(TECHNIQUE_CHECKPOINTS_DIR, 'camlds_technique_wide_{}_epoch*.pt'.format(run_tag)))
    for p in epoch_ckpts:
        os.remove(p)
    if epoch_ckpts:
        print('  Removed {} periodic checkpoint(s) -- kept only the final one.'.format(len(epoch_ckpts)))

def save_combined_result(run_tag):
    os.makedirs(COMBINE_RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    ckpt_src = os.path.join(TECHNIQUE_CHECKPOINTS_DIR, 'camlds_technique_wide_matcher_{}.pt'.format(run_tag))
    if os.path.exists(ckpt_src):
        ckpt_dst = os.path.join(CHECKPOINTS_DIR, 'camlds_technique_wide_matcher_{}.pt'.format(run_tag))
        shutil.copy(ckpt_src, ckpt_dst)
        print('  Checkpoint    -> {}'.format(ckpt_dst))
    else:
        print('  WARNING: expected checkpoint file not found: {}'.format(ckpt_src))

    test_src = os.path.join(RESULTS_DIR, 'camlds_technique_wide_test_results_{}.json'.format(run_tag))
    if os.path.exists(test_src):
        test_dst = os.path.join(COMBINE_RESULTS_DIR, '{}.json'.format(run_tag))
        shutil.copy(test_src, test_dst)
        print('  Test results  -> {}'.format(test_dst))
    else:
        print('  WARNING: expected test results file not found: {}'.format(test_src))

    hist_src = os.path.join(RESULTS_DIR, 'train_history_camlds_technique_wide_{}.json'.format(run_tag))
    if os.path.exists(hist_src):
        hist_dst = os.path.join(COMBINE_RESULTS_DIR, '{}_train_history.json'.format(run_tag))
        shutil.copy(hist_src, hist_dst)
        print('  Train history -> {}'.format(hist_dst))
    else:
        print('  WARNING: expected train history file not found: {}'.format(hist_src))

@combination('mymodel', 'mymodel', 'mymodel')
def run_mymodel_mymodel_mymodel(args, run_tag):
    print('  template=mymodel : MITRE technique templates (templates_all92, default) -> RoBERTa, plain')
    print('  graph=mymodel    : RAW sequence text (sequences_raw/, not generalized) -> RoBERTa, plain, no wide features')
    print('  training=mymodel : your contrastive-prototype loss')

    camlds_train.MAX_CHUNKS = args.max_chunks if args.max_chunks > 0 else None
    seq_dir = graph_sequences_dir(args)
    if args.skip_train:
        print('  Skipping training -- reusing existing checkpoint for run_tag={}'.format(run_tag))
    else:
        header('Training')
        camlds_train.run_contrastive_train(
            test_file_match=args.test_file,
            run_tag=run_tag,
            n_epochs=args.n_epochs,
            sequences_dir=seq_dir,
            use_wide=args.use_wide,
            k_per_technique=camlds_train.K_PER_TACTIC,
            freeze_template_encoder=args.freeze_template_encoder,
        )

    header('Testing')
    camlds_test.run(
        test_file_match=args.test_file,
        run_tag=run_tag,
        sequences_dir=seq_dir,
    )

    clean_up_checkpoints(run_tag)
    save_combined_result(run_tag)

def _import_zoomer_losses():
    saved_data_utils = sys.modules.pop('data_utils', None)
    from Train_TTP_Recognition_Multilabel import (
        ttp_recognition_loss, masked_single_label_loss, N_EPISODES, support_and_query_size, K_SHOT,
        MAX_QUERY_PER_CLASS, LOG_EVERY, PATIENCE, MIN_DELTA, LR,
    )
    if saved_data_utils is not None:
        sys.modules['data_utils'] = saved_data_utils
    return (ttp_recognition_loss, masked_single_label_loss, N_EPISODES, support_and_query_size, K_SHOT,
            MAX_QUERY_PER_CLASS, LOG_EVERY, PATIENCE, MIN_DELTA, LR)

(zoomer_ttp_recognition_loss, zoomer_masked_single_label_loss, ZOOMER_N_EPISODES, zoomer_support_and_query_size,
 ZOOMER_K_SHOT, ZOOMER_MAX_QUERY_PER_CLASS, ZOOMER_LOG_EVERY, ZOOMER_PATIENCE, ZOOMER_MIN_DELTA,
 ZOOMER_LR) = _import_zoomer_losses()

def _import_zoomer_test_rollup():
    saved_data_utils = sys.modules.pop('data_utils', None)
    from data_utils import load_tactic_map, technique_tactics
    from Test_TTP_Recognition_Multilabel import (
        stage_scores_from_tactic_scores_max,
        print_technique_results_table, print_tactic_results_table, print_stage_results_table,
    )
    from stage_accuracy import (
        technique_top_k_accuracy, technique_accuracy_at_n_true,
        tactic_top_k_accuracy, tactic_accuracy_at_n_true,
        stage_top_k_accuracy, stage_accuracy_at_n_true,
    )
    from tactic_to_stage import tactics_to_stages
    if saved_data_utils is not None:
        sys.modules['data_utils'] = saved_data_utils
    return {
        'load_tactic_map': load_tactic_map, 'technique_tactics': technique_tactics,
        'stage_scores_from_tactic_scores_max': stage_scores_from_tactic_scores_max,
        'print_technique_results_table': print_technique_results_table,
        'print_tactic_results_table': print_tactic_results_table,
        'print_stage_results_table': print_stage_results_table,
        'technique_top_k_accuracy': technique_top_k_accuracy,
        'technique_accuracy_at_n_true': technique_accuracy_at_n_true,
        'tactic_top_k_accuracy': tactic_top_k_accuracy,
        'tactic_accuracy_at_n_true': tactic_accuracy_at_n_true,
        'stage_top_k_accuracy': stage_top_k_accuracy,
        'stage_accuracy_at_n_true': stage_accuracy_at_n_true,
        'tactics_to_stages': tactics_to_stages,
    }

ZOOMER_TEST = _import_zoomer_test_rollup()

def print_seen_vs_overall(rows, tactic_rows, stage_rows, train_tech_counts):
    seen_mask = [camlds_test.row_is_seen(r['true_techniques'], train_tech_counts) for r in rows]
    n_seen = sum(seen_mask)
    seen_rows = camlds_test.filter_rows_by_mask(rows, seen_mask)
    seen_tactic_rows = camlds_test.filter_rows_by_mask(tactic_rows, seen_mask)
    seen_stage_rows = camlds_test.filter_rows_by_mask(stage_rows, seen_mask)

    seen_tech = camlds_test.compute_metrics_block(
        seen_rows, 'true_techniques', ZOOMER_TEST['technique_top_k_accuracy'], ZOOMER_TEST['technique_accuracy_at_n_true'])
    seen_tac = camlds_test.compute_metrics_block(
        seen_tactic_rows, 'true_tactics', ZOOMER_TEST['tactic_top_k_accuracy'], ZOOMER_TEST['tactic_accuracy_at_n_true'])
    seen_stg = camlds_test.compute_metrics_block(
        seen_stage_rows, 'true_stages', ZOOMER_TEST['stage_top_k_accuracy'], ZOOMER_TEST['stage_accuracy_at_n_true'])
    overall_tech = camlds_test.compute_metrics_block(
        rows, 'true_techniques', ZOOMER_TEST['technique_top_k_accuracy'], ZOOMER_TEST['technique_accuracy_at_n_true'])
    overall_tac = camlds_test.compute_metrics_block(
        tactic_rows, 'true_tactics', ZOOMER_TEST['tactic_top_k_accuracy'], ZOOMER_TEST['tactic_accuracy_at_n_true'])
    overall_stg = camlds_test.compute_metrics_block(
        stage_rows, 'true_stages', ZOOMER_TEST['stage_top_k_accuracy'], ZOOMER_TEST['stage_accuracy_at_n_true'])

    print()
    print('  ── Seen (trained on every true label) vs Overall ──')
    print('  "Seen" = every true label for that step has >=1 training example (n={}/{} steps).'.format(
        n_seen, len(rows)))
    print()
    row_fmt = '  {:<22}{:>9}{:>9}{:>13}{:>13}'
    print(row_fmt.format('', 'top-1', 'at-n', 'top-3-any', 'top-3-all'))
    print('  Technique')
    print(row_fmt.format('    Seen ({})'.format(n_seen), *['{:.1f}%'.format(v * 100) for v in seen_tech]))
    print(row_fmt.format('    Overall ({})'.format(len(rows)), *['{:.1f}%'.format(v * 100) for v in overall_tech]))
    print('  Tactic')
    print(row_fmt.format('    Seen ({})'.format(n_seen), *['{:.1f}%'.format(v * 100) for v in seen_tac]))
    print(row_fmt.format('    Overall ({})'.format(len(rows)), *['{:.1f}%'.format(v * 100) for v in overall_tac]))
    print('  Stage')
    print(row_fmt.format('    Seen ({})'.format(n_seen), *['{:.1f}%'.format(v * 100) for v in seen_stg]))
    print(row_fmt.format('    Overall ({})'.format(len(rows)), *['{:.1f}%'.format(v * 100) for v in overall_stg]))

    all_techniques_u = sorted({t for r in rows for t in r['true_techniques']} | {t for r in rows for (t, _) in r['ranked']})
    all_tactics_u = sorted({t for r in tactic_rows for t in r['true_tactics']} | {t for r in tactic_rows for (t, _) in r['ranked']})
    all_stages_u = sorted({t for r in stage_rows for t in r['true_stages']} | {t for r in stage_rows for (t, _) in r['ranked']})
    seen_lrap_tech = camlds_test.compute_lrap_generic(seen_rows, 'true_techniques', all_techniques_u)
    seen_aupr_tech = camlds_test.compute_aupr_generic(seen_rows, 'true_techniques', all_techniques_u)
    overall_lrap_tech = camlds_test.compute_lrap_generic(rows, 'true_techniques', all_techniques_u)
    overall_aupr_tech = camlds_test.compute_aupr_generic(rows, 'true_techniques', all_techniques_u)
    seen_lrap_tac = camlds_test.compute_lrap_generic(seen_tactic_rows, 'true_tactics', all_tactics_u)
    seen_aupr_tac = camlds_test.compute_aupr_generic(seen_tactic_rows, 'true_tactics', all_tactics_u)
    overall_lrap_tac = camlds_test.compute_lrap_generic(tactic_rows, 'true_tactics', all_tactics_u)
    overall_aupr_tac = camlds_test.compute_aupr_generic(tactic_rows, 'true_tactics', all_tactics_u)
    seen_lrap_stg = camlds_test.compute_lrap_generic(seen_stage_rows, 'true_stages', all_stages_u)
    seen_aupr_stg = camlds_test.compute_aupr_generic(seen_stage_rows, 'true_stages', all_stages_u)
    overall_lrap_stg = camlds_test.compute_lrap_generic(stage_rows, 'true_stages', all_stages_u)
    overall_aupr_stg = camlds_test.compute_aupr_generic(stage_rows, 'true_stages', all_stages_u)

    print()
    print('  ── LRAP / AUPR: Seen vs Overall ──')
    lrap_fmt = '  {:<22}{:>9}{:>9}'
    print(lrap_fmt.format('', 'LRAP', 'AUPR'))
    print('  Technique')
    print(lrap_fmt.format('    Seen ({})'.format(n_seen), '{:.1f}%'.format(seen_lrap_tech * 100), '{:.1f}%'.format(seen_aupr_tech * 100)))
    print(lrap_fmt.format('    Overall ({})'.format(len(rows)), '{:.1f}%'.format(overall_lrap_tech * 100), '{:.1f}%'.format(overall_aupr_tech * 100)))
    print('  Tactic')
    print(lrap_fmt.format('    Seen ({})'.format(n_seen), '{:.1f}%'.format(seen_lrap_tac * 100), '{:.1f}%'.format(seen_aupr_tac * 100)))
    print(lrap_fmt.format('    Overall ({})'.format(len(rows)), '{:.1f}%'.format(overall_lrap_tac * 100), '{:.1f}%'.format(overall_aupr_tac * 100)))
    print('  Stage')
    print(lrap_fmt.format('    Seen ({})'.format(n_seen), '{:.1f}%'.format(seen_lrap_stg * 100), '{:.1f}%'.format(seen_aupr_stg * 100)))
    print(lrap_fmt.format('    Overall ({})'.format(len(rows)), '{:.1f}%'.format(overall_lrap_stg * 100), '{:.1f}%'.format(overall_aupr_stg * 100)))

    return {
        'n_seen': n_seen,
        'seen_vs_overall': {
            'technique': {'seen': list(seen_tech), 'overall': list(overall_tech)},
            'tactic': {'seen': list(seen_tac), 'overall': list(overall_tac)},
            'stage': {'seen': list(seen_stg), 'overall': list(overall_stg)},
        },
        'lrap_aupr_seen_vs_overall': {
            'technique': {'seen': [seen_lrap_tech, seen_aupr_tech], 'overall': [overall_lrap_tech, overall_aupr_tech]},
            'tactic': {'seen': [seen_lrap_tac, seen_aupr_tac], 'overall': [overall_lrap_tac, overall_aupr_tac]},
            'stage': {'seen': [seen_lrap_stg, seen_aupr_stg], 'overall': [overall_lrap_stg, overall_aupr_stg]},
        },
    }

def print_accuracy_summary(tech_acc, tech_at_n, tech_lrap, tech_aupr,
                            tac_acc, tac_at_n, tac_lrap, tac_aupr,
                            stage_acc, stage_at_n, stage_lrap, stage_aupr):
    print()
    print('  Technique  top-1: {:.1f}%  top-3: {:.1f}%  at-n: {:.1f}%  LRAP: {:.1f}%  AUPR: {:.1f}%'.format(
        tech_acc[1] * 100, tech_acc[3] * 100, tech_at_n * 100, tech_lrap * 100, tech_aupr * 100))
    print('  Tactic     top-1: {:.1f}%  top-3: {:.1f}%  at-n: {:.1f}%  LRAP: {:.1f}%  AUPR: {:.1f}%'.format(
        tac_acc[1] * 100, tac_acc[3] * 100, tac_at_n * 100, tac_lrap * 100, tac_aupr * 100))
    print('  Stage      top-1: {:.1f}%  top-3: {:.1f}%  at-n: {:.1f}%  LRAP: {:.1f}%  AUPR: {:.1f}%'.format(
        stage_acc[1] * 100, stage_acc[3] * 100, stage_at_n * 100, stage_lrap * 100, stage_aupr * 100))

def save_checkpoint(ckpt_path, modules, extra=None):
    os.makedirs(COMBINE_RESULTS_DIR, exist_ok=True)
    state = {name: m.state_dict() for name, m in modules.items()}
    if extra:
        state.update(extra)
    torch.save(state, ckpt_path)
    print('  Checkpoint -> {}  (final only -- no periodic snapshots were ever written)'.format(ckpt_path))

def load_checkpoint(ckpt_path, modules, logit_scale=None):
    if not os.path.exists(ckpt_path):
        print('  ERROR: --skip-train requested but checkpoint not found: {}'.format(ckpt_path))
        sys.exit(1)
    state = torch.load(ckpt_path, map_location='cpu')
    for name, m in modules.items():
        m.load_state_dict(state[name])
    if logit_scale is not None:
        with torch.no_grad():
            logit_scale.copy_(state['logit_scale'])
    print('  Skipped training -- loaded checkpoint from {}  (best_loss={})'.format(ckpt_path, state.get('best_loss')))
    return state.get('history', []), state.get('best_loss')

def save_train_history(run_tag, history):
    hist_path = os.path.join(COMBINE_RESULTS_DIR, '{}_train_history.json'.format(run_tag))
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print('  Train history -> {}'.format(hist_path))
    return hist_path

def save_results_json(run_tag, out):
    results_path = os.path.join(COMBINE_RESULTS_DIR, '{}.json'.format(run_tag))
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('  Test results -> {}'.format(results_path))
    return results_path

def graph_sequences_dir(args):
    override = getattr(args, 'sequences_dir_override', None)
    if override:
        if not os.path.isdir(override):
            raise SystemExit(
                'FATAL: --sequences-dir-override {!r} is not a directory -- refusing to silently '
                'fall back to raw/generalized sequences.'.format(override))
        return override
    return SEQUENCES_RAWFULL_DIR if args.graph_sequences == 'raw' else None

def load_entries_and_templates(args, sequences_dir=None):
    entries = camlds_train.load_sequences(sequences_dir=sequences_dir)
    all_techniques = sorted({t for e in entries for t in e['techniques']})
    technique_to_col = {t: i for i, t in enumerate(all_techniques)}
    print('  Techniques: {}'.format(len(all_techniques)))

    train_entries, test_entries = camlds_train.leave_out_split(entries, args.test_file)
    train_entries = [e for e in train_entries if e['techniques']]

    exclude = getattr(args, 'exclude_from_training', None)
    if exclude:
        exclude_set = set(s.strip() for s in exclude.split(',') if s.strip())
        before = len(train_entries)
        train_entries = [e for e in train_entries if e['file'] not in exclude_set]
        removed = before - len(train_entries)
        print('  --exclude-from-training: removed {} of {} requested step(s) from training ({} requested, '
              '{} not found/already outside training): {}'.format(
                  removed, len(exclude_set), len(exclude_set), len(exclude_set) - removed, sorted(exclude_set)))

    print('  Train sequences: {}  Test sequences: {}'.format(len(train_entries), len(test_entries)))

    templates = camlds_train.load_templates()
    print('  Templates: {} (dir={})'.format(len(templates), camlds_train.TEMPLATE_DIR))

    return all_techniques, technique_to_col, train_entries, test_entries, templates

def setup_roberta_encoder(device, out_dim):
    encoder = RobertaModel.from_pretrained(camlds_train.ROBERTA_MODEL).to(device)
    camlds_train.freeze_lower_layers(encoder, n_freeze=camlds_train.N_FREEZE)
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    encoder.train()
    proj = camlds_train.ProjectionNetwork(camlds_train.EMB_DIM, out_dim, camlds_train.DROPOUT).to(device)
    return encoder, proj

def make_knowhow_vectorizer(fasttext_model, dim=128):
    cache = {}
    def vectorize(key, text):
        if key not in cache:
            words = text.split()
            vectors = [fasttext_model.wv[w] for w in words if w in fasttext_model.wv]
            if vectors:
                avg = np.mean(vectors, axis=0).astype('float32')
            else:
                avg = np.zeros(dim, dtype='float32')
            cache[key] = torch.from_numpy(avg)
        return cache[key]
    return vectorize

def stratified_batch_indices_capped(technique_pools, all_techniques, k_per_technique, rng):
    chosen = set()
    for t in all_techniques:
        pool = technique_pools[t]
        if not pool:
            continue
        take = min(k_per_technique, len(pool))
        chosen.update(rng.sample(pool, take))
    return sorted(chosen)

def run_training_loop(args, label, modules, trainable_params, technique_pools, train_entries,
                       all_techniques, compute_loss_fn, use_amp, logit_scale=None):
    optimizer = torch.optim.Adam(trainable_params, lr=camlds_train.LR)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    batch_size = min(len(train_entries), camlds_train.K_PER_TACTIC * len(all_techniques))
    steps_per_epoch = max(1, -(-len(train_entries) // batch_size))

    best_loss = float('inf')
    no_improve = 0
    best_state = None
    history = []
    train_start = time.time()

    for epoch in range(1, args.n_epochs + 1):
        epoch_start = time.time()
        for m in modules.values():
            m.train()
        epoch_losses = []

        for _ in range(steps_per_epoch):
            batch_idxs = stratified_batch_indices_capped(
                technique_pools, all_techniques, camlds_train.K_PER_TACTIC, random)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = compute_loss_fn(batch_idxs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = compute_loss_fn(batch_idxs)
                loss.backward()
                optimizer.step()
            if logit_scale is not None:
                with torch.no_grad():
                    logit_scale.clamp_(0, math.log(100))
            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = (time.time() - epoch_start) / 60
        history.append({'epoch': epoch, 'loss': round(epoch_loss, 6), 'n_steps': len(epoch_losses)})

        if epoch % 5 == 0 or epoch == 1:
            print('  {0} {1:>4d}  loss={2:.6f}  best_loss={3:.6f}  {0}_time={4:.2f}m'.format(
                label, epoch, epoch_loss, best_loss, epoch_time))

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            no_improve = 0
            best_state = {name: {k: v.clone() for k, v in m.state_dict().items()} for name, m in modules.items()}
            if logit_scale is not None:
                best_state['logit_scale'] = logit_scale.detach().clone()
        else:
            no_improve += 1
            if no_improve >= camlds_train.PATIENCE:
                print('  Early stopping at {0} {1} (no improvement for {2} {0}s)'.format(
                    label, epoch, camlds_train.PATIENCE))
                break

    if best_state:
        for name, m in modules.items():
            m.load_state_dict(best_state[name])
        if logit_scale is not None:
            with torch.no_grad():
                logit_scale.copy_(best_state['logit_scale'])

    total_time = (time.time() - train_start) / 60
    print('  Best train loss: {:.6f}  Total time: {:.2f}m'.format(best_loss, total_time))
    return history, best_loss

def build_episodic_batch(technique_pools, all_techniques, rng):
    support_idxs_by_class = []
    used_techniques = []
    query_items = []
    for t in all_techniques:
        pool = list(technique_pools[t])
        if not pool:
            continue
        rng.shuffle(pool)
        support_size, query_size = zoomer_support_and_query_size(len(pool))
        support = pool[:support_size]
        remaining = pool[support_size:support_size + query_size]
        class_idx = len(used_techniques)
        used_techniques.append(t)
        support_idxs_by_class.append(support)
        for i in remaining:
            query_items.append((i, class_idx))
    return support_idxs_by_class, used_techniques, query_items

def run_episodic_training(args, embed_entry_fn, technique_pools, train_entries, all_techniques, modules,
                           trainable_params, use_amp=False, loss_style='zoomer', logit_scale=None):
    """Mirrors ZOOMER's own episodic training (run_episode: prototype = average of a few real support
    examples), but with a pluggable embed_entry_fn, a pluggable loss ('zoomer' = their own squared-distance
    cross-entropy, 'mymodel' = your own multilabel_prototype_loss + uniformity_loss), and ONE batched
    optimizer step per episode (summing every query's loss term) instead of ZOOMER's literal
    one-optimizer-step-per-query-label-occurrence -- that part of their loop assumes a cheap GNN forward
    pass; doing it literally with a RoBERTa encoder would mean hundreds of thousands of sequential
    forward+backward passes and be impractically slow."""
    optimizer = torch.optim.Adam(trainable_params, lr=ZOOMER_LR if loss_style == 'zoomer' else camlds_train.LR)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    best_loss = float('inf')
    best_state = None
    stale_windows = 0
    recent_losses = []
    train_start = time.time()
    history = []

    def compute_episode_loss():
        support_idxs_by_class, used_techniques, query_items = build_episodic_batch(
            technique_pools, all_techniques, random)
        prototypes = torch.stack([
            torch.stack([embed_entry_fn(train_entries[i]) for i in support]).mean(dim=0)
            for support in support_idxs_by_class
        ])

        by_entry = {}
        for (i, class_idx) in query_items:
            by_entry.setdefault(i, []).append(class_idx)
        query_embeds_cache = {i: embed_entry_fn(train_entries[i]) for i in by_entry}

        if loss_style == 'mymodel':
            entry_idxs = list(by_entry.keys())
            z = torch.stack([query_embeds_cache[i] for i in entry_idxs])
            pos_mask = torch.zeros(len(entry_idxs), len(used_techniques), device=prototypes.device)
            for row, i in enumerate(entry_idxs):
                for t in train_entries[i]['techniques']:
                    if t in used_techniques:
                        pos_mask[row, used_techniques.index(t)] = 1
            main_loss = camlds_train.multilabel_prototype_loss(z, prototypes, pos_mask, logit_scale)
            return main_loss + camlds_train.LAMBDA_UNIFORM * camlds_train.uniformity_loss(z)

        losses = []
        single_idxs = [i for i, cidxs in by_entry.items() if len(cidxs) == 1]
        if single_idxs:
            query_embeds = torch.stack([query_embeds_cache[i] for i in single_idxs])
            query_labels = torch.tensor([by_entry[i][0] for i in single_idxs], device=prototypes.device)
            losses.append(zoomer_ttp_recognition_loss(prototypes, query_embeds, query_labels))
        for i, cidxs in by_entry.items():
            if len(cidxs) <= 1:
                continue
            for true_idx in cidxs:
                exempt = set(cidxs) - {true_idx}
                losses.append(zoomer_masked_single_label_loss(query_embeds_cache[i], prototypes, true_idx, exempt))
        return torch.stack(losses).mean()

    log_every = min(ZOOMER_LOG_EVERY, max(1, args.n_epochs // 10))

    for episode in range(1, args.n_epochs + 1):
        for m in modules.values():
            m.train()

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                loss = compute_episode_loss()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = compute_episode_loss()
            loss.backward()
            optimizer.step()
        if logit_scale is not None:
            with torch.no_grad():
                logit_scale.clamp_(0, math.log(100))
        episode_loss = loss.item()
        recent_losses.append(episode_loss)

        if episode % log_every == 0 or episode == 1:
            avg_loss = sum(recent_losses) / len(recent_losses)
            history.append({'epoch': episode, 'loss': round(avg_loss, 6), 'n_steps': len(recent_losses)})
            recent_losses = []
            if avg_loss < best_loss - ZOOMER_MIN_DELTA:
                best_loss = avg_loss
                stale_windows = 0
                best_state = {name: {k: v.clone() for k, v in m.state_dict().items()} for name, m in modules.items()}
                if logit_scale is not None:
                    best_state['logit_scale'] = logit_scale.detach().clone()
            else:
                stale_windows += 1
            print('  episode {:5d}  loss {:.4f}  best_loss {:.4f}'.format(episode, avg_loss, best_loss))
            if stale_windows >= ZOOMER_PATIENCE:
                print("  loss hasn't improved for {} checks (best={:.4f}) -- stopping early at episode {}.".format(
                    ZOOMER_PATIENCE, best_loss, episode))
                break

    if best_state:
        for name, m in modules.items():
            m.load_state_dict(best_state[name])
        if logit_scale is not None:
            with torch.no_grad():
                logit_scale.copy_(best_state['logit_scale'])

    total_time = (time.time() - train_start) / 60
    print('  Best train loss: {:.6f}  Total time: {:.2f}m'.format(best_loss, total_time))
    return history, best_loss

def build_final_episodic_prototypes(embed_entry_fn, technique_pools, train_entries, all_techniques):
    prototypes = []
    used_techniques = []
    for t in all_techniques:
        pool = technique_pools[t]
        if not pool:
            continue
        embeds = torch.stack([embed_entry_fn(train_entries[i]) for i in pool])
        prototypes.append(embeds.mean(dim=0))
        used_techniques.append(t)
    return torch.stack(prototypes), used_techniques

def mymodel_style_build_results(test_entries, all_techniques, embed_entry_fn, zt_all, display_scale):
    results = []
    skipped = 0
    for e in test_entries:
        z = embed_entry_fn(e)
        if z is None:
            skipped += 1
            continue
        z_s = F.normalize(z, dim=-1)
        logits = display_scale * (z_s @ zt_all.T)
        probs = [round(p, 6) for p in F.softmax(logits, dim=-1).tolist()]
        scores = dict(zip(all_techniques, probs))
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results.append({
            'file': e['file'], 'true_techniques': e['techniques_all'], 'scores': scores,
            'ranked': [{'technique': t, 'score': s} for t, s in ranked], 'n_events': e.get('n_events'),
        })
    if skipped:
        print('  WARNING: {} test steps had no graph file, skipped.'.format(skipped))
    return results

def zoomer_style_build_results(test_entries, all_techniques, embed_entry_fn, prototypes):
    results = []
    skipped = 0
    for e in test_entries:
        z = embed_entry_fn(e)
        if z is None:
            skipped += 1
            continue
        dists = (torch.cdist(z.unsqueeze(0), prototypes).squeeze(0) ** 2)
        scores = (-dists).tolist()
        ranked_idx = torch.argsort(dists).tolist()
        ranked = [{'technique': all_techniques[i], 'score': scores[i]} for i in ranked_idx]
        results.append({
            'file': e['file'], 'true_techniques': e['techniques_all'],
            'scores': dict(zip(all_techniques, scores)), 'ranked': ranked,
            'n_events': e.get('n_events'),
        })
    if skipped:
        print('  WARNING: {} test steps had no graph file, skipped.'.format(skipped))
    return results

def mymodel_style_test_rollup(out, results, all_techniques, train_entries):
    technique_to_tactics = camlds_test.load_technique_to_tactics()
    technique_to_tactics_all = camlds_test.load_technique_to_tactics(restrict_to_scope=False)
    technique_rows, tactic_rows, stage_rows = camlds_test.build_tactic_and_stage_rows(
        results, technique_to_tactics, technique_to_tactics_all)
    files = [r['file'] for r in results]

    all_reachable_tactics = sorted({tac for tacs in technique_to_tactics.values() for tac in tacs})
    all_reachable_stages = camlds_test.tactics_to_stages(all_reachable_tactics)
    camlds_test.print_tactic_results_table(tactic_rows, files, len(all_reachable_tactics))
    camlds_test.print_stage_results_table(stage_rows, files)

    tech_acc = camlds_test.technique_top_k_accuracy(technique_rows)
    tech_at_n = camlds_test.technique_accuracy_at_n_true(technique_rows)
    tac_acc = camlds_test.tactic_top_k_accuracy(tactic_rows)
    tac_at_n = camlds_test.tactic_accuracy_at_n_true(tactic_rows)
    stage_acc = camlds_test.stage_top_k_accuracy(stage_rows)
    stage_at_n = camlds_test.stage_accuracy_at_n_true(stage_rows)
    tac_lrap = camlds_test.compute_lrap_generic(tactic_rows, 'true_tactics', all_reachable_tactics)
    tac_aupr = camlds_test.compute_aupr_generic(tactic_rows, 'true_tactics', all_reachable_tactics)
    stage_lrap = camlds_test.compute_lrap_generic(stage_rows, 'true_stages', all_reachable_stages)
    stage_aupr = camlds_test.compute_aupr_generic(stage_rows, 'true_stages', all_reachable_stages)

    print_accuracy_summary(tech_acc, tech_at_n, out['lrap'], out['aupr'],
                            tac_acc, tac_at_n, tac_lrap, tac_aupr,
                            stage_acc, stage_at_n, stage_lrap, stage_aupr)

    train_tech_counts = {}
    for e in train_entries:
        for t in e['techniques']:
            train_tech_counts[t] = train_tech_counts.get(t, 0) + 1
    seen_overall = print_seen_vs_overall(technique_rows, tactic_rows, stage_rows, train_tech_counts)

    out.update({
        'tech_acc_top1': tech_acc[1], 'tech_acc_top3': tech_acc[3], 'tech_acc_at_n': tech_at_n,
        'tac_acc_top1': tac_acc[1], 'tac_acc_top3': tac_acc[3], 'tac_acc_at_n': tac_at_n,
        'tac_lrap': tac_lrap, 'tac_aupr': tac_aupr,
        'stage_acc_top1': stage_acc[1], 'stage_acc_top3': stage_acc[3], 'stage_acc_at_n': stage_at_n,
        'stage_lrap': stage_lrap, 'stage_aupr': stage_aupr,
        'techniques': all_techniques,
    })
    out.update(seen_overall)
    return out

def zoomer_style_test_rollup(results, all_techniques, train_entries):
    tactic_map = ZOOMER_TEST['load_tactic_map']()
    technique_tactics_fn = ZOOMER_TEST['technique_tactics']
    class_tactics_all = {t: technique_tactics_fn(tactic_map, t) for t in all_techniques}
    all_tactics = sorted({tac for tacs in class_tactics_all.values() for tac in tacs})

    rows, tactic_rows, stage_rows = [], [], []
    for r in results:
        true_techniques = set(r['true_techniques'])
        ranked_tech = [(d['technique'], d['score']) for d in r['ranked']]
        rows.append({'file': r['file'], 'true_techniques': true_techniques, 'ranked': ranked_tech})

        true_tactics = set()
        for t in true_techniques:
            true_tactics |= technique_tactics_fn(tactic_map, t)
        tactic_scores = {t: -1e9 for t in all_tactics}
        for (c, s) in ranked_tech:
            for t in class_tactics_all.get(c, set()):
                if s > tactic_scores[t]:
                    tactic_scores[t] = s
        tactic_ranked = sorted(tactic_scores.items(), key=lambda x: x[1], reverse=True)
        tactic_rows.append({'file': r['file'], 'true_tactics': true_tactics, 'ranked': tactic_ranked})

        stage_scores = ZOOMER_TEST['stage_scores_from_tactic_scores_max'](tactic_scores)
        stage_ranked = sorted(stage_scores.items(), key=lambda x: x[1], reverse=True)
        true_stages = set(ZOOMER_TEST['tactics_to_stages'](true_tactics))
        stage_rows.append({'file': r['file'], 'true_stages': true_stages, 'ranked': stage_ranked})

    ZOOMER_TEST['print_technique_results_table'](rows, len(all_techniques))
    ZOOMER_TEST['print_tactic_results_table'](tactic_rows, len(all_tactics))
    ZOOMER_TEST['print_stage_results_table'](stage_rows)

    tech_acc = ZOOMER_TEST['technique_top_k_accuracy'](rows)
    tech_at_n = ZOOMER_TEST['technique_accuracy_at_n_true'](rows)
    tac_acc = ZOOMER_TEST['tactic_top_k_accuracy'](tactic_rows)
    tac_at_n = ZOOMER_TEST['tactic_accuracy_at_n_true'](tactic_rows)
    stage_acc = ZOOMER_TEST['stage_top_k_accuracy'](stage_rows)
    stage_at_n = ZOOMER_TEST['stage_accuracy_at_n_true'](stage_rows)

    all_stages = sorted({s for row in stage_rows for s in row['true_stages']} |
                         {s for row in stage_rows for (s, _) in row['ranked']})
    tech_lrap = camlds_test.compute_lrap_generic(rows, 'true_techniques', all_techniques)
    tech_aupr = camlds_test.compute_aupr_generic(rows, 'true_techniques', all_techniques)
    tac_lrap = camlds_test.compute_lrap_generic(tactic_rows, 'true_tactics', all_tactics)
    tac_aupr = camlds_test.compute_aupr_generic(tactic_rows, 'true_tactics', all_tactics)
    stage_lrap = camlds_test.compute_lrap_generic(stage_rows, 'true_stages', all_stages)
    stage_aupr = camlds_test.compute_aupr_generic(stage_rows, 'true_stages', all_stages)

    print_accuracy_summary(tech_acc, tech_at_n, tech_lrap, tech_aupr,
                            tac_acc, tac_at_n, tac_lrap, tac_aupr,
                            stage_acc, stage_at_n, stage_lrap, stage_aupr)

    train_tech_counts = {}
    for e in train_entries:
        for t in e['techniques']:
            train_tech_counts[t] = train_tech_counts.get(t, 0) + 1
    seen_overall = print_seen_vs_overall(rows, tactic_rows, stage_rows, train_tech_counts)

    out = {
        'tech_acc_top1': tech_acc[1], 'tech_acc_top3': tech_acc[3], 'tech_acc_at_n': tech_at_n,
        'tech_lrap': tech_lrap, 'tech_aupr': tech_aupr,
        'tac_acc_top1': tac_acc[1], 'tac_acc_top3': tac_acc[3], 'tac_acc_at_n': tac_at_n,
        'tac_lrap': tac_lrap, 'tac_aupr': tac_aupr,
        'stage_acc_top1': stage_acc[1], 'stage_acc_top3': stage_acc[3], 'stage_acc_at_n': stage_at_n,
        'stage_lrap': stage_lrap, 'stage_aupr': stage_aupr,
        'techniques': all_techniques,
    }
    out.update(seen_overall)
    return out

@combination('mymodel', 'mymodel', 'zoomer')
def run_mymodel_mymodel_zoomer(args, run_tag):
    print('  template=mymodel : MITRE technique templates (templates_all92, default) -> RoBERTa, plain')
    print('  graph=mymodel    : RAW sequence text (sequences_raw/, not generalized) -> RoBERTa, plain, no wide features')
    print('  training=zoomer  : prototype = template embedding ONLY (no example-averaging), '
          "ZOOMER's squared-distance cross-entropy loss")

    device = torch.device('cuda')
    camlds_train.MAX_CHUNKS = args.max_chunks if args.max_chunks > 0 else None

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(
        args, sequences_dir=graph_sequences_dir(args))

    if args.use_wide:
        step_to_path = camlds_train.build_step_to_graph_path()
        train_steps = [e['file'] for e in train_entries if e['techniques']]
        bins, masks, wide_dim = camlds_train.fit_wide_bins_and_masks(train_steps, step_to_path, seed=camlds_train.SEED)
        print('  ZOOMER wide bins fit on {} training graphs, wide_dim={}'.format(len(train_steps), wide_dim))
        wide_model = camlds_train.WideModel(wide_dim, camlds_train.WIDE_DIM_OUT).to(device)
        template_out_dim = camlds_train.COMBINED_DIM
    else:
        step_to_path, bins, masks, wide_model = None, None, None, None
        template_out_dim = camlds_train.PROJ_DIM

    tokenizer = RobertaTokenizer.from_pretrained(camlds_train.ROBERTA_MODEL)
    seq_encoder, log_proj = setup_roberta_encoder(device, camlds_train.PROJ_DIM)
    tmpl_encoder, text_proj = setup_roberta_encoder(device, template_out_dim)
    print('  log_proj out_dim={}  text_proj out_dim={}  (must match -- they do)'.format(
        template_out_dim, template_out_dim))

    trainable_params = (
        [p for p in seq_encoder.parameters() if p.requires_grad] + list(log_proj.parameters()) +
        [p for p in tmpl_encoder.parameters() if p.requires_grad] + list(text_proj.parameters())
    )
    if wide_model is not None:
        trainable_params += list(wide_model.parameters())

    def embed_sequence(entry):
        return camlds_train.encode_sequence_combined(
            tokenizer, seq_encoder, log_proj, wide_model, entry, step_to_path, bins, masks, device)

    def embed_prototypes():
        return text_proj(torch.stack(
            [camlds_train.encode_one(tokenizer, tmpl_encoder, templates[t], device) for t in all_techniques]))

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        prototypes = embed_prototypes()
        step_losses = []
        for e in batch_entries:
            embed = embed_sequence(e)
            true_idxs = [technique_to_col[t] for t in e['techniques']]
            exempt_all = set(true_idxs)
            for true_idx in true_idxs:
                step_losses.append(zoomer_masked_single_label_loss(
                    embed, prototypes, true_idx, exempt_all - {true_idx}))
        return torch.stack(step_losses).mean()

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'seq_encoder': seq_encoder, 'tmpl_encoder': tmpl_encoder, 'log_proj': log_proj, 'text_proj': text_proj}
    if wide_model is not None:
        modules['wide_model'] = wide_model
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'camlds_camlds_zoomer_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules)
    else:
        history, best_loss = run_training_loop(
            args, 'episode', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=True)
        save_checkpoint(ckpt_path, modules, extra={
            'techniques': all_techniques, 'best_loss': best_loss, 'history': history,
            'run_tag': run_tag, 'test_file': args.test_file,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    with torch.no_grad():
        prototypes = embed_prototypes()
        results = zoomer_style_build_results(test_entries, all_techniques, embed_sequence, prototypes)

    out = {'run_tag': run_tag, 'test_file': args.test_file, 'n_total': len(results)}
    out.update(zoomer_style_test_rollup(results, all_techniques, train_entries))
    out['results'] = results
    save_results_json(run_tag, out)

def _import_zoomer_graph_encoder():
    saved_data_utils = sys.modules.pop('data_utils', None)
    from Deep_Wide_Model import TSGModel
    from data_utils import GraphTensorCache
    from Train_TTP_Recognition_Multilabel import IN_DIM
    if saved_data_utils is not None:
        sys.modules['data_utils'] = saved_data_utils
    return TSGModel, GraphTensorCache, IN_DIM

ZoomerTSGModel, ZoomerGraphTensorCache, ZOOMER_IN_DIM = _import_zoomer_graph_encoder()

def setup_zoomer_graph_encoder(train_entries, device):
    step_to_path = camlds_train.build_step_to_graph_path()
    train_steps = [e['file'] for e in train_entries if e['techniques']]
    bins, masks, wide_dim = camlds_train.fit_wide_bins_and_masks(train_steps, step_to_path, seed=camlds_train.SEED)
    print('  ZOOMER wide bins fit on {} training graphs, wide_dim={}'.format(len(train_steps), wide_dim))

    cache = ZoomerGraphTensorCache(bins, masks, wide_features='all')
    graph_model = ZoomerTSGModel(deep_in_dim=ZOOMER_IN_DIM, wide_in_dim=wide_dim).to(device)

    def embed_graph(path):
        (h, adjacency, wide_x) = cache.get(path)
        return graph_model(h.to(device), adjacency.to(device), wide_x.to(device))

    return step_to_path, graph_model, embed_graph, wide_dim

@combination('mymodel', 'zoomer', 'mymodel')
def run_mymodel_zoomer_mymodel(args, run_tag):
    print('  template=mymodel : MITRE technique templates (templates_all92, default) -> RoBERTa, plain (192-dim)')
    print('  graph=zoomer     : real provenance graph -> ZOOMER\'s TSGModel (GAT deep + wide linear, 192-dim)')
    print('  training=mymodel : your contrastive-prototype loss (multilabel_prototype_loss + uniformity_loss)')

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(
        args, sequences_dir=SEQUENCES_RAWFULL_DIR)

    step_to_path, graph_model, embed_graph, wide_dim = setup_zoomer_graph_encoder(train_entries, device)

    tmpl_encoder, text_proj = setup_roberta_encoder(device, camlds_train.COMBINED_DIM)
    tokenizer = RobertaTokenizer.from_pretrained(camlds_train.ROBERTA_MODEL)
    print('  graph_model out_dim={}  text_proj out_dim={}  (must match -- they do)'.format(
        camlds_train.COMBINED_DIM, camlds_train.COMBINED_DIM))

    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    trainable_params = (
        list(graph_model.parameters()) +
        [p for p in tmpl_encoder.parameters() if p.requires_grad] + list(text_proj.parameters()) +
        [logit_scale]
    )

    def embed_prototypes():
        return text_proj(torch.stack(
            [camlds_train.encode_one(tokenizer, tmpl_encoder, templates[t], device) for t in all_techniques]))

    train_pos_mask = camlds_train.build_pos_mask(train_entries, all_techniques, technique_to_col).to(device)

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        pos_mask_batch = train_pos_mask[batch_idxs]
        z = torch.stack([embed_graph(step_to_path[e['file']]) for e in batch_entries])
        prototypes = embed_prototypes()
        main_loss = camlds_train.multilabel_prototype_loss(z, prototypes, pos_mask_batch, logit_scale)
        return main_loss + camlds_train.LAMBDA_UNIFORM * camlds_train.uniformity_loss(z)

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'graph_model': graph_model, 'tmpl_encoder': tmpl_encoder, 'text_proj': text_proj}
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'camlds_zoomer_camlds_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules, logit_scale=logit_scale)
    else:
        history, best_loss = run_training_loop(
            args, 'epoch', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=False, logit_scale=logit_scale)
        save_checkpoint(ckpt_path, modules, extra={
            'logit_scale': logit_scale, 'techniques': all_techniques, 'best_loss': best_loss,
            'history': history, 'run_tag': run_tag, 'test_file': args.test_file, 'wide_dim': wide_dim,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    display_scale = logit_scale.exp().item()
    with torch.no_grad():
        zt_all = F.normalize(embed_prototypes(), dim=-1)

        def embed_entry(e):
            path = step_to_path.get(e['file'])
            return None if path is None else embed_graph(path)

        results = mymodel_style_build_results(test_entries, all_techniques, embed_entry, zt_all, display_scale)

    out = camlds_test.print_table(results, all_techniques)
    out = mymodel_style_test_rollup(out, results, all_techniques, train_entries)
    out.update({'run_tag': run_tag, 'test_file': args.test_file})
    save_results_json(run_tag, out)

@combination('mymodel', 'zoomer', 'zoomer')
def run_mymodel_zoomer_zoomer(args, run_tag):
    print('  template=mymodel : MITRE technique templates (templates_all92, default) -> RoBERTa, plain (192-dim)')
    print('  graph=zoomer     : real provenance graph -> ZOOMER\'s TSGModel (GAT deep + wide linear, 192-dim)')
    print('  training=zoomer  : prototype = template embedding ONLY (no example-averaging), '
          "ZOOMER's squared-distance cross-entropy loss")

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(
        args, sequences_dir=SEQUENCES_RAWFULL_DIR)

    step_to_path, graph_model, embed_graph, wide_dim = setup_zoomer_graph_encoder(train_entries, device)

    tmpl_encoder, text_proj = setup_roberta_encoder(device, camlds_train.COMBINED_DIM)
    tokenizer = RobertaTokenizer.from_pretrained(camlds_train.ROBERTA_MODEL)
    print('  graph_model out_dim={}  text_proj out_dim={}  (must match -- they do)'.format(
        camlds_train.COMBINED_DIM, camlds_train.COMBINED_DIM))

    trainable_params = (
        list(graph_model.parameters()) +
        [p for p in tmpl_encoder.parameters() if p.requires_grad] + list(text_proj.parameters())
    )

    def embed_prototypes():
        return text_proj(torch.stack(
            [camlds_train.encode_one(tokenizer, tmpl_encoder, templates[t], device) for t in all_techniques]))

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        prototypes = embed_prototypes()
        step_losses = []
        for e in batch_entries:
            embed = embed_graph(step_to_path[e['file']])
            true_idxs = [technique_to_col[t] for t in e['techniques']]
            exempt_all = set(true_idxs)
            for true_idx in true_idxs:
                step_losses.append(zoomer_masked_single_label_loss(
                    embed, prototypes, true_idx, exempt_all - {true_idx}))
        return torch.stack(step_losses).mean()

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'graph_model': graph_model, 'tmpl_encoder': tmpl_encoder, 'text_proj': text_proj}
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'camlds_zoomer_zoomer_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules)
    else:
        history, best_loss = run_training_loop(
            args, 'episode', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=False)
        save_checkpoint(ckpt_path, modules, extra={
            'techniques': all_techniques, 'best_loss': best_loss, 'history': history,
            'run_tag': run_tag, 'test_file': args.test_file, 'wide_dim': wide_dim,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    with torch.no_grad():
        prototypes = embed_prototypes()

        def embed_entry(e):
            path = step_to_path.get(e['file'])
            return None if path is None else embed_graph(path)

        results = zoomer_style_build_results(test_entries, all_techniques, embed_entry, prototypes)

    out = {'run_tag': run_tag, 'test_file': args.test_file, 'n_total': len(results)}
    out.update(zoomer_style_test_rollup(results, all_techniques, train_entries))
    out['results'] = results
    save_results_json(run_tag, out)

def _load_knowhow_fasttext():
    from gensim.models import FastText
    path = os.path.join(CAM_LDS_DIR, 'knowhow', 'KNOWHOW-seng402', 'technique-embedding-128.model')
    model = FastText.load(path)
    print('  KnowHow FastText model loaded (technique-embedding-128.model)')
    return model

@combination('mymodel', 'knowhow', 'mymodel')
def run_mymodel_knowhow_mymodel(args, run_tag):
    print('  template=mymodel : MITRE technique templates (templates_all92, default) -> RoBERTa, plain (128-dim)')
    print('  graph=knowhow    : whole-step generalized sequence text (sequences/) -> KnowHow\'s real FastText '
          'word-average (fixed, 128-dim) -> one small trainable adapter layer')
    print('  training=mymodel : your contrastive-prototype loss (multilabel_prototype_loss + uniformity_loss)')

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(args)

    fasttext_model = _load_knowhow_fasttext()
    knowhow_graph_vec = make_knowhow_vectorizer(fasttext_model)

    tmpl_encoder, text_proj = setup_roberta_encoder(device, camlds_train.PROJ_DIM)
    tokenizer = RobertaTokenizer.from_pretrained(camlds_train.ROBERTA_MODEL)
    graph_adapter = nn.Linear(128, camlds_train.PROJ_DIM).to(device)
    print('  graph_adapter out_dim={}  text_proj out_dim={}  (must match -- they do)'.format(
        camlds_train.PROJ_DIM, camlds_train.PROJ_DIM))

    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    trainable_params = (
        list(graph_adapter.parameters()) +
        [p for p in tmpl_encoder.parameters() if p.requires_grad] + list(text_proj.parameters()) +
        [logit_scale]
    )

    def embed_graph(entry):
        return graph_adapter(knowhow_graph_vec(entry['file'], entry['sequence']).to(device))

    def embed_prototypes():
        return text_proj(torch.stack(
            [camlds_train.encode_one(tokenizer, tmpl_encoder, templates[t], device) for t in all_techniques]))

    train_pos_mask = camlds_train.build_pos_mask(train_entries, all_techniques, technique_to_col).to(device)

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        pos_mask_batch = train_pos_mask[batch_idxs]
        z = torch.stack([embed_graph(e) for e in batch_entries])
        prototypes = embed_prototypes()
        main_loss = camlds_train.multilabel_prototype_loss(z, prototypes, pos_mask_batch, logit_scale)
        return main_loss + camlds_train.LAMBDA_UNIFORM * camlds_train.uniformity_loss(z)

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'graph_adapter': graph_adapter, 'tmpl_encoder': tmpl_encoder, 'text_proj': text_proj}
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'camlds_knowhow_camlds_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules, logit_scale=logit_scale)
    else:
        history, best_loss = run_training_loop(
            args, 'epoch', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=True, logit_scale=logit_scale)
        save_checkpoint(ckpt_path, modules, extra={
            'logit_scale': logit_scale, 'techniques': all_techniques, 'best_loss': best_loss,
            'history': history, 'run_tag': run_tag, 'test_file': args.test_file,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    display_scale = logit_scale.exp().item()
    with torch.no_grad():
        zt_all = F.normalize(embed_prototypes(), dim=-1)
        results = mymodel_style_build_results(
            test_entries, all_techniques, lambda e: embed_graph(e), zt_all, display_scale)

    out = camlds_test.print_table(results, all_techniques)
    out = mymodel_style_test_rollup(out, results, all_techniques, train_entries)
    out.update({'run_tag': run_tag, 'test_file': args.test_file})
    save_results_json(run_tag, out)

@combination('knowhow', 'mymodel', 'mymodel')
def run_knowhow_mymodel_mymodel(args, run_tag):
    print('  template=knowhow : MITRE technique templates -> KnowHow\'s real FastText word-average '
          '(fixed, 128-dim) -> one small trainable adapter layer')
    print('  graph=mymodel    : generalized sequence text (sequences/) -> RoBERTa, plain (128-dim)')
    print('  training=mymodel : your contrastive-prototype loss (multilabel_prototype_loss + uniformity_loss)')

    device = torch.device('cuda')
    camlds_train.MAX_CHUNKS = args.max_chunks if args.max_chunks > 0 else None

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(
        args, sequences_dir=graph_sequences_dir(args))

    fasttext_model = _load_knowhow_fasttext()
    knowhow_template_vec = make_knowhow_vectorizer(fasttext_model)

    if getattr(args, 'trainable_knowhow_templates', False):
        template_params = nn.ParameterDict({
            t: nn.Parameter(knowhow_template_vec(t, templates[t]).clone()) for t in all_techniques
        }).to(device)
        print('  KnowHow templates: TRAINABLE (initialized from FastText average, then free to move)')
    else:
        template_params = None
        print('  KnowHow templates: frozen (FastText average, fixed)')

    if args.use_wide:
        (fit_wide_fn, wide_vector_fn) = camlds_train.WIDE_FEATURE_FNS[args.wide_features]
        print('  Wide features pipeline: {}'.format(args.wide_features))
        step_to_path = camlds_train.build_step_to_graph_path()
        train_steps = [e['file'] for e in train_entries if e['techniques']]
        bins, masks, wide_dim = fit_wide_fn(train_steps, step_to_path, seed=camlds_train.SEED)
        print('  ZOOMER wide bins fit on {} training graphs, wide_dim={}'.format(len(train_steps), wide_dim))
        wide_model = camlds_train.WideModel(wide_dim, camlds_train.WIDE_DIM_OUT).to(device)
        template_out_dim = camlds_train.COMBINED_DIM
    else:
        step_to_path, bins, masks, wide_model, wide_vector_fn = None, None, None, None, None
        template_out_dim = camlds_train.PROJ_DIM

    tokenizer = RobertaTokenizer.from_pretrained(camlds_train.ROBERTA_MODEL)
    seq_encoder, log_proj = setup_roberta_encoder(device, camlds_train.PROJ_DIM)
    template_adapter = nn.Linear(128, template_out_dim).to(device)
    print('  log_proj out_dim={}  template_adapter out_dim={}  (must match -- they do)'.format(
        template_out_dim, template_out_dim))

    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    trainable_params = (
        [p for p in seq_encoder.parameters() if p.requires_grad] + list(log_proj.parameters()) +
        list(template_adapter.parameters()) + [logit_scale]
    )
    if wide_model is not None:
        trainable_params += list(wide_model.parameters())
    if template_params is not None:
        trainable_params += list(template_params.parameters())

    def embed_sequence(entry):
        kwargs = {'vector_fn': wide_vector_fn} if wide_vector_fn is not None else {}
        return camlds_train.encode_sequence_combined(
            tokenizer, seq_encoder, log_proj, wide_model, entry, step_to_path, bins, masks, device, **kwargs)

    def embed_prototypes():
        if template_params is not None:
            raw = torch.stack([template_params[t] for t in all_techniques])
        else:
            raw = torch.stack([knowhow_template_vec(t, templates[t]) for t in all_techniques]).to(device)
        return template_adapter(raw)

    train_pos_mask = camlds_train.build_pos_mask(train_entries, all_techniques, technique_to_col).to(device)

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        pos_mask_batch = train_pos_mask[batch_idxs]
        z = torch.stack([embed_sequence(e) for e in batch_entries])
        prototypes = embed_prototypes()
        main_loss = camlds_train.multilabel_prototype_loss(z, prototypes, pos_mask_batch, logit_scale)
        return main_loss + camlds_train.LAMBDA_UNIFORM * camlds_train.uniformity_loss(z)

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'seq_encoder': seq_encoder, 'log_proj': log_proj, 'template_adapter': template_adapter}
    if wide_model is not None:
        modules['wide_model'] = wide_model
    if template_params is not None:
        modules['template_params'] = template_params
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'knowhow_camlds_camlds_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules, logit_scale=logit_scale)
    else:
        history, best_loss = run_training_loop(
            args, 'epoch', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=True, logit_scale=logit_scale)
        save_checkpoint(ckpt_path, modules, extra={
            'logit_scale': logit_scale, 'techniques': all_techniques, 'best_loss': best_loss,
            'history': history, 'run_tag': run_tag, 'test_file': args.test_file,
            'wide_features': args.wide_features if args.use_wide else None,
            'trainable_knowhow_templates': template_params is not None,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    display_scale = logit_scale.exp().item()
    with torch.no_grad():
        zt_all = F.normalize(embed_prototypes(), dim=-1)
        results = mymodel_style_build_results(
            test_entries, all_techniques, embed_sequence, zt_all, display_scale)

    out = camlds_test.print_table(results, all_techniques)
    out = mymodel_style_test_rollup(out, results, all_techniques, train_entries)
    out.update({'run_tag': run_tag, 'test_file': args.test_file,
                'wide_features': args.wide_features if args.use_wide else None,
                'trainable_knowhow_templates': template_params is not None})
    save_results_json(run_tag, out)

@combination('knowhow', 'mymodel', 'zoomer')
def run_knowhow_mymodel_zoomer(args, run_tag):
    print('  template=knowhow : MITRE technique templates -> KnowHow\'s real FastText word-average '
          '(fixed, 128-dim) -> one small trainable adapter layer')
    print('  graph=mymodel    : generalized sequence text (sequences/) -> RoBERTa, plain (128-dim)')
    print('  training=zoomer  : prototype = template embedding ONLY (no example-averaging), '
          "ZOOMER's squared-distance cross-entropy loss")

    device = torch.device('cuda')
    camlds_train.MAX_CHUNKS = args.max_chunks if args.max_chunks > 0 else None

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(
        args, sequences_dir=graph_sequences_dir(args))

    fasttext_model = _load_knowhow_fasttext()
    knowhow_template_vec = make_knowhow_vectorizer(fasttext_model)

    if args.use_wide:
        step_to_path = camlds_train.build_step_to_graph_path()
        train_steps = [e['file'] for e in train_entries if e['techniques']]
        bins, masks, wide_dim = camlds_train.fit_wide_bins_and_masks(train_steps, step_to_path, seed=camlds_train.SEED)
        print('  ZOOMER wide bins fit on {} training graphs, wide_dim={}'.format(len(train_steps), wide_dim))
        wide_model = camlds_train.WideModel(wide_dim, camlds_train.WIDE_DIM_OUT).to(device)
        template_out_dim = camlds_train.COMBINED_DIM
    else:
        step_to_path, bins, masks, wide_model = None, None, None, None
        template_out_dim = camlds_train.PROJ_DIM

    tokenizer = RobertaTokenizer.from_pretrained(camlds_train.ROBERTA_MODEL)
    seq_encoder, log_proj = setup_roberta_encoder(device, camlds_train.PROJ_DIM)
    template_adapter = nn.Linear(128, template_out_dim).to(device)
    print('  log_proj out_dim={}  template_adapter out_dim={}  (must match -- they do)'.format(
        template_out_dim, template_out_dim))

    trainable_params = (
        [p for p in seq_encoder.parameters() if p.requires_grad] + list(log_proj.parameters()) +
        list(template_adapter.parameters())
    )
    if wide_model is not None:
        trainable_params += list(wide_model.parameters())

    def embed_sequence(entry):
        return camlds_train.encode_sequence_combined(
            tokenizer, seq_encoder, log_proj, wide_model, entry, step_to_path, bins, masks, device)

    def embed_prototypes():
        return template_adapter(torch.stack(
            [knowhow_template_vec(t, templates[t]) for t in all_techniques]).to(device))

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        prototypes = embed_prototypes()
        step_losses = []
        for e in batch_entries:
            embed = embed_sequence(e)
            true_idxs = [technique_to_col[t] for t in e['techniques']]
            exempt_all = set(true_idxs)
            for true_idx in true_idxs:
                step_losses.append(zoomer_masked_single_label_loss(
                    embed, prototypes, true_idx, exempt_all - {true_idx}))
        return torch.stack(step_losses).mean()

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'seq_encoder': seq_encoder, 'log_proj': log_proj, 'template_adapter': template_adapter}
    if wide_model is not None:
        modules['wide_model'] = wide_model
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'knowhow_camlds_zoomer_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules)
    else:
        history, best_loss = run_training_loop(
            args, 'episode', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=True)
        save_checkpoint(ckpt_path, modules, extra={
            'techniques': all_techniques, 'best_loss': best_loss, 'history': history,
            'run_tag': run_tag, 'test_file': args.test_file,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    with torch.no_grad():
        prototypes = embed_prototypes()
        results = zoomer_style_build_results(
            test_entries, all_techniques, embed_sequence, prototypes)

    out = {'run_tag': run_tag, 'test_file': args.test_file, 'n_total': len(results)}
    out.update(zoomer_style_test_rollup(results, all_techniques, train_entries))
    out['results'] = results
    save_results_json(run_tag, out)

@combination('knowhow', 'zoomer', 'mymodel')
def run_knowhow_zoomer_mymodel(args, run_tag):
    print('  template=knowhow : MITRE technique templates -> KnowHow\'s real FastText word-average '
          '(fixed, 128-dim) -> one small trainable adapter layer (192-dim out)')
    print('  graph=zoomer     : real provenance graph -> ZOOMER\'s TSGModel (GAT deep + wide linear, 192-dim)')
    print('  training=mymodel : your contrastive-prototype loss (multilabel_prototype_loss + uniformity_loss)')

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(
        args, sequences_dir=SEQUENCES_RAWFULL_DIR)

    fasttext_model = _load_knowhow_fasttext()
    knowhow_template_vec = make_knowhow_vectorizer(fasttext_model)

    step_to_path, graph_model, embed_graph, wide_dim = setup_zoomer_graph_encoder(train_entries, device)

    template_adapter = nn.Linear(128, camlds_train.COMBINED_DIM).to(device)
    print('  graph_model out_dim={}  template_adapter out_dim={}  (must match -- they do)'.format(
        camlds_train.COMBINED_DIM, camlds_train.COMBINED_DIM))

    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    trainable_params = (
        list(graph_model.parameters()) + list(template_adapter.parameters()) + [logit_scale]
    )

    def embed_prototypes():
        return template_adapter(torch.stack(
            [knowhow_template_vec(t, templates[t]) for t in all_techniques]).to(device))

    train_pos_mask = camlds_train.build_pos_mask(train_entries, all_techniques, technique_to_col).to(device)

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        pos_mask_batch = train_pos_mask[batch_idxs]
        z = torch.stack([embed_graph(step_to_path[e['file']]) for e in batch_entries])
        prototypes = embed_prototypes()
        main_loss = camlds_train.multilabel_prototype_loss(z, prototypes, pos_mask_batch, logit_scale)
        return main_loss + camlds_train.LAMBDA_UNIFORM * camlds_train.uniformity_loss(z)

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'graph_model': graph_model, 'template_adapter': template_adapter}
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'knowhow_zoomer_camlds_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules, logit_scale=logit_scale)
    else:
        history, best_loss = run_training_loop(
            args, 'epoch', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=False, logit_scale=logit_scale)
        save_checkpoint(ckpt_path, modules, extra={
            'logit_scale': logit_scale, 'techniques': all_techniques, 'best_loss': best_loss,
            'history': history, 'run_tag': run_tag, 'test_file': args.test_file, 'wide_dim': wide_dim,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    display_scale = logit_scale.exp().item()
    with torch.no_grad():
        zt_all = F.normalize(embed_prototypes(), dim=-1)

        def embed_entry(e):
            path = step_to_path.get(e['file'])
            return None if path is None else embed_graph(path)

        results = mymodel_style_build_results(test_entries, all_techniques, embed_entry, zt_all, display_scale)

    out = camlds_test.print_table(results, all_techniques)
    out = mymodel_style_test_rollup(out, results, all_techniques, train_entries)
    out.update({'run_tag': run_tag, 'test_file': args.test_file})
    save_results_json(run_tag, out)

@combination('knowhow', 'zoomer', 'zoomer')
def run_knowhow_zoomer_zoomer(args, run_tag):
    print('  template=knowhow : MITRE technique templates -> KnowHow\'s real FastText word-average '
          '(fixed, 128-dim) -> one small trainable adapter layer (192-dim out)')
    print('  graph=zoomer     : real provenance graph -> ZOOMER\'s TSGModel (GAT deep + wide linear, 192-dim)')
    print('  training=zoomer  : prototype = template embedding ONLY (no example-averaging), '
          "ZOOMER's squared-distance cross-entropy loss")

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(
        args, sequences_dir=SEQUENCES_RAWFULL_DIR)

    fasttext_model = _load_knowhow_fasttext()
    knowhow_template_vec = make_knowhow_vectorizer(fasttext_model)

    step_to_path, graph_model, embed_graph, wide_dim = setup_zoomer_graph_encoder(train_entries, device)

    template_adapter = nn.Linear(128, camlds_train.COMBINED_DIM).to(device)
    print('  graph_model out_dim={}  template_adapter out_dim={}  (must match -- they do)'.format(
        camlds_train.COMBINED_DIM, camlds_train.COMBINED_DIM))

    trainable_params = list(graph_model.parameters()) + list(template_adapter.parameters())

    def embed_prototypes():
        return template_adapter(torch.stack(
            [knowhow_template_vec(t, templates[t]) for t in all_techniques]).to(device))

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        prototypes = embed_prototypes()
        step_losses = []
        for e in batch_entries:
            embed = embed_graph(step_to_path[e['file']])
            true_idxs = [technique_to_col[t] for t in e['techniques']]
            exempt_all = set(true_idxs)
            for true_idx in true_idxs:
                step_losses.append(zoomer_masked_single_label_loss(
                    embed, prototypes, true_idx, exempt_all - {true_idx}))
        return torch.stack(step_losses).mean()

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'graph_model': graph_model, 'template_adapter': template_adapter}
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'knowhow_zoomer_zoomer_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules)
    else:
        history, best_loss = run_training_loop(
            args, 'episode', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=False)
        save_checkpoint(ckpt_path, modules, extra={
            'techniques': all_techniques, 'best_loss': best_loss, 'history': history,
            'run_tag': run_tag, 'test_file': args.test_file, 'wide_dim': wide_dim,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    with torch.no_grad():
        prototypes = embed_prototypes()

        def embed_entry(e):
            path = step_to_path.get(e['file'])
            return None if path is None else embed_graph(path)

        results = zoomer_style_build_results(test_entries, all_techniques, embed_entry, prototypes)

    out = {'run_tag': run_tag, 'test_file': args.test_file, 'n_total': len(results)}
    out.update(zoomer_style_test_rollup(results, all_techniques, train_entries))
    out['results'] = results
    save_results_json(run_tag, out)

@combination('knowhow', 'knowhow', 'mymodel')
def run_knowhow_knowhow_mymodel(args, run_tag):
    print('  template=knowhow : MITRE technique templates -> KnowHow\'s real FastText word-average '
          '(fixed, 128-dim) -> one small trainable adapter layer')
    print('  graph=knowhow    : whole-step generalized sequence text (sequences/) -> KnowHow\'s real FastText '
          'word-average (fixed, 128-dim) -> one small trainable adapter layer')
    print('  training=mymodel : your contrastive-prototype loss (multilabel_prototype_loss + uniformity_loss)')

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(args)

    fasttext_model = _load_knowhow_fasttext()
    knowhow_template_vec = make_knowhow_vectorizer(fasttext_model)
    knowhow_graph_vec = make_knowhow_vectorizer(fasttext_model)

    if getattr(args, 'trainable_knowhow_templates', False):
        template_params = nn.ParameterDict({
            t: nn.Parameter(knowhow_template_vec(t, templates[t]).clone()) for t in all_techniques
        }).to(device)
        print('  KnowHow templates: TRAINABLE (initialized from FastText average, then free to move)')
    else:
        template_params = None
        print('  KnowHow templates: frozen (FastText average, fixed)')

    template_adapter = nn.Linear(128, camlds_train.PROJ_DIM).to(device)
    graph_adapter = nn.Linear(128, camlds_train.PROJ_DIM).to(device)
    print('  graph_adapter out_dim={}  template_adapter out_dim={}  (must match -- they do)'.format(
        camlds_train.PROJ_DIM, camlds_train.PROJ_DIM))

    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    trainable_params = list(template_adapter.parameters()) + list(graph_adapter.parameters()) + [logit_scale]
    if template_params is not None:
        trainable_params += list(template_params.parameters())

    def embed_sequence(entry):
        return graph_adapter(knowhow_graph_vec(entry['file'], entry['sequence']).to(device))

    def embed_prototypes():
        if template_params is not None:
            raw = torch.stack([template_params[t] for t in all_techniques])
        else:
            raw = torch.stack([knowhow_template_vec(t, templates[t]) for t in all_techniques]).to(device)
        return template_adapter(raw)

    train_pos_mask = camlds_train.build_pos_mask(train_entries, all_techniques, technique_to_col).to(device)

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        pos_mask_batch = train_pos_mask[batch_idxs]
        z = torch.stack([embed_sequence(e) for e in batch_entries])
        prototypes = embed_prototypes()
        main_loss = camlds_train.multilabel_prototype_loss(z, prototypes, pos_mask_batch, logit_scale)
        return main_loss + camlds_train.LAMBDA_UNIFORM * camlds_train.uniformity_loss(z)

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'template_adapter': template_adapter, 'graph_adapter': graph_adapter}
    if template_params is not None:
        modules['template_params'] = template_params
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'knowhow_knowhow_camlds_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules, logit_scale=logit_scale)
    else:
        history, best_loss = run_training_loop(
            args, 'epoch', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=False, logit_scale=logit_scale)
        save_checkpoint(ckpt_path, modules, extra={
            'logit_scale': logit_scale, 'techniques': all_techniques, 'best_loss': best_loss,
            'history': history, 'run_tag': run_tag, 'test_file': args.test_file,
            'trainable_knowhow_templates': template_params is not None,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    display_scale = logit_scale.exp().item()
    with torch.no_grad():
        zt_all = F.normalize(embed_prototypes(), dim=-1)
        results = mymodel_style_build_results(
            test_entries, all_techniques, lambda e: embed_sequence(e), zt_all, display_scale)

    out = camlds_test.print_table(results, all_techniques)
    out = mymodel_style_test_rollup(out, results, all_techniques, train_entries)
    out.update({'run_tag': run_tag, 'test_file': args.test_file,
                'trainable_knowhow_templates': template_params is not None})
    save_results_json(run_tag, out)

@combination('knowhow', 'knowhow', 'zoomer')
def run_knowhow_knowhow_zoomer(args, run_tag):
    print('  template=knowhow : MITRE technique templates -> KnowHow\'s real FastText word-average '
          '(fixed, 128-dim) -> one small trainable adapter layer')
    print('  graph=knowhow    : whole-step generalized sequence text (sequences/) -> KnowHow\'s real FastText '
          'word-average (fixed, 128-dim) -> one small trainable adapter layer')
    print('  training=zoomer  : prototype = template embedding ONLY (no example-averaging), '
          "ZOOMER's squared-distance cross-entropy loss")

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(args)

    fasttext_model = _load_knowhow_fasttext()
    knowhow_template_vec = make_knowhow_vectorizer(fasttext_model)
    knowhow_graph_vec = make_knowhow_vectorizer(fasttext_model)

    template_adapter = nn.Linear(128, camlds_train.PROJ_DIM).to(device)
    graph_adapter = nn.Linear(128, camlds_train.PROJ_DIM).to(device)
    print('  graph_adapter out_dim={}  template_adapter out_dim={}  (must match -- they do)'.format(
        camlds_train.PROJ_DIM, camlds_train.PROJ_DIM))

    trainable_params = list(template_adapter.parameters()) + list(graph_adapter.parameters())

    def embed_sequence(entry):
        return graph_adapter(knowhow_graph_vec(entry['file'], entry['sequence']).to(device))

    def embed_prototypes():
        return template_adapter(torch.stack(
            [knowhow_template_vec(t, templates[t]) for t in all_techniques]).to(device))

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        prototypes = embed_prototypes()
        step_losses = []
        for e in batch_entries:
            embed = embed_sequence(e)
            true_idxs = [technique_to_col[t] for t in e['techniques']]
            exempt_all = set(true_idxs)
            for true_idx in true_idxs:
                step_losses.append(zoomer_masked_single_label_loss(
                    embed, prototypes, true_idx, exempt_all - {true_idx}))
        return torch.stack(step_losses).mean()

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'template_adapter': template_adapter, 'graph_adapter': graph_adapter}
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'knowhow_knowhow_zoomer_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules)
    else:
        history, best_loss = run_training_loop(
            args, 'episode', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=False)
        save_checkpoint(ckpt_path, modules, extra={
            'techniques': all_techniques, 'best_loss': best_loss, 'history': history,
            'run_tag': run_tag, 'test_file': args.test_file,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    with torch.no_grad():
        prototypes = embed_prototypes()
        results = zoomer_style_build_results(test_entries, all_techniques, lambda e: embed_sequence(e), prototypes)

    out = {'run_tag': run_tag, 'test_file': args.test_file, 'n_total': len(results)}
    out.update(zoomer_style_test_rollup(results, all_techniques, train_entries))
    out['results'] = results
    save_results_json(run_tag, out)

@combination('mymodel', 'knowhow', 'zoomer')
def run_mymodel_knowhow_zoomer(args, run_tag):
    print('  template=mymodel : MITRE technique templates (templates_all92, default) -> RoBERTa, plain (128-dim)')
    print('  graph=knowhow    : whole-step generalized sequence text (sequences/) -> KnowHow\'s real FastText '
          'word-average (fixed, 128-dim) -> one small trainable adapter layer')
    print('  training=zoomer  : prototype = template embedding ONLY (no example-averaging), '
          "ZOOMER's squared-distance cross-entropy loss")

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, templates = load_entries_and_templates(args)

    fasttext_model = _load_knowhow_fasttext()
    knowhow_graph_vec = make_knowhow_vectorizer(fasttext_model)

    tmpl_encoder, text_proj = setup_roberta_encoder(device, camlds_train.PROJ_DIM)
    tokenizer = RobertaTokenizer.from_pretrained(camlds_train.ROBERTA_MODEL)
    graph_adapter = nn.Linear(128, camlds_train.PROJ_DIM).to(device)
    print('  graph_adapter out_dim={}  text_proj out_dim={}  (must match -- they do)'.format(
        camlds_train.PROJ_DIM, camlds_train.PROJ_DIM))

    trainable_params = (
        [p for p in tmpl_encoder.parameters() if p.requires_grad] + list(text_proj.parameters()) +
        list(graph_adapter.parameters())
    )

    def embed_sequence(entry):
        return graph_adapter(knowhow_graph_vec(entry['file'], entry['sequence']).to(device))

    def embed_prototypes():
        return text_proj(torch.stack(
            [camlds_train.encode_one(tokenizer, tmpl_encoder, templates[t], device) for t in all_techniques]))

    def compute_loss(batch_idxs):
        batch_entries = [train_entries[i] for i in batch_idxs]
        prototypes = embed_prototypes()
        step_losses = []
        for e in batch_entries:
            embed = embed_sequence(e)
            true_idxs = [technique_to_col[t] for t in e['techniques']]
            exempt_all = set(true_idxs)
            for true_idx in true_idxs:
                step_losses.append(zoomer_masked_single_label_loss(
                    embed, prototypes, true_idx, exempt_all - {true_idx}))
        return torch.stack(step_losses).mean()

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'tmpl_encoder': tmpl_encoder, 'text_proj': text_proj, 'graph_adapter': graph_adapter}
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'camlds_knowhow_zoomer_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules)
    else:
        history, best_loss = run_training_loop(
            args, 'episode', modules, trainable_params, technique_pools, train_entries, all_techniques,
            compute_loss, use_amp=True)
        save_checkpoint(ckpt_path, modules, extra={
            'techniques': all_techniques, 'best_loss': best_loss, 'history': history,
            'run_tag': run_tag, 'test_file': args.test_file,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    with torch.no_grad():
        prototypes = embed_prototypes()
        results = zoomer_style_build_results(test_entries, all_techniques, lambda e: embed_sequence(e), prototypes)

    out = {'run_tag': run_tag, 'test_file': args.test_file, 'n_total': len(results)}
    out.update(zoomer_style_test_rollup(results, all_techniques, train_entries))
    out['results'] = results
    save_results_json(run_tag, out)

@combination('none', 'mymodel', 'zoomer')
def run_none_mymodel_zoomer(args, run_tag):
    print('  template=none    : no MITRE templates at all -- a technique\'s prototype is the AVERAGE of a '
          "few of its own real training examples (ZOOMER's own K-shot episodic construction)")
    print('  graph=mymodel    : RAW sequence text (sequences_rawfull/ by default) -> RoBERTa, plain'
          + (' + ZOOMER wide features' if args.use_wide else ', no wide features'))
    print("  training=zoomer  : same source embeds both the averaged prototype and the query -- "
          "ZOOMER's squared-distance cross-entropy loss")

    device = torch.device('cuda')
    camlds_train.MAX_CHUNKS = args.max_chunks if args.max_chunks > 0 else None

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, _templates = load_entries_and_templates(
        args, sequences_dir=graph_sequences_dir(args))

    if args.use_wide:
        step_to_path = camlds_train.build_step_to_graph_path()
        train_steps = [e['file'] for e in train_entries if e['techniques']]
        bins, masks, wide_dim = camlds_train.fit_wide_bins_and_masks(train_steps, step_to_path, seed=camlds_train.SEED)
        print('  ZOOMER wide bins fit on {} training graphs, wide_dim={}'.format(len(train_steps), wide_dim))
        wide_model = camlds_train.WideModel(wide_dim, camlds_train.WIDE_DIM_OUT).to(device)
    else:
        step_to_path, bins, masks, wide_model = None, None, None, None

    tokenizer = RobertaTokenizer.from_pretrained(camlds_train.ROBERTA_MODEL)
    seq_encoder, log_proj = setup_roberta_encoder(device, camlds_train.PROJ_DIM)

    trainable_params = [p for p in seq_encoder.parameters() if p.requires_grad] + list(log_proj.parameters())
    if wide_model is not None:
        trainable_params += list(wide_model.parameters())

    def embed_sequence(entry):
        return camlds_train.encode_sequence_combined(
            tokenizer, seq_encoder, log_proj, wide_model, entry, step_to_path, bins, masks, device)

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'seq_encoder': seq_encoder, 'log_proj': log_proj}
    if wide_model is not None:
        modules['wide_model'] = wide_model
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'none_camlds_zoomer_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules)
    else:
        history, best_loss = run_episodic_training(
            args, embed_sequence, technique_pools, train_entries, all_techniques, modules, trainable_params,
            use_amp=True)
        save_checkpoint(ckpt_path, modules, extra={
            'techniques': all_techniques, 'best_loss': best_loss, 'history': history,
            'run_tag': run_tag, 'test_file': args.test_file,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    with torch.no_grad():
        prototypes, used_techniques = build_final_episodic_prototypes(
            embed_sequence, technique_pools, train_entries, all_techniques)
        results = zoomer_style_build_results(test_entries, used_techniques, embed_sequence, prototypes)

    out = {'run_tag': run_tag, 'test_file': args.test_file, 'n_total': len(results)}
    out.update(zoomer_style_test_rollup(results, used_techniques, train_entries))
    out['results'] = results
    save_results_json(run_tag, out)

@combination('none', 'zoomer', 'zoomer')
def run_none_zoomer_zoomer(args, run_tag):
    print('  template=none    : no MITRE templates at all -- a technique\'s prototype is the AVERAGE of a '
          "few of its own real training examples (ZOOMER's own K-shot episodic construction)")
    print('  graph=zoomer     : real provenance graph -> ZOOMER\'s TSGModel (GAT deep + wide linear, 192-dim)')
    print("  training=zoomer  : same source embeds both the averaged prototype and the query -- "
          "ZOOMER's squared-distance cross-entropy loss")

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, _templates = load_entries_and_templates(
        args, sequences_dir=SEQUENCES_RAWFULL_DIR)

    step_to_path, graph_model, embed_graph, wide_dim = setup_zoomer_graph_encoder(train_entries, device)

    def embed_entry_train(e):
        return embed_graph(step_to_path[e['file']])

    def embed_entry_test(e):
        path = step_to_path.get(e['file'])
        return None if path is None else embed_graph(path)

    trainable_params = list(graph_model.parameters())
    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'graph_model': graph_model}
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'none_zoomer_zoomer_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules)
    else:
        history, best_loss = run_episodic_training(
            args, embed_entry_train, technique_pools, train_entries, all_techniques, modules, trainable_params,
            use_amp=False)
        save_checkpoint(ckpt_path, modules, extra={
            'techniques': all_techniques, 'best_loss': best_loss, 'history': history,
            'run_tag': run_tag, 'test_file': args.test_file, 'wide_dim': wide_dim,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    with torch.no_grad():
        prototypes, used_techniques = build_final_episodic_prototypes(
            embed_entry_train, technique_pools, train_entries, all_techniques)
        results = zoomer_style_build_results(test_entries, used_techniques, embed_entry_test, prototypes)

    out = {'run_tag': run_tag, 'test_file': args.test_file, 'n_total': len(results)}
    out.update(zoomer_style_test_rollup(results, used_techniques, train_entries))
    out['results'] = results
    save_results_json(run_tag, out)

@combination('none', 'mymodel', 'mymodel')
def run_none_mymodel_mymodel(args, run_tag):
    print('  template=none    : no MITRE templates at all -- a technique\'s prototype is the AVERAGE of a '
          "few of its own real training examples (ZOOMER's own K-shot episodic construction)")
    print('  graph=mymodel    : RAW sequence text (sequences_rawfull/ by default) -> RoBERTa, plain'
          + (' + ZOOMER wide features' if args.use_wide else ', no wide features'))
    print('  training=mymodel : same source embeds both the averaged prototype and the query -- '
          'your contrastive-prototype loss (multilabel_prototype_loss + uniformity_loss)')

    device = torch.device('cuda')
    camlds_train.MAX_CHUNKS = args.max_chunks if args.max_chunks > 0 else None

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, _templates = load_entries_and_templates(
        args, sequences_dir=graph_sequences_dir(args))

    if args.use_wide:
        step_to_path = camlds_train.build_step_to_graph_path()
        train_steps = [e['file'] for e in train_entries if e['techniques']]
        bins, masks, wide_dim = camlds_train.fit_wide_bins_and_masks(train_steps, step_to_path, seed=camlds_train.SEED)
        print('  ZOOMER wide bins fit on {} training graphs, wide_dim={}'.format(len(train_steps), wide_dim))
        wide_model = camlds_train.WideModel(wide_dim, camlds_train.WIDE_DIM_OUT).to(device)
    else:
        step_to_path, bins, masks, wide_model = None, None, None, None

    tokenizer = RobertaTokenizer.from_pretrained(camlds_train.ROBERTA_MODEL)
    seq_encoder, log_proj = setup_roberta_encoder(device, camlds_train.PROJ_DIM)
    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    trainable_params = (
        [p for p in seq_encoder.parameters() if p.requires_grad] + list(log_proj.parameters()) + [logit_scale]
    )
    if wide_model is not None:
        trainable_params += list(wide_model.parameters())

    def embed_sequence(entry):
        return camlds_train.encode_sequence_combined(
            tokenizer, seq_encoder, log_proj, wide_model, entry, step_to_path, bins, masks, device)

    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'seq_encoder': seq_encoder, 'log_proj': log_proj}
    if wide_model is not None:
        modules['wide_model'] = wide_model
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'none_camlds_camlds_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules, logit_scale=logit_scale)
    else:
        history, best_loss = run_episodic_training(
            args, embed_sequence, technique_pools, train_entries, all_techniques, modules, trainable_params,
            use_amp=True, loss_style='mymodel', logit_scale=logit_scale)
        save_checkpoint(ckpt_path, modules, extra={
            'logit_scale': logit_scale, 'techniques': all_techniques, 'best_loss': best_loss,
            'history': history, 'run_tag': run_tag, 'test_file': args.test_file,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    display_scale = logit_scale.exp().item()
    with torch.no_grad():
        prototypes, used_techniques = build_final_episodic_prototypes(
            embed_sequence, technique_pools, train_entries, all_techniques)
        zt_all = F.normalize(prototypes, dim=-1)
        results = mymodel_style_build_results(test_entries, used_techniques, embed_sequence, zt_all, display_scale)

    out = camlds_test.print_table(results, used_techniques)
    out = mymodel_style_test_rollup(out, results, used_techniques, train_entries)
    out.update({'run_tag': run_tag, 'test_file': args.test_file})
    save_results_json(run_tag, out)

@combination('none', 'zoomer', 'mymodel')
def run_none_zoomer_mymodel(args, run_tag):
    print('  template=none    : no MITRE templates at all -- a technique\'s prototype is the AVERAGE of a '
          "few of its own real training examples (ZOOMER's own K-shot episodic construction)")
    print('  graph=zoomer     : real provenance graph -> ZOOMER\'s TSGModel (GAT deep + wide linear, 192-dim)')
    print('  training=mymodel : same source embeds both the averaged prototype and the query -- '
          'your contrastive-prototype loss (multilabel_prototype_loss + uniformity_loss)')

    device = torch.device('cuda')

    header('Training')

    all_techniques, technique_to_col, train_entries, test_entries, _templates = load_entries_and_templates(
        args, sequences_dir=SEQUENCES_RAWFULL_DIR)

    step_to_path, graph_model, embed_graph, wide_dim = setup_zoomer_graph_encoder(train_entries, device)
    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    def embed_entry_train(e):
        return embed_graph(step_to_path[e['file']])

    def embed_entry_test(e):
        path = step_to_path.get(e['file'])
        return None if path is None else embed_graph(path)

    trainable_params = list(graph_model.parameters()) + [logit_scale]
    technique_pools = camlds_train.build_technique_pools(train_entries, all_techniques)
    modules = {'graph_model': graph_model}
    ckpt_path = os.path.join(CHECKPOINTS_DIR, 'none_zoomer_camlds_matcher_{}.pt'.format(run_tag))

    if args.skip_train:
        history, best_loss = load_checkpoint(ckpt_path, modules, logit_scale=logit_scale)
    else:
        history, best_loss = run_episodic_training(
            args, embed_entry_train, technique_pools, train_entries, all_techniques, modules, trainable_params,
            use_amp=False, loss_style='mymodel', logit_scale=logit_scale)
        save_checkpoint(ckpt_path, modules, extra={
            'logit_scale': logit_scale, 'techniques': all_techniques, 'best_loss': best_loss,
            'history': history, 'run_tag': run_tag, 'test_file': args.test_file, 'wide_dim': wide_dim,
        })
        save_train_history(run_tag, history)

    header('Testing')
    for m in modules.values():
        m.eval()
    display_scale = logit_scale.exp().item()
    with torch.no_grad():
        prototypes, used_techniques = build_final_episodic_prototypes(
            embed_entry_train, technique_pools, train_entries, all_techniques)
        zt_all = F.normalize(prototypes, dim=-1)
        results = mymodel_style_build_results(test_entries, used_techniques, embed_entry_test, zt_all, display_scale)

    out = camlds_test.print_table(results, used_techniques)
    out = mymodel_style_test_rollup(out, results, used_techniques, train_entries)
    out.update({'run_tag': run_tag, 'test_file': args.test_file})
    save_results_json(run_tag, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--template-encoding', choices=['mymodel', 'knowhow', 'zoomer', 'none'], default='mymodel',
                     help='"none" means no template at all -- prototypes are the average of a few real '
                          'training examples instead (episodic, ZOOMER-style construction either way); '
                          '--training then picks the loss used against those prototypes.')
    ap.add_argument('--graph-encoding', choices=['mymodel', 'knowhow', 'zoomer'], default='mymodel')
    ap.add_argument('--training', choices=['mymodel', 'zoomer', 'none'], default='mymodel')
    ap.add_argument('--test-file', type=str, default=DEFAULT_TEST_FILE,
                     help='Held out as test; everything else trains. Default: {}'.format(DEFAULT_TEST_FILE))
    ap.add_argument('--run-tag', type=str, default=None,
                     help='Default: <template>_<graph>_<training>_<test-file>')
    ap.add_argument('--max-chunks', type=int, default=DEFAULT_MAX_CHUNKS)
    ap.add_argument('--n-epochs', type=int, default=None,
                     help='Default: {} for training=zoomer (matching ZOOMER\'s own N_EPISODES), '
                          '{} otherwise (matching your own N_EPOCHS).'.format(
                              ZOOMER_N_EPISODES, camlds_train.N_EPOCHS))
    ap.add_argument('--skip-train', action='store_true',
                     help='Skip training and reuse the existing checkpoint for this run_tag -- test only.')
    ap.add_argument('--graph-sequences', choices=['raw', 'generalized'], default='raw',
                     help='Only affects graph-encoding=mymodel: raw sequence text (sequences_rawfull/, '
                          'default) vs generalized (sequences/).')
    ap.add_argument('--sequences-dir-override', type=str, default=None,
                     help='Only affects graph-encoding=mymodel: read graph/sequence text from this exact '
                          'folder instead, overriding --graph-sequences entirely (e.g. sequences_dev/).')
    ap.add_argument('--exclude-from-training', type=str, default=None,
                     help='Comma-separated exact step id(s) to additionally drop from training, on top of '
                          'the normal --test-file holdout -- use this to remove training steps that are '
                          'byte-identical duplicates of a held-out test step under a different scenario name '
                          '(e.g. "6_macro_cron-3,6_macro_cron-4").')
    ap.add_argument('--k-per-technique', type=int, default=4,
                     help='How many training examples per technique go into each batch. Default: 4.')
    ap.add_argument('--use-wide', action='store_true',
                     help='Only affects graph-encoding=mymodel combos: also concatenate ZOOMER\'s wide features '
                          'onto the sequence embedding (default off).')
    ap.add_argument('--wide-features', type=str, default='all', choices=['all', 'node_edge', 'v2'],
                     help='Only used when --use-wide is set, on combos that support it (currently '
                          'template=knowhow graph=mymodel training=mymodel). "all" = original ZOOMER wide '
                          'pipeline (default). "v2" = the new abstract_node_v2/discretize_features_v2 '
                          'pipeline (updated file/process categories, 20 HNODE_DIMS instead of 18).')
    ap.add_argument('--trainable-knowhow-templates', action='store_true',
                     help='Only affects template=knowhow combos: make each technique\'s KnowHow template '
                          'vector a trainable parameter (initialized from FastText\'s fixed word-average) '
                          'instead of keeping it frozen. Ablation to test whether letting the template '
                          'side move during training helps or hurts, mirroring how template=mymodel '
                          '(RoBERTa) trains its own template encoder end to end.')
    ap.add_argument('--freeze-template-encoder', action='store_true',
                     help='Only affects template=mymodel (RoBERTa) combos: fully freeze the template '
                          'RoBERTa encoder (no training at all on it) so only the text_proj adapter on '
                          'top of it learns -- the reverse ablation of --trainable-knowhow-templates, '
                          'mirroring template=knowhow\'s fixed FastText + one trainable adapter structure.')
    ap.add_argument('--output-dir-override', type=str, default=None,
                     help='Write checkpoints and results under <this path>/checkpoint_combinations/ and '
                          '<this path>/combination_results/ instead of the shared top-level folders -- use '
                          'this to keep a different scenario\'s runs (e.g. scenario 4) fully separate.')
    args = ap.parse_args()

    if args.output_dir_override:
        global CHECKPOINTS_DIR, COMBINE_RESULTS_DIR
        CHECKPOINTS_DIR = os.path.join(args.output_dir_override, 'checkpoint_combinations')
        COMBINE_RESULTS_DIR = os.path.join(args.output_dir_override, 'combination_results')
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(COMBINE_RESULTS_DIR, exist_ok=True)
        print('--output-dir-override: checkpoints -> {}  results -> {}'.format(CHECKPOINTS_DIR, COMBINE_RESULTS_DIR))

    if args.n_epochs is None:
        args.n_epochs = ZOOMER_N_EPISODES if args.training == 'zoomer' else camlds_train.N_EPOCHS

    camlds_train.K_PER_TACTIC = args.k_per_technique

    key = (args.template_encoding, args.graph_encoding, args.training)
    fn = COMBINATIONS.get(key)
    if fn is None:
        print('Combination not built yet: template={} graph={} training={}'.format(*key))
        print('Built so far: {}'.format(sorted(COMBINATIONS.keys())))
        sys.exit(1)

    default_run_tag = 'graph_encoder_{}_template_encoder_{}_training_{}_{}'.format(
        args.graph_encoding, args.template_encoding, args.training, args.test_file)
    if args.graph_encoding == 'mymodel' and args.graph_sequences == 'generalized':
        default_run_tag += '_graphseq_generalized'
    if args.use_wide:
        default_run_tag += '_wide'
    if args.k_per_technique is not None:
        default_run_tag += '_k{}'.format(args.k_per_technique)
    run_tag = args.run_tag or default_run_tag
    print('Combination: template={} graph={} training={}  (run_tag={})'.format(*key, run_tag))

    fn(args, run_tag)

    print()
    print('Done.')

if __name__ == '__main__':
    main()
