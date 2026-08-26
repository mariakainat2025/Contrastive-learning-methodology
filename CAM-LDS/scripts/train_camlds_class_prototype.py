
import os
import sys
import json
import math
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

PROJECT_ROOT = '/csse/research/contructive-learning'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scripts.config import ROBERTA_MODEL
from prototype_multilabel_loss import multilabel_prototype_loss
from test_split_presets import resolve_test_file_arg

from train_camlds_matcher import (
    ProjectionNetwork, encode_one, load_sequences, leave_out_split, random_split,
    build_pos_mask, build_tactic_pools, stratified_batch_indices, freeze_lower_layers,
    load_templates,
    CAM_LDS_DIR, RESULTS_DIR, OUTPUT_TRAINING,
    SEED, LR, DROPOUT, N_EPOCHS, EMB_DIM, PROJ_DIM, PATIENCE, N_FREEZE, K_PER_TACTIC,
)

BETA_EMA = 0.999         
RIDGE_EPS = 1e-3         


class LearnedClassPrototypes:

    def __init__(self, n_classes, dim, device, beta=BETA_EMA):
        cp = torch.empty(n_classes, dim)
        nn.init.orthogonal_(cp)          
        self.cp = F.normalize(cp, dim=-1).to(device)
        self.beta = beta
        self.n_classes = n_classes
        self.dim = dim

    @torch.no_grad()
    def update(self, z, pos_mask):
        L = pos_mask.float()
        LtL = L.T @ L                                              
        reg = RIDGE_EPS * torch.eye(LtL.size(0), device=LtL.device)
        LtL_inv = torch.linalg.pinv(LtL + reg)
        cp_star = LtL_inv @ L.T @ z.detach()                       
        cp_star = F.normalize(cp_star, dim=-1)

        has_signal = L.sum(dim=0) > 0                               
        new_cp = self.cp.clone()
        new_cp[has_signal] = self.beta * self.cp[has_signal] + (1 - self.beta) * cp_star[has_signal]
        self.cp = F.normalize(new_cp, dim=-1)
        return self.cp

    def state_dict(self):
        return {'cp': self.cp.clone(), 'beta': self.beta}

    def load_state_dict(self, state):
        self.cp = state['cp'].clone()
        self.beta = state['beta']


def run_contrastive_train(proto_mode='class', n_epochs=N_EPOCHS, test_file_match=None, patience=PATIENCE,
                           k_per_tactic=K_PER_TACTIC, test_size=0.2, split_seed=SEED, run_tag=None,
                           min_events=None, beta_ema=BETA_EMA):
    assert proto_mode in ('class', 'template'), "proto_mode must be 'class' or 'template'"

    device = torch.device('cuda')
    print('  Device     : {}'.format(device))
    print('  Proto mode : {}'.format(proto_mode))
    print()

    entries = load_sequences(min_events=min_events)

    all_tactics = sorted({t for e in entries for t in e['tactics']})
    tactic_to_col = {t: i for i, t in enumerate(all_tactics)}
    print('  Tactics (from full true multi-label membership of our {} steps): {}'.format(
        len(entries), ', '.join(all_tactics)))

    if run_tag is None:
        run_tag = test_file_match if test_file_match else 'seed{}'.format(split_seed)
    run_tag = '{}_{}'.format(run_tag, proto_mode)

    if test_file_match:
        train_entries, test_entries = leave_out_split(entries, test_file_match)
        print('  Split mode: leave-out — test = every file matching "{}"'.format(test_file_match))
    else:
        train_entries, test_entries = random_split(entries, test_size=test_size, seed=split_seed)
        print('  Split mode: random {:.0f}/{:.0f} (split_seed={})'.format(
            (1 - test_size) * 100, test_size * 100, split_seed))

    tactic_counts = {t: 0 for t in all_tactics}
    for e in train_entries:
        for t in e['tactics']:
            tactic_counts[t] += 1
    print('  Train sequences: {}  Test sequences: {}'.format(len(train_entries), len(test_entries)))
    for t in all_tactics:
        print('    {} : {} train'.format(t, tactic_counts[t]))
    if not train_entries:
        print('  ERROR: no training sequences found.')
        return

    print('  Loading RoBERTa encoder...')
    tokenizer   = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    seq_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    freeze_lower_layers(seq_encoder, n_freeze=N_FREEZE)
    seq_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    seq_encoder.train()

    log_proj = ProjectionNetwork(EMB_DIM, PROJ_DIM, DROPOUT).to(device)
    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    trainable_params = [p for p in seq_encoder.parameters() if p.requires_grad] + list(log_proj.parameters()) + [logit_scale]

    class_protos = None
    templates = tmpl_encoder = text_proj = None

    if proto_mode == 'class':
        class_protos = LearnedClassPrototypes(len(all_tactics), PROJ_DIM, device, beta=beta_ema)
        print('  Class prototypes: LEARNED (orthogonal init, ridge + EMA update, beta={})'.format(beta_ema))
    else:
        templates = load_templates()
        print('  Templates: {}'.format(', '.join(sorted(templates.keys()))))
        tmpl_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
        freeze_lower_layers(tmpl_encoder, n_freeze=N_FREEZE)
        tmpl_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        tmpl_encoder.train()
        text_proj = ProjectionNetwork(EMB_DIM, PROJ_DIM, DROPOUT).to(device)
        trainable_params += [p for p in tmpl_encoder.parameters() if p.requires_grad] + list(text_proj.parameters())

    print()

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    best_loss  = float('inf')
    no_improve = 0
    best_state = None
    history    = []

    train_start = time.time()
    batch_size = min(len(train_entries), k_per_tactic * len(all_tactics))
    print('  Training  epochs={} lr={} dropout={} patience={} batch_size={}'.format(
        n_epochs, LR, DROPOUT, patience, batch_size))
    print('  Start time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print()

    train_pos_mask = build_pos_mask(train_entries, all_tactics, tactic_to_col).to(device)
    tactic_pools = build_tactic_pools(train_entries, all_tactics)
    steps_per_epoch = max(1, -(-len(train_entries) // batch_size))
    rng = random.Random(split_seed)

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        seq_encoder.train(); log_proj.train()
        if proto_mode == 'template':
            tmpl_encoder.train(); text_proj.train()

        epoch_losses = []

        for _ in range(steps_per_epoch):
            batch_idxs = stratified_batch_indices(tactic_pools, all_tactics, k_per_tactic, rng)
            texts = [train_entries[i]['sequence'] for i in batch_idxs]
            pos_mask_batch = train_pos_mask[batch_idxs]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                z = log_proj(torch.stack([encode_one(tokenizer, seq_encoder, t, device) for t in texts]))

                if proto_mode == 'class':
                    prototypes = class_protos.update(z, pos_mask_batch)   
                else:
                    prototypes = text_proj(torch.stack(
                        [encode_one(tokenizer, tmpl_encoder, templates[t], device) for t in all_tactics]))

                loss = multilabel_prototype_loss(z, prototypes, pos_mask_batch, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                logit_scale.clamp_(0, math.log(100))

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        history.append({'epoch': epoch, 'loss': round(epoch_loss, 6), 'n_steps': len(epoch_losses),
                         'logit_scale': round(logit_scale.exp().item(), 4)})

        epoch_time = (time.time() - epoch_start) / 60
        elapsed    = (time.time() - train_start) / 60
        if epoch % 5 == 0 or epoch == 1:
            print('  epoch {:>4d}  loss={:.6f}  steps={}  logit_scale={:.4f}  epoch_time={:.2f}m  total_time={:.2f}m'.format(
                epoch, epoch_loss, len(epoch_losses), logit_scale.exp().item(), epoch_time, elapsed))

        if epoch_loss < best_loss - 1e-6:
            best_loss  = epoch_loss
            no_improve = 0
            best_state = {
                'seq_encoder' : {k: v.clone() for k, v in seq_encoder.state_dict().items()},
                'log_proj'    : {k: v.clone() for k, v in log_proj.state_dict().items()},
                'logit_scale' : logit_scale.detach().clone(),
                'epoch'       : epoch,
            }
            if proto_mode == 'class':
                best_state['class_protos'] = class_protos.state_dict()
            else:
                best_state['tmpl_encoder'] = {k: v.clone() for k, v in tmpl_encoder.state_dict().items()}
                best_state['text_proj']    = {k: v.clone() for k, v in text_proj.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print('  Early stopping at epoch {} (no improvement for {} epochs)'.format(epoch, patience))
                break

    if best_state:
        seq_encoder.load_state_dict(best_state['seq_encoder'])
        log_proj.load_state_dict(best_state['log_proj'])
        with torch.no_grad():
            logit_scale.copy_(best_state['logit_scale'])
        if proto_mode == 'class':
            class_protos.load_state_dict(best_state['class_protos'])
        else:
            tmpl_encoder.load_state_dict(best_state['tmpl_encoder'])
            text_proj.load_state_dict(best_state['text_proj'])

    total_time = (time.time() - train_start) / 60
    print()
    print('  Best train loss : {:.6f}'.format(best_loss))
    print('  logit_scale     : {:.4f}'.format(logit_scale.exp().item()))
    print('  End time        : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('  Total time      : {:.2f}m'.format(total_time))

    model_path = os.path.join(OUTPUT_TRAINING, 'camlds_classproto_matcher_{}.pt'.format(run_tag))
    save_dict = {
        'seq_encoder' : seq_encoder.state_dict(),
        'log_proj'    : log_proj.state_dict(),
        'logit_scale' : logit_scale,
        'proj_dims'   : (EMB_DIM, PROJ_DIM),
        'tactics'     : all_tactics,
        'proto_mode'  : proto_mode,
        'best_loss'   : best_loss,
        'best_epoch'  : best_state['epoch'] if best_state else None,
        'history'     : history,
        'seed'        : SEED,
        'split_seed'  : split_seed,
        'test_size'   : test_size,
        'run_tag'     : run_tag,
        'test_file_match': test_file_match,
    }
    if proto_mode == 'class':
        save_dict['class_prototypes'] = class_protos.cp.detach().cpu()
    else:
        save_dict['tmpl_encoder'] = tmpl_encoder.state_dict()
        save_dict['text_proj']    = text_proj.state_dict()
    torch.save(save_dict, model_path)

    hist_path = os.path.join(RESULTS_DIR, 'train_history_camlds_classproto_{}.json'.format(run_tag))
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print('  Checkpoint    → {}'.format(model_path))
    print('  Train history → {}'.format(hist_path))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto-mode', type=str, choices=['class', 'template'], default='class',
                     help="'class' = paper's learned class prototypes (orthogonal init + ridge/EMA update). "
                          "'template' = original text-template-derived prototypes, for comparison.")
    ap.add_argument('--k-per-tactic', type=int, default=K_PER_TACTIC)
    ap.add_argument('--test-file', type=str, default=None)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=SEED)
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--min-events', type=int, default=None)
    ap.add_argument('--beta-ema', type=float, default=BETA_EMA,
                     help='EMA decay for class prototype updates (only used in --proto-mode class).')
    args = ap.parse_args()
    test_files = resolve_test_file_arg(args.test_file)

    run_tag = args.run_tag
    if run_tag is None and args.test_file:
        run_tag = args.test_file

    run_contrastive_train(proto_mode=args.proto_mode, test_file_match=test_files, k_per_tactic=args.k_per_tactic,
                           test_size=args.test_size, split_seed=args.split_seed, run_tag=run_tag,
                           min_events=args.min_events, beta_ema=args.beta_ema)

