
import os
import sys
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import (
    ROBERTA_MODEL, MAX_LEN, STRIDE,
    OUTPUT_TRAINING,
)
from scripts.encoder_utils import embed_text

SEED        = 42
BATCH_SIZE  = 64
LR          = 1e-5
DROPOUT     = 0.5
N_EPOCHS    = 100
EMB_DIM     = 768
PROJ_DIM    = 128
PATIENCE    = 20
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEP_TO_CTI = {
    'browser_extension': {
        (19710, 9): [
            'dep19710_browser_extension_drakon',
            'dep19710_browser_extension_drakon_aug1',
            'dep19710_browser_extension_drakon_aug2',
            'dep19710_browser_extension_drakon_aug3',
        ],
        (19821, 3): [
            'dep19821_profile_implant',
            'dep19821_profile_implant_aug1',
            'dep19821_profile_implant_aug2',
            'dep19821_profile_implant_aug3',
        ],
        (22391, 0): [
            'dep22391_micro_apt_portscan',
            'dep22391_micro_apt_portscan_aug1',
            'dep22391_micro_apt_portscan_aug2',
            'dep22391_micro_apt_portscan_aug3',
        ],
    },
    'firefox_backdoor': {
        (19710, 0): [
            'firefox_backdoor_dep19710_part0_fluxbox',
            'firefox_backdoor_dep19710_part0_fluxbox_aug1',
            'firefox_backdoor_dep19710_part0_fluxbox_aug2',
            'firefox_backdoor_dep19710_part0_fluxbox_aug3',
        ],
        (19710, 1): [
            'firefox_backdoor_dep19710_part1_fluxbox',
            'firefox_backdoor_dep19710_part1_fluxbox_aug1',
            'firefox_backdoor_dep19710_part1_fluxbox_aug2',
            'firefox_backdoor_dep19710_part1_fluxbox_aug3',
        ],
        (19799, 0): [
            'firefox_backdoor_dep19799_part0_home_admin_clean',
            'firefox_backdoor_dep19799_part0_home_admin_clean_aug1',
            'firefox_backdoor_dep19799_part0_home_admin_clean_aug2',
            'firefox_backdoor_dep19799_part0_home_admin_clean_aug3',
        ],
        (19821, 0): [
            'firefox_backdoor_dep19821_part0_home_admin_profile',
            'firefox_backdoor_dep19821_part0_home_admin_profile_aug1',
            'firefox_backdoor_dep19821_part0_home_admin_profile_aug2',
            'firefox_backdoor_dep19821_part0_home_admin_profile_aug3',
        ],
    },
}

ALL_SCENARIOS = list(DEP_TO_CTI.keys())


class ProjectionNetwork(nn.Module):
    def __init__(self, in_dim=EMB_DIM, out_dim=PROJ_DIM, dropout=DROPOUT):
        super().__init__()
        self.fc1     = nn.Linear(in_dim, out_dim)
        self.fc2     = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        return h1 + h2


def info_nce_loss(z_L, z_T, logit_scale):
    scale  = logit_scale.exp()
    S      = scale * (z_L @ z_T.T)
    labels = torch.arange(len(z_L), device=z_L.device)
    L_l2t  = F.cross_entropy(S,   labels)
    L_t2l  = F.cross_entropy(S.T, labels)
    return (L_l2t + L_t2l) / 2


def run_contrastive_train():
    os.chdir(PROJECT_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('  Device   : {}'.format(device))
    print('  Scenarios: {}'.format(', '.join(ALL_SCENARIOS)))
    print()

    
    tok_path = os.path.join(OUTPUT_TRAINING, 'tokenized.pt')
    if not os.path.exists(tok_path):
        print('  ERROR: {} not found — run Stage 8 (tokenize) first'.format(tok_path))
        return
    print('  Loading pre-tokenized data from {}'.format(tok_path))
    tok_data = torch.load(tok_path, map_location='cpu')

    ben_ids      = tok_data['benign']['input_ids'][:10000]   
    ben_mask     = tok_data['benign']['attention_mask'][:10000]
    ben_cti_ids  = tok_data['benign_cti']['input_ids'][0] 
    ben_cti_mask = tok_data['benign_cti']['attention_mask'][0]
    n_benign     = len(ben_ids)

    cti_key_to_idx    = {k: i for i, k in enumerate(tok_data['cti_keys'])}
    atk_log_ids_list  = []
    atk_log_mask_list = []
    atk_cti_ids_list  = []
    atk_cti_mask_list = []

    for scenario in ALL_SCENARIOS:
        dep_map = DEP_TO_CTI[scenario]
        if scenario not in tok_data['attack']:
            print('  WARNING: {} not found in tokenized.pt'.format(scenario))
            continue
        atk_enc  = tok_data['attack'][scenario]['enc']
        atk_meta = tok_data['attack'][scenario]['meta']
        n_loaded = 0
        for seq_idx, meta in enumerate(atk_meta):
            dep_id   = meta['dep_id']
            part_idx = meta['part_idx']
            cti_keys = dep_map.get((dep_id, part_idx), [])
            if not cti_keys:
                print('  WARNING: no CTI for scenario={} dep_id={} part_idx={}'.format(
                    scenario, dep_id, part_idx))
                continue
            for cti_key in cti_keys:
                cti_idx = cti_key_to_idx.get(cti_key)
                if cti_idx is None:
                    print('  WARNING: CTI key {} not in tokenized.pt'.format(cti_key))
                    continue
                atk_log_ids_list.append(atk_enc['input_ids'][seq_idx])   
                atk_log_mask_list.append(atk_enc['attention_mask'][seq_idx])
                atk_cti_ids_list.append(tok_data['cti']['input_ids'][cti_idx])
                atk_cti_mask_list.append(tok_data['cti']['attention_mask'][cti_idx])
            n_loaded += 1
        print('  {} attack seqs loaded: {}'.format(scenario, n_loaded))

   

    n_attack           = len(atk_log_ids_list)
    n_benign_per_batch = BATCH_SIZE - n_attack

    print('  Attack sequences : {}'.format(n_attack))
    print('  Benign sequences : {}'.format(n_benign))
    print('  Attack per batch : {}'.format(n_attack))
    print('  Benign per batch : {}'.format(n_benign_per_batch))
    print()

    print('  Loading RoBERTa encoders...')
    tokenizer    = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    log_encoder  = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    text_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    # Freeze first 8 transformer layers (0-7) + embeddings; train layers 8-11
    def freeze_lower_layers(model, n_freeze=8):
        for param in model.embeddings.parameters():
            param.requires_grad = False
        for i in range(n_freeze):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = False

    freeze_lower_layers(log_encoder)
    freeze_lower_layers(text_encoder)


    log_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    text_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    log_encoder.train()
    text_encoder.train()
    print('  Encoders loaded — layers 0-7 frozen, layers 8-11 trainable, gradient checkpointing ON.')
    print()

    log_proj  = ProjectionNetwork(EMB_DIM, PROJ_DIM, DROPOUT).to(device)
    text_proj = ProjectionNetwork(EMB_DIM, PROJ_DIM, DROPOUT).to(device)

    logit_scale = nn.Parameter(
        torch.ones([], device=device) * np.log(1 / 0.07)
    )

    optimizer = torch.optim.Adam(
        [p for p in log_encoder.parameters()  if p.requires_grad] +
        [p for p in text_encoder.parameters() if p.requires_grad] +
        list(log_proj.parameters())     +
        list(text_proj.parameters())    +
        [logit_scale],
        lr=LR,
    )

   
    scaler = torch.amp.GradScaler('cuda')

    best_loss  = float('inf')
    no_improve = 0
    best_state = None
    history    = []

    train_start = time.time()
    print('  Training  epochs={} lr={} dropout={} batch={}'.format(
        N_EPOCHS, LR, DROPOUT, BATCH_SIZE))
    print('  Start time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print()

    for epoch in range(1, N_EPOCHS + 1):
        epoch_start = time.time()

        perm         = torch.randperm(n_benign).tolist()
        epoch_losses = []

        for start in range(0, n_benign, n_benign_per_batch):
            b_idx = perm[start: start + n_benign_per_batch]
            if not b_idx:
                continue

           
            b_log_seq  = [ben_ids[i]  for i in b_idx]
            b_log_msk  = [ben_mask[i] for i in b_idx]
            b_cti_seq  = [ben_cti_ids]  * len(b_idx)
            b_cti_msk  = [ben_cti_mask] * len(b_idx)

            # Pad attack + benign together (variable-length → same width per batch)
            all_log_seqs = atk_log_ids_list  + b_log_seq
            all_log_msks = atk_log_mask_list + b_log_msk
            all_cti_seqs = atk_cti_ids_list  + b_cti_seq
            all_cti_msks = atk_cti_mask_list + b_cti_msk

            batch_log_ids  = torch.nn.utils.rnn.pad_sequence(
                all_log_seqs, batch_first=True, padding_value=1).to(device)
            batch_log_mask = torch.nn.utils.rnn.pad_sequence(
                all_log_msks, batch_first=True, padding_value=0).to(device)
            batch_cti_ids  = torch.nn.utils.rnn.pad_sequence(
                all_cti_seqs, batch_first=True, padding_value=1).to(device)
            batch_cti_mask = torch.nn.utils.rnn.pad_sequence(
                all_cti_msks, batch_first=True, padding_value=0).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                v = embed_text(log_encoder,  tokenizer, batch_log_ids,  batch_log_mask, device, truncate=True)
                u = embed_text(text_encoder, tokenizer, batch_cti_ids,  batch_cti_mask, device, truncate=True)

                v = log_proj(v)
                u = text_proj(u)

                v = F.normalize(v, dim=-1)
                u = F.normalize(u, dim=-1)

                loss = info_nce_loss(v, u, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                logit_scale.clamp_(0, np.log(100))

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        history.append({
            'epoch'      : epoch,
            'loss'       : round(epoch_loss, 6),
            'logit_scale': round(logit_scale.exp().item(), 4),
        })

        epoch_time = (time.time() - epoch_start) / 60
        if epoch == 1 or epoch % 10 == 0:
            elapsed = (time.time() - train_start) / 60
            print('  epoch {:>4d}  loss={:.6f}  logit_scale={:.4f}  epoch_time={:.2f}m  total_time={:.2f}m'.format(
                epoch, epoch_loss, logit_scale.exp().item(), epoch_time, elapsed))

        if epoch_loss < best_loss - 1e-6:
            best_loss  = epoch_loss
            no_improve = 0
            best_state = {
                'log_encoder' : {k: v.clone() for k, v in log_encoder.state_dict().items()},
                'text_encoder': {k: v.clone() for k, v in text_encoder.state_dict().items()},
                'log_proj'    : {k: v.clone() for k, v in log_proj.state_dict().items()},
                'text_proj'   : {k: v.clone() for k, v in text_proj.state_dict().items()},
                'logit_scale' : logit_scale.detach().clone(),
            }
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print('  Early stopping at epoch {} (no improvement for {} epochs)'.format(
                    epoch, PATIENCE))
                break

    if best_state:
        log_encoder.load_state_dict(best_state['log_encoder'])
        text_encoder.load_state_dict(best_state['text_encoder'])
        log_proj.load_state_dict(best_state['log_proj'])
        text_proj.load_state_dict(best_state['text_proj'])
        with torch.no_grad():
            logit_scale.copy_(best_state['logit_scale'])

    total_time = (time.time() - train_start) / 60
    print()
    print('  Best train loss : {:.6f}'.format(best_loss))
    print('  logit_scale     : {:.4f}'.format(logit_scale.exp().item()))
    print('  End time        : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('  Total time      : {:.2f}m'.format(total_time))

    os.makedirs(OUTPUT_TRAINING, exist_ok=True)

    model_path = os.path.join(OUTPUT_TRAINING, 'theia.pt')
    torch.save({
        'log_encoder' : log_encoder.state_dict(),
        'text_encoder': text_encoder.state_dict(),
        'log_proj'    : log_proj.state_dict(),
        'text_proj'   : text_proj.state_dict(),
        'logit_scale' : logit_scale,
        'proj_dims'   : (EMB_DIM, PROJ_DIM),
        'scenarios'   : ALL_SCENARIOS,
        'best_loss'   : best_loss,
        'history'     : history,
        'seed'        : SEED,
    }, model_path)

    hist_path = os.path.join(OUTPUT_TRAINING, 'train_history_theia.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print('  theia.pt                 → {}'.format(model_path))
    print('  train_history_theia.json → {}'.format(hist_path))


if __name__ == '__main__':
    run_contrastive_train()
