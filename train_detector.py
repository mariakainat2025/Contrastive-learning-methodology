import os
import sys
import json
import random
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import show, OUTPUT_EMBEDDINGS, OUTPUT_BENIGN, OUTPUT_CTI, OUTPUT_TRAINING


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


class GraphProjector(nn.Module):
    """64-dim GAT window embedding → 128-dim shared contrastive space."""
    def __init__(self, in_dim=64, hidden_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TextProjector(nn.Module):
    """768-dim SecureBERT [CLS] embedding → 128-dim shared contrastive space."""
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)



def info_nce_loss(z_G, z_T, logit_scale, Display=False, label=''):
    """
    Bidirectional InfoNCE loss following paper equations (5), (6), (7).

    Paper formula:
      Loss_l2t = -∑ log( exp(vi^T ui / τ) / ∑ exp(vi^T uj / τ) )
      Loss_t2l = -∑ log( exp(ui^T vi / τ) / ∑ exp(ui^T vj / τ) )
      Loss     = (Loss_l2t + Loss_t2l) / 2


    Sources:
      CLIP (Radford et al., 2021):
        logits = img_feats @ text_feats.T * exp(temperature)
        labels = torch.arange(batch_size)
        loss   = CrossEntropyLoss(logits, labels)

      CLIProv (2024):
        Bidirectional version: L = (L_G2T + L_T2G) / 2
    """
    # Adapted from: https://github.com/openai/CLIP  (clip/model.py)
    #   logit_scale      = self.logit_scale.exp()
    #   logits_per_image = logit_scale * image_features @ text_features.t()
    #   logits_per_text  = logits_per_image.t()
    #   labels = torch.arange(n, device=device)
    #   loss_i = F.cross_entropy(logits_per_image, labels)
    #   loss_t = F.cross_entropy(logits_per_text,  labels)
    #   loss   = (loss_i + loss_t) / 2
    logit_scale_val  = logit_scale.exp()
    S                = logit_scale_val * (z_G @ z_T.T)
    labels           = torch.arange(len(z_G), device=z_G.device)
    L_G2T            = F.cross_entropy(S,   labels)
    L_T2G            = F.cross_entropy(S.T, labels)
    loss             = (L_G2T + L_T2G) / 2

    if Display:
        raw   = (z_G @ z_T.T).detach()
        S_d   = S.detach()
        names = ['benign', 'malicious']
        title = f'    InfoNCE loss step-by-step  {label}  (N={len(z_G)} batch)   '
        print(f'\n{title}')
        print(f'  Step 1 | logit_scale = exp({logit_scale.item():.4f}) = {logit_scale_val.item():.4f}')
        print(f'  Step 2 | Raw cosine similarity  z_G @ z_T.T:')
        for i in range(len(z_G)):
            row = '   '.join(f'{raw[i,j].item():+.4f}' for j in range(len(z_G)))
            print(f'           row {i} ({names[i]:>9} graph): [{row}]  →  col label = text[{i}] ({names[i]})')
        print(f'  Step 3 | Scaled S = {logit_scale_val.item():.4f} × raw:')
        for i in range(len(z_G)):
            row = '   '.join(f'{S_d[i,j].item():+.4f}' for j in range(len(z_G)))
            print(f'           row {i}: [{row}]')
        print(f'  Step 4 | Labels = {labels.tolist()}  (row i should prefer column i)')
        print(f'  Step 5 | L_G2T = CrossEntropy(S,   labels) = {L_G2T.item():.6f}')
        print(f'           L_T2G = CrossEntropy(S.T, labels) = {L_T2G.item():.6f}')
        print(f'  Step 6 | loss  = (L_G2T + L_T2G) / 2      = {loss.item():.6f}')
        print(f'  {"─" * (len(title) - 2)}\n')

    return loss

TRAIN_MAL_KEYS = ['window_1', 'window_1', 'window_1', 'window_1', 'window_2', 'window_2', 'window_2', 'window_2',]
TRAIN_CTI_KEYS = ['theia1', 'theia1_aug1', 'theia1_aug2', 'theia1_aug3', 'theia2' ,'theia2_aug1', 'theia2_aug2', 'theia2_aug3', ]

TRAIN_BENIGN_KEYS = None
def run_contrastive_train():
    """
    Paper attributions:
      CLIProv (2024)     — graph-text contrastive alignment, benign text strategy
      CLIP (Radford 2021)— InfoNCE loss, τ=0.07 initialization
      TRAP               — Adam lr=0.004, weight_decay=5e-4
    """

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    os.chdir(PROJECT_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device : {device}')
    print(f'  Seed   : {SEED}')

    #load embeddings
    mal_graph_path    = OUTPUT_EMBEDDINGS + 'window_embeddings.pkl'
    benign_graph_path = OUTPUT_EMBEDDINGS + 'benign_window_embeddings.pkl'
    cti_path          = OUTPUT_CTI        + 'cti_embeddings.pkl'

    for p in [mal_graph_path, benign_graph_path, cti_path]:
        if not os.path.exists(p):
            print(f'  ERROR: {p} not found — run earlier stages first')
            return

    with open(mal_graph_path,    'rb') as f: window_emb_all = pkl.load(f)
    with open(benign_graph_path, 'rb') as f: benign_emb_all = pkl.load(f)
    with open(cti_path,          'rb') as f: cti_emb_all    = pkl.load(f)

   
    all_benign_keys = sorted(benign_emb_all.keys())
    TRAIN_BENIGN_KEYS = all_benign_keys[:-1]   
    print(f'  Benign train keys : {TRAIN_BENIGN_KEYS}')
    print(f'  Benign test  key  : {all_benign_keys[-1]}  (excluded from training)')


    for k in set(TRAIN_MAL_KEYS):
        if k not in window_emb_all:
            print(f'  ERROR: {k} not in window_embeddings.pkl'); return
    for k in TRAIN_BENIGN_KEYS:
        if k not in benign_emb_all:
            print(f'  ERROR: {k} not in benign_window_embeddings.pkl'); return
    for k in TRAIN_CTI_KEYS + ['benign1']:
        if k not in cti_emb_all:
            print(f'  ERROR: {k} not in cti_embeddings.pkl'); return

    #  build training tensors
    mal_graph_train    = torch.stack(
        [window_emb_all[k] for k in TRAIN_MAL_KEYS]).to(device)
    benign_graph_train = torch.stack(
        [benign_emb_all[k] for k in TRAIN_BENIGN_KEYS]).to(device)
    mal_text_train     = torch.stack(
        [cti_emb_all[k]    for k in TRAIN_CTI_KEYS]).to(device)
    # one shared benign text for all benign windows (CLIProv)
    benign_text_vec    = cti_emb_all['benign1'].to(device)

    n_mal           = len(TRAIN_MAL_KEYS)
    n_benign        = len(TRAIN_BENIGN_KEYS)
    steps_per_epoch = n_mal * n_benign

    print(f'  mal_graph_train   : {list(mal_graph_train.shape)}  ({n_mal} pairs)')
    print(f'  benign_graph_train: {list(benign_graph_train.shape)}')
    print(f'  mal_text_train    : {list(mal_text_train.shape)}')
    print(f'  benign_text_vec   : {list(benign_text_vec.shape)}')
    print()

    #  model init
    torch.manual_seed(SEED)
    g_proj = GraphProjector(in_dim=64,  hidden_dim=256, out_dim=128).to(device)
    t_proj = TextProjector (in_dim=768, hidden_dim=256, out_dim=128).to(device)


    # Adapted from: https://github.com/openai/CLIP  (clip/model.py)
    #   self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    # Create directly on device — do NOT use .to() reassignment because
    # .to() on an nn.Parameter returns a plain Tensor, breaking grad tracking
    # and making results non-deterministic across runs.
    logit_scale = nn.Parameter(
        torch.ones([], device=device) * np.log(1 / 0.07)
    )

    #  optimizer — projectors + logit_scale 
    optimizer = torch.optim.Adam(
        list(g_proj.parameters()) + list(t_proj.parameters()) + [logit_scale],
        lr=1e-4, weight_decay=5e-4,
    )

    # training loop
    n_epochs   = 200
    patience   = 50
    best_loss  = float('inf')
    no_improve = 0
    best_state = None
    history    = []

    print(f'  n_epochs={n_epochs}  patience={patience}  '
          f'batch N=2  ({steps_per_epoch} steps/epoch = {n_mal} mal × {n_benign} benign)')
    print()

    g_proj.train(); t_proj.train()

    first_G_batch = None
    first_T_batch = None

    for epoch in range(1, n_epochs + 1):
        epoch_losses = []

        # Outer loop: each malicious (graph, text) pair
        for pair_idx in range(n_mal):
            m_graph = mal_graph_train[pair_idx].unsqueeze(0)
            m_text  = mal_text_train[pair_idx].unsqueeze(0)

            # Inner loop: each benign window
            for b_idx in range(n_benign):
                b_graph = benign_graph_train[b_idx].unsqueeze(0)
                b_text  = benign_text_vec.unsqueeze(0)

                # N=2 batch: [benign, malicious]
                G_batch = torch.cat([b_graph, m_graph], dim=0)
                T_batch = torch.cat([b_text,  m_text ], dim=0)


                if first_G_batch is None:
                    first_G_batch = G_batch.detach().clone()
                    first_T_batch = T_batch.detach().clone()

                z_G = F.normalize(g_proj(G_batch), dim=-1)
                z_T = F.normalize(t_proj(T_batch), dim=-1)

                Display = (epoch == 1 and pair_idx == 0 and b_idx == 0)
                loss = info_nce_loss(z_G, z_T, logit_scale, Display=Display,
                                     label='   Epoch 1 (before training)')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Clamp logit_scale to [0, log(100)] — CLIP convention to
                # prevent scale from exploding to infinity during training.
                with torch.no_grad():
                    logit_scale.clamp_(0, np.log(100))

                epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logit_scale_val = logit_scale.exp().item()
        history.append({'epoch': epoch, 'loss': round(epoch_loss, 6),
                        'logit_scale': round(logit_scale_val, 4)})

        if epoch == 1 or epoch % 20 == 0:
            print(f'  epoch {epoch:>4d}  loss={epoch_loss:.6f}  logit_scale={logit_scale_val:.4f}')

        # early stopping
        if epoch_loss < best_loss - 1e-6:
            best_loss  = epoch_loss
            no_improve = 0
            # Adapted from: https://github.com/openai/CLIP (clip/model.py)
            # CLIP stores logit_scale inside the model state_dict so it is
            # always checkpointed together with the encoder weights.
            # We replicate that by saving logit_scale alongside g_proj/t_proj.
            best_state = {
                'g_proj'      : {k: v.clone() for k, v in g_proj.state_dict().items()},
                't_proj'      : {k: v.clone() for k, v in t_proj.state_dict().items()},
                'logit_scale' : logit_scale.detach().clone(),   
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at epoch {epoch} '
                      f'(no improvement for {patience} epochs)')
                break


    if best_state:
        g_proj.load_state_dict(best_state['g_proj'])
        t_proj.load_state_dict(best_state['t_proj'])
        with torch.no_grad():
            logit_scale.copy_(best_state['logit_scale'])   

    print(f'\n  Best train loss  : {best_loss:.4f}')
    print(f'  logit_scale      : {logit_scale.exp().item():.4f}  (learned — started at 14.2857)')

            
    g_proj.eval(); t_proj.eval()
    with torch.no_grad():
        z_G_final = F.normalize(g_proj(first_G_batch), dim=-1)
        z_T_final = F.normalize(t_proj(first_T_batch), dim=-1)
        info_nce_loss(z_G_final, z_T_final, logit_scale, Display=True,
                      label='   Final model (best epoch)')

    #  save
    os.makedirs(OUTPUT_TRAINING, exist_ok=True)

    save_path = OUTPUT_TRAINING + 'contrastive_model.pt'
    torch.save({
        'g_proj'           : g_proj.state_dict(),
        't_proj'           : t_proj.state_dict(),
        'logit_scale'      : logit_scale,
        'g_proj_dims'      : (64,  256, 128),
        't_proj_dims'      : (768, 256, 128),
        'train_mal_keys'   : TRAIN_MAL_KEYS,
        'train_benign_keys': TRAIN_BENIGN_KEYS,
        'train_cti_keys'   : TRAIN_CTI_KEYS,
        'best_train_loss'  : best_loss,
        'history'          : history,
        'seed'             : SEED,
    }, save_path)
    print(f'contrastive_model.pt  →  {save_path}')

    hist_path = OUTPUT_TRAINING + 'train_history.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'train_history.json    →  {hist_path}')
    print()
    show('train_detector.py — DONE')


run_training = run_contrastive_train


if __name__ == '__main__':
    run_contrastive_train()