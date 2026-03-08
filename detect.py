''"""
Paper attributions:
  CLIProv (2024)      — anomaly score formula, retrieval-based detection
  CLIP (Radford 2021) — zero-shot retrieval, L2-normalise + dot-product
"""
import os
import sys
import json
import pickle as pkl

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import (show,
                             OUTPUT_EMBEDDINGS, OUTPUT_BENIGN, OUTPUT_CTI, OUTPUT_TRAINING)
from scripts.train_detector import GraphProjector, TextProjector

# ── Test split
TEST_MAL_KEYS    = ['window_3']

TEST_BENIGN_KEYS = None   

# CTI retrieval library
RETRIEVAL_CTI_KEYS = [
    'theia1',
    'theia2',
    'theia3',
]


def run_contrastive_test():
    
    os.chdir(PROJECT_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device : {device}')

    model_path = OUTPUT_TRAINING + 'contrastive_model.pt'
    if not os.path.exists(model_path):
        print(f'  ERROR: {model_path} not found — run train_detector first')
        return

    ckpt        = torch.load(model_path, map_location=device)
    g_dims      = ckpt.get('g_proj_dims', (64,  256, 128))
    t_dims      = ckpt.get('t_proj_dims', (768, 256, 128))
    g_proj      = GraphProjector(*g_dims).to(device)
    t_proj      = TextProjector (*t_dims).to(device)
    g_proj.load_state_dict(ckpt['g_proj'])
    t_proj.load_state_dict(ckpt['t_proj'])
    logit_scale = ckpt['logit_scale'].to(device)
    g_proj.eval(); t_proj.eval()

    print(f'  Loaded model  (best_train_loss={ckpt.get("best_train_loss","?"):.4f})')
    print(f'  Temperature   τ = {1.0 / logit_scale.exp().item():.4f}')
    print()

  
    mal_graph_path    = OUTPUT_EMBEDDINGS + 'window_embeddings.pkl'
    benign_graph_path = OUTPUT_EMBEDDINGS + 'benign_window_embeddings.pkl'
    cti_path          = OUTPUT_CTI        + 'cti_embeddings.pkl'

    for p in [mal_graph_path, benign_graph_path, cti_path]:
        if not os.path.exists(p):
            print(f'  ERROR: {p} not found'); return

    with open(mal_graph_path,    'rb') as f: window_emb_all = pkl.load(f)
    with open(benign_graph_path, 'rb') as f: benign_emb_all = pkl.load(f)
    with open(cti_path,          'rb') as f: cti_emb_all    = pkl.load(f)


    mal_node_path    = OUTPUT_EMBEDDINGS + 'node_embeddings.pkl'
    benign_node_path = OUTPUT_EMBEDDINGS + 'benign_node_embeddings.pkl'
    node_emb_all = {}
    if os.path.exists(mal_node_path):
        with open(mal_node_path, 'rb') as f:
            node_emb_all.update(pkl.load(f))
    if os.path.exists(benign_node_path):
        with open(benign_node_path, 'rb') as f:
            node_emb_all.update(pkl.load(f))


    TEST_BENIGN_KEYS = [sorted(benign_emb_all.keys())[-1]]

    print(f'  Test malicious windows : {TEST_MAL_KEYS}')
    print(f'  Test benign windows    : {TEST_BENIGN_KEYS}')
    print()

    active_cti_keys = [k for k in RETRIEVAL_CTI_KEYS if k in cti_emb_all]
    missing_cti     = [k for k in RETRIEVAL_CTI_KEYS if k not in cti_emb_all]
    if missing_cti:
        print(f'  NOTE: CTI keys missing from pkl: {missing_cti}')
    print(f'  Active CTI library : {len(active_cti_keys)} keys: {active_cti_keys}')
    print()

     
    with torch.no_grad():
        cti_lib_vecs = torch.stack(
            [cti_emb_all[k] for k in active_cti_keys]).to(device)
        z_cti_lib    = F.normalize(t_proj(cti_lib_vecs), dim=-1)    

    k_cap = min(5, len(active_cti_keys))

   
    test_cases = []
    for wk in TEST_MAL_KEYS:
        if wk in window_emb_all:
            test_cases.append((wk, window_emb_all))
        else:
            print(f'  WARNING: {wk} not in window_embeddings.pkl — skipping')

    for wk in TEST_BENIGN_KEYS:
        if wk in benign_emb_all:
            test_cases.append((wk, benign_emb_all))
        else:
            print(f'  WARNING: {wk} not in benign_window_embeddings.pkl — skipping')

    print(f'  Detection mode : CLIP-only')
    print(f'  ── Scores ({len(test_cases)} windows) ─────────────────────────────────')

    results = []

    with torch.no_grad():
        for wkey, emb_dict in test_cases:

            # clip
            g_vec            = emb_dict[wkey].unsqueeze(0).to(device)
            z_G              = F.normalize(g_proj(g_vec), dim=-1).squeeze(0)
            sim_cti          = (z_G @ z_cti_lib.T)         

            topk_vals, topk_idxs = torch.topk(sim_cti, k=k_cap, largest=True)
            clip_top_matches = [(active_cti_keys[i.item()], topk_vals[j].item())
                                for j, i in enumerate(topk_idxs)]
            best_cti         = clip_top_matches[0][0]

           
            best_cti_idx = active_cti_keys.index(best_cti)
            z_cti_best   = z_cti_lib[best_cti_idx]          

            top_nodes = []
            if wkey in node_emb_all and node_emb_all[wkey]:
                raw   = node_emb_all[wkey]
                entry = raw[0] if isinstance(raw, list) else raw
                node_emb_t = entry['node_emb'].to(device)   
                node_map   = entry['node_map']             
                node_texts = entry.get('node_texts', {})
                z_nodes    = F.normalize(g_proj(node_emb_t), dim=-1)  
                sim_nodes  = (z_nodes @ z_cti_best)                
                top_n      = min(5, len(node_map))
                topk_vals_n, topk_idxs_n = torch.topk(sim_nodes, k=top_n)
                idx_to_uuid = {v: k for k, v in node_map.items()}
                for val, idx in zip(topk_vals_n, topk_idxs_n):
                    uuid = idx_to_uuid.get(idx.item(), '?')
                    text = node_texts.get(uuid, uuid[:20])
                    top_nodes.append((val.item(), uuid, text))

            # ── display ───────────────────────────────────────────────────────
            print()
            print(f'  ┌─ {wkey}')
            print(f'  │  CLIP top-{k_cap} candidates:')
            for rank, (ckey, csim) in enumerate(clip_top_matches, 1):
                print(f'  │    #{rank:<2}  {ckey:<25}  clip_sim={csim:+.4f}')
            print(f'  │  Best CTI → {best_cti}')
            if top_nodes:
                print(f'  │  Top nodes contributing to score:')
                for rank, (sim_val, uuid, text) in enumerate(top_nodes, 1):
                    print(f'  │    #{rank}  sim={sim_val:+.4f}  uuid={uuid}  {text}')
            print(f'  └─')

            results.append({
                'window'    : wkey,
                'clip_top_k': [{'cti_key': nm, 'clip_sim': round(sv, 6)}
                               for nm, sv in clip_top_matches],
                'best_cti'  : best_cti,
            })


    os.makedirs(OUTPUT_CTI, exist_ok=True)
    save_path = OUTPUT_CTI + 'test_results.json'
    with open(save_path, 'w') as f:
        json.dump({
            'test_mal_keys'     : TEST_MAL_KEYS,
            'test_benign_keys'  : TEST_BENIGN_KEYS,
            'retrieval_cti_keys': active_cti_keys,
            'per_window'        : results,
        }, f, indent=2)

    print()
    print(f' test_results.json  →  {save_path}')
    print()
    show('detect.py — DONE')
    return results


run_detection = run_contrastive_test

if __name__ == '__main__':
    run_contrastive_test()
