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
                             OUTPUT_EMBEDDINGS, OUTPUT_BENIGN, OUTPUT_CTI)

TEST_BENIGN_KEYS  = ['benign_8']
TEST_MAL_ORIG     = ['file1_window']

RETRIEVAL_CTI_KEYS = [
    'theia1',
    'theia2',
    'theia3',
    'file1',
]

TOP_K = len(RETRIEVAL_CTI_KEYS)

FILIP_SEED_ONLY = True

_TRAIN_MAL_KEYS    = ['window_1', 'window_2', 'window_3']
_TRAIN_CTI_KEYS    = ['theia1',   'theia2',   'theia3']
_TRAIN_BENIGN_KEYS = ['benign_1', 'benign_2', 'benign_3',
                      'benign_4', 'benign_5', 'benign_6']

_FILIP_PUNCT = frozenset({
    '-', '.', ',', '!', '?', ';', ':', "'", '"',
    '(', ')', '[', ']', '{', '}', '/', '\\',
    '`', '~', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>',
    '|', '_', '\u2013', '\u2014',
})

def filip_sim_precomputed(z_G_nodes, z_T_tokens):
    sim_matrix    = z_G_nodes @ z_T_tokens.T
    node_max_sims = sim_matrix.max(dim=1).values
    return node_max_sims.mean().item()

def clip_explain(z_G_seeds, seed_texts, z_cti_vec, top_n=5):
    rows = []
    for emb, text in zip(z_G_seeds, seed_texts):
        sim = (emb.squeeze(0) @ z_cti_vec).item()
        rows.append({'seed_text': text, 'clip_sim': round(sim, 4)})
    rows.sort(key=lambda r: r['clip_sim'], reverse=True)
    return rows[:top_n]

def filip_explain(z_G_nodes, z_T_tokens, node_texts, cti_tokens, top_n=5):
    sim_matrix = z_G_nodes @ z_T_tokens.T
    rows = []
    for i in range(sim_matrix.shape[0]):
        node_sims   = sim_matrix[i]
        sorted_idxs = node_sims.argsort(descending=True).tolist()
        chosen_tok = (cti_tokens[sorted_idxs[0]]
                      if sorted_idxs[0] < len(cti_tokens)
                      else f'tok_{sorted_idxs[0]}')
        chosen_sim = node_sims[sorted_idxs[0]].item()
        for idx in sorted_idxs:
            tok_str = cti_tokens[idx] if idx < len(cti_tokens) else f'tok_{idx}'
            if not all(c in _FILIP_PUNCT for c in tok_str):
                chosen_tok = tok_str
                chosen_sim = node_sims[idx].item()
                break
        rows.append({
            'node_text'     : node_texts[i] if i < len(node_texts) else f'node_{i}',
            'best_cti_token': chosen_tok,
            'sim'           : round(chosen_sim, 4),
        })
    rows.sort(key=lambda r: r['sim'], reverse=True)
    return rows[:top_n]

def run_contrastive_test():
    os.chdir(PROJECT_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device : {device}')

    model_path = OUTPUT_CTI + 'contrastive_model.pt'
    if not os.path.exists(model_path):
        print(f'  ERROR: {model_path} not found — run contrastive_train first')
        return

    ckpt   = torch.load(model_path, map_location=device)
    g_dims = ckpt.get('g_proj_dims', (64,  256, 128))
    t_dims = ckpt.get('t_proj_dims', (768, 256, 128))

    g_proj = GraphProjector(*g_dims).to(device)
    t_proj = TextProjector (*t_dims).to(device)
    g_proj.load_state_dict(ckpt['g_proj'])
    t_proj.load_state_dict(ckpt['t_proj'])
    logit_scale = ckpt['logit_scale'].to(device)

    g_proj.eval(); t_proj.eval()
    print(f'  Loaded model  (best_train_loss={ckpt.get("best_train_loss","?"):.4f})')
    print(f'  Temperature   τ = {1.0/logit_scale.exp().item():.4f}')
    print()

    mal_graph_path    = OUTPUT_EMBEDDINGS + 'window_embeddings.pkl'
    benign_graph_path = OUTPUT_BENIGN     + 'benign_window_embeddings.pkl'
    cti_path          = OUTPUT_CTI        + 'cti_embeddings.pkl'

    for p in [mal_graph_path, benign_graph_path, cti_path]:
        if not os.path.exists(p):
            print(f'  ERROR: {p} not found'); return

    with open(mal_graph_path,    'rb') as f: window_emb_all = pkl.load(f)
    with open(benign_graph_path, 'rb') as f: benign_emb_all = pkl.load(f)
    with open(cti_path,          'rb') as f: cti_emb_all    = pkl.load(f)

    print(f'  Test malicious windows : {TEST_MAL_ORIG}')
    print(f'  Test benign windows    : {TEST_BENIGN_KEYS}')
    print()

    cti_readable_path = OUTPUT_CTI + 'cti_embeddings_readable.json'
    cti_token_strings = {}
    if os.path.exists(cti_readable_path):
        with open(cti_readable_path, 'r') as f:
            cti_readable = json.load(f)
        for k, v in cti_readable.items():
            toks = v.get('H_Ti', {}).get('tokens', [])
            cti_token_strings[k] = [t['token'] for t in toks]
        print(f'  CTI token strings loaded for {len(cti_token_strings)} keys')

    active_cti_keys = [k for k in RETRIEVAL_CTI_KEYS if k in cti_emb_all]
    missing_cti     = [k for k in RETRIEVAL_CTI_KEYS if k not in cti_emb_all]
    if missing_cti:
        print(f'  NOTE: {len(missing_cti)} CTI keys missing: {missing_cti}')
        print(f'  Add .txt files to input/cti_reports/ and re-run report_embedding.py')
        print()
    print(f'  Active CTI library : {len(active_cti_keys)} keys: {active_cti_keys}')
    print()

    with torch.no_grad():
        cti_lib_vecs  = torch.stack(
            [cti_emb_all[k] for k in active_cti_keys]).to(device)
        z_cti_lib     = F.normalize(t_proj(cti_lib_vecs), dim=-1)
        z_benign_text = F.normalize(
            t_proj(cti_emb_all['benign1'].unsqueeze(0).to(device)), dim=-1
        ).squeeze(0)

    k_cap = min(5, len(active_cti_keys))

    mal_node_path    = OUTPUT_EMBEDDINGS + 'node_embeddings.pkl'
    benign_node_path = OUTPUT_BENIGN     + 'benign_node_embeddings.pkl'
    cti_tok_path     = OUTPUT_CTI        + 'cti_token_embeddings.pkl'

    filip_available = all(os.path.exists(p)
                          for p in [mal_node_path, benign_node_path, cti_tok_path])

    if filip_available:
        with open(mal_node_path,    'rb') as f: mal_node_emb_all    = pkl.load(f)
        with open(benign_node_path, 'rb') as f: benign_node_emb_all = pkl.load(f)
        with open(cti_tok_path,     'rb') as f: cti_tok_emb_all     = pkl.load(f)
        print(f'  FILIP pkl loaded — {len(mal_node_emb_all)} mal windows, '
              f'{len(benign_node_emb_all)} benign, {len(cti_tok_emb_all)} CTI texts')
    else:
        print(f'  WARNING: FILIP pkl files missing — CLIP-only mode')

    all_graph_emb = {}
    all_graph_emb.update(benign_emb_all)
    all_graph_emb.update(window_emb_all)

    all_node_emb = {}
    if filip_available:
        all_node_emb.update(benign_node_emb_all)
        all_node_emb.update(mal_node_emb_all)

    filip_cti_proj    = {}
    filip_benign_proj = None

    active_filip_keys = [k for k in active_cti_keys
                         if filip_available and k in cti_tok_emb_all]
    if filip_available:
        with torch.no_grad():
            for k in set(active_filip_keys + _TRAIN_CTI_KEYS + ['benign1']):
                if k in cti_tok_emb_all:
                    h_T = cti_tok_emb_all[k].to(device)
                    filip_cti_proj[k] = F.normalize(t_proj(h_T), dim=-1)
        if 'benign1' in filip_cti_proj:
            filip_benign_proj = filip_cti_proj['benign1']
        else:
            filip_available = False

    benign_sg_nodes = {}
    benign_sg_path  = OUTPUT_BENIGN + 'benign_window_subgraphs.json'
    if os.path.exists(benign_sg_path):
        with open(benign_sg_path, 'r') as f:
            _bsg = json.load(f)
        for bk, bv in _bsg.items():
            benign_sg_nodes[bk] = {
                nd['uuid']: _build_node_text(nd)
                for nd in bv.get('subgraph_nodes', [])
            }
        print(f'  Benign node texts loaded for {len(benign_sg_nodes)} windows')
        print()

    test_cases = []
    for wk in TEST_MAL_ORIG:
        if wk in all_graph_emb:
            test_cases.append((wk, 'malicious', all_graph_emb, all_node_emb, 'THEIA'))
        else:
            print(f'  WARNING: {wk} not found in embeddings — skipping')

    for wk in TEST_BENIGN_KEYS:
        if wk in all_graph_emb:
            test_cases.append((wk, 'benign', all_graph_emb, all_node_emb, 'benign'))
        else:
            print(f'  WARNING: {wk} not found in embeddings — skipping')

    mode_str = 'FILIP+CLIP two-stage' if filip_available else 'CLIP-only'
    print(f'  Detection mode : {mode_str}')
    print(f'  ── Scores ({len(test_cases)} windows) ─────────────────────────────────')

    results = []

    with torch.no_grad():

        for wkey, truth_str, emb_dict, node_dict, dataset_tag in test_cases:

            g_vec           = emb_dict[wkey].unsqueeze(0).to(device)
            z_G_pool        = F.normalize(g_proj(g_vec), dim=-1).squeeze(0)
            sim_cti         = (z_G_pool @ z_cti_lib.T)
            sim_benign_clip = (z_G_pool @ z_benign_text).item()

            topk_vals, topk_idxs = torch.topk(sim_cti, k=k_cap, largest=True)
            clip_top_matches = [(active_cti_keys[i.item()], topk_vals[j].item())
                                for j, i in enumerate(topk_idxs)]
            clip_max_cti_sim = topk_vals[0].item()
            clip_score       = clip_max_cti_sim - sim_benign_clip

            use_filip = (filip_available and wkey in node_dict)

            filip_score       = clip_score
            filip_top_matches = clip_top_matches
            filip_max_cti_sim = clip_max_cti_sim
            fil_ben_sim       = sim_benign_clip
            z_G_nodes         = None
            all_node_texts    = []
            can_filip         = use_filip

            if can_filip:
                entries = node_dict.get(wkey)

                if isinstance(entries, list):
                    if FILIP_SEED_ONLY:
                        seed_emb_list, seed_text_list = [], []
                        for e in entries:
                            su = e.get('seed_uuid', '')
                            nm = e.get('node_map', {})
                            nt = e.get('node_texts', {})
                            si = nm.get(su)
                            sv = (e['node_emb'][si].unsqueeze(0)
                                  if si is not None and si < e['node_emb'].shape[0]
                                  else e['node_emb'].mean(0, keepdim=True))
                            seed_emb_list.append(sv)
                            seed_text_list.append(
                                nt.get(su, su[:8]) if su else 'seed')
                        node_emb       = torch.cat(seed_emb_list, dim=0).to(device)
                        all_node_texts = seed_text_list
                    else:
                        node_emb = torch.cat(
                            [e['node_emb'] for e in entries], dim=0).to(device)
                        for e in entries:
                            nm = e.get('node_map', {})
                            nt = e.get('node_texts', {})
                            for uuid, idx in sorted(nm.items(), key=lambda x: x[1]):
                                all_node_texts.append(nt.get(uuid, uuid[:8]))
                elif isinstance(entries, dict):
                    node_emb   = entries['node_emb'].to(device)
                    nm_ben     = entries.get('node_map', {})
                    uuid_order = sorted(nm_ben, key=lambda u: nm_ben[u])
                    uid2text   = benign_sg_nodes.get(wkey, {})
                    all_node_texts = [uid2text.get(u, u[:8]) for u in uuid_order]
                    if not all_node_texts:
                        all_node_texts = [f'node_{i}' for i in range(node_emb.shape[0])]
                else:
                    can_filip = False

                z_G_nodes = F.normalize(g_proj(node_emb), dim=-1)

                fil_cand_sims = [
                    (ck, filip_sim_precomputed(z_G_nodes, filip_cti_proj[ck]))
                    for ck, _ in clip_top_matches if ck in filip_cti_proj
                ]
                fil_ben_sim = filip_sim_precomputed(z_G_nodes, filip_benign_proj)

                if fil_cand_sims:
                    filip_top_matches = sorted(
                        fil_cand_sims, key=lambda x: x[1], reverse=True)[:3]
                    filip_max_cti_sim = filip_top_matches[0][1]
                    filip_score       = filip_max_cti_sim - fil_ben_sim

            final_score = filip_score if can_filip else clip_score

            clip_seed_explain_per = {}
            if isinstance(node_dict.get(wkey), list):
                entries = node_dict[wkey]
                z_seed_list, seed_labels = [], []
                for e in entries:
                    nm = e.get('node_map', {})
                    nt = e.get('node_texts', {})
                    su = e.get('seed_uuid', '')
                    se = e.get('node_emb')
                    if se is None: continue
                    si = nm.get(su)
                    sv = (se[si].unsqueeze(0).to(device)
                          if si is not None and si < se.shape[0]
                          else se.mean(0, keepdim=True).to(device))
                    z_seed_list.append(F.normalize(g_proj(sv), dim=-1))
                    seed_labels.append(nt.get(su, su[:8]) if su else 'unknown')
                if z_seed_list:
                    for ckey, _ in clip_top_matches:
                        if ckey in cti_emb_all:
                            z_cti_vec = F.normalize(
                                t_proj(cti_emb_all[ckey].unsqueeze(0).to(device)),
                                dim=-1).squeeze(0)
                            clip_seed_explain_per[ckey] = clip_explain(
                                z_seed_list, seed_labels, z_cti_vec, top_n=5)

            filip_node_explain_per = {}
            if can_filip and z_G_nodes is not None:
                for fw, _ in filip_top_matches:
                    if fw in filip_cti_proj:
                        tok_strings = cti_token_strings.get(fw, [])
                        filip_node_explain_per[fw] = filip_explain(
                            z_G_nodes, filip_cti_proj[fw],
                            all_node_texts, tok_strings, top_n=5)

            print()
            print(f'  ┌─ {wkey}  [{dataset_tag}]  │  truth: {truth_str}')
            print(f'  │')
            print(f'  │  Stage 1 — CLIP top-{k_cap} candidates:')
            for rank, (ckey, csim) in enumerate(clip_top_matches, 1):
                print(f'  │    #{rank:<2}  {ckey:<25}  clip_sim={csim:.4f}')
                if ckey in clip_seed_explain_per:
                    print(f'  │       └─ Top contributing seeds:')
                    for r in clip_seed_explain_per[ckey]:
                        print(f'  │            {r["clip_sim"]:+.4f}  {r["seed_text"]}')

            if can_filip and filip_top_matches:
                print(f'  │')
                print(f'  │  Stage 2 — FILIP top-{len(filip_top_matches)} (token-wise re-rank):')
                for rank, (fw, fs) in enumerate(filip_top_matches, 1):
                    print(f'  │    #{rank}  {fw:<25}  filip_sim={fs:.4f}')
                    if fw in filip_node_explain_per:
                        print(f'  │       └─ Top node→token matches:')
                        for r in filip_node_explain_per[fw]:
                            print(f'  │            sim={r["sim"]:.4f}  '
                                  f'[{r["node_text"]}]  →  "{r["best_cti_token"]}"')
                print(f'  │')
                print(f'  │  filip_score  = {filip_score:.4f}  '
                      f'(winner_sim={filip_max_cti_sim:.4f} - '
                      f'benign_sim={fil_ben_sim:.4f})')
            else:
                print(f'  │  clip_score   = {clip_score:.4f}  '
                      f'benign_sim={sim_benign_clip:.4f}')

            attr = (filip_top_matches[0][0] if can_filip and filip_top_matches
                    else clip_top_matches[0][0])
            print(f'  └─ Final score = {final_score:.4f}  '
                  f'(best match → {attr})')

            results.append({
                'window'              : wkey,
                'dataset'             : dataset_tag,
                'truth'               : truth_str,
                'clip_score'          : round(clip_score,        6),
                'clip_max_cti_sim'    : round(clip_max_cti_sim,  6),
                'clip_top_k'          : [{'cti_key': nm, 'clip_sim': round(sv, 6)}
                                          for nm, sv in clip_top_matches],
                'clip_seed_explain'   : clip_seed_explain_per,
                'filip_score'         : round(filip_score,        6),
                'filip_max_cti_sim'   : round(filip_max_cti_sim,  6),
                'filip_benign_sim'    : round(fil_ben_sim,         6),
                'filip_top_k_reranked': [{'cti_key': nm, 'filip_sim': round(sv, 6)}
                                          for nm, sv in (filip_top_matches
                                                         if can_filip else [])],
                'filip_node_explain'  : filip_node_explain_per,
                'sim_cti_clip'        : {k: round(v, 6) for k, v in
                                          zip(active_cti_keys, sim_cti.tolist())},
                'sim_benign_clip'     : round(sim_benign_clip, 6),
                'final_score'         : round(final_score,     6),
                'attribution'         : attr,
            })

    os.makedirs(OUTPUT_CTI, exist_ok=True)
    save_path = OUTPUT_CTI + 'test_results.json'
    with open(save_path, 'w') as f:
        json.dump({
            'test_mal_keys'    : TEST_MAL_ORIG,
            'test_benign_keys' : TEST_BENIGN_KEYS,
            'retrieval_cti_keys': active_cti_keys,
            'detection_mode'   : mode_str,
            'filip_available'  : filip_available,
            'per_window'       : results,
        }, f, indent=2)
    print()
    print(f'  ✅ test_results.json  →  {save_path}')
    print()
    show('detect.py — DONE')
    return results

run_detection = run_contrastive_test

if __name__ == '__main__':
    run_contrastive_test()