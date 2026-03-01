import os
import sys
import json
import pickle as pkl
import argparse
import torch
import torch.nn as nn
import dgl
from dgl.ops import edge_softmax
import dgl.function as fn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config        import show, OUTPUT_IOC, OUTPUT_GRAPHS, OUTPUT_FEATURES, OUTPUT_EMBEDDINGS
from scripts.feature_embed import build_node_text, BERTTextEncoder, EDGE_TYPE_TEXT

def build_args():
    parser = argparse.ArgumentParser(description='subgraph embedding')
    parser.add_argument('--hidden_dim',     type=int,   default=64)
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--n_heads',        type=int,   default=4)
    parser.add_argument('--feat_drop',      type=float, default=0.0)
    parser.add_argument('--attn_drop',      type=float, default=0.0)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--text_model',     type=str,   default='ehsanaghaei/SecureBERT',
    help='HuggingFace model id for BERT text encoding (f_T)')
    parser.add_argument('--device',         type=int,   default=0)
    return parser.parse_args([])

def nx_to_dgl(g_nx):
    if g_nx.number_of_edges() == 0:
        return None

    src_list, dst_list, etypes = [], [], []
    for u, v, _key, data in g_nx.edges(keys=True, data=True):
        src_list.append(u)
        dst_list.append(v)
        etypes.append(data['type'])

    g_dgl = dgl.graph((src_list, dst_list), num_nodes=g_nx.number_of_nodes())
    g_dgl.edata['type'] = torch.tensor(etypes, dtype=torch.long)
    return g_dgl


class GATConv(nn.Module):
    def __init__(self, in_dim, e_dim, out_dim, n_heads,
                 feat_drop=0.0, attn_drop=0.0, negative_slope=0.2,
                 residual=False, activation=None, norm=None, concat_out=True):
        super().__init__()
        self.n_heads    = n_heads
        self.out_feat   = out_dim
        self.concat_out = concat_out

        self.fc      = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.edge_fc = nn.Linear(e_dim,  out_dim * n_heads, bias=False)

        self.attn_h  = nn.Parameter(torch.FloatTensor(1, n_heads, out_dim))
        self.attn_t  = nn.Parameter(torch.FloatTensor(1, n_heads, out_dim))
        self.attn_e  = nn.Parameter(torch.FloatTensor(1, n_heads, out_dim))
        self.bias    = nn.Parameter(torch.FloatTensor(1, n_heads, out_dim))

        self.feat_drop  = nn.Dropout(feat_drop)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            self.res_fc = (nn.Linear(in_dim, out_dim * n_heads, bias=False)
                           if in_dim != out_dim * n_heads else nn.Identity())
        else:
            self.res_fc = None

        self.activation = activation
        self.norm       = norm(out_dim * n_heads) if norm else None
        self._reset()

    def _reset(self):
        gain = nn.init.calculate_gain('relu')
        for w in [self.fc.weight, self.edge_fc.weight,
                  self.attn_h, self.attn_t, self.attn_e]:
            nn.init.xavier_normal_(w, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g, feat):
        edge_feat = g.edata['attr'].float()
        with g.local_scope():
            h     = self.feat_drop(feat)
            feat_ = self.fc(h).view(-1, self.n_heads, self.out_feat)

            eh = (feat_ * self.attn_h).sum(-1, keepdim=True)
            et = (feat_ * self.attn_t).sum(-1, keepdim=True)

            feat_e = self.edge_fc(edge_feat).view(-1, self.n_heads, self.out_feat)
            ee     = (feat_e * self.attn_e).sum(-1, keepdim=True)

            g.srcdata.update({'hs': feat_, 'eh': eh})
            g.dstdata.update({'et': et})
            g.edata.update({'ee': ee})

            g.apply_edges(fn.u_add_e('eh', 'ee', 'ee'))
            g.apply_edges(fn.e_add_v('ee', 'et', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))

            g.update_all(fn.u_mul_e('hs', 'a', 'm'), fn.sum('m', 'hs'))
            rst = g.dstdata['hs'].view(-1, self.n_heads, self.out_feat) + self.bias

            if self.res_fc is not None:
                rst = rst + self.res_fc(h).view(-1, self.n_heads, self.out_feat)

            rst = rst.flatten(1) if self.concat_out else rst.mean(1)
            if self.norm:       rst = self.norm(rst)
            if self.activation: rst = self.activation(rst)
        return rst

class GAT(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, out_dim,
                 n_layers, n_heads,
                 feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        per_head    = hidden_dim // n_heads

        if n_layers == 1:
            self.layers.append(GATConv(
                n_dim, e_dim, out_dim // n_heads, n_heads,
                feat_drop, attn_drop, negative_slope,
                residual=residual, concat_out=True,
            ))
        else:
            self.layers.append(GATConv(
                n_dim, e_dim, per_head, n_heads,
                feat_drop, attn_drop, negative_slope,
                residual=residual, activation=nn.PReLU(),
                norm=nn.BatchNorm1d, concat_out=True,
            ))
            for _ in range(1, n_layers - 1):
                self.layers.append(GATConv(
                    hidden_dim, e_dim, per_head, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual=residual, activation=nn.PReLU(),
                    norm=nn.BatchNorm1d, concat_out=True,
                ))
            self.layers.append(GATConv(
                hidden_dim, e_dim, out_dim // n_heads, n_heads,
                feat_drop, attn_drop, negative_slope,
                residual=residual, concat_out=True,
            ))

    def forward(self, g, x):
        h = x
        for layer in self.layers:
            h = layer(g, h)
        return h   


def seed_node_embedding(node_emb, seed_idx):
    if seed_idx is not None:
        return node_emb[seed_idx].unsqueeze(0)  
    return node_emb.mean(0, keepdim=True)         


def mean_pool_embedding(node_emb):
    return node_emb.mean(0, keepdim=True)        


def main():
    args   = build_args()
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    )
    enc_dim = args.hidden_dim

    os.makedirs(OUTPUT_EMBEDDINGS, exist_ok=True)

    etd_path = OUTPUT_GRAPHS + 'edge_type_dict.json'
    if not os.path.exists(etd_path):
        print('edge_type_dict.json not found — run graph_builder first.')
        return
    with open(etd_path, 'r') as f:
        edge_type_dict = json.load(f)   
    print(f'edge_type_dict : {len(edge_type_dict)} edge types')

    # ── load seed graphs (NetworkX DiGraph, from graph_builder) ───────────
    sg_path = OUTPUT_GRAPHS + 'seed_graphs.pkl'
    if not os.path.exists(sg_path):
        print('seed_graphs.pkl not found — run graph_builder first.')
        return
    show('Loading seed_graphs.pkl ...')
    with open(sg_path, 'rb') as f:
        seed_nx_graphs = pkl.load(f)

    total_seeds = sum(len(v) for v in seed_nx_graphs.values())
    print(f'  Windows: {len(seed_nx_graphs)}   Total seeds: {total_seeds}')
    print()

    # ── load node attribute text (from window_subgraphs.json) ─────────────
    show('Loading window_subgraphs.json for node attribute text ...')
    with open(OUTPUT_IOC + 'window_subgraphs.json', 'r') as f:
        window_subgraphs = json.load(f)

 
    win_node_attrs = {
        wkey: {nd['uuid']: nd for nd in wdata.get('subgraph_nodes', [])}
        for wkey, wdata in window_subgraphs.items()
    }
    total_nodes = sum(len(v) for v in win_node_attrs.values())
    print(f'  Windows: {len(win_node_attrs)}   Total nodes with attributes: {total_nodes}')
    print()

    show('Building BERTTextEncoder (feature_embed.py) ...')
    text_enc = BERTTextEncoder(enc_dim, model_name=args.text_model).to(device)
    print(f'  text_dim → enc_dim  :  {text_enc.text_dim} → {enc_dim}')
    print(f'  proj params         :  {text_enc.proj.weight.numel():,}')
    print()

    show('Collecting unique node texts ...')
    all_texts      = set()
    win_node_texts = {}   # wkey → {uuid: text}
    for wkey in sorted(seed_nx_graphs.keys()):
        win_node_texts[wkey] = {}
        uid2attr = win_node_attrs.get(wkey, {})
        for entry in seed_nx_graphs[wkey]:
            for uuid in entry['node_map']:
                if uuid not in win_node_texts[wkey]:
                    nd   = uid2attr.get(uuid, {})
                    text = build_node_text(nd) if nd else 'unknown'
                    win_node_texts[wkey][uuid] = text
                    all_texts.add(text)
    for t in EDGE_TYPE_TEXT.values():
        all_texts.add(t)
    print(f'  Unique strings : {len(all_texts)}')

    show('SecureBERT-encoding all unique texts ...')
    text_enc.warm_cache(list(all_texts))
    print(f'  Cache size : {text_enc.cache_size}')

    show('Tokenizing all unique texts for inspection logs ...')
    token_info = {t: text_enc.tokenize(t) for t in sorted(all_texts)}
    print(f'  Tokenized  : {len(token_info)} unique strings')
    print()

    # ── build edge lookup table ───────────────────────────────────────────
    show('Building edge-type lookup table ...')
    edge_table = text_enc.build_edge_table(edge_type_dict, EDGE_TYPE_TEXT, device)
    print(f'  edge_table : {list(edge_table.shape)}')
    print()

    # ── build GAT encoder (f_G) ───────────────────────────────────────────
    encoder = GAT(
        n_dim=enc_dim, e_dim=enc_dim,
        hidden_dim=enc_dim, out_dim=enc_dim,
        n_layers=args.n_layers, n_heads=args.n_heads,
        feat_drop=args.feat_drop, attn_drop=args.attn_drop,
        negative_slope=args.negative_slope, residual=True,
    ).to(device)

    gat_params = sum(p.numel() for p in encoder.parameters())
    print(f'  GAT parameters : {gat_params:,}')
    print()

    # ── embedding loop ────────────────────────────────────────────────────
    show('Embedding seed subgraphs')
    encoder.eval()
    text_enc.eval()

    all_node_emb     = {}
    all_subgraph_emb = {}
    all_bert_emb     = {}  
    all_window_emb   = {}   
    summary          = []

    with torch.no_grad():
        for wkey in sorted(seed_nx_graphs.keys()):
            seeds = seed_nx_graphs[wkey]
            all_node_emb[wkey]     = []
            all_subgraph_emb[wkey] = []
            all_bert_emb[wkey]     = []
            print(f'  {wkey}  ({len(seeds)} seeds)')

            for entry in seeds:
                seed_uuid   = entry['seed_uuid']
                seed_reason = entry['seed_reason']
                node_map    = entry['node_map']   
                g_nx        = entry['graph']    

                # convert NetworkX → DGL
                g_dgl = nx_to_dgl(g_nx)
                if g_dgl is None:
                   
                    seed_text = win_node_texts[wkey].get(seed_uuid, 'unknown')
                    x = text_enc.encode_nodes([seed_text], device)   

                    all_bert_emb[wkey].append({
                        'seed_uuid'  : seed_uuid,
                        'seed_reason': seed_reason,
                        'node_map'   : {seed_uuid: 0},
                        'node_texts' : {seed_uuid: seed_text},
                        'bert_emb'   : x.cpu(),
                    })
                    all_node_emb[wkey].append({
                        'seed_uuid'  : seed_uuid,
                        'seed_reason': seed_reason,
                        'node_map'   : {seed_uuid: 0},
                        'node_texts' : {seed_uuid: seed_text},
                        'node_emb'   : x.cpu(),
                        'n_nodes'    : 1,
                        'n_edges'    : 0,
                    })
                    all_subgraph_emb[wkey].append({
                        'seed_uuid'    : seed_uuid,
                        'seed_reason'  : seed_reason,
                        'seed_node_emb': x.cpu(),
                        'subgraph_emb' : x.cpu(),
                        'seed_in_graph': True,
                        'n_nodes'      : 1,
                        'n_edges'      : 0,
                    })
                    summary.append({
                        'window'          : wkey,
                        'seed_uuid'       : seed_uuid,
                        'seed_reason'     : seed_reason,
                        'n_nodes'         : 1,
                        'n_edges'         : 0,
                        'seed_in_graph'   : True,
                        'node_emb_dim'    : enc_dim,
                        'subgraph_emb_dim': enc_dim,
                    })
                    continue

                g_dgl = g_dgl.to(device)

                n_nodes = g_dgl.num_nodes()
                n_edges = g_dgl.num_edges()

                # Order nodes by their integer index (same order as DGL graph)
                uuid_by_idx = sorted(node_map.keys(), key=lambda u: node_map[u])
                node_texts  = [
                    win_node_texts[wkey].get(u, 'unknown')
                    for u in uuid_by_idx
                ]
                x = text_enc.encode_nodes(node_texts, device)  

      
                all_bert_emb[wkey].append({
                    'seed_uuid'  : seed_uuid,
                    'seed_reason': seed_reason,
                    'node_map'   : node_map,
                    'node_texts' : dict(zip(uuid_by_idx, node_texts)),
                    'bert_emb'   : x.cpu(),     
                })

             
                g_dgl.edata['attr'] = edge_table[g_dgl.edata['type']]

            
                node_emb = encoder(g_dgl, x)

               
                seed_idx      = node_map.get(seed_uuid)
                seed_node_emb = seed_node_embedding(node_emb, seed_idx) 

              
                subgraph_emb  = mean_pool_embedding(node_emb)           

               
                all_node_emb[wkey].append({
                    'seed_uuid'  : seed_uuid,
                    'seed_reason': seed_reason,
                    'node_map'   : node_map,
                    'node_texts' : dict(zip(uuid_by_idx, node_texts)),
                    'node_emb'   : node_emb.cpu(),   
                    'n_nodes'    : n_nodes,
                    'n_edges'    : n_edges,
                })

                all_subgraph_emb[wkey].append({
                    'seed_uuid'    : seed_uuid,
                    'seed_reason'  : seed_reason,
                    'seed_node_emb': seed_node_emb.cpu(), 
                    'subgraph_emb' : subgraph_emb.cpu(),   
                    'seed_in_graph': seed_idx is not None,
                    'n_nodes'      : n_nodes,
                    'n_edges'      : n_edges,
                })

                summary.append({
                    'window'          : wkey,
                    'seed_uuid'       : seed_uuid,
                    'seed_reason'     : seed_reason,
                    'n_nodes'         : n_nodes,
                    'n_edges'         : n_edges,
                    'seed_in_graph'   : seed_idx is not None,
                    'node_emb_dim'    : enc_dim,
                    'subgraph_emb_dim': enc_dim,
                })

            # ── window-level embedding  ───
            if all_subgraph_emb[wkey]:
                sg_stack   = torch.cat(
                    [e['subgraph_emb'] for e in all_subgraph_emb[wkey]], dim=0
                )                                          
                window_emb = sg_stack.mean(0)            
            else:
                window_emb = torch.zeros(enc_dim)
            all_window_emb[wkey] = window_emb
            print(f'    window_emb shape : {list(window_emb.shape)}  '
                  f'(mean of {len(all_subgraph_emb[wkey])} seed subgraph embs)')

   
    os.makedirs(OUTPUT_FEATURES, exist_ok=True)
    print()
    show('Saving feature embeddings to OUTPUT_FEATURES ...')


    node_tokens_json = {
        wkey: {
            uuid: token_info[text]
            for uuid, text in uuid_text_map.items()
        }
        for wkey, uuid_text_map in win_node_texts.items()
    }
    with open(OUTPUT_FEATURES + 'node_tokens.json', 'w') as f:
        json.dump(node_tokens_json, f, indent=2)
    print('  ✅ node_tokens.json')


    edge_tokens_json = {
        etype: token_info[EDGE_TYPE_TEXT.get(etype, etype)]
        for etype in edge_type_dict
    }
    with open(OUTPUT_FEATURES + 'edge_tokens.json', 'w') as f:
        json.dump(edge_tokens_json, f, indent=2)
    print('  ✅ edge_tokens.json')


    with open(OUTPUT_FEATURES + 'node_bert_embeddings.pkl', 'wb') as f:
        pkl.dump(all_bert_emb, f)
    print('  ✅ node_bert_embeddings.pkl')


    with open(OUTPUT_FEATURES + 'edge_bert_table.pkl', 'wb') as f:
        pkl.dump(edge_table.cpu(), f)
    print('  ✅ edge_bert_table.pkl')


    idx2type      = sorted(edge_type_dict.keys(), key=lambda k: edge_type_dict[k])
    edge_bert_json = [
        {
            'edge_type': etype,
            'text'     : EDGE_TYPE_TEXT.get(etype, etype),
            'type_id'  : edge_type_dict[etype],
            'emb'      : [round(v, 4) for v in
                          edge_table.cpu()[edge_type_dict[etype]].tolist()],
        }
        for etype in idx2type
    ]
    with open(OUTPUT_FEATURES + 'edge_bert_table.json', 'w') as f:
        json.dump(edge_bert_json, f, indent=2)
    print('edge_bert_table.json')

    bert_json = {}
    for wkey, entries in all_bert_emb.items():
        bert_json[wkey] = [
            {
                'seed_uuid'  : e['seed_uuid'],
                'seed_reason': e['seed_reason'],
                'nodes': [
                    {
                        'uuid' : uuid,
                        'text' : e['node_texts'][uuid],
                        'emb'  : [round(v, 4) for v in
                                  e['bert_emb'][e['node_map'][uuid]].tolist()],
                    }
                    for uuid in sorted(e['node_map'], key=lambda u: e['node_map'][u])
                ],
            }
            for e in entries
        ]
    with open(OUTPUT_FEATURES + 'node_bert_embeddings.json', 'w') as f:
        json.dump(bert_json, f, indent=2)
    print(' node_bert_embeddings.json')

    # ── save OUTPUT_EMBEDDINGS (f_G outputs: GAT embeddings) ─────────────
    print()
    show('Saving GAT embeddings to OUTPUT_EMBEDDINGS ...')

    # 7. GAT node embeddings (binary)
    with open(OUTPUT_EMBEDDINGS + 'node_embeddings.pkl', 'wb') as f:
        pkl.dump(all_node_emb, f)
    print('  ✅ node_embeddings.pkl')

    # 8. GAT node embeddings (readable JSON)
    gat_json = {}
    for wkey, entries in all_node_emb.items():
        gat_json[wkey] = [
            {
                'seed_uuid'  : e['seed_uuid'],
                'seed_reason': e['seed_reason'],
                'n_nodes'    : e['n_nodes'],
                'n_edges'    : e['n_edges'],
                'nodes': [
                    {
                        'uuid': uuid,
                        'text': e['node_texts'][uuid],
                        'emb' : [round(v, 4) for v in
                                 e['node_emb'][e['node_map'][uuid]].tolist()],
                    }
                    for uuid in sorted(e['node_map'], key=lambda u: e['node_map'][u])
                ],
            }
            for e in entries
        ]
    with open(OUTPUT_EMBEDDINGS + 'node_embeddings.json', 'w') as f:
        json.dump(gat_json, f, indent=2)
    print('  ✅ node_embeddings.json')

    # 9. subgraph embeddings (binary)
    with open(OUTPUT_EMBEDDINGS + 'subgraph_embeddings.pkl', 'wb') as f:
        pkl.dump(all_subgraph_emb, f)
    print('  ✅ subgraph_embeddings.pkl')

    # 9b. subgraph embeddings (readable JSON)
    sg_emb_json = {}
    for wkey, entries in all_subgraph_emb.items():
        sg_emb_json[wkey] = [
            {
                'seed_uuid'    : e['seed_uuid'],
                'seed_reason'  : e['seed_reason'],
                'n_nodes'      : e['n_nodes'],
                'n_edges'      : e['n_edges'],
                'seed_in_graph': e['seed_in_graph'],
                'seed_node_emb': [round(v, 4) for v in e['seed_node_emb'][0].tolist()],
                'subgraph_emb' : [round(v, 4) for v in e['subgraph_emb'][0].tolist()],
            }
            for e in entries
        ]
    with open(OUTPUT_EMBEDDINGS + 'subgraph_embeddings.json', 'w') as f:
        json.dump(sg_emb_json, f, indent=2)
    print('  ✅ subgraph_embeddings.json')

    # 10. window embeddings (binary + readable JSON)
    with open(OUTPUT_EMBEDDINGS + 'window_embeddings.pkl', 'wb') as f:
        pkl.dump(all_window_emb, f)
    window_emb_json = {
        wkey: [round(v, 6) for v in emb.tolist()]
        for wkey, emb in all_window_emb.items()
    }
    with open(OUTPUT_EMBEDDINGS + 'window_embeddings.json', 'w') as f:
        json.dump(window_emb_json, f, indent=2)
    print('  ✅ window_embeddings.pkl / .json')

    # 11. summary
    with open(OUTPUT_EMBEDDINGS + 'embeddings_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('  ✅ embeddings_summary.json')

    # ── final summary ─────────────────────────────────────────────────────
    n_no_seed = sum(1 for s in summary if not s['seed_in_graph'])
    print()
    print('=' * 65)
    print(f'  Windows embedded        : {len(all_window_emb)}')
    print(f'  Total seeds embedded    : {len(summary)}')
    print(f'  Seeds with no edges     : {sum(1 for s in summary if s["n_edges"] == 0)}')
    print(f'  Seed not found in graph : {n_no_seed}  (fallback = mean pooling)')
    print()
    print('  ── feature_embed.py ─────────────────────────────')
    print(f'  BERT model   : {args.text_model}')
    print(f'  text_dim     : {text_enc.text_dim}  →  proj  →  enc_dim={enc_dim}')
    print(f'  node features: BERT(build_node_text(node_attrs))')
    print(f'  edge features: BERT(EDGE_TYPE_TEXT[edge_type])  via edge_table')
    print()
    print('  ── embed.py─────────────────────────────────────')
    print(f'  GAT  n_layers={args.n_layers}  n_heads={args.n_heads}  enc_dim={enc_dim}')
    print(f'  H_Gi shape        : [n_nodes, {enc_dim}]   (all 2-hop neighbourhood nodes)')
    print(f'  seed_node_emb     : [1, {enc_dim}]   H_Gi[seed_idx]  (reference)')
    print(f'  subgraph_emb      : [1, {enc_dim}]   mean(H_Gi)      (neighbourhood pool)')
    print(f'  window_emb        : [{enc_dim}]       mean(subgraph_emb per seed in window)')
    print('=' * 65)


if __name__ == '__main__':
    main()
