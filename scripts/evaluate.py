
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import (
    ROBERTA_MODEL, MAX_LEN, OUTPUT_TRAINING,
    OUTPUT_TEST, INPUT_TEST,
)
from scripts.encoder_utils import embed_text
from scripts.subgraph_sequence_builder import _rebuild_graph, extract_triples, triples_to_text
from transformers import RobertaTokenizer, RobertaModel

EMB_DIM  = 768
PROJ_DIM = 128
DROPOUT  = 0.5


TEST_SCENARIOS = [
    {
        'name'        : 'Phishing_Email_Link',
        'atk_file'    : 'attack_subgraphs_phishing_email_link.json',
        'ground_truth': 'phishing_email_link',
    },
    {
        'name'        : 'Phishing_Email_Executable_Attachment',
        'atk_file'    : 'attack_subgraphs_phishing_email_executable_attachment.json',
        'ground_truth': 'phishing_email_executable_attachment',
    },
]
TEST_BEN_SG_FILE = 'benign_subgraphs.json'


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


def tokenize_and_encode(tokenizer, encoder, proj, text, device):
    enc  = tokenizer(text, padding=False, truncation=False,
                     return_tensors='pt')
    ids  = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        emb = embed_text(encoder, tokenizer, ids, mask, device, truncate=False)
        emb = proj(emb)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu()


def sg_to_text(sg):
    G       = _rebuild_graph(sg)
    triples = extract_triples(G)
    text    = triples_to_text(triples)
    return ' '.join(text) if isinstance(text, list) else text


def evaluate():
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(OUTPUT_TRAINING, 'theia.pt')

    print('  Device: {}'.format(device))

    # ── Load trained model ────────────────────────────────────────────────────
    checkpoint   = torch.load(model_path, map_location=device)
    tokenizer    = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    log_encoder  = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    text_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    log_proj     = ProjectionNetwork().to(device)
    text_proj    = ProjectionNetwork().to(device)

    log_encoder.load_state_dict(checkpoint['log_encoder'])
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    log_proj.load_state_dict(checkpoint['log_proj'])
    text_proj.load_state_dict(checkpoint['text_proj'])

    log_encoder.eval()
    text_encoder.eval()
    log_proj.eval()
    text_proj.eval()
    print('  Model loaded.')
    print()

    cti_keys  = []
    cti_embs  = []
    for fname in sorted(os.listdir(INPUT_TEST)):
        if fname.endswith('.txt') and not fname.startswith('.'):
            key  = fname.replace('.txt', '')
            with open(os.path.join(INPUT_TEST, fname)) as f:
                text = f.read().strip()
            emb = tokenize_and_encode(tokenizer, text_encoder, text_proj, text, device)
            cti_keys.append(key)
            cti_embs.append(emb)
    cti_embs = torch.cat(cti_embs, dim=0)  # (n_cti, PROJ_DIM)
    print('  CTI reports loaded: {}'.format(len(cti_keys)))
    for k in cti_keys:
        print('    - {}'.format(k))
    print()

    with open(os.path.join(INPUT_TEST, TEST_BEN_SG_FILE)) as f:
        ben_data = json.load(f)
    ben_subgraphs = ben_data['subgraphs'] if isinstance(ben_data, dict) else ben_data

    ben_embs = []
    for sg in ben_subgraphs:
        text = sg_to_text(sg)
        emb  = tokenize_and_encode(tokenizer, log_encoder, log_proj, text, device)
        ben_embs.append(emb)
    ben_embs = torch.cat(ben_embs, dim=0)
    n_ben = len(ben_embs)
    print('  Benign subgraphs encoded: {}'.format(n_ben))
    print()

   
    all_atk_labels = []
    all_atk_embs   = [] 
    for scenario in TEST_SCENARIOS:
        with open(os.path.join(INPUT_TEST, scenario['atk_file'])) as f:
            atk_data = json.load(f)
        atk_subgraphs = atk_data['subgraphs'] if isinstance(atk_data, dict) else atk_data
        for sg in atk_subgraphs:
            text     = sg_to_text(sg)
            enc_tmp  = tokenizer(text, padding=False, truncation=False, return_tensors='pt')
            n_tokens = enc_tmp['input_ids'].shape[1]
            from scripts.config import MAX_LEN, STRIDE
            n_chunks = max(1, (n_tokens - MAX_LEN) // STRIDE + 2) if n_tokens > MAX_LEN else 1
            print('    dep={} part={}  tokens={:,}  chunks={}'.format(
                sg['dep_id'], sg.get('part_idx'), n_tokens, n_chunks))
            emb  = tokenize_and_encode(tokenizer, log_encoder, log_proj, text, device)
            all_atk_embs.append(emb)
            all_atk_labels.append('{} dep={} part={} seed={}'.format(
                scenario['name'], sg['dep_id'], sg['part_idx'], sg.get('seed_name', '')))
    all_atk_embs = torch.cat(all_atk_embs, dim=0)
    print('  Attack subgraphs encoded: {}'.format(len(all_atk_embs)))
    print()

   
    # atk_scores_matrix: (n_atk, n_cti)
    atk_scores_matrix = all_atk_embs @ cti_embs.T
    ben_scores_matrix = ben_embs @ cti_embs.T
    # Max score across all CTI reports (for threshold detection)
    ben_scores_max = ben_scores_matrix.max(dim=1).values

    threshold = 0.3

    print()
    print('  ── Benign subgraphs vs ALL CTI reports (max score) ──────────────')
    print()
    print('    max : {:.4f}'.format(ben_scores_max.max().item()))
    print('    min : {:.4f}'.format(ben_scores_max.min().item()))
    print()
    print('    Top 3 benign scores:')
    top10_vals, top10_idx = torch.topk(ben_scores_max, min(3, n_ben))
    for rank, (score, idx) in enumerate(zip(top10_vals, top10_idx), 1):
        sg      = ben_subgraphs[idx.item()]
        best_cti_idx = ben_scores_matrix[idx.item()].argmax().item()
        best_cti = cti_keys[best_cti_idx]
        print('      #{:02d}  sim={:.4f}  dep={}  part={}  seed={}  → {}'.format(
            rank, score.item(), sg.get('dep_id', '?'), sg.get('part_idx', '?'),
            sg.get('seed_name', '?'), best_cti))
    print()

   
    atk_scores_max = atk_scores_matrix.max(dim=1).values
    correct  = (ben_scores_max < threshold).sum().item()
    flagged  = (ben_scores_max >= threshold).sum().item()
    detected = (atk_scores_max >= threshold).sum().item()
    missed   = (atk_scores_max < threshold).sum().item()
    n_atk    = len(all_atk_embs)
    precision = detected / (detected + flagged) if (detected + flagged) > 0 else 0.0
    recall    = detected / (detected + missed)  if (detected + missed)  > 0 else 0.0

    # ── Ground truth vs Technique comparison ─────────────────────────────────
    # All technique/extra CTI keys (everything except ground truths)
    gt_keys_set = {s['ground_truth'] for s in TEST_SCENARIOS}
    EXCLUDE_CTI = {'apt32_network_scanning', 'green_lambert_persistence', 'mustard_tempest_driveby'}
    all_tq_keys = [k for k in cti_keys if k not in gt_keys_set and k not in EXCLUDE_CTI]

    print()
    print('  ── Ground Truth vs MITRE Technique Scores ───────────────────────')
    print()
    CTI_DISPLAY_NAMES = {
        'applejeus_spearphishing'   : 'Phishing',
         'Drive-by Compromise': 'Drive-by Compromise',
         'Event Triggered Execution': 'Event Triggered Execution',
         'Network Service Discovery': 'Network Service Discovery',
         'Content Injection': 'Content Injection',
         
    }
    COL = 16
    LBL = 40
    header = '  {:<{}}'.format('Attack Subgraph', LBL)
    header += '  {:>12}'.format('GT Score')
    for tq in all_tq_keys:
        display = CTI_DISPLAY_NAMES.get(tq, tq)[:COL]
        header += '  {:>{}}'.format(display, COL)
    print(header)
    print('  ' + '-' * (LBL + 12 + COL * len(all_tq_keys) + 4 * (len(all_tq_keys) + 1)))
    for i, label in enumerate(all_atk_labels):
        scenario  = TEST_SCENARIOS[i]
        gt_key    = scenario['ground_truth']
        gt_score  = atk_scores_matrix[i, cti_keys.index(gt_key)].item() if gt_key in cti_keys else -1
        row = '  {:<{}}'.format(label[:LBL], LBL)
        row += '  {:>12.4f}'.format(gt_score)
        for tq in all_tq_keys:
            tq_score = atk_scores_matrix[i, cti_keys.index(tq)].item() if tq in cti_keys else -1
            row += '  {:>{}.4f}'.format(tq_score, COL)
        print(row)
    print()

    print('  ── Detection (threshold={:.1f}, max across all CTI) ──────────────'.format(threshold))
    print('    Benign correctly classified : {}/{}'.format(correct, n_ben))
    print('    Attack correctly detected   : {}/{}'.format(detected, n_atk))
    print('    Precision : {:.4f}'.format(precision))
    print('    Recall    : {:.4f}'.format(recall))
    print()

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_TEST, exist_ok=True)
    out = {
        'cti_reports': cti_keys,
        'attacks': [
            {
                'label' : all_atk_labels[i],
                'scores': {cti_keys[j]: atk_scores_matrix[i, j].item() for j in range(len(cti_keys))},
            }
            for i in range(n_atk)
        ],
        'benign': {'max_score': ben_scores_max.max().item(), 'min_score': ben_scores_max.min().item()},
        'threshold' : threshold,
        'precision' : precision,
        'recall'    : recall,
    }
    out_path = os.path.join(OUTPUT_TEST, 'evaluation_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('  Saved: {}'.format(out_path))


if __name__ == '__main__':
    evaluate()
