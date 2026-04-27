
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
from scripts.node_abstraction import abstract_node_name
from transformers import RobertaTokenizer, RobertaModel, logging as hf_logging
hf_logging.set_verbosity_error()

EMB_DIM  = 768
PROJ_DIM = 128
DROPOUT  = 0.5


TEST_SCENARIOS = [
    {
        'name'        : 'Phishing_Email_Link_full_chain',
        'atk_file'    : 'phishing_link_attack_chain.json',
        'ground_truth': 'phishing_email_link',
    },
    {
        'name'        : 'attachment',
        'atk_file'    : 'phishing_executable_attachment_attack_chain.json',
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



def abstract_and_save(in_path, out_path):
    """Read subgraphs from original in_path, apply abstraction, save to out_path.
    Always abstracts from the original so test matches training abstraction rules."""
    with open(in_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        subgraphs = data
    elif 'subgraphs' in data:
        subgraphs = data['subgraphs']
    else:
        subgraphs = [data]

    print('  Abstracting: {}'.format(os.path.basename(in_path)))
    for sg in subgraphs:
        for entry in sg.get('nodes', []):
            node = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else entry
            if isinstance(node, dict) and 'name' in node and 'type' in node:
                node['name'] = abstract_node_name(node['name'], node['type'])

    with open(out_path, 'w') as f:
        json.dump({'total_subgraphs': len(subgraphs), 'subgraphs': subgraphs}, f, indent=2)
    print('  Saved: {}'.format(out_path))
    return subgraphs


def sg_to_text(sg):
    G       = _rebuild_graph(sg)
    triples = extract_triples(G)
    text    = triples_to_text(triples)
    return ' '.join(text) if isinstance(text, list) else text


CHECKPOINTS      = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
THRESHOLD        = 0.3
USE_ABSTRACTION  = False  # set True to abstract node names, False for raw

CTI_DISPLAY_NAMES = {
    'applejeus_spearphishing': 'Phishing',
    'Drive-by Compromise'    : 'Drive-by Compromise',
    'Event Triggered Execution': 'Event Triggered Execution',
    'Network Service Discovery': 'Network Service Discovery',
    'Content Injection'      : 'Content Injection',
}
EXCLUDE_CTI = {'apt32_network_scanning', 'green_lambert_persistence', 'mustard_tempest_driveby'}


def _load_model(ckpt_path, device):
    checkpoint   = torch.load(ckpt_path, map_location=device, weights_only=False)
    log_encoder  = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    text_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    log_proj     = ProjectionNetwork().to(device)
    text_proj    = ProjectionNetwork().to(device)
    log_encoder.load_state_dict(checkpoint['log_encoder'])
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    log_proj.load_state_dict(checkpoint['log_proj'])
    text_proj.load_state_dict(checkpoint['text_proj'])
    log_encoder.eval(); text_encoder.eval()
    log_proj.eval();    text_proj.eval()
    return log_encoder, text_encoder, log_proj, text_proj


def _load_texts(use_abstraction):
    """Load benign and attack texts with or without node abstraction."""
    if use_abstraction:
        ben_subgraphs = abstract_and_save(
            in_path=os.path.join(INPUT_TEST, TEST_BEN_SG_FILE),
            out_path=os.path.join(INPUT_TEST, 'abstracted_' + TEST_BEN_SG_FILE)
        )
    else:
        with open(os.path.join(INPUT_TEST, TEST_BEN_SG_FILE)) as f:
            data = json.load(f)
        ben_subgraphs = data['subgraphs'] if isinstance(data, dict) else data

    ben_texts  = [sg_to_text(sg) for sg in ben_subgraphs]
    atk_texts  = []
    atk_labels = []

    for scenario in TEST_SCENARIOS:
        if use_abstraction:
            atk_subgraphs = abstract_and_save(
                in_path=os.path.join(INPUT_TEST, scenario['atk_file']),
                out_path=os.path.join(INPUT_TEST, 'abstracted_' + scenario['atk_file'])
            )
        else:
            with open(os.path.join(INPUT_TEST, scenario['atk_file'])) as f:
                data = json.load(f)
            atk_subgraphs = data['subgraphs'] if (isinstance(data, dict) and 'subgraphs' in data) else [data]

        for sg in atk_subgraphs:
            atk_texts.append(sg_to_text(sg))
            atk_labels.append('{} dep={} part={} seed={}'.format(
                scenario['name'], sg['dep_id'], sg.get('part_idx', 0), sg.get('seed_name', '')))

    return ben_subgraphs, ben_texts, atk_texts, atk_labels


def _run_checkpoints(device, tokenizer, ben_texts, atk_texts, atk_labels, cti_keys, cti_texts):
    """Evaluate all checkpoints and return results list."""
    os.makedirs(OUTPUT_TEST, exist_ok=True)
    results = []

    print('  {:<30}  {:>12}  {:>12}  {:>14}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
        'Checkpoint', 'Link→Link', 'Attach→Attach', 'Link→Attach', 'Attach→Link', 'Precision', 'Recall', 'Ben Max', 'Ben Min'))
    print('  ' + '-' * 130)

    best_link_score = -1
    best_ckpt       = None

    for epoch in CHECKPOINTS:
        ckpt_path = os.path.join(OUTPUT_TRAINING, 'theia_epoch{}.pt'.format(epoch))
        if not os.path.exists(ckpt_path):
            continue

        ckpt_name = 'theia_epoch{}.pt'.format(epoch)
        log_encoder, text_encoder, log_proj, text_proj = _load_model(ckpt_path, device)

        cti_embs = torch.cat([tokenize_and_encode(tokenizer, text_encoder, text_proj, t, device)
                               for t in cti_texts], dim=0)
        ben_embs = torch.cat([tokenize_and_encode(tokenizer, log_encoder, log_proj, t, device)
                               for t in ben_texts], dim=0)
        atk_embs = torch.cat([tokenize_and_encode(tokenizer, log_encoder, log_proj, t, device)
                               for t in atk_texts], dim=0)

        atk_scores     = atk_embs @ cti_embs.T
        ben_scores_max = (ben_embs @ cti_embs.T).max(dim=1).values
        atk_scores_max = atk_scores.max(dim=1).values

        flagged   = (ben_scores_max >= THRESHOLD).sum().item()
        detected  = (atk_scores_max >= THRESHOLD).sum().item()
        missed    = (atk_scores_max <  THRESHOLD).sum().item()
        precision = detected / (detected + flagged) if (detected + flagged) > 0 else 0.0
        recall    = detected / (detected + missed)  if (detected + missed)  > 0 else 0.0
        ben_max   = ben_scores_max.max().item()
        ben_min   = ben_scores_max.min().item()

        # get per-scenario scores (ground truth + cross scores)
        link_score      = -1
        attach_score    = -1
        link_to_attach  = -1  # link subgraph vs attach CTI
        attach_to_link  = -1  # attach subgraph vs link CTI

        link_gt_key   = TEST_SCENARIOS[0]['ground_truth'] if len(TEST_SCENARIOS) > 0 else ''
        attach_gt_key = TEST_SCENARIOS[1]['ground_truth'] if len(TEST_SCENARIOS) > 1 else ''

        link_idx   = cti_keys.index(link_gt_key)   if link_gt_key   in cti_keys else -1
        attach_idx = cti_keys.index(attach_gt_key) if attach_gt_key in cti_keys else -1

        if 0 < len(atk_texts) and link_idx >= 0:
            link_score     = atk_scores[0, link_idx].item()
            attach_to_link = atk_scores[1, link_idx].item() if len(atk_texts) > 1 else -1
        if 1 < len(atk_texts) and attach_idx >= 0:
            attach_score   = atk_scores[1, attach_idx].item()
            link_to_attach = atk_scores[0, attach_idx].item()

        print('  {:<30}  {:>12.4f}  {:>12.4f}  {:>14.4f}  {:>14.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
            ckpt_name, link_score, attach_score, link_to_attach, attach_to_link,
            precision, recall, ben_max, ben_min))

        if link_score > best_link_score:
            best_link_score = link_score
            best_ckpt       = ckpt_name

        results.append({
            'epoch': epoch, 'checkpoint': ckpt_name,
            'link_score': link_score, 'attach_score': attach_score,
            'precision': precision, 'recall': recall,
            'ben_max': ben_max, 'ben_min': ben_min,
            'attacks': [{'label': atk_labels[i],
                         'scores': {cti_keys[j]: atk_scores[i, j].item() for j in range(len(cti_keys))}}
                        for i in range(len(atk_labels))]
        })

        del log_encoder, text_encoder, log_proj, text_proj
        torch.cuda.empty_cache()

    print()
    print('  Best checkpoint for phishing link: {} (score={:.4f})'.format(best_ckpt, best_link_score))
    return results


def evaluate():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    print('  Device: {}'.format(device))
    print()

    # load CTI reports
    cti_keys, cti_texts = [], []
    for fname in sorted(os.listdir(INPUT_TEST)):
        if fname.endswith('.txt') and not fname.startswith('.'):
            with open(os.path.join(INPUT_TEST, fname)) as f:
                cti_texts.append(f.read().strip())
            cti_keys.append(fname.replace('.txt', ''))
    print('  CTI reports: {}'.format(len(cti_keys)))
    print()

    # load test data
    mode = 'WITH Node Abstraction (stem + role: e.g. libc library, passwd config, tcexec download)' \
           if USE_ABSTRACTION else 'WITHOUT Node Abstraction (raw node names)'
    print('=' * 100)
    print('  {}'.format(mode))
    print('=' * 100)

    ben_subgraphs, ben_texts, atk_texts, atk_labels = _load_texts(use_abstraction=USE_ABSTRACTION)
    print('  Benign: {}  Attack: {}'.format(len(ben_texts), len(atk_texts)))
    print()

    # save sequences
    os.makedirs(INPUT_TEST, exist_ok=True)
    prefix = 'abstracted_' if USE_ABSTRACTION else ''
    with open(os.path.join(INPUT_TEST, f'{prefix}benign_sequences.json'), 'w') as f:
        json.dump([{'dep_id': ben_subgraphs[i].get('dep_id'), 'sequence': ben_texts[i]}
                   for i in range(len(ben_texts))], f, indent=2)
    with open(os.path.join(INPUT_TEST, f'{prefix}attack_sequences.json'), 'w') as f:
        json.dump([{'label': atk_labels[i], 'sequence': atk_texts[i]}
                   for i in range(len(atk_texts))], f, indent=2)

    os.makedirs(OUTPUT_TEST, exist_ok=True)
    results = _run_checkpoints(device, tokenizer, ben_texts, atk_texts,
                               atk_labels, cti_keys, cti_texts)

    print()
    out_path = os.path.join(OUTPUT_TEST, 'evaluation_results.json')
    with open(out_path, 'w') as f:
        json.dump({'mode': mode, 'results': results}, f, indent=2)
    print('  Saved: {}'.format(out_path))


if __name__ == '__main__':
    evaluate()