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
from scripts.deduplicate_sequence import deduplicate_sequence, _deduplicate_with_report, save_report, SIMILARITY_THRESHOLD
from scripts.abstract_cti import abstract_cti_text
from transformers import RobertaTokenizer, RobertaModel, logging as hf_logging
hf_logging.set_verbosity_error()

EMB_DIM  = 768
PROJ_DIM = 128
DROPOUT  = 0.5


TEST_SCENARIOS = [
    {
        'name'        : 'Phishing_Email_Link_full_chain',
        'display'     : 'Subgraph 1: Phishing_Email_Link',
        'atk_file'    : 'phishing_link_attack_chain.json',
        'ground_truth': 'phishing_email_link',
    },
    {
        'name'        : 'attachment',
        'display'     : 'Subgraph 2: Phishing_Email_Attachment',
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


CHECKPOINTS        = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
THRESHOLD          = 0.3
USE_ABSTRACTION    = True   # set True to abstract node names, False for raw
USE_DEDUPLICATION  = True   # remove exact duplicate sentences from sequences

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
    prefix = 'abstracted_' if use_abstraction else ''

    # ── Benign ────────────────────────────────────────────────────────────────
    original_ben_path = os.path.join(INPUT_TEST, TEST_BEN_SG_FILE)

    if use_abstraction:
        ben_subgraphs = abstract_and_save(
            in_path=original_ben_path,
            out_path=os.path.join(INPUT_TEST, 'abstracted_' + TEST_BEN_SG_FILE)
        )
    else:
        with open(original_ben_path) as f:
            data = json.load(f)
        ben_subgraphs = data['subgraphs'] if isinstance(data, dict) else data

    # Build sequences and save before dedup
    ben_raw_texts = [sg_to_text(sg) for sg in ben_subgraphs]
    abs_ben_file  = '{}benign_sequences.json'.format(prefix)
    abs_ben_path  = os.path.join(INPUT_TEST, abs_ben_file)
    with open(abs_ben_path, 'w') as f:
        json.dump([{'dep_id': ben_subgraphs[i].get('dep_id'), 'sequence': ben_raw_texts[i]}
                   for i in range(len(ben_raw_texts))], f, indent=2)
    print('  Saved: {}'.format(abs_ben_path))

    # Dedup from abstracted sequences file and save
    if USE_DEDUPLICATION:
        print('  Deduplicating: {}'.format(abs_ben_file))
    ben_texts = [deduplicate_sequence(t) if USE_DEDUPLICATION else t for t in ben_raw_texts]
    if USE_DEDUPLICATION:
        dedup_ben_path = os.path.join(INPUT_TEST, '{}benign_sequences_duplicate_removed.json'.format(prefix))
        with open(dedup_ben_path, 'w') as f:
            json.dump([{'dep_id': ben_subgraphs[i].get('dep_id'), 'sequence': ben_texts[i]}
                       for i in range(len(ben_texts))], f, indent=2)
        print('  Saved: {}'.format(dedup_ben_path))

    # ── Attack ────────────────────────────────────────────────────────────────
    atk_raw_texts = []
    atk_texts     = []
    atk_labels    = []

    for scenario in TEST_SCENARIOS:
        original_atk_path = os.path.join(INPUT_TEST, scenario['atk_file'])

        if use_abstraction:
            atk_subgraphs = abstract_and_save(
                in_path=original_atk_path,
                out_path=os.path.join(INPUT_TEST, 'abstracted_' + scenario['atk_file'])
            )
        else:
            with open(original_atk_path) as f:
                data = json.load(f)
            atk_subgraphs = data['subgraphs'] if (isinstance(data, dict) and 'subgraphs' in data) else [data]

        for sg in atk_subgraphs:
            atk_raw_texts.append(sg_to_text(sg))
            atk_labels.append('{} dep={} part={} seed={}'.format(
                scenario['name'], sg['dep_id'], sg.get('part_idx', 0), sg.get('seed_name', '')))

    # Save attack sequences before dedup
    abs_atk_file = '{}attack_sequences.json'.format(prefix)
    abs_atk_path = os.path.join(INPUT_TEST, abs_atk_file)
    with open(abs_atk_path, 'w') as f:
        json.dump([{'label': atk_labels[i], 'sequence': atk_raw_texts[i]}
                   for i in range(len(atk_raw_texts))], f, indent=2)
    print('  Saved: {}'.format(abs_atk_path))

    # Dedup from abstracted attack sequences file and save
    if USE_DEDUPLICATION:
        print('  Deduplicating: {}'.format(abs_atk_file))
    atk_texts = [deduplicate_sequence(t) if USE_DEDUPLICATION else t for t in atk_raw_texts]
    if USE_DEDUPLICATION:
        dedup_atk_path = os.path.join(INPUT_TEST, '{}attack_sequences_duplicate_removed.json'.format(prefix))
        with open(dedup_atk_path, 'w') as f:
            json.dump([{'label': atk_labels[i], 'sequence': atk_texts[i]}
                       for i in range(len(atk_texts))], f, indent=2)
        print('  Saved: {}'.format(dedup_atk_path))

    return ben_subgraphs, ben_texts, atk_texts, atk_labels


def _run_checkpoints(device, tokenizer, ben_texts, atk_texts, atk_labels, cti_keys, cti_texts):
    """Evaluate all checkpoints and return results list."""
    os.makedirs(OUTPUT_TEST, exist_ok=True)
    results = []
    TOP_K = 3  # show top-3 matches per attack in terminal

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
        ben_max        = ben_scores_max.max().item()
        ben_min        = ben_scores_max.min().item()

        # resolve ground truth key — prefer abstracted version if present
        link_gt_key = TEST_SCENARIOS[0]['ground_truth'] if len(TEST_SCENARIOS) > 0 else ''
        if link_gt_key not in cti_keys:
            link_gt_key = link_gt_key + '_abstracted'
        link_score = atk_scores[0, cti_keys.index(link_gt_key)].item() \
                     if link_gt_key in cti_keys and len(atk_texts) > 0 else -1

        # ── Terminal: top-3 per attack ────────────────────────────────────────
        print('  {} '.format(ckpt_name))
        for i, label in enumerate(atk_labels):
            display = TEST_SCENARIOS[i]['display'] if i < len(TEST_SCENARIOS) else label[:40]
            scores_i  = [(cti_keys[j], atk_scores[i, j].item()) for j in range(len(cti_keys))]
            scores_i.sort(key=lambda x: -x[1])
            top3 = '  |  '.join('{}:{:.4f}'.format(k, v) for k, v in scores_i[:TOP_K])
            print('    {}  →  {}'.format(display, top3))
        print()

        if link_score > best_link_score:
            best_link_score = link_score
            best_ckpt       = ckpt_name

        results.append({
            'epoch'     : epoch,
            'checkpoint': ckpt_name,
            'ben_max'   : ben_max,
            'ben_min'   : ben_min,
            'attacks'   : [{'label': atk_labels[i],
                            'scores': {cti_keys[j]: round(atk_scores[i, j].item(), 4)
                                       for j in range(len(cti_keys))}}
                           for i in range(len(atk_labels))]
        })

        del log_encoder, text_encoder, log_proj, text_proj
        torch.cuda.empty_cache()

    print()
    print('  Best checkpoint: {}  (Link score={:.4f})'.format(best_ckpt, best_link_score))
    return results


def evaluate():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    print('  Device: {}'.format(device))
    print()

    # load CTI reports — prefer _abstracted.txt if it exists, else abstract on the fly
    cti_keys, cti_texts = [], []
    for fname in sorted(os.listdir(INPUT_TEST)):
        if not fname.endswith('.txt') or fname.startswith('.') or fname.endswith('_abstracted.txt'):
            continue
        abs_fname = fname.replace('.txt', '_abstracted.txt')
        abs_path  = os.path.join(INPUT_TEST, abs_fname)
        if os.path.exists(abs_path):
            # use saved abstracted version — key uses abstracted filename
            with open(abs_path) as f:
                text = f.read().strip()
            key = abs_fname.replace('.txt', '')
            print('  [cti] using abstracted: {}'.format(abs_fname))
        else:
            # abstract on the fly and save if changed
            with open(os.path.join(INPUT_TEST, fname)) as f:
                raw = f.read().strip()
            text = abstract_cti_text(raw)
            if text != raw:
                with open(abs_path, 'w') as f:
                    f.write(text)
                key = abs_fname.replace('.txt', '')
                print('  [cti abstracted] {}  →  {}'.format(fname, abs_fname))
            else:
                key = fname.replace('.txt', '')
                print('  [cti] {}'.format(fname))
        cti_texts.append(text)
        cti_keys.append(key)
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

    os.makedirs(OUTPUT_TEST, exist_ok=True)
    results = _run_checkpoints(device, tokenizer, ben_texts, atk_texts,
                               atk_labels, cti_keys, cti_texts)

    # Save JSON
    print()
    out_path = os.path.join(OUTPUT_TEST, 'evaluation_results.json')
    with open(out_path, 'w') as f:
        json.dump({'mode': mode, 'cti_reports': cti_keys, 'results': results}, f, indent=2)
    print('  Saved: {}'.format(out_path))

    # Save human-readable text report
    txt_path = os.path.join(OUTPUT_TEST, 'evaluation_report.txt')
    lines = []
    lines.append('=' * 100)
    lines.append('  Evaluation Report')
    lines.append('  Mode: {}'.format(mode))
    lines.append('  CTI Reports: {}'.format(', '.join(cti_keys)))
    lines.append('=' * 100)
    for r in results:
        lines.append('')
        lines.append('  Checkpoint: {}  (Ben Max={:.4f}  Ben Min={:.4f})'.format(
            r['checkpoint'], r['ben_max'], r['ben_min']))
        for idx, atk in enumerate(r['attacks']):
            display = TEST_SCENARIOS[idx]['display'] if idx < len(TEST_SCENARIOS) else atk['label']
            lines.append('    {}'.format(display))
            sorted_scores = sorted(atk['scores'].items(), key=lambda x: -x[1])
            for rank, (k, v) in enumerate(sorted_scores, 1):
                marker = ' ←' if rank == 1 else ''
                lines.append('      {:2d}. {:45s}  {:.4f}{}'.format(rank, k, v, marker))
    lines.append('')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
    print('  Report: {}'.format(txt_path))


if __name__ == '__main__':
    evaluate()