
import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import (
    ROBERTA_MODEL,
    OUTPUT_SEQUENCES, OUTPUT_TRAINING, OUTPUT_BENIGN, CTI_REPORTS_DIR,
)
from scripts.subgraph_sequence_builder import _rebuild_graph, extract_triples, triples_to_text
from transformers import RobertaTokenizer

SCENARIOS = {
    'browser_extension': 'Browser_Extension_Drakon_Dropper',
    'firefox_backdoor' : 'Firefox_Backdoor_Drakon_In_Memory',
    # phishing_email_credential_harvest is kept as test-only — not tokenized here
}

BENIGN_TEXT = 'This is a benign sequence.'
BATCH_SIZE  = 64


def tokenize_texts(tokenizer, texts, desc):
    all_ids   = []
    all_masks = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[i: i + BATCH_SIZE]
        enc   = tokenizer(
            batch,
            padding       = True,
            truncation    = True,
            max_length    = 512,
            return_tensors= 'pt',
        )
        for j in range(len(batch)):
            real_len = int(enc['attention_mask'][j].sum())
            all_ids.append(enc['input_ids'][j][:real_len].clone())
            all_masks.append(enc['attention_mask'][j][:real_len].clone())

    # Return as lists of 1-D tensors (variable length).
    # Padding to the longest sequence would create a huge tensor for 10k sequences.
    # The training loop pads each mini-batch on-the-fly instead.
    return {'input_ids': all_ids, 'attention_mask': all_masks}


def tokenize_all():
    os.makedirs(OUTPUT_TRAINING, exist_ok=True)

    print('  Loading tokenizer: {}'.format(ROBERTA_MODEL))
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)

    # ── Benign sequences ──────────────────────────────────────────────────────
    ben_path = os.path.join(OUTPUT_SEQUENCES, 'sequences_benign.json')
    with open(ben_path, 'r') as f:
        benign_seqs = json.load(f)

    benign_texts = []
    for seq in benign_seqs:
        raw  = seq.get('sequence') or seq.get('text', '')
        text = ' '.join(raw) if isinstance(raw, list) else raw
        benign_texts.append(text)

    print('  Tokenizing {} benign sequences...'.format(len(benign_texts)))
    benign_enc = tokenize_texts(tokenizer, benign_texts, '  benign')

    # ── Attack sequences ──────────────────────────────────────────────────────
    attack_enc = {}
    for scenario, tag in SCENARIOS.items():
        seq_path = os.path.join(OUTPUT_SEQUENCES, 'sequences_{}.json'.format(tag))
        if not os.path.exists(seq_path):
            print('  WARNING: {} not found'.format(seq_path))
            continue
        with open(seq_path, 'r') as f:
            attack_seqs = json.load(f)

        texts = []
        meta  = []
        for seq in attack_seqs:
            raw  = seq.get('sequence') or seq.get('text', '')
            text = ' '.join(raw) if isinstance(raw, list) else raw
            texts.append(text)
            meta.append({
                'dep_id'  : seq['dep_id'],
                'part_idx': seq.get('part_idx', 0),
            })

        print('  Tokenizing {} {} attack sequences...'.format(len(texts), scenario))
        enc = tokenize_texts(tokenizer, texts, '  {}'.format(scenario))
        attack_enc[scenario] = {'enc': enc, 'meta': meta}

    # ── CTI reports ───────────────────────────────────────────────────────────
    cti_keys  = []
    cti_texts = []
    for fname in sorted(os.listdir(CTI_REPORTS_DIR)):
        if fname.endswith('.txt') and not fname.startswith('.'):
            key   = fname.replace('.txt', '')
            fpath = os.path.join(CTI_REPORTS_DIR, fname)
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            if text:
                cti_keys.append(key)
                cti_texts.append(text)

    print('  Tokenizing {} CTI reports...'.format(len(cti_texts)))
    cti_enc = tokenize_texts(tokenizer, cti_texts, '  CTI')

    # ── Benign CTI text ───────────────────────────────────────────────────────
    benign_cti_enc = tokenize_texts(tokenizer, [BENIGN_TEXT], '  benign_cti')

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_TRAINING, 'tokenized.pt')
    torch.save({
        'benign'    : benign_enc,
        'attack'    : attack_enc,
        'cti_keys'  : cti_keys,
        'cti'       : cti_enc,
        'benign_cti': benign_cti_enc,
    }, out_path)

    print()
    print('  Saved: {}'.format(out_path))
    print('  benign     : {} sequences'.format(len(benign_enc['input_ids'])))
    print('  CTI        : {} reports'.format(len(cti_enc['input_ids'])))
    for scenario, data in attack_enc.items():
        print('  {}  : {} sequences'.format(scenario, len(data['enc']['input_ids'])))


def tokenize_benign_testing():
    """Tokenize benign_testing.json subgraphs for evaluation."""
    os.makedirs(OUTPUT_TRAINING, exist_ok=True)

    testing_path = os.path.join(OUTPUT_BENIGN, 'benign_testing.json')
    if not os.path.exists(testing_path):
        print('  WARNING: {} not found'.format(testing_path))
        return

    print('  Loading tokenizer: {}'.format(ROBERTA_MODEL))
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)

    with open(testing_path, 'r') as f:
        data = json.load(f)

    subgraphs = data['subgraphs']
    texts = []
    for sg in subgraphs:
        G       = _rebuild_graph(sg)
        triples = extract_triples(G)
        tokens  = triples_to_text(triples)
        texts.append(' '.join(tokens) if tokens else 'empty subgraph')

    print('  Tokenizing {} benign_testing subgraphs...'.format(len(texts)))
    enc = tokenize_texts(tokenizer, texts, '  benign_testing')

    out_path = os.path.join(OUTPUT_TRAINING, 'tokenized_benign_testing.pt')
    torch.save({'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}, out_path)

    print()
    print('  Saved: {}'.format(out_path))
    print('  benign_testing: {} sequences'.format(len(enc['input_ids'])))


if __name__ == '__main__':
    tokenize_all()
