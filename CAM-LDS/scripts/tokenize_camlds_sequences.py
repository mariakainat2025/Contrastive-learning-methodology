
import os
import sys
import json
import glob

import torch
from transformers import RobertaTokenizer

PROJECT_ROOT = '/csse/research/contructive-learning'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from scripts.config import ROBERTA_MODEL, show

CAM_LDS_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCES_DIR  = os.path.join(CAM_LDS_DIR, "sequences")
TOKENIZED_DIR  = os.path.join(CAM_LDS_DIR, "tokenized")
TOKENIZED_OUT  = os.path.join(TOKENIZED_DIR, "tokenized.pt")
TEMPLATE_DIR   = os.path.join(PROJECT_ROOT, "output", "theia", "tactic_data", "templates")

TACTIC_LABELS = {
    'Initial_Access': 'TA0001', 'Execution': 'TA0002', 'Persistence': 'TA0003',
    'Privilege_Escalation': 'TA0004', 'Stealth': 'TA0005', 'Defense_Impairment': 'TA0112',
    'Credential_Access': 'TA0006', 'Discovery': 'TA0007', 'Lateral_Movement': 'TA0008',
    'Collection': 'TA0009', 'Command_and_Control': 'TA0011', 'Exfiltration': 'TA0010',
    'Impact': 'TA0040', 'Reconnaissance': 'TA0043', 'Resource_Development': 'TA0042',
}


def tokenize_texts(tokenizer, texts, desc=''):
    all_ids, all_masks = [], []
    for i, text in enumerate(texts):
        enc = tokenizer(text, padding=False, truncation=False, return_tensors='pt')
        real_len = int(enc['attention_mask'][0].sum())
        all_ids.append(enc['input_ids'][0][:real_len].clone())
        all_masks.append(enc['attention_mask'][0][:real_len].clone())
        if desc and (i + 1) % 32 == 0:
            print(f'  {desc}: {i+1}/{len(texts)}')
    return {'input_ids': all_ids, 'attention_mask': all_masks}


def load_templates():
    templates = {}
    for fname in sorted(os.listdir(TEMPLATE_DIR)):
        if not fname.endswith('.txt'):
            continue
        label = None
        for lbl, tid in TACTIC_LABELS.items():
            if tid in fname:
                label = lbl
                break
        if label is None:
            continue
        with open(os.path.join(TEMPLATE_DIR, fname), encoding='utf-8') as f:
            templates[label] = f.read().strip()
    return templates


def tokenize_all(force=False):
    os.makedirs(TOKENIZED_DIR, exist_ok=True)

    if os.path.exists(TOKENIZED_OUT) and not force:
        print(f'  [cache] skipping tokenize — found {TOKENIZED_OUT}')
        return torch.load(TOKENIZED_OUT, map_location='cpu')

    print(f'  Loading tokenizer: {ROBERTA_MODEL}')
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)

    files = sorted(glob.glob(os.path.join(SEQUENCES_DIR, '*', '*', 'sequence_*.json')))
    print(f'  Found {len(files)} sequence JSON files')

    texts, meta = [], []
    for fpath in files:
        with open(fpath) as f:
            seq = json.load(f)
        texts.append(' '.join(seq['sequence']))
        meta.append({
            'tactic'   : seq['tactic'],
            'technique': seq['technique'],
            'step'     : seq['step'],
            'host'     : seq['host'],
            'file'     : os.path.relpath(fpath, SEQUENCES_DIR),
        })

    print(f'  Tokenizing {len(texts)} CAM-LDS sequences...')
    seq_enc = tokenize_texts(tokenizer, texts, '  sequences')

    templates = load_templates()
    tmpl_labels = list(templates.keys())
    tmpl_texts  = [templates[l] for l in tmpl_labels]
    print(f'  Tokenizing {len(tmpl_texts)} MITRE tactic templates...')
    tmpl_enc = tokenize_texts(tokenizer, tmpl_texts, '  templates')

    data = {
        'sequences': {'enc': seq_enc, 'meta': meta},
        'templates': {'enc': tmpl_enc, 'labels': tmpl_labels},
    }
    torch.save(data, TOKENIZED_OUT)

    print()
    print(f'  Saved -> {TOKENIZED_OUT}')
    print(f'  sequences: {len(seq_enc["input_ids"])}')
    print(f'  templates: {len(tmpl_enc["input_ids"])}')
    return data


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='re-tokenize even if tokenized.pt already exists')
    args = ap.parse_args()

    show('Stage 4 / 5 — tokenize sequences')
    print()
    tokenize_all(force=args.force)
    print()

