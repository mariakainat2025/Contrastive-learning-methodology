import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import ROBERTA_MODEL, OUTPUT_TRAINING, INPUT_TEST
from scripts.encoder_utils import embed_text
from transformers import RobertaTokenizer, RobertaModel, logging as hf_logging
hf_logging.set_verbosity_error()

EMB_DIM  = 768
PROJ_DIM = 128
DROPOUT  = 0.5


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


def _encode(tokenizer, encoder, proj, text, device):
    enc  = tokenizer(text, padding=False, truncation=False, max_length=False,
                     return_tensors='pt')
    ids  = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        emb = embed_text(encoder, tokenizer, ids, mask, device, truncate=False)
        emb = proj(emb)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu()


def _split_sentences(text):
    return [s.strip() for s in str(text).split('. ') if s.strip()]


def _split_cti(text):
    import re
    text = re.sub(r'\n(?![A-Z])', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('i.e.,', 'IE_ABBR,').replace('e.g.,', 'EG_ABBR,')
    text = re.sub(r'\d+\.\d+\.\d+\.\d+', lambda m: m.group().replace('.', '__DOT__'), text)
    text = re.sub(r'www\.\S+',            lambda m: m.group().replace('.', '__DOT__'), text)
    parts = re.split(r'\.\s+(?=[A-Z])', text)
    result = []
    for p in parts:
        p = (p.strip().rstrip('.')
               .replace('__DOT__', '.')
               .replace('IE_ABBR', 'i.e.')
               .replace('EG_ABBR', 'e.g.'))
        if len(p) > 10:
            result.append(p)
    return result


def _load_cti_reports(cti_filter=None):
    cti_reports = {}
    for fname in sorted(os.listdir(INPUT_TEST)):
        if fname.endswith('.txt') and not fname.startswith('.'):
            key = fname.replace('.txt', '')
            if cti_filter and key not in cti_filter:
                continue 
            if not key.endswith('_abstracted'):
                abstracted_path = os.path.join(INPUT_TEST, f'{key}_abstracted.txt')
                if os.path.exists(abstracted_path):
                    continue
            with open(os.path.join(INPUT_TEST, fname)) as f:
                cti_reports[key] = f.read().strip()
    return cti_reports


def _load_atk_seqs():
    atk_path = os.path.join(INPUT_TEST, 'abstracted_attack_sequences_duplicate_removed.json')
    with open(atk_path) as f:
        return json.load(f)


def explain_with_summary(ckpt_name, top_k, cti_filter, lines_acc):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)

    ckpt_path = os.path.join(OUTPUT_TRAINING, ckpt_name)
    log_encoder, text_encoder, log_proj, text_proj = _load_model(ckpt_path, device)

    atk_seqs    = _load_atk_seqs()
    cti_reports = _load_cti_reports(cti_filter)
    cti_keys    = list(cti_reports.keys())

    cti_sents_all = {k: _split_cti(v) for k, v in cti_reports.items()}

    cti_embs_all = {}
    for k, sents in cti_sents_all.items():
        cti_embs_all[k] = torch.cat(
            [_encode(tokenizer, text_encoder, text_proj, s, device) for s in sents], dim=0)

    atk_data = []
    for atk in atk_seqs:
        label     = atk['label']
        log_sents = _split_sentences(atk['sequence'])
        log_embs  = torch.cat(
            [_encode(tokenizer, log_encoder, log_proj, s, device) for s in log_sents], dim=0)
        atk_data.append((label, log_sents, log_embs))

    lines_acc.append(f'\n  === {ckpt_name} ===')
    print(f'\n  === {ckpt_name} ===')

    for label, log_sents, log_embs in atk_data:

        row_scores = {}
        for k in cti_keys:
            score = (log_embs @ cti_embs_all[k].T).max(dim=1).values.mean().item()
            row_scores[k] = score

        all_scores = sorted(row_scores.items(), key=lambda x: x[1], reverse=True)
        print(f'\n    {label}')
        lines_acc.append(f'\n    {label}')
        for k, score in all_scores:
            print(f'      {k:<40} {score:.4f}')
            lines_acc.append(f'      {k:<40} {score:.4f}')

        top2_ctis = all_scores[:2]
        for top_cti_key, _ in top2_ctis:
            cti_sents  = cti_sents_all[top_cti_key]
            sim_matrix = log_embs @ cti_embs_all[top_cti_key].T
            flat       = sim_matrix.flatten()
            vals, idxs = flat.topk(min(10, flat.numel()))

            all_pairs = []
            for val, idx in zip(vals.tolist(), idxs.tolist()):
                li, ci = idx // len(cti_sents), idx % len(cti_sents)
                all_pairs.append((val, log_sents[li], cti_sents[ci]))

            all_pairs.sort(key=lambda x: x[0], reverse=True)
            seen, rank = set(), 0
            lines_acc.append(f'\n    Top matching sentences for [{top_cti_key}]:')
            print(f'\n    Top matching sentences for [{top_cti_key}]:')
            for val, log_s, cti_s in all_pairs:
                pair = (log_s, cti_s)
                if pair in seen:
                    continue
                seen.add(pair)
                rank += 1
                lines_acc.append(f'      [{rank}] Score : {val:.4f}')
                lines_acc.append(f'          LOG : {log_s}')
                lines_acc.append(f'          CTI : {cti_s}')
                print(f'      [{rank}] Score : {val:.4f}')
                print(f'          LOG : {log_s}')
                print(f'          CTI : {cti_s}')
                if rank >= top_k:
                    break

    del log_encoder, text_encoder, log_proj, text_proj
    torch.cuda.empty_cache()


def main():
    from scripts.config import OUTPUT_TEST

    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=2)
    ap.add_argument('--cti', nargs='+', default=None)
    args = ap.parse_args()

    epochs    = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    lines_acc = []

    for epoch in epochs:
        ckpt_name = f'theia_epoch{epoch}.pt'
        ckpt_path = os.path.join(OUTPUT_TRAINING, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f'  skipping (not found): {ckpt_name}')
            continue
        explain_with_summary(ckpt_name=ckpt_name, top_k=args.top, cti_filter=args.cti, lines_acc=lines_acc)

    os.makedirs(OUTPUT_TEST, exist_ok=True)
    report_path = os.path.join(OUTPUT_TEST, 'explain_all_epochs_summary.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines_acc))
    print(f'\n  Report saved : {report_path}')


if __name__ == '__main__':
    main()