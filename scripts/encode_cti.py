
import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import (
    ROBERTA_MODEL,
    MAX_LEN,
    STRIDE,
    CTI_REPORTS_DIR,
    OUTPUT_CTI,
    OUTPUT_EMBEDDINGS,
)


class CTIEncoder:

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('  loading RoBERTa: {}'.format(ROBERTA_MODEL))
        self.tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
        self.model     = RobertaModel.from_pretrained(ROBERTA_MODEL)
        self.model.eval()
        self.model.to(self.device)
        print('  device: {}'.format(self.device))

    def embed_text(self, text, max_len=MAX_LEN, stride=STRIDE):
        device = self.device
        bs     = len(text['input_ids'])

        batch_sequence_embeddings = [None] * bs
        batch_sentence_embeddings = [None] * bs

        short_texts   = []
        long_texts    = []
        short_indices = []
        long_indices  = []

        for i in range(bs):
            real_len  = int(text['attention_mask'][i].sum())
            input_ids = text['input_ids'][i][:real_len]
            att_mask  = text['attention_mask'][i][:real_len]

            if real_len <= max_len:
                short_texts.append((input_ids, att_mask))
                short_indices.append(i)
            else:
                long_texts.append((input_ids, att_mask, i))
                long_indices.append(i)

        if short_texts:
            short_input_ids = torch.nn.utils.rnn.pad_sequence(
                [t[0] for t in short_texts], batch_first=True, padding_value=1
            ).to(device)

            short_attention_masks = torch.nn.utils.rnn.pad_sequence(
                [t[1] for t in short_texts], batch_first=True, padding_value=0
            ).to(device)

            with torch.no_grad():
                outputs = self.model(
                    short_input_ids,
                    attention_mask=short_attention_masks,
                    return_dict=True,
                )

            for idx, orig_idx in enumerate(short_indices):
                actual_len = int(text['attention_mask'][orig_idx].sum())
                batch_sequence_embeddings[orig_idx] = outputs.last_hidden_state[idx, 1:actual_len].cpu()
                batch_sentence_embeddings[orig_idx] = outputs.last_hidden_state[idx, 0].cpu()

        if long_texts:
            all_chunks     = []
            chunk_metadata = []

            cls_id  = self.tokenizer.bos_token_id
            cls_att = 1

            for input_ids, attention_mask, orig_idx in long_texts:
                seq_len       = len(input_ids)
                window_starts = list(range(1, seq_len, stride))

                for start in window_starts:
                    end = min(start + (max_len - 1), seq_len)

                    chunk_tokens = input_ids[start:end]
                    chunk_mask   = attention_mask[start:end]

                    chunk_input_ids = torch.cat([
                        torch.tensor([cls_id], device=input_ids.device),
                        chunk_tokens
                    ])
                    chunk_attention_mask = torch.cat([
                        torch.tensor([cls_att], device=attention_mask.device),
                        chunk_mask
                    ])

                    pad_len = max_len - len(chunk_input_ids)
                    if pad_len > 0:
                        chunk_input_ids      = F.pad(chunk_input_ids,      (0, pad_len), value=1)
                        chunk_attention_mask = F.pad(chunk_attention_mask, (0, pad_len), value=0)
                    else:
                        chunk_input_ids      = chunk_input_ids[:max_len]
                        chunk_attention_mask = chunk_attention_mask[:max_len]

                    all_chunks.append((chunk_input_ids, chunk_attention_mask))
                    chunk_metadata.append((orig_idx, start, end))

            chunk_input_ids       = torch.stack([c[0] for c in all_chunks]).to(device)
            chunk_attention_masks = torch.stack([c[1] for c in all_chunks]).to(device)

            with torch.no_grad():
                chunk_outputs = self.model(
                    chunk_input_ids,
                    attention_mask=chunk_attention_masks,
                    return_dict=True,
                )

            hidden_size           = chunk_outputs.last_hidden_state.size(-1)
            long_text_aggregation = {}

            for chunk_idx, (orig_idx, start, end) in enumerate(chunk_metadata):
                if orig_idx not in long_text_aggregation:
                    seq_len = int(text['attention_mask'][orig_idx].sum().item())
                    long_text_aggregation[orig_idx] = {
                        'embeddings'    : torch.zeros(seq_len, hidden_size),
                        'counts'        : torch.zeros(seq_len),
                        'cls_embeddings': [],
                    }

                data         = long_text_aggregation[orig_idx]
                chunk_output = chunk_outputs.last_hidden_state[chunk_idx].cpu()

                data['cls_embeddings'].append(chunk_output[0])

                real_tokens = chunk_output[1:1 + (end - start)]
                data['embeddings'][start:end] += real_tokens
                data['counts'][start:end]     += 1

            for orig_idx, data in long_text_aggregation.items():
                data['counts'] = torch.clamp(data['counts'], min=1)
                batch_sequence_embeddings[orig_idx] = (
                    data['embeddings'][1:] / data['counts'][1:].unsqueeze(-1)
                )
                batch_sentence_embeddings[orig_idx] = (
                    torch.stack(data['cls_embeddings']).mean(0)
                )

        batch_sequence_embeddings = torch.nn.utils.rnn.pad_sequence(
            batch_sequence_embeddings, batch_first=True, padding_value=0
        )
        batch_sentence_embeddings = torch.stack(batch_sentence_embeddings, dim=0)

        return batch_sequence_embeddings, batch_sentence_embeddings

    def encode_files(self, reports_dir, batch_size=8):
        files = sorted([
            f for f in os.listdir(reports_dir)
            if f.endswith('.txt') and not f.startswith('.')
        ])

        if not files:
            print('  ERROR: no .txt files found in {}'.format(reports_dir))
            return None, None, []

        print('  found {} CTI report(s) in {}'.format(len(files), reports_dir))

        texts    = []
        metadata = []
        for fname in files:
            fpath = os.path.join(reports_dir, fname)
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read().strip()
            if not txt:
                print('  WARNING: {} is empty, skipping'.format(fname))
                continue
            texts.append(txt)
            metadata.append({
                'key'     : fname.replace('.txt', ''),
                'filename': fname,
                'n_chars' : len(txt),
            })

        tokenized = self.tokenizer(
            texts,
            padding       = True,
            truncation    = False,
            return_tensors= 'pt',
        )

        n            = len(texts)
        all_sentence = []
        all_sequence = []

        for start in tqdm(range(0, n, batch_size), desc='  encoding CTI'):
            end   = min(start + batch_size, n)
            batch = {
                'input_ids'     : tokenized['input_ids'][start:end],
                'attention_mask': tokenized['attention_mask'][start:end],
            }
            seq_emb, sent_emb = self.embed_text(batch)
            all_sentence.append(sent_emb)
            all_sequence.append(seq_emb)

        sentence_embeddings = torch.cat(all_sentence, dim=0)

        max_seq_len = max(e.size(1) for e in all_sequence)
        padded_seq  = []
        for e in all_sequence:
            pad_size = max_seq_len - e.size(1)
            if pad_size > 0:
                e = F.pad(e, (0, 0, 0, pad_size))
            padded_seq.append(e)
        sequence_embeddings = torch.cat(padded_seq, dim=0)

        return sentence_embeddings, sequence_embeddings, metadata


def encode_cti():
    os.makedirs(OUTPUT_EMBEDDINGS, exist_ok=True)

    encoder = CTIEncoder()

    sent_emb, seq_emb, metadata = encoder.encode_files(CTI_REPORTS_DIR, batch_size=8)

    if sent_emb is None:
        return

    pt_path = os.path.join(OUTPUT_EMBEDDINGS, 'cti_embeddings.pt')
    torch.save({
        'sentence_embeddings': sent_emb,
        'sequence_embeddings': seq_emb,
        'metadata'           : metadata,
    }, pt_path)

    json_path = os.path.join(OUTPUT_EMBEDDINGS, 'cti_embeddings.json')
    json_out  = []
    for i, m in enumerate(metadata):
        entry = dict(m)
        entry['sentence_embedding'] = sent_emb[i].tolist()
        json_out.append(entry)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2)

    print('  saved: {}'.format(pt_path))
    print('  saved: {}'.format(json_path))
    print('  sentence_embeddings : {}'.format(list(sent_emb.shape)))
    print('  sequence_embeddings : {}'.format(list(seq_emb.shape)))
    print('  reports : {}'.format([m['key'] for m in metadata]))


if __name__ == '__main__':
    encode_cti()
