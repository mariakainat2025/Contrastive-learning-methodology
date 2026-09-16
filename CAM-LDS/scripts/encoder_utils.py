
import torch
import torch.nn.functional as F

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import MAX_LEN, STRIDE


def embed_text(model, tokenizer, input_ids, attention_mask, device,
               max_len=MAX_LEN, stride=STRIDE, truncate=False):

    bs = input_ids.size(0)

   
    if truncate:
        ids_t  = input_ids[:, :max_len].to(device)
        mask_t = attention_mask[:, :max_len].to(device)
        out    = model(input_ids=ids_t, attention_mask=mask_t, return_dict=True)
        return out.last_hidden_state[:, 0]   # CLS token for every sequence

   
    sentence_embeddings = [None] * bs
    short_texts         = []
    long_texts          = []
    short_indices       = []

    for i in range(bs):
        real_len = int(attention_mask[i].sum())
        ids_i    = input_ids[i][:real_len]
        mask_i   = attention_mask[i][:real_len]
        if real_len <= max_len:
            short_texts.append((ids_i, mask_i))
            short_indices.append(i)
        else:
            long_texts.append((ids_i, mask_i, i))

  
    if short_texts:
        s_ids  = torch.nn.utils.rnn.pad_sequence(
            [t[0] for t in short_texts], batch_first=True, padding_value=1
        ).to(device)
        s_mask = torch.nn.utils.rnn.pad_sequence(
            [t[1] for t in short_texts], batch_first=True, padding_value=0
        ).to(device)
        out = model(input_ids=s_ids, attention_mask=s_mask, return_dict=True)
        for idx, orig_idx in enumerate(short_indices):
            sentence_embeddings[orig_idx] = out.last_hidden_state[idx, 0]


    if long_texts:
        all_chunks = []
        chunk_meta = []   # orig_idx for each chunk
        cls_id     = tokenizer.bos_token_id

        for ids_i, mask_i, orig_idx in long_texts:
            seq_len = len(ids_i)
            for start in range(1, seq_len, stride):
                end        = min(start + (max_len - 1), seq_len)
                chunk_ids  = torch.cat([
                    torch.tensor([cls_id], device=ids_i.device),
                    ids_i[start:end]
                ])
                chunk_mask = torch.cat([
                    torch.tensor([1], device=mask_i.device),
                    mask_i[start:end]
                ])
                pad_len = max_len - len(chunk_ids)
                if pad_len > 0:
                    chunk_ids  = F.pad(chunk_ids,  (0, pad_len), value=1)
                    chunk_mask = F.pad(chunk_mask, (0, pad_len), value=0)
                else:
                    chunk_ids  = chunk_ids[:max_len]
                    chunk_mask = chunk_mask[:max_len]
                all_chunks.append((chunk_ids, chunk_mask))
                chunk_meta.append(orig_idx)

        cls_per_orig = {}
        CHUNK_BATCH = 32   # process at most 32 chunks at a time to avoid OOM
        for cb_start in range(0, len(all_chunks), CHUNK_BATCH):
            cb_end  = min(cb_start + CHUNK_BATCH, len(all_chunks))
            c_ids   = torch.stack([c[0] for c in all_chunks[cb_start:cb_end]]).to(device)
            c_mask  = torch.stack([c[1] for c in all_chunks[cb_start:cb_end]]).to(device)
            c_out   = model(input_ids=c_ids, attention_mask=c_mask, return_dict=True)
            for i, orig_idx in enumerate(chunk_meta[cb_start:cb_end]):
                cls_per_orig.setdefault(orig_idx, []).append(
                    c_out.last_hidden_state[i, 0]
                )

        for orig_idx, cls_list in cls_per_orig.items():
            sentence_embeddings[orig_idx] = torch.stack(cls_list).mean(0)

    return torch.stack(sentence_embeddings)
