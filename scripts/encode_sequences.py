
import os
import json
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

from scripts.config import (
    ROBERTA_MODEL,
    MAX_LEN,
    STRIDE,
    EMB_DIM,
    OUTPUT_SEQUENCES,
    OUTPUT_EMBEDDINGS,
)



def load_model(device):
    print('  loading tokenizer and model: {}'.format(ROBERTA_MODEL))
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    model     = RobertaModel.from_pretrained(ROBERTA_MODEL)
    model.eval()
    model.to(device)
    return tokenizer, model




def encode_text(text, tokenizer, model, device):
    tokens = tokenizer(
        text,
        return_tensors    = 'pt',
        truncation        = False,
        add_special_tokens= True,
    )
    input_ids = tokens['input_ids'][0]         
    n_tokens  = len(input_ids)

    if n_tokens <= MAX_LEN:
        input_ids      = input_ids.unsqueeze(0).to(device)        
        attention_mask = torch.ones_like(input_ids).to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = output.last_hidden_state[0, 0, :]               
        return cls_vec.cpu()

    else:
        
        cls_vecs = []
        start    = 0
        while start < n_tokens:
            end        = min(start + MAX_LEN, n_tokens)
            chunk_ids  = input_ids[start:end].unsqueeze(0).to(device)   
            attn_mask  = torch.ones_like(chunk_ids).to(device)
            with torch.no_grad():
                output = model(input_ids=chunk_ids, attention_mask=attn_mask)
            cls_vecs.append(output.last_hidden_state[0, 0, :].cpu())  
            if end == n_tokens:
                break
            start += STRIDE   


        cls_vec = torch.stack(cls_vecs, dim=0).mean(dim=0)          
        return cls_vec



def encode_sequences(tag='theia'):
    seq_path = os.path.join(OUTPUT_SEQUENCES, 'log_sequences_{}.json'.format(tag))
    out_path = os.path.join(OUTPUT_EMBEDDINGS, 'log_embeddings_{}.pt'.format(tag))

    os.makedirs(OUTPUT_EMBEDDINGS, exist_ok=True)

    if not os.path.exists(seq_path):
        print('  ERROR: {} not found'.format(seq_path))
        return

    with open(seq_path, 'r', encoding='utf-8') as f:
        sequences = json.load(f)

    print('  loaded {} sequences from {}'.format(len(sequences), seq_path))

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('  device: {}'.format(device))

    tokenizer, model = load_model(device)

    embeddings = []
    metadata   = []

    for seq in tqdm(sequences, desc='  encoding'):
        raw  = seq.get('text') or seq.get('sequence', '')
        text = ' '.join(raw) if isinstance(raw, list) else raw
        vec     = encode_text(text, tokenizer, model, device)           
        embeddings.append(vec)
        metadata.append({
            'idx'        : seq['idx'],
            'dep_id'     : seq.get('dep_id',      0),
            'part_idx'   : seq.get('part_idx',    0),
            'total_parts': seq.get('total_parts', 1),
            'seed_uuid'  : seq.get('seed_uuid',   ''),
            'seed_name'  : seq.get('seed_name',   ''),
            'n_triples'  : seq.get('n_triples',   0),
        })

    embeddings_tensor = torch.stack(embeddings, dim=0)                 

    torch.save({
        'embeddings': embeddings_tensor,
        'metadata'  : metadata,
    }, out_path)

    # save embeddings as JSON
    json_path = os.path.join(OUTPUT_EMBEDDINGS, 'log_embeddings_{}.json'.format(tag))
    json_out  = []
    for i, m in enumerate(metadata):
        entry = dict(m)
        entry['embedding'] = embeddings_tensor[i].tolist()
        json_out.append(entry)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2)

    # save tokens as JSON for inspection
    token_path = os.path.join(OUTPUT_EMBEDDINGS, 'log_tokens_{}.json'.format(tag))
    token_out  = []
    for seq in sequences:
        raw  = seq.get('text') or seq.get('sequence', '')
        text = ' '.join(raw) if isinstance(raw, list) else raw
        tokens     = tokenizer(text, truncation=False, add_special_tokens=True)
        input_ids  = tokens['input_ids']
        token_strs = tokenizer.convert_ids_to_tokens(input_ids)
        token_out.append({
            'idx'        : seq['idx'],
            'seed_name'  : seq.get('seed_name', ''),
            'n_tokens'   : len(input_ids),
            'n_chunks'   : max(1, (len(input_ids) - MAX_LEN) // STRIDE + 2) if len(input_ids) > MAX_LEN else 1,
            'token_ids'  : input_ids,
            'tokens'     : token_strs,
        })
    with open(token_path, 'w', encoding='utf-8') as f:
        json.dump(token_out, f, indent=2)
    print('  log_tokens_{}.json     saved : {}'.format(tag, token_path))

    print('  log_embeddings_{}.pt   saved : {}'.format(tag, out_path))
    print('  log_embeddings_{}.json saved : {}'.format(tag, json_path))
    print('  shape : {}'.format(list(embeddings_tensor.shape)))
    print('  dtype : {}'.format(embeddings_tensor.dtype))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='theia')
    args = parser.parse_args()
    encode_sequences(args.tag)
