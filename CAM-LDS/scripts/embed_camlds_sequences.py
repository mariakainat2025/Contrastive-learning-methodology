
import os
import sys

import torch

PROJECT_ROOT = '/csse/research/contructive-learning'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from scripts.config import ROBERTA_MODEL
from scripts.encoder_utils import embed_text

from transformers import RobertaTokenizer, RobertaModel

CAM_LDS_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZED_PATH = os.path.join(CAM_LDS_DIR, "tokenized", "tokenized.pt")
EMBEDDINGS_OUT = os.path.join(CAM_LDS_DIR, "embeddings.pt")


def encode_batch(model, tokenizer, ids_list, masks_list, device, desc=''):
    embeddings = []
    for i, (ids, mask) in enumerate(zip(ids_list, masks_list)):
        emb = embed_text(model, tokenizer, ids.unsqueeze(0).to(device),
                         mask.unsqueeze(0).to(device), device)
        embeddings.append(emb.squeeze(0).cpu())
        if desc and (i + 1) % 32 == 0:
            print(f'  {desc}: {i+1}/{len(ids_list)}')
    return torch.stack(embeddings)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading {TOKENIZED_PATH}...')
    data = torch.load(TOKENIZED_PATH, map_location='cpu')

    print(f'Loading RoBERTa: {ROBERTA_MODEL}')
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    model     = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    model.eval()

    seq_enc  = data['sequences']['enc']
    seq_meta = data['sequences']['meta']
    tmpl_enc    = data['templates']['enc']
    tmpl_labels = data['templates']['labels']

    with torch.no_grad():
        print(f'Encoding {len(seq_enc["input_ids"])} sequences...')
        seq_embs = encode_batch(model, tokenizer, seq_enc['input_ids'],
                                seq_enc['attention_mask'], device, 'sequences')

        print(f'Encoding {len(tmpl_enc["input_ids"])} templates...')
        tmpl_embs = encode_batch(model, tokenizer, tmpl_enc['input_ids'],
                                 tmpl_enc['attention_mask'], device, 'templates')

    torch.save({
        'sequence_embeddings': seq_embs,
        'sequence_meta'      : seq_meta,
        'template_embeddings': tmpl_embs,
        'template_labels'    : tmpl_labels,
        'emb_dim'            : seq_embs.shape[1],
    }, EMBEDDINGS_OUT)

    print()
    print(f'Saved -> {EMBEDDINGS_OUT}')
    print(f'  sequence_embeddings: {tuple(seq_embs.shape)}')
    print(f'  template_embeddings: {tuple(tmpl_embs.shape)}')


if __name__ == '__main__':
    main()
