
# Code inspiration : https://github.com/nanda-rani/TTPHunter-Automated-Extraction-of-Actionable-Intelligence-as-TTPs-from-Narrative-Threat-Reports
import os
import re
import json
import pickle as pkl

import torch
import torch.nn as nn
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import RobertaTokenizer, RobertaModel

from scripts.config import show, OUTPUT_CTI, load_window_cti

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)


WINDOW_CTI = load_window_cti()

def remove_stopwords(text):
    sw           = set(stopwords.words('english'))
    words        = word_tokenize(text)
    kept         = [w for w in words if w.lower() not in sw]
    removed      = [w for w in words if w.lower() in sw]
    cleaned_text = ' '.join(kept)
    stats = {
        'original_count' : len(words),
        'cleaned_count'  : len(kept),
        'removed_count'  : len(removed),
    }
    return cleaned_text, stats


class TextEncoder(nn.Module):

    def __init__(
        self,
        model_name    = 'ehsanaghaei/SecureBERT',
        max_length    = 512,
        freeze_layers = 10,
    ):
        super().__init__()
        self.max_length = max_length

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.bert      = RobertaModel.from_pretrained(model_name)

        self._freeze(freeze_layers)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  [TextEncoder] total params={total:,}  trainable={trainable:,}  '
              f'(layers {freeze_layers}–11 unfrozen)')

    def forward(self, input_ids, attention_mask):
        out     = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            return_dict    = True,
        )
        H_Ti    = out.last_hidden_state
        h_t_cls = H_Ti[:, 0, :]
        return H_Ti, h_t_cls

    def encode(self, text, device):
        enc = self.tokenizer(
            text,
            return_tensors = 'pt',
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
        )
        input_ids      = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        with torch.no_grad():
            H_Ti, h_t_cls = self.forward(input_ids, attention_mask)

        H_Ti    = H_Ti[0]
        h_t_cls = h_t_cls[0]
        tokens  = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return {
            'H_Ti'    : H_Ti,
            'h_t_cls' : h_t_cls,
            'tokens'  : tokens,
            'seq_len' : len(tokens),
        }

    def _freeze(self, n):
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False


def encode_cti_windows(encoder, device, window_cti=None, save=True):

    if window_cti is None:
        window_cti = WINDOW_CTI

    os.makedirs(OUTPUT_CTI, exist_ok=True)

    cti_embeddings       = {}
    cti_token_embeddings = {}   
    readable_json        = {}
    summary              = []

    for w_id in sorted(window_cti.keys()):
        raw_text = window_cti[w_id]

        cleaned, sw_stats = remove_stopwords(raw_text)

        out     = encoder.encode(cleaned, device)
        H_Ti    = out['H_Ti']
        h_t_cls = out['h_t_cls']
        tokens  = out['tokens']
        seq_len = out['seq_len']

        cti_embeddings[w_id]       = h_t_cls.cpu()  
        cti_token_embeddings[w_id] = H_Ti.cpu()     

        readable_json[w_id] = {
            'cleaned_text'     : cleaned.strip(),
            'stopwords_removed': sw_stats['removed_count'],
            'h_t_cls': {
                'shape'       : '(768,)',
                'first_8_dims': [round(x, 4) for x in h_t_cls[:8].tolist()],
                'full_vector' : [round(x, 6) for x in h_t_cls.tolist()],
            },
            'H_Ti': {
                'shape'   : f'({seq_len}, 768)',
                'n_tokens': seq_len,
                'tokens'  : [
                    {
                        'position'    : i,
                        'token'       : tok,
                        'notation'    : 'h_t[CLS]' if i == 0 else f'h_t{i}',
                        'full_768_dim': [round(x, 6) for x in H_Ti[i].tolist()],
                    }
                    for i, tok in enumerate(tokens)
                ],
            },
        }

        print(f'\n  {w_id}')
        print(f'    H_Ti shape        : ({seq_len}, 768)  ← all tokens incl. CLS')
        print(f'    h_t[CLS]          : (768,)            ← global representation')
        print(f'    tokens            : {seq_len}')
        print(f'    stopwords removed : {sw_stats["removed_count"]}')
        print(f'    all tokens        : {tokens}')

        summary.append({
            'window_id'    : w_id,
            'raw_words'    : sw_stats['original_count'],
            'cleaned_words': sw_stats['cleaned_count'],
            'n_tokens'     : seq_len,
            'H_Ti_shape'   : [seq_len, 768],
            'h_t_cls_shape': [768],
            'h_t_cls_norm' : round(h_t_cls.norm().item(), 4),
            'cleaned_text' : cleaned.strip(),
        })

    if save:
        with open(OUTPUT_CTI + 'cti_embeddings.pkl', 'wb') as f:
            pkl.dump(cti_embeddings, f)
        print(f'\ncti_embeddings.pkl            {len(cti_embeddings)} windows')

        with open(OUTPUT_CTI + 'cti_token_embeddings.pkl', 'wb') as f:
            pkl.dump(cti_token_embeddings, f)
        print(f'cti_token_embeddings.pkl      {len(cti_token_embeddings)} texts')

        with open(OUTPUT_CTI + 'cti_embeddings_readable.json', 'w') as f:
            json.dump(readable_json, f, indent=2)
        print(f'cti_embeddings_readable.json')

        with open(OUTPUT_CTI + 'cti_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'cti_summary.json')

    return cti_embeddings



def run_cti_encoding():
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device : {device}')

    encoder = TextEncoder(
        model_name    = 'ehsanaghaei/SecureBERT',
        max_length    = 512,
        freeze_layers = 10,
    ).to(device)

    return encode_cti_windows(encoder, device, save=True)


run_encode_cti = run_cti_encoding


if __name__ == '__main__':
    run_cti_encoding()
