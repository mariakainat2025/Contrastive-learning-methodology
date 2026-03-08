import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel



EDGE_TYPE_TEXT = {
    'EVENT_EXECUTE'  : 'execute',
    'EVENT_READ'     : 'read',
    'EVENT_WRITE'    : 'write',
    'EVENT_OPEN'     : 'open',
    'EVENT_CLOSE'    : 'close',
    'EVENT_CONNECT'  : 'connect',
    'EVENT_SENDTO'   : 'send',
    'EVENT_RECVFROM' : 'receive',
    'EVENT_CLONE'    : 'clone',
    'EVENT_MMAP'     : 'memory map',
    'EVENT_MPROTECT' : 'memory protect',
    'EVENT_UNLINK'   : 'delete',
    'EVENT_RENAME'   : 'rename',
    'EVENT_FORK'     : 'fork',
    'EVENT_EXIT'     : 'exit',
    'EVENT_CREATE'   : 'create',
}


def _ip_scope(ip):
    if not ip or ip.lower() in ('na', 'none', ''):
        return ''
    try:
        parts = ip.split('.')
        if len(parts) != 4:
            return ''
        a, b = int(parts[0]), int(parts[1])
        if a == 127:
            return 'loopback'
        if a == 10:
            return 'internal'
        if a == 172 and 16 <= b <= 31:
            return 'internal'
        if a == 192 and b == 168:
            return 'internal'
        return 'external'
    except (ValueError, IndexError):
        return ''


def build_node_text(node_dict):
    ntype = node_dict.get('node_type', '')

    if ntype == 'SUBJECT_PROCESS':
        exe   = node_dict.get('exe_path', '').strip()
        cmd   = node_dict.get('cmdline',  '').strip()
        parts = ['process']
        if exe:
            parts.append(exe)
        if cmd and cmd != exe:
            parts.append('cmd')
            parts.append(cmd)
        return ' '.join(parts)

    elif 'FILE' in ntype:
        fp = node_dict.get('file_path', '').strip()
        return ('file ' + fp) if fp else 'file'

    elif ntype == 'NetFlowObject':
        rip   = node_dict.get('remote_ip',   '').strip()
        rport = str(node_dict.get('remote_port', '')).strip()
        lip   = node_dict.get('local_ip',    '').strip()
        lport = str(node_dict.get('local_port',  '')).strip()


        scope = _ip_scope(rip)

        parts = ['network']
        if scope:
            parts.append(scope)      
        parts.append('connection')
        if rip:                      parts.append(rip)
        if rport:                    parts.append('port ' + rport)
        if lip:                      parts.append('local ' + lip)
        if lport and lport != '0':   parts.append(lport)
        return ' '.join(parts)

    elif ntype == 'MemoryObject':
        addr = node_dict.get('memory_address', '').strip()
        return ('memory ' + addr) if addr else 'memory'

    else:
        return ntype.lower().replace('_', ' ')




def _load_text_model(model_name):
    print('  [TextEncoder] Loading "{}" ...'.format(model_name))
    tok      = AutoTokenizer.from_pretrained(model_name)
    mdl      = AutoModel.from_pretrained(model_name).eval()
    text_dim = mdl.config.hidden_size
    print('  [TextEncoder] {}  →  hidden_size = {:,}'.format(model_name, text_dim))

    def encode_fn(texts, batch_size=64):
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inp   = tok(batch, return_tensors='pt', truncation=True,
                        max_length=128, padding=True)
            with torch.no_grad():
                out = mdl(**inp)
            # [CLS] token as the sequence-level representation
            vecs.append(out.last_hidden_state[:, 0, :].cpu().float())
        return torch.cat(vecs, dim=0)

    return encode_fn, text_dim, tok


# SecureBERT node/edge text encoder with caching + projection ───────────────

class BERTTextEncoder(nn.Module):
    def __init__(self, enc_dim, model_name='ehsanaghaei/SecureBERT'):
        super().__init__()
        self.enc_dim    = enc_dim
        self.model_name = model_name

        encode_fn, text_dim, tok = _load_text_model(model_name)
        self._encode_fn  = encode_fn
        self._text_dim   = text_dim
        self._tokenizer  = tok


        self.proj = nn.Linear(text_dim, enc_dim, bias=False)
        nn.init.xavier_normal_(self.proj.weight)

        self._cache: dict = {}  

    # ── Public API ────────────────────────────────────────────────────────────

    def warm_cache(self, texts):

        new_texts = [t for t in texts if t not in self._cache]
        if not new_texts:
            return
        n_cached = len(texts) - len(new_texts)
        print('  [BERTTextEncoder] encoding {:,} new strings '
              '({:,} already cached) ...'.format(len(new_texts), n_cached))
        raw = self._encode_fn(new_texts)
        for t, vec in zip(new_texts, raw):
            self._cache[t] = vec.cpu()

    def tokenize(self, text):
        enc    = self._tokenizer([text], return_tensors='pt',
                                 truncation=True, max_length=128, padding=False)
        ids    = enc['input_ids'][0].tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(ids)
        return {'text': text, 'tokens': tokens, 'token_ids': ids}

    def encode_nodes(self, texts, device):
        new_texts = [t for t in texts if t not in self._cache]
        if new_texts:
            raw_new = self._encode_fn(new_texts)
            for t, vec in zip(new_texts, raw_new):
                self._cache[t] = vec.cpu()

        raw = torch.stack([self._cache[t] for t in texts]).to(device)
        with torch.no_grad():
            return self.proj(raw)  

    def build_edge_table(self, edge_type_dict, edge_type_text_map, device):
        idx2type = sorted(edge_type_dict.keys(), key=lambda k: edge_type_dict[k])
        texts    = [edge_type_text_map.get(t, t) for t in idx2type]
        self.warm_cache(texts)
        raw = torch.stack([self._cache[t] for t in texts]).to(device)
        with torch.no_grad():
            return self.proj(raw)   


    @property
    def text_dim(self):
        """Raw BERT hidden size before projection (e.g. 768 for RoBERTa-base)."""
        return self._text_dim

    @property
    def cache_size(self):
        """Number of unique strings currently in the encoding cache."""
        return len(self._cache)
