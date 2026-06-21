"""
Cosine similarity: each Execution subgraph edge vs each Persistence template sentence.
Uses mean pooling RoBERTa.
"""
import torch, json, os, re
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXEC_SG   = os.path.join(BASE, 'output/theia/tactic_data/abstract_sequnce/abstract_command_and_control_backdoor.json')
PERS_TMPL = os.path.join(BASE, 'output/theia/tactic_data/templates/TA0011_Command_and_Control.txt')

with open(EXEC_SG) as f:
    exec_edges = json.load(f)['sequence']

with open(PERS_TMPL) as f:
    raw = f.read()

paragraphs = [p.strip() for p in raw.split('\n\n') if p.strip()]
sentences  = []
for p in paragraphs:
    p = re.sub(r'[=\-]{4,}', '', p).strip()
    if p and len(p) > 20 and 'MITRE' not in p and 'ID ' not in p and 'NAME' not in p:
        sentences.append(p.replace('\n', ' ').strip())

print(f"Execution edges: {len(exec_edges)}")
print(f"Persistence template sentences: {len(sentences)}\n")

print("Loading RoBERTa (mean pooling)...")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model     = RobertaModel.from_pretrained('roberta-base').to(DEVICE)
model.eval()

def encode(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = model(**tokens)
    mask = tokens['attention_mask'].unsqueeze(-1).float()
    emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
    return F.normalize(emb, dim=-1)

# Pre-encode all sentences
sent_embs = [encode(s) for s in sentences]

print("=" * 100)
print(f"{'Score':>7}  Execution Edge  →  Best Matching Persistence Template Sentence")
print("=" * 100)
for edge in exec_edges:
    e_emb = encode(edge)
    sims  = [(( e_emb @ s_emb.T).item(), s) for s_emb, s in zip(sent_embs, sentences)]
    sims.sort(reverse=True)
    best_sim, best_sent = sims[0]
    print(f"\n  EXEC EDGE : {edge}")
    print(f"  BEST MATCH: {best_sent[:120]}")
    print(f"  SCORE     : {best_sim:+.4f}")
    print(f"  2nd MATCH : {sims[1][1][:120]}  ({sims[1][0]:+.4f})")
