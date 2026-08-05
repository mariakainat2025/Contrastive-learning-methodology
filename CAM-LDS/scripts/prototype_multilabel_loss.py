import torch
import torch.nn.functional as F


# Lmlc(x) = -1/|Pml(x)| * sum_{k in Pml(x)} log [ exp(sim(z,cpk)/tau) / sum_{c-j in C\Pml(x)} exp(sim(z,cp_c-j)/tau) ]
def multilabel_prototype_loss(z, prototypes, pos_mask, logit_scale):
    z = F.normalize(z, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    scale = logit_scale.exp().clamp(max=100)
    sims = scale * (z @ prototypes.T)  

    pos_mask = pos_mask.bool()
    neg_sims = sims.masked_fill(pos_mask, float('-inf'))
    neg_logsumexp = torch.logsumexp(neg_sims, dim=-1, keepdim=True)

    per_class_loss = neg_logsumexp - sims
    per_anchor_loss = (per_class_loss * pos_mask).sum(dim=-1) / pos_mask.sum(dim=-1).clamp(min=1)
    per_anchor_loss = per_anchor_loss.clamp(min=0)

    valid = pos_mask.sum(dim=-1) > 0
    return per_anchor_loss[valid].mean()
