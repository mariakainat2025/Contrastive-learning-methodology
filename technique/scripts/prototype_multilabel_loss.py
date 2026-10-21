import torch
import torch.nn.functional as F


def multilabel_prototype_loss(z, prototypes, pos_mask, logit_scale, class_weights=None):
    z = F.normalize(z, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    scale = logit_scale.exp().clamp(max=100)
    sims = scale * (z @ prototypes.T)

    pos_mask = pos_mask.bool()
    neg_sims = sims.masked_fill(pos_mask, float('-inf'))
    neg_logsumexp = torch.logsumexp(neg_sims, dim=-1, keepdim=True)

    per_class_loss = neg_logsumexp - sims
    if class_weights is not None:
        per_class_loss = per_class_loss * class_weights.unsqueeze(0)
    per_anchor_loss = (per_class_loss * pos_mask).sum(dim=-1) / pos_mask.sum(dim=-1).clamp(min=1)
    per_anchor_loss = per_anchor_loss.clamp(min=0)

    valid = pos_mask.sum(dim=-1) > 0
    return per_anchor_loss[valid].mean()


def asymmetric_loss(z, prototypes, pos_mask, logit_scale, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
    z = F.normalize(z, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    scale = logit_scale.exp().clamp(max=100)
    sims = scale * (z @ prototypes.T)
    p = torch.sigmoid(sims)

    pos_mask = pos_mask.float()
    p_pos = p.clamp(min=eps, max=1 - eps)
    loss_pos = pos_mask * torch.log(p_pos) * (1 - p_pos).pow(gamma_pos)

    p_neg = (p - clip).clamp(min=0, max=1 - eps)
    loss_neg = (1 - pos_mask) * torch.log((1 - p_neg).clamp(min=eps)) * p_neg.pow(gamma_neg)

    loss = -(loss_pos + loss_neg)
    return loss.sum(dim=-1).mean()


def uniformity_loss(z, t=2.0):
    z = F.normalize(z, dim=-1)
    sq_dists = torch.pdist(z, p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log()
