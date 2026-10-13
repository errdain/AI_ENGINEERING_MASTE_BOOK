"""Preference / alignment math: Bradley-Terry reward loss (10.21), DPO (10.25), distillation (10.26)."""
import torch
import torch.nn.functional as F


def bradley_terry_loss(r_chosen, r_rejected):
    """-log sigmoid(r_c - r_r): reward-model loss. P(chosen beats rejected)=sigmoid(r_c - r_r)."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1):
    """DPO: -log sigmoid( beta * [ (log pi(yc)-log ref(yc)) - (log pi(yr)-log ref(yr)) ] ).
    Inputs are *sequence* log-probs (sum over response tokens). Returns (loss, stats)."""
    rc, rr = beta * (pol_c - ref_c), beta * (pol_r - ref_r)     # implicit rewards
    margin = rc - rr
    loss = -F.logsigmoid(margin).mean()
    return loss, {"margin": margin.mean().item(), "acc": (margin > 0).float().mean().item(),
                  "reward_chosen": rc.mean().item(), "reward_rejected": rr.mean().item()}


def distill_kl(student_logits, teacher_logits, T=2.0):
    """KL(teacher || student) on temperature-softened distributions, scaled by T^2 (Hinton et al.)."""
    p = F.softmax(teacher_logits / T, -1)
    return F.kl_div(F.log_softmax(student_logits / T, -1), p, reduction="batchmean") * T * T
