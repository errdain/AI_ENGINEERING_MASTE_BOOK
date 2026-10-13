import copy
import math
import random

import pytest
import torch
import torch.nn.functional as F

from alignment import bradley_terry_loss, distill_kl, dpo_loss
from data import (END, PAD, USER, ASST, collate, decode, encode, render_chat, sft_example, split_combos, answer)
from lora import (LoRALinear, apply_lora, dequantize_4bit, finetune_memory_gb, lora_param_count, merge_all,
                  plain_state_dict, quantize_4bit)
from merge import add_task_vectors, lerp_state, task_vector
from tiny_lm import TinyLM, lm_loss, n_params, seq_logprob

torch.manual_seed(0)


def fresh():
    torch.manual_seed(0)
    return TinyLM()


def randomize_adapters(model):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.B, std=0.1)


X = torch.randint(4, 20, (3, 12))


# ---------------------------------------------------------------- LoRA
def test_lora_init_is_identity():
    base = fresh().eval()
    ref = base(X)
    wrapped = copy.deepcopy(base)
    apply_lora(wrapped, ("q_proj", "v_proj"), r=4, alpha=8)
    assert torch.allclose(wrapped.eval()(X), ref, atol=1e-6)


def test_lora_grads_at_init_only_B():
    m = fresh()
    apply_lora(m, ("q_proj",), r=4, alpha=8)
    lm_loss(m, X[:, :-1], X[:, 1:]).backward()
    for l in (x for x in m.modules() if isinstance(x, LoRALinear)):
        assert l.B.grad.abs().sum() > 0          # B receives gradient
        assert l.A.grad.abs().sum() == 0          # A gets zero grad while B == 0
        assert l.base.weight.grad is None         # frozen


def test_merged_weights_equal_adapter_output():
    m = fresh()
    apply_lora(m, ("q_proj", "v_proj", "fc1"), r=4, alpha=16)
    randomize_adapters(m)
    m.eval()
    with torch.no_grad():
        before = m(X)
        merge_all(m)
        after = m(X)
    assert torch.allclose(before, after, atol=1e-5)
    assert not torch.allclose(before, fresh().eval()(X), atol=1e-3)  # adapter really changed the function


def test_merge_unmerge_roundtrip_and_plain_state():
    m = fresh()
    w0 = m.blocks[0].q_proj.weight.clone()
    apply_lora(m, ("q_proj",), r=2, alpha=4)
    randomize_adapters(m)
    l = m.blocks[0].q_proj
    l.merge(); assert not torch.allclose(l.base.weight, w0)
    sd = plain_state_dict(m)
    assert "blocks.0.q_proj.weight" in sd and not any(k.endswith((".A", ".B")) for k in sd)
    fresh_model = fresh(); fresh_model.load_state_dict(sd)           # loads into a plain model
    l.unmerge(); assert torch.allclose(l.base.weight, w0, atol=1e-6)


def test_delta_weight_formula():
    m = fresh(); apply_lora(m, ("q_proj",), r=3, alpha=9)
    randomize_adapters(m)
    l = m.blocks[0].q_proj
    assert l.delta_weight().shape == l.base.weight.shape
    assert torch.allclose(l.delta_weight(), 3.0 * l.B @ l.A)          # alpha/r = 3
    assert torch.linalg.matrix_rank(l.delta_weight()) <= 3


@pytest.mark.parametrize("targets,r", [(("q_proj", "v_proj"), 4), (("fc1", "fc2"), 2), (("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"), 8)])
def test_trainable_param_count_formula(targets, r):
    m = fresh()
    shapes = [(l.out_features, l.in_features) for mod in m.modules() for n, l in mod.named_children()
              if n in targets and isinstance(l, torch.nn.Linear)]
    apply_lora(m, targets, r=r, alpha=2 * r)
    assert n_params(m, trainable_only=True) == lora_param_count(shapes, r)
    # only adapters train
    assert all(n.endswith((".A", ".B")) for n, p in m.named_parameters() if p.requires_grad)


def test_lora_short_training_reduces_loss():
    m = fresh(); apply_lora(m, ("q_proj", "v_proj", "fc1", "fc2"), r=4, alpha=8)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x, y = X[:, :-1], X[:, 1:]
    l0 = lm_loss(m, x, y).item()
    for _ in range(40):
        opt.zero_grad(); lm_loss(m, x, y).backward(); opt.step()
    assert lm_loss(m, x, y).item() < l0 - 0.3


# ---------------------------------------------------------------- QLoRA-lite
def test_quantize_4bit_levels_and_error():
    w = torch.randn(10, 37)
    q, absmax, shape, pad = quantize_4bit(w, block=16)
    assert q.min() >= -8 and q.max() <= 7
    assert len(torch.unique(q)) <= 16
    back = dequantize_4bit(q, absmax, shape, pad)
    assert back.shape == w.shape
    assert (back - w).abs().max() <= absmax.max() / 7 / 2 + 1e-6      # half a step at most
    q2 = quantize_4bit(back, block=16)
    assert torch.allclose(dequantize_4bit(*q2), back, atol=1e-6)      # idempotent


def test_memory_calculator_ordering():
    full = finetune_memory_gb(7, "full_fp16_adam")
    lora = finetune_memory_gb(7, "lora_fp16", r_params_m=20)
    q = finetune_memory_gb(7, "qlora_4bit", r_params_m=20)
    assert full == pytest.approx(7 * 16)                              # 16 bytes/param
    assert q < lora < full
    assert q < 6                                                       # ~3.7GB weights + adapter state


# ---------------------------------------------------------------- alignment math
def test_dpo_identity_at_reference_is_log2():
    a = torch.randn(8)
    loss, st = dpo_loss(a, a - 1, a, a - 1, beta=0.3)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert st["margin"] == 0


def test_dpo_antisymmetry_and_monotonic():
    pc, pr, rc, rr = (torch.tensor([x]) for x in (-3.0, -5.0, -4.0, -4.5))
    l_fwd, st = dpo_loss(pc, pr, rc, rr, 0.5)
    l_swp, _ = dpo_loss(pr, pc, rr, rc, 0.5)                         # swap chosen/rejected -> margin negated
    assert (l_swp - l_fwd).item() == pytest.approx(st["margin"], abs=1e-6)   # L(-m) - L(m) = m
    losses = [dpo_loss(torch.tensor([-3.0 + d]), pr, rc, rr, 0.5)[0].item() for d in (-1, 0, 1, 2)]
    assert losses == sorted(losses, reverse=True)                     # raising chosen log-prob lowers loss


def test_dpo_gradient_directions_and_beta_zero():
    pc = torch.tensor([-3.0], requires_grad=True); pr = torch.tensor([-5.0], requires_grad=True)
    ref = torch.tensor([-4.0])
    dpo_loss(pc, pr, ref, ref, 0.5)[0].backward()
    assert pc.grad < 0 and pr.grad > 0                                # descent: raise chosen, lower rejected
    loss0, _ = dpo_loss(torch.tensor([-1.0]), torch.tensor([-9.0]), ref, ref, beta=0.0)
    assert loss0.item() == pytest.approx(math.log(2))                 # beta=0 -> no signal


def test_bradley_terry():
    z = torch.zeros(4)
    assert bradley_terry_loss(z, z).item() == pytest.approx(math.log(2))
    assert bradley_terry_loss(z + 2, z).item() < bradley_terry_loss(z + 1, z).item() < math.log(2)
    assert bradley_terry_loss(z, z + 2).item() > math.log(2)
    p = torch.sigmoid(torch.tensor(1.5) - torch.tensor(0.5))          # P(chosen wins) = sigmoid(diff)
    assert p.item() == pytest.approx(1 / (1 + math.exp(-1.0)))


def test_distill_kl():
    t = torch.randn(5, 10)
    assert distill_kl(t, t).item() == pytest.approx(0, abs=1e-6)
    assert distill_kl(torch.randn(5, 10), t).item() > 0


def test_seq_logprob_matches_manual():
    m = fresh().eval()
    ids, mask = X.clone(), torch.zeros_like(X, dtype=torch.float)
    mask[:, 6:] = 1
    lp = torch.log_softmax(m(ids[:, :-1]), -1)
    manual = torch.stack([sum(lp[b, t - 1, ids[b, t]] for t in range(6, 12)) for b in range(3)])
    assert torch.allclose(seq_logprob(m, ids, mask), manual, atol=1e-5)


# ---------------------------------------------------------------- data / templates / merging
def test_chat_template_and_special_tokens():
    s = render_chat([{"role": "user", "content": "up cat"}, {"role": "assistant", "content": "CAT"}])
    assert s == "<|user|>up cat<|asst|>CAT<|end|>"
    ids = encode(s)
    assert ids[0] == USER and ids[-1] == END and ids.count(ASST) == 1
    assert decode(ids) == s
    assert render_chat([{"role": "user", "content": "x"}], True).endswith("<|asst|>")
    assert encode("<|end|>") == [END]    # user text containing a special string becomes the control token (injection risk)


def test_sft_loss_mask_covers_only_response():
    ids, mask = sft_example("up", ["cat", "dog"])
    n_resp = len(encode("CAT DOG")) + 1
    assert sum(mask) == n_resp and mask[-n_resp:] == [1] * n_resp
    x, y, m = collate([(ids, mask)])
    assert m.sum() == n_resp and y[0][m[0] > 0][-1] == END


def test_split_is_disjoint_and_answers():
    tr, te = split_combos()
    assert not {tuple(w) for w in tr} & {tuple(w) for w in te}
    assert answer("up", ["cat", "dog"]) == "CAT DOG" and answer("rv", ["cat", "dog"]) == "dog cat"


def test_lerp_and_task_arithmetic():
    a, b = fresh().state_dict(), TinyLM().state_dict()
    for k in a:
        assert torch.equal(lerp_state(a, b, 0.0)[k], a[k])
        assert torch.allclose(lerp_state(a, b, 1.0)[k], b[k])
    tv = task_vector(b, a)
    rec = add_task_vectors(a, [tv], [1.0])
    assert all(torch.allclose(rec[k], b[k], atol=1e-6) for k in a)
    half = add_task_vectors(a, [tv], [0.5])
    assert all(torch.allclose(half[k], lerp_state(a, b, .5)[k], atol=1e-6) for k in a)
