"""Hand-written LoRA (10.11-10.14) and a QLoRA-lite 4-bit blockwise quantizer (10.15).

  h = W x + (alpha / r) * B (A x)         W: frozen (out,in)   A: (r,in)   B: (out,r)   B starts at 0
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base, self.r, self.alpha = base, r, alpha
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero => delta W = 0 at init (identity)
        self.merged = False

    @property
    def scale(self):
        return self.alpha / self.r

    def delta_weight(self):
        return self.scale * (self.B @ self.A)

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        return y + self.scale * ((x @ self.A.t()) @ self.B.t())

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False


def apply_lora(model: nn.Module, targets=("q_proj", "v_proj"), r=4, alpha=8.0, freeze_rest=True):
    """Replace every nn.Linear whose attribute name is in `targets`. Returns the list of LoRALinear."""
    if freeze_rest:
        for p in model.parameters():
            p.requires_grad_(False)
    layers = []
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                l = LoRALinear(child, r, alpha)
                setattr(mod, name, l)
                layers.append(l)
    return layers


def lora_param_count(shapes, r):
    """Formula: sum over adapted (out,in) layers of r*(in+out)."""
    return sum(r * (i + o) for o, i in shapes)


def merge_all(model):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


# ---------------------------------------------------------------- QLoRA-lite (uniform int4, absmax per block)
def quantize_4bit(w: torch.Tensor, block=64):
    """Blockwise absmax quantization to 16 signed levels (-8..7). NOT the NF4 codebook that real QLoRA uses
    (NF4 places levels at normal-distribution quantiles); this is the simpler uniform variant."""
    flat = w.detach().reshape(-1)
    pad = (-flat.numel()) % block
    flat = torch.cat([flat, flat.new_zeros(pad)]).view(-1, block)
    absmax = flat.abs().amax(1, keepdim=True).clamp(min=1e-12)
    q = torch.clamp(torch.round(flat / absmax * 7), -8, 7).to(torch.int8)
    return q, absmax, w.shape, pad


def dequantize_4bit(q, absmax, shape, pad):
    flat = (q.float() / 7 * absmax).reshape(-1)
    return (flat[: flat.numel() - pad] if pad else flat).view(shape)


@torch.no_grad()
def quantize_model_inplace(model, targets=("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"), block=64):
    """Store base weights as 4-bit then write the dequantized values back (simulated QLoRA base)."""
    for mod in model.modules():
        for name, child in mod.named_children():
            if name in targets and isinstance(child, nn.Linear):
                child.weight.copy_(dequantize_4bit(*quantize_4bit(child.weight, block)))


def finetune_memory_gb(n_params_b: float, method: str, r_params_m: float = 0.0, block=64):
    """Rough weights+grads+optimizer memory (GB), ignoring activations/KV/CUDA overhead.
    n_params_b: base params in billions. method in {'full_fp16_adam','lora_fp16','qlora_4bit'}.
    Full FT with Adam, mixed precision: fp16 weights(2) + fp16 grads(2) + fp32 master(4) + Adam m,v(8) = 16 B/param."""
    GB = 1e9
    n = n_params_b * 1e9
    lora = r_params_m * 1e6
    if method == "full_fp16_adam":
        return n * 16 / GB
    lora_bytes = lora * (4 + 4 + 8)          # fp32 adapter + grad + Adam m,v
    if method == "lora_fp16":
        return (n * 2 + lora_bytes) / GB
    if method == "qlora_4bit":
        return (n * (0.5 + 4 / block) + lora_bytes) / GB   # 4 bits + one fp32 absmax per block
    raise ValueError(method)


def plain_state_dict(model):
    """State dict with LoRA wrappers folded away: 'x.q_proj.base.weight' -> 'x.q_proj.weight', A/B dropped.
    Call merge_all(model) first so the base weights already contain the adapter delta."""
    out = {}
    for k, v in model.state_dict().items():
        if k.endswith((".A", ".B")):
            continue
        out[k.replace(".base.", ".")] = v.detach().clone()
    return out
