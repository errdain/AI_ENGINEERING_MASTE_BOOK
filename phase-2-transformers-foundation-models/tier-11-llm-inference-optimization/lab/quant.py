"""Weight-only quantization: symmetric INT8 (per-tensor / per-channel) and group-wise INT4 (simulated in int8 storage)."""
import torch


def quantize_sym(w, bits=8, dim=None, group=None):
    """Return (q int tensor, scale). dim=None per-tensor; dim=0 per-output-channel (row); group=g groups along last dim."""
    qmax = 2 ** (bits - 1) - 1
    if group:
        shp = w.shape
        wg = w.reshape(-1, group)
        s = wg.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
        q = torch.clamp(torch.round(wg / s), -qmax - 1, qmax)
        return q.to(torch.int8).reshape(shp), s
    if dim is None:
        s = w.abs().max().clamp_min(1e-12) / qmax
    else:
        s = w.abs().amax(1 - dim if w.dim() == 2 else dim, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(w / s), -qmax - 1, qmax)
    return q.to(torch.int8), s


def dequantize(q, s, group=None):
    if group:
        return (q.reshape(-1, group).float() * s).reshape(q.shape)
    return q.float() * s


def quant_error(w, bits=8, dim=None, group=None):
    q, s = quantize_sym(w, bits, dim, group)
    d = dequantize(q, s, group)
    err = (w - d)
    return dict(max_abs=float(err.abs().max()), rmse=float(err.pow(2).mean().sqrt()),
                step=float(s.max()), rel=float(err.norm() / w.norm()))


class QLinear(torch.nn.Module):
    """Linear with int8 per-output-channel weights, dequantized on the fly (weight-only quantization)."""
    def __init__(self, lin: torch.nn.Linear, bits=8, group=None):
        super().__init__()
        self.group = group
        self.q, self.s = quantize_sym(lin.weight.data, bits, dim=0 if not group else None, group=group)
        self.bias = lin.bias

    def forward(self, x):
        return torch.nn.functional.linear(x, dequantize(self.q, self.s, self.group), self.bias)


def quantize_model(model, bits=8, group=None):
    """Replace every nn.Linear (except nothing excluded) in-place with QLinear. Returns the model."""
    for name, m in list(model.named_children()):
        if isinstance(m, torch.nn.Linear) and (not group or m.weight.shape[-1] % group == 0):
            setattr(model, name, QLinear(m, bits, group))
        else:
            quantize_model(m, bits, group)
    return model
