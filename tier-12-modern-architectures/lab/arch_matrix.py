"""Architecture comparison matrix (Tier 12 build).

Pure-Python formulas for: parameters (total / active), KV-cache (or recurrent state)
memory, forward FLOPs per generated token, and context length -- for dense, GQA,
sliding-window, MoE, SSM and hybrid layouts.

Conventions (all documented so tests can hand-check them):
  * bias terms and norm weights are ignored except RMSNorm-style 2*d per layer + d final
    (so counts are within ~1% of published numbers for bias-free models).
  * FLOPs = 2 * multiply-accumulates.  Only matmul-like work is counted.
  * "FLOPs/token" = decode-time forward cost for ONE new token at context length n.
  * Case-study presets are CASE STUDIES, not authoritative: every number is
    REVIEW_REQUIRED against the official model card / config before external use.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, field
import math


@dataclass(frozen=True)
class ArchSpec:
    name: str
    kind: str                      # "decoder" | "encoder" | "encdec"
    d_model: int
    n_layers: int                  # decoder layers (or encoder layers for kind="encoder")
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab: int
    head_dim: int = 0              # 0 -> d_model // n_heads
    gated_ffn: bool = True         # SwiGLU-style 3 matrices; False -> 2 matrices
    tie_embeddings: bool = False
    n_experts: int = 1             # >1 -> sparse MoE FFN in every layer
    top_k: int = 1
    pattern: str = ""              # per-layer mixer: A=global attn, L=local (sliding) attn, S=SSM. "" -> all A
    window: int = 4096             # sliding window size for 'L' layers
    ssm_state: int = 16            # N
    ssm_expand: int = 2            # E (inner width = E*d)
    ssm_conv: int = 4
    ffn_in_ssm_layers: bool = False  # Mamba-1 style: SSM layer has no separate FFN
    n_enc_layers: int = 0          # encdec only: encoder depth (n_layers = decoder depth)
    learned_pos: int = 0           # max positions for learned absolute embeddings (0 = RoPE/none)
    context: int = 4096            # trained / advertised context (REVIEW_REQUIRED for presets)
    status: str = "ILLUSTRATIVE"   # ILLUSTRATIVE | CASE_STUDY (REVIEW_REQUIRED)

    @property
    def dh(self) -> int:
        return self.head_dim or self.d_model // self.n_heads

    def layers(self) -> str:
        p = self.pattern or "A" * self.n_layers
        if len(p) != self.n_layers:
            raise ValueError(f"pattern length {len(p)} != n_layers {self.n_layers}")
        return p


def tile_pattern(unit: str, n_layers: int) -> str:
    """Repeat `unit` (e.g. 'LLLLLA') until n_layers characters."""
    return (unit * math.ceil(n_layers / len(unit)))[:n_layers]


# ---------------------------------------------------------------- parameters
def attn_params(s: ArchSpec) -> int:
    d, dh = s.d_model, s.dh
    return d * s.n_heads * dh + 2 * d * s.n_kv_heads * dh + s.n_heads * dh * d


def ffn_params_one(s: ArchSpec) -> int:
    return (3 if s.gated_ffn else 2) * s.d_model * s.d_ff


def ffn_params_layer(s: ArchSpec, active: bool) -> int:
    if s.n_experts == 1:
        return ffn_params_one(s)
    k = s.top_k if active else s.n_experts
    return k * ffn_params_one(s) + s.d_model * s.n_experts  # + router


def ssm_params(s: ArchSpec) -> int:
    """Mamba-1-style selective SSM block (approximate; see Tier 12 sec. 12)."""
    d, e, n = s.d_model, s.ssm_expand, s.ssm_state
    di = e * d
    dt_rank = math.ceil(d / 16)
    return (d * 2 * di              # in_proj (x and gate)
            + di * s.ssm_conv       # depthwise conv
            + di * (dt_rank + 2 * n)  # x_proj -> (dt, B, C)
            + dt_rank * di          # dt_proj
            + di * n + di           # A (log) and D
            + di * d)               # out_proj


def embedding_params(s: ArchSpec) -> int:
    return s.vocab * s.d_model + s.learned_pos * s.d_model


def head_params(s: ArchSpec) -> int:
    return 0 if (s.tie_embeddings or s.kind == "encoder") else s.vocab * s.d_model


def _layer_params(s: ArchSpec, mixer: str, active: bool) -> int:
    mix = ssm_params(s) if mixer == "S" else attn_params(s)
    ffn = ffn_params_layer(s, active) if (mixer != "S" or s.ffn_in_ssm_layers) else 0
    return mix + ffn + 2 * s.d_model


def total_params(s: ArchSpec, active: bool = False) -> int:
    n = embedding_params(s) + head_params(s) + s.d_model
    n += sum(_layer_params(s, m, active) for m in s.layers())
    if s.kind == "encdec":
        enc = attn_params(s) + ffn_params_layer(s, active) + 2 * s.d_model
        n += s.n_enc_layers * enc
        n += s.n_layers * (attn_params(s) + s.d_model)   # cross-attention in each decoder layer
    return n


def active_params(s: ArchSpec) -> int:
    """Parameters touched per token (== total for dense models)."""
    return total_params(s, active=True)


# ---------------------------------------------------------------- memory
def kv_bytes_per_token(s: ArchSpec, dtype_bytes: int = 2) -> int:
    """MARGINAL cache growth per extra context token (global-attention layers only)."""
    if s.kind == "encoder":
        return 0
    n_global = s.layers().count("A")
    return 2 * n_global * s.n_kv_heads * s.dh * dtype_bytes


def ssm_state_bytes(s: ArchSpec, dtype_bytes: int = 2) -> int:
    """Fixed recurrent state (+conv buffer) per sequence, independent of context length."""
    di = s.ssm_expand * s.d_model
    per = di * s.ssm_state + di * (s.ssm_conv - 1)
    return s.layers().count("S") * per * dtype_bytes


def cache_bytes(s: ArchSpec, n: int, dtype_bytes: int = 2) -> int:
    """Total per-sequence decode-time memory (KV + state) at context length n."""
    if s.kind == "encoder":
        return 0
    per_tok_layer = 2 * s.n_kv_heads * s.dh * dtype_bytes
    p = s.layers()
    total = p.count("A") * n * per_tok_layer
    total += p.count("L") * min(n, s.window) * per_tok_layer
    return total + ssm_state_bytes(s, dtype_bytes)


def weight_bytes(s: ArchSpec, dtype_bytes: int = 2) -> int:
    return total_params(s) * dtype_bytes


def max_concurrent_sequences(s: ArchSpec, gpu_gb: float, n: int, dtype_bytes: int = 2) -> int:
    free = gpu_gb * 1e9 - weight_bytes(s, dtype_bytes)
    c = cache_bytes(s, n, dtype_bytes)
    return max(0, int(free // c)) if c > 0 else 0


# ---------------------------------------------------------------- compute
def matmul_params_active(s: ArchSpec) -> int:
    """Active parameters that take part in matmuls for one decoded token.
    Excludes embedding lookup and norm weights; includes the output head (tied or not)."""
    n = s.vocab * s.d_model
    for m in s.layers():
        mix = ssm_params(s) if m == "S" else attn_params(s)
        ffn = ffn_params_layer(s, True) if (m != "S" or s.ffn_in_ssm_layers) else 0
        n += mix + ffn
    return n


def flops_per_token(s: ArchSpec, n: int) -> int:
    """Decode-time forward FLOPs for one new token with n tokens already in context."""
    f = 2 * matmul_params_active(s)
    for m in s.layers():
        if m == "A":
            f += 4 * n * s.n_heads * s.dh              # QK^T and AV: 2*n*H*dh each
        elif m == "L":
            f += 4 * min(n, s.window) * s.n_heads * s.dh
        else:                                          # selective scan update + readout
            f += 6 * s.ssm_expand * s.d_model * s.ssm_state
    return f


# ---------------------------------------------------------------- matrix / reporting
def comparison_rows(specs, n: int = 8192, dtype_bytes: int = 2, gpu_gb: float = 80.0):
    rows = []
    for s in specs:
        rows.append({
            "name": s.name, "status": s.status, "kind": s.kind,
            "total_params_B": total_params(s) / 1e9,
            "active_params_B": active_params(s) / 1e9,
            "kv_kib_per_token": kv_bytes_per_token(s, dtype_bytes) / 1024,
            "cache_gib_at_n": cache_bytes(s, n, dtype_bytes) / 2**30,
            "gflops_per_token": flops_per_token(s, n) / 1e9,
            "context": s.context,
            "weights_gib": weight_bytes(s, dtype_bytes) / 2**30,
            "max_seqs": max_concurrent_sequences(s, gpu_gb, n, dtype_bytes),
        })
    return rows


COLUMNS = [("name", "Model / archetype"), ("kind", "Kind"), ("total_params_B", "Total params (B)"),
           ("active_params_B", "Active params (B)"), ("kv_kib_per_token", "KV KiB/token"),
           ("cache_gib_at_n", "Cache GiB @ n"), ("gflops_per_token", "GFLOPs/token @ n"),
           ("context", "Context"), ("max_seqs", "Max seqs (80GB)")]


def to_markdown(rows) -> str:
    out = ["| " + " | ".join(h for _, h in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        cells = []
        for k, _ in COLUMNS:
            v = r[k]
            cells.append(f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def to_html(rows, n: int, title="Architecture comparison matrix") -> str:
    """Self-contained HTML (light/dark), heat-shaded numeric columns."""
    num = [k for k, _ in COLUMNS if k not in ("name", "kind")]
    mx = {k: max(r[k] for r in rows) or 1 for k in num}
    head = "".join(f"<th>{h}</th>" for _, h in COLUMNS)
    body = ""
    for r in rows:
        tds = ""
        for k, _ in COLUMNS:
            v = r[k]
            if k in num:
                a = 0.10 + 0.5 * (v / mx[k]) if k != "max_seqs" else 0.10 + 0.5 * (v / mx[k])
                txt = f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
                tds += f'<td class="n" style="--a:{a:.2f}">{txt}</td>'
            else:
                tds += f"<td>{v}</td>"
        tag = "case study" if r["status"] == "CASE_STUDY" else "illustrative"
        body += f'<tr class="{r["status"]}">{tds}<td class="tag">{tag}</td></tr>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>:root{{--bg:#fff;--fg:#1d2430;--mut:#667;--card:#f4f6fa;--ac:#4C78A8;--grid:#d8dde6}}@media(prefers-color-scheme:dark){{:root{{--bg:#14181f;--fg:#e6e9ef;--mut:#9aa3b2;--card:#1e242e;--grid:#343c4a}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1200px;margin:auto}}h1{{font-size:20px}}.w{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:5px 9px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}}th{{background:var(--card);position:sticky;top:0}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;background:color-mix(in srgb,var(--ac) calc(var(--a)*100%),transparent)}}td.tag{{color:var(--mut);font-size:12px}}p{{color:var(--mut)}}</style></head><body>
<h1>{title}</h1><p>Context length n = {n:,} tokens, fp16/bf16 (2 bytes). Generated by <code>lab/arch_matrix.py</code>. Rows labelled <b>case study</b> use published-looking configs that are
<b>REVIEW_REQUIRED</b>; <b>illustrative</b> rows are made-up archetypes for comparing ideas. Cells shaded by size within their column.</p>
<div class="w"><table><thead><tr>{head}<th>Status</th></tr></thead><tbody>{body}</tbody></table></div></body></html>"""


# ---------------------------------------------------------------- presets
def _cs(**kw):
    return ArchSpec(status="CASE_STUDY", **kw)


PRESETS = {
    # CASE STUDIES -- every number REVIEW_REQUIRED against the official config
    "BERT-base (case study)": _cs(name="BERT-base (case study)", kind="encoder", d_model=768, n_layers=12, n_heads=12,
                                  n_kv_heads=12, d_ff=3072, vocab=30522, gated_ffn=False, tie_embeddings=True,
                                  learned_pos=512, context=512),
    "GPT-2 small (case study)": _cs(name="GPT-2 small (case study)", kind="decoder", d_model=768, n_layers=12, n_heads=12,
                                    n_kv_heads=12, d_ff=3072, vocab=50257, gated_ffn=False, tie_embeddings=True,
                                    learned_pos=1024, context=1024),
    "T5-base-like (case study)": _cs(name="T5-base-like (case study)", kind="encdec", d_model=768, n_layers=12, n_enc_layers=12,
                                     n_heads=12, n_kv_heads=12, d_ff=3072, vocab=32128, gated_ffn=False,
                                     tie_embeddings=True, context=512),
    "Llama-2-7B-shape (case study)": _cs(name="Llama-2-7B-shape (case study)", kind="decoder", d_model=4096, n_layers=32,
                                         n_heads=32, n_kv_heads=32, d_ff=11008, vocab=32000, context=4096),
    "Llama-3-8B-shape (case study)": _cs(name="Llama-3-8B-shape (case study)", kind="decoder", d_model=4096, n_layers=32,
                                         n_heads=32, n_kv_heads=8, d_ff=14336, vocab=128256, context=8192),
    "Qwen2.5-7B-shape (case study)": _cs(name="Qwen2.5-7B-shape (case study)", kind="decoder", d_model=3584, n_layers=28,
                                         n_heads=28, n_kv_heads=4, d_ff=18944, vocab=152064, context=32768),
    "Gemma-2-9B-shape (case study)": _cs(name="Gemma-2-9B-shape (case study)", kind="decoder", d_model=3584, n_layers=42,
                                         n_heads=16, n_kv_heads=8, head_dim=256, d_ff=14336, vocab=256000,
                                         tie_embeddings=True, pattern=tile_pattern("LA", 42), window=4096, context=8192),
    "Mixtral-8x7B-shape (case study)": _cs(name="Mixtral-8x7B-shape (case study)", kind="decoder", d_model=4096, n_layers=32,
                                           n_heads=32, n_kv_heads=8, d_ff=14336, vocab=32000, n_experts=8, top_k=2,
                                           context=32768),
    # ILLUSTRATIVE archetypes (not any real model)
    "Dense MHA (illustrative)": ArchSpec(name="Dense MHA (illustrative)", kind="decoder", d_model=4096, n_layers=32,
                                            n_heads=32, n_kv_heads=32, d_ff=11008, vocab=32000, context=8192),
    "Dense GQA-8 (illustrative)": ArchSpec(name="Dense GQA-8 (illustrative)", kind="decoder", d_model=4096, n_layers=32,
                                              n_heads=32, n_kv_heads=8, d_ff=11008, vocab=32000, context=8192),
    "Dense MQA (illustrative)": ArchSpec(name="Dense MQA (illustrative)", kind="decoder", d_model=4096, n_layers=32,
                                            n_heads=32, n_kv_heads=1, d_ff=11008, vocab=32000, context=8192),
    "SWA (illustrative)": ArchSpec(name="SWA (illustrative)", kind="decoder", d_model=4096, n_layers=32, n_heads=32,
                                      n_kv_heads=8, d_ff=11008, vocab=32000, pattern="L" * 32, window=4096, context=131072),
    "Pure SSM (illustrative)": ArchSpec(name="Pure SSM (illustrative)", kind="decoder", d_model=4096, n_layers=48,
                                           n_heads=32, n_kv_heads=8, d_ff=11008, vocab=32000, pattern="S" * 48,
                                           ssm_state=16, context=131072),
    "Hybrid 1A:7S (illustrative)": ArchSpec(name="Hybrid 1A:7S (illustrative)", kind="decoder", d_model=4096, n_layers=32,
                                               n_heads=32, n_kv_heads=8, d_ff=11008, vocab=32000,
                                               pattern=tile_pattern("SSSSSSSA", 32), ffn_in_ssm_layers=True, context=131072),
}


def default_matrix(n: int = 8192):
    return comparison_rows(list(PRESETS.values()), n)


if __name__ == "__main__":
    print(to_markdown(default_matrix()))
