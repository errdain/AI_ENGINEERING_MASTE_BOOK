"""KV-cache memory calculator (11.17)."""
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


def kv_bytes(layers, kv_heads, head_dim, seq_len, batch=1, dtype="fp16"):
    """2 (K and V) * layers * kv_heads * head_dim * seq_len * batch * bytes_per_elem"""
    return 2 * layers * kv_heads * head_dim * seq_len * batch * DTYPE_BYTES[dtype]


def kv_bytes_per_token(layers, kv_heads, head_dim, dtype="fp16"):
    return kv_bytes(layers, kv_heads, head_dim, 1, 1, dtype)


def max_concurrent(gpu_gb, weights_gb, per_seq_bytes, overhead_frac=0.1):
    free = gpu_gb * (1 - overhead_frac) - weights_gb
    return max(0, int(free * 1e9 // per_seq_bytes))
