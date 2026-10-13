"""Weight-space merging (10.30-10.32): linear interpolation and task arithmetic."""
import torch


def task_vector(finetuned: dict, base: dict):
    return {k: finetuned[k] - base[k] for k in base}


def lerp_state(a: dict, b: dict, t: float):
    """(1-t)*a + t*b  -- 'model soup' / checkpoint interpolation."""
    return {k: (1 - t) * a[k] + t * b[k] for k in a}


def add_task_vectors(base: dict, vectors: list, weights: list):
    out = {k: v.clone() for k, v in base.items()}
    for tv, w in zip(vectors, weights):
        for k in out:
            out[k] += w * tv[k]
    return out


def average_states(states: list):
    return {k: sum(s[k] for s in states) / len(states) for k in states[0]}
