"""Tiny reverse-mode (Value) and forward-mode (Dual) automatic differentiation for scalars (Tier 3.19-3.21)."""
from __future__ import annotations

import math


class Value:
    """A scalar node in a computational graph. `backward()` applies the chain rule in reverse topological order."""

    def __init__(self, data: float, parents: tuple["Value", ...] = (), op: str = ""):
        self.data, self.grad, self._parents, self._op, self._backward = float(data), 0.0, parents, op, lambda: None

    def _wrap(o): return o if isinstance(o, Value) else Value(o)

    def __add__(self, o):
        o = Value._wrap(o); out = Value(self.data + o.data, (self, o), "+")
        def bw(): self.grad += out.grad; o.grad += out.grad
        out._backward = bw; return out

    def __mul__(self, o):
        o = Value._wrap(o); out = Value(self.data * o.data, (self, o), "*")
        def bw(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = bw; return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")
        def bw(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw; return out

    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-Value._wrap(o))
    def __truediv__(self, o): return self * Value._wrap(o) ** -1
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __rsub__(self, o): return Value._wrap(o) - self
    def __rtruediv__(self, o): return Value._wrap(o) / self

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def bw(): self.grad += out.data * out.grad
        out._backward = bw; return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def bw(): self.grad += out.grad / self.data
        out._backward = bw; return out

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), "tanh")
        def bw(): self.grad += (1 - t**2) * out.grad
        out._backward = bw; return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def bw(): self.grad += (self.data > 0) * out.grad
        out._backward = bw; return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data)) if self.data >= 0 else math.exp(self.data) / (1 + math.exp(self.data)); out = Value(s, (self,), "sigmoid")
        def bw(): self.grad += s * (1 - s) * out.grad
        out._backward = bw; return out

    def _topo(self):
        """Iterative post-order DFS (a recursive version overflows the Python stack on long chains of operations)."""
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            v, done = stack.pop()
            if done: topo.append(v); continue
            if id(v) in seen: continue
            seen.add(id(v)); stack.append((v, True)); stack.extend((p, False) for p in v._parents)
        return topo

    def backward(self):
        topo = self._topo(); self.grad = 1.0
        for v in reversed(topo): v._backward()

    def graph_size(self) -> int: return len(self._topo())

    def __repr__(self): return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"


class Dual:
    """Forward mode: carries (value, derivative with respect to ONE chosen input) through each operation."""

    def __init__(self, v: float, d: float = 0.0): self.v, self.d = float(v), float(d)
    def _w(o): return o if isinstance(o, Dual) else Dual(o, 0.0)
    def __add__(self, o): o = Dual._w(o); return Dual(self.v + o.v, self.d + o.d)
    def __mul__(self, o): o = Dual._w(o); return Dual(self.v * o.v, self.d * o.v + self.v * o.d)
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __sub__(self, o): o = Dual._w(o); return Dual(self.v - o.v, self.d - o.d)
    def __pow__(self, k): return Dual(self.v ** k, k * self.v ** (k - 1) * self.d)
    def exp(self): e = math.exp(self.v); return Dual(e, e * self.d)
    def tanh(self): t = math.tanh(self.v); return Dual(t, (1 - t**2) * self.d)
    def log(self): return Dual(math.log(self.v), self.d / self.v)
