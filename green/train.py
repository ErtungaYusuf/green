"""Training a neural network with compute bought on the network.

Almost all the arithmetic of training is matrix products: per layer one in the
forward pass (X·W) and two in the backward pass (the weight gradient Xᵀ·G and
the gradient for the layer below G·Wᵀ). Those go to providers as int8 products
(green/compute.py), checked by Freivalds' test as they come back; everything
else (biases, ReLU, softmax, the loss, the update) is O(n^2) and stays with the
buyer, which also keeps the float32 master copy of the weights.

Each operand is quantized to int8 with one scale per tensor. Gradients use
stochastic rounding (round up with probability equal to the fraction), which
is unbiased, so small gradients are not rounded away and SGD still converges;
the int32 products are exact, so the only error is the quantization itself.

`matmul` is any function computing an exact int8 product: local_matmul, a
ComputeChannel's matmul, or a ComputeCluster's. When it belongs to an object
with matmul_many (channels and clusters), the two products of a layer's
backward pass, which do not depend on each other, are sent together: over the
internet a step then costs 2 round trips per layer instead of 3.
"""

import numpy as np

from . import work

# 5x7 bitmaps of the digits 0-9, for a small dataset that needs no download.
FONT = [
    "01110100011001110101110011000101110", "00100011000010000100001000010001110",
    "01110100010000100010001000100011111", "11111000100010000010000011000101110",
    "00010001100101010010111110001000010", "11111100001111000001000011000101110",
    "00110010001000011110100011000101110", "11111000010001000100010000100001000",
    "01110100011000101110100011000101110", "01110100011000101111000010001001100",
]
SIDE = 16                          # images are SIDE x SIDE


def local_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The exact int8 product on this machine (float64 is exact for these sums)."""
    return work.blas_matmul(a, b, np.int64)


def digits(n: int, seed: int = 0, noise: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """`n` noisy 16x16 images of digits (glyphs drawn at twice their size, shifted at
    random, with Gaussian noise and flipped pixels) and their labels."""
    rng = np.random.default_rng(seed)
    glyphs = [np.kron(np.array([int(c) for c in g], dtype=np.float32).reshape(7, 5), np.ones((2, 2), np.float32))
              for g in FONT]                                          # 14 x 10 each
    y = rng.integers(0, 10, size=n)
    x = np.zeros((n, SIDE, SIDE), dtype=np.float32)
    for i, label in enumerate(y):
        top, left = rng.integers(0, SIDE - 14 + 1), rng.integers(0, SIDE - 10 + 1)
        x[i, top:top + 14, left:left + 10] = glyphs[label] * rng.uniform(0.6, 1.0)
    flips = rng.random(x.shape) < 0.04
    x = np.where(flips, 1.0 - x, x) + rng.normal(0.0, noise, x.shape).astype(np.float32)
    return x.reshape(n, SIDE * SIDE), y


def quantize(x: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Symmetric int8 with one scale per tensor; stochastic rounding when `rng` is given."""
    top = float(np.abs(x).max(initial=0.0))
    if top == 0.0:
        return np.zeros(x.shape, dtype=np.int8), 1.0
    scale = top / 127.0
    y = x / scale
    y = np.floor(y + rng.random(x.shape)) if rng is not None else np.rint(y)
    return np.clip(y, -127, 127).astype(np.int8), scale


def _many(matmul, pairs: list) -> list:
    owner = getattr(matmul, "__self__", None)
    if owner is not None and hasattr(owner, "matmul_many"):
        return owner.matmul_many(pairs)
    return [matmul(a, b) for a, b in pairs]


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """A multilayer perceptron (ReLU, softmax output) trained with SGD and momentum.

    With matmul=None every product is float32 on this machine (the reference);
    otherwise every product is int8 through `matmul`."""

    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0.0, np.sqrt(2.0 / i), (i, o)).astype(np.float32) for i, o in zip(sizes, sizes[1:])]
        self.b = [np.zeros(o, dtype=np.float32) for o in sizes[1:]]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.rng = np.random.default_rng(seed + 1)          # for stochastic rounding
        self.products = 0

    def _product(self, x, w, matmul, stochastic=False):
        """x @ w and what the backward pass needs: (result, quantized x and w with scales)."""
        if matmul is None:
            return x @ w, (x, 1.0, w, 1.0)
        xq, sx = quantize(x, self.rng if stochastic else None)
        wq, sw = quantize(w)
        self.products += 1
        return matmul(xq, wq).astype(np.float32) * np.float32(sx * sw), (xq, sx, wq, sw)

    def forward(self, x: np.ndarray, matmul=None):
        cache, h = [], x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z, q = self._product(h, w, matmul)
            z = z + b
            cache.append((q, z))
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
        return h, cache

    def step(self, x: np.ndarray, y: np.ndarray, matmul=None, lr: float = 0.05, momentum: float = 0.9) -> float:
        logits, cache = self.forward(x, matmul)
        p = softmax(logits)
        n = len(y)
        loss = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
        g = p
        g[np.arange(n), y] -= 1.0
        g /= n                                              # dLoss/dlogits
        for i in reversed(range(len(self.W))):
            (xq, sx, wq, sw), z = cache[i]
            if i < len(self.W) - 1:
                g = g * (z > 0)                             # through the ReLU
            if matmul is None:
                gq, sg = g, 1.0
            else:
                gq, sg = quantize(g, self.rng)              # unbiased rounding for gradients
            db = g.sum(axis=0)
            if matmul is None:
                dw = xq.T @ gq
                if i > 0:
                    g = gq @ wq.T
            else:
                # The weight gradient and the gradient for the layer below are independent.
                pairs = [(np.ascontiguousarray(xq.T), gq)] + ([(gq, np.ascontiguousarray(wq.T))] if i > 0 else [])
                out = _many(matmul, pairs)
                self.products += len(pairs)
                dw = out[0].astype(np.float32) * np.float32(sx * sg)
                if i > 0:
                    g = out[1].astype(np.float32) * np.float32(sg * sw)
            self.vW[i] = momentum * self.vW[i] - lr * dw
            self.vb[i] = momentum * self.vb[i] - lr * db
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]
        return loss

    def accuracy(self, x: np.ndarray, y: np.ndarray, matmul=None) -> float:
        logits, _ = self.forward(x, matmul)
        return float((logits.argmax(axis=1) == y).mean())


def train(model: MLP, x: np.ndarray, y: np.ndarray, steps: int, matmul=None, batch: int = 128,
          lr: float = 0.05, seed: int = 0, report=None, every: int = 50) -> list[float]:
    """Minibatch SGD for `steps` steps; returns the losses. `report(step, loss)` is called every `every` steps."""
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(y), size=batch)
        losses.append(model.step(x[idx], y[idx], matmul, lr))
        if report is not None and (step % every == 0 or step == steps):
            report(step, float(np.mean(losses[-every:])))
    return losses
