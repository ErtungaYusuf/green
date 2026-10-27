"""A transformer (Qwen2 / Llama layout) as int8 layers whose every step can be checked.

Every matrix product inside a layer is int8 x int8 with exact integer sums, the same
operation Green's mining runs. Weights are int8 per output channel; activations are
int8 per group of GROUP channels, so one outlier channel spoils only its own group.
The residual stream is fixed point (FP units per 1.0). Done this way a layer's output
is the same on every machine, up to one unit of rounding in the steps between the
products, and anyone holding the weights can check a layer's work:

- each product Y = X @ W^T with Freivalds' test against a secret vector r: Y @ r must
  equal X @ (W^T r), and W^T r is computed once per weight matrix, so checking costs
  about b * (k + n) where computing cost b * k * n (hundreds of times less);
- the steps between products (norms, attention, SiLU, rounding) by doing them again,
  which is cheap next to the products.

A layer's work for one pass is a Record: its input, everything it computed, its output.
Its products need not be committed to: Freivalds' test pins each one to its committed
input, so a miner opening a layer cannot offer any product but the true one.
"""

import json
import os
import struct

import numpy as np

GROUP = 128                  # activation channels sharing one int8 scale
FP = 4096                    # residual stream: fixed point, FP units per 1.0
TOL = 1                      # rounding slack allowed on a recomputed integer (one unit)
STOL = 1e-9                  # relative slack on a recomputed scale


# --- loading -----------------------------------------------------------------------------

def read_safetensors(path: str) -> dict:
    """Tensors as float32, without the safetensors package (bf16 is widened by hand)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        buf = raw[a:b]
        if info["dtype"] == "BF16":
            arr = (np.frombuffer(buf, "<u2").astype(np.uint32) << 16).view(np.float32)
        elif info["dtype"] == "F16":
            arr = np.frombuffer(buf, "<f2").astype(np.float32)
        elif info["dtype"] == "F32":
            arr = np.frombuffer(buf, "<f4").copy()
        else:
            raise ValueError(f"unsupported dtype {info['dtype']} for {name}")
        out[name] = arr.reshape(info["shape"])
    return out


class QLinear:
    """y = x @ W^T (+ bias) with W as int8 per output channel."""

    def __init__(self, w: np.ndarray, bias: np.ndarray | None = None):
        n, k = w.shape
        if k % GROUP:
            raise ValueError(f"input width {k} is not a multiple of {GROUP}")
        s = np.abs(w).max(axis=1).astype(np.float64) / 127
        s[s == 0] = 1.0
        q = np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int8)
        self.n, self.k, self.groups = n, k, k // GROUP
        self.s = s
        self.bias = None if bias is None else bias.astype(np.float64)
        # (groups, GROUP, n) as float32: every partial sum is an integer below 2^24, so exact
        self.wg = q.astype(np.float32).reshape(n, self.groups, GROUP).transpose(1, 2, 0).copy()

    def product(self, xq: np.ndarray) -> np.ndarray:
        """xq (b, groups, GROUP) int8 -> Y (b, groups, n) int32: exact partial products."""
        p = np.matmul(np.ascontiguousarray(xq.transpose(1, 0, 2), dtype=np.float32), self.wg)
        return p.transpose(1, 0, 2).astype(np.int32)

    def value(self, y: np.ndarray, sx: np.ndarray) -> np.ndarray:
        out = np.matmul(sx[:, None, :], y.astype(np.float64))[:, 0, :] * self.s
        return out if self.bias is None else out + self.bias

    def sketch(self, r: np.ndarray) -> np.ndarray:
        """W^T r per group, (groups, GROUP): what Freivalds' test needs, computed once."""
        return np.matmul(self.wg.astype(np.float64), r.astype(np.float64))


def quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x (b, k) float -> int8 (b, k/GROUP, GROUP) and a scale per group (b, k/GROUP)."""
    b, k = x.shape
    g = x.reshape(b, k // GROUP, GROUP)
    s = np.abs(g).max(axis=2) / 127
    s[s == 0] = 1.0
    return np.clip(np.round(g / s[:, :, None]), -127, 127).astype(np.int8), s


def rmsnorm(h: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    return h / np.sqrt((h * h).mean(axis=1, keepdims=True) + eps) * gamma


def rope(x: np.ndarray, pos: np.ndarray, theta: float) -> np.ndarray:
    """x (b, heads, hd), rotate-half layout."""
    hd = x.shape[-1]
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class Layer:
    LINEARS = ("q", "k", "v", "o", "gate", "up", "down")

    def __init__(self, t: dict, p: str, cfg: dict):
        self.cfg = cfg
        self.ln1 = t[p + "input_layernorm.weight"].astype(np.float64)
        self.ln2 = t[p + "post_attention_layernorm.weight"].astype(np.float64)
        a, m = p + "self_attn.", p + "mlp."
        self.q = QLinear(t[a + "q_proj.weight"], t.get(a + "q_proj.bias"))
        self.k = QLinear(t[a + "k_proj.weight"], t.get(a + "k_proj.bias"))
        self.v = QLinear(t[a + "v_proj.weight"], t.get(a + "v_proj.bias"))
        self.o = QLinear(t[a + "o_proj.weight"])
        self.gate = QLinear(t[m + "gate_proj.weight"])
        self.up = QLinear(t[m + "up_proj.weight"])
        self.down = QLinear(t[m + "down_proj.weight"])


class Model:
    """The int8 model. A miner needs only its own layers; the buyer needs the embedding
    and the head, and the layers it wants to check."""

    def __init__(self, path: str, layers: range | None = None, head: bool = True):
        with open(os.path.join(path, "config.json")) as f:
            self.cfg = cfg = json.load(f)
        t = read_safetensors(os.path.join(path, "model.safetensors"))
        self.n_layers = cfg["num_hidden_layers"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = cfg["hidden_size"] // self.heads
        self.eps, self.theta = cfg["rms_norm_eps"], cfg.get("rope_theta", 10000.0)
        wanted = range(self.n_layers) if layers is None else layers
        self.layers = {i: Layer(t, f"model.layers.{i}.", cfg) for i in wanted}
        if head:
            self.embed = t["model.embed_tokens.weight"].astype(np.float32)
            self.norm = t["model.norm.weight"].astype(np.float64)
            self.lm_head = self.embed if cfg.get("tie_word_embeddings") else t["lm_head.weight"].astype(np.float32)

    # --- the buyer's ends ---

    def embed_tokens(self, ids) -> np.ndarray:
        return np.round(self.embed[np.asarray(ids)].astype(np.float64) * FP).astype(np.int64)

    def logits(self, h: np.ndarray) -> np.ndarray:
        x = rmsnorm(h.astype(np.float64) / FP, self.norm, self.eps).astype(np.float32)
        return x @ self.lm_head.T

    # --- one layer, one pass ---

    def forward(self, i: int, h_in: np.ndarray, pos: np.ndarray, past: list) -> dict:
        """Run layer i on h_in (b, d) int64 at positions pos, given the earlier rows of
        its cache (`past`: [(x1q, s1, yk, yv, positions)]). Returns the Record."""
        L = self.layers[i]
        x1q, s1 = quantize(rmsnorm(h_in / FP, L.ln1, self.eps))
        yq, yk, yv = L.q.product(x1q), L.k.product(x1q), L.v.product(x1q)
        aq, sa = quantize(self._attention(L, yq, yk, yv, s1, pos, past))
        yo = L.o.product(aq)
        h_mid = h_in + np.round(L.o.value(yo, sa) * FP).astype(np.int64)
        x2q, s2 = quantize(rmsnorm(h_mid / FP, L.ln2, self.eps))
        yg, yu = L.gate.product(x2q), L.up.product(x2q)
        mq, sm = quantize(self._act(L, yg, yu, s2))
        yd = L.down.product(mq)
        h_out = h_mid + np.round(L.down.value(yd, sm) * FP).astype(np.int64)
        return {"h_in": h_in, "pos": pos, "x1q": x1q, "s1": s1, "yq": yq, "yk": yk, "yv": yv, "aq": aq, "sa": sa,
                "yo": yo, "h_mid": h_mid, "x2q": x2q, "s2": s2, "yg": yg, "yu": yu, "mq": mq, "sm": sm, "yd": yd,
                "h_out": h_out}

    @staticmethod
    def _act(L: Layer, yg, yu, s2) -> np.ndarray:
        g, u = L.gate.value(yg, s2), L.up.value(yu, s2)
        return g / (1 + np.exp(-g)) * u

    def _kv(self, L: Layer, yk, yv, s1, pos):
        b = yk.shape[0]
        k = rope(L.k.value(yk, s1).reshape(b, self.kv_heads, self.hd), pos, self.theta)
        return k, L.v.value(yv, s1).reshape(b, self.kv_heads, self.hd)

    def _attention(self, L: Layer, yq, yk, yv, s1, pos, past) -> np.ndarray:
        b = yq.shape[0]
        q = rope(L.q.value(yq, s1).reshape(b, self.heads, self.hd), pos, self.theta)
        ks, vs, ps = [], [], []
        for (px1q, ps1, pyk, pyv, ppos) in past:
            k, v = self._kv(L, pyk, pyv, ps1, ppos)
            ks.append(k), vs.append(v), ps.append(ppos)
        k, v = self._kv(L, yk, yv, s1, pos)
        K, V, P = np.concatenate(ks + [k]), np.concatenate(vs + [v]), np.concatenate(ps + [pos])
        rep = self.heads // self.kv_heads
        K, V = np.repeat(K, rep, axis=1), np.repeat(V, rep, axis=1)          # (T, heads, hd)
        att = np.einsum("bhd,thd->bht", q, K) / np.sqrt(self.hd)
        att = np.where(P[None, None, :] <= pos[:, None, None], att, -np.inf)  # causal
        att = np.exp(att - att.max(axis=2, keepdims=True))
        att /= att.sum(axis=2, keepdims=True)
        return np.einsum("bht,thd->bhd", att, V).reshape(b, self.heads * self.hd)

    # --- checking a layer's work ---

    def sketches(self, i: int, r: np.ndarray) -> dict:
        """W^T r for each of layer i's weight matrices: computed once per secret r."""
        L = self.layers[i]
        out = {}
        for name in Layer.LINEARS:
            lin = getattr(L, name)
            out[name] = lin.sketch(r[:lin.n])
        return out

    def check(self, i: int, rec: dict, past: list, r: np.ndarray, sk: dict) -> str | None:
        """None if the Record is layer i's honest work on rec["h_in"] (given `past`, the
        cache rows the buyer has already checked against their commitments), else what
        is wrong. Products: Freivalds' test; everything else: done again."""
        L = self.layers[i]

        def freivalds(name, xq, y):
            lin = getattr(L, name)
            lhs = (y.astype(np.float64) @ r[:lin.n].astype(np.float64))                # (b, groups)
            rhs = np.einsum("bgk,gk->bg", xq.astype(np.float64), sk[name])
            return None if np.array_equal(lhs, rhs) else f"layer {i}: the {name} product is wrong"

        def close(name, got, want):
            if got.shape != want.shape:
                return f"layer {i}: {name} has the wrong shape"
            if got.dtype.kind == "f":
                ok = np.all(np.abs(got - want) <= STOL * np.abs(want))
            else:
                ok = np.all(np.abs(got.astype(np.int64) - want.astype(np.int64)) <= TOL)
            return None if ok else f"layer {i}: {name} does not follow from its inputs"

        h_in, pos = rec["h_in"], rec["pos"]
        x1q, s1 = quantize(rmsnorm(h_in / FP, L.ln1, self.eps))
        checks = [lambda: close("x1q", rec["x1q"], x1q), lambda: close("s1", rec["s1"], s1)]
        for name, y in (("q", "yq"), ("k", "yk"), ("v", "yv")):
            checks.append(lambda name=name, y=y: freivalds(name, rec["x1q"], rec[y]))
        for (px1q, ps1, pyk, pyv, ppos) in past:                     # the cache rows' own products
            checks.append(lambda a=px1q, b=pyk: freivalds("k", a, b))
            checks.append(lambda a=px1q, b=pyv: freivalds("v", a, b))
        for c in checks:
            if (err := c()) is not None:
                return err
        aq, sa = quantize(self._attention(L, rec["yq"], rec["yk"], rec["yv"], rec["s1"], pos, past))
        for err in (close("aq", rec["aq"], aq), close("sa", rec["sa"], sa), freivalds("o", rec["aq"], rec["yo"])):
            if err:
                return err
        h_mid = h_in + np.round(L.o.value(rec["yo"], rec["sa"]) * FP).astype(np.int64)
        x2q, s2 = quantize(rmsnorm(rec["h_mid"] / FP, L.ln2, self.eps))
        for err in (close("h_mid", rec["h_mid"], h_mid), close("x2q", rec["x2q"], x2q), close("s2", rec["s2"], s2),
                    freivalds("gate", rec["x2q"], rec["yg"]), freivalds("up", rec["x2q"], rec["yu"])):
            if err:
                return err
        mq, sm = quantize(self._act(L, rec["yg"], rec["yu"], rec["s2"]))
        for err in (close("mq", rec["mq"], mq), close("sm", rec["sm"], sm), freivalds("down", rec["mq"], rec["yd"])):
            if err:
                return err
        h_out = rec["h_mid"] + np.round(L.down.value(rec["yd"], rec["sm"]) * FP).astype(np.int64)
        return close("h_out", rec["h_out"], h_out)
