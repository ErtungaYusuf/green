"""Run a quantized neural network's forward pass on the Green network.

The heavy part of neural-network inference is the linear layers -- large int8
matrix multiplications. Those are exactly Green work units, so a forward pass
can be computed by the mining network: each layer's W @ X is submitted as a
compute job, miners deliver it (and earn), and the thin glue between layers
(requantize + ReLU, cheap integer ops) runs on the client. This is what makes
the proof-of-useful-work actually useful: the work that secures the chain is a
real neural network running.

Everything here is integer arithmetic, so the result a miner delivers is
bit-identical to the local reference -- there is no quantization drift between
"computed on Green" and "computed here".

Delivery is optimistic: validators only see result hashes, so the client is
the verifier. It checks every unit it receives with Freivalds (private
randomness, well under a millisecond per unit) and disputes a wrong one on
chain, which costs the cheating miner the bond of its whole batch; the unit is
then redone. A result is used only after it passed that check, so the client
does not have to wait for the challenge window to close.
"""

import os
import time
import urllib.error

import numpy as np

from . import params, tx as txmod, work
from .crypto import hhex
from .node import http_get, http_post, wait_for_block
from .tx import make_tx

N = params.UNIT


def _pad(m: np.ndarray) -> np.ndarray:
    r = -(-m.shape[0] // N) * N
    c = -(-m.shape[1] // N) * N
    out = np.zeros((r, c), dtype=np.int8)
    out[:m.shape[0], :m.shape[1]] = m
    return out


class MatmulClient:
    """Submit A @ B to a running node and get the int64 result back."""

    def __init__(self, node_url: str, secret_hex: str):
        from .crypto import Key
        self.node = node_url.rstrip("/")
        self.key = Key.from_hex(secret_hex)
        self.chain_id = http_get(f"{self.node}/status")["genesis"]
        self._inputs: dict[str, tuple] = {}      # job id -> padded (A, B), to verify results
        self.disputes = 0

    def submit(self, a: np.ndarray, b: np.ndarray, price: int) -> tuple[str, tuple[int, int]]:
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"inner dims differ: {a.shape} x {b.shape}")
        shape = (a.shape[0], b.shape[1])
        pa, pb = _pad(a.astype(np.int8)), _pad(b.astype(np.int8))
        blob = txmod.encode_job(pa, pb)
        dims = txmod.job_dims_units(pa, pb)
        job_id = http_post(f"{self.node}/blob", blob, raw=True)["hash"]
        assert job_id == hhex(blob)
        self._inputs[job_id] = (pa, pb)
        nonce = http_get(f"{self.node}/account/{self.key.address}")["pending_nonce"]
        fields = {"job": job_id, "dims": dims, "price": price}
        t = make_tx(self.key, self.chain_id, "job_post", nonce,
                    txmod.min_fee({"type": "job_post", **fields}), **fields)
        try:
            http_post(f"{self.node}/tx", t)
        except urllib.error.HTTPError as e:
            # Jobs are content-addressed: an identical matmul is the same job.
            # If it was already posted (e.g. a re-run), just use the existing one.
            if e.code == 400 and b"already posted" in e.read():
                return job_id, shape
            raise
        return job_id, shape

    def result(self, job_id: str, shape: tuple[int, int], timeout: float = 120) -> np.ndarray:
        """Wait for every unit, verify each one, dispute wrong or withheld ones; return A @ B."""
        end = time.time() + timeout
        verified: dict[str, tuple[str, np.ndarray]] = {}     # unit key -> (result hash, result)
        disputed: set[str] = set()                            # unit keys we disputed
        tip = None
        while time.time() < end:
            try:
                job = http_get(f"{self.node}/job/{job_id}")
            except urllib.error.HTTPError as e:
                if e.code == 404:                  # not mined into a block yet
                    time.sleep(0.5)
                    continue
                raise
            wrong: dict[str, list] = {}            # batch -> units that failed the check
            for key, rh in job["results"].items():
                if key in verified and verified[key][0] == rh:
                    continue
                c = self._fetch_result(rh)
                if c is None:
                    continue                       # not reachable yet
                unit = [int(x) for x in key.split(",")]
                if not self._check(job_id, unit, c):
                    if key in job["pending"]:
                        wrong.setdefault(job["pending"][key], []).append(unit)
                    continue
                verified[key] = (rh, c)
            # Data that is still missing late in a window counts as withheld.
            for b in job["batches"]:
                if b["status"] != "revealed" or job["height"] + 1 < b["settle_at"] - params.JOB_CHALLENGE_BLOCKS // 2:
                    continue
                for key in b["units"]:
                    if key not in verified and key not in b["proven"]:
                        wrong.setdefault(b["id"], []).append([int(x) for x in key.split(",")])
            for bid, units in wrong.items():
                units = [u for u in units if txmod.unit_key(u) not in disputed]
                if units:
                    disputed.update(txmod.unit_key(u) for u in units)
                    self._dispute(job_id, job["dims"], bid, units)
            if len(verified) == job["units_total"]:
                mu, ku, nu = job["dims"]
                out = np.zeros((mu * N, nu * N), dtype=np.int64)
                for key, (_, c) in verified.items():
                    i, l, j = (int(x) for x in key.split(","))
                    out[i * N:(i + 1) * N, j * N:(j + 1) * N] += c
                return out[:shape[0], :shape[1]]
            tip = wait_for_block(self.node, tip, 0.5)       # wakes early when a block arrives
        raise TimeoutError(f"job {job_id[:12]} not finished in {timeout}s")

    def _fetch_result(self, result_hash: str) -> np.ndarray | None:
        try:
            data = http_get(f"{self.node}/blob/{result_hash}", raw=True)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        if hhex(data) != result_hash:
            return None
        try:
            return txmod.decode_result(data)
        except ValueError:
            return np.full((N, N), np.iinfo(np.int32).max, dtype=np.int32)    # garbage: fails the check

    def _check(self, job_id: str, unit, c: np.ndarray) -> bool:
        inputs = self._inputs.get(job_id)
        if inputs is None:                         # not our submission: nothing to check against
            return True
        ua, ub = txmod.unit_inputs(*inputs, unit)
        return work.freivalds_check(ua, ub, c, os.urandom(32))

    def _dispute(self, job_id: str, dims, batch_id: str, units) -> None:
        nonce = http_get(f"{self.node}/account/{self.key.address}")["pending_nonce"]
        t = make_tx(self.key, self.chain_id, "job_challenge", nonce, txmod.job_challenge_fee(len(units)),
                    job=job_id, batch=batch_id, units=txmod.units_bitmap(units, dims))
        try:
            http_post(f"{self.node}/tx", t)
            self.disputes += 1
        except urllib.error.HTTPError:
            pass                                   # e.g. someone else disputed these units first

    def matmul(self, a: np.ndarray, b: np.ndarray, price: int, timeout: float = 120) -> np.ndarray:
        job_id, shape = self.submit(a, b, price)
        return self.result(job_id, shape, timeout)


def requantize(acc: np.ndarray, shift: int, relu: bool) -> np.ndarray:
    """int32/64 accumulator -> int8 activations for the next layer.

    Rounded arithmetic shift (standard in int8 inference), clamped to int8.
    ReLU is a clamp at zero. Pure integer, so it is deterministic everywhere.
    """
    if relu:
        acc = np.maximum(acc, 0)
    scaled = (acc + (1 << (shift - 1))) >> shift if shift > 0 else acc
    return np.clip(scaled, -127, 127).astype(np.int8)


class QuantMLP:
    """A simple int8-quantized multilayer perceptron.

    Each layer is y = requantize(W @ x), with ReLU on the hidden layers. The
    W @ x matmuls are the work; everything else is cheap integer glue.
    """

    def __init__(self, weights: list[np.ndarray], shifts: list[int]):
        assert len(weights) == len(shifts)
        for w in weights:
            assert w.dtype == np.int8
        self.weights = weights
        self.shifts = shifts

    @classmethod
    def random(cls, dims: list[int], seed: int = 0, target_std: float = 32.0) -> "QuantMLP":
        """Random int8 weights with per-layer shifts calibrated so activations
        keep a healthy int8 range instead of collapsing to zero or saturating.
        This is the same idea as real int8-quantization calibration."""
        rng = np.random.default_rng(seed)
        weights = [rng.integers(-32, 33, size=(dims[i + 1], dims[i]), dtype=np.int8)
                   for i in range(len(dims) - 1)]
        xcal = rng.integers(-127, 128, size=(dims[0], 64), dtype=np.int8)
        shifts, h = [], xcal
        for i, w in enumerate(weights):
            acc = w.astype(np.int64) @ h.astype(np.int64)
            std = float(acc.std()) or 1.0
            shift = max(0, int(round(np.log2(std / target_std))))
            shifts.append(shift)
            h = requantize(acc, shift, relu=(i < len(weights) - 1))
        return cls(weights, shifts)

    def forward_local(self, x: np.ndarray) -> np.ndarray:
        """Reference forward pass, all on this machine. x: int8 (in_dim, batch)."""
        h = x.astype(np.int8)
        for i, (w, shift) in enumerate(zip(self.weights, self.shifts)):
            acc = w.astype(np.int64) @ h.astype(np.int64)
            h = requantize(acc, shift, relu=(i < len(self.weights) - 1))
        return h

    def forward_green(self, client: MatmulClient, x: np.ndarray, price: int,
                      timeout: float = 120, log=lambda m: None) -> np.ndarray:
        """Forward pass whose matmuls are computed by the Green network."""
        h = x.astype(np.int8)
        for i, (w, shift) in enumerate(zip(self.weights, self.shifts)):
            log(f"layer {i + 1}/{len(self.weights)}: {w.shape} @ {h.shape} -> sent to the network")
            acc = client.matmul(w, h, price, timeout)
            h = requantize(acc, shift, relu=(i < len(self.weights) - 1))
            log(f"layer {i + 1} done, activation range [{int(h.min())}, {int(h.max())}]")
        return h


def predictions(logits: np.ndarray) -> np.ndarray:
    """Argmax over classes (rows) for each sample (column)."""
    return logits.argmax(axis=0)
