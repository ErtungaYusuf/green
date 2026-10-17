"""GPU proof-of-useful-work via WebGPU (wgpu / Vulkan).

The whole work unit is exact integer arithmetic: inputs are int8, the
seed-derived noise is rank-32 ternary, so a1 = A + U@V and b1 = B + U'@V' are
exact integers in about [-160, 160], and every 32x32 tile product fits in
int32. An i32 GPU kernel therefore reproduces the CPU result bit for bit on any
device, which is what consensus needs.

Throughput comes from doing BOTH the matmul and the transcript digest on the
GPU, so the 2 MB product stream never crosses PCIe -- only the 32-byte digest
is read back per nonce. Each thread computes one tile-product element, mixes it
(lowbias32, see green.work.transcript_digest) and XORs it into a
workgroup-local 8-lane accumulator; one thread folds that into the global
digest with atomics. XOR is commutative, so the parallel result equals the CPU
pass exactly.

The useful result A@B does not need the product stream: summing the tiles over
the shared index is just a1 @ b1, which the CPU gets from BLAS in a fraction of
a millisecond, and only when a unit is actually being delivered.
"""

import numpy as np

from . import params, work
from .crypto import h

N = params.UNIT
BLK = params.BLOCK
T = N // BLK
LANES = work.DIGEST_LANES

_SHADER = f"""
const N  : u32 = {N}u;
const BLK: u32 = {BLK}u;
const T  : u32 = {T}u;
const GOLDEN : u32 = 0x9E3779B9u;
const M1 : u32 = 0x7FEB352Du;
const M2 : u32 = 0x846CA68Bu;

@group(0) @binding(0) var<storage, read>       a1 : array<i32>;
@group(0) @binding(1) var<storage, read>       b1 : array<i32>;
@group(0) @binding(2) var<storage, read_write> digest : array<atomic<u32>>;

var<workgroup> ta : array<i32, {BLK * BLK}u>;    // the tile A'(i, l)
var<workgroup> tb : array<i32, {BLK * BLK}u>;    // the tile B'(l, j)
var<workgroup> red : array<u32, {BLK * BLK}u>;   // mixed products, folded into the lanes

fn mix(value : u32, idx : u32) -> u32 {{
    var h : u32 = value ^ (idx * GOLDEN);   // lowbias32 of the value and its position
    h ^= h >> 16u;
    h *= M1;
    h ^= h >> 15u;
    h *= M2;
    h ^= h >> 16u;
    return h;
}}

// One workgroup per tile product (i, l, j) of one nonce of the batch (wg.y). The two
// tiles are loaded into workgroup memory with coalesced reads (neighbouring threads,
// neighbouring addresses); thread (x, y) then computes product element (row y,
// column x). Its stream position's lane is x mod 8 = lidx mod 8, so a tree fold over
// the other bits of lidx leaves each lane's XOR in red[lane], and the workgroup adds
// 8 atomics to the batch's digest instead of 1024.
@compute @workgroup_size(BLK, BLK, 1)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) li : vec3<u32>,
        @builtin(local_invocation_index) lidx : u32) {{
    let ilj = wg.x;
    let n = wg.y;                           // which nonce of the batch
    let j = ilj % T;
    let l = (ilj / T) % T;
    let i = ilj / (T * T);
    let off = n * N * N;
    ta[li.y * BLK + li.x] = a1[off + (i * BLK + li.y) * N + l * BLK + li.x];
    tb[li.y * BLK + li.x] = b1[off + (l * BLK + li.y) * N + j * BLK + li.x];
    workgroupBarrier();

    let r = li.y;
    let c = li.x;
    var acc : i32 = 0;
    for (var k : u32 = 0u; k < BLK; k = k + 1u) {{
        acc = acc + ta[r * BLK + k] * tb[k * BLK + c];
    }}
    let base = ((((i * T) + l) * T + j) * BLK + r) * BLK + c;
    red[lidx] = mix(bitcast<u32>(acc), base);
    workgroupBarrier();
    for (var s : u32 = {BLK * BLK // 2}u; s >= {LANES}u; s = s / 2u) {{
        if (lidx < s) {{ red[lidx] = red[lidx] ^ red[lidx + s]; }}
        workgroupBarrier();
    }}
    if (lidx < {LANES}u) {{ atomicXor(&digest[n * {LANES}u + lidx], red[lidx]); }}
}}
"""


BATCH = 16                         # nonces per GPU submission


class GPUMiner:
    """Reusable GPU context, one per process (not thread-safe).

    Nonces go to the GPU in batches, and there are two sets of buffers: the miner
    submits one batch, then collects the previous one, so the CPU prepares the next
    batch's inputs (SHAKE-256 filler and noise) while the GPU digests the last. One
    nonce at a time, each waited for, left the GPU idle most of the time (measured in
    a mining node: 358 units/s against ~1,300 when fed back to back)."""

    def __init__(self, batch: int = BATCH):
        import wgpu

        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        self.name = self.adapter.info["device"]
        self._wgpu = wgpu
        self.batch = batch
        shader = self.device.create_shader_module(code=_SHADER)
        self.pipeline = self.device.create_compute_pipeline(
            layout=wgpu.enums.AutoLayoutMode.auto, compute={"module": shader, "entry_point": "main"})
        self._slots = [self._slot() for _ in range(2)]
        self._next = 0
        self._stage_a = np.empty((batch, N, N), np.int32)
        self._stage_b = np.empty((batch, N, N), np.int32)
        self._zeros = np.zeros(batch * LANES, dtype=np.uint32)

    def _slot(self) -> dict:
        usage = self._wgpu.BufferUsage
        size = self.batch * N * N * 4
        slot = {"a": self.device.create_buffer(size=size, usage=usage.STORAGE | usage.COPY_DST),
                "b": self.device.create_buffer(size=size, usage=usage.STORAGE | usage.COPY_DST),
                "d": self.device.create_buffer(size=self.batch * LANES * 4,
                                               usage=usage.STORAGE | usage.COPY_DST | usage.COPY_SRC),
                "r": self.device.create_buffer(size=self.batch * LANES * 4, usage=usage.COPY_DST | usage.MAP_READ),
                "seeds": None}
        slot["bind"] = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": slot["a"], "offset": 0, "size": size}},
                     {"binding": 1, "resource": {"buffer": slot["b"], "offset": 0, "size": size}},
                     {"binding": 2, "resource": {"buffer": slot["d"], "offset": 0, "size": self.batch * LANES * 4}}])
        return slot

    @staticmethod
    def factors(a: np.ndarray, b: np.ndarray, seed: bytes) -> tuple[np.ndarray, np.ndarray]:
        """A' and B' as int32, ready for the GPU (the noise products are exact in
        float32: |entries| <= RANK). Thread-safe: miners prepare tries in threads."""
        u, v, uf, vf = work.noise_factors(seed)
        a1, b1 = np.empty((N, N), np.int32), np.empty((N, N), np.int32)
        a1[...] = a + u @ v
        b1[...] = b + uf @ vf
        return a1, b1

    def submit(self, items: list) -> int:
        """Start the transcripts of up to `batch` tries, each (A, B, seed), on the GPU;
        returns a handle for collect(). At most two batches can be outstanding."""
        return self.submit_prepared([(*self.factors(a, b, seed), seed) for a, b, seed in items])

    def submit_prepared(self, items: list) -> int:
        """submit() for tries whose (A', B', seed) the caller prepared with factors()."""
        if not 0 < len(items) <= self.batch:
            raise ValueError(f"1..{self.batch} tries per batch")
        handle = self._next
        slot = self._slots[handle]
        if slot["seeds"] is not None:
            raise RuntimeError("collect this slot's previous batch first")
        self._next ^= 1
        n = len(items)
        for i, (a1, b1, _) in enumerate(items):
            self._stage_a[i] = a1
            self._stage_b[i] = b1
        q = self.device.queue
        q.write_buffer(slot["a"], 0, self._stage_a[:n])
        q.write_buffer(slot["b"], 0, self._stage_b[:n])
        q.write_buffer(slot["d"], 0, self._zeros[:n * LANES])
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.pipeline)
        cp.set_bind_group(0, slot["bind"])
        cp.dispatch_workgroups(T * T * T, n, 1)
        cp.end()
        enc.copy_buffer_to_buffer(slot["d"], 0, slot["r"], 0, n * LANES * 4)
        q.submit([enc.finish()])
        slot["seeds"] = [seed for _, _, seed in items]
        return handle

    def collect(self, handle: int) -> list[tuple[str, int]]:
        """(transcript, ticket value) of every try of a submitted batch, in order."""
        slot = self._slots[handle]
        seeds, slot["seeds"] = slot["seeds"], None
        if seeds is None:
            raise RuntimeError("nothing submitted in this slot")
        size = len(seeds) * LANES * 4
        slot["r"].map_sync(self._wgpu.MapMode.READ, 0, size)
        data = bytes(slot["r"].read_mapped(0, size))
        slot["r"].unmap()
        out = []
        for i, seed in enumerate(seeds):
            transcript = data[i * LANES * 4:(i + 1) * LANES * 4].hex()
            out.append((transcript, work.ticket_value(seed, transcript)))
        return out

    def tickets(self, items: list) -> list[tuple[str, int]]:
        return self.collect(self.submit(items))

    def discard(self) -> None:
        """Forget any outstanding batch (after an error, before starting over)."""
        for slot in self._slots:
            if slot["seeds"] is not None:
                slot["seeds"] = None
                try:
                    slot["r"].map_sync(self._wgpu.MapMode.READ)
                    slot["r"].unmap()
                except Exception:
                    pass

    def ticket(self, a: np.ndarray, b: np.ndarray, seed: bytes) -> tuple[str, int]:
        """Transcript and ticket value for one nonce. The hot path: no A@B."""
        return self.tickets([(a, b, seed)])[0]

    def unit(self, a: np.ndarray, b: np.ndarray, seed: bytes):
        """Full unit including the useful result. Returns (transcript, ticket, A@B)."""
        transcript, ticket = self.ticket(a, b, seed)
        return transcript, ticket, work.recover_useful(a, b, seed)


_MINER = None


def get_miner():
    """Shared GPU miner, or None if no GPU / wgpu is available."""
    global _MINER
    if _MINER is None:
        try:
            _MINER = GPUMiner()
        except Exception:
            _MINER = False
    return _MINER or None
