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
@group(0) @binding(2) var<storage, read_write> digest : array<atomic<u32>, {LANES}u>;

var<workgroup> wacc : array<atomic<u32>, {LANES}u>;

fn mix(value : u32, idx : u32) -> u32 {{
    var h : u32 = value ^ (idx * GOLDEN);   // lowbias32 of the value and its position
    h ^= h >> 16u;
    h *= M1;
    h ^= h >> 15u;
    h *= M2;
    h ^= h >> 16u;
    return h;
}}

@compute @workgroup_size(BLK, BLK, 1)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) li : vec3<u32>,
        @builtin(local_invocation_index) lidx : u32) {{
    if (lidx < {LANES}u) {{ atomicStore(&wacc[lidx], 0u); }}
    workgroupBarrier();

    let ilj = wg.x;
    let j = ilj % T;
    let l = (ilj / T) % T;
    let i = ilj / (T * T);
    let r = li.x;
    let c = li.y;
    let a_row = (i * BLK + r) * N + l * BLK;
    let b_col = j * BLK + c;
    var acc : i32 = 0;
    for (var k : u32 = 0u; k < BLK; k = k + 1u) {{
        acc = acc + a1[a_row + k] * b1[(l * BLK + k) * N + b_col];
    }}
    let base = ((((i * T) + l) * T + j) * BLK + r) * BLK + c;
    atomicXor(&wacc[base & ({LANES}u - 1u)], mix(bitcast<u32>(acc), base));
    workgroupBarrier();

    if (lidx < {LANES}u) {{ atomicXor(&digest[lidx], atomicLoad(&wacc[lidx])); }}
}}
"""


class GPUMiner:
    """Reusable GPU context. One instance per process; call unit() per nonce."""

    def __init__(self):
        import wgpu

        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        self.name = self.adapter.info["device"]
        self._wgpu = wgpu

        shader = self.device.create_shader_module(code=_SHADER)
        usage = wgpu.BufferUsage
        self.buf_a = self.device.create_buffer(size=N * N * 4, usage=usage.STORAGE | usage.COPY_DST)
        self.buf_b = self.device.create_buffer(size=N * N * 4, usage=usage.STORAGE | usage.COPY_DST)
        self.buf_digest = self.device.create_buffer(
            size=LANES * 4, usage=usage.STORAGE | usage.COPY_DST | usage.COPY_SRC)
        self.buf_read = self.device.create_buffer(size=LANES * 4, usage=usage.COPY_DST | usage.MAP_READ)
        self._zeros = np.zeros(LANES, dtype=np.uint32).tobytes()

        self.pipeline = self.device.create_compute_pipeline(
            layout=wgpu.enums.AutoLayoutMode.auto, compute={"module": shader, "entry_point": "main"})
        self.bind = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": self.buf_a, "offset": 0, "size": N * N * 4}},
                     {"binding": 1, "resource": {"buffer": self.buf_b, "offset": 0, "size": N * N * 4}},
                     {"binding": 2, "resource": {"buffer": self.buf_digest, "offset": 0, "size": LANES * 4}}])

    def _digest(self, a1: np.ndarray, b1: np.ndarray) -> str:
        q = self.device.queue
        q.write_buffer(self.buf_a, 0, np.ascontiguousarray(a1, dtype=np.int32))
        q.write_buffer(self.buf_b, 0, np.ascontiguousarray(b1, dtype=np.int32))
        q.write_buffer(self.buf_digest, 0, self._zeros)
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.pipeline)
        cp.set_bind_group(0, self.bind)
        cp.dispatch_workgroups(T * T * T, 1, 1)
        cp.end()
        enc.copy_buffer_to_buffer(self.buf_digest, 0, self.buf_read, 0, LANES * 4)
        q.submit([enc.finish()])
        self.buf_read.map_sync(self._wgpu.MapMode.READ)
        data = bytes(self.buf_read.read_mapped())
        self.buf_read.unmap()
        return data.hex()

    @staticmethod
    def _factors(a, b, seed):
        """A' and B' as int32 (the noise products are exact in float32: |entries| <= RANK)."""
        u, v, uf, vf = work.noise_factors(seed)
        return (a + u @ v).astype(np.int32), (b + uf @ vf).astype(np.int32)

    def ticket(self, a: np.ndarray, b: np.ndarray, seed: bytes) -> tuple[str, int]:
        """Transcript and ticket value for one nonce. The hot path: no A@B."""
        a1, b1 = self._factors(a, b, seed)
        transcript = self._digest(a1, b1)
        return transcript, work.ticket_value(seed, transcript)

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
