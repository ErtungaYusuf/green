"""Tests for the deterministic WebAssembly sandbox (green/wasm.py)."""

import pytest

wasmtime = pytest.importorskip("wasmtime")
from green import wasm  # noqa: E402

# Sum the input bytes into an i32, store at 0, return len 4.
SUM = wasmtime.wat2wasm(r"""(module
  (memory (export "memory") 2)
  (func (export "green_run") (param $p i32) (param $n i32) (result i64)
    (local $i i32) (local $acc i32)
    (block $done (loop $l
      (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
      (local.set $acc (i32.add (local.get $acc)
        (i32.load8_u (i32.add (local.get $p) (local.get $i)))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l)))
    (i32.store (i32.const 0) (local.get $acc))
    (i64.const 4)))""")

# Float work that would differ if NaN bits leaked: produce a NaN, store its bits.
NAN = wasmtime.wat2wasm(r"""(module
  (memory (export "memory") 1)
  (func (export "green_run") (param i32 i32) (result i64)
    (i32.store (i32.const 0)
      (i32.reinterpret_f32 (f32.div (f32.const 0) (f32.const 0))))
    (i64.const 4)))""")

LOOP = wasmtime.wat2wasm(r"""(module (memory (export "memory") 1)
  (func (export "green_run") (param i32 i32) (result i64) (loop $l (br $l)) (i64.const 0)))""")

GROW = wasmtime.wat2wasm(r"""(module (memory (export "memory") 1)
  (func (export "green_run") (param i32 i32) (result i64)
    (drop (memory.grow (i32.const 100000))) (i64.const 0)))""")

IMPORTS = wasmtime.wat2wasm(r"""(module (import "env" "f" (func)) (memory (export "memory") 1)
  (func (export "green_run") (param i32 i32) (result i64) (i64.const 0)))""")

NO_ENTRY = wasmtime.wat2wasm(r"""(module (memory (export "memory") 1)
  (func (export "other") (result i64) (i64.const 0)))""")

OOB = wasmtime.wat2wasm(r"""(module (memory (export "memory") 1)
  (func (export "green_run") (param i32 i32) (result i64)
    (i64.or (i64.shl (i64.const 1000000) (i64.const 32)) (i64.const 16))))""")


def test_sum_and_determinism():
    out, fuel = wasm.run(SUM, bytes([1, 2, 3, 4, 5]))
    assert int.from_bytes(out, "little") == 15
    out2, fuel2 = wasm.run(SUM, bytes([1, 2, 3, 4, 5]))
    assert out == out2 and fuel == fuel2            # bit-identical, same cost
    out3, _ = wasm.run(SUM, bytes([10, 20]))
    assert int.from_bytes(out3, "little") == 30


def test_verify_accepts_right_and_rejects_wrong():
    out, _ = wasm.run(SUM, b"hello")
    assert wasm.verify(SUM, b"hello", wasm.result_hash(out), wasm.DEFAULT_FUEL)
    assert not wasm.verify(SUM, b"hello", "00" * 32, wasm.DEFAULT_FUEL)
    assert not wasm.verify(SUM, b"world", wasm.result_hash(out), wasm.DEFAULT_FUEL)


def test_nan_is_canonical():
    out, _ = wasm.run(NAN, b"")
    assert out.hex() == "0000c07f"                  # canonical quiet NaN, never a device-specific pattern


def test_infinite_loop_runs_out_of_fuel():
    with pytest.raises(wasm.ComputeError, match="fuel"):
        wasm.run(LOOP, b"", fuel=1_000_000)


def test_memory_bomb_is_capped():
    # 100000 pages ~ 6.5 GiB requested; the cap makes grow fail (returns -1), program returns 0.
    out, _ = wasm.run(GROW, b"", max_memory=16 << 20)
    assert out == b""


def test_sandbox_rejects_imports_and_missing_entry():
    with pytest.raises(wasm.ComputeError, match="import"):
        wasm.validate_module(IMPORTS)
    with pytest.raises(wasm.ComputeError, match="green_run"):
        wasm.validate_module(NO_ENTRY)
    with pytest.raises(wasm.ComputeError, match="not a WebAssembly module"):
        wasm.validate_module(b"not wasm")


def test_output_outside_memory_is_rejected():
    with pytest.raises(wasm.ComputeError, match="outside memory"):
        wasm.run(OOB, b"")


def test_tight_fuel_budget_traps():
    _, fuel = wasm.run(SUM, bytes([1] * 200))
    assert fuel > 10
    with pytest.raises(wasm.ComputeError, match="fuel"):
        wasm.run(SUM, bytes([1] * 200), fuel=5)
