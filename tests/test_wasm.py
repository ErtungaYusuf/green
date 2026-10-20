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


# --- a stack limit that is the same on every machine (green/wasm_stack.py) ----------

import struct  # noqa: E402

from green import wasm_stack  # noqa: E402


def recursive(locals_="", value="(i32.add (i32.const 1) {rec})"):
    rec = "(call $rec (i32.sub (local.get $n) (i32.const 1)))"
    return wasmtime.wat2wasm(f"""(module (memory (export "memory") 2)
      (func $rec (param $n i32) (result i32) {locals_}
        (if (result i32) (i32.eqz (local.get $n)) (then (i32.const 0)) (else {value.format(rec=rec)})))
      (func (export "green_run") (param i32 i32) (result i64)
        (i32.store (i32.const 0) (call $rec (i32.load (local.get 0)))) (i64.const 4)))""")


def finishes(prog, depth) -> bool:
    try:
        wasm.run(prog, struct.pack("<i", depth))
        return True
    except wasm.ComputeError:
        return False


def deepest(prog) -> int:
    lo, hi = 0, 4_000_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if finishes(prog, mid) else (lo, mid - 1)
    return lo


@pytest.fixture
def machine(monkeypatch):
    """Run on an engine with a different native stack: a machine whose frames are
    bigger runs out of native stack sooner, exactly like a smaller stack."""
    def use(max_wasm_stack):
        monkeypatch.setattr(wasm, "_ENGINE", wasm._engine(max_wasm_stack))
    yield use


def test_without_the_limit_the_verdict_depends_on_the_machine(machine, monkeypatch):
    """The bug the limit fixes: with the counter off, how deep a program may recurse is
    whatever the native stack allows, a fraction of the fuel budget, so the same task
    finishes on one machine and traps on another."""
    monkeypatch.setattr(wasm_stack, "STACK_LIMIT", 2**31 - 1)
    monkeypatch.setattr(wasm, "_INSTRUMENTED", {})
    prog = recursive()
    machine(wasm.MAX_WASM_STACK)
    here = deepest(prog)
    machine(wasm.MAX_WASM_STACK // 2)                   # frames twice as big
    there = deepest(prog)
    assert there < here and finishes(prog, here) is False
    machine(wasm.MAX_WASM_STACK)
    assert wasm.run(prog, struct.pack("<i", here))[1] < wasm.DEFAULT_FUEL // 100      # fuel had no say


def test_recursion_stops_at_the_same_depth_on_every_machine(machine):
    prog = recursive()
    cost = wasm_stack.costs(prog)[0]
    limit = wasm_stack.STACK_LIMIT // cost - 1            # depth d is d + 1 calls, each adding its cost
    for stack in (wasm.MAX_WASM_STACK, wasm.MAX_WASM_STACK // 2, wasm.MAX_WASM_STACK // 4):
        machine(stack)
        assert finishes(prog, limit) and not finishes(prog, limit + 1)
        assert wasm.run(prog, struct.pack("<i", limit))[0] == struct.pack("<i", limit)


# Frames Cranelift has to make big: values live across the recursive call.
SPILLS = {
    "i64 locals": (" ".join(f"(local $l{i} i64)" for i in range(24)),
                   " ".join(f"(local.set $l{i} (i64.extend_i32_u (i32.add (local.get $n) (i32.const {i}))))"
                            for i in range(24))
                   + "(i32.add {rec} (i32.wrap_i64 " + "".join(f"(i64.add (local.get $l{i}) " for i in range(23))
                   + "(local.get $l23)" + ")" * 23 + "))"),
    "v128 locals": (" ".join(f"(local $v{i} v128)" for i in range(32)),
                    " ".join(f"(local.set $v{i} (i32x4.splat (i32.add (local.get $n) (i32.const {i}))))"
                             for i in range(32))
                    + "(i32.add {rec} (i32x4.extract_lane 0 " + "".join(f"(i32x4.add (local.get $v{i}) " for i in range(31))
                    + "(local.get $v31)" + ")" * 31 + "))"),
    "v128 operand stack": ("", "(i32x4.extract_lane 0 " + "".join(
        f"(i32x4.add (i32x4.splat (i32.add (local.get $n) (i32.const {i}))) " for i in range(32))
        + "(i32x4.splat {rec})" + ")" * 32 + ")"),
}


@pytest.mark.parametrize("name", list(SPILLS))
def test_the_native_stack_holds_the_limit_with_room_to_spare(name, monkeypatch):
    """Measure how much native stack a slot of cost really takes, with the counter
    off, for frames full of spilled values; the limit must need under half the native
    stack even at four times the worst measured (it is 3.6 bytes a slot on x86-64,
    against 16 assumed)."""
    locals_, value = SPILLS[name]
    prog = recursive(locals_, value)
    cost, limit = wasm_stack.costs(prog)[0], wasm_stack.STACK_LIMIT
    monkeypatch.setattr(wasm_stack, "STACK_LIMIT", 2**31 - 1)
    monkeypatch.setattr(wasm, "_INSTRUMENTED", {})
    per_slot = wasm.MAX_WASM_STACK / (deepest(prog) * cost)
    assert 4 * per_slot * limit <= wasm.MAX_WASM_STACK / 2, per_slot


def test_call_indirect_is_counted_too():
    prog = wasmtime.wat2wasm(r"""(module (memory (export "memory") 1)
      (type $t (func (param i32) (result i32)))
      (table 1 funcref) (elem (i32.const 0) $rec)
      (func $rec (type $t) (param $n i32) (result i32)
        (if (result i32) (i32.eqz (local.get $n)) (then (i32.const 0))
          (else (i32.add (i32.const 1)
                  (call_indirect (type $t) (i32.sub (local.get $n) (i32.const 1)) (i32.const 0))))))
      (func (export "green_run") (param i32 i32) (result i64)
        (i32.store (i32.const 0) (call $rec (i32.load (local.get 0)))) (i64.const 4)))""")
    worst = max(wasm_stack.costs(prog))
    first = wasm_stack.costs(prog)[0]
    limit = (wasm_stack.STACK_LIMIT - first) // worst + 1
    assert finishes(prog, limit - 1) and not finishes(prog, limit)


def test_the_rewrite_keeps_what_programs_compute():
    """Blocks with results, br_table, multi-value, SIMD, bulk memory and a start
    function: the instrumented module computes what the original does."""
    prog = wasmtime.wat2wasm(r"""(module (memory (export "memory") 1)
      (global $g (mut i32) (i32.const 7))
      (func $two (param i32) (result i32 i32) (local.get 0) (i32.mul (local.get 0) (i32.const 2)))
      (func $pick (param i32) (result i32)
        (block $c (block $b (block $a (br_table $a $b $c (local.get 0)))
          (return (i32.const 10))) (return (i32.const 20))) (i32.const 30))
      (func $start (global.set $g (i32.add (global.get $g) (i32.const 1))))
      (start $start)
      (func (export "green_run") (param i32 i32) (result i64)
        (local v128)
        (memory.fill (i32.const 64) (i32.const 3) (i32.const 16))
        (local.set 2 (i32x4.add (v128.load (i32.const 64)) (i32x4.splat (global.get $g))))
        (i32.store (i32.const 0) (i32.add (call $two (i32.const 5))))
        (i32.store (i32.const 4) (i32.add (call $pick (i32.const 1)) (call $pick (i32.const 9))))
        (i32.store (i32.const 8) (i32x4.extract_lane 2 (local.get 2)))
        (i64.const 12)))""")
    out, _ = wasm.run(prog, b"")
    assert struct.unpack("<3i", out) == (15, 50, 0x03030303 + 8)
    assert wasm.run(SUM, bytes(range(10)))[0] == struct.pack("<i", 45)


def test_modules_get_their_limit_when_posted():
    wasm.validate_module(recursive())                  # instrumentable: fine
    with pytest.raises(wasm.ComputeError):
        wasm.validate_module(b"\x00asm\x01\x00\x00\x00\x0a\x03\x01\x00\x0b")    # code without functions


def test_the_decoder_agrees_with_the_engine_on_every_instruction():
    """For every instruction in wasm_stack's tables, a function made of exactly the
    operands the table says it pops (of some type) and its immediates is accepted by
    the engine and can be instrumented: the table's immediates and stack effects are
    the engine's. (Instructions that need locals, globals, tables or segments are
    covered by the programs above.)"""
    import itertools
    eng = wasm._engine()
    consts = [b"\x41\x00", b"\x42\x00", b"\x43" + bytes(4), b"\x44" + bytes(8), b"\xfd\x0c" + bytes(16)]
    imm = {wasm_stack.NONE: b"", wasm_stack.MEMARG: b"\x00\x00", wasm_stack.BYTE: b"\x00", wasm_stack.S32: b"\x00",
           wasm_stack.S64: b"\x00", wasm_stack.F32: bytes(4), wasm_stack.F64: bytes(8), "L": b"\x00", "V": bytes(16)}

    def module(body):
        code = b"\x00" + body + b"\x0b"
        sec = lambda i, b: bytes([i]) + wasm_stack._u32_bytes(len(b)) + b  # noqa: E731
        return (b"\x00asm\x01\x00\x00\x00" + sec(1, b"\x01\x60\x00\x00") + sec(3, b"\x01\x00")
                + sec(5, b"\x01\x00\x01") + sec(10, b"\x01" + wasm_stack._u32_bytes(len(code)) + code))
    cases = [(bytes([op]) + imm[k], pops, pushes) for op, (pops, pushes, k) in wasm_stack._SIMPLE.items()
             if op not in (0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0xD0, 0xD1, 0xD2)]
    cases += [(b"\xfd" + wasm_stack._u32_bytes(op) + b"".join(imm[k] for k in imms), pops, pushes)
              for op, (pops, pushes, imms) in wasm_stack._FD.items()]
    cases += [(b"\xfc" + bytes([op]), 1, 1) for op in range(8)]
    assert len(wasm_stack._FD) == 236                       # every fixed-width SIMD instruction
    for ins, pops, pushes in cases:
        for operands in itertools.product(consts, repeat=pops):
            m = module(b"".join(operands) + ins + b"\x1a" * pushes)
            try:
                wasmtime.Module.validate(eng, m)
            except Exception:
                continue
            wasmtime.Module.validate(eng, wasm_stack.instrument(m))
            break
        else:
            raise AssertionError(f"the engine accepts no function made of {ins.hex()}")
