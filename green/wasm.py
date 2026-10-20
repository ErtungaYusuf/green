"""Deterministic, sandboxed WebAssembly execution for the general compute market.

Matrix units are verified cheaply with Freivalds, but an arbitrary program is
not checkable without running it. The way to make an arbitrary result trustable
is to make execution *deterministic*: identical module + input + fuel give
identical output and identical fuel-used on every machine. Then any node can
verify a claimed result by re-running it, and a dispute has a single right
answer the chain can adjudicate.

Determinism and safety come from the engine configuration:
  * fuel metering bounds CPU and turns infinite loops into a trap, so a result
    always comes with an exact, reproducible fuel cost;
  * NaN canonicalization removes the one source of float nondeterminism in core
    WebAssembly;
  * no WASI and no host imports, so the guest has no clock, no randomness, no
    I/O, no threads -- nothing nondeterministic and no way out of the sandbox;
  * a hard memory cap stops memory bombs;
  * a stack limit counted in the module itself (green/wasm_stack.py), far below the
    native stack, so how deep a program may recurse is the same on every machine;
  * every WebAssembly proposal not listed below is off, so the instruction set is
    exactly the one wasm_stack.py decodes.

ABI: the host writes the input at IN_PTR, then calls
    green_run(in_ptr: i32, in_len: i32) -> i64
which returns (out_ptr << 32) | out_len; the host reads the output from guest
memory. No multi-value or reference-type features are needed.
"""

import atexit
import hashlib
import threading

from . import params, wasm_stack

IN_PTR = 0x10000                 # input is written at 64 KiB, leaving page 0 for the guest
DEFAULT_MAX_MEMORY = 256 << 20   # 256 MiB
DEFAULT_FUEL = params.MAX_TASK_FUEL
ENTRY = "green_run"
# Native stack for WASM code, and the thread that runs it. wasm_stack.STACK_LIMIT
# (65,536 slots) needs at most 1 MiB even if every slot were a spilled v128 (16 bytes);
# the worst measured on x86-64 is 3.6 bytes a slot (tests/test_wasm.py), so this
# native limit is never what stops a program.
MAX_WASM_STACK = 2 << 20          # the most wasmtime allows (its async stack size)
THREAD_STACK = MAX_WASM_STACK + (16 << 20)
WASM_MAGIC = b"\x00asm\x01\x00\x00\x00"
_SECTION_IMPORT, _SECTION_EXPORT = 2, 7
_EXPORT_FUNC, _EXPORT_MEMORY = 0, 2


class ComputeError(Exception):
    """The program trapped, ran out of fuel/memory, or broke the ABI."""


class EngineMismatch(RuntimeError):
    """The installed wasmtime is not the one the network's consensus is pinned to.

    Whether a program finishes inside its fuel budget depends on wasmtime's fuel
    accounting, which can change between releases. A node on another version
    could reach a different verdict on a challenge and silently fork, so it
    refuses to run instead.
    """


def installed_version() -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version("wasmtime")
    except Exception:
        return None


def check_engine() -> None:
    have = installed_version()
    if have != params.WASMTIME_VERSION:
        raise EngineMismatch(
            f"Green consensus requires wasmtime {params.WASMTIME_VERSION}, installed: {have or 'none'}. "
            f"To install it: pip install wasmtime=={params.WASMTIME_VERSION}")


def program_id(module_bytes: bytes) -> str:
    return hashlib.blake2b(module_bytes, digest_size=32).hexdigest()


def result_hash(output: bytes) -> str:
    return hashlib.blake2b(output, digest_size=32).hexdigest()


def _engine(max_wasm_stack: int = MAX_WASM_STACK):
    import wasmtime

    cfg = wasmtime.Config()
    cfg.consume_fuel = True
    cfg.cranelift_nan_canonicalization = True     # the only float nondeterminism in core wasm
    cfg.wasm_threads = False                       # shared memory is nondeterministic
    cfg.wasm_bulk_memory = True
    cfg.wasm_multi_value = True
    cfg.wasm_simd = True                           # fixed-width SIMD is deterministic
    cfg.wasm_relaxed_simd = False                  # relaxed SIMD is NOT deterministic
    # Everything else off: the instruction set must be the one wasm_stack.py decodes.
    # (Reference types stay on: wasmtime's component-model-async needs them.)
    for name in ("wasm_tail_call", "wasm_multi_memory", "wasm_memory64", "wasm_gc",
                 "wasm_function_references", "wasm_exceptions", "wasm_wide_arithmetic", "wasm_custom_page_sizes",
                 "wasm_stack_switching", "wasm_component_model", "gc_support"):
        setattr(cfg, name, False)
    cfg.max_wasm_stack = max_wasm_stack
    return wasmtime.Engine(cfg)


_ENGINE = None


def engine():
    global _ENGINE
    if _ENGINE is None:
        check_engine()
        _ENGINE = _engine()
        atexit.register(_close_engine)
    return _ENGINE


def _close_engine() -> None:
    # Release the engine while the wasmtime module still exists; otherwise its
    # finalizer runs during interpreter teardown and prints a spurious traceback.
    global _ENGINE
    if _ENGINE is not None:
        try:
            _ENGINE.close()
        except Exception:
            pass
        _ENGINE = None


def _leb_u32(b: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = b[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, i
        shift += 7
        if shift > 28:
            raise ComputeError("bad LEB128 in module")


def _sections(b: bytes):
    i = len(WASM_MAGIC)
    while i < len(b):
        sid = b[i]
        size, i = _leb_u32(b, i + 1)
        yield sid, b[i:i + size]
        i += size


def validate_module(module_bytes: bytes) -> None:
    """Reject anything that is not a well-formed module for our sandbox.

    Runs in consensus (every task_post), so it must stay cheap: wasmtime's
    validator parses without compiling (~1 ns/byte, vs ~110 ns/byte to
    compile), then the import and export sections are read directly.
    """
    import wasmtime

    if len(module_bytes) > params.MAX_TASK_PROGRAM_BYTES:
        raise ComputeError(f"program larger than {params.MAX_TASK_PROGRAM_BYTES} bytes")
    if not module_bytes.startswith(WASM_MAGIC):
        raise ComputeError("not a WebAssembly module")
    try:
        wasmtime.Module.validate(engine(), module_bytes)
    except Exception as e:
        raise ComputeError(f"invalid module: {str(e).splitlines()[0]}") from None
    exports: dict[str, int] = {}
    try:
        for sid, body in _sections(module_bytes):
            if sid == _SECTION_IMPORT:
                count, _ = _leb_u32(body, 0)
                if count:
                    # No imports = no host functions = no escape hatch, no clock, no randomness.
                    raise ComputeError("module must not import anything")
            elif sid == _SECTION_EXPORT:
                count, j = _leb_u32(body, 0)
                for _ in range(count):
                    n, j = _leb_u32(body, j)
                    name = body[j:j + n].decode("utf-8")
                    kind = body[j + n]
                    _, j = _leb_u32(body, j + n + 1)
                    exports[name] = kind
    except (IndexError, UnicodeDecodeError):
        raise ComputeError("malformed module sections") from None
    if exports.get(ENTRY) != _EXPORT_FUNC or exports.get("memory") != _EXPORT_MEMORY:
        raise ComputeError(f"module must export function '{ENTRY}' and memory 'memory'")
    _instrumented(module_bytes)                    # refused now, not when a challenge re-runs it


_INSTRUMENTED: dict[str, bytes] = {}
_INSTRUMENTED_LOCK = threading.Lock()


def _instrumented(module_bytes: bytes) -> bytes:
    """The module with its deterministic stack limit (cached by its hash)."""
    key = program_id(module_bytes)
    with _INSTRUMENTED_LOCK:
        done = _INSTRUMENTED.get(key)
    if done is not None:
        return done
    try:
        done = wasm_stack.instrument(module_bytes)
    except (wasm_stack.Uninstrumentable, IndexError) as e:
        raise ComputeError(f"module cannot be given a stack limit: {e}") from None
    with _INSTRUMENTED_LOCK:
        if len(_INSTRUMENTED) > 64:
            _INSTRUMENTED.clear()
        _INSTRUMENTED[key] = done
    return done


_THREAD_LOCK = threading.Lock()


def _on_big_stack(fn):
    """Run fn() in a thread whose native stack holds MAX_WASM_STACK with room to spare
    (a worker thread's own stack can be as small as 512 KiB, on macOS)."""
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:                     # handed back to the caller
            box["error"] = e
    with _THREAD_LOCK:
        old = threading.stack_size()
        threading.stack_size(THREAD_STACK)
        try:
            t = threading.Thread(target=target, name="wasm", daemon=True)
            t.start()
        finally:
            threading.stack_size(old)
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def run(module_bytes: bytes, input_bytes: bytes, fuel: int = DEFAULT_FUEL,
        max_memory: int = DEFAULT_MAX_MEMORY) -> tuple[bytes, int]:
    """Execute deterministically. Returns (output, fuel_used) or raises ComputeError.

    The same arguments always give the same return value on any machine, which
    is what makes a result verifiable by re-execution.
    """
    if not module_bytes.startswith(WASM_MAGIC):
        raise ComputeError("not a WebAssembly module")
    try:
        import wasmtime
        wasmtime.Module.validate(engine(), module_bytes)
    except Exception as e:
        raise ComputeError(f"invalid module: {str(e).splitlines()[0]}") from None
    instrumented = _instrumented(module_bytes)
    return _on_big_stack(lambda: _run(instrumented, input_bytes, fuel, max_memory))


def _run(module_bytes: bytes, input_bytes: bytes, fuel: int, max_memory: int) -> tuple[bytes, int]:
    import wasmtime

    try:
        module = wasmtime.Module(engine(), module_bytes)
    except Exception as e:
        raise ComputeError(f"invalid module: {e}") from None

    store = wasmtime.Store(engine())
    store.set_limits(memory_size=max_memory)
    store.set_fuel(fuel)
    try:
        instance = wasmtime.Instance(store, module, [])     # [] imports: nothing is provided
    except Exception as e:
        raise ComputeError(f"instantiation failed: {e}") from None

    exports = instance.exports(store)
    memory = exports.get("memory")
    entry = exports.get(ENTRY)
    if memory is None or entry is None:
        raise ComputeError(f"module must export '{ENTRY}' and 'memory'")

    if input_bytes:
        needed = IN_PTR + len(input_bytes)
        if memory.data_len(store) < needed:
            grow_pages = -(-(needed - memory.data_len(store)) // 65536)
            try:
                memory.grow(store, grow_pages)
            except Exception as e:
                raise ComputeError(f"cannot grow memory for input: {e}") from None
        memory.write(store, input_bytes, IN_PTR)

    try:
        packed = entry(store, IN_PTR, len(input_bytes))
    except wasmtime.Trap as e:
        if store.get_fuel() == 0:
            raise ComputeError("out of fuel (program ran too long)") from None
        raise ComputeError(f"trap: {str(e).splitlines()[0]}") from None
    except Exception as e:
        raise ComputeError(f"execution failed: {e}") from None

    fuel_used = fuel - store.get_fuel()
    if not isinstance(packed, int):
        raise ComputeError(f"{ENTRY} must return i64")
    packed &= (1 << 64) - 1
    out_ptr, out_len = packed >> 32, packed & 0xFFFFFFFF
    if out_ptr + out_len > memory.data_len(store):
        raise ComputeError("output region is outside memory")
    output = bytes(memory.read(store, out_ptr, out_ptr + out_len))
    return output, fuel_used


def verify(module_bytes: bytes, input_bytes: bytes, claimed_output_hash: str,
           fuel: int, max_memory: int = DEFAULT_MAX_MEMORY) -> bool:
    """Re-run and check the claimed output hash. The adjudication primitive."""
    try:
        output, _ = run(module_bytes, input_bytes, fuel, max_memory)
    except ComputeError:
        return False
    return result_hash(output) == claimed_output_hash
