"""A deterministic stack limit for WASM tasks.

Fuel makes every instruction cost the same on every machine, but how deep a program
may recurse is decided by the native stack: wasmtime traps when its stack
(`max_wasm_stack`) runs out, and how much each call takes depends on the machine
code Cranelift produced, which differs between x86-64 and AArch64. A task sitting at
that edge would finish on one node and trap on another, and a challenge of it would
split the network (SECURITY.md, W-stack).

So every module is rewritten before it runs, the way Substrate and NEAR do it: each
function gets a cost, its parameters + locals + the most values its body ever keeps
on the operand stack + a fixed frame overhead, and every call adds the callee's cost
to a counter (a new global) before the call and takes it off after. A call that
would take the counter past STACK_LIMIT traps with `unreachable`, the same way
everywhere. STACK_LIMIT is set far below what the native stack holds
(wasm.MAX_WASM_STACK), so the native stack never decides anything.

`call_indirect` is charged the most any function costs (the callee is not known).
The rewrite is a pure function of the module's bytes. It knows the instruction set
the engine accepts: core WebAssembly, sign extension, saturating conversions, bulk
memory, multi-value, reference types and fixed-width SIMD (wasm.py turns every
other proposal off), and a module it cannot decode is refused when the task is
posted, never at adjudication.
"""

STACK_LIMIT = 65_536            # values (slots); each call adds its callee's cost
FRAME_OVERHEAD = 16             # slots per call for the return address, saved registers, spills
PRELUDE_HEIGHT = 2              # the check before a call puts two values on the caller's stack


class Uninstrumentable(ValueError):
    pass


# --- reading ---------------------------------------------------------------------

def _u32(b: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if i >= len(b):
            raise Uninstrumentable("truncated integer")
        byte = b[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, i
        shift += 7
        if shift > 35:
            raise Uninstrumentable("bad LEB128")


def _skip_leb(b: bytes, i: int, max_bytes: int) -> int:
    for n in range(max_bytes):
        if i >= len(b):
            raise Uninstrumentable("truncated integer")
        i += 1
        if b[i - 1] < 0x80:
            return i
    raise Uninstrumentable("bad LEB128")


def _s33(b: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = b[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            if byte & 0x40:
                result -= 1 << shift
            return result, i
        if shift > 35:
            raise Uninstrumentable("bad block type")


def _u32_bytes(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _s32_bytes(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if (n == 0 and not byte & 0x40) or (n == -1 and byte & 0x40):
            out.append(byte)
            return bytes(out)
        out.append(byte | 0x80)


# --- the instruction set: (pops, pushes) and immediates ---------------------------

MEMARG, NONE, BYTE, IDX, S32, S64, F32, F64 = range(8)

_SIMPLE: dict[int, tuple[int, int, int]] = {0x01: (0, 0, NONE), 0x1A: (1, 0, NONE), 0x1B: (3, 1, NONE),
                                           0x20: (0, 1, IDX), 0x21: (1, 0, IDX), 0x22: (1, 1, IDX),
                                           0x23: (0, 1, IDX), 0x24: (1, 0, IDX),
                                           0x3F: (0, 1, BYTE), 0x40: (1, 1, BYTE),
                                           0x41: (0, 1, S32), 0x42: (0, 1, S64), 0x43: (0, 1, F32), 0x44: (0, 1, F64),
                                           # reference types: table.get/set, ref.null/is_null/func
                                           0x25: (1, 1, IDX), 0x26: (2, 0, IDX), 0xD0: (0, 1, BYTE),
                                           0xD1: (1, 1, NONE), 0xD2: (0, 1, IDX)}
for _op in range(0x28, 0x36):
    _SIMPLE[_op] = (1, 1, MEMARG)                       # loads
for _op in range(0x36, 0x3F):
    _SIMPLE[_op] = (2, 0, MEMARG)                       # stores
for _lo, _hi, _pops in ((0x45, 0x45, 1), (0x46, 0x4F, 2), (0x50, 0x50, 1), (0x51, 0x66, 2), (0x67, 0x69, 1),
                        (0x6A, 0x78, 2), (0x79, 0x7B, 1), (0x7C, 0x8A, 2), (0x8B, 0x91, 1), (0x92, 0x98, 2),
                        (0x99, 0x9F, 1), (0xA0, 0xA6, 2), (0xA7, 0xC4, 1)):
    for _op in range(_lo, _hi + 1):
        _SIMPLE[_op] = (_pops, 1, NONE)

# 0xFC prefix: saturating truncation, bulk memory and the table instructions
_FC = {**{op: (1, 1, ()) for op in range(8)},
       8: (3, 0, (IDX, BYTE)), 9: (0, 0, (IDX,)), 10: (3, 0, (BYTE, BYTE)), 11: (3, 0, (BYTE,)),
       12: (3, 0, (IDX, IDX)), 13: (0, 0, (IDX,)), 14: (3, 0, (IDX, IDX)),
       15: (2, 1, (IDX,)), 16: (0, 1, (IDX,)), 17: (3, 0, (IDX,))}

# 0xFD prefix: fixed-width SIMD. Immediates: M memarg, L lane index, V 16 bytes.
_FD: dict[int, tuple[int, int, tuple]] = {}
for _op in range(0x00, 0x0B):
    _FD[_op] = (1, 1, (MEMARG,))                        # loads, extending loads, splat loads
_FD[0x0B] = (2, 0, (MEMARG,))
_FD[0x0C] = (0, 1, ("V",))
_FD[0x0D] = (2, 1, ("V",))
_FD[0x0E] = (2, 1, ())
for _op in range(0x0F, 0x15):
    _FD[_op] = (1, 1, ())                               # splats
for _op in (0x15, 0x16, 0x18, 0x19, 0x1B, 0x1D, 0x1F, 0x21):
    _FD[_op] = (1, 1, ("L",))                           # extract_lane
for _op in (0x17, 0x1A, 0x1C, 0x1E, 0x20, 0x22):
    _FD[_op] = (2, 1, ("L",))                           # replace_lane
for _op in range(0x23, 0x4D):
    _FD[_op] = (2, 1, ())                               # comparisons
_FD[0x4D] = (1, 1, ())
for _op in range(0x4E, 0x52):
    _FD[_op] = (2, 1, ())
_FD[0x52] = (3, 1, ())                                  # bitselect
_FD[0x53] = (1, 1, ())
for _op in range(0x54, 0x58):
    _FD[_op] = (2, 1, (MEMARG, "L"))                    # load_lane
for _op in range(0x58, 0x5C):
    _FD[_op] = (2, 0, (MEMARG, "L"))                    # store_lane
_FD[0x5C] = _FD[0x5D] = (1, 1, (MEMARG,))               # load_zero
_UNARY = {0x5E, 0x5F, 0x60, 0x61, 0x62, 0x63, 0x64, 0x67, 0x68, 0x69, 0x6A, 0x74, 0x75, 0x7A, 0x7C, 0x7D, 0x7E,
          0x7F, 0x80, 0x81, 0x83, 0x84, 0x87, 0x88, 0x89, 0x8A, 0x94, 0xA0, 0xA1, 0xA3, 0xA4, 0xA7, 0xA8, 0xA9, 0xAA,
          0xC0, 0xC1, 0xC3, 0xC4, 0xC7, 0xC8, 0xC9, 0xCA, 0xE0, 0xE1, 0xE3, 0xEC, 0xED, 0xEF,
          *range(0xF8, 0x100)}
_BINARY = {0x65, 0x66, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x76, 0x77, 0x78, 0x79, 0x7B, 0x82,
           0x85, 0x86, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x90, 0x91, 0x92, 0x93, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9B,
           0x9C, 0x9D, 0x9E, 0x9F, 0xAB, 0xAC, 0xAD, 0xAE, 0xB1, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBC, 0xBD,
           0xBE, 0xBF, 0xCB, 0xCC, 0xCD, 0xCE, 0xD1, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE,
           0xDF, *range(0xE4, 0xEC), *range(0xF0, 0xF8)}
for _op in _UNARY:
    _FD[_op] = (1, 1, ())
for _op in _BINARY:
    _FD[_op] = (2, 1, ())

_VALTYPES = {0x7F, 0x7E, 0x7D, 0x7C, 0x7B, 0x70, 0x6F}


def _immediate(b: bytes, i: int, kind) -> int:
    if kind == NONE:
        return i
    if kind == BYTE:
        return i + 1
    if kind == IDX:
        return _skip_leb(b, i, 5)
    if kind == MEMARG:
        return _skip_leb(b, _skip_leb(b, i, 5), 10)
    if kind == S32:
        return _skip_leb(b, i, 5)
    if kind == S64:
        return _skip_leb(b, i, 10)
    if kind == F32:
        return i + 4
    if kind == F64:
        return i + 8
    if kind == "L":
        return i + 1
    if kind == "V":
        return i + 16
    raise AssertionError(kind)


# --- one function body -------------------------------------------------------------

def _block_type(b: bytes, i: int, types: list) -> tuple[int, int, int]:
    """(params, results, next offset)."""
    if b[i] == 0x40:
        return 0, 0, i + 1
    if b[i] in _VALTYPES:
        return 0, 1, i + 1
    idx, j = _s33(b, i)
    if not 0 <= idx < len(types):
        raise Uninstrumentable("bad block type")
    return types[idx][0], types[idx][1], j


def scan_body(b: bytes, start: int, end: int, types: list, func_types: list):
    """The instructions of an expression b[start:end]: returns (max operand stack
    height, [(offset, size, kind, callee)] for every call and call_indirect)."""
    i = start
    h = top = 0
    frames = [[0, 0, 0, False]]                         # [start height, params, results, unreachable]
    calls = []

    def pop(n):
        nonlocal h
        f = frames[-1]
        h = max(f[0], h - n) if f[3] else h - n
        if h < f[0] and not f[3]:
            raise Uninstrumentable("operand stack underflow")

    def push(n):
        nonlocal h, top
        h += n
        top = max(top, h)

    def unreachable():
        nonlocal h
        frames[-1][3] = True
        h = frames[-1][0]

    while i < end:
        at = i
        op = b[i]
        i += 1
        if op in _SIMPLE:
            pops, pushes, imm = _SIMPLE[op]
            i = _immediate(b, i, imm)
            pop(pops)
            push(pushes)
        elif op in (0x02, 0x03, 0x04):                  # block, loop, if
            if op == 0x04:
                pop(1)
            params, results, i = _block_type(b, i, types)
            pop(params)
            frames.append([h, params, results, False])
            push(params)
        elif op == 0x05:                                # else
            f = frames[-1]
            h = f[0] + f[1]
            f[3] = False
        elif op == 0x0B:                                # end
            f = frames.pop()
            h = f[0]
            if not frames:
                if i != end:
                    raise Uninstrumentable("code after the end of a function")
                return top, calls
            push(f[2])
        elif op == 0x1C:                                # select with its type
            n, i = _u32(b, i)
            i += n
            pop(3)
            push(1)
        elif op == 0x00:                                # unreachable
            unreachable()
        elif op == 0x0C:                                # br
            i = _skip_leb(b, i, 5)
            unreachable()
        elif op == 0x0D:                                # br_if
            i = _skip_leb(b, i, 5)
            pop(1)
        elif op == 0x0E:                                # br_table
            n, i = _u32(b, i)
            for _ in range(n + 1):
                i = _skip_leb(b, i, 5)
            pop(1)
            unreachable()
        elif op == 0x0F:                                # return
            unreachable()
        elif op == 0x10:                                # call
            idx, i = _u32(b, i)
            if not 0 <= idx < len(func_types):
                raise Uninstrumentable("call to an unknown function")
            params, results = types[func_types[idx]]
            calls.append((at, i - at, "call", idx))
            pop(params)
            push(results)
        elif op == 0x11:                                # call_indirect
            tidx, i = _u32(b, i)
            i = _skip_leb(b, i, 5)
            if not 0 <= tidx < len(types):
                raise Uninstrumentable("call_indirect with an unknown type")
            params, results = types[tidx]
            calls.append((at, i - at, "call_indirect", None))
            pop(params + 1)
            push(results)
        elif op == 0xFC:
            sub, i = _u32(b, i)
            if sub not in _FC:
                raise Uninstrumentable(f"unsupported instruction 0xfc {sub}")
            pops, pushes, imms = _FC[sub]
            for k in imms:
                i = _immediate(b, i, k)
            pop(pops)
            push(pushes)
        elif op == 0xFD:
            sub, i = _u32(b, i)
            if sub not in _FD:
                raise Uninstrumentable(f"unsupported instruction 0xfd {sub}")
            pops, pushes, imms = _FD[sub]
            for k in imms:
                i = _immediate(b, i, k)
            pop(pops)
            push(pushes)
        else:
            raise Uninstrumentable(f"unsupported instruction 0x{op:02x}")
    raise Uninstrumentable("function body without its end")


# --- the module ----------------------------------------------------------------------

_ORDER = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 12: 10, 10: 11, 11: 12}


def _sections(m: bytes) -> list[tuple[int, bytes]]:
    if m[:8] != b"\x00asm\x01\x00\x00\x00":
        raise Uninstrumentable("not a WebAssembly module")
    out, i = [], 8
    while i < len(m):
        sid = m[i]
        size, i = _u32(m, i + 1)
        if i + size > len(m):
            raise Uninstrumentable("truncated section")
        out.append((sid, m[i:i + size]))
        i += size
    return out


def _types(body: bytes) -> list[tuple[int, int]]:
    n, i = _u32(body, 0)
    out = []
    for _ in range(n):
        if body[i] != 0x60:
            raise Uninstrumentable("only function types are supported")
        params, i = _u32(body, i + 1)
        i += params
        results, i = _u32(body, i)
        i += results
        out.append((params, results))
    return out


def _vec_u32(body: bytes) -> list[int]:
    n, i = _u32(body, 0)
    out = []
    for _ in range(n):
        v, i = _u32(body, i)
        out.append(v)
    return out


def _section(sid: int, body: bytes) -> bytes:
    return bytes([sid]) + _u32_bytes(len(body)) + body


def costs(module: bytes) -> list[int]:
    """Each defined function's cost in slots (exposed for tests)."""
    return _analyse(_sections(module))[0]


def _analyse(sections):
    by_id = {}
    for sid, body in sections:
        if sid != 0:
            by_id[sid] = body
    if 2 in by_id and _u32(by_id[2], 0)[0]:
        raise Uninstrumentable("imports are not allowed")
    types = _types(by_id[1]) if 1 in by_id else []
    func_types = _vec_u32(by_id[3]) if 3 in by_id else []
    if any(not 0 <= t < len(types) for t in func_types):
        raise Uninstrumentable("function with an unknown type")
    code = by_id.get(10, b"\x00")
    n, i = _u32(code, 0)
    if n != len(func_types):
        raise Uninstrumentable("function and code sections disagree")
    bodies, costs_, scans = [], [], []
    for f in range(n):
        size, i = _u32(code, i)
        start, end = i, i + size
        groups, j = _u32(code, start)
        locals_ = 0
        for _ in range(groups):
            count, j = _u32(code, j)
            locals_ += count
            j += 1
        height, calls = scan_body(code, j, end, types, func_types)
        params = types[func_types[f]][0]
        costs_.append(params + locals_ + height + PRELUDE_HEIGHT + FRAME_OVERHEAD)
        bodies.append((start, j, end))
        scans.append(calls)
        i = end
    return costs_, bodies, scans, code, by_id


def instrument(module: bytes) -> bytes:
    """`module` with the deterministic stack limit added (see the module docstring)."""
    sections = _sections(module)
    costs_, bodies, scans, code, by_id = _analyse(sections)
    nglobals = _u32(by_id[6], 0)[0] if 6 in by_id else 0
    g = _u32_bytes(nglobals)
    worst = max(costs_, default=FRAME_OVERHEAD)

    def check(cost: int) -> tuple[bytes, bytes]:
        c = _s32_bytes(cost)
        before = (b"\x23" + g + b"\x41" + c + b"\x6a" + b"\x24" + g          # depth += cost
                  + b"\x23" + g + b"\x41" + _s32_bytes(STACK_LIMIT) + b"\x4b"  # depth > limit?
                  + b"\x04\x40\x00\x0b")                                       # then trap
        after = b"\x23" + g + b"\x41" + c + b"\x6b" + b"\x24" + g             # depth -= cost
        return before, after

    new_bodies = []
    for (start, j, end), calls in zip(bodies, scans):
        out = bytearray(code[start:j])                  # the locals, unchanged
        pos = j
        for at, size, kind, callee in calls:
            out += code[pos:at]
            before, after = check(costs_[callee] if kind == "call" else worst)
            out += before + code[at:at + size] + after
            pos = at + size
        out += code[pos:end]
        new_bodies.append(_u32_bytes(len(out)) + bytes(out))
    new_code = _u32_bytes(len(new_bodies)) + b"".join(new_bodies)

    counter = b"\x7f\x01\x41\x00\x0b"                   # (global (mut i32) (i32.const 0))
    if 6 in by_id:
        body = by_id[6]
        _, i = _u32(body, 0)
        new_globals = _u32_bytes(nglobals + 1) + body[i:] + counter
    else:
        new_globals = _u32_bytes(1) + counter
    out = bytearray(module[:8])
    placed = False
    for sid, body in sections:
        if sid != 0 and sid != 6 and not placed and _ORDER.get(sid, 0) > _ORDER[6]:
            out += _section(6, new_globals)              # the first section that comes after globals
            placed = True
        if sid == 6:
            body, placed = new_globals, True
        elif sid == 10:
            body = new_code
        out += _section(sid, body)
    if not placed:
        out += _section(6, new_globals)
    return bytes(out)
