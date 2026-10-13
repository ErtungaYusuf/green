"""Transactions and the binary formats of job / result data ("blobs").

Account model (balances + nonces). Transaction types:

  transfer       move coins
  job_post       buy compute: lock price * units in escrow for a matmul job
  job_commit     sell compute, step 1: commit to hash(miner, job, units, root)
                 for a batch of units (a bitmap over the job) and lock a bond
                 per unit; reserves the units for the miner
  job_reveal     sell compute, step 2: reveal the Merkle root of the result
                 hashes (plus a hint where the full list is); the batch is paid
                 when the challenge window closes
  job_challenge  dispute units of a revealed batch (anyone; in practice the
                 buyer, whose client checks every result with Freivalds)
  job_respond    answer a disputed unit with its result hash and a Merkle
                 proof; every validator fetches the result and Freivalds-checks
                 it. One unanswered unit forfeits the whole batch
  job_cancel     buyer closes the job; takes effect CANCEL_DELAY blocks later
  guard_set      put the sender under a guardian with a spending limit per
                 period (or tighten its existing guard)
  guard_update   the guardian loosens or tightens the limit, freezes or
                 unfreezes the account, or hands the guard to someone else
  guard_release  the guardian removes the guard

Senders are single keys or multisignature accounts. A multisig sender's `pub`
is {"k": k, "keys": [n sorted, distinct public keys]} and its `sig` a list of
n entries, exactly k of them signatures (the rest null) by the matching keys.
Its address is the hash of k and the keys, so it looks like any other address:
anyone can pay it, and k of the n key holders together spend from it. Each
cosigner signs the same bytes on their own device (add_signature).

Signatures cover the chain id (the genesis hash), so a transaction signed for
one network is never valid on another.

Large data (the job's matrices, the results) never goes into blocks. Blocks
only carry their hashes; nodes exchange the bytes separately as blobs. Result
blobs (256 KB each) and the manifest (the list of a batch's result hashes) are
fetched by the buyer from the miner; validators only fetch a disputed result.
"""

import struct

import numpy as np

from . import params
from .crypto import (ADDRESS_HRP, Key, address_of, bech32m_encode, canonical, h, hhex, is_address, is_hex,
                     merkle_root, verify)

N = params.UNIT
JOB_MAGIC = b"GJOB"
RESULT_BYTES = N * N * 4
HASH_BYTES = 32
MAX_BITMAP_HEX = 2 * -(-params.MAX_JOB_UNITS // 8)
MAX_PROOF_LEN = (params.MAX_BATCH_UNITS - 1).bit_length()

FIELDS = {
    "transfer": {"to", "amount"},
    "job_post": {"job", "dims", "price"},
    "job_commit": {"job", "units", "commitment"},
    "job_reveal": {"job", "batch", "root", "manifest"},
    "job_challenge": {"job", "batch", "units"},
    "job_respond": {"job", "batch", "unit", "result", "proof"},
    "job_cancel": {"job"},
    "task_post": {"prog", "input", "fuel", "price"},
    "task_commit": {"task", "commitment"},
    "task_reveal": {"task", "output"},
    "task_challenge": {"task"},
    "guard_set": {"guardian", "limit", "period"},
    "guard_update": {"account", "guardian", "limit", "period", "frozen"},
    "guard_release": {"account"},
    "channel_open": {"provider", "amount", "expires"},
    "channel_claim": {"channel", "total", "voucher"},
}
COMMON = {"type", "pub", "nonce", "fee", "sig"}
# Fields a transaction may carry or leave out. A job_commit's payees share the
# batch's payment (a pool paying the miners who computed its units).
OPTIONAL = {"job_commit": {"payees"}, "channel_claim": {"payees", "close"}}


class InvalidTx(Exception):
    pass


class FutureNonce(InvalidTx):
    """The sender has an earlier transaction this node has not seen yet: valid later, perhaps."""


def txid(tx: dict) -> str:
    return hhex(canonical(tx))


def signing_bytes(tx: dict, chain_id: str) -> bytes:
    return canonical({"chain": chain_id, "tx": {k: v for k, v in tx.items() if k != "sig"}})


def make_tx(key: Key, chain_id: str, tx_type: str, nonce: int, fee: int = params.MIN_TX_FEE, **fields) -> dict:
    tx = {"type": tx_type, "pub": key.pub_hex, "nonce": nonce, "fee": fee, **fields}
    tx["sig"] = key.sign(signing_bytes(tx, chain_id))
    return tx


def is_multisig(pub) -> bool:
    """A well-formed multisig descriptor: 1 <= k <= n, 2 <= n <= MAX_MULTISIG_KEYS, keys sorted and distinct."""
    if not isinstance(pub, dict) or set(pub) != {"k", "keys"}:
        return False
    keys, k = pub["keys"], pub["k"]
    return (isinstance(keys, list) and 2 <= len(keys) <= params.MAX_MULTISIG_KEYS
            and all(is_hex(x, 64) for x in keys) and keys == sorted(set(keys))
            and is_uint(k) and 1 <= k <= len(keys))


def multisig_address(k: int, keys: list[str]) -> str:
    return bech32m_encode(ADDRESS_HRP, h(canonical(["grn-multisig", k, sorted(keys)]))[:20])


def pub_address(pub) -> str:
    return address_of(pub) if isinstance(pub, str) else multisig_address(pub["k"], pub["keys"])


def sender(tx: dict) -> str:
    return pub_address(tx["pub"])


def sig_count(tx: dict) -> int:
    """Signatures a validator checks for this transaction (a claim's voucher is one more)."""
    return (tx["pub"]["k"] if isinstance(tx["pub"], dict) else 1) + (tx.get("type") == "channel_claim")


def voucher_bytes(chain_id: str, channel_id: str, total: int) -> bytes:
    """What a channel's buyer signs: everything it owes the provider so far."""
    return canonical(["grn-voucher", chain_id, channel_id, total])


def sign_voucher(key: Key, chain_id: str, channel_id: str, total: int) -> str:
    return key.sign(voucher_bytes(chain_id, channel_id, total))


def make_multisig_tx(chain_id: str, k: int, keys: list[str], tx_type: str, nonce: int,
                     fee: int = params.MIN_TX_FEE, **fields) -> dict:
    """An unsigned multisig transaction; each cosigner then calls add_signature."""
    keys = sorted(keys)
    return {"type": tx_type, "pub": {"k": k, "keys": keys}, "nonce": nonce, "fee": fee, **fields,
            "sig": [None] * len(keys)}


def add_signature(tx: dict, key: Key, chain_id: str) -> dict:
    """The transaction with `key`'s signature in its slot (a copy)."""
    keys = tx["pub"]["keys"]
    if key.pub_hex not in keys:
        raise InvalidTx("this key is not one of the account's keys")
    sigs = list(tx["sig"])
    sigs[keys.index(key.pub_hex)] = key.sign(signing_bytes(tx, chain_id))
    return {**tx, "sig": sigs}


def _check_auth(tx: dict, chain_id: str) -> None:
    pub, sig = tx["pub"], tx["sig"]
    if isinstance(pub, str):
        if not is_hex(pub, 64):
            raise InvalidTx("bad public key")
        if not verify(pub, signing_bytes(tx, chain_id), sig):
            raise InvalidTx("bad signature")
        return
    if not is_multisig(pub):
        raise InvalidTx(f"bad multisig account: k of n sorted distinct keys, 2 <= n <= {params.MAX_MULTISIG_KEYS}")
    if not isinstance(sig, list) or len(sig) != len(pub["keys"]):
        raise InvalidTx("a multisig transaction needs one signature slot per key")
    present = [(key, s) for key, s in zip(pub["keys"], sig) if s is not None]
    if len(present) != pub["k"]:
        raise InvalidTx(f"a {pub['k']}-of-{len(pub['keys'])} account needs exactly {pub['k']} signatures")
    message = signing_bytes(tx, chain_id)
    if not all(verify(key, message, s) for key, s in present):
        raise InvalidTx("bad signature")


def commitment(addr: str, job_id: str, bitmap: str, root: str) -> str:
    """Binds a batch to its miner: the commit carries this, and it doubles as the batch id."""
    return hhex(canonical(["green-commit", addr, job_id, bitmap, root]))


def task_id_of(tx: dict) -> str:
    """A task's id is the hash of its post (unique per sender+nonce)."""
    return hhex(canonical(["green-task", tx["pub"], tx["nonce"], tx["prog"], tx["input"], tx["fuel"], tx["price"]]))


def task_commitment(addr: str, task_id: str, output_hash: str) -> str:
    return hhex(canonical(["green-task-commit", addr, task_id, output_hash]))


def is_uint(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 2**63


def _is_unit(u) -> bool:
    return isinstance(u, list) and len(u) == 3 and all(is_uint(x) and x < params.MAX_JOB_UNITS_PER_DIM for x in u)


def is_payee_list(x) -> bool:
    """[[address, weight], ...]: 2 to MAX_PAYEES entries sorted by address, addresses
    distinct, weights in 1..MAX_PAYEE_WEIGHT. Splits a block's reward (the header's
    `miner`) or a batch's payment (a job_commit's `payees`)."""
    if not isinstance(x, list) or not 2 <= len(x) <= params.MAX_PAYEES:
        return False
    for entry in x:
        if not (isinstance(entry, list) and len(entry) == 2 and is_address(entry[0])
                and is_uint(entry[1]) and 1 <= entry[1] <= params.MAX_PAYEE_WEIGHT):
            return False
    addrs = [a for a, _ in x]
    return addrs == sorted(set(addrs))


def check_format(tx: dict, chain_id: str) -> None:
    """Stateless checks: shape of the fields and the signature."""
    if not isinstance(tx, dict) or tx.get("type") not in FIELDS:
        raise InvalidTx("unknown transaction type")
    expected = FIELDS[tx["type"]] | COMMON
    optional = OPTIONAL.get(tx["type"], set())
    if not expected <= set(tx) <= expected | optional:
        raise InvalidTx(f"fields must be exactly {sorted(expected)}"
                        + (f", optionally with {sorted(optional)}" if optional else ""))
    if not is_uint(tx["nonce"]) or not is_uint(tx["fee"]):
        raise InvalidTx("nonce and fee must be non-negative integers")
    kind = tx["type"]
    if kind == "transfer":
        if not is_address(tx["to"]):
            raise InvalidTx("bad recipient address")
        if not is_uint(tx["amount"]) or tx["amount"] == 0:
            raise InvalidTx("amount must be a positive integer")
    elif kind.startswith("job_") and not is_hex(tx["job"], 64):
        raise InvalidTx("bad job id")
    if kind == "job_post":
        dims = tx["dims"]
        if (not isinstance(dims, list) or len(dims) != 3
                or not all(is_uint(d) and 1 <= d <= params.MAX_JOB_UNITS_PER_DIM for d in dims)):
            raise InvalidTx(f"dims must be 3 integers in 1..{params.MAX_JOB_UNITS_PER_DIM}")
        if not is_uint(tx["price"]) or tx["price"] == 0:
            raise InvalidTx("price must be a positive integer")
    if kind in ("job_commit", "job_challenge") and not _is_bitmap(tx["units"]):
        raise InvalidTx("units must be a non-empty bitmap (lowercase hex)")
    if kind == "job_commit" and not is_hex(tx["commitment"], 64):
        raise InvalidTx("bad commitment")
    if kind in ("job_commit", "channel_claim") and "payees" in tx and not is_payee_list(tx["payees"]):
        raise InvalidTx("payees must be 2 to MAX_PAYEES [address, weight] pairs sorted by address")
    if kind == "channel_open":
        if not is_address(tx["provider"]):
            raise InvalidTx("bad provider address")
        if not is_uint(tx["amount"]) or tx["amount"] == 0 or not is_uint(tx["expires"]):
            raise InvalidTx("amount must be a positive integer and expires a height")
        if is_multisig(tx["pub"]):
            raise InvalidTx("a channel's buyer signs vouchers with a single key")
    if kind == "channel_claim":
        if not is_hex(tx["channel"], 64) or not is_uint(tx["total"]) or tx["total"] == 0:
            raise InvalidTx("bad channel id or total")
        if not is_hex(tx["voucher"], 128):
            raise InvalidTx("bad voucher signature")
        if "close" in tx and tx["close"] is not True:
            raise InvalidTx("close is true or left out")      # one spelling per transaction
    if kind in ("job_reveal", "job_challenge", "job_respond") and not is_hex(tx["batch"], 64):
        raise InvalidTx("bad batch id")
    if kind == "job_reveal" and not (is_hex(tx["root"], 64) and is_hex(tx["manifest"], 64)):
        raise InvalidTx("bad root or manifest hash")
    if kind == "job_respond":
        if not _is_unit(tx["unit"]) or not is_hex(tx["result"], 64):
            raise InvalidTx("bad unit or result hash")
        proof = tx["proof"]
        if not isinstance(proof, list) or len(proof) > MAX_PROOF_LEN or not all(is_hex(p, 64) for p in proof):
            raise InvalidTx(f"proof must be at most {MAX_PROOF_LEN} hashes")
    if kind == "task_post":
        if not is_hex(tx["prog"], 64) or not is_hex(tx["input"], 64):
            raise InvalidTx("prog and input must be blob hashes")
        if not is_uint(tx["fuel"]) or not 1 <= tx["fuel"] <= params.MAX_TASK_FUEL:
            raise InvalidTx(f"fuel must be in 1..{params.MAX_TASK_FUEL}")
        if not is_uint(tx["price"]) or tx["price"] == 0:
            raise InvalidTx("price must be a positive integer")
    if kind in ("task_commit", "task_reveal", "task_challenge") and not is_hex(tx["task"], 64):
        raise InvalidTx("bad task id")
    if kind == "task_commit" and not is_hex(tx["commitment"], 64):
        raise InvalidTx("bad commitment")
    if kind in ("guard_set", "guard_update"):
        if not is_address(tx["guardian"]):
            raise InvalidTx("bad guardian address")
        if not is_uint(tx["limit"]) or not is_uint(tx["period"]) or not 1 <= tx["period"] <= params.MAX_GUARD_PERIOD:
            raise InvalidTx(f"limit must be a non-negative integer, period in 1..{params.MAX_GUARD_PERIOD}")
    if kind in ("guard_update", "guard_release") and not is_address(tx["account"]):
        raise InvalidTx("bad account address")
    if kind == "guard_update" and not isinstance(tx["frozen"], bool):
        raise InvalidTx("frozen must be true or false")
    if kind == "task_reveal" and not is_hex(tx["output"], 64):
        raise InvalidTx("bad output hash")
    _check_auth(tx, chain_id)


def min_fee(tx: dict) -> int:
    """Lower bound computable from the transaction alone.

    task_post and task_challenge also pay for data / adjudication, which needs
    chain state; the chain checks those (see task_post_fee, params.challenge_fee).
    """
    fee = params.MIN_TX_FEE
    if tx["type"] == "job_post":
        fee += params.DATA_FEE_PER_BYTE * job_blob_size(tx["dims"])
    elif tx["type"] == "job_challenge":
        fee = job_challenge_fee(bitmap_count(tx["units"]))
    return fee


def job_challenge_fee(units: int = 1) -> int:
    """Each disputed unit makes every validator fetch one result blob, so a
    dispute pays for those bytes. There is no challenger bond: answering costs
    the miner only its response fee, so withholding data until disputed earns a
    miner nothing, and a griefer pays far more than the honest miner it bothers."""
    return params.MIN_TX_FEE + params.DATA_FEE_PER_BYTE * RESULT_BYTES * units


def tx_size(tx: dict) -> int:
    return len(canonical(tx))


def outflow(tx: dict) -> int:
    """What a transaction spends, for a guarded account's limit: value that
    leaves for good (transfers, escrow) plus the fee. Bonds are not counted:
    they come back unless the sender is proven to have cheated."""
    out = tx["fee"]
    if tx["type"] == "transfer":
        out += tx["amount"]
    elif tx["type"] == "job_post":
        mu, ku, nu = tx["dims"]
        out += tx["price"] * mu * ku * nu
    elif tx["type"] == "task_post":
        out += tx["price"]
    elif tx["type"] == "channel_open":
        out += tx["amount"]
    return out


def task_post_fee(program_bytes: int, input_bytes: int) -> int:
    """Program and input are stored by every node, so they pay per byte like job data."""
    return params.MIN_TX_FEE + params.DATA_FEE_PER_BYTE * (program_bytes + input_bytes)


# --- Job blobs -----------------------------------------------------------------
# Job blob:      b"GJOB" | M, K, N as big-endian uint32 | A (int8, M*K) | B (int8, K*N)
# Result blob:   one unit's partial product A_il @ B_lj as int32, N*N, little endian.
# Manifest blob: the 32-byte result hashes of a batch, in the order of its units.

def job_blob_size(dims) -> int:
    mu, ku, nu = dims
    return 16 + (mu * ku + ku * nu) * N * N


def encode_job(a: np.ndarray, b: np.ndarray) -> bytes:
    if a.dtype != np.int8 or b.dtype != np.int8:
        raise ValueError("job matrices must be int8")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dimensions differ")
    if m % N or k % N or n % N:
        raise ValueError(f"dimensions must be multiples of {N} (pad with zeros)")
    return JOB_MAGIC + struct.pack(">III", m, k, n) + a.tobytes() + b.tobytes()


def decode_job(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(blob) < 16 or blob[:4] != JOB_MAGIC:
        raise ValueError("not a job blob")
    m, k, n = struct.unpack(">III", blob[4:16])
    if not (m and k and n) or m % N or k % N or n % N:
        raise ValueError("bad job dimensions")
    if len(blob) != 16 + m * k + k * n:
        raise ValueError("job blob has wrong length")
    a = np.frombuffer(blob, dtype=np.int8, count=m * k, offset=16).reshape(m, k)
    b = np.frombuffer(blob, dtype=np.int8, count=k * n, offset=16 + m * k).reshape(k, n)
    return a, b


def job_dims_units(a: np.ndarray, b: np.ndarray) -> list[int]:
    return [a.shape[0] // N, a.shape[1] // N, b.shape[1] // N]


def unit_inputs(a: np.ndarray, b: np.ndarray, unit) -> tuple[np.ndarray, np.ndarray]:
    i, l, j = unit
    return a[i * N:(i + 1) * N, l * N:(l + 1) * N], b[l * N:(l + 1) * N, j * N:(j + 1) * N]


def encode_result(c: np.ndarray) -> bytes:
    return c.astype("<i4").tobytes()


def decode_result(blob: bytes) -> np.ndarray:
    if len(blob) != RESULT_BYTES:
        raise ValueError("result blob has wrong length")
    return np.frombuffer(blob, dtype="<i4").reshape(N, N)


# --- Unit bitmaps --------------------------------------------------------------
# A set of units of a job as a bitmap: unit index k = (i * K + l) * N + j (the
# order of all_units) is bit 7 - k % 8 of byte k // 8, hex encoded. Exactly
# ceil(units / 8) bytes, no bits past the last unit, so one set has one spelling.

def _is_bitmap(x) -> bool:
    return (isinstance(x, str) and 2 <= len(x) <= MAX_BITMAP_HEX and len(x) % 2 == 0
            and all(c in "0123456789abcdef" for c in x) and int(x, 16) != 0)


def unit_index(unit, dims) -> int:
    i, l, j = unit
    return (i * dims[1] + l) * dims[2] + j


def units_bitmap(units, dims) -> str:
    total = dims[0] * dims[1] * dims[2]
    bits = bytearray(-(-total // 8))
    for u in units:
        k = unit_index(u, dims)
        bits[k // 8] |= 0x80 >> (k % 8)
    return bits.hex()


def bitmap_units(bitmap: str, dims) -> list[list[int]] | None:
    """The units in a bitmap, in ascending index order; None if it does not fit `dims`."""
    total = dims[0] * dims[1] * dims[2]
    if not _is_bitmap(bitmap) or len(bitmap) != 2 * -(-total // 8):
        return None
    value = int(bitmap, 16)
    width = len(bitmap) * 4
    if value & ((1 << (width - total)) - 1):
        return None                                   # bits past the last unit
    return [u for k, u in enumerate(all_units(dims)) if value >> (width - 1 - k) & 1]


def bitmap_count(bitmap: str) -> int:
    return bin(int(bitmap, 16)).count("1") if _is_bitmap(bitmap) else 0


def encode_manifest(result_hashes: list[str]) -> bytes:
    return b"".join(bytes.fromhex(r) for r in result_hashes)


def decode_manifest(blob: bytes, count: int) -> list[str]:
    if len(blob) != HASH_BYTES * count:
        raise ValueError("manifest has wrong length")
    return [blob[i:i + HASH_BYTES].hex() for i in range(0, len(blob), HASH_BYTES)]


def manifest_leaves(blob: bytes | None, count: int, root: str) -> list[str] | None:
    """A batch's result hashes from its manifest, only if they match the revealed root."""
    if blob is None:
        return None
    try:
        leaves = decode_manifest(blob, count)
    except ValueError:
        return None
    return leaves if merkle_root(leaves) == root else None


def unit_key(unit) -> str:
    return ",".join(str(x) for x in unit)


def all_units(dims) -> list[list[int]]:
    mu, ku, nu = dims
    return [[i, l, j] for i in range(mu) for l in range(ku) for j in range(nu)]
