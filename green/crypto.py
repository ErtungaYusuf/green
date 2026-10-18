"""Hashing, canonical encoding, Ed25519 signatures and wallet encryption."""

import functools
import hashlib
import json
import os
import re

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

ZERO_HASH = "00" * 32


def h(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=32).digest()


def hhex(data: bytes) -> str:
    return h(data).hex()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


# --- Addresses: bech32m (BIP-350) --------------------------------------------
# "grn1" + 32 data characters + 6 checksum characters. The BCH checksum detects
# every error that touches up to 4 characters, and any other error with
# probability ~1 - 2^-30, so a mistyped address is rejected instead of sending
# coins to a key nobody holds. The alphabet has no 1/b/i/o, and the string is
# all lowercase, so it reads aloud and survives copying.

ADDRESS_HRP = "grn"
_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_BECH32M_CONST = 0x2BC830A3
_GEN = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)


def _polymod(values) -> int:
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            if (top >> i) & 1:
                chk ^= _GEN[i]
    return chk


def _hrp_expand(hrp: str) -> list[int]:
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def _convertbits(data, frombits: int, tobits: int, pad: bool) -> list[int] | None:
    acc = bits = 0
    out = []
    maxv = (1 << tobits) - 1
    for value in data:
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            out.append((acc >> bits) & maxv)
    if pad:
        if bits:
            out.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return out


def bech32m_encode(hrp: str, payload: bytes) -> str:
    data = _convertbits(payload, 8, 5, True)
    polymod = _polymod(_hrp_expand(hrp) + data + [0] * 6) ^ _BECH32M_CONST
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(_CHARSET[d] for d in data + checksum)


def bech32m_decode(text: str) -> tuple[str, bytes] | None:
    """(hrp, payload), or None if `text` is not valid bech32m."""
    if not isinstance(text, str) or len(text) > 90 or text.lower() != text and text.upper() != text:
        return None
    text = text.lower()
    pos = text.rfind("1")
    if pos < 1 or pos + 7 > len(text) or any(not 33 <= ord(c) <= 126 for c in text):
        return None
    hrp = text[:pos]
    if any(c not in _CHARSET for c in text[pos + 1:]):
        return None
    data = [_CHARSET.find(c) for c in text[pos + 1:]]
    if _polymod(_hrp_expand(hrp) + data) != _BECH32M_CONST:
        return None
    payload = _convertbits(data[:-6], 5, 8, False)
    return None if payload is None else (hrp, bytes(payload))


@functools.lru_cache(maxsize=1 << 16)       # every transaction asks for its sender's address
def address_of(pub_hex: str) -> str:
    return bech32m_encode(ADDRESS_HRP, h(bytes.fromhex(pub_hex))[:20])


_LOWER_HEX = re.compile(r"[0-9a-f]*")


def is_hex(text, length: int) -> bool:
    return isinstance(text, str) and len(text) == length and _LOWER_HEX.fullmatch(text) is not None


def is_address(text) -> bool:
    """Canonical form only (lowercase): addresses are ledger keys, so one account
    must never have two spellings."""
    return isinstance(text, str) and len(text) <= 90 and _is_address(text)


@functools.lru_cache(maxsize=1 << 16)
def _is_address(text: str) -> bool:
    if text != text.lower():
        return False
    decoded = bech32m_decode(text)
    return decoded is not None and decoded[0] == ADDRESS_HRP and len(decoded[1]) == 20


class Key:
    def __init__(self, private: Ed25519PrivateKey):
        self._private = private
        self.pub_hex = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        self.address = address_of(self.pub_hex)

    @classmethod
    def generate(cls) -> "Key":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_hex(cls, secret_hex: str) -> "Key":
        return cls(Ed25519PrivateKey.from_private_bytes(bytes.fromhex(secret_hex)))

    def secret_hex(self) -> str:
        return self._private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()

    def sign(self, message: bytes) -> str:
        return self._private.sign(message).hex()


# Signatures already found good, so a transaction checked on arrival in the mempool is
# not checked again when its block comes (Bitcoin's sigcache). Only successes are kept,
# keyed by a hash of key, signature and message together: a hit means exactly this
# triple verified before. Bounded; emptied when full.
SIG_CACHE_SIZE = 200_000
_good_sigs: set[bytes] = set()


def verify(pub_hex: str, message: bytes, sig_hex: str) -> bool:
    if not is_hex(pub_hex, 64) or not is_hex(sig_hex, 128):
        return False
    key = h(bytes.fromhex(pub_hex) + bytes.fromhex(sig_hex) + message)    # 32 + 64 bytes, then the message
    if key in _good_sigs:
        return True
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex)).verify(bytes.fromhex(sig_hex), message)
    except (InvalidSignature, ValueError):
        return False
    if len(_good_sigs) >= SIG_CACHE_SIZE:
        _good_sigs.clear()
    _good_sigs.add(key)
    return True


def merkle_root(leaves_hex: list[str]) -> str:
    """Merkle root that cannot be mutated.

    Leaves and inner nodes use different prefixes, an odd node is carried up
    unchanged instead of being duplicated, and the leaf count is mixed into the
    root. So no two different transaction lists share a root (Bitcoin's
    CVE-2012-2459 came from duplicating the last node).
    """
    level = [h(b"\x00" + bytes.fromhex(x)) for x in leaves_hex]
    while len(level) > 1:
        nxt = [h(b"\x01" + level[i] + level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    top = level[0] if level else bytes(32)
    return h(b"\x02" + len(leaves_hex).to_bytes(8, "big") + top).hex()


def merkle_proof(leaves_hex: list[str], index: int) -> list[str]:
    """Sibling hashes from leaf `index` up to the top, for merkle_verify."""
    level = [h(b"\x00" + bytes.fromhex(x)) for x in leaves_hex]
    proof = []
    i = index
    while len(level) > 1:
        if i ^ 1 < len(level):            # an odd last node has no sibling: it is carried up
            proof.append(level[i ^ 1].hex())
        nxt = [h(b"\x01" + level[k] + level[k + 1]) for k in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
        i //= 2
    return proof


def merkle_verify(leaf_hex: str, index: int, count: int, proof: list[str], root_hex: str) -> bool:
    """Is `leaf_hex` the leaf at `index` of a `count`-leaf tree with this merkle_root?

    The shape of the tree follows from `count`, so a proof has exactly one valid
    length and each position decides left/right; leaves and inner nodes are
    domain separated. No other (index, leaf) pair can produce the same root.
    """
    if not (0 <= index < count) or not is_hex(leaf_hex, 64) or not all(is_hex(p, 64) for p in proof):
        return False
    node = h(b"\x00" + bytes.fromhex(leaf_hex))
    i, n, used = index, count, 0
    while n > 1:
        if i ^ 1 < n:
            if used == len(proof):
                return False
            sib = bytes.fromhex(proof[used])
            used += 1
            node = h(b"\x01" + (sib + node if i & 1 else node + sib))
        i //= 2
        n = (n + 1) // 2
    return used == len(proof) and h(b"\x02" + count.to_bytes(8, "big") + node).hex() == root_hex


# --- Wallet encryption ---------------------------------------------------------
# scrypt (memory-hard, slows down password guessing) + AES-256-GCM.

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**15, 8, 1


class WrongPassword(Exception):
    pass


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(password.encode())


def encrypt_secret(secret_hex: str, password: str) -> dict:
    salt, nonce = os.urandom(16), os.urandom(12)
    key = _derive(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    ct = AESGCM(key).encrypt(nonce, bytes.fromhex(secret_hex), b"green-wallet")
    return {"kdf": "scrypt", "n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P,
            "salt": salt.hex(), "nonce": nonce.hex(), "ciphertext": ct.hex()}


def decrypt_secret(box: dict, password: str) -> str:
    key = _derive(password, bytes.fromhex(box["salt"]), box["n"], box["r"], box["p"])
    try:
        return AESGCM(key).decrypt(bytes.fromhex(box["nonce"]), bytes.fromhex(box["ciphertext"]),
                                   b"green-wallet").hex()
    except InvalidTag:
        raise WrongPassword("wrong password") from None
