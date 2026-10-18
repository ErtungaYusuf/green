"""Sealed requests: a buyer's request to a provider that only the provider can read,
and an answer only that buyer can read, whoever carries them in between.

The provider advertises an X25519 key in its signed offer. For every request the
buyer makes a fresh X25519 key pair, derives a key with the provider's (HKDF-SHA256)
and encrypts the request with ChaCha20-Poly1305; the provider answers under the same
key. A relay (p2p.py) sees only the provider's address and opaque bytes; the provider
sees neither who asked nor from where, only the relay that brought it. Each request's
key is used for one request and one answer, so nothing links two requests either,
except what the request says (its channel).

    sealed request : ephemeral public key (32) | nonce (12) | ciphertext
    sealed answer  : nonce (12) | ciphertext
    plaintext      : header length (4, big-endian) | JSON header | body
"""

import json
import os
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

INFO = b"green sealed request v1"
MAX_SEALED = 4 * 1024 * 1024 - 4096          # fits one P2P frame with room for the relay's header


class SealError(ValueError):
    pass


def provider_key(secret_hex: str) -> X25519PrivateKey:
    """A provider's X25519 key, derived from its wallet key (nothing more to keep safe)."""
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"green provider kx v1")
    return X25519PrivateKey.from_private_bytes(hkdf.derive(bytes.fromhex(secret_hex)))


def public_hex(key: X25519PrivateKey) -> str:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _key(shared: bytes, eph_pub: bytes, provider_pub: bytes) -> ChaCha20Poly1305:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=eph_pub + provider_pub, info=INFO)
    return ChaCha20Poly1305(hkdf.derive(shared))


def _pack(header: dict, body: bytes) -> bytes:
    h = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(h)) + h + body


def _unpack(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 4:
        raise SealError("truncated")
    n = struct.unpack(">I", data[:4])[0]
    if n > 64 * 1024 or 4 + n > len(data):
        raise SealError("bad header")
    try:
        header = json.loads(data[4:4 + n])
    except ValueError:
        raise SealError("bad header") from None
    if not isinstance(header, dict):
        raise SealError("bad header")
    return header, data[4 + n:]


def seal_request(provider_pub_hex: str, header: dict, body: bytes) -> tuple[bytes, ChaCha20Poly1305]:
    """(sealed request, the key to open its answer with)."""
    provider_pub = bytes.fromhex(provider_pub_hex)
    eph = X25519PrivateKey.generate()
    eph_pub = eph.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    key = _key(eph.exchange(X25519PublicKey.from_public_bytes(provider_pub)), eph_pub, provider_pub)
    nonce = os.urandom(12)
    sealed = eph_pub + nonce + key.encrypt(nonce, _pack(header, body), b"request")
    if len(sealed) > MAX_SEALED:
        raise SealError(f"request larger than {MAX_SEALED} bytes: split it")
    return sealed, key


def open_request(provider: X25519PrivateKey, sealed: bytes) -> tuple[dict, bytes, ChaCha20Poly1305]:
    """(header, body, the key to seal the answer with), or SealError."""
    if not 44 + 16 <= len(sealed) <= MAX_SEALED:
        raise SealError("bad size")
    eph_pub, nonce, ct = sealed[:32], sealed[32:44], sealed[44:]
    provider_pub = provider.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    try:
        key = _key(provider.exchange(X25519PublicKey.from_public_bytes(eph_pub)), eph_pub, provider_pub)
        plain = key.decrypt(nonce, ct, b"request")
    except (InvalidTag, ValueError):
        raise SealError("cannot open: not for us, or altered") from None
    header, body = _unpack(plain)
    return header, body, key


def seal_answer(key: ChaCha20Poly1305, header: dict, body: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + key.encrypt(nonce, _pack(header, body), b"answer")


def open_answer(key: ChaCha20Poly1305, sealed: bytes) -> tuple[dict, bytes]:
    if len(sealed) < 12 + 16:
        raise SealError("bad size")
    try:
        plain = key.decrypt(sealed[:12], sealed[12:], b"answer")
    except InvalidTag:
        raise SealError("the answer was altered on the way") from None
    return _unpack(plain)
