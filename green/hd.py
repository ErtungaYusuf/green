"""Recovery phrases (BIP-39) and hierarchical keys (SLIP-0010, Ed25519).

A wallet is a list of 24 common English words. Written on paper (or stamped in
steel), it restores the wallet on any machine, with no file and no password:

    words  --BIP-39 PBKDF2-->  64-byte seed  --SLIP-0010-->  m/44'/coin'/account'  -->  Ed25519 key

BIP-39: 256 bits of entropy plus an 8-bit SHA-256 checksum, split into 11-bit
indexes into the standard 2048-word list (green/bip39_english.txt, checked
against its published SHA-256 when loaded). The checksum catches a wrong or
misspelled word, and words can be given by their first four letters, which are
unique. An optional passphrase (the "25th word") gives a different wallet.

SLIP-0010 is BIP-32 for Ed25519, the scheme hardware wallets use for Ed25519
coins. Ed25519 has only hardened derivation, so every path element is
hardened; one phrase gives any number of independent accounts.
"""

import hashlib
import hmac
import os
import unicodedata
from pathlib import Path

from . import params
from .crypto import Key

WORDLIST_SHA256 = "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda"
HARDENED = 0x80000000
WORD_COUNTS = (12, 15, 18, 21, 24)


def _load_words() -> list[str]:
    data = Path(__file__).with_name("bip39_english.txt").read_bytes()
    if hashlib.sha256(data).hexdigest() != WORDLIST_SHA256:
        raise RuntimeError("green/bip39_english.txt is damaged: recovery phrases would be wrong. Reinstall Green.")
    words = data.decode("ascii").split()
    assert len(words) == 2048 and words == sorted(words)
    return words


WORDS = _load_words()
INDEX = {w: i for i, w in enumerate(WORDS)}


class BadPhrase(ValueError):
    pass


def _nfkd(text: str) -> str:
    return unicodedata.normalize("NFKD", text)


def entropy_to_phrase(entropy: bytes) -> str:
    if len(entropy) not in (16, 20, 24, 28, 32):
        raise ValueError("entropy must be 128 to 256 bits in steps of 32")
    bits = len(entropy) * 8
    cs = bits // 32
    n = int.from_bytes(entropy, "big") << cs | hashlib.sha256(entropy).digest()[0] >> (8 - cs)
    count = (bits + cs) // 11
    return " ".join(WORDS[(n >> 11 * (count - 1 - i)) & 2047] for i in range(count))


def new_phrase(words: int = 24) -> str:
    if words not in WORD_COUNTS:
        raise ValueError(f"a phrase has {', '.join(map(str, WORD_COUNTS))} words")
    return entropy_to_phrase(os.urandom(words * 4 // 3))


def words_of(phrase: str) -> list[str]:
    """The phrase's words in standard form. A word may be given by its first four
    letters (or any longer unambiguous start); anything else is refused with the
    position of the word, never silently corrected."""
    if not isinstance(phrase, str):
        raise BadPhrase("the phrase must be text")
    out = []
    for pos, w in enumerate(_nfkd(phrase).lower().split(), 1):
        if w not in INDEX:
            matches = [x for x in WORDS if x.startswith(w)] if len(w) >= 4 else []
            if len(matches) != 1:
                raise BadPhrase(f"word {pos} ('{w}') is not in the BIP-39 English word list")
            w = matches[0]
        out.append(w)
    if len(out) not in WORD_COUNTS:
        raise BadPhrase(f"the phrase has {len(out)} words; it should have 12, 15, 18, 21 or 24")
    return out


def phrase_to_entropy(phrase: str) -> bytes:
    words = words_of(phrase)
    n = 0
    for w in words:
        n = n << 11 | INDEX[w]
    total = len(words) * 11
    cs = total // 33
    entropy = (n >> cs).to_bytes((total - cs) // 8, "big")
    if n & ((1 << cs) - 1) != hashlib.sha256(entropy).digest()[0] >> (8 - cs):
        raise BadPhrase("the phrase's checksum does not match: a word is wrong or out of order")
    return entropy


def normalize_phrase(phrase: str) -> str:
    """Checked, standard spelling (full words, single spaces)."""
    phrase_to_entropy(phrase)
    return " ".join(words_of(phrase))


def phrase_to_seed(phrase: str, passphrase: str = "") -> bytes:
    """BIP-39 seed. The phrase is checked first: a typo must not silently open an empty wallet."""
    text = normalize_phrase(phrase)
    return hashlib.pbkdf2_hmac("sha512", _nfkd(text).encode(), ("mnemonic" + _nfkd(passphrase)).encode(), 2048)


# --- SLIP-0010 (Ed25519) ---------------------------------------------------------

def master_key(seed: bytes) -> tuple[bytes, bytes]:
    i = hmac.new(b"ed25519 seed", seed, hashlib.sha512).digest()
    return i[:32], i[32:]


def child_key(key: bytes, chain_code: bytes, index: int) -> tuple[bytes, bytes]:
    if not HARDENED <= index < 2**32:
        raise ValueError("Ed25519 keys only have hardened children")
    i = hmac.new(chain_code, b"\x00" + key + index.to_bytes(4, "big"), hashlib.sha512).digest()
    return i[:32], i[32:]


def parse_path(path: str) -> list[int]:
    """"m/44'/7777'/0'" -> hardened indexes. Every element must be hardened (' or h)."""
    parts = path.split("/") if isinstance(path, str) else []
    if not parts or parts[0] != "m":
        raise ValueError(f"bad derivation path {path!r}: it starts with m/")
    out = []
    for p in parts[1:]:
        if not p or p[-1] not in "'hH" or not p[:-1].isdigit() or int(p[:-1]) >= HARDENED:
            raise ValueError(f"bad derivation path {path!r}: every element is hardened, like 44'")
        out.append(int(p[:-1]) | HARDENED)
    return out


def derive(seed: bytes, path: str) -> bytes:
    key, chain_code = master_key(seed)
    for index in parse_path(path):
        key, chain_code = child_key(key, chain_code, index)
    return key


def account_path(account: int = 0) -> str:
    if not 0 <= account < HARDENED:
        raise ValueError("account must be between 0 and 2^31 - 1")
    return f"m/44'/{params.HD_COIN_TYPE}'/{account}'"


def key_from_phrase(phrase: str, passphrase: str = "", path: str | None = None) -> Key:
    return Key.from_hex(derive(phrase_to_seed(phrase, passphrase), path or account_path(0)).hex())
