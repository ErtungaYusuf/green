"""Signed releases: how a user checks that a node release comes from the maintainers.

A release ships a SHA256SUMS file (the format `sha256sum` writes and checks:
one "<hex digest>  <file name>" line per file) and detached Ed25519
signatures of it, one per maintainer. A release is genuine when at least
RELEASE_THRESHOLD different known maintainer keys signed that exact SHA256SUMS
and every listed file still has its digest. Signing with several independent
keys means one stolen key cannot ship a malicious release on its own.

A signature file is one line: "<maintainer public key hex> <signature hex>".
The signed message is "grn-release-v1\\n" followed by the SHA256SUMS bytes, so a
release signature can never be mistaken for a transaction signature.
"""

import hashlib
import os

from .crypto import Key, is_hex, verify

DOMAIN = b"grn-release-v1\n"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_sums(paths: list[str]) -> str:
    """SHA256SUMS text for the files (names relative to their common folder)."""
    lines = []
    for p in sorted(paths, key=lambda x: os.path.basename(x)):
        name = os.path.basename(p)
        if any(c.isspace() for c in name):
            raise ValueError(f"file names in a release cannot contain spaces: {name!r}")
        lines.append(f"{file_sha256(p)}  {name}")
    return "\n".join(lines) + "\n"


def parse_sums(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if not sep or not is_hex(digest, 64) or not name or name in out or "/" in name or "\\" in name:
            raise ValueError(f"bad SHA256SUMS line: {line!r}")
        out[name] = digest
    if not out:
        raise ValueError("SHA256SUMS lists no files")
    return out


def sign(key: Key, sums: bytes) -> str:
    return f"{key.pub_hex} {key.sign(DOMAIN + sums)}\n"


def verify_release(sums: bytes, signatures: list[str], keys: list[str], threshold: int,
                   folder: str | None = None) -> dict:
    """Check the signatures (and, given the folder, every file's digest).

    Returns {"signers": [...], "ok_signatures": bool, "files": {name: "ok"|"missing"|"changed"}, "ok": bool}.
    """
    known = {k.lower() for k in keys if is_hex(k.lower(), 64)}
    signers = set()
    for text in signatures:
        parts = text.split()
        if len(parts) == 2 and parts[0] in known and verify(parts[0], DOMAIN + sums, parts[1]):
            signers.add(parts[0])
    files = {}
    listed = parse_sums(sums.decode())
    if folder is not None:
        for name, digest in listed.items():
            path = os.path.join(folder, name)
            files[name] = "missing" if not os.path.exists(path) else ("ok" if file_sha256(path) == digest else "changed")
    ok_sigs = threshold > 0 and len(signers) >= threshold
    return {"signers": sorted(signers), "ok_signatures": ok_sigs, "files": files,
            "ok": ok_sigs and all(v == "ok" for v in files.values())}
