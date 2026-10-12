"""Signed releases: enough distinct maintainer keys, over exactly these files."""

import os
import subprocess
import sys

import pytest

from green import release
from green.crypto import Key


def make_release(tmp_path):
    folder = tmp_path / "release"
    folder.mkdir()
    for name, body in (("green-0.8.0.tar.gz", b"source"), ("grn-0.8.0-win64.zip", b"binary")):
        (folder / name).write_bytes(body)
    sums = release.make_sums([str(p) for p in folder.iterdir()]).encode()
    (folder / "SHA256SUMS").write_bytes(sums)
    return folder, sums


def test_a_release_needs_enough_distinct_maintainers_and_unchanged_files(tmp_path):
    folder, sums = make_release(tmp_path)
    maintainers = [Key.generate() for _ in range(3)]
    keys = [k.pub_hex for k in maintainers]
    sigs = [release.sign(maintainers[0], sums), release.sign(maintainers[2], sums)]
    r = release.verify_release(sums, sigs, keys, 2, folder=str(folder))
    assert r["ok"] and len(r["signers"]) == 2 and set(r["files"].values()) == {"ok"}

    assert not release.verify_release(sums, sigs[:1], keys, 2)["ok_signatures"]          # one key is not enough
    assert not release.verify_release(sums, sigs[:1] * 2, keys, 2)["ok_signatures"]      # nor the same key twice
    outsider = release.sign(Key.generate(), sums)
    assert not release.verify_release(sums, [sigs[0], outsider], keys, 2)["ok_signatures"]  # unknown key ignored
    other = release.sign(maintainers[1], sums.replace(b"  ", b"  x", 1))
    assert not release.verify_release(sums, [sigs[0], other], keys, 2)["ok_signatures"]  # signed something else
    tx_style = maintainers[1].sign(sums)                                                 # no release domain
    assert not release.verify_release(sums, [sigs[0], f"{keys[1]} {tx_style}"], keys, 2)["ok_signatures"]

    (folder / "grn-0.8.0-win64.zip").write_bytes(b"trojan")
    r = release.verify_release(sums, sigs, keys, 2, folder=str(folder))
    assert not r["ok"] and r["files"]["grn-0.8.0-win64.zip"] == "changed"


def test_sums_files_are_parsed_strictly():
    good = "ab" * 32
    for bad in ("", f"{good} one-space", f"{good}  ../etc/passwd", f"zz  file", f"{good}  a\n{good}  a"):
        with pytest.raises(ValueError):
            release.parse_sums(bad)


def test_release_commands_end_to_end(tmp_path):
    folder, _ = make_release(tmp_path)
    run = lambda *a: subprocess.run([sys.executable, "-m", "green", *a], capture_output=True, text=True,  # noqa: E731
                                    cwd=os.path.dirname(os.path.dirname(__file__)), timeout=60)
    sums = run("release", "sums", *[str(folder / n) for n in sorted(os.listdir(folder)) if n != "SHA256SUMS"])
    assert sums.returncode == 0
    (folder / "SHA256SUMS").write_text(sums.stdout, newline="")
    pubs = []
    for who in ("m1", "m2"):
        assert run("--data", str(tmp_path / who), "wallet", "new", "--no-password").returncode == 0
        out = run("--data", str(tmp_path / who), "release", "sign", str(folder / "SHA256SUMS"),
                  "--out", str(folder / f"{who}.sig"))
        assert out.returncode == 0, out.stderr
        pubs.append(out.stdout.split()[2])
    ok = run("release", "verify", str(folder / "SHA256SUMS"), str(folder / "m1.sig"), str(folder / "m2.sig"),
             "--keys", ",".join(pubs), "--threshold", "2")
    assert ok.returncode == 0 and "Genuine release." in ok.stdout, ok.stdout + ok.stderr
    (folder / "green-0.8.0.tar.gz").write_bytes(b"tampered")
    bad = run("release", "verify", str(folder / "SHA256SUMS"), str(folder / "m1.sig"), str(folder / "m2.sig"),
              "--keys", ",".join(pubs), "--threshold", "2")
    assert bad.returncode != 0 and "changed" in bad.stdout and "NOT a genuine release" in bad.stdout + bad.stderr
