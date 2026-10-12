"""Recovery phrases (BIP-39) and key derivation (SLIP-0010 Ed25519): published
test vectors, typo handling, and the wallet commands built on them."""

import io
import json

import pytest

from green import cli, hd
from green.crypto import Key

# BIP-39 reference vectors (trezor/python-mnemonic vectors.json, passphrase "TREZOR").
BIP39 = [
    ("00" * 16, "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
     "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e53495531f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04"),
    ("7f" * 16, "legal winner thank year wave sausage worth useful legal winner thank yellow",
     "2e8905819b8723fe2c1d161860e5ee1830318dbf49a83bd451cfb8440c28bd6fa457fe1296106559a3c80937a1c1069be3a3a5bd381ee6260e8d9739fce1f607"),
    ("80" * 16, "letter advice cage absurd amount doctor acoustic avoid letter advice cage above",
     "d71de856f81a8acc65e6fc851a38d4d7ec216fd0796d0a6827a3ad6ed5511a30fa280f12eb2e47ed2ac03b5c462a0358d18d69fe4f985ec81778c1b370b652a8"),
    ("ff" * 16, "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong",
     "ac27495480225222079d7be181583751e86f571027b0497b5b5d11218e0a8a13332572917f0f8e5a589620c6f15b11c61dee327651a14c34e18231052e48c069"),
    ("00" * 32, " ".join(["abandon"] * 23 + ["art"]),
     "bda85446c68413707090a52022edd26a1c9462295029f2e60cd7c4f2bbd3097170af7a4d73245cafa9c3cca8d561a7c3de6f5d4a10be8ed2a5e608d68f92fcc8"),
]

# SLIP-0010 test vector 1 for Ed25519: (path, private key, public key without SLIP-10's 00 prefix).
SLIP10_SEED = "000102030405060708090a0b0c0d0e0f"
SLIP10 = [
    ("m", "2b4be7f19ee27bbf30c667b642d5f4aa69fd169872f8fc3059c08ebae2eb19e7",
     "a4b2856bfec510abab89753fac1ac0e1112364e7d250545963f135f2a33188ed"),
    ("m/0'", "68e0fe46dfb67e368c75379acec591dad19df3cde26e63b93a8e704f1dade7a3",
     "8c8a13df77a28f3445213a0f432fde644acaa215fc72dcdf300d5efaa85d350c"),
    ("m/0'/1'", "b1d0bad404bf35da785a64ca1ac54b2617211d2777696fbffaf208f746ae84f2",
     "1932a5270f335bed617d5b935c80aedb1a35bd9fc1e31acafd5372c30f5c1187"),
    ("m/0'/1'/2'", "92a5b23c0b8a99e37d07df3fb9966917f5d06e02ddbd909c7e184371463e9fc9",
     "ae98736566d30ed0e9d2f4486a64bc95740d89c7db33f52121f8ea8f76ff0fc1"),
]


@pytest.mark.parametrize("entropy,phrase,seed", BIP39)
def test_bip39_reference_vectors(entropy, phrase, seed):
    assert hd.entropy_to_phrase(bytes.fromhex(entropy)) == phrase
    assert hd.phrase_to_entropy(phrase).hex() == entropy
    assert hd.phrase_to_seed(phrase, "TREZOR").hex() == seed


def test_slip10_ed25519_reference_vector():
    seed = bytes.fromhex(SLIP10_SEED)
    for path, private, public in SLIP10:
        key = hd.derive(seed, path)
        assert key.hex() == private, path
        assert Key.from_hex(key.hex()).pub_hex == public, path
    with pytest.raises(ValueError, match="hardened"):
        hd.derive(seed, "m/0")                      # Ed25519 has no unhardened children
    for bad in ("44'/0'", "m/x'", "m//0'", f"m/{2**31}'"):
        with pytest.raises(ValueError):
            hd.parse_path(bad)


def test_typos_are_caught_not_guessed():
    phrase = BIP39[1][1]
    words = phrase.split()
    swapped = " ".join([words[1], words[0]] + words[2:])
    with pytest.raises(hd.BadPhrase, match="checksum"):
        hd.phrase_to_seed(swapped)                  # two words out of order
    with pytest.raises(hd.BadPhrase, match="word 3"):
        hd.phrase_to_entropy(phrase.replace("thank", "thnak"))
    with pytest.raises(hd.BadPhrase, match="11 words"):
        hd.phrase_to_entropy(" ".join(words[:11]))
    # The first four letters identify a word; spacing and case do not matter.
    short = "  LEGA winn than year wave saus wort usef lega winn than yell "
    assert hd.normalize_phrase(short) == phrase
    assert hd.phrase_to_seed(short, "TREZOR").hex() == BIP39[1][2]
    with pytest.raises(hd.BadPhrase):
        hd.phrase_to_entropy(phrase.replace("legal", "leg"))   # three letters are ambiguous


def test_new_phrases_and_accounts():
    a, b = hd.new_phrase(), hd.new_phrase()
    assert a != b and len(a.split()) == 24 and len(hd.new_phrase(12).split()) == 12
    keys = {hd.key_from_phrase(a, "", hd.account_path(i)).address for i in range(3)}
    keys.add(hd.key_from_phrase(a, "extra word").address)       # a passphrase is another wallet
    assert len(keys) == 4


def _valid(phrase: str) -> bool:
    try:
        hd.phrase_to_entropy(phrase)
        return True
    except hd.BadPhrase:
        return False


def run(args, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    return cli.main(args)


def test_a_wallet_restores_from_its_phrase(tmp_path, monkeypatch, capsys):
    a, b, c = (str(tmp_path / x) for x in "abc")
    assert cli.main(["--data", a, "wallet", "new", "--no-password"]) == 0
    with open(f"{a}/wallet.json") as f:
        made = json.load(f)
    shown = capsys.readouterr().out
    assert all(w in shown for w in made["phrase"].split()) and made["address"] in shown
    # restored without the file: same address
    assert run(["--data", b, "wallet", "restore", "--no-password"], made["phrase"] + "\n", monkeypatch) == 0
    with open(f"{b}/wallet.json") as f:
        assert json.load(f)["address"] == made["address"]
    # encrypted: the phrase is not in the file, and the password opens it
    monkeypatch.setenv("GREEN_WALLET_PASSWORD", "correct horse")
    assert run(["--data", c, "wallet", "restore", "--account", "2"], made["phrase"] + "\n", monkeypatch) == 0
    text = open(f"{c}/wallet.json").read()
    assert made["phrase"] not in text and "encrypted_phrase" in text
    key = cli.load_wallet(c)
    assert key.address == hd.key_from_phrase(made["phrase"], "", hd.account_path(2)).address != made["address"]
    capsys.readouterr()
    assert cli.main(["--data", c, "wallet", "phrase"]) == 0
    assert made["phrase"].split()[-1] in capsys.readouterr().out
    monkeypatch.setenv("GREEN_WALLET_PASSWORD", "wrong")
    assert cli.main(["--data", c, "wallet", "phrase"]) == 1
    # a damaged phrase is refused, and no wallet is written
    words = made["phrase"].split()
    bad = next(p for p in (" ".join(words[:-1] + [w]) for w in hd.WORDS) if not _valid(p))
    assert run(["--data", str(tmp_path / "d"), "wallet", "restore", "--no-password"], bad, monkeypatch) == 1
    assert not (tmp_path / "d" / "wallet.json").exists()


def test_older_wallets_still_open(tmp_path, capsys):
    key = Key.generate()
    (tmp_path / "wallet.json").write_text(json.dumps({"address": key.address, "public_key": key.pub_hex,
                                                      "secret": key.secret_hex()}))
    assert cli.load_wallet(str(tmp_path)).address == key.address
    assert cli.main(["--data", str(tmp_path), "wallet", "phrase"]) == 1    # it has none: told what to do
    assert cli.main(["--data", str(tmp_path), "wallet", "show"]) == 0
    assert "none" in capsys.readouterr().out
