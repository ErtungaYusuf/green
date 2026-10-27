"""Command line interface: `python -m green <command>`."""

import argparse
import getpass
import json
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
import tomllib
import urllib.error

import numpy as np

from . import hd, params, tx as txmod, work
from .chain import block_hash, make_genesis
from .crypto import Key, WrongPassword, canonical, decrypt_secret, encrypt_secret, hhex, is_address
from .node import Node, http_get, http_post
from .tx import make_tx

N = params.UNIT
DEFAULT_NODE = f"http://127.0.0.1:{params.DEFAULT_PORT}"
PASSWORD_ENV = "GREEN_WALLET_PASSWORD"
PASSWORD_FILE_ENV = "GREEN_WALLET_PASSWORD_FILE"   # a file holding it (e.g. a systemd credential)


class CliError(Exception):
    pass


# --- wallet --------------------------------------------------------------------

def wallet_path(data: str) -> str:
    return os.path.join(data, "wallet.json")


def ask_password(confirm: bool = False, label: str = "Wallet password") -> str:
    """Wallet password from the terminal, or for automation from the file named by
    GREEN_WALLET_PASSWORD_FILE (preferred) or GREEN_WALLET_PASSWORD."""
    path = os.environ.get(PASSWORD_FILE_ENV)
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read().rstrip("\r\n")
        except OSError as e:
            raise CliError(f"cannot read the wallet password file {path}: {e}") from None
    env = os.environ.get(PASSWORD_ENV)
    if env is not None:
        return env
    if not sys.stdin.isatty():
        raise CliError(f"wallet password required: run the command from a terminal "
                       f"or set the {PASSWORD_ENV} environment variable")
    password = getpass.getpass(f"{label}: ")
    if confirm:
        if len(password) < 8:
            raise CliError("the password must be at least 8 characters")
        if getpass.getpass("Repeat the password: ") != password:
            raise CliError("passwords do not match")
    return password


def _read_wallet(data: str) -> dict | None:
    path = wallet_path(data)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _unencrypted_warning() -> None:
    print("WARNING: this wallet is stored unencrypted. Create a new encrypted wallet and move your coins to it.",
          file=sys.stderr)


def wallet_phrase(w: dict) -> dict:
    """{"phrase", "passphrase"} of a wallet made from a recovery phrase (asks for the password)."""
    if "phrase" in w:
        _unencrypted_warning()
        return {"phrase": w["phrase"], "passphrase": w.get("passphrase", "")}
    try:
        return json.loads(bytes.fromhex(decrypt_secret(w["encrypted_phrase"], ask_password())))
    except WrongPassword:
        raise CliError("wrong wallet password") from None


def load_wallet(data: str) -> Key | None:
    w = _read_wallet(data)
    if w is None:
        return None
    if "path" in w:                                   # made from a recovery phrase
        secret = wallet_phrase(w)
        key = hd.key_from_phrase(secret["phrase"], secret["passphrase"], w["path"])
    elif "secret" in w:
        _unencrypted_warning()
        return Key.from_hex(w["secret"])
    else:
        try:
            key = Key.from_hex(decrypt_secret(w["encrypted"], ask_password()))
        except WrongPassword:
            raise CliError("wrong wallet password") from None
    if key.address != w["address"]:
        raise CliError("wallet file is corrupt: the address does not match the key")
    return key


def create_wallet(data: str, password: str | None, phrase: str | None = None, passphrase: str = "",
                  account: int = 0) -> Key:
    """A wallet whose key comes from a recovery phrase (a new one unless `phrase` is
    given); the file keeps the phrase, encrypted with `password` unless it is None."""
    os.makedirs(data, exist_ok=True)
    phrase = hd.normalize_phrase(phrase) if phrase is not None else hd.new_phrase()
    path = hd.account_path(account)
    key = hd.key_from_phrase(phrase, passphrase, path)
    record = {"address": key.address, "public_key": key.pub_hex, "path": path, "words": len(phrase.split())}
    secret = {"phrase": phrase, "passphrase": passphrase}
    if password is None:
        record.update(secret)
    else:
        record["encrypted_phrase"] = encrypt_secret(canonical(secret).hex(), password)
    fd = os.open(wallet_path(data), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(record, f, indent=2)
    return key


def print_phrase(phrase: str) -> None:
    words = phrase.split()
    for r in range(0, len(words), 4):              # in reading order: 1 2 3 4 / 5 6 7 8 ...
        print("   " + "".join(f"{i + 1:>3}. {words[i]:<10}" for i in range(r, min(r + 4, len(words)))))


def require_wallet(data: str) -> Key:
    key = load_wallet(data)
    if key is None:
        raise CliError(f"no wallet in '{data}'. First run: python -m green --data {data} wallet new")
    return key


def wallet_address(data: str) -> str:
    """The wallet's address, without asking for the password."""
    path = wallet_path(data)
    if not os.path.exists(path):
        raise CliError(f"no wallet in '{data}'. First run: python -m green --data {data} wallet new")
    with open(path) as f:
        return json.load(f)["address"]


# --- helpers -------------------------------------------------------------------

def api_get(args, path: str, raw: bool = False):
    try:
        return http_get(args.node.rstrip("/") + path, timeout=30, raw=raw)
    except urllib.error.HTTPError as e:
        raise CliError(f"node error: {e.read().decode(errors='replace')}") from None
    except (OSError, urllib.error.URLError):
        raise CliError(f"no node running at {args.node}. Start one with: python -m green node") from None


def api_post(args, path: str, body, raw: bool = False):
    try:
        return http_post(args.node.rstrip("/") + path, body, timeout=60, raw=raw)
    except urllib.error.HTTPError as e:
        raise CliError(f"node rejected it: {e.read().decode(errors='replace')}") from None
    except (OSError, urllib.error.URLError):
        raise CliError(f"no node running at {args.node}. Start one with: python -m green node") from None


def send_tx(args, key: Key, tx_type: str, required_fee: int | None = None, **fields) -> str:
    chain_id = api_get(args, "/status")["genesis"]
    nonce = api_get(args, f"/account/{key.address}")["pending_nonce"]
    fee = max(txmod.min_fee({"type": tx_type, **fields}), required_fee or 0)
    if getattr(args, "fee", None):
        fee = max(fee, params.parse_amount(args.fee))
    else:
        # The node's estimate for the next 2 blocks; the size barely depends on the fee.
        rate = api_get(args, "/v1/fee?blocks=2")["fee_rate"]
        size = txmod.tx_size(make_tx(key, chain_id, tx_type, nonce, fee, **fields))
        fee = max(fee, int(rate * size) + 1 if rate else 0)
    t = make_tx(key, chain_id, tx_type, nonce, fee, **fields)
    return api_post(args, "/tx", t)["txid"]


def pad_to_unit(m: np.ndarray) -> np.ndarray:
    rows = -(-m.shape[0] // N) * N
    cols = -(-m.shape[1] // N) * N
    out = np.zeros((rows, cols), dtype=np.int8)
    out[:m.shape[0], :m.shape[1]] = m
    return out


def load_int8(path: str) -> np.ndarray:
    m = np.load(path)
    if m.ndim != 2:
        raise CliError(f"{path}: must be a 2-dimensional matrix")
    if m.dtype != np.int8:
        if not np.issubdtype(m.dtype, np.integer) or m.min() < -128 or m.max() > 127:
            raise CliError(f"{path}: values must be int8 (-128..127). Quantize the model to int8 first.")
        m = m.astype(np.int8)
    return m


# --- commands ------------------------------------------------------------------

def _ask_passphrase(args) -> str:
    if not args.passphrase:
        return ""
    if not sys.stdin.isatty():
        raise CliError("--passphrase needs a terminal")
    first = getpass.getpass("Passphrase (an extra word for the phrase; forget it and the coins are gone): ")
    if args.action == "new" and getpass.getpass("Repeat the passphrase: ") != first:
        raise CliError("passphrases do not match")
    return first


def _read_phrase() -> str:
    if sys.stdin.isatty():
        return getpass.getpass("Recovery phrase (the words, separated by spaces; input hidden): ")
    return sys.stdin.readline()


def cmd_wallet(args):
    if args.action in ("new", "restore"):
        if os.path.exists(wallet_path(args.data)):
            raise CliError(f"{wallet_path(args.data)} already exists; not overwriting it.")
        if args.action == "restore":
            try:
                phrase = hd.normalize_phrase(_read_phrase())
            except hd.BadPhrase as e:
                raise CliError(f"that phrase cannot be used: {e}") from None
        else:
            phrase = hd.new_phrase(args.words)
        passphrase = _ask_passphrase(args)
        password = None if args.no_password else ask_password(confirm=True)
        key = create_wallet(args.data, password, phrase, passphrase, args.account)
        print(f"{'Restored' if args.action == 'restore' else 'New'} wallet: {wallet_path(args.data)}")
        print(f"Address: {key.address}   (account {args.account}, path {hd.account_path(args.account)})")
        if args.action == "new":
            print(f"\nRecovery phrase ({len(phrase.split())} words). Write it on paper, in order, and keep it "
                  f"offline.\nWith it the wallet can be restored anywhere (grn wallet restore), even without this "
                  f"file or its password.\nAnyone who sees it can take your coins; nobody, not even Green's "
                  f"developers, can recover it for you.\n")
            print_phrase(phrase)
            print("\n(`grn wallet phrase` shows it again.)")
        if password is None:
            print("WARNING: the wallet file is NOT encrypted. Anyone who gets the file gets your coins. Test use only.")
        else:
            print("The wallet file is encrypted with your password (scrypt + AES-256-GCM).")
    elif args.action == "phrase":
        w = _read_wallet(args.data)
        if w is None:
            raise CliError(f"no wallet in '{args.data}'")
        if "path" not in w:
            raise CliError("this wallet was made before recovery phrases (v0.14). Create a new wallet in another "
                           "folder and move the coins to it: then a phrase restores it.")
        secret = wallet_phrase(w)
        print(f"Recovery phrase for {w['address']} (path {w['path']}"
              f"{', with a passphrase' if secret['passphrase'] else ''}). Keep it secret:\n")
        print_phrase(secret["phrase"])
    else:
        print(f"Address    : {wallet_address(args.data)}")
        w = _read_wallet(args.data)
        if w.get("public_key"):
            print(f"Public key : {w['public_key']}   (give this to co-signers of a multisig account)")
        print(f"Recovery   : {w['words']}-word phrase, path {w['path']} (grn wallet phrase)" if "path" in w else
              "Recovery   : none (an older wallet: back up the file, or move the coins to a new wallet)")


def cmd_multisig(args):
    """k-of-n accounts: an unsigned transaction file goes from cosigner to cosigner."""
    if args.ms_cmd == "address":
        if not 1 <= args.k <= len(args.keys):
            raise CliError("k must be between 1 and the number of keys")
        print(txmod.multisig_address(args.k, args.keys))
        return
    if args.ms_cmd == "new":
        st = api_get(args, "/status")
        addr = txmod.multisig_address(args.k, args.keys)
        if args.to:
            kind, fields = "transfer", {"to": args.to, "amount": params.parse_amount(args.amount)}
        else:
            kind, fields = args.type, json.loads(args.fields or "{}")
        nonce = api_get(args, f"/account/{addr}")["pending_nonce"]
        fee = None if args.fee is None else params.parse_amount(args.fee)    # None: the least, per signature
        t = txmod.make_multisig_tx(st["genesis"], args.k, args.keys, kind, nonce, fee, **fields)
        with open(args.out, "w") as f:
            json.dump({"chain": st["genesis"], "tx": t}, f, indent=1)
        print(f"Unsigned {kind} from {addr} written to {args.out}; it needs {args.k} signatures (multisig sign)")
        return
    with open(args.file) as f:
        doc = json.load(f)
    if args.ms_cmd == "sign":
        key = require_wallet(args.data)
        try:
            doc["tx"] = txmod.add_signature(doc["tx"], key, doc["chain"])
        except txmod.InvalidTx as e:
            raise CliError(str(e)) from None
        with open(args.file, "w") as f:
            json.dump(doc, f, indent=1)
        have = sum(s is not None for s in doc["tx"]["sig"])
        print(f"Signed: {have} of {doc['tx']['pub']['k']} signatures")
    else:
        print(f"Sent: {api_post(args, '/tx', doc['tx'])['txid']}")


def cmd_genesis(args):
    address = wallet_address(args.data)
    path = os.path.join(args.data, "genesis.json")
    if os.path.exists(path) and not args.force:
        raise CliError(f"{path} already exists. Use --force to start a new network.")
    print(f"Mining the genesis block (first reward {params.fmt_amount(params.emission(params.TOTAL_SUPPLY)[0])} "
          f"-> {address})")
    print(f"Message: {args.msg!r}")
    started = time.time()
    interactive = sys.stdout.isatty()

    def progress(nonce):
        if interactive and nonce % 25 == 0:
            print(f"  {nonce + 1} work units tried...", end="\r", flush=True)

    if args.checkpoints:
        key_file = checkpoint_key_path(args.data)            # (not `path`: that is where the genesis goes)
        if not os.path.exists(key_file):
            raise CliError(f"--checkpoints needs a checkpoint key: grn --data {args.data} checkpoint key")
        with open(key_file, encoding="utf-8") as f:
            pub = json.load(f)["pub"]
        until = args.checkpoint_until or params.CHECKPOINT_DEFAULT_BLOCKS
        args.msg = f"{args.msg} | checkpoints {pub} until {until}"
        if len(args.msg) > 200:
            raise CliError("the message is too long with the checkpoint key in it (200 characters at most)")
        print(f"Checkpoints: signed by {pub[:16]}... until block {until} (or until you end them)")
        print(f"Message: {args.msg!r}")
    target = params.INITIAL_TARGET
    if args.units_per_block:
        target = min(target, params.MAX_TARGET // args.units_per_block)
    genesis = make_genesis(address, args.msg, on_try=progress, target=target)
    with open(path, "w") as f:
        json.dump(genesis, f, indent=2)
    if interactive:
        print(" " * 40, end="\r")
    print(f"Genesis found! {genesis['header']['nonce'] + 1} work units, {time.time() - started:.1f} s")
    print(f"Hash: {block_hash(genesis)}")
    print(f"Saved: {path}")
    print(f"Now start the node: python -m green --data {args.data} node --mine")


CHECKPOINT_KEY_FILE = "checkpoint_key.json"
CHECKPOINT_PASSWORD_ENV = "GREEN_CHECKPOINT_PASSWORD"


def checkpoint_key_path(data: str) -> str:
    return os.path.join(data, CHECKPOINT_KEY_FILE)


def load_checkpoint_key(data: str) -> Key:
    """The founder's checkpoint key (its password from GREEN_CHECKPOINT_PASSWORD or the terminal)."""
    path = checkpoint_key_path(data)
    if not os.path.exists(path):
        raise CliError(f"no checkpoint key in {data}: make one with: grn --data {data} checkpoint key")
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    if "secret" in record:
        key = Key.from_hex(record["secret"])
    else:
        password = os.environ.get(CHECKPOINT_PASSWORD_ENV)
        if password is None:
            if not sys.stdin.isatty():
                raise CliError(f"checkpoint key password required: set {CHECKPOINT_PASSWORD_ENV} or use a terminal")
            password = getpass.getpass("Checkpoint key password: ")
        try:
            key = Key.from_hex(decrypt_secret(record["encrypted"], password))
        except WrongPassword:
            raise CliError("wrong checkpoint key password") from None
    if key.pub_hex != record["pub"]:
        raise CliError(f"{path} is corrupt: the key does not match its public key")
    return key


def cmd_checkpoint(args):
    from .chain import checkpoint_end_bytes
    if args.action == "key":
        path = checkpoint_key_path(args.data)
        if os.path.exists(path):
            raise CliError(f"{path} already exists")
        os.makedirs(args.data, exist_ok=True)
        key = Key.generate()
        record = {"pub": key.pub_hex}
        if args.no_password:
            record["secret"] = key.secret_hex()
        else:
            password = os.environ.get(CHECKPOINT_PASSWORD_ENV) or ask_password(confirm=True, label="Checkpoint key password")
            record["encrypted"] = encrypt_secret(key.secret_hex(), password)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
        print(f"Checkpoint key: {path}")
        print(f"Public key: {key.pub_hex}")
        print("Back it up and keep it secret: whoever holds it decides which chain is final until the "
              "checkpoint period ends. Next: grn genesis --msg \"...\" --checkpoints")
    elif args.action == "status":
        st = api_get(args, "/checkpoint")
        if st["authority"] is None:
            print("This network has no checkpoints: the chain with the most work is the chain.")
            return
        print(f"Authority : {st['authority']}")
        print(f"Until     : block {st['until']}" + ("  (ended by the authority)" if st["ended"] else ""))
        cp = st["checkpoint"]
        print(f"Latest    : " + (f"block {cp['height']} {cp['block'][:16]}..." if cp else "none yet"))
    else:                                           # end
        if not args.yes:
            raise CliError("this ends the checkpoint period for good: from then on the chain with the most work "
                           "decides alone. Run again with --yes to do it.")
        key = load_checkpoint_key(args.data)
        chain_id = api_get(args, "/status")["genesis"]
        r = api_post(args, "/checkpoint", {"end": True, "sig": key.sign(checkpoint_end_bytes(chain_id))})
        print("The checkpoint period is over: every node now follows the chain with the most work."
              if r.get("new") else "The checkpoint period had already ended.")


def node_genesis(args) -> dict:
    """The genesis a node starts from: --genesis, the data folder's, or the main network's."""
    os.makedirs(args.data, exist_ok=True)
    local_genesis = os.path.join(args.data, "genesis.json")
    source = args.genesis or local_genesis
    if not os.path.exists(source):
        from . import mainnet
        genesis = mainnet.genesis()
        if genesis is None or args.genesis:
            raise CliError(f"no genesis file: {source}. For a new network: python -m green genesis --msg \"...\" "
                           f"or pass --genesis <file> to join an existing one.")
        with open(local_genesis, "w") as f:              # no genesis given: the Green main network
            json.dump(genesis, f, indent=2)
        source = local_genesis
        print("Joining the Green main network")
    with open(source) as f:
        genesis = json.load(f)
    if source != local_genesis and not os.path.exists(local_genesis):
        with open(local_genesis, "w") as f:
            json.dump(genesis, f, indent=2)
    return genesis


def cmd_node(args):
    genesis = node_genesis(args)
    key = require_wallet(args.data)
    checkpoint_key = load_checkpoint_key(args.data) if args.checkpoints else None
    logger = setup_logging(args.log_level, args.log_file)
    log = logger.info

    node = Node(args.data, genesis, port=args.port, host=args.host, peers=args.peer, miner_key=key, log=log,
                debug=logger.debug if logger.isEnabledFor(logging.DEBUG) else None,
                use_gpu=args.gpu, p2p_port=args.p2p_port, listen=not args.no_listen, fast_sync=args.fast_sync,
                p2p_host=args.p2p_host, no_signal=args.no_signal, prune=args.prune, prune_keep=args.prune_keep,
                dandelion=not args.no_dandelion, portmap=args.portmap, checkpoint_key=checkpoint_key)
    if checkpoint_key is not None:
        if node.chain.authority is None or node.chain.authority[0] != checkpoint_key.pub_hex:
            raise CliError("this node's checkpoint key is not the one this network's genesis names")
        log(f"Signing this network's checkpoints (until block {node.chain.authority[1]})")
    if not 1 <= args.intensity <= 100:
        raise CliError("--intensity is a percentage, 1..100")
    node.intensity = args.intensity
    node.start(mine=args.mine)
    pool = None
    if args.pool_port is not None:
        from .pool import Pool
        pool = Pool(node, key.address, host=args.pool_host, port=args.pool_port, fee_percent=args.pool_fee)
        pool.start()
        log(f"Pool open on {args.pool_host}:{pool.port}: miners run "
            f"`grn pool-mine --pool http://<this host>:{pool.port} --address <theirs>`; "
            f"blocks pay them directly, the pool keeps {args.pool_fee}%")
    provider = None
    if args.provide_port is not None:
        from .compute import Provider
        provider = Provider(node, key, host=args.provide_host, port=args.provide_port, price=args.provide_price,
                            url=args.provide_url)
        provider.start()
        log(f"Selling matrix products on {args.provide_host}:{provider.port} at {args.provide_price} yaprak per "
            f"unit; " + (f"advertised to the network as {provider.url} (once this wallet holds 0.1 GRN)"
                         if provider.url else "not advertised: set --provide-url so buyers can find it"))
    log(f"{params.COIN_NAME} node running. API: {node.url}  (genesis {node.chain.genesis_hash[:16]}...)")
    if node.p2p.listen:
        log(f"P2P: {node.p2p.address}  -- other nodes connect with --peer {node.p2p.full_address}")
    log(f"Height {node.chain.height}, mining: {'on' if args.mine else 'off'}")
    stop = node.shutdown_requested               # set by POST /shutdown (`grn stop`) or a signal
    for name in ("SIGTERM", "SIGBREAK"):         # SIGBREAK: Ctrl+Break, or a launcher's stop on Windows
        if hasattr(signal, name) and threading.current_thread() is threading.main_thread():
            signal.signal(getattr(signal, name), lambda *_: stop.set())
    try:
        while not stop.wait(args.report):
            st = node.status()
            acct = node.account(key.address)
            log(f"height {st['height']} | {st['units_per_second']:.1f} units/s | "
                f"~{st['units_per_block']:.0f} units per block | peers {len(st['peers'])} | "
                f"mempool {st['mempool']} | balance {params.fmt_amount(acct['balance'])}")
    except KeyboardInterrupt:
        pass
    log("Shutting down...")
    watchdog = threading.Timer(SHUTDOWN_SECONDS, _stuck_shutdown, args=(logger,))
    watchdog.daemon = True
    watchdog.start()
    if pool is not None:
        pool.stop()
    if provider is not None:
        provider.claim_due(force=True)           # collect what is owed before going away
        provider.stop()
    node.stop()
    watchdog.cancel()
    log("Stopped.")


SHUTDOWN_SECONDS = 45          # a node that has not stopped by then logs where it is stuck and exits


def _stuck_shutdown(logger) -> None:
    """The shutdown hangs: say where every thread is (the log is how such a bug gets found),
    then end the process; the store's write-ahead log keeps the chain safe."""
    from .node import thread_stacks
    logger.error(f"[error] shutdown has not finished after {SHUTDOWN_SECONDS} s; where every thread is:")
    for name, stack in thread_stacks().items():
        logger.error(f"--- {name}\n" + "".join(stack))
    logging.shutdown()
    os._exit(3)


def setup_logging(level: str, path: str | None) -> logging.Logger:
    """Console lines with the time; optionally a log file rotated at 10 MB (5 kept)."""
    logger = logging.getLogger("green")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(console)
    if path:
        f = logging.handlers.RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        f.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(f)
    return logger


# Settings a node reads from <data>/green.toml ([node] table); flags on the command line win.
CONFIG_KEYS = {"port": int, "host": str, "p2p_port": int, "p2p_host": str, "peer": list, "genesis": str,
               "mine": bool, "gpu": bool, "fast_sync": bool, "no_listen": bool, "no_signal": list,
               "report": (int, float), "log_level": str, "log_file": str, "prune": bool, "prune_keep": int,
               "pool_port": int, "pool_host": str, "pool_fee": int, "no_dandelion": bool,
               "provide_port": int, "provide_host": str, "provide_url": str, "provide_price": int,
               "portmap": bool, "intensity": int, "checkpoints": bool}


def load_config(data: str) -> dict:
    path = os.path.join(data, "green.toml")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f).get("node", {})
    except tomllib.TOMLDecodeError as e:
        raise CliError(f"{path}: {e}") from None
    for k, v in cfg.items():
        if k not in CONFIG_KEYS:
            raise CliError(f"{path}: unknown setting {k!r} (known: {', '.join(sorted(CONFIG_KEYS))})")
        if not isinstance(v, CONFIG_KEYS[k]) or (CONFIG_KEYS[k] is int and isinstance(v, bool)):
            raise CliError(f"{path}: {k} has the wrong type")
    if cfg.get("log_level", "info").lower() not in ("debug", "info", "warning", "error"):
        raise CliError(f"{path}: log_level must be debug, info, warning or error")
    return cfg


def cmd_tx(args):
    v = api_get(args, f"/v1/tx/{args.txid}")
    if v["status"] == "pending":
        print(f"{args.txid}: pending in the mempool")
    else:
        print(f"{args.txid}: in block {v['height']} ({v['block'][:16]}...), position {v['position']}, "
              f"{v['confirmations']} confirmations{' (body pruned on this node)' if v['pruned'] else ''}")
    if v.get("tx"):
        print(json.dumps(v["tx"], indent=2))


def cmd_history(args):
    addr = args.address or wallet_address(args.data)
    h = api_get(args, f"/v1/address/{addr}/history?limit={args.limit}")
    for p in h["pending"]:
        print(f"pending   {p['type']:<14} {p['role']:<9} {p['txid']}")
    for it in h["items"]:
        ref = it.get("txid") or it.get("block")
        print(f"{it['height']:>8}  {it['type']:<14} {it['role']:<9} {ref}")
    if not h["pending"] and not h["items"]:
        print("No activity.")


def cmd_pool_mine(args):
    from .pool import PoolMiner
    gpu = None
    if args.gpu:
        from . import gpu as gpumod
        gpu = gpumod.get_miner()
        if gpu is None:
            raise CliError("no suitable GPU found")
    miner = PoolMiner(args.pool, args.address or wallet_address(args.data), gpu=gpu,
                      log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    print(f"Mining for {args.pool}; payouts to {miner.address}, in the blocks themselves. Ctrl+C stops.")
    try:
        miner.run()
    except KeyboardInterrupt:
        print(f"Stopped: {miner.shares} shares accepted, {miner.blocks} blocks found, "
              f"{miner.useful} job units computed and paid for, {miner.rejected} refused")


def cmd_bump(args):
    """Replace a pending transaction of this wallet with the same one at a higher fee."""
    key = require_wallet(args.data)
    view = api_get(args, f"/v1/tx/{args.txid}")
    if view.get("status") != "pending":
        raise CliError("that transaction is not pending: it is confirmed already, or this node does not know it")
    t = view["tx"]
    if t.get("pub") != key.pub_hex:
        raise CliError("that transaction was not sent by this wallet")
    need = t["fee"] + max(params.MIN_TX_FEE, t["fee"] // 10)      # what replace-by-fee accepts
    if args.fee:
        fee = params.parse_amount(args.fee)
    else:
        rate = api_get(args, "/v1/fee?blocks=1")["fee_rate"]
        fee = max(need, int(rate * txmod.tx_size(t)) + 1)
    if fee < need:
        raise CliError(f"a replacement must pay at least {params.fmt_amount(need)}")
    fields = {k: v for k, v in t.items() if k not in txmod.COMMON}
    new = make_tx(key, api_get(args, "/status")["genesis"], t["type"], t["nonce"], fee, **fields)
    txid = api_post(args, "/tx", new)["txid"]
    print(f"Replaced: fee {params.fmt_amount(t['fee'])} -> {params.fmt_amount(fee)}. New txid: {txid}")


def cmd_providers(args):
    offers = api_get(args, "/v1/providers")
    if not offers:
        print("No compute on offer that this node knows of.")
        return
    print(f"{'price/unit':>11}  {'max dim':>7}  provider                                     url")
    for o in offers:
        print(f"{o['price']:>11}  {o['max_dim']:>7}  {o['provider']}  {o['url']}")
    print("(A unit is 2^24 multiply-adds; green.compute.ComputeCluster.from_market opens channels to them.)")


def cmd_fee(args):
    est = api_get(args, f"/v1/fee?blocks={args.blocks}")
    print(f"To be included within {est['blocks']} block(s): {est['fee_rate']} yaprak per byte "
          f"(a transfer: {params.fmt_amount(est['transfer_fee'])}; based on {est['basis']}, "
          f"{est['mempool']} transactions waiting)")


def cmd_status(args):
    st = api_get(args, "/status")
    print(f"Network    : {st['coin']} (genesis {st['genesis'][:16]}...)")
    print(f"Height     : {st['height']}  tip {st['tip'][:16]}...")
    print(f"Difficulty : ~{st['units_per_block']:.0f} work units per block")
    print(f"Mining     : {'on' if st['mining'] else 'off'}  ({st['units_per_second']:.1f} units/s)")
    print(f"Work units : {st['stats']['units']} total, {st['stats']['useful_units']} useful, "
          f"{st['stats']['blocks_found']} blocks found, {st['stats']['results_revealed']} results delivered")
    p2p = st["p2p"]
    print(f"Peers      : {p2p['outbound']} outbound, {p2p['inbound']} inbound, "
          f"{p2p['known_addresses']} addresses known")
    if st.get("pruned"):
        print(f"Pruned     : block bodies from height {st['first_block']} on")
    if p2p["best_header_height"] > st["height"]:
        print(f"Syncing    : headers to {p2p['best_header_height']}, {p2p['blocks_in_flight']} blocks in flight")
    print(f"Node id    : {p2p['node_id']}")
    print(f"Mempool    : {st['mempool']} transactions")
    for w in st.get("warnings", []):
        print(f"WARNING    : {w}")


def cmd_balance(args):
    addr = args.address or wallet_address(args.data)
    if not is_address(addr):
        raise CliError("invalid address")
    acct = api_get(args, f"/account/{addr}")
    print(f"{addr}")
    print(f"  balance         : {params.fmt_amount(acct['balance'])}")
    if acct["pending_balance"] != acct["balance"]:
        print(f"  with pending    : {params.fmt_amount(acct['pending_balance'])}")


def cmd_send(args):
    key = require_wallet(args.data)
    if not is_address(args.to):
        raise CliError("invalid recipient address")
    amount = params.parse_amount(args.amount)
    tid = send_tx(args, key, "transfer", to=args.to, amount=amount)
    print(f"Sent: {params.fmt_amount(amount)} -> {args.to}\nTransaction: {tid}")


def cmd_mine(args):
    body = {"on": args.state == "on"}
    if args.intensity is not None:
        body["intensity"] = args.intensity
    if args.gpu is not None:
        body["gpu"] = args.gpu == "on"
    res = api_post(args, "/mining", body)
    print(f"Mining {'on' if res['mining'] else 'off'}, {res.get('intensity', 100)}% of the time, "
          f"on the {res['gpu'] or 'CPU'}")


def cmd_stop(args):
    api_post(args, "/shutdown", {})
    print(f"The node at {args.node} is stopping.")


def cmd_app(args):
    from . import app as appmod
    try:
        if args.localnet:
            accounts, buyer = appmod.localnet_accounts(args.localnet)
        else:
            key = require_wallet(args.data)
            accounts, buyer = [appmod.Account("me", "this wallet", key, args.node, args.data)], "me"
        a = appmod.App(accounts, buyer=buyer, port=args.port, allow_local=bool(args.localnet))
        a.start()
    except (appmod.AppError, OSError) as e:
        raise CliError(f"cannot start the app: {e}") from None
    print(f"Green app: {a.url}   (compute proxy: POST {a.url}/v1/matmul)")
    print("Ctrl+C to stop; open channels are closed on the way out.")
    if not args.no_browser:
        import webbrowser
        webbrowser.open(a.url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Closing channels...")
        a.stop()


def cmd_localnet(args):
    from . import localnet
    try:
        if args.action == "start":
            print(f"Local network in {args.dir}:")
            localnet.start(args.dir, gpu=not args.cpu, base_port=args.base_port, intensity=args.intensity,
                           units_per_block=args.units_per_block)
            print()
            localnet.print_status(localnet.status(args.dir), args.dir)
        elif args.action == "status":
            localnet.print_status(localnet.status(args.dir), args.dir)
        elif args.action == "stop":
            localnet.stop(args.dir)
        else:
            localnet.reset(args.dir)
    except localnet.LocalnetError as e:
        raise CliError(str(e)) from None


def cmd_job_submit(args):
    key = require_wallet(args.data)
    if args.a or args.b:
        if not (args.a and args.b):
            raise CliError("--a and --b must be given together")
        a, b = load_int8(args.a), load_int8(args.b)
        if a.shape[1] != b.shape[0]:
            raise CliError(f"shapes do not match: {a.shape} x {b.shape}")
    else:
        m, k, n = args.size
        rng = np.random.default_rng()
        a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
        b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
    shape = (a.shape[0], b.shape[1])
    pa, pb = pad_to_unit(a), pad_to_unit(b)
    blob = txmod.encode_job(pa, pb)
    dims = txmod.job_dims_units(pa, pb)
    if max(dims) > params.MAX_JOB_UNITS_PER_DIM:
        raise CliError(f"job too large: at most {params.MAX_JOB_UNITS_PER_DIM * N} per dimension")
    price = params.parse_amount(args.price)
    units = dims[0] * dims[1] * dims[2]

    job_id = api_post(args, "/blob", blob, raw=True)["hash"]
    if job_id != hhex(blob):
        raise CliError("the node stored corrupted data")
    tid = send_tx(args, key, "job_post", job=job_id, dims=dims, price=price)

    os.makedirs(os.path.join(args.data, "myjobs"), exist_ok=True)
    np.savez(os.path.join(args.data, "myjobs", job_id + ".npz"), a=a, b=b, shape=np.array(shape))
    print(f"Job posted: {job_id}")
    print(f"  {a.shape[0]}x{a.shape[1]} @ {b.shape[0]}x{b.shape[1]}  -> {units} work units ({dims})")
    print(f"  price per unit {params.fmt_amount(price)}, escrow {params.fmt_amount(price * units)}")
    print(f"  transaction {tid}")
    print("Your node checks every delivered result and disputes wrong ones automatically.")
    print(f"To fetch the result: python -m green --data {args.data} job get {job_id[:12]}")


def find_job(args, prefix: str) -> dict:
    jobs = api_get(args, "/jobs")
    matches = [j for j in jobs if j["id"].startswith(prefix)]
    if len(matches) != 1:
        raise CliError("job not found (it can take a block to get on chain)" if not matches
                       else "ambiguous id, type more of it")
    return matches[0]


def cmd_job_list(args):
    jobs = api_get(args, "/jobs")
    if not jobs:
        print("No jobs on chain.")
    for j in jobs:
        if j["closed"]:
            state = "closed"
        elif j["closes_at"] is not None:
            state = f"cancel #{j['closes_at']}"
        else:
            state = "done" if j["units_done"] == j["units_total"] else "open"
        print(f"{j['id'][:16]}  {state:12}  {j['units_done']}/{j['units_total']} units final, "
              f"{j['units_pending']} in challenge window, {j['units_reserved']} committed  "
              f"price {params.fmt_amount(j['price'])}/unit  buyer {j['buyer'][:12]}...")


def _local_inputs(args, job_id: str):
    """(padded A, padded B, original shape) if this job was submitted from this data folder."""
    local = os.path.join(args.data, "myjobs", job_id + ".npz")
    if not os.path.exists(local):
        return None
    saved = np.load(local)
    return pad_to_unit(saved["a"]), pad_to_unit(saved["b"]), tuple(saved["shape"])


def cmd_job_get(args):
    job = find_job(args, args.job)
    mu, ku, nu = job["dims"]
    print(f"Job {job['id'][:16]}: {job['units_done']}/{job['units_total']} units final, "
          f"{job['units_pending']} in the challenge window")
    if len(job["results"]) < job["units_total"]:
        withheld = sorted(k for k in job["pending"] if k not in job["results"])
        if withheld:
            print(f"{len(withheld)} delivered units have no reachable data yet: {', '.join(withheld[:8])}")
            print(f"If that persists, dispute them (the miner must publish them or lose the batch bond):\n"
                  f"  python -m green --data {args.data} job challenge {job['id'][:12]} {' '.join(withheld[:8])}")
        raise CliError("not finished yet. Try again a bit later.")
    inputs = _local_inputs(args, job["id"])
    c = np.zeros((mu * N, nu * N), dtype=np.int64)
    wrong = []
    for key, result_hash in sorted(job["results"].items()):
        i, l, j = (int(x) for x in key.split(","))
        data = api_get(args, f"/blob/{result_hash}", raw=True)
        if hhex(data) != result_hash:
            raise CliError("the node returned corrupted result data")
        unit_c = txmod.decode_result(data)
        if inputs is not None:
            ua, ub = txmod.unit_inputs(inputs[0], inputs[1], [i, l, j])
            if not work.freivalds_check(ua, ub, unit_c, os.urandom(32)):
                wrong.append(key)
        c[i * N:(i + 1) * N, j * N:(j + 1) * N] += unit_c
    if wrong:
        print(f"WRONG RESULT: {len(wrong)} units failed the Freivalds check: {', '.join(wrong[:8])}")
        open_ = [k for k in wrong if k in job["pending"]]
        if open_:
            print(f"To dispute them (the miner loses the bond of the whole batch if it cannot answer):\n"
                  f"  python -m green --data {args.data} job challenge {job['id'][:12]} {' '.join(open_[:8])}")
        raise CliError("result not saved")
    if inputs is not None:
        rows, cols = inputs[2]
        c = c[:rows, :cols]
        print("Local check: every unit verified with Freivalds (error probability <= 2^-124)")
    else:
        print("Warning: this job's inputs are not in this data folder, the result was not verified.")
    out = args.out or f"result_{job['id'][:12]}.npy"
    np.save(out, c)
    print(f"Result saved: {out}  ({c.shape[0]}x{c.shape[1]}, int64)")


def cmd_job_challenge(args):
    key = require_wallet(args.data)
    job = find_job(args, args.job)
    keys = [u.replace(" ", "") for u in args.units]
    batches = {job["pending"].get(k) for k in keys}
    if None in batches:
        raise CliError("a unit is not in a challenge window (already final, or not delivered yet)")
    if len(batches) != 1:
        raise CliError("these units belong to different batches; dispute each batch separately")
    batch = batches.pop()
    units = [[int(x) for x in k.split(",")] for k in keys]
    inputs = _local_inputs(args, job["id"])
    if inputs is not None and not args.force:
        # Check locally first: disputing a correct, available result only costs you the fee.
        for k, unit in zip(keys, units):
            data = None
            if k in job["results"]:
                try:
                    data = api_get(args, f"/blob/{job['results'][k]}", raw=True)
                except CliError:
                    pass                            # withheld: worth disputing
            if data is not None:
                ua, ub = txmod.unit_inputs(inputs[0], inputs[1], unit)
                if work.freivalds_check(ua, ub, txmod.decode_result(data), os.urandom(32)):
                    raise CliError(f"unit {k} is correct and available; disputing it would cost the fee "
                                   f"for nothing (use --force to send it anyway)")
    tid = send_tx(args, key, "job_challenge", job=job["id"], batch=batch,
                  units=txmod.units_bitmap(units, job["dims"]))
    print(f"Dispute sent for {len(units)} units (fee {params.fmt_amount(txmod.job_challenge_fee(len(units)))}). "
          f"If the miner cannot prove every one of them before its deadline, the batch's bond goes to you "
          f"and its units reopen.\nTransaction: {tid}")


def cmd_job_cancel(args):
    key = require_wallet(args.data)
    job = find_job(args, args.job)
    tid = send_tx(args, key, "job_cancel", job=job["id"])
    print(f"Cancel sent. It takes effect after {params.CANCEL_DELAY} blocks "
          f"(~{params.CANCEL_DELAY * params.TARGET_BLOCK_SECONDS // 60} min); batches committed before then "
          f"are settled as usual and the rest of the escrow is refunded.\nTransaction: {tid}")


def cmd_task_post(args):
    from . import wasm
    key = require_wallet(args.data)
    try:
        with open(args.program, "rb") as f:
            prog = f.read()
        with open(args.input, "rb") as f:
            inp = f.read()
    except OSError as e:
        raise CliError(str(e)) from None
    try:
        wasm.validate_module(prog)
    except wasm.ComputeError as e:
        raise CliError(f"not a valid Green WASM module: {e}") from None
    if len(inp) > params.MAX_TASK_INPUT_BYTES:
        raise CliError(f"the input can be at most {params.MAX_TASK_INPUT_BYTES} bytes")
    if not 1 <= args.fuel <= params.MAX_TASK_FUEL:
        raise CliError(f"fuel must be in 1..{params.MAX_TASK_FUEL}")
    price = params.parse_amount(args.price)
    prog_hash = api_post(args, "/blob", prog, raw=True)["hash"]
    input_hash = api_post(args, "/blob", inp, raw=True)["hash"]
    chain_id = api_get(args, "/status")["genesis"]
    nonce = api_get(args, f"/account/{key.address}")["pending_nonce"]
    fee = max(txmod.task_post_fee(len(prog), len(inp)), params.parse_amount(args.fee or "0"))
    t = make_tx(key, chain_id, "task_post", nonce, fee, prog=prog_hash, input=input_hash,
                fuel=args.fuel, price=price)
    api_post(args, "/tx", t)
    task_id = txmod.task_id_of(t)
    print(f"Task posted: {task_id}")
    print(f"  program {len(prog)} bytes, input {len(inp)} bytes, fuel {args.fuel:,}, reward {params.fmt_amount(price)}, "
          f"fee {params.fmt_amount(fee)}")
    print(f"To fetch the result: python -m green --data {args.data} task get {task_id[:12]}")


def cmd_task_list(args):
    tasks = api_get(args, "/tasks")
    if not tasks:
        print("No tasks on chain.")
    for t in tasks:
        print(f"{t['id'][:16]}  {t['status']:8}  fuel {t['fuel']}  price {params.fmt_amount(t['price'])}  "
              f"worker {(t['worker'] or '-')[:12]}")


def find_task(args, prefix: str) -> dict:
    tasks = api_get(args, "/tasks")
    matches = [t for t in tasks if t["id"].startswith(prefix)]
    if len(matches) != 1:
        raise CliError("task not found" if not matches else "ambiguous id, type more of it")
    return matches[0]


def cmd_task_challenge(args):
    from . import wasm
    key = require_wallet(args.data)
    t = find_task(args, args.task)
    if t["status"] != "revealed":
        raise CliError(f"no revealed result to dispute (status: {t['status']})")
    # Check locally first: a wrong challenge costs the bond.
    prog = api_get(args, f"/blob/{t['prog']}", raw=True)
    inp = api_get(args, f"/blob/{t['input']}", raw=True)
    if wasm.verify(prog, inp, t["output"], t["fuel"]) and not args.force:
        raise CliError("re-running it locally found the worker's result CORRECT; a challenge would cost your bond. "
                       "Use --force if you still want to")
    tid = send_tx(args, key, "task_challenge", required_fee=t["challenge_fee"], task=t["id"])
    print(f"Challenge sent (fee {params.fmt_amount(t['challenge_fee'])}). Transaction: {tid}")


def cmd_task_get(args):
    t = find_task(args, args.task)
    print(f"Task {t['id'][:16]}: status {t['status']}")
    if t["status"] not in ("revealed", "confirmed", "done"):
        raise CliError("the result has not been revealed yet")
    data = api_get(args, f"/blob/{t['output']}", raw=True)
    if hhex(data) != t["output"]:
        raise CliError("the node returned unverifiable result data (the worker may not have published it)")
    out = args.out or f"task_{t['id'][:12]}.out"
    with open(out, "wb") as f:
        f.write(data)
    print(f"Result saved: {out}  ({len(data)} bytes)")


def cmd_agent(args):
    from .agent import Agent, Goal
    from .nn import QuantMLP
    key = require_wallet(args.data)
    api_get(args, "/status")                       # fail early if no node is running
    mlp = QuantMLP.random([256, 512, 256, 10], seed=7)
    rng = np.random.default_rng()
    goals = [Goal(f"inference-{i}", mlp, rng.integers(-127, 128, size=(256, args.samples), dtype=np.int8))
             for i in range(args.goals)]
    agent = Agent(args.node, key.secret_hex(), reserve=params.parse_amount(args.reserve),
                  price=params.parse_amount(args.price), log=print)
    print(f"Agent {key.address}: {args.goals} goals, reserve {args.reserve} GRN")
    print("(To earn, a node mining with this wallet must be running.)")
    done = agent.run(goals, poll=2.0, max_wait=args.max_wait)
    print(f"{len(done)}/{args.goals} goals completed.")


def cmd_supply(args):
    s = api_get(args, "/supply")
    print(f"Height   : {s['height']}")
    print(f"Balances : {params.fmt_amount(s['balances'])}")
    print(f"Locked   : {params.fmt_amount(s['locked'])}  (escrow, bonds and channels)")
    print(f"Reserve  : {params.fmt_amount(s['reserve'])}  (not yet paid out as rewards, and fees flowing back)")
    print(f"Total    : {params.fmt_amount(s['total'])}")
    print(f"Burned   : {params.fmt_amount(s.get('burned', 0))}  ({params.FEE_BURN_PERCENT}% of every fee is destroyed)")
    print(f"Supply   : {params.fmt_amount(s['supply'])}  (never grows: the fixed total minus what was burned)")
    if not s["ok"]:
        raise CliError("SUPPLY MISMATCH: coins were created from nothing, or coins are missing")
    print("OK: every coin is accounted for, and no coin was ever created")


def cmd_guard(args):
    key = require_wallet(args.data)
    if args.guard_cmd == "set":
        tid = send_tx(args, key, "guard_set", guardian=args.guardian, limit=params.parse_amount(args.limit),
                      period=args.period)
        print(f"Your account now answers to {args.guardian}: at most {args.limit} GRN per {args.period} blocks. "
              f"Only the guardian can loosen or remove this.\nTransaction: {tid}")
        return
    current = api_get(args, f"/account/{args.account}")["guard"]
    if current is None:
        raise CliError("that account has no guardian")
    if args.guard_cmd == "release":
        tid = send_tx(args, key, "guard_release", account=args.account)
        print(f"Guard removed from {args.account}.\nTransaction: {tid}")
        return
    frozen = current["frozen"] if args.freeze is None else args.freeze
    tid = send_tx(args, key, "guard_update", account=args.account,
                  guardian=args.new_guardian or current["guardian"],
                  limit=params.parse_amount(args.limit) if args.limit else current["limit"],
                  period=args.period or current["period"], frozen=frozen)
    print(f"Guard of {args.account} updated{' (FROZEN)' if frozen else ''}.\nTransaction: {tid}")


def cmd_guard_show(args):
    addr = args.address or wallet_address(args.data)
    g = api_get(args, f"/account/{addr}")["guard"]
    if g is None:
        print(f"{addr} has no guardian.")
        return
    print(f"{addr}")
    print(f"  guardian   : {g['guardian']}")
    print(f"  limit      : {params.fmt_amount(g['limit'])} per {g['period']} blocks")
    print(f"  left now   : {params.fmt_amount(g['spend_left'])}")
    print(f"  frozen     : {'YES' if g['frozen'] else 'no'}")


def cmd_release(args):
    from . import release
    if args.release_cmd == "sums":
        sys.stdout.write(release.make_sums(args.files))
    elif args.release_cmd == "sign":
        key = require_wallet(args.data)
        with open(args.sums, "rb") as f:
            sig = release.sign(key, f.read())
        out = args.out or f"{args.sums}.{key.pub_hex[:16]}.sig"
        with open(out, "w") as f:
            f.write(sig)
        print(f"Signed with {key.pub_hex} -> {out}")
    else:
        keys = [k.strip() for k in args.keys.split(",")] if args.keys else list(params.RELEASE_KEYS)
        threshold = args.threshold if args.threshold is not None else params.RELEASE_THRESHOLD
        if not keys:
            raise CliError("no maintainer keys: this network has none yet; pass --keys")
        with open(args.sums, "rb") as f:
            sums = f.read()
        sigs = []
        for path in args.sigs:
            with open(path) as f:
                sigs.append(f.read())
        try:
            r = release.verify_release(sums, sigs, keys, threshold, folder=os.path.dirname(os.path.abspath(args.sums)))
        except ValueError as e:
            raise CliError(str(e)) from None
        print(f"Valid maintainer signatures: {len(r['signers'])} of {threshold} needed")
        for name, status in sorted(r["files"].items()):
            print(f"  {status:<8} {name}")
        if not r["ok"]:
            raise CliError("NOT a genuine release: do not install it")
        print("Genuine release.")


def cmd_deployments(args):
    rows = api_get(args, "/deployments")
    if not rows:
        print("No rule changes are scheduled.")
    for d in rows:
        line = f"{d['name']:<16} bit {d['bit']:<2} {d['state']:<10} start {d['start']}, timeout {d['timeout']}"
        if d["min_activation"]:
            line += f", not active before {d['min_activation']}"
        if d["lock_in_on_timeout"]:
            line += ", mandatory at timeout"
        if d["signalled"] is not None:
            line += f" | this period: {d['signalled']} of {d['threshold']} needed ({d['period']}-block periods)"
        print(line)
    for w in api_get(args, "/status").get("warnings", []):
        print(f"WARNING: {w}")


def cmd_peers(args):
    if args.add:
        if not api_post(args, "/peers", {"addr": args.add})["added"]:
            raise CliError("not a peer address (host:port or <node id>@host:port)")
        print(f"Connecting to {args.add}")
    peers = api_get(args, "/peers")
    if not peers:
        print("No peers connected.")
    for p in peers:
        ping = f"{p['ping_ms']:.0f} ms" if p["ping_ms"] is not None else "-"
        kind = "in " if p["inbound"] else "out"
        relay = " blocks-only" if p["block_only"] else ""
        print(f"{kind} {p['listen'] or p['addr']:<22} id {p['id'][:16]}  height {p['height']:<7} ping {ping:<7} "
              f"session {p['session']}{relay}")


# --- parser --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    prog = "grn" if os.path.basename(sys.argv[0]).lower().startswith("grn") else "python -m green"
    p = argparse.ArgumentParser(prog=prog,
                                description=f"{params.COIN_NAME} ({params.TICKER}): a blockchain secured by proof of useful work")
    p.add_argument("--data", default="data", help="folder for the wallet, blocks and data (default: data)")
    p.add_argument("--node", default=DEFAULT_NODE, help=f"node to talk to (default: {DEFAULT_NODE})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("wallet", help="create, restore or show the wallet")
    s.add_argument("action", choices=["new", "restore", "show", "phrase"],
                   help="new: a wallet with a recovery phrase; restore: from a phrase (read from the terminal or "
                        "stdin); show: address and public key; phrase: show the recovery phrase")
    s.add_argument("--no-password", action="store_true", help="unencrypted wallet (testing only)")
    s.add_argument("--words", type=int, default=24, choices=hd.WORD_COUNTS, help="phrase length for new (default 24)")
    s.add_argument("--account", type=int, default=0, help="account number on the phrase (default 0)")
    s.add_argument("--passphrase", action="store_true", help="add a passphrase to the phrase (asked for)")
    s.set_defaults(fn=cmd_wallet)

    s = sub.add_parser("genesis", help="start a new network: mine the genesis block")
    s.add_argument("--msg", required=True, help="message written into the genesis block")
    s.add_argument("--checkpoints", action="store_true",
                   help="this folder's checkpoint key signs which blocks are final while the network is young "
                        "(grn checkpoint key makes one)")
    s.add_argument("--checkpoint-until", type=int, metavar="HEIGHT",
                   help="last block a checkpoint may name (default: a year of blocks)")
    s.add_argument("--units-per-block", type=int, metavar="N",
                   help="starting difficulty: about N work units a block (the miners' units/s times 15); "
                        "the genesis block itself takes this much work to find")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_genesis)

    s = sub.add_parser("node", help="run a node")
    s.add_argument("--port", type=int, default=params.DEFAULT_PORT)
    s.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    s.add_argument("--p2p-port", type=int, default=params.DEFAULT_P2P_PORT, help="port for other nodes (P2P)")
    s.add_argument("--p2p-host", help="address the P2P port listens on (default: same as --host); "
                                      "e.g. 0.0.0.0 to be reachable by other nodes while the API stays local")
    s.add_argument("--no-listen", action="store_true", help="do not accept connections from other nodes")
    s.add_argument("--checkpoints", action="store_true",
                   help="sign this network's checkpoints with the folder's checkpoint key (the founder's node)")
    s.add_argument("--portmap", action="store_true",
                   help="ask the home router to forward the P2P port (and the --provide-port) with PCP or "
                        "NAT-PMP, so other nodes and buyers can reach this one")
    s.add_argument("--no-dandelion", action="store_true",
                   help="announce this node's transactions at once instead of passing them along a stem first "
                        "(faster, but shows the network where they came from)")
    s.add_argument("--no-signal", action="append", default=[], metavar="DEPLOYMENT",
                   help="do not signal readiness for this rule change when mining (see: deployments)")
    s.add_argument("--fast-sync", action="store_true",
                   help="new node: start from a ledger snapshot verified against the header chain, "
                        "instead of replaying all of history")
    s.add_argument("--peer", action="append", default=[],
                   help="a node to stay connected to: host:port, or <node id>@host:port to also check its key")
    s.add_argument("--genesis", help="genesis.json of an existing network")
    s.add_argument("--mine", action="store_true", help="start mining")
    s.add_argument("--gpu", action="store_true", help="mine on the GPU (WebGPU/Vulkan)")
    s.add_argument("--intensity", type=int, default=100, metavar="PERCENT",
                   help="mine this percent of the time and let the GPU/CPU rest the rest (default 100)")
    s.add_argument("--report", type=float, default=15, help="seconds between status lines")
    s.add_argument("--provide-port", type=int, help="also sell matrix products through channels on this port")
    s.add_argument("--provide-host", default="127.0.0.1", help="address the seller listens on (0.0.0.0: everyone)")
    s.add_argument("--provide-url", help="public URL buyers reach it at, e.g. http://203.0.113.7:8548 "
                                         "(advertised over the network)")
    s.add_argument("--provide-price", type=int, default=1000, help="yaprak per 2^24 multiply-adds (default 1000)")
    s.add_argument("--pool-port", type=int, help="also run a non-custodial mining pool on this port")
    s.add_argument("--pool-host", default="127.0.0.1", help="address the pool listens on (0.0.0.0 for remote miners)")
    s.add_argument("--pool-fee", type=int, default=1, help="the pool's percentage of each block (default 1)")
    s.add_argument("--prune", action="store_true",
                   help="keep only recent block bodies and the data still in use (OPERATIONS.md)")
    s.add_argument("--prune-keep", type=int, default=5760, help="blocks of bodies a pruned node keeps")
    s.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    s.add_argument("--log-file", help="also log to this file (rotated at 10 MB, 5 kept)")
    s.set_defaults(fn=cmd_node)
    p.node_parser = s

    s = sub.add_parser("pool-mine", help="mine for a pool (paid in the blocks themselves)")
    s.add_argument("--pool", required=True, help="the pool's URL, e.g. http://203.0.113.7:8547")
    s.add_argument("--address", help="payout address (default: this wallet)")
    s.add_argument("--gpu", action="store_true")
    s.set_defaults(fn=cmd_pool_mine)

    s = sub.add_parser("tx", help="look up a transaction")
    s.add_argument("txid")
    s.set_defaults(fn=cmd_tx)

    s = sub.add_parser("history", help="what an address took part in")
    s.add_argument("address", nargs="?")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_history)

    s = sub.add_parser("bump", help="pay a higher fee for a pending transaction of this wallet")
    s.add_argument("txid")
    s.add_argument("--fee", help="the new fee in GRN (default: what the next block needs, at least +10%%)")
    s.set_defaults(fn=cmd_bump)

    s = sub.add_parser("providers", help="compute on offer: nodes selling matrix products through channels")
    s.set_defaults(fn=cmd_providers)

    s = sub.add_parser("fee", help="fee estimate for the next blocks")
    s.add_argument("--blocks", type=int, default=2)
    s.set_defaults(fn=cmd_fee)

    s = sub.add_parser("status", help="node status")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("balance", help="balance")
    s.add_argument("address", nargs="?")
    s.set_defaults(fn=cmd_balance)

    s = sub.add_parser("send", help="send coins")
    s.add_argument("to")
    s.add_argument("amount", help="in GRN, e.g. 1.5")
    s.add_argument("--fee", default="0")
    s.set_defaults(fn=cmd_send)

    s = sub.add_parser("mine", help="turn mining on/off on a running node")
    s.add_argument("state", choices=["on", "off"])
    s.add_argument("--intensity", type=int, metavar="PERCENT", help="mine this percent of the time (1..100)")
    s.add_argument("--gpu", choices=["on", "off"], help="mine on the GPU or on the CPU")
    s.set_defaults(fn=cmd_mine)

    s = sub.add_parser("stop", help="stop a running node cleanly (it collects what buyers owe it first)")
    s.set_defaults(fn=cmd_stop)

    s = sub.add_parser("app", help="the app: mine, send coins, buy compute, and a compute proxy (in a browser)")
    s.add_argument("--localnet", nargs="?", const="localnet", metavar="DIR",
                   help="use the five people of a local network (default folder: localnet)")
    s.add_argument("--port", type=int, default=8600, help="port of the app on this computer (default 8600)")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser")
    s.set_defaults(fn=cmd_app)

    s = sub.add_parser("localnet", help="a network of five people on this computer, to try everything out")
    s.add_argument("action", choices=["start", "status", "stop", "reset"],
                   help="start: create it the first time and start every node; status: who is who; "
                        "stop: stop every node cleanly; reset: stop and delete it")
    s.add_argument("--dir", default="localnet", help="folder of the local network (default: localnet)")
    s.add_argument("--cpu", action="store_true", help="the miners mine on the CPU (default: the GPU, if there is one)")
    s.add_argument("--intensity", type=int, default=50, metavar="PERCENT",
                   help="each miner mines this percent of the time (default 50: three miners share one GPU)")
    s.add_argument("--base-port", type=int, default=18500,
                   help="ports used: base+1.. (API), base+101.. (P2P), base+201.. (compute); default 18500")
    s.add_argument("--units-per-block", type=int, default=10_000, metavar="N",
                   help="starting difficulty of a new network, in work units a block (default 10000)")
    s.set_defaults(fn=cmd_localnet)

    ms = sub.add_parser("multisig", help="k-of-n accounts").add_subparsers(dest="ms_cmd", required=True)
    s = ms.add_parser("address", help="the address of a k-of-n account")
    s.add_argument("k", type=int)
    s.add_argument("keys", nargs="+", help="the public keys (wallet show)")
    s.set_defaults(fn=cmd_multisig)
    s = ms.add_parser("new", help="write an unsigned transaction from a k-of-n account")
    s.add_argument("k", type=int)
    s.add_argument("keys", nargs="+")
    s.add_argument("--to", help="transfer: recipient")
    s.add_argument("--amount", help="transfer: amount in GRN")
    s.add_argument("--type", default="transfer", help="any transaction type (with --fields)")
    s.add_argument("--fields", help="the type's fields as JSON")
    s.add_argument("--fee", help="fee in GRN (default: the minimum)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_multisig)
    s = ms.add_parser("sign", help="add this wallet's signature to a transaction file")
    s.add_argument("file")
    s.set_defaults(fn=cmd_multisig)
    s = ms.add_parser("send", help="submit a fully signed transaction file")
    s.add_argument("file")
    s.set_defaults(fn=cmd_multisig)

    rel = sub.add_parser("release", help="sign and verify releases").add_subparsers(dest="release_cmd", required=True)
    s = rel.add_parser("sums", help="print SHA256SUMS for files")
    s.add_argument("files", nargs="+")
    s.set_defaults(fn=cmd_release)
    s = rel.add_parser("sign", help="sign a SHA256SUMS file with your wallet key")
    s.add_argument("sums")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_release)
    s = rel.add_parser("verify", help="check a release: maintainer signatures and every file's digest")
    s.add_argument("sums")
    s.add_argument("sigs", nargs="+")
    s.add_argument("--keys", help="comma-separated maintainer public keys (default: the network's)")
    s.add_argument("--threshold", type=int, help="signatures needed (default: the network's)")
    s.set_defaults(fn=cmd_release)

    s = sub.add_parser("deployments", help="scheduled rule changes and how far their signalling is")
    s.set_defaults(fn=cmd_deployments)

    s = sub.add_parser("peers", help="list / add peers")
    s.add_argument("--add", help="host:port or <node id>@host:port")
    s.set_defaults(fn=cmd_peers)

    job = sub.add_parser("job", help="buy compute: matrix multiplication").add_subparsers(dest="job_cmd", required=True)
    s = job.add_parser("submit", help="post a matrix multiplication job (A @ B)")
    s.add_argument("--a", help="matrix A (.npy, int8)")
    s.add_argument("--b", help="matrix B (.npy, int8)")
    s.add_argument("--size", type=int, nargs=3, metavar=("M", "K", "N"), default=[512, 512, 512],
                   help="size of random demo matrices when no files are given")
    s.add_argument("--price", default="0.01", help="GRN per work unit")
    s.add_argument("--fee", default="0")
    s.set_defaults(fn=cmd_job_submit)
    s = job.add_parser("list", help="jobs on chain")
    s.set_defaults(fn=cmd_job_list)
    s = job.add_parser("get", help="download a finished job's result and verify it")
    s.add_argument("job")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_job_get)
    s = job.add_parser("challenge", help="dispute wrong or withheld results (checks locally first)")
    s.add_argument("job")
    s.add_argument("units", nargs="+", help="units of one batch, e.g. 0,1,0 0,1,1")
    s.add_argument("--force", action="store_true", help="dispute even if the local check finds it correct")
    s.add_argument("--fee", default="0")
    s.set_defaults(fn=cmd_job_challenge)
    s = job.add_parser("cancel", help="cancel the job, get the remaining escrow back")
    s.add_argument("job")
    s.add_argument("--fee", default="0")
    s.set_defaults(fn=cmd_job_cancel)

    s = sub.add_parser("checkpoint", help="the founder's checkpoints: make the key, see them, end them")
    s.add_argument("action", choices=["key", "status", "end"],
                   help="key: make this folder's checkpoint key; status: the network's checkpoints; "
                        "end: end the checkpoint period for good (with --yes)")
    s.add_argument("--no-password", action="store_true", help="key: store it unencrypted (testing only)")
    s.add_argument("--yes", action="store_true", help="end: really end the checkpoint period")
    s.set_defaults(fn=cmd_checkpoint)

    s = sub.add_parser("supply", help="audit that every coin is accounted for")
    s.set_defaults(fn=cmd_supply)

    guard = sub.add_parser("guard", help="guardians for autonomous agents: spending limits and a freeze switch")
    gsub = guard.add_subparsers(dest="guard_cmd", required=True)
    s = gsub.add_parser("set", help="put your account under a guardian (you can only tighten it later)")
    s.add_argument("guardian")
    s.add_argument("limit", help="GRN per period")
    s.add_argument("period", type=int, help="period in blocks")
    s.add_argument("--fee", default="0")
    s.set_defaults(fn=cmd_guard)
    s = gsub.add_parser("update", help="as the guardian: change the limit, freeze or unfreeze, hand over")
    s.add_argument("account")
    s.add_argument("--limit", help="GRN per period")
    s.add_argument("--period", type=int)
    s.add_argument("--freeze", dest="freeze", action="store_true", default=None)
    s.add_argument("--unfreeze", dest="freeze", action="store_false")
    s.add_argument("--new-guardian")
    s.add_argument("--fee", default="0")
    s.set_defaults(fn=cmd_guard)
    s = gsub.add_parser("release", help="as the guardian: remove the guard")
    s.add_argument("account")
    s.add_argument("--fee", default="0")
    s.set_defaults(fn=cmd_guard)
    s = gsub.add_parser("show", help="show an account's guard")
    s.add_argument("address", nargs="?")
    s.set_defaults(fn=cmd_guard_show)

    s = sub.add_parser("agent", help="autonomous agent: spends what it earns on its own inference")
    s.add_argument("--goals", type=int, default=3)
    s.add_argument("--samples", type=int, default=4)
    s.add_argument("--reserve", default="2", help="balance it never goes below (GRN)")
    s.add_argument("--price", default="0.01")
    s.add_argument("--max-wait", type=float, default=300, help="longest wait for funds (s)")
    s.set_defaults(fn=cmd_agent)

    task = sub.add_parser("task", help="buy general compute (WASM)").add_subparsers(dest="task_cmd", required=True)
    s = task.add_parser("post", help="post a WASM program + input to be run")
    s.add_argument("program", help="deterministic WASM module (.wasm) exporting 'green_run' and 'memory'")
    s.add_argument("input", help="input file (raw bytes)")
    s.add_argument("--fuel", type=int, default=100_000_000, help="fuel budget (bound on execution steps)")
    s.add_argument("--price", default="0.01", help="reward in GRN")
    s.add_argument("--fee", default="0")
    s.set_defaults(fn=cmd_task_post)
    s = task.add_parser("list", help="tasks on chain")
    s.set_defaults(fn=cmd_task_list)
    s = task.add_parser("get", help="download a finished task's result")
    s.add_argument("task")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_task_get)
    s = task.add_parser("challenge", help="dispute a wrong result (checks locally first)")
    s.add_argument("task")
    s.add_argument("--force", action="store_true", help="dispute even if the local re-run finds it correct")
    s.add_argument("--fee", default="0")
    s.set_defaults(fn=cmd_task_challenge)
    return p


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "node":
            # <data>/green.toml supplies defaults; whatever the command line says wins.
            cfg = load_config(args.data)
            if cfg:
                parser.node_parser.set_defaults(**cfg)
                args = parser.parse_args(argv)
        args.fn(args)
    except CliError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0
