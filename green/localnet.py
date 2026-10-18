"""A whole network on one computer: five people, each running their own node.

    grn localnet start [--dir localnet] [--cpu]   create it the first time, start everyone
    grn localnet status                           who is who: height, peers, mining, balance, buying
    grn localnet stop                             stop everyone cleanly
    grn localnet reset                            stop, then delete the network

Three miners (alice, bob, dave) mine on the GPU (on the CPU without one, or with --cpu)
and sell compute; two buyers (carol, erin) keep asking the network for compute: each
runs a buyer process next to its node that trains small models and buys products
through payment channels, and posts matrix jobs to the on-chain market.

Each persona's node is a separate process with its own wallet, data folder, ports and
log file, talking to the others over the real encrypted P2P protocol, on a network of
its own (its own genesis block). Nothing is special-cased: these are the nodes anyone
would run, only every address is 127.0.0.1. The wallets are stored unencrypted -- this
network is for trying things out, and its coins are worth nothing.

`grn app --localnet` opens the app on it: mine, send coins, buy compute.
"""

import faulthandler
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import params
from .chain import block_hash, make_genesis
from .node import http_get, http_post

PERSONAS = [
    {"name": "alice", "role": "miner, sells compute", "mine": True, "price": 1000, "buys": False},
    {"name": "bob", "role": "miner, sells compute (dearer)", "mine": True, "price": 1500, "buys": False},
    {"name": "carol", "role": "AI researcher: trains models on bought compute", "mine": False, "price": None,
     "buys": True},
    {"name": "dave", "role": "miner, sells compute", "mine": True, "price": 1200, "buys": False},
    {"name": "erin", "role": "startup: buys compute, posts jobs", "mine": False, "price": None, "buys": True},
]
BASE_PORT = 18500           # persona i: API BASE+1+i, P2P BASE+101+i, compute BASE+201+i
START_SECONDS = 90          # how long `start` waits for every node to answer
STOP_SECONDS = 60           # how long `stop` waits for every node to exit


class LocalnetError(Exception):
    pass


def network_file(root: str) -> str:
    return os.path.join(root, "network.json")


def _pids_file(root: str) -> str:
    return os.path.join(root, "pids.json")


def _alive(pid: int) -> bool:
    """Whether a process with this id exists (only ever used to wait, never to kill)."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)              # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259                # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def load(root: str) -> dict | None:
    try:
        with open(network_file(root), encoding="utf-8") as f:
            net = json.load(f)
    except FileNotFoundError:
        return None
    for p in net["personas"]:              # each persona lives in the network's folder,
        p["data"] = os.path.join(os.path.abspath(root), p["name"])  # wherever we run from
    return net


def create(root: str, base_port: int = BASE_PORT, log=print) -> dict:
    """Wallets, a genesis block and the plan of the network. Carol mines the genesis
    block, so she starts with its reward and can buy compute at once."""
    from .cli import create_wallet
    if load(root) is not None:
        raise LocalnetError(f"{root} already holds a network (grn localnet reset deletes it)")
    os.makedirs(root, exist_ok=True)
    people = []
    for i, p in enumerate(PERSONAS):
        data = os.path.join(root, p["name"])
        key = create_wallet(data, None)
        people.append({**p, "address": key.address, "data": data, "api": base_port + 1 + i,
                       "p2p": base_port + 101 + i, "compute": base_port + 201 + i if p["price"] else None})
    carol = next(p for p in people if p["name"] == "carol")
    log(f"Mining the genesis block of a new local network (reward to carol, {carol['address'][:16]}...)")
    genesis = make_genesis(carol["address"], f"Green local network, {time.strftime('%Y-%m-%d %H:%M')}")
    with open(os.path.join(root, "genesis.json"), "w", encoding="utf-8") as f:
        json.dump(genesis, f, indent=2)
    net = {"genesis": block_hash(genesis), "created": int(time.time()), "personas": people}
    with open(network_file(root), "w", encoding="utf-8") as f:
        json.dump(net, f, indent=2)
    return net


def url(p: dict) -> str:
    return f"http://127.0.0.1:{p['api']}"


def _listening(port: int) -> bool:
    """Whether anything accepts connections on this local port. Quick: on Windows a
    refused connection to a local port otherwise takes about two seconds."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def node_status(net: dict, p: dict, timeout: float = 2.0) -> dict | None:
    """The persona's node status, or None if no node of this network answers there."""
    if not _listening(p["api"]):
        return None
    try:
        st = http_get(f"{url(p)}/status", timeout=timeout)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return st if st.get("genesis") == net["genesis"] else None


def _command(root: str, net: dict, p: dict, gpu: bool, intensity: int = 100) -> list[str]:
    """The node of persona `p` (mining on the GPU if `gpu`: a node without a usable GPU
    says so in its log and mines on the CPU)."""
    cmd = [sys.executable, "-m", "green", "--data", os.path.abspath(p["data"]), "node",
           "--port", str(p["api"]), "--p2p-port", str(p["p2p"]),
           "--genesis", os.path.abspath(os.path.join(root, "genesis.json")), "--report", "30"]
    for other in net["personas"]:
        if other is not p:
            cmd += ["--peer", f"127.0.0.1:{other['p2p']}"]
    if p["mine"]:
        cmd += ["--mine", "--intensity", str(intensity)]
    if gpu:
        cmd.append("--gpu")
    if p["compute"]:
        cmd += ["--provide-port", str(p["compute"]), "--provide-url", f"http://127.0.0.1:{p['compute']}",
                "--provide-price", str(p["price"])]
    return cmd


def _spawn(cmd: list[str], log_path: str) -> subprocess.Popen:
    """Start a node that keeps running after this command returns."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([package_parent] + [x for x in [env.get("PYTHONPATH")] if x])
    out = open(log_path, "ab")
    kw = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": subprocess.STDOUT, "env": env}
    try:
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000          # CREATE_NO_WINDOW
            try:                                                              # out of a job that would end it with us
                return subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kw)   # CREATE_BREAKAWAY_FROM_JOB
            except OSError:
                return subprocess.Popen(cmd, creationflags=flags, **kw)
        return subprocess.Popen(cmd, start_new_session=True, **kw)
    finally:
        out.close()


def start(root: str, gpu: bool = True, base_port: int = BASE_PORT, log=print, intensity: int = 50) -> dict:
    net = load(root)
    if net is None:
        taken = [base_port + 1 + i for i in range(len(PERSONAS)) if _answers({"api": base_port + 1 + i})]
        if taken:
            raise LocalnetError(f"port {taken[0]} is another network's node (another localnet folder?): stop it, "
                                f"or use --base-port for a network of your own")
        net = create(root, base_port, log)
    started = {}
    for p in net["personas"]:
        if node_status(net, p) is not None:
            log(f"  {p['name']:6s} already running at {url(p)}")
            continue
        if _answers(p):
            raise LocalnetError(f"{url(p)} is another network's node (another localnet folder?): stop it, or "
                                f"use --base-port for a network of your own")
        proc = _spawn(_command(root, net, p, gpu and p["mine"], intensity), os.path.join(p["data"], "node.log"))
        started[p["name"]] = proc
        log(f"  {p['name']:6s} starting ({p['role']}), API {url(p)}")
    end = time.time() + START_SECONDS
    waiting = set(started)
    while waiting and time.time() < end:
        for name in list(waiting):
            p = next(x for x in net["personas"] if x["name"] == name)
            if node_status(net, p) is not None:
                waiting.discard(name)
            elif started[name].poll() is not None:
                raise LocalnetError(f"{name}'s node exited at start; see {os.path.join(p['data'], 'node.log')}")
        time.sleep(0.5)
    if waiting:
        for name in waiting:                     # only processes started by this very call
            started[name].kill()
        raise LocalnetError(f"no answer from {', '.join(sorted(waiting))} after {START_SECONDS} s; see their node.log")
    pids = _read_pids(root)
    pids.update({name: _real_pid(next(p for p in net["personas"] if p["name"] == name), proc.pid)
                 for name, proc in started.items()})
    for p in net["personas"]:                    # the buyers' processes, next to their nodes
        key = f"{p['name']}-buyer"
        if p.get("buys") and not (key in pids and _alive(pids[key])):
            proc = _spawn([sys.executable, "-m", "green.localnet", "buyer", os.path.abspath(root), p["name"]],
                          os.path.join(p["data"], "buyer.log"))
            pids[key] = proc.pid
            log(f"  {p['name']:6s} starts asking the network for compute (log: {os.path.join(p['data'], 'buyer.log')})")
    with open(_pids_file(root), "w", encoding="utf-8") as f:
        json.dump(pids, f)
    return net


def _real_pid(p: dict, fallback: int) -> int:
    """The node's own process id (on Windows a virtual environment's python.exe is a
    launcher that runs the real interpreter as a child: waiting on the launcher, or
    stopping it, leaves the node running)."""
    req = urllib.request.Request(f"{url(p)}/debug/process", headers={"X-Green-Client": "1"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return int(json.loads(r.read())["pid"])
    except (OSError, ValueError, KeyError):
        return fallback


def _read_pids(root: str) -> dict:
    try:
        with open(_pids_file(root), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def status(root: str) -> list[dict]:
    net = load(root)
    if net is None:
        raise LocalnetError(f"no local network in {root}: grn localnet start creates one")
    rows = []
    for p in net["personas"]:
        st = node_status(net, p)
        row = {"name": p["name"], "role": p["role"], "address": p["address"], "url": url(p), "running": st is not None}
        if st is not None:
            try:
                acct = http_get(f"{url(p)}/account/{p['address']}", timeout=2)
            except (OSError, ValueError):
                acct = {}
            row.update(height=st["height"], peers=len(st["peers"]), mining=st["mining"],
                       units_per_second=st["units_per_second"], balance=acct.get("balance"))
        if p.get("buys"):
            try:
                with open(os.path.join(p["data"], "buyer.json"), encoding="utf-8") as f:
                    row["buying"] = json.load(f)
            except (FileNotFoundError, ValueError):
                row["buying"] = None
        rows.append(row)
    return rows


def stop(root: str, log=print) -> None:
    """Ask every node to stop (each collects what buyers owe it first) and wait for them.
    Processes are never killed by a remembered PID: it could belong to something else by now."""
    net = load(root)
    if net is None:
        return
    asked = []
    for p in net["personas"]:
        if node_status(net, p) is None:
            continue
        try:
            http_post(f"{url(p)}/shutdown", {}, timeout=5)
            asked.append(p)
            log(f"  {p['name']:6s} stopping")
        except (OSError, ValueError) as e:
            log(f"  {p['name']:6s} did not take the stop request: {e}")
    pids = _read_pids(root)
    end = time.time() + STOP_SECONDS
    while asked and time.time() < end:           # until each has exited, not just closed its API
        asked = [p for p in asked if _answers(p) or (p["name"] in pids and _alive(pids[p["name"]]))]
        time.sleep(0.5)
    for p in asked:
        log(f"  {p['name']:6s} still running after {STOP_SECONDS} s; stop it by hand (see {p['data']}/node.log)")
    buyers = [k for k in pids if k.endswith("-buyer")]
    while buyers and time.time() < end:          # a buyer ends once its node is gone (BUYER_GRACE)
        buyers = [k for k in buyers if _alive(pids[k])]
        time.sleep(0.5)
    if not asked and not buyers and os.path.exists(_pids_file(root)):
        os.remove(_pids_file(root))


def _answers(p: dict) -> bool:
    if not _listening(p["api"]):
        return False
    try:
        http_get(f"{url(p)}/status", timeout=1)
        return True
    except (OSError, ValueError):
        return False


def reset(root: str, log=print) -> None:
    """Stop the network and delete its folder -- only a folder that holds a local network."""
    net = load(root)
    if net is None:
        if os.path.exists(root):
            raise LocalnetError(f"{root} does not hold a local network (no network.json); not deleting it")
        return
    stop(root, log)
    if any(_answers(p) for p in net["personas"]):
        raise LocalnetError("some nodes are still running; not deleting their data")
    shutil.rmtree(root)
    log(f"Deleted {root}")


def print_status(rows: list[dict], root: str) -> None:
    print(f"{'persona':8s} {'role':38s} {'height':>6s} {'peers':>5s} {'mining':>8s} {'balance':>16s}  node")
    for r in rows:
        if not r["running"]:
            print(f"{r['name']:8s} {r['role']:38s} {'-':>6s} {'-':>5s} {'-':>8s} {'(stopped)':>16s}  {r['url']}")
            continue
        bal = params.fmt_amount(r["balance"]) if r["balance"] is not None else "?"
        mining = f"{r['units_per_second']:.0f}/s" if r["mining"] else "off"
        print(f"{r['name']:8s} {r['role']:38s} {r['height']:>6d} {r['peers']:>5d} "
              f"{mining:>8s} {bal:>16s}  {r['url']}")
    buyers = [r for r in rows if "buying" in r]
    if buyers:
        print("\nCompute bought:")
        for r in buyers:
            b = r["buying"]
            print(f"  {r['name']:6s} " + ("nothing yet" if not b else
                  f"{b['products']} products, {b['trainings']} models trained, {b['jobs']} jobs posted, "
                  f"paid {params.fmt_amount(b['paid'])}" + (f"; last: {b['last']}" if b.get("last") else "")))
    print(f"\nLogs: {os.path.join(root, '<persona>', 'node.log')} and buyer.log.  App: grn app --localnet {root}")


# --- the buyers ------------------------------------------------------------------------

BUYER_GRACE = 15            # a buyer process ends once its node has been gone this long
BUYER_STUCK = 300           # ...and reports where it is if one purchase takes longer than this
LOW_BALANCE = params.YAPRAK_PER_COIN // 2      # below this a buyer buys coins from a miner...
COINS_BOUGHT = 3 * params.YAPRAK_PER_COIN      # ...this many
JOB_PRICE = params.YAPRAK_PER_COIN // 500      # per unit, for jobs posted on chain


def run_buyer(root: str, name: str) -> None:
    """A compute buyer's life (its own process, started by `start`). When its balance
    runs low it buys coins from the richest miner (on this test network, a transfer the
    miner signs). Otherwise, every 10-30 s, it asks the network for compute: trains a
    small model (every product bought from the miners through payment channels and
    checked), buys a few products, or posts a matrix job to the on-chain market."""
    import random

    from . import app as appmod
    net = load(root)
    me = next(p for p in net["personas"] if p["name"] == name)
    accounts, _ = appmod.localnet_accounts(root)
    app = appmod.App(accounts, buyer=name, log=_say)
    stats_path = os.path.join(me["data"], "buyer.json")
    try:
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    except (FileNotFoundError, ValueError):
        stats = {"products": 0, "trainings": 0, "jobs": 0, "paid": 0, "coins_bought": 0, "last": ""}
    rng = random.Random()
    gone = None
    _say(f"{name} starts buying compute on the network")
    while True:
        if not _answers(me):
            gone = gone or time.time()
            if time.time() - gone > BUYER_GRACE:
                _say(f"{name}'s node is gone; stopping")
                try:
                    app.close(name)
                except (OSError, ValueError):
                    pass
                return
            time.sleep(1)
            continue
        gone = None
        # A purchase that takes over BUYER_STUCK seconds writes every thread's stack to the log.
        faulthandler.dump_traceback_later(BUYER_STUCK, file=sys.stdout)
        try:
            _buy_once(app, me, net, stats, rng)
        except Exception as e:                   # keep going: the network is allowed to be busy
            _say(f"could not buy this time: {appmod.describe(e)}")
            time.sleep(10)
        finally:
            faulthandler.cancel_dump_traceback_later()
        with open(stats_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(stats, f)
        os.replace(stats_path + ".tmp", stats_path)
        time.sleep(rng.uniform(10, 30))


def _buy_once(app, me: dict, net: dict, stats: dict, rng) -> None:
    name = me["name"]
    balance = http_get(f"{url(me)}/account/{me['address']}", timeout=10)["pending_balance"]
    if balance < LOW_BALANCE:
        miners = [(http_get(f"{url(p)}/account/{p['address']}", timeout=10)["pending_balance"], p["name"])
                  for p in net["personas"] if p["mine"] and _answers(p)]
        rich = [m for m in sorted(miners, reverse=True) if m[0] > 4 * COINS_BOUGHT]
        if not rich:
            _say(f"{params.fmt_amount(balance)} left and no miner has coins to sell yet; waiting")
            return
        app.send(rich[0][1], name, str(COINS_BOUGHT / params.YAPRAK_PER_COIN))
        stats["coins_bought"] += COINS_BOUGHT
        stats["last"] = f"bought {params.fmt_amount(COINS_BOUGHT)} from {rich[0][1]}"
        _say(stats["last"] + " (a transfer on this test network)")
        return
    if not http_get(f"{url(me)}/providers", timeout=10):
        _say("no compute on offer yet (providers advertise once they hold 0.1 GRN); waiting")
        return
    paid0 = app.paid(app.account(name))
    roll = rng.random()
    if roll < 0.4:
        steps = rng.choice([50, 100, 200])
        job = {"id": "buyer", "kind": "train", "account": name, "steps": steps, "state": "running",
               "lines": [], "started": time.time()}
        app._train(job, app.account(name))
        if job["state"] != "done":
            raise RuntimeError(job["lines"][-1] if job["lines"] else "training failed")
        r = job["result"]
        stats["trainings"] += 1
        stats["products"] += r["products"]
        stats["last"] = f"trained a digit model ({steps} steps, {r['products']} products, accuracy {r['accuracy']:.2f})"
    elif roll < 0.8:
        count, size = rng.randint(3, 8), rng.choice([256, 512, 1024])
        privacy = rng.choice(["split", "split", "secret"])
        size = min(size, 512) if privacy == "secret" else size     # (secret sharing costs 12x)
        for _ in range(count):
            app.product(name, size, size, size, privacy)
        stats["products"] += count
        stats["last"] = f"bought {count} products of {size}x{size}x{size} ({privacy}, sealed through a peer)"
    else:
        r = app.post_job(name, 512, 512, 512, JOB_PRICE)
        stats["jobs"] += 1
        stats["last"] = f"posted job {r['job'][:12]} on chain ({r['units']} units, escrow {params.fmt_amount(r['escrow'])})"
    stats["paid"] += app.paid(app.account(name)) - paid0
    _say(stats["last"])


def _say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


if __name__ == "__main__":                        # python -m green.localnet buyer <root> <name>
    if len(sys.argv) == 4 and sys.argv[1] == "buyer":
        run_buyer(sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: python -m green.localnet buyer <root> <name>   (started by grn localnet start)")
