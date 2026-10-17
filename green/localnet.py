"""A whole network on one computer: five people, each running their own node.

    grn localnet start [--dir localnet] [--gpu NAME]   create it the first time, start every node
    grn localnet status                                who is who: height, peers, mining, balance
    grn localnet stop                                  stop every node cleanly
    grn localnet reset                                 stop, then delete the network

Each persona is a separate process with its own wallet, data folder, ports and log
file, talking to the others over the real encrypted P2P protocol, on a network of its
own (its own genesis block). Nothing is special-cased: these are the nodes anyone
would run, only every address is 127.0.0.1. The wallets are stored unencrypted --
this network is for trying things out, and its coins are worth nothing.

`grn app --localnet` opens the app on it: mine, send coins, buy compute.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error

from . import params
from .chain import block_hash, make_genesis
from .node import http_get, http_post

PERSONAS = [
    {"name": "alice", "role": "miner who sells compute", "mine": True, "price": 1000},
    {"name": "bob", "role": "miner who sells compute, dearer", "mine": True, "price": 1500},
    {"name": "carol", "role": "AI researcher who buys compute", "mine": False, "price": None},
    {"name": "dave", "role": "small miner, paid through shares", "mine": True, "price": None},
    {"name": "erin", "role": "full node: validates, relays, watches", "mine": False, "price": None},
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
            return json.load(f)
    except FileNotFoundError:
        return None


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


def _command(root: str, net: dict, p: dict, gpu: bool) -> list[str]:
    cmd = [sys.executable, "-m", "green", "--data", os.path.abspath(p["data"]), "node",
           "--port", str(p["api"]), "--p2p-port", str(p["p2p"]),
           "--genesis", os.path.abspath(os.path.join(root, "genesis.json")), "--report", "30"]
    for other in net["personas"]:
        if other is not p:
            cmd += ["--peer", f"127.0.0.1:{other['p2p']}"]
    if p["mine"]:
        cmd.append("--mine")
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


def start(root: str, gpu: str | None = None, base_port: int = BASE_PORT, log=print) -> dict:
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
        proc = _spawn(_command(root, net, p, gpu == p["name"]), os.path.join(p["data"], "node.log"))
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
    pids.update({name: proc.pid for name, proc in started.items()})
    with open(_pids_file(root), "w", encoding="utf-8") as f:
        json.dump(pids, f)
    return net


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
    if not asked and os.path.exists(_pids_file(root)):
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
    print(f"{'persona':8s} {'role':38s} {'height':>6s} {'peers':>5s} {'mining':>6s} {'balance':>16s}  node")
    for r in rows:
        if not r["running"]:
            print(f"{r['name']:8s} {r['role']:38s} {'-':>6s} {'-':>5s} {'-':>6s} {'(stopped)':>16s}  {r['url']}")
            continue
        bal = params.fmt_amount(r["balance"]) if r["balance"] is not None else "?"
        print(f"{r['name']:8s} {r['role']:38s} {r['height']:>6d} {r['peers']:>5d} "
              f"{'on' if r['mining'] else 'off':>6s} {bal:>16s}  {r['url']}")
    print(f"\nLogs: {os.path.join(root, '<persona>', 'node.log')}.  App: grn app --localnet {root}")
