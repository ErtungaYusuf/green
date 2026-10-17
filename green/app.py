"""The Green app: use the network from one place -- mine, send coins, buy compute --
and a compute proxy for other programs on this computer.

    grn app --localnet [DIR]     the five people of a local network (grn localnet start)
    grn app                      this wallet (--data) and its node (--node)

Then open http://127.0.0.1:8600. Programs use the proxy like a local matrix
multiplier; it buys each product on the network through payment channels, checks
it with Freivalds' test, and answers in the providers' own wire format:

    POST http://127.0.0.1:8600/v1/matmul
         X-Shape: m,k,n         body: A (m*k int8, row-major) then B (k*n int8)
    ->   m*n little-endian int32

The app listens on this computer only. Requests that spend coins need a header a
web page cannot send cross-site (the page's token, or X-Shape for the proxy) and a
Host that names this machine, so a web site open in a browser here cannot use it.
"""

import json
import os
import secrets
import socket
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np

from . import params, tx as txmod, work
from .compute import MAX_DIM, ComputeCluster, PaymentRequired, ProviderError, WrongResult
from .crypto import Key, is_address
from .node import BoundedServer, http_get, http_post, names_this_machine
from .train import MLP, digits, train

DEFAULT_PORT = 8600
CHANNEL_AMOUNT = params.YAPRAK_PER_COIN // 20   # locked with each provider when an account starts buying
PROVIDERS = 2                                    # channels an account opens, to the cheapest providers
MAX_BODY = 2 * MAX_DIM * MAX_DIM
TRAIN_STEPS_MAX = 2000
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.html")


class AppError(Exception):
    pass


class Account:
    """Someone using the network: a key and the node it talks to."""

    def __init__(self, name: str, role: str, key: Key, node: str, data: str | None = None):
        self.name, self.role, self.key, self.node, self.data = name, role, key, node.rstrip("/"), data
        self.cluster: ComputeCluster | None = None
        self.lock = threading.Lock()                # one channel setup at a time
        self.spent = 0                              # paid through closed clusters


class App:
    def __init__(self, accounts: list[Account], buyer: str | None = None, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, allow_local: bool = True, log=print):
        if not accounts:
            raise AppError("no accounts")
        self.accounts = {a.name: a for a in accounts}
        self.buyer = buyer if buyer in self.accounts else accounts[0].name
        self.host, self.port, self.allow_local, self.log = host, port, allow_local, log
        self.token = secrets.token_hex(16)
        self.jobs: dict[str, dict] = {}
        self.jobs_lock = threading.Lock()
        self.proxy = {"products": 0, "macs": 0, "seconds": 0.0, "errors": 0}
        self.proxy_lock = threading.Lock()
        self.server = None

    # --- reading ------------------------------------------------------------------------

    def account(self, name: str | None) -> Account:
        a = self.accounts.get(name or self.buyer)
        if a is None:
            raise AppError(f"no account called {name!r}")
        return a

    def names(self) -> dict:
        """address -> account name, to show who mined what."""
        return {a.key.address: a.name for a in self.accounts.values()}

    def overview(self) -> dict:
        rows = []
        for a in self.accounts.values():
            row = {"name": a.name, "role": a.role, "address": a.key.address, "node": a.node, "online": False}
            try:
                if not _reachable(a.node):
                    raise OSError("not listening")
                st = http_get(f"{a.node}/status", timeout=2)
                acct = http_get(f"{a.node}/account/{a.key.address}", timeout=2)
                row.update(online=True, height=st["height"], peers=len(st["peers"]), mining=st["mining"],
                           can_mine=st["miner"] == a.key.address, units_per_second=st["units_per_second"],
                           balance=acct["balance"], pending=acct["pending_balance"], genesis=st["genesis"],
                           reserve=st["reserve"], mempool=st["mempool"], blocks_found=st["stats"].get("blocks_found", 0))
            except (OSError, ValueError, KeyError):
                pass
            cl = a.cluster
            row["buying"] = None if cl is None else {
                "providers": len(cl.channels), "locked": sum(c.amount for c in cl.channels), "owed": cl.owed,
                "products": sum(c.stats["products"] for c in cl.channels)}
            row["spent"] = self.paid(a)
            rows.append(row)
        live = next((r for r in rows if r["online"]), None)
        out = {"accounts": rows, "buyer": self.buyer, "proxy": dict(self.proxy),
               "proxy_url": f"http://{self.host}:{self.port}/v1/matmul", "jobs": self.job_list()}
        if live is not None:
            node = self.accounts[live["name"]].node
            try:
                out["offers"] = http_get(f"{node}/providers", timeout=2)
                out["blocks"] = self.recent_blocks(node, live["height"])
            except (OSError, ValueError):
                pass
        return out

    def recent_blocks(self, node: str, height: int, count: int = 8) -> list[dict]:
        names, out = self.names(), []
        for h in range(height, max(-1, height - count), -1):
            try:
                b = http_get(f"{node}/height/{h}", timeout=2, limit=16 * 1024 * 1024)
            except (OSError, ValueError):
                break
            hdr = b["header"]
            miner = hdr["miner"] if isinstance(hdr["miner"], str) else "pool"
            out.append({"height": h, "time": hdr["time"], "miner": names.get(miner, miner[:14] + "..."),
                        "txs": len(b.get("txs", [])), "shares": len(b.get("shares", [])),
                        "work": "job" if hdr["work"]["job"] else "filler"})
        return out

    # --- acting -------------------------------------------------------------------------

    def set_mining(self, name: str, on: bool) -> bool:
        a = self.account(name)
        return http_post(f"{a.node}/mining", {"on": bool(on)}, timeout=10)["mining"]

    def send(self, frm: str, to: str, amount_text: str) -> str:
        a = self.account(frm)
        to_addr = self.accounts[to].key.address if to in self.accounts else to
        if not is_address(to_addr):
            raise AppError("send to an account name or an address")
        amount = params.parse_amount(str(amount_text))
        if amount <= 0:
            raise AppError("the amount must be positive")
        return self._submit(a, "transfer", to=to_addr, amount=amount)

    def _submit(self, a: Account, kind: str, **fields) -> str:
        chain_id = http_get(f"{a.node}/status", timeout=10)["genesis"]
        nonce = http_get(f"{a.node}/account/{a.key.address}", timeout=10)["pending_nonce"]
        fee = txmod.min_fee({"type": kind, **fields})
        try:
            rate = http_get(f"{a.node}/v1/fee?blocks=2", timeout=10)["fee_rate"]
            size = txmod.tx_size(txmod.make_tx(a.key, chain_id, kind, nonce, fee, **fields))
            fee = max(fee, int(rate * size) + 1 if rate else 0)
        except (OSError, ValueError, KeyError):
            pass
        return http_post(f"{a.node}/tx", txmod.make_tx(a.key, chain_id, kind, nonce, fee, **fields), timeout=10)["txid"]

    def cluster(self, a: Account) -> ComputeCluster:
        """The account's channels to providers, opened on first use (one block)."""
        with a.lock:
            if a.cluster is None:
                self.log(f"[app] {a.name}: opening channels to {PROVIDERS} providers "
                         f"({params.fmt_amount(CHANNEL_AMOUNT)} each; confirmed in the next block)")
                a.cluster = ComputeCluster.from_market(a.node, a.key, CHANNEL_AMOUNT, count=PROVIDERS,
                                                       allow_local=self.allow_local)
            return a.cluster

    @staticmethod
    def paid(a: Account) -> int:
        """Everything the account has paid for compute through this app."""
        cl = a.cluster
        return a.spent + (cl.owed if cl is not None else 0)

    def matmul(self, name: str | None, a8: np.ndarray, b8: np.ndarray) -> np.ndarray:
        """a8 @ b8 bought on the network; a spent or ended channel is replaced once."""
        acct = self.account(name)
        for attempt in (1, 2):
            cl = self.cluster(acct)
            try:
                return cl.matmul(a8, b8)
            except (PaymentRequired, ProviderError) as e:
                if attempt == 2:
                    raise AppError(f"no provider could do it: {e}") from None
                self.log(f"[app] {acct.name}: {e}; opening new channels")
                self.close(acct.name)

    def product(self, name: str | None, m: int, k: int, n: int) -> dict:
        if not all(1 <= d <= MAX_DIM for d in (m, k, n)):
            raise AppError(f"dimensions must be 1..{MAX_DIM}")
        acct = self.account(name)
        rng = np.random.default_rng()
        a8 = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b8 = rng.integers(-128, 128, (k, n), dtype=np.int8)
        opened = acct.cluster is None
        t = time.perf_counter()
        self.cluster(acct)
        setup = time.perf_counter() - t
        paid = self.paid(acct)
        t = time.perf_counter()
        c = self.matmul(name, a8, b8)
        took = time.perf_counter() - t
        t = time.perf_counter()
        local = work.blas_matmul(a8, b8, np.int64)            # the same product on this machine, to compare
        local_s = time.perf_counter() - t
        return {"account": acct.name, "shape": [m, k, n], "seconds": took, "setup_seconds": setup if opened else 0,
                "cost": self.paid(acct) - paid, "checked": True,
                "matches_local": bool(np.array_equal(c, local)), "local_seconds": local_s,
                "sample": c[:2, :4].tolist()}

    def close(self, name: str | None) -> int:
        """Close the account's channels: the providers claim what was used, the rest comes back."""
        a = self.account(name)
        with a.lock:
            cl, a.cluster = a.cluster, None
        if cl is None:
            return 0
        a.spent += cl.owed
        return cl.close()

    # --- training jobs ------------------------------------------------------------------

    def job_list(self) -> list[dict]:
        with self.jobs_lock:
            return [dict(j, lines=list(j["lines"][-12:])) for j in self.jobs.values()][-5:]

    def start_training(self, name: str | None, steps: int) -> str:
        acct = self.account(name)
        steps = max(10, min(TRAIN_STEPS_MAX, int(steps)))
        jid = secrets.token_hex(4)
        job = {"id": jid, "kind": "train", "account": acct.name, "steps": steps, "state": "running",
               "lines": [], "started": time.time()}
        with self.jobs_lock:
            self.jobs[jid] = job
        threading.Thread(target=self._train, args=(job, acct), daemon=True, name="app-train").start()
        return jid

    def _train(self, job: dict, acct: Account) -> None:
        def say(text):
            with self.jobs_lock:
                job["lines"].append(f"[{time.strftime('%H:%M:%S')}] {text}")
        try:
            if acct.cluster is None:
                say("Opening payment channels to the cheapest providers (one transaction each, confirmed in the next block)...")
            cl = self.cluster(acct)
            say(f"{len(cl.channels)} providers: " + ", ".join(
                f"{self.names().get(c.offer['provider'], c.offer['provider'][:12])} at {c.offer['price']} yaprak/unit"
                for c in cl.channels))
            x, y = digits(6000, seed=1)
            xt, yt = digits(1500, seed=2)
            model = MLP([256, 128, 10], seed=3)
            paid0 = self.paid(acct)
            started = time.time()
            say(f"Training a 256-128-10 network on 16x16 digits, {job['steps']} steps; every product is computed by miners and checked here")

            def report(step, loss):
                say(f"step {step:4d}  loss {loss:.3f}  test accuracy {model.accuracy(xt, yt):.3f}  "
                    f"paid {params.fmt_amount(self.paid(acct) - paid0)}")
            train(model, x, y, job["steps"], matmul=lambda a, b: self.matmul(acct.name, a, b), lr=0.1, seed=4,
                  report=report, every=max(10, job["steps"] // 8))
            result = {"accuracy": model.accuracy(xt, yt), "products": model.products,
                      "seconds": time.time() - started, "paid": self.paid(acct) - paid0}
            say(f"Done in {result['seconds']:.1f} s: {result['products']} products, test accuracy "
                f"{result['accuracy']:.3f}, paid {params.fmt_amount(result['paid'])}")
            with self.jobs_lock:
                job.update(state="done", result=result)
        except Exception as e:                       # shown to the user, not raised in a thread
            say(f"Failed: {e}")
            with self.jobs_lock:
                job["state"] = "failed"

    # --- the server ---------------------------------------------------------------------

    def start(self) -> None:
        self.server = BoundedServer((self.host, self.port), _handler(self))
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True, name="app-http").start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        for a in self.accounts.values():
            if a.cluster is not None:
                try:
                    self.close(a.name)
                except (OSError, ValueError, ProviderError):
                    pass

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _handler(app: App):
    with open(PAGE, encoding="utf-8") as f:
        page = f.read().replace("{{TOKEN}}", app.token).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 60

        def log_message(self, *args):
            pass

        def _send(self, code: int, body, ctype: str = "application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def _refuse(self, code: int, error: str):
            self.close_connection = True
            raw = self.headers.get("Content-Length", "0")
            if raw.isdigit() and int(raw) <= MAX_BODY:
                self.rfile.read(int(raw))                # a closed socket with data unread resets the answer
            self._send(code, {"error": error})

        def _json(self) -> dict:
            raw = self.headers.get("Content-Length", "0")
            if not raw.isdigit() or int(raw) > 64 * 1024:
                raise AppError("bad request body")
            body = json.loads(self.rfile.read(int(raw)) or b"{}")
            if not isinstance(body, dict):
                raise AppError("bad request body")
            return body

        def do_GET(self):
            if not names_this_machine(self.headers.get("Host")):
                return self._refuse(403, "use http://127.0.0.1")
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(200, app.overview())
            if path == "/v1/status":
                return self._send(200, {"proxy": dict(app.proxy), "buyer": app.buyer})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not names_this_machine(self.headers.get("Host")):
                return self._refuse(403, "use http://127.0.0.1")
            path = urlparse(self.path).path
            try:
                if path == "/v1/matmul":
                    return self._proxy()
                if self.headers.get("X-App-Token") != app.token:
                    return self._refuse(403, "missing or wrong X-App-Token (open the app at its own address)")
                body = self._json()
                if path == "/api/mining":
                    return self._send(200, {"mining": app.set_mining(body["account"], bool(body["on"]))})
                if path == "/api/send":
                    return self._send(200, {"txid": app.send(body["from"], body["to"], body["amount"])})
                if path == "/api/product":
                    return self._send(200, app.product(body.get("account"), int(body["m"]), int(body["k"]),
                                                       int(body["n"])))
                if path == "/api/train":
                    return self._send(200, {"job": app.start_training(body.get("account"), body.get("steps", 200))})
                if path == "/api/close":
                    return self._send(200, {"closed": app.close(body.get("account"))})
                if path == "/api/buyer":
                    app.buyer = app.account(body["account"]).name
                    return self._send(200, {"buyer": app.buyer})
                return self._send(404, {"error": "not found"})
            except (AppError, PaymentRequired, ProviderError, WrongResult) as e:
                return self._send(400, {"error": str(e)})
            except urllib.error.HTTPError as e:
                try:
                    detail = json.loads(e.read()).get("error", str(e))
                except (ValueError, OSError):
                    detail = str(e)
                return self._send(400, {"error": detail})
            except (KeyError, TypeError, ValueError) as e:
                return self._send(400, {"error": f"bad request: {e}"})
            except OSError as e:
                return self._send(502, {"error": f"the node did not answer: {e}"})

        def _proxy(self):
            shape = self.headers.get("X-Shape", "")
            raw = self.headers.get("Content-Length", "")
            try:
                m, k, n = (int(x) for x in shape.split(","))
            except ValueError:
                return self._refuse(400, "X-Shape must be m,k,n")
            if not all(1 <= d <= MAX_DIM for d in (m, k, n)):
                return self._refuse(400, f"dimensions must be 1..{MAX_DIM}")
            if not raw.isdigit() or int(raw) != m * k + k * n:
                return self._refuse(400, "the body must be A (m*k bytes) then B (k*n bytes), int8")
            body = self.rfile.read(int(raw))
            a8 = np.frombuffer(body, np.int8, m * k).reshape(m, k)
            b8 = np.frombuffer(body, np.int8, k * n, offset=m * k).reshape(k, n)
            t = time.perf_counter()
            try:
                c = app.matmul(self.headers.get("X-Account"), a8, b8)
            except Exception as e:
                with app.proxy_lock:
                    app.proxy["errors"] += 1
                return self._send(502, {"error": str(e)})
            with app.proxy_lock:
                app.proxy["products"] += 1
                app.proxy["macs"] += m * k * n
                app.proxy["seconds"] += time.perf_counter() - t
            return self._send(200, np.ascontiguousarray(c, dtype="<i4").tobytes(), "application/octet-stream")

    return Handler


def _reachable(node: str) -> bool:
    """A quick check that something listens at the node's address (a refused local
    connection otherwise takes about two seconds on Windows, and the page asks often)."""
    u = urlparse(node)
    try:
        with socket.create_connection((u.hostname or "127.0.0.1", u.port or 80), timeout=0.5):
            return True
    except OSError:
        return False


def localnet_accounts(root: str) -> tuple[list[Account], str]:
    """The personas of a local network (their wallets are unencrypted), and its buyer."""
    from . import localnet
    net = localnet.load(root)
    if net is None:
        raise AppError(f"no local network in {root}: start one with grn localnet start")
    accounts = []
    for p in net["personas"]:
        with open(os.path.join(p["data"], "wallet.json"), encoding="utf-8") as f:
            w = json.load(f)
        if "phrase" not in w:
            raise AppError(f"{p['name']}'s wallet is encrypted; the local network's are not")
        from . import hd
        key = hd.key_from_phrase(w["phrase"], w.get("passphrase", ""), w["path"])
        accounts.append(Account(p["name"], p["role"], key, localnet.url(p), p["data"]))
    return accounts, "carol"
