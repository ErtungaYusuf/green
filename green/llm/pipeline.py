"""A language model split across miners, as a pipeline, with every miner's work checkable.

    buyer: tokens -> embedding ──▶ stage A (layers 0-5) ──▶ stage B (6-11) ──▶ ... ──▶ buyer: head -> next tokens

Each stage is a miner holding a few of the model's layers. A pass carries the hidden
states of a few tokens (a few KB each) from stage to stage, and each stage hands its
result straight to the next, so a pass costs one network hop per stage and no more.

Latency is hidden two ways. Speculative decoding: the buyer guesses the next few
tokens cheaply (here by finding the recent tokens earlier in the text) and one pass
checks all the guesses at once, so a pass yields several tokens when the guesses are
right, and exactly the tokens the model would have produced either way. And the
stages work on other buyers' passes while one is in flight.

Cheating is caught by the buyer. Each stage signs a commitment per pass: the hash of
the hidden state it took and the one it gave after every layer, of each layer's whole
Record, and of each layer's new cache rows. The buyer checks that the commitments link
up (each stage's input is the previous stage's output). Then, off the critical path,
it picks a layer at random and asks the stage to open it: the Record must hash to
what was committed, and Model.check re-verifies it (Freivalds' test with the buyer's
secret vector for every product, the cheap steps done again). The stage cannot know
which layer will be opened, so skipping or faking any layer is caught with a chance of
one in its layer count on every pass, and a signed commitment to wrong work is proof
of it. A caught stage is dropped and the text is made again on an honest one.
"""

import base64
import hashlib
import itertools
import json
import queue
import random
import secrets
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ..crypto import Key, canonical, verify
from .model import Model

# What a stage commits to per layer: the products (y*) are left out, since Freivalds' test
# ties each one to its committed input; this keeps a commitment to tens of KB per token.
RECORD_KEYS = ("h_in", "pos", "x1q", "s1", "aq", "sa", "h_mid", "x2q", "s2", "mq", "sm", "h_out")
ROW_KEYS = ("x1q", "s1", "yk", "yv")                 # a layer's cache rows (yk, yv opened, not committed)
ROW_COMMIT = ("x1q", "s1")


# --- wire format: JSON with arrays as base64 ---------------------------------------------

def pack(obj):
    if isinstance(obj, np.ndarray):
        return {"nd": [obj.dtype.str, list(obj.shape), base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode()]}
    if isinstance(obj, dict):
        return {k: pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [pack(v) for v in obj]
    return obj


def unpack(obj):
    if isinstance(obj, dict):
        if set(obj) == {"nd"}:
            dt, shape, data = obj["nd"]
            return np.frombuffer(base64.b64decode(data), dtype=np.dtype(dt)).reshape(shape).copy()
        return {k: unpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unpack(v) for v in obj]
    return obj


def digest(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(a.dtype.str.encode() + repr(a.shape).encode())
        m.update(memoryview(a).cast("B"))
    return m.hexdigest()


def record_hash(rec: dict) -> str:
    return digest(*(rec[k] for k in RECORD_KEYS))


def rows_hash(rec: dict) -> str:
    return digest(*(rec[k] for k in ROW_COMMIT))


def post(url: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(pack(body)).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return unpack(json.loads(r.read()))


class _Server:
    """A small HTTP server whose POST handlers are methods named on_<path>."""

    def serve(self, host: str = "127.0.0.1", port: int = 0) -> str:
        owner = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    body = unpack(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                    out = getattr(owner, "on_" + self.path.strip("/"))(body)
                    data, code = json.dumps(pack(out)).encode(), 200
                except Exception as e:                       # noqa: BLE001  (reported to the caller)
                    data, code = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), 400
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer((host, port), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://{host}:{self.httpd.server_address[1]}"
        return self.url

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class _Sender:
    """Delivers messages after a simulated one-way network delay, without holding up
    the work that comes next."""

    def __init__(self, latency: float):
        self.latency, self.q = latency, queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, url: str, body: dict):
        self.q.put((time.perf_counter() + self.latency, url, body))

    def _run(self):
        while True:
            at, url, body = self.q.get()
            if (wait := at - time.perf_counter()) > 0:
                time.sleep(wait)
            try:
                post(url, body)
            except OSError:
                pass


# --- a miner's stage ----------------------------------------------------------------------

class Stage(_Server):
    """A miner serving layers [first, last) of a model. `cheat` is for tests and demos:
    "skip" fakes the stage's last layer instead of computing it."""

    def __init__(self, model: Model, layers: range, key: Key, latency: float = 0.0, cheat: str | None = None):
        self.model, self.layers, self.key, self.cheat = model, layers, key, cheat
        self.sender = _Sender(latency)
        self.sessions: dict[str, list] = {}               # session -> its passes, in order
        self.work = queue.Queue()
        self.lock = threading.Lock()
        self.passes = self.layer_runs = 0
        threading.Thread(target=self._worker, daemon=True).start()

    @property
    def info(self) -> dict:
        return {"url": self.url, "layers": [self.layers.start, self.layers.stop], "key": self.key.pub_hex}

    def on_forward(self, m: dict) -> dict:
        self.work.put(m)                                  # one queue: the stage works pass by pass
        return {"queued": True}

    def _past(self, passes: list, i: int) -> list:
        """Layer i's cache rows as of now: each earlier pass's rows, cut to what the buyer kept."""
        return [tuple(p["rows"][i][k][:p["keep"]] for k in ROW_KEYS) + (p["pos"][:p["keep"]],)
                for p in passes if p["keep"]]

    def _run_layer(self, i: int, h: np.ndarray, pos: np.ndarray, past: list) -> dict:
        if self.cheat == "skip" and i == self.layers.stop - 1:
            rec = self.model.forward(i, h, pos, past)     # a cheater's record: nothing done, input passed on
            rec = {k: (np.zeros_like(v) if k.startswith("y") else v) for k, v in rec.items()}
            rec["h_mid"] = rec["h_out"] = h
            return rec
        return self.model.forward(i, h, pos, past)

    def _worker(self):
        while True:
            m = self.work.get()
            try:
                self._do(m)
            except Exception as e:                        # noqa: BLE001
                print(f"stage {self.layers}: {type(e).__name__}: {e}")

    def _do(self, m: dict):
        with self.lock:
            passes = self.sessions.setdefault(m["session"], [])
            if passes and passes[-1]["keep"] is None:     # the buyer says how much of the last pass it kept
                last = passes[-1]
                last["keep"] = int(max(0, min(len(last["pos"]), m["cache_len"] - int(last["pos"][0]))))
        h, pos = m["h"], m["pos"]
        bounds, leaves, kv, entry = [digest(h)], [], [], {"no": m["pass"], "pos": pos, "keep": None, "rows": {},
                                                          "inputs": {}}
        for i in self.layers:
            rec = self._run_layer(i, h, pos, self._past(passes, i))
            entry["inputs"][i] = h
            entry["rows"][i] = {k: rec[k] for k in ROW_KEYS}
            h = rec["h_out"]
            bounds.append(digest(h))
            leaves.append(record_hash(rec))
            kv.append(rows_hash(rec))
            self.layer_runs += len(pos)
        with self.lock:
            passes.append(entry)
            self.passes += 1
        c = {"session": m["session"], "pass": m["pass"], "key": self.key.pub_hex,
             "layers": [self.layers.start, self.layers.stop], "bounds": bounds, "leaves": leaves, "kv": kv}
        c["sig"] = self.key.sign(canonical(c))
        commits = m["commits"] + [c]
        if m["route"]:
            self.sender.send(m["route"][0] + "/forward", {**m, "h": h, "route": m["route"][1:], "commits": commits})
        else:
            self.sender.send(m["reply"] + "/result", {"session": m["session"], "pass": m["pass"], "h": h,
                                                      "commits": commits})

    def on_open(self, m: dict) -> dict:
        """Open layer i of a pass: its Record (made again from the stored input: the work
        is deterministic) and every earlier pass's cache rows for that layer, whole."""
        i, no = m["layer"], m["pass"]
        with self.lock:
            passes = list(self.sessions[m["session"]])
        idx = next(n for n, p in enumerate(passes) if p["no"] == no)
        p = passes[idx]
        # the cache as it stood for that pass: the earlier passes, cut as they were then
        past = self._past(passes[:idx], i)
        rec = self._run_layer(i, p["inputs"][i], p["pos"], past)
        return {"record": rec, "past": [{"pass": q["no"], "pos": q["pos"], **q["rows"][i]} for q in passes[:idx]]}


# --- the buyer -----------------------------------------------------------------------------

class CheatingStage(Exception):
    def __init__(self, stage: dict, why: str):
        super().__init__(f"the stage for layers {stage['layers']} cheated: {why}")
        self.stage, self.why = stage, why


class Pipeline(_Server):
    """The buyer's side: embedding and head here, the layers on the stages, and audits."""

    def __init__(self, model: Model, stages: list[dict], latency: float = 0.0, audit: float = 1.0, opens: int = 2,
                 seed=None, audit_later: bool = False):
        """Each pass, with chance `audit`, open `opens` random layers of every stage: a stage
        faking one of n layers then escapes a pass with chance 1 - opens/n, and n passes
        in a row with chance (1 - opens/n)^n. audit_later: hold the audits until
        finish_audits (a simulation on one computer, where the buyer's audits would
        otherwise take the CPU from the miners)."""
        self.model, self.stages, self.latency, self.audit, self.opens = model, stages, latency, audit, opens
        self.later = threading.Event() if audit_later else None
        self.rng = random.Random(seed)
        # the secret vector of Freivalds' test: never leaves the buyer
        self.r = np.frombuffer(secrets.token_bytes(4 * 8192), dtype="<u4").astype(np.int64) & ((1 << 18) - 1)
        self.sketches = {i: model.sketches(i, self.r) for i in model.layers}
        self.results: dict = {}
        self.cond = threading.Condition()
        self.audits = queue.Queue()
        self.audit_log: list = []
        self.caught: CheatingStage | None = None
        self.serve()
        threading.Thread(target=self._auditor, daemon=True).start()

    def on_result(self, m: dict) -> dict:
        with self.cond:
            self.results[(m["session"], m["pass"])] = m
            self.cond.notify_all()
        return {}

    def _check_links(self, h_sent: np.ndarray, m: dict) -> list:
        commits = m["commits"]
        if len(commits) != len(self.stages):
            raise CheatingStage(self.stages[-1], "a stage's commitment is missing")
        expect = digest(h_sent)
        for st, c in zip(self.stages, commits):
            body = {k: v for k, v in c.items() if k != "sig"}
            n = st["layers"][1] - st["layers"][0]
            if (c["key"] != st["key"] or not verify(c["key"], canonical(body), c["sig"], cache=False)
                    or c["layers"] != st["layers"] or len(c["bounds"]) != n + 1 or len(c["leaves"]) != n
                    or len(c["kv"]) != n):
                raise CheatingStage(st, "its commitment is not signed or not complete")
            if c["bounds"][0] != expect:
                raise CheatingStage(st, "it did not start from the previous stage's output")
            expect = c["bounds"][-1]
        if digest(m["h"]) != expect:
            raise CheatingStage(self.stages[-1], "it delivered something other than what it committed to")
        return commits

    def _auditor(self):
        if self.later is not None:
            self.later.wait()
        while True:
            job = self.audits.get()
            if job is None:
                return
            try:
                self._audit(*job)
            except CheatingStage as e:
                self.caught = self.caught or e
            except Exception as e:                        # noqa: BLE001  (an unreachable stage is no proof)
                self.audit_log.append(("error", str(e)))
            finally:
                self.audits.task_done()

    def _audit(self, sess: dict, no: int, s: int, i: int):
        st, c = self.stages[s], sess["commits"][no][s]
        j = i - st["layers"][0]
        out = post(st["url"] + "/open", {"session": sess["id"], "pass": no, "layer": i})
        rec, past = out["record"], out["past"]
        if record_hash(rec) != c["leaves"][j]:
            raise CheatingStage(st, f"layer {i} opened to something it had not committed to")
        if digest(rec["h_in"]) != c["bounds"][j] or digest(rec["h_out"]) != c["bounds"][j + 1]:
            raise CheatingStage(st, f"layer {i}'s input or output is not the committed one")
        if not np.array_equal(rec["pos"], sess["pos"][no]) or rows_hash(rec) != c["kv"][j]:
            raise CheatingStage(st, f"layer {i}'s positions or cache rows do not match")
        rows = []
        for q in past:                                    # every earlier pass, checked against its commitment
            p = q["pass"]
            if p >= no or rows_hash(q) != sess["commits"][p][s]["kv"][j] \
                    or not np.array_equal(q["pos"], sess["pos"][p]):
                raise CheatingStage(st, f"layer {i}'s cache rows from pass {p} are not the committed ones")
            keep = sess["keep"][p]
            if keep:
                rows.append(tuple(q[k][:keep] for k in ROW_KEYS) + (q["pos"][:keep],))
        if len(past) != no:
            raise CheatingStage(st, f"layer {i}'s cache is missing passes")
        if (err := self.model.check(i, rec, rows, self.r, self.sketches[i])) is not None:
            raise CheatingStage(st, err)
        self.audit_log.append(("ok", s, i, no))

    def _pass(self, sess: dict, ids: list, cache_len: int) -> np.ndarray:
        no = len(sess["pos"])
        pos = np.arange(cache_len, cache_len + len(ids), dtype=np.int64)
        h = self.model.embed_tokens(ids)
        sess["pos"].append(pos)
        if no:
            sess["keep"][no - 1] = cache_len - int(sess["pos"][no - 1][0])
        sess["keep"].append(None)
        msg = {"session": sess["id"], "pass": no, "cache_len": cache_len, "pos": pos, "h": h,
               "route": [st["url"] for st in self.stages[1:]], "reply": self.url, "commits": []}
        time.sleep(self.latency)                          # the first hop
        post(self.stages[0]["url"] + "/forward", msg)
        with self.cond:
            if not self.cond.wait_for(lambda: (sess["id"], no) in self.results, timeout=120):
                raise TimeoutError("the pipeline did not answer")
            m = self.results.pop((sess["id"], no))
        sess["commits"].append(self._check_links(h, m))
        for s, st in enumerate(self.stages):              # audits, off the critical path
            if self.rng.random() < self.audit:
                layers = range(*st["layers"])
                for i in self.rng.sample(layers, min(self.opens, len(layers))):
                    self.audits.put((sess, no, s, i))
        return self.model.logits(m["h"])

    def generate(self, prompt: list, max_new: int = 64, draft: int = 4, stop=()) -> tuple[list, dict]:
        """Greedy decoding with speculation: each pass sends the last token and up to
        `draft` guesses; the logits accept the guesses that match and add one token."""
        sess = {"id": secrets.token_hex(8), "pos": [], "keep": [], "commits": []}
        t0 = time.perf_counter()
        logits = self._pass(sess, list(prompt), 0)
        out, cache_len = [int(np.argmax(logits[-1]))], len(prompt)
        t_first, passes = time.perf_counter() - t0, 1
        while len(out) < max_new and out[-1] not in stop:
            if self.caught:
                raise self.caught
            guess = lookup_draft(list(prompt) + out, draft)[:max(0, max_new - len(out) - 1)]
            logits = self._pass(sess, [out[-1]] + guess, cache_len)
            passes += 1
            n = 0
            while n < len(guess) and int(np.argmax(logits[n])) == guess[n]:
                n += 1
            cache_len += 1 + n
            out += guess[:n] + [int(np.argmax(logits[n]))]
            if any(t in stop for t in out):
                out = out[:next(k for k, t in enumerate(out) if t in stop) + 1]
                break
        took = time.perf_counter() - t0
        return out[:max_new], {"tokens": len(out[:max_new]), "passes": passes, "seconds": took,
                               "first_token_s": t_first, "tokens_per_s": len(out) / took, "session": sess}

    def finish_audits(self):
        if self.later is not None:
            self.later.set()
        self.audits.join()
        if self.caught:
            raise self.caught


def lookup_draft(ids: list, k: int, ngram: int = 3) -> list:
    """Guess the next k tokens by finding the last few tokens earlier in the text and
    copying what followed them (prompt lookup): free, and right surprisingly often."""
    if k <= 0:
        return []
    for n in range(ngram, 0, -1):
        if len(ids) <= n:
            continue
        tail = ids[-n:]
        for start in range(len(ids) - n - 1, -1, -1):
            if ids[start:start + n] == tail:
                follow = ids[start + n:start + n + k]
                if follow:
                    return follow
    return []


def local_stages(model: Model, splits: list[range], latency: float = 0.0, cheat: dict | None = None) -> list[Stage]:
    """Stages on this computer (a simulation of miners), one per range of layers."""
    stages = []
    for n, rng in enumerate(splits):
        st = Stage(model, rng, Key.generate(), latency=latency, cheat=(cheat or {}).get(n))
        st.serve()
        stages.append(st)
    return stages


def even_splits(n_layers: int, parts: int) -> list[range]:
    cuts = [round(n_layers * k / parts) for k in range(parts + 1)]
    return [range(a, b) for a, b in itertools.pairwise(cuts)]
