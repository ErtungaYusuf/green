"""An OpenAI-compatible endpoint on the buyer's computer, in front of a Pipeline, so
existing programs and agents use the network's model without changing a line:

    grn llm --model DIR                                 # the model's layers bought on the network
    POST http://127.0.0.1:<port>/v1/chat/completions   {"messages": [...], "max_tokens": 128}
    GET  http://127.0.0.1:<port>/v1/models

Only greedy decoding (temperature 0): that is what the miners' work can be checked for.
The buyer keeps the weights: it needs them to check the miners' work.
"""

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import Model, model_id
from .pipeline import Pipeline

IM_END = "<|im_end|>"


def chat_prompt(messages: list) -> str:
    """The ChatML layout Qwen models are trained on."""
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + list(messages)
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    return text + "<|im_start|>assistant\n"


def load_tokenizer(path: str):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise RuntimeError("a language model needs the tokenizers package: pip install tokenizers") from None
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json"))


class NetworkModel:
    """A language model whose layers are bought on the network: the weights stay here (to
    check the miners' work), the layers run on miners found through the offers, paid
    through channels of `per_stage` yaprak each, opened on the first request."""

    def __init__(self, path: str, node_url: str, key, per_stage: int, draft: int = 6):
        self.model, self.tok, self.id = Model(path), load_tokenizer(path), model_id(path)
        self.name = os.path.basename(os.path.normpath(path))
        self.node_url, self.key, self.per_stage, self.draft = node_url, key, per_stage, draft
        self.end = self.tok.token_to_id(IM_END)
        self.pipe: Pipeline | None = None
        self.lock = threading.Lock()                       # one generation at a time per pipeline

    def owed(self) -> int:
        return sum(st["owed"] for st in self.pipe.stages) if self.pipe else 0

    def complete(self, req: dict) -> dict:
        with self.lock:
            if self.pipe is None:
                self.pipe = Pipeline.from_market(self.node_url, self.key, self.model, self.id, self.per_stage)
            return complete(self.pipe, self.tok, self.end, self.name, self.draft, req)

    def close(self) -> dict:
        with self.lock:
            paid, self.pipe = (self.pipe.close() if self.pipe else {}), None
            return paid


def complete(pipe: Pipeline, tok, end: int, name: str, draft: int, req: dict) -> dict:
    """One chat completion, in OpenAI's shape, with what it took on the network."""
    if not isinstance(req.get("messages"), list) or not req["messages"]:
        raise ValueError("messages: a list of {role, content}")
    ids = tok.encode(chat_prompt(req["messages"])).ids
    max_new = int(req.get("max_tokens") or 256)
    before = sum(st.get("owed", 0) for st in pipe.stages)
    out, st = pipe.generate(ids, max_new=max_new, draft=draft, stop=(end,))
    done = bool(out) and out[-1] == end
    text = tok.decode(out[:-1] if done else out)
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
            "model": name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop" if done else "length"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)},
            "green": {"passes": st["passes"], "seconds": round(st["seconds"], 3), "stages": len(pipe.stages),
                      "cost": sum(s.get("owed", 0) for s in pipe.stages) - before, "replaced": st.get("replaced", 0)}}


class ChatServer:
    """`pipe` is a Pipeline (with `tokenizer` and `name`), or a NetworkModel."""

    def __init__(self, pipe, tokenizer=None, name: str = "", host: str = "127.0.0.1", port: int = 0, draft: int = 6):
        if isinstance(pipe, NetworkModel):
            self.complete, self.name = pipe.complete, pipe.name
        else:
            end, lock = tokenizer.token_to_id(IM_END), threading.Lock()

            def local(req):
                with lock:
                    return complete(pipe, tokenizer, end, name, draft, req)
            self.complete, self.name = local, name
        owner = self

        class H(BaseHTTPRequestHandler):
            def _send(self, code: int, body: dict):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/") == "/v1/models":
                    return self._send(200, {"object": "list", "data": [{"id": owner.name, "object": "model",
                                                                         "owned_by": "green"}]})
                self._send(404, {"error": {"message": "not found"}})

            def do_POST(self):
                if self.path.rstrip("/") != "/v1/chat/completions":
                    return self._send(404, {"error": {"message": "not found"}})
                try:
                    req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    self._send(200, owner.complete(req))
                except (ValueError, KeyError, TypeError) as e:
                    self._send(400, {"error": {"message": str(e)}})
                except Exception as e:                     # noqa: BLE001  (a cheater caught, a stage gone)
                    self._send(502, {"error": {"message": f"{type(e).__name__}: {e}"}})

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer((host, port), H)
        self.url = f"http://{host}:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
