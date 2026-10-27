"""An OpenAI-compatible endpoint on the buyer's computer, in front of a Pipeline, so
existing programs and agents use the network's model without changing a line:

    POST http://127.0.0.1:<port>/v1/chat/completions   {"messages": [...], "max_tokens": 128}
    GET  http://127.0.0.1:<port>/v1/models

Only greedy decoding (temperature 0): that is what the miners' work can be checked for.
"""

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .pipeline import Pipeline

IM_END = "<|im_end|>"


def chat_prompt(messages: list) -> str:
    """The ChatML layout Qwen models are trained on."""
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + list(messages)
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    return text + "<|im_start|>assistant\n"


class ChatServer:
    def __init__(self, pipe: Pipeline, tokenizer, name: str, host: str = "127.0.0.1", port: int = 0, draft: int = 6):
        self.pipe, self.tok, self.name, self.draft = pipe, tokenizer, name, draft
        self.end = tokenizer.token_to_id(IM_END)
        self.lock = threading.Lock()                       # one generation at a time per pipeline
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

    def complete(self, req: dict) -> dict:
        ids = self.tok.encode(chat_prompt(req["messages"])).ids
        max_new = int(req.get("max_tokens") or 256)
        with self.lock:
            out, st = self.pipe.generate(ids, max_new=max_new, draft=self.draft, stop=(self.end,))
        done = bool(out) and out[-1] == self.end
        text = self.tok.decode(out[:-1] if done else out)
        return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
                "model": self.name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop" if done else "length"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                          "total_tokens": len(ids) + len(out)},
                "green": {"passes": st["passes"], "seconds": round(st["seconds"], 3), "stages": len(self.pipe.stages)}}

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
