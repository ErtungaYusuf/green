"""A language model run across several miners, with their work checked.

    python examples/distributed_llm.py --model <folder with a Qwen2.5 model> [--latency 0.02]

The model's layers are split across four stages (miners; here on this computer, with a
simulated network delay on every hop). The buyer keeps the embedding and the head,
sends hidden states through the stages, and checks each stage's work at random with
Freivalds' test. Three runs:

1. one token per pass: the plain pipeline;
2. speculative decoding: several guessed tokens checked per pass, same text;
3. a cheating stage that skips a layer: caught, named, and dropped.

Download the model (about 1 GB) from https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
(config.json, tokenizer.json, model.safetensors). Needs `pip install tokenizers`.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tokenizers import Tokenizer  # noqa: E402

from green.llm.model import Model  # noqa: E402
from green.llm.pipeline import CheatingStage, Pipeline, even_splits, local_stages  # noqa: E402

END = 151645                  # <|im_end|>


def chat(tok, question: str) -> list:
    return tok.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                      f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n").ids


def run(model, tok, ids, latency, draft, cheat=None, max_new=80):
    stages = local_stages(model, even_splits(model.n_layers, 4), latency=latency, cheat=cheat)
    # On one computer the buyer's audits would take the CPU from the miners: run them after.
    pipe = Pipeline(model, [s.info for s in stages], latency=latency, audit_later=True)
    try:
        out, st = pipe.generate(ids, max_new=max_new, draft=draft, stop=(END,))
        try:
            pipe.finish_audits()
        except CheatingStage as e:
            return tok.decode(out), st, len([a for a in pipe.audit_log if a[0] == "ok"]), e
        return tok.decode(out), st, len([a for a in pipe.audit_log if a[0] == "ok"]), None
    except CheatingStage as e:
        return None, None, len([a for a in pipe.audit_log if a[0] == "ok"]), e
    finally:
        pipe.stop()
        for s in stages:
            s.stop()


def serve(model, tok, args):
    from green.llm.api import ChatServer
    stages = local_stages(model, even_splits(model.n_layers, 4), latency=args.latency)
    pipe = Pipeline(model, [s.info for s in stages], latency=args.latency)
    srv = ChatServer(pipe, tok, os.path.basename(args.model.rstrip("/\\")), port=args.serve)
    print(f"OpenAI-compatible API: {srv.url}/v1/chat/completions (Ctrl+C stops)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--latency", type=float, default=0.02, help="one-way delay per hop, seconds")
    ap.add_argument("--serve", type=int, metavar="PORT",
                    help="instead of the demo, serve an OpenAI-compatible API on this port")
    ap.add_argument("--question", default="Fix the spelling in this text and give back the whole text: "
                                          "'The quick brown fox jumpd over the lazy dog. Every morning the fox "
                                          "runs thru the forrest, looking for food and water for it's family.'")
    args = ap.parse_args()
    t = time.time()
    model = Model(args.model)
    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    print(f"Model loaded in {time.time() - t:.0f} s: {model.n_layers} layers over 4 stages, "
          f"{args.latency * 1000:.0f} ms per hop\n")
    if args.serve:
        return serve(model, tok, args)
    ids = chat(tok, args.question)

    texts = {}
    for draft in (0, 6):
        text, st, audits, err = run(model, tok, ids, args.latency, draft)
        texts[draft] = text
        label = "one token per pass" if draft == 0 else f"speculative, up to {draft} guesses"
        print(f"--- {label}: {st['tokens']} tokens in {st['passes']} passes, {st['seconds']:.1f} s "
              f"({st['tokens_per_s']:.1f} tokens/s, {st['tokens'] / st['passes']:.2f} per pass), "
              f"{audits} layers audited, all honest")
    print(texts[6])
    print(f"\nSame text both ways: {texts[0] == texts[6]}\n")

    print("--- stage 3 of 4 cheats: it skips its last layer")
    text, st, audits, err = run(model, tok, ids, args.latency, 6, cheat={2: "skip"})
    print(f"Caught after {audits} honest audits: {err}" if err else "NOT caught")


if __name__ == "__main__":
    main()
