# Language models across miners

A language model on Green runs on many miners at once: each holds a few of its layers,
and a buyer's text flows through them. The buyer checks their work, so a miner that
skips or fakes any part of it is caught and named. This is phase 1: it runs on one
computer as a simulation of the network (`green/llm/`, `examples/distributed_llm.py`,
`tests/test_llm.py`); payment and penalties on chain come in phase 2.

## Splitting a model

```
buyer: text -> tokens -> embedding ──▶ miner A: layers 0-5 ──▶ B: 6-11 ──▶ C: 12-17 ──▶ D: 18-23 ──▶ buyer: head -> next token
```

- A miner holds whole layers. What travels between miners is the hidden state of the
  tokens in flight: 7 KB per token for a 0.5B model, about 32 KB for an 8B one. The
  weights never travel.
- Each miner hands its result straight to the next, and the last one to the buyer, so a
  pass costs one network hop per miner. A miner keeps the attention cache of its own
  layers, so a pass carries only the new tokens.
- The embedding and the head stay with the buyer: miners see hidden states, not the text.
- Splitting each product across machines (tensor parallelism) would need several
  network round trips per layer; whole layers per miner need one hop per miner.

## Hiding the latency

The network adds a delay per hop that no protocol removes. Green hides it:

- **Speculative decoding.** The buyer guesses the next few tokens cheaply and one pass
  checks all the guesses at once: the logits accept the guesses that match the model's
  own choice and add one more token. A pass then yields several tokens when the guesses
  are good, and the text is exactly what the model would produce one token at a time
  (`test_the_pipeline_gives_the_models_own_tokens_and_speculation_changes_nothing`).
  Phase 1 guesses by finding the last few tokens earlier in the text (prompt lookup),
  which is free and strong when the answer repeats its input (editing, code, summaries).
  A small draft model on the buyer's own computer guesses well on any text.
- **Full pipelines.** A miner works on the next buyer's pass while one is in flight, so
  the network's throughput does not wait for the latency.
- **Nearby miners.** A buyer picks miners close to each other; within a region a hop is
  10-20 ms.

Measured on one computer (Qwen2.5-0.5B-Instruct, int8, 4 stages of 6 layers, 20 ms per
hop, numpy on the CPU), fixing the spelling of a 30-word text:

| | passes | tokens per pass | time |
|---|---|---|---|
| one token per pass | 29 | 1.00 | 18.5 s |
| speculative (up to 6 guesses) | 12 | 2.42 | 13.0 s |

Both produce the same text. On this CPU a pass costs about 0.3 s of computing and 0.1 s
of network; on miners' GPUs the computing shrinks to milliseconds and the network
dominates: a 4-miner pass at 20 ms per hop takes about 0.1 s, which at 2.4 tokens per
pass is about 24 tokens per second. That figure is an estimate, not yet a measurement.

## Why a miner cannot cheat

**Exact integer layers.** Every product inside a layer is int8 times int8 with exact
integer sums, the same operation Green's mining runs. Weights are int8 per output
channel, activations int8 per group of 128 channels, the residual stream fixed point.
A layer's output is then the same on every machine, up to one unit of rounding in the
cheap steps between products. On Qwen2.5-0.5B-Instruct the int8 model chose the same
next token as the bfloat16 original at every position of a test answer; a wider
comparison is part of phase 2.

**Signed commitments.** For every pass each miner signs, for each of its layers, the hash
of the hidden state it took and the one it gave, the hash of the layer's whole work (its
Record: inputs to every product, every rounding step, the output) and of the layer's
new cache rows. The buyer checks that the commitments link up: each miner started from
the previous miner's output and the last delivered what it committed to.

**Random openings.** After each pass, off the critical path, the buyer picks layers at
random (two per miner by default) and asks for their Records. The Record must hash to
what was committed, and the buyer checks it:

- each product Y = X W^T with Freivalds' test against a secret random vector r: Y r must
  equal X (W^T r). The buyer computes W^T r once per weight matrix, so checking a product
  costs about b(k + n) operations where computing it cost b k n: hundreds of times less
  for a 0.5B model, thousands for an 8B one. The miner never learns r, so it cannot make
  a wrong product pass (the chance is 2^-18 per check);
- the steps between products (norms, attention, SiLU, rounding) by doing them again;
- the attention cache rows the layer used, against the commitments of the passes that
  made them, and their products with Freivalds' test too.

A miner does not know which layers will be opened, so faking one of its n layers escapes
a pass with chance 1 - 2/n, and k passes in a row with chance (1 - 2/n)^k: under one in
a hundred thousand after 30 passes for a 6-layer miner. Skipping more layers, the only
way to save real work, is caught sooner. The products themselves need not be committed:
Freivalds' test pins each to its committed input. Re-opening is cheap for the miner,
since a layer's work is deterministic: it keeps only the layer inputs and recomputes.

**A caught miner** is named by its own signature: a signed commitment to a Record that
fails the check is proof anyone can verify. In phase 1 the buyer stops and drops it; in
phase 2 that proof takes the miner's bond on chain.

The buyer needs the weights to check (it computes W^T r from them). A buyer that cannot
hold them hands the openings to an auditor: another miner holding the same layers,
chosen at random and paid for it.

## Phase 2: on the network, paid on chain (in progress)

Built on what Green already has: signed compute offers relayed by every node, payment
channels, vouchers and the provider's claims.

- **Serving layers.** `grn node --provide-port P --provide-url URL --llm-model DIR
  --llm-layers 0-12 --llm-price N` runs a Stage inside the miner's node, next to its
  Provider. The signed offer gains an optional `llm` field: `{url, model, layers, price}`,
  where `model` is the SHA-256 of the weights file and `price` is yaprak per token per
  layer. Nodes from this version relay offers with the field (not a consensus change;
  the founder's seed is upgraded with it).
- **Routes.** The buyer reads `/providers` from its own node, keeps the offers for its
  model, measures each stage's round trip, and picks a chain of stages covering every
  layer at the lowest price and latency, with a spare per range.
- **No inbound connections at the buyer.** It sends a pass to the first stage and waits
  on the last stage (`/llm/wait`, a long poll): one hop back, and it works behind NAT.
- **Payment.** The buyer opens a channel to each stage's address (`channel_open`). Each
  pass carries, per stage, the channel id, a voucher for everything delivered before
  this pass, and the buyer's signature over the pass (so nobody can run up another
  buyer's channel). A stage works only while the unvouched amount stays under a small
  credit and the channel holds enough; the Provider's existing books claim the vouchers
  on chain and close the channel when the buyer is done (`/close`). A buyer that catches
  a stage stops paying it: its loss is at most one credit.
- **Test:** separate node processes on a local network, each with its own wallet,
  finding each other through offers, paid through channels, the claims checked in the
  blocks.

## Phase 3: bonds and penalties (a consensus change, planned)

A stage that signs a commitment to wrong work hands the buyer proof of it. Phase 3 lets
that proof cost the stage its bond.

- **`llm_bond` transaction:** locks an amount (at least `LLM_MIN_BOND`, proposed 10 GRN)
  for a model hash and layer range; unlocks only `LLM_UNBOND_BLOCKS` (proposed 40,320,
  about a week) after an `llm_unbond`, so proof found later can still reach it. The
  offer names the bond; buyers use bonded stages only.
- **`llm_slash` transaction:** carries the evidence: the stage's signed commitment for a
  pass, the opened Record, the cache rows it relied on with their own signed
  commitments, and the challenger's vector r. Every node re-runs `Model.check` on that
  one layer (it needs that layer's weights, about 15 MB for a 0.5B model; the model hash
  in the bond pins which). The r is revealed only here, after the fact, so it is no
  longer secret but the commitment was already signed.
- **Who checks evidence:** nodes that hold the bonded model's layers; the rule is that a
  block with an `llm_slash` is valid only if the check fails, and a node lacking the
  weights fetches the layer by hash from the network before validating.
- **Where the bond goes:** half to the challenger (it paid to audit), half burned, so a
  stage cannot slash itself for profit through a second identity.
- **Weakness covered:** the one-unit rounding slack is fixed in the rule, so honest
  machines never get slashed for floating-point differences.
- **Rollout:** a versionbits deployment, as with earlier rule changes, after tests that
  replay honest and dishonest Records on Linux, Windows and macOS.

## Try it

```
pip install tokenizers
python examples/distributed_llm.py --model <folder with Qwen2.5-0.5B-Instruct>
python examples/distributed_llm.py --model <folder> --serve 8700   # an OpenAI-compatible API
```

The model folder needs `config.json`, `tokenizer.json` and `model.safetensors` from
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct (about 1 GB).
