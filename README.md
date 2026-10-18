# Green (GRN)

> The energy burned to secure the world should also produce the intelligence the world needs, and the source of that intelligence should belong to everyone, not to a few companies.

A blockchain prototype where miners do AI work (integer matrix multiplication) instead of hashing, and compute is bought and sold with the coin.

- Every work unit is both a real result delivered to a buyer and a ticket in the block lottery.
- A whole job is delivered with **two transactions** and settled **optimistically**: validators only see a 32-byte Merkle root, the buyer checks every unit with Freivalds (0.4 ms per unit, error <= 2^-124), and a wrong or withheld unit costs the miner the bond of its whole batch.
- A separate **WASM compute market** covers every other kind of computation: the buyer publishes a deterministic program, workers run it, and the result is verified by re-execution.
- **GPU miner** (WebGPU/Vulkan) on AMD/NVIDIA/Intel GPUs: ~1,300 work units per second on an RX 9070 XT, against ~480 on one CPU core.
- **Instant compute through payment channels:** anyone can pay miners to multiply matrices or train a model, product by product, each result checked on arrival.
- Nodes talk over an **encrypted, authenticated peer-to-peer protocol**; every block header **commits to the whole ledger**, so a new node can start from a verified snapshot.
- Everything runs on this computer. By default a node listens only on `127.0.0.1`.

Production status of every area, with evidence and blockers: [PRODUCTION.md](PRODUCTION.md). How rules change: [UPGRADES.md](UPGRADES.md). Design and known limitations: [DESIGN.md](DESIGN.md). Protocol specification: [SPEC.md](SPEC.md). HTTP API: [API.md](API.md). Security reviews and fixes: [SECURITY.md](SECURITY.md). Release notes: [CHANGELOG.md](CHANGELOG.md). Against Bitcoin and Solana: [COMPARISON.md](COMPARISON.md).

## Try it: a whole network on your computer

Five people, each running their own node, on a network of their own: alice, bob and dave mine on the GPU (on the CPU without one, or with `--cpu`) and sell compute; carol, an AI researcher, and erin, a startup, keep asking the network for compute: next to each of their nodes a buyer process trains small models and buys products through payment channels, and posts matrix jobs to the on-chain market. `grn localnet status` shows what they have bought.

```bash
grn localnet start      # create the network the first time, start the five nodes
grn app --localnet      # the app opens in your browser
```

In the app you can turn mining on and off for anyone, send coins between them, buy a matrix product from the miners (paid through payment channels, checked on arrival), and train a small neural network whose every product is bought from the miners. Other programs on your computer can use the network through the app's **compute proxy**, like a local matrix multiplier:

```python
import numpy as np, urllib.request
a = np.random.randint(-128, 128, (512, 512), dtype=np.int8)
b = np.random.randint(-128, 128, (512, 512), dtype=np.int8)
req = urllib.request.Request("http://127.0.0.1:8600/v1/matmul", a.tobytes() + b.tobytes(), {"X-Shape": "512,512,512"})
c = np.frombuffer(urllib.request.urlopen(req).read(), "<i4").reshape(512, 512)    # computed by the network
```

`grn localnet status` shows who is who, `grn localnet stop` stops every node cleanly, `grn localnet reset` deletes the network. The nodes are ordinary nodes (their logs are in `localnet/<name>/node.log`); every command below works on them with `--data localnet/<name> --node http://127.0.0.1:1850N` (alice is 18501, erin 18505; the buyers' logs are `localnet/<name>/buyer.log`). Their wallets are unencrypted: the local network is for trying things, its coins are worth nothing.

## What's new in v0.21

- **Anonymous compute:** requests are sealed to the provider and carried by a random peer, so providers never learn who is asking and only the buyer can read the result; the data is shuffled and spread over providers (split) or secret-shared so each provider sees only random numbers (secret). The app does this by default.
- **Mining intensity:** miners can work a set percentage of the time and switch GPU/CPU while running; the local network's GPU miners run at 50%.

## What's new in v0.20

- **GPU mining that keeps the GPU busy:** a mining node does ~800 units/s on an RX 9070 XT instead of 358 (batches, a pipeline, a faster kernel, and the slow bookkeeping around each try removed); CPU mining uses several cores (398 instead of 137).
- **A local network with buyers:** the miners mine on the GPU, and two buyers keep the network busy with real compute demand: training, products and on-chain jobs.

## What's new in v0.19

- **A network on your computer** (`grn localnet`) and **the app** (`grn app`): mine, send, buy compute and train a model from a browser, and a compute proxy for other programs.
- **Clean stops:** `grn stop`, and the node now shuts down cleanly on SIGTERM too (`systemctl stop`, `docker stop`): providers collect what buyers owe before going away.
- **Admin endpoints closed to web pages:** a page open in a browser on the node's computer could call them; they now need a header browsers cannot send cross-site and a local Host name.

## What's new in v0.18

- **Work units 4× faster:** mining or checking a unit takes ~2.1 ms on one core instead of ~8.4 ms, and a GPU does ~1,280 per second instead of ~700. The digest is simpler and better mixed, and the products are computed and digested while they are in cache, bit-exact on every machine.
- **New nodes catch up 9× faster:** since v0.17 every block carries ~32 shares, each a work unit to check. A long sync now checks them in parallel worker processes: 300 such blocks in 7.2 s instead of 66 s.

## What's new in v0.17

- **Money that neither inflates nor halves:** the 21M GRN supply is fixed and starts in the protocol's reserve; each block pays out a fixed fraction of what is left (a smooth curve, no halving cliffs), and half of every fee flows back in, so rewards never dry up and the supply never grows.
- **Paid for work, not luck:** 80% of each block's reward goes to *shares*, tickets 32 times easier than a block, so every miner is paid in proportion to its work, block after block, without a pool. A block that loses a race still counts as a share.

## What's new in v0.16

- **Reachable from home:** `--portmap` asks the router to forward the node's port (PCP, with NAT-PMP as a fallback), so nodes and compute providers behind a home router can be reached.
- **Hardened compute channels:** providers check a request before reading its body, limit concurrent products, keep their books through reorgs, and close channels on request so the unspent part returns at once; the two products of a layer's backward pass go out together (a fifth fewer round trips per step over the internet).
- **Transactions do not get stuck:** the node pays the fee market's rate for its own deadline-bound transactions and raises the fee of any still waiting after 3 blocks; `python -m green bump <txid>` does it for wallet transactions.
- **Gossip cannot point a node into its own network:** addresses from the internet must be public IP addresses.

## What's new in v0.15

- **Train a model by paying miners, instantly:** compute channels. Lock coins for a miner in one transaction, then send it matrix products directly; every result is checked locally with Freivalds' test before it is used and paid by a signed voucher, and the miner claims its earnings on chain. A 256-128-10 network trains in 300 steps in about 4 seconds on products computed by two miners ([examples/train_on_green.py](examples/train_on_green.py)).
- **Find miners on the network:** nodes that sell compute advertise signed offers over P2P; `python -m green providers` lists them.

## What's new in v0.14

- **Recovery phrases:** a wallet is 24 words you can write down (`python -m green wallet new`); `wallet restore` brings it back anywhere (BIP-39 with SLIP-0010 Ed25519 keys, the scheme hardware wallets use).
- **Private by default:** your node's transactions take a Dandelion++ stem before anyone announces them, so the network cannot tell which node sent them.
- **Compact blocks:** new blocks cross the network as a few kilobytes of short ids, rebuilt from each node's mempool.
- **Pools do useful work:** pool miners compute AI jobs, and each batch pays the miners who computed it, in consensus. Neither block rewards nor job payments ever pass through the pool.

## What's new in v0.13

- **Pools that never hold your coins:** a block can pay up to 32 miners by weight, so a pool's blocks pay its miners directly (`python -m green pool-mine --pool <url>`). Bitcoin's large pools are custodial; a Green pool cannot run off with rewards.

## What's new in v0.12

- **Light wallets:** `green.light.LightClient` checks headers (including a sample of their work) and Merkle proofs itself, so a phone wallet can verify a payment without running a node or trusting one.
- **Explorer API:** look up any transaction (`python -m green tx <txid>`) and an address's history (`python -m green history`), kept consistent with the main chain through reorgs.

## What's new in v0.11

- **Pruned nodes** (`--prune`): keep recent block bodies and the data still in use, delete the rest after a grace period, and give the space back to the disk; a pruned node still helps new nodes join by fast sync. Archive nodes (the default) keep everything. See [OPERATIONS.md](OPERATIONS.md).

## What's new in v0.10

- **For developers:** a stable `/v1` API, fee estimates (`python -m green fee`), and long-polling for new blocks so clients and agents react at once instead of polling.
- **For operators:** a config file (`green.toml`), log levels and rotated log files, a hardened systemd unit, Prometheus alert rules, and [OPERATIONS.md](OPERATIONS.md) with backup, restore and upgrade procedures.

## What's new in v0.9

- **Multisignature accounts:** k of n keys (up to 16) spend together, each cosigner signing on their own device; the address looks like any other. A guardian of an autonomous agent can now be a 2-of-3 account instead of a single key.

## What's new in v0.8

- **Upgrades without splitting the network:** rule changes activate by miner signalling in the block version (90% of a ~3.5-day period), as in Bitcoin's BIP 9/BIP 8; nodes that do not upgrade keep following the chain, and they are warned when an unknown change is being signalled. `python -m green deployments` shows what is scheduled.
- **Signed releases:** `python -m green release verify SHA256SUMS a.sig b.sig` checks that enough maintainers signed a release and that no file was changed. The whole process is in [UPGRADES.md](UPGRADES.md).

## What's new in v0.7

- **Encrypted peer-to-peer network:** X25519 key exchange, ChaCha20-Poly1305 on every frame, node identities that can be pinned, announce-then-fetch relay (every transaction crosses each link once), address gossip with eclipse protections, peer bans. The old plain-HTTP gossip is gone.
- **Headers-first sync from many peers:** headers are checked (with a random sample of their work recomputed) before any block is fetched; blocks come from several peers at once and their work is checked in parallel.
- **Every header commits to the ledger** (`state_root`, a lattice hash updated per block), so nodes that disagree on any balance fail at that block, and **fast sync** starts a new node from a snapshot verified against that root.
- **Work verification 1.9× faster on one core** (bit-identical). Full sync re-executes about 144 blocks/s on this machine (v0.6: about 41 with the same benchmark), fast sync is about 3× faster still (`bench/sync_bench.py`).
- **Measured on an 8-node network** (`bench/net_sim.py`): a block reaches every node in a median of 0.03 s; after a network split the halves rejoin one chain with identical ledgers within about a second.
- This is a consensus change (a new header field): networks started with an earlier version are not compatible.

## What's new in v0.6

- **Per-block difficulty (ASERT):** blocks stay at 15 s and issuance on schedule even when miners come and go; the old rule drifted 9.3 hours in 6,000 blocks in that case.
- **An off switch for autonomous agents:** a guardian can cap an agent's spending and freeze it, enforced by consensus; the agent can only tighten its own guard.
- **Operations:** supply audit, Prometheus metrics, Docker image, documented API, and [PRODUCTION.md](PRODUCTION.md).

## What's new in v0.5

- **A ledger that scales:** each block's state is a thin diff layer over a flat base, so a block costs the same with 10 or 1,000,000 accounts. With 1M accounts, memory held by 20 blocks fell from 587 MB to 0.1 MB.
- **Atomic SQLite storage:** one transaction per block, crash-safe; a restart takes 0.04 s at any chain length (v0.4 re-executed the whole chain on every start).
- **A consensus fix:** the reorg-depth rule now checks where a fork leaves the main chain; v0.4 let a side chain forked long ago keep growing.
- Benchmark: `python bench/state_bench.py`.

## What's new in v0.4

- **Whole jobs with two transactions.** A batch is a bitmap over the job's units plus the Merkle root of their results. On-chain data per delivered unit fell from 262,144 B (v0.2) to 44 B (v0.3) to **0.49 B**; the network's delivery ceiling rose from 16.7 to **68,267 units/s**.
- **A safer dispute game.** Four holes found in v0.3, including one that let a miner be paid for data it withheld, are closed (SECURITY.md, "Fourth review").
- **Checksummed addresses** (`grn1...`, bech32m): a typo is refused instead of burning coins.
- **A real fee market:** blocks filled by fee per byte, mempool eviction, replace-by-fee.
- **A written protocol specification** ([SPEC.md](SPEC.md)), an installable package with a `grn` command, an MIT license and CI.

This is a consensus change: networks started with an earlier version are not compatible.

## Setup

Python 3.11 or newer. Install Green into a virtual environment; this also installs the `grn` command (a shortcut for `python -m green`):

```bash
git clone <this repository> green && cd green
python -m venv .venv
.venv/bin/pip install -e ".[dev,gpu]"          # Windows: .venv\Scripts\pip install -e ".[dev,gpu]"
```

Then activate it (`source .venv/bin/activate`; on Windows `.venv\Scripts\Activate.ps1`).

`wgpu` (the `gpu` extra) is optional; without it Green mines on the CPU. `wasmtime` is required and pinned, because WASM adjudication is part of consensus.

## Mining the first blocks

**1. Create a wallet.** It shows a 24-word recovery phrase: write it on paper and keep it offline. The phrase alone restores the wallet on any machine (`python -m green wallet restore`); the wallet file itself is encrypted with a password:

```bash
python -m green wallet new
```

**2. Start the network: mine the genesis block.** The message is written into the genesis block forever (like Bitcoin's newspaper headline):

```bash
python -m green genesis --msg "Your message"
```

**3. Start the node with mining** (add `--gpu` for the GPU):

```bash
python -m green node --mine --gpu
```

Every block found prints `*** BLOCK FOUND`. `Ctrl+C` stops it; on restart the chain is loaded from disk.

## From another terminal

```bash
python -m green status
```
```bash
python -m green balance
```
```bash
python -m green send <address> 1.5
```

## Buying compute: matrix multiplication

```bash
python -m green job submit --size 512 512 512 --price 0.01
```
```bash
python -m green job list
```
```bash
python -m green job get <job-id>
```

You can also send your own int8 matrices with `--a A.npy --b B.npy`. `job get` downloads the result, checks every unit with Freivalds and saves it as `.npy`.

Your node checks every result delivered to your wallet's jobs and disputes wrong or withheld ones on its own (the buyer watchtower). To dispute units by hand (any number from one batch):

```bash
python -m green job challenge <job-id> 0,1,0 0,1,1
```

It checks the units locally first, so you do not pay the dispute fee for correct results. If the miner cannot prove every disputed unit before its deadline (20 blocks, longer when many disputes are open), the bond of its whole batch goes to you and the units reopen for other miners. Undisputed batches are paid after a 40-block window (~10 min).

Addresses look like `grn1trzpxwy4uqzw383a42sm6vnr5taudcqxae6wmz`; a mistyped one is rejected.

## Buying compute: any program (WASM)

A deterministic WASM module must export `green_run(in_ptr, in_len) -> i64` (packed out_ptr/out_len) and `memory`, and must import nothing.

```bash
python -m green task post program.wasm input.bin --fuel 100000000 --price 0.05
```
```bash
python -m green task list
```
```bash
python -m green task get <task-id>
```
```bash
python -m green task challenge <task-id>
```

A mining node also runs open tasks to earn rewards, and acts as a **watchtower** that re-runs every revealed result locally: when it sees a wrong result it challenges it on its own and takes the cheater's bond. `task challenge` does the same by hand, after checking locally first so you do not lose your bond on a wrong challenge.

Note: adjudication must give the same answer on every node, so the wasmtime version is pinned (`requirements.txt`); a node with another version will not start.

## Keeping an autonomous agent accountable

An agent's wallet can answer to a person. The agent (or anyone setting it up) names a guardian and a spending limit per period:

```bash
python -m green --data agent guard set <guardian-address> 0.5 5760
```

That allows at most 0.5 GRN per 5,760 blocks (about a day). The agent can only make this stricter. The guardian can loosen it, freeze the account entirely, hand the role to someone else, or remove it:

```bash
python -m green --data guardian guard update <agent-address> --freeze
```
```bash
python -m green guard show <agent-address>
```

A frozen account can send nothing; incoming coins are never blocked. The agent SDK respects its guard on its own: it waits when its allowance is used up and stops when frozen.

## Training a model by paying miners

The job market above settles in blocks, which is right for big batch jobs but far too slow for training, where every step needs the result of the last. For that there are **compute channels**: the buyer locks coins for one miner (`channel_open`, one block), then sends products straight to it. Every result is checked on arrival with Freivalds' test (milliseconds, and a wrong result passes with probability below 2^-120), so correctness needs no chain; payment is a signed voucher for the running total, which the miner claims on chain when it likes. What it never claims returns to the buyer when the channel expires.

```python
from green.compute import ComputeCluster
from green.train import MLP, digits, train

cluster = ComputeCluster.from_market("http://127.0.0.1:8545", my_key, amount_each=10_000_000, count=3)
x, y = digits(8000)
model = MLP([256, 128, 10])
train(model, x, y, steps=300, matmul=cluster.matmul)     # every product computed by miners, checked here
cluster.settle()
```

Miners sell compute with `grn node --mine --provide-port 8548 --provide-host 0.0.0.0 --provide-url http://<public address>:8548`; their offers spread over the network once their wallet holds 0.1 GRN. `python examples/train_on_green.py` runs the whole thing on a local network.

## Mining in a pool

A pool operator adds `--pool-port 8547 --pool-host 0.0.0.0` to its node. Miners point at it:

```bash
python -m green pool-mine --pool http://<pool host>:8547 --address <your address>
```

Every block the pool finds pays the miners of its recent shares by weight, in the block itself; the pool keeps its fee (1% by default) and never holds anyone's coins. When buyers have posted jobs, the pool also hands its miners units to compute; each batch of results pays the miners who computed it when it settles (the pool posts the bond from its own balance).

## Accounts that need several signatures

Three people (or devices) make a 2-of-3 account from their public keys (`python -m green wallet show` prints yours):

```bash
python -m green multisig address 2 <key1> <key2> <key3>
```

Anyone can pay that address. To spend, one of them writes the transaction, two sign it in turn with their own wallets, and anyone sends it:

```bash
python -m green multisig new 2 <key1> <key2> <key3> --to <address> --amount 1.5 --out pay.json
```
```bash
python -m green --data alice multisig sign pay.json
```
```bash
python -m green --data bob multisig sign pay.json
```
```bash
python -m green multisig send pay.json
```

A multisig address can be an agent's guardian, so freezing the agent takes the agreement of its holders, and one stolen key cannot loosen its limits.

## Operations

```bash
python -m green supply
```

checks that every coin is accounted for: balances plus escrow, bonds and channel funds plus the reserve equal the fixed supply, exactly. `GET /metrics` serves Prometheus metrics. Running a node as a service, its settings file, monitoring, backups and upgrades: [OPERATIONS.md](OPERATIONS.md). A Docker image: see the comments at the top of the [Dockerfile](Dockerfile).

## Running a neural network on the network

Runs the forward pass of a quantized (int8) neural network on Green: each layer's matrix multiplication is posted as a job, miners compute and deliver it, the client verifies every unit, and the activation in between runs on the client. The result is bit-identical to the local reference.

With a mining node running:

```bash
python examples/mlp_demo.py --node http://127.0.0.1:8545 --data data
```

This shows the point of PoUW: the work that secures the chain is a working neural network. The SDK is in [green/nn.py](green/nn.py).

## Autonomous agent

An agent that exists on the network with its own wallet, earns with its idle compute and spends its earnings on its own AI work. No human in the loop: it mines, watches its balance, has an inference done by the network when it has money above its reserve, and otherwise keeps earning.

A live demo that starts its own network in one file:

```bash
python examples/autonomous_agent.py
```

Or connected to a running node (with a node mining with this wallet):

```bash
python -m green agent --goals 3 --reserve 2
```

The heart of the vision: autonomous AIs can live on the network on their own. The SDK is in [green/agent.py](green/agent.py).

## A second node on the same computer (network test)

The first node prints its P2P address when it starts (`P2P: 127.0.0.1:8546 -- other nodes connect with --peer <id>@127.0.0.1:8546`). Start a second one with its own ports:

```bash
python -m green --data data2 node --port 8547 --p2p-port 8548 --peer 127.0.0.1:8546 --genesis data/genesis.json --mine
```

The two nodes set up an encrypted connection and sync the chain. With `--peer <id>@host:port` a node also checks the other's key, so nothing in between can pose as it. `python -m green peers` lists the connections (with session ids you can compare on both ends).

To be reachable by other computers while keeping the API private, listen on all interfaces for P2P only: `--p2p-host 0.0.0.0`.

**Joining an existing network quickly:** a new node started with `--fast-sync` downloads the header chain, then the newest snapshot of the ledger that another node offers, checks it against the state root in that block's header, and validates every block after it. Without a usable snapshot it syncs from genesis.

## Tests

```bash
python -m pytest -q tests
```

264 tests: consensus, difficulty, guarded accounts, security regressions, addresses, the fee market, the state engine and storage (restarts, crashes, reorgs across base moves, a differential check against full copies), the state commitment, snapshots and fast sync, rule changes by version bits, signed releases, multisignature accounts, pruning, the transaction index and the light client, non-custodial pools, recovery phrases, Dandelion++ and compact blocks, compute channels and training, port mapping, the reserve and shares, transaction atomicity, the P2P protocol (encryption on the wire, identity pinning, relay, headers-first sync, and what happens to peers that lie or flood), the network (watchtowers, dispute answers and reorgs included), the GPU (if present), the WASM sandbox, both compute markets, neural-network inference, the autonomous agent, the app and its compute proxy, and a local network of five node processes started and stopped as a user does it. Two fuzz tests check conservation of money, the exact escrow of every job, the dispute backlog and transaction atomicity at every block, and every block any test builds checks its incremental state root against a full recomputation; mutation testing confirmed they catch injected accounting bugs. CI runs the suite on Linux, Windows and macOS ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Project layout

| File | Contents |
|---|---|
| `green/work.py` | Proof of useful work: noisy matrix multiplication, transcript digest, ticket, Freivalds over a prime field |
| `green/gpu.py` | WebGPU miner (matmul + digest on the GPU, bit-identical to the CPU) |
| `green/wasm.py` | Deterministic, sandboxed WebAssembly runner |
| `green/nn.py` | Quantized neural-network SDK (layers computed on Green, every unit verified, wrong ones disputed) |
| `green/agent.py` | Autonomous agent: earn/spend decision loop |
| `examples/mlp_demo.py` | MLP forward pass on the network |
| `examples/autonomous_agent.py` | Autonomous agent demo that starts its own node, earns and spends |
| `green/chain.py` | Blocks, ledger, validation, difficulty, fork choice, job and task markets |
| `green/tx.py` | Transaction types; job, result and manifest data formats |
| `green/node.py` | HTTP API, mempool, miner, batch delivery, watchtowers, snapshots |
| `green/p2p.py` | Encrypted P2P protocol: handshake, relay, address book, headers-first and fast sync, bans |
| `green/verify.py` | Worker processes that check work units in parallel during a long sync |
| `green/localnet.py` | A network of five people on one computer (`grn localnet`) |
| `green/app.py`, `green/app.html` | The app in the browser and its compute proxy (`grn app`) |
| `green/snapshot.py` | Ledger snapshots: writing, checking against a header's state root, rebuilding |
| `green/versionbits.py` | Rule changes by miner signalling: deployment states per period |
| `green/release.py` | Signed releases: SHA256SUMS, maintainer signatures, verification |
| `green/prune.py` | Pruned nodes: which data is still in use, two-phase collection of the rest |
| `green/light.py` | Light client: verified headers and payment proofs without a full node |
| `green/pool.py` | Non-custodial mining pool (server and miner); pool miners compute jobs and are paid per unit |
| `green/hd.py` | Recovery phrases (BIP-39) and key derivation (SLIP-0010 Ed25519) |
| `green/compute.py` | Compute channels: providers selling products, buyers checking and paying per result |
| `green/train.py` | Training neural networks on int8 products bought through channels |
| `green/portmap.py` | Port forwarding on the home router (PCP, NAT-PMP) |
| `UPGRADES.md` | How Green changes its rules and ships releases |
| `OPERATIONS.md` | Running a node: settings, service, monitoring, backup, upgrades |
| `contrib/` | systemd unit and Prometheus alert rules |
| `green/cli.py` | Command line |
| `green/params.py` | Network constants (changing them = a new network) |
| `green/crypto.py` | Hashing, Ed25519, bech32m addresses, Merkle trees and proofs, wallet encryption |
| `green/state.py` | The ledger as copy-on-write diff layers over a flat base, and its lattice-hash commitment |
| `green/store.py` | SQLite storage: blocks, main chain, state diffs, base state |
| `bench/state_bench.py` | How block cost, memory and restart time scale |
| `bench/difficulty_sim.py` | Difficulty algorithms under hashrate shocks and switching miners |
| `bench/sync_bench.py` | How fast a new node catches up: full sync versus fast sync |
| `bench/net_sim.py` | Eight nodes on this machine: propagation, a network split and its healing |
| `PRODUCTION.md` | Status of every production area, with evidence and blockers |
| `API.md` | The node's HTTP API |
| `Dockerfile` | Container image for a node |
| `SPEC.md` | The consensus rules, independent of the code |

> This is a prototype and has not been independently audited. An audit is mandatory before any network carries real value.
