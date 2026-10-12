# Green (GRN): design of a Proof of Useful Work

## 1. The idea

Bitcoin miners compute billions of SHA-256 hashes that are good for nothing except securing the chain. Meanwhile AI needs enormous amounts of compute, and that compute is concentrated in a handful of companies.

In Green, miners do **integer matrix multiplication, the core workload of AI**, instead of hashing:

- Anyone who wants compute (a person or an autonomous AI agent) posts a job and pays for it in GRN.
- Miners split the job into units and compute them. Every unit is both a real result delivered to the buyer and a ticket in the block lottery.
- When nobody is buying, miners multiply random "filler" matrices (like Bitcoin); when demand arrives, the same work becomes useful.

On top of that there is a second market for **any computation that is not a matrix multiplication**: the buyer publishes a deterministic WebAssembly program and an input, workers run it, and the result is verified by re-execution (§7). Some buy matrix products, some run any program they like.

## 2. The hard part: Bitcoin's "wasted" work is the security

Hashing is useless on purpose. A PoW puzzle has to have these properties:

| Requirement | Why | SHA-256 | Green (matrix) |
|---|---|---|---|
| Expensive to solve, cheap to verify | Everyone checks every block instantly | 1 hash | The winning unit is recomputed once (~15 ms) |
| Not precomputable | Work cannot be stockpiled | Bound to the previous block hash | Seed bound to the header |
| No shortcut, every try costs the same | A fair lottery | Yes | Noise rank >= tile size |
| Adjustable difficulty | Steady block time | Target | Target |
| Deterministic | Everyone sees the same result | Yes | Integer arithmetic |

Taken literally, "miners do AI work" breaks in three places: (1) verifying AI work is expensive; (2) a miner can post fake work and pay the reward to itself; (3) floating point gives different results on different hardware. Green solves all three.

## 3. The core: Proof of Useful Work with a noisy transcript

A simplified construction inspired by Komargodski, Schen and Weinstein, *"Proofs of Useful Work from Arbitrary Matrix Multiplication"* (2025):

1. `seed = H(header)` binds the previous block, the transactions, **the miner's address** and the nonce.
2. Low-rank ternary noise is derived from the seed: `E = U·V`, `F = U'·V'`.
3. The miner multiplies `A' = A+E` by `B' = B+F` in 32×32 tiles; **every tile product** goes into the transcript.
4. `ticket = BLAKE2b(seed ‖ digest(transcript))`. The block is valid if `ticket <= target` (Bitcoin's `hash <= target`).
5. The useful result is recovered cheaply: `A·B = A'B' − A·F − E·B'` (the correction terms are low rank).

**Determinism (pitfall 3):** inputs are int8 and the noise is rank-32 ternary, so `A'`, `B'` and every tile product are integers (|value| < 200,000, fits in int32). Integer arithmetic is bit-for-bit identical on every device.

**No gain from buying your own job (pitfall 2):** the noise rank is at least the tile size (`NOISE_RANK >= BLOCK`), so knowing A·B in advance does not make the transcript cheaper. Every ticket costs one full multiplication, whoever owns the matrices.

**Cheap block verification (pitfall 1):** a validator only recomputes the winning ticket (one unit).

## 4. Delivering results: whole jobs, optimistic settlement, fraud proofs

Version 0.2 had every node check every delivered unit with Freivalds and store its 256 KB result. Measuring it showed two problems that capped the useful work far below the network's hashrate:

- **Checking was not cheap.** Freivalds with 64 rounds of {0,1} vectors costs 6.45 ms per unit, about 70% of simply recomputing it (9.1 ms).
- **Delivery did not scale.** Each unit took two transactions (commit + reveal) and every node had to download its 256 KB result. With 500 transactions per block the whole network could deliver at most 250 units per block, 16.7 units/s, while a single RX 9070 XT computes ~700 units/s. Even with unlimited demand at most ~2% of one GPU's work could become useful.

Version 0.3 moved verification to the buyer; version 0.4 lets a whole job be delivered with two transactions and closes the holes found in the v0.3 dispute game. This is principle 3 of the manifesto, "don't trust, verify", applied to delivery: **the buyer verifies, the chain only adjudicates fraud.**

### 4.1 Freivalds over a prime field

Each delivered unit is checked by Freivalds: for random `r`, `A·(B·r) = C·r`. Green works modulo the prime `p = 2^31 − 1` with `r` drawn from all of Z_p, not from {0,1}. A wrong result survives one round with probability <= 1/p instead of 1/2, so 4 rounds give **2^-124** where 64 rounds gave 2^-64. Cost: **0.37 ms per unit, 17× faster**, about 25× cheaper than recomputing.

Reducing mod p is only sound if a wrong result stays wrong mod p. A cheater could deliver `C = AB − p·E` (equal to AB mod p) and pass every round. So `|C|` is first checked against the largest value `A·B` can hold (`k · max|A| · max|B|` = 2^22 for a unit); then `|AB − C| < p` and a nonzero difference stays nonzero mod p (`test_S20_freivalds_cannot_be_fooled_by_modular_wraparound`).

### 4.2 Batches: a bitmap and a Merkle root

A miner sells any set of a job's units, up to the whole job (4,096 units), as one batch:

1. **commit:** `job_commit {units, commitment}`. `units` is a bitmap over the job (512 bytes for the largest job), `commitment = H(miner, job, units, root)` and `root` is the Merkle root of the units' 32-byte result hashes. The miner locks a bond per unit; the units are reserved for it for `RESERVE_BLOCKS`. The commitment is also the batch id.
2. **reveal:** `job_reveal {batch, root, manifest}` in a later block. Validators check the root against the commitment and start the challenge window. They fetch **nothing**: the manifest (the list of result hashes) is only a hint telling the buyer where to find it.

Because the commitment contains the miner's address, nobody can steal results from the mempool: a thief who sees the root can neither reveal someone else's batch nor commit the reserved units.

### 4.3 Optimistic settlement and disputes

The buyer fetches the manifest and the results straight from the miner's node, checks that the manifest hashes to the revealed root, and checks every unit with Freivalds.

- No dispute during `JOB_CHALLENGE_BLOCKS` (40 blocks, ~10 min): the miner is paid price + bond for every unit.
- **Disputes:** anyone (in practice the buyer) sends `job_challenge {batch, units}` naming **any number** of units at once (a bitmap), paying a fee that covers the bytes validators will fetch for each one. There is no challenger bond.
- The miner must answer **every** disputed unit with `job_respond {unit, result, proof}`: the unit's result hash, its Merkle proof against the root (at most 12 hashes), and the result blob. Only then does every validator fetch that one result and check it with Freivalds. A proven unit cannot be disputed again.
- **One unanswered unit forfeits the whole batch:** its bond goes to the challenger and its units reopen for other miners. If every disputed unit is answered, the batch settles normally.
- **Capacity:** at most `MAX_BLOCK_JOB_RESPONSES` (32) answers fit in a block (~8 MB of results). Each new dispute's deadline is pushed back by `open disputes / 32` blocks, so there is always room to answer every open dispute before its deadline.

Why these rules (each closes a hole found in v0.3; see SECURITY.md, "Fourth review"):

- **Parallel disputes** stop *partial withholding*. With one dispute per batch, a miner could withhold most of a batch, publish only the unit someone disputed, and be paid in full for data the buyer never got.
- **No challenger bond** makes withholding pointless. When the bond went to a miner that answered, withholding data until disputed was a way to earn the buyer's bonds. Now answering costs the miner a fee and earns nothing; griefing an honest miner costs the griefer ~27× more (the data fee) than the miner (the response fee).
- **Deadlines that stretch with the backlog** stop *capacity griefing*: with a fixed deadline, a flood of disputes could fill the response budget so honest miners missed their deadlines and lost their bonds to the attacker. Now a flood can only delay settlement.

Who checks: `green.nn.MatmulClient` verifies every unit before using it and disputes wrong or missing ones, and every node runs a **buyer watchtower** for its own wallet's jobs. Since a result is used only after it passed the check, the buyer never waits for the window to close.

**Security assumption:** the buyer, or a watchtower acting for it, is online at some point during the challenge window, as with Lightning and optimistic rollups. A buyer that goes offline for the whole window can be paid in garbage.

**Known cost:** a miner that withholds a whole manifest gains nothing, but the buyer has to dispute every unit to get the data published, paying the data fee for each. Real data-availability guarantees without that cost (erasure coding with sampling) are future work.

### 4.4 What changed, measured

| | v0.2 | v0.3 | v0.4 |
|---|---|---|---|
| Freivalds per unit | 6.45 ms, error 2^-64 | 0.37 ms, 2^-124 | 0.37 ms, 2^-124 |
| Transactions to deliver a 4,096-unit job | 8,192 | 32 | **2** |
| On-chain data per delivered unit | 262,144 B | 44 B | **0.49 B** |
| Delivery ceiling (network) | 16.7 units/s | 4,267 units/s | **68,267 units/s** (~98 RX 9070 XT) |
| Partial withholding | n/a (all on chain) | possible | prevented |

The ceiling is now set by the transaction count (250 batches per block), and each batch can be a whole job.

## 5. Economics and the "wasted work" problem

- **Block reward (lottery):** 1.25 GRN, halving every ~4 years (8.4M blocks), ~21M GRN in total. The same pace and supply as Bitcoin.
- **Job payment:** the per-unit price set by the buyer, paid from escrow to the miner when the batch settles.
- **Transaction and data fees:** go to the block's miner; the per-byte data fee prevents storage spam.
- **Block size:** at most 1 MB of transactions per block (`MAX_BLOCK_BYTES`), which bounds what every node must download and relay, like Bitcoin's limit.
- **Fee market:** block space is sold by fee per byte. Miners fill blocks with the best-paying transactions while keeping each sender's nonce order; a full mempool evicts the cheapest transaction instead of refusing new ones; a stuck transaction can be replaced by paying at least 10% more (replace-by-fee); one sender can have at most 100 pending transactions.

**The honest framing:** a GPU computes thousands of units per second. `_choose_work` skips units that are already delivered or reserved, so a job is computed and delivered roughly **once** across the network, and once the jobs run out the remaining hashing goes back to filler. So:

- When demand >= hashrate (up to the delivery ceiling in §4.4), almost all work is useful.
- When demand < hashrate, the surplus is filler (as in Bitcoin), but the work **has the shape of AI** (same hardware, same energy), and all demanded compute is served by the very work that secures the chain, with no separate spending. As demand grows the waste shrinks.

The lottery (every unit is a ticket) and delivery (batched commit-reveal) are separate: a miner sells a bounded number of units at a time (`COMMIT_INFLIGHT_CAP`); the rest are tickets only.

## 6. GPU miner

Matrix multiplication is what GPUs do best, but a naive speedup does not help. The transcript that PoUW hashes is proportional to the multiplication work (`UNIT³/BLOCK`), so **the hashing must happen on the GPU too**. Otherwise moving the 2 MB transcript to the CPU for every nonce slows everything down (measured: ~1.2×).

The solution is a transcript digest that gives bit-identical results on CPU and GPU and parallelizes well: a u32 function with MurmurHash3-style per-element nonlinear mixing and order-independent XOR, 256 bits wide. The GPU kernel computes each tile-product element, mixes it and accumulates it with a workgroup-local atomic XOR; only a 32-byte digest goes back to the node, and the 2 MB never crosses PCIe. The useful result is just `a1@b1`, which the CPU gets cheaply from BLAS.

Measured (AMD RX 9070 XT, WebGPU/Vulkan): CPU ~50–115 units/s, GPU ~600–820 units/s, **~12–17×** depending on hardware and bottleneck. The remaining bottleneck is CPU-side noise generation (SHAKE) and per-nonce wgpu call overhead; generating the noise on the GPU and batching nonces would give more (future work).

**Safety:** a GPU that produced a different result would spend its work on blocks every other node rejects. So the node checks the GPU bit-for-bit against the CPU at startup and falls back to the CPU on any mismatch. The security analysis of using a custom transcript digest instead of BLAKE2b is in [SECURITY.md](SECURITY.md).

## 7. General compute market (WebAssembly)

A matrix unit can be checked cheaply with Freivalds, but an **arbitrary program** cannot be checked without running it. The solution is to make execution **deterministic**: the same program + input + fuel gives the same output on every machine, so any node can verify a result by re-running it.

- **Sandbox (wasmtime):** no WASI and no host imports. Guest code has no clock, no randomness, no I/O, no threads, and cannot leave the sandbox.
- **Determinism:** fuel metering (turns infinite loops into a trap), NaN canonicalization, relaxed SIMD disabled.
- **Resource limits:** fuel bounds CPU, a memory cap bounds memory.
- **Market (optimistic, with fraud proofs):**
  1. The buyer sends `task_post`: program + input, fuel, price. The program and input are **stored by every node** (paid per byte), so fraud can always be proven even if the buyer goes offline. The program is validated at post time without compiling it (no imports, the right exports).
  2. A worker runs the program and sends `task_commit` with a commitment to the output hash and a bond.
  3. `task_reveal` reveals the output and opens the challenge window.
  4. During the window **anyone** can trigger on-chain re-execution with `task_challenge`. If the result is wrong, the worker's bond goes to the challenger and the escrow back to the buyer; if it is right, the challenger's bond goes to the worker and the result is **final** (it cannot be challenged again).
  5. When the window closes without a challenge (or with a confirmed result), the worker is paid.
- **Adjudication is bounded, because it loads the whole network:** at most one re-execution per task; a per-block adjudication budget (like Ethereum's gas limit; cost = fuel + program bytes × compile cost); and the challenger pays a fee proportional to that cost.
- **Watchtower:** an optimistic system is only safe if someone actually checks. Every node re-runs revealed results locally, challenges only the results it has proven wrong itself, and earns the cheater's bond. Honest results never reach on-chain adjudication.

Adjudication must give the same answer on every node. "Did the program finish within its fuel?" depends on fuel accounting, so the wasmtime version is a consensus constant (`params.WASMTIME_VERSION`, like Ethereum pinning the EVM) and a node on another version refuses to start. Measurements and limits are in [SECURITY.md](SECURITY.md).

## 8. The chain

- **Account model:** balance + nonce (like Ethereum). Signatures cover the network id (the genesis hash).
- **Signatures:** Ed25519. **Hash:** BLAKE2b-256. **Wallet:** scrypt + AES-256-GCM.
- **Addresses:** bech32m (BIP-350), `grn1...`. The checksum detects every typo of up to 4 characters, so a mistyped address is refused instead of sending coins to a key nobody holds.
- **Specification:** the consensus rules are written down independently of the code in [SPEC.md](SPEC.md).
- **State and storage:** each block's state is a thin copy-on-write diff layer over its parent's; layers that leave the reorg window are merged into one flat base (as in Ethereum's snapshot layers), so a block costs O(keys it changes) at any ledger size. Blocks, the main-chain index, recent diffs and the base live in SQLite (`chain.db`), one transaction per block: a crash leaves the last complete block, and a restart loads the base plus recent diffs without executing anything (0.04 s at any height).
- **State commitment:** every header carries `state_root`, a lattice hash (LtHash) of every ledger entry. Adding or removing an entry is one vector addition, so the root is updated per block in O(entries changed), independent of the order of writes. Nodes that disagree on any balance reject each other's block at once, and a snapshot of the ledger can be checked against a header instead of trusted (SPEC.md §6.1).
- **Rule changes:** soft forks activate by miner signalling in the header version (version bits, as in Bitcoin's BIP 9/BIP 8), so the network can upgrade without splitting; see [UPGRADES.md](UPGRADES.md).
- **Fork choice:** the chain with the most accumulated work wins; forks that leave the main chain more than 100 blocks below the tip are refused.
- **Difficulty:** set for every block by ASERT (aserti3-2d, anchored at genesis, half-life 30 minutes): each half-life behind the ideal schedule halves the difficulty, each half-life ahead doubles it. It keeps issuance on schedule and cannot be gamed by miners who come and go; the simulation behind the choice is in `bench/difficulty_sim.py` and SECURITY.md (Z1). Timestamps follow the median-time-past rule.
- **Accountability for autonomous agents:** an account can name a guardian who caps its spending per period and can freeze it; the account can only tighten its own guard (SPEC.md §3.4). This is the manifesto's off switch, enforced by consensus rather than by a company.
- **Block size:** at most 500 transactions and 1 MB of transaction data.
- **Big data stays out of blocks:** job matrices, results, manifests, WASM programs and inputs are content-addressed blobs shared between nodes; blocks only carry their hashes. Validators fetch job data and task programs; manifests and matmul results are only fetched by the buyer, and a result by validators when it is disputed.
- **Network:** nodes talk over a dedicated encrypted protocol (SPEC.md §9): X25519 + ChaCha20-Poly1305 with node identities, announce-then-fetch relay, address gossip with eclipse limits, and bans for invalid data. The HTTP API is only for wallets, tools and agents.
- **Sync:** headers first (a random sample of work units recomputed per message), blocks from several peers at once with their work units checked in parallel threads, and an opt-in fast sync from a ledger snapshot verified against a header's state root. A fast-synced node never needs the matrices of jobs finished before its snapshot.

## 9. Data privacy: the honest status

Goal: random miners should not learn who computes what.

| Property | Status |
|---|---|
| Buyer identity | Partly hidden: only a pseudonym (key) is visible; transactions are public |
| Distribution of work | Yes: each unit lands with a random miner |
| Confidentiality of the data (the "what") | **No:** a miner sees the matrices of the units it computes |

Real confidentiality needs secure hardware (the confidential-computing mode of data-center GPUs, a TEE), secret sharing / MPC, or fully homomorphic encryption (10^4–10^6× slower today, not practical). TEEs are the most realistic. None of this is in v0.3.

## 10. Known limitations (prototype)

- Python reference implementation: CPU + numpy and a single GPU kernel. The GPU miner reaches ~700 units/s, a small fraction of the card's int8 peak; a native miner is needed.
- The base state is held in memory (one copy; a disk-backed base loaded on demand is future work) and so is the main-chain hash index.
- The transcript digest is an analyzed custom function, not an audited hash (§6, SECURITY.md).
- Matmul delivery and WASM tasks are optimistic: they assume the buyer or a watchtower is online during the challenge window (§4.3, §7). Withholding a whole manifest costs the buyer dispute fees (§4.3).
- WASM adjudication depends on the wasmtime version (an upgrade is a hard fork); there is no data-availability proof for WASM *outputs* (§7).
- No data privacy (§9).
- The network layer has no NAT traversal, DNS seeds or Tor/I2P support yet, and the protocol can be recognised (though not read) by an observer; results reach the buyer by asking a few peers, not through a content-routing layer.
- The code has not been independently audited. That is mandatory before any network carries real value.

## 11. Roadmap

Done: ~~prime-field Freivalds and buyer-verified delivery (v0.3)~~, ~~whole-job Merkle batches and the dispute fixes (v0.4)~~, ~~checksummed addresses, fee market, replace-by-fee (v0.4)~~, ~~diff-layer state engine and atomic SQLite storage (v0.5)~~, ~~the quantized-model SDK~~ ([green/nn.py](green/nn.py)), ~~the autonomous agent~~ ([green/agent.py](green/agent.py)), ~~a written protocol specification~~ ([SPEC.md](SPEC.md)).

The complete, area-by-area status is in [PRODUCTION.md](PRODUCTION.md). Done in v0.6: ~~per-block ASERT difficulty~~, ~~guarded accounts for agents~~, ~~supply audit and metrics~~, ~~Docker image~~. Done in v0.7: ~~encrypted P2P protocol with inv/getdata relay and eclipse limits~~, ~~headers-first sync with parallel verification~~, ~~a state commitment in every header~~, ~~fast sync from verified snapshots~~. Done in v0.8: ~~version-bits activation of rule changes~~, ~~signed releases and a written upgrade process~~ ([UPGRADES.md](UPGRADES.md)).

Toward production, in order:

1. **Storage, next steps:** a disk-backed base state loaded on demand, pruning of old blobs and block bodies, a Merkle or Verkle state commitment for light clients.
2. **Network layer, next steps:** NAT traversal, DNS seeds, a Kademlia DHT for blobs so a buyer can always find a miner's data, compact blocks, Tor/I2P.
3. **Native node and miner (Rust):** an implementation of SPEC.md, a GPU miner that generates the noise on the GPU and batches nonces, and conformance tests shared with the Python reference.
4. **Data availability** for matmul and WASM outputs (erasure coding with sampling), removing the cost of manifest withholding.
5. **Wallets:** HD wallets (BIP-32/39 style seed phrases), hardware-wallet support, a JSON-RPC API.
6. Privacy: secret sharing first, then TEE attestation.
7. A public testnet, an independent security audit, and a bug bounty, before any value.
8. Legal: advice before anything carries real monetary value (e.g. in Turkey, Law No. 7518 and the Capital Markets Board).

## 12. Related projects and how Green differs

Akash, Render and io.net are compute markets with a separate consensus; Bittensor distributes rewards through validator scoring; Gensyn works on verifiable ML training. **Green's difference: the consensus itself is the useful work.** The work that secures the block and the work that is sold are the same matrix multiplication.
