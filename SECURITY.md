# Green security notes

This document records the security reviews done so far: the vulnerabilities found, the fixes applied and the known limitations. Every finding has a permanent test in `tests/` that keeps it from coming back.

> **Warning:** Green is a prototype and has not been through an independent security audit. A professional audit is mandatory before using it on a network that carries real monetary value.

## Method

Eight attack scenarios were written against the first version (v0.1); all eight succeeded. A code review then found nine more issues. All seventeen were fixed, each with a regression test.

The WASM compute market added later went through a second design review ("Second review of the WASM market" below); nine more issues were found and fixed. Decisions were made by measuring, not assuming.

Version 0.3 reworked how matrix results are delivered and went through a third review ("Third review: delivering matmul results" below). Six more issues were found and fixed, and mutation testing was used to check that the tests actually catch the bugs they are meant to catch.

Version 0.4 went through a fourth review ("Fourth review" below), which found four holes in the v0.3 dispute game, among them one that let a miner be paid for data it never delivered, plus problems in addresses and the mempool.

Version 0.5 replaced the state engine and storage and went through a fifth review ("Fifth review" below), which found a consensus bug in the reorg-depth rule.

Version 0.6 replaced the difficulty algorithm and added guarded accounts for autonomous agents (sixth review below).

Version 0.7 replaced the network layer (plain HTTP) with an encrypted, authenticated peer-to-peer protocol, added headers-first and snapshot sync, and put a commitment to the whole ledger in every block header (seventh review below).

Version 0.8 added a way to change the rules without splitting the network, and signed releases (eighth review below).

Version 0.9 added multisignature accounts (ninth review below). Version 0.10 added operator and developer tooling (tenth review below). Version 0.11 added pruned nodes (R1–R3 in the tenth review). Version 0.12 added a transaction index and a light client (L1–L2 in the tenth review). Version 0.13 added non-custodial pools (eleventh review below). Version 0.14 added recovery phrases, Dandelion++, compact blocks and pools that sell compute (twelfth review below). All 214 tests pass; two of them are fuzz tests that check conservation of money at every block (one for jobs, one for tasks), and the jobs fuzz also checks every job's exact escrow, the network-wide dispute backlog, and that every rejected transaction leaves the state untouched.

## Fixed vulnerabilities

### Consensus and ledger
- **S1: Merkle mutation (like CVE-2012-2459).** By sending a corrupted copy of a block first, an attacker could get the honest block blacklisted forever. Fix: leaves and inner nodes of the Merkle tree use different prefixes, an odd node is not duplicated, the leaf count is mixed into the root; and only *permanent* invalidity (the header itself is bad) is remembered, not a body mismatch.
- **S4: Cross-network replay.** A transaction signed on one network was valid on another. Fix: signatures cover the network id (the genesis hash).
- **S9: Timestamp manipulation.** Fix: median-time-past rule; a block's time must be above the median of the previous 11 blocks.
- **S10: Blacklisting future blocks.** A block that is only invalid *for now* (time too far ahead) is no longer treated as permanently invalid.
- **S11: Long-range forks.** Fix: forks that branch off deeper than 100 blocks are refused. (The v0.4 check measured the wrong block; see Y1.)
- **S16: Conservation of money.** A fuzz test with random transaction sequences checks that no coin is ever created or lost at any step (transfers, jobs, tasks), and that every job's escrow is exactly what it still owes.

### Compute market (job escrow)
- **S2: Result theft.** A miner could copy a result from the mempool and get paid. Fix: commit-reveal. The miner first commits to `hash(address, job, units, root)`, the units are reserved for it, and it reveals only in a later block. The commitment contains the miner's address, so nobody else can use it.
- **S3: Dodging payment by cancelling.** A buyer could see the result and cancel the job to avoid paying. Fix: a cancel takes effect only after `CANCEL_DELAY` (40 blocks), and batches committed before it keep their share of the escrow and are settled as usual.
- **S7: Free storage spam.** Large job data could be written to every node for free. Fix: a minimum fee per transaction plus a per-byte fee for job data and for the result bytes a dispute makes validators fetch.

### Network layer
- **S5: Fake peer injection / SSRF / eclipse.** A single HTTP request could fill the peer list. Fix: peers are added only after a handshake (proving with `/status` that they run the same network); a request header can never add a peer or choose where we connect. Peer URLs are parsed strictly (user/password/query/path rejected).
- **S6: Malformed Content-Length.** A negative or garbage length locked a server thread. Fix: the length is validated and sockets have timeouts.
- **S12: Admin endpoints.** Mining on/off, adding peers and uploading blobs only work from 127.0.0.1.
- **S13: Leaking unrevealed results.** Committed but unrevealed results are not served to the network.
- **S14/S17: Response/request size limits.** Every network read has an upper bound.
- **S15: Mempool overflow.** The mempool size is bounded.
- **S18: Connection floods.** The number of concurrent connections is bounded; the rest are dropped.
- **S19: Cheap checks first.** Data (blob) downloads happen only after a transaction passes the cheap checks (signature, balance, fee), so an invalid transaction cannot make a node download data.

### Economics / wallet
- **S8: Emission schedule bug.** A calculation error minted 99% of the supply in ~8 months. Fix: block reward 1.25 GRN, halving every ~4 years, ~21M GRN in total (the same pace as Bitcoin). `test_emission_matches_bitcoin_schedule` checks it.
- **Wallet encryption.** The secret key is now encrypted with a password (scrypt + AES-256-GCM). The unencrypted mode is only available with `--no-password` and an explicit warning.

## Integrity of the Proof of Useful Work

- **You cannot steal a ticket:** the seed binds the header, the miner's address included, so another miner cannot claim a valid ticket. (`test_tampered_blocks_are_rejected`)
- **Buying your own job gives no advantage:** the noise rank is at least the tile size (`NOISE_RANK >= BLOCK`), so knowing A·B in advance does not make the transcript cheaper. Every ticket costs one full multiplication.
- **A wrong result is never paid:** a buyer or watchtower that checks it disputes it, the miner cannot produce a valid answer (the result must sit under the committed Merkle root, and every validator runs Freivalds; a wrong result passes with probability <= 2^-124), and the whole batch is forfeit. (`test_wrong_result_is_disputed_and_the_whole_batch_is_forfeit`, `test_buyer_watchtower_catches_a_cheating_miner`)

### About the transcript digest (a design decision for the GPU)

To make GPU mining possible, the transcript digest is computed with a parallel u32 function that gives bit-identical results on CPU and GPU (MurmurHash3-style, per-element nonlinear mixing plus order-independent XOR accumulation, 256 bits) instead of BLAKE2b. The final ticket still goes through BLAKE2b (`ticket = BLAKE2b(seed || digest)`).

Security argument: the digest's role in PoW is to bind the miner to actually computing every tile product. Per-element nonlinear mixing rules out linear shortcuts such as "sum = row sum · column sum", so every product value is genuinely needed to produce the digest. The protocol inputs (seed, A, B) determine the products, so the miner has no freedom to choose them. Validators recompute the winning block's transcript from scratch, so there is no "trust the digest" step.

This is a deliberate trade of an audited hash for a custom function: it has been analyzed but not proven. The conservative alternative is to compute BLAKE2b directly on the GPU (future work).

### GPU correctness

A GPU that produced a different result than the CPU would spend work on blocks every node rejects. To prevent that, the node runs a self-test at startup that compares the GPU bit-for-bit with the CPU; on a mismatch the GPU is disabled and mining continues on the CPU. (`green/node.py::_init_gpu`, `tests/test_gpu.py`)

## General compute (WASM) security

- **Sandbox:** wasmtime runs without WASI and without host imports. Guest code has no clock, no randomness, no I/O, no threads, and cannot leave the sandbox. A module that imports anything is rejected.
- **Determinism:** fuel metering (infinite loops trap), NaN canonicalization (the only float nondeterminism), relaxed SIMD disabled. The same program + input + fuel gives the same output on every machine, which is what makes verification by re-execution possible.
- **Resource limits:** fuel bounds CPU, a memory cap (256 MB by default) bounds memory.
- **Fraud game:** the worker delivers the result with commit-reveal and a bond. During the challenge window anyone can prove a wrong result by on-chain re-execution and take the worker's bond; a wrong challenger's bond goes to the worker.

### Second review of the WASM market

In an optimistic market, *every validator* pays for adjudication. So the real question is "how much work can an attacker make the whole network do, for how little money?" Measured first (wasmtime 49, i5-14600KF):

| Measurement | Result | Conclusion |
|---|---|---|
| Simple loop | ~0.03 ns/fuel | fuel bounds time well |
| Cache-missing random memory access | **~0.7 ns/fuel** (worst observed) | 2 billion fuel ≈ 1.4 s |
| `memory.fill` (bulk memory) | 1 fuel / byte | wasmtime charges bulk ops correctly; the concern was unfounded |
| Module parsing | ~1 ms / MB | safe to do in consensus |
| Module compilation | **~110 ms / MB** | not metered by fuel, must be bounded separately |

Findings and fixes (`tests/test_compute.py`, `tests/test_network.py`):

- **W1: Unlimited challenges of an honest result (consensus DoS).** A result could be challenged again after surviving a challenge. If the worker and challenger were the same person, the challenge bond came back, so each challenge cost only the fee; 500 challenges per block × ~1.4 s ≈ **~12 minutes per block per validator**. Fix: a result that survives a challenge becomes `confirmed` and cannot be challenged again; at most one re-execution per task.
- **W2: No per-block adjudication limit.** Fix: a `MAX_BLOCK_ADJUDICATION` budget, like Ethereum's gas limit. Cost = `fuel + program_bytes × 200` (counting compile time too), worst case ~1 s per block. A block over budget is invalid; the block template leaves the excess for the next block, the mempool does not drop it.
- **W3: Adjudication was free.** Fix: a challenge fee proportional to the adjudication cost (`challenge_fee`), paid to the validators who re-run it.
- **W4: Program/input availability was not required.** If the buyer went offline nobody could prove fraud, and a cheating worker would be paid; the whole model depended on this. Fix: `task_post` is valid only if the program and input are available on every node (as with job data), and it pays a per-byte data fee; sizes are bounded (program <= 1 MB, input <= 4 MB).
- **W5: DoS through compile cost and invalid programs.** Fix: `task_post` validation **does not compile** the program; it only parses it (~1 ms/MB) and reads the import/export sections directly. Invalid or importing programs cannot be posted.
- **W6: The wasmtime version was not pinned.** "Did the program finish within its fuel?" depends on fuel accounting, which can change between versions, so different versions could silently split the chain. Fix: `params.WASMTIME_VERSION` is a consensus constant; a node with another version refuses to start (stopping loudly instead of diverging silently), and `requirements.txt` pins the version.
- **W7: An atomicity bug.** The challenge bond was deducted *before* adjudication; if adjudication stopped for missing data, the mempool and block template were left with a wrong balance. Fix: adjudicate first, then change state.
- **W8: Nobody challenged automatically.** An optimistic system is only safe if someone actually checks; otherwise a lying worker would always be paid. Fix: every node runs a **watchtower** that re-runs revealed results locally, challenges only the results it has proven wrong itself, and earns the cheater's bond. Honest results never reach on-chain adjudication. Security assumption: at least one honest watchtower is online during the challenge window (the same assumption as optimistic rollups).
- **W9: The worker forgot its own commit while it waited in the mempool.** With realistic block times a commit sits in the mempool for a while; the worker deleted its record in the meantime, never revealed, lost its bond and committed the same task a second time. Instant blocks in the tests hid this; it showed up in a live node's log. Fix: the worker keeps its own commit while it is in the mempool. The regression test (`test_worker_commits_once_when_blocks_are_slow`) runs with realistic block times and was confirmed to fail when the fix is reverted.

Also: transactions in blocks abandoned by a reorg now go back to the mempool (`test_reorg_returns_abandoned_transactions_to_mempool`), and a task that cannot be run is no longer retried every second.

### Known WASM limitations
- **Consensus depends on the wasmtime version.** An upgrade is a hard fork and must be coordinated.
- **Data availability of the output:** if a worker reveals the hash of the right output but never publishes the bytes, a challenge says "the worker is honest" but the buyer has no bytes. Since the program and input are on every node, the buyer (or anyone) can reproduce the output deterministically; still, in production adjudication should publish the output bytes too.
- **Fuel is not a time guarantee:** ~0.7 ns/fuel is the worst *observed* value, not a proven upper bound. The budget was chosen with a margin; it should be re-measured on other hardware.
- **Environment errors:** a node that cannot allocate memory during adjudication (at the OS level) could decide differently from the others. The 256 MB memory cap makes this unlikely but not impossible.

## Third review: delivering matmul results (v0.3)

Version 0.2 had every node check every delivered unit and store its result. The review started from the same question as the WASM review: what does each delivered unit cost every validator, and what can an attacker make the network do for little money? Measured first (numpy, i5-14600KF):

| Measurement | Result | Conclusion |
|---|---|---|
| Mining one unit (CPU, with transcript) | 14.8 ms | the cost of one ticket |
| Recomputing a unit (int64 A·B) | 9.1 ms | the cost of re-doing the work |
| Freivalds, 64 rounds of {0,1} vectors | **6.45 ms** | ~70% of re-doing it: not cheap |
| Freivalds over Z_p, p = 2^31 − 1, 4 rounds | **0.37 ms** | 17× faster, error 2^-124 instead of 2^-64 |
| Result per unit | 262,144 B | every node downloaded it |

Findings and fixes (`tests/test_chain.py`, `tests/test_security.py`, `tests/test_network.py`):

- **V1: Verification was not cheap.** "Cheap to verify" was the premise of the design, yet Freivalds with {0,1} vectors needed 64 rounds and cost 70% of recomputing the unit. Fix: Freivalds over a prime field with random vectors from all of Z_p; 4 rounds give 2^-124. (`test_S21_freivalds_is_much_cheaper_than_recomputing`)
- **V2: Modular wraparound (introduced by the V1 fix and closed with it).** Working mod p, a forged result `C = AB − p·E` equals `AB` mod p and would pass every round. Fix: `|C|` is checked against the largest value `A·B` can take (`k · max|A| · max|B|`), which keeps `|AB − C| < p`, so a wrong result stays wrong mod p. Values are widened to int64 before taking `abs`, because `abs(INT32_MIN)` overflows in int32 and would slip past the bound. (`test_S20_freivalds_cannot_be_fooled_by_modular_wraparound`)
- **V3: Bandwidth DoS for the price of fees.** Job data paid per byte, but delivered results did not, and every node had to download them. Posting a job at 1 yaprak per unit and delivering it to yourself made every node download up to 62.5 MB per block for ~0.05 GRN in fees. It also capped the whole network's useful work at 16.7 units/s, about 2% of one GPU. Fix: optimistic, batched delivery. Validators only fetch the manifest (32 bytes per unit, paid per byte) and, for a disputed unit, one result; the challenger's fee pays for those bytes and responses are capped at 32 per block (~8 MB). The ceiling rose 256× and per-unit data for every node fell 5,900× (DESIGN.md §4.4). (`test_validators_never_need_the_results`, `test_responses_per_block_are_capped`)
- **V4: Withholding.** Under optimistic delivery a miner can publish a manifest but not the results. Fix: a dispute forces a response, a response is only valid if every validator can fetch the result and it passes Freivalds, and without one in time the whole batch is forfeit to the challenger. (`test_withheld_result_cannot_be_answered`)
- **V5: Blocks had no byte limit.** Only the transaction count was bounded; batch commits are ~2.7 KB, and a large block could exceed what peers accept over HTTP (2 MB). Fix: `MAX_BLOCK_BYTES` (1 MB) as a consensus rule; the block template respects it and an oversized block is permanently invalid. (`test_block_size_is_limited`)
- **V6: The money fuzz test had blind spots.** Mutation testing (injecting a bug and checking that a test fails) showed that the conservation fuzz test never forfeited a batch or dropped one after a job closed, so it missed both injected bugs. Conservation of supply also cannot catch coins that stay locked in escrow forever. Fix: the fuzz now models a buyer watchtower and defending miners, asserts that every settlement path ran, and checks the exact escrow invariant (an open job's escrow = price × undelivered units; a closed job's escrow = price × units in live batches). Both injected bugs are now caught.

Note: V4's fix was incomplete for batches of more than one unit; see X1 below.

New trust assumption, stated plainly: **the buyer, or a watchtower acting for it, must be online at some point during the challenge window** (40 blocks, ~10 min). The SDK (`green.nn.MatmulClient`) checks every unit before using it and every node watches its own wallet's jobs, so this holds whenever the buyer's software is running. A buyer that is offline for the whole window can be paid in garbage, as in the WASM market, Lightning or optimistic rollups.

## Fourth review: the dispute game, addresses and the mempool (v0.4)

Version 0.4 delivers a whole job as one batch (a bitmap of units and the Merkle root of their result hashes; validators see only the root). Designing it meant asking of the v0.3 dispute game: what can a miner get paid for without delivering, and what can an attacker make honest miners lose? Four holes came out of that, plus problems outside the job market.

Findings and fixes (`tests/test_chain.py`, `tests/test_security.py`, `tests/test_network.py`, `tests/test_addresses.py`, `tests/test_fees.py`):

- **X1: Partial withholding (a flaw in v0.3's own fix for V4).** Only one dispute could be open per batch, and a dispute was answered by publishing that one unit. A miner could withhold 255 of 256 units, answer each dispute by publishing the disputed unit, and be paid in full for data the buyer never received; the v0.3 test used a one-unit batch and missed it. Fix: a dispute names any number of units at once, every one must be answered, and one unanswered unit forfeits the whole batch. (`test_partial_withholding_cannot_get_paid`)
- **X2: Response-capacity griefing.** Answers are capped per block (each makes every validator fetch 256 KB) but the deadline was fixed. Opening more disputes than fit in the blocks before the deadline (~640 for about 1.7 GRN in fees) made honest miners miss deadlines, and the attacker, as challenger, collected their batch bonds. Fix: a new dispute's deadline is pushed back by the network-wide backlog of open disputes divided by the per-block cap, so every open dispute always has room to be answered; a flood can delay settlement but not cause forfeits. (`test_dispute_deadlines_stretch_with_the_backlog`)
- **X3: Withholding was profitable.** A challenger posted a bond that went to the miner when it answered, so a miner could withhold data, wait for disputes and pocket the bonds, while the buyer paid to get what it had already bought. Fix: no challenger bond; a dispute costs only a fee covering the bytes validators fetch. Answering costs the miner its response fee and earns nothing, and griefing an honest miner costs the griefer ~27× more than the miner. (`test_answering_every_dispute_publishes_the_data_and_earns_nothing_extra`, `test_honest_result_survives_a_dispute`)
- **X4: Disputing the same unit forever.** After an answer, the same unit could be disputed again, delaying settlement indefinitely. Fix: a proven unit cannot be disputed again. (`test_dispute_rules`)
- **X5: Unchecked addresses.** Addresses were `g` + 40 hex digits with no checksum; a single mistyped character sent coins to an address nobody holds, irreversibly. Fix: bech32m (BIP-350) addresses, `grn1...`; the checksum detects every error affecting up to 4 characters (all single-character substitutions and adjacent swaps are tested exhaustively), and only the lowercase spelling is accepted so one account has one ledger key. Checked against the BIP-350 test vectors. (`tests/test_addresses.py`)
- **X6: Mempool flooding and lost transactions.** A full mempool refused every new transaction, so filling it with minimum-fee transactions blocked everyone else; one sender could fill it alone; a stuck transaction could not be replaced; and after every block the mempool was re-sorted by nonce, which dropped a new account's spend when it was ordered before the payment funding it. Fix: eviction of the lowest fee-rate transaction when full, at most 100 pending per sender, replace-by-fee, and rebuilding in arrival order. The chained-payment test was confirmed to fail on the old ordering. (`tests/test_fees.py`)

Mutation testing on the new design: three injected bugs (the forfeited bond vanishes, a closed job's escrow share is not refunded, the dispute backlog is not decremented) are each caught by the fuzz test.

Residual risks, stated plainly:
- **Manifest withholding still costs the buyer.** A miner that withholds a whole manifest gains nothing (answering earns nothing), but the buyer must dispute every unit and pay the data fee for each to get the data published. Proper data availability (erasure coding with sampling) is future work.
- **Censorship of responses** by a majority of block producers would make honest miners forfeit; like a 51% attack on Bitcoin, this is outside the threat model.
- **Timing:** answers must be included in blocks in time. Block producers order by fee rate, so a miner close to its deadline may need to pay more for its answer.

## Fifth review: state engine and storage (v0.5)

Version 0.5 replaced how the ledger is held (diff layers over a flat base) and stored (SQLite). The review asked what an attacker, or plain time, does to a node's memory, CPU and restart, and whether the code really guarantees what the rest of the system assumes.

Findings and fixes (`tests/test_storage.py`, `tests/test_state.py`, `tests/test_atomicity.py`):

- **Y1: The reorg-depth rule checked the wrong block (consensus).** S11 was meant to refuse forks that leave the main chain more than MAX_REORG_DEPTH blocks below the tip, but v0.4 compared only the new block's *parent* height with the tip. A side chain forked long ago could keep growing block by block next to the main chain, and once heavier it would rewrite deep history, which is exactly what the rule exists to stop. Fix: walk back to the fork point and compare its height. The new test was run against the v0.4 code and fails there. (`test_side_chain_that_forked_too_deep_is_refused_block_by_block`)
- **Y2: Memory and CPU grew with the number of accounts.** Every block copied the whole ledger and the last 120 copies were kept, so with a million accounts a node held ~3.5 GB of copies. Creating a million accounts costs about 100 GRN in fees, a cheap way to push small nodes out. Fix: copy-on-write diff layers over one flat base; a block costs O(keys it touches). Measured: 587 MB → 0.1 MB for 20 blocks at 1M accounts, ledger work 0.5–0.8 ms per block at any size. (`bench/state_bench.py`)
- **Y3: Restart time grew with the chain's age.** A node re-executed every block since genesis on every start, re-running WASM adjudications and needing every blob again: about 18 hours after 4 years of blocks. Fix: SQLite storage of the base state and recent diffs; restart takes 0.04 s at any height, and the test proves no block is executed while opening. (`test_restart_opens_the_same_chain_without_re_executing`)
- **Y4: Atomicity was assumed, never tested.** The code relied on "a rejected transaction changes nothing" (and v0.5's mempool now applies transactions in place). Fix: a test drives all ten transaction types to their latest failure point and checks the state is bit-for-bit unchanged, and the fuzz test checks it for every rejected transaction. (`tests/test_atomicity.py`)
- **Y5: Committed states could be modified.** Nothing stopped code from writing into a past block's state. Fix: a state with children is frozen and every write raises; the full suite runs under this rule.
- **Y6: Crash safety.** Blocks were written as separate JSON files with no link to the state. Fix: one SQLite transaction per block (block, state diff, main-chain update, base move); a simulated power cut mid-block reopens at the last complete block. (`test_a_crash_mid_block_leaves_the_last_complete_block`)

A differential test replays the same blocks with full copies and compares: the layered chain, through reorgs and dozens of base moves, ends with the identical state. (`test_layered_state_matches_a_full_copy_replay`)

Residual limits: the base state is held fully in memory (one copy, not 120; a lazily loaded, disk-backed base is future work), and the main-chain hash index is in memory (~0.8 GB after 4 years of blocks).

## Sixth review: difficulty and accountability (v0.6)

- **Z1: The windowed difficulty rule could be gamed and drifted.** Retargeting every 30 blocks over a 30-block window lets miners time their participation: join while difficulty is low, leave when it rises. Simulated (`bench/difficulty_sim.py`) with 100 units/s always on and 500 units/s that come and go: blocks averaged 20.6 s instead of 15, 15-minute averages were off by up to 24.9 s, and the chain fell 9.3 hours behind its issuance schedule in 6,000 blocks. Even with steady hashrate the rule ran 4.7% slow (15.7 s blocks), which stretches every halving. Fix: per-block ASERT (aserti3-2d) anchored at genesis, half-life 30 minutes. Same simulation: 15.0 s average, worst 15-minute average off by 4.1 s, zero schedule drift; recovery from a 100× hashrate drop in 4.0 h instead of 8.8 h. (`test_asert_matches_the_formula`, `test_difficulty_follows_the_schedule_every_block`; the live-node WASM test now also checks that ASERT slowed instant blocks down.)
  Residual: after a sudden 10× surge ASERT needs about an hour to return to 15 s blocks. A miner can date its block up to 60 s ahead (MAX_FUTURE_SECONDS), which makes the next target at most 2^(60/1800) ≈ 2.3% easier; the median-time rule keeps timestamps from running backwards.
- **Z2: A self-funding agent had no off switch.** A misbehaving or compromised agent key could spend everything it held, and nobody could stop it. Fix: guarded accounts (SPEC.md §3.4). Design choices, each tested: the account can only make its own guard stricter, so a compromised agent cannot lift it; a frozen account can send nothing at all; only the guardian can loosen, freeze, hand over or remove; spending counts transfers, escrow and fees, checked before any change so a rejected transaction leaves no trace; incoming coins are never blocked. (`tests/test_guard.py`, guard cases in `tests/test_atomicity.py` and in the money fuzz test, `test_a_guarded_agent_obeys_its_guardian`.)
  Residual: the guardian is a single key; whoever holds it can freeze or loosen the account (multisignature guardians are future work). Bonds are not counted as spending, because they come back unless fraud is proven; a frozen miner may lose the bonds of batches it can no longer reveal, which is the guardian's call.
- **Z3: Supply could only be checked by the test suite.** A running node had no way to show that no coin was created from nothing. Fix: `GET /supply` and `grn supply` check that balances plus escrow plus bonds equal the sum of all block rewards, exactly; `GET /metrics` exposes the node's health for monitoring. (`test_supply_audit_and_metrics_on_a_live_node`)

## Seventh review: the network, sync and a state commitment (v0.7)

- **P1: Nodes talked plain, unauthenticated HTTP.** Anyone on the path could read every transaction and block, change them, or drop them, and a node trusted an `X-Green-From` header to decide which peer a message came from. Fix: P2P v1 (SPEC.md §9, `green/p2p.py`): ephemeral X25519 per connection, keys from HKDF salted with the genesis hash (another network fails at the first frame), every frame sealed with ChaCha20-Poly1305 including its length, rekeying every 4,096 frames, and an Ed25519 node identity that signs the session id with its role. A peer given as `<id>@host:port` must present that key. The HTTP gossip endpoints are gone; the HTTP API is for wallets and tools only. (`test_nothing_on_the_wire_is_readable` records the bytes between two nodes and finds no transaction, address, genesis hash or message name in them; `test_a_corrupted_byte_drops_the_connection_and_it_recovers`; `test_a_pinned_peer_must_present_its_key`; `test_nodes_of_another_network_never_connect`; `test_frames_round_trip_and_rekey_in_step`; `test_S5_http_requests_cannot_inject_peers_or_blocks`.)
- **P2: Every block and transaction was pushed in full to every peer.** Bandwidth grew with the number of peers times the data, whether they had it or not. Fix: announce by hash, fetch only what is missing, ask another announcer after a timeout. (`test_transactions_and_blocks_cross_each_link_once`: in a line of three nodes each node receives the transaction body exactly once; `test_a_silent_peer_is_skipped_for_another_announcer`.)
- **P3: A transaction's origin was easy to find.** A node pushed its own transactions to everyone at once. Fix: transaction announcements leave each peer's queue at exponentially distributed intervals, so the first node to announce is not necessarily the source. Residual: this is the trickle Bitcoin uses, weaker than Dandelion++.
- **P4: Nothing stopped an eclipse.** Any number of announced addresses competed for the peer list, outbound peers could all sit in one network, and inbound slots were first come, first served. Fix: an address book that one source network group can fill only up to 64 entries; outbound peers spread over /16 (IPv4) and /32 (IPv6) groups; at most 4 inbound peers per group; when inbound slots are full, the fastest, most recently useful and longest-connected peers are protected and the youngest peer of the most crowded group is evicted; two of eight outbound links relay blocks only, which hides them from transaction-based topology probing. (`test_one_source_cannot_fill_the_address_book`, `test_eviction_protects_good_peers_and_hits_the_crowded_group`, `test_network_groups_and_local_addresses`, `test_addresses_spread_and_nodes_find_each_other`.)
- **P5: A new protocol is a new DoS surface.** Fix: a frame's sealed length is checked before its body is read (4 MiB cap; larger only for a blob this node asked that peer for); every read has a deadline, handshakes must finish in 10 s, at most 1,000 messages wait per peer, a peer that does not read what we send is dropped at 64 MiB, blob uploads are budgeted per peer, and unsolicited data, oversized frames, invalid headers and blocks are banned for 24 hours. Malformed messages add up to a ban. (`test_an_oversized_frame_is_refused_unread`, `test_data_nobody_asked_for_gets_the_peer_banned`, `test_a_flood_beyond_the_queue_is_cut_off`, `test_a_silent_connection_is_closed_after_the_handshake_deadline`, `test_malformed_messages_add_up_to_a_ban`, `test_bans_survive_a_restart`.)
- **P6: Sync was serial, single-peer and slow.** A new node fetched the hash list and then every block from one peer, so a slow or lying peer stalled it, and each block cost a 17 ms work-unit recomputation on one core. Fix: headers first, each header checked against every rule except its work unit, the work units of a random sample plus the last header recomputed before any block is requested; blocks downloaded from several peers in a 256-block window; work units recomputed in parallel threads before the blocks are connected in order. The transcript digest itself is now computed in cache-sized chunks: 3.6x faster, bit for bit the same (checked against the definition), so a unit verifies in 9 ms on one core and ~4 ms per unit with 8 threads, versus 17 ms before. (`test_headers_first_sync_from_several_peers`, `test_fake_headers_are_caught_before_any_block_is_fetched`, `test_headers_that_break_a_rule_get_the_peer_banned`, `test_headers_from_the_future_are_declined_without_a_ban`, `test_fast_digest_equals_the_definition`, `test_work_unit_transcript_is_unchanged`.)
  Why sampling is enough: a forged header costs a hash instead of a matmul, but a chain whose forged fraction is f survives one message's 9 samples with probability (1−f)^9, and over thousands of messages only a negligible f goes unnoticed, which saves the forger a negligible part of the work. Blocks are always fully checked before they are connected.
- **P7: Nothing committed to the ledger.** Two nodes (or two implementations) could disagree on a balance and not notice until some later transaction forked them, and a snapshot could only be trusted through a hash shipped in the software. Fix: every header carries `state_root`, a lattice hash (LtHash, 1,024 16-bit lanes) of every ledger entry, updated per block in O(entries changed) and checked by every validator. Every block built anywhere in the test suite also compares the incremental root with a full recomputation. Writing the exact state layout into SPEC.md §6.1 found a quirk that would have become consensus: crediting zero coins created an empty balance entry; balances now hold non-zero amounts only. (`test_a_block_must_commit_to_the_state_it_produces`, `test_state_root_is_incremental_and_order_independent`, `test_state_root_sees_every_change`, the check in `tests/helpers.py`.)
- **P8: History depended on old job data forever.** Re-validating an old useful-work block needs that job's matrices (up to 32 MiB); if nobody kept them, new nodes could not sync at all. Fix: snapshot sync. Nodes write the ledger every SNAPSHOT_INTERVAL blocks as content-addressed chunks; a new node with `--fast-sync` takes the newest final snapshot on offer, checks every chunk against the manifest and the lattice of all entries against the snapshot header's `state_root`, then validates every later block. A snapshot that does not match gets its server banned and the node falls back to a full sync. (`test_a_snapshot_rebuilds_the_exact_ledger_and_the_chain_continues`, `test_a_tampered_snapshot_is_refused`, `test_a_new_node_joins_from_a_snapshot`, `test_a_lying_snapshot_server_is_banned_and_sync_falls_back`.)
- **P9: Decoded job matrices were cached without a bound.** Each decoded job can take 32 MiB and none was ever released, so posting many jobs could exhaust a node's memory. Fix: a least-recently-used cache of 8 jobs. (`test_decoded_job_data_is_cached_within_a_bound`.)
- **P10: An unauthenticated connection could evict a good peer (found re-reviewing this release).** With inbound slots full, the node made room as soon as a TCP connection arrived, before its handshake, so an attacker could keep opening connections that never authenticate and push honest peers out one by one; handshake threads were also unbounded. Fix: eviction happens only when a newcomer has proven its key; at most 32 handshakes may be in progress (4 per address); peers that do not deliver requested blocks are penalised; headers of forks below the reorg window are pruned so they cannot fill the header index. (`test_only_an_authenticated_newcomer_can_evict` fails with the old rule restored; `test_unfinished_handshakes_are_capped`, `test_a_peer_that_stalls_block_requests_is_penalised`, `test_stale_fork_headers_are_pruned`.)

Residual risks of this layer:
- X25519 keys are not uniformly random bytes (no Elligator encoding as in BIP324) and frame sizes are visible, so a determined observer can recognise the protocol, though not read it.
- Encryption to a peer that was not pinned is opportunistic: an active man in the middle on the very first connection could sit in between. Pin important peers (`<id>@host:port`) or compare session ids (`grn peers`).
- Fast-synced nodes trust the heaviest header chain for history before the snapshot (the snap-sync model); they never re-execute those blocks.
- Peers on local addresses are banned by node id, which a peer on the same LAN can change.
- No Tor/I2P, NAT traversal, DNS seeds (SEED_NODES is empty until a public network launches) or compact block relay yet.
- The state root proves the whole ledger, not a single account: light clients need a Merkle or Verkle commitment.

## Eighth review: changing the rules (v0.8)

- **U1: A live network could not change a single rule without splitting.** Every release so far started a new network; after launch any rule change would have forced everyone to switch at once, and anyone who did not would have been on another chain. Fix: version-bits deployments (SPEC.md §5.5, UPGRADES.md). A rule activates only after 90% of a 20,160-block period's blocks signal it, one period after locking in; it must be a soft fork, so nodes that never upgraded keep following the chain. Deployment state depends only on a block's ancestors and is cached by block hash, so a reorg across a period boundary follows the other branch. (`tests/test_versionbits.py`; the exact-threshold test was checked to fail when `>=` is mutated to `>`.)
- **U2: A user could not tell a real release from a tampered one.** Fix: releases carry `SHA256SUMS` signed by several maintainer keys (green/release.py); a release is genuine only with RELEASE_THRESHOLD distinct known signers and every file matching its digest. Release signatures are domain-separated from transaction signatures. (`tests/test_release.py`.)
- **U3: A node would not notice an upgrade it does not know.** Fix: when most of the last 1,000 blocks set a signal bit the node has no deployment for, `grn status` and `GET /status` warn the operator to upgrade before it activates. (`test_a_node_warns_about_an_unknown_rule_change_and_reports_deployments`.)

Residual risks: miners can signal readiness without enforcing the rule ("false signalling"); after activation, a non-enforcing majority could extend a chain that upgraded nodes reject. The 90% threshold and the warning to old nodes reduce, but do not remove, that risk. A `lock_in_on_timeout` deployment can split the network if miners and users disagree, which is why UPGRADES.md treats it as a last resort. Release security rests on how maintainers keep their keys; reproducible builds are still missing.

## Ninth review: multisignature accounts (v0.9)

- **M1: Every account was one key away from loss.** A stolen key emptied an account, and an agent's guardian was a single key that could loosen or remove the guard alone (a residual of Z2). Fix: k-of-n accounts; a guardian can be one. (`test_k_of_n_keys_spend_from_a_multisig_account`, `test_a_multisig_guardian_needs_its_quorum_to_freeze_an_agent`, `test_cli_multisig_workflow_end_to_end`.)
- **M2: Ambiguous multisig encodings would give one account several addresses or let signatures be padded.** Fix: the keys must be sorted and distinct, so an account has exactly one address; exactly k signatures, no more, each in its key's slot; a malformed account object is refused before any signature is checked. (`test_malformed_multisig_accounts_are_refused`, the too-many and too-few cases above.)
- **M3: Signature checking could become a block-level DoS.** Up to 16 signatures per transaction times 500 transactions would be 8,000 checks per block. Fix: at most MAX_BLOCK_SIGS = 2,000 signatures per block (about 0.1 s), enforced by validators and respected by block templates. (`test_a_block_has_a_signature_budget`.)

Residual risks: a spend reveals all n keys and k on chain (no key aggregation such as MuSig, so multisig spends are recognisable and larger); there are no time locks, so an account whose holders lose more than n − k keys is lost for good.

## Tenth review: operations and the API (v0.10)

- **O1: The wallet password of an unattended node sat in its environment.** Any process of the same user (and root) can read another process's environment. Fix: `GREEN_WALLET_PASSWORD_FILE`; the systemd unit hands the password over as a systemd credential, a file only the node can read. (`test_wallet_password_from_a_file`.)
- **O2: Long-polling could tie up the API.** Every waiting client holds a server thread. Fix: at most 16 waiters (the next gets 503 at once) and at most 60 s per wait; stopping the node releases them. (`test_long_poll_returns_on_the_next_block`.)
- **R1: Disk use grew without a bound.** Every block body and every data blob was kept forever, so a long chain (or many large jobs) would eventually fill any disk. Fix: pruned nodes (OPERATIONS.md); archive nodes stay the default so history remains available. (`test_old_bodies_go_headers_stay_and_the_chain_goes_on`.)
- **R2: Deleting data that is still needed would break the node or a buyer.** Fix: a blob is deleted only if nothing in the ledger, the mempool, the recent blocks, the snapshots or this node's own pending work references it, and only after it stayed unreferenced for a retention period that is far longer than the reorg window. (`test_unused_data_is_deleted_only_after_the_retention_period`.)
- **R3: Peers asking a pruned node for old blocks would stall sync.** Fix: the handshake says where a node's bodies start, block requests go only to peers that have them, and a "not found" answer moves that boundary for a peer that pruned after connecting. A pruned node still serves snapshots. (`test_a_pruned_node_still_lets_new_nodes_join`.)
- **L1: A wallet had to trust its node about payments.** Fix: inclusion proofs and a light client that checks every header rule, recomputes the work of a random sample plus the newest header, and checks the Merkle proof against a verified header with enough confirmations. (`test_a_light_client_verifies_payments_itself`; `test_a_light_client_catches_a_lying_node`: a proof for another transaction and a header with undone work are both rejected, and nothing of a bad batch is kept.) Residual: like any SPV client it trusts that the heaviest header chain is valid, and with sampled work a forger must still do nearly all of the work.
- **L2: An index that disagrees with the chain would show payments that are not there.** Fix: the index is written in the same database transaction as the block and rewritten from the fork point on a reorg. (`test_the_index_follows_the_main_chain_through_a_reorg`.)
- **O4: One unexpected error stopped a node's miner for good, silently.** The mining and task-worker loops ran in threads that died on any exception. Fix: they are supervised; an error is logged and the loop restarts (the task worker keeps its record of commitments, so it still reveals them). (`test_S22_a_node_loop_survives_an_unexpected_error`, which fails with the supervision removed.)
- **O3: A typo in a settings file would be ignored silently.** A misspelt `p2p_host` or a string where a number belongs could leave a node listening where the operator did not intend. Fix: unknown keys, wrong types and bad log levels stop the node with a message. (`test_config_file_settings_and_command_line_overrides`.)

## Eleventh review: pools (v0.13)

- **K1: Pooled mining meant trusting the pool with the rewards.** In a custodial pool the operator receives every block reward and pays miners later, or not. Fix: a block's `miner` can list up to 32 payees with weights, and consensus credits each one; the pool's templates pay the miners of recent shares directly, so the operator never holds their coins. A miner sees its weight in every template and can leave a pool that underpays it. (`test_a_block_can_pay_several_miners_by_weight`, `test_a_pool_pays_its_miners_in_the_blocks_themselves`.)
- **K2: Ambiguous payee lists.** Fix: exactly one valid form (2 to 32 entries, sorted, distinct addresses, positive weights; a single payee is a plain address), and a fixed rounding rule (the remainder goes to the first payee). (`test_payee_lists_have_one_valid_form`, `test_a_block_with_a_malformed_payee_list_is_rejected`.)
- **K3: Share checking is a DoS target.** Every share costs the pool a work-unit recomputation. Fix: stale, duplicate and wrong shares are refused, and a client (by IP; payout addresses cost nothing to change) that sends 20 bad shares is paused for 10 minutes. (`test_bad_shares_are_refused`.)

Residual risks: the pool still chooses each block's transactions (miners cannot yet negotiate them, as Stratum V2 allows); payee lists hold 32 entries, so a pool with a long tail of tiny miners pays the largest 31 of the window per block; pool miners do filler work only.

## Twelfth review: wallets, relay privacy, compact blocks, pooled compute (v0.14)

- **H1: A typo in a recovery phrase must not open an empty wallet.** A restore from a phrase with one wrong word would derive a valid key for an empty account, and the user would believe the coins gone. Fix: every phrase is checked against its BIP-39 checksum before use; unknown words are refused with their position; only an exact word or an unambiguous prefix of four or more letters is accepted, never a guess. A 24-word phrase (the default) has an 8-bit checksum, so a random substitution passes with probability 1/256 (1/16 for 12 words). (`test_typos_are_caught_not_guessed`.)
- **H2: A damaged word list would silently make wrong phrases.** Fix: `green/bip39_english.txt` is checked against the SHA-256 published with BIP-39 when the module loads; any change stops the wallet. (`green/hd.py`; the BIP-39 and SLIP-0010 reference vectors in `tests/test_hd.py`.)
- **D1: The first node to announce a transaction is usually its source.** Trickled announcements only blur this. Fix: Dandelion++ stems (SPEC.md §9.3). (`test_own_transactions_take_a_stem_and_are_not_announced`, `test_a_stem_crosses_relays_and_is_announced_where_it_ends`.)
- **D2: Probing the stem.** A spy could ask nodes for a transaction to find where the stem has reached. Fix: stem transactions live in a separate pool, are never served to `getdata`, and are not shown by the API unless they are the node's own. (Same test: `getdata` gets `notfound`.)
- **D3: A stem node that drops the transaction.** Fix: every node on the stem keeps it under a random embargo and announces it itself when the embargo runs out; a stem that loops back ends at once. (`test_a_stem_that_loops_back_ends_there`; the embargo in the first D1 test.)
- **D4: Stems and the node's own deadlines.** A batch commit or a dispute answer held on a stem could miss its deadline, and the loops that send them look for them in the mempool, so they would send them again. Fix: the node's protocol transactions skip the stem (their sender is the miner address blocks already show), and so do fee bumps, which only the mempool's rules can judge. (`test_the_nodes_own_protocol_transactions_skip_the_stem`; `test_replace_by_fee` failed before this.)
- **C1: Short-id collisions.** 48-bit ids could collide by accident or by design. Fix: ids are keyed by the block and a fresh random salt per message, so colliding transactions cannot be prepared in advance; a rebuilt block is checked against `tx_root`, and a mismatch fetches the whole block instead of punishing anyone. (`test_a_bad_compact_block_falls_back_to_the_whole_block`.)
- **C2: Compact block spam.** A `cmpct` costs the receiver a pass over its mempool. Fix: only blocks that extend a block it has are rebuilt (others trigger header sync), the block is then fully validated, and an invalid header gets the peer banned as for a full block.
- **J1: Pool miners' job payments.** If the pool committed results in its own name, it would receive every unit payment and pay miners later, or not. Fix: an optional `payees` list on `job_commit` (the same form as a header's), and consensus splits the batch's payment among them when it settles; the bond (the pool's risk) stays the committer's. (`test_a_batch_can_pay_the_miners_who_computed_it`, `test_batch_payees_have_one_valid_form`, `test_pool_miners_compute_jobs_and_the_batches_pay_them`.)
- **J2: Wrong results would cost the pool its bond.** Fix: the pool checks each result with Freivalds' test under a fresh secret seed (a wrong result passes with probability 2^-124) and only accepts results for units it assigned to that address; bad results count toward the client's pause, like bad shares. (`test_wrong_or_unassigned_results_are_refused`.)
- **J3: A batch can name 32 payees.** Fix: a batch takes units from at most 31 miners besides the pool's fee; the others' units wait in the buffer for the next batch, since they are owed. (`test_a_batch_pays_at_most_31_contributors_and_the_rest_wait`.)
- Found while testing: stale shares counted as invalid could pause an honest pool miner for ten minutes (now answered as stale, not held against it); a batch flush popped entries while iterating and lost accepted results.

Residual risks: a stem is only as private as its relays are honest (a spy controlling a node's two stem destinations learns that it was the source, as in Dandelion++); the stem pool holds 1,000 transactions, beyond which new ones are announced directly; pool miners trust the pool to pay them for units by naming them in commits, which they can check on chain for every batch and leave a pool that does not.

## Still open (needed before production)

The full list, by area, is in [PRODUCTION.md](PRODUCTION.md).

- An independent security audit and a public testnet with a bug bounty.
- Data privacy: a miner sees the matrices of the units it computes (see [DESIGN.md](DESIGN.md) §9). The fix is secure hardware (TEE) or secret sharing.
- The residual risks of the network layer listed above.
- Data availability of WASM outputs and matmul manifests (above).
- Matmul results reach the buyer through a direct fetch from up to four random peers; a real network needs a content-routing layer (e.g. a DHT) so the buyer can always find the miner's data.
