# Production readiness

Where Green stands on every aspect a production cryptocurrency needs, what backs each claim, and what is still missing. "Done" means implemented, tested and measured in this repository; it does not mean audited, battle-tested or ready to carry real value. Nothing here is ready for real money until the items marked **blocker** are done.

Status as of v0.14 (October 2026). 214 automated tests; twelve internal security reviews (SECURITY.md).

## Summary

| Area | Status | Blockers before real value |
|---|---|---|
| 1. Consensus rules and specification | Done for a testnet | external review of SPEC.md |
| 2. Proof of useful work | Done; custom digest unaudited | audit of the transcript digest |
| 3. Difficulty and issuance | Done (ASERT, 21M cap, supply audit) | — |
| 4. Compute markets (matmul, WASM) | Done; optimistic, documented limits | data availability for manifests and WASM outputs |
| 5. Accountability for autonomous agents | Done (guarded accounts) | — |
| 6. State, storage, crash safety | Done (with a state commitment in every header) | — |
| 7. Peer-to-peer networking | Done for a testnet (encrypted, authenticated, eclipse limits, Dandelion++, compact blocks) | NAT traversal, DNS seeds, Tor/I2P |
| 8. Initial sync of new nodes | Done (headers-first, parallel verification, verified snapshots) | — |
| 9. Mining software | Partial (CPU/GPU miners, non-custodial pools that sell compute; GPU far below hardware peak) | native GPU miner |
| 10. Wallets and keys | Done for a testnet (recovery phrases, encrypted files, checksummed addresses, multisig) | hardware wallets, a registered coin type |
| 11. APIs and developer tools | Done for a testnet (versioned API, fee estimates, long-poll, Python SDK) | SDKs in other languages, explorer API |
| 12. Operations | Done for a testnet (config file, logs, systemd, metrics and alerts, runbook) | — |
| 13. Testing and QA | Partial (214 tests, fuzz, mutation testing, 8-node simulation) | multi-node testnet, soak tests |
| 14. Security assurance | Partial (internal reviews only) | **blocker**: independent audit, bug bounty |
| 15. Upgrades and governance | Done (version-bits activation, signed releases, published process) | reproducible builds |
| 16. Privacy | Partial (transaction origin: Dandelion++; data and amounts public) | — (documented) |
| 17. Performance and implementation | Partial (Python reference) | native node for scale |
| 18. Source control and releases | Partial (git, tagged versions; not published, CI not yet run) | **blocker**: public repository, CI, signed tags |
| 19. Economics and launch | Partial (supply set; no launch plan) | fair-launch plan, public testnet |
| 20. Legal and compliance | **Missing** | **blocker** |
| 21. Ecosystem | **Missing** (no explorer, exchanges, website) | — |

## 1. Consensus rules and specification

**Done.** The rules are written down independently of the code in [SPEC.md](SPEC.md): encodings, addresses, every transaction type and its validity rules, end-of-block events, Merkle trees, the work unit bit for bit, difficulty, fork choice, block limits, WASM execution and constants. Block size is bounded (500 transactions, 1 MB); forks that leave the main chain more than 100 blocks below the tip are refused (measured at the fork point since v0.5).

**Evidence:** every rule has tests (`tests/test_chain.py`, `tests/test_security.py`); the work-unit description in SPEC was checked line by line against the code.

**Missing:** a second, independent implementation built from SPEC.md alone (the strongest test of a spec), and outside review of the spec.

## 2. Proof of useful work

**Done.** Integer matrix multiplication with seed-derived low-rank noise; every unit is both a lottery ticket and a deliverable; the useful result is recovered cheaply; knowing A·B gives no shortcut (noise rank ≥ tile size); bit-exact on CPU and GPU (the GPU is checked against the CPU at startup). Since v0.7 the digest is computed in cache-sized chunks: a unit verifies in 9 ms on one core (was 17 ms), bit for bit the same.

**Evidence:** `tests/test_chain.py` (tampered blocks, stolen tickets), `tests/test_gpu.py` (bit-for-bit), `tests/test_work.py` (the fast digest against its definition), DESIGN.md §3.

**Missing:** the transcript digest is a custom MurmurHash3-style function chosen for GPU speed, analyzed but not proven (SECURITY.md); either an audit or BLAKE2b/BLAKE3 on the GPU.

## 3. Difficulty and issuance

**Done.** Per-block ASERT difficulty since v0.6: in simulation with miners that come and go, the old windowed rule ran 37% slow and drifted 9.3 hours behind schedule in 6,000 blocks; ASERT held 15.0 s and zero drift (`bench/difficulty_sim.py`). Issuance: 1.25 GRN per block, halving every 8.4M blocks, cap 20,999,998.908 GRN. Every node can audit the supply (`grn supply`, `GET /supply`): balances plus escrow plus bonds must equal the sum of block rewards, exactly.

**Evidence:** `test_asert_matches_the_formula`, `test_difficulty_follows_the_schedule_every_block`, `test_emission_matches_bitcoin_schedule`, the supply audit on a live node (`test_supply_audit_and_metrics_on_a_live_node`), and two fuzz tests that check conservation of money at every block.

**Known trade-off:** after a sudden 10× hashrate surge ASERT takes ~1 hour to reach 15 s blocks again (the old rule took minutes, but drifted for good).

## 4. Compute markets

**Done.** Matmul jobs: a whole job delivered with two transactions (bitmap + Merkle root), buyer-verified with prime-field Freivalds (2^-124), disputes for any number of units, whole-batch forfeit, backlog-stretched deadlines. WASM tasks: deterministic sandbox, optimistic with on-chain re-execution, watchtowers, per-block adjudication budget.

**Evidence:** `tests/test_chain.py`, `tests/test_compute.py`, `tests/test_network.py` (a cheating miner caught by the buyer's node; a miner answering a dispute with a Merkle proof), SECURITY.md reviews W, V and X.

**Missing:** real data availability. A miner that withholds a whole manifest gains nothing but costs the buyer dispute fees; WASM outputs have no availability proof. Erasure coding with sampling is the known fix. Delivery ceiling: 68,267 units/s (~98 RX 9070 XT), set by the transaction count.

## 5. Accountability for autonomous agents

**Done (v0.6).** The manifesto's question, "who holds the off switch of a system that funds itself?", answered in consensus: an account can name a guardian who caps its spending per period and can freeze it. The account can only tighten its own guard; loosening, freezing, handing over and removing belong to the guardian. Incoming coins are never blocked. The agent SDK reads its guard: it waits when the allowance is used up and stops when frozen.

**Evidence:** `tests/test_guard.py`, guard cases in `tests/test_atomicity.py` and in the money fuzz test (no guarded account ever overspends), `test_a_guarded_agent_obeys_its_guardian` and `test_cli_guard_commands_end_to_end` on live nodes.

## 6. State, storage, crash safety

**Done (v0.5, commitment v0.7).** Diff layers over a flat base; SQLite storage with one transaction per block; restart without re-executing; frozen committed states. Every header carries `state_root`, a lattice hash of the whole ledger updated per block in O(entries changed), and the exact state layout is specified (SPEC.md §6.1).

**Evidence:** 1M accounts: 0.1 MB held per 20 blocks (v0.4: 587 MB), ledger work 1.1–1.5 ms per block at any size including the state root, restart 0.04 s at any height (`bench/state_bench.py`); `tests/test_storage.py` (differential test against full copies through reorgs and base moves, restart without replay, simulated power cut); every block built by any test checks its incremental root against a full recomputation; a real node killed mid-run reopened and settled its pending job.

Pruned nodes (v0.11, green/prune.py): every header but only recent block bodies, the ledger and its snapshots, and the data blobs still in use; unused blobs are deleted after a retention period, and freed space returns to the disk (`tests/test_prune.py`).

**Missing:** a disk-backed base loaded on demand (the base is one copy in memory), a Merkle or Verkle commitment so light clients can verify single accounts (the lattice hash proves the whole state only).

## 7. Peer-to-peer networking

**Done for a testnet (v0.7).** A dedicated protocol (SPEC.md §9, `green/p2p.py`) on its own port: X25519 per connection, keys salted with the genesis hash, ChaCha20-Poly1305 on every frame including its length, rekeying, Ed25519 node identities that can be pinned (`<id>@host:port`). Persistent connections with announce-then-fetch relay; transaction announcements trickled at random intervals; address gossip with per-source limits; outbound peers spread over network groups, two of them block-relay-only; inbound limits per group with eviction; frame sizes checked before reading; per-peer queues, deadlines and upload budgets; bans for invalid data that survive restarts. The HTTP API is for clients only.

**Dandelion++ and compact blocks (v0.14):** a node's own transactions travel a stem before anyone announces them (routes fixed per epoch and incoming link, stem pool never served, loops end the stem, embargo timers on every stem node); new blocks cross a link as salted short ids and are rebuilt from the mempool, so a 20-transaction block costs under a quarter of its size and usually no round trip (`tests/test_relay.py`, 7 tests, each mutation-checked).

**Evidence:** `tests/test_p2p.py` (27 tests: nothing readable on the wire, a corrupted byte drops the link and it recovers, pinning, other networks refused, each transaction crosses each link once, silent announcers skipped, nodes find each other through gossip, fake headers banned, oversized frames, unsolicited data, floods and silent handshakes cut off, unfinished handshakes capped, only authenticated newcomers can evict, stalling peers penalised, eviction and address-book rules); `bench/net_sim.py`; SECURITY.md P1–P5, P10.

**Missing:** NAT traversal (UPnP/PCP) and DNS seeds (`SEED_NODES` is empty until launch); Tor/I2P; Elligator-encoded keys and traffic padding against protocol fingerprinting; a content-routing layer (DHT) so a buyer can always find a miner's results.

## 8. Initial sync

**Done (v0.7).** Headers first, each checked against every rule except its work unit, with a random sample of work units recomputed per message before any block is fetched; blocks downloaded from several peers at once in a 256-block window; each block's work unit sent to 8 verification threads the moment it arrives while a separate thread connects blocks in order. Measured with `bench/sync_bench.py` on 600 blocks: full sync 4.2 s (144 blocks/s; before this pipeline 14.6 s), fast sync 1.4 s. Fast sync: nodes write the ledger every 2,880 blocks as content-addressed chunks, and a new node starts from the newest final snapshot after checking it against the `state_root` in that block's header, then validates every later block. This is trustless relative to the heaviest header chain (Ethereum's snap-sync model), where Bitcoin's assumeutxo trusts a hash shipped with the software. It also removes the need for old job matrices to join the network.

**Evidence:** `test_headers_first_sync_from_several_peers` (headers in rounds, both peers serve blocks, work units checked in parallel), `test_fake_headers_are_caught_before_any_block_is_fetched`, `tests/test_snapshot.py` (a snapshot rebuilds the exact ledger and the chain continues; tampered snapshots refused; a new node joins from a snapshot over the network; a lying snapshot server is banned and sync falls back), `bench/sync_bench.py`.

**Missing:** GPU verification of work units during sync; background validation of pre-snapshot history for nodes that want it; pruned nodes cannot serve blocks below their snapshot (they say so in their handshake).

## 9. Mining software

**Today:** CPU and GPU (WebGPU/Vulkan) miners, bit-exact, ~12–17× GPU speed-up; miners pick the best-paying open units and deliver them in batches.

**Non-custodial pools (v0.13):** a block can pay up to 32 payees by weight, so a pool's blocks pay its miners directly; `green/pool.py` has the pool server (PPLNS templates, every share's work recomputed, bad shares refused and their senders paused) and a pool miner (`tests/test_pool.py`).

**Pools sell compute (v0.14):** a pool assigns its miners units of open jobs, checks each result with Freivalds' test, commits them in batches and posts the bond; each commit names the miners who computed its units as payees, so consensus pays them when the batch settles (`test_pool_miners_compute_jobs_and_the_batches_pay_them`: the buyer gets correct products, the pool keeps exactly its fee).

**Missing:** the GPU miner reaches ~700 units/s, a small fraction of a modern card's int8 throughput (noise is generated on the CPU, one nonce per call); a native miner with on-GPU noise and batched nonces. Pool miners compute job units on the CPU even when they mine on a GPU; miners cannot yet choose a pool block's transactions (as Stratum V2 allows); results travel to the pool over plain HTTP (256 KiB per unit).

## 10. Wallets and keys

**Today:** wallets made from a 24-word recovery phrase (v0.14: BIP-39, SLIP-0010 Ed25519 derivation, optional passphrase, numbered accounts; the official word list is checked against its published hash when loaded); wallet files encrypted with scrypt + AES-256-GCM; bech32m addresses that reject typos; k-of-n multisignature accounts (v0.9) with a cosigning workflow (`grn multisig`), usable as an agent's guardian.

**Evidence:** `tests/test_hd.py` (BIP-39 reference vectors, SLIP-0010 test vector 1, typos caught and never guessed, restore with and without a password, older wallets still open).

Since v0.12 a wallet can track an address's history and verify its payments as a light client.

**Missing:** hardware-wallet support; a registered SLIP-0044 coin type (7777 is provisional); watch-only wallets in the command line.

## 11. APIs and developer tools

**Today (v0.10):** every HTTP endpoint documented in [API.md](API.md) under a stable `/v1` prefix; fee estimates from the mempool and recent blocks (`GET /v1/fee`, used by the command line); long-polling for new blocks (`GET /v1/wait`), used by the SDKs; a Python SDK for buying compute (`green.nn.MatmulClient`), autonomous agents (`green.agent`) and multisig (`green.tx`); a command line covering every transaction type.

**Evidence:** `tests/test_api.py` (the versioned paths, the fee estimate against a full mempool, long-poll wake-up, timeout and waiter cap).

Since v0.12: a transaction and address index kept consistent through reorgs (`/v1/tx`, `/v1/address/.../history`), inclusion proofs, and a light client (`green.light`) that verifies headers and payments itself (`tests/test_light.py`).

**Missing:** SDKs in other languages; notifications for transactions to an address (not only blocks); a graphical block explorer.

## 12. Operations

**Today (v0.10):** a config file (`green.toml`, strictly checked), log levels and a rotated log file, a hardened systemd unit that passes the wallet password as a systemd credential, Prometheus metrics with alert rules (supply audit, peers, stalls, sync, warnings, ban storms), a runbook ([OPERATIONS.md](OPERATIONS.md): settings, firewall, service, monitoring, backup and restore, upgrades, troubleshooting), a Docker image (built and smoke-tested by CI; not built on the development machine because Docker was not running there), and an installable `grn` command.

**Evidence:** `tests/test_api.py` (config file precedence and refusal of bad settings, the password file, the log file, the supply-audit metric).

**Missing:** a public status page; packages for operating systems (deb, rpm, Homebrew, Windows installer).

## 13. Testing and QA

**Today:** 192 tests across consensus, security regressions, addresses, fees, state and storage, the state commitment and snapshots, the P2P protocol, atomicity, networks of live nodes, GPU, WASM, both markets, the agent and its guardian; two fuzz tests with exact accounting invariants; mutation testing proved the fuzz catches injected accounting bugs; a differential test against a reference replay; an 8-node network simulation with a split and a heal (`bench/net_sim.py`: blocks reach every node in a median of 0.03 s, the halves converge in about 1 s with identical state roots); CI configured for Linux, Windows and macOS.

**Missing:** CI has never run (the code is not on any git host yet); multi-node testnets with tens of nodes; network partition and latency simulation; week-long soak tests; property-based testing at larger scale; conformance tests shared with a second implementation.

## 14. Security assurance — blocker

**Today:** eleven internal reviews with every finding fixed and a regression test for each (SECURITY.md: S1–S19, W1–W9, V1–V6, X1–X6, Y1–Y6, Z1–Z3, P1–P10, U1–U3, M1–M3, O1–O4, R1–R3, L1–L2, K1–K3), including bugs found in this project's own earlier fixes.

**Missing:** an independent professional audit of consensus, cryptography and the dispute games; a public bug bounty; a responsible-disclosure policy; formal analysis of the economic games (disputes, delivery, guards).

## 15. Upgrades and governance

**Done (v0.8).** Rule changes activate by version bits (green/versionbits.py, SPEC.md §5.5): miners signal a deployment in the header version, 90% of a 20,160-block period locks it in, it activates one period later (optionally not before a minimum height, as in Speedy Trial), it fails at its timeout or, if scheduled so, locks in there regardless. Activated rules must be soft forks, so nodes that never upgraded keep following the chain, and a node warns its operator when most blocks signal a change it does not know. Releases are signed by several maintainer keys over a `SHA256SUMS` file (`grn release sums|sign|verify`, green/release.py). The process for soft forks, hard forks, emergency fixes and releases is written down in [UPGRADES.md](UPGRADES.md).

**Evidence:** `tests/test_versionbits.py` (lock-in and activation, failure at the timeout, mandatory activation, minimum activation height, a reorg across a period boundary, the exact threshold, header version rules, old nodes following an activated soft fork, the unknown-bit warning on a live node), `tests/test_release.py` (threshold, duplicate and outside keys, tampered files, the commands end to end).

**Missing:** reproducible builds; the maintainer keys themselves (`RELEASE_KEYS` is empty until there are maintainers).

## 16. Privacy

**Transaction origin (v0.14):** Dandelion++ hides which node a transaction came from from observers of the network (§7).

**Not provided, and documented as such** (DESIGN.md §9): transactions and amounts are public, and a miner sees the matrices it computes. Future work: secret sharing, then attested secure hardware (TEE).

## 17. Performance and implementation

**Today:** a Python reference implementation; measured: 1.1–1.5 ms of ledger work per block (state root included), ~0.6 ms of storage per block, 9 ms to verify a block's work unit on one core (about 4 ms each with 8 threads during sync), restart in 0.04 s.

**Missing:** a native node (Rust) implementing SPEC.md, for throughput, memory safety and smaller attack surface; the Python node remains the readable reference and test oracle.

## 18. Source control and releases — blocker

**Today (v0.14):** the repository is under git; v0.13.0 is the first commit and every version is tagged. The CI workflow (Linux, Windows and macOS, Python 3.12 and 3.13, plus a Docker smoke test) is in `.github/workflows/`.

**Missing:** a public repository with a protected main branch, CI actually running on it, signed tags and reviewed merges.

## 19. Economics and launch

**Today:** fixed supply schedule with no premine (the genesis reward is the only coinbase not mined in competition, 1.25 GRN), fees to miners, data fees against storage spam, bonds and forfeits in both markets.

**Missing:** a fair-launch plan (announced genesis date, open-source release beforehand, no founder allocation); a public testnet run long enough to observe fees, delivery and disputes under real use; economic modeling of miner incentives when compute demand is low or bursty.

## 20. Legal and compliance — blocker

Not started. Before anything carries monetary value: legal advice in each jurisdiction involved (in Turkey, Law No. 7518 on crypto assets and Capital Markets Board rules), terms for running public infrastructure, and a decision on what entity, if any, publishes releases.

## 21. Ecosystem

**Missing:** a block explorer, a website with documentation, wallets for phones and desktops, exchange integrations, community channels, a foundation or maintainer group.

## What to do next, in order

Done in v0.7: ~~peer-to-peer transport with encryption and announce-then-fetch relay (§7)~~, ~~headers-first and snapshot sync with parallel work-unit verification (§8)~~. Done in v0.8: ~~an activation mechanism for upgrades and a signed release process (§15)~~.

Done in v0.14: ~~version control (§18)~~, ~~seed-phrase HD wallets (§10)~~.

1. Publish the repository and let CI run (§18).
2. NAT traversal and seed nodes, then a public testnet with many independent nodes, run for months (§7, §13, §19).
3. Independent audit and bug bounty (§14); legal review (§20).
4. Native node and miner from SPEC.md (§9, §17).
