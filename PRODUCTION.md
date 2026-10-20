# Production readiness

Where Green stands on every aspect a production cryptocurrency needs, what backs each claim, and what is still missing. "Done" means implemented, tested and measured in this repository; it does not mean audited, battle-tested or ready to carry real value. Nothing here is ready for real money until the items marked **blocker** are done.

Status as of v0.24.1 (October 2026). 334 automated tests (330 pass and 4 GPU tests skip on a machine without a GPU); internal security reviews only (SECURITY.md).

## Summary

| Area | Status | Blockers before real value |
|---|---|---|
| 1. Consensus rules and specification | Done for a testnet | external review of SPEC.md |
| 2. Proof of useful work | Done; custom digest unaudited | audit of the transcript digest |
| 3. Difficulty and issuance | Done (ASERT, fixed 21M supply paid from a reserve, fee recycling, shares, supply audit) | — |
| 4. Compute markets (matmul jobs, instant channels, WASM) | Done; optimistic, documented limits | data availability for manifests and WASM outputs |
| 5. Accountability for autonomous agents | Done (guarded accounts) | — |
| 6. State, storage, crash safety | Done (with a state commitment in every header) | — |
| 7. Peer-to-peer networking | Done for a testnet (encrypted, authenticated, eclipse limits, Dandelion++, compact blocks, PCP/NAT-PMP) | DNS seeds, Tor/I2P |
| 8. Initial sync of new nodes | Done (headers-first, parallel verification, verified snapshots) | — |
| 9. Mining software | Partial (CPU/GPU miners, non-custodial pools that sell compute; GPU far below hardware peak) | native GPU miner |
| 10. Wallets and keys | Done for a testnet (recovery phrases, encrypted files, checksummed addresses, multisig) | hardware wallets, a registered coin type |
| 11. APIs and developer tools | Done for a testnet (versioned API, fee estimates, long-poll, Python SDK) | SDKs in other languages, explorer API |
| 12. Operations | Done for a testnet (config file, logs, systemd, metrics and alerts, runbook) | — |
| 13. Testing and QA | Partial (334 tests, fuzz of transactions and P2P messages, mutation testing, 8-node simulation) | multi-node testnet, soak tests |
| 14. Security assurance | Partial (internal reviews only) | **blocker**: independent audit, bug bounty |
| 15. Upgrades and governance | Done (version-bits activation, signed releases, published process) | reproducible builds |
| 16. Privacy | Partial (transaction origin: Dandelion++; compute requests sealed and relayed, data split or secret-shared; amounts public) | private payments |
| 17. Performance and implementation | Partial (Python reference) | native node for scale |
| 18. Source control and releases | Partial (git, tagged versions, private GitHub repository; CI jobs have not run since v0.23.3 because the account's Actions minutes ran out) | **blocker**: public repository, signed tags |
| 19. Economics and launch | Partial (supply set; no launch plan) | fair-launch plan, public testnet |
| 20. Legal and compliance | **Missing** | **blocker** |
| 21. Ecosystem | **Missing** (no explorer, exchanges, website) | — |

## 1. Consensus rules and specification

**Done.** The rules are written down independently of the code in [SPEC.md](SPEC.md): encodings, addresses, every transaction type and its validity rules, end-of-block events, Merkle trees, the work unit bit for bit, difficulty, fork choice, block limits, WASM execution and constants. Block size is bounded (3,000 transactions, 1 MB; 500 transactions before v0.22); forks that leave the main chain more than 100 blocks below the tip are refused (measured at the fork point since v0.5).

**Evidence:** every rule has tests (`tests/test_chain.py`, `tests/test_security.py`); the work-unit description in SPEC was checked line by line against the code.

**Missing:** a second, independent implementation built from SPEC.md alone (the strongest test of a spec), and outside review of the spec.

## 2. Proof of useful work

**Done.** Integer matrix multiplication with seed-derived low-rank noise; every unit is both a lottery ticket and a deliverable; the useful result is recovered cheaply; knowing A·B gives no shortcut (noise rank ≥ tile size); bit-exact on CPU and GPU (the GPU is checked against the CPU at startup). A unit is mined or checked in 2.1 ms on one core (17 ms in v0.6, 8.4 ms in v0.17): since v0.18 the tile products are computed one column block at a time in float32, which is exact for them, and digested while in cache, bit for bit the definition.

**Evidence:** `tests/test_chain.py` (tampered blocks, stolen tickets), `tests/test_gpu.py` (bit-for-bit), `tests/test_work.py` (the fast digest against its definition), DESIGN.md §3.

**Missing:** the transcript digest is a custom function (lowbias32 per element since v0.18) chosen for GPU speed, analyzed but not proven (SECURITY.md); either an audit or BLAKE2b/BLAKE3 on the GPU.

## 3. Difficulty and issuance

**Done.** Per-block ASERT difficulty since v0.6: in simulation with miners that come and go, the old windowed rule ran 37% slow and drifted 9.3 hours behind schedule in 6,000 blocks; ASERT held 15.0 s and zero drift (`bench/difficulty_sim.py`). Issuance (v0.17): a fixed 21M GRN, all of it in the protocol's reserve at genesis; each block pays out 1/2^24 of the reserve (~1.25 GRN at first, a smooth curve with no halvings). Since v0.23, 99% of each block's emission pays shares (tickets 128 times easier than a block, ~128 a block) so miners are paid in proportion to their work, and of every fee 50% goes to the miner, 30% back into the reserve and 20% is burned. Every node can audit the supply (`grn supply`, `GET /supply`): balances, escrow, bonds, channel funds and the reserve must add up to the fixed supply minus what was burned, exactly (`tests/test_economics.py`).

**Evidence:** `test_asert_matches_the_formula`, `test_difficulty_follows_the_schedule_every_block`, `test_emission_matches_bitcoin_schedule`, the supply audit on a live node (`test_supply_audit_and_metrics_on_a_live_node`), and two fuzz tests that check conservation of money at every block.

**Known trade-off:** after a sudden 10× hashrate surge ASERT takes ~1 hour to reach 15 s blocks again (the old rule took minutes, but drifted for good).

## 4. Compute markets

**Done.** Matmul jobs: a whole job delivered with two transactions (bitmap + Merkle root), buyer-verified with prime-field Freivalds (2^-124), disputes for any number of units, whole-batch forfeit, backlog-stretched deadlines. WASM tasks: deterministic sandbox, optimistic with on-chain re-execution, watchtowers, per-block adjudication budget.

**Evidence:** `tests/test_chain.py`, `tests/test_compute.py`, `tests/test_network.py` (a cheating miner caught by the buyer's node; a miner answering a dispute with a Merkle proof), SECURITY.md reviews W, V and X.

**Instant compute (v0.15):** compute channels (DESIGN.md §4.5): a buyer locks coins for a provider in one transaction, then buys products directly, checks each with Freivalds' test and pays by voucher; the provider claims on chain. Providers advertise signed, funded offers over P2P. A model trains on products computed by miners at about 12 ms a step (`examples/train_on_green.py`; `tests/test_channels.py`, `tests/test_channel_compute.py`: the model trained remotely is bit-identical to the one trained locally, wrong products are caught and not paid, credit is limited, offers must be signed and funded).

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

**Dandelion++ and compact blocks (v0.14):** a node's own transactions travel a stem before anyone announces them (routes fixed per epoch and incoming link, stem pool never served, loops end the stem, embargo timers on every stem node); new blocks cross a link as salted short ids and are rebuilt from the mempool, so a 20-transaction block costs under a quarter of its size and usually no round trip (`tests/test_relay.py`, 7 tests, each mutation-checked). Measured on 8 nodes (`bench/net_sim.py` with and without `--no-dandelion`): transactions reach every mempool in a median of 0.28 s with stems and 0.27 s without, so the privacy costs no noticeable delay.

**Evidence:** `tests/test_p2p.py` (27 tests: nothing readable on the wire, a corrupted byte drops the link and it recovers, pinning, other networks refused, each transaction crosses each link once, silent announcers skipped, nodes find each other through gossip, fake headers banned, oversized frames, unsolicited data, floods and silent handshakes cut off, unfinished handshakes capped, only authenticated newcomers can evict, stalling peers penalised, eviction and address-book rules); `bench/net_sim.py`; SECURITY.md P1–P5, P10.

**Port mapping (v0.16):** `--portmap` asks the home router to forward the P2P port and a provider's port with PCP (RFC 6887), falling back to NAT-PMP (RFC 6886), renews the mapping and removes it on shutdown; a router behind carrier-grade NAT is detected and nothing is advertised. Tested against fake gateways only (`tests/test_portmap.py`): real routers vary, so this needs a field test. Gossip can no longer make nodes dial private addresses or look up names.

**Missing:** DNS seeds (`SEED_NODES` is empty until launch); UPnP (deliberately left out: a much larger attack surface); Tor/I2P; Elligator-encoded keys and traffic padding against protocol fingerprinting; a content-routing layer (DHT) so a buyer can always find a miner's results.

## 8. Initial sync

**Done (v0.7).** Headers first, each checked against every rule except its work unit, with a random sample of work units recomputed per message before any block is fetched; blocks downloaded from several peers at once in a 256-block window; each block's work unit sent to 8 verification threads the moment it arrives while a separate thread connects blocks in order. Measured with `bench/sync_bench.py` on 600 blocks: full sync 4.2 s (144 blocks/s; before this pipeline 14.6 s), fast sync 1.4 s. Since v0.18 a long sync checks the work units of blocks and their shares in worker processes (green/verify.py): 300 blocks with 32 shares each sync in 7.2 s (66 s in v0.17), or 5.2 s from a snapshot. Fast sync: nodes write the ledger every 2,880 blocks as content-addressed chunks, and a new node starts from the newest final snapshot after checking it against the `state_root` in that block's header, then validates every later block. This is trustless relative to the heaviest header chain (Ethereum's snap-sync model), where Bitcoin's assumeutxo trusts a hash shipped with the software. It also removes the need for old job matrices to join the network.

**Evidence:** `test_headers_first_sync_from_several_peers` (headers in rounds, both peers serve blocks, work units checked in parallel), `test_fake_headers_are_caught_before_any_block_is_fetched`, `tests/test_snapshot.py` (a snapshot rebuilds the exact ledger and the chain continues; tampered snapshots refused; a new node joins from a snapshot over the network; a lying snapshot server is banned and sync falls back), `bench/sync_bench.py`.

**Missing:** GPU verification of work units during sync; background validation of pre-snapshot history for nodes that want it; pruned nodes cannot serve blocks below their snapshot (they say so in their handshake).

## 9. Mining software

**Today:** CPU and GPU (WebGPU/Vulkan) miners, bit-exact; a mining node does ~800 units/s on an RX 9070 XT and ~400 on the CPU (v0.20: batched, pipelined, threaded); miners pick the best-paying open units and deliver them in batches.

**Non-custodial pools (v0.13):** a block can pay up to 32 payees by weight, so a pool's blocks pay its miners directly; `green/pool.py` has the pool server (PPLNS templates, every share's work recomputed, bad shares refused and their senders paused) and a pool miner (`tests/test_pool.py`).

**Pools sell compute (v0.14):** a pool assigns its miners units of open jobs, checks each result with Freivalds' test, commits them in batches and posts the bond; each commit names the miners who computed its units as payees, so consensus pays them when the batch settles (`test_pool_miners_compute_jobs_and_the_batches_pay_them`: the buyer gets correct products, the pool keeps exactly its fee).

**Missing:** a mining node feeds its GPU ~800 units/s (v0.20), a small fraction of a modern card's int8 throughput, held back by preparing the SHAKE-256 inputs and noise on the CPU in Python; those generated on the GPU, or a native miner. Pool miners compute job units on the CPU even when they mine on a GPU; miners cannot yet choose a pool block's transactions (as Stratum V2 allows); results travel to the pool over plain HTTP (256 KiB per unit).

## 10. Wallets and keys

**Today:** wallets made from a 24-word recovery phrase (v0.14: BIP-39, SLIP-0010 Ed25519 derivation, optional passphrase, numbered accounts; the official word list is checked against its published hash when loaded); wallet files encrypted with scrypt + AES-256-GCM; bech32m addresses that reject typos; k-of-n multisignature accounts (v0.9) with a cosigning workflow (`grn multisig`), usable as an agent's guardian.

**Evidence:** `tests/test_hd.py` (BIP-39 reference vectors, SLIP-0010 test vector 1, typos caught and never guessed, restore with and without a password, older wallets still open).

Since v0.12 a wallet can track an address's history and verify its payments as a light client.

**Missing:** hardware-wallet support; a registered SLIP-0044 coin type (7777 is provisional); watch-only wallets in the command line.

## 11. APIs and developer tools

**Today (v0.19):** `grn app`, a local app in the browser (mine, send, buy compute, train a model) with a compute proxy that other programs call like a local matrix multiplier; `grn localnet`, a network of five nodes on one computer to try everything. **Since v0.10:** every HTTP endpoint documented in [API.md](API.md) under a stable `/v1` prefix; fee estimates from the mempool and recent blocks (`GET /v1/fee`, used by the command line); long-polling for new blocks (`GET /v1/wait`), used by the SDKs; a Python SDK for buying compute (`green.nn.MatmulClient`), autonomous agents (`green.agent`) and multisig (`green.tx`); a command line covering every transaction type.

**Evidence:** `tests/test_api.py` (the versioned paths, the fee estimate against a full mempool, long-poll wake-up, timeout and waiter cap).

Since v0.12: a transaction and address index kept consistent through reorgs (`/v1/tx`, `/v1/address/.../history`), inclusion proofs, and a light client (`green.light`) that verifies headers and payments itself (`tests/test_light.py`).

**Missing:** SDKs in other languages; notifications for transactions to an address (not only blocks); a graphical block explorer.

## 12. Operations

**Today (v0.10):** clean stops with `grn stop` or SIGTERM (v0.19), a config file (`green.toml`, strictly checked), log levels and a rotated log file, a hardened systemd unit that passes the wallet password as a systemd credential, Prometheus metrics with alert rules (supply audit, peers, stalls, sync, warnings, ban storms), a runbook ([OPERATIONS.md](OPERATIONS.md): settings, firewall, service, monitoring, backup and restore, upgrades, troubleshooting), a Docker image (built and smoke-tested by CI; not built on the development machine because Docker was not running there), and an installable `grn` command.

**Evidence:** `tests/test_api.py` (config file precedence and refusal of bad settings, the password file, the log file, the supply-audit metric).

**Missing:** a public status page; packages for operating systems (deb, rpm, Homebrew, Windows installer).

## 13. Testing and QA

**Today:** 334 tests across consensus, security regressions, addresses, fees, state and storage, the state commitment and snapshots, the P2P protocol, atomicity, networks of live nodes, GPU, WASM, both markets, the agent and its guardian; two fuzz tests with exact accounting invariants; mutation testing proved the fuzz catches injected accounting bugs; a differential test against a reference replay; an 8-node network simulation with a split and a heal (`bench/net_sim.py`, v0.14: blocks reach every node in a median of 0.02 s, the halves converge in 0.6 s with identical state roots); CI configured for Linux, Windows and macOS.

**Since v0.19** CI runs on GitHub (a private repository for now): every push is tested on Linux, Windows and macOS with Python 3.12 to 3.14, plus a Docker smoke test; the first run (3.12 and 3.13) passed everywhere. Since v0.23.3 every job has ended within seconds without starting (no logs), the sign of exhausted Actions minutes, not of failing tests: the full suite passes locally on v0.24.1 (330 passed, 4 GPU tests skipped, Linux, Python 3.13).

**Soak (v0.24.1, October 2026):** the local network (five nodes, three CPU miners, two buyers training models and posting jobs) ran 1 h 40 min in two runs with a clean stop and restart between them: 289 blocks; every node on the same chain at every one-minute sample (state roots compared) and the supply audit passing at every sample; 52,651 products bought in the first hour, 95 models trained, 38 jobs posted; 3 of ~150 purchases failed: two because a buyer's job post never left its node and its channel opens waited behind it (fixed: data-carrying posts skip the Dandelion stem), one while the providers restarted. It also found two threads storing one blob failing (fixed). Threads and open files stayed flat; resident memory grew in the first ten minutes and then levelled off (nodes ~250-300 MB, buyers ~300-490 MB), with a slow drift on the nodes that a run of days must settle. The 8-node simulation (`bench/net_sim.py`) still splits and heals to one chain with identical ledgers.

**Missing:** multi-node testnets with tens of nodes on separate machines; week-long soak tests (memory over days); property-based testing at larger scale; conformance tests shared with a second implementation.

## 14. Security assurance — blocker

**Today:** seventeen internal reviews with every finding fixed and, where it can be tested, a regression test checked to fail without its fix (SECURITY.md), including bugs found in this project's own earlier fixes and by running a network of five nodes (v0.19).

**Missing:** an independent professional audit of consensus, cryptography and the dispute games; a public bug bounty; a responsible-disclosure policy; formal analysis of the economic games (disputes, delivery, guards).

## 15. Upgrades and governance

**Done (v0.8).** Rule changes activate by version bits (green/versionbits.py, SPEC.md §5.5): miners signal a deployment in the header version, 90% of a 20,160-block period locks it in, it activates one period later (optionally not before a minimum height, as in Speedy Trial), it fails at its timeout or, if scheduled so, locks in there regardless. Activated rules must be soft forks, so nodes that never upgraded keep following the chain, and a node warns its operator when most blocks signal a change it does not know. Releases are signed by several maintainer keys over a `SHA256SUMS` file (`grn release sums|sign|verify`, green/release.py). The process for soft forks, hard forks, emergency fixes and releases is written down in [UPGRADES.md](UPGRADES.md).

**Evidence:** `tests/test_versionbits.py` (lock-in and activation, failure at the timeout, mandatory activation, minimum activation height, a reorg across a period boundary, the exact threshold, header version rules, old nodes following an activated soft fork, the unknown-bit warning on a live node), `tests/test_release.py` (threshold, duplicate and outside keys, tampered files, the commands end to end).

**Missing:** reproducible builds; the maintainer keys themselves (`RELEASE_KEYS` is empty until there are maintainers).

## 16. Privacy

**Transaction origin (v0.14):** Dandelion++ hides which node a transaction came from from observers of the network (§7).

**Not provided, and documented as such** (DESIGN.md §9): transactions and amounts are public, and a miner sees the matrices it computes. Future work: secret sharing, then attested secure hardware (TEE).

## 17. Performance and implementation

**Today:** a Python reference implementation; measured: 1.1–1.5 ms of ledger work per block (state root included), ~0.6 ms of storage per block, 2.1 ms to check a work unit on one core (0.75 ms each across 8 threads, ~0.45 ms across 8 worker processes during sync), restart in 0.04 s.

**Missing:** a native node (Rust) implementing SPEC.md, for throughput, memory safety and smaller attack surface; the Python node remains the readable reference and test oracle.

## 18. Source control and releases — blocker

**Today (v0.19):** the repository is under git with every version tagged (v0.13.0 is the first commit), pushed to a private GitHub repository where CI tests every push (Linux, Windows and macOS, Python 3.12 to 3.14, plus a Docker smoke test); runs since v0.23.3 have not started for lack of Actions minutes.

**Missing:** making the repository public, with a protected main branch, signed tags and reviewed merges.

## 19. Economics and launch

**Today:** a fixed supply with no premine (the genesis block's reward is the only one not mined in competition, ~0.0125 GRN), paid out from the reserve with no halvings and no inflation, 30% of every fee recycled into the reserve and 20% burned, fees of 0.000005 GRN per signature, data fees against storage spam, bonds and forfeits in both markets.

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
