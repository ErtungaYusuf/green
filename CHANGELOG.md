# Changelog

Every release so far is a consensus change: nodes of different versions do not form one network.

## 0.23.3

Checked end to end before publishing:
- **A peer that relays a transaction with a bad signature is penalized** (two strikes ban it). Every refusal used to be treated as a policy difference between nodes, so junk cost the sender nothing while each bad signature cost us a check; with fees of 0.000005 GRN that mattered more. An honest node checks signatures before it relays, so a bad one is proof; refusals that depend on state (nonce, balance) or policy still carry no penalty. (`test_a_relayed_bad_signature_counts_against_the_peer_but_policy_does_not`.)
- **The per-parent share cap of v0.23.2 would have cost miners ~13% of their share income.** The number of shares before a block is random (geometric, mean 128); 13% of blocks see more than 256, and later blocks of the window pay that backlog. The cap is now 1,024, which drops 0.03%. (`test_the_share_cap_drops_almost_nothing`, which fails with the old cap.)
- **Miners slowed down 4-5x within hours.** A job stays in the state after all its units are delivered, until its buyer cancels it, and a miner listed every job's units for every try to pick its work: after five hours of the local network's buyers posting jobs (257, all finished), a GPU miner did ~80 units a second instead of ~440. A finished job now answers at once, and the options are worked out once per batch of tries instead of once per try; restarted on the same chain, the miners went back to ~440 a second. (`test_finished_jobs_cost_a_miner_nothing`, which fails without the fix.)
- The signature cache no longer keeps the vouchers a provider checks once (30,000 an hour on the local network) and holds 50,000 signatures instead of 200,000, still 8 full blocks.
- **A pool could hand the same unit to two miners.** It released a unit the moment it accepted the result, before the node had taken the result in, and a miner asking for work in between got that unit again: computed twice, the second miner's work thrown away unpaid. Found as a test that failed one run in ten; the unit is now held until the node has the result. The same gap was there a step later: a batch's units, taken out of the buffer to be committed, looked free until the commit reached the mempool; they now count as taken until then. (`test_a_unit_being_handed_in_is_not_given_to_a_second_miner`, `test_units_on_their_way_to_a_commit_stay_taken`; each fails without its fix every time.)
- Every `grn` command written in the documentation was parsed with the real command line: three said `wallet create` (it is `wallet new`). The multi-computer steps of OPERATIONS.md were followed with two nodes (wallets, genesis, `--peer`, mining, a payment from one to the other, a restart that finds its peer again from `peers.json`, supply audit, clean stop), and the four repositories, cloned fresh, pass all their tests.
- PRODUCTION.md still described the economics of v0.17; the README links the three interfaces.

## 0.23.2

- **Found on the local network: the network split in two within five minutes of starting.** A new network starts at a difficulty of ~200 work units a block, so with three GPU miners blocks came every second, and with 128 shares a block (since v0.23) more than 100 shares a second. The buyers' nodes, busy checking bought compute, could not keep up, took their honest peers for attackers ("flooding faster than it can be processed") and banned them for 24 hours: carol and erin went on alone at block 181 while the miners reached 275. Three fixes:
  - A node that falls behind drops shares, transactions and offers (the network carries them again) instead of blaming the peer; only blocks and headers count. (`test_a_node_that_falls_behind_drops_shares_instead_of_blaming_peers`.)
  - A node checks and relays at most 1,024 shares on the same parent block (4 x MAX_BLOCK_SHARES; v0.23.2 had 256, which would have dropped ~13% of all shares, since the count before a block is random and later blocks of the window pay the rest). (`test_a_burst_of_shares_on_one_parent_is_not_all_checked`.)
  - A network chooses its starting difficulty in its genesis block (never easier than INITIAL_TARGET): `grn genesis --units-per-block N`, and the local network starts at 10,000 units a block, so it no longer pays out its first hour's coins in minutes. (`test_a_network_chooses_its_starting_difficulty`.)
- Both tests fail without their fix.

## 0.23.1

- A node that was killed (or crashed) left its 8 block-checking worker processes running, idle, holding its log file open: 40 of them had been left by local-network nodes stopped hours earlier. Workers now exit as soon as the node that started them is gone. (`test_workers_die_with_a_killed_node`, which fails without the fix.)
- DNS seeds: a seed given by name is resolved and up to 16 of the addresses it answers with are tried, as Bitcoin's DNS seeds work, so one name gives a new node many peers.
- OPERATIONS.md: how to run one network across several computers (genesis, `--p2p-host 0.0.0.0`, `--portmap`, `--peer`, seeds).

## 0.23.0

- **Miners earn for the work they do, not for luck.** 99% of each block's emission now goes to shares (was 80%), and shares are 128 times easier than a block (was 32): about 128 a block, one every ~0.12 s across the network. Finding the block is worth about one share plus a 5% cut of the shares it includes and half of the fees. On the local network over 300 blocks, each miner's part of the rewards already matched its part of the work within a point; with the new setting a GPU with 0.01% of a large network gets ~74 shares a day, and its daily income stays within ~12% of what its work deserves (32% before; `bench/economics_sim.py`). Nobody needs a pool. Checking 128 shares costs a node ~0.3 s per 15 s block (2.4 ms each).
- **A slow, small deflation.** Of every fee, 50% goes to the block's miner, 30% back into the reserve (paid out again as rewards, so rewards never dry up) and 20% is burned (state entry `burned`). No coin is ever created; the supply is 21M minus what was burned, and shrinks a little with use: at 10 transfers a second, ~0.0015% a year. `GET /supply` and `grn supply` show what was burned.
- [green-hub](https://github.com/ErtungaYusuf/green-hub): wallet, compute buying and mining in one page in the browser.
- Consensus change: a new network (the payout and the supply rule differ from 0.22).

## 0.22.0

- **Fees as low as Solana's.** The minimum fee is per signature a validator checks: 500 yaprak (0.000005 GRN, as Solana's 5,000 lamports) for a plain transaction, k times that for a k-of-n multisig one, twice for a channel claim (its voucher). It was a flat 10,000 yaprak. When blocks fill, transactions still compete by fee rate (the priority fee), and half of every fee still goes back to the reserve.
- **A transfer that opens an account sends at least 0.0001 GRN** (`NEW_ACCOUNT_MIN`), so cheap fees do not make filling the ledger with empty accounts cheap. It plays the part of Solana's rent-exempt minimum; the coins are the recipient's.
- **Six times the transactions per block:** at most 3,000 (was 500), 200 transfers a second, and 6,000 signatures (was 2,000). The 1 MB limit on a block's transactions, which bounds what nodes download and store, is unchanged. Measured with the new `bench/tps_bench.py`: a full block of transfers is checked in ~0.33 s cold and ~0.15 s when its transactions were seen in the mempool, on one core.
- A signature cache, as in Bitcoin: a transaction checked when it reached the mempool is not checked again when its block comes. Addresses derived from keys and address checks are cached, and the hex check is a regular expression: a block's transactions are applied 2.5x faster (8,000 to 20,000 transfers/s warm).
- [COMPARISON.md](COMPARISON.md): Green against Bitcoin and Solana: the problem each solves, the numbers side by side, and what Green still has to catch up on.
- The wallet and the miner pages now live in their own repositories, [green-wallet](https://github.com/ErtungaYusuf/green-wallet) and [green-miner](https://github.com/ErtungaYusuf/green-miner).
- Consensus change: nodes before 0.22 refuse blocks with more than 500 transactions and transactions below the old minimum fee.

## 0.21.7

- `grn app --localnet` failed with "no such file" when started from another folder than the one the local network was made in: the plan kept the personas' folders relative to where `grn localnet start` ran. They are now found inside the network's own folder, so existing networks work from anywhere.
- The app's secret mode always failed with "secret sharing needs three providers": an account opened channels to only two providers. It now opens three, as the local network's buyers already did.
- Checked by hand on the local network: the miner and wallet pages of green-ui (a received payment shows in the history, a sent one confirms in the next block) and a secret-shared purchase from the app page.

## 0.21.6

- A local network's buyer that spends more than five minutes on one purchase writes where every thread is to its log (`faulthandler`), as nodes do when their shutdown hangs.
- Measured on the local network with v0.21.5, in seven minutes: two buyers trained 10 models through sealed requests carried by peers (93% test accuracy at 200 steps), bought dozens of products split and secret-shared, and posted jobs on chain, with no refusals.

## 0.21.5

- Training through sealed requests often failed with "busy": a product is cut into tiles sent at once, more than one peer would carry for us at a time, and the first refusal ended the request. A busy peer now hands the request to the next one, and when all are busy it waits a moment and tries again until its deadline; a peer carries up to 4 of a node's requests at once (was 2).

## 0.21.4

- Found with the thread dump of v0.21.2: a node too busy to carry another sealed request refused it while holding the peer's lock, and sending takes that lock, so the connection's reader locked itself up for good. The node stopped hearing that peer (the stalls of the last runs), and its shutdown waited forever on the same lock. Introduced in v0.21; the refusal is now sent after the lock is released. (`test_a_refused_relay_does_not_lock_up_the_connection`, which hangs without the fix.)
- 274 tests.

## 0.21.3

- On Windows a virtual environment's `python.exe` is a launcher that runs the real interpreter as its child. The local network recorded the launcher's process id, so `stop` waited on the wrong process, and a node stopped by its id kept running behind it: an old, stuck node of the local network went on answering on its port while its replacement never started (the "node that keeps stalling" of the last run). The local network now asks each node for its own id (`GET /debug/process`, admin).
- A node whose shutdown has not finished after 45 s writes where every thread is to its log and exits: one did hang after "Shutting down...", and the log is how the cause will be found. The store's write-ahead log keeps the chain safe.
- 273 tests.

## 0.21.2

- A buyer whose own node could not carry a sealed request (it was restarting) counted that against the providers, gave up all three channels as dishonest and opened new ones, locking more coins. Failures to carry a request are now their own error (`RelayError`) and leave the channels alone.
- `GET /debug/threads` (admin): where every thread of the node is, for diagnosing a node that hangs. A buyer's node on the local network stalled for four minutes during the catch-up after an outage and recovered when its peers reconnected; the cause is not found yet, and this is how it will be.

## 0.21.1

Found when the local network came back after 15 hours off (ASERT then lowers the difficulty as far as it goes, and blocks come many per second until the chain is back on schedule):
- A miner whose chain's clock ran ahead of real time waited for it while holding the node's lock (v0.20): the node stopped handling blocks and peers, and banned its honest peers as flooders. It now waits outside the lock. (`test_a_miner_waiting_for_the_clock_does_not_starve_the_node`, which fails with the wait back under the lock.)
- A peer whose messages pile up faster than the node handles them is now disconnected, and banned only the third time within an hour; a node that fell behind used to ban every honest peer at once. (`test_a_flood_beyond_the_queue_is_cut_off`.)
- 271 tests.

## 0.21.0

**Anonymous compute**
- Sealed requests (`green/sealed.py`): providers advertise an X25519 key in their signed offers; every request is sealed to it with a fresh key pair (HKDF-SHA256, ChaCha20-Poly1305) and its answer comes back under the same key, so only the buyer can read the result.
- Carried by a peer (`relay`/`relayed`, P2P messages 25 and 26; `POST /relay` on the buyer's node): the provider never learns where a request comes from, the peer carrying it never learns what it is, and the buyer's node never contacts the provider. A node carries requests only to providers on offer, within per-peer limits.
- Data hidden from the providers (`green/private.py`), per request: *split* (secret row and column shuffles, a grid of tiles over the providers: nobody sees the whole problem; same cost) or *secret* (replicated secret sharing mod two primes over three providers: each sees only uniformly random shares; 12 times the compute). Providers serve exact modular products (`POST /matmul_mod`) and buyers check them with Freivalds' test mod p.
- The app uses all of it by default (sealed through a peer, split), with a privacy selector and `X-Privacy` on the compute proxy; the local network's buyers use split and secret.

**Mining**
- `--intensity PERCENT` and `POST /mining {"intensity", "gpu"}`: a miner works that percent of the time, and switches between GPU and CPU while running; `grn mine on --intensity 50 --gpu on`. The local network runs its three GPU miners at 50% by default.

**Also**
- Fixed, from v0.20's pipelined miner: units its batches were trying were not counted as busy, so a node that ran a pool could hand the same unit to a pool miner; the two results then collided in one batch, which the chain refused. In-flight units are busy now, and a batch never takes a unit twice. And the GPU check on switching to the GPU ran on the GPU object the miner thread was using (it is not thread-safe); each node now keeps its own checked GPU.
- Eighteenth security review (SECURITY.md, B1-B5). 270 tests. Consensus is unchanged; the P2P protocol gains two optional messages and offers an optional field, which older nodes ignore or refuse to relay.

## 0.20.0

**Mining that keeps the GPU busy**
- A mining node fed its GPU one nonce at a time and waited for each, and around every try it checked the whole share pool, often rebuilt its block template, and counted the unit by rebuilding a list of the last 30 seconds of units (the faster the miner, the slower each count). Measured in a mining node on an RX 9070 XT: 358 units/s, where the GPU alone does several thousand.
- Now the miner works in batches of 16 tries: the GPU digests one batch while the next is prepared (SHAKE-256 inputs and noise, in three threads), with two sets of GPU buffers; the template is rebuilt when the tip or the mempool changes, and for new shares at most once a second; the unit counter is constant time. The GPU kernel loads each pair of tiles into workgroup memory with coalesced reads and folds the 1,024 mixed products in a tree, instead of 1,024 atomic writes to 8 addresses. Bit for bit the same transcripts (tests compare every try of full and partial batches with the CPU; the node still checks its GPU against the CPU at start).
- Measured: one mining node 358 → ~800 units/s on the GPU; three GPU miners of the local network ~630 units/s each, sharing one card. CPU mining also uses the threads: 137 → 398 units/s. The rest is preparing inputs in Python (SHAKE-256 holds the interpreter lock); generating them on the GPU is the next step.

**The local network: miners and buyers**
- Three miners (alice, bob, dave) mine on the GPU by default (`--cpu` to use the CPU; a node without a usable GPU says so and mines on the CPU) and sell compute; two buyers (carol, erin) each run a buyer process next to their node that keeps asking the network for compute: it buys coins from the richest miner when it runs low, then trains small models and buys products through payment channels, and posts matrix jobs to the on-chain market. `grn localnet status` shows what they bought. In a 45-minute run: 65,000 products, about 100 models trained (93% test accuracy at 200 steps) and 54 jobs posted, for about 1 GRN.
- The app can post on-chain jobs (`App.post_job`) and open channels to more providers; errors in the app and the buyers' logs carry the node's own explanation.

**Also**
- A miner no longer mines blocks its peers would refuse as from the future: when blocks come many per second (each must be later than the median of the last 11) the chain's clock runs ahead of real time, and past MAX_FUTURE_SECONDS the miner now waits instead of churning out rejected blocks. Found when the faster miner stalled a test's chain.
- 265 tests. Consensus is unchanged.

## 0.19.1

- Verification workers start with `spawn` on every platform. On Linux, Python before 3.14 forks by default, copying a node whose other threads may hold locks the child would wait on forever.
- CI runs on GitHub (a private repository for now); its first run passed on Linux, Windows and macOS with Python 3.12 and 3.13 and the Docker smoke test. Python 3.14 joins the matrix, and the workflow uses the current versions of the checkout and setup-python actions.

## 0.19.0

**A network on your computer, and an app to use it**
- `grn localnet start | status | stop | reset`: five people, each running their own node as a separate process, with their own wallet, data folder, ports and log, on a network of their own: alice and bob mine and sell compute, carol buys it (she mined the genesis block, so she can start at once), dave is a small miner paid through shares, erin runs a full node. Nothing is special-cased; they talk over the real encrypted P2P protocol. `stop` asks every node to shut down and waits until each has exited; it never kills a process by a remembered id, which could belong to something else by then. `reset` deletes only a folder that holds a local network.
- `grn app` (`--localnet` for the five people): a page in the browser to turn mining on and off for anyone, send coins, buy a matrix product from the miners (paid through payment channels opened on first use, checked with Freivalds' test, compared with the same product computed locally) and train a small network whose every product is bought from the miners. Measured on the local network: a 1024³ product in 47 ms split over two providers, channels opened in 1.2 s; 200 training steps (1,000 products) in 1.4 s for 0.0125 GRN, 93% test accuracy.
- The app's **compute proxy**: `POST /v1/matmul` with `X-Shape: m,k,n` and A, B as int8 returns the int32 product, bought on the network, checked and paid by the app (API.md). Other programs use the network like a local matrix multiplier.

**Stopping cleanly**
- `grn stop` (admin `POST /shutdown`), and the node now shuts down cleanly on SIGTERM and Ctrl+Break as well as Ctrl+C. Until now `systemctl stop` and `docker stop` ended it without its shutdown: a provider did not claim what buyers owed it until the next start.

**Security**
- Admin endpoints (mining switch, peers, blob upload, shutdown) only checked that the caller was on the node's machine, and a web page open in a browser there is: it could add peers of its choosing (an eclipse attack) or switch mining off. They now also need an `X-Green-Client` header, which a page cannot send cross-site without a CORS preflight, and a Host naming the machine, which a DNS-rebinding page does not. Green's own tools send the header.
- Seventeenth security review (SECURITY.md, A1–A6).

**Found by running the five nodes**
- `grn peers` showed each peer's height from the handshake (0 or 2 on a chain at 512); it now follows the blocks each peer sends.
- A transaction still waiting showed its recipient as "named"; it now has the role it will have once confirmed (recipient, account, guardian).
- `grn fee` printed "GRN GRN".
- Starting a local network on ports another one uses said only that a node exited; it now says which port and why, before creating anything.
- A node in fast sync chose its snapshot from the peers connected at the moment its headers settled; a peer that offered one but connected a moment later was missed, and the node replayed history from genesis instead. It now waits for a second peer, or 10 seconds. (Found as a rare failure of the lying-snapshot test in a full run.)
- A network test waited for two tips to be equal while one node kept finding a block every few milliseconds (v0.18 made it four times faster), which only luck could satisfy; it now stops mining first.

**Also**
- The README starts with the local network and the app; its setup no longer assumes one particular computer.
- 264 tests (the app, the local network started and stopped as a user does it, the admin guards). Consensus is unchanged from v0.18.

## 0.18.0

**Work units 4× faster**
- The transcript digest mixes each value with lowbias32 (two xorshift-multiply rounds) instead of the MurmurHash3-style mix: lower measured avalanche bias, half the operations. This is the consensus change of this release, pinned by a test vector (SPEC.md §5.2).
- The tile products are computed one column block at a time (`A1[:, l] · B1[l, :]` holds every tile pair of one `l`: 8 BLAS calls instead of 512 tiny products), in float32, and each block is digested while it is still in cache, so the 2 MB transcript is never held in memory. float32 is exact for tile products (at most 819,200 < 2^24, 20× headroom, asserted when `green.work` loads); v0.7 rejected it for the whole product, which has under 3×.
- Mining and checking skip A·B; the deliverable comes straight from the BLAS when it is needed.
- One BLAS thread by default (`green/__init__.py`, before numpy loads; an operator's OPENBLAS_NUM_THREADS etc. wins): a BLAS thread per core made each unit slower and serialized the node's threads. Large products (providers, training) are split into row or column blocks computed on every core in threads instead (`work.blas_matmul`): 4096³ in 1.06 s against 1.13 s with a BLAS thread per core, a 63×4096×4096 product in 29 ms. Buyers check a 2048³ product in 32–36 ms instead of 52–61 (`bench/compute_bench.py`, interleaved with v0.17); a 300-step training run takes 1.8–2.0 s instead of 2.1–2.4.
- Measured on one core (v0.17 and v0.18 interleaved, best of several runs): mining or checking a unit 8.4 → 2.1 ms (4.1×), a unit with its product 8.4 → 3.0 ms, across 8 threads 4.0 → 0.75 ms per unit. The GPU miner makes its noise in float32 too: ~700 → ~1,280 tickets/s on an RX 9070 XT.

**New nodes catch up 9× faster**
- Since v0.17 a block carries ~32 shares, each a work unit every syncing node checks. When 32 or more downloaded blocks wait to be connected, a pool of worker processes (green/verify.py: at most 8, each with one BLAS thread, stopped after 30 s idle) recomputes the transcripts of the blocks and their shares; the node alone decides what counts as checked. Threads could not do this: the interpreter lock caps them, and the 2 MB transcript made even separate processes run each unit 3× slower once 8 ran at once (memory bandwidth). The new unit's working set, about 1.5 MB, stays in a core's cache: 8 processes check a unit every ~0.45 ms.
- `bench/sync_bench.py [blocks] [shares per block] [--threads]`. 300 blocks with 32 shares each: full sync 66.1 → 7.2 s (9.2×), fast sync 44.1 → 5.2 s (8.5×), building them 376 → 123 s. 600 blocks without shares: full sync 6.3–6.8 → 3.1–3.3 s, fast sync 2.2 → 1.5–1.7 s, building them 23 → 6.8 s.

**Also**
- Every verified share goes through one bounded cache (a node's own shares were added without the bound).
- Two fetches of one blob from the same peer at once (the buyer's watchtower and its API client asking for one result) made the second give up on that peer, so a node with one peer answered 404. The second now waits for the first one's answer. (A full test run failed once with such a 404 after blocks got faster; this race is the likely cause, shown by a test that fails without the fix, though that run was not reproduced.)
- Found by a full test run, present since v0.16: a provider refusing a product from its headers hung up while the body (sent after the headers) was still arriving; the reset that a closed socket answers with can discard, on Windows, the refusal the buyer had not read yet, so a buyer whose channel the provider's node had not seen yet got "connection aborted" instead of a 404 to retry. The provider now drains and drops the body (at most the largest valid body, for at most 10 s) before hanging up; it still answers before reading a byte of it.
- Moving the reorg base keeps its states in one height-ordered pass instead of walking each state's ancestry (O(window log window) instead of O(window²) per block).
- `bench/compute_bench.py` funded its buyer from the genesis reward, which v0.17 cut to a fifth (the rest pays shares), and failed with a 400 since; it now turns shares off like the tests.
- Sixteenth security review (SECURITY.md, G1–G6). 257 tests. Consensus change (the transcript digest): v0.17 nodes reject v0.18 blocks.

## 0.17.0

**Money: no halvings, no inflation**
- The supply is fixed at 21M GRN and starts in the protocol's reserve (state map `protocol`). Each block pays out 1/2^24 of the reserve: ~1.25 GRN at first, then a smooth curve with no halving cliffs. Half of every block's fees flow back into the reserve, so rewards never dry up: after the initial distribution miners are paid from what users pay, smoothed over years. The supply audit (`GET /supply`, `grn supply`) checks balances, locked funds and the reserve against the fixed total. (SPEC.md §3.5.)

**Shares: paid for work, not luck**
- 80% of each block's emission pays *shares*: headers mined on one of the last 32 blocks whose ticket meets 32 times the block target. Blocks include them (at most 64, committed in the new header field `shares_root`); each share's miner is paid from the reserve and the including block gets 5% of it. A miner is paid in proportion to its work without a pool; a block that loses a race counts as a share. Nodes keep a share pool, miners publish their near-misses as shares, shares travel over P2P (`share`, checked before relay; fake ones get the sender banned) and inside compact blocks; their work units are checked in parallel during sync.
- The miner reuses its block template while the tip, the mempool and the shares are unchanged (it rebuilt it for every try). `bench/economics_sim.py` shows the emission curve, where rewards settle on fees, and a small miner's income with and without shares.
- Pools stay useful for selling compute and finer payouts; their own shares are now 1024 times easier than blocks, and a pool share that is also a protocol share pays the pool's payees.
- Found by the network simulation, present since v0.14: a sender's transaction could take a Dandelion stem while its predecessor was still being announced, reach a node that had not seen the predecessor, and be dropped there; every later transaction followed, and all waited ~20 s for the embargo (half the simulation runs). A transaction now takes a stem only if its predecessor is confirmed or on the same stem, and a transaction with a future nonce is no longer marked rejected, so it is fetched again once its predecessor arrives.
- Fifteenth security review (SECURITY.md, E1-E9). 249 tests. Consensus change (header field, rewards): v0.16 nodes reject v0.17 blocks.

## 0.16.0

**Hardening the compute channels**
- Providers admit a product request from its headers (shape, channel, voucher, credit, funds) before reading the body; at most 2 products compute at once; kept-alive connections idle for 30 s are closed; a request that breaks off is not billed; books are kept until a channel is past expiry by more than a reorg can undo.
- Cooperative close (consensus: optional `close` on `channel_claim`): the unspent part of a channel returns to the buyer at once. `ComputeChannel.close`, `ComputeCluster.close`, provider `POST /close`; a claim still waiting in the mempool is replaced by the closing one.
- `matmul_many`: independent products in parallel, when the measured time per product shows the link is slow enough to gain (on a fast local link threads cost more than they save); training sends each layer's two backward products together (221 instead of 270 ms a step over a 40 ms, 100 Mbit/s link). Channel opens and claims pay the node's fee estimate. `bench/compute_bench.py --rtt --mbit` emulates an internet link.

**Reachability**
- `--portmap`: PCP (RFC 6887) with a NAT-PMP (RFC 6886) fallback forwards the P2P port and the provider's port on the home router, renews and removes the mapping, and advertises the external address unless it is not global (carrier-grade NAT). `green/portmap.py`.

**Fees**
- The node pays the fee estimate for its own protocol transactions and raises the fee of any still waiting after 3 blocks (at most 0.1 GRN). `grn bump <txid>` for wallet transactions; a bump of a transaction on its Dandelion stem replaces it.

**Security**
- Gossiped addresses must be global IP addresses (no names, no private ranges) unless both the address and its source are local; offer URLs must name an IP address.
- Found on the way: Python's `is_private` does not cover 100.64.0.0/10. And the rare thread warning in test runs since v0.11, never pinned down before: a dial thread still in its handshake read the database after its node had closed it on shutdown. Handshakes now stop when the node is stopping, and connection threads log unexpected errors instead of dying.
- Fourteenth security review (SECURITY.md, F1-F8). 241 tests. Consensus change (`close` on claims): v0.15 nodes reject such claims.

## 0.15.0

**Instant compute: train a model by paying miners**
- Compute channels (consensus: `channel_open`, `channel_claim`, channels ending at expiry; SPEC.md §3.2, §3.3, §6.1). A buyer locks coins for a provider in one transaction, then buys matrix products straight from it, checks each with Freivalds' test and pays with signed vouchers for the running total; the provider claims on chain, with the fee taken from the claim (or splits it among payees); what is never claimed returns to the buyer at expiry.
- `green/compute.py`: the provider (`grn node --provide-port ...`, books kept on disk, automatic claims, limited credit), the buyer's `ComputeChannel` (kept-alive connection, every product checked), `ComputeCluster` (several providers, large products split by rows, a cheating provider dropped and its share recomputed), and `ComputeCluster.from_market` (find providers through the node and open channels).
- `green/train.py`: neural networks trained on int8 products (per-tensor scales, stochastic rounding for gradients) with float32 master weights at the buyer; it reaches the same accuracy as float32 training, and a model trained through a channel is bit-identical to one trained locally.
- Provider discovery: providers advertise signed offers over P2P (`offers`, kept only while the provider holds 0.1 GRN); `GET /v1/providers`, `grn providers`.
- `examples/train_on_green.py`: two miners sell compute on a local network and a buyer trains a digit classifier through them (300 steps in about 4 s, 93% test accuracy, the miners paid on chain). `bench/compute_bench.py`.
- Found on the way: Freivalds' check needed a variant for large products (the market's one is exact only up to about 1,000 output columns), and the first version was slower than computing the product (int64 products bypass the BLAS); `np.abs` of an int8 -128 is -128, which made the size bound too small.
- Thirteenth security review (SECURITY.md, CH1-CH7). 225 tests. Consensus change (new transaction types): v0.14 nodes reject them.

## 0.14.0

**Recovery phrases**
- `grn wallet new` makes a wallet from a 24-word BIP-39 phrase; `grn wallet restore` brings it back on any machine, without the file or its password; `grn wallet phrase` shows it again. Keys are derived with SLIP-0010 for Ed25519 along `m/44'/7777'/account'` (`--account`), with an optional passphrase. `green/hd.py`; the official word list ships in the package and is checked against its published SHA-256 when loaded. Passes the BIP-39 reference vectors and SLIP-0010 test vector 1. Words can be given by their first four letters; a wrong or swapped word is caught by the checksum and refused, never guessed. Wallet files from earlier versions still open.

**Transaction origin privacy (Dandelion++)**
- A node's own transactions pass along a stem of single hops (`dtx`) before anyone announces them, so the announcement does not point at the node they came from. Stem routes are fixed per epoch and per incoming link, stem transactions are never served to `getdata`, loops end the stem, and every stem node announces the transaction itself when its random embargo runs out (SPEC.md §9.3). `--no-dandelion` turns it off. The node's own commits, reveals and challenges, and fee bumps, are announced directly.

**Compact block relay**
- New blocks travel as a header plus 6-byte salted transaction ids (`cmpct`); the receiver rebuilds them from its mempool and asks only for what it lacks (`getblocktxn`/`blocktxn`); a short-id collision falls back to the whole block. A 20-transaction block crosses a link in under a quarter of its size, usually with no round trip. Peers that do not list the feature get an `inv` as before.

**Useful work through pools**
- A `job_commit` can name payees (SPEC.md §3.2): the batch's payment is split among them when it settles, and the bond stays the committer's. Pools now sell compute: with each template a miner gets units of open jobs (as far as the pool can bond them), computes them, and sends the results, which the pool checks with Freivalds' test before batching them. Each batch pays the miners who computed its units, plus the pool's fee, in consensus. Pool miners now do useful AI work instead of filler, and the pool still never holds what they earn.
- Pool fixes found on the way: stale shares (work on a template a new block replaced) counted as invalid, so an honest miner could be paused for ten minutes; miners now refresh templates every second and stale shares are answered as such. A batch flush popped entries while iterating, losing accepted results.

**Git and CI**
- The repository is under git (v0.13.0 is the first commit, tagged); the CI workflow runs once it is pushed.
- Twelfth security review (SECURITY.md, H1–H2, D1–D4, C1–C2, J1–J3). 214 tests. Consensus change (optional `payees` on `job_commit`): v0.13 nodes reject such commits.

## 0.13.0

**Non-custodial pooled mining**
- A block's `miner` can be a list of up to 32 payees with weights; consensus splits the reward and fees among them (SPEC.md §5.1, §3.3). A pool pays its miners in the block itself: it never holds their coins and sends no payout transactions.
- `green/pool.py`: a pool server (`grn node --pool-port 8547`) whose templates pay the miners of the last 1,000 shares by weight plus the operator's fee, recomputes the work of every share, refuses stale, duplicate and wrong shares, and pauses clients that send many bad ones; a pool miner (`grn pool-mine --pool <url>`, CPU or GPU).
- Found by its first test: the pool's status call deadlocked (it took its own lock twice). A pool miner now keeps retrying when the pool restarts or is unreachable instead of stopping.
- The node's miner and task worker now restart after an unexpected error instead of their thread dying silently (traced from a rare test warning: a thread that died with an exception); the task worker keeps what it committed across a restart, so it still reveals and keeps its bond.
- Eleventh security review (SECURITY.md, K1–K3). 192 tests. Consensus change (the header's `miner` field): v0.12 networks are not compatible.

## 0.12.0

**Light wallets and explorers**
- A transaction and address index, written in the same database transaction as each block and rewritten from the fork point on a reorg, so it always matches the main chain: `GET /v1/tx/<txid>`, `GET /v1/address/<address>/history` (paged, with mining rewards and pending transactions), `grn tx`, `grn history`.
- Inclusion proofs (`GET /v1/proof/tx/<txid>`) and headers (`GET /v1/headers`).
- `green.light.LightClient`: Bitcoin-style SPV for Green. It downloads headers, checks every header rule and recomputes the work of a random sample plus the newest header, and verifies a payment by its Merkle proof against a header with enough confirmations. The node is trusted only for availability; wrong headers, forged work and wrong proofs are detected.
- A data folder made by an earlier version gets its index built from the stored blocks when it is opened.
- Found while checking why the suite had slowed down: a storage test called `monkeypatch.undo()`, which also undid the test fixtures' easy target, so the rest of it ground ~67 work units per block (104 s); it now restores only what it patched (4 s). The same mistake was caught in a new light-client test.
- 185 tests. Not a consensus change.

## 0.11.0

**Pruned nodes**
- `--prune` (or `prune = true`): keep every header but block bodies only for the last `--prune-keep` blocks (default 5,760, about a day; at least the reorg window); keep the ledger and its snapshots, so a pruned node still serves fast sync.
- Unused data blobs (matrices of finished jobs, results of finished jobs, programs of settled tasks, ...) are collected in two phases: marked when nothing references them, deleted a retention period later if still unused.
- New databases give freed space back to the disk (SQLite incremental auto-vacuum). Found while testing: the vacuum pragma frees one page per result row, so without reading its rows it freed almost nothing (271 → 270 pages; with the fix 271 → 171).
- Peers learn where a node's bodies start from its handshake, and from its "not found" answers if it pruned after connecting.
- 180 tests. Not a consensus change.

## 0.10.0

**For developers**
- A stable, versioned API under `/v1` (the unprefixed paths still work).
- Fee estimates (`GET /v1/fee`, `grn fee`): where a transaction would land in the mempool's fee order and what recent full blocks needed; `grn send` and the other commands now pay the estimate instead of always the minimum.
- Long-polling for blocks (`GET /v1/wait`, `green.node.wait_for_block`): the agent and matmul SDKs now wake when a block arrives instead of polling on a timer. At most 16 clients wait at once.

**For operators**
- A config file, `<data>/green.toml` (`[node]` table); the command line overrides it, and an unknown key or a wrong type stops the node with a message.
- Log levels and a rotated log file (`--log-level`, `--log-file`); `debug` shows the P2P layer's details.
- The wallet password can come from a file (`GREEN_WALLET_PASSWORD_FILE`), so a service manager can hand it over without putting it in the environment.
- [contrib/green.service](contrib/green.service) (a hardened systemd unit using a systemd credential for the password), [contrib/prometheus-alerts.yml](contrib/prometheus-alerts.yml), new metrics `green_supply_ok` and `green_warnings`, and [OPERATIONS.md](OPERATIONS.md): install, settings, firewall, service, monitoring, backup and restore, upgrades.
- 177 tests. Not a consensus change: v0.9 and v0.10 nodes form one network.

## 0.9.0

**Multisignature accounts**
- A sender can be a k-of-n account (2 ≤ n ≤ 16): its `pub` is `{"k", "keys"}` with sorted distinct keys, its `sig` one slot per key with exactly k signatures. Its address is the hash of k and the keys, so anyone can pay it like any other address.
- Cosigners sign one at a time: `grn multisig address|new|sign|send` passes one transaction file around; `grn wallet show` prints the public key to share. The SDK has `make_multisig_tx` and `add_signature`.
- A guardian (v0.6) can be a multisig account, so freezing or loosening an agent needs its quorum.
- A block carries at most 2,000 signatures in total, which bounds signature checking to about 0.1 s per block whatever the transactions.
- The mempool groups and evicts by sender address instead of public key, so multisig senders get the same per-sender limits and replace-by-fee.
- Ninth security review (SECURITY.md, M1–M3). 170 tests. Consensus change (transaction format): v0.8 networks are not compatible.

## 0.8.0

**Changing the rules without splitting the network**
- Version-bits deployments (green/versionbits.py, SPEC.md §5.5): miners signal a rule change in the header version; 90% of a 20,160-block period locks it in, and it activates one period later (optionally not before a minimum height); it fails at its timeout, or locks in there if scheduled as mandatory. Header versions are now `0x20000000` plus signal bits.
- Miners signal every scheduled deployment automatically (`--no-signal <name>` opts out); `grn deployments` and `GET /deployments` show the states; a node warns when most recent blocks signal a change it does not know.
- `testdummy`, an example deployment for test networks, shows the pattern: it tightens the header message limit once active, and nodes that never heard of it keep following the chain.

**Signed releases**
- `grn release sums|sign|verify` (green/release.py): maintainers sign a `SHA256SUMS` file; a release is genuine only with enough distinct maintainer signatures and every file unchanged.
- [UPGRADES.md](UPGRADES.md): how soft forks, hard forks, emergency fixes and releases are handled.

**Also**
- Two tests that had become timing-dependent were fixed after tracing them, and each was checked to still catch the bug it guards (by reintroducing that bug): the WASM worker test now leaves the reveal 4 blocks of slack instead of 1 (faster verification since v0.7 made its test blocks arrive up to twice as fast), the eviction test waits for the peer list, which settles a moment after a socket closes, and the relay test keeps its line of three nodes a line (the far node no longer listens, so the first cannot dial it directly).
- Eighth security review (SECURITY.md, U1–U3). 165 tests.
- This is a consensus change (the header version rule): networks started with v0.7 are not compatible.

## 0.7.0

**Every header commits to the ledger**
- New header field `state_root`: a lattice hash (LtHash, 1,024 16-bit lanes) of every ledger entry, updated per block in O(entries changed) and checked by every validator. Nodes that disagree on any balance now fail at that block instead of forking later. The exact state layout is part of SPEC.md (§6.1); writing it down found that a zero credit left an empty balance entry, so balances now hold non-zero amounts only.
- Ledger work per block: 1.1–1.5 ms (was 0.5–0.8 ms), still independent of the number of accounts (`bench/state_bench.py`).

**An encrypted peer-to-peer network** (`green/p2p.py`, SPEC.md §9)
- Replaces the plain-HTTP gossip. Per connection: X25519 key exchange, HKDF keys salted with the genesis hash, ChaCha20-Poly1305 on every frame including its length, rekeying every 4,096 frames. Each node has an Ed25519 identity (`node_key`) that signs the session; `--peer <id>@host:port` pins it.
- Announce-then-fetch relay (`inv`/`getdata`): every transaction and block crosses each link once. Transaction announcements leave at random intervals, which hides their origin.
- Address book with per-source limits, outbound peers spread over network groups (two of them block-relay-only), inbound limits per group with eviction, rate-limited address gossip, misbehaviour bans that survive restarts.
- DoS limits: frame size checked before reading (large frames only for requested data), read deadlines, handshake timeout, per-peer message and send queues, blob upload budgets.
- New options: `--p2p-port` (default 8546), `--p2p-host`, `--no-listen`; `grn peers` shows ids, ping and session ids.

**Sync**
- Headers first, with a random sample of work units recomputed per message, then blocks from several peers at once in a 256-block window, their work units checked in parallel threads before they are connected in order.
- Fast sync (`--fast-sync`): nodes write the ledger every 2,880 blocks as content-addressed chunks; a new node takes the newest final snapshot, checks it against the snapshot block's `state_root`, and validates every block after it. A lying snapshot server is banned and the node falls back to a full sync. Old job matrices are no longer needed to join the network.
- `bench/sync_bench.py` compares full and fast sync: 600 blocks in 4.2 s (full) and 1.4 s (fast) on the development machine.

**Faster work verification**
- The transcript digest is computed in cache-sized chunks with in-place operations: 3.6× faster, bit for bit the same (checked against the definition and the GPU). Verifying a work unit takes 9 ms on one core instead of 17 ms, and about 4 ms per unit with 8 threads. A float32 variant was measured too and rejected: exact today, but with less than a 2× margin, which is too thin for consensus.

**Also**
- Decoded job matrices are cached with a bound (8 jobs), not forever.
- Headers are stored apart from block bodies, so serving 2,000 headers does not read 2,000 blocks.
- Removed: the HTTP endpoints `POST /block` and `POST /announce`, the `X-Green-From` header, and the one-time import of v0.4 block folders (v0.7 headers differ, so old data cannot be imported).
- Sync pipeline: each downloaded block's work unit goes to the verification threads on arrival and a separate thread connects blocks in order. Profiling found the first version never checked in parallel (blocks from one peer were connected one by one inside the message handler); full sync of 600 blocks went from 14.6 s to 4.2 s.
- Inbound eviction only for authenticated newcomers, a cap on unfinished handshakes, penalties for peers that stall block requests, pruning of stale fork headers.
- `bench/net_sim.py`: eight nodes, a split and a heal; blocks reach every node in a median of 0.03 s.
- Seventh security review (SECURITY.md, P1–P10). 154 tests.

## 0.6.0

**Difficulty: per-block ASERT**
- Replaces the 30-block windowed retarget with aserti3-2d (half-life 30 minutes, anchored at genesis). Simulated with miners that come and go: the old rule averaged 20.6 s blocks and fell 9.3 hours behind the issuance schedule in 6,000 blocks; ASERT holds 15.0 s with zero drift, and recovers from a 100× hashrate drop in 4.0 h instead of 8.8 h. Trade-off: about an hour to absorb a sudden 10× surge (`bench/difficulty_sim.py`, SECURITY.md Z1).

**Guarded accounts: an off switch for autonomous agents**
- New transactions `guard_set`, `guard_update`, `guard_release`: a guardian caps an account's spending per period and can freeze it; the account can only tighten its own guard. The agent SDK waits when its allowance is used up and stops when frozen. CLI: `grn guard set|update|release|show`.

**Operations**
- Supply audit (`grn supply`, `GET /supply`): balances + escrow + bonds must equal the sum of block rewards, exactly.
- Prometheus metrics (`GET /metrics`).
- Docker image, built and smoke-tested in CI.
- [API.md](API.md) documents every endpoint; [PRODUCTION.md](PRODUCTION.md) states the status of every production area, with evidence and blockers.

**Found while changing it**
- A live-node test passed only because the old difficulty rule happened to slow its instant blocks down within seconds; it now sets the conditions it needs explicitly and checks that ASERT does the slowing.
- 117 tests.

## 0.5.0

**State engine: the ledger no longer costs O(accounts) per block**
- Each block's state is a thin diff layer over its parent's (only the keys it changed); layers that leave the reorg window are merged into one flat base, like Ethereum's snapshot layers. v0.4 copied the whole ledger for every block and kept 120 copies.
- Measured with 1,000,000 accounts (`bench/state_bench.py`): memory held by 20 blocks 587 MB → 0.1 MB; ledger work per block 0.5–0.8 ms at any ledger size.
- A state that has children is frozen: any attempt to modify it raises, instead of silently rewriting history.
- The main chain is updated at its changed end only; v0.4 walked back to genesis on every new tip.

**Storage: SQLite, atomic, restart without re-executing**
- Blocks, the main-chain index, recent state diffs and the base state live in `chain.db` (standard-library SQLite, WAL). Every block is written in one transaction, so a crash leaves the last complete block.
- Restart loads the base and recent diffs: 0.04 s at any height. v0.4 re-executed every block since genesis (7.8 ms per block of history, plus every WASM adjudication again), which after 4 years of blocks would take about 18 hours.
- Old blocks leave memory and are served from the store. A v0.4 data folder (one JSON file per block) is imported once.

**Consensus fix**
- The reorg-depth rule now measures where a fork leaves the main chain. v0.4 checked only the new block's parent, so a side chain forked long ago could keep growing next to the main chain (SECURITY.md, "Fifth review", Y1). Confirmed: the new test fails on v0.4.

**Proven, not assumed**
- The mempool now applies transactions in place (constant depth instead of one layer per pending transaction). That relies on every transaction failing atomically, which is now tested for all ten types, including late failures, and checked on every rejected transaction in the fuzz test.
- Differential test: the layered chain, with reorgs and dozens of base moves, matches a full-copy replay of the same blocks.
- 107 tests.

## 0.4.0

**Delivery: whole jobs with two transactions**
- A batch is a bitmap over the job's units (up to all 4,096) plus the Merkle root of their result hashes. Validators see only the root; the list of hashes (the manifest) travels from the miner to the buyer like the results.
- On-chain data per delivered unit: 0.49 B (v0.3: 44 B, v0.2: 262,144 B). Delivery ceiling: 68,267 units/s (v0.3: 4,267; v0.2: 16.7).
- `job_respond` carries the unit's result hash and a Merkle proof (at most 12 hashes).

**Security fixes in the dispute game** (details in SECURITY.md, "Fourth review")
- X1, partial withholding: v0.3 allowed one open dispute per batch, so a miner could withhold most of a batch, publish only the disputed unit and still be paid in full. Any number of units can now be disputed at once, and every one must be answered.
- X2, response-capacity griefing: a flood of disputes could fill the per-block response budget so honest miners missed their deadlines and lost their bonds. Deadlines now stretch with the backlog of open disputes.
- X3, withholding was profitable: the challenger's bond went to a miner that answered, so withholding data until disputed earned money. Disputes now cost only a fee, and answering earns nothing.
- X4, re-disputing a proven unit: a proven unit can no longer be disputed again.

**Addresses**
- Bech32m (BIP-350) addresses, `grn1...`: every typo of up to 4 characters is detected, so a mistyped address is refused instead of burning coins. The old unchecked `g...` hex format is gone.

**Fee market and mempool**
- Blocks are filled by fee per byte, keeping each sender's nonce order.
- A full mempool evicts the lowest fee-rate transaction instead of refusing everything new.
- Replace-by-fee: the same sender and nonce, paying at least 10% (and at least the minimum fee) more.
- At most 100 pending transactions per sender.
- Fixed: the mempool was re-sorted by nonce after every block, which dropped a new account's spend when it came before the payment funding it.

**Product**
- Installable package (`pip install .`, command `grn`), MIT license, CI on Linux/Windows/macOS, LF line endings enforced.
- 93 tests.

## 0.3.0

- Freivalds over the prime field Z_p (p = 2^31 − 1): 0.37 ms per unit instead of 6.45 ms, error 2^-124 instead of 2^-64, with a range check against modular wraparound.
- Optimistic, batched delivery: up to 256 units per commit/reveal; the buyer verifies, validators only adjudicate disputes.
- 1 MB block size limit.
- Buyer-side watchtower in every node; the SDK verifies every unit before using it.
- The whole project translated to English.

## 0.2.0

- WASM compute market with on-chain adjudication, watchtowers and per-block adjudication budget.
- GPU miner (WebGPU/Vulkan), quantized neural-network SDK, autonomous agent.
- Security reviews S1–S19 and W1–W9.
