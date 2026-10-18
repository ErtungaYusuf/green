# Green, Bitcoin and Solana

What each one solves, how they compare in numbers, and where Green still falls short. Sources: the Bitcoin whitepaper (Nakamoto, 2008) and Bitcoin Core's consensus rules; the Solana whitepaper (v0.8.13) and the Solana documentation on fees and transactions; Green's own numbers come from this repository's benches, run on an i5-14600KF with an RX 9070 XT.

## The problem each one solves

**Bitcoin** removes the trusted third party from online payments. Transactions are put in order by a timestamp server made of proof of work: each block commits to the one before, and the chain with the most work is the history. An attacker with less than half the work falls behind exponentially; the whitepaper's §11 table says how many blocks a merchant should wait (z = 5 against an attacker with 10% of the work, z = 24 against 30%, for a success chance below 0.1%). The work itself is SHA-256 hashing that is useful for nothing else.

**Solana** asks how fast a replicated ledger can go. Proof of History is a sequential SHA-256 chain that works as a clock everyone can check, so validators agree on the order of events without exchanging messages about time; proof of stake votes on top of it (Tower BFT). The paper projects 710,000 transactions per second on a 1 Gbps network, with signatures checked on GPUs, and suggests that contracts call batched GPU "intrinsics". Every validator still executes every transaction: computing power goes into replicating the same execution, not into work someone else needs.

**Green** asks that the work securing the ledger is the compute people need. Mining is integer matrix multiplication, the operation at the core of AI training and inference. A block's ticket comes out of a work unit, and the same unit can be a buyer's result. Anyone can buy that compute with the coin: whole jobs on chain, products one by one through payment channels, or any program through the WASM market. Results are checked in O(n²) with Freivalds' test, not recomputed in O(n³). Compute is done once by whoever sells it, not repeated by every node. This is **proof of given compute**: the network is paid in coin for compute it gives away, and the coin is secured by that same compute.

## In numbers

| | Bitcoin | Solana | Green (v0.22) |
|---|---|---|---|
| Sybil resistance | proof of work (SHA-256) | proof of stake + Proof of History | proof of useful work (int8 matmul, GPU or CPU) |
| What the work produces | nothing else | a clock (PoH), replicated execution | matrix products buyers paid for |
| Block / slot time | 600 s | 400 ms slots | 15 s |
| Difficulty | every 2016 blocks, x4 clamp | n/a (leader schedule) | every block, ASERT (half-life 30 min) |
| Confirmation | probabilistic; 6 blocks ≈ 60 min | optimistic in under a second, finalized after 32 slots (≈ 13 s) | probabilistic; 6 blocks ≈ 90 s (the §11 math is per block, so 40x faster than Bitcoin at the same depth) |
| Throughput on chain | ~7 transfers/s | thousands of transactions/s measured (60M compute units per block) | **200 transfers/s** limit (3,000 per block); one core checks ~8,000/s cold, ~20,000/s with signatures cached |
| Off chain | Lightning | none needed at its speed | payment channels for compute (each result checked, paid per product) |
| Transaction size | ~140–250 B (P2WPKH) | ≤ 1,232 B | ~320 B transfer |
| Minimum fee | market, ~1 sat/vB floor for relay | 5,000 lamports per signature (0.000005 SOL) + priority fee | **500 yaprak per signature (0.000005 GRN)** + priority by fee rate |
| Where fees go | miner | half of the base fee burned, rest + priority to the leader | half back into the reserve (paid out again as rewards), half to the miner |
| Account spam | UTXO dust limits | rent-exempt minimum balance | a transfer that opens an account sends ≥ 0.0001 GRN |
| Supply | 21M, halving every 210,000 blocks | inflationary (decreasing yearly rate) | 21M, fixed; every block pays 1/2^24 of the reserve, no halving cliff |
| Signatures | ECDSA/Schnorr secp256k1 | Ed25519 | Ed25519, k-of-n multisig, guardians |
| Light clients | SPV (headers + Merkle proofs) | RPC trust or light-client research | headers + Merkle proofs, and a state root in every header for verified snapshots |
| Programs | Script (not Turing complete) | eBPF/SBF programs, re-executed by all | WASM tasks run once off chain, settled optimistically with fraud proofs; matmul jobs checked by Freivalds |
| Privacy | pseudonymous; new key per payment | pseudonymous | pseudonymous; Dandelion++ relay; compute requests sealed, relayed, split or secret-shared across providers |
| Validator hardware | a laptop | 12+ cores, 256 GB+ RAM, NVMe, 1 Gbps+ | a desktop; a GPU to mine competitively |
| Maturity | since 2009, many implementations | since 2020, several multi-hour halts (2021–2024) | **prototype**: Python reference implementation, private, not audited |

## Transactions on Green

Yes, GRN moves between people like Bitcoin or SOL. A `transfer` carries sender key, nonce, fee, recipient and amount. It is signed with Ed25519, relayed through Dandelion++, and included in the next block when its fee rate makes the cut. The wallet ([green-wallet](https://github.com/ErtungaYusuf/green-wallet)), the command line (`grn send`) and the local network's app all send them. On the local network a 0.5 GRN payment confirmed in the next block. The minimum fee for a plain transfer is 0.000005 GRN; a k-of-n multisig transfer pays it k times; when blocks fill, the fee estimate (`GET /v1/fee`) says what gets in.

## Where Green has to catch up

Against **Bitcoin**: fifteen years of adversarial testing, a hash rate no newcomer can attack, independent implementations, hardware wallets and an ecosystem. Green is unaudited Python. Its security margin is only as large as the GPUs mining it, and on a small network a rented GPU farm could outmine it (see PRODUCTION.md).

Against **Solana**: speed of confirmation (15 s blocks and probabilistic finality against sub-second optimistic confirmation) and raw throughput (200/s against thousands). General on-chain programs are also missing: WASM tasks are off-chain compute, not contracts that change the ledger.

What closes the gap, in order:

1. **A native node** (Rust or Go). The Python node spends most of a block on canonical JSON and hashing the state. Ten times the throughput limit is in reach once checking is native and parallel, still inside 1 MB blocks (already the storage bound).
2. **Faster confirmation from shares.** About 32 shares are mined per block (one every ~0.5 s), and each commits to its parent. A wallet can show a payment as "seen by miners" in seconds, before the block (weak, FruitChains-style confirmations).
3. **A public testnet, an audit and a bug bounty**, then main-net.
4. **More compute for sale:** whole-model training jobs, conv and transformer layers, GPU providers through channels.
5. **Private payments** (a shielded pool), so paying for compute does not link buyer and provider on chain.

## How the numbers were measured

- Throughput: `python bench/tps_bench.py 4000` builds one block of 4,000 transfers from 4,000 accounts. Signatures: ~13,500/s on one core (74 µs each). Whole block: ~8,000 transfers/s when each signature is checked, ~20,000/s when the transactions were already checked on arrival in the mempool (signature cache). A full block of 3,000 transfers takes ~0.4 s cold and ~0.15 s warm.
- Confirmation depth: the whitepaper's attacker formula depends on the attacker's share of the work, not on the block time, so 6 Green blocks give the same protection as 6 Bitcoin blocks against an attacker with the same share. Faster blocks lose more work to stale blocks; Green's 15 s with compact blocks keeps that small on a local or regional network.
- Bitcoin and Solana figures are their published parameters (block time, retarget rule, fee per signature, transaction size limit, compute units per block), not new measurements.
