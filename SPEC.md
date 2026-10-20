# Green protocol specification (v0.17)

This document defines the consensus rules of the Green network precisely enough to write an independent, compatible node. Where it and the reference implementation (`green/`) disagree, that is a bug in one of them. Constants are named as in `green/params.py`; their values are listed in §10.

Notation: `H(x)` is BLAKE2b with a 32-byte digest. `hex(x)` is lowercase hexadecimal. `‖` is byte concatenation. Integers in hashes are big-endian unless stated otherwise. "yaprak" is the base unit: 1 GRN = 10^8 yaprak.

## 1. Encodings

- **Canonical JSON** `C(obj)`: JSON with keys sorted, separators `,` and `:` (no spaces), ASCII only (`\uXXXX` escapes). Every hash over a structured object is `H(C(obj))`.
- **Hashes** in objects are 64-character lowercase hex strings.
- **Integers** in objects are JSON integers in `[0, 2^63)`; booleans are not integers.

## 2. Keys and addresses

- Signatures are Ed25519. A public key is 32 raw bytes, written as 64 hex characters.
- An address is `bech32m("grn", H(pubkey)[0:20])` (BIP-350): `grn1` followed by 38 characters. Only the lowercase form is a valid address, so every account has exactly one spelling.
- A **multisig account** is `{"k": k, "keys": [pubkey, ...]}` with 2 ≤ n ≤ MAX_MULTISIG_KEYS distinct keys sorted ascending and 1 ≤ k ≤ n. Its address is `bech32m("grn", H(C(["grn-multisig", k, keys]))[0:20])`; it looks like any other address.

## 3. Transactions

A transaction is a JSON object with exactly the common fields `type`, `pub`, `nonce`, `fee`, `sig` plus the fields of its type (§3.2), and optionally the optional fields of its type; any other field set is invalid.

- `txid = hex(H(C(tx)))`.
- Signing: `sig = Ed25519(secret, C({"chain": chain_id, "tx": tx without "sig"}))`, 128 hex characters. `chain_id` is the genesis block hash, so a transaction is valid on one network only.
- For a multisig sender, `pub` is the account object (§2) and `sig` a list with one entry per key, in the same order: exactly `k` entries are signatures, each by the key in its position over the bytes above, and the others are `null`.
- The sender is `address(pub)` (a single key's or a multisig account's address). `nonce` must equal the sender's account nonce; applying the transaction increments it.
- `fee >= MIN_TX_FEE · signatures` (1 for a single-key transaction, `k` for a k-of-n one, plus 1 for a channel claim's voucher), plus the data fees listed per type. The fee is paid to the block's miner.
- A transaction is applied atomically: if any rule fails, the state is unchanged and the transaction, and therefore the block containing it, is invalid.
- **Guarded accounts.** If the sender has a guard (§3.4), before any other effect: a frozen sender's transaction is invalid; otherwise, with `w = floor(height / period)`, `spent' = (spent if window == w else 0) + outflow(tx)` must be `<= limit`, and on success the guard's `window := w`, `spent := spent'` (a `guard_set` instead writes its own new guard). `outflow(tx)` = fee + `amount` (transfer and channel_open) + `price · mu·ku·nu` (job_post) + `price` (task_post); a `job_commit`, `task_commit` or `task_challenge` also adds its bond to `spent'`, which must still be `<= limit` (since v0.24.1: a bond the sender does not redeem goes to someone else).

### 3.1 Blobs

Large data travels between nodes as content-addressed blobs (`hash = hex(H(bytes))`), never inside blocks.

| Blob | Format |
|---|---|
| Job | `"GJOB"` ‖ M ‖ K ‖ N (uint32 BE each, multiples of UNIT) ‖ A (int8, M×K row-major) ‖ B (int8, K×N row-major) |
| Result | one unit's partial product `A_il · B_lj`: int32 little-endian, UNIT×UNIT row-major |
| Manifest | the 32-byte result hashes of a batch, concatenated in ascending unit index order |
| WASM program / input | raw bytes |

A transaction that needs a blob for validation (§3.2) is only valid if the validator can obtain that blob; validators fetch it from peers.

### 3.2 Transaction types

Units of a job with dimensions `(mu, ku, nu)` (in units) are triples `[i, l, j]`. Unit index `k = (i·ku + l)·nu + j`. A **unit bitmap** is the hex encoding of exactly `ceil(mu·ku·nu / 8)` bytes in which unit k is bit `7 − k mod 8` of byte `k div 8`, with no bits set past the last unit and at least one bit set.

| Type | Fields | Rules (in addition to §3) |
|---|---|---|
| `transfer` | `to`, `amount` | `to` is an address, `amount >= 1`, balance ≥ amount + fee; if `to` has no balance and nonce 0, `amount >= NEW_ACCOUNT_MIN` |
| `job_post` | `job`, `dims`, `price` | `job` is a job blob hash that is not yet a job; `dims` 3 integers in `1..MAX_JOB_UNITS_PER_DIM` matching the blob; `price >= 1` per unit; locks `escrow = price · mu·ku·nu`; fee ≥ MIN_TX_FEE + DATA_FEE_PER_BYTE · blob size |
| `job_commit` | `job`, `units`, `commitment`; optional `payees` | job open and not cancelled; `units` a bitmap of units that are neither delivered nor reserved; `commitment` not already a batch id of this job (live or delivered); `payees`, if present, is `[[address, weight], ...]` in the form of a header's payee list (§5.1); locks `bond = bond_for(price) · count`; creates batch `id = commitment`, status *committed*, `expires = height + RESERVE_BLOCKS`, recording `payees` if present, and reserves its units |
| `job_reveal` | `job`, `batch`, `root`, `manifest` | sender is the batch's miner; status *committed*; `batch.height < height <= expires`; `commitment == hex(H(C(["green-commit", sender, job, units, root])))`. Sets status *revealed*, `settle_at = height + JOB_CHALLENGE_BLOCKS`. `manifest` is an unverified hint |
| `job_challenge` | `job`, `batch`, `units` | batch *revealed*; `height <= settle_at`; `units` a bitmap of the same length that is a subset of the batch's; none of them disputed or proven; fee ≥ MIN_TX_FEE + DATA_FEE_PER_BYTE · RESULT_BYTES · count. Opens a dispute per unit with `deadline = height + JOB_RESPONSE_BLOCKS + floor(open_disputes' / MAX_BLOCK_JOB_RESPONSES)`, where `open_disputes'` is the network-wide count after adding these |
| `job_respond` | `job`, `batch`, `unit`, `result`, `proof` | the unit has an open dispute; `height <= deadline`; `merkle_verify(result, index of unit in the batch, count, proof, root)` (§4); fewer than MAX_BLOCK_JOB_RESPONSES responses earlier in this block; the result blob is available and passes Freivalds (§5.4). Closes the dispute and marks the unit proven; if no dispute is left and `height >= settle_at`, the batch is paid |
| `job_cancel` | `job` | sender is the buyer; not already cancelled; the job closes at `height + CANCEL_DELAY` |
| `task_post` | `prog`, `input`, `fuel`, `price` | program and input blobs available; program ≤ MAX_TASK_PROGRAM_BYTES, input ≤ MAX_TASK_INPUT_BYTES; program is a valid module with no imports exporting function `green_run` and memory `memory`; `1 <= fuel <= MAX_TASK_FUEL`; locks `price`; fee ≥ MIN_TX_FEE + DATA_FEE_PER_BYTE · (program + input bytes). Task id `= hex(H(C(["green-task", pub, nonce, prog, input, fuel, price])))` |
| `task_commit` | `task`, `commitment` | task open and not reserved (or reservation expired, whose bond then goes to the buyer); locks `task_bond(price)`; reserves for TASK_RESERVE_BLOCKS |
| `task_reveal` | `task`, `output` | sender holds the reservation; `reservation.height < height <= expires`; `commitment == hex(H(C(["green-task-commit", sender, task, output])))`; opens a window until `height + TASK_CHALLENGE_BLOCKS` |
| `task_challenge` | `task` | status *revealed* (not *confirmed*); within the window; fee ≥ `challenge_fee(fuel, program bytes)`; locks `task_bond(price)`; adjudication budget of the block not exceeded. Every validator re-runs the task (§7): wrong output → worker's bond and challenger's bond to the challenger, escrow to the buyer, status *failed*; right output → challenger's bond to the worker, status *confirmed* |
| `guard_set` | `guardian`, `limit`, `period` | `guardian` an address other than the sender; `limit >= 0`; `1 <= period <= MAX_GUARD_PERIOD`. Unguarded sender: creates `{guardian, limit, period, frozen: false, window: floor(height/period), spent: 0}`. Guarded sender: allowed only with the same guardian, `limit <= old limit` and `period >= old period` (stricter); keeps `spent'` and sets `window := floor(height/period)` |
| `guard_update` | `account`, `guardian`, `limit`, `period`, `frozen` | sender is `account`'s guardian; new `guardian` is not `account`; replaces guardian, limit, period and frozen (window and spent are kept) |
| `guard_release` | `account` | sender is `account`'s guardian; removes the guard |
| `channel_open` | `provider`, `amount`, `expires` | the sender is a single key (not multisig) and not `provider`; `amount >= 1`; `height + MIN_CHANNEL_BLOCKS <= expires <= height + MAX_CHANNEL_BLOCKS`; balance ≥ amount + fee; locks `amount` in channel `id = txid` `{buyer, pub (the sender's key), provider, amount, paid: 0, expires}`, ending at `expires + 1` |
| `channel_claim` | `channel`, `total`, `voucher`; optional `payees`, `close` | sender is the channel's provider; `height <= expires`; `paid < total <= amount`; `voucher = Ed25519(buyer's key, C(["grn-voucher", chain_id, channel, total]))`; `total − paid >= fee`. Sets `paid := total` and credits `total − paid − fee` to the provider, or splits it among `payees` (a payee list as in §5.1). The fee comes out of the claim. With `close` (which, if present, is `true`), the buyer also receives `amount − total` and the channel is removed at once. Counts as two signatures toward MAX_BLOCK_SIGS |

### 3.3 End-of-block events

After a block's transactions, at height `h`, in this order:

1. **Job events** scheduled for `h`, in scheduling order:
   - *expire* (scheduled at `expires + 1`): a batch still *committed* is dropped; its bond goes to the buyer.
   - *settle* (scheduled at `settle_at`): a *revealed* batch with no open dispute is paid: the escrow decreases by `price · count`, which goes to the miner or, if the batch has payees, is split among them as in §3.5; the miner gets its bond back; its units become delivered.
   - *deadline* (scheduled at `deadline + 1`): if a dispute of the batch with that deadline is still open, the batch is forfeit: its bond goes to the challenger of the earliest-opened such dispute and the batch is dropped.
   A dropped batch releases its units (they reopen) and its open disputes; if the job is closed, `price · count` of escrow is refunded to the buyer.
2. **Job closings** scheduled for `h`: refund `escrow − price · (units in live batches)` to the buyer and mark the job closed. Live batches settle later as usual.
3. **Task settlements** scheduled for `h`: a *revealed* or *confirmed* task pays the worker `price + bond`.
4. **Channel ends** scheduled for `h` (channels whose `expires` is `h − 1`), in opening order: the buyer receives `amount − paid` and the channel is removed.
5. **Shares**: the ids of shares included at `h − SHARE_WINDOW − 1` are forgotten, and those of this block's shares recorded at `h`.
6. **Rewards** (§3.5). Whenever an amount is paid to a `miner` field (a block's or a share's) or to payees, an address receives all of it, and a payee list `[[a_i, w_i]]` gives `a_i` `floor(total · w_i / Σw)`, the first payee also the remainder.

### 3.5 Money: the reserve, rewards and shares

The total supply is fixed at TOTAL_SUPPLY. Before the genesis block all of it is in the protocol's **reserve** (state map `protocol`, key `"reserve"`); no coin is ever created, and a part of every fee is destroyed (recorded under key `"burned"`), so `Σ balances + escrows + bonds + channel funds + reserve = TOTAL_SUPPLY − burned` at every block. There is no halving and no inflation; the supply shrinks slowly with use.

A block whose parent state holds reserve `R` has emission `E = floor(R / EMISSION_DIVISOR)`, of which `per_share = floor(floor(E · SHARES_PERCENT / 100) / SHARE_FACTOR)`, `cut = floor(per_share · SHARE_INCLUSION_PERCENT / 100)` and `finder = E − SHARE_FACTOR · per_share`. With `F` the block's fees and `n` its shares:
- each share's `miner` receives `per_share − cut`;
- the block's `miner` receives `finder + n · cut + F − floor(F · FEE_RESERVE_PERCENT / 100) − floor(F · FEE_BURN_PERCENT / 100)`;
- the reserve changes by `floor(F · FEE_RESERVE_PERCENT / 100) − finder − n · per_share`;
- `burned` grows by `floor(F · FEE_BURN_PERCENT / 100)` (the entry is absent while it is 0).

The finder's part does not depend on `n`: a block that leaves shares out keeps nothing more (what the missing shares would have been paid stays in the reserve), and one that includes them earns their cut.

A **share** is a header (§5.1, all fields, including `transcript`) whose ticket meets `min(MAX_TARGET, target · SHARE_FACTOR)`. A block at height `H` on parent `P` may include up to MAX_BLOCK_SHARES distinct shares, listed in the body as `shares` (present only when not empty) and committed to by `shares_root = merkle_root(share ids)`, where a share's id is `hex(H(C(share)))`. Each share must satisfy: `H − SHARE_WINDOW <= share.height <= H`; `share.prev` is the block at height `share.height − 1` of the branch ending at `P`; the share is not itself the block at its height on that branch; `target` is the target its `prev` calls for (§5.3); its id was not recorded by an earlier block within the window (§3.3); a job named in its `work` exists in the parent state; and its work unit recomputes to its `transcript` and meets the share target. Its `time`, `tx_root`, `shares_root`, `state_root`, `msg` and `version` are not otherwise checked: a share is evidence of work done for a payout address, and the seed covers every field, so neither the address nor the work can be changed. A block of a losing branch is a valid share.

### 3.4 Guarded accounts

The state maps an address to at most one guard `{guardian, limit, period, frozen, window, spent}`. A guard lets a person keep an autonomous agent accountable: the agent cannot spend more than `limit` yaprak per `period` blocks (spending as defined by `outflow`, §3), cannot send anything while `frozen`, and can only make its own guard stricter (`guard_set`); loosening, freezing, handing over and removing are the guardian's (`guard_update`, `guard_release`). Incoming coins are never restricted.

## 4. Merkle trees

`merkle_root(leaves)` over hex leaves: level 0 is `H(0x00 ‖ leaf)`; each next level pairs nodes left to right as `H(0x01 ‖ left ‖ right)`, and an odd last node is carried up unchanged. With top node `t` (32 zero bytes for no leaves), the root is `H(0x02 ‖ uint64_BE(len(leaves)) ‖ t)`.

A proof for leaf `i` of `n` lists the sibling at each level where one exists (an odd last node has none). `merkle_verify` recomputes the root from the leaf, the index and the count, using exactly `len(proof)` siblings. The count fixes the tree's shape, so each (count, index) has one valid proof length.

The block's `tx_root` is `merkle_root(txids)`. A batch's `root` is `merkle_root(result hashes in ascending unit index order)`.

## 5. Proof of Useful Work

### 5.1 Header and hashes

A block is `{"header": header, "txs": [...]}`. The header has exactly these fields:

| Field | Value |
|---|---|
| `version` | `0x20000000` plus signal bits 0–28 (§5.5); the top three bits must be `001` |
| `height` | parent height + 1 (genesis 0) |
| `prev` | parent block hash (genesis: 64 zeros) |
| `time` | seconds; greater than the median of the previous MTP_WINDOW block times; nodes reject (temporarily) a time more than MAX_FUTURE_SECONDS ahead of their clock |
| `target` | 64 hex digits; must equal the expected target (§5.3) |
| `tx_root` | `merkle_root(txids)` |
| `shares_root` | `merkle_root(share ids)` of the shares in the body (§3.5) |
| `state_root` | the lattice root of the state after this block (§6.1) |
| `miner` | the address that receives reward and fees, or `[[address, weight], ...]`: 2 to MAX_PAYEES entries sorted by address, addresses distinct, weights in 1..MAX_PAYEE_WEIGHT (a pool paying its miners) |
| `work` | `{"job": null, "unit": null}` (filler) or `{"job": job id posted before this block, "unit": [i, l, j] inside it}` |
| `nonce` | integer |
| `msg` | string, at most 200 characters |
| `transcript` | 64 hex digits (§5.2) |

`block_hash = hex(H(C(header)))`. `seed = H(C(header without "transcript"))`, so the seed binds the parent, the transactions, the resulting state, the miner and the nonce.

### 5.2 The work unit

`N = UNIT`, `T = N / BLOCK`, `R = NOISE_RANK`. `prg(seed, label, n)` is the first n bytes of SHAKE-256(seed ‖ label).

1. Inputs: for filler work, `A = prg(seed, "fillA", N²)` and `B = prg(seed, "fillB", N²)` read as int8, row-major. For job work, `A` is block `(i, l)` and `B` block `(l, j)` of the job's matrices.
2. Noise: `t(label, r, c) = (prg(seed, label, r·c) mod 3) − 1` reshaped row-major. `U = t("U", N, R)`, `V = t("V", R, N)`, `U' = t("Uf", N, R)`, `V' = t("Vf", R, N)`. `A1 = A + U·V`, `B1 = B + U'·V'` (exact integers).
3. Products: for every `(i, l, j)` in `T³` in row-major order, the tile product `P = A1[i, l] · B1[l, j]` (BLOCK×BLOCK). Flatten all products in order `(i, l, j, r, c)` into a sequence of int32 values `v_0, v_1, ...`.
4. Digest (since v0.18): for each index `k`, with all arithmetic mod 2^32 (lowbias32 of the value XOR its position):
   ```
   y = v_k (as uint32) XOR (k * 0x9E3779B9)
   y ^= y >> 16;  y *= 0x7FEB352D;  y ^= y >> 15;  y *= 0x846CA68B;  y ^= y >> 16
   lane[k mod 8] ^= y
   ```
   `transcript = hex(lane_0 ‖ ... ‖ lane_7)`, each lane uint32 little-endian, all lanes starting at 0. XOR is order-independent, so an implementation may compute the products and fold them in any order (the reference computes, for each `l`, all `(i, j)` products at once as `A1[:, l] · B1[l, :]`, and the GPU in parallel). Test vector: for filler work with `seed = 0x07` repeated 32 times, `transcript = 22f044a7ddbf20f469bd9cdec7aad9723b48abc62bbbd2b7e58a6cafbfd87a2e`. Before v0.18 the mix was MurmurHash3-style (`x = rotl(v·0xCC9E2D51, 15)·0x1B873593; y = rotl((k·0x9E3779B9) XOR x, 13)·5 + 0xE6546B64`, then the fmix32 finalizer).
5. `ticket = int_BE(H(seed ‖ bytes(transcript)))`. The block is valid only if `transcript` matches the recomputation and `ticket <= target`.

### 5.3 Difficulty and fork choice

- `work(target) = floor(2^256 / (target + 1))`; a chain's chainwork is the sum over its blocks (genesis included).
- The target of every block is set by ASERT (aserti3-2d) from its parent `p` and the anchor, the genesis block `g`: `e = floor(((p.time − (g.time − T)) − T · (p.height + 1)) · 65536 / ASERT_HALFLIFE)` with `T = TARGET_BLOCK_SECONDS` (floor division, rounding toward minus infinity); `shifts = e >> 16` (arithmetic), `frac = e & 0xFFFF`; `factor = 65536 + ((195766423245049·frac + 971821376·frac² + 5127·frac³ + 2^47) >> 48)`; `t = g.target · factor`, shifted left by `shifts` if `shifts >= 0`, else right by `−shifts`; `target = clamp(t >> 16, 1, MAX_TARGET)`. Block 1 therefore has the genesis target.
- A node follows the valid chain with the most chainwork; on a tie the first one seen stays. It refuses a block whose fork point (its last ancestor on the node's main chain) is more than MAX_REORG_DEPTH blocks below the node's tip.

### 5.4 Freivalds check of a result

For unit inputs `a` (N×K), `b` (K×N) and claimed result `c` (N×N): let `bound = K · max|a| · max|b|`. Reject if any `|c_xy| > bound`. With `p = FREIVALDS_PRIME`, take `prg(H(C(["freivalds", job, unit, result_hash])), "freivalds", 8·N·FREIVALDS_ROUNDS)` as little-endian uint64 values, reduce them mod p, and arrange them as an N×FREIVALDS_ROUNDS matrix `r`. Accept iff `a·((b·r) mod p) ≡ c·r (mod p)`.

### 5.5 Rule changes: version bits

A deployment is `(name, bit, start, timeout, min_activation = 0, lock_in_on_timeout = false)` with heights that are multiples of SIGNAL_PERIOD; the active set is `DEPLOYMENTS` (empty: none is scheduled). A block signals a deployment if bit `bit` of its version is set. The state of a deployment for a block at height `h` is the state of period `k = floor(h / SIGNAL_PERIOD)` on that block's branch:

- period 0 is DEFINED;
- for `k >= 1`, with `s` the state of period `k−1`, `H = k · SIGNAL_PERIOD`, and `n` the number of blocks at heights `H − SIGNAL_PERIOD … H − 1` on the branch that signal the deployment:
  - DEFINED → STARTED if `H >= start`;
  - STARTED → LOCKED_IN if `n >= SIGNAL_THRESHOLD`; otherwise, if `H >= timeout`, LOCKED_IN when `lock_in_on_timeout`, else FAILED;
  - LOCKED_IN → ACTIVE if `H >= min_activation`;
  - ACTIVE and FAILED never change.

A rule added by a deployment applies to blocks whose state is ACTIVE and must only make fewer blocks valid. Unknown signal bits are valid. The example deployment `testdummy`, scheduled only on test networks, limits a header's `msg` to 100 characters while ACTIVE.

### 5.6 Checkpoints (optional, for a young network)

A network may name a checkpoint authority in its genesis message: the text `checkpoints <K> until <U>`, with `K` an Ed25519 public key (64 lowercase hex) and `U` a block height (the first match counts). Without it there are no checkpoints and §5.3 alone decides. With it:

- A **checkpoint** is `{"height": h, "block": hash, "sig"}` with `sig = Ed25519(K, C(["green-checkpoint", chain_id, h, hash]))`, `1 <= h <= U`. The **end** is `{"end": true, "sig"}` with `sig = Ed25519(K, C(["green-checkpoint-end", chain_id]))`; after it no checkpoint is taken, and the last one taken stays.
- A node keeps the checkpoint with the greatest height it has taken (and the end, if any) and passes new ones on. A checkpoint whose block the node knows must descend from the previous checkpoint's block, or it is refused.
- While a node holds a checkpoint `(h, hash)`, a header at height `h` must have that hash, and a block above `h` must have it as its ancestor at `h`: no chain that leaves out the latest checkpoint is valid, whatever its chainwork. A new checkpoint whose block is on a side branch moves the node's tip to the most-work block descending from it.
- The authority's node signs the block `CHECKPOINT_DEPTH` (3) below its tip every `CHECKPOINT_EVERY` (4) blocks. A majority of the work can then rewrite at most the blocks after the latest checkpoint, about a minute and a half, instead of MAX_REORG_DEPTH. The authority cannot make an invalid block valid, move coins or create them; it only chooses which valid chain is final, and only until `U` or the end.

## 6. Blocks

A block is `{"header", "txs"}` plus `"shares"` when it includes any. It is valid if its header is valid (§5), its transactions match `tx_root` and its shares `shares_root`, its shares are valid (§3.5), it has at most MAX_BLOCK_TXS transactions with distinct txids, `len(C(txs)) <= MAX_BLOCK_BYTES` and at most MAX_BLOCK_SIGS signatures in total (1 per single-key transaction, `k` per multisig one, plus 1 per channel claim for its voucher), a job named in `work` exists in the parent state, every transaction applies in order (§3) followed by the events of §3.3, and the root of the resulting state equals `state_root` (§6.1). The genesis block has height 0, no transactions, no shares, filler work and a target `0 < target <= INITIAL_TARGET` (the network's starting difficulty, chosen when it is made); it is applied to the state whose reserve is TOTAL_SUPPLY and pays its finder's part (§3.5).

### 6.1 The state and its commitment

The state is a set of maps plus one counter. Their exact content is consensus, because the header commits to it:

| Map | Key | Value |
|---|---|---|
| `balances` | address | balance in yaprak, **non-zero only** (an entry whose balance becomes 0 is removed) |
| `nonces` | address | number of transactions the address has sent (present once ≥ 1) |
| `jobs` | job id | `{"buyer", "dims" [mu,ku,nu], "price", "escrow", "done": {unit key: batch id}, "delivered": {batch id: {"miner", "units", "count", "root", "manifest", "payees"*}}, "reserved": {unit key: batch id}, "batches": {batch id: {"miner", "units" (bitmap), "count", "height", "expires", "bond", "status" ("committed"/"revealed"), "root", "manifest", "settle_at", "disputes": {unit key: {"by", "deadline"}}, "proven": set of unit keys, "payees"*}}, "closes_at" (height or null), "closed" (bool)}` |
| `tasks` | task id | `{"buyer", "prog", "input", "prog_size", "fuel", "price", "escrow", "status" ("open"/"revealed"/"confirmed"/"failed"/"done"), "worker", "output", "reserved" (null or {"worker", "commitment", "height", "expires", "bond"}), "settle_at"}` |
| `closing` | height (integer) | job ids whose cancellation completes at that height, in the order the cancels were applied |
| `job_events` | height (integer) | `[kind, job id, batch id, null]` entries (§3.3) in the order they were scheduled |
| `task_settle` | height (integer) | task ids settling at that height, in scheduling order |
| `guards` | address | `{"guardian", "limit", "period", "frozen", "window", "spent"}` (§3.4) |
| `channels` | channel id | `{"buyer", "pub", "provider", "amount", "paid", "expires"}` |
| `channel_expiry` | height (integer) | channel ids ending at that height, in opening order |
| `protocol` | `"reserve"` | the reserve (§3.5) |
| `share_seen` | share id | height of the block that included it |
| `share_expiry` | height (integer) | share ids included at that height, in block order |

A unit key is `"i,l,j"` (decimal). Fields marked * are present only when the batch's commit named payees. Entries of `closing`, `job_events`, `task_settle`, `channel_expiry` and `share_expiry` are created when the first item is scheduled for their height and removed when that height is processed; a removed job batch (paid or dropped) leaves `batches`. The counter `open_disputes` is the number of open job disputes network wide.

**Lattice hash** (LtHash with 1024 16-bit lanes, as in Lewi et al., "Securing Update Propagation with Homomorphic Hashing", 2019). For a map `M`, key `k` and value `v`, `entry(M, k, v)` is the 2,048-byte output of SHAKE-256(`"grn-state-v1:"` ‖ `C([M, k, plain(v)])`) read as 1,024 little-endian uint16 lanes, where `plain` turns sets into lists sorted by code point and dictionary keys into strings. The state's lattice is the lane-wise sum, modulo 2^16, of `entry(M, k, v)` for every entry of every map plus `entry("meta", "open_disputes", open_disputes)`. `state_root = hex(H(lattice as 1,024 uint16 little-endian))`.

Adding an entry adds its vector and removing one subtracts it, so a node updates the lattice in O(entries a block changed) and the result does not depend on the order of writes. Two states have the same root exactly when they have the same content (up to a collision of the lattice hash).

### 6.2 Snapshots

Not consensus, but defined here so that nodes can exchange them. A snapshot of the state after block `b` at height `h` is a list of chunks and a manifest. The entries `[M, k, v]` (maps in the order of the table above, keys sorted, integer keys before string keys) are cut into chunks of about 1 MiB; a chunk is the JSON text (sorted keys, no spaces, sets written as `{"$set": [...]}`) of its entries list. The manifest is `C({"height": h, "hash": b, "root": state_root, "open_disputes": n, "chunks": [hash of each chunk]})`. Chunks and manifests travel as blobs. A node that receives one checks every chunk against the manifest and the lattice of all entries against the `state_root` of header `b`, which must be on its best header chain at least MAX_REORG_DEPTH blocks below the tip. Nodes write a snapshot when their base passes a multiple of SNAPSHOT_INTERVAL.

## 7. WASM execution

Tasks run on wasmtime WASMTIME_VERSION (a consensus constant) with fuel metering, NaN canonicalization, threads and relaxed SIMD disabled, and no imports. The host writes the input at address 65,536 (growing memory if needed), calls `green_run(65536, len)`, and reads `(out_ptr, out_len) = (ret >> 32, ret & 0xFFFFFFFF)` from the i64 return value. A trap, running out of fuel, or an output outside memory is a failed run. A claimed output is right iff the run succeeds and `hex(H(output))` equals it. Adjudication cost is `fuel + program_bytes · COMPILE_COST_PER_BYTE`, at most MAX_BLOCK_ADJUDICATION per block.

## 8. Not consensus (policy of the reference node)

These rules are local; a node may choose differently without forking:
- Blocks are filled by fee per byte, keeping each sender's transactions in nonce order.
- The mempool holds at most 5,000 transactions and 100 per sender; when full, the lowest fee-rate transaction that is last in its sender's queue is evicted for a better one.
- Replace-by-fee: a transaction with the same sender and nonce replaces a pending one if it pays at least 10% and at least MIN_TX_FEE more.
- Blob relay: job, task and manifest blobs of revealed batches, results listed in a manifest the node holds, and snapshot chunks and manifests are served to peers; committed but unrevealed results are not.

## 9. Peer-to-peer protocol (P2P v1)

Not consensus: it decides how nodes talk, not which chain is valid. Default port 8546 (the HTTP API on 8545 is for wallets and tools, see API.md).

### 9.1 Transport

1. Each side sends 32 bytes: a fresh X25519 public key with its top bit set at random (the receiver clears it).
2. Both compute the X25519 shared secret (all zeros is refused) and `okm = HKDF-SHA256(ikm = secret, salt = "grn-p2p-v1" ‖ genesis hash (32 bytes), info = initiator key ‖ responder key, 160 bytes)`, split into five 32-byte values: initiator→responder length key, initiator→responder body key, the same two responder→initiator, and the session id.
3. A frame is `ChaCha20-Poly1305(length key, n, uint32_LE(len(body)))` (20 bytes) followed by `ChaCha20-Poly1305(body key, n, body)`, where the nonce `n` is 4 zero bytes ‖ uint64_LE(frame counter). After REKEY_FRAMES frames in a direction both of its keys become `HKDF-Expand-SHA256(key, "grn-p2p-rekey", 32)` and the counter restarts at 0.
4. A body is one byte of message type followed by the payload: canonical JSON for every message except `blob`, whose payload is the raw blob.
5. The first frame each way is `version`: `{"proto": 1, "agent", "id": node public key, "sig", "port": listening port or 0, "relay": wants transaction announcements, "height", "tip", "work": chainwork as a decimal string, "you": the address the sender sees for the receiver, "time", "first": lowest height it has block bodies for, "features": optional messages it understands}`. Features are `"dtx"` (message 19), `"cmpct"` (20 to 22), `"offers"` (23), `"shares"` (24), `"relay"` (25, 26) and `"checkpoints"` (27); a `cmpct` carries the block's shares whole; a node sends an optional message only to a peer that listed it, and ignores features it does not know. `sig = Ed25519(node key, "grn-p2p-auth-v1" ‖ session id ‖ ("I" if the sender initiated, else "R"))`. A connection is up once each side has checked the other's signature. Peers given as `<id>@host:port` must present that id.

A node on another network derives different keys and fails at the first frame.

### 9.2 Messages

| Type | Payload | Meaning |
|---|---|---|
| 1 `version` | see above | handshake |
| 2 `ping`, 3 `pong` | `{"n"}` | liveness and latency |
| 4 `getaddr`, 5 `addr` | `{}`, `{"addrs": [{"host", "port", "id", "seen"}]}` | address gossip, at most 1,000 per message; one getaddr answered per connection |
| 6 `inv` | `{"items": [["tx" or "block", hash]]}` | announce, at most 1,000 items |
| 7 `getdata`, 8 `notfound` | same shape | request announced items / say they are gone |
| 9 `tx`, 10 `block` | `{"tx": ...}`, `{"block": ...}` | the item |
| 11 `getheaders` | `{"locator": [hashes]}` (at most 64) | the answer starts after the first locator hash on the responder's main chain |
| 12 `headers` | `{"headers": [header, ...]}` | at most 2,000 consecutive headers |
| 13 `getblob`, 14 `blob`, 15 `noblob` | `{"hash", "max"}`, raw bytes, `{"hash"}` | fetch a public blob of at most `max` bytes |
| 16 `getmempool` | `{}` | announce your mempool to me (once per connection) |
| 17 `getsnapinfo`, 18 `snapinfo` | `{}`, `{"snapshots": [{"id", "height", "hash", "root"}]}` | snapshots on offer (§6.2) |
| 19 `dtx` | `{"tx": ...}` | a transaction in its stem phase (Dandelion++, §9.3): pass it on or announce it, never serve it to `getdata` |
| 20 `cmpct` | `{"header", "salt": 16 hex, "ids": [12 hex], "fill": [[index, tx]]}` | a new block: its header, the transactions given in `fill` at their indexes, and for the others, in order, `short_id = BLAKE2b-48(txid bytes, key = block hash bytes ‖ salt bytes)` |
| 21 `getblocktxn`, 22 `blocktxn` | `{"block", "idx": [indexes]}`, `{"block", "txs": [...]}` | the transactions of a compact block the receiver could not find, in the order asked |
| 24 `share` | `{"share": header}` | a share (§3.5), checked (its work unit recomputed) before it is passed on; a fake one gets the sender banned |
| 23 `offers` | `{"offers": [offer, ...]}` (at most 100) | providers selling compute (§12): `{"provider", "pub", "url", "price", "max_dim", "credit", "time", "sig"}` and since v0.21 optionally `"kx"` (the provider's X25519 key for sealed requests, 64 hex digits), `sig = Ed25519(pub, C(["grn-offer", chain_id, offer without sig]))`, `provider = address(pub)` |
| 27 `checkpoint` | `{"cp": checkpoint or end}` | a checkpoint of the network's authority (§5.6), sent when the connection opens and whenever a newer one is taken; a forged one counts against the sender |
| 25 `relay`, 26 `relayed` | raw: `id (8) ‖ len (1) ‖ provider address ‖ sealed request`; raw: `id (8) ‖ status (1: 0 carried, 1 refused) ‖ sealed answer or reason` | carry a sealed request (§12) to a provider on offer and bring back its answer; at most 4 MiB each way |

### 9.3 Behaviour of the reference node

- **Relay**: new transactions are announced to each peer in batches at exponentially distributed intervals (mean 0.5 s). A node requests an announced item from one peer and asks the next announcer after a timeout or `notfound`.
- **Transaction origin (Dandelion++)**: transactions submitted to a node through its API are not announced. The node checks them against its mempool plus its stem pool and sends them as `dtx` to one of two outbound peers (that listed `dtx`) drawn per 10-minute epoch; each incoming link, and the node's own transactions, map to one of the two for the whole epoch. A node receiving `dtx` passes it on the same way, unless it is a diffuser this epoch (probability 0.1, drawn per epoch), has nowhere to pass it, or already holds it in its stem pool (a loop): then it adds it to its mempool and announces it to every peer. Every node on a stem sets an embargo of 15 s plus an exponential delay (mean 5 s) and announces the transaction itself if it has not seen it announced by then. The node's own protocol transactions (batch commits, reveals, answers, challenges) and replace-by-fee replacements are announced directly.
- **Blocks (compact relay)**: a new block goes at once to peers that listed `cmpct` as a `cmpct` message (transactions the peer is known to have seen as short ids, the rest whole), to the others as an `inv`. The receiver rebuilds the block from its mempool and stem pool, asks with `getblocktxn` for what it lacks, and if the rebuilt transactions do not match `tx_root` (a short-id collision) fetches the whole block with `getdata`. A collision is not misbehaviour.
- **Addresses from gossip**: only IP addresses (no names) are kept, and only global ones, unless the peer that sent them and the address are both local (loopback, private or link-local); otherwise any peer could make nodes dial into their own networks.
- **Compute offers**: a node keeps one offer per provider address (at most 1,000, the least recently renewed dropped first) if it is validly signed, names an IP address (not a host name) in its URL, is at most an hour old, and its provider holds at least 0.1 GRN on the node's chain (an offer that arrives before the block funding it waits and is checked again); it relays new offers to peers that listed `offers`, sends its list to such a peer when it connects, and serves it at `GET /v1/providers`. Forged offers add up to a ban; offers are rate-limited per peer.
- **Carrying sealed requests**: a node carries a peer's `relay` only to a provider it holds a current offer for (never to an arbitrary address), POSTs it to the offer's URL `/sealed`, and returns the answer as `relayed` with the same id; it refuses (status 1) a provider whose address is not public when the asking peer is remote, more than 4 requests in flight per peer, 8 in all, or more than 256 MiB per peer per minute. It never sends its own requests itself: a node's `POST /relay` (admin) hands a sealed request to a random peer that listed `relay`.
- **Sync**: a node that sees a peer with more chainwork asks for headers, checks each one against every rule of §5 except the work unit, recomputes the work unit of a random sample plus the last header of every message, then downloads the blocks of the best header chain from several peers at once (a window of 256 blocks) and connects them in order, their work units recomputed in parallel beforehand.
- **Fast sync** (opt in): a new node first gets the header chain, then the newest final snapshot on offer (§6.2), and from there fully validates every block; with no usable snapshot it syncs from genesis.
- **Limits**: frames above 4 MiB are refused before they are read unless that peer owes a blob we asked for; handshakes must finish in 10 s; at most 1,000 messages wait per peer; at most 32 inbound connections, at most 4 per network group (/16 IPv4, /32 IPv6; local addresses exempt), the youngest peer of the most crowded group evicted for a newcomer while the fastest, most useful and longest-connected ones are protected; 8 outbound connections spread over network groups, 2 of them relaying blocks only; address gossip rate-limited to 0.1 per second per peer after a burst; one source network group can hold at most 64 new address-book entries.
- **Misbehaviour**: invalid blocks or headers, failed work samples, bad snapshots, data nobody asked for and oversized frames get a peer banned for 24 h; malformed messages add up to a ban. Local addresses are banned by node id, others by IP.

## 10. Constants

| Name | Value |
|---|---|
| UNIT, BLOCK, NOISE_RANK | 256, 32, 32 |
| FREIVALDS_PRIME, FREIVALDS_ROUNDS | 2^31 − 1, 4 |
| TARGET_BLOCK_SECONDS, ASERT_HALFLIFE | 15, 1,800 |
| MAX_TARGET, INITIAL_TARGET | 2^256 − 1, MAX_TARGET / 200 |
| CHECKPOINT_EVERY, CHECKPOINT_DEPTH (policy of the authority's node) | 4, 3 |
| MTP_WINDOW, MAX_FUTURE_SECONDS, MAX_REORG_DEPTH | 11, 60, 100 |
| SNAPSHOT_INTERVAL (not consensus) | 2,880 |
| SIGNAL_PERIOD, SIGNAL_THRESHOLD | 20,160, 18,144 (90%) |
| P2P: REKEY_FRAMES, MAX_FRAME, MAX_HEADERS, MAX_INV | 4,096, 4 MiB, 2,000, 1,000 |
| P2P (not consensus): Dandelion++ epoch, fluff probability, destinations, embargo | 600 s, 0.1, 2, 15 s + Exp(mean 5 s) |
| TOTAL_SUPPLY, EMISSION_DIVISOR, FEE_RESERVE_PERCENT, FEE_BURN_PERCENT | 2,100,000,000,000,000 yaprak (21,000,000 GRN), 2^24, 30, 20 |
| SHARE_FACTOR, SHARES_PERCENT, SHARE_INCLUSION_PERCENT, MAX_BLOCK_SHARES, SHARE_WINDOW | 128, 99, 5, 256, 32 |
| MAX_BLOCK_TXS, MAX_BLOCK_BYTES | 3,000, 1,000,000 |
| MAX_JOB_UNITS_PER_DIM, MAX_BATCH_UNITS | 16, 4,096 |
| MIN_TX_FEE (per signature), DATA_FEE_PER_BYTE, NEW_ACCOUNT_MIN | 500, 1, 10,000 |
| RESERVE_BLOCKS, JOB_CHALLENGE_BLOCKS, JOB_RESPONSE_BLOCKS | 20, 40, 20 |
| MAX_BLOCK_JOB_RESPONSES, BOND_DIVISOR, CANCEL_DELAY | 32, 10, 40 |
| `bond_for(price)` | ceil(price / BOND_DIVISOR) |
| WASMTIME_VERSION | 49.0.0 |
| MAX_GUARD_PERIOD | 100,000,000 |
| MAX_MULTISIG_KEYS, MAX_BLOCK_SIGS | 16, 6,000 |
| MAX_PAYEES, MAX_PAYEE_WEIGHT | 32, 2^32 |
| MIN_CHANNEL_BLOCKS, MAX_CHANNEL_BLOCKS | 240, 161,280 |
| MAX_TASK_FUEL, MAX_TASK_PROGRAM_BYTES, MAX_TASK_INPUT_BYTES | 500,000,000, 2^20, 2^22 |
| COMPILE_COST_PER_BYTE, MAX_BLOCK_ADJUDICATION, ADJUDICATION_FEE_PER_MUNIT | 200, 1,500,000,000, 1,000 |
| TASK_RESERVE_BLOCKS, TASK_CHALLENGE_BLOCKS, TASK_BOND_DIVISOR | 20, 40, 2 |
| `task_bond(price)` | max(1, ceil(price / TASK_BOND_DIVISOR)) |

## 11. Wallet keys (informative)

Not consensus; defined so that wallets agree. A wallet's key comes from a BIP-39 recovery phrase (12 to 24 words of the standard English list, whose file's SHA-256 is `2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda`; 24 words by default) and an optional passphrase: `seed = PBKDF2-HMAC-SHA512(NFKD(words), "mnemonic" ‖ NFKD(passphrase), 2048, 64 bytes)`. The Ed25519 secret is derived from the seed with SLIP-0010 along `m/44'/HD_COIN_TYPE'/account'`, every step hardened (Ed25519 has no other kind). HD_COIN_TYPE is 7777, provisional until a SLIP-0044 number is registered before a public launch; a wallet file records its path.

## 12. Compute channels (informative)

Not consensus beyond the two transactions of §3.2; defined so that buyers and providers agree. A provider serves HTTP: `GET /offer`, `POST /matmul` with headers `X-Channel`, `X-Shape: m,k,n` and optionally `X-Voucher-Total`, `X-Voucher`, a body of A (`m·k` int8) then B (`k·n` int8) in row-major order, answered with C = A·B as `m·n` little-endian int32 (or 402: pay first or the channel is spent; 404: unknown channel; 409: the channel ends within 20 blocks), `POST /voucher`, and `POST /close` (a final voucher, after which the provider claims with `close`). A provider answers a product request from its headers alone (shape, channel, voucher, credit and funds) before it reads the body. A product costs `price · max(1, ceil(m·k·n / 2^24))`. The buyer checks each C with Freivalds' test over the integers (a secret random matrix with entries in `[0, 2^s)`, `s` the largest value that keeps every sum below 2^53, and `ceil(120 / s)` columns, so a wrong C passes with probability below 2^-120), then signs a voucher for the total it owes for every result checked so far. A provider delivers at most `credit` results ahead of vouchers; it claims vouchers worth at least 40·MIN_TX_FEE, and anything left when the channel is near its end.

Since v0.21 a provider also serves `POST /matmul_mod` (headers as `/matmul` plus `X-Modulus: p`, `2 <= p <= 2^20` and `k·(p-1)^2 < 2^53`; a body of A then B as little-endian int32 with entries in `[0, p)`; answered with `(A·B) mod p` as `m·n` int32, priced like `/matmul`; the buyer checks it with Freivalds' test mod p, 12 columns of entries in `[0, 2^10)`), and `POST /sealed`: a request sealed to the offer's `kx`: `eph ‖ nonce ‖ ChaCha20-Poly1305(key, nonce, plaintext, aad "request")` with `key = HKDF-SHA256(X25519(eph, kx), salt = eph ‖ kx, info "green sealed request v1")` and `plaintext = len (4, big-endian) ‖ JSON {"path", "headers"} ‖ body`, for paths `/matmul`, `/matmul_mod`, `/voucher` and `/close`; the answer is `nonce ‖ ChaCha20-Poly1305(key, nonce, len ‖ JSON {"status", "headers"} ‖ body, aad "answer")`. The provider's `kx` key is derived from its wallet key: `X25519(HKDF-SHA256(secret, info "green provider kx v1"))`.
| `challenge_fee(fuel, bytes)` | MIN_TX_FEE + ceil(adjudication_cost · ADJUDICATION_FEE_PER_MUNIT / 10^6) |
