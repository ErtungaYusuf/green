# Green node HTTP API (v0.13)

Every node serves this API on its port (default 8545, `127.0.0.1` unless started with `--host`). Requests and responses are JSON unless noted. Amounts are integers in yaprak (1 GRN = 10^8 yaprak). Errors come back as `{"error": "..."}` with status 400 (bad request or invalid transaction), 403 (admin endpoint called remotely), 404 (unknown), 409 (data the node does not have yet), 413 (too large) or 500.

This is the reference node's interface for wallets, tools and agents, not consensus (SPEC.md is). **Use the `/v1` prefix** (`/v1/status`, `/v1/tx`, ...): it is the stable version, and later incompatible changes will get a new prefix. The same paths without a prefix still work for old clients. Nodes do not use it to talk to each other: since v0.7 they use the encrypted P2P protocol on port 8546 (SPEC.md §9), and nothing sent to this API can add a peer or inject a block from outside the node's machine.

## Chain

| Method and path | Returns |
|---|---|
| `GET /status` | `coin`, `genesis` (network id), `height`, `tip`, `time`, `chainwork` (string), `target`, `units_per_block`, `peers` (their P2P addresses), `p2p` (`node_id`, `listen`, `inbound`, `outbound`, `known_addresses`, `headers_ahead`, `best_header_height`, `blocks_in_flight`, `fast_sync`, `stats`), `mempool`, `mining`, `miner`, `units_per_second`, `stats`, `warnings` (e.g. an unknown rule change signalled by most recent blocks), `pruned`, `first_block` (lowest height whose body this node has) |
| `GET /chain?from=<height>` | `{"hashes": [...]}`: up to 5,000 main-chain block hashes from that height |
| `GET /block/<hash>` | the block `{"header": {...}, "txs": [...]}` (old blocks come from the store; 404 for blocks below the snapshot a fast-synced node started from) |
| `GET /height/<n>` | the main-chain block at height n |
| `GET /headers?from=<n>&count=<k>` | up to 2,000 main-chain headers from height n (light clients) |
| `GET /tx/<txid>` | `status` (`pending` or `confirmed`), `tx`, and when confirmed `height`, `block`, `position`, `confirmations`, `pruned` (the body is gone on a pruned node) |
| `GET /proof/tx/<txid>` | a Merkle proof that the transaction is in a main-chain block: `txid`, `block`, `height`, `index`, `count`, `proof`, `header`. Check it with `green.crypto.merkle_verify(txid, index, count, proof, header["tx_root"])`, or let `green.light.LightClient` do it against headers it verified itself |
| `GET /address/<address>/history?limit=<n>&before=<height>` | what the address took part in, newest first: `items` (`height`, `position`, `role`: `sender`, `recipient`, `account`, `guardian` or `miner`; `type`; `txid`, or `block` for a block reward), `pending` (mempool), `next` (pass as `before` for the next page) |
| `GET /fee?blocks=<n>` | fee estimate for inclusion within n blocks (default 2, at most 100): `fee_rate` (yaprak per byte, 0 when the minimum is enough), `transfer_fee` (for a typical transfer, at least `min_fee`), `min_fee`, `mempool` (transactions waiting), `basis` (`mempool`, `recent blocks` or `minimum`) |
| `GET /wait?tip=<hash>&timeout=<s>` | long-poll: answers as soon as the tip is not `<hash>` any more, or after the timeout (at most 60 s): `tip`, `height`, `changed`, `mempool`. 503 if too many clients are already waiting. Clients use this instead of polling (`green.node.wait_for_block`) |
| `GET /deployments` | scheduled rule changes (UPGRADES.md): `name`, `bit`, `state` (`defined`, `started`, `locked_in`, `active`, `failed`), `start`, `timeout`, `min_activation`, `lock_in_on_timeout`, `period_start`, `signalled` (blocks of this period so far, while started), `threshold`, `period` |
| `GET /supply` | supply audit: `balances`, `locked` (escrow and bonds), `total`, `minted` (sum of block rewards), `ok` (true if total equals minted) |
| `GET /metrics` | Prometheus text format (`text/plain; version=0.0.4`): height, base height, peers (total, inbound, outbound), mempool, mining, units per second, open disputes, blocks found, work units, useful units, results delivered, frauds caught, headers ahead of the chain, blocks in flight, peers banned, P2P bytes received and sent, supply audit (`green_supply_ok`), warnings (`green_warnings`). Alert rules: [contrib/prometheus-alerts.yml](contrib/prometheus-alerts.yml) |

## Accounts and transactions

| Method and path | Returns |
|---|---|
| `GET /account/<address>` | `balance`, `nonce` (confirmed), `pending_balance`, `pending_nonce` (including the mempool; use `pending_nonce` for the next transaction), `guard` (null, or `guardian`, `limit`, `period`, `frozen`, `window`, `spent`, `spend_left`) |
| `GET /mempool` | the pending transactions |
| `POST /tx` | body: a signed transaction (SPEC.md §3). Returns `{"txid": ...}` or 400 with the reason. Data a transaction needs (job matrices, WASM programs, results) must have been uploaded or be fetchable from peers (409 otherwise) |

Transactions are built and signed client side: `green.tx.make_tx(key, chain_id, type, nonce, fee, **fields)`, with `chain_id` = `genesis` from `/status`. From a multisig account: `green.tx.make_multisig_tx(chain_id, k, keys, type, nonce, fee, **fields)`, then `green.tx.add_signature(tx, key, chain_id)` by k of the key holders; the account's address (`green.tx.multisig_address(k, keys)`) works with `/account` like any other. Minimum fees: `green.tx.min_fee(tx)`; `task_post` and `task_challenge` also pay for data and adjudication (`green.tx.task_post_fee`, `green.params.challenge_fee`).

## Compute markets

| Method and path | Returns |
|---|---|
| `GET /jobs` | every matmul job (same fields as below, without fetching manifests) |
| `GET /job/<id>` | `buyer`, `dims`, `price`, `escrow`, `units_total`, `units_done` (final), `units_pending` (in a challenge window), `units_reserved` (committed), `results` (unit key → result hash, for every delivered unit whose manifest this node holds), `pending` (unit key → batch id), `batches` (live batches: `id`, `miner`, `status`, `units`, `root`, `manifest`, `expires`, `settle_at`, `disputes`, `proven`), `height`. Asked from the node's own machine, it first fetches missing manifests from peers |
| `GET /tasks` | every WASM task |
| `GET /task/<id>` | `buyer`, `prog`, `input`, `prog_size`, `fuel`, `price`, `status` (`open`, `revealed`, `confirmed`, `failed`, `done`), `worker`, `output`, `settle_at`, `challenge_fee` |
| `GET /blob/<hash>` | raw bytes (`application/octet-stream`) of public data: job matrices, WASM programs and inputs, manifests and results of revealed batches, snapshot chunks and manifests. Committed but unrevealed results are never served. Asked from the node's own machine, a missing public blob is fetched from peers |

## Peers

| Method and path | Returns |
|---|---|
| `GET /peers` | the connected peers: `addr` (socket address), `listen` (its P2P address, if it listens), `id` (node key), `inbound`, `block_only`, `agent`, `height`, `ping_ms`, `session` (first 16 hex digits of the session id, to compare out of band), `connected` (seconds), `bytes_in`, `bytes_out`, `score` (misbehaviour points; 100 is a ban) |

## Admin (only from the node's own machine)

| Method and path | Use |
|---|---|
| `POST /blob` | raw body: store data (job matrices, a WASM program or input) before posting the transaction that needs it; returns `{"hash": ...}` |
| `POST /peers` | `{"addr": "host:port"}` or `{"addr": "<node id>@host:port"}`: keep a P2P connection to that node (with an id, it must present that key). Returns `{"added": bool}` |
| `POST /mining` | `{"on": true \| false}`: switch mining |
