# Green node HTTP API (v0.21)

Every node serves this API on its port (default 8545, `127.0.0.1` unless started with `--host`). Requests and responses are JSON unless noted. Amounts are integers in yaprak (1 GRN = 10^8 yaprak). Errors come back as `{"error": "..."}` with status 400 (bad request or invalid transaction), 403 (admin endpoint called remotely), 404 (unknown), 409 (data the node does not have yet), 413 (too large) or 500.

This is the reference node's interface for wallets, tools and agents, not consensus (SPEC.md is). **Use the `/v1` prefix** (`/v1/status`, `/v1/tx`, ...): it is the stable version, and later incompatible changes will get a new prefix. The same paths without a prefix still work for old clients. Nodes do not use it to talk to each other: since v0.7 they use the encrypted P2P protocol on port 8546 (SPEC.md §9), and nothing sent to this API can add a peer or inject a block from outside the node's machine.

## Chain

| Method and path | Returns |
|---|---|
| `GET /status` | `coin`, `genesis` (network id), `height`, `tip`, `time`, `chainwork` (string), `target`, `units_per_block`, `peers` (their P2P addresses), `p2p` (`node_id`, `listen`, `inbound`, `outbound`, `known_addresses`, `headers_ahead`, `best_header_height`, `blocks_in_flight`, `fast_sync`, `stats`), `mempool`, `mining`, `miner`, `units_per_second`, `stats`, `warnings` (e.g. an unknown rule change signalled by most recent blocks), `pruned`, `first_block` (lowest height whose body this node has) |
| `GET /chain?from=<height>` | `{"hashes": [...]}`: up to 5,000 main-chain block hashes from that height |
| `GET /debug/memory` | admin (this machine, with the client header): what the node holds in memory: Python's allocated blocks, the commonest object types, the sizes of its collections; with `PYTHONTRACEMALLOC=n` also the allocation sites that grew since the previous call |
| `GET /block/<hash>` | the block `{"header": {...}, "txs": [...]}` (old blocks come from the store; 404 for blocks below the snapshot a fast-synced node started from) |
| `GET /height/<n>` | the main-chain block at height n |
| `GET /headers?from=<n>&count=<k>` | up to 2,000 main-chain headers from height n (light clients) |
| `GET /tx/<txid>` | `status` (`pending` or `confirmed`), `tx`, and when confirmed `height`, `block`, `position`, `confirmations`, `pruned` (the body is gone on a pruned node). A transaction this node is still passing along its Dandelion++ stem has `"relay": "stem"`; it reaches the mempool once announced (seconds) |
| `GET /proof/tx/<txid>` | a Merkle proof that the transaction is in a main-chain block: `txid`, `block`, `height`, `index`, `count`, `proof`, `header`. Check it with `green.crypto.merkle_verify(txid, index, count, proof, header["tx_root"])`, or let `green.light.LightClient` do it against headers it verified itself |
| `GET /address/<address>/history?limit=<n>&before=<height>` | what the address took part in, newest first: `items` (`height`, `position`, `role`: `sender`, `recipient`, `account`, `guardian` or `miner`; `type`; `txid`, or `block` for a block reward), `pending` (mempool), `next` (pass as `before` for the next page) |
| `GET /fee?blocks=<n>` | fee estimate for inclusion within n blocks (default 2, at most 100): `fee_rate` (yaprak per byte, 0 when the minimum is enough), `transfer_fee` (for a typical transfer, at least `min_fee`), `min_fee`, `mempool` (transactions waiting), `basis` (`mempool`, `recent blocks` or `minimum`) |
| `GET /wait?tip=<hash>&timeout=<s>` | long-poll: answers as soon as the tip is not `<hash>` any more, or after the timeout (at most 60 s): `tip`, `height`, `changed`, `mempool`. 503 if too many clients are already waiting. Clients use this instead of polling (`green.node.wait_for_block`) |
| `GET /deployments` | scheduled rule changes (UPGRADES.md): `name`, `bit`, `state` (`defined`, `started`, `locked_in`, `active`, `failed`), `start`, `timeout`, `min_activation`, `lock_in_on_timeout`, `period_start`, `signalled` (blocks of this period so far, while started), `threshold`, `period` |
| `GET /supply` | supply audit: `balances`, `locked` (escrow, bonds, channel funds), `reserve`, `total`, `supply` (the fixed total supply), `ok` (true if they match exactly) |
| `GET /metrics` | Prometheus text format (`text/plain; version=0.0.4`): height, base height, peers (total, inbound, outbound), mempool, mining, units per second, open disputes, blocks found, work units, useful units, results delivered, frauds caught, headers ahead of the chain, blocks in flight, peers banned, P2P bytes received and sent, supply audit (`green_supply_ok`), warnings (`green_warnings`). Alert rules: [contrib/prometheus-alerts.yml](contrib/prometheus-alerts.yml) |

## Accounts and transactions

| Method and path | Returns |
|---|---|
| `GET /providers` | compute on offer (SPEC.md §12): signed offers `{"provider", "pub", "url", "price", "max_dim", "credit", "time", "sig"}` this node has heard over the network, cheapest first |
| `GET /channel/<id>` | an open compute channel: `buyer`, `pub`, `provider`, `amount`, `paid`, `left`, `expires`, `height`; 404 before it is confirmed and after it ends |
| `GET /account/<address>` | `balance`, `nonce` (confirmed), `pending_balance`, `pending_nonce` (including the mempool and this node's own transactions still on their stem; use `pending_nonce` for the next transaction), `guard` (null, or `guardian`, `limit`, `period`, `frozen`, `window`, `spent`, `spend_left`) |
| `GET /mempool` | the pending transactions |
| `POST /tx` | body: a signed transaction (SPEC.md §3). Returns `{"txid": ...}` or 400 with the reason. The node first passes it along a Dandelion++ stem (unless started with `--no-dandelion`), so for a few seconds it is not in the mempool of any node. Data a transaction needs (job matrices, WASM programs, results) must have been uploaded or be fetchable from peers (409 otherwise) |

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
| `GET /peers` | the connected peers: `addr` (socket address), `listen` (its P2P address, if it listens), `id` (node key), `inbound`, `block_only`, `agent`, `height` (the highest block it has sent or announced to us), `ping_ms`, `session` (first 16 hex digits of the session id, to compare out of band), `connected` (seconds), `bytes_in`, `bytes_out`, `score` (misbehaviour points; 100 is a ban) |

## Admin (only from the node's own machine)

An admin request must come from the node's machine, carry the header `X-Green-Client: 1`, and name the machine in `Host` (`127.0.0.1`, `localhost` or `::1`, with any port); otherwise the answer is 403. A web page open in a browser on the same machine can reach the loopback address too, but it cannot add that header to a cross-site request (it would need a CORS preflight, which the node never grants), and a DNS-rebinding page sends its own name as `Host`. Green's own tools (`grn`, `green.node.http_post`) send the header.

| Method and path | Use |
|---|---|
| `POST /blob` | raw body: store data (job matrices, a WASM program or input) before posting the transaction that needs it; returns `{"hash": ...}` |
| `POST /peers` | `{"addr": "host:port"}` or `{"addr": "<node id>@host:port"}`: keep a P2P connection to that node (with an id, it must present that key). Returns `{"added": bool}` |
| `POST /mining` | any of `{"on": bool, "intensity": 1..100, "gpu": bool}`: switch mining, mine that percent of the time, mine on the GPU (checked against the CPU first) or the CPU. Returns `{"mining", "intensity", "gpu"}` (the GPU's name or null) |
| `POST /relay` | header `X-Relay-To: <provider address>`, raw body: a sealed request (SPEC.md §12); a random peer carries it to that provider and the sealed answer comes back raw (503 if no peer could) |
| `GET /debug/process` | `{"pid"}`: the node's own process id (launchers wait on it) |
| `GET /debug/threads` | the stack of every thread of the node (thread name -> its last frames), to diagnose a node that hangs; admin rules as above |
| `POST /shutdown` | `{}`: stop the node cleanly (as `grn stop` does); a provider first claims what buyers owe it. Returns `{"stopping": true}` |

## The app's compute proxy (`grn app`)

Not part of the node: `grn app` serves it on `127.0.0.1:8600`. It buys each product on the network through the selected account's payment channels (opened on first use, to the cheapest providers), checks it with Freivalds' test, and answers in the providers' wire format.

| Method and path | Use |
|---|---|
| `POST /v1/matmul` | headers `X-Shape: m,k,n` (each 1..4096) and optionally `X-Account: <name>` and `X-Privacy: split \| secret \| none` (default split; see SECURITY.md B4); body A (m·k int8, row-major) then B (k·n int8). Returns m·n little-endian int32. 400 if the body does not match the shape exactly, 502 if no provider could do it |
| `GET /v1/status` | `{"proxy": {"products", "macs", "seconds", "errors"}, "buyer"}` |

The app's own page uses `/api/*` endpoints that need the page's token (`X-App-Token`); every request must name this machine in `Host`.
