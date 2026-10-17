# Running a Green node

How to install, configure, monitor, back up, restore and upgrade a node. For the HTTP API see [API.md](API.md); for what changes between releases, [UPGRADES.md](UPGRADES.md).

## Install

```bash
python -m pip install .
```

This installs the `grn` command (the same as `python -m green`). Python 3.11 or newer. The GPU miner needs the optional `gpu` extra. A container image is described at the top of the [Dockerfile](Dockerfile).

## The data folder

Everything a node keeps is in one folder (`--data`, default `data`):

| File | What | Back up? |
|---|---|---|
| `wallet.json` | the wallet key, encrypted with your password | **Yes, and the password.** Without both, the coins are gone |
| `genesis.json` | the network you are on | Yes (small; also available from any node of the network) |
| `node_key` | the node's P2P identity (not a wallet; holds no coins) | Recommended, so peers that pinned your node still recognise it |
| `green.toml` | your settings | Yes |
| `chain.db`, `blobs/` | the chain, the ledger and the data blobs | Optional: a node can resync (or fast-sync) them |
| `peers.json`, `banlist.json` | address book and bans | No |

## Settings

Settings go in `<data>/green.toml`; command-line flags override them:

```toml
[node]
p2p_host = "0.0.0.0"       # reachable by other nodes; the API stays on 127.0.0.1
p2p_port = 8546
port = 8545                # HTTP API
peer = ["<node id>@203.0.113.7:8546"]
mine = true
gpu = false
log_level = "info"         # debug, info, warning, error
log_file = "node.log"      # rotated at 10 MB, 5 files kept
```

Every key: `port`, `host`, `p2p_port`, `p2p_host`, `peer`, `genesis`, `mine`, `gpu`, `fast_sync`, `no_listen`, `no_signal`, `report`, `log_level`, `log_file`, `prune`, `prune_keep`, `pool_port`, `pool_host`, `pool_fee`, `no_dandelion`, `provide_port`, `provide_host`, `provide_url`, `provide_price`, `portmap`. An unknown key or a wrong type stops the node with a message instead of being ignored.

## Archive or pruned

By default a node is an **archive** node: it keeps every block and every data blob, so other nodes can sync all of history from it. With `prune = true` (or `--prune`) it keeps:

- every header, and block bodies only for the last `prune_keep` blocks (default 5,760, about a day; at least the reorg window);
- the ledger and its snapshots, so it still serves fast sync to new nodes;
- the data still in use: matrices of unfinished jobs and of jobs used by recent blocks, results of live batches and of unfinished jobs, programs of unsettled tasks, data named by pending transactions, and its own unrevealed work.

Data nothing uses any more is marked, and deleted about a day later if it is still unused, so buyers have time to fetch results. Freed space goes back to the disk. Peers learn from the handshake where a pruned node's bodies start and ask others for older blocks. A pruned node cannot serve a full sync from genesis; new nodes joining through it use `fast_sync`.

## Running a pool

`pool_port = 8547` (with `pool_host = "0.0.0.0"` for miners on other machines) makes the node a pool. Its blocks pay the miners of the last 1,000 shares by weight, plus `pool_fee` percent (default 1) to the node's wallet, all inside the block: the pool never holds miners' coins. When buyers have posted jobs, the pool also sells its miners' compute: it hands them units, checks every result, and commits batches whose payment goes to the miners who computed them (plus the pool's fee). The bonds come from the node's wallet, so a pool sells only as many units as its balance can bond; with an empty wallet its miners just mine. Open the pool port in the firewall. The pool recomputes every share (about 2 ms each), and pauses a client that sends 20 bad shares for 10 minutes. `GET /status` on the pool port shows miners, shares, blocks and the next payee list.

## Selling compute

`provide_port = 8548`, `provide_host = "0.0.0.0"` and `provide_url = "http://<public address>:8548"` make the node sell matrix products through compute channels at `provide_price` yaprak per 2^24 multiply-adds (default 1,000). The offer is advertised over the network once the wallet holds 0.1 GRN and renewed every 20 minutes. Buyers pay with vouchers; the node claims them on chain when they are worth 20 transaction fees, before a channel ends, and on shutdown. Its books are in `provider.json` in the data folder: back it up with the wallet (vouchers not yet claimed are money). Open the port in the firewall; each request is at most 32 MiB.

## Behind a home router

`portmap = true` asks the router to forward the P2P port (and the provider's port) with PCP or NAT-PMP, renews the mapping every half hour and removes it when the node stops; the log says whether the router agreed. Routers without PCP or NAT-PMP (or with it switched off) need a manual port forward. Behind carrier-grade NAT (the router's own address is in 100.64.0.0/10 or private) no mapping helps, and the node advertises nothing.

## Fees

The node pays the fee market's rate for its own commits, reveals, answers and claims, and raises the fee of any still waiting after 3 blocks, up to 0.1 GRN. For a wallet transaction stuck in a busy mempool: `grn bump <txid>` (optionally `--fee`).

## Network and firewall

- **8546/tcp (P2P)**: open it inbound if you want other nodes to connect to you (recommended for the network's health). Traffic is encrypted.
- **8545/tcp (HTTP API)**: keep it local (the default). It is for your own wallet, tools and agents; admin endpoints only answer from the node's own machine, but the API is not meant to face the internet.

## Running as a service

On Linux with systemd, [contrib/green.service](contrib/green.service) runs the node as an unprivileged user with a read-only system, and hands the wallet password to it as a systemd credential (a file only the node can read), not through the environment:

```bash
sudo systemctl enable --now green
```
```bash
journalctl -u green -f
```

For other setups: the node reads the password from the file named by `GREEN_WALLET_PASSWORD_FILE` (preferred) or from `GREEN_WALLET_PASSWORD`.

**Stopping:** `grn stop`, Ctrl+C, or SIGTERM (`systemctl stop green`, `docker stop`) all run the same clean shutdown: a node that sells compute first claims what buyers owe it, then the store closes. Give it the unit's `TimeoutStopSec` (60 s).

## Monitoring

- `grn status`: height, peers, sync progress, warnings.
- `GET /metrics`: Prometheus metrics. [contrib/prometheus-alerts.yml](contrib/prometheus-alerts.yml) has alert rules for a failed supply audit (critical: coins not accounted for), no peers, a stalled chain, falling behind, operator warnings and ban storms.
- `grn supply`: checks that every coin is accounted for.
- `grn deployments`: scheduled rule changes and their signalling.

Act on every warning `grn status` shows. The most important one says that most blocks signal a rule change your node does not know: upgrade before it activates.

## Backup and restore

Write down the wallet's recovery phrase (`grn wallet phrase`) and keep it offline: it alone restores the wallet (`grn wallet restore`), with any number of copies of the file lost. Also back up `wallet.json` (with its password, stored separately), `genesis.json`, `node_key` and `green.toml`. To restore, put them in an empty data folder and start the node; it syncs the chain from its peers (`fast_sync = true` makes a new node start from a verified ledger snapshot instead of replaying all of history). Copying `chain.db` and `blobs/` of a stopped node also works and avoids the resync.

## Upgrading

1. Download the release and its signatures; check them: `grn release verify SHA256SUMS <maintainer>.sig ...`. Install only a "Genuine release."
2. Stop the node, install the new version, start it. The data folder is kept; a release that needs a new data format says so in [CHANGELOG.md](CHANGELOG.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| No peers | Is `peer` set or are seed nodes known? Is 8546 reachable? `grn peers` |
| Stuck at a height | `grn status` (syncing?), `GET /metrics` `green_headers_ahead`, the log at `log_level = "debug"` |
| "this data directory belongs to another network" | The data folder was made for another genesis; use another folder |
| "wasmtime ... required" | Install the exact pinned wasmtime version (`requirements.txt`): task adjudication is consensus |
