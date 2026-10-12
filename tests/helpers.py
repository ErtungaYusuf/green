"""Shared helpers for the tests: in-memory chains and quick block building."""

import numpy as np

from green import params, tx as txmod
from green.chain import BlobStore, Chain, block_hash, make_genesis, solve
from green.crypto import hhex, merkle_proof, merkle_root
from green.tx import make_tx

N = params.UNIT
T0 = 1_000_000


def new_chain(key, msg="test genesis"):
    return Chain(make_genesis(key.address, msg, timestamp=T0), BlobStore())


def mine(chain, key, txs=(), work=None, t=None, parent=None):
    """Build and solve a block (tests run with an always-winning target)."""
    if parent is not None:
        saved = chain.tip
        chain.tip = parent
    try:
        header, chosen, _ = chain.build_template(key.address, list(txs), now=t or T0 + 10 * (chain.height + 1))
    finally:
        if parent is not None:
            chain.tip = saved
    if work is not None:
        header["work"] = work
    return grind(chain, header, chosen)


def grind(chain, header, txs):
    """Find a nonce whose ticket meets the target, like a real miner."""
    while True:
        header["transcript"], ticket, useful = solve(header, chain.blobs)
        if ticket <= int(header["target"], 16):
            return {"header": header, "txs": txs}, useful
        header["nonce"] += 1


def add(chain, key, txs=(), **kw):
    block, useful = mine(chain, key, txs, **kw)
    assert chain.add_block(block) == "added"
    assert len(block["txs"]) == len(txs), "a transaction was left out of the block"
    # Every block any test builds also checks the state commitment: the hash kept
    # up to date per block must equal the hash recomputed over the whole ledger.
    state = chain.states[block_hash(block)]
    assert np.array_equal(state.lattice, state.recompute_lattice()), "incremental state root drifted"
    return block, useful


def tx(chain, key, kind, **fields):
    """Signed transaction with the sender's next nonce and the minimum fee."""
    nonce = fields.pop("nonce", chain.state.nonce(key.address))
    fee = fields.pop("fee", txmod.min_fee({"type": kind, **fields}))
    return make_tx(key, chain.chain_id, kind, nonce, fee, **fields)


def post_job(chain, buyer, rng, dims=(1, 2, 1), price=1000):
    a = rng.integers(-128, 128, size=(dims[0] * N, dims[1] * N), dtype=np.int8)
    b = rng.integers(-128, 128, size=(dims[1] * N, dims[2] * N), dtype=np.int8)
    job_id = chain.blobs.put(txmod.encode_job(a, b))
    add(chain, buyer, [tx(chain, buyer, "job_post", job=job_id, dims=list(dims), price=price)])
    return a, b, job_id


def unit_result(a, b, unit):
    ua, ub = txmod.unit_inputs(a, b, unit)
    return ua.astype(np.int64) @ ub.astype(np.int64)


class Sale:
    """What a miner knows about a batch it sold: units (in index order), result hashes, root, manifest."""

    def __init__(self, units, leaves, root, manifest):
        self.units, self.leaves, self.root, self.manifest = units, leaves, root, manifest


SALES: dict[str, Sale] = {}       # batch id -> Sale, so tests can answer disputes like a miner


def commit(chain, miner, job_id, units, results, publish=True, publish_manifest=True, **kw):
    """job_commit for a batch. Returns (tx, Sale); the tx's commitment is the batch id.

    With publish=False the result blobs never reach the node, and with
    publish_manifest=False neither does the list of their hashes, like a miner
    that withholds its data.
    """
    dims = chain.state.jobs[job_id]["dims"]
    order = sorted(range(len(units)), key=lambda i: txmod.unit_index(units[i], dims))
    units = [list(units[i]) for i in order]
    blobs = [txmod.encode_result(results[i]) for i in order]
    leaves = [chain.blobs.put(x) if publish else hhex(x) for x in blobs]
    root = merkle_root(leaves)
    m = txmod.encode_manifest(leaves)
    manifest = chain.blobs.put(m) if publish_manifest else hhex(m)
    bitmap = txmod.units_bitmap(units, dims)
    cm = txmod.commitment(miner.address, job_id, bitmap, root)
    SALES[cm] = sale = Sale(units, leaves, root, manifest)
    return tx(chain, miner, "job_commit", job=job_id, units=bitmap, commitment=cm, **kw), sale


def reveal(chain, miner, job_id, batch, sale=None, **kw):
    sale = sale or SALES[batch]
    return tx(chain, miner, "job_reveal", job=job_id, batch=batch, root=sale.root, manifest=sale.manifest, **kw)


def challenge(chain, key, job_id, batch, units, **kw):
    dims = chain.state.jobs[job_id]["dims"]
    return tx(chain, key, "job_challenge", job=job_id, batch=batch, units=txmod.units_bitmap(units, dims), **kw)


def respond(chain, key, job_id, batch, unit, **kw):
    """Answer a dispute with the unit's result hash and Merkle proof, as its miner would."""
    sale = SALES[batch]
    i = sale.units.index(list(unit))
    return tx(chain, key, "job_respond", job=job_id, batch=batch, unit=list(unit), result=sale.leaves[i],
              proof=merkle_proof(sale.leaves, i), **kw)


def sell(chain, miner, job_id, units, results, settle=True, **commit_kw):
    """Commit in one block, reveal in the next, optionally wait out the
    challenge window. Returns the batch id."""
    commit_tx, sale = commit(chain, miner, job_id, units, results, **commit_kw)
    add(chain, miner, [commit_tx])
    bid = commit_tx["commitment"]
    add(chain, miner, [reveal(chain, miner, job_id, bid, sale)])
    if settle:
        settle_batch(chain, miner, job_id, bid)
    return bid


def settle_batch(chain, key, job_id, bid):
    while bid in chain.state.jobs[job_id]["batches"]:
        add(chain, key)


def supply(chain) -> int:
    """Every coin that exists must be in a balance, an escrow or a bond."""
    s = chain.state
    return sum(s.balances.values()) + s.locked_total()


def minted(chain) -> int:
    return sum(params.block_reward(h) for h in range(chain.height + 1))
