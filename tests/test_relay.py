"""Transaction origin privacy (Dandelion++ stems) and compact block relay."""

import json
import socket
import threading
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from green import p2p, tx as txmod
from green.chain import block_hash, make_genesis
from green.crypto import Key, canonical
from green.node import http_post
from green.tx import make_tx
from helpers import grind
from test_p2p import RawPeer, connected, nodes, wait_for  # noqa: F401  (nodes is a fixture)


class Listener:
    """A peer the node dials: it does the answering side of the handshake, records
    every message, and passes nothing on (a black hole, unless a test sends)."""

    def __init__(self, genesis_hash: str, features=("dtx", "cmpct")):
        self.genesis_hash = genesis_hash
        self.features = list(features)
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(4)
        self.port = self.server.getsockname()[1]
        self.got: list[tuple] = []
        self.ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self.sock, _ = self.server.accept()
            self.send_c, self.recv_c, sid = p2p.key_exchange(self.sock, False, self.genesis_hash)
            key = Ed25519PrivateKey.generate()
            self.send("version", {"proto": p2p.PROTOCOL_VERSION, "agent": "listener", "id": p2p.pub_hex(key),
                                  "sig": key.sign(p2p.auth_message(sid, False)).hex(), "port": 0, "relay": True,
                                  "height": 0, "tip": self.genesis_hash, "work": "0", "you": "127.0.0.1",
                                  "time": int(time.time()), "features": self.features})
            self.ready.set()
            while True:
                length = self.recv_c.open_length(p2p._recv_exact(self.sock, 20, time.time() + 600))
                body = self.recv_c.open_body(p2p._recv_exact(self.sock, length + 16, time.time() + 60))
                cmd = p2p.NAME[body[0]]
                self.got.append((cmd, body[1:] if cmd == "blob" else json.loads(body[1:])))
        except Exception:
            pass

    def send(self, cmd, payload=None):
        self.sock.sendall(self.send_c.seal(p2p.encode(cmd, {} if payload is None else payload)))

    def messages(self, cmd):
        return [m for c, m in list(self.got) if c == cmd]

    def close(self):
        self.server.close()


def invs(listener, kind="tx"):
    return {h for m in listener.messages("inv") for k, h in m["items"] if k == kind}


def test_own_transactions_take_a_stem_and_are_not_announced(nodes, monkeypatch):
    monkeypatch.setattr(p2p, "TRICKLE_MEAN_SECONDS", 0.1)
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "stem")
    hole = Listener(block_hash(genesis))
    a = nodes("a", genesis, key=miner, peers=[f"127.0.0.1:{hole.port}"], listen=False)
    wait_for(lambda: hole.ready.is_set() and a.p2p.all_peers(), what="the node to dial its peer")
    pays = []
    for amount in (1, 2):                    # the second one builds on the first, still on its stem
        pay = make_tx(miner, a.chain.chain_id, "transfer", a.pending_nonce(miner.address),
                      to=payee.address, amount=amount)
        http_post(f"{a.url}/tx", pay)
        pays.append(txmod.txid(pay))
    wait_for(lambda: len(hole.messages("dtx")) == 2, what="both to be passed along the stem")
    assert [txmod.txid(m["tx"]) for m in hole.messages("dtx")] == pays
    assert not set(pays) & set(a.mempool) and not set(pays) & invs(hole)    # nobody was told
    assert a.tx_view(pays[0])["relay"] == "stem"
    hole.send("getdata", {"items": [["tx", pays[0]]]})
    wait_for(lambda: hole.messages("notfound"), what="notfound")
    assert hole.messages("tx") == []                                        # stem transactions are not served
    # The next hop swallowed them: when the embargo runs out the node announces them itself.
    with a.lock:                                         # (fast-forward the embargo clock)
        for e in a.p2p.stempool.values():
            e["embargo"] = time.time()
    wait_for(lambda: set(pays) <= set(a.mempool), timeout=15, what="the embargo to fire")
    wait_for(lambda: set(pays) <= invs(hole), what="the announcement")
    assert a.p2p.stats["embargo_fired"] == 2 and not a.p2p.stempool


def test_a_stem_crosses_relays_and_is_announced_where_it_ends(nodes, monkeypatch):
    monkeypatch.setattr(p2p, "DANDELION_FLUFF_PROB", 0.0)     # relays pass it on; the end of the line diffuses
    monkeypatch.setattr(p2p, "MAX_OUTBOUND", 1)
    monkeypatch.setattr(p2p, "TRICKLE_MEAN_SECONDS", 0.1)
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "line")
    c = nodes("c", genesis)
    b = nodes("b", genesis, peers=[c.p2p.address])
    a = nodes("a", genesis, key=miner, peers=[b.p2p.address], listen=False)
    wait_for(lambda: connected(a, b) and connected(b, c), what="the line")
    pay = make_tx(miner, a.chain.chain_id, "transfer", 0, to=payee.address, amount=5)
    tid = txmod.txid(pay)
    http_post(f"{a.url}/tx", pay)
    wait_for(lambda: all(tid in n.mempool for n in (a, b, c)), what="the transaction everywhere")
    assert a.p2p.stats["stem_relayed"] == 1 and b.p2p.stats["stem_relayed"] == 1
    assert c.p2p.stats["stem_fluffed"] == 1 and b.p2p.stats["stem_fluffed"] == 0
    wait_for(lambda: not a.p2p.stempool and not b.p2p.stempool, what="the stem pools to empty")
    assert all(n.p2p.stats["embargo_fired"] == 0 for n in (a, b, c))      # announced, not rescued


def test_a_stem_that_loops_back_ends_there(nodes, monkeypatch):
    monkeypatch.setattr(p2p, "DANDELION_FLUFF_PROB", 0.0)
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "loop")
    hole = Listener(block_hash(genesis))
    n = nodes("n", genesis, peers=[f"127.0.0.1:{hole.port}"])
    wait_for(lambda: hole.ready.is_set() and n.p2p.all_peers(), what="the node to dial its peer")
    raw = RawPeer(n, features=["dtx"])
    pay = make_tx(miner, n.chain.chain_id, "transfer", 0, to=payee.address, amount=3)
    raw.send("dtx", {"tx": pay})
    wait_for(lambda: hole.messages("dtx"), what="the stem to go on")
    assert txmod.txid(pay) not in n.mempool
    raw.send("dtx", {"tx": pay})                         # it came around again
    wait_for(lambda: txmod.txid(pay) in n.mempool, what="the loop to end the stem")
    assert len(hole.messages("dtx")) == 1


def test_compact_blocks_are_rebuilt_from_the_mempool(nodes, monkeypatch):
    monkeypatch.setattr(p2p, "TRICKLE_MEAN_SECONDS", 0.1)
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "compact")
    a = nodes("a", genesis, key=miner)
    b = nodes("b", genesis, peers=[a.p2p.address])
    wait_for(lambda: connected(a, b), what="the link")
    pays = [make_tx(miner, a.chain.chain_id, "transfer", i, to=payee.address, amount=i + 1) for i in range(20)]
    for pay in pays:
        http_post(f"{b.url}/tx", pay)
    wait_for(lambda: len(a.mempool) == 20 and len(b.mempool) == 20, what="both mempools")
    a.mining = True
    wait_for(lambda: b.chain.state.balance(payee.address) == sum(range(1, 21)), what="the block at b")
    a.mining = False
    st = b.p2p.stats
    assert st["cmpct_rebuilt"] >= 1 and st["cmpct_txs_requested"] == 0 and st["cmpct_fallbacks"] == 0
    assert st["blocks_downloaded"] == 0                   # no block was fetched whole
    block = next(a.chain.block(h) for h in a.chain.main[1:] if len(a.chain.block(h)["txs"]) == 20)
    peer = next(p for p in a.p2p.all_peers() if p.node_id == b.p2p.node_id)
    small = len(canonical(a.p2p._compact(block, peer)))
    assert small * 4 < len(canonical(block)), (small, len(canonical(block)))


def test_a_compact_block_asks_only_for_what_is_missing(nodes):
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "missing")
    n = nodes("n", genesis)
    sender, old, new = RawPeer(n, features=["cmpct"]), RawPeer(n), RawPeer(n, features=["cmpct"])
    pay = make_tx(miner, n.chain.chain_id, "transfer", 0, to=payee.address, amount=4)
    with n.lock:
        header, chosen, _ = n.chain.build_template(miner.address, [pay])
    block, _ = grind(n.chain, header, chosen)
    bh, salt = block_hash(block), "ab" * 8
    sid = p2p.short_id(p2p.compact_key(bh, salt), txmod.txid(pay))
    sender.send("cmpct", {"header": block["header"], "salt": salt, "ids": [sid], "fill": []})
    assert sender.recv_until("getblocktxn") == {"block": bh, "idx": [0]}
    sender.send("blocktxn", {"block": bh, "txs": [pay]})
    wait_for(lambda: n.chain.tip == bh, what="the rebuilt block")
    assert n.p2p.stats["cmpct_txs_requested"] == 1 and n.p2p.stats["cmpct_rebuilt"] == 1
    # It is relayed on: compact to a peer that speaks it (with the transaction it
    # never saw sent whole), an inv to one that does not.
    m = new.recv_until("cmpct")
    assert block_hash({"header": m["header"]}) == bh and m["ids"] == [] and m["fill"] == [[0, pay]]
    assert ["block", bh] in old.recv_until("inv")["items"]


def test_a_bad_compact_block_falls_back_to_the_whole_block(nodes):
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "fallback")
    n = nodes("n", genesis)
    raw = RawPeer(n, features=["cmpct"])
    pay = make_tx(miner, n.chain.chain_id, "transfer", 0, to=payee.address, amount=4)
    wrong = make_tx(miner, n.chain.chain_id, "transfer", 0, to=payee.address, amount=5)
    with n.lock:
        header, chosen, _ = n.chain.build_template(miner.address, [pay])
    block, _ = grind(n.chain, header, chosen)
    bh = block_hash(block)
    raw.send("cmpct", {"header": block["header"], "salt": "00" * 8, "ids": [], "fill": [[0, wrong]]})
    assert raw.recv_until("getdata") == {"items": [["block", bh]]}     # not a ban: a collision looks the same
    raw.send("block", {"block": block})
    wait_for(lambda: n.chain.tip == bh, what="the block")
    assert n.p2p.stats["cmpct_fallbacks"] == 1 and raw.node_id in {p.node_id for p in n.p2p.all_peers()}


def test_a_compact_block_without_its_work_is_refused_before_rebuilding(nodes):
    """Rebuilding hashes the whole mempool under the node's lock and holds the block's
    slots in `partial`: a header with a valid parent but no real work used to get all
    of that (and a getblocktxn back) for free. Its work is checked first now."""
    miner = Key.generate()
    n = nodes("n", make_genesis(miner.address, "no work"))
    raw = RawPeer(n, features=["cmpct"])
    with n.lock:
        header, _, _ = n.chain.build_template(miner.address, [])
    header["transcript"] = "00" * 32                       # made up: no work behind it
    raw.send("cmpct", {"header": header, "salt": "00" * 8, "ids": ["00" * p2p.SHORT_ID_BYTES] * 3000, "fill": []})
    assert raw.closed()
    assert n.p2p.partial == {} and n.p2p.stats["cmpct_txs_requested"] == 0
    assert n.p2p.is_banned("127.0.0.1", raw.node_id)


def test_the_nodes_own_protocol_transactions_skip_the_stem(nodes):
    """Commits, reveals and challenges have deadlines and the loops that send them
    look for them in the mempool: they are announced directly."""
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "protocol")
    hole = Listener(block_hash(genesis))
    n = nodes("n", genesis, key=miner, peers=[f"127.0.0.1:{hole.port}"], listen=False)
    wait_for(lambda: hole.ready.is_set() and n.p2p.all_peers(), what="the node to dial its peer")
    assert n._try_tx("transfer", to=payee.address, amount=1)
    assert len(n.mempool) == 1 and not n.p2p.stempool
    wait_for(lambda: invs(hole), what="the announcement")
    assert hole.messages("dtx") == []


def test_a_fee_bump_replaces_a_transaction_still_on_its_stem(nodes):
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "bump")
    hole = Listener(block_hash(genesis))
    a = nodes("a", genesis, key=miner, peers=[f"127.0.0.1:{hole.port}"], listen=False)
    wait_for(lambda: hole.ready.is_set() and a.p2p.all_peers(), what="the node to dial its peer")
    slow = make_tx(miner, a.chain.chain_id, "transfer", 0, to=payee.address, amount=1)
    http_post(f"{a.url}/tx", slow)
    wait_for(lambda: hole.messages("dtx"), what="the stem")
    fast = make_tx(miner, a.chain.chain_id, "transfer", 0, 3 * p2p.params.MIN_TX_FEE, to=payee.address, amount=1)
    http_post(f"{a.url}/tx", fast)                       # the same nonce, paying more
    assert list(a.mempool) == [txmod.txid(fast)] and not a.p2p.stempool


def test_a_transaction_does_not_overtake_its_predecessor_on_a_stem(nodes, monkeypatch):
    """Found by bench/net_sim.py: a sender's first transaction was announced (its node had no
    stem yet), its second took a fresh stem, reached a node that had not seen the first,
    and was dropped there; every later one followed, and all waited ~20 s for the embargo."""
    monkeypatch.setattr(p2p, "TRICKLE_MEAN_SECONDS", 1.0)
    monkeypatch.setattr(p2p, "DANDELION_FLUFF_PROB", 0.0)
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "order")
    b = nodes("b", genesis)
    a = nodes("a", genesis, key=miner, peers=[b.p2p.address], listen=False)
    wait_for(lambda: connected(a, b), what="the link")
    first = make_tx(miner, a.chain.chain_id, "transfer", 0, to=payee.address, amount=1)
    a.submit_tx(first, stem=False)                       # announced (slowly), not stemmed
    second = make_tx(miner, a.chain.chain_id, "transfer", 1, to=payee.address, amount=2)
    http_post(f"{a.url}/tx", second)                     # must not take a stem past the first
    wait_for(lambda: len(b.mempool) == 2, timeout=10, what="both, in order, at b")
    assert a.p2p.stats["stem_relayed"] == 0 and b.p2p.stats["embargo_fired"] == 0


def test_a_future_nonce_is_not_marked_rejected(nodes):
    miner, payee = Key.generate(), Key.generate()
    n = nodes("n", make_genesis(miner.address, "future"))
    raw = RawPeer(n)
    first = make_tx(miner, n.chain.chain_id, "transfer", 0, to=payee.address, amount=1)
    second = make_tx(miner, n.chain.chain_id, "transfer", 1, to=payee.address, amount=2)
    raw.send("tx", {"tx": second})                       # arrives before its predecessor
    raw.send("tx", {"tx": first})
    wait_for(lambda: txmod.txid(first) in n.mempool, what="the first")
    assert txmod.txid(second) not in n.p2p.rejected      # so a later announcement fetches it
    raw.send("tx", {"tx": second})
    wait_for(lambda: txmod.txid(second) in n.mempool, what="the second, once it can be accepted")
