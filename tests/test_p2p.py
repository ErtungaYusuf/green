"""The peer-to-peer protocol: encryption, identity, relay, headers-first sync,
and what it does to peers that misbehave."""

import json
import random
import socket
import struct
import threading
import time

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from green import p2p, params
from green.chain import block_hash, make_genesis
from green.crypto import Key
from green.node import Node, http_post
from green.tx import make_tx
from helpers import T0, add, tx

QUIET = lambda m: None  # noqa: E731


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(cond, timeout=30.0, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def start_node(path, genesis, key=None, peers=(), mine=False, **kw):
    n = Node(str(path), genesis, port=free_port(), miner_key=key, log=QUIET, peers=peers, **kw)
    n.start(mine=mine)
    return n


def connected(a: Node, b: Node) -> bool:
    return b.p2p.node_id in {p.node_id for p in a.p2p.all_peers()}


@pytest.fixture
def nodes(tmp_path):
    started = []

    def make(name, genesis, **kw):
        n = start_node(tmp_path / name, genesis, **kw)
        started.append(n)
        return n
    yield make
    for n in started:
        n.stop()


class RawPeer:
    """A hand-driven peer: does the real handshake, then sends whatever a test wants."""

    def __init__(self, node: Node, work: int = 0, genesis_hash: str | None = None, port: int = 0,
                 features: list | None = None, key: Ed25519PrivateKey | None = None):
        self.sock = socket.create_connection(("127.0.0.1", node.p2p.port), timeout=10)
        self.send_c, self.recv_c, self.sid = p2p.key_exchange(self.sock, True, genesis_hash or node.chain.genesis_hash)
        self.key = key or Ed25519PrivateKey.generate()
        self.node_id = p2p.pub_hex(self.key)
        self.send("version", {"proto": p2p.PROTOCOL_VERSION, "agent": "raw", "id": self.node_id,
                              "sig": self.key.sign(p2p.auth_message(self.sid, True)).hex(), "port": port,
                              "relay": True, "height": 0, "tip": node.chain.genesis_hash, "work": str(work),
                              "you": "127.0.0.1", "time": int(time.time()),
                              **({"features": features} if features is not None else {})})
        cmd, self.version = self.recv()
        assert cmd == "version"

    def send(self, cmd, payload=None):
        self.sock.sendall(self.send_c.seal(p2p.encode(cmd, {} if payload is None else payload)))

    def recv(self, timeout=10):
        deadline = time.time() + timeout
        length = self.recv_c.open_length(p2p._recv_exact(self.sock, 20, deadline))
        body = self.recv_c.open_body(p2p._recv_exact(self.sock, length + 16, deadline))
        cmd = p2p.NAME[body[0]]
        return cmd, (body[1:] if cmd in p2p.BINARY else json.loads(body[1:]))

    def recv_until(self, want, timeout=10):
        deadline = time.time() + timeout
        while True:
            cmd, m = self.recv(max(0.1, deadline - time.time()))
            if cmd == want:
                return m

    def closed(self, timeout=10) -> bool:
        """True once the node hangs up on us."""
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                self.recv(max(0.1, deadline - time.time()))
        except TimeoutError:                             # (a subclass of OSError) still open
            return False
        except (ConnectionError, OSError, InvalidTag, struct.error):
            return True
        return False


# --- the transport ------------------------------------------------------------------

def test_frames_round_trip_and_rekey_in_step(monkeypatch):
    monkeypatch.setattr(p2p, "REKEY_FRAMES", 5)
    a, b = socket.socketpair()
    out = {}
    t = threading.Thread(target=lambda: out.update(r=p2p.key_exchange(b, False, "11" * 32)))
    t.start()
    send, _, sid = p2p.key_exchange(a, True, "11" * 32)
    t.join()
    _, recv, sid_b = out["r"]
    assert sid == sid_b
    for i in range(23):                                  # across four rekeys
        frame = send.seal(b"message %d" % i)
        assert recv.open_length(frame[:20]) == len(b"message %d" % i)
        assert recv.open_body(frame[20:]) == b"message %d" % i
    frame = bytearray(send.seal(b"tampered"))
    frame[25] ^= 1
    recv.open_length(bytes(frame[:20]))
    with pytest.raises(InvalidTag):
        recv.open_body(bytes(frame[20:]))
    a.close()
    b.close()


def test_another_network_cannot_read_the_first_frame():
    a, b = socket.socketpair()
    out = {}
    t = threading.Thread(target=lambda: out.update(r=p2p.key_exchange(b, False, "22" * 32)))
    t.start()
    send, _, _ = p2p.key_exchange(a, True, "11" * 32)
    t.join()
    _, recv, _ = out["r"]
    with pytest.raises(InvalidTag):
        recv.open_length(send.seal(b"hello")[:20])
    a.close()
    b.close()


class Tap:
    """A TCP relay between two nodes that records (and can corrupt) the bytes."""

    def __init__(self, target_port: int, flip_after: int | None = None):
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(4)
        self.port = self.server.getsockname()[1]
        self.target = target_port
        self.flip_after = flip_after
        self.seen = bytearray()
        self.connections = 0
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                c, _ = self.server.accept()
            except OSError:
                return
            self.connections += 1
            try:
                u = socket.create_connection(("127.0.0.1", self.target))
            except OSError:                              # the node behind us is gone (test teardown)
                c.close()
                continue
            threading.Thread(target=self._pipe, args=(c, u, False), daemon=True).start()
            threading.Thread(target=self._pipe, args=(u, c, True), daemon=True).start()

    def _pipe(self, src, dst, downstream):
        sent = 0
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                self.seen += data
                if downstream and self.flip_after is not None and sent <= self.flip_after < sent + len(data):
                    data = bytearray(data)
                    data[self.flip_after - sent] ^= 0x40
                    data = bytes(data)
                    self.flip_after = None              # corrupt one byte, once
                sent += len(data)
                dst.sendall(data)
        except OSError:
            pass
        for s in (src, dst):
            try:
                s.close()
            except OSError:
                pass

    def close(self):
        self.server.close()


def test_nothing_on_the_wire_is_readable(nodes):
    miner, payer = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "wire")
    a = nodes("a", genesis, key=miner, mine=True)
    tap = Tap(a.p2p.port)
    b = nodes("b", genesis, peers=[f"127.0.0.1:{tap.port}"])
    wait_for(lambda: b.chain.height >= 3, what="sync through the tap")
    pay = make_tx(miner, a.chain.chain_id, "transfer", a.pending_nonce(miner.address), to=payer.address, amount=5)
    http_post(f"{a.url}/tx", pay)
    wait_for(lambda: payer.address in b.chain.state.balances or pay["sig"] in str(b.mempool), what="tx relay")
    wire = bytes(tap.seen)
    for secret in (b"transfer", payer.address.encode(), miner.address.encode(), a.chain.genesis_hash.encode(),
                   b'"header"', b"version", pay["sig"].encode()):
        assert secret not in wire, secret
    tap.close()


def test_a_corrupted_byte_drops_the_connection_and_it_recovers(nodes):
    miner = Key.generate()
    genesis = make_genesis(miner.address, "corrupt")
    a = nodes("a", genesis, key=miner, mine=True)
    tap = Tap(a.p2p.port, flip_after=2000)               # past the handshake, inside some frame
    b = nodes("b", genesis, peers=[f"127.0.0.1:{tap.port}"])
    wait_for(lambda: tap.connections >= 2, what="the corrupted link to be dropped and redialled")
    wait_for(lambda: b.chain.height >= a.chain.height - 1 and b.chain.height >= 3, what="sync after the redial")
    assert b.p2p.stats["banned"] == 0                    # corruption is not proof of malice
    tap.close()


def test_a_pinned_peer_must_present_its_key(nodes):
    miner = Key.generate()
    genesis = make_genesis(miner.address, "pin")
    a = nodes("a", genesis)
    wrong = "ab" * 32
    # (b and c do not listen, so the only possible link is their own pinned dial)
    b = nodes("b", genesis, peers=[f"{wrong}@{a.p2p.address}"], listen=False)
    time.sleep(1.5)
    assert not connected(b, a)                           # a man in the middle would look like this
    c = nodes("c", genesis, peers=[a.p2p.full_address], listen=False)
    wait_for(lambda: connected(c, a), what="pinned connection")


def test_nodes_of_another_network_never_connect(nodes):
    a = nodes("a", make_genesis(Key.generate().address, "net A"))
    b = nodes("b", make_genesis(Key.generate().address, "net B"), peers=[a.p2p.address])
    time.sleep(1.5)
    assert a.p2p.all_peers() == [] and b.p2p.all_peers() == []


def test_a_node_does_not_connect_to_itself(nodes):
    a = nodes("a", make_genesis(Key.generate().address, "self"))
    a.p2p.add_manual(f"localhost:{a.p2p.port}")
    time.sleep(1.5)
    assert a.p2p.all_peers() == []


# --- relay ----------------------------------------------------------------------------

def test_transactions_and_blocks_cross_each_link_once(nodes, monkeypatch):
    """A line of three nodes: a transaction entered at one end reaches the other,
    and every node receives its body exactly once (announce, then fetch)."""
    monkeypatch.setattr(p2p, "MAX_OUTBOUND", 1)          # keep the line a line
    monkeypatch.setattr(p2p, "TRICKLE_MEAN_SECONDS", 0.1)
    received: dict[str, int] = {}
    on_tx = p2p.P2P._on_tx

    def counting(self, peer, m):
        received[self.node_id] = received.get(self.node_id, 0) + 1
        return on_tx(self, peer, m)
    monkeypatch.setattr(p2p.P2P, "_on_tx", counting)
    miner, payee = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "line")
    a = nodes("a", genesis, key=miner, mine=True)
    b = nodes("b", genesis, peers=[a.p2p.address])
    # c does not listen: otherwise its self-advertisement can reach a through b, and
    # a (with a free outbound slot) may dial c directly, which turns the line into a triangle.
    c = nodes("c", genesis, peers=[b.p2p.address], listen=False)
    wait_for(lambda: c.chain.height >= 3, what="blocks to reach the far end")
    a.mining = False
    wait_for(lambda: c.chain.tip == a.chain.tip, what="same tip")
    assert not connected(c, a)
    pay = make_tx(miner, a.chain.chain_id, "transfer", a.pending_nonce(miner.address), to=payee.address, amount=7)
    http_post(f"{a.url}/tx", pay)
    wait_for(lambda: len(c.mempool) == 1, what="the transaction to reach the far end")
    time.sleep(1.0)                                      # any duplicate would arrive by now
    assert received == {b.p2p.node_id: 1, c.p2p.node_id: 1}
    a.mining = True
    wait_for(lambda: c.chain.state.balance(payee.address) == 7, what="confirmation everywhere")


def test_a_silent_peer_is_skipped_for_another_announcer(nodes, monkeypatch):
    monkeypatch.setattr(p2p, "TX_REQUEST_SECONDS", 1)
    a = nodes("a", make_genesis(Key.generate().address, "silent"))
    tid = "cd" * 32
    first, second = RawPeer(a), RawPeer(a)
    first.send("inv", {"items": [["tx", tid]]})
    assert first.recv_until("getdata")["items"] == [["tx", tid]]
    second.send("inv", {"items": [["tx", tid]]})
    started = time.time()
    assert second.recv_until("getdata", timeout=5)["items"] == [["tx", tid]]   # first never answered
    assert time.time() - started >= 0.5


def test_addresses_spread_and_nodes_find_each_other(nodes):
    genesis = make_genesis(Key.generate().address, "gossip")
    a = nodes("a", genesis)
    b = nodes("b", genesis, peers=[a.p2p.address])
    wait_for(lambda: connected(a, b), what="b to reach a")
    c = nodes("c", genesis, peers=[a.p2p.address])       # c is only told about a
    wait_for(lambda: connected(c, b), timeout=30, what="c to learn about b from a and connect")


def test_two_requests_for_one_blob_share_the_answer(nodes, monkeypatch):
    """A buyer's watchtower and its API client can ask the node's only peer for the
    same result at once: the second request waits for the first one's answer instead
    of giving up on the peer (which made the client see a 404)."""
    genesis = make_genesis(Key.generate().address, "blobs")
    a = nodes("a", genesis)
    b = nodes("b", genesis, peers=[a.p2p.address])
    wait_for(lambda: connected(b, a), what="b to reach a")
    data = random.randbytes(100_000)
    h = a.blobs.put(data)
    monkeypatch.setattr(a, "is_public_blob", lambda blob: blob == h)
    on_getblob = p2p.P2P._on_getblob

    def slow(self, peer, m):                             # the answer takes a while
        time.sleep(0.5)
        return on_getblob(self, peer, m)
    monkeypatch.setattr(p2p.P2P, "_on_getblob", slow)
    got = []
    threads = [threading.Thread(target=lambda: got.append(b.p2p.fetch_blob(h, 1 << 20))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert got == [data, data]


def test_a_refused_relay_does_not_lock_up_the_connection(nodes, monkeypatch):
    """Refusing a sealed request it is too busy to carry, a node answered while holding the
    peer's lock, and sending takes that lock: the connection's reader locked itself up for
    good (the node stopped hearing that peer, and could not shut down)."""
    monkeypatch.setattr(p2p, "RELAY_PER_PEER", 0)            # every request is refused as busy
    a = nodes("a", make_genesis(Key.generate().address, "relay busy"))
    peer = RawPeer(a, features=["relay"])
    rid = b"\x01" * 8
    to = Key.generate().address.encode()
    peer.send("relay", rid + bytes([len(to)]) + to + b"\x00" * 100)
    answer = peer.recv_until("relayed", timeout=5)
    assert answer[:8] == rid and answer[8] == 1 and b"busy" in answer[9:]
    peer.send("ping", {"n": 7})
    assert peer.recv_until("pong", timeout=5) == {"n": 7}      # the connection still works


# --- headers-first sync ------------------------------------------------------------------

def test_headers_first_sync_from_several_peers(nodes, tmp_path, monkeypatch):
    """A new node fetches headers in rounds, samples their work, downloads blocks
    from two peers at once and checks their work units in parallel."""
    monkeypatch.setattr(p2p, "MAX_HEADERS", 16)
    monkeypatch.setattr(p2p, "BLOCKS_PER_PEER", 4)
    keys = [Key.generate() for _ in range(2)]
    genesis = make_genesis(keys[0].address, "ibd", timestamp=T0)
    served: dict[str, int] = {}
    on_getdata = p2p.P2P._on_getdata

    def counting(self, peer, m):
        served[self.node_id] = served.get(self.node_id, 0) + sum(1 for k, _ in m["items"] if k == "block")
        return on_getdata(self, peer, m)
    monkeypatch.setattr(p2p.P2P, "_on_getdata", counting)
    sources = []
    for name in ("s1", "s2"):
        n = Node(str(tmp_path / name), genesis, port=free_port(), log=QUIET)
        sources.append(n)
    for i in range(60):
        add(sources[0].chain, keys[0], [tx(sources[0].chain, keys[0], "transfer", to=keys[1].address, amount=1)])
    for bh in sources[0].chain.main[1:]:
        assert sources[1].chain.add_block(sources[0].chain.block(bh)) == "added"
    for n in sources:
        n.start()
    try:
        fresh = nodes("fresh", genesis, peers=[sources[0].p2p.address, sources[1].p2p.address])
        wait_for(lambda: fresh.chain.tip == sources[0].chain.tip, timeout=60, what="initial sync")
        st = fresh.p2p.stats
        assert st["headers_received"] >= 60 and st["work_samples"] >= 4
        assert st["prechecked"] + st["checked_in_processes"] >= 30   # most work units were checked in parallel
        assert len([v for v in served.values() if v > 0]) == 2, served   # both peers served blocks
        assert fresh.chain.state.root() == sources[0].chain.state.root()
    finally:
        for n in sources:
            n.stop()


def fake_headers(node: Node, count: int) -> list[dict]:
    """Headers that pass every cheap rule but whose work was never done."""
    chain = node.chain
    out, prev = [], chain.genesis["header"]
    lookup = {chain.genesis_hash: prev}
    for i in range(count):
        hdr = dict(prev, height=prev["height"] + 1, prev=block_hash({"header": prev}), time=prev["time"] + 15,
                   nonce=i, msg="", work={"job": None, "unit": None})
        hdr["target"] = f"{chain.expected_target(hdr['prev'], lambda h: lookup[h]):064x}"
        hdr["transcript"] = "00" * 32                    # tests' target accepts any ticket
        lookup[block_hash({"header": hdr})] = hdr
        out.append(hdr)
        prev = hdr
    return out


def test_fake_headers_are_caught_before_any_block_is_fetched(nodes):
    a = nodes("a", make_genesis(Key.generate().address, "fake", timestamp=T0))
    liar = RawPeer(a, work=10**40)
    liar.recv_until("getheaders")
    liar.send("headers", {"headers": fake_headers(a, 50)})
    assert liar.closed()
    assert a.p2p.stats["banned"] == 1 and a.p2p.hindex == {} and a.p2p.requested == {}
    assert a.p2p.is_banned("127.0.0.1", liar.node_id)


def test_headers_from_the_future_are_declined_without_a_ban(nodes):
    a = nodes("a", make_genesis(Key.generate().address, "future"))   # genesis is now: 50 x 15 s ahead
    liar = RawPeer(a, work=10**40)
    liar.recv_until("getheaders")
    liar.send("headers", {"headers": fake_headers(a, 50)})
    time.sleep(1)
    assert not liar.closed(timeout=1) and a.p2p.hindex == {} and a.p2p.stats["banned"] == 0


def test_headers_that_break_a_rule_get_the_peer_banned(nodes):
    a = nodes("a", make_genesis(Key.generate().address, "bad headers", timestamp=T0))
    liar = RawPeer(a, work=10**40)
    liar.recv_until("getheaders")
    hdrs = fake_headers(a, 3)
    hdrs[0]["target"] = f"{params.MAX_TARGET:064x}"       # not the ASERT target
    liar.send("headers", {"headers": hdrs})
    assert liar.closed()


# --- defence ------------------------------------------------------------------------------

def test_an_oversized_frame_is_refused_unread(nodes):
    a = nodes("a", make_genesis(Key.generate().address, "big"))
    peer = RawPeer(a)
    sealed_len = peer.send_c.len_aead.encrypt(peer.send_c._nonce(), struct.pack("<I", p2p.MAX_FRAME + 1), None)
    peer.sock.sendall(sealed_len)                        # announce 4 MiB + 1 and send nothing more
    assert peer.closed(timeout=5)
    assert a.p2p.is_banned("127.0.0.1", peer.node_id)


def test_data_nobody_asked_for_gets_the_peer_banned(nodes):
    a = nodes("a", make_genesis(Key.generate().address, "blob"))
    peer = RawPeer(a)
    peer.send("blob", b"x" * 1000)
    assert peer.closed(timeout=5)


def test_malformed_messages_add_up_to_a_ban(nodes):
    a = nodes("a", make_genesis(Key.generate().address, "malformed"))
    peer = RawPeer(a)
    for _ in range(6):
        peer.send("inv", {"items": [["tx", "not a hash"]]})
    assert peer.closed(timeout=5)


def test_a_flood_beyond_the_queue_is_cut_off(nodes, monkeypatch):
    """Disconnected at once, banned the third time within an hour: banning at once hit
    honest peers whenever the node itself fell behind."""
    monkeypatch.setattr(p2p, "MAX_PEER_QUEUE", 5)
    monkeypatch.setattr(p2p, "WORKERS", 0)               # nothing drains the queue
    a = nodes("a", make_genesis(Key.generate().address, "flood"))
    key = Ed25519PrivateKey.generate()
    for time_ in range(3):
        peer = RawPeer(a, key=key)
        try:
            for _ in range(10):
                peer.send("getaddr")
        except OSError:
            pass                                         # cut off before the flood was even sent
        assert peer.closed(timeout=5)
        banned = a.p2p.is_banned("127.0.0.1", peer.node_id)
        assert banned == (time_ == 2), time_


def test_a_silent_connection_is_closed_after_the_handshake_deadline(nodes, monkeypatch):
    monkeypatch.setattr(p2p, "HANDSHAKE_SECONDS", 1)
    a = nodes("a", make_genesis(Key.generate().address, "slow"))
    s = socket.create_connection(("127.0.0.1", a.p2p.port), timeout=5)
    started = time.time()
    try:
        while s.recv(100):
            pass
    except OSError:
        pass
    assert time.time() - started < 4
    assert a.p2p.all_peers() == []


def test_bans_survive_a_restart(tmp_path):
    genesis = make_genesis(Key.generate().address, "banlist")
    a = start_node(tmp_path / "a", genesis)
    peer = RawPeer(a)
    peer.send("blob", b"unasked")
    assert peer.closed(timeout=5)
    a.stop()
    again = Node(str(tmp_path / "a"), genesis, port=free_port(), log=QUIET)
    assert again.p2p.is_banned("127.0.0.1", peer.node_id)
    again.close()


# --- eclipse resistance (pure rules) -------------------------------------------------------

def test_network_groups_and_local_addresses():
    assert p2p.netgroup("8.8.1.1") == p2p.netgroup("8.8.200.3") != p2p.netgroup("8.9.1.1")
    assert p2p.netgroup("2001:db8:1::1") == p2p.netgroup("2001:db8:ffff::2") != p2p.netgroup("2001:db9::1")
    assert p2p.netgroup("::ffff:8.8.1.1") == p2p.netgroup("8.8.1.1")
    assert p2p.is_local("127.0.0.1") and p2p.is_local("192.168.1.5") and p2p.is_local("::1")
    assert not p2p.is_local("8.8.8.8")


def test_one_source_cannot_fill_the_address_book():
    book = p2p.AddrMan(None)
    added = sum(book.add(f"9.{i // 250}.{i % 250}.1", 8546, source="66.1.2.3") for i in range(500))
    assert added == p2p.ADDR_PER_SOURCE
    assert book.add("10.20.30.40", 8546, source="77.1.2.3")         # another source still gets in


class FakePeer:
    def __init__(self, host, ping=None, last_block=0.0, last_tx=0.0, age=0.0):
        self.host, self.ping_ms, self.last_block, self.last_tx = host, ping, last_block, last_tx
        self.connected_at = 1000.0 - age


def test_eviction_protects_good_peers_and_hits_the_crowded_group():
    honest = [FakePeer(f"{i}.1.1.1", ping=10 + i, last_block=100 + i, last_tx=50 + i, age=500 + i) for i in range(16)]
    sybils = [FakePeer(f"66.6.{i}.1", age=i) for i in range(16)]
    victim = p2p.P2P._eviction_candidate(honest + sybils)
    assert victim in sybils and victim.connected_at == max(p.connected_at for p in sybils)   # youngest sybil
    assert p2p.P2P._eviction_candidate(honest[:10]) is None           # all protected


def test_a_peer_that_stalls_block_requests_is_penalised(nodes, monkeypatch):
    monkeypatch.setattr(p2p, "BLOCK_REQUEST_SECONDS", 1)
    a = nodes("a", make_genesis(Key.generate().address, "stall"))
    staller = RawPeer(a)
    staller.send("inv", {"items": [["block", "ef" * 32]]})
    assert staller.recv_until("getdata")["items"] == [["block", "ef" * 32]]   # and never answers
    peer = next(p for p in a.p2p.all_peers() if p.node_id == staller.node_id)
    wait_for(lambda: peer.score >= 10, timeout=10, what="the stall to be noticed")
    assert "ef" * 32 not in a.p2p.requested                                  # free for another peer


def test_stale_fork_headers_are_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "MAX_REORG_DEPTH", 3)
    key = Key.generate()
    n = Node(str(tmp_path / "n"), make_genesis(key.address, "prune", timestamp=T0), port=free_port(), log=QUIET)
    for _ in range(10):
        add(n.chain, key)
    old = dict(n.chain.header(n.chain.main[2]), nonce=999)        # a fork header far below the tip
    recent = dict(n.chain.tip_header(), nonce=7)                  # one that could still win
    n.p2p.hindex.update({"f" * 64: {"header": old, "work": 1}, "e" * 64: {"header": recent, "work": 1}})
    with n.lock:
        n.p2p._prune_index()
    assert "f" * 64 not in n.p2p.hindex and "e" * 64 in n.p2p.hindex
    n.close()


def test_unfinished_handshakes_are_capped(nodes, monkeypatch):
    monkeypatch.setattr(p2p, "MAX_PENDING_HANDSHAKES", 3)
    a = nodes("a", make_genesis(Key.generate().address, "pending"))
    silent = [socket.create_connection(("127.0.0.1", a.p2p.port), timeout=5) for _ in range(3)]
    time.sleep(0.3)
    extra = socket.create_connection(("127.0.0.1", a.p2p.port), timeout=5)
    started = time.time()
    try:
        assert extra.recv(100) == b""                    # closed at once, not after the handshake timeout
    except OSError:
        pass
    assert time.time() - started < 2
    for s in silent + [extra]:
        s.close()
    wait_for(lambda: a.p2p.pending == {}, timeout=15, what="the silent handshakes to time out")
    RawPeer(a)                                           # and a real peer gets in afterwards


def test_only_an_authenticated_newcomer_can_evict(nodes, monkeypatch):
    monkeypatch.setattr(p2p, "MAX_INBOUND", 2)
    monkeypatch.setattr(p2p.P2P, "_eviction_candidate", staticmethod(lambda inbound: inbound[0]))
    a = nodes("a", make_genesis(Key.generate().address, "evict"))
    first, second = RawPeer(a), RawPeer(a)
    wait_for(lambda: len(a.p2p.all_peers()) == 2, what="two inbound peers")
    silent = socket.create_connection(("127.0.0.1", a.p2p.port), timeout=5)   # never handshakes
    time.sleep(1)
    assert len(a.p2p.all_peers()) == 2 and not first.closed(timeout=0.5)
    third = RawPeer(a)                                   # proves its key: now someone makes room
    assert first.closed(timeout=5)
    # (the socket closes a moment before the peer leaves the node's list)
    wait_for(lambda: {p.node_id for p in a.p2p.all_peers()} == {second.node_id, third.node_id},
             timeout=5, what="the evicted peer to leave")
    silent.close()


def test_gossip_cannot_point_us_into_our_own_network():
    """Addresses from the internet must be public IPs: a private one would have every
    node dial into its own LAN, a name would make it look up whatever a stranger chose."""
    assert p2p.gossip_ok("8.8.8.8", "203.0.114.9")
    for host in ("192.168.1.1", "10.0.0.5", "127.0.0.1", "100.64.0.9", "169.254.1.1", "evil.example"):
        assert not p2p.gossip_ok(host, "8.8.4.4"), host
    assert p2p.gossip_ok("192.168.1.20", "192.168.1.7")         # a LAN peer may name LAN nodes
    assert not p2p.gossip_ok("evil.example", "192.168.1.7")


def test_a_dns_seed_gives_many_addresses():
    from green.p2p import MAX_SEED_ADDRESSES, seed_hosts
    assert seed_hosts("203.0.113.5", 8546) == ["203.0.113.5"]          # an IP is used as it is
    answers = [(2, 1, 6, "", (f"198.51.100.{i % 40}", 8546)) for i in range(60)]
    got = seed_hosts("seed.example", 8546, resolve=lambda *a, **k: answers)
    assert len(got) == MAX_SEED_ADDRESSES and len(set(got)) == len(got)

    def fails(*a, **k):
        raise OSError("no such name")
    assert seed_hosts("seed.example", 8546, resolve=fails) == []


def test_a_node_that_falls_behind_drops_shares_instead_of_blaming_peers():
    """Busy nodes counted honest peers' shares as flooding and banned them: the network split."""
    import threading
    from types import SimpleNamespace
    from green.p2p import MAX_PEER_QUEUE, P2P
    blamed = []
    me = SimpleNamespace(stats={}, _flooded=blamed.append, _ready=None)
    peer = SimpleNamespace(lock=threading.Lock(), in_q=[("x", b"")] * MAX_PEER_QUEUE, scheduled=True)
    for cmd in ("share", "tx", "dtx", "offers"):
        P2P._receive(me, peer, cmd, b"{}")
    assert blamed == [] and me.stats["dropped_when_busy"] == 4
    P2P._receive(me, peer, "headers", b"{}")              # what cannot be dropped still counts
    assert blamed == [peer]


def test_a_relayed_bad_signature_counts_against_the_peer_but_policy_does_not():
    """Cheap fees must not make junk free to send: a bad signature is proof the peer did not
    check what it relays; a refusal that depends on our state or policy is not."""
    import threading
    from types import SimpleNamespace
    import pytest
    from green import tx as txmod
    from green.p2p import Misbehaving, P2P
    key = Key.generate()
    good = txmod.make_tx(key, "ab" * 32, "transfer", 0, to=Key.generate().address, amount=10**6)
    forged = dict(good, amount=10**8)                      # the signature no longer matches
    node = SimpleNamespace(lock=threading.Lock())
    me = SimpleNamespace(node=node, tx_wait={}, rejected=set())

    def submit(t, source=None, stem=False):
        txmod.check_format(t, "ab" * 32)
        raise txmod.InvalidTx("insufficient balance")      # a state-dependent refusal
    node.submit_tx = submit
    peer = SimpleNamespace(known=set(), last_tx=0)
    P2P._on_tx(me, peer, {"tx": good})                     # no penalty
    with pytest.raises(Misbehaving):
        P2P._on_tx(me, peer, {"tx": forged})
    assert txmod.txid(forged) in me.rejected


def test_a_node_past_its_upload_budget_stops_serving(tmp_path):
    """A rented seed must not run up a traffic bill: past its monthly budget it serves
    nothing heavy and takes no new inbound peers, and the count survives a restart."""
    key = Key.generate()
    g = make_genesis(key.address, "budget")
    a = start_node(tmp_path / "a", g, key=key, mine=True, upload_cap_gb=0.000002)     # 2 kB
    try:
        wait_for(lambda: a.chain.height >= 3, what="blocks")
        a.p2p.count_upload(5000)
        assert a.p2p.over_upload_budget()
        b = start_node(tmp_path / "b", g, peers=[a.p2p.address])
        try:
            time.sleep(3)
            assert b.chain.height == 0 and not b.p2p.all_peers()       # refused, nothing served
        finally:
            b.stop()
    finally:
        a.stop()
    again = start_node(tmp_path / "a", g, upload_cap_gb=0.000002)
    try:
        assert again.p2p.over_upload_budget()                          # this month's count was kept
    finally:
        again.stop()
