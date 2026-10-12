"""Opening a port on the router: PCP, the NAT-PMP fallback, renewal and removal,
against a fake gateway (nothing is sent to a real router)."""

import ipaddress
import socket
import struct
import threading
import time

import pytest

from green import portmap
from green.chain import make_genesis
from green.crypto import Key
from green.node import Node
from green.portmap import PortMapError, PortMapper

EXTERNAL = "1.2.3.4"           # (public: a private one would mean another NAT outside, and is not advertised)


class FakeGateway:
    """Speaks PCP ("pcp"), only NAT-PMP ("natpmp"), refuses ("refuse") or says nothing ("silent")."""

    def __init__(self, mode: str, external: str = EXTERNAL):
        self.mode, self.external = mode, external
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.requests: list[tuple] = []       # (protocol, internal port, suggested port, lifetime, nonce)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1100)
            except OSError:
                return
            reply = self._answer(data)
            if reply is not None:
                self.sock.sendto(reply, addr)

    def _answer(self, d: bytes):
        if self.mode == "silent":
            return None
        if d[0] == 2:                                           # PCP
            if self.mode == "natpmp":
                return struct.pack("!BBHI", 0, 128 + d[1], 1, 0)  # NAT-PMP: unsupported version
            lifetime = struct.unpack("!I", d[4:8])[0]
            nonce, internal, suggested = d[24:36], *struct.unpack("!HH", d[40:44])
            self.requests.append(("pcp", internal, suggested, lifetime, nonce))
            result = 8 if self.mode == "refuse" else 0            # NOT_AUTHORIZED
            port = suggested or 40000 + internal % 20000
            header = struct.pack("!BBBBII", 2, 0x81, 0, result, lifetime, 1) + bytes(12)
            return header + nonce + struct.pack("!B3xHH", 6, internal, port) + \
                ipaddress.IPv6Address(f"::ffff:{self.external}").packed
        if d[1] == 0:                                           # NAT-PMP: external address
            return struct.pack("!BBHI", 0, 128, 0, 1) + socket.inet_aton(self.external)
        if d[1] == 2:                                           # NAT-PMP: map TCP
            internal, suggested, lifetime = struct.unpack("!HHI", d[4:12])
            self.requests.append(("nat-pmp", internal, suggested, lifetime, None))
            return struct.pack("!BBHIHHI", 0, 130, 0, 1, internal, suggested or 0, lifetime)
        return None

    def close(self):
        self.sock.close()


def test_pcp_maps_renews_with_the_same_nonce_and_removes(monkeypatch):
    gw = FakeGateway("pcp")
    m = PortMapper(8546, gateway="127.0.0.1", gateway_port=gw.port, lifetime=2)
    assert m.start()
    assert (m.protocol, m.external_ip, m.external_port) == ("pcp", EXTERNAL, 48546)
    deadline = time.time() + 5
    while len(gw.requests) < 2:                                 # renewed at half the lifetime
        assert time.time() < deadline, "no renewal"
        time.sleep(0.05)
    first, renewal = gw.requests[:2]
    assert renewal[4] == first[4] and renewal[2] == 48546       # same nonce, asks to keep its port
    m.stop()
    assert gw.requests[-1][3] == 0                              # lifetime 0: removed
    gw.close()


def test_a_nat_pmp_router_is_used_when_pcp_is_not_spoken():
    gw = FakeGateway("natpmp")
    m = PortMapper(8546, gateway="127.0.0.1", gateway_port=gw.port)
    ip, port, life = m.map_once()
    assert (m.protocol, ip, port, life) == ("nat-pmp", EXTERNAL, 8546, portmap.LIFETIME)
    m.running = True
    m.stop()
    assert gw.requests[-1][1:4] == (8546, 0, 0)                 # deletion: external port and lifetime 0
    gw.close()


def test_a_refusal_or_silence_leaves_the_node_unmapped(monkeypatch):
    monkeypatch.setattr(portmap, "TRIES", (0.1,))
    for mode in ("refuse", "silent"):
        gw = FakeGateway(mode)
        m = PortMapper(8546, gateway="127.0.0.1", gateway_port=gw.port)
        if mode == "refuse":
            with pytest.raises(PortMapError, match="refused"):
                m.map_once()                                    # a refusal is final: no NAT-PMP retry
        assert not m.start() and m.external_port is None
        gw.close()
    assert portmap.default_gateway() is None or ipaddress.ip_address(portmap.default_gateway())


@pytest.mark.parametrize("external", [EXTERNAL, "100.64.0.9"])       # 100.64/10: carrier-grade NAT
def test_a_node_advertises_its_mapped_address(tmp_path, external):
    gw = FakeGateway("pcp", external)
    key = Key.generate()
    n = Node(str(tmp_path / "n"), make_genesis(key.address, "nat"), port=_free(), log=lambda m: None,
             portmap=True, portmap_gateway=("127.0.0.1", gw.port))
    n.start()
    try:
        public = n.p2p.status()["public"]
        # Behind a second NAT the router's "external" address is private: nothing to advertise.
        assert public == (f"{EXTERNAL}:{40000 + n.p2p.port % 20000}" if external == EXTERNAL else None)
    finally:
        n.stop()
    assert gw.requests[-1][3] == 0                              # the node removed it on stop
    gw.close()


def _free() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
