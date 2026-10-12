"""Opening a port on the home router: PCP (RFC 6887) with a NAT-PMP (RFC 6886) fallback.

A node behind a home router can dial out but cannot be reached, so it never
serves other nodes, and a miner selling compute from home cannot be found by
buyers. Most routers can forward a port on request: PCP is the current
standard (Bitcoin Core uses it since v29), NAT-PMP its predecessor that many
routers still speak. Both are a single UDP request to the gateway on port 5351;
the mapping lasts a lifetime and is renewed at half of it, and is removed when
the node stops. UPnP is not used: it is a much larger attack surface for the
same result.

Opt in (`--portmap`): it changes the router's state, so it is the operator's call.
"""

import ipaddress
import os
import platform
import re
import socket
import struct
import subprocess
import threading
import time

PORT = 5351
LIFETIME = 3600                 # seconds asked for; renewed at half
TRIES = (0.25, 0.5, 1.0)        # reply timeouts (RFC 6886 starts at 250 ms and doubles)
TCP = 6

PCP_VERSION = 2
PCP_MAP = 1
PCP_UNSUPP_VERSION = 1


class PortMapError(Exception):
    pass


def default_gateway() -> str | None:
    """The IPv4 default gateway, read from the routing table (no packets sent)."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/net/route") as f:
                for line in f.readlines()[1:]:
                    fields = line.split()
                    if fields[1] == "00000000" and int(fields[3], 16) & 2:      # default route, RTF_GATEWAY
                        return socket.inet_ntoa(struct.pack("<I", int(fields[2], 16)))
        elif system == "Windows":
            return _windows_gateway()
        else:                                                                   # macOS, BSDs
            out = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True,
                                 timeout=5).stdout
            m = re.search(r"gateway:\s*([0-9.]+)", out)
            return m.group(1) if m else None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return None


def _windows_gateway() -> str | None:
    import ctypes
    from ctypes import wintypes

    class MIB_IPFORWARDROW(ctypes.Structure):
        _fields_ = [(name, wintypes.DWORD) for name in (
            "dest", "mask", "policy", "next_hop", "if_index", "type", "proto", "age",
            "next_hop_as", "metric1", "metric2", "metric3", "metric4", "metric5")]
    row = MIB_IPFORWARDROW()
    dest = struct.unpack("<I", socket.inet_aton("8.8.8.8"))[0]                 # any public address
    if ctypes.windll.iphlpapi.GetBestRoute(dest, 0, ctypes.byref(row)) != 0:
        return None
    hop = socket.inet_ntoa(struct.pack("<I", row.next_hop))
    return None if hop == "0.0.0.0" else hop


def _mapped(ip: str) -> bytes:
    """An IPv4 address as PCP writes it: IPv4-mapped IPv6 (::ffff:a.b.c.d)."""
    return ipaddress.IPv6Address(f"::ffff:{ip}").packed


def _unmapped(raw: bytes) -> str:
    addr = ipaddress.IPv6Address(raw)
    return str(addr.ipv4_mapped) if addr.ipv4_mapped else str(addr)


class PortMapper:
    """Keeps TCP `internal_port` forwarded on the gateway while running."""

    def __init__(self, internal_port: int, gateway: str | None = None, gateway_port: int = PORT,
                 lifetime: int = LIFETIME, log=lambda m: None):
        self.internal_port = internal_port
        self.gateway = gateway
        self.gateway_port = gateway_port
        self.lifetime = lifetime
        self.log = log
        self.protocol: str | None = None          # "pcp" or "nat-pmp", once mapped
        self.external_ip: str | None = None
        self.external_port: int | None = None
        self.nonce = os.urandom(12)               # PCP: the same nonce renews the same mapping
        self.running = False
        self.wake = threading.Event()

    # -- one exchange --

    def _ask(self, request: bytes, parse):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((self.gateway, self.gateway_port))
            for timeout in TRIES:
                s.send(request)
                s.settimeout(timeout)
                end = time.time() + timeout
                while time.time() < end:
                    try:
                        reply = s.recv(1100)
                    except (TimeoutError, socket.timeout):
                        break
                    except OSError as e:                # e.g. ICMP port unreachable
                        raise PortMapError(f"gateway {self.gateway} refused: {e}") from None
                    result = parse(reply)
                    if result is not None:
                        return result
        raise PortMapError(f"no answer from {self.gateway}:{self.gateway_port}")

    def _local_ip(self) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((self.gateway, self.gateway_port))
            return s.getsockname()[0]

    def _pcp(self, lifetime: int) -> tuple[str, int, int]:
        request = (struct.pack("!BBHI", PCP_VERSION, PCP_MAP, 0, lifetime) + _mapped(self._local_ip())
                   + self.nonce + struct.pack("!B3xHH", TCP, self.internal_port, self.external_port or 0)
                   + _mapped(self.external_ip or "0.0.0.0"))

        def parse(r: bytes):
            if len(r) >= 2 and r[0] == 0:
                raise PortMapError("unsupported version")   # a NAT-PMP-only gateway answers in its format
            if len(r) < 60 or r[0] != PCP_VERSION or r[1] != PCP_MAP | 0x80 or r[24:36] != self.nonce:
                return None
            if r[3] == PCP_UNSUPP_VERSION:
                raise PortMapError("unsupported version")
            if r[3] != 0:
                raise PortMapError(f"gateway refused the mapping (PCP result {r[3]})")
            life = struct.unpack("!I", r[4:8])[0]
            port = struct.unpack("!H", r[42:44])[0]
            return _unmapped(r[44:60]), port, life
        return self._ask(request, parse)

    def _natpmp(self, lifetime: int) -> tuple[str, int, int]:
        def parse_ip(r: bytes):
            if len(r) < 12 or r[0] != 0 or r[1] != 128:
                return None
            if struct.unpack("!H", r[2:4])[0] != 0:
                raise PortMapError("gateway refused to tell its address (NAT-PMP)")
            return socket.inet_ntoa(r[8:12])

        def parse_map(r: bytes):
            if len(r) < 16 or r[0] != 0 or r[1] != 130 or struct.unpack("!H", r[8:10])[0] != self.internal_port:
                return None
            code = struct.unpack("!H", r[2:4])[0]
            if code != 0:
                raise PortMapError(f"gateway refused the mapping (NAT-PMP result {code})")
            port, life = struct.unpack("!HI", r[10:16])
            return port, life
        ip = self._ask(struct.pack("!BB", 0, 0), parse_ip) if lifetime else self.external_ip
        suggested = (self.external_port or self.internal_port) if lifetime else 0   # a deletion asks for 0
        port, life = self._ask(struct.pack("!BBHHHI", 0, 2, 0, self.internal_port, suggested, lifetime), parse_map)
        return ip, port, life

    def map_once(self, lifetime: int | None = None) -> tuple[str, int, int]:
        """Create or renew the mapping; returns (external ip, external port, lifetime granted)."""
        if self.gateway is None:
            self.gateway = default_gateway()
            if self.gateway is None:
                raise PortMapError("no default gateway found")
        lifetime = self.lifetime if lifetime is None else lifetime
        if self.protocol != "nat-pmp":
            try:
                ip, port, life = self._pcp(lifetime)
                self.protocol = "pcp"
            except PortMapError as e:
                if self.protocol == "pcp" or "unsupported version" not in str(e) and "no answer" not in str(e):
                    raise
                ip, port, life = self._natpmp(lifetime)
                self.protocol = "nat-pmp"
        else:
            ip, port, life = self._natpmp(lifetime)
        if lifetime:
            self.external_ip, self.external_port = ip, port
        return ip, port, life

    # -- keeping it --

    def start(self) -> bool:
        """Map now (False if no gateway would), then keep renewing in the background."""
        try:
            ip, port, life = self.map_once()
        except PortMapError as e:
            self.log(f"[portmap] could not open port {self.internal_port} on the router: {e}")
            return False
        self.log(f"[portmap] the router forwards {ip}:{port} to port {self.internal_port} "
                 f"({self.protocol}, {life} s)")
        self.running = True
        threading.Thread(target=self._renew_loop, args=(life,), daemon=True, name="portmap").start()
        return True

    def _renew_loop(self, life: int) -> None:
        while self.running:
            if self.wake.wait(max(1.0, life / 2)):
                return
            try:
                _, _, life = self.map_once()
            except PortMapError as e:
                self.log(f"[portmap] renewing the mapping failed ({e}); trying again")
                life = 120

    def stop(self) -> None:
        """Remove the mapping (lifetime 0)."""
        if not self.running:
            return
        self.running = False
        self.wake.set()
        try:
            self.map_once(lifetime=0)
        except PortMapError:
            pass                                      # it expires by itself
