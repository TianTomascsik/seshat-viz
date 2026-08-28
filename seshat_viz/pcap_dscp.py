"""
DSCP evidence extraction from the wire benchmark's far-side packet capture.

The QOS-001 claim — that the gateway's DSCP marks survive a physical link — is
proven by the capture `wire_peer.sh --capture` takes on the *far* machine
(``peer-out/wire.pcap``, tcpdump snaplen 96: headers only, by design). This
module turns that capture into per-port DSCP histograms for figure F27.

Pure stdlib on purpose: neither tshark nor scapy/dpkt is installed on the
machines involved, and a 96-byte snaplen means every record is trivially small
(Ethernet 14 + IPv4 ≥20 + TCP/UDP header always fit, so the TOS byte at
ip_offset+1 is always present).

Both capture formats are handled: classic pcap (µs and ns magics, either
endianness) and pcapng (a minimal block walk — Section Header for endianness,
Interface Description for the linktype, Enhanced/Simple Packet Blocks for the
frames), since a distro tcpdump may emit either.

Only ``to_peer`` packets (destination port ∈ the mid-hop ports) feed the
evidence histogram: those are the marked data packets crossing the cable toward
the capture point. Reverse-direction packets (ACKs, DTLS handshake replies) are
counted separately and never plotted.
"""

from __future__ import annotations

import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

#: The wire benchmark's inter-gateway ("mid") ports — the only flows on the cable.
#: bulk (TCP, BE), safety (TCP, EF), datagram (UDP, EF), nprobe control (TCP, BE).
PORTS_DEFAULT: Tuple[int, ...] = (21100, 21101, 21102, 21103)

_PCAP_MAGICS = {
    0xA1B2C3D4: "<",  # little-endian, µs timestamps
    0xD4C3B2A1: ">",  # big-endian, µs
    0xA1B23C4D: "<",  # little-endian, ns
    0x4D3CB2A1: ">",  # big-endian, ns
}
_PCAPNG_MAGIC = 0x0A0D0D0A
_LINKTYPE_ETHERNET = 1


class PcapFormatError(ValueError):
    """The file is not a capture this parser understands."""


def _iter_classic(data: bytes) -> Iterator[bytes]:
    """Yield raw frames from a classic pcap byte string."""
    magic = struct.unpack("<I", data[:4])[0]
    endian = _PCAP_MAGICS.get(magic)
    if endian is None:
        magic_be = struct.unpack(">I", data[:4])[0]
        endian = _PCAP_MAGICS.get(magic_be)
        if endian is None:
            raise PcapFormatError(f"unknown pcap magic 0x{magic:08x}")
    linktype = struct.unpack(endian + "I", data[20:24])[0]
    if linktype != _LINKTYPE_ETHERNET:
        raise PcapFormatError(f"unsupported linktype {linktype} (need 1 = Ethernet)")
    off = 24
    while off + 16 <= len(data):
        caplen = struct.unpack(endian + "I", data[off + 8: off + 12])[0]
        frame_start = off + 16
        yield data[frame_start: frame_start + caplen]
        off = frame_start + caplen


def _iter_pcapng(data: bytes) -> Iterator[bytes]:
    """Yield raw frames from a pcapng byte string (minimal block walk)."""
    off = 0
    endian = "<"
    linktype: Optional[int] = None
    while off + 12 <= len(data):
        block_type = struct.unpack(endian + "I", data[off: off + 4])[0]
        if block_type == _PCAPNG_MAGIC:  # Section Header Block: re-derive endianness
            bom = data[off + 8: off + 12]
            endian = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            block_type = struct.unpack(endian + "I", data[off: off + 4])[0]
        block_len = struct.unpack(endian + "I", data[off + 4: off + 8])[0]
        if block_len < 12 or off + block_len > len(data):
            break  # torn tail — stop rather than mis-walk
        body = data[off + 8: off + block_len - 4]
        if block_type == 0x00000001:  # Interface Description Block
            linktype = struct.unpack(endian + "H", body[:2])[0]
            if linktype != _LINKTYPE_ETHERNET:
                raise PcapFormatError(f"unsupported linktype {linktype} (need 1 = Ethernet)")
        elif block_type == 0x00000006:  # Enhanced Packet Block
            caplen = struct.unpack(endian + "I", body[12:16])[0]
            yield body[20: 20 + caplen]
        elif block_type == 0x00000003:  # Simple Packet Block
            yield body[4:]
        off += block_len


def _iter_frames(path: Path) -> Iterator[bytes]:
    data = path.read_bytes()
    if len(data) < 24:
        raise PcapFormatError("file too short to be a capture")
    first = struct.unpack("<I", data[:4])[0]
    if first == _PCAPNG_MAGIC:
        yield from _iter_pcapng(data)
    else:
        yield from _iter_classic(data)


def parse_pcap(path: Path | str, ports: Sequence[int] = PORTS_DEFAULT) -> Dict:
    """Per-port DSCP histograms for the given L4 ports.

    Returns a JSON-ready dict; see the module docstring for the direction
    semantics. Never raises on individual malformed/truncated records — those
    are skip-counted so the evidence stays auditable.
    """
    path = Path(path)
    ports_set = set(int(p) for p in ports)
    skipped = Counter()
    total = 0
    per_port: Dict[int, Dict] = {}

    for frame in _iter_frames(path):
        total += 1
        if len(frame) < 14:
            skipped["truncated"] += 1
            continue
        ethertype = struct.unpack("!H", frame[12:14])[0]
        ip_off = 14
        if ethertype == 0x8100:  # one VLAN tag tolerated
            if len(frame) < 18:
                skipped["truncated"] += 1
                continue
            ethertype = struct.unpack("!H", frame[16:18])[0]
            ip_off = 18
        if ethertype != 0x0800:
            skipped["non_ipv4"] += 1
            continue
        if len(frame) < ip_off + 20:
            skipped["truncated"] += 1
            continue
        tos = frame[ip_off + 1]
        proto = frame[ip_off + 9]
        if proto not in (6, 17):
            skipped["non_tcp_udp"] += 1
            continue
        ihl = (frame[ip_off] & 0x0F) * 4
        l4 = ip_off + ihl
        if len(frame) < l4 + 4:
            skipped["truncated"] += 1
            continue
        src, dst = struct.unpack("!HH", frame[l4: l4 + 4])
        if dst in ports_set:
            port, direction = dst, "to_peer"
        elif src in ports_set:
            port, direction = src, "from_peer"
        else:
            skipped["other_port"] += 1
            continue
        entry = per_port.setdefault(
            port,
            {"proto": "tcp" if proto == 6 else "udp",
             "dscp": Counter(), "ecn": Counter(),
             "direction": Counter()},
        )
        entry["direction"][direction] += 1
        if direction == "to_peer":  # only marked data packets are evidence
            entry["dscp"][tos >> 2] += 1
            entry["ecn"][tos & 0x03] += 1

    return {
        "path": str(path),
        "packets_total": total,
        "packets_skipped": dict(skipped),
        "ports": {
            port: {
                "proto": e["proto"],
                "dscp": {str(k): v for k, v in sorted(e["dscp"].items())},
                "ecn": {str(k): v for k, v in sorted(e["ecn"].items())},
                "direction": dict(e["direction"]),
            }
            for port, e in sorted(per_port.items())
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m seshat_viz.pcap_dscp",
        description="Extract per-port DSCP histograms from a wire-benchmark capture.",
    )
    parser.add_argument("pcap", help="capture file (classic pcap or pcapng, Ethernet)")
    parser.add_argument("--out", default=None,
                        help="write the JSON here (default: print to stdout)")
    parser.add_argument("--ports", default=",".join(str(p) for p in PORTS_DEFAULT),
                        help="comma-separated L4 ports to attribute (default: the mid-hop ports)")
    args = parser.parse_args(argv)

    try:
        ports = tuple(int(p) for p in args.ports.split(",") if p.strip())
        result = parse_pcap(Path(args.pcap), ports)
    except (OSError, PcapFormatError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        matched = sum(sum(p["direction"].values()) for p in result["ports"].values())
        print(f"wrote {args.out} ({result['packets_total']} packets, {matched} on the mid-hop ports)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
