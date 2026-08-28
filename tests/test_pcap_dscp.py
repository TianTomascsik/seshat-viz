"""
Tests for seshat_viz.pcap_dscp — the pure-python DSCP evidence parser.

Synthesises captures byte-by-byte (struct.pack), so the expected histograms are
known exactly: TOS values on each mid-hop port, both directions, one VLAN-tagged
frame, one truncated record, one non-IPv4 frame, a minimal pcapng file, and a
bad magic. Runnable under pytest or as a plain script.
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz.pcap_dscp import PcapFormatError, parse_pcap  # noqa: E402


def _eth(payload: bytes, ethertype: int = 0x0800, vlan: bool = False) -> bytes:
    hdr = b"\xaa" * 6 + b"\xbb" * 6
    if vlan:
        return hdr + struct.pack("!HH", 0x8100, 0) + struct.pack("!H", ethertype) + payload
    return hdr + struct.pack("!H", ethertype) + payload


def _ipv4(tos: int, proto: int, sport: int, dport: int) -> bytes:
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, tos, 40, 0, 0, 64, proto, 0,
                     bytes([10, 9, 0, 1]), bytes([10, 9, 0, 2]))
    l4 = struct.pack("!HH", sport, dport) + b"\x00" * 16
    return ip + l4


def _classic(frames: list[bytes]) -> bytes:
    out = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 96, 1)
    for f in frames:
        out += struct.pack("<IIII", 0, 0, len(f), len(f)) + f
    return out


def _pcapng(frames: list[bytes]) -> bytes:
    shb = struct.pack("<IIIHHq", 0x0A0D0D0A, 28, 0x1A2B3C4D, 1, 0, -1) + struct.pack("<I", 28)
    idb = struct.pack("<IIHHI", 1, 20, 1, 0, 96) + struct.pack("<I", 20)
    out = shb + idb
    for f in frames:
        pad = (4 - len(f) % 4) % 4
        blen = 32 + len(f) + pad
        out += (struct.pack("<IIIIII", 6, blen, 0, 0, 0, len(f)) + struct.pack("<I", len(f))
                + f + b"\x00" * pad + struct.pack("<I", blen))
    return out


_EF = 46 << 2  # TOS byte for DSCP EF


def test_classic_histograms_directions_and_skips():
    frames = [
        _eth(_ipv4(_EF, 6, 50000, 21101)),          # safety, to_peer, EF
        _eth(_ipv4(_EF, 6, 50000, 21101)),
        _eth(_ipv4(0, 6, 21101, 50000)),            # reverse (ACK) — never evidence
        _eth(_ipv4(0, 6, 50001, 21100)),            # bulk, to_peer, BE
        _eth(_ipv4(_EF | 0b01, 17, 50002, 21102)),  # dgram, EF with ECN bit set
        _eth(_ipv4(_EF, 6, 50003, 21101), vlan=True),  # VLAN-tagged safety frame
        _eth(_ipv4(0, 6, 50004, 9999)),             # unrelated port
        _eth(b"\x00" * 8, ethertype=0x86DD),        # IPv6 → non_ipv4
        b"\xaa" * 10,                               # truncated record
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wire.pcap"
        path.write_bytes(_classic(frames))
        out = parse_pcap(path)
    assert out["packets_total"] == len(frames)
    safety = out["ports"][21101]
    assert safety["dscp"] == {"46": 3}, "EF count must include the VLAN frame"
    assert safety["direction"] == {"to_peer": 3, "from_peer": 1}
    assert out["ports"][21100]["dscp"] == {"0": 1}
    dgram = out["ports"][21102]
    assert dgram["proto"] == "udp" and dgram["dscp"] == {"46": 1}
    assert dgram["ecn"] == {"1": 1}, "ECN bits must not bleed into the DSCP histogram"
    assert out["packets_skipped"] == {"other_port": 1, "non_ipv4": 1, "truncated": 1}


def test_pcapng_and_bad_magic():
    frames = [_eth(_ipv4(_EF, 6, 50000, 21101)), _eth(_ipv4(0, 6, 50001, 21103))]
    with tempfile.TemporaryDirectory() as tmp:
        ng = Path(tmp) / "wire.pcapng"
        ng.write_bytes(_pcapng(frames))
        out = parse_pcap(ng)
        assert out["ports"][21101]["dscp"] == {"46": 1}
        assert out["ports"][21103]["dscp"] == {"0": 1}, "nprobe control port must be attributed"

        bad = Path(tmp) / "bad.pcap"
        bad.write_bytes(b"\xde\xad\xbe\xef" + b"\x00" * 64)
        try:
            parse_pcap(bad)
        except PcapFormatError:
            pass
        else:
            raise AssertionError("bad magic must raise PcapFormatError")


def test_empty_capture_is_valid_absent_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.pcap"
        path.write_bytes(_classic([]))
        out = parse_pcap(path)
    assert out["ports"] == {} and out["packets_total"] == 0


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_pcap_dscp: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
