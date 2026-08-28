"""
Tests for SCG-SESHAT/scripts/merge_peer_out.py — the peer sink-report merge.

The merge is positional (burst k ↔ k-th traffic-bearing cell) with byte-count
corroboration, so these tests guard the refusal paths above all: a shifted or
short sequence must abort with nothing written, and the original CSV must be
byte-identical after every invocation, successful or not.

Runnable under pytest or as a plain script.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wire_fixtures import make_campaign, row  # noqa: E402

_SCRIPT = (Path(__file__).resolve().parents[2] / "SCG-SESHAT" / "scripts" /
           "merge_peer_out.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("merge_peer_out", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _campaign(tmp: Path) -> Path:
    rows = [
        row("qos-safety-alone", "wire", connections="", rtt_us_p50=300, rtt_us_p99=310),
        row("tput-tls-c1", "wire", throughput_gbps_mean=0.948, cpu_pct_mean=4.8),
        row("sweep-udp-500", "wire", transport="udp", protocol="dtls",
            traffic_class="safety", message_bytes=1400, offered_mbps=500),
    ]
    sidecars = {
        "tput-tls-c1": {"sent_bytes": 1_000_000_000, "sender_gbps": 0.948},
        "sweep-udp-500": {"sent_bytes": 500_000_000},
    }
    return make_campaign(tmp, "wire-run", rows, sidecars=sidecars)


def _peer_out(tmp: Path, bulk: list[dict], dgram: list[dict]) -> Path:
    peer = tmp / "peer-out"
    peer.mkdir(exist_ok=True)
    (peer / "sink-bulk.jsonl").write_text("\n".join(json.dumps(r) for r in bulk) + "\n")
    (peer / "sink-dgram.jsonl").write_text("\n".join(json.dumps(r) for r in dgram) + "\n")
    return peer


def _burst(i, byte_count, delivered, loss=0.0, lost=0, dscp=None):
    out = {"burst": i, "bytes": byte_count, "count": 1000, "delivered_gbps": delivered,
           "loss_pct": loss, "lost": lost, "observed_s": 10.0}
    if dscp is not None:
        out.update(dscp_observed=1000, dscp_matched=dscp, dscp_preserved=dscp == 1000)
    return out


def test_happy_path_fills_delivered_and_link_limited():
    mod = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cdir = _campaign(tmp)
        original = (cdir / "wire_summary.csv").read_bytes()
        peer = _peer_out(tmp, [_burst(0, 1_002_000_000, 0.947)],
                         [_burst(0, 490_000_000, 0.489, loss=2.0, lost=20, dscp=1000)])
        out = mod.merge(cdir, peer, "wire_summary_merged.csv", force=False)
        assert (cdir / "wire_summary.csv").read_bytes() == original, "original mutated!"
        import csv as _csv

        merged = {r["scenario"]: r for r in _csv.DictReader(open(out))}
        tput = merged["tput-tls-c1"]
        assert tput["delivered_gbps"] == "0.947000"
        assert tput["link_limited"] == "true" and tput["bottleneck"] == "link"
        udp = merged["sweep-udp-500"]
        assert udp["loss_pct"] == "2.0000" and udp["total_lost"] == "20"
        assert udp["dscp_preserved"] == "true"
        assert udp["link_limited"] == "false"
        assert merged["qos-safety-alone"]["delivered_gbps"] == "", "RTT-only cell touched"
        # refusal to overwrite without --force
        try:
            mod.merge(cdir, peer, "wire_summary_merged.csv", force=False)
        except SystemExit:
            pass
        else:
            raise AssertionError("existing merged CSV must require --force")
        mod.merge(cdir, peer, "wire_summary_merged.csv", force=True)


def _expect_refusal(mod, cdir, peer, why):
    original = (cdir / "wire_summary.csv").read_bytes()
    try:
        mod.merge(cdir, peer, "wire_summary_merged.csv", force=True)
    except SystemExit:
        assert (cdir / "wire_summary.csv").read_bytes() == original
        assert not (cdir / "wire_summary_merged.csv").exists(), f"{why}: partial output left"
    else:
        raise AssertionError(f"{why}: merge did not refuse")


def test_tcp_byte_mismatch_refuses():
    mod = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cdir = _campaign(tmp)
        peer = _peer_out(tmp, [_burst(0, 700_000_000, 0.66)],  # 30% off → shifted sequence
                         [_burst(0, 490_000_000, 0.489)])
        _expect_refusal(mod, cdir, peer, "TCP >5% byte mismatch")


def test_burst_count_mismatch_refuses():
    mod = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cdir = _campaign(tmp)
        peer = _peer_out(tmp, [_burst(0, 1_002_000_000, 0.947), _burst(1, 5, 0.001)],
                         [_burst(0, 490_000_000, 0.489)])
        _expect_refusal(mod, cdir, peer, "extra burst record")


def test_udp_sink_exceeding_sender_refuses():
    mod = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cdir = _campaign(tmp)
        peer = _peer_out(tmp, [_burst(0, 1_002_000_000, 0.947)],
                         [_burst(0, 600_000_000, 0.6)])  # sink > sent × 1.05 → impossible
        _expect_refusal(mod, cdir, peer, "UDP sink exceeds sender")


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_merge_peer_out: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
