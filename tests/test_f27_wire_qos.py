"""
Tests for F27 (traffic-class prioritisation on the wire, QOS-001).

Guards the honesty rails: without the qdisc campaign the takeaway must say
"indistinguishable" and never claim prioritisation; the wire-qos3-qdisc dir
auto-appears as the third media-condition and flips the takeaway branch; the
"DSCP 46" evidence comes only from the capture (never from the CSV count
columns); no qos rows → record_skip. Runnable under pytest or as a plain script.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wire_fixtures import make_campaign, row  # noqa: E402

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import wire_qos as f27  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402
from seshat_viz.wire import load_wire  # noqa: E402


def _bundle(wire) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(run_dir=Path("20260101-000000"), summary=empty, runs=empty,
                     sysmetrics=empty, saturation=empty, skipped=empty,
                     sysinfo={"hostname": "test"}, wire=wire)


def _qos_rows(medium, alone, contended, normal):
    out = []
    for cell, vals, cls in (("qos-safety-alone", alone, "safety"),
                            ("qos-safety-contended", contended, "safety"),
                            ("qos-normal-contended", normal, "normal")):
        for i, p99 in enumerate(vals, 1):
            out.append(row(f"{cell}#r{i}", medium, traffic_class=cls, message_bytes=256,
                           connections="" if cell.endswith("alone") else 4,
                           rtt_us_p50=p99 * 0.8, rtt_us_p99=p99, cpu_pct_mean=5.0))
    return out


def _chrome(saver):
    return {c["kind"]: c["text"] for c in saver.manifest[-1]["chrome"]}


def _base(tmp: Path):
    make_campaign(tmp, "lo-qos3", _qos_rows("loopback", [88, 89, 88], [107, 108, 108],
                                            [110, 111, 110]))
    # the honest chaos: safety and normal overlap/invert across replicates
    make_campaign(tmp, "wire-qos3", _qos_rows("wire", [310, 312, 310],
                                              [4291, 6531, 4370], [524, 3277, 6522]))


def test_no_qdisc_takeaway_never_claims_prioritisation():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
        _base(Path(tmp))
        wb = load_wire(tmp)
        saver = theme.Saver(Path(out), formats=("png",))
        f27.make(_bundle(wb), saver)
        chrome = _chrome(saver)
        assert "indistinguishable" in chrome["takeaway"]
        assert "reproducing the loopback isolation" not in chrome["takeaway"]
        assert "pending" in chrome["method"]  # both qdisc campaign and evidence
        assert "packet counts" in chrome["method"]


def test_qdisc_campaign_auto_appears_and_flips_the_branch():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
        tmp = Path(tmp)
        _base(tmp)
        qdir = make_campaign(tmp, "wire-qos3-qdisc",
                             _qos_rows("wire", [310, 311, 310], [355, 360, 358],
                                       [2100, 2400, 2250]))
        # far-side evidence next to the qdisc campaign
        peer = qdir / "peer-out"
        peer.mkdir()
        (peer / "dscp_evidence.json").write_text(json.dumps({
            "ports": {"21101": {"proto": "tcp", "dscp": {"46": 12345}, "ecn": {},
                                "direction": {"to_peer": 12345}},
                      "21100": {"proto": "tcp", "dscp": {"0": 999}, "ecn": {},
                                "direction": {"to_peer": 999}}}}))
        wb = load_wire(tmp)
        assert bool(wb.df[wb.df["campaign"] == "wire-qos3-qdisc"]["qdisc"].iloc[0])
        saver = theme.Saver(Path(out), formats=("png",))
        f27.make(_bundle(wb), saver)
        chrome = _chrome(saver)
        assert "priority qdisc" in chrome["takeaway"]
        assert "reproducing the loopback isolation result" in chrome["takeaway"]
        # the code point comes from the evidence file, with its packet count
        assert "12,345" in chrome["takeaway"] and "DSCP 46" in chrome["takeaway"]
        assert "pending" not in chrome["takeaway"]


def test_skip_without_qos_rows():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
        make_campaign(Path(tmp), "knee-wire", [
            row("sweep-tcp-500", "wire", offered_mbps=500, throughput_gbps_mean=0.5,
                cpu_pct_mean=9)])
        wb = load_wire(tmp)
        saver = theme.Saver(Path(out), formats=("png",))
        f27.make(_bundle(wb), saver)
        assert saver.manifest[-1]["skipped"] == "no qos-* rows in the wire campaigns"


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_f27_wire_qos: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
