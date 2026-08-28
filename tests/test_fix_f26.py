"""
Tests for F26 (wire/loopback sweep — loopback-testbed realism).

Guards: the skip path without wire data; the coarse/fine 950-point dedupe
preferring the knee campaign; the knee detection; the takeaway quoting the
clamp plateau (never a >100%-of-ceiling single point); contaminated rows never
reaching the panels. Runnable under pytest or as a plain script.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wire_fixtures import make_campaign, row  # noqa: E402

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import wire_loopback_sweep as f26  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402
from seshat_viz.wire import load_wire  # noqa: E402


def _bundle(wire) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(run_dir=Path("20260101-000000"), summary=empty, runs=empty,
                     sysmetrics=empty, saturation=empty, skipped=empty,
                     sysinfo={"hostname": "test"}, wire=wire)


def _sweep_row(offered, medium, achieved, lag=54.0, p99=650.0, cpu=10.0):
    return row(f"sweep-tcp-{offered}", medium, offered_mbps=offered,
               throughput_gbps_mean=achieved, rtt_us_p50=p99 * 0.5, rtt_us_p99=p99,
               send_lag_mean_us=lag, send_lag_max_us=lag * 3, cpu_pct_mean=cpu,
               message_bytes=65536)


def _wire_bundle(tmp: Path):
    coarse_w = [_sweep_row(o, "wire", o / 1000) for o in range(50, 951, 100)]
    coarse_w[-1] = _sweep_row(950, "wire", 0.999, lag=1100)  # coarse 950: the loser of the dedupe
    coarse_l = [_sweep_row(o, "loopback", o / 1000, cpu=9 + o / 100) for o in range(50, 951, 100)]
    make_campaign(tmp, "wire-run", coarse_w)
    make_campaign(tmp, "wire-loopback-baseline", coarse_l)
    knee_w = [_sweep_row(o, "wire", min(o / 1000, 0.944),
                         lag=54.0 if o <= 950 else 300000.0) for o in range(880, 1001, 10)]
    knee_l = [_sweep_row(o, "loopback", o / 1000) for o in range(880, 1001, 10)]
    make_campaign(tmp, "knee-wire", knee_w)
    make_campaign(tmp, "knee-loopback", knee_l)
    return load_wire(tmp)


def _chrome(saver):
    return {c["kind"]: c["text"] for c in saver.manifest[-1]["chrome"]}


def test_skip_without_wire_data():
    with tempfile.TemporaryDirectory() as out:
        saver = theme.Saver(Path(out), formats=("png",))
        f26.make(_bundle(None), saver)
        assert saver.manifest[-1]["skipped"].startswith("no wire campaign dirs")


def test_dedupe_prefers_knee_and_takeaway_uses_plateau():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
        wb = _wire_bundle(Path(tmp))
        # unit: the 950 duplicate keeps the knee row (0.944), not the coarse 0.999
        sel = f26._sweep_rows(wb.df, "sweep-tcp-")
        p950 = sel[(sel["medium"] == "wire") & (sel["offered_gbps"] == 0.95)]
        assert len(p950) == 1 and float(p950["throughput_gbps_mean"].iloc[0]) == 0.944
        knee = f26._knee(sel[sel["medium"] == "wire"])
        assert knee is not None and knee["offered_gbps"] == 0.96

        theme.set_variant("thesis")
        try:
            saver = theme.Saver(Path(out), formats=("png",))
            f26.make(_bundle(wb), saver)
        finally:
            theme.set_variant("full")
        chrome = _chrome(saver)
        # plateau (0.944), never the coarse 0.999 single point; no >100% claim
        assert "0.944" in chrome["takeaway"]
        assert "100." not in chrome["takeaway"].split("ceiling")[0]
        assert "knee" in chrome["takeaway"] or "960" in chrome["takeaway"]
        assert "sender-side" in chrome["method"] or "sender" in chrome["method"]
        assert "wire campaigns:" in chrome["provenance"]


def test_contaminated_rows_never_plotted():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
        tmp = Path(tmp)
        _ = _wire_bundle(tmp)
        # add a pre-guard contaminated 64B RTT row to wire-run (sidecar without rtt_resyncs)
        make_campaign(tmp, "wire-rtt3", [
            row("rtt-tls-64#r1", "wire", message_bytes=64, connections="",
                rtt_us_p50=7.7, rtt_us_p99=300)],
            sidecars={"rtt-tls-64#r1": {"rtt_n": 1000}})
        wb = load_wire(tmp)
        assert "rtt-tls-64" not in set(wb.df["cell"])
        saver = theme.Saver(Path(out), formats=("png",))
        f26.make(_bundle(wb), saver)
        assert "chrome" in saver.manifest[-1]


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_fix_f26: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
