"""
Regression tests for F8 (saturation knee).

Guards the audited takeaway/disclosure defects:
  * F8-1 (P0): the takeaway pooled max() over ALL loss-shedding sweeps but attributed the
    result to "only the DTLS/UDP path" — a chimera (routing-UDP's knee + DTLS's loss on one
    path). Each shedding path must be named with ITS OWN knee and peak loss.
  * F8-2 (P1): sweeps flagged `harness_limited` by summary.csv rendered with no disclosure;
    their plateaus are lower bounds, and the "rows decompose ... cost" method-note sentence
    must not be claimed on such data.
  * F8-3 (P2): the p99 floor 8567 µs printed as "0.0 s" (%.1f), and the "(TCP)" label
    excluded the loopback-UDP panel that produced the floor.
  * F8-4 (P3): panels must lay out as the factorial (rows loopback/routing/crypto ×
    columns TCP/UDP), not in loader order.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f8.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import saturation  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402


def _sweep(scenario: str, transport: str, protocol: str, *, thr, loss, p99) -> pd.DataFrame:
    """One saturation.csv sweep (offered 0.5..3.0 Gbps) with joined summary columns."""
    n = len(thr)
    return pd.DataFrame({
        "scenario": [scenario] * n,
        "offered_mbps": [500.0 * (i + 1) for i in range(n)],
        "throughput_gbps": thr,
        "loss_pct": loss,
        "latency_p99_us": p99,
        "transport": [transport] * n,
        "protocol": [protocol] * n,
        "message_bytes": [1024] * n,
    })


def _sat_frame() -> pd.DataFrame:
    """Four sweeps mirroring the canonical run's shape: two latency-degrading panels
    (loopback TCP + loopback UDP — UDP owns the 8.6 ms p99 floor) and two loss-shedding
    panels with DIFFERENT knees/losses (plaintext routing-UDP 2.00/36%, DTLS 1.49/53%)."""
    return pd.concat([
        _sweep("sat_loopback_tcp_1KB", "tcp", "none",
               thr=[0.5, 1.0, 1.5, 1.62, 1.62, 1.62],
               loss=[0.0] * 6,
               p99=[50.0, 60.0, 405887.0, 900000.0, 1594966.4, 1500000.0]),
        _sweep("sat_loopback_udp_1KB", "udp", "none",
               thr=[0.5, 1.0, 1.5, 2.0, 2.5, 2.95],
               loss=[0.0, 0.0, 0.01, 0.0, 0.02, 0.0],
               p99=[50.0, 60.0, 8566.52, 9000.0, 1.0e6, 1.2e6]),
        _sweep("sat_gw_routing_udp_1KB", "udp", "none",
               thr=[0.5, 1.0, 1.5, 2.0034, 2.0, 2.0],
               loss=[0.0, 0.0, 0.0, 0.5, 20.0, 35.64],
               p99=[50.0, 60.0, 70.0, 80.0, 90.0, 100.0]),
        _sweep("sat_gw_dtls12_udp_1KB", "udp", "dtls/1.2",
               thr=[0.5, 1.0, 1.4921, 1.7, 1.7, 1.7],
               loss=[0.0, 0.0, 0.8, 30.0, 50.0, 52.65],
               p99=[50.0, 60.0, 70.0, 80.0, 90.0, 100.0]),
    ], ignore_index=True)


def _summary(limited: dict) -> pd.DataFrame:
    """summary.csv slice with the per-scenario harness_limited verdict. Values are given
    as CSV-style strings AND bools on purpose — both load shapes occur in the wild."""
    rows = []
    for scen, lim in limited.items():
        rows.append({
            "scenario": scen,
            "harness_limited": "True" if lim else False,
            "connections": 1,
            "message_bytes": 1024,
        })
    return pd.DataFrame(rows)


def _bundle(sat: pd.DataFrame, summary: pd.DataFrame) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=summary,
        runs=empty,
        sysmetrics=empty,
        saturation=sat,
        skipped=empty,
        sysinfo={"hostname": "test"},
    )


def _render(sat: pd.DataFrame, summary: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        saturation.make(_bundle(sat, summary), saver)
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F8 unexpectedly skipped: {entry.get('skipped')}"
    chrome = {c["kind"]: c["text"] for c in entry.get("chrome", [])}
    return entry, chrome


_LIMITED = {
    "sat_loopback_tcp_1KB": False,
    "sat_loopback_udp_1KB": False,
    "sat_gw_routing_udp_1KB": True,
    "sat_gw_dtls12_udp_1KB": True,
}


def test_takeaway_names_each_shedding_path():
    """F8-1: per-path knee + loss, no pooled chimera, no hardcoded DTLS attribution."""
    _, chrome = _render(_sat_frame(), _summary(_LIMITED))
    take = chrome["takeaway"]
    assert "only the DTLS/UDP path" not in take
    # Both shedding paths named, each with its OWN knee and its OWN peak loss.
    assert "gw routing udp 1KB" in take and "gw dtls12 udp 1KB" in take
    assert "2.00" in take and "36%" in take   # routing-UDP's knee/loss
    assert "1.49" in take and "53%" in take   # DTLS's knee/loss
    # The old pooled rendering ("knee ≈ 2.0 Gbps, up to ~53% loss" once) is gone: the
    # 53% figure must not be the only loss number.
    assert take.count("% loss") >= 2
    # Both shedding cells are gateway-UDP (plaintext + crypto) → the property is named.
    assert "gateway-UDP" in take


def test_harness_limited_disclosure():
    """F8-2: limited sweeps are starred and disclosed; the cost-decomposition sentence
    is only claimed when NO sweep is harness-limited."""
    _, chrome = _render(_sat_frame(), _summary(_LIMITED))
    assert "harness-limited" in chrome["method"]
    assert "2/4" in chrome["method"]
    assert "lower bound" in chrome["method"]
    assert "decompose" not in chrome["method"]  # contradicted by limited plateaus
    assert "*" in chrome["takeaway"] and "lower bound" in chrome["takeaway"]

    # All-clean run: no stars, and the factorial decomposition may be advertised.
    clean = _summary({k: False for k in _LIMITED})
    _, chrome = _render(_sat_frame(), clean)
    assert "harness-limited" not in chrome["method"]
    assert "decompose" in chrome["method"]
    assert "*" not in chrome["takeaway"]


def test_latency_floor_formats_as_ms_and_names_loopback_udp():
    """F8-3: an 8.6 ms p99 floor renders as '9 ms' (never '0.0 s') and the latency
    clause credits the loopback-UDP panel that produced it."""
    assert saturation._fmt_dur_us(8566.52) == "9 ms"
    assert saturation._fmt_dur_us(1594966.4) == "1.6 s"
    assert saturation._fmt_dur_us(405887.0) == "0.4 s"
    _, chrome = _render(_sat_frame(), _summary(_LIMITED))
    take = chrome["takeaway"]
    assert "9 ms" in take and "1.6 s" in take
    assert "0.0 s" not in take
    assert "loopback UDP" in take


def test_factorial_panel_order():
    """F8-4: panels sort into the factorial grid (rows loopback/routing/crypto ×
    columns TCP/UDP) regardless of the loader's alphabetical order."""
    sat = pd.concat([
        _sat_frame(),
        _sweep("sat_gw_routing_tcp_1KB", "tcp", "none",
               thr=[0.5, 1.0, 1.5, 2.0, 2.18, 2.1], loss=[0.0] * 6,
               p99=[50.0, 60.0, 555899.0, 800000.0, 900000.0, 950000.0]),
        _sweep("sat_gw_tls13_tcp_1KB", "tcp", "tls/1.3",
               thr=[0.5, 1.0, 1.5, 3.0, 4.08, 3.5], loss=[0.0] * 6,
               p99=[50.0, 60.0, 54369.0, 700000.0, 800000.0, 850000.0]),
    ], ignore_index=True)
    ordered = sorted(
        sat["scenario"].dropna().unique(),
        key=lambda s: (saturation._factorial_key(s, sat[sat["scenario"] == s]), str(s)),
    )
    assert ordered == [
        "sat_loopback_tcp_1KB", "sat_loopback_udp_1KB",       # row 0: no gateway
        "sat_gw_routing_tcp_1KB", "sat_gw_routing_udp_1KB",   # row 1: gateway plaintext
        "sat_gw_tls13_tcp_1KB", "sat_gw_dtls12_udp_1KB",      # row 2: gateway crypto
    ]


def test_missing_summary_still_renders():
    """No summary at all (older run): no harness verdicts, figure must still render."""
    _, chrome = _render(_sat_frame(), pd.DataFrame())
    assert "harness-limited" not in chrome["method"]
    assert "takeaway" in chrome


if __name__ == "__main__":
    test_takeaway_names_each_shedding_path()
    test_harness_limited_disclosure()
    test_latency_floor_formats_as_ms_and_names_loopback_udp()
    test_factorial_panel_order()
    test_missing_summary_still_renders()
    print("ok")
