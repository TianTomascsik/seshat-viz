"""
Regression tests for the F4 honesty fixes (protocol × size heatmaps).

Guards four caption/disclosure defects:
  * F4-1 — the scg-preference dedup silently mixed topologies: a cell with no 2-gateway
    scg run fell back to 1-gateway direct with no per-cell marking. `_pivot` must flag
    those cells, `_annot` must dagger them, and the footer must carry the legend.
  * F4-2 — jitter conclusions were generalized from a single transport; the jitter clause
    must be scoped to the transports that actually carry PDV data, and its magnitude /
    separation claims must be computed from the rendered pivots (with a guard that drops
    the "separates the protocols" claim when the spread is within 2×).
  * F4-3 — the RTT grid is single-gateway (scg-direct) next to an scg-preferred throughput
    column; the method note must disclose it and the "scg path preferred" footer must be
    scoped away from the latency panels.
  * F4-4 — the takeaway hardcoded "~5–10×" / "flat"; the throughput clause must compute
    routing's scaling range on stream transports, report datagram transports apart, and
    bound the encrypted rows' residual growth.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f4.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import heatmaps  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402


def _bundle(summary: pd.DataFrame) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=summary,
        runs=empty,
        sysmetrics=empty,
        saturation=empty,
        skipped=empty,
        sysinfo={"hostname": "test"},
    )


def _chrome(saver: theme.Saver, kind: str) -> str:
    """Concatenated chrome text of one kind (headline/takeaway/method/provenance) from the
    last manifest entry — chrome is recorded unconditionally, so this works chrome-on/off."""
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F4 unexpectedly skipped: {entry.get('skipped')}"
    return " ".join(c["text"] for c in entry.get("chrome", []) if c["kind"] == kind)


def _mixed_chain_rows() -> list[dict]:
    """The real-run TPROXY shape: routing exists ONLY as 1-gateway direct, the encrypted
    row at both chains (2-gw scg must win); a second transport is uniformly scg."""
    rows = []
    for size in (64, 16384):
        rows.append({"scenario": f"m_none_tproxy_{size}_direct", "family": "matrix",
                     "transport": "tproxy", "protocol": "none", "message_bytes": size,
                     "connections": 1, "chain": "direct", "throughput_gbps_mean": 12.0,
                     "latency_p99_us_mean": 40.0, "jitter_us_mean": np.nan})
        for chain, tput in (("scg", 8.0), ("direct", 9.0)):
            rows.append({"scenario": f"m_tls_tproxy_{size}_{chain}", "family": "matrix",
                         "transport": "tproxy", "protocol": "tls/1.3", "message_bytes": size,
                         "connections": 1, "chain": chain, "throughput_gbps_mean": tput,
                         "latency_p99_us_mean": 50.0, "jitter_us_mean": np.nan})
        for protocol in ("none", "tls/1.3"):
            rows.append({"scenario": f"m_{protocol}_tcp_{size}_scg", "family": "matrix",
                         "transport": "tcp", "protocol": protocol, "message_bytes": size,
                         "connections": 1, "chain": "scg", "throughput_gbps_mean": 10.0,
                         "latency_p99_us_mean": 45.0, "jitter_us_mean": np.nan})
    return rows


def _rtt_rows() -> list[dict]:
    """A matrix_lat_* closed-loop grid covering every cell, all single-gateway direct."""
    rows = []
    for transport in ("tcp", "tproxy"):
        for protocol in ("none", "tls/1.3"):
            for size in (64, 16384):
                rows.append({"scenario": f"matrix_lat_{protocol}_{transport}_{size}",
                             "family": "matrix-latency", "transport": transport,
                             "protocol": protocol, "message_bytes": size, "connections": 1,
                             "chain": "direct", "throughput_gbps_mean": np.nan,
                             "latency_p99_us_mean": np.nan, "jitter_us_mean": np.nan,
                             "rtt_us_p99": 55.0})
    return rows


# ---------------------------------------------------------------------------- F4-1


def test_pivot_flags_direct_fallback_cells_and_prefers_scg():
    """Cells with no scg row are flagged as fallback; where both chains exist, the scg
    value wins and the cell is NOT flagged."""
    sub = pd.DataFrame([r for r in _mixed_chain_rows() if r["transport"] == "tproxy"])
    piv, fb = heatmaps._pivot(sub, "throughput_gbps_mean")
    assert piv.loc["tls/1.3", 64] == 8.0, "scg value must displace the direct duplicate"
    assert bool(fb.loc["none", 64]) and bool(fb.loc["none", 16384]), "direct-only cells must be flagged"
    assert not fb.loc["tls/1.3"].any(), "scg-backed cells must not be flagged"


def test_annot_daggers_only_fallback_cells():
    sub = pd.DataFrame([r for r in _mixed_chain_rows() if r["transport"] == "tproxy"])
    piv, fb = heatmaps._pivot(sub, "throughput_gbps_mean")
    annot, fmt, flagged = heatmaps._annot(piv, fb, ".2f")
    assert flagged and fmt == ""
    i_none, i_tls = piv.index.get_loc("none"), piv.index.get_loc("tls/1.3")
    j64 = piv.columns.get_loc(64)
    assert annot[i_none][j64] == "12.00†"
    assert annot[i_tls][j64] == "8.00"


def test_annot_stays_numeric_without_fallback():
    sub = pd.DataFrame([r for r in _mixed_chain_rows() if r["transport"] == "tcp"])
    piv, fb = heatmaps._pivot(sub, "throughput_gbps_mean")
    annot, fmt, flagged = heatmaps._annot(piv, fb, ".2f")
    assert annot is True and fmt == ".2f" and not flagged


def test_footer_carries_dagger_legend_when_fallback_rendered():
    bundle = _bundle(pd.DataFrame(_mixed_chain_rows()))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        heatmaps.make(bundle, saver)
    assert "† = 1-gateway direct" in _chrome(saver, "provenance")


def test_no_scg_anywhere_means_no_daggers_and_no_preference_claim():
    """With no scg rows at all nothing was 'preferred' — the footer must not claim it and
    no cell may carry a dagger legend."""
    rows = [dict(r, chain="direct") for r in _mixed_chain_rows()]
    bundle = _bundle(pd.DataFrame(rows))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        heatmaps.make(bundle, saver)
    foot = _chrome(saver, "provenance")
    assert "scg path preferred" not in foot
    assert "†" not in foot


# ---------------------------------------------------------------------------- F4-2


def test_jitter_clause_is_scoped_and_computed():
    """Jitter on one of two rendered transports → the clause names its true scope, and the
    band / separation numbers come from the data (not a hardcoded '~150 µs' / fixed ordering)."""
    piv = pd.DataFrame(
        {64.0: [0.5, 0.6], 65536.0: [20.0, 150.0]},
        index=["none", "tls/1.2+integrity"],
    )
    clause = heatmaps._jitter_clause({"tproxy": piv}, ["tcp", "tproxy"])
    assert "measured only on TPROXY in this run" in clause
    assert "0.50–150 µs" in clause
    assert "DOES grow with size" in clause
    # Separation names the computed extremes at the largest covered size.
    assert "TLS 1.2 (integrity) 150 µs vs routing 20 µs" in clause
    assert "64KB" in clause


def test_jitter_clause_drops_unsupported_claims():
    """Near-equal protocols with no size growth → neither 'grow with size' nor 'separate
    the protocols' may be claimed; full coverage → no scope disclaimer."""
    piv = pd.DataFrame({64.0: [5.0, 5.5], 65536.0: [6.0, 6.5]}, index=["none", "tls/1.3"])
    clause = heatmaps._jitter_clause({"tproxy": piv}, ["tproxy"])
    assert "separate the protocols" not in clause
    assert "grow with size" not in clause
    assert "measured only" not in clause


def test_takeaway_and_method_note_scope_partial_jitter():
    """End-to-end: a frame with jitter populated only on tproxy must scope both the
    takeaway clause and the method note to TPROXY."""
    rows = []
    for transport in ("tcp", "tproxy"):
        for protocol, j64, j16k in (("none", 2.0, 8.0), ("tls/1.3", 2.2, 9.0)):
            for size, jit in ((64, j64), (16384, j16k)):
                rows.append({"scenario": f"m_{protocol}_{transport}_{size}", "family": "matrix",
                             "transport": transport, "protocol": protocol, "message_bytes": size,
                             "connections": 1, "chain": "scg", "throughput_gbps_mean": 10.0,
                             "latency_p99_us_mean": 50.0,
                             "jitter_us_mean": jit if transport == "tproxy" else np.nan})
    bundle = _bundle(pd.DataFrame(rows))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        heatmaps.make(bundle, saver)
    assert "measured only on TPROXY in this run" in _chrome(saver, "takeaway")
    assert "populated only for TPROXY in this run" in _chrome(saver, "method")


# ---------------------------------------------------------------------------- F4-3


def test_rtt_chain_disclosed_and_footer_scoped():
    """An all-direct matrix_lat_* grid next to scg-preferred throughput panels: the method
    note must disclose the single-gateway RTT topology and the footer must scope the
    'scg path preferred' claim away from the latency column."""
    bundle = _bundle(pd.DataFrame(_mixed_chain_rows() + _rtt_rows()))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        heatmaps.make(bundle, saver)
    method = _chrome(saver, "method")
    assert "single-gateway (scg-direct)" in method
    assert "differ in topology" in method
    foot = _chrome(saver, "provenance")
    assert "scg path preferred (throughput" in foot
    assert "single-gateway scg-direct" in _chrome(saver, "takeaway")


# ---------------------------------------------------------------------------- F4-4


def test_tput_clause_computes_ranges_and_splits_datagram():
    """Routing scaling is computed per transport (stream range + datagram outlier apart)
    and the encrypted near-flat claim carries a computed bound — no hardcoded '~5–10×'."""
    piv_tcp = pd.DataFrame(
        {64.0: [5.0, 10.2], 256.0: [12.0, 10.0], 16384.0: [25.0, 10.5]},
        index=["none", "tls/1.3"],
    )
    piv_udp = pd.DataFrame({64.0: [0.1], 16384.0: [30.0]}, index=["none"])
    clause = heatmaps._tput_clause({"tcp": piv_tcp, "udp": piv_udp})
    assert "~5.0×" in clause and "on stream transports" in clause
    assert "300× on UDP" in clause and "datagram" in clause
    assert "near-flat above 256 B (≤1.05×" in clause
    assert "5–10×" not in clause


def test_tput_clause_flatness_needs_two_large_sizes():
    """A single ≥256 B column cannot support a flatness claim — the clause must drop it."""
    piv = pd.DataFrame({64.0: [5.0, 10.0], 16384.0: [25.0, 10.5]}, index=["none", "tls/1.3"])
    clause = heatmaps._tput_clause({"tcp": piv})
    assert "near-flat" not in clause
    assert "routing scales ~5.0×" in clause


if __name__ == "__main__":
    test_pivot_flags_direct_fallback_cells_and_prefers_scg()
    test_annot_daggers_only_fallback_cells()
    test_annot_stays_numeric_without_fallback()
    test_footer_carries_dagger_legend_when_fallback_rendered()
    test_no_scg_anywhere_means_no_daggers_and_no_preference_claim()
    test_jitter_clause_is_scoped_and_computed()
    test_jitter_clause_drops_unsupported_claims()
    test_takeaway_and_method_note_scope_partial_jitter()
    test_rtt_chain_disclosed_and_footer_scoped()
    test_tput_clause_computes_ranges_and_splits_datagram()
    test_tput_clause_flatness_needs_two_large_sizes()
    print("ok")
