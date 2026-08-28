"""
Regression tests for F16 (closed-loop RTT grid + coordinated-omission inflation).

Guards the two audited defects of the inflation panel:

  * F16-1 — the per-protocol dumbbell must pair BOTH endpoints from the same
    (protocol, interface) cell. The old min-RTT / max-blast aggregation crossed interfaces
    (fastest interface's RTT ÷ slowest interface's blast) and overstated every encrypted
    protocol's ratio by 1.3–1.7×.
  * F16-2 — the open-loop blast baseline must be like-for-like: 1-connection scg-direct
    rows of the throughput matrix family only. The generic pool blended cipher-sweep rows,
    two-gateway `chain=='scg'` rows and the iface/profile families into the divisor.
  * F16-3 — the takeaway names the grid-wide worst matched cell explicitly instead of a
    bare "(bottom)" pointer at a panel that shows a smaller maximum.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f16.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import closed_loop_rtt  # noqa: E402
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


def _rtt_row(proto: str, transport: str, size: int, rtt: float) -> dict:
    """A closed-loop matrix_lat_* grid cell (scg-direct, 1 conn, ping-pong)."""
    return {
        "scenario": f"matrix_lat_{proto.replace('/', '')}_{transport}_{transport}_{size}B_direct_1c",
        "family": "matrix-latency",
        "chain": "direct",
        "mode": "pingpong",
        "transport": transport,
        "protocol": proto,
        "message_bytes": size,
        "connections": 1,
        "rtt_us_p50": rtt * 0.8,
        "rtt_us_p99": rtt,
        "latency_p99_us_mean": np.nan,
    }


def _blast_row(proto: str, transport: str, size: int, p99: float, *,
               family: str = "matrix", chain: str = "direct",
               connections: int = 1, rtt: float = np.nan) -> dict:
    """A sustained-blast row; family/chain/connections/rtt vary to build contaminants."""
    return {
        "scenario": f"{family}_{proto.replace('/', '')}_{transport}_{transport}_{size}B_{chain}_{connections}c",
        "family": family,
        "chain": chain,
        "mode": "throughput",
        "transport": transport,
        "protocol": proto,
        "message_bytes": size,
        "connections": connections,
        "rtt_us_p50": np.nan,
        "rtt_us_p99": rtt,
        "latency_p99_us_mean": p99,
    }


def _chrome(saver: theme.Saver, kind: str) -> str:
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F16 unexpectedly skipped: {entry.get('skipped')}"
    texts = [c["text"] for c in entry.get("chrome", []) if c["kind"] == kind]
    assert texts, f"no '{kind}' chrome recorded"
    return " ".join(texts)


# --------------------------------------------------------------------------------------
# F16-2: the matched blast baseline pool
# --------------------------------------------------------------------------------------

def test_baseline_excludes_contaminant_pools():
    """Only 1-conn scg-direct matrix blast rows enter the baseline: cipher-family,
    two-gateway scg, iface-family, multi-connection and closed-loop (rtt-carrying) rows
    at the same (protocol, transport, size) key must all be ignored."""
    rows = [
        _blast_row("tls/1.2", "tcp", 1024, 100.0),                              # the clean row
        _blast_row("tls/1.2", "tcp", 1024, 9000.0, family="cipher", chain="n/a"),
        _blast_row("tls/1.2", "tcp", 1024, 9000.0, chain="scg"),
        _blast_row("tls/1.2", "tcp", 1024, 9000.0, family="iface", chain="n/a"),
        _blast_row("tls/1.2", "tcp", 1024, 9000.0, connections=4),
        _blast_row("tls/1.2", "tcp", 1024, 9000.0, rtt=50.0),                   # closed-loop leak
    ]
    base = closed_loop_rtt._matched_blast_baseline(pd.DataFrame(rows))
    assert ("tls/1.2", "tcp", 1024.0) in base.index
    assert base[("tls/1.2", "tcp", 1024.0)] == 100.0  # not a blend with any 9000 µs row


def test_baseline_has_no_fallback_blend():
    """A cell whose only blast rows are contaminated gets NO baseline (NaN downstream),
    never a protocol-only or cross-size blend."""
    rows = [
        _blast_row("tls/1.2", "tcp", 1024, 100.0),
        # tls/1.3 exists only as a cipher row; shm exists only at another size.
        _blast_row("tls/1.3", "tcp", 1024, 200.0, family="cipher", chain="n/a"),
        _blast_row("tls/1.2", "shm", 4096, 300.0),
    ]
    base = closed_loop_rtt._matched_blast_baseline(pd.DataFrame(rows))
    assert ("tls/1.3", "tcp", 1024.0) not in base.index
    assert ("tls/1.2", "shm", 1024.0) not in base.index


# --------------------------------------------------------------------------------------
# F16-1 + F16-2 + F16-3 end-to-end: the rendered grid figure
# --------------------------------------------------------------------------------------

def _grid_summary() -> pd.DataFrame:
    """Grid where cross-interface pairing and pool contamination each produce a distinct,
    detectably wrong number:

      tls/1.2  tcp 1KB: rtt 10, blast 1000  -> matched 100   (+ cipher row 50000 µs: blended
                                                pool would give 25500/10 = 2550x)
      tls/1.2  shm 1KB: rtt 50, blast 6000  -> matched 120   (the honest per-protocol max)
      cross-interface min-RTT/max-blast     -> 6000/10 = 600x (the F16-1 defect)
      tls/1.2  shm 64B: rtt  5, blast 20000 -> matched 4000  (grid-wide max, NOT at the
                                                panel's 1KB slice — the F16-3 shape)
    """
    rows = [
        _rtt_row("tls/1.2", "tcp", 1024, 10.0),
        _rtt_row("tls/1.2", "shm", 1024, 50.0),
        _rtt_row("tls/1.2", "shm", 64, 5.0),
        _rtt_row("none", "tcp", 1024, 20.0),
        _rtt_row("none", "shm", 1024, 30.0),
        _blast_row("tls/1.2", "tcp", 1024, 1000.0),
        _blast_row("tls/1.2", "shm", 1024, 6000.0),
        _blast_row("tls/1.2", "shm", 64, 20000.0),
        _blast_row("none", "tcp", 1024, 2000.0),
        _blast_row("none", "shm", 1024, 3000.0),
        _blast_row("tls/1.2", "tcp", 1024, 50000.0, family="cipher", chain="n/a"),
    ]
    return pd.DataFrame(rows)


def test_grid_takeaway_uses_matched_cell_not_cross_pairing():
    """The headline inflation is the grid-wide max of SAME-CELL ratios (4000x at 64B/SHM),
    never the cross-interface 600x pairing nor the cipher-blended 2550x."""
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        closed_loop_rtt.make(_bundle(_grid_summary()), saver)
        take = _chrome(saver, "takeaway")
    assert "≥4000×" in take
    assert "≥600×" not in take and "≥2550×" not in take


def test_grid_takeaway_names_worst_cell_no_bare_bottom_pointer():
    """F16-3: the worst cell (64B over SHM) is named with a grid-wide qualifier and the
    panel's slice is stated; the old unverifiable '(bottom)' pointer is gone."""
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        closed_loop_rtt.make(_bundle(_grid_summary()), saver)
        take = _chrome(saver, "takeaway")
    assert "64B" in take and "SHM" in take and "grid-wide" in take
    assert "1KB slice" in take        # the bottom panel's representative size, disclosed
    assert "(bottom)" not in take


def test_grid_method_note_discloses_matched_pairing_and_baseline_scope():
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        closed_loop_rtt.make(_bundle(_grid_summary()), saver)
        note = _chrome(saver, "method")
    assert "largest matched ratio" in note
    assert "scg-direct matrix blast" in note


def test_grid_without_clean_blast_renders_top_only():
    """No like-for-like blast row at all (only cipher/scg pools): the inflation panel and
    the inflation takeaway sentence must vanish rather than ride a blended baseline."""
    rows = [
        _rtt_row("tls/1.2", "tcp", 1024, 10.0),
        _rtt_row("tls/1.2", "shm", 1024, 50.0),
        _rtt_row("none", "tcp", 1024, 20.0),
        _rtt_row("none", "shm", 1024, 30.0),
        _blast_row("tls/1.2", "tcp", 1024, 50000.0, family="cipher", chain="n/a"),
        _blast_row("tls/1.2", "shm", 1024, 6000.0, chain="scg"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        closed_loop_rtt.make(_bundle(pd.DataFrame(rows)), saver)
        take = _chrome(saver, "takeaway")
    assert "Open-loop blast" not in take  # no inflation claim without an honest baseline


# --------------------------------------------------------------------------------------
# Legacy dumbbell fallback (runs without the matrix_lat_* grid) still renders
# --------------------------------------------------------------------------------------

def test_legacy_dumbbell_still_renders():
    rows = [
        {
            "scenario": "pp_tcp_tls12_1KB", "family": "pp", "chain": "n/a",
            "mode": "pingpong", "transport": "tcp", "protocol": "tls/1.2",
            "message_bytes": 1024, "connections": 1,
            "rtt_us_p50": 8.0, "rtt_us_p99": 10.0, "latency_p99_us_mean": np.nan,
        },
        _blast_row("tls/1.2", "tcp", 1024, 500.0, family="baseline", chain="n/a"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        closed_loop_rtt.make(_bundle(pd.DataFrame(rows)), saver)
    assert "skipped" not in saver.manifest[-1]


if __name__ == "__main__":
    test_baseline_excludes_contaminant_pools()
    test_baseline_has_no_fallback_blend()
    test_grid_takeaway_uses_matched_cell_not_cross_pairing()
    test_grid_takeaway_names_worst_cell_no_bare_bottom_pointer()
    test_grid_method_note_discloses_matched_pairing_and_baseline_scope()
    test_grid_without_clean_blast_renders_top_only()
    test_legacy_dumbbell_still_renders()
    print("ok")
