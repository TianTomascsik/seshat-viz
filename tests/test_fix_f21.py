"""
Regression tests for F21 (parallel-coordinates trade-off map).

Guards the two audited failure modes of the "most balanced" takeaway and the drawn pool:

* F21-2 — the NaN→0.5 mid-axis fill let a config whose metric was never measured carry a
  guaranteed 0.5 floor into the min-score, crowning an unmeasured config ("no axis below
  50%" was the placeholder itself) with an arbitrary row-order tie-break. Now: an axis
  measured for fewer than half the configs is dropped (and disclosed), a missing value is
  a line gap that never scores, the prize requires every drawn axis measured (with a
  disclosed "no measured axis" fallback), and exact rank ties break deterministically.
* F21-3 — n_gateways==0 loopback-baseline rows blended into the routing polylines' means
  and shifted every config's rank slot. Now they never enter the canvas.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f21.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import parallel_coords  # noqa: E402
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


def _row(transport: str, protocol: str, *, tput: float, p99: float,
         rss: float = np.nan, jitter: float = np.nan, n_gateways: int = 1) -> dict:
    tag = protocol.replace("/", "").replace(".", "").replace("+", "_")
    return {
        "scenario": f"matrix_{tag}_{transport}_256B_scg_1c_gw{n_gateways}",
        "transport": transport,
        "protocol": protocol,
        "message_bytes": 256,
        "connections": 1,
        "n_gateways": n_gateways,
        "throughput_gbps_mean": tput,
        "latency_p99_us_mean": p99,
        "rss_peak_kib": rss,
        "jitter_us_mean": jitter,
    }


def _render(summary: pd.DataFrame) -> dict:
    """Run the figure end-to-end into a temp dir; return the manifest entry."""
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        parallel_coords.make(_bundle(summary), saver)
    return saver.manifest[-1]


def _chrome(entry: dict, kind: str) -> str:
    assert "skipped" not in entry, f"F21 unexpectedly skipped: {entry.get('skipped')}"
    return next(c["text"] for c in entry["chrome"] if c["kind"] == kind)


# ----------------------------------------------------------------------------------------
# F21-2: low-coverage axis gate
# ----------------------------------------------------------------------------------------

def test_low_coverage_axis_dropped_and_disclosed():
    """Jitter measured for 2/6 configs (<half) → the axis is dropped from the canvas and
    the method note discloses the coverage instead of drawing fabricated mid-axis points."""
    rows = [
        _row("tcp", "none", tput=20, p99=100, rss=8000, jitter=5.0),
        _row("tcp", "tls/1.2", tput=10, p99=200, rss=9000, jitter=10.0),
        _row("tcp", "tls/1.3", tput=11, p99=210, rss=9100),
        _row("tcp", "ktls/1.2", tput=12, p99=180, rss=9200),
        _row("tcp", "ktls/1.3", tput=13, p99=190, rss=9300),
        _row("tcp", "dtls/1.2", tput=9, p99=250, rss=9400),
    ]
    entry = _render(pd.DataFrame(rows))
    method = _chrome(entry, "method")
    assert "axes drawn: throughput, p99 latency, peak RSS (" in method
    assert "low-coverage axes dropped: jitter (measured 2/6 configs)" in method
    take = _chrome(entry, "takeaway")
    assert "Most balanced crypto config" in take
    assert "no axis below" in take  # kept axes are fully measured → no softened wording


# ----------------------------------------------------------------------------------------
# F21-2: a gap never scores and never wins
# ----------------------------------------------------------------------------------------

def _grp_norm(rows: list) -> tuple:
    grp = pd.DataFrame(rows)
    metrics = [(c, lbl, hi) for c, lbl, hi in parallel_coords._AXES
               if c in grp.columns and grp[c].notna().any()]
    norm, metrics, low_cov = parallel_coords._rank_norm(grp, metrics)
    return grp, norm, metrics, low_cov


def test_gap_config_cannot_take_prize():
    """Jitter kept (3/4 coverage) but the eligible config with the best measured min-score
    has a jitter gap: it must not be crowned (candidacy needs every drawn axis measured),
    and its missing vertex must be NaN — not a 0.5 floor — in the normalized table."""
    grp, norm, metrics, low_cov = _grp_norm([
        {"transport": "tcp", "protocol": "none", "throughput_gbps_mean": 20,
         "latency_p99_us_mean": 100, "jitter_us_mean": 5.0},
        {"transport": "tcp", "protocol": "tls/1.3", "throughput_gbps_mean": 15,
         "latency_p99_us_mean": 120, "jitter_us_mean": np.nan},
        {"transport": "tcp", "protocol": "tls/1.2", "throughput_gbps_mean": 10,
         "latency_p99_us_mean": 200, "jitter_us_mean": 10.0},
        {"transport": "tcp", "protocol": "ktls/1.2", "throughput_gbps_mean": 12,
         "latency_p99_us_mean": 300, "jitter_us_mean": 8.0},
    ])
    assert not low_cov and len(metrics) == 3
    i_gap = grp.index[grp["protocol"] == "tls/1.3"][0]
    assert pd.isna(norm.loc[i_gap, "jitter_us_mean"])  # gap, not a fabricated 0.5
    # its measured min (0.667) tops every fully-measured eligible config's min (0.0) …
    assert float(norm.min(axis=1).loc[i_gap]) > 0.5
    pick = parallel_coords._most_balanced(grp, norm)
    assert pick is not None and pick["softened"] is False
    # … yet the prize goes to a fully-measured config (mean-rank tie-break → ktls/1.2).
    assert str(grp.loc[pick["idx"], "protocol"]) == "ktls/1.2"


def test_softened_claim_when_no_fully_measured_candidate():
    """Every eligible config misses the jitter axis (kept at exactly half coverage): the
    takeaway must claim 'no measured axis below', not 'no axis below'."""
    rows = [
        _row("tcp", "none", tput=20, p99=100, jitter=5.0),
        _row("tcp", "tls/1.3", tput=15, p99=150),
        _row("tcp", "dtls/1.2", tput=10, p99=200),
        _row("udp", "none", tput=5, p99=300, jitter=4.0),
    ]
    entry = _render(pd.DataFrame(rows))
    take = _chrome(entry, "takeaway")
    assert "no measured axis below" in take
    assert "TLS 1.3" in take  # best measured min among eligible configs
    method = _chrome(entry, "method")
    assert "miss an axis (line gap, not scored)" in method


def test_tie_break_deterministic_and_disclosed():
    """Two eligible configs with identical measurements tie exactly on every rank: the
    winner must not depend on row order, and the co-leader must be disclosed."""
    rows = [
        {"transport": "tcp", "protocol": "none", "throughput_gbps_mean": 20,
         "latency_p99_us_mean": 100, "rss_peak_kib": 8000},
        {"transport": "tcp", "protocol": "tls/1.2", "throughput_gbps_mean": 10,
         "latency_p99_us_mean": 200, "rss_peak_kib": 9000},
        {"transport": "tcp", "protocol": "tls/1.3", "throughput_gbps_mean": 10,
         "latency_p99_us_mean": 200, "rss_peak_kib": 9000},
    ]
    for order in (rows, list(reversed(rows))):
        grp, norm, metrics, _ = _grp_norm(order)
        pick = parallel_coords._most_balanced(grp, norm)
        assert pick is not None
        assert str(grp.loc[pick["idx"], "protocol"]) == "tls/1.2"  # label order, not row order
        assert [str(grp.loc[i, "protocol"]) for i in pick["ties"]] == ["tls/1.3"]
    # end-to-end: the tie is named in the takeaway
    entry = _render(pd.DataFrame([_row(r["transport"], r["protocol"],
                                       tput=r["throughput_gbps_mean"],
                                       p99=r["latency_p99_us_mean"],
                                       rss=r["rss_peak_kib"]) for r in rows]))
    take = _chrome(entry, "takeaway")
    assert "tied with TLS 1.3 · TCP" in take


# ----------------------------------------------------------------------------------------
# F21-3: loopback baselines stay off the canvas
# ----------------------------------------------------------------------------------------

def test_loopback_rows_excluded_from_canvas():
    """A 0-gw loopback row for (tcp, none) would drag that polyline's mean below tls/1.2
    and hand tls/1.2 the top throughput rank (takeaway '100%'); with the row excluded the
    honest claim is '50%'. The exclusion must also be disclosed in the method note."""
    rows = [
        _row("tcp", "none", tput=20, p99=300, rss=10000),
        _row("tcp", "tls/1.2", tput=15, p99=150, rss=8000),
        _row("tcp", "tls/1.3", tput=10, p99=200, rss=9000),
        _row("tcp", "none", tput=1, p99=300, rss=10000, n_gateways=0),  # loopback baseline
    ]
    entry = _render(pd.DataFrame(rows))
    take = _chrome(entry, "takeaway")
    assert "TLS 1.2 · TCP" in take
    assert "no axis below 50% of best" in take, take
    method = _chrome(entry, "method")
    assert "loopback (0-gw) rows excluded: 1" in method


def test_gateway_count_blending_disclosed():
    """When a config's mean averages 1-gw and 2-gw rows the method note must say so."""
    rows = [
        _row("tcp", "none", tput=20, p99=100, rss=8000),
        _row("tcp", "none", tput=18, p99=110, rss=8100, n_gateways=2),
        _row("tcp", "tls/1.2", tput=10, p99=200, rss=9000),
        _row("tcp", "tls/1.2", tput=9, p99=220, rss=9100, n_gateways=2),
        _row("tcp", "tls/1.3", tput=11, p99=210, rss=9200),
    ]
    entry = _render(pd.DataFrame(rows))
    method = _chrome(entry, "method")
    assert "polyline means blend n_gateways=1/2 rows (2/3 configs)" in method


if __name__ == "__main__":
    test_low_coverage_axis_dropped_and_disclosed()
    test_gap_config_cannot_take_prize()
    test_softened_claim_when_no_fully_measured_candidate()
    test_tie_break_deterministic_and_disclosed()
    test_loopback_rows_excluded_from_canvas()
    test_gateway_count_blending_disclosed()
    print("ok")
