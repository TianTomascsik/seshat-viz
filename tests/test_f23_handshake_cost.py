"""
Regression tests for F23 (handshake-algorithm cost).

Guards the invariants of F23:

  * F23-1 — cell selection used a keyless `iloc[0]` behind an always-NaN `conn_threads`
    facet, so which replicate got rendered depended on summary.csv row order: reversing the
    frame silently swapped every 4-client row for its 1-client sibling. The figure must key
    cells on `connections` (the real replicate dimension) and refuse ambiguous cells rather
    than let frame order pick.
  * F23-2 — the closed-loop client count was undisclosed, so the absolute conns/s read as a
    gateway-capacity figure. The method note / provenance / takeaway must carry the client
    counts computed from the data.
  * F23-3 — a log latency axis over a ~1.5× span emits colliding minor tick labels; narrow
    spans must fall back to linear.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f23_handshake_cost.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import handshake_cost  # noqa: E402
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


# 4-client vs 1-client rates chosen so the cert-family ratio DIFFERS by client count:
# 2.00× at 4 clients vs 1.00× at 1 client — an order-dependent pick is unmissable.
_ROWS = [
    # scenario, connections, conns_per_sec, ci95, hs_p50_us, hs_p99_us
    ("handshake_tls13_ecdsa", 4, 4000.0, 20.0, 1000.0, 1200.0),
    ("handshake_tls13_ecdsa_1c", 1, 990.0, 5.0, 1010.0, 1050.0),
    ("handshake_tls13_rsa", 4, 2000.0, 90.0, 1400.0, 1800.0),
    ("handshake_tls13_rsa_1c", 1, 990.0, 30.0, 1500.0, 1700.0),
    ("handshake_kex_x25519", 4, 4100.0, 30.0, 940.0, 1160.0),
    ("handshake_kex_x25519_1c", 1, 995.0, 3.0, 1020.0, 1060.0),
    ("handshake_kex_p256", 4, 4050.0, 40.0, 990.0, 1160.0),
    ("handshake_kex_p256_1c", 1, 990.0, 7.0, 1025.0, 1060.0),
]


def _handshake_summary(*, one_client_first: bool = False, extra_rows: list | None = None) -> pd.DataFrame:
    rows = [
        {
            "scenario": name,
            "family": "handshake",
            "protocol": "tls/1.3",
            "transport": "tcp",
            "chain": "direct",
            "connections": conns,
            "conn_threads": np.nan,  # always unset on handshake rows in real runs
            "message_bytes": 64,
            "conns_per_sec": rate,
            "conns_per_sec_ci95": ci,
            "conn_handshake_p50_us": p50,
            "conn_handshake_p99_us": p99,
        }
        for name, conns, rate, ci, p50, p99 in _ROWS + (extra_rows or [])
    ]
    df = pd.DataFrame(rows)
    return df.iloc[::-1].reset_index(drop=True) if one_client_first else df


def _render(summary: pd.DataFrame) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp), formats=("png",))
        handshake_cost.make(_bundle(summary), saver)
    return saver.manifest[-1]


def _chrome(entry: dict, kind: str) -> str:
    return next(rec["text"] for rec in entry.get("chrome", []) if rec["kind"] == kind)


def test_row_order_does_not_change_the_takeaway():
    """F23-1: the takeaway ratio comes from the widest client count regardless of summary.csv
    row order (the old keyless iloc[0] rendered the 1-client 1.00× on the reversed frame)."""
    for flipped in (False, True):
        entry = _render(_handshake_summary(one_client_first=flipped))
        assert "skipped" not in entry, entry.get("skipped")
        take = _chrome(entry, "takeaway")
        assert "ECDSA-P256 vs RSA-2048: 2.00×" in take, f"flipped={flipped}: {take}"


def test_cell_keys_on_facet_not_frame_order():
    """F23-1: _cell must return the row for the requested client count even when a different
    count's row comes first in the frame."""
    sub = pd.DataFrame({
        "scenario": ["handshake_tls13_ecdsa_1c", "handshake_tls13_ecdsa"],
        "label": ["ECDSA-P256", "ECDSA-P256"],
        "facet": [1.0, 4.0],
        "conns_per_sec": [990.0, 4000.0],
    })
    cell = handshake_cost._cell(sub, "ECDSA-P256", 4.0)
    assert cell is not None and cell["conns_per_sec"] == 4000.0
    assert handshake_cost._cell(sub, "ECDSA-P256", 8.0) is None


def test_cell_refuses_ambiguous_rows():
    """F23-1: two rows on the same (label, facet) cell must raise, not silently pick iloc[0]."""
    sub = pd.DataFrame({
        "scenario": ["a", "b"],
        "label": ["ECDSA-P256", "ECDSA-P256"],
        "facet": [4.0, 4.0],
        "conns_per_sec": [1.0, 2.0],
    })
    try:
        handshake_cost._cell(sub, "ECDSA-P256", 4.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on ambiguous (label, facet) rows")


def test_make_skips_on_ambiguous_rows():
    """F23-1: a second scenario landing on an existing (algorithm, client-count) cell must skip
    the figure with an explicit reason instead of rendering an arbitrary row."""
    dup = [("handshake_tls13_ecdsa_dup", 4, 3000.0, 10.0, 1100.0, 1300.0)]
    entry = _render(_handshake_summary(extra_rows=dup))
    assert "skipped" in entry
    assert "ambiguous" in str(entry["skipped"])
    assert "handshake_tls13_ecdsa_dup" in str(entry["skipped"])


def test_closed_loop_concurrency_is_disclosed():
    """F23-2: client counts and message size are computed from the data and disclosed."""
    entry = _render(_handshake_summary())
    method = _chrome(entry, "method")
    prov = _chrome(entry, "provenance")
    take = _chrome(entry, "takeaway")
    headline = _chrome(entry, "headline")
    assert "closed-loop" in method.lower() and "churn rate" in method
    assert "closed-loop" in prov and "1/4 client" in prov and "64 B" in prov
    assert "(closed-loop, 4 clients)" in take
    # Headline context comes from the plotted rows, not hardcoded assumptions.
    assert "TLS 1.3" in headline and "TCP" in headline


def test_latency_scale_narrow_span_is_linear():
    """F23-3: sub-decade spans use a linear axis; only wide spans keep log."""
    assert handshake_cost._latency_scale([940.0, 1200.0, 1800.0]) == "linear"
    assert handshake_cost._latency_scale([100.0, 5000.0]) == "log"
    assert handshake_cost._latency_scale([]) == "linear"
    assert handshake_cost._latency_scale([np.nan, 1000.0]) == "linear"


if __name__ == "__main__":
    test_row_order_does_not_change_the_takeaway()
    test_cell_keys_on_facet_not_frame_order()
    test_cell_refuses_ambiguous_rows()
    test_make_skips_on_ambiguous_rows()
    test_closed_loop_concurrency_is_disclosed()
    test_latency_scale_narrow_span_is_linear()
    print("ok")
