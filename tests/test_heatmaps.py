"""
Regression tests for F4 (protocol × size heatmaps).

Guards the crash where a metric (e.g. jitter/PDV) is populated for only *some* transports:
the `have_*` gates are global over the frame, but a single transport's pivot is then empty,
and seaborn.heatmap raises `zero-size array to reduction operation fmin` on the all-NaN
array before the LogNorm can bail. F4 must draw a blank panel for that transport instead of
sinking the whole figure.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_heatmaps.py`) so it needs no extra dev dependency.
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


def _matrix_summary(*, jitter_only_on: str | None) -> pd.DataFrame:
    """Two transports × two protocols × two sizes, all with throughput+latency; jitter is
    populated only on `jitter_only_on` (or nowhere if None) — the asymmetry that crashed F4."""
    rows = []
    for transport in ("tcp", "tproxy"):
        for protocol in ("none", "tls/1.3"):
            for size in (64, 16384):
                jit = 5.0 if transport == jitter_only_on else np.nan
                rows.append({
                    "scenario": f"matrix_{protocol}_{transport}_{size}",
                    "family": "matrix",
                    "transport": transport,
                    "protocol": protocol,
                    "message_bytes": size,
                    "connections": 1,
                    "chain": "scg",
                    "throughput_gbps_mean": 10.0,
                    "latency_p99_us_mean": 50.0,
                    "jitter_us_mean": jit,
                })
    return pd.DataFrame(rows)


def test_f4_jitter_present_on_one_transport_only():
    """The real-run shape: jitter only on tproxy → tcp's jitter pivot is empty. Must not raise."""
    bundle = _bundle(_matrix_summary(jitter_only_on="tproxy"))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        heatmaps.make(bundle, saver)  # must not raise
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F4 unexpectedly skipped: {entry.get('skipped')}"
    assert entry["id"] == "F4"


def test_f4_no_jitter_anywhere_still_renders():
    """All-NaN jitter across every transport (have_jit false) still produces a figure."""
    bundle = _bundle(_matrix_summary(jitter_only_on=None))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        heatmaps.make(bundle, saver)
    assert "skipped" not in saver.manifest[-1]


def test_has_cells_helper():
    assert heatmaps._has_cells(pd.DataFrame({"a": [1.0, np.nan]})) is True
    assert heatmaps._has_cells(pd.DataFrame({"a": [np.nan, np.nan]})) is False
    assert heatmaps._has_cells(pd.DataFrame()) is False
    assert heatmaps._has_cells(None) is False


if __name__ == "__main__":
    test_f4_jitter_present_on_one_transport_only()
    test_f4_no_jitter_anywhere_still_renders()
    test_has_cells_helper()
    print("ok")
