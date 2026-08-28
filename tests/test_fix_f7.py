"""
Regression tests for F7 (latency tail shape, p50-normalized CCDF facets).

Guards the two P1 defects from the 2026-07-07 audit:
  * F7-1 — the per-protocol scenario picker preferred chain=='scg' only "if any", so a
    protocol with no scg scenario silently fell back to a chain=direct (1-gateway) run
    inside an otherwise all-scg (2-gateway) facet. The chain is now pinned figure-wide;
    combos existing only under another chain are dropped and disclosed.
  * F7-2 — tail curves were built from the MEAN of each percentile across repetitions,
    which fabricates a curve no run produced when reps are bimodal (observed 1.7×/18×/20×
    rendered as ~14×). Each curve is now one scenario's median-tail repetition; zero-work
    (dead) repetitions are excluded before selection.
And the P2 disclosure F7-3: when the plaintext curves are harness-limited while the
encrypted ones are not, the asymmetry must appear in the figure's method notes.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f7.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import latency_tails  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402


def _rep_row(scenario: str, transport: str, protocol: str, chain: str, run: int,
             p50: float, ratio: float, *, messages: int = 100_000) -> dict:
    """One runs.csv repetition with a monotone percentile ladder reaching p999 = ratio*p50."""
    return {
        "scenario": scenario,
        "family": "matrix",
        "transport": transport,
        "protocol": protocol,
        "chain": chain,
        "message_bytes": 16384,
        "run": run,
        "messages": messages,
        "latency_p50_us": p50,
        "latency_p90_us": p50 * (1 + 0.3 * (ratio - 1)),
        "latency_p95_us": p50 * (1 + 0.5 * (ratio - 1)),
        "latency_p99_us": p50 * (1 + 0.8 * (ratio - 1)),
        "latency_p999_us": p50 * ratio,
    }


def _runs_frame() -> pd.DataFrame:
    """The audit shape in miniature: tcp has scg scenarios for routing (bimodal reps:
    1.7×/18×/20×) and tls/1.3 (homogeneous ~1.2×); tproxy has an scg tls/1.3 scenario but
    routing exists ONLY as a chain=direct run (the F7-1 fallback trap)."""
    rows = []
    for run, ratio in enumerate((1.7, 18.0, 20.0), start=1):
        rows.append(_rep_row("matrix_routing_tcp_tcp_16KB_scg_1c", "tcp", "none", "scg",
                             run, p50=100.0, ratio=ratio))
    for run in (1, 2, 3):
        rows.append(_rep_row("matrix_tls13_tcp_tcp_16KB_scg_1c", "tcp", "tls/1.3", "scg",
                             run, p50=12000.0, ratio=1.2))
        rows.append(_rep_row("matrix_tls13_tproxy_tproxy_16KB_scg_1c", "tproxy", "tls/1.3",
                             "scg", run, p50=12500.0, ratio=1.25))
        rows.append(_rep_row("matrix_routing_tproxy_tproxy_16KB_direct_1c", "tproxy", "none",
                             "direct", run, p50=110.0, ratio=16.0))
    return pd.DataFrame(rows)


def _summary_frame() -> pd.DataFrame:
    """harness_limited=True on plaintext only — the F7-3 asymmetry."""
    return pd.DataFrame([
        {"scenario": "matrix_routing_tcp_tcp_16KB_scg_1c", "harness_limited": True},
        {"scenario": "matrix_tls13_tcp_tcp_16KB_scg_1c", "harness_limited": False},
        {"scenario": "matrix_tls13_tproxy_tproxy_16KB_scg_1c", "harness_limited": False},
        {"scenario": "matrix_routing_tproxy_tproxy_16KB_direct_1c", "harness_limited": True},
    ])


def _bundle(runs: pd.DataFrame, summary: pd.DataFrame | None = None) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=summary if summary is not None else empty,
        runs=runs,
        sysmetrics=empty,
        saturation=empty,
        skipped=empty,
        sysinfo={"hostname": "test"},
    )


def _chrome_text(entry: dict, kind: str) -> str:
    return " | ".join(c["text"] for c in entry.get("chrome", []) if c["kind"] == kind)


def _render(runs: pd.DataFrame, summary: pd.DataFrame | None = None) -> dict:
    bundle = _bundle(runs, summary)
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp), formats=("png",))
        latency_tails.make(bundle, saver)
    return saver.manifest[-1]


# ---------------------------------------------------------------------- F7-1: chain pin


def test_pin_chain_prefers_scg_and_drops_direct_only_combos():
    runs = _runs_frame()
    pinned, pin = latency_tails._pin_chain(runs)
    assert pin == "scg"
    assert (pinned["chain"] == "scg").all()
    # The direct-only tproxy routing scenario must be gone from the plotting pool …
    assert "matrix_routing_tproxy_tproxy_16KB_direct_1c" not in set(pinned["scenario"])
    # … and reported as a dropped combo for the disclosure note.
    dropped = latency_tails._dropped_combos(runs, pinned)
    assert dropped == [("tproxy", "none")]


def test_pin_chain_direct_only_run_stays_homogeneous():
    """A run with no scg rows at all must still render — pinned to its own single chain."""
    runs = _runs_frame()
    runs = runs[runs["chain"] == "direct"]
    pinned, pin = latency_tails._pin_chain(runs)
    assert pin == "direct"
    assert len(pinned) == len(runs)
    assert latency_tails._dropped_combos(runs, pinned) == []


def test_make_excludes_direct_fallback_curve_and_discloses_it():
    entry = _render(_runs_frame(), _summary_frame())
    assert "skipped" not in entry, f"F7 unexpectedly skipped: {entry.get('skipped')}"
    method = _chrome_text(entry, "method")
    assert "chain pinned to scg" in method
    assert "omitted" in method and "TPROXY" in method.upper()
    # The direct scenario's 16× tail must NOT drive the takeaway: plaintext max comes from
    # the scg tcp scenario's median rep (18×), not from a mean (~13×) or the direct run.
    takeaway = _chrome_text(entry, "takeaway")
    assert "~18.0×" in takeaway
    assert "TCP" in takeaway


# --------------------------------------------------------------- F7-2: median repetition


def test_median_rep_tails_picks_observed_median_not_mean():
    runs = _runs_frame()
    tcp_routing = runs[runs["scenario"] == "matrix_routing_tcp_tcp_16KB_scg_1c"]
    tails, spread = latency_tails._median_rep_tails(tcp_routing)
    g = tails.sort_values("percentile")
    p50 = float(g[g["percentile"] == 0.5]["latency_us"].iloc[0])
    p999 = float(g[g["percentile"] == 0.999]["latency_us"].iloc[0])
    # Median rep of ratios (1.7, 18, 20) is 18 — an observed run. The old mean gave ~13.2.
    assert abs(p999 / p50 - 18.0) < 1e-9
    lo, hi = spread["matrix_routing_tcp_tcp_16KB_scg_1c"]
    assert abs(lo - 1.7) < 1e-9 and abs(hi - 20.0) < 1e-9


def test_median_rep_tails_excludes_dead_repetitions():
    """A zero-work rep (messages=0, garbage percentiles) must not participate in selection."""
    reps = [
        _rep_row("s", "tcp", "none", "scg", 1, p50=100.0, ratio=50.0, messages=0),  # dead
        _rep_row("s", "tcp", "none", "scg", 2, p50=100.0, ratio=2.0),
        _rep_row("s", "tcp", "none", "scg", 3, p50=100.0, ratio=3.0),
    ]
    tails, spread = latency_tails._median_rep_tails(pd.DataFrame(reps))
    g = tails.sort_values("percentile")
    ratio = float(g[g["percentile"] == 0.999]["latency_us"].iloc[0]) / \
        float(g[g["percentile"] == 0.5]["latency_us"].iloc[0])
    # Two live reps remain (2×, 3×); the lower-middle pick is 2× and 50× never appears.
    assert abs(ratio - 2.0) < 1e-9
    assert spread["s"] == (2.0, 3.0)


def test_median_rep_tails_nonpositive_p50_dropped():
    reps = [
        _rep_row("s", "tcp", "none", "scg", 1, p50=100.0, ratio=2.0),
        _rep_row("s", "tcp", "none", "scg", 2, p50=0.0, ratio=2.0),  # invalid median
        _rep_row("s", "tcp", "none", "scg", 3, p50=np.nan, ratio=2.0),
    ]
    tails, _spread = latency_tails._median_rep_tails(pd.DataFrame(reps))
    assert len(tails[tails["percentile"] == 0.5]) == 1
    assert float(tails[tails["percentile"] == 0.5]["latency_us"].iloc[0]) == 100.0


# ------------------------------------------------------- F7-3: harness-limited disclosure


def test_harness_limited_asymmetry_disclosed():
    entry = _render(_runs_frame(), _summary_frame())
    method = _chrome_text(entry, "method")
    assert "harness-limited" in method
    assert "1/1" in method and "0/2" in method  # plaintext vs encrypted curve counts
    assert "harness-limited" in _chrome_text(entry, "takeaway")


def test_no_harness_note_when_summary_lacks_column():
    entry = _render(_runs_frame(), pd.DataFrame({"scenario": ["x"]}))
    assert "harness-limited" not in _chrome_text(entry, "method")


if __name__ == "__main__":
    test_pin_chain_prefers_scg_and_drops_direct_only_combos()
    test_pin_chain_direct_only_run_stays_homogeneous()
    test_make_excludes_direct_fallback_curve_and_discloses_it()
    test_median_rep_tails_picks_observed_median_not_mean()
    test_median_rep_tails_excludes_dead_repetitions()
    test_median_rep_tails_nonpositive_p50_dropped()
    test_harness_limited_asymmetry_disclosed()
    test_no_harness_note_when_summary_lacks_column()
    print("ok")
