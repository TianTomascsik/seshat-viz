"""
Regression tests for F12 (system-metrics timeline, transports compared).

Guards three invariants of F12:

* F12-1 — the comparable-panel picker admitted every scenario carrying a /proc timeseries,
  so its max-throughput preference crowned the SHM zero-copy slot-ring *microbenchmark*
  (shmzc_*, non-default data path, 0s-cooldown schedule) over the like-for-like default-path
  blast row. Eligibility must be restricted to sustained-blast rows (paced_*/shmzc_*/
  handshake_*/pp_*/conn_* are a different measurement class).
* F12-2 — 4/5 panels came from harness-limited rows at achieved loads spanning tens of Gbps
  with no disclosure. The method note must carry a *computed* achieved-load span and
  harness-limited count for the rows actually plotted.
* F12-3 — the "SHM spins ~1.5 cores even between reps (RC4 busy-poll)" claim was hardcoded
  and would render unchanged whatever the data said (and the default SHM path idles to
  ~0.15 cores between reps). Every so-what number must come from the plotted trace.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f12_timeline.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import timeline  # noqa: E402
from seshat_viz.loader import RunBundle, TRANSPORT_ORDER  # noqa: E402


def _bundle(summary: pd.DataFrame, sysmetrics: pd.DataFrame) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=summary,
        runs=empty,
        sysmetrics=sysmetrics,
        saturation=empty,
        sysinfo={"hostname": "test"},
    )


def _summary() -> pd.DataFrame:
    """One pinned cell (routing · 1 gateway · 1 conn · 16 KiB) with the real-run shape:
    a like-for-like blast row per transport, plus the higher-throughput shmzc slot-ring
    microbenchmark and a paced row — the exact bait the picker used to take (F12-1)."""
    rows = [
        ("iface_shm_throughput_16KB_1c", "shm", 31.3, True),
        ("iface_tcp_scg_throughput_16KB_1c", "tcp", 46.3, True),
        # Non-default measurement classes in the SAME pinned cell, posting the top numbers:
        ("shmzc_shm_scale_16KB_1c", "shm", 47.4, False),
        ("paced_udp_16KB_1c", "udp", 99.0, False),
    ]
    return pd.DataFrame(
        [
            {
                "scenario": scen,
                "transport": transport,
                "protocol": "none",
                "n_gateways": 1,
                "connections": 1,
                "message_bytes": 16384,
                "throughput_gbps_mean": tput,
                "harness_limited": limited,
            }
            for scen, transport, tput, limited in rows
        ]
    )


def _trace(scenario: str, *, idle_gap: bool) -> pd.DataFrame:
    """A synthetic /proc timeseries: warm-up ramp, one or two 150% bursts, teardown tail.
    With `idle_gap` the bursts are separated by a 15% inter-repetition dip."""
    rows = []
    for i, t_s in enumerate(np.arange(0.0, 20.5, 0.5)):
        if t_s < 1.0:
            cpu = 5.0  # warm-up ramp
        elif t_s <= 8.0:
            cpu = 150.0
        elif t_s < 12.0:
            cpu = 15.0 if idle_gap else 150.0  # inter-rep gap (or busy-poll through it)
        elif t_s <= 19.0:
            cpu = 150.0
        else:
            cpu = 3.0  # teardown tail
        rows.append(
            {"scenario": scenario, "elapsed_ms": t_s * 1000.0, "cpu_pct": cpu, "rss_kib": 100 * 1024.0}
        )
    return pd.DataFrame(rows)


def test_picker_rejects_variant_and_paced_rows():
    """F12-1: shmzc (top throughput) and paced rows share the pinned cell but must lose to
    the like-for-like blast row / drop out entirely."""
    summ = _summary()
    have_sys = set(summ["scenario"])  # all four carry a timeseries
    scenarios, chosen, ok = timeline._pin_and_pick(
        summ,
        have_sys,
        vary="transport",
        pin={"protocol": "none", "n_gateways": 1},
        order=TRANSPORT_ORDER,
        max_scen=5,
    )
    assert ok, "two blast transports survive, the pick must be comparable"
    assert "iface_shm_throughput_16KB_1c" in scenarios, "the like-for-like SHM row must win"
    assert not any(s.startswith(("shmzc_", "paced_", "handshake_")) for s in scenarios), scenarios
    assert chosen["connections"] == 1 and chosen["message_bytes"] == 16384


def test_steady_spans_split_on_the_inter_burst_gap():
    """The steady-window recovery still splits on the idle gap (the warm-up ramp and
    teardown tail stay excluded). The figure deliberately has no per-panel so-what line
    (no _trace_floor/_sowhat helpers — red italic in-axes prose is banned);
    interpretation lives in the caption/body text."""
    g = _trace("x", idle_gap=True)
    t = g["elapsed_ms"].values / 1000.0
    spans = timeline._steady_spans(t, g["cpu_pct"].values)
    assert len(spans) == 2, spans
    assert not hasattr(timeline, "_sowhat") and not hasattr(timeline, "_trace_floor")


def test_load_disclosure_computes_span_and_harness_count():
    """F12-2: the disclosure clause is derived from the plotted rows only."""
    summ = _summary()
    plotted = ["iface_shm_throughput_16KB_1c", "iface_tcp_scg_throughput_16KB_1c"]
    note = timeline._load_disclosure(summ, plotted)
    assert "achieved load differs per panel (31–46 Gbps)" in note, note
    assert "2/2 panels are harness-limited" in note, note
    # Without the columns there is nothing honest to say — the clause must vanish.
    bare = summ.drop(columns=["throughput_gbps_mean", "harness_limited"])
    assert timeline._load_disclosure(bare, plotted) == ""


def test_make_end_to_end_method_note_and_no_boilerplate():
    """Full make(): comparable pick, computed disclosure in the method note, and no
    hardcoded SHM claim anywhere in the recorded chrome."""
    summ = _summary()
    sysm = pd.concat(
        [
            _trace("iface_shm_throughput_16KB_1c", idle_gap=True),
            _trace("iface_tcp_scg_throughput_16KB_1c", idle_gap=True),
            _trace("shmzc_shm_scale_16KB_1c", idle_gap=False),  # in have_sys, still excluded
        ],
        ignore_index=True,
    )
    bundle = _bundle(summ, sysm)
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp), formats=("png",))
        timeline.make(bundle, saver)
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F12 unexpectedly skipped: {entry.get('skipped')}"
    assert entry["id"] == "F12"
    chrome = {c["kind"]: c["text"] for c in entry["chrome"]}
    method = chrome["method"]
    assert "panels ARE comparable" in method
    assert "achieved load differs per panel (31–46 Gbps)" in method, method
    assert "2/2 panels are harness-limited" in method, method
    for text in chrome.values():
        assert "~1.5 cores" not in text and "RC4" not in text, text


if __name__ == "__main__":
    test_picker_rejects_variant_and_paced_rows()
    test_steady_spans_split_on_the_inter_burst_gap()
    test_load_disclosure_computes_span_and_harness_count()
    test_make_end_to_end_method_note_and_no_boilerplate()
    print("ok")
