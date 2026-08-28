"""
Regression tests for F6 (second-gateway insertion cost).

Guards the pair-hygiene rules that keep the chart's sorted extremes honest:

- a matched pair with a load-generator-bound side never enters Δ, and a row where EVERY
  pair is bound makes no Δ claim at all (the +13% "gain" between two harness floors);
- scenarios with a zero-work repetition are excluded before pairing (the bimodal SHM/UDS
  multi-connection means that once crowned the chart with +623%);
- the printed Δ is always the delta of the drawn dumbbell (the median-ratio pair of the
  harness-clean sweep), so number and dots can never contradict;
- a sweep whose clean ratios spread >2× falls back to the lowest-concurrency pair;
- groups with one topology entirely absent (TPROXY 2-gw) are disclosed, not silently dropped.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f6.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import derive, theme  # noqa: E402
from seshat_viz.figures import gateway_cost  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402


def _bundle(summary: pd.DataFrame, runs: pd.DataFrame | None = None) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=summary,
        runs=runs if runs is not None else empty,
        sysmetrics=empty,
        saturation=empty,
        skipped=empty,
        sysinfo={"hostname": "test"},
    )


def _row(transport, protocol, size, conns, chain, gbps, *, hl=False, scenario=None):
    return {
        "scenario": scenario or f"matrix_{protocol}_{transport}_{size}B_{chain}_{conns}c",
        "family": "matrix",
        "transport": transport,
        "protocol": protocol,
        "message_bytes": size,
        "connections": conns,
        "chain": chain,
        "throughput_gbps_mean": gbps,
        "latency_p99_us_mean": 50.0,
        "harness_limited": hl,
    }


def _summary() -> pd.DataFrame:
    """Five groups, one per guarded pathology (values mimic the audited run's shapes)."""
    rows = []
    # A: clean 3-pair sweep, ratios 0.9 / 1.0 / 1.3 → median pair is 4c, Δ = 0%.
    for conns, d, s in [(1, 10.0, 9.0), (4, 20.0, 20.0), (16, 30.0, 39.0)]:
        rows.append(_row("tcp", "tls/1.3", 64, conns, "direct", d))
        rows.append(_row("tcp", "tls/1.3", 64, conns, "scg", s))
    # B: every matched pair load-generator bound (the fake "+13% routing gain" shape).
    for conns, d, s in [(1, 48.8, 47.4), (4, 113.5, 110.7)]:
        rows.append(_row("tcp", "none", 65536, conns, "direct", d, hl=True))
        rows.append(_row("tcp", "none", 65536, conns, "scg", s, hl=True))
    # C: sane 1c pair + a 4c pair whose scg side has a zero-work repetition (dead).
    rows.append(_row("shm", "tls/1.2", 64, 1, "direct", 10.0))
    rows.append(_row("shm", "tls/1.2", 64, 1, "scg", 10.1))
    rows.append(_row("shm", "tls/1.2", 64, 4, "direct", 10.0))
    rows.append(_row("shm", "tls/1.2", 64, 4, "scg", 100.0, scenario="matrix_dead_scg_4c"))
    # D: one topology entirely absent (TPROXY has no 2-gateway data).
    for size in (64, 256):
        rows.append(_row("tproxy", "none", size, 1, "direct", 9.4))
    # E: clean but bimodal sweep (ratios 1.007 / 0.097 / 17.8, spread ≫ 2×).
    for conns, d, s in [(1, 5.38, 5.41), (4, 18.41, 1.78), (16, 1.74, 30.9)]:
        rows.append(_row("unix", "ktls/1.3", 64, conns, "direct", d))
        rows.append(_row("unix", "ktls/1.3", 64, conns, "scg", s))
    return pd.DataFrame(rows)


def _rows_by_group(tbl: pd.DataFrame) -> dict:
    return {(r["transport"], r["protocol"], r["message_bytes"]): r for _, r in tbl.iterrows()}


def test_all_bound_group_makes_no_delta_claim():
    """F6-3: 12/12 harness-limited constituent rows must not render a +13% 'gain'."""
    tbl, _ = gateway_cost._pair_rows(_summary(), set())
    r = _rows_by_group(tbl)[("tcp", "none", 65536)]
    assert r["flag"] == "bound"
    assert not np.isfinite(r["delta_pct"])
    assert r["n_clean"] == 0 and r["n_pairs"] == 2
    text, _color = gateway_cost._annotation(r)
    assert "load-gen bound" in text and "%" not in text


def test_dead_repeat_scenario_excluded_from_pairing():
    """F6-1: the zero-work-repetition side must not turn a ~+1% row into a +450% one."""
    dead = {"matrix_dead_scg_4c"}
    tbl, _ = gateway_cost._pair_rows(_summary(), dead)
    r = _rows_by_group(tbl)[("shm", "tls/1.2", 64)]
    # Only the 1c pair survives (4c lost its scg side): Δ is the honest +1%.
    assert r["conns"] == 1 and r["n_pairs"] == 1
    assert abs(r["delta_pct"] - 1.0) < 0.2
    # Without the exclusion the 4c garbage pair (ratio 10×) contaminates the group.
    tbl_raw, _ = gateway_cost._pair_rows(_summary(), set())
    r_raw = _rows_by_group(tbl_raw)[("shm", "tls/1.2", 64)]
    assert r_raw["n_pairs"] == 2  # proves the dead pair only vanishes via the dead set


def test_dead_repeat_detection_via_runs_frame():
    """derive.dead_repeat_scenarios wiring: a zero-work repetition flags its scenario."""
    runs = pd.DataFrame({
        "scenario": ["matrix_dead_scg_4c", "matrix_dead_scg_4c", "other"],
        "throughput_gbps": [0.0, 12.0, 5.0],
        "messages": [0, 100, 100],
    })
    assert derive.dead_repeat_scenarios(runs) == {"matrix_dead_scg_4c"}


def test_annotation_is_the_drawn_pair():
    """F6-2: Δ must be the delta of the dumbbell actually drawn (median-ratio pair)."""
    tbl, _ = gateway_cost._pair_rows(_summary(), set())
    r = _rows_by_group(tbl)[("tcp", "tls/1.3", 64)]
    assert r["flag"] == "ok"
    # Ratios 0.9/1.0/1.3 → median pair is the 4c one (20 → 20 Gbps, Δ = 0%).
    assert r["conns"] == 4
    assert r["direct"] == 20.0 and r["scg"] == 20.0
    drawn_delta = (r["scg"] / r["direct"] - 1.0) * 100.0
    assert abs(r["delta_pct"] - drawn_delta) < 1e-9
    text, _color = gateway_cost._annotation(r)
    assert "@4c" in text and text.startswith(f"{drawn_delta:+.0f}%")


def test_bimodal_sweep_falls_back_to_lowest_pair():
    """F6-1: a >2× ratio spread disqualifies the sweep median; the 1c pair is shown, flagged."""
    tbl, _ = gateway_cost._pair_rows(_summary(), set())
    r = _rows_by_group(tbl)[("unix", "ktls/1.3", 64)]
    assert r["flag"] == "unstable"
    assert r["conns"] == 1
    assert abs(r["delta_pct"] - (5.41 / 5.38 - 1.0) * 100.0) < 1e-9  # ~+0.6%, not +600%
    text, _color = gateway_cost._annotation(r)
    assert "†" in text


def test_missing_topology_is_disclosed_not_silent():
    """F6-4: groups whose scg side never ran are excluded AND surfaced for the method note."""
    tbl, excl = gateway_cost._pair_rows(_summary(), set())
    assert not any(tbl["transport"] == "tproxy")
    (label, n_sizes), = excl["missing_side"].items()
    assert "TPROXY" in label and n_sizes == 2


def test_make_renders_and_discloses():
    """End-to-end: figure renders, exclusions land in the method note, takeaway is computed."""
    runs = pd.DataFrame({
        "scenario": ["matrix_dead_scg_4c"],
        "throughput_gbps": [0.0],
        "messages": [0],
    })
    bundle = _bundle(_summary(), runs=runs)
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp), formats=("png",))
        gateway_cost.make(bundle, saver)  # must not raise
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F6 unexpectedly skipped: {entry.get('skipped')}"
    assert entry["id"] == "F6"
    chrome = {c["kind"]: [] for c in entry["chrome"]}
    for c in entry["chrome"]:
        chrome[c["kind"]].append(c["text"])
    methods = " ".join(chrome.get("method", []))
    assert "lacks one topology entirely" in methods
    assert "zero-work repetition" in methods
    assert "SAME matched pair" in methods
    takeaways = " ".join(chrome.get("takeaway", []))
    assert "harness-clean" in takeaways and "%" in takeaways


if __name__ == "__main__":
    test_all_bound_group_makes_no_delta_claim()
    test_dead_repeat_scenario_excluded_from_pairing()
    test_dead_repeat_detection_via_runs_frame()
    test_annotation_is_the_drawn_pair()
    test_bimodal_sweep_falls_back_to_lowest_pair()
    test_missing_topology_is_disclosed_not_silent()
    test_make_renders_and_discloses()
    print("ok")
