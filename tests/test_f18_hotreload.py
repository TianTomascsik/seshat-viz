"""
Regression tests for F18 (hot-reload robustness).

Guards two invariants of F18:

* F18-1 — the takeaway multiplied the reload-event count by the repetition count
  ("~288 reload events"), but the harness arms exactly ONE reload per scenario (the
  reload timer fires once, during the first measurement run). Events must equal
  scenarios, never scenarios × runs.
* F18-2 — retention was the scenario mean over all measurement runs, of which only the
  first contains the reload, diluting the reload window 2:1 with steady state. When
  per-run data is available the figure must report the reload run against the same
  scenario's reload-free runs; when it is not, the dilution must be disclosed.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f18_hotreload.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import hotreload  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402

# One reload run at 8.0 Gbps against two reload-free runs at 10.0 Gbps: the undiluted
# (reload-run) retention is 80%, the diluted 3-run mean vs the 10.0 matrix baseline is 93%.
_RELOAD_TP, _STEADY_TP, _BASELINE_TP = 8.0, 10.0, 10.0
_SCENARIOS = [
    ("hotreload_tls12_tcp_add_connection_saturation_1c", "tls/1.2", 1, "add_connection"),
    ("hotreload_tls12_tcp_invalid_config_saturation_16c", "tls/1.2", 16, "invalid_config"),
    ("hotreload_ktls13_tcp_add_connection_saturation_1c", "ktls/1.3", 1, "add_connection"),
    ("hotreload_ktls13_tcp_invalid_config_saturation_16c", "ktls/1.3", 16, "invalid_config"),
]


def _summary(*, with_loss_counters: bool = True, total_lost: int = 0) -> pd.DataFrame:
    rows = []
    for name, proto, conns, trig in _SCENARIOS:
        row = {
            "scenario": name,
            "family": "hotreload",
            "protocol": proto,
            "transport": "tcp",
            "message_bytes": 4096,
            "connections": conns,
            "chain": "direct",
            "reload_trigger": trig,
            "reload_load": "saturation",
            "throughput_gbps_mean": (_RELOAD_TP + 2 * _STEADY_TP) / 3,  # diluted 3-run mean
        }
        if with_loss_counters:
            row.update({"total_lost": total_lost, "integrity_failures": 0,
                        "boundary_violations": 0})
        rows.append(row)
        # Matched matrix baseline row (same protocol/transport/size/connections, chain=direct).
        rows.append({
            "scenario": f"matrix_{proto}_tcp_{conns}c",
            "family": "matrix",
            "protocol": proto,
            "transport": "tcp",
            "message_bytes": 4096,
            "connections": conns,
            "chain": "direct",
            "reload_trigger": np.nan,
            "reload_load": np.nan,
            "throughput_gbps_mean": _BASELINE_TP,
        })
    return pd.DataFrame(rows)


def _runs(scenario_names=None) -> pd.DataFrame:
    """Per-run rows: run 1 carries the reload dip, runs 2-3 are steady state."""
    names = scenario_names if scenario_names is not None else [s[0] for s in _SCENARIOS]
    rows = []
    for name in names:
        for run, tp in ((1, _RELOAD_TP), (2, _STEADY_TP), (3, _STEADY_TP)):
            rows.append({"scenario": name, "run": run, "throughput_gbps": tp})
    return pd.DataFrame(rows)


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


def _render(bundle: RunBundle) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        hotreload.make(bundle, saver)
    return saver.manifest[-1]


def _chrome(entry: dict, kind: str) -> str:
    return " ".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == kind)


def test_f18_event_count_is_one_per_scenario():
    """F18-1: 4 scenarios → '4 reload events (one per scenario)', never ~12 (scenarios × runs)."""
    entry = _render(_bundle(_summary(), _runs()))
    assert "skipped" not in entry, f"F18 unexpectedly skipped: {entry.get('skipped')}"
    takeaway = _chrome(entry, "takeaway")
    assert "4 reload events (one per scenario)" in takeaway, takeaway
    assert "12" not in takeaway and "~" not in takeaway, takeaway


def test_f18_retention_uses_reload_run_not_diluted_mean():
    """F18-2: with per-run data the median must be the reload-run 80%, not the diluted 93%."""
    entry = _render(_bundle(_summary(), _runs()))
    takeaway = _chrome(entry, "takeaway")
    assert "80%" in takeaway, takeaway
    assert "93%" not in takeaway, takeaway
    method = _chrome(entry, "method")
    assert "reload-free runs" in method, method


def test_f18_fallback_without_runs_discloses_dilution():
    """No per-run data → the diluted matrix-baseline metric renders, with the dilution disclosed."""
    entry = _render(_bundle(_summary()))  # runs empty
    assert "skipped" not in entry, f"F18 unexpectedly skipped: {entry.get('skipped')}"
    takeaway = _chrome(entry, "takeaway")
    assert "93%" in takeaway, takeaway  # diluted 3-run mean vs baseline
    assert "diluted" in _chrome(entry, "method")


def test_f18_partial_run_coverage_falls_back_whole_figure():
    """Per-run data for only some scenarios must not mix metrics — all cells fall back."""
    partial = _runs([s[0] for s in _SCENARIOS[:2]])
    entry = _render(_bundle(_summary(), partial))
    takeaway = _chrome(entry, "takeaway")
    assert "93%" in takeaway and "80%" not in takeaway, takeaway


def test_f18_zero_loss_claim_needs_counter():
    """A missing loss counter must not be presented as '0 frames lost'."""
    entry = _render(_bundle(_summary(with_loss_counters=False), _runs()))
    takeaway = _chrome(entry, "takeaway")
    assert "0 frames lost" not in takeaway, takeaway


def test_f18_nonzero_loss_drops_nondisruptive_claim():
    entry = _render(_bundle(_summary(total_lost=7), _runs()))
    takeaway = _chrome(entry, "takeaway")
    assert "non-disruptive" not in takeaway and "0 frames lost" not in takeaway, takeaway


def test_reload_run_retention_helper_guards():
    """Single-run and zero-work-steady scenarios are omitted; the normal case is exact."""
    runs = pd.DataFrame({
        "scenario": ["a", "a", "a", "b", "c", "c"],
        "run": [1, 2, 3, 1, 1, 2],
        "throughput_gbps": [8.0, 10.0, 10.0, 9.0, 8.0, 0.0],
    })
    scn = pd.Series(["a", "b", "c"])
    ret = hotreload._reload_run_retention(runs, scn)
    assert set(ret.index) == {"a"}          # b: single run; c: zero-work steady run
    assert abs(ret["a"] - 80.0) < 1e-9
    assert hotreload._reload_run_retention(pd.DataFrame(), scn).empty


if __name__ == "__main__":
    test_f18_event_count_is_one_per_scenario()
    test_f18_retention_uses_reload_run_not_diluted_mean()
    test_f18_fallback_without_runs_discloses_dilution()
    test_f18_partial_run_coverage_falls_back_whole_figure()
    test_f18_zero_loss_claim_needs_counter()
    test_f18_nonzero_loss_drops_nondisruptive_claim()
    test_reload_run_retention_helper_guards()
    print("ok")
