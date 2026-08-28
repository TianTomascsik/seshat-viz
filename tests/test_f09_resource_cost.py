"""
Regression tests for F9 (resource cost of security).

Guards the user-scope-demoted perf-counter invariant: an unprivileged
perf run (perf_event_paranoid >= 2) silently demotes every event to ':u', so the
cycles/byte ladder compares *where* work runs (user vs kernel) instead of what it costs
— kernel-offload rungs read near-zero while userspace crypto reads full cost — and
ctx-switches is definitionally zero. F9 must withhold those panels behind a labelled
placeholder, keep the cache-miss *rate* only as a user-scope metric, and its method
note must describe the actual panel state (placeholder wording only when panels really
fell back, never a hardcoded "shown as placeholders" over populated panels).

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f09_resource_cost.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import resource_cost  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402

# One shared task-clock so the implied clock rate (cycles / task-clock) is easy to pin:
# 35 s of CPU → 3.5e10 "cycle slots" per GHz.
_TASK_CLOCK_MS = 35_000.0
_DEMOTED_CYCLES = 1.4e9   # implies 0.04 GHz — the ':u' fingerprint (kernel share missing)
_PLAUSIBLE_CYCLES = 8.8e10  # implies ~2.5 GHz — a full kernel+user count


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


def _matrix_summary(*, perf: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    F9's slice: matrix / TCP / 1c, one shared payload size, >=3 encrypted protocols (so
    size_match_for_protocol_compare keeps the size). `perf` selects the counter state:
    None (procfs run, no counters), 'demoted' (user-scope ':u' fingerprints: implausible
    implied clock + all-zero ctx-switches), or 'kernel' (plausible clock, nonzero ctxsw).
    Returns (summary, runs) — runs carries the bytes/messages work denominators.
    """
    srows, rrows = [], []
    for protocol, tag in (("none", "routing"), ("tls/1.2", "tls12"),
                          ("tls/1.3", "tls13"), ("ktls/1.3", "ktls13")):
        for chain in ("direct", "scg"):
            scen = f"matrix_{tag}_tcp_tcp_4096B_{chain}_1c"
            row = {
                "scenario": scen,
                "family": "matrix",
                "transport": "tcp",
                "protocol": protocol,
                "message_bytes": 4096,
                "connections": 1,
                "chain": chain,
                "throughput_gbps_mean": 10.0,
                "gbps_per_core": 8.0,
                "rss_peak_kib": 24_000.0,
            }
            if perf is not None:
                row.update({
                    "perf_cycles": _DEMOTED_CYCLES if perf == "demoted" else _PLAUSIBLE_CYCLES,
                    "perf_task_clock_ms": _TASK_CLOCK_MS,
                    "perf_context_switches": 0.0 if perf == "demoted" else 1500.0,
                    "perf_cache_references": 1.0e9,
                    "perf_cache_misses": 4.0e7,
                })
            srows.append(row)
            rrows.append({"scenario": scen, "bytes": 4.0e10, "messages": 1.0e7})
    return pd.DataFrame(srows), pd.DataFrame(rrows)


def _chrome(saver: theme.Saver, kind: str) -> str:
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F9 unexpectedly skipped: {entry.get('skipped')}"
    assert entry["id"] == "F9"
    return "\n".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == kind)


def _render(perf: str | None) -> tuple[theme.Saver, list[str]]:
    """Run F9's make() on a synthetic bundle, recording which panels were scope-withheld."""
    summary, runs = _matrix_summary(perf=perf)
    bundle = _bundle(summary, runs)
    withheld: list[str] = []
    orig = resource_cost._scope_placeholder
    resource_cost._scope_placeholder = lambda ax, label: (withheld.append(label), orig(ax, label))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            saver = theme.Saver(Path(tmp))
            resource_cost.make(bundle, saver)
    finally:
        resource_cost._scope_placeholder = orig
    return saver, withheld


# -- the scope detector itself ---------------------------------------------------------

def test_detector_flags_implausible_implied_clock():
    """cycles/task-clock ~0.04 GHz on most rows = the ':u' demotion fingerprint."""
    summary, _ = _matrix_summary(perf="demoted")
    assert resource_cost._perf_user_scope_only(summary) is True


def test_detector_flags_all_zero_ctx_switches():
    """Even with a plausible clock, all-zero ctx-switches across the slice means ':u'."""
    summary, _ = _matrix_summary(perf="kernel")
    summary["perf_context_switches"] = 0.0
    assert resource_cost._perf_user_scope_only(summary) is True


def test_detector_accepts_kernel_scope_counters():
    summary, _ = _matrix_summary(perf="kernel")
    assert resource_cost._perf_user_scope_only(summary) is False


def test_detector_ignores_missing_counters():
    """No counters at all is the perf_missing (placeholder) path, not a scope problem."""
    summary, _ = _matrix_summary(perf=None)
    assert resource_cost._perf_user_scope_only(summary) is False
    summary["perf_cycles"] = np.nan  # column present but empty behaves the same
    assert resource_cost._perf_user_scope_only(summary) is False


# -- end-to-end panel + method-note behavior -------------------------------------------

def test_user_scope_run_withholds_cycles_and_ctxsw_panels():
    """The 110x-artifact scenario: cycles/byte and ctx-switches must NOT render
    as bar measurements; the method note must disclose the scope instead of claiming
    the panels are procfs placeholders."""
    saver, withheld = _render(perf="demoted")
    assert any("cycles" in w for w in withheld), f"cycles/byte not withheld: {withheld}"
    assert any("ctx-switches" in w for w in withheld), f"ctx-switches not withheld: {withheld}"
    method = _chrome(saver, "method")
    assert "user-scope only" in method
    assert "shown as placeholders" not in method  # the F9-3 self-contradiction
    assert "user-scope only" in _chrome(saver, "provenance")


def test_procfs_run_method_note_names_the_placeholders():
    """No counters → placeholder wording is accurate and names all three panels."""
    saver, withheld = _render(perf=None)
    assert withheld == []
    method = _chrome(saver, "method")
    assert "shown as placeholders" in method
    for name in ("cycles/byte", "cache-miss", "ctx-switches"):
        assert name in method, f"{name} missing from method note: {method}"
    assert "need a perf run" in _chrome(saver, "provenance")
    assert "user-scope" not in method


def test_kernel_scope_run_renders_counters_as_measurements():
    """Trustworthy counters render as bars, with a perf-stat attribution and neither the
    placeholder claim nor the user-scope caveat."""
    saver, withheld = _render(perf="kernel")
    assert withheld == []
    method = _chrome(saver, "method")
    assert "measured by perf stat" in method
    assert "shown as placeholders" not in method
    assert "user-scope" not in method


if __name__ == "__main__":
    test_detector_flags_implausible_implied_clock()
    test_detector_flags_all_zero_ctx_switches()
    test_detector_accepts_kernel_scope_counters()
    test_detector_ignores_missing_counters()
    test_user_scope_run_withholds_cycles_and_ctxsw_panels()
    test_procfs_run_method_note_names_the_placeholders()
    test_kernel_scope_run_renders_counters_as_measurements()
    print("ok")
