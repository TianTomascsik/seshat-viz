"""
Regression tests for F13 (suite coverage, scenario-coverage fallback mode).

Guards two invariants of F13:

* F13-1 — the denominator was silently `executed + skip-logged`, letting a run whose skip
  records were lost from accounting (e.g. across `--resume`) masquerade as ~complete. F13
  must (a) label the plan-less denominator as *recorded only / upper bound*, and (b) when a
  plan manifest (plan.json / planned_scenarios.txt) is present, use the plan as denominator
  and count planned-but-unaccounted scenarios explicitly (excluding everything already
  executed or skip-logged).
* F13-2 — the harness-limited share used all executed rows as denominator, counting rows
  that carry NO saturation determination (NA) as not-limited. The share must be computed
  over determination-carrying rows only, with the NA count disclosed.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f13_coverage.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import coverage  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402


def _bundle(summary: pd.DataFrame, skipped: pd.DataFrame, run_dir: Path) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=run_dir,
        summary=summary,
        runs=empty,
        sysmetrics=empty,
        saturation=empty,
        skipped=skipped,
        sysinfo={"hostname": "test"},
    )


def _summary(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame({
        "scenario": [f"matrix_tls13_tcp_scg_1KB_{i}c" for i in range(1, n + 1)],
        "family": ["matrix"] * n,
        "throughput_gbps_mean": [1.0] * n,
    })


def _skipped() -> pd.DataFrame:
    return pd.DataFrame({
        "scenario": ["matrix_tls13_shm_scg_1KB_64c", "matrix_lat_tls13_shm_scg_1KB_1c"],
        "reason": ["gateway run did not complete", "gateway run did not complete"],
        "family": ["matrix", "matrix-latency"],
        "connections": [64, 1],
    })


def _render(bundle: RunBundle) -> dict:
    """Render F13 into a temp dir and return its manifest entry (chrome included)."""
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        coverage.make(bundle, saver)
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F13 unexpectedly skipped: {entry.get('skipped')}"
    assert entry["id"] == "F13"
    return entry


def _chrome(entry: dict, kind: str) -> str:
    return " ".join(c["text"] for c in entry.get("chrome", []) if c["kind"] == kind)


def test_no_plan_denominator_disclosed_as_recorded_upper_bound():
    """Without a plan manifest, F13 must not claim plan coverage: the takeaway/provenance
    say 'recorded' + 'upper bound' and a method note explains the invisible-skip caveat."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "20260101-000000"
        run_dir.mkdir()
        entry = _render(_bundle(_summary(8), _skipped(), run_dir))
    take = _chrome(entry, "takeaway")
    assert "8/10 recorded scenarios executed" in take
    assert "upper bound" in take
    assert "planned" not in take  # no plan → no plan claim
    assert "upper bound" in _chrome(entry, "provenance")
    method = _chrome(entry, "method")
    assert "no plan manifest" in method and "upper bound" in method


def test_plan_names_extend_denominator_and_exclude_recorded():
    """A plan.json with scenario names: the unaccounted pool is plan − executed −
    skip-logged (here exactly the 2 extra names), and the denominator becomes the plan."""
    summary, skipped = _summary(8), _skipped()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "20260101-000000"
        run_dir.mkdir()
        planned = (list(summary["scenario"]) + list(skipped["scenario"])
                   + ["matrix_lost_a_1c", "matrix_lost_b_1c"])
        (run_dir / "plan.json").write_text(json.dumps({"scenarios": planned}))
        entry = _render(_bundle(summary, skipped, run_dir))
    take = _chrome(entry, "takeaway")
    assert "8/12 planned scenarios executed" in take
    assert "2 unaccounted" in take
    assert "upper bound" not in take  # plan-based denominator is exact, not a bound
    assert "plan.json" in _chrome(entry, "method")
    assert "8/12" in _chrome(entry, "provenance")


def test_plan_total_only_and_inconsistent_plan_clamped():
    """Count-only manifest: total 12 → 2 unaccounted. A too-small (inconsistent) total
    must clamp to 0 unaccounted instead of going negative or shrinking the denominator."""
    for total, expect in ((12, "8/12 planned"), (6, "8/10 planned")):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "20260101-000000"
            run_dir.mkdir()
            (run_dir / "plan.json").write_text(json.dumps({"total": total}))
            entry = _render(_bundle(_summary(8), _skipped(), run_dir))
        assert expect in _chrome(entry, "takeaway"), (total, _chrome(entry, "takeaway"))


def test_harness_limited_share_uses_flagged_rows_only():
    """F13-2: NA determinations must not inflate the denominator — 2 limited of 3 flagged
    (not of 6 executed), with the 3 determination-free rows disclosed."""
    summary = _summary(6)
    summary["harness_limited"] = pd.array([True, True, False, pd.NA, pd.NA, pd.NA],
                                          dtype="boolean")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "20260101-000000"
        run_dir.mkdir()
        entry = _render(_bundle(summary, _skipped(), run_dir))
    take = _chrome(entry, "takeaway")
    assert "2/3 rows with a saturation determination are harness-limited" in take
    assert "(3 rows carry none)" in take
    assert "2/6" not in take


def test_load_plan_txt_variant_and_absent():
    """planned_scenarios.txt (one name per line) parses; an empty run dir yields None."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "20260101-000000"
        run_dir.mkdir()
        assert coverage._load_plan(run_dir) is None
        (run_dir / "planned_scenarios.txt").write_text("a_1c\nb_1c\n\nc_1c\n")
        total, names, src = coverage._load_plan(run_dir)
        assert total == 3 and names == {"a_1c", "b_1c", "c_1c"}
        assert src.name == "planned_scenarios.txt"


if __name__ == "__main__":
    test_no_plan_denominator_disclosed_as_recorded_upper_bound()
    test_plan_names_extend_denominator_and_exclude_recorded()
    test_plan_total_only_and_inconsistent_plan_clamped()
    test_harness_limited_share_uses_flagged_rows_only()
    test_load_plan_txt_variant_and_absent()
    print("ok")
