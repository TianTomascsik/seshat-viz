"""
Regression tests for F11 (measurement validity: headroom & bottleneck attribution).

Guards the honesty of the figure's data pool and disclosures:

  * A FAILED ceiling probe (ceiling_gbps <= 0 → headroom 0) must be excluded from the
    headroom pool and counted in the headline, not binned as a "< 1× probe under-read":
    a headroom of exactly 0 is unplottable on the log ECDF (matplotlib drops x=0 while
    the axis lower bound collapses to 1e-3, stretching empty decades) and renders as a
    zero-length bar in the worst slice.
  * Rows that never received a ceiling probe (headroom absent) must be disclosed — the
    "run-wide" claim otherwise silently means "the probed subset" while the unprobed
    throughput/paced rows feed other figures (F1-F22).
  * The sub-1× mechanism note must cover encrypted paths, not routing only — a third of
    the sub-1× band in a full run is TLS/kTLS at 64B.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f11_validity.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import validity  # noqa: E402
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


def _row(scenario: str, *, transport: str, protocol: str, size: int, tput: float,
         ceiling: float, mode: str = "throughput", harness_limited: bool = False,
         bottleneck: str = "", dut: str = "scg") -> dict:
    headroom = ceiling / tput if (pd.notna(ceiling) and tput) else np.nan
    return {
        "scenario": scenario,
        "transport": transport,
        "protocol": protocol,
        "message_bytes": size,
        "connections": 1,
        "n_gateways": 1,
        "chain": "scg",
        "mode": mode,
        "datapath": "gateway",
        "dut": dut,
        "throughput_gbps_mean": tput,
        "ceiling_gbps": ceiling,
        "headroom": headroom,
        "harness_limited": harness_limited,
        "bottleneck": bottleneck,
    }


def _validity_summary(*, corrupt: bool = True, n_unassessed: int = 0) -> pd.DataFrame:
    """Five well-formed headroom rows spanning the three bins, optionally plus the real-run
    pathology (a 0-Gbps ceiling probe on a 44-Gbps measurement) and rows with no probe."""
    rows = []
    if corrupt:
        # The shmzc_shm_slot_tput_1MB shape: probe returned nothing, measurement is real.
        rows.append(_row("shmzc_shm_slot_tput_1MB", transport="shm", protocol="none",
                         size=1048576, tput=44.18, ceiling=0.0, bottleneck="scg-cpu"))
    # Genuine probe under-read: encrypted 64B beating the 2-thread null probe.
    rows.append(_row("matrix_tls13_tcp_64B", transport="tcp", protocol="tls/1.3", size=64,
                     tput=1.0, ceiling=0.9, harness_limited=True, bottleneck="harness-io"))
    # Below the gate but plausible.
    rows.append(_row("matrix_none_tcp_16KB", transport="tcp", protocol="none", size=16384,
                     tput=5.0, ceiling=10.0, harness_limited=True, bottleneck="harness-io"))
    # Passing rows.
    rows.append(_row("matrix_tls13_tcp_16KB", transport="tcp", protocol="tls/1.3", size=16384,
                     tput=2.0, ceiling=7.0))
    rows.append(_row("matrix_ktls13_tcp_16KB", transport="tcp", protocol="ktls/1.3", size=16384,
                     tput=2.0, ceiling=10.0))
    rows.append(_row("matrix_none_tproxy_64KB", transport="tproxy", protocol="none", size=65536,
                     tput=1.0, ceiling=12.0))
    # Rows with a measured throughput but no ceiling probe at all (headroom NaN) — half of
    # them genuine throughput measurements, half closed-loop pingpong.
    for i in range(n_unassessed):
        mode = "throughput" if i % 2 == 0 else "pingpong"
        rows.append(_row(f"hotreload_none_tcp_{i}", transport="tcp", protocol="none", size=4096,
                         tput=3.0, ceiling=np.nan, mode=mode))
    return pd.DataFrame(rows)


class _CaptureSaver(theme.Saver):
    """Saver that snapshots panel internals before `save` closes the figure."""

    def save(self, fig, name, *, fig_id="", title=""):
        ax_cdf, ax_bars = fig.get_axes()[:2]
        self.cdf_xlim = ax_cdf.get_xlim()
        self.cdf_title = ax_cdf.get_title()
        self.bar_widths = [p.get_width() for p in ax_bars.patches]
        self.bar_labels = [t.get_text() for t in ax_bars.get_yticklabels()]
        return super().save(fig, name, fig_id=fig_id, title=title)


def _chrome(saver: theme.Saver, kind: str) -> str:
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F11 unexpectedly skipped: {entry.get('skipped')}"
    return " ".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == kind)


def _render(summary: pd.DataFrame) -> _CaptureSaver:
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CaptureSaver(Path(tmp))
        validity.make(_bundle(summary), saver)
    return saver


def test_failed_ceiling_probe_excluded_from_pool():
    """A ceiling_gbps=0 row must leave every rendered element: not in the <1× bin, not a
    zero-length worst-slice bar, not the ECDF axis anchor — and be counted in the headline."""
    saver = _render(_validity_summary(corrupt=True))
    headline = _chrome(saver, "headline")
    # Counted as a probe failure, not as an under-read: only the genuine 0.9× row sits <1×.
    assert "1 failed ceiling probe excluded" in headline
    assert "(1 below 1×" in headline
    assert "3/5 clear" in headline  # pool is the 5 good rows, not 6
    # The corrupt row (transport shm) is out of the worst slice, and no bar is zero-length.
    assert not any(lbl.upper().startswith("SHM") for lbl in saver.bar_labels)
    assert len(saver.bar_labels) == 5
    assert min(saver.bar_widths) > 0
    # ECDF x-axis anchors at the true minimum (0.9×), not at the 1e-3 fallback for x=0.
    assert saver.cdf_xlim[0] > 0.5


def test_no_probe_failures_keeps_headline_clean():
    saver = _render(_validity_summary(corrupt=False))
    assert "failed ceiling probe" not in _chrome(saver, "headline")


def test_all_probes_failed_skips():
    df = _validity_summary(corrupt=False)
    df["ceiling_gbps"] = 0.0
    df["headroom"] = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        validity.make(_bundle(df), saver)
    entry = saver.manifest[-1]
    assert "ceiling" in str(entry.get("skipped", ""))


def test_unprobed_rows_disclosed():
    """Rows without any ceiling probe must be counted in the method note and the run-wide
    denominator must appear in the ECDF panel title; the takeaway scopes itself to the
    probed subset."""
    saver = _render(_validity_summary(corrupt=False, n_unassessed=4))
    method = _chrome(saver, "method")
    assert "4 rows without a ceiling probe" in method
    assert "(2 of them throughput/paced measurements)" in method
    assert "5 finite-headroom rows of 9 in the run" in saver.cdf_title
    assert "ceiling-probed" in _chrome(saver, "takeaway")


def test_fully_probed_run_has_no_unprobed_clause():
    saver = _render(_validity_summary(corrupt=False, n_unassessed=0))
    assert "without a ceiling probe" not in _chrome(saver, "method")


def test_sub1x_note_covers_encrypted_paths():
    """The sub-1× mechanism note must not scope the effect to routing/passthrough only."""
    saver = _render(_validity_summary())
    method = _chrome(saver, "method")
    assert "encrypted" in method and "routing" in method
    assert "On passthrough (routing) paths headroom" not in method


if __name__ == "__main__":
    test_failed_ceiling_probe_excluded_from_pool()
    test_no_probe_failures_keeps_headline_clean()
    test_all_probes_failed_skips()
    test_unprobed_rows_disclosed()
    test_fully_probed_run_has_no_unprobed_clause()
    test_sub1x_note_covers_encrypted_paths()
    print("ok")
