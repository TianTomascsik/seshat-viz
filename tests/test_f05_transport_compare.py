"""
Regression tests for F5 (transport radar + absolute bars).

Guards the invariants of F5:
  * F5-1 (P0): the "low loss" radar axis crowned the ONLY lossy transport "best" — a
    ratio-to-best over positive values used the lossy row's own 43.67% as the reference,
    so every vertex (zero-loss and 43%-loss alike) landed at 1.0. loss_pct now maps to
    the delivered fraction (1 - loss/100): measured loss pulls the vertex inward.
  * F5-3 (P1): a transport with no measurement on an axis had its vertex invented by
    fillna(median) (UDP got a third-best "low jitter" score with zero data behind it).
    Missing values now stay NaN — a polygon gap — and the method note names the gap.
  * F5-2 (P1): the shmzc_* zero-copy family must never be averaged into the SHM blast
    bar (derive.throughput_scenarios_only guards it upstream; the render must agree).
  * F5-4 (P2): when every plotted row is harness-limited the method note must say the
    absolute Gbps panel is a load-generator lower bound.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f05_transport_compare.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import derive  # noqa: E402
from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import transport_compare  # noqa: E402
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


class _CapturingSaver(theme.Saver):
    """Saver that snapshots the rendered artists before the figure is closed."""

    def __init__(self, out_dir: Path):
        super().__init__(out_dir)
        self.bar_heights: list[float] = []
        self.radar_lines: dict[str, np.ndarray] = {}

    def save(self, fig, name, *, fig_id="", title=""):
        axr, axb = fig.axes[0], fig.axes[1]
        self.bar_heights = [p.get_height() for p in axb.patches]
        self.radar_lines = {ln.get_label(): np.asarray(ln.get_ydata(), dtype=float)
                            for ln in axr.get_lines() if not ln.get_label().startswith("_")}
        return super().save(fig, name, fig_id=fig_id, title=title)


def _routing_summary() -> pd.DataFrame:
    """The real-run shape in miniature: three transports at one shared routing size/1c,
    all harness-limited; UDP is lossy and has NO jitter measurement; a shmzc zero-copy
    row survives every F5 slice pin except the blast-family guard (the D1-5 leak)."""
    rows = [
        # scenario, transport, tput, p99, jitter, cpu, loss, harness_limited
        ("iface_shm_throughput_16KB_1c", "shm", 30.0, 5600.0, 8.0, 260.0, 0.0, True),
        ("matrix_routing_shm_shm_16KB_direct_1c", "shm", 31.0, 5500.0, np.nan, 250.0, 0.0, True),
        ("matrix_routing_tcp_tcp_16KB_direct_1c", "tcp", 46.0, 1400.0, 4.0, 200.0, 0.0, True),
        ("matrix_routing_udp_udp_16KB_direct_1c", "udp", 20.0, 4700.0, np.nan, 150.0, 43.67, True),
        # The leak: zero-copy microbenchmark, mode 'throughput', all slice pins pass.
        ("shmzc_shm_scale_16KB_1c", "shm", 47.0, 4400.0, np.nan, 100.0, 0.0, False),
    ]
    return pd.DataFrame(
        [
            {
                "scenario": s,
                "mode": "throughput",
                "transport": t,
                "protocol": "none",
                "message_bytes": 16384,
                "connections": 1,
                "chain": "direct",
                "datapath": "gateway",
                "n_gateways": 1,
                "throughput_gbps_mean": tput,
                "latency_p99_us_mean": p99,
                "jitter_us_mean": jit,
                "cpu_pct_mean": cpu,
                "loss_pct": loss,
                "harness_limited": hl,
            }
            for (s, t, tput, p99, jit, cpu, loss, hl) in rows
        ]
    )


# ---------------------------------------------------------------------------------
# _axis_norm — the normalization rules themselves (F5-1 / F5-3)
# ---------------------------------------------------------------------------------

def test_loss_axis_maps_measured_loss_inward():
    """F5-1: v=[0,0,0,0,43.67] must keep the zero-loss transports at 1.0 and pull the
    lossy one to its delivered fraction — NOT crown it 'best' at 1.0."""
    scaled = transport_compare._axis_norm([0.0, 0.0, 0.0, 0.0, 43.67], False, "loss_pct")
    assert np.allclose(scaled[:4], 1.0)
    assert abs(scaled[4] - (1.0 - 43.67 / 100.0)) < 1e-9
    assert scaled[4] < 1.0


def test_lower_is_better_zero_best_generic():
    """Generic lower-is-better with an achieved 0: only 0 earns 1.0; positives collapse."""
    scaled = transport_compare._axis_norm([0.0, 5.0], False, "latency_p99_us_mean")
    assert scaled[0] == 1.0
    assert scaled[1] == 0.0


def test_lower_is_better_ratio_preserved():
    """The baseline min/v ratio scheme is unchanged when every value is positive."""
    scaled = transport_compare._axis_norm([4.0, 8.0], False, "jitter_us_mean")
    assert np.allclose(scaled, [1.0, 0.5])


def test_missing_metric_stays_nan():
    """F5-3: NaN must survive normalization (a gap), never be imputed or become 0."""
    scaled = transport_compare._axis_norm([4.0, np.nan, 6.0], False, "jitter_us_mean")
    assert scaled[0] == 1.0
    assert np.isnan(scaled[1])
    assert abs(scaled[2] - 4.0 / 6.0) < 1e-9
    # higher-is-better path too
    scaled = transport_compare._axis_norm([np.nan, 10.0, 5.0], True, "throughput_gbps_mean")
    assert np.isnan(scaled[0]) and scaled[1] == 1.0 and scaled[2] == 0.5


# ---------------------------------------------------------------------------------
# blast pool — the shmzc zero-copy family stays out of the SHM mean (F5-2)
# ---------------------------------------------------------------------------------

def test_blast_pool_excludes_shmzc():
    d = derive.throughput_scenarios_only(_routing_summary())
    assert not d["scenario"].astype(str).str.startswith("shmzc_").any()
    assert len(d) == 4


# ---------------------------------------------------------------------------------
# end-to-end render — gaps drawn, leak excluded, disclosures present
# ---------------------------------------------------------------------------------

def test_f5_renders_honest_radar_and_bars():
    bundle = _bundle(_routing_summary())
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CapturingSaver(Path(tmp))
        transport_compare.make(bundle, saver)
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F5 unexpectedly skipped: {entry.get('skipped')}"
    assert entry["id"] == "F5"

    # F5-2: the SHM bar is the blast-only mean (30+31)/2, not the shmzc-inflated 36.
    # Bars follow the fixed display order SHM, TCP, UDP.
    assert abs(saver.bar_heights[0] - 30.5) < 1e-9
    assert abs(saver.bar_heights[1] - 46.0) < 1e-9
    assert abs(saver.bar_heights[2] - 20.0) < 1e-9

    # Radar vertices follow _AXES order minus unavailable columns; here gbps_per_core is
    # absent, so: throughput, low latency, low jitter, low CPU use, low loss (+ closure).
    udp = saver.radar_lines["UDP"]
    assert len(udp) == 6
    # F5-3: UDP's jitter vertex is a gap, not a median-imputed value.
    assert np.isnan(udp[2])
    # F5-1: UDP's loss vertex sits at its delivered fraction, well inside the outer ring.
    assert abs(udp[4] - (1.0 - 43.67 / 100.0)) < 1e-9
    # Zero-loss transports do sit at the outer ring on the loss axis.
    assert saver.radar_lines["TCP"][4] == 1.0 and saver.radar_lines["SHM"][4] == 1.0

    # F5-3/F5-4 disclosures, computed from the plotted slice at render time.
    method = " ".join(c["text"] for c in entry["chrome"] if c["kind"] == "method")
    assert "delivered fraction" in method
    assert "unmeasured" in method and "UDP" in method
    assert "harness-limited" in method and "lower bound" in method


def test_f5_partial_harness_limit_counts_rows():
    """A mixed harness_limited slice reports the honest n/m count, not the blanket claim."""
    df = _routing_summary()
    df.loc[df["scenario"] == "matrix_routing_tcp_tcp_16KB_direct_1c", "harness_limited"] = False
    bundle = _bundle(df)
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CapturingSaver(Path(tmp))
        transport_compare.make(bundle, saver)
    entry = saver.manifest[-1]
    method = " ".join(c["text"] for c in entry["chrome"] if c["kind"] == "method")
    assert "3/4 plotted rows are harness-limited" in method


if __name__ == "__main__":
    test_loss_axis_maps_measured_loss_inward()
    test_lower_is_better_zero_best_generic()
    test_lower_is_better_ratio_preserved()
    test_missing_metric_stays_nan()
    test_blast_pool_excludes_shmzc()
    test_f5_renders_honest_radar_and_bars()
    test_f5_partial_harness_limit_counts_rows()
    print("ok")
