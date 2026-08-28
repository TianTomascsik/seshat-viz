"""
Regression tests for F1 (throughput–latency landscape).

Guards the headline invariants of F1:
- the sustained-blast slice must exclude ``paced_*`` (rate-capped latency runs) and
  ``shmzc_*`` (zero-copy slot microbenchmarks) rows, so neither can fabricate the point
  cloud's low-latency frontier nor crown a callout;
- the extreme callouts must crown trusted rows WITH a gateway in the path
  (``n_gateways >= 1``) — loopback baselines are plotted but never headline.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f01_landscape.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import landscape  # noqa: E402
from seshat_viz.loader import RunBundle, transport_label  # noqa: E402


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
    """Saver that snapshots the main axes' annotations/limits before the figure closes."""

    def save(self, fig, name, *, fig_id="", title=""):
        ax = fig.get_axes()[0]
        self.callouts = [t.get_text() for t in ax.texts]  # only the extreme annotations
        self.xlim = ax.get_xlim()
        self.n_points = len(ax.collections)  # one scatter call per plotted row (no errorbar)
        return super().save(fig, name, fig_id=fig_id, title=title)


def _landscape_summary() -> pd.DataFrame:
    """The crown-hazard shape in miniature: leaked paced/shmzc rows holding the
    global extremes, trusted loopback rows holding the leak-free extremes, a faster
    harness-limited gateway row, and the two honest gateway rows that must crown."""
    cols = ("scenario", "transport", "protocol", "message_bytes", "connections",
            "n_gateways", "harness_limited", "throughput_gbps_mean", "latency_p99_us_mean")
    rows = [
        # leaked classes — must not be plotted at all
        ("paced_ale_tls13_udp_1400B_direct_1c", "udp", "tls/1.3", 1400, 1, 1, False, 0.33, 12.5),
        ("shmzc_shm_slot_tput_64KB", "shm", "none", 65536, 1, 1, False, 60.0, 3081.6),
        # trusted loopback (n_gateways=0) — plotted, but must never headline
        ("iface_tcp_loopback_throughput_64KB_1c", "tcp", "none", 65536, 1, 0, False, 44.0, 373.0),
        ("iface_tcp_loopback_throughput_4096B_1c", "tcp", "none", 4096, 1, 0, False, 38.5, 64.8),
        # harness-limited gateway row faster than every honest row — must never headline
        ("matrix_routing_tcp_16KB_scg_1c", "tcp", "none", 16384, 1, 2, True, 47.0, 900.0),
        # the honest gateway crowns
        ("matrix_routing_uds_unix_1KB_scg_1c", "uds", "none", 1024, 1, 2, False, 39.7, 1171.0),
        ("matrix_routing_uds_unix_1KB_direct_1c", "uds", "none", 1024, 1, 1, False, 39.3, 167.5),
        # two encrypted rows so the zoom panel renders
        ("matrix_tls13_tcp_16KB_scg_1c", "tcp", "tls/1.3", 16384, 1, 2, False, 8.0, 500.0),
        ("matrix_tls12_tcp_16KB_scg_1c", "tcp", "tls/1.2", 16384, 1, 2, False, 6.0, 600.0),
    ]
    return pd.DataFrame(rows, columns=cols)


def test_callout_pool_requires_gateway_in_path():
    """_callout_pool keeps only trusted n_gateways>=1 rows, degrading gracefully."""
    d = pd.DataFrame({
        "scenario": ["loopback", "gw", "limited"],
        "_hstate": ["trusted", "trusted", "limited"],
        "n_gateways": [0, 2, 1],
    })
    pool = landscape._callout_pool(d)
    assert list(pool["scenario"]) == ["gw"]

    # no trusted gateway row → fall back to trusted-any (loopback), not the limited row
    pool = landscape._callout_pool(d[d["scenario"] != "gw"])
    assert list(pool["scenario"]) == ["loopback"]

    # nothing trusted at all → full slice
    pool = landscape._callout_pool(d[d["_hstate"] == "limited"])
    assert list(pool["scenario"]) == ["limited"]

    # n_gateways column absent → trusted-any (guard cannot apply)
    pool = landscape._callout_pool(d.drop(columns=["n_gateways"]))
    assert set(pool["scenario"]) == {"loopback", "gw"}


def test_f1_crowns_gateway_rows_not_loopback_or_microbench():
    """End-to-end: both rendered callouts crown the UDS gateway rows — not the leaked
    shmzc 60 Gbps / paced 12.5 µs rows and not the faster trusted loopback rows."""
    bundle = _bundle(_landscape_summary())
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CapturingSaver(Path(tmp), formats=("png",))
        landscape.make(bundle, saver)
    assert "skipped" not in saver.manifest[-1]
    callouts = [t for t in saver.callouts if "Gbps @" in t]
    assert len(callouts) == 2, f"expected 2 extreme callouts, got: {saver.callouts}"
    for text in callouts:
        assert text.startswith(transport_label("uds")), f"non-gateway row crowned: {text!r}"
    joined = "\n".join(callouts)
    assert "60 Gbps" not in joined      # shmzc microbenchmark max
    assert "12.5" not in joined         # paced pacing-rate latency min
    assert transport_label("shm") not in joined
    assert transport_label("tcp") not in joined  # loopback / harness-limited rows


def test_f1_point_cloud_excludes_paced_and_shmzc():
    """The paced/shmzc rows are absent from the plot itself: point count matches the
    blast-only slice and the inverted log x-axis is not stretched to the paced 12.5 µs."""
    bundle = _bundle(_landscape_summary())
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CapturingSaver(Path(tmp), formats=("png",))
        landscape.make(bundle, saver)
    assert saver.n_points == 7  # 9 synthetic rows minus the paced_ and shmzc_ leaks
    # x-limits derive from the plotted data (min/2 on the right, inverted axis): with the
    # leak the right edge would be 12.5/2; blast-only data floors it at 64.8/2 = 32.4 µs.
    assert min(saver.xlim) > 30.0, f"x-axis stretched by leaked sub-blast rows: {saver.xlim}"


def test_f1_method_note_discloses_callout_basis():
    """The method note states the crown rule so the figure is honest about why the callout
    is not the visually-topmost (loopback/harness-limited) point."""
    bundle = _bundle(_landscape_summary())
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CapturingSaver(Path(tmp), formats=("png",))
        landscape.make(bundle, saver)
    chrome = saver.manifest[-1].get("chrome", [])
    methods = [c["text"] for c in chrome if c.get("kind") == "method"]
    assert methods and "gateway-in-path" in methods[0]


if __name__ == "__main__":
    test_callout_pool_requires_gateway_in_path()
    test_f1_crowns_gateway_rows_not_loopback_or_microbench()
    test_f1_point_cloud_excludes_paced_and_shmzc()
    test_f1_method_note_discloses_callout_basis()
    print("ok")
