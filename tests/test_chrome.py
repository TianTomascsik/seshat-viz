"""
Unit tests for the --no-chrome figure-chrome switch (theme.set_chrome / set_headline /
footer helpers / Saver manifest / cli captions writer).

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_chrome.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (theme forces the Agg backend)


def _with_chrome(enabled: bool):
    """Context manager restoring the module-global chrome switch."""
    class _Ctx:
        def __enter__(self):
            self.prev = theme.chrome_enabled()
            theme.set_chrome(enabled)

        def __exit__(self, *exc):
            theme.set_chrome(self.prev)

    return _Ctx()


def test_headline_drawn_when_chrome_on():
    with _with_chrome(True):
        fig, ax = plt.subplots()
        theme.set_headline(fig, "Suptitle headline", y=1.02)
        assert fig.get_suptitle() == "Suptitle headline"
        theme.set_headline(fig, "Axes headline", ax=ax)
        assert ax.get_title() == "Axes headline"
        plt.close(fig)


def test_headline_suppressed_and_recorded_when_chrome_off():
    with _with_chrome(False):
        fig, ax = plt.subplots()
        theme.set_headline(fig, "Suptitle headline", y=1.02)
        theme.set_headline(fig, "Axes headline", ax=ax)
        assert fig.get_suptitle() == ""
        assert ax.get_title() == ""
        kinds = [r["kind"] for r in fig._seshat_chrome]
        texts = [r["text"] for r in fig._seshat_chrome]
        assert kinds == ["headline", "headline"]
        assert texts == ["Suptitle headline", "Axes headline"]
        plt.close(fig)


def test_strip_editorial_axis_label():
    # The "higher/lower is better" cue is editorial chrome, stripped for --no-chrome.
    strip = theme.strip_editorial_axis_label
    assert strip("p99 latency (µs, log) — lower is better →") == "p99 latency (µs, log)"
    assert strip("throughput (Gbps) — higher is better ↑") == "throughput (Gbps)"
    # A plain label is untouched.
    assert strip("connections (log)") == "connections (log)"


def test_saver_strips_editorial_labels_only_when_chrome_off(tmp_path):
    # Chrome on: the editorial cue survives.
    with _with_chrome(True):
        fig, ax = plt.subplots()
        ax.set_xlabel("p99 latency (µs, log) — lower is better →")
        theme.Saver(tmp_path, formats=("png",)).save(fig, "on")
    with _with_chrome(False):
        fig, ax = plt.subplots()
        ax.set_xlabel("p99 latency (µs, log) — lower is better →")
        ax.set_ylabel("throughput (Gbps) — higher is better ↑")
        # `save` closes the fig, but the local `ax` ref keeps the object readable afterward.
        theme.Saver(tmp_path, formats=("png",)).save(fig, "off")
        assert ax.get_xlabel() == "p99 latency (µs, log)"
        assert ax.get_ylabel() == "throughput (Gbps)"


def test_footers_suppressed_and_recorded():
    with _with_chrome(False):
        fig, _ax = plt.subplots()
        theme.add_provenance(fig, "host line")
        theme.add_method_note(fig, "method line")
        theme.add_takeaway(fig, "the point")
        assert len(fig.texts) == 0
        recorded = {r["kind"]: r["text"] for r in fig._seshat_chrome}
        assert recorded == {"provenance": "host line", "method": "method line",
                            "takeaway": "the point"}
        plt.close(fig)
    with _with_chrome(True):
        fig, _ax = plt.subplots()
        theme.add_provenance(fig, "host line")
        theme.add_method_note(fig, "method line")
        theme.add_takeaway(fig, "the point")
        assert len(fig.texts) == 3
        plt.close(fig)


def test_saver_manifest_carries_chrome():
    with _with_chrome(False), tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp), formats=("png",))
        fig, _ax = plt.subplots()
        theme.set_headline(fig, "H")
        theme.add_takeaway(fig, "T")
        saver.save(fig, "test_fig", fig_id="F0", title="H")
        entry = saver.manifest[0]
        assert (Path(tmp) / "test_fig.png").exists()
        assert {r["kind"] for r in entry["chrome"]} == {"headline", "takeaway"}


def test_cli_captions_writer():
    from seshat_viz.cli import _write_captions

    with _with_chrome(False), tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp), formats=("png",))
        fig, _ax = plt.subplots()
        theme.set_headline(fig, "Headline text")
        theme.add_method_note(fig, "method text")
        saver.save(fig, "test_fig", fig_id="F0", title="Headline text")
        saver.record_skip("F9", "skipped_fig", "no data")
        path = _write_captions(saver)
        content = path.read_text(encoding="utf-8")
        assert "F0  test_fig" in content
        assert "headline:   Headline text" in content
        assert "method:     method text" in content
        assert "takeaway:   (none)" in content
        assert "skipped_fig" not in content  # skipped figures carry no captions


def test_figure_module_end_to_end_records_headline():
    """Render F1 from a tiny synthetic bundle with chrome off; manifest carries the text."""
    import pandas as pd

    from seshat_viz.figures import landscape
    from seshat_viz.loader import RunBundle, _enrich_factors

    summary = _enrich_factors(pd.DataFrame({
        "scenario": ["matrix_routing_tcp_tcp_4096B_direct_1c",
                     "matrix_tls13_tcp_tcp_4096B_direct_1c"],
        "transport": ["tcp", "tcp"],
        "protocol": ["none", "tls/1.3"],
        "message_bytes": [4096, 4096],
        "connections": [1, 1],
        "throughput_gbps_mean": [10.0, 5.0],
        "latency_p99_us_mean": [100.0, 200.0],
    }))
    bundle = RunBundle(run_dir=Path("."), summary=summary, runs=pd.DataFrame(),
                       sysmetrics=pd.DataFrame(), saturation=pd.DataFrame(),
                       skipped=pd.DataFrame(columns=["scenario", "reason"]))
    with _with_chrome(False), tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp), formats=("png",))
        landscape.make(bundle, saver)
        entry = saver.manifest[0]
        assert "skipped" not in entry, entry
        heads = [r["text"] for r in entry["chrome"] if r["kind"] == "headline"]
        assert heads and heads[0].startswith(landscape.TITLE)


def test_payload_scaling_adds_rtt_row_when_ping_pong_data_present():
    """F2 renders a 3rd (honest closed-loop RTT) row iff the run carries rtt_us_p99."""
    import numpy as np
    import pandas as pd

    from seshat_viz.figures import payload_scaling
    from seshat_viz.loader import RunBundle, _enrich_factors

    class _CaptureSaver(theme.Saver):
        captured_axes = None

        def save(self, fig, name, **kw):
            self.captured_axes = len(fig.get_axes())
            return super().save(fig, name, **kw)

    sizes = [64, 1024, 4096]
    # matrix (blast) rows: throughput + open-loop blast p99, no closed-loop rtt.
    blast = {
        "scenario": [f"matrix_routing_tcp_tcp_{s}B_direct_1c" for s in sizes],
        "transport": ["tcp"] * len(sizes),
        "protocol": ["none"] * len(sizes),
        "message_bytes": list(sizes),
        "connections": [1] * len(sizes),
        "throughput_gbps_mean": [10.0, 12.0, 14.0],
        "latency_p99_us_mean": [500.0, 800.0, 1200.0],
        "rtt_us_p99": [np.nan] * len(sizes),
    }
    # matrix-latency (honest RTT) rows: closed-loop rtt_us_p99, no throughput/blast latency.
    rtt = {
        "scenario": [f"matrix_lat_routing_tcp_{s}B_direct_1c" for s in sizes],
        "transport": ["tcp"] * len(sizes),
        "protocol": ["none"] * len(sizes),
        "message_bytes": list(sizes),
        "connections": [1] * len(sizes),
        "throughput_gbps_mean": [np.nan] * len(sizes),
        "latency_p99_us_mean": [np.nan] * len(sizes),
        "rtt_us_p99": [22.0, 24.0, 31.0],
    }

    def _bundle(frames):
        summary = _enrich_factors(pd.concat([pd.DataFrame(f) for f in frames], ignore_index=True))
        return RunBundle(run_dir=Path("."), summary=summary, runs=pd.DataFrame(),
                         sysmetrics=pd.DataFrame(), saturation=pd.DataFrame(),
                         skipped=pd.DataFrame(columns=["scenario", "reason"]))

    # With ping-pong data → throughput + blast + honest-RTT = 3 rows (× 1 transport = 3 axes).
    with _with_chrome(False), tempfile.TemporaryDirectory() as tmp:
        saver = _CaptureSaver(Path(tmp), formats=("png",))
        payload_scaling.make(_bundle([blast, rtt]), saver)
        entry = saver.manifest[0]
        assert "skipped" not in entry, entry
        assert saver.captured_axes == 3, saver.captured_axes
        methods = " ".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == "method")
        assert "closed-loop ping-pong RTT" in methods, methods

    # Without ping-pong data → throughput + blast = 2 rows, and no honest-RTT note.
    with _with_chrome(False), tempfile.TemporaryDirectory() as tmp:
        saver = _CaptureSaver(Path(tmp), formats=("png",))
        payload_scaling.make(_bundle([blast]), saver)
        entry = saver.manifest[0]
        assert "skipped" not in entry, entry
        assert saver.captured_axes == 2, saver.captured_axes
        methods = " ".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == "method")
        assert "closed-loop ping-pong RTT" not in methods, methods


def test_closed_loop_rtt_grid_when_matrix_lat_present():
    """F16 draws the per-interface RTT-vs-payload grid (one facet per interface + an
    inflation panel) when the matrix_lat_* closed-loop grid is present, and falls back to
    the per-profile dumbbell (2 panels) when only the loopback ping-pong rows exist."""
    import numpy as np
    import pandas as pd

    from seshat_viz.figures import closed_loop_rtt
    from seshat_viz.loader import RunBundle, _enrich_factors

    class _CaptureSaver(theme.Saver):
        captured_axes = None

        def save(self, fig, name, **kw):
            self.captured_axes = len(fig.get_axes())
            return super().save(fig, name, **kw)

    def _bundle(frame):
        summary = _enrich_factors(pd.DataFrame(frame))
        return RunBundle(run_dir=Path("."), summary=summary, runs=pd.DataFrame(),
                         sysmetrics=pd.DataFrame(), saturation=pd.DataFrame(),
                         skipped=pd.DataFrame(columns=["scenario", "reason"]))

    # --- Grid path: matrix_lat_* rows over 2 interfaces × 2 protocols × 2 sizes, plus the
    #     matched throughput/blast rows so the inflation panel has a baseline. ---
    grid_rows = {"scenario": [], "transport": [], "protocol": [], "message_bytes": [],
                 "connections": [], "throughput_gbps_mean": [], "latency_p99_us_mean": [],
                 "rtt_us_p50": [], "rtt_us_p99": []}
    for iface, unix in (("tcp", "tcp"), ("unix", "unix")):
        for proto, pid in (("none", "routing"), ("tls/1.3", "tls13")):
            for sz in (64, 1024):
                # closed-loop RTT row (matrix-latency family)
                grid_rows["scenario"].append(f"matrix_lat_{pid}_{iface}_{unix}_{sz}B_direct_1c")
                grid_rows["transport"].append(iface); grid_rows["protocol"].append(proto)
                grid_rows["message_bytes"].append(sz); grid_rows["connections"].append(1)
                grid_rows["throughput_gbps_mean"].append(np.nan)
                grid_rows["latency_p99_us_mean"].append(np.nan)
                grid_rows["rtt_us_p50"].append(20.0 + sz / 100.0)
                grid_rows["rtt_us_p99"].append(25.0 + sz / 80.0)
                # matched open-loop blast row (matrix family) for the inflation baseline —
                # must be chain 'direct': the corrected baseline rightly rejects 2-gateway
                # blast rows as a baseline for direct-chain RTT (audit F16-2).
                grid_rows["scenario"].append(f"matrix_{pid}_{iface}_{unix}_{sz}B_direct")
                grid_rows["transport"].append(iface); grid_rows["protocol"].append(proto)
                grid_rows["message_bytes"].append(sz); grid_rows["connections"].append(1)
                grid_rows["throughput_gbps_mean"].append(5.0)
                grid_rows["latency_p99_us_mean"].append(9000.0 + sz)
                grid_rows["rtt_us_p50"].append(np.nan); grid_rows["rtt_us_p99"].append(np.nan)

    with _with_chrome(False), tempfile.TemporaryDirectory() as tmp:
        saver = _CaptureSaver(Path(tmp), formats=("png",))
        closed_loop_rtt.make(_bundle(grid_rows), saver)
        entry = saver.manifest[0]
        assert "skipped" not in entry, entry
        # 2 interface facets + 1 inflation panel (the shared legend is a figure legend, not an axis).
        assert saver.captured_axes == 3, saver.captured_axes
        methods = " ".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == "method")
        assert "vs payload size" in methods, methods
        heads = [r["text"] for r in entry["chrome"] if r["kind"] == "headline"]
        assert heads and heads[0].startswith(closed_loop_rtt.TITLE)

    # --- Dumbbell fallback: only the loopback ping-pong rows (no matrix_lat_* grid). ---
    dumbbell = {
        "scenario": ["pp_tcp_loopback_64B", "pp_tcp_loopback_1KB"],
        "transport": ["tcp", "tcp"], "protocol": ["none", "none"],
        "message_bytes": [64, 1024], "connections": [1, 1],
        "throughput_gbps_mean": [np.nan, np.nan], "latency_p99_us_mean": [np.nan, np.nan],
        "rtt_us_p50": [11.0, 12.0], "rtt_us_p99": [12.8, 13.0],
    }
    with _with_chrome(False), tempfile.TemporaryDirectory() as tmp:
        saver = _CaptureSaver(Path(tmp), formats=("png",))
        closed_loop_rtt.make(_bundle(dumbbell), saver)
        entry = saver.manifest[0]
        assert "skipped" not in entry, entry
        assert saver.captured_axes == 1, saver.captured_axes  # no blast baseline → single panel
        methods = " ".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == "method")
        assert "per-profile ping-pong rows" in methods, methods


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
