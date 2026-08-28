"""
F29 relay-backend A/B — port of the io-uring branch's plot_relay_backend_ab.py.

Covers the port's load-bearing behaviours with synthetic campaign trees:
- source discovery next to the bundle's run_dir (procfs pass vs eBPF pass by `metric`);
- panel D omits poll+read/write (read()/write() are outside the eBPF probe set) while
  the figure-level legend still keys all four backends;
- syscalls-per-message accounting divides eBPF counters by the runs.csv message count;
- the skip path when no relay-backend campaign exists;
- the placeholder path when only the procfs pass exists.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pandas as pd

from seshat_viz import theme
from seshat_viz.figures import relay_backend
from seshat_viz.loader import RunBundle

_AGG_COLS = ["metric", "backend", "scenario", "throughput_gbps", "ctx_switches",
             "peak_threads", "mem_splice", "mem_poll", "mem_io_uring_enter"]

_BACKENDS = ["splice", "readwrite", "iouring_splice", "iouring_rw"]
_TPUT = {"splice": 60.0, "readwrite": 30.0, "iouring_splice": 55.0, "iouring_rw": 28.0}
_SIZES = ["64B", "16KB", "256KB"]


def _write_agg(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_AGG_COLS)
        w.writeheader()
        w.writerows(rows)


def _procfs_rows() -> list[dict]:
    rows = []
    for b in _BACKENDS:
        for s in _SIZES:
            for c in (1, 4):
                rows.append({
                    "metric": "procfs", "backend": b,
                    "scenario": f"relaybackend_routing_tcp_{s}_{c}c",
                    "throughput_gbps": _TPUT[b], "ctx_switches": 1_000_000,
                    "peak_threads": 100 if b.startswith("iouring") else 4,
                    "mem_splice": "", "mem_poll": "", "mem_io_uring_enter": "",
                })
    return rows


def _ebpf_rows() -> list[dict]:
    rows = []
    for b in _BACKENDS:
        for s in _SIZES:
            r = {
                "metric": "ebpf", "backend": b,
                "scenario": f"relaybackend_routing_tcp_{s}_1c",
                "throughput_gbps": _TPUT[b], "ctx_switches": 1_000_000,
                "peak_threads": 100,
                "mem_splice": "", "mem_poll": "", "mem_io_uring_enter": "",
            }
            if b == "splice":
                r["mem_splice"], r["mem_poll"] = 1000, 500     # 1500 calls / 750 msgs = 2.0
            else:
                r["mem_io_uring_enter"] = 3000                 # 3000 / 750 = 4.0
            rows.append(r)
    return rows


def _make_tree(root: Path, *, with_ebpf: bool = True) -> Path:
    """Build results/{relay dirs} + a campaign run dir two levels below results/."""
    results = root / "results"
    _write_agg(results / "relay-backend-ab-20260101-000001" / "aggregate.csv", _procfs_rows())
    if with_ebpf:
        ebpf_dir = results / "relay-backend-ab-20260101-000002"
        _write_agg(ebpf_dir / "aggregate.csv", _ebpf_rows())
        for r in _ebpf_rows():
            scen = r["scenario"]
            rc = (ebpf_dir / "ebpf" / r["backend"] / scen / "20260101-000000"
                  / "scenarios" / scen / "runs.csv")
            rc.parent.mkdir(parents=True, exist_ok=True)
            rc.write_text("messages\n750\n")
    run_dir = results / "campaign" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _bundle(run_dir: Path) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(run_dir=run_dir, summary=empty, runs=empty, sysmetrics=empty,
                     saturation=empty, sysinfo={"hostname": "test"})


class _CaptureSaver(theme.Saver):
    """Snapshots per-axes patch counts and figure legend labels before save closes."""

    def save(self, fig, name, **kw):
        self.axes_patches = [len(ax.patches) for ax in fig.get_axes()]
        self.legend_labels = [
            t.get_text() for leg in fig.legends for t in leg.get_texts()
        ]
        return super().save(fig, name, fig_id=kw.get("fig_id", ""),
                            title=kw.get("title", ""))


def _chrome(saver: theme.Saver, kind: str) -> str:
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F29 unexpectedly skipped: {entry.get('skipped')}"
    return " ".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == kind)


def _render(run_dir: Path, out: Path) -> _CaptureSaver:
    saver = _CaptureSaver(out, formats=("png",))
    relay_backend.make(_bundle(run_dir), saver)
    return saver


def test_renders_with_both_passes():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_tree(Path(tmp))
        saver = _render(run_dir, Path(tmp) / "out")
        entry = saver.manifest[-1]
        assert "skipped" not in entry, entry
        assert entry["id"] == "F29"
        assert (Path(tmp) / "out" / "f29_relay_backend_ab.png").is_file()


def test_panel_d_omits_readwrite_but_legend_keys_it():
    """Panel A bars 4 backends × 3 sizes; panel D only 3 backends (readwrite uncounted);
    the shared legend still names all four."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_tree(Path(tmp))
        saver = _render(run_dir, Path(tmp) / "out")
        ax_a, _b, _c, ax_d = saver.axes_patches
        assert ax_a == 4 * 3, saver.axes_patches
        assert ax_d == 3 * 3, saver.axes_patches
        for lbl in ("poll+splice", "poll+read/write", "io_uring splice",
                    "io_uring recv/send"):
            assert lbl in saver.legend_labels, saver.legend_labels
        method = _chrome(saver, "method")
        assert "outside the eBPF probe" in method, method


def test_takeaway_computes_backend_ratio():
    """splice 60 vs io_uring splice 55 Gbit/s at 256 KiB → 1.1×, computed not hardcoded."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_tree(Path(tmp))
        saver = _render(run_dir, Path(tmp) / "out")
        take = _chrome(saver, "takeaway")
        assert "1.1×" in take, take


def test_skip_without_relay_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "results" / "campaign" / "run"
        run_dir.mkdir(parents=True)
        saver = _CaptureSaver(Path(tmp) / "out", formats=("png",))
        relay_backend.make(_bundle(run_dir), saver)
        entry = saver.manifest[-1]
        assert "skipped" in entry, entry
        assert "relay-backend" in entry["skipped"], entry


def test_procfs_only_renders_with_placeholder_panel_d():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_tree(Path(tmp), with_ebpf=False)
        saver = _render(run_dir, Path(tmp) / "out")
        entry = saver.manifest[-1]
        assert "skipped" not in entry, entry
        assert saver.axes_patches[3] == 0, saver.axes_patches  # placeholder, no bars


def _main() -> int:
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    return failed


if __name__ == "__main__":
    raise SystemExit(_main())
