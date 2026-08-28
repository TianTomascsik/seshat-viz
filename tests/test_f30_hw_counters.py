"""
F30 hardware-counter figure — honesty gates and normalisation.

The figure draws from TWO kernel-scope perf campaigns (relay pass + ladder slice) and
must refuse — not placeholder — anything less: missing sources, counter-free runs, and
user-scope-demoted data. Also pins
the cycles/byte normalisation, the F29 palette contract, the two-build disclosure, and
that F29's and F30's source discovery can never cross-match.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pandas as pd

from seshat_viz import theme
from seshat_viz.figures import hw_counters, relay_backend
from seshat_viz.loader import RunBundle

# Kernel-plausible vs user-scope-demoted counter magnitudes (same constants family as
# test_f09_resource_cost): 8.8e10 cycles over 35 s task-clock = 2.5 GHz; 1.4e9 = 0.04 GHz.
_TASK_CLOCK_MS = 35_000.0
_PLAUSIBLE_CYCLES = 8.8e10
_DEMOTED_CYCLES = 1.4e9


def _ladder_summary(*, demoted: bool = False, with_perf: bool = True) -> pd.DataFrame:
    rows = []
    cyc = _DEMOTED_CYCLES if demoted else _PLAUSIBLE_CYCLES
    ctx = 0.0 if demoted else 25_000.0
    for tr, tok in (("shm", "shm_shm"), ("shm-slot", "shmslot_shm"), ("tcp", "tcp_tcp")):
        for proto, ptok in (("none", "routing"), ("tls/1.3", "tls13"), ("ktls/1.3", "ktls13")):
            for size in (4096, 16384, 65536):
                r = {
                    "scenario": f"matrix_{ptok}_{tok}_{size}B_direct_1c",
                    "transport": tr, "protocol": proto, "family": "matrix",
                    "chain": "direct", "connections": 1, "message_bytes": size,
                    "throughput_gbps_mean": 10.0,
                }
                if with_perf:
                    r.update({
                        "perf_cycles": cyc, "perf_instructions": cyc * 0.9,
                        "perf_ipc": 0.9, "perf_cache_references": 1e9,
                        "perf_cache_misses": 6.8e7,
                        "perf_task_clock_ms": _TASK_CLOCK_MS,
                        "perf_context_switches": ctx,
                    })
                rows.append(r)
    return pd.DataFrame(rows)


def _write_relay_tree(root: Path, *, demoted: bool = False,
                      backends=("splice", "readwrite", "iouring_splice", "iouring_rw"),
                      cycles: float = _PLAUSIBLE_CYCLES, total_bytes: float = 1e9) -> Path:
    relay = root / "relaybackend-perf-20260101-000000"
    cyc = _DEMOTED_CYCLES if demoted else cycles
    ctx = 0.0 if demoted else 2_500_000.0
    for be in backends:
        for size in ("64B", "16KB", "256KB"):
            scen = f"relaybackend_routing_tcp_{size}_1c"
            cell = relay / "perf" / be / scen / "20260101-000001"
            (cell / "scenarios" / scen).mkdir(parents=True, exist_ok=True)
            with open(cell / "summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "scenario", "perf_cycles", "perf_cache_misses",
                    "perf_cache_references", "perf_task_clock_ms",
                    "perf_context_switches"])
                w.writeheader()
                w.writerow({"scenario": scen, "perf_cycles": cyc,
                            "perf_cache_misses": 6.8e7, "perf_cache_references": 1e9,
                            "perf_task_clock_ms": _TASK_CLOCK_MS,
                            "perf_context_switches": ctx})
            (cell / "scenarios" / scen / "runs.csv").write_text(
                f"run,bytes\n1,{total_bytes:.0f}\n")
    return relay


def _bundle(run_dir: Path, summary: pd.DataFrame) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(run_dir=run_dir, summary=summary, runs=empty, sysmetrics=empty,
                     saturation=empty, sysinfo={"hostname": "test"})


class _CaptureSaver(theme.Saver):
    def save(self, fig, name, **kw):
        self.axes_patches = [len(ax.patches) for ax in fig.get_axes()]
        self.panel_a_heights = [p.get_height() for p in fig.get_axes()[0].patches]
        self.legend_labels = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
        return super().save(fig, name, fig_id=kw.get("fig_id", ""),
                            title=kw.get("title", ""))


def _chrome(saver: theme.Saver, kind: str) -> str:
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F30 unexpectedly skipped: {entry.get('skipped')}"
    return " ".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == kind)


def _run(root: Path, summary: pd.DataFrame) -> _CaptureSaver:
    run_dir = root / "campaign" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    saver = _CaptureSaver(root / "out", formats=("png",))
    hw_counters.make(_bundle(run_dir, summary), saver)
    return saver


def test_renders_with_both_kernel_scope_sources():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_relay_tree(root / "campaign")
        saver = _run(root, _ladder_summary())
        entry = saver.manifest[-1]
        assert "skipped" not in entry, entry
        assert entry["id"] == "F30"
        assert (root / "out" / "f30_hw_counters.png").is_file()


def test_refuses_demoted_ladder_source():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_relay_tree(root / "campaign")
        saver = _run(root, _ladder_summary(demoted=True))
        entry = saver.manifest[-1]
        assert "skipped" in entry, entry
        assert "ladder" in entry["skipped"] and "user-scope" in entry["skipped"], entry


def test_refuses_demoted_relay_source():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_relay_tree(root / "campaign", demoted=True)
        saver = _run(root, _ladder_summary())
        entry = saver.manifest[-1]
        assert "skipped" in entry, entry
        assert "relay" in entry["skipped"] and "user-scope" in entry["skipped"], entry


def test_refuses_missing_sources():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # no relay tree at all
        saver = _run(root, _ladder_summary())
        assert "relaybackend-perf" in saver.manifest[-1]["skipped"]
        # counter-free ladder
        _write_relay_tree(root / "campaign")
        saver2 = _run(root, _ladder_summary(with_perf=False))
        assert "no hardware counters" in saver2.manifest[-1]["skipped"]


def test_cycles_per_byte_normalisation_pinned():
    """8.8e10 cycles over 1e9 runs.csv bytes -> an 88 cycles/byte bar in panel A."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_relay_tree(root / "campaign")
        saver = _run(root, _ladder_summary())
        expect = _PLAUSIBLE_CYCLES / 1e9
        assert any(abs(h - expect) < 1e-6 for h in saver.panel_a_heights), \
            saver.panel_a_heights


def test_missing_backend_omitted_not_fabricated():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_relay_tree(root / "campaign",
                          backends=("splice", "iouring_splice", "iouring_rw"))
        saver = _run(root, _ladder_summary())
        entry = saver.manifest[-1]
        assert "skipped" not in entry, entry
        # panel A: 3 backends x 3 sizes, not 4 x 3 (and no NaN ghost patches)
        assert saver.axes_patches[0] == 9, saver.axes_patches
        # the absent backend is keyed nowhere and disclosed in the method note
        assert "poll+read/write" not in saver.legend_labels, saver.legend_labels
        method = _chrome(saver, "method")
        assert "omitted" in method and "poll+read/write" in method, method


def test_palette_contract_matches_f29():
    assert hw_counters._BACKEND_COLOR is relay_backend._COLOR
    assert hw_counters._BACKEND_COLOR["splice"] == theme.CATEGORY[0]
    assert hw_counters._BACKEND_COLOR["readwrite"] == theme.GREYS["baseline"]
    assert hw_counters._BACKEND_COLOR["iouring_splice"] == theme.CATEGORY[2]
    assert hw_counters._BACKEND_COLOR["iouring_rw"] == theme.CATEGORY[1]


def test_chrome_discloses_two_builds_and_both_sources():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_relay_tree(root / "campaign")
        saver = _run(root, _ladder_summary())
        method = _chrome(saver, "method")
        assert "--features io_uring" in method, method
        assert "mainline gateway build" in method, method
        assert "different gateway builds" in method, method
        prov = _chrome(saver, "provenance")
        assert "relaybackend-perf-20260101-000000" in prov, prov


def test_f29_f30_source_discovery_never_crosses():
    """A results root holding BOTH a relay-backend-ab-* tree and a
    relaybackend-perf-* tree: each figure's discovery picks only its own."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        results = root / "results"
        july = results / "relay-backend-ab-20260101-000000"
        july.mkdir(parents=True)
        (july / "aggregate.csv").write_text(
            "metric,backend,scenario,throughput_gbps\n"
            "procfs,splice,relaybackend_routing_tcp_64B_1c,10\n")
        _write_relay_tree(results)
        run_dir = results / "campaign" / "run"
        run_dir.mkdir(parents=True)
        b = _bundle(run_dir, pd.DataFrame())
        procfs_dir, _ebpf = relay_backend._find_sources(b)
        assert procfs_dir is not None and procfs_dir.name.startswith("relay-backend-ab-")
        relay_dir = hw_counters._find_relay_source(b)
        assert relay_dir is not None and relay_dir.name.startswith("relaybackend-perf-")


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
