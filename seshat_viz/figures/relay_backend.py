"""
F29 — Relay-backend A/B on the SCG routing path (splice / read-write / io_uring).

Compares the gateway's selectable relay backends — the shipped poll+splice zero-copy
path against the copying poll+read/write path and the two io_uring variants — on
throughput, worker-thread footprint, context switches, and syscalls per message.

Reads the two ``relay-backend-ab-*`` SESHAT result trees that live next to the bundle's
campaign (no hardcoded numbers) and emits a 2×2 figure:

  A  routing throughput by message size          (procfs pass, unperturbed)
  B  peak io-wq worker threads vs connections    (procfs pass)
  C  context switches vs connections             (procfs pass)
  D  system calls per message by size            (eBPF pass, syscall counters)

Panels A–C use the procfs pass because eBPF tracing depresses throughput a few percent.
Panel D needs the eBPF pass, the only one carrying per-syscall counters. ``read()``/
``write()`` are outside the eBPF probe set (mem_syscalls.bt counts sendmsg/recvmsg/splice/
poll/ppoll/io_uring_enter), so the copying poll+read/write backend cannot be shown fairly
on panel D and is omitted there; the zero-copy pair (splice vs io_uring-splice) and
io_uring recv/send are fully counted.

Backend identity uses the scoped CATEGORY palette (backends never share an axes with
transport/protocol series): the shipped poll+splice baseline takes CATEGORY[0], the two
io_uring variants CATEGORY[1]/[2], and the copying poll+read/write reference the baseline
grey — four backends, one legend, keyed once for the whole figure.
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import theme as T
from ..loader import RunBundle

FIG_ID = "F29"
NAME = "f29_relay_backend_ab"
TITLE = "Relay-backend A/B on the SCG routing path"

_BACKENDS = ["splice", "readwrite", "iouring_splice", "iouring_rw"]
_LABEL = {
    "splice": "poll+splice",
    "readwrite": "poll+read/write",
    "iouring_splice": "io_uring splice",
    "iouring_rw": "io_uring recv/send",
}
_COLOR = {
    "splice": T.CATEGORY[0],
    "readwrite": T.GREYS["baseline"],
    "iouring_splice": T.CATEGORY[2],
    "iouring_rw": T.CATEGORY[1],
}
_SIZES = [("64B", 64), ("16KB", 16384), ("256KB", 262144)]
_CONNS = [1, 4, 16, 64]
# read()/write() are outside the eBPF probe set, so readwrite is unfair on panel D.
_D_BACKENDS = ["splice", "iouring_splice", "iouring_rw"]


def _parse_scenario(name: str) -> tuple[str, str, int]:
    """relaybackend_<path>_tcp_<size>_<conns>c -> (path, size, conns)."""
    parts = name.split("_")
    return parts[1], parts[3], int(parts[4].rstrip("c"))


def _load_aggregate(path: Path, metric: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("metric") == metric:
                rows.append(r)
    return rows


def _fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _find_sources(bundle: RunBundle) -> tuple[Path | None, Path | None]:
    """
    Locate the procfs and eBPF relay-backend campaigns beside the bundle's run.

    The relay-backend A/B dirs are siblings of the campaign directory (the results root),
    so look one and two levels above the bundle's run_dir. Source selection is
    deterministic: the newest dir whose aggregate carries procfs rows supplies A–C, the
    newest with eBPF rows supplies D (the dedicated eBPF pass is newer than the combined
    one, so the original script's pairing is reproduced from data, not hardcoded names).
    """
    run_dir = Path(bundle.run_dir)
    candidates: list[Path] = []
    for root in (run_dir.parent, run_dir.parent.parent):
        candidates += [Path(p) for p in sorted(glob.glob(str(root / "relay-backend-ab-*")))]
    procfs_dir = ebpf_dir = None
    for d in candidates:  # sorted ascending → the last hit is the newest
        agg = d / "aggregate.csv"
        if not agg.is_file():
            continue
        try:
            metrics = {r.get("metric") for r in csv.DictReader(open(agg, newline=""))}
        except OSError:
            continue
        if "procfs" in metrics:
            procfs_dir = d
        if "ebpf" in metrics:
            ebpf_dir = d
    return procfs_dir, ebpf_dir


def make(bundle: RunBundle, saver: T.Saver) -> None:
    procfs_dir, ebpf_dir = _find_sources(bundle)
    if procfs_dir is None:
        saver.record_skip(
            FIG_ID, NAME,
            "no relay-backend-ab-* campaign with procfs rows next to this run "
            "(run the relay-backend A/B with an io_uring-enabled gateway build)")
        return

    procfs = _load_aggregate(procfs_dir / "aggregate.csv", "procfs")
    ebpf = _load_aggregate(ebpf_dir / "aggregate.csv", "ebpf") if ebpf_dir else []

    # Messages per scenario from the eBPF run's runs.csv (authoritative count for panel D).
    msgs_by_scen: dict[str, float] = defaultdict(float)
    if ebpf_dir is not None:
        for rc in glob.glob(os.path.join(str(ebpf_dir), "ebpf", "*", "*", "*",
                                         "scenarios", "*", "runs.csv")):
            scen = os.path.basename(os.path.dirname(rc))
            with open(rc, newline="") as f:
                for r in csv.DictReader(f):
                    msgs_by_scen[scen] += _fnum(r.get("messages"))

    # Panel A: routing throughput by size (mean over conns).
    thr: dict[tuple[str, str], list] = defaultdict(list)
    for r in procfs:
        p, s, _c = _parse_scenario(r["scenario"])
        if p == "routing":
            thr[(r["backend"], s)].append(_fnum(r["throughput_gbps"]))
    thr_a = {k: float(np.mean(v)) for k, v in thr.items()}

    # Panels B/C: peak threads + context switches vs conns (mean over all sizes+paths).
    threads: dict[tuple[str, int], list] = defaultdict(list)
    ctxsw: dict[tuple[str, int], list] = defaultdict(list)
    for r in procfs:
        _p, _s, c = _parse_scenario(r["scenario"])
        threads[(r["backend"], c)].append(_fnum(r["peak_threads"]))
        ctxsw[(r["backend"], c)].append(_fnum(r["ctx_switches"]))
    threads_b = {k: float(np.mean(v)) for k, v in threads.items()}
    ctxsw_c = {k: float(np.mean(v)) for k, v in ctxsw.items()}

    # Panel D: syscalls per message by size (eBPF, all paths pooled).
    sysc: dict[tuple[str, str], float] = defaultdict(float)
    msgs: dict[tuple[str, str], float] = defaultdict(float)
    for r in ebpf:
        _p, s, _c = _parse_scenario(r["scenario"])
        b = r["backend"]
        if b == "splice":
            n = _fnum(r["mem_splice"]) + _fnum(r["mem_poll"])
        else:  # iouring_* driven through io_uring_enter; readwrite uncounted (omitted)
            n = _fnum(r["mem_io_uring_enter"])
        if not np.isnan(n):
            sysc[(b, s)] += n
        msgs[(b, s)] += msgs_by_scen.get(r["scenario"], 0.0)
    sysc_d: dict[tuple[str, str], float] = {}
    for b in _D_BACKENDS:
        for s, _ in _SIZES:
            m = msgs.get((b, s), 0.0)
            sysc_d[(b, s)] = (sysc.get((b, s), 0.0) / m) if m else float("nan")
    have_d = any(np.isfinite(v) for v in sysc_d.values())

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    def _grouped_bars(ax, groups, backends, value, ylabel, title, *, logy=False,
                      label_fmt="{:.0f}"):
        x = np.arange(len(groups))
        w = 0.8 / len(backends)
        for i, b in enumerate(backends):
            vals = [value.get((b, g), float("nan")) for g in groups]
            xpos = x + (i - (len(backends) - 1) / 2) * w
            ax.bar(xpos, vals, w, color=_COLOR[b], label=_LABEL[b],
                   edgecolor="white", linewidth=0.6)
            for xi, v in zip(xpos, vals):
                if np.isfinite(v):
                    # Alternate label heights across the group: neighbouring bars of
                    # near-equal height would otherwise run their labels together.
                    T.annotate_value(ax, xi, v, label_fmt.format(v),
                                     stagger=(i % 2) * 7.0)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=T.FS["small"])
        ax.set_ylabel(ylabel, fontsize=T.FS["small"])
        T.panel_title(ax, title)
        if logy:
            ax.set_yscale("log")
        else:
            ax.margins(y=0.20)

    def _lines(ax, xs, backends, value, ylabel, title, *, yscale_m=1.0):
        # Fixed per-series vertical nudge for the end-of-line labels: the two copy
        # backends converge at high connection counts and their labels would collide.
        dys = [0, -7, 0, 7]
        for i, b in enumerate(backends):
            ys = [value.get((b, x), float("nan")) / yscale_m for x in xs]
            ax.plot(xs, ys, "-o", color=_COLOR[b], label=_LABEL[b], lw=2.0, ms=5)
            ax.annotate(f"{ys[-1]:.0f}" if yscale_m == 1 else f"{ys[-1]:.1f}",
                        (xs[-1], ys[-1]), color=T.GREYS["ink"], fontsize=T.FS["annot"],
                        xytext=(4, dys[i % len(dys)]), textcoords="offset points",
                        va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=T.FS["small"])
        ax.set_xlabel("concurrent connections", fontsize=T.FS["small"])
        ax.set_ylabel(ylabel, fontsize=T.FS["small"])
        T.panel_title(ax, title)
        ax.margins(x=0.12)

    _grouped_bars(ax_a, [s for s, _ in _SIZES], _BACKENDS, thr_a,
                  "throughput (Gbit/s)", "A · routing throughput by size")
    _lines(ax_b, _CONNS, _BACKENDS, threads_b, "peak worker threads",
           "B · io-wq worker threads")
    _lines(ax_c, _CONNS, _BACKENDS, ctxsw_c, "context switches (millions)",
           "C · context switches", yscale_m=1e6)
    if have_d:
        _grouped_bars(ax_d, [s for s, _ in _SIZES], _D_BACKENDS, sysc_d,
                      "system calls per message", "D · syscalls per message (eBPF)",
                      logy=True, label_fmt="{:.2g}")
        ax_d.set_ylim(top=ax_d.get_ylim()[1] * 4)  # headroom for the in-bar value labels
    else:
        T.perf_placeholder(ax_d, "syscalls per message\n(needs the eBPF relay-backend pass)")

    handles = [plt.matplotlib.patches.Patch(facecolor=_COLOR[b], edgecolor=T.GREYS["edge"],
                                            linewidth=0.5, label=_LABEL[b])
               for b in _BACKENDS]
    T.legend_below(fig, handles, ncol=4)

    T.set_headline(fig, f"{TITLE}  ·  loopback")
    spl = thr_a.get(("splice", "256KB"), float("nan"))
    iou = thr_a.get(("iouring_splice", "256KB"), float("nan"))
    if np.isfinite(spl) and np.isfinite(iou) and iou > 0:
        T.add_takeaway(
            fig,
            f"The shipped poll+splice backend moves {spl:.0f} Gbit/s at 256 KiB vs "
            f"{iou:.0f} Gbit/s for io_uring splice ({spl / iou:.1f}×): the io_uring "
            "variants pay an io-wq worker-thread pool and its context-switch bill "
            "without reducing syscalls per message enough to win — the measured basis "
            "for rejecting the io_uring relay.")
    T.add_method_note(
        fig,
        "panels A–C: procfs pass (eBPF tracing depresses throughput a few percent); "
        "panel D: eBPF pass syscall counters. read()/write() are outside the eBPF probe "
        "set, so poll+read/write is omitted from panel D only. Panel A is the plaintext "
        "routing path; B–D pool routing+kTLS paths and all sizes at each connection count.")
    T.add_provenance(
        fig,
        f"procfs: {procfs_dir.name} · eBPF: {ebpf_dir.name if ebpf_dir else '—'} "
        "· relay-backend A/B campaigns (io_uring-enabled gateway build)")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
