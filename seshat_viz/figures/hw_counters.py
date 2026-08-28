"""
F30 — Hardware-counter evidence: relay copy cost and the protection ladder.

Two claims that throughput numbers alone cannot attribute get their direct
measurement here: the "CPU cost per byte" term of the relay-backend (splice vs
io_uring) comparison, and the "memory and kernel bound" attribution of the plaintext
stream path vs the AEAD-bound encrypted path. Both are read from two kernel-scope
`perf stat` campaigns:

  A  CPU cycles per payload byte, by relay backend and message size   (relay perf pass)
  B  cache misses per payload byte, by relay backend and message size (relay perf pass)
  C  cache-miss rate (misses ÷ references) across the protection ladder (ladder slice)
  D  instructions per cycle across the same ladder                      (ladder slice)

Panels A/B compare the four relay backends on the plaintext routing path —
zero-copy (poll+splice, io_uring splice) against copying (poll+read/write, io_uring
recv/send) — normalised by the bytes each cell actually moved (runs.csv totals) and
averaged over the 1/4/16/64-connection cells. Panels C/D climb routing → TLS 1.3 →
kTLS 1.3 on the shared-memory rings and TCP at a matched 16 KiB, 1 connection.

Honesty gates: the figure REFUSES to render (record_skip, no placeholder panels) unless
both campaign sources exist AND both pass :func:`derive.perf_user_scope_only` — an
unprivileged (paranoid>=2) perf run silently demotes every event to user scope, which
turns these panels into user-vs-kernel attribution artifacts. The two campaigns run two
gateway builds (an io_uring-enabled build vs mainline) under one harness build;
no panel mixes rows across builds, and the method note discloses the split.

Sources: the ladder slice is the run directory this figure is invoked on (RUN_PERF in
the export script); the relay pass is found by globbing ``relaybackend-perf-*`` beside
it — deliberately NOT matching F29's ``relay-backend-ab-*`` trees, so the two
figures can never read each other's campaign.
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label
from .relay_backend import _COLOR as _BACKEND_COLOR
from .relay_backend import _LABEL as _BACKEND_LABEL

FIG_ID = "F30"
NAME = "f30_hw_counters"
TITLE = "Hardware-counter evidence: relay copy cost & protection ladder"

_BACKENDS = ["splice", "readwrite", "iouring_splice", "iouring_rw"]
_SIZES = ["64B", "16KB", "256KB"]
_LADDER_PROTOS = ["none", "tls/1.3", "ktls/1.3"]
_LADDER_TRANSPORTS = ["shm", "shm-slot", "tcp"]
_LADDER_SIZE = 16384


def _fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _find_relay_source(bundle: RunBundle) -> Path | None:
    """Newest relaybackend-perf-* campaign beside the bundle's run (F29's discovery
    pattern with a glob that can never match F29's own relay-backend-ab-* dirs)."""
    run_dir = Path(bundle.run_dir)
    hits: list[Path] = []
    for root in (run_dir.parent, run_dir.parent.parent):
        hits += [Path(p) for p in sorted(glob.glob(str(root / "relaybackend-perf-*")))]
    return hits[-1] if hits else None


def _load_relay(relay_dir: Path) -> pd.DataFrame:
    """One row per (backend, scenario) cell: perf counters from the cell's summary.csv,
    total bytes moved from its runs.csv (summed across repetitions)."""
    rows = []
    for summ in glob.glob(str(relay_dir / "perf" / "*" / "*" / "*" / "summary.csv")):
        p = Path(summ)
        if p.parent.parent.parent.parent.name != "perf":
            continue  # only the per-cell top-level summary, not scenarios/*/summary.csv
        backend, scenario = p.parent.parent.parent.name, p.parent.parent.name
        with open(p, newline="") as f:
            recs = [r for r in csv.DictReader(f) if r.get("scenario") == scenario]
        if not recs:
            continue
        r = recs[0]
        total_bytes = 0.0
        runs_csv = p.parent / "scenarios" / scenario / "runs.csv"
        if runs_csv.is_file():
            with open(runs_csv, newline="") as f:
                for rr in csv.DictReader(f):
                    b = _fnum(rr.get("bytes"))
                    if b == b:
                        total_bytes += b
        parts = scenario.split("_")  # relaybackend_<path>_tcp_<size>_<conns>c
        rows.append({
            "backend": backend, "scenario": scenario, "path": parts[1],
            "size": parts[3], "connections": int(parts[4].rstrip("c")),
            "bytes": total_bytes,
            "perf_cycles": _fnum(r.get("perf_cycles")),
            "perf_cache_misses": _fnum(r.get("perf_cache_misses")),
            "perf_cache_references": _fnum(r.get("perf_cache_references")),
            "perf_task_clock_ms": _fnum(r.get("perf_task_clock_ms")),
            "perf_context_switches": _fnum(r.get("perf_context_switches")),
        })
    return pd.DataFrame(rows)


def _relay_cell(df: pd.DataFrame, backend: str, size: str, counter: str) -> float:
    """Mean of counter/bytes over the connection cells of (backend, size), routing path."""
    g = df[(df["backend"] == backend) & (df["size"] == size) & (df["path"] == "routing")]
    g = g[(g["bytes"] > 0) & g[counter].notna()]
    if g.empty:
        return float("nan")
    return float((g[counter] / g["bytes"]).mean())


def make(bundle: RunBundle, saver: T.Saver) -> None:
    # ---- source 1: the ladder slice (the run this figure is invoked on) --------------
    summ = bundle.summary
    if summ.empty or "perf_cycles" not in summ.columns \
            or not pd.to_numeric(summ["perf_cycles"], errors="coerce").notna().any():
        saver.record_skip(FIG_ID, NAME,
                          "ladder source carries no hardware counters — invoke on the "
                          "kernel-scope perf run (ladder-perf-*, --metrics-backend perf)")
        return
    if derive.perf_user_scope_only(summ):
        saver.record_skip(FIG_ID, NAME,
                          "ladder source counters are user-scope demoted (unprivileged "
                          "perf at paranoid>=2) — rerun at kernel scope (paranoid<=1)")
        return

    # ---- source 2: the relay perf pass ----------------------------------------------
    relay_dir = _find_relay_source(bundle)
    if relay_dir is None:
        saver.record_skip(FIG_ID, NAME,
                          "no relaybackend-perf-* campaign found beside the run "
                          "(run the relay-backend perf pass)")
        return
    relay = _load_relay(relay_dir)
    if relay.empty:
        saver.record_skip(FIG_ID, NAME, f"relay perf pass at {relay_dir.name} holds no cells")
        return
    if derive.perf_user_scope_only(relay):
        saver.record_skip(FIG_ID, NAME,
                          f"relay perf pass {relay_dir.name} is user-scope demoted — "
                          "rerun at kernel scope (paranoid<=1)")
        return

    # ---- ladder frame: matched 1-connection direct cells, normalised ------------------
    lad = derive.add_normalized_costs(derive.attach_bytes_from_runs(summ, bundle.runs))
    m = pd.Series(True, index=lad.index)
    for col, want in (("family", "matrix"), ("chain", "direct")):
        if col in lad.columns:
            m &= lad[col].astype(str) == want
    if "connections" in lad.columns:
        m &= pd.to_numeric(lad["connections"], errors="coerce") == 1
    lad = lad[m]
    lad16 = lad[pd.to_numeric(lad.get("message_bytes"), errors="coerce") == _LADDER_SIZE]

    def _lcell(transport: str, proto: str, col: str) -> float:
        g = lad16[(lad16["transport"].astype(str) == transport)
                  & (lad16["protocol"].astype(str) == proto)]
        v = pd.to_numeric(g.get(col), errors="coerce").dropna() if not g.empty else []
        return float(v.mean()) if len(v) else float("nan")

    # Ranking stability of the ladder ordering across the >=4 KiB sizes (method-note
    # clause — computed, never asserted): compare each transport's protocol ordering by
    # cache-miss rate at every size >= 4096 against the 16 KiB ordering.
    stable = True
    sizes_ge4k = sorted(set(pd.to_numeric(lad.get("message_bytes"), errors="coerce")
                            .dropna().astype(int)) & {4096, 16384, 65536})
    for tr in _LADDER_TRANSPORTS:
        orders = []
        for sz in sizes_ge4k:
            g = lad[(lad["transport"].astype(str) == tr)
                    & (pd.to_numeric(lad["message_bytes"], errors="coerce") == sz)]
            vals = {p: pd.to_numeric(
                g[g["protocol"].astype(str) == p].get("cache_miss_rate"),
                errors="coerce").mean() for p in _LADDER_PROTOS}
            if all(v == v for v in vals.values()):
                orders.append(tuple(sorted(_LADDER_PROTOS, key=lambda p: vals[p])))
        if len({o for o in orders}) > 1:
            stable = False

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # Only backends the relay pass actually measured: a missing backend must be absent
    # from bars AND legend, never a silent NaN ghost.
    present = [be for be in _BACKENDS
               if any(np.isfinite(_relay_cell(relay, be, s, "perf_cycles"))
                      for s in _SIZES)]

    def _relay_panel(ax, counter, ylabel, title, fmt):
        x = np.arange(len(_SIZES))
        w = 0.8 / len(present)
        groups: dict[int, list] = defaultdict(list)
        for i, be in enumerate(present):
            vals = [_relay_cell(relay, be, s, counter) for s in _SIZES]
            xpos = x + (i - (len(present) - 1) / 2) * w
            ax.bar(xpos, vals, w, color=_BACKEND_COLOR[be], label=_BACKEND_LABEL[be],
                   edgecolor="white", linewidth=0.6)
            for gi, (xi, v) in enumerate(zip(xpos, vals)):
                if np.isfinite(v):
                    groups[gi].append((xi, v))
        # Collision-free labels on the log axis: within each size group (left to right)
        # a label sits just above its own bar AND at least a fixed log-interval above
        # the previous label — stagger-by-index cannot guarantee that when value gaps
        # oppose the stagger direction.
        top = 0.0
        for cells in groups.values():
            prev = 0.0
            for xi, v in sorted(cells):
                lv = max(v * 1.10, prev * 1.32)
                ax.annotate(fmt.format(v), (xi, lv), ha="center", va="bottom",
                            fontsize=T.FS["annot"], color=T.GREYS["ink"])
                prev = lv
                top = max(top, lv)
        ax.set_xticks(x)
        ax.set_xticklabels(_SIZES, fontsize=T.FS["small"])
        ax.set_ylabel(ylabel, fontsize=T.FS["small"])
        ax.set_yscale("log")
        if top > 0:
            ax.set_ylim(top=top * 1.7)  # labels clear the panel title
        T.panel_title(ax, title)

    _relay_panel(ax_a, "perf_cycles", "CPU cycles / byte",
                 "A · relay cycles per byte", "{:.2g}")
    _relay_panel(ax_b, "perf_cache_misses", "cache misses / byte",
                 "B · relay cache misses per byte", "{:.2g}")

    def _ladder_panel(ax, col, ylabel, title, fmt):
        x = np.arange(len(_LADDER_TRANSPORTS))
        w = 0.8 / len(_LADDER_PROTOS)
        for i, proto in enumerate(_LADDER_PROTOS):
            vals = [_lcell(tr, proto, col) for tr in _LADDER_TRANSPORTS]
            xpos = x + (i - (len(_LADDER_PROTOS) - 1) / 2) * w
            ax.bar(xpos, vals, w, color=T.protocol_color(proto),
                   label=protocol_label(proto), edgecolor="white", linewidth=0.6)
            for xi, v in zip(xpos, vals):
                if np.isfinite(v):
                    T.annotate_value(ax, xi, v, fmt.format(v), stagger=(i % 2) * 7.0)
        ax.set_xticks(x)
        ax.set_xticklabels([transport_label(t) for t in _LADDER_TRANSPORTS],
                           fontsize=T.FS["small"])
        ax.set_ylabel(ylabel, fontsize=T.FS["small"])
        ax.margins(y=0.22)
        T.panel_title(ax, title)

    _ladder_panel(ax_c, "cache_miss_rate", "cache-miss rate (miss / ref)",
                  "C · ladder cache-miss rate (16 KiB)", "{:.3f}")
    _ladder_panel(ax_d, "ipc", "instructions / cycle",
                  "D · ladder IPC (16 KiB)", "{:.2f}")

    from matplotlib.patches import Patch

    handles = ([Patch(facecolor=_BACKEND_COLOR[b], edgecolor=T.GREYS["edge"],
                      linewidth=0.5, label=_BACKEND_LABEL[b]) for b in present]
               + [Patch(facecolor=T.protocol_color(p), edgecolor=T.GREYS["edge"],
                        linewidth=0.5, label=protocol_label(p)) for p in _LADDER_PROTOS])
    T.legend_below(fig, handles, ncol=4)

    T.set_headline(fig, f"{TITLE}  ·  kernel-scope perf")

    # ---- computed takeaway ------------------------------------------------------------
    take = ""
    spl = _relay_cell(relay, "splice", "256KB", "perf_cache_misses")
    rw = _relay_cell(relay, "readwrite", "256KB", "perf_cache_misses")
    if np.isfinite(spl) and np.isfinite(rw) and spl > 0:
        take = (f"At 256 KiB the copying poll+read/write relay moves {rw / spl:.1f}× the "
                f"cache misses per byte of zero-copy poll+splice")
    r_ipc = _lcell("tcp", "none", "ipc")
    t_ipc = _lcell("tcp", "tls/1.3", "ipc")
    if np.isfinite(r_ipc) and np.isfinite(t_ipc):
        take += (f"; on TCP the AEAD rung lifts IPC from {r_ipc:.2f} (routing, "
                 f"memory/kernel-bound) to {t_ipc:.2f} (compute-bound crypto)")
    if take:
        T.add_takeaway(fig, take + ".")

    note = (
        "A/B: relay-backend perf pass, plaintext routing path, gateway built with "
        "io_uring enabled (--features io_uring); counters divided by the bytes "
        "each cell moved (runs.csv totals) and averaged over the 1/4/16/64-connection "
        "cells; 64 B bars are per-message-framing dominated. C/D: protection-ladder slice "
        "at 16 KiB, 1 connection, single gateway, mainline gateway build; miss rate = "
        "misses ÷ references, IPC = instructions ÷ cycles. "
        + ("The rung ordering is stable across the 4–64 KiB sizes. " if stable
           else "The rung ordering varies across the 4–64 KiB sizes; 16 KiB is one slice. ")
        + ("" if len(present) == len(_BACKENDS) else
           "Backends absent from the relay pass are omitted: "
           + ", ".join(_BACKEND_LABEL[b] for b in _BACKENDS if b not in present) + ". ")
        + "Both campaigns: one harness build, unprivileged kernel-scope perf "
        "(perf_event_paranoid=1); counter windows span warmup and, on encrypted rungs, "
        "the handshake — identical fractions within each panel. The two campaigns use "
        "different gateway builds, so no panel compares across them."
    )
    T.add_method_note(fig, note)
    T.add_provenance(
        fig,
        f"ladder: {Path(bundle.run_dir).parent.name}/{Path(bundle.run_dir).name} · "
        f"relay: {relay_dir.name} · kernel-scope perf stat, whole-scenario windows")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
