"""
F28 — kTLS vs user-space TLS on a physical NIC (the wire A/B).

Chapter 9 records the kernel/user-space TLS parity as "a property of the
loopback testbed [that] says nothing about offload gains behind a real NIC".
This figure is that measurement: the same A/B (only the encrypt gateway's
`prefer_ktls` flag differs; the peer stays fixed) on both media, 3 replicates
per cell.

The result has two halves and the layout follows them:

  a. where the link does NOT clamp (loopback), kTLS is faster outright —
     the published "parity" understated it;
  b. where the link DOES clamp (wire), both arms sit at line rate and the
     entire difference moves into CPU: kTLS delivers identical goodput for
     roughly a quarter less processor. That reframes the open question from
     "kTLS behind a NIC" to specifically "kTLS with hardware offload".
  c. closed-loop RTT p50 by message size, both media (the wire 64 B point does
     not exist — those cells died under the probe's desync guard — so the wire
     series starts at 1 KiB, disclosed in the method note).

Here the arm colour IS protocol identity (kTLS mTLS vs user-space mTLS), so
colour separation follows the house palette; medium stays linestyle. Neither
NIC has TLS hardware offload: this compares the kernel's software record layer
against user-space TLS, which the method note states plainly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle, protocol_label
from ..wire import aggregate

FIG_ID = "F28"
NAME = "f28_wire_ktls_ab"
TITLE = "kTLS vs user-space TLS on a physical NIC"

_ARMS = [("ktls", "ktls/1.3+mtls"), ("user", "tls/1.3+mtls")]
_ARM_LABEL = {"ktls": "kTLS (kernel record layer)", "user": "user-space TLS"}
_TPUT_CELLS = ["tput-tls-c1", "tput-tls-c4"]
_CELL_LABEL = {"tput-tls-c1": "1 conn", "tput-tls-c4": "4 conns"}


def _bars(ax, agg: pd.DataFrame, ycol: str, *, annotate_fmt: str = "{:.2f}") -> None:
    """Grouped bars (cell × arm) with CI whiskers and replicate dots."""
    x = np.arange(len(_TPUT_CELLS))
    bw = 0.8 / len(_ARMS)
    for j, (arm, proto) in enumerate(_ARMS):
        xpos = x + (j - (len(_ARMS) - 1) / 2.0) * bw
        med, err = [], []
        for cell in _TPUT_CELLS:
            row = agg[(agg["arm"] == arm) & (agg["cell"] == cell)]
            med.append(float(row[f"{ycol}_med"].iloc[0]) if not row.empty else np.nan)
            err.append(float(row[f"{ycol}_ci95"].iloc[0]) if not row.empty else 0.0)
        ax.bar(xpos, med, width=bw, color=T.protocol_color(proto), edgecolor=T.GREYS["edge"],
               linewidth=0.5, label=_ARM_LABEL[arm], yerr=err,
               error_kw=dict(ecolor=T.GREYS["annot"], elinewidth=0.9, capsize=2.5))
        for xi, v, e in zip(xpos, med, err):
            if np.isfinite(v):
                T.annotate_value(ax, xi, v, annotate_fmt.format(v),
                                 yerr=e if np.isfinite(e) else 0.0)
    ax.set_xticks(x)
    ax.set_xticklabels([_CELL_LABEL[c] for c in _TPUT_CELLS], fontsize=T.FS["small"])


def _dots(ax, df: pd.DataFrame, ycol: str) -> None:
    """Overlay the individual replicates on grouped bars (same geometry as _bars)."""
    x = np.arange(len(_TPUT_CELLS))
    bw = 0.8 / len(_ARMS)
    for j, (arm, _proto) in enumerate(_ARMS):
        for i, cell in enumerate(_TPUT_CELLS):
            vals = pd.to_numeric(
                df[(df["arm"] == arm) & (df["cell"] == cell)][ycol], errors="coerce"
            ).dropna()
            xpos = x[i] + (j - (len(_ARMS) - 1) / 2.0) * bw
            ax.scatter([xpos] * len(vals), vals, s=7, color=T.GREYS["ink"], zorder=4, alpha=0.75)


def make(bundle: RunBundle, saver: T.Saver) -> None:
    wb = getattr(bundle, "wire", None)
    if wb is None or wb.df.empty:
        saver.record_skip(FIG_ID, NAME,
                          "no wire campaign dirs found (pass --wire-results SCG-SESHAT/results)")
        return
    ab = wb.df[wb.df["role"] == "ab"]
    if ab.empty:
        saver.record_skip(FIG_ID, NAME, "no ab-* (kTLS A/B) campaigns in the wire results")
        return
    in_print = T.print_variant()

    tput = ab[ab["cell"].isin(_TPUT_CELLS)]
    lo_tput = tput[tput["medium"] == "loopback"]
    wi_tput = tput[tput["medium"] == "wire"]
    rtt = ab[ab["cell"].str.startswith("rtt-")].copy()
    rtt["message_bytes"] = pd.to_numeric(rtt["message_bytes"], errors="coerce")

    agg_lo = aggregate(lo_tput, ["arm", "cell"], ["throughput_gbps_mean"])
    agg_wi = aggregate(wi_tput, ["arm", "cell"],
                       ["throughput_gbps_mean", "cpu_pct_mean", "gbps_per_core_filled"])
    agg_rtt = aggregate(rtt, ["arm", "medium", "message_bytes"], ["rtt_us_p50"])

    ceiling = pd.to_numeric(wi_tput["ceiling_gbps"], errors="coerce").dropna()
    ceiling = float(ceiling.iloc[0]) if not ceiling.empty else 0.9493

    import matplotlib.pyplot as plt

    if in_print:
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), squeeze=False)
        axa, axcpu, axrtt, axwt = axes[0][0], axes[0][1], axes[0][2], None
    else:
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8), squeeze=False)
        axa, axwt = axes[0][0], axes[0][1]
        axcpu, axrtt = axes[1][0], axes[1][1]

    # --- (a) loopback throughput: the unclamped delta -------------------------------------
    _bars(axa, agg_lo, "throughput_gbps_mean", annotate_fmt="{:.1f}")
    _dots(axa, lo_tput, "throughput_gbps_mean")
    axa.set_ylabel("throughput (Gbit/s)")
    T.panel_title(axa, "loopback (unclamped)")
    top = pd.to_numeric(agg_lo["throughput_gbps_mean_med"], errors="coerce").max()
    if np.isfinite(top):
        axa.set_ylim(0, top * 1.30)  # headroom so the value labels stay inside the panel

    # --- (full) wire throughput: four bars pinned at line rate ----------------------------
    if axwt is not None:
        _bars(axwt, agg_wi, "throughput_gbps_mean", annotate_fmt="{:.3f}")
        _dots(axwt, wi_tput, "throughput_gbps_mean")
        axwt.axhline(ceiling, color=T.ACCENT, lw=1.2, ls=(0, (6, 2)))
        axwt.set_ylim(0, 1.05)
        axwt.set_ylabel("throughput (Gbit/s)")
        T.panel_title(axwt, "wire (link-clamped)")

    # --- (b) wire CPU: where the wire difference lives ------------------------------------
    _bars(axcpu, agg_wi, "cpu_pct_mean", annotate_fmt="{:.1f}")
    _dots(axcpu, wi_tput, "cpu_pct_mean")
    axcpu.set_ylabel("gateway CPU (%)")
    T.panel_title(axcpu, "wire CPU at line rate")
    cpu_top = pd.to_numeric(agg_wi["cpu_pct_mean_med"], errors="coerce").max()
    if np.isfinite(cpu_top):
        axcpu.set_ylim(0, cpu_top * 1.22)  # headroom for the value labels
    wire_range = pd.to_numeric(wi_tput["throughput_gbps_mean"], errors="coerce")
    percore = {arm: agg_wi[(agg_wi["arm"] == arm) & (agg_wi["cell"] == "tput-tls-c1")]
               for arm, _ in _ARMS}
    pc_txt = ""
    if all(not v.empty for v in percore.values()):
        pc_txt = (f"{percore['ktls']['gbps_per_core_filled_med'].iloc[0]:.1f} vs "
                  f"{percore['user']['gbps_per_core_filled_med'].iloc[0]:.1f} Gbit/s per core")

    # --- (c) closed-loop RTT p50 by size --------------------------------------------------
    for arm, proto in _ARMS:
        for medium, ls in (("wire", "-"), ("loopback", ":")):
            sel = agg_rtt[(agg_rtt["arm"] == arm) & (agg_rtt["medium"] == medium)]
            sel = sel.sort_values("message_bytes")
            if sel.empty or sel["rtt_us_p50_med"].isna().all():
                continue
            axrtt.errorbar(
                sel["message_bytes"], sel["rtt_us_p50_med"], yerr=sel["rtt_us_p50_ci95"],
                ls=ls, marker="o" if medium == "wire" else ".", ms=4.5, lw=1.5,
                color=T.protocol_color(proto), capsize=2,
            )
    T.byte_axis(axrtt, "x")
    axrtt.set_yscale("log")
    axrtt.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
    axrtt.set_ylabel("RTT p50 (µs, log)")
    axrtt.set_xlabel("message size")
    T.panel_title(axrtt, "closed-loop RTT")
    axrtt.grid(axis="y", which="both", alpha=0.5)

    # One figure-level key: arm = colour (all panels), medium = linestyle (RTT panel),
    # plus the ceiling guide when the full variant draws it.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=T.protocol_color(proto), edgecolor=T.GREYS["edge"], linewidth=0.5,
              label=_ARM_LABEL[arm])
        for arm, proto in _ARMS
    ] + [
        Line2D([0], [0], color=T.GREYS["annot"], ls="-", marker="o", ms=4.5, label="wire"),
        Line2D([0], [0], color=T.GREYS["annot"], ls=":", marker=".", ms=4.5, label="loopback"),
    ]
    if axwt is not None:
        handles.append(Line2D([0], [0], color=T.ACCENT, lw=1.2, ls=(0, (6, 2)),
                              label=f"goodput ceiling {ceiling:.4f} Gbit/s"))
    T.legend_below(fig, handles, ncol=min(4, len(handles)))

    # --- computed takeaway ----------------------------------------------------------------
    take = ""
    cpu = {arm: agg_wi[(agg_wi["arm"] == arm) & (agg_wi["cell"] == "tput-tls-c1")]
           for arm, _ in _ARMS}
    lo = {arm: {c: agg_lo[(agg_lo["arm"] == arm) & (agg_lo["cell"] == c)]
                for c in _TPUT_CELLS} for arm, _ in _ARMS}
    if all(not v.empty for v in cpu.values()) and wire_range.notna().any():
        k_cpu = float(cpu["ktls"]["cpu_pct_mean_med"].iloc[0])
        u_cpu = float(cpu["user"]["cpu_pct_mean_med"].iloc[0])
        take = (f"On the wire both arms saturate the link "
                f"({wire_range.min():.3f}–{wire_range.max():.3f} Gbit/s); kTLS delivers it "
                f"for {100 * (u_cpu - k_cpu) / u_cpu:.0f}% less CPU ({k_cpu:.1f}% vs "
                f"{u_cpu:.1f}% at 1 conn" + (f"; {pc_txt}" if pc_txt else "") + ")")
        deltas = []
        for c in _TPUT_CELLS:
            a, b = lo["ktls"][c], lo["user"][c]
            if not a.empty and not b.empty:
                deltas.append(
                    (c, 100 * (a["throughput_gbps_mean_med"].iloc[0]
                               / b["throughput_gbps_mean_med"].iloc[0] - 1))
                )
        if deltas:
            take += (". Where the link does not clamp (loopback) kTLS is "
                     + " / ".join(f"+{d:.0f}% ({_CELL_LABEL[c]})" for c, d in deltas))
        take += "."

    wire_sizes = sorted(
        rtt[(rtt["medium"] == "wire")]["message_bytes"].dropna().unique().tolist()
    )
    T.set_headline(
        fig,
        f"{TITLE}  ·  {protocol_label('ktls/1.3+mtls')} vs {protocol_label('tls/1.3+mtls')}",
        y=1.03,
    )
    if take:
        T.add_takeaway(fig, take)
    note = (
        "Same campaign with only the encrypt gateway's prefer_ktls flag toggled; the peer "
        "and every other variable held fixed. 3 replicates per cell: bars = median, "
        "whiskers = t-based CI95, dots = individual replicates. Neither NIC offers TLS "
        "hardware offload — this compares the kernel's software record layer against "
        "user-space TLS; hardware offload remains unmeasured. Throughput is sender-side "
        "(TCP backpressure bounds the difference to one socket buffer)."
    )
    if wire_sizes:
        note += (
            f" The wire RTT series starts at {wire_sizes[0]:.0f} B: the 64 B cells died "
            "under the probe's desync guard (no samples kept rather than corrupted ones) "
            "and are absent, not zero."
        )
    T.add_method_note(fig, note)
    T.add_provenance(fig, bundle.caption() + "  ·  " + wb.provenance({"ab"}))
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
