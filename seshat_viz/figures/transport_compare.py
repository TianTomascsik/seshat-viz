"""
F5 — Transport comparison: multi-metric radar + absolute bars.

For a matched workload (routing at a shared payload size), each transport's normalized
profile across six axes — throughput, low latency, low jitter, CPU efficiency, low CPU
use, low loss — on a radar, with a companion grouped bar panel giving the absolute
throughput and latency numbers (with CI95). On a single loopback host there is no NIC for
SHM/UDS to bypass, so the figure's job is to show that no single transport wins every
axis — not to demonstrate a shared-memory advantage that only exists on a real wire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, transport_label

FIG_ID = "F5"
NAME = "f05_transport_radar"
TITLE = "Transport comparison: normalized profile + absolute metrics"

# (column, higher_is_better, axis label)
_AXES = [
    ("throughput_gbps_mean", True, "throughput"),
    ("latency_p99_us_mean", False, "low latency"),
    ("jitter_us_mean", False, "low jitter"),
    ("gbps_per_core", True, "CPU efficiency"),
    ("cpu_pct_mean", False, "low CPU use"),
    ("loss_pct", False, "low loss"),
]


def _pick_size(d: pd.DataFrame) -> int | None:
    """
    The payload size shared by the most transports for routing — and, among ties, the
    LARGEST such size, since bulk-throughput differences (e.g. SHM's advantage) are most
    representative away from the tiny-message overhead-dominated regime.
    """
    routing = d[d["protocol"].astype(str) == "none"]
    if routing.empty:
        routing = d  # fall back to any protocol
    best, best_key = None, (0, -1)
    for size, g in routing.groupby("message_bytes", observed=True):
        n = g["transport"].nunique()
        key = (n, int(size))  # maximize transports first, then prefer the larger size
        if key > best_key:
            best, best_key = size, key
    return int(best) if best is not None and best_key[0] >= 2 else None


def _axis_norm(values, higher_is_better: bool, col: str) -> np.ndarray:
    """
    Normalize one radar axis to [0, 1] with outer = best.

    Baseline scheme: ratio to the best transport (higher_is_better: v/max, else min/v) so
    the ordering structure is preserved and the worst transport does NOT collapse to the
    center (min-max would map it to exactly 0 regardless of the real gap). Three honesty
    rules on top:

      * NaN stays NaN — a transport with no measurement on an axis gets a polygon gap,
        never an imputed vertex (F5-3: median-filling invented UDP a "low jitter" score);
      * ``loss_pct`` is a bounded percentage whose achievable best is 0, so it maps
        absolutely to the delivered fraction (1 - loss/100). A ratio over the positive
        values only would crown the sole lossy transport "best" while the true zero-loss
        rows also land at 1.0 — erasing and inverting the axis (F5-1);
      * generic lower-is-better with best == 0 ("free"): only 0 earns 1.0; any positive
        value is unboundedly worse than free and collapses to the center.
    """
    v = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    nan_mask = ~np.isfinite(v)
    if nan_mask.all():
        return np.full(len(v), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        if col == "loss_pct":
            scaled = 1.0 - v / 100.0
        elif higher_is_better:
            best = np.nanmax(v)
            scaled = v / best if best else np.ones(len(v))
        else:
            best = np.nanmin(v)
            if best > 0:
                scaled = best / v
            else:
                scaled = np.where(v == 0.0, 1.0, 0.0)
    scaled = np.asarray(scaled, dtype=float)
    scaled[nan_mask] = np.nan  # np.where above may have filled gaps — reassert them
    return np.clip(scaled, 0.0, 1.0)


def make(bundle: RunBundle, saver: T.Saver) -> None:
    df = bundle.summary
    if not {"transport", "protocol", "message_bytes", "throughput_gbps_mean"}.issubset(df.columns):
        saver.record_skip(FIG_ID, NAME, "needs transport/protocol/size/throughput")
        return
    # Sustained-blast routing rows only: paced (lat_/iface-latency) and ping-pong rows would
    # otherwise dilute TCP's mean (its true routing throughput is ~35 Gbps, not the ~12 Gbps
    # the paced rows produced) and inject multi-second "latency" into the profile.
    d = derive.throughput_scenarios_only(df)
    d = d[d["message_bytes"].notna()].copy()
    size = _pick_size(d)
    if size is None:
        saver.record_skip(FIG_ID, NAME, "no payload size shared by ≥2 transports")
        return

    sel = d[(d["message_bytes"] == size) & (d["protocol"].astype(str) == "none")]
    # Pin connections==1 so all transports compare like-for-like: TCP rows otherwise
    # aggregate several connection counts while SHM/UDS/UDP only have single-conn rows.
    if "connections" in sel.columns:
        conns = pd.to_numeric(sel["connections"], errors="coerce")
        sel = sel[conns.isin([1]) | conns.isna()]
    # Hold EVERY transport at a real ≥1-gateway hop BEFORE choosing its topology. The raw
    # zero-gateway loopback baseline (iface_tcp_loopback: n_gateways==0, datapath 'loopback')
    # never traverses the SCG — its 368µs "p99" and NaN CPU would otherwise win TCP's per-
    # transport min below and confound the profile against the 1-hop gateway rows the other
    # transports are pinned at (making TCP read as artificially fast, low-latency, and CPU-free).
    if "datapath" in sel.columns:
        sel = sel[sel["datapath"].astype(str) != "loopback"]
    if "n_gateways" in sel.columns and sel["n_gateways"].notna().any():
        ng_all = pd.to_numeric(sel["n_gateways"], errors="coerce")
        sel = sel[(ng_all >= 1) | ng_all.isna()]
    # Then prefer a single-gateway topology per transport (fewest *secured* hops available) so
    # the profile is not confounded by a 1-hop scg-direct row on one transport vs a 2-hop
    # scg-scg matrix row on another.
    if "n_gateways" in sel.columns and sel["n_gateways"].notna().any():
        keep = []
        for _t, g in sel.groupby("transport", observed=True):
            ng = pd.to_numeric(g["n_gateways"], errors="coerce")
            keep.append(g[ng == ng.min()])
        sel = pd.concat(keep, ignore_index=True) if keep else sel
    # harness_limited vanishes in the numeric mean below — capture the row-level flags of
    # the plotted rows first so the method note can bound the absolute-Gbps claim.
    sel = sel[sel["transport"].notna()]
    hl = sel["harness_limited"].astype("boolean") if "harness_limited" in sel.columns else None
    sel = sel.groupby("transport", observed=True).mean(numeric_only=True).reset_index()
    sel = sel[sel["transport"].notna()]
    order = [t for t in T.TRANSPORT_ORDER if t in set(sel["transport"].astype(str))]
    order += [t for t in sel["transport"].astype(str).unique() if t not in order]
    sel["__o"] = sel["transport"].astype(str).map({t: i for i, t in enumerate(order)})
    sel = sel.sort_values("__o").reset_index(drop=True)
    if len(sel) < 2:
        saver.record_skip(FIG_ID, NAME, "fewer than 2 transports at the matched size")
        return

    # Which radar axes are actually available AND discriminating? Drop an axis whose values
    # are all equal across transports (e.g. loss_pct is 0 for every reliable transport): a
    # degenerate axis min-max-normalizes to all-1.0 and falsely reads as "measured & maximal".
    axes_used = []
    for (c, hib, lbl) in _AXES:
        if c not in sel.columns or not sel[c].notna().any():
            continue
        v = pd.to_numeric(sel[c], errors="coerce").dropna()
        if len(v) < 2:
            continue  # measured on <2 transports: nothing to compare (gaps aren't imputed)
        if float(np.ptp(v.to_numpy())) < 1e-12:
            continue  # non-discriminating axis
        axes_used.append((c, hib, lbl))
    if len(axes_used) < 3:
        saver.record_skip(FIG_ID, NAME, "fewer than 3 discriminating radar axes")
        return

    # Normalize each axis (see _axis_norm for the scheme and its honesty rules). A missing
    # per-transport value stays NaN and is tracked so the method note can name the gaps.
    norm = {}
    missing: dict[str, list[str]] = {}  # axis label -> transports with no measurement
    for col, higher, lbl in axes_used:
        vals = pd.to_numeric(sel[col], errors="coerce")
        gaps = [transport_label(str(t)) for t, bad in zip(sel["transport"], vals.isna()) if bad]
        if gaps:
            missing[lbl] = gaps
        norm[lbl] = _axis_norm(vals, higher, col)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 4.8))
    axr = fig.add_subplot(1, 2, 1, polar=True)
    labels = [lbl for _c, _h, lbl in axes_used]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    for i, t in enumerate(sel["transport"].astype(str)):
        vals = np.array([norm[lbl][i] for lbl in labels])
        vals_closed = np.concatenate([vals, vals[:1]])
        # NaN vertices break the outline into a visible GAP (unmeasured, not a value); the
        # small markers keep any isolated measured vertex visible. Fill only fully-measured
        # polygons so no edge is ever drawn across an unmeasured axis.
        axr.plot(angles_closed, vals_closed, color=T.transport_color(t), lw=2,
                 marker="o", ms=3, label=transport_label(t))
        if np.isfinite(vals).all():
            axr.fill(angles_closed, vals_closed, color=T.transport_color(t), alpha=0.12)
    axr.set_xticks(angles)
    axr.set_xticklabels(labels, fontsize=T.FS["small"])
    axr.set_yticks([0.25, 0.5, 0.75, 1.0])
    axr.set_yticklabels(["¼", "½", "¾", "best"], fontsize=T.FS["annot"],
                        color=T.GREYS["muted"])
    axr.set_ylim(0, 1.05)
    T.panel_title(axr, "profile (normalized per axis; outer = best)")
    axr.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=len(sel),
               fontsize=T.FS["small"])

    # Companion absolute bars: throughput (left axis) + p99 latency (right axis).
    axb = fig.add_subplot(1, 2, 2)
    x = np.arange(len(sel))
    w = 0.38
    colors = [T.transport_color(str(t)) for t in sel["transport"]]
    yerr = sel.get("throughput_gbps_ci95")
    axb.bar(x - w / 2, sel["throughput_gbps_mean"], width=w, color=colors,
            yerr=yerr if yerr is not None else None, capsize=3, label="throughput")
    axb.set_ylabel("throughput (Gbps)")
    axb.set_xticks(x)
    axb.set_xticklabels([transport_label(str(t)) for t in sel["transport"]])
    if "latency_p99_us_mean" in sel.columns:
        axb2 = axb.twinx()
        axb2.plot(x, sel["latency_p99_us_mean"], "o--", color=T.GREYS["ink"],
                  label="p99 latency")
        axb2.set_yscale("log")
        axb2.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
        axb2.set_ylabel("p99 latency (µs, log)")
        axb2.grid(False)
        # The ink-grey line is a second data series on a twin axis: key it together
        # with the throughput bars in one in-axes legend on the bars axes.
        handles = (axb.get_legend_handles_labels()[0]
                   + axb2.get_legend_handles_labels()[0])
        T.legend_inline(axb, handles=handles, loc="upper center")
    T.panel_title(axb, "absolute throughput & latency")

    T.set_headline(fig, f"{TITLE} — routing · {T.fmt_bytes(size)}B")

    # Data-driven takeaway: the *measured* transport order, and why it is not the assumed
    # SHM>UDS>TCP>TPROXY>UDP "IPC-locality" hierarchy. On a single loopback host there is no NIC
    # for SHM/UDS to bypass, so in-kernel TCP loopback legitimately matches/beats them; the
    # locality hierarchy only emerges on a bandwidth-bound wire.
    order_t = sel.sort_values("throughput_gbps_mean", ascending=False)
    tput_order = " > ".join(transport_label(str(t)) for t in order_t["transport"])
    take = f"No single ranking on loopback: by throughput {tput_order}"
    if "latency_p99_us_mean" in sel.columns and sel["latency_p99_us_mean"].notna().any():
        order_l = sel.sort_values("latency_p99_us_mean")
        lat_order = " > ".join(transport_label(str(t)) for t in order_l["transport"])
        take += f"; by low p99 latency {lat_order}"
    take += (" — with no wire for SHM/UDS to bypass, the SHM>UDS>TCP>TPROXY>UDP locality order "
             "appears only on a bandwidth-bound NIC, not here.")
    T.add_takeaway(fig, take)

    # Two stacked footer lines (savefig.bbox='tight' widens the canvas to the longest text
    # line, so one concatenated note would dominate the figure width): line 1 = the matched
    # cell and its caveats; line 2 = how the radar/bars must be read.
    slice_note = (
        "routing · " + T.fmt_cell({"message_bytes": size, "connections": 1})
        + f" · payload pinned to {T.fmt_bytes(size)}B — the largest size shared by every transport "
        "(ties among equally-shared sizes break toward the larger one, where bulk-throughput gaps "
        "dominate over tiny-message overhead); every transport held at a real ≥1-gateway hop. "
        + T.SHM_STALL_NOTE + " · " + T.BLAST_LATENCY_NOTE
    )
    reading = []
    if any(c == "loss_pct" for c, _h, _l in axes_used):
        reading.append("'low loss' = delivered fraction (1 − loss %): zero-loss transports sit at "
                       "the outer ring and measured loss pulls the vertex inward")
    if missing:
        gaps = "; ".join(f"{lbl} unmeasured for {', '.join(ts)}" for lbl, ts in missing.items())
        reading.append(f"an unmeasured axis is a polygon gap, never imputed ({gaps})")
    # Bound the absolute panel honestly: when the load generator (not the gateway) is the
    # bottleneck, the Gbps bars are lower bounds and only the relative ordering is a claim.
    if hl is not None and hl.notna().any():
        n_lim, n_rows = int(hl.fillna(False).sum()), int(len(hl))
        if n_lim == n_rows:
            reading.append("all plotted rows are harness-limited — absolute Gbps is a "
                           "load-generator lower bound; compare transports relatively only")
        elif n_lim:
            reading.append(f"{n_lim}/{n_rows} plotted rows are harness-limited — their absolute "
                           "Gbps are load-generator lower bounds")
    reading.append("display order (SHM,UDS,TCP,TPROXY,UDP) is a fixed IPC-locality legend, not a ranking.")
    # Constrained layout parks the radar legend at y≈0.02–0.035 regardless of its anchor,
    # so the notes stack ABOVE that band (the default y=0.020 renders under the swatches).
    T.add_method_note(fig, slice_note, y=0.052)
    T.add_method_note(fig, " · ".join(reading), y=0.036)
    T.add_provenance(fig, bundle.caption() + f"  ·  {bundle.label}")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
