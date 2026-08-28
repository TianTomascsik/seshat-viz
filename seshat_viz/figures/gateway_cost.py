"""
F6 — Second-gateway insertion cost: 1 gateway (scg-direct) vs 2 gateways (scg-scg).

Pairs scenarios that share (transport, protocol, size) but differ only in gateway count:
'direct' == scg-direct (ONE gateway process), 'scg' == scg-scg (TWO gateway processes).
BOTH are through a gateway — this isolates the cost of inserting a *second* gateway, NOT
proxy-vs-no-proxy (the true no-gateway baseline is the 'loopback' family). Dumbbells show
1-gateway → 2-gateway for throughput and p99 latency.

Pair hygiene — each guard exists because without it the sorted extremes of this chart are
measurement pathology, not topology cost:

- Scenarios with a zero-work repetition are excluded before pairing: their summary means
  average dead SHM/UDS multi-connection repetitions with live ones, and whether the dead
  level lands on the 1-gw or 2-gw side is arbitrary (audit F6-1/D2-1).
- A matched pair with a load-generator-bound side carries no cost information — the ratio
  of two harness floors is noise — so such pairs never enter Δ. Rows where EVERY matched
  pair is bound are drawn grey with no Δ claim (audit F6-3, F15's bottleneck convention).
- Δ and the drawn dumbbell are the SAME experimental pair: the matched connection count
  whose 2gw/1gw ratio is the median of the harness-clean sweep, so the number printed is
  always the delta of the dots next to it (audit F6-2). If the sweep's ratios spread
  beyond ``_SPREAD_MAX``, no single summary is defensible; the lowest-concurrency pair
  (least exposed to multi-connection pathology) is shown and flagged '†'.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F6"
NAME = "f06_gateway_insertion_cost"
TITLE = "Second-gateway insertion cost: 1 gateway (scg-direct) vs 2 gateways (scg-scg)"

# A genuine insertion cost is roughly consistent across the matched connection sweep; a
# max/min ratio spread beyond this is the bimodal multi-connection signature (observed up
# to 158× on contaminated sweeps) and disqualifies the sweep median.
_SPREAD_MAX = 2.0
_GREY = T.GREYS["faint"]  # rows/endpoints with no harness-clean pair (lower bounds only)
_ANNOT = T.GREYS["annot"]
_ANNOT_GREY = T.GREYS["muted"]

_KEYS = ["transport", "protocol", "message_bytes"]
_METRIC = "throughput_gbps_mean"
_LAT = "latency_p99_us_mean"


def _label(row) -> str:
    return f"{transport_label(str(row['transport']))} · {protocol_label(str(row['protocol']))} · {T.fmt_bytes(row['message_bytes'])}B"


def _hl_by_conns(side: pd.DataFrame) -> pd.Series:
    """Per connection count: True if any constituent row is load-generator bound."""
    if "harness_limited" not in side.columns:
        return pd.Series(False, index=side["connections"].unique())
    flags = side["harness_limited"].map(lambda v: bool(v) if pd.notna(v) else False)
    return flags.groupby(side["connections"], observed=True).any()


def _pair_rows(summary: pd.DataFrame, dead: set) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Form direct/scg pairs per (transport, protocol, message_bytes), matched on connection
    count, with the hygiene guards documented in the module docstring. `dead` is the
    zero-work-repetition scenario set from :func:`derive.dead_repeat_scenarios`.

    Returns ``(rows, exclusions)``. `rows` columns: the three keys, plus
    ``conns/direct/scg`` (the drawn pair), ``d_hl/s_hl`` (endpoint bound flags),
    ``lat_direct/lat_scg`` (p99 at the same pair, NaN if absent), ``delta_pct`` (NaN when
    no clean pair exists), ``n_pairs/n_clean``, and ``flag`` in {'ok','unstable','bound'}.
    `exclusions` carries what could NOT be paired, for the method note:
    ``missing_side`` maps 'transport · protocol' → #sizes with one topology entirely
    absent; ``dead_groups`` counts groups un-pairable only after the dead-repeat drop;
    ``dead_scenarios`` counts excluded matrix scenarios.
    """
    excl: Dict[str, object] = {"missing_side": {}, "dead_groups": 0, "dead_scenarios": 0}
    needed = {*_KEYS, "chain", "connections", _METRIC}
    if not needed.issubset(summary.columns):
        return pd.DataFrame(), excl
    m = summary
    if "family" in m.columns:
        # Matrix family only: there '_direct'/'_scg' are explicit topology tokens and both
        # sides are sustained blast at the same shape (elsewhere 'chain' is a loader guess).
        m = m[m["family"].astype(str) == "matrix"]
    if m.empty:
        return pd.DataFrame(), excl
    if dead and "scenario" in m.columns:
        excl["dead_scenarios"] = int(m["scenario"].astype(str).isin(dead).sum())

    rows: List[dict] = []
    for kv, grp in m.groupby(_KEYS, observed=True):
        d_all = grp[grp["chain"] == "direct"]
        s_all = grp[grp["chain"] == "scg"]
        if d_all.empty or s_all.empty:
            # One topology never ran (e.g. TPROXY 2-gw): disclose, aggregated per
            # transport·protocol so the note stays short.
            lbl = f"{transport_label(str(kv[0]))} · {protocol_label(str(kv[1]))}"
            excl["missing_side"][lbl] = excl["missing_side"].get(lbl, 0) + 1
            continue
        if dead and "scenario" in grp.columns:
            d_all = d_all[~d_all["scenario"].astype(str).isin(dead)]
            s_all = s_all[~s_all["scenario"].astype(str).isin(dead)]
        if d_all.empty or s_all.empty:
            excl["dead_groups"] += 1
            continue

        dmap = d_all.groupby("connections", observed=True)[_METRIC].mean()
        smap = s_all.groupby("connections", observed=True)[_METRIC].mean()
        dlat = (d_all.groupby("connections", observed=True)[_LAT].mean()
                if _LAT in d_all.columns else pd.Series(dtype=float))
        slat = (s_all.groupby("connections", observed=True)[_LAT].mean()
                if _LAT in s_all.columns else pd.Series(dtype=float))
        dhl = _hl_by_conns(d_all)
        shl = _hl_by_conns(s_all)

        shared = sorted(c for c in dmap.index if c in smap.index
                        and np.isfinite(dmap[c]) and np.isfinite(smap[c]) and dmap[c] > 0)
        if not shared:
            excl["dead_groups"] += 1
            continue
        pairs = [(int(c), float(dmap[c]), float(smap[c]), float(smap[c] / dmap[c]),
                  bool(dhl.get(c, False)), bool(shl.get(c, False))) for c in shared]
        clean = [p for p in pairs if not (p[4] or p[5])]

        if clean:
            ratios = [p[3] for p in clean]
            if max(ratios) / min(ratios) > _SPREAD_MAX:
                rep, flag = clean[0], "unstable"
            else:
                med = float(np.median(ratios))
                rep, flag = min(clean, key=lambda p: abs(p[3] - med)), "ok"
            delta = (rep[3] - 1.0) * 100.0
        else:
            # Every matched pair has a load-generator-bound side: both values are lower
            # bounds, their ratio is noise — draw the lowest-concurrency pair, claim nothing.
            rep, flag, delta = pairs[0], "bound", float("nan")

        c0 = rep[0]
        rows.append({
            **dict(zip(_KEYS, kv)),
            "conns": c0,
            "direct": rep[1],
            "scg": rep[2],
            "d_hl": rep[4],
            "s_hl": rep[5],
            "lat_direct": float(dlat[c0]) if c0 in dlat.index and np.isfinite(dlat.get(c0, np.nan)) else np.nan,
            "lat_scg": float(slat[c0]) if c0 in slat.index and np.isfinite(slat.get(c0, np.nan)) else np.nan,
            "delta_pct": delta,
            "n_pairs": len(pairs),
            "n_clean": len(clean),
            "flag": flag,
        })
    return pd.DataFrame(rows), excl


def _annotation(r) -> Tuple[str, str]:
    """(text, color) printed left of a dumbbell; always describes the drawn pair."""
    if r["flag"] == "bound":
        return "no Δ — load-gen bound", _ANNOT_GREY
    mark = "†" if r["flag"] == "unstable" else ("*" if r["n_clean"] == 1 else "")
    return f"{r['delta_pct']:+.0f}%{mark} @{r['conns']}c", _ANNOT


def make(bundle: RunBundle, saver: T.Saver) -> None:
    df = bundle.summary
    if "chain" not in df.columns:
        saver.record_skip(FIG_ID, NAME, "no direct/scg chain information")
        return

    dead = derive.dead_repeat_scenarios(bundle.runs)
    tbl, excl = _pair_rows(df, dead)
    if tbl.empty:
        saver.record_skip(FIG_ID, NAME, "no direct/scg pairs share (transport,protocol,size)")
        return

    tbl = tbl.assign(_lbl=tbl.apply(_label, axis=1))
    # Order: quantified rows sorted so the largest second-gateway PENALTY sits at the top
    # (y grows upward); unquantifiable (all-bound) rows form a grey block at the bottom.
    scored = tbl[tbl["delta_pct"].notna()].sort_values("delta_pct", ascending=False)
    bound = tbl[tbl["delta_pct"].isna()].sort_values(_KEYS)
    plot = pd.concat([bound, scored], ignore_index=True)
    y = np.arange(len(plot))

    import matplotlib.pyplot as plt

    have_lat = bool((plot["lat_direct"].gt(0) & plot["lat_scg"].gt(0)).any())
    ncol = 2 if have_lat else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.0 * ncol, 0.46 * len(plot) + 2.0), squeeze=False)

    axT = axes[0][0]
    # Shared x-range, for placing each row's annotation on whichever side of the dumbbell
    # has more room (low-Gbps rows would otherwise print into the y tick labels).
    xlo = float(plot[["direct", "scg"]].min().min())
    xhi = float(plot[["direct", "scg"]].max().max())
    for i, (_, r) in enumerate(plot.iterrows()):
        is_bound = r["flag"] == "bound"
        color = _GREY if is_bound else T.transport_color(str(r["transport"]))
        axT.plot([r["direct"], r["scg"]], [i, i], color=color, lw=2.2,
                 ls=(":" if is_bound else "-"), solid_capstyle="round", zorder=2)
        axT.scatter([r["direct"]], [i], facecolor="white", edgecolor=color, s=55, zorder=3)
        axT.scatter([r["scg"]], [i], color=color, s=70, zorder=4)
        text, tcolor = _annotation(r)
        lo, hi = min(r["direct"], r["scg"]), max(r["direct"], r["scg"])
        if (xhi - hi) >= (lo - xlo):
            anchor, offset, align = hi, 6, "left"
        else:
            anchor, offset, align = lo, -6, "right"
        axT.annotate(text, (anchor, i),
                     xytext=(offset, 0), textcoords="offset points", ha=align, va="center",
                     fontsize=T.FS["annot"], color=tcolor)
    # Tight top margin; keep a proportional band clear at the bottom — the method notes
    # (figure coordinates) render over it on tall figures.
    ylim = (-(0.05 * len(plot) + 1.0), len(plot) - 0.2)
    axT.set_ylim(*ylim)
    axT.set_yticks(y)
    axT.set_yticklabels(plot["_lbl"], fontsize=T.FS["annot"])
    axT.set_xlabel("throughput (Gbps)")
    T.panel_title(axT, "Throughput")
    axT.grid(axis="x")

    if ncol == 2:
        axL = axes[0][1]
        for i, (_, r) in enumerate(plot.iterrows()):
            if not (r["lat_direct"] > 0 and r["lat_scg"] > 0):
                continue
            color = _GREY if r["flag"] == "bound" else T.transport_color(str(r["transport"]))
            axL.plot([r["lat_direct"], r["lat_scg"]], [i, i], color=color, lw=2.2,
                     ls=(":" if r["flag"] == "bound" else "-"), solid_capstyle="round", zorder=2)
            axL.scatter([r["lat_direct"]], [i], facecolor="white", edgecolor=color, s=55, zorder=3)
            axL.scatter([r["lat_scg"]], [i], color=color, s=70, zorder=4)
        axL.set_xscale("log")
        axL.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
        axL.set_ylim(*ylim)
        axL.set_yticks(y)
        axL.set_yticklabels([])
        axL.set_xlabel("p99 latency (µs, log)")
        T.panel_title(axL, "p99 latency")
        axL.grid(axis="x")

    handles = [
        plt.matplotlib.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                                    markeredgecolor=T.GREYS["annot"], markersize=8,
                                    label="1 gateway (scg-direct)"),
        plt.matplotlib.lines.Line2D([0], [0], marker="o", color="none",
                                    markerfacecolor=T.GREYS["annot"],
                                    markeredgecolor=T.GREYS["annot"], markersize=8,
                                    label="2 gateways (scg-scg)"),
        plt.matplotlib.lines.Line2D([0], [0], color=_GREY, ls=":", lw=2.2,
                                    label="load-generator-bound (both are lower bounds — no Δ)"),
    ]
    T.legend_below(fig, handles, ncol=3)
    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.02)

    # Two method lines: pairing semantics, then coverage exclusions (all counts computed
    # from this bundle). Stack the first a text-height above the default slot.
    gap = 0.11 / max(fig.get_size_inches()[1], 1.0)
    T.add_method_note(
        fig,
        "'direct'=scg-direct (1 gateway process), 'scg'=scg-scg (2 gateways); no-gateway baseline is "
        "the loopback family. Δ and dots are the SAME matched pair (equal connection count on both "
        "sides), chosen as the median 2gw/1gw throughput ratio of the harness-clean sweep; "
        "'@Nc'=its connection count, *=single clean pair (no sweep support), "
        f"†=sweep ratios spread >{_SPREAD_MAX:g}× (lowest-concurrency pair shown), "
        "grey=every matched pair load-generator-bound (no Δ).",
        y=0.020 + gap,
    )
    excl_bits = []
    if excl["dead_scenarios"]:
        excl_bits.append(f"{excl['dead_scenarios']} scenarios with a zero-work repetition dropped")
    if excl["dead_groups"]:
        excl_bits.append(f"{excl['dead_groups']} groups thereby left unpairable")
    for lbl, n in sorted(excl["missing_side"].items()):
        excl_bits.append(f"{lbl} lacks one topology entirely ({n} sizes)")
    excl_txt = ("Excluded: " + "; ".join(excl_bits) + ". ") if excl_bits else ""
    T.add_method_note(fig, excl_txt + T.BLAST_LATENCY_NOTE, y=0.020)
    if len(scored):
        med = float(scored["delta_pct"].median())
        T.add_takeaway(
            fig,
            f"Median 2-gateway vs 1-gateway throughput delta across {len(scored)} harness-clean "
            f"matched groups: {med:+.1f}% — the quantifiable per-hop insertion cost.",
        )
    T.add_provenance(fig, bundle.caption())
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
