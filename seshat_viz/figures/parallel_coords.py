"""
F21 — Every configuration at once: a parallel-coordinates trade-off map.

The radar (F5) compares a handful of transports; this puts *every* (transport, protocol)
configuration on one canvas. Each polyline crosses a set of normalized axes — throughput,
Gbps/core, p99 latency, jitter, cycles/byte, peak RSS, loss — all oriented so that **up is
better**. A line that stays high everywhere is a config with no weak axis; a line that dives
on one axis exposes exactly where that configuration pays. This is the multi-objective view
a single scatter or bar cannot give.

Axes are **per-axis rank-normalized**: min-max against UDP's outlier loss/latency
collapses every reliable transport into a thin band near the top, so ranking is used instead —
it spreads the configs evenly on each axis and keeps their real trade-offs visible. An axis
measured for fewer than half the configs is dropped (disclosed in the method note) rather than
padded, and a config's missing value on a kept axis draws as a *gap*, never a fabricated
mid-axis point. The "most balanced" takeaway is scored only over genuine gateway *crypto*
configs with **every drawn axis measured**: routing-only rows — a single-protocol
transport that carries only a ``none`` row, or any ``none`` routing config — are excluded,
since with no encrypted axis they look balanced for the wrong reason, and ``n_gateways==0``
loopback-baseline rows never enter the canvas at all (they are not gateway configurations).
SHM's latency/jitter axes carry ``theme.SHM_STALL_NOTE``: those points are a harness
receive-poll stall, not the SHM transport's capability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F21"
NAME = "f21_parallel_coordinates"
TITLE = "Multi-objective trade-off (parallel coordinates)"

# (column, short axis label, higher_is_better)
_AXES = [
    ("throughput_gbps_mean", "throughput", True),
    ("gbps_per_core", "Gbps/core", True),
    ("latency_p99_us_mean", "p99 latency", False),
    ("jitter_us_mean", "jitter", False),
    ("cycles_per_byte", "cycles/byte", False),
    ("rss_peak_kib", "peak RSS", False),
    ("loss_pct", "loss", False),
]

_Metric = Tuple[str, str, bool]


def _rank_norm(
    grp: pd.DataFrame, metrics: List[_Metric]
) -> Tuple[pd.DataFrame, List[_Metric], List[Tuple[str, int, int]]]:
    """
    Rank-normalize each metric axis across the config table, "up = better".
    Returns ``(norm, kept_metrics, low_coverage)``.

    Two gates run first. An axis whose values are (near-)constant across configs
    normalizes to a meaningless flat column (e.g. loss_pct is 0 for every reliable
    transport when UDP is absent) — dropped. An axis measured for fewer than half the
    configs would put a placeholder on most polylines and decide the balance prize by
    who was *not* measured — dropped, and reported in ``low_coverage`` as
    ``(label, measured, total)`` so the method note can disclose it.

    A missing value stays NaN: it renders as a gap in the polyline and never enters a
    score. A NaN→0.5 mid-axis fill may look neutral but is not — under min-score
    selection a guaranteed 0.5 floor on an unmeasured axis would shield that config
    from ever ranking below the middle, while every measured competitor can.
    """
    kept: List[_Metric] = []
    low_cov: List[Tuple[str, int, int]] = []
    for col, lbl, hi in metrics:
        v = pd.to_numeric(grp[col], errors="coerce").dropna()
        if len(v) < 2 or float(np.ptp(v.to_numpy())) <= 1e-9:
            continue  # non-discriminating: a constant/singleton axis carries no ranking
        if 2 * len(v) < len(grp):
            low_cov.append((lbl, int(len(v)), int(len(grp))))
            continue
        kept.append((col, lbl, hi))

    norm = pd.DataFrame(index=grp.index)
    for col, _lbl, higher in kept:
        v = pd.to_numeric(grp[col], errors="coerce")
        m = int(v.notna().sum())  # ≥ 2 — guaranteed by the gates above
        # ascending rank: for an up-is-better axis the largest raw value ranks best, for a
        # down-is-better axis the smallest does. Ties (e.g. every reliable transport at 0%
        # loss) share the average rank. Scale rank 1..m onto 0..1 (best = 1).
        r = v.rank(method="average", ascending=higher)
        norm[col] = (r - 1.0) / (m - 1.0)
    return norm, kept, low_cov


def _most_balanced(grp: pd.DataFrame, norm: pd.DataFrame) -> Optional[Dict[str, object]]:
    """
    The "most balanced" config: highest MINIMUM normalized score (no weak axis), computed
    from what is actually plotted rather than hard-coded. Returns ``None`` when no config
    may honestly wear the label, else ``{"idx", "ties", "score", "softened"}``.

    Only a genuine gateway *crypto* config is a candidate. A routing-only row
    trivially lacks a crypto axis to trade off, so it looks balanced for the wrong
    reason — single-protocol transports (only a ``none`` row, no encrypted config to
    expose a crypto cost) and any ``none`` routing config are out. With no crypto config
    at all there is no winner: a routing row must never be captioned as one.

    Candidacy further requires every drawn axis measured: a min over fewer axes has fewer
    chances to be low, so a config with a gap could win on coverage rather than balance.
    Only if *no* eligible config is fully measured does the pick fall back to
    the min over measured axes, flagged ``softened`` so the takeaway claims "no measured
    axis" instead of overclaiming.

    Rank ties are exact (shared average ranks), so ``idxmax``'s row-order pick would be
    arbitrary; ties break by best mean rank, then by label, and surviving co-leaders are
    returned in ``ties`` for disclosure.
    """
    protos_per_transport = grp.groupby("transport", observed=True)["protocol"].nunique()
    multi_proto = set(protos_per_transport[protos_per_transport >= 2].index.astype(str))
    elig = grp.apply(
        lambda row: str(row["protocol"]) != "none" and str(row["transport"]) in multi_proto,
        axis=1,
    )
    if not bool(elig.any()):
        return None

    min_scores = norm.min(axis=1)  # skips NaN: a gap never scores
    fully = norm.notna().all(axis=1)
    softened = False
    cand = min_scores[elig & fully]
    if cand.empty:
        cand = min_scores[elig].dropna()
        softened = True
    if cand.empty:
        return None

    top = list(cand[cand >= cand.max() - 1e-9].index)
    mean_scores = norm.mean(axis=1)
    best_mean = float(mean_scores.loc[top].max())
    top = [i for i in top if float(mean_scores.loc[i]) >= best_mean - 1e-9]
    top.sort(key=lambda i: (str(grp.loc[i, "protocol"]), str(grp.loc[i, "transport"])))
    return {"idx": top[0], "ties": top[1:], "score": float(min_scores.loc[top[0]]), "softened": softened}


def make(bundle: RunBundle, saver: T.Saver) -> None:
    if "protocol" not in bundle.summary.columns:
        saver.record_skip(FIG_ID, NAME, "no protocol column")
        return
    d = derive.attach_bytes_from_runs(bundle.summary, bundle.runs)
    d = derive.throughput_scenarios_only(d)
    d = derive.size_match_for_protocol_compare(d, min_protocols=2)
    if d.empty:
        saver.record_skip(FIG_ID, NAME, "no size-matched throughput scenarios")
        return

    # n_gateways==0 loopback baselines are not gateway configurations at all — left in,
    # they blend into the routing polylines' means and shift every config's rank
    # slot. They are excluded from the canvas, not merely from the prize.
    n_loopback = 0
    if "n_gateways" in d.columns:
        gw_rows = pd.to_numeric(d["n_gateways"], errors="coerce").fillna(1) != 0
        n_loopback = int((~gw_rows).sum())
        d = d[gw_rows]
    if d.empty:
        saver.record_skip(FIG_ID, NAME, "only loopback-baseline rows; no gateway configurations")
        return

    # Pin to a single matched 1-conn cell so every polyline is a comparable slice
    # (otherwise TCP's multi-conn aggregate is normalized against single-conn others).
    d, chosen = derive.matched_cell(d, ["transport", "protocol"], fixed={"connections": 1})
    if d.empty:
        saver.record_skip(FIG_ID, NAME, "no matched 1-conn cell across configurations")
        return

    metrics = [(c, lbl, hi) for c, lbl, hi in _AXES if c in d.columns and d[c].notna().any()]
    if len(metrics) < 3:
        saver.record_skip(FIG_ID, NAME, "fewer than 3 usable metric axes present")
        return

    # One row per (transport, protocol): mean of each metric.
    grp = d.groupby(["transport", "protocol"], observed=True)[[c for c, _, _ in metrics]].mean().reset_index()
    grp = grp.dropna(subset=[c for c, _, _ in metrics], how="all")
    if len(grp) < 2:
        saver.record_skip(FIG_ID, NAME, "fewer than 2 configurations to compare")
        return

    norm, metrics, low_cov = _rank_norm(grp, metrics)
    if len(metrics) < 3:
        saver.record_skip(FIG_ID, NAME, "fewer than 3 discriminating metric axes")
        return
    # A config measured only on dropped axes has no drawable vertex left — omit it rather
    # than carry a legend entry for an invisible line.
    visible = norm.notna().any(axis=1)
    grp, norm = grp[visible], norm[visible]
    if len(grp) < 2:
        saver.record_skip(FIG_ID, NAME, "fewer than 2 configurations to compare")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.7 * len(metrics) + 2.0, 6.0))
    xs = np.arange(len(metrics))

    # vertical axis guides
    for j in xs:
        ax.axvline(j, color=T.GREYS["faint"], lw=0.8, zorder=1)

    protos_seen: set[str] = set()
    for idx, r in grp.iterrows():
        proto = str(r["protocol"])
        protos_seen.add(proto)
        # A NaN vertex breaks the line there — an honest gap, not a fabricated point.
        ys = [norm.loc[idx, col] for col, _, _ in metrics]
        ax.plot(xs, ys, color=T.protocol_color(proto), alpha=0.85, lw=1.7,
                marker=T.transport_marker(str(r["transport"])), ms=6,
                markeredgecolor=T.GREYS["ink"], markeredgewidth=0.4, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lbl}\n{'↑' if hi else '↓'} rank" for _c, lbl, hi in metrics], fontsize=T.FS["small"])
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["worst", "mid", "best"], fontsize=T.FS["small"])
    ax.set_ylabel("normalized (up = better)")
    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", ax=ax)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)

    # legends reserved outside the axes by constrained layout (never clipped at the edge).
    ordered = [p for p in T.PROTOCOL_ORDER if p in protos_seen]
    phandles = [plt.matplotlib.patches.Patch(facecolor=T.protocol_color(p), label=protocol_label(p)) for p in ordered]
    transports = [t for t in T.TRANSPORT_ORDER if t in set(grp["transport"].astype(str))]
    thandles = [plt.matplotlib.lines.Line2D([0], [0], marker=T.transport_marker(t), color=T.GREYS["annot"],
                markerfacecolor=T.GREYS["annot"], markersize=8, lw=0, label=transport_label(t)) for t in transports]
    T.legend_right(fig, phandles, title="protocol")
    fig.legend(handles=thandles, title="transport", loc="outside right lower",
               fontsize=T.FS["small"], title_fontsize=T.FS["small"])

    # Data-driven takeaway (crypto-only + measured-axes candidacy — see _most_balanced).
    take = "A line high on every axis has no weak point; a dip marks exactly where a configuration trades off."
    pick = _most_balanced(grp, norm)
    if pick is not None:
        r = grp.loc[pick["idx"]]
        qual = " measured" if pick["softened"] else ""
        tie_txt = ""
        if pick["ties"]:
            tie_txt = "; tied with " + ", ".join(
                f"{protocol_label(str(grp.loc[i, 'protocol']))} · {transport_label(str(grp.loc[i, 'transport']))}"
                for i in pick["ties"])
        take = (f"Most balanced crypto config across all objectives: {protocol_label(str(r['protocol']))} · "
                f"{transport_label(str(r['transport']))} (no{qual} axis below "
                f"{pick['score']:.0%} of best{tie_txt}); a dip on any line marks that config's weakest objective. "
                "Routing-only rows (single-protocol transports, any `none` config) are excluded — no crypto axis to trade off.")
    T.add_takeaway(fig, take)

    axes_txt = ", ".join(lbl for _c, lbl, _h in metrics)
    note = (f"per-axis rank-normalized over the matched {T.fmt_cell(chosen)} slice (rank, not min-max: "
            f"min-max against UDP's outlier flattens the reliable transports); axes drawn: {axes_txt} "
            "(non-discriminating or perf-only axes omitted)")
    if low_cov:
        note += "; low-coverage axes dropped: " + ", ".join(
            f"{lbl} (measured {m}/{n} configs)" for lbl, m, n in low_cov)
    gaps = int((~norm.notna().all(axis=1)).sum())
    if gaps:
        note += f"; {gaps} config(s) miss an axis (line gap, not scored)"
    if "n_gateways" in d.columns:
        # The pinned cell controls connections and size but not chain length: disclose that
        # a polyline's mean averages its 1-gw and 2-gw rows.
        ngu = d.groupby(["transport", "protocol"], observed=True)["n_gateways"].nunique()
        gw_vals = sorted({int(v) for v in pd.to_numeric(d["n_gateways"], errors="coerce").dropna().unique()})
        blended = int((ngu > 1).sum())
        if blended and len(gw_vals) > 1:
            note += (f"; polyline means blend n_gateways={'/'.join(str(v) for v in gw_vals)} rows "
                     f"({blended}/{len(ngu)} configs)")
        if n_loopback:
            note += f"; loopback (0-gw) rows excluded: {n_loopback}"
    note += (". TRANSPORT_ORDER (SHM,UDS,TCP,TPROXY,UDP) is a fixed IPC-locality legend, not a ranking — "
             "the measured routing order is TPROXY≈TCP>UDS>SHM (see F5).")
    metric_cols = {c for c, _, _ in metrics}
    if "shm" in set(grp["transport"].astype(str)) and (metric_cols & {"latency_p99_us_mean", "jitter_us_mean"}):
        note += "  ·  " + T.SHM_STALL_NOTE  # SHM's latency/jitter axes are a harness stall, not capability
    T.add_method_note(fig, note)
    T.add_provenance(fig, bundle.caption() + "  ·  axes rank-normalized per objective across shown configs; size-matched throughput runs")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
