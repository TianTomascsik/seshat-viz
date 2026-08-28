"""
F12 — System-metrics timeline, transports compared.

For each transport, the /proc timeseries sampled during the run: gateway CPU% and resident
memory (twin axes) on top, and the context-switch rate below. Shows the warm-up ramp,
steady-state plateau, any memory growth, and scheduling pressure over the measured window —
the temporal view the single-number summaries cannot give.

Panels ARE mutually comparable here: the selection pins protocol (routing/plaintext), a
single gateway, connection count and message size to one shared value, and varies *only the
transport* (see :func:`_pin_and_pick`). Every CPU / RSS / context-switch difference is then
attributable to the transport mechanism alone (kernel TCP splice vs UDS vs the SHM ring vs
the UDP datagram path vs TPROXY) rather than confounded by a different protocol or topology.
Eligibility is restricted to sustained-blast rows on the default data path
(:func:`derive.throughput_scenarios_only`): paced runs, connection-rate sweeps and the SHM
zero-copy slot-ring microbenchmarks carry /proc timeseries too, but run a different data
path and/or schedule, so admitting them would silently swap a non-default variant into a
panel. Its sibling F22 holds the transport fixed and varies the protection mode instead
(TLS vs kTLS vs mTLS). When no comparable cross-transport slice exists, F12 falls back to
one highest-throughput blast scenario per transport and labels itself NOT comparable.

Each panel is titled transport · protocol · gateway-count · number of PIDs summed · achieved
Gbps (+ a harness-limited tag when the load generator, not the gateway, was the bottleneck —
pinning the configured workload does not pin the achieved one). The /proc timeseries carries
no phase marker, so the steady-state measurement window(s) are recovered from the CPU trace
itself and shaded (see :func:`_steady_spans`). Per-repetition gateway restarts reset the
cumulative context-switch counter, so naive-diff negatives are masked as gaps rather than
plotted. The three overlaid series (CPU / RSS / ctxsw) and the steady-state shading are
keyed by a figure-level legend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import TRANSPORT_ORDER, RunBundle

FIG_ID = "F12"
NAME = "f12_system_metrics_timeline"
TITLE = "System-metrics timeline — transports compared"

# All transports fit (5): the whole point of the interface view is to line every transport
# up under the identical workload. The protocol sibling (F22) curates a shorter crypto ladder.
_MAX_SCEN = 5


def _ctxsw_rate(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Total context-switch rate (per second) from cumulative voluntary+involuntary counts."""
    cols = [c for c in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches") if c in g.columns]
    if not cols or "elapsed_ms" not in g.columns:
        return np.array([]), np.array([])
    total = g[cols].sum(axis=1).values.astype(float)
    t = g["elapsed_ms"].values.astype(float) / 1000.0
    if len(t) < 2:
        return np.array([]), np.array([])
    dt = np.diff(t)
    dcs = np.diff(total)
    rate = np.where(dt > 0, dcs / dt, 0.0)
    # Per-repetition gateway restarts reset the cumulative counter to ~0, so a naive diff
    # yields huge NEGATIVE "rates" that wreck the axis. Mask negatives as gaps (a restart is
    # not a real rate sample) rather than plotting physically impossible values.
    rate = np.where(rate < 0, np.nan, rate)
    return t[1:], rate


def _steady_spans(t: np.ndarray, cpu: np.ndarray) -> list[tuple[float, float]]:
    """
    Contiguous time spans where the gateway is at working load — the steady-state
    "measurement windows" the single-number summaries average over. The /proc timeseries
    carries no phase marker (warmup/measure/cooldown boundaries are not recorded), so we
    recover the windows from the CPU trace itself: a sample counts as *measuring* once CPU
    rises past 40% of the plateau (its p90), which excludes the initial warm-up ramp and
    the brief inter-repetition restart dips. Returns one (t0, t1) per contiguous window —
    several for transports that idle between reps, a single span when the trace never drops
    below the threshold (e.g. a busy-polling path that keeps every rep boundary hot).
    """
    cpu = np.asarray(cpu, dtype=float)
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(cpu)
    if finite.sum() < 3 or len(t) != len(cpu):
        return []
    thr = 0.40 * float(np.nanpercentile(cpu[finite], 90))
    if not np.isfinite(thr) or thr <= 0:
        return []
    active = finite & (cpu >= thr)
    spans: list[tuple[float, float]] = []
    i, n = 0, len(t)
    while i < n:
        if active[i]:
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            if t[j] - t[i] > 0.3:  # drop single-sample blips
                spans.append((float(t[i]), float(t[j])))
            i = j + 1
        else:
            i += 1
    return spans


def _load_disclosure(summ: pd.DataFrame, scenarios: list) -> str:
    """
    Method-note clause disclosing the panels' *achieved* operating points: pinning the
    configured workload does not pin the achieved one, and rows flagged harness-limited hit
    the load generator's ceiling, not the gateway's (audit F12-2). Everything here is
    computed from the summary rows actually plotted; empty when the columns are absent.
    """
    if summ is None or summ.empty or "scenario" not in summ.columns:
        return ""
    rows = summ[summ["scenario"].isin(scenarios)].drop_duplicates("scenario")
    if rows.empty:
        return ""
    parts: list[str] = []
    if "throughput_gbps_mean" in rows.columns:
        tput = pd.to_numeric(rows["throughput_gbps_mean"], errors="coerce").dropna()
        tput = tput[tput > 0]
        if len(tput) > 1 and float(tput.max()) - float(tput.min()) >= 0.5:
            parts.append(f"achieved load differs per panel ({tput.min():.0f}–{tput.max():.0f} Gbps)")
    if "harness_limited" in rows.columns:
        n_hl = int(sum(1 for v in rows["harness_limited"] if isinstance(v, (bool, np.bool_)) and v))
        if n_hl:
            parts.append(
                f"{n_hl}/{len(rows)} panels are harness-limited (the generator, not the "
                "gateway, was the bottleneck)"
            )
    if not parts:
        return ""
    return (
        " · " + " · ".join(parts)
        + " — each panel's CPU is read at its own achievable operating point"
    )


def _pin_and_pick(
    summ: pd.DataFrame,
    have_sys: set,
    *,
    vary: str,
    pin: dict,
    order: list,
    max_scen: int,
) -> tuple[list, dict, bool]:
    """
    Select a *comparable* panel set where only ``vary`` differs across panels.

    ``pin`` maps each confounding column to the single value it is held at (e.g.
    ``{"protocol": "none", "n_gateways": 1}`` for the interface view, or
    ``{"transport": "tcp", "n_gateways": 1}`` for the protocol view). Connection count and
    message size are pinned too — ``connections==1`` (the only count every transport shares)
    plus the message size with maximal ``vary`` coverage, via :func:`derive.matched_cell`.
    With every other dimension fixed, the panels differ *only* in ``vary``, so the timeline
    differences are attributable to it alone.

    Only scenarios that actually carry a /proc timeseries (``have_sys``) are eligible, and
    only sustained-blast rows on the default data path: comparable panels must share the
    measurement *class*, not just the workload cell. ``order`` is the canonical value order
    for ``vary`` (``TRANSPORT_ORDER`` / a crypto ladder) giving a stable, narrative panel
    order. Returns ``(scenarios, chosen, ok)`` where ``chosen`` is the pinned cell (for a
    provenance stamp) and ``ok`` is True once at least two panels survive (a single panel
    is no comparison).
    """
    if summ is None or summ.empty or "scenario" not in summ.columns or vary not in summ.columns:
        return [], {}, False
    d = summ[summ["scenario"].isin(have_sys)].copy()
    # Paced/rate-capped runs, connection-rate sweeps and the SHM zero-copy slot-ring
    # microbenchmarks (non-default data path AND a 0s-cooldown schedule) carry /proc
    # timeseries too, and the max-throughput preference below would happily crown one of
    # them over the like-for-like default-path row (audit F12-1). They are a different
    # measurement class, so they are not eligible panel candidates.
    d = derive.throughput_scenarios_only(d)
    chosen: dict = {}
    for col, val in pin.items():
        if col in d.columns:
            d = d[d[col] == val]
            chosen[col] = val
    if d.empty:
        return [], chosen, False
    cell, mc = derive.matched_cell(d, [vary], fixed={"connections": 1})
    chosen.update(mc)
    if cell.empty or vary not in cell.columns:
        return [], chosen, False
    # One scenario per `vary` value; within the now fully-pinned cell prefer the
    # highest-throughput representative so a value with several rows resolves deterministically.
    if "throughput_gbps_mean" in cell.columns:
        cell = cell.sort_values("throughput_gbps_mean", ascending=False)
    reps = cell.drop_duplicates(vary).copy()
    rank = {v: i for i, v in enumerate(order)}
    # `vary` (transport/protocol) is an ordered Categorical, and `.map` on a Categorical
    # returns a *Categorical* ordered by the column's own categories — which would ignore our
    # `order` entirely. Materialize the rank as plain ints and sort on a plain-string secondary
    # key so the panel order actually follows `order` (e.g. TLS/kTLS grouped by version).
    labels = reps[vary].astype(str)
    reps["_o"] = [rank.get(v, len(order)) for v in labels]
    reps["_lbl"] = labels.values
    reps = reps.sort_values(["_o", "_lbl"])
    scenarios = reps["scenario"].tolist()[:max_scen]
    return scenarios, chosen, len(scenarios) >= 2


def make(bundle: RunBundle, saver: T.Saver) -> None:
    sysm = bundle.sysmetrics
    if sysm is None or sysm.empty or "elapsed_ms" not in sysm.columns:
        saver.record_skip(FIG_ID, NAME, "no system_metrics timeseries in this run")
        return

    # Comparable interface view: pin routing (plaintext 'none') on a single gateway, then vary
    # only the transport, so the panels line up the SAME workload across transports and every
    # CPU/RSS/ctxsw difference is the transport's own (not a confounded protocol/topology).
    have_sys = set(sysm["scenario"])
    summ = bundle.summary
    scenarios, chosen, comparable = _pin_and_pick(
        summ,
        have_sys,
        vary="transport",
        pin={"protocol": "none", "n_gateways": 1},
        order=TRANSPORT_ORDER,
        max_scen=_MAX_SCEN,
    )
    if not comparable:
        # Fallback: one highest-throughput scenario per transport — protocol/topology/size all
        # differ, so the panels are NOT comparable (labeled as such in the method note). Still
        # blast rows only: a paced run or a zero-copy microbenchmark under a transport's label
        # would misattribute a non-default variant's resource profile to the transport.
        scenarios = derive.representative_scenarios(derive.throughput_scenarios_only(summ))
        scenarios = [s for s in scenarios if s in have_sys][:_MAX_SCEN]
    if not scenarios:
        scenarios = list(pd.unique(sysm["scenario"]))[:_MAX_SCEN]
    if not scenarios:
        saver.record_skip(FIG_ID, NAME, "no scenarios with system metrics")
        return

    if comparable:
        method_note = (
            "panels ARE comparable — identical protocol (routing) · single gateway · matched "
            + T.fmt_cell(chosen)
            + "; only the transport varies, so every CPU/RSS/ctxsw difference is the transport "
            "mechanism's own · sustained-blast default-path rows only (paced/zero-copy/connrate "
            "excluded) · each panel sums CPU/RSS over its gateway PIDs · shaded bands = "
            "steady-state windows recovered from the CPU trace (no /proc phase marker) · "
            "per-rep ctxsw counter resets masked as gaps"
            + _load_disclosure(summ, scenarios)
            + "."
        )
    else:
        method_note = (
            "panels are NOT mutually comparable — no matched cross-transport routing slice in "
            "this run, so this is the highest-throughput blast scenario per transport and "
            "protocol/topology/size all differ · each panel sums CPU/RSS over its gateway PIDs · "
            "shaded bands = steady-state windows recovered from the CPU trace · per-rep "
            "ctxsw counter resets masked"
            + _load_disclosure(summ, scenarios)
            + "."
        )
    _render(bundle, saver, scenarios, fig_id=FIG_ID, name=NAME, title=TITLE,
            method_note=method_note, ctxsw_takeaway=comparable)


def _render(
    bundle: RunBundle,
    saver: T.Saver,
    scenarios: list,
    *,
    fig_id: str,
    name: str,
    title: str,
    method_note: str,
    ctxsw_takeaway: bool = False,
) -> None:
    sysm = bundle.sysmetrics
    summ = bundle.summary

    import matplotlib.pyplot as plt

    # Per-scenario identity (transport / protocol / gateway-count) for the panel titles,
    # plus the achieved load and bottleneck ownership (harness_limited) so each panel says
    # at what operating point its resource profile was measured. n_gateways etc. are
    # enriched only onto the summary, so look them up here rather than from the sysmetrics
    # frame (which carries only transport/protocol via the loader join).
    meta_by_scen: dict[str, pd.Series] = {}
    if not summ.empty and "scenario" in summ.columns:
        keep = [
            c
            for c in ("scenario", "transport", "protocol", "n_gateways", "throughput_gbps_mean", "harness_limited")
            if c in summ.columns
        ]
        for _, row in summ[keep].drop_duplicates("scenario").iterrows():
            meta_by_scen[str(row["scenario"])] = row

    nrow = len(scenarios)
    fig, axes = plt.subplots(nrow, 1, figsize=(8.6, 2.5 * nrow), squeeze=False, sharex=False)

    ctxsw_stats: list[tuple[str, float]] = []  # (panel label, median steady ctxsw/s)
    for r, scen in enumerate(scenarios):
        g = sysm[sysm["scenario"] == scen].sort_values("elapsed_ms")
        # Aggregate across PIDs at each timestamp (sum CPU/RSS over gateway processes).
        if "pid" in g.columns and g["pid"].nunique() > 1:
            agg = g.groupby("elapsed_ms", as_index=False).agg(
                cpu_pct=("cpu_pct", "sum") if "cpu_pct" in g.columns else ("elapsed_ms", "size"),
                rss_kib=("rss_kib", "sum") if "rss_kib" in g.columns else ("elapsed_ms", "size"),
            )
            ctx_src = g.groupby("elapsed_ms", as_index=False)[
                [c for c in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches") if c in g.columns]
            ].sum()
        else:
            agg = g
            ctx_src = g
        t = agg["elapsed_ms"].values / 1000.0

        ax = axes[r][0]

        # Panel identity: transport · protocol · gateway-count · #PIDs summed. Prefer the
        # summary (only place n_gateways lives); fall back to the merged sysmetrics columns.
        info = meta_by_scen.get(scen)

        def _fact(col: str):
            if info is not None and col in info.index and pd.notna(info[col]):
                return info[col]
            if col in g.columns and g[col].notna().any():
                return g[col].dropna().iloc[0]
            return None

        tr = _fact("transport")
        pr = _fact("protocol")
        ngw = _fact("n_gateways")
        npids = int(g["pid"].nunique()) if "pid" in g.columns else 1
        tlabel = T.transport_label(str(tr)) if tr is not None else ""
        plabel = T.protocol_label(str(pr)) if pr is not None else ""
        try:
            ngw_i = int(ngw) if ngw is not None else None
        except (TypeError, ValueError):
            ngw_i = None
        gw_txt = f"{ngw_i} gateway" + ("s" if ngw_i != 1 else "") if ngw_i is not None else "gateway"
        pid_txt = f"Σ{npids} PID" + ("s" if npids != 1 else "")
        ident = " · ".join(x for x in (tlabel, plabel) if x) or scen

        # Achieved load + bottleneck ownership per panel: the pinned cell fixes the
        # *configured* workload, not the throughput the harness actually reached, so
        # cross-panel CPU readings must be interpretable at each panel's own operating
        # point (audit F12-2).
        title_extras = [gw_txt, pid_txt]
        try:
            tput = float(_fact("throughput_gbps_mean"))
        except (TypeError, ValueError):
            tput = float("nan")
        if np.isfinite(tput) and tput > 0:
            title_extras.append(f"{tput:.1f} Gbps achieved")
        hl = _fact("harness_limited")
        if isinstance(hl, (bool, np.bool_)) and hl:
            title_extras.append("harness-limited")

        # Shade the recovered steady-state measurement window(s) behind the traces (there is
        # no phase marker in /proc, so these are inferred from the CPU trace — see helper).
        spans = _steady_spans(t, agg["cpu_pct"].values) if "cpu_pct" in agg.columns else []
        for s0, s1 in spans:
            ax.axvspan(s0, s1, color=T.GREYS["faint"], alpha=0.28, lw=0, zorder=0)

        if "cpu_pct" in agg.columns:
            ax.plot(t, agg["cpu_pct"], color=T.METRIC["cpu"], lw=1.6)
            ax.set_ylabel("CPU % (100=1 core)", fontsize=T.FS["small"])
        T.panel_title(ax, f"{ident}   ({' · '.join(title_extras)})")
        ax.grid(axis="y")

        if "rss_kib" in agg.columns:
            ax2 = ax.twinx()
            ax2.plot(t, agg["rss_kib"] / 1024.0, color=T.METRIC["rss"], lw=1.4, ls="--")
            ax2.set_ylabel("RSS (MiB)", fontsize=T.FS["small"])
            ax2.grid(False)

        # Context-switch rate as a faint filled area on a third (offset) scale — but only
        # when there is real signal (these /proc counters are often near-constant for the
        # gateway, which would otherwise add a flat, distracting zero axis).
        ct, rate = _ctxsw_rate(ctx_src)
        # Median steady-state ctxsw/s per panel for the (optional) computed takeaway —
        # restricted to the recovered steady windows so warmup/cooldown don't dilute it.
        if ctxsw_takeaway and len(ct) and np.isfinite(rate).any():
            in_steady = np.zeros_like(ct, dtype=bool)
            for s0, s1 in spans:
                in_steady |= (ct >= s0) & (ct <= s1)
            pool = rate[in_steady & np.isfinite(rate)]
            if not pool.size:
                pool = rate[np.isfinite(rate)]
            if pool.size:
                ctxsw_stats.append((tlabel or str(scen), float(np.median(pool))))
        if len(ct) and np.isfinite(rate).any() and np.nanmax(rate) > 5.0:
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.18))
            ax3.fill_between(ct, rate, color=T.METRIC["ctxsw"], alpha=0.12)
            ax3.plot(ct, rate, color=T.METRIC["ctxsw"], lw=0.8)
            ax3.set_ylabel("ctxsw/s", fontsize=T.FS["small"])
            ax3.tick_params(axis="y", labelsize=T.FS["annot"])
            ax3.grid(False)

        if r == nrow - 1:
            ax.set_xlabel("elapsed time (s)")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color=T.METRIC["cpu"], lw=1.6, label="CPU % (left axis)"),
        Line2D([], [], color=T.METRIC["rss"], lw=1.4, ls="--", label="RSS MiB (right axis)"),
        Patch(facecolor=T.METRIC["ctxsw"], alpha=0.25, edgecolor=T.METRIC["ctxsw"],
              label="ctxsw/s (outer right axis)"),
        Patch(facecolor=T.GREYS["faint"], alpha=0.28, edgecolor="none",
              label="steady-state window"),
    ]
    T.legend_right(fig, handles)
    T.set_headline(fig, f"{title}  ·  {bundle.label}")
    if ctxsw_takeaway and ctxsw_stats:
        def _fmt_rate(v: float) -> str:
            return f"~{v / 1000.0:.0f}k/s" if v >= 1000.0 else f"~{v:.0f}/s"

        pairs = " · ".join(f"{lbl} {_fmt_rate(v)}" for lbl, v in ctxsw_stats)
        T.add_takeaway(
            fig,
            "Steady-state context-switch rate tracks IPC locality, read at each panel's own "
            f"achieved load: {pairs}.",
        )
    T.add_method_note(fig, method_note)
    T.add_provenance(fig, bundle.caption() + f"  ·  {bundle.label}  ·  CPU/RSS summed across gateway PIDs · sampling spans warmup→measure→cooldown")
    saver.save(fig, name, fig_id=fig_id, title=title)
