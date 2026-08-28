"""
F9 — The resource cost of security, in one place.

Consolidates what used to be three separate figures (efficiency, perf-counter breakdown,
memory footprint) into a single small-multiples grid keyed to the security ladder, so the
whole question — *what does each security tier cost across every resource dimension?* — is
answered at one glance: throughput, CPU efficiency (Gbps/core), CPU cycles per byte,
cache-miss rate, context switches per thousand messages, and peak memory. A final scatter
ties peak memory footprint (RSS) to delivered throughput. The hardware-counter panels
(cycles/byte, cache-miss, ctx-switches) need a ``--metrics-backend perf`` run; on a procfs
run they render as labelled placeholders (``theme.perf_placeholder``) rather than vanishing,
so the panel honestly reads "no counters here" instead of "measured & zero".

Populated counters are additionally *scope-checked*: an unprivileged perf run
(perf_event_paranoid >= 2) silently demotes every event to user scope (``:u``), which turns
the cycles/byte ladder into a user-vs-kernel *attribution* artifact — kernel-offload rungs
read near-zero while userspace crypto reads full cost — and makes ctx-switches definitionally
zero. Those panels are then withheld with the reason rather than rendered as measurements.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label

FIG_ID = "F9"
NAME = "f09_resource_cost_of_security"
TITLE = "Resource cost of security (throughput · CPU · memory)"

_KIB_PER_MIB = 1024.0

# Counters whose cross-rung comparison is meaningless under user-only scope: cycles misses
# the kernel-side share (the bulk of kTLS/routing work), ctx-switches happen *at* the kernel
# transition so a ':u' event counts none. The cache-miss *rate* (a ratio of two same-scope
# counts) survives as a user-mode locality metric and is only relabelled.
_SCOPE_BLIND = {"cycles_per_byte", "ctxsw_per_kmsg"}


# Lives in derive.perf_user_scope_only (F30 shares it); the module-local name is
# kept so this figure's call sites and tests read unchanged.
_perf_user_scope_only = derive.perf_user_scope_only


def _scope_placeholder(ax, metric_label: str) -> None:
    """
    Placeholder for a counter panel whose data exists but was collected user-scope only.
    Cannot delegate to ``theme.perf_placeholder``: that helper hardcodes the "procfs run,
    no hardware counters" reason (false here — the counters exist but are untrustworthy),
    and its ``metric_label`` doubles as the panel title, so the differing reason cannot be
    smuggled through it. Minimal replicate of its look, with the true withholding reason —
    rendering the numbers would show privilege-domain attribution, not resource cost.
    """
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(T.GREYS["faint"])
    T.panel_title(ax, metric_label)
    ax.text(
        0.5, 0.5,
        f"{metric_label}\ncounters were user-scope only (unprivileged perf) —\n"
        "kernel-side work invisible, not comparable across the ladder\n"
        "needs a kernel-scope (CAP_PERFMON / paranoid≤1) perf run",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=T.FS["small"], color=T.GREYS["muted"], style="italic",
    )


def make(bundle: RunBundle, saver: T.Saver) -> None:
    df = bundle.summary
    if "protocol" not in df.columns:
        saver.record_skip(FIG_ID, NAME, "needs protocol column")
        return

    d = derive.add_normalized_costs(df)
    d = derive.attach_bytes_from_runs(d, bundle.runs)
    d = derive.throughput_scenarios_only(d)
    # Pin ONE transport so the per-protocol "cost of security" bars are like-for-like: the
    # kernel-offload (kTLS) advantage is a TCP-path property, and pooling protocols over
    # different transport sets (TLS's UDP-over-TLS rows vs kTLS's SHM/UDS rows) manufactured a
    # spurious efficiency gap. TCP carries the full protocol ladder; matrix = sustained blast.
    if "family" in d.columns and (d["family"].astype(str) == "matrix").any():
        d = d[d["family"].astype(str) == "matrix"]
    if "transport" in d.columns and (d["transport"].astype(str) == "tcp").any():
        d = d[d["transport"].astype(str) == "tcp"]
    d = derive.size_match_for_protocol_compare(d, min_protocols=3)
    if d.empty:
        saver.record_skip(FIG_ID, NAME, "no matched throughput scenarios to aggregate")
        return

    # Pin single-connection rows before the per-protocol aggregation so a 1c row is never
    # averaged with a 1024c aggregate (1c is the shared per-message anchor).
    if "connections" in d.columns:
        conn = d["connections"]
        d = d[conn.isin([1]) | conn.isna()]
        if d.empty:
            saver.record_skip(FIG_ID, NAME, "no single-connection rows to aggregate")
            return

    # Per-protocol aggregation (mean across transports/sizes that carry each metric).
    scalar_panels = [
        ("throughput_gbps_mean", "throughput (Gbps)", False),
        ("gbps_per_core", "Gbps / core", False),
        ("cycles_per_byte", "CPU cycles / byte", True),
        ("cache_miss_rate", "cache-miss rate (miss/ref)", True),
        ("ctxsw_per_kmsg", "ctx-switches / 1k msg", True),
    ]
    present = [(c, t, lb) for c, t, lb in scalar_panels if c in d.columns and d[c].notna().any()]
    # The hardware-counter metrics are all-NaN on a procfs run. Rather than silently dropping
    # their panels (which reads as "measured & zero"), keep them as labelled placeholders so the
    # grid honestly shows the panel exists but needs a --metrics-backend perf run.
    _PERF_METRICS = {"cycles_per_byte", "cache_miss_rate", "ctxsw_per_kmsg"}
    perf_missing = [(c, t) for c, t, _ in scalar_panels
                    if c in _PERF_METRICS and (c not in d.columns or not d[c].notna().any())]
    # Populated counters from an unprivileged perf run are user-scope only: the scope-blind
    # panels get withheld with the reason instead of rendered as measurements.
    user_scope = _perf_user_scope_only(d)
    scope_withheld = [(c, t) for c, t, _ in present if user_scope and c in _SCOPE_BLIND]
    if scope_withheld:
        withheld_cols = {c for c, _ in scope_withheld}
        present = [(c, t, lb) for c, t, lb in present if c not in withheld_cols]
    grp = d.groupby("protocol", observed=True)
    agg = grp.agg(**{c: (c, "mean") for c, _, _ in present}).reset_index()

    # memory (max peak per protocol).
    mem_cols = [c for c in ("rss_peak_kib", "pss_peak_kib") if c in d.columns and d[c].notna().any()]
    mem = grp[mem_cols].max().reset_index() if mem_cols else pd.DataFrame()

    # order by the security ladder.
    order = [p for p in T.PROTOCOL_ORDER if p in set(agg["protocol"].astype(str))]
    order += [p for p in agg["protocol"].astype(str).unique() if p not in order]
    pos = {p: i for i, p in enumerate(order)}
    agg["__o"] = agg["protocol"].astype(str).map(pos)
    agg = agg.sort_values("__o").reset_index(drop=True)
    if not mem.empty:
        mem["__o"] = mem["protocol"].astype(str).map(pos)
        mem = mem.sort_values("__o").reset_index(drop=True)

    have_mem = not mem.empty

    # ---- Thesis variant: memory ladder + CPU-efficiency panel, nothing else --------------
    # The hardware-counter placeholders and the RSS-vs-throughput scatter are exploration
    # aids; the print figure carries the two loopback-robust facts (RSS separates the
    # security tiers, Gbps/core does not) and omits placeholder panels entirely.
    if T.thesis_variant():
        if not have_mem:
            saver.record_skip(FIG_ID, NAME, "thesis variant needs memory columns")
            return
        _make_thesis(bundle, saver, agg=agg, mem=mem, d=d)
        return

    # Scatter ties peak memory footprint (RSS, present on every run) to delivered throughput —
    # the "cost vs benefit" panel, keyed on RSS rather than a perf-only proxy so it stays
    # populated on a procfs run.
    have_scatter = have_mem and "rss_peak_kib" in mem.columns \
        and "throughput_gbps_mean" in agg.columns and agg["throughput_gbps_mean"].notna().any()
    # Keep the ladder order of scalar panels stable whichever class each falls into (bar,
    # scope-withheld placeholder, missing-counter placeholder) so the grid shape reads the
    # same across procfs and perf runs.
    bar_meta = {c: (t, lb) for c, t, lb in present}
    withheld_meta = dict(scope_withheld)
    missing_meta = dict(perf_missing)
    panel_render = []
    for col, _title, _lb in scalar_panels:
        if col in bar_meta:
            panel_render.append(("bar", col))
        elif col in withheld_meta:
            panel_render.append(("withheld", col))
        elif col in missing_meta:
            panel_render.append(("missing", col))
    n_panels = len(panel_render) + (1 if have_mem else 0) + (1 if have_scatter else 0)
    if n_panels == 0:
        saver.record_skip(FIG_ID, NAME, "no resource metrics available to plot")
        return

    ncol = min(3, n_panels)
    nrow = math.ceil(n_panels / ncol)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.7 * nrow), squeeze=False)
    flat = [axes[r][c] for r in range(nrow) for c in range(ncol)]
    x = np.arange(len(agg))
    colors = [T.protocol_color(str(p)) for p in agg["protocol"]]
    labels = [protocol_label(str(p)) for p in agg["protocol"]]
    pi = 0

    def _bar_axis(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=T.FS["annot"])
        ax.grid(axis="y")

    # scalar panels — bars, or labelled placeholders where the counters are absent (procfs
    # run) / untrustworthy (user-scope-demoted perf run)
    for kind, col in panel_render:
        ax = flat[pi]; pi += 1
        if kind == "withheld":
            _scope_placeholder(ax, withheld_meta[col])
            continue
        if kind == "missing":
            T.perf_placeholder(ax, missing_meta[col])
            continue
        title, lower_better = bar_meta[col]
        if user_scope and col == "cache_miss_rate":
            # A ratio of two same-scope counts stays meaningful, but only for user-mode work.
            title = "cache-miss rate (miss/ref, user-scope)"
        vals = agg[col]
        ax.bar(x, vals, color=colors, edgecolor=T.GREYS["edge"], linewidth=0.5)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}", (i, v), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=T.FS["annot"])
        T.panel_title(ax, title + ("  ↓ better" if lower_better else "  ↑ better"))
        _bar_axis(ax)

    # memory panel (grouped RSS/PSS)
    if have_mem:
        ax = flat[pi]; pi += 1
        series = [(c, lbl, clr) for c, lbl, clr in (
            ("rss_peak_kib", "RSS", T.METRIC["rss"]),
            ("pss_peak_kib", "PSS", T.METRIC["pss"])) if c in mem.columns]
        width = 0.8 / max(len(series), 1)
        xm = np.arange(len(mem))
        for k, (c, lbl, clr) in enumerate(series):
            ax.bar(xm + (k - (len(series) - 1) / 2) * width, mem[c] / _KIB_PER_MIB, width,
                   label=lbl, color=clr, edgecolor=T.GREYS["edge"], linewidth=0.4)
        ax.set_xticks(xm)
        ax.set_xticklabels([protocol_label(str(p)) for p in mem["protocol"]], rotation=40,
                           ha="right", fontsize=T.FS["annot"])
        T.panel_title(ax, "peak memory (MiB)  ↓ better")
        T.legend_inline(ax)
        ax.grid(axis="y")

    # scatter: delivered throughput vs peak memory footprint (RSS). Keyed on RSS (present on
    # every run) rather than a perf-only proxy, so the trade-off panel stays populated here.
    if have_scatter:
        ax = flat[pi]; pi += 1
        sc = agg.merge(mem[["protocol", "rss_peak_kib"]], on="protocol", how="inner")
        for _, r in sc.iterrows():
            mx = r["rss_peak_kib"] / _KIB_PER_MIB
            ty = r["throughput_gbps_mean"]
            if not (np.isfinite(mx) and np.isfinite(ty)):
                continue
            ax.scatter(mx, ty, s=90, color=T.protocol_color(str(r["protocol"])),
                       edgecolor=T.GREYS["edge"], linewidth=0.6, zorder=3)
            ax.annotate(protocol_label(str(r["protocol"])), (mx, ty),
                        xytext=(5, 2), textcoords="offset points", fontsize=T.FS["annot"])
        ax.set_xlabel("peak RSS (MiB)")
        ax.set_ylabel("throughput (Gbps)")
        T.panel_title(ax, "throughput vs memory footprint")
        ax.grid(True)

    # hide any unused axes
    for ax in flat[pi:]:
        ax.axis("off")

    # data-driven takeaway: two coupled facts on this loopback path — CPU efficiency is *flat*
    # across the crypto ladder (kTLS ≈ userspace TLS ≈ ~8 Gbps/core, because the path is
    # harness/bandwidth-bound, not cipher-bound — bottleneck attribution), while peak memory
    # footprint *does* climb the ladder. Both clauses are computed from the aggregated frame.
    take = "Each panel reads along the security ladder — where a tier spends its CPU and memory."

    def _eff(proto):
        if "gbps_per_core" not in agg.columns:
            return np.nan
        row = agg[agg["protocol"].astype(str) == proto]
        return float(row["gbps_per_core"].iloc[0]) if len(row) else np.nan

    def _rss_mib(proto):
        if mem.empty or "rss_peak_kib" not in mem.columns:
            return np.nan
        row = mem[mem["protocol"].astype(str) == proto]
        return float(row["rss_peak_kib"].iloc[0]) / _KIB_PER_MIB if len(row) else np.nan

    kt, tl = _eff("ktls/1.3"), _eff("tls/1.3")
    r_none, r_tls, r_mtls = _rss_mib("none"), _rss_mib("tls/1.3"), _rss_mib("tls/1.3+mtls")
    hl_frac = float(pd.to_numeric(d.get("harness_limited"), errors="coerce").fillna(0).mean()) \
        if "harness_limited" in d.columns else 0.0

    # CPU-efficiency clause, phrased to the actual gap with bottleneck attribution.
    cpu_clause = ""
    if np.isfinite(kt) and np.isfinite(tl) and tl > 0:
        gap = (kt / tl - 1) * 100
        if abs(gap) < 3:
            why = "harness/bandwidth-bound" if hl_frac > 0.2 else "not cipher-bound"
            cpu_clause = (f"CPU efficiency is flat up the ladder — kTLS 1.3 ≈ userspace TLS 1.3 "
                          f"(~{kt:.0f} Gbps/core) because this TCP-loopback path is {why}; the "
                          f"kernel-offload payoff needs a bandwidth-bound path (see F4/F5)")
        else:
            cpu_clause = (f"kTLS 1.3 delivers {kt:.1f} Gbps/core vs userspace TLS 1.3's {tl:.1f} "
                          f"({gap:+.0f}%) — the kernel offload's CPU-efficiency payoff")

    # Memory clause: footprint *does* separate the tiers where efficiency does not.
    mem_clause = ""
    if np.isfinite(r_none) and np.isfinite(r_tls) and np.isfinite(r_mtls):
        mem_clause = (f"memory *does* separate them (peak RSS routing {r_none:.0f} < TLS "
                      f"{r_tls:.1f} < mTLS {r_mtls:.0f} MiB)")

    if cpu_clause and mem_clause:
        take = f"{cpu_clause}; {mem_clause}."
    elif cpu_clause:
        take = cpu_clause + "."
    elif mem_clause:
        take = "Up the security ladder, " + mem_clause + "."

    # Method note & provenance describe the *actual* state of the hardware-counter panels
    # rather than a hardcoded "placeholders" claim that a populated render would
    # contradict: placeholder wording only when panels really fell back, scope caveat only
    # when the counters were user-scope-demoted, a plain perf-stat attribution otherwise.
    _SHORT = {"cycles_per_byte": "cycles/byte", "cache_miss_rate": "cache-miss",
              "ctxsw_per_kmsg": "ctx-switches"}
    method = ("matrix family · TCP · single-connection (1c), size-matched across the "
              "compared protocols")
    prov = bundle.caption() + "  ·  matrix TCP 1c sustained-blast runs"
    if perf_missing:
        names = "/".join(_SHORT[c] for c, _ in perf_missing)
        method += (f" · hardware-counter panels ({names}) shown as placeholders — they need "
                   f"a --metrics-backend perf run")
        prov += f"; {names} need a perf run"
    if user_scope:
        method += (" · perf counters were user-scope only (unprivileged perf): cycles/byte & "
                   "ctx-switches withheld, cache-miss rate covers user-mode references only")
        prov += "; perf counters user-scope only"
    elif any(c in _PERF_METRICS for c, _, _ in present):
        method += " · hardware counters measured by perf stat"

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.02)
    T.add_takeaway(fig, take)
    T.add_method_note(fig, method)
    T.add_provenance(fig, prov)
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)


def _make_thesis(bundle: RunBundle, saver: T.Saver, *, agg: pd.DataFrame,
                 mem: pd.DataFrame, d: pd.DataFrame) -> None:
    """Print variant: peak-memory ladder + Gbps/core, sized for a 15 cm text column."""
    import matplotlib.pyplot as plt

    have_eff = "gbps_per_core" in agg.columns and agg["gbps_per_core"].notna().any()
    ncol = 2 if have_eff else 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.9 * ncol, 3.4), squeeze=False)
    ax_mem = axes[0][0]

    series = [(c, lbl, clr) for c, lbl, clr in (
        ("rss_peak_kib", "RSS", T.METRIC["rss"]),
        ("pss_peak_kib", "PSS", T.METRIC["pss"]))
        if c in mem.columns and mem[c].notna().any()]
    width = 0.8 / max(len(series), 1)
    xm = np.arange(len(mem))
    for k, (c, lbl, clr) in enumerate(series):
        vals = mem[c] / _KIB_PER_MIB
        ax_mem.bar(xm + (k - (len(series) - 1) / 2) * width, vals, width,
                   label=lbl, color=clr, edgecolor=T.GREYS["edge"], linewidth=0.4)
        if c == "rss_peak_kib":
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    ax_mem.annotate(f"{v:.1f}", (xm[i] + (k - (len(series) - 1) / 2) * width, v),
                                    xytext=(0, 2), textcoords="offset points",
                                    ha="center", fontsize=T.FS["annot"])
    ax_mem.set_xticks(xm)
    ax_mem.set_xticklabels([protocol_label(str(p)) for p in mem["protocol"]],
                           rotation=40, ha="right", fontsize=T.FS["small"])
    ax_mem.set_ylabel("peak memory (MiB) — lower is better")
    if len(series) > 1:
        T.legend_inline(ax_mem)
    ax_mem.grid(axis="y")

    if have_eff:
        ax_eff = axes[0][1]
        xe = np.arange(len(agg))
        ax_eff.bar(xe, agg["gbps_per_core"],
                   color=[T.protocol_color(str(p)) for p in agg["protocol"]],
                   edgecolor=T.GREYS["edge"], linewidth=0.4)
        for i, v in enumerate(agg["gbps_per_core"]):
            if np.isfinite(v):
                ax_eff.annotate(f"{v:.1f}", (i, v), xytext=(0, 2),
                                textcoords="offset points", ha="center", fontsize=T.FS["annot"])
        ax_eff.set_xticks(xe)
        ax_eff.set_xticklabels([protocol_label(str(p)) for p in agg["protocol"]],
                               rotation=40, ha="right", fontsize=T.FS["small"])
        ax_eff.set_ylabel("Gbps per gateway core")
        ax_eff.grid(axis="y")

    def _rss_mib(proto: str) -> float:
        if "rss_peak_kib" not in mem.columns:
            return float("nan")
        row = mem[mem["protocol"].astype(str) == proto]
        return float(row["rss_peak_kib"].iloc[0]) / _KIB_PER_MIB if len(row) else float("nan")

    def _eff_of(proto: str) -> float:
        if "gbps_per_core" not in agg.columns:
            return float("nan")
        row = agg[agg["protocol"].astype(str) == proto]
        return float(row["gbps_per_core"].iloc[0]) if len(row) else float("nan")

    # Memory ladder over the four canonical rungs, computed each render; fall back to a
    # min→max span over whatever protocols are present if a rung is missing.
    rungs = [("none", "routing"), ("tls/1.3", "TLS 1.3"),
             ("tls/1.3+mtls", "mTLS 1.3"), ("tls/1.2+integrity", "TLS 1.2 (integrity)")]
    vals = [(lbl, _rss_mib(p)) for p, lbl in rungs]
    vals = [(lbl, v) for lbl, v in vals if np.isfinite(v)]
    if len(vals) >= 3:
        mem_clause = "peak RSS climbs the security ladder: " + " < ".join(
            f"{lbl} {v:.1f} MiB" if lbl == "routing" else f"{lbl} {v:.1f}"
            for lbl, v in vals)
    else:
        rall = (mem["rss_peak_kib"] / _KIB_PER_MIB) if "rss_peak_kib" in mem.columns else pd.Series(dtype=float)
        mem_clause = (f"peak RSS spans {rall.min():.1f}–{rall.max():.1f} MiB across the "
                      f"{len(mem)} protection modes") if rall.notna().any() else ""

    kt, tl = _eff_of("ktls/1.3"), _eff_of("tls/1.3")
    cpu_clause = ""
    if np.isfinite(kt) and np.isfinite(tl) and tl > 0 and abs(kt / tl - 1) < 0.1:
        cpu_clause = (f" while Gbps/core stays flat (kTLS 1.3 {kt:.1f} ≈ userspace TLS 1.3 "
                      f"{tl:.1f}) because this TCP-loopback path is harness/bandwidth-bound, "
                      f"not cipher-bound")
    elif np.isfinite(kt) and np.isfinite(tl) and tl > 0:
        cpu_clause = f" while kTLS 1.3 delivers {kt:.1f} vs TLS 1.3's {tl:.1f} Gbps/core"

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.02)
    T.add_takeaway(fig, (mem_clause + (";" + cpu_clause if cpu_clause else "") + ".")
                   if mem_clause else "Peak memory separates the protection modes.")
    T.add_method_note(
        fig,
        "matrix family · TCP · single-connection (1c), size-matched across the compared "
        "protocols · peak memory = max RSS/PSS over the scenario's gateway processes · "
        "hardware counters (cycles/byte, cache misses) were not measured on this run "
        "and their panels are omitted here",
    )
    T.add_provenance(fig, bundle.caption() + "  ·  matrix TCP 1c sustained-blast runs")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
