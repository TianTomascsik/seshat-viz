"""
F15 — Concurrency scaling: does the gateway keep up as connections multiply?

The nightly matrix sweeps 1 / 4 / 16 (/64) connections, but no figure put connections on an
axis. Three stacked panels per transport. Top: scaling *efficiency* (achieved speedup ÷ the
ideal-linear diagonal, %) — 100% hugs the diagonal (perfect scaling), a fall-off is hitting a
shared bottleneck; the achieved efficiency at peak concurrency is annotated. Middle: ABSOLUTE
throughput (Gbps) — the efficiency axis normalizes each series to its own 1-connection point
and so hides the LEVEL gap between transports (UDS plateaus ~38 Gbps, SHM ~28–37 then collapses
once encrypted, routed TCP climbs into the 100s); the absolute panel restores it. Bottom: p99
tail latency under a saturating blast (queueing-dominated), where added concurrency is paid for.

The print variant keeps the efficiency row only and splits it into one row per protection
group — plaintext routing on top, the user-space TLS profiles in the middle, the kernel TLS
(kTLS) profiles at the bottom — with every row of a column sharing the union connection range,
so a routing sweep truncated at 16c is drawn against its encrypted siblings' 64c reach rather
than stretched to look converged; a group absent from the whole run drops its row.

Why curves flatten (or, once encrypted, decline) is the point of this figure, so each point is
drawn with its SESHAT bottleneck class: a **hollow** marker is load-generator / host bound
(`harness-io` / `host-saturated`, or flagged `harness_limited`), a **filled** marker is
gateway-path bound (`scg` / `scg-cpu`, the latter a single per-connection relay thread pegging
one core). The causal prose is computed from the run's own telemetry at render time, never
asserted: host_busy_frac_p95 over the plotted rows decides whether "not core count" is
defensible — typically the host is near-idle at 1–4c (there the serial single-thread-per-
connection data plane on a no-NIC loopback host is the ceiling) but climbs toward saturation
at higher concurrency, where host CPU becomes a co-limit that must be named. The takeaway
compares series at the deepest connection count they ALL reach: a max over per-series
endpoints would crown whichever sweep was truncated shallowest (e.g. a series whose 64c
scenarios were all skipped). Series that end early because their higher-concurrency scenarios
were skipped, or that ride the single-gateway 'direct' chain because no 2-gateway data exists,
are disclosed with the reason spelled out in the method note. Transports run at a
single connection only appear as labelled placeholder columns; the high-concurrency points the
suite could not run (the coverage wall) are counted in the caption.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F15"
NAME = "f15_concurrency_scaling"
TITLE = "Concurrency scaling: throughput speedup & tail latency vs connections"

# SESHAT bottleneck classes that are NOT the gateway's fan-out limit: the harness load
# generator (`harness-io`) or the whole box (`host-saturated`). On a single loopback host
# these are testbed ceilings — the honest reading of a flat curve.
_LOADGEN_BOTTLENECKS = {"harness-io", "host-saturated"}

# Per-transport reason a transport appears as a single-connection placeholder column. Only
# transports actually resolved into `single_only` are explained — a fixed sentence naming
# absent transports goes stale the moment one of them starts sweeping connections (a stale
# "TPROXY pinned to 1c" once contradicted the TPROXY panel in this very figure; F15-3).
_SINGLE_ONLY_REASONS = {
    "udp": "the 'dtls_multi_connection' limitation (one shared backend datagram flow cannot "
           "report parallel connections)",
}

# Matrix scenario names end `_<size>_<chain>_<n>c`; stripping that suffix yields the series
# identity a skipped row shares with its measured siblings, so a series' endpoint can be
# checked against the skip register (was it truncated, or did the sweep really end there?).
_SERIES_SUFFIX_RE = re.compile(r"_\d+(?:[KM])?B_(?:scg|direct)_\d+c$")


def _series_stem(name: str) -> str:
    """Matrix scenario name minus its `_<size>_<chain>_<n>c` suffix."""
    return _SERIES_SUFFIX_RE.sub("", str(name))


def _is_loadgen(bottleneck: object, harness_limited: object) -> bool:
    """True when a point's ceiling is the load generator / host, not the gateway's fan-out."""
    # `harness_limited` arrives as a Python or numpy bool (derive aggregates with .any());
    # an identity check against True would silently ignore np.True_.
    if isinstance(harness_limited, (bool, np.bool_)) and bool(harness_limited):
        return True
    # derive.scaling_table aggregates an all-NaN bottleneck column to pd.NA, whose truth value
    # raises ("boolean value of NA is ambiguous"); an unrecorded class is not load-generator.
    if bottleneck is None or (np.ndim(bottleneck) == 0 and pd.isna(bottleneck)):
        return False
    return str(bottleneck) in _LOADGEN_BOTTLENECKS


def _preferred_chain(g: pd.DataFrame) -> pd.DataFrame:
    """
    One series per (transport, protocol): prefer the gateway (scg) path, else whatever exists.

    Grouping per (transport, protocol) — not per protocol globally — is essential: the old
    per-protocol pick chose 'scg' for a protocol because TCP had it, then dropped every SHM/UDS
    series (which are direct-only), collapsing the whole figure to a single TCP panel and
    making the "only TCP sweeps connections" caption false.
    """
    if "chain" not in g.columns:
        return g
    out = []
    for _key, pg in g.groupby(["transport", "protocol"], observed=True):
        chains = set(pg["chain"].astype(str))
        pick = "scg" if "scg" in chains else sorted(chains)[0]
        out.append(pg[pg["chain"].astype(str) == pick])
    return pd.concat(out, ignore_index=True) if out else g


def _series_rows(summary: pd.DataFrame, tbl: pd.DataFrame) -> pd.DataFrame:
    """
    The raw summary rows behind the plotted series — same family scope, series keys and
    per-series payload that `derive.scaling_table` + `_preferred_chain` selected. The scaling
    aggregation drops host telemetry and scenario names; this recovers them so the causal
    prose (host_busy_frac_p95, cpu_hot_thread_pct_p95) and the skip-truncation check are
    computed from exactly the rows on the canvas, not from the whole run.
    """
    need = {"transport", "protocol", "connections", "throughput_gbps_mean",
            "message_bytes", "scenario"}
    if tbl.empty or not need.issubset(summary.columns):
        return pd.DataFrame()
    d = derive.throughput_scenarios_only(summary)
    if "family" in d.columns and (d["family"].astype(str) == "matrix").any():
        d = d[d["family"].astype(str) == "matrix"]
    d = d[d["connections"].notna() & d["throughput_gbps_mean"].notna()].copy()
    if "chain" not in d.columns:
        d["chain"] = "scg"

    def key(df: pd.DataFrame) -> pd.Series:
        mb = pd.to_numeric(df["message_bytes"], errors="coerce").astype("Float64").astype(str)
        return (df["transport"].astype(str) + "|" + df["protocol"].astype(str) + "|"
                + df["chain"].astype(str) + "|" + mb)

    return d[key(d).isin(set(key(tbl)))]


def _busy_profile(plotted: pd.DataFrame) -> "pd.Series | None":
    """Median host_busy_frac_p95 per connection count over the plotted rows, or None."""
    if plotted.empty or "host_busy_frac_p95" not in plotted.columns:
        return None
    hb = plotted[["connections", "host_busy_frac_p95"]].apply(pd.to_numeric, errors="coerce").dropna()
    if hb.empty:
        return None
    return hb.groupby("connections")["host_busy_frac_p95"].median().sort_index()


# Print-variant row groups, top to bottom. Membership is decided by protocol prefix so a
# profile a later run adds lands in the right row without a table edit.
_ROW_GROUPS = (
    ("routing", "plaintext routing"),
    ("user", "user-space TLS"),
    ("kernel", "kernel TLS (kTLS)"),
)


def _row_group(protocol: object) -> str:
    """Print-variant row a protection profile belongs to: routing / user-space / kernel."""
    p = str(protocol)
    if p == "none":
        return "routing"
    return "kernel" if p.startswith("ktls/") else "user"


def _efficiency_panel(ax: "plt.Axes", sub: pd.DataFrame, xs: list, *,
                      has_bottleneck: bool, declutter: bool,
                      busy_rows: "pd.DataFrame | None" = None,
                      busy_label: bool = False) -> "tuple[set[str], bool]":
    """
    One scaling-efficiency panel: the ideal-linear guide (labelled at xs[-1]), one line per
    protocol with hollow/filled boundness faces, endpoint % labels (thinned to >=4 % apart when
    `declutter`), log2 x with `xs` as ticks, a 0–112 % y axis and the grid. When `busy_rows` is
    given, the median host_busy_frac_p95 per connection count over those rows is drawn on a
    right axis (label + ticks only when `busy_label`). Returns (protocols drawn, busy drawn).
    """
    import matplotlib.ticker as mticker

    protos: set[str] = set()
    ann_ys: list[float] = []  # endpoint labels already placed (declutter)
    # Plot scaling EFFICIENCY (achieved speedup ÷ ideal-linear speedup, %) rather than raw
    # speedup: a shared ideal diagonal to 1024× would crush every real curve (which peak
    # below ~10×) onto the x-axis. 100% = perfect linear scaling; the fall-off to the
    # bottleneck is read directly on a bounded 0–100% axis.
    ax.axhline(100.0, ls=":", color=T.GREYS["faint"], lw=0.8, zorder=1)
    ax.annotate("ideal (linear)", (xs[-1], 100.0), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=T.FS["annot"],
                color=T.GREYS["muted"], style="italic")
    for proto, pg in sub.groupby("protocol", observed=True):
        pg = pg.sort_values("connections")
        protos.add(str(proto))
        c = T.protocol_color(str(proto))
        eff = (pg["tput_norm"] / pg["ideal_norm"]).where(pg["ideal_norm"] != 0) * 100.0
        eff = pd.to_numeric(eff, errors="coerce").astype(float)  # NA → NaN, never ambiguous
        ax.plot(pg["connections"], eff, color=c, lw=1.8, zorder=3,
                label=protocol_label(str(proto)))
        # Overlay per-point bottleneck markers: hollow = load-generator/host bound (not the
        # gateway's fan-out limit), filled = gateway-path bound. This is the answer to "why
        # doesn't it scale" drawn on the data rather than asserted in prose.
        if has_bottleneck:
            bvals = pg.get("bottleneck", pd.Series([pd.NA] * len(pg)))
            hvals = pg.get("harness_limited", pd.Series([pd.NA] * len(pg)))
            face = [("#FFFFFF" if _is_loadgen(b, h) else c) for b, h in zip(bvals, hvals)]
        else:
            face = c
        ax.scatter(pg["connections"], eff, facecolors=face, edgecolors=c,
                   linewidths=1.3, s=42, zorder=4)
        last_eff = eff.iloc[-1]
        if np.isfinite(last_eff):
            # Truncated / chain-fallback series are disclosed in the method note (F15-1);
            # the historical endpoint dagger is retired — the hollow/filled marker already
            # encodes boundness and "†" is a retired encoding.
            # Declutter (print variant): skip labels that would overprint a neighbour
            # (<4% apart) — the clustered series share the same reading anyway.
            if not (declutter and any(abs(last_eff - y0) < 4.0 for y0 in ann_ys)):
                ann_ys.append(float(last_eff))
                ax.annotate(f"{last_eff:.0f}%", (pg["connections"].iloc[-1], last_eff),
                            xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=T.FS["annot"], color=T.GREYS["ink"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(0, 112)
    ax.margins(x=0.16)
    ax.grid(True, which="both")

    # Host-busy overlay (print variant): the median p95 profile over `busy_rows` on a right
    # axis, so the high-concurrency co-limit is visible in-panel.
    drawn = False
    prof = _busy_profile(busy_rows) if busy_rows is not None else None
    if prof is not None and len(prof):
        ax_b = ax.twinx()
        ax_b.plot(prof.index, prof.to_numpy() * 100.0, color=T.GREYS["annot"],
                  ls="--", lw=1.2, zorder=2)
        drawn = True
        ax_b.set_ylim(0, 112)
        ax_b.grid(False)
        ax_b.spines["right"].set_visible(busy_label)
        if busy_label:
            ax_b.set_ylabel("host busy, p95 (%)", fontsize=T.FS["small"],
                            color=T.GREYS["annot"])
            ax_b.tick_params(axis="y", labelsize=T.FS["tick"], colors=T.GREYS["annot"])
        else:
            ax_b.set_yticks([])
    return protos, drawn


def make(bundle: RunBundle, saver: T.Saver) -> None:
    tbl = derive.scaling_table(bundle.summary)
    if tbl.empty:
        saver.record_skip(FIG_ID, NAME, "no multi-connection scaling series in the matrix")
        return
    tbl = _preferred_chain(tbl)
    has_bottleneck = "bottleneck" in tbl.columns or "harness_limited" in tbl.columns
    plotted = _series_rows(bundle.summary, tbl)

    # Per-series disclosure flags (reason spelled out in the method note). A series can
    # end early because its higher-concurrency scenarios were SKIPPED (truncated, not
    # converged), and it can ride the single-gateway 'direct' chain because no 2-gateway data
    # exists for it — both change what its curve means next to its neighbours (F15-1).
    scg_exists = "chain" in tbl.columns and (tbl["chain"].astype(str) == "scg").any()
    sk = bundle.skipped
    sk_ok = (not sk.empty) and {"scenario", "connections", "chain"}.issubset(sk.columns)
    sk_stems = sk["scenario"].astype(str).map(_series_stem) if sk_ok else None
    flags: dict[tuple[str, str], str] = {}
    for (tr_, pr_), g in tbl.groupby(["transport", "protocol"], observed=True):
        g = g.sort_values("connections")
        ch = str(g["chain"].iloc[0]) if "chain" in g.columns else "scg"
        last_c = int(g["connections"].iloc[-1])
        reasons = []
        if scg_exists and ch != "scg":
            reasons.append(f"single-gateway '{ch}' chain (no 2-gateway scg scenarios exist for it)")
        if sk_ok and not plotted.empty:
            rows = plotted[(plotted["transport"].astype(str) == str(tr_))
                           & (plotted["protocol"].astype(str) == str(pr_))]
            stems = set(rows["scenario"].astype(str).map(_series_stem))
            n_hi = int((sk_stems.isin(stems)
                        & (sk["chain"].astype(str) == ch)
                        & (pd.to_numeric(sk["connections"], errors="coerce") > last_c)).sum())
            if n_hi:
                reasons.append(f"ends at {last_c}c because its {n_hi} higher-concurrency "
                               "scenario(s) were skipped (truncated, not converged)")
        if reasons:
            flags[(str(tr_), str(pr_))] = (f"{protocol_label(str(pr_))} · "
                                           f"{transport_label(str(tr_))}: " + "; ".join(reasons))

    transports = [t for t in T.TRANSPORT_ORDER if t in set(tbl["transport"].astype(str))]
    transports += [t for t in tbl["transport"].astype(str).unique() if t not in transports]

    # Transports present in the matrix but excluded from the scaling table because they were run
    # at ONLY a single connection (so there is no scaling curve). Surface them as labelled
    # placeholder columns instead of silently dropping them — otherwise a reader asks "where is
    # UDP?". The explanation is assembled per transport from _SINGLE_ONLY_REASONS so it stays
    # honest across runs (F15-3).
    swept = set(transports)
    single_only: list[str] = []
    sm = bundle.summary
    if {"transport", "connections"}.issubset(sm.columns):
        mm = sm[sm["family"].astype(str) == "matrix"] if "family" in sm.columns else sm
        ordered_tr = [t for t in T.TRANSPORT_ORDER if t in set(mm["transport"].astype(str))]
        ordered_tr += [t for t in mm["transport"].astype(str).unique() if t not in ordered_tr]
        for t in ordered_tr:
            if t in swept:
                continue
            cc = mm[mm["transport"].astype(str) == t]["connections"].dropna()
            if len(cc) and cc.max() <= 1:
                single_only.append(t)
    columns = transports + single_only

    # Print variant: the efficiency row alone (the absolute-throughput level gap is stated
    # in prose, the blast-latency row is CO-uncorrected and banned from the print variant), no
    # placeholder columns (their reason moves to the method note), plus a per-panel host-busy
    # line so the co-limit is visible on the figure itself. That efficiency row is split into
    # one row per protection group (_ROW_GROUPS): nine near-coincident lines in a 5 cm panel
    # were unreadable in print, so plaintext routing, user-space TLS and kernel TLS each get
    # their own axes. A group absent from the whole run drops its row.
    in_print = T.print_variant()
    if in_print:
        columns = transports
        grp = tbl["protocol"].map(_row_group)
        row_groups = [(key, label, grp == key) for key, label in _ROW_GROUPS
                      if bool((grp == key).any())]
    else:
        row_groups = [("all", "", pd.Series(True, index=tbl.index))]
    split_rows = in_print and len(row_groups) >= 2

    have_lat = "latency_p99_us_mean" in tbl.columns and tbl["latency_p99_us_mean"].notna().any()
    have_lat = have_lat and not in_print
    # Rows (full variant): (0) scaling efficiency %, (1) ABSOLUTE throughput in Gbps, (2) p99
    # tail latency when present. The efficiency axis normalizes each series to its own
    # 1-connection throughput, which HIDES that UDS plateaus at ~38 Gbps and SHM at ~28–37 while
    # routed TCP climbs into the 100s; the absolute panel restores that transport-level
    # difference the % axis flattens away.
    row_abs = None if in_print else 1
    row_lat = 2 if have_lat else None
    nrows = len(row_groups) if in_print else (3 if have_lat else 2)

    import matplotlib.pyplot as plt

    # sharex=False: each transport is drawn over ONLY its measured connection range. SHM/UDS
    # stop at 16c and TCP sweeps to 1024c; a shared axis stretched the SHM/UDS panels to 1024
    # with three points bunched at the left, visually implying "SHM died early" when it was
    # simply never swept that far. Per-panel ranges make the coverage difference honest.
    # WITHIN a column the print rows do share x (Axes.sharex below): a routing sweep truncated
    # at 16c is then drawn over the same 1–64c range as its encrypted siblings, so the
    # truncation stays visible instead of being stretched to look converged.
    if in_print:
        figsize = (1.95 * len(columns) + 1.5, 3.4 + 2.8 * (len(row_groups) - 1))
    else:
        figsize = (4.7 * len(columns) + 1.4, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, len(columns), figsize=figsize, squeeze=False, sharex=False)
    if split_rows:
        for col in range(len(columns)):
            for r in range(1, len(row_groups)):
                axes[r][col].sharex(axes[0][col])

    protos_seen: set[str] = set()
    busy_drawn = False  # print-variant host-busy overlay actually drawn → key it in the legend
    for col, tr in enumerate(columns):
        # Single-connection-only transport: no scaling series exists. Draw a labelled placeholder
        # spanning the column so the transport is visibly accounted for, not silently absent.
        if tr in single_only:
            for r in range(nrows):
                axp = axes[r][col]
                axp.set_xticks([])
                axp.set_yticks([])
                for sp in axp.spines.values():
                    sp.set_visible(False)
            axtop = axes[0][col]
            T.panel_title(axtop, transport_label(tr))
            axtop.text(0.5, 0.5, "single-connection only\n— no scaling sweep\n(see caption)",
                       transform=axtop.transAxes, ha="center", va="center",
                       fontsize=T.FS["small"], color=T.GREYS["muted"], style="italic")
            continue
        sub_tr = tbl[tbl["transport"].astype(str) == tr]
        # Union of the connection counts over every row of this column (per-column x range).
        conns = sorted(int(c) for c in sub_tr["connections"].unique())
        pt = plotted[plotted["transport"].astype(str) == tr] if not plotted.empty else plotted
        for r, (gkey, glabel, gmask) in enumerate(row_groups):
            ax_t = axes[r][col]
            sub = sub_tr[gmask.loc[sub_tr.index]]
            if r == 0:
                T.panel_title(ax_t, transport_label(tr))
            if col == 0:
                ax_t.set_ylabel((glabel + "\n" if split_rows else "")
                                + "scaling efficiency\n(% of ideal-linear)")
            if sub.empty:
                # This transport has no scaling series in the group: keep the frame (shared x
                # range, same y axis) so the gap reads as "not measured here", not as a
                # missing panel.
                ax_t.set_ylim(0, 112)
                ax_t.grid(True, which="both")
                ax_t.text(0.5, 0.5, f"no {glabel} series\n(see caption)",
                          transform=ax_t.transAxes, ha="center", va="center",
                          fontsize=T.FS["small"], color=T.GREYS["muted"], style="italic")
                continue
            # Print variant: this row's host-busy profile (median p95 over its plotted rows for
            # this transport) on a right axis, so the high-concurrency co-limit is in-panel.
            busy_rows = None
            if in_print and not pt.empty and "host_busy_frac_p95" in pt.columns:
                busy_rows = pt[pt["protocol"].map(_row_group) == gkey] if split_rows else pt
            protos, drawn = _efficiency_panel(
                ax_t, sub, conns, has_bottleneck=has_bottleneck, declutter=in_print,
                busy_rows=busy_rows, busy_label=(col == len(columns) - 1))
            protos_seen |= protos
            busy_drawn = busy_drawn or drawn

        if in_print:
            bottom = axes[len(row_groups) - 1][col]
            bottom.set_xlabel("connections (log)")
            if len(conns) > 4:
                bottom.tick_params(axis="x", labelrotation=45)
            for r in range(len(row_groups) - 1):
                axes[r][col].tick_params(axis="x", labelbottom=False)
            continue

        sub = sub_tr
        # ---- Row 1: ABSOLUTE throughput (Gbps). The efficiency panel above normalizes each
        # series to its own single-connection point, which flattens away the LEVEL gap between
        # transports: routed TCP sits far above the encrypted / IPC bands (UDS ~38, SHM ~28–37).
        # Plot raw Gbps so the plateau heights are read directly, with the SAME hollow/filled
        # bottleneck encoding as the efficiency panel (hollow = load-generator/host bound).
        ax_a = axes[row_abs][col]
        for proto, pg in sub.groupby("protocol", observed=True):
            pg = pg.sort_values("connections")
            c = T.protocol_color(str(proto))
            ax_a.plot(pg["connections"], pg["throughput_gbps_mean"], color=c, lw=1.8, zorder=3)
            if has_bottleneck:
                bvals = pg.get("bottleneck", pd.Series([pd.NA] * len(pg)))
                hvals = pg.get("harness_limited", pd.Series([pd.NA] * len(pg)))
                face = [("#FFFFFF" if _is_loadgen(b, h) else c) for b, h in zip(bvals, hvals)]
            else:
                face = c
            ax_a.scatter(pg["connections"], pg["throughput_gbps_mean"], facecolors=face,
                         edgecolors=c, linewidths=1.3, s=42, zorder=4)
        ax_a.set_xscale("log", base=2)
        ax_a.set_xticks(conns)
        ax_a.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax_a.set_ylim(bottom=0)
        ax_a.margins(x=0.16)
        if col == 0:
            ax_a.set_ylabel("throughput (Gbps)")
        ax_a.grid(True, which="both")

        if have_lat:
            ax_l = axes[row_lat][col]
            for proto, pg in sub.groupby("protocol", observed=True):
                pg = pg.sort_values("connections")
                ax_l.plot(pg["connections"], pg["latency_p99_us_mean"], marker="s",
                          color=T.protocol_color(str(proto)), zorder=3)
            ax_l.set_xscale("log", base=2)
            ax_l.set_yscale("log")
            ax_l.set_xticks(conns)
            ax_l.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
            ax_l.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
            ax_l.set_xlabel("connections (log)")
            ax_l.margins(x=0.16)
            if col == 0:
                # Relabel: this is open-loop blast p99 under a saturating sender — the tail is
                # queueing delay, not service time (see the BLAST/SHM notes in the method footer).
                ax_l.set_ylabel("p99 under saturating blast\n(queueing-dominated, µs, log)")
            ax_l.grid(True, which="both")
        else:
            ax_a.set_xlabel("connections (log)")

    # Single shared protocol legend + an encoding key (boundness markers, ideal / host-busy
    # guide lines), stacked outside the right edge.
    ordered = [p for p in T.PROTOCOL_ORDER if p in protos_seen]
    handles = [plt.matplotlib.patches.Patch(facecolor=T.protocol_color(p), label=protocol_label(p)) for p in ordered]
    key = []
    if has_bottleneck:
        key += [
            plt.matplotlib.lines.Line2D([0], [0], marker="o", color="none",
                                        markerfacecolor="#FFFFFF",
                                        markeredgecolor=T.GREYS["edge"], markeredgewidth=1.3,
                                        markersize=8,
                                        label="hollow = load-generator / host-bound"),
            plt.matplotlib.lines.Line2D([0], [0], marker="o", color="none",
                                        markerfacecolor=T.GREYS["annot"],
                                        markeredgecolor=T.GREYS["edge"], markeredgewidth=1.3,
                                        markersize=8,
                                        label="filled = gateway-bound"),
        ]
    key.append(plt.matplotlib.lines.Line2D([0], [0], ls=":", color=T.GREYS["faint"], lw=0.8,
                                           label="ideal (linear)"))
    if busy_drawn:
        key.append(plt.matplotlib.lines.Line2D([0], [0], ls="--", color=T.GREYS["annot"],
                                               lw=1.2, label="host busy p95 (right axis)"))
    # Two stacked right-margin legends: two margin-reserving "outside right upper" calls
    # would overprint each other, so only ONE reserves — and it must be the WIDER of the
    # two (the boundness/guide key), or the narrow protocol column reserves a margin the
    # key then overflows into the last panel (seen as the key overprinting the TPROXY
    # x-axis in the 11.25-in print variant). The protocol legend anchors below the key
    # inside the same reserved margin (an explicit bbox_to_anchor opts it out of
    # reserving again); the anchor drops by the key's estimated height, so the placement
    # holds across the published evaluation (3.4 in) and full (7–10.5 in) figure heights.
    T.legend_right(fig, key)
    row_frac = (T.FS["small"] * 1.7) / (fig.get_size_inches()[1] * 72.0)
    T.legend_right(fig, handles, title="protocol",
                   bbox_to_anchor=(0.998, 1.0 - (len(key) * row_frac + 0.02)))

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}")
    skipped_hi = ""
    if not bundle.skipped.empty and "connections" in bundle.skipped.columns:
        hi = bundle.skipped[bundle.skipped["connections"].fillna(0) >= 64]
        if len(hi):
            skipped_hi = f"  ·  {len(hi)} scenario(s) at ≥64c skipped (coverage wall — see F13)"
    swept = ", ".join(transport_label(t) for t in transports)
    single_txt = ""
    if single_only:
        reasons = "; ".join(
            f"{transport_label(t)} — "
            f"{_SINGLE_ONLY_REASONS.get(t, 'no multi-connection scenarios in this run')}"
            for t in single_only)
        where = "not drawn" if in_print else "placeholder column"
        single_txt = (f"Run at a single connection only ({where}, no scaling "
                      f"series): {reasons}. ")

    # The causal reading is MEASURED from the plotted rows, not asserted: host_busy_frac_p95
    # decides whether "not core count" is defensible, and the per-connection relay-thread claim
    # carries its own statistic. When the host nears saturation at high concurrency, core count
    # is a live co-limit and the prose must say so (F15-2).
    busy = _busy_profile(plotted)
    hot_txt = ""
    if not plotted.empty and "cpu_hot_thread_pct_p95" in plotted.columns:
        hot = pd.to_numeric(plotted["cpu_hot_thread_pct_p95"], errors="coerce").dropna()
        if len(hot):
            hot_txt = (" while one serial relay thread per connection pegs a core "
                       f"(cpu_hot_thread_pct_p95 median {hot.median():.0f}%)")
    if busy is not None and len(busy):
        lo_c, pk_c = int(busy.index.min()), int(busy.idxmax())
        lo_b, pk_b = float(busy.iloc[0]), float(busy.max())
        if pk_b <= 0.5:
            busy_method = (f"the host stays mostly idle (host_busy_frac_p95 median {lo_b:.0%} at "
                           f"{lo_c}c, peak {pk_b:.0%} at {pk_c}c){hot_txt} — the ceiling is the "
                           "serial single-thread-per-connection data plane, not core count")
            busy_take = (f"The host peaks at {pk_b:.0%} busy (p95): the ceiling is the serial "
                         "single-thread-per-connection data plane on a no-NIC loopback host, "
                         "not core count.")
        else:
            busy_method = ("host_busy_frac_p95 (median over plotted rows) rises from "
                           f"{lo_b:.0%} at {lo_c}c to {pk_b:.0%} at {pk_c}c{hot_txt} — the serial "
                           "single-thread-per-connection data plane explains the low-concurrency "
                           "flattening, but near-saturated host CPU is a co-limit at high concurrency")
            busy_take = (f"Host p95 busy climbs from {lo_b:.0%} ({lo_c}c) to {pk_b:.0%} ({pk_c}c): "
                         "the serial per-connection data plane drives the early flattening, with "
                         "host saturation a co-limit at high concurrency.")
    else:
        busy_method = ("the gateway data plane is serial per connection; host-utilization "
                       "telemetry (host_busy_frac_p95) is absent from this run, so no "
                       "core-count claim is made")
        busy_take = ""
    flag_txt = ("".join(f"{s}. " for s in flags.values())) if flags else ""

    if in_print:
        axis_txt = ("y = achieved throughput ÷ ideal-linear (%); dashed grey line (right "
                    "axis) = median host busy p95 per connection count. ")
        if split_rows:
            axis_txt += ("Rows per transport, top to bottom: "
                         + ", ".join(label for _key, label, _mask in row_groups)
                         + ", sharing each column's x range and efficiency axis; the "
                           "host-busy line is the median over that row's series. ")
    else:
        axis_txt = ("Top y = achieved throughput ÷ ideal-linear (%); middle y = absolute "
                    "throughput (Gbps), the level gap the efficiency axis normalizes away. ")
    lat_notes = "" if in_print else (T.BLAST_LATENCY_NOTE + " · " + T.SHM_STALL_NOTE)
    T.add_method_note(fig, f"matrix family, one topology per (transport, protocol); {swept} sweep "
                           "connection count here. " + single_txt + flag_txt +
                           f"Single loopback host, no NIC; {busy_method}. " + axis_txt
                           + lat_notes)

    # Data-driven takeaway compared at the deepest connection count EVERY series reaches: a max
    # over per-series endpoints crowned whichever sweep was truncated shallowest (a harness-bound
    # 16c-ending series once "won" against series that kept scaling down to 64/1024c; F15-1).
    take = "Hollow = load-generator bound, filled = gateway-relay bound."

    def _eff_pct(row: pd.Series) -> float:
        return float(row["tput_norm"] / row["ideal_norm"] * 100.0) if row["ideal_norm"] else np.nan

    series_conns = [set(g["connections"].dropna().astype(int))
                    for _, g in tbl.groupby(["transport", "protocol"], observed=True)]
    common = set.intersection(*series_conns) if series_conns else set()
    if common:
        cc = max(common)
        at_cc: list[tuple[float, str, str, bool]] = []
        deeper: list[tuple[float, int]] = []
        for (tr_, pr_), g in tbl.groupby(["transport", "protocol"], observed=True):
            g = g.sort_values("connections")
            rc = g[g["connections"] == cc].iloc[0]
            e_cc = _eff_pct(rc)
            if np.isfinite(e_cc):
                at_cc.append((e_cc, protocol_label(str(pr_)), transport_label(str(tr_)),
                              _is_loadgen(rc.get("bottleneck"), rc.get("harness_limited"))))
            last = g.iloc[-1]
            e_last = _eff_pct(last)
            if int(last["connections"]) > cc and np.isfinite(e_last):
                deeper.append((e_last, int(last["connections"])))
        if at_cc:
            best = max(at_cc, key=lambda x: x[0])
            # Only classify the winner when the run recorded bottleneck telemetry at all.
            bn = (("load-generator-bound even there" if best[3] else "gateway-bound")
                  if has_bottleneck else "bottleneck class unrecorded")
            take = (f"At {cc} connections — the deepest sweep every series reaches — the best "
                    f"series attains {best[0]:.0f}% of ideal-linear ({best[1]} · {best[2]}, {bn})")
            if deeper:
                lo, hi = min(e for e, _ in deeper), max(e for e, _ in deeper)
                cmin, cmax = min(c for _, c in deeper), max(c for _, c in deeper)
                rng = f"{lo:.1f}–{hi:.1f}%" if hi - lo > 0.05 else f"{hi:.1f}%"
                span = f"{cmin}–{cmax}" if cmin != cmax else f"{cmax}"
                take += f"; series swept further end at {rng} ({span} connections)"
            take += (". " + (busy_take + " " if busy_take else "")
                     + "Hollow = load-generator bound, filled = gateway-relay bound.")
    T.add_takeaway(fig, take)
    T.add_provenance(fig, bundle.caption() + "  ·  gateway path preferred per (transport,protocol); single payload per series" + skipped_hi)
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
