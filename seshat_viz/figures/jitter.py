"""
F19 — Jitter & determinism: the metric that matters for ETCS/EuroRadio signaling.

Railway signaling cares less about peak throughput than about *predictability* — the packet
delay variation (jitter) a control loop must tolerate. SESHAT records `jitter_us_mean` (mean
absolute consecutive-sample difference, i.e. PDV) for every scenario, but it only ever
appeared as one radar spoke. Left: jitter per protocol, grouped by transport, lowest (most
deterministic) first. Right: a determinism quadrant — throughput against jitter — so the
configurations that are both fast *and* steady are visible at a glance.

Jitter is recorded on only a subset of scenarios, so the matched cell is voted on the
jitter-BEARING rows (F19-1): voting on the full measurement pool can pin a payload size
where entire transports carry no encrypted jitter, silently collapsing the crypto story to
a single transport. The comparison also pins topology: 0-gateway loopback/baseline
rows are excluded and the gateway count joins the matched-cell controls, so a no-gateway
baseline never stands in for a gateway configuration.

Both jitter axes are log-scaled so the routing cluster separates from the crypto/DTLS
outliers instead of collapsing onto the axis floor. SHM rows draw hollow, keyed by a legend
entry in each panel: PDV cannot see SHM's *steady* multi-ms stall offset — jitter is
the wrong lens for SHM until that harness stall is fixed (see theme.SHM_STALL_NOTE).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F19"
NAME = "f19_jitter_determinism"
TITLE = "Jitter (packet delay variation) & determinism"

# Above this best-encrypted / plaintext-routing PDV ratio the takeaway may not claim that
# encryption "adds little" — it states the computed multiplier instead. The old hardcoded
# phrasing generalized a single-transport artifact to the gateway (F19-1).
_ADDS_LITTLE_RATIO = 1.5


def _gateway_blast_pool(summary: pd.DataFrame) -> pd.DataFrame:
    """Sustained-blast rows that actually traverse a gateway.

    A per-configuration gateway figure must not let a 0-gateway loopback/baseline row stand
    in for a gateway config — e.g. `baseline_udp_loopback_*` as the only jitter-bearing
    "UDP · routing" row silently becoming that bar (F19-2). Prefers the loader's
    `n_gateways`; falls back to the naming convention when the summary was loaded without
    `_enrich_factors`.
    """
    pool = derive.throughput_scenarios_only(summary)
    if "n_gateways" in pool.columns:
        return pool[~(pd.to_numeric(pool["n_gateways"], errors="coerce") == 0)]
    if "scenario" in pool.columns:
        name = pool["scenario"].astype(str)
        return pool[~(name.str.startswith("baseline_") | name.str.contains("loopback"))]
    return pool


def _matched_jitter(
    summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """The confound-controlled jitter slice: (jitter rows, chosen cell, all rows in cell).

    The cell is voted on jitter-BEARING rows only — `matched_cell` on the full pool picks
    the size with the broadest *measurement* coverage, which can be a size where every
    encrypted row of some transports carries NaN jitter (F19-1: the crypto bars degrade to
    one transport at a jitter-pessimal size). `n_gateways` joins the controls so topology
    is held fixed too (F19-2); coverage ties prefer the fewest gateways. The third return
    value is the same cell WITHOUT the jitter dropna, so the caller can disclose which
    measured configs carry no jitter at all (gaps, not zeros).
    """
    pool = _gateway_blast_pool(summary)
    if "jitter_us_mean" not in pool.columns or not pool["jitter_us_mean"].notna().any():
        return pool.iloc[0:0], {}, pool.iloc[0:0]
    jit = pool.dropna(subset=["jitter_us_mean"])
    controls = ("connections", "n_gateways", "message_bytes")
    df, chosen = derive.matched_cell(
        jit, ["transport", "protocol"], controls=controls,
        fixed={"connections": 1}, prefer={"n_gateways": "min"},
    )
    if df.empty:
        # No 1-connection jitter rows at all (degenerate run): vote connections too rather
        # than skip the figure — the cell stamp still discloses whatever was chosen.
        df, chosen = derive.matched_cell(
            jit, ["transport", "protocol"], controls=controls,
            prefer={"connections": "min", "n_gateways": "min"},
        )
    cell = pool
    for col, val in chosen.items():
        if col in cell.columns:
            cell = cell[cell[col] == val]
    return df, chosen, cell


def make(bundle: RunBundle, saver: T.Saver) -> None:
    # Match on a single connection count, gateway count and shared size so the quadrant/bars
    # are like-for-like — otherwise UDP's jitter is a loopback baseline, TCP's throughput a
    # multi-conn aggregate, and the crypto bars one transport (see _matched_jitter).
    df, chosen, cell = _matched_jitter(bundle.summary)
    if df.empty:
        saver.record_skip(FIG_ID, NAME, "no jitter_us_mean on gateway blast rows")
        return
    tbl = derive.jitter_table(df)
    if tbl.empty:
        saver.record_skip(FIG_ID, NAME, "no jitter_us_mean in the matched cell")
        return
    tbl = tbl.dropna(subset=["jitter_us_mean"]).copy()

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def _stall_handle(wrap: bool = False) -> Line2D:
        # The one stall-affected encoding: hollow marks, keyed in every panel legend.
        # `wrap` breaks the label over two lines for narrow in-axes legends.
        label = ("hollow = stall-affected\n(harness receive-poll)" if wrap
                 else "hollow = stall-affected (harness receive-poll)")
        return Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                      markeredgecolor=T.GREYS["annot"], markeredgewidth=1.3, markersize=8,
                      label=label)

    # Adaptive unit formatter so the sub-µs cluster stays legible (ns below 1 µs, µs above) —
    # reused for the Panel-A bar annotations and the takeaway sentence.
    def _us(v: float) -> str:
        return f"{v*1000:.0f} ns" if abs(v) < 1.0 else f"{v:.1f} µs"

    have_tput = "throughput_gbps_mean" in tbl.columns and tbl["throughput_gbps_mean"].notna().any()
    fig, axes = plt.subplots(1, 2 if have_tput else 1, figsize=(12.0 if have_tput else 6.6, 5.2), squeeze=False)
    # Reserve a bottom strip for the two method-note lines + provenance: constrained layout
    # cannot see fig.text, so without this the panels' x-labels land on top of the notes.
    try:
        fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))
    except AttributeError:
        pass  # no constrained layout engine (non-print rcParams): notes keep the old spot

    # --- Panel A ---
    axa = axes[0][0]
    if T.print_variant():
        # Print variant: the full ladder (50+ rows at print size) overwhelms the page.
        # Draw a per-interface dot plot instead: one row per transport (locality order),
        # every row carrying the SAME fixed representative protocol set — the chapter's
        # family representatives extended by TLS 1.2 and the DTLS variants so every value
        # the prose cites is a member. A thin range line spans each row's min→max, so the
        # per-interface spread reads directly and rows stay like-for-like comparable.
        # The complete ladder remains available in the full render.
        _THESIS_PROTOS = ["none", "tls/1.2", "tls/1.3", "ktls/1.3",
                          "tls/1.3+mtls", "tls/1.2+integrity", "dtls/1.0", "dtls/1.2"]
        dp = tbl[tbl["protocol"].astype(str).isin(_THESIS_PROTOS)].copy()
        rows = [t for t in ["shm", "shm-slot", "unix", "tcp", "tproxy", "udp"]
                if t in set(dp["transport"].astype(str))]
        rows += [t for t in dp["transport"].astype(str).unique() if t not in rows]
        lo_all = float(dp["jitter_us_mean"].min())
        hi_all = float(dp["jitter_us_mean"].max())
        seen_protos: list[str] = []
        for yi, tr in enumerate(rows):
            g = dp[dp["transport"].astype(str) == tr]
            stall = tr in ("shm", "shm-slot")
            gmin, gmax = float(g["jitter_us_mean"].min()), float(g["jitter_us_mean"].max())
            axa.plot([gmin, gmax], [yi, yi], color=T.GREYS["faint"], lw=1.4, zorder=1)
            for _, r in g.iterrows():
                p = str(r["protocol"])
                # Stall-affected rows draw hollow (legend-keyed): their jitter is real but
                # misleading — the multi-ms SHM latency is a *steady* stall offset that
                # never surfaces as PDV.
                if stall:
                    axa.scatter([r["jitter_us_mean"]], [yi], facecolor="white",
                                edgecolor=T.protocol_color(p), s=72, linewidth=1.3, zorder=3)
                else:
                    axa.scatter([r["jitter_us_mean"]], [yi], color=T.protocol_color(p), s=72,
                                edgecolor=T.GREYS["edge"], linewidth=0.5, zorder=3)
                if p not in seen_protos:
                    seen_protos.append(p)
            # Annotate the row extremes only, so six rows carry twelve numbers, not fifty.
            axa.annotate(_us(gmin), (gmin, yi), xytext=(-6, 0), textcoords="offset points",
                         ha="right", va="center", fontsize=T.FS["annot"], color=T.GREYS["ink"])
            axa.annotate(_us(gmax), (gmax, yi), xytext=(7, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=T.FS["annot"], color=T.GREYS["ink"])
        axa.set_yticks(np.arange(len(rows)))
        axa.set_yticklabels([transport_label(t) for t in rows], fontsize=T.FS["tick"])
        axa.invert_yaxis()
        axa.set_xscale("log")
        axa.set_xlim(lo_all * 0.35, hi_all * 5.0)  # room for the end-of-line value labels
        axa.set_xlabel("jitter / PDV (µs, log — lower = more deterministic)")
        T.panel_title(axa, "Jitter per interface — fixed representative protocol set")
        handles_a = [Line2D([0], [0], marker="o", color="none",
                            markerfacecolor=T.protocol_color(p), markeredgecolor=T.GREYS["edge"],
                            markeredgewidth=0.5, markersize=8, label=protocol_label(p))
                     for p in seen_protos]
        handles_a.append(_stall_handle(wrap=True))
        T.legend_inline(axa, handles=handles_a, loc="upper right",
                        handletextpad=0.3, borderaxespad=0.2)
        axa.grid(axis="x", which="both", alpha=0.5)
    else:
        # Full render: jitter per (transport, protocol), ascending (lowest = most deterministic).
        a = tbl.sort_values("jitter_us_mean").reset_index(drop=True)
        y = np.arange(len(a))
        # Both SHM ring variants go through the same harness receive path, so the
        # stall-offset caveat applies to the slot ring exactly as to the byte-stream one.
        a_shm = a["transport"].astype(str).isin(["shm", "shm-slot"])
        # Stall-affected rows draw hollow (legend-keyed): their jitter is real but misleading —
        # the multi-ms SHM latency is a *steady* stall offset, so it never surfaces as PDV (see note).
        colors = ["white" if s else T.protocol_color(str(p)) for p, s in zip(a["protocol"], a_shm)]
        edges = [T.protocol_color(str(p)) if s else T.GREYS["edge"]
                 for p, s in zip(a["protocol"], a_shm)]
        widths = [1.1 if s else 0.5 for s in a_shm]
        axa.barh(y, a["jitter_us_mean"], color=colors, edgecolor=edges, linewidth=widths)
        axa.set_yticks(y)
        axa.set_yticklabels([f"{transport_label(str(t))} · {protocol_label(str(p))}"
                             for t, p in zip(a["transport"], a["protocol"])], fontsize=T.FS["annot"])
        axa.invert_yaxis()
        # Log-x so the routing cluster separates from the crypto/DTLS outliers instead of
        # collapsing against the left spine.
        axa.set_xscale("log")
        lo, hi = float(a["jitter_us_mean"].min()), float(a["jitter_us_mean"].max())
        axa.set_xlim(lo * 0.5, hi * 4.0)  # room for the right-hand value labels
        axa.set_xlabel("jitter / PDV (µs, log — lower = more deterministic)")
        T.panel_title(axa, "Per-configuration jitter")
        for i, v in enumerate(a["jitter_us_mean"]):
            T.annotate_value(axa, v, i, _us(v), horizontal=True)
        T.legend_inline(
            axa,
            handles=[Patch(facecolor="white", edgecolor=T.GREYS["annot"], linewidth=1.1,
                           label="hollow = stall-affected (harness receive-poll)")],
            loc="upper right")
        axa.grid(axis="x")

    # --- Panel B: determinism quadrant (throughput vs jitter) ---
    if have_tput:
        axb = axes[0][1]
        for _, r in tbl.iterrows():
            # Same stall encoding as Panel A: draw the point hollow (legend-keyed) so its
            # jitter isn't read as "SHM is the steadiest path" when its mean latency is a
            # multi-ms stall offset.
            if str(r["transport"]) in ("shm", "shm-slot"):
                axb.scatter(r["throughput_gbps_mean"], r["jitter_us_mean"],
                            facecolor="white",
                            edgecolor=T.protocol_color(str(r["protocol"])),
                            marker=T.transport_marker(str(r["transport"])),
                            s=95, linewidth=1.3, zorder=3)
            else:
                axb.scatter(r["throughput_gbps_mean"], r["jitter_us_mean"],
                            color=T.protocol_color(str(r["protocol"])),
                            marker=T.transport_marker(str(r["transport"])),
                            s=95, edgecolor=T.GREYS["edge"], linewidth=0.5, zorder=3)
        # Log-y for the same reason as Panel A: without it the worst outliers flatten the
        # entire routing cluster onto the axis floor.
        axb.set_yscale("log")
        med_j = float(np.nanmedian(tbl["jitter_us_mean"]))
        axb.axhline(med_j, color=T.GREYS["faint"], ls="--", lw=0.9, zorder=1)
        axb.annotate("more deterministic ↓", (axb.get_xlim()[1], med_j), xytext=(-4, -10),
                     textcoords="offset points", ha="right", fontsize=T.FS["annot"],
                     color=T.GREYS["annot"])
        axb.set_xlabel("throughput (Gbps)")
        axb.set_ylabel("jitter / PDV (µs, log)")
        T.panel_title(axb, "Determinism quadrant — fast (right) & steady (low)")
        axb.grid(True)
        # Transport key OUTSIDE the quadrant (Questions.md: the in-panel legend was blocking
        # data points), as the sanctioned figure-level right-edge column.
        present = [t for t in T.TRANSPORT_ORDER if t in set(tbl["transport"].astype(str))]
        thandles = [Line2D([0], [0], marker=T.transport_marker(t), color="none",
                    markerfacecolor=T.transport_color(t), markeredgecolor=T.transport_color(t),
                    markersize=8, label=transport_label(t)) for t in present]
        thandles.append(_stall_handle())
        T.legend_right(fig, thandles, title="transport")

    # Data-driven takeaway, formatted so sub-µs values don't round to "0". Compare within the
    # stream transports (TCP), where encryption's added PDV is the meaningful signal — across
    # transports SHM routing can jitter more than encrypted TCP, so a blanket "routing is most
    # deterministic" claim would be false. Whether encryption "adds little" is itself computed
    # (_ADDS_LITTLE_RATIO), and the worst-encrypted config is named from the data, not assumed
    # to be DTLS. (_us is defined above and reused here.)
    tcp = tbl[tbl["transport"].astype(str) == "tcp"]
    src = tcp if not tcp.empty else tbl
    base = src[src["protocol"].astype(str) == "none"]["jitter_us_mean"].min()
    enc = src[src["protocol"].astype(str) != "none"]["jitter_us_mean"].min()
    enc_all = tbl[tbl["protocol"].astype(str) != "none"]
    take = "Lower jitter = more predictable delivery — the determinism a signaling control loop must tolerate."
    if np.isfinite(base) and np.isfinite(enc):
        scope = "On TCP, plaintext" if not tcp.empty else "Plaintext"
        if base > 0 and enc / base > _ADDS_LITTLE_RATIO:
            mid = f"encryption multiplies PDV ≈ {enc / base:.1f}× (best encrypted ≈ {_us(enc)})"
        else:
            mid = f"stream encryption adds little (best encrypted ≈ {_us(enc)})"
        tail = ""
        if not enc_all.empty:
            w = enc_all.loc[enc_all["jitter_us_mean"].idxmax()]
            tail = (f"; worst encrypted ({transport_label(str(w['transport']))} · "
                    f"{protocol_label(str(w['protocol']))}) ≈ {_us(float(w['jitter_us_mean']))}")
        take = f"{scope} routing jitters least ({_us(base)}); {mid}{tail}."

    # Cell stamp: render n_gateways as "N gw" rather than fmt_cell's raw "n_gateways=N".
    stamp = dict(chosen)
    gw_val = stamp.pop("n_gateways", None)
    cell_txt = T.fmt_cell(stamp)
    if gw_val is not None:
        try:
            cell_txt += f" · {int(gw_val)} gw"
        except (TypeError, ValueError):
            cell_txt += f" · {gw_val} gw"

    # Coverage disclosure (F19-1/F19-3): jitter exists for only a subset of the configs
    # measured in this cell — name the transports whose configs are missing (entirely, or
    # all their encrypted rows) so absent bars read as data gaps, not zeros.
    plotted = {(str(t), str(p)) for t, p in zip(tbl["transport"], tbl["protocol"])}
    measured = plotted | {(str(t), str(p)) for t, p in zip(cell["transport"], cell["protocol"])}
    plotted_tr = {t for t, _ in plotted}
    gaps = []
    absent_tr = sorted({t for t, _ in measured} - plotted_tr)
    if absent_tr:
        gaps.append("/".join(transport_label(t) for t in absent_tr))
    enc_absent = sorted({t for t, p in measured - plotted
                         if p != "none" and t in plotted_tr
                         and not any(pt == t and pp != "none" for pt, pp in plotted)})
    if enc_absent:
        gaps.append("encrypted " + "/".join(transport_label(t) for t in enc_absent))
    gap_txt = f" (none for {', '.join(gaps)})" if gaps else ""
    if len(plotted) < len(measured):
        coverage = (f"Jitter is recorded for a subset of scenarios only: {len(plotted)}/{len(measured)} "
                    f"configs measured in this cell carry it{gap_txt} — gaps are missing data, not zeros.")
    else:
        coverage = (f"Jitter is recorded on all {len(plotted)}/{len(measured)} configs measured in "
                    f"this cell{gap_txt} — absent dots are combinations not measured, not zeros.")

    # Harness-limited disclosure (F19-2/F19-3): a plotted config whose blast throughput hit
    # the harness ceiling must not be read as gateway capacity on the quadrant's x axis.
    hl_txt = ""
    if have_tput and "harness_limited" in df.columns:
        flags = df["harness_limited"].astype(str).str.lower() == "true"
        n_hl = int(df.assign(_hl=flags)
                     .groupby(["transport", "protocol"], observed=True)["_hl"].any().sum())
        if n_hl:
            hl_txt = (f" {n_hl}/{len(tbl)} plotted configs are harness-limited — their throughput "
                      "positions are a harness ceiling, not gateway capacity.")

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.03)
    T.add_takeaway(fig, take)
    reduction_txt = ("Left panel draws a fixed representative protocol set per interface "
                     "(absent dots are unmeasured cells, not zeros); the full per-configuration "
                     "ladder is in the repository render. " if T.print_variant() else "")
    T.add_method_note(fig, "matched " + cell_txt + ", 0-gateway loopback baselines excluded; "
                      "jitter derived from open-loop one-way latencies (relative), log axes; each "
                      "bar/point averages all scenarios at that cell (cipher sweeps included). "
                      + reduction_txt +
                      "Transport order (SHM,SHM (slot),UDS,TCP,TPROXY,UDP) is a locality legend, "
                      "not a ranking (see F5).", y=0.036)
    T.add_method_note(fig, coverage + hl_txt +
                      " * PDV cannot see SHM's *steady* multi-ms stall offset — "
                      + T.SHM_STALL_NOTE, y=0.020)
    T.add_provenance(fig, bundle.caption() + "  ·  jitter = mean |Δ| of consecutive one-way latencies (PDV, not stddev)")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
