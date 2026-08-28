"""
F26 — Loopback-testbed realism on a physical 1 GbE path (the wire campaign).

The thesis' entire measurement campaign ran across loopback; this figure is the
empirical answer to "how realistic was that?". The same offered-load grid was
driven through the same gateway pair twice, changing exactly one variable — the
inter-gateway hop's address (127.0.0.1 vs the peer's link IP) — so every panel
overlays the two media at matched offered load:

  a. achieved vs offered: both media track the offered=achieved diagonal until
     the wire clamps at the 1 GbE goodput ceiling (0.9493 Gbit/s for TCP/IPv4 at
     MTU 1500); loopback continues through 1 Gbit/s. The knee is annotated with
     the sender-side back-pressure evidence (send-lag explosion).
  b. closed-loop RTT p99 under load — the curve loopback structurally cannot
     produce (no queue to build). Self-timing on the sender's clock, so it is
     valid across hosts with unsynchronised clocks.
  c. CPU cost at matched offered load — where the two media agree, the loopback
     campaign characterised the gateway rather than the medium.

Honesty rails: everything here is sender-side (`measurement_side=sender`; the
peer's delivered/loss reports merge in separately and do not change this
figure's metric), grids are single-shot (no fabricated error bars), the
contaminated pre-guard 64 B RTT cells never reach this module (dropped by the
wire loader), and the link-limited clamp is encoded as the ACCENT ceiling guide
— never with the hollow-marker/hatch/dagger conventions, which keep their
published harness-limited meaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle, protocol_label

FIG_ID = "F26"
NAME = "f26_wire_loopback_sweep"
TITLE = "Loopback-testbed realism on a physical 1 GbE path"

_MEDIUM_STYLE = {
    # House rule: colour stays protocol identity; the medium is linestyle/marker.
    "wire": {"ls": "-", "marker": "o", "mfc": None, "label": "wire (1 GbE)"},
    "loopback": {"ls": ":", "marker": ".", "mfc": None, "label": "loopback"},
}
_TCP_PROTO = "ktls/1.3+mtls"
_UDP_PROTO = "dtls/1.2+mtls"


def _sweep_rows(wire_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Coarse + fine sweep rows for one cell prefix, deduped per (medium, offered).

    The coarse grid (baseline campaigns, 50–950 step 100) and the fine knee grid
    (880–1000 step 10) overlap at 950; the knee campaign ran later with the
    hardened probe, so it wins the duplicate.
    """
    df = wire_df[wire_df["role"].isin(("baseline", "knee"))]
    df = df[df["cell"].str.startswith(prefix)].copy()
    if df.empty:
        return df
    df["offered_gbps"] = pd.to_numeric(df["offered_mbps"], errors="coerce") / 1000.0
    df["_pref"] = (df["role"] == "knee").astype(int)
    df = (
        df.sort_values("_pref")
        .drop_duplicates(subset=["medium", "offered_gbps"], keep="last")
        .sort_values("offered_gbps")
    )
    return df.drop(columns="_pref")


def _knee(df_wire: pd.DataFrame) -> dict | None:
    """First offered point where the wire sender's pacing lag explodes (>10x base)."""
    lag = pd.to_numeric(df_wire.get("send_lag_mean_us"), errors="coerce")
    sel = df_wire.assign(_lag=lag).dropna(subset=["_lag"]).sort_values("offered_gbps")
    base = sel[sel["offered_gbps"] <= 0.9]["_lag"]
    if base.empty:
        return None
    base_lag = float(base.median())
    above = sel[sel["_lag"] > 10 * max(base_lag, 1e-9)]
    if above.empty:
        return None
    first = above.iloc[0]
    return {
        "offered_gbps": float(first["offered_gbps"]),
        "base_lag_us": base_lag,
        "peak_lag_us": float(sel["_lag"].max()),
        "achieved": float(first["throughput_gbps_mean"]),
    }


def _plot_medium(ax, df: pd.DataFrame, ycol: str, color: str, *, label_prefix: str = "") -> None:
    for medium, style in _MEDIUM_STYLE.items():
        sel = df[df["medium"] == medium].sort_values("offered_gbps")
        y = pd.to_numeric(sel.get(ycol), errors="coerce")
        mask = y.notna()
        if not mask.any():
            continue
        ax.plot(
            sel["offered_gbps"][mask], y[mask],
            ls=style["ls"], marker=style["marker"], color=color,
            markersize=4.5, lw=1.6, label=f"{label_prefix}{style['label']}",
        )


def make(bundle: RunBundle, saver: T.Saver) -> None:
    wb = getattr(bundle, "wire", None)
    if wb is None or wb.df.empty:
        saver.record_skip(FIG_ID, NAME,
                          "no wire campaign dirs found (pass --wire-results SCG-SESHAT/results)")
        return
    tcp = _sweep_rows(wb.df, "sweep-tcp-")
    if tcp.empty or (tcp["medium"] == "wire").sum() == 0:
        saver.record_skip(FIG_ID, NAME, "no wire sweep-tcp rows in the wire campaigns")
        return
    thesis = T.thesis_variant()
    udp = pd.DataFrame() if thesis else _sweep_rows(wb.df, "sweep-udp-")

    ceiling = pd.to_numeric(tcp["ceiling_gbps"], errors="coerce").dropna()
    ceiling = float(ceiling.iloc[0]) if not ceiling.empty else 0.9493
    knee = _knee(tcp[tcp["medium"] == "wire"])
    color = T.protocol_color(_TCP_PROTO)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nrows = 3 if thesis else 4
    fig, axes = plt.subplots(nrows, 1, figsize=(7.6, 2.2 * nrows), sharex=True, squeeze=False)
    axa, axb, axc = axes[0][0], axes[1][0], axes[2][0]

    # --- (a) achieved vs offered ----------------------------------------------------------
    lim = max(1.05, float(tcp["offered_gbps"].max()) + 0.05)
    axa.plot([0, lim], [0, lim], ls=":", color=T.GREYS["faint"], lw=1.0,
             label="offered = achieved")
    axa.axhline(ceiling, color=T.ACCENT, lw=1.2, ls=(0, (6, 2)),
                label=f"1 GbE goodput ceiling ({ceiling:.4f} Gbit/s)")
    _plot_medium(axa, tcp, "throughput_gbps_mean", color)
    if not udp.empty:
        _plot_medium(axa, udp, "throughput_gbps_mean", T.protocol_color(_UDP_PROTO),
                     label_prefix="UDP · ")
    axa.set_ylabel("achieved (Gbit/s, sender)")
    axa.set_ylim(0, lim)
    if knee is not None:
        axa.annotate(
            f"knee at {knee['offered_gbps'] * 1000:.0f} Mbit/s:\nopen-loop back-pressure, "
            f"send-lag {knee['base_lag_us']:.0f} µs → {knee['peak_lag_us'] / 1000:.0f} ms",
            xy=(knee["offered_gbps"], knee["achieved"]),
            xytext=(0.42, 0.30), textcoords="axes fraction", fontsize=T.FS["annot"],
            color=T.GREYS["ink"],
            arrowprops=dict(arrowstyle="->", color=T.GREYS["annot"], lw=0.9),
        )
    handles = [
        Line2D([0], [0], ls="-", marker="o", color=color, markersize=4.5, lw=1.6,
               label=_MEDIUM_STYLE["wire"]["label"]),
        Line2D([0], [0], ls=":", marker=".", color=color, markersize=4.5, lw=1.6,
               label=_MEDIUM_STYLE["loopback"]["label"]),
    ]
    if not udp.empty:
        udp_color = T.protocol_color(_UDP_PROTO)
        handles += [
            Line2D([0], [0], ls="-", marker="o", color=udp_color, markersize=4.5, lw=1.6,
                   label=f"UDP · {_MEDIUM_STYLE['wire']['label']}"),
            Line2D([0], [0], ls=":", marker=".", color=udp_color, markersize=4.5, lw=1.6,
                   label=f"UDP · {_MEDIUM_STYLE['loopback']['label']}"),
        ]
    handles += [
        Line2D([0], [0], ls=(0, (6, 2)), color=T.ACCENT, lw=1.2,
               label=f"1 GbE goodput ceiling ({ceiling:.4f} Gbit/s)"),
        Line2D([0], [0], ls=":", color=T.GREYS["faint"], lw=1.0, label="offered = achieved"),
    ]
    T.legend_right(fig, handles)

    # --- (b) closed-loop RTT p99 under load ----------------------------------------------
    _plot_medium(axb, tcp, "rtt_us_p99", color)
    axb.set_yscale("log")
    axb.yaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{v:g}")
    )
    axb.set_ylabel("round-trip p99 (µs, log)")
    axb.grid(axis="y", which="both", alpha=0.5)

    # --- (c) CPU at matched offered load --------------------------------------------------
    _plot_medium(axc, tcp, "cpu_pct_mean", color)
    if not udp.empty:
        _plot_medium(axc, udp, "cpu_pct_mean", T.protocol_color(_UDP_PROTO),
                     label_prefix="UDP · ")
    axc.set_ylabel("gateway CPU (%)")
    axc.set_ylim(bottom=0)

    # --- (d, full variant) send-lag — the knee mechanism ----------------------------------
    if not thesis:
        axd = axes[3][0]
        _plot_medium(axd, tcp, "send_lag_mean_us", color)
        if not udp.empty:
            _plot_medium(axd, udp, "send_lag_mean_us", T.protocol_color(_UDP_PROTO),
                         label_prefix="UDP · ")
        axd.set_yscale("log")
        axd.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
        axd.set_ylabel("send-lag mean (µs, log)")
        axd.grid(axis="y", which="both", alpha=0.5)

    axes[-1][0].set_xlabel("offered load (Gbit/s)")

    # --- computed takeaway ----------------------------------------------------------------
    wire_t = tcp[tcp["medium"] == "wire"]
    lo_t = tcp[tcp["medium"] == "loopback"]
    take = ""
    if not wire_t.empty and not lo_t.empty:
        achieved = pd.to_numeric(wire_t["throughput_gbps_mean"], errors="coerce")
        # The clamp plateau, not the single best point: a 10 s window can buffer
        # itself momentarily above the theoretical goodput line, and quoting that
        # as ">100% of ceiling" would be indefensible.
        if knee is not None:
            plateau = achieved[wire_t["offered_gbps"] >= knee["offered_gbps"]]
        else:
            plateau = achieved
        wmax = float(plateau.median()) if plateau.notna().any() else float(achieved.max())
        below = wire_t[wire_t["offered_gbps"] <= 0.9]
        cpu_max = float(pd.to_numeric(below["cpu_pct_mean"], errors="coerce").max())
        matched = wire_t.merge(lo_t, on="offered_gbps", suffixes=("_w", "_l"))
        p99_w = pd.to_numeric(matched["rtt_us_p99_w"], errors="coerce")
        p99_l = pd.to_numeric(matched["rtt_us_p99_l"], errors="coerce")
        ratio = (p99_w / p99_l).dropna()
        take = (
            f"On the physical 1 GbE path the gateway clamps at {wmax:.3f} Gbit/s = "
            f"{100 * wmax / ceiling:.1f}% of the line-rate goodput ceiling with CPU ≤ "
            f"{cpu_max:.0f}% below the knee"
        )
        if knee is not None:
            take += (
                f"; past {knee['offered_gbps'] * 1000:.0f} Mbit/s only sender back-pressure "
                f"grows (send-lag ×{knee['peak_lag_us'] / max(knee['base_lag_us'], 1e-9):.0f})"
            )
        take += (
            ". Loopback tracks offered load through 1 Gbit/s — loopback results transfer "
            "to the wire everywhere below the link ceiling; RTT under load is "
            f"×{ratio.median():.1f} on the wire."
        )

    pre = wb.preflight
    pf_note = ""
    if not pre.empty:
        parts = []
        for medium in ("wire", "loopback"):
            sel = pre[pre["medium"] == medium]
            if not sel.empty and sel["rtt_us_mean"].notna().any():
                parts.append(f"{medium} {sel['rtt_us_mean'].median():.0f} µs")
        if parts:
            pf_note = f" 2 s preflight RTT baselines: {' / '.join(parts)}."

    T.set_headline(fig, f"{TITLE}  ·  {protocol_label(_TCP_PROTO)} over TCP", y=1.02)
    if take:
        T.add_takeaway(fig, take)
    T.add_method_note(
        fig,
        "Identical offered-load grid on both media; only the inter-gateway hop address "
        "differs (127.0.0.1 vs the peer's link IP; mutual TLS on both, M-13). All values "
        "sender-side (measurement_side=sender): TCP backpressure makes sender bytes ≈ "
        "delivered within one socket buffer; the peer's delivered/loss reports are merged "
        "separately and do not alter this figure. Coarse (50–950, step 100 Mbit/s) and fine "
        "knee (880–1000, step 10) grids merged; the duplicated 950 point keeps the fine-grid "
        "value. One measurement per grid point — no error bars are drawn or implied. RTT is "
        "a concurrent paced closed-loop probe timed entirely on the sender's clock (one-way "
        "latency is unmeasurable across unsynchronised hosts and is reported empty). "
        "Pre-guard 64 B RTT cells are excluded (echo-reframing artefact)." + pf_note,
    )
    T.add_provenance(fig, bundle.caption() + "  ·  " + wb.provenance({"baseline", "knee"}))
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
