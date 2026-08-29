"""
F2 — Payload-size scaling small-multiples.

One column per transport; top row throughput vs message size (log-x), middle row p99
latency under saturating blast, bottom row the honest closed-loop ping-pong RTT — both
latencies vs message size. One line per protocol (color), 2-gateway path solid / 1-gateway
path dashed, with CI95 throughput bands (shaded for multi-size series, whiskers for a
lone size). Reveals where per-message overhead dominates (small messages, syscall-bound)
versus where crypto/link capacity dominates (large messages).

The MIDDLE row is OPEN-LOOP *blast* p99 — queue depth under an unthrottled sender,
NOT service time; only its ranking is meaningful. The BOTTOM row is the
coordinated-omission-free closed-loop RTT — the honest service latency, at the same
payload sizes, so the blast/RTT gap per size is visible directly (the per-size,
per-transport companion to F16). The blast sweep is scoped to the matrix family at a
single connection; the RTT row is drawn from closed-loop ping-pong rows only
(mode=='pingpong': matrix_lat_* / pp_*), with the linestyle keyed to the measured
gateway-hop count and no-gateway loopback probes split off into an explicitly-labelled
dotted reference. Both latency rows are optional — the figure drops to whichever of the
two the run actually measured. Every quoted range in the notes/takeaway is computed from
the rows actually plotted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F2"
NAME = "f02_payload_scaling"
TITLE = "Payload-size scaling: throughput, blast latency & closed-loop RTT vs message size"
THESIS_TITLE = "Payload-size scaling: throughput vs message size"


def _series(grp, xcol, ycol, errcol=None):
    """Collapse to one point per x value (mean) so duplicate scenarios don't zig-zag."""
    g = grp.dropna(subset=[xcol, ycol])
    if g.empty:
        return np.array([]), np.array([]), np.array([])
    agg = {ycol: (ycol, "mean")}
    if errcol and errcol in g.columns:
        agg[errcol] = (errcol, "mean")
    out = g.groupby(xcol, observed=True).agg(**agg).reset_index().sort_values(xcol)
    err = out[errcol].values if (errcol and errcol in out.columns) else None
    return out[xcol].values, out[ycol].values, err


def _rtt_pools(df):
    """Split the run's honest closed-loop RTT rows into (gateway, loopback) pools.

    Only mode=='pingpong' rows are honest service latency — paced iface_*/lat_* rows also
    carry rtt percentiles but measure a rate-capped open-loop stream, and blending them
    would bend the routing curves. Loopback (n_gateways==0) ping-pong probes carry
    no gateway hop, so they are returned separately: their scenario names carry the same
    `_direct`/loopback chain token as the 1-gateway path, and chain-keyed styling would
    present a no-gateway baseline as the fastest gateway interface. When the
    canonical matrix_lat_* grid is present, the gateway pool is scoped to it — QoS-profile-
    tuned and zero-copy-variant ping-pongs measure differently-configured gateways and
    would fold into the same per-size mean. Either pool is None when empty.
    """
    if "rtt_us_p99" not in df.columns or not df["rtt_us_p99"].notna().any():
        return None, None
    r = df[df["rtt_us_p99"].notna() & df["message_bytes"].notna()].copy()
    if "mode" in r.columns:
        r = r[r["mode"].astype(str) == "pingpong"]
    # A clean size-scaling curve needs a fixed stream count (same rule as the blast rows).
    if "connections" in r.columns and (r["connections"] == 1).any():
        r = r[(r["connections"] == 1) | (r["connections"].isna())]
    if "n_gateways" in r.columns:
        n_gw = pd.to_numeric(r["n_gateways"], errors="coerce")
        gw, lo = r[n_gw >= 1], r[n_gw == 0]
    else:
        gw, lo = r, r.iloc[0:0]
    if "family" in gw.columns and (gw["family"].astype(str) == "matrix-latency").any():
        gw = gw[gw["family"].astype(str) == "matrix-latency"]
    return (gw if not gw.empty else None), (lo if not lo.empty else None)


def _hop_style(key) -> str:
    """Linestyle from the gateway-hop group key: 2 hops solid, 1 hop dashed — the same
    semantics as the chain legend (`n_gateways` int preferred; chain-string fallback)."""
    if isinstance(key, str):
        return "-" if key == "scg" else "--"
    return "-" if float(key) >= 2 else "--"


def _fmt_us_range(lo: float, hi: float) -> str:
    """'~7.7–326 µs' from the plotted extremes — the quoted band is never hardcoded."""
    if hi >= 1000:
        return f"~{T.fmt_latency_value(lo)}–{T.fmt_latency_value(hi)}"

    def one(v: float) -> str:
        return f"{v:.1f}" if v < 10 else f"{v:.0f}"

    a, b = one(lo), one(hi)
    return f"~{a} µs" if a == b else f"~{a}–{b} µs"


def make(bundle: RunBundle, saver: T.Saver) -> None:
    df = bundle.summary
    needed = {"transport", "protocol", "message_bytes", "throughput_gbps_mean"}
    if not needed.issubset(df.columns):
        saver.record_skip(FIG_ID, NAME, "needs transport/protocol/message_bytes/throughput")
        return

    d = df[df["message_bytes"].notna()].copy()
    # Restrict to the matrix family: it is the only sustained-blast payload sweep with an
    # explicit topology token. Without this, paced/pingpong/connrate/hot-reload/cipher/iface
    # rows fold into each (transport,protocol,chain,size) cell — the mid-size throughput dips
    # and >1 s latency spikes in the older render were pure blending artifacts.
    if "family" in d.columns and (d["family"].astype(str) == "matrix").any():
        d = d[d["family"].astype(str) == "matrix"]
    # A clean size-scaling curve needs a fixed stream count; use single-connection rows
    # (the canonical case) so 4c/16c variants don't stack vertically at the same size.
    if "connections" in d.columns and (d["connections"] == 1).any():
        d = d[(d["connections"] == 1) | d["connections"].isna()]
    transports = [t for t in ["shm", "shm-slot", "unix", "tcp", "udp"] if t in set(d["transport"].dropna().astype(str))]
    transports += [t for t in d["transport"].dropna().astype(str).unique() if t not in transports]
    if not transports:
        saver.record_skip(FIG_ID, NAME, "no transports present")
        return

    # ---- Print variant: the throughput row alone -----------------------------------------
    # The blast-p99 row is CO-uncorrected (banned from the print variant) and the honest RTT
    # story is carried by F16's print figure; what this figure owes the chapter is the
    # per-message-vs-per-byte crypto-cost claim, which lives entirely in the throughput row.
    if T.print_variant():
        _make_print(bundle, saver, d=d, transports=transports)
        return

    # Honest closed-loop RTT rows carry `rtt_us_p99` — the coordinated-omission-free service
    # latency. They are a DIFFERENT family than the sustained-blast matrix, so select from the
    # full summary (not the matrix-scoped `d`, which excludes family=="matrix-latency").
    # Gateway curves and no-gateway loopback references come back as separate pools so the
    # chain legend never claims a gateway hop that was not measured.
    d_rtt, d_rtt_lo = _rtt_pools(df)
    has_rtt = d_rtt is not None or d_rtt_lo is not None

    import matplotlib.pyplot as plt

    ncol = len(transports)
    has_lat = "latency_p99_us_mean" in d.columns and d["latency_p99_us_mean"].notna().any()
    # Rows, top→bottom: throughput, blast p99 (if measured), honest closed-loop RTT (if measured).
    row_t = 0
    row_l = 1 if has_lat else None
    row_r = (1 + int(has_lat)) if has_rtt else None
    nrow = 1 + int(has_lat) + int(has_rtt)
    bottom_row = row_r if row_r is not None else (row_l if row_l is not None else row_t)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), sharex=True, squeeze=False
    )

    seen_protos = set()
    drew_loopback = False
    rtt_min = rtt_max = None  # extremes of the *plotted* RTT points, for the notes/takeaway
    for c, transport in enumerate(transports):
        sub = d[d["transport"].astype(str) == transport]
        ax_t = axes[row_t][c]
        ax_l = axes[row_l][c] if row_l is not None else None
        ax_r = axes[row_r][c] if row_r is not None else None

        for proto, pg in sub.groupby("protocol", observed=True):
            proto = str(proto)
            color = T.protocol_color(proto)
            for chain, cg in pg.groupby("chain", observed=True) if "chain" in pg.columns else [("scg", pg)]:
                style = "-" if chain == "scg" else "--"
                x, y, ci = _series(cg, "message_bytes", "throughput_gbps_mean", "throughput_gbps_ci95")
                if len(x) == 0:
                    continue
                seen_protos.add(proto)
                if len(x) == 1:
                    # Single-size series: no band to fill, so show the throughput CI95 as a
                    # vertical whisker (lower arm clamped to 0 like the multi-point band below).
                    if ci is not None:
                        c0 = float(np.nan_to_num(ci)[0])
                        ax_t.errorbar(x, y, yerr=[[min(c0, float(y[0]))], [c0]], fmt="none",
                                      ecolor=color, elinewidth=1.0, capsize=3, alpha=0.6, zorder=2)
                    ax_t.scatter(x, y, color=color, marker=T.transport_marker(transport), s=42, zorder=3)
                else:
                    ax_t.plot(x, y, style, color=color, marker=".", zorder=3)
                    if ci is not None:
                        ci = np.nan_to_num(ci)
                        # Clamp the lower band to 0: degenerate CIs (ci95 >= mean on 2-3-rep
                        # single-conn rows) would otherwise paint an impossible negative-Gbps
                        # region and stretch the panel's y-axis below zero.
                        ax_t.fill_between(x, np.maximum(y - ci, 0.0), y + ci, color=color, alpha=0.15, zorder=1)
                if ax_l is not None:
                    lx, ly, _ = _series(cg, "message_bytes", "latency_p99_us_mean")
                    if len(lx) == 1:
                        ax_l.scatter(lx, ly, color=color, marker=T.transport_marker(transport), s=42, zorder=3)
                    elif len(lx) > 1:
                        ax_l.plot(lx, ly, style, color=color, marker=".", zorder=3)

        # Honest closed-loop RTT panel — its own proto loop because the ping-pong family's
        # protocol set differs from the blast `sub` (and its rows live in `d_rtt`, not `d`).
        # Linestyle is keyed to the measured gateway-hop count, NOT the raw chain token:
        # loopback probes are `_direct`-named too and must never wear the 1-gateway style.
        rtt_has_data = False
        if ax_r is not None:
            if d_rtt is not None:
                sub_r = d_rtt[d_rtt["transport"].astype(str) == transport]
                for proto, pg in sub_r.groupby("protocol", observed=True):
                    proto = str(proto)
                    color = T.protocol_color(proto)
                    gcol = "n_gateways" if "n_gateways" in pg.columns else (
                        "chain" if "chain" in pg.columns else None)
                    for hops, cg in (pg.groupby(gcol, observed=True) if gcol else [(1, pg)]):
                        rx, ry, _ = _series(cg, "message_bytes", "rtt_us_p99")
                        if len(rx) == 0:
                            continue
                        rtt_has_data = True
                        seen_protos.add(proto)
                        rtt_min = float(ry.min()) if rtt_min is None else min(rtt_min, float(ry.min()))
                        rtt_max = float(ry.max()) if rtt_max is None else max(rtt_max, float(ry.max()))
                        if len(rx) == 1:
                            ax_r.scatter(rx, ry, color=color, marker=T.transport_marker(transport), s=42, zorder=3)
                        else:
                            ax_r.plot(rx, ry, _hop_style(hops), color=color, marker=".", zorder=3)
            if d_rtt_lo is not None:
                # No-gateway loopback ping-pong reference: dotted, its own legend entry — a
                # baseline, never comparable to the gateway curves' service latency.
                sub_lo = d_rtt_lo[d_rtt_lo["transport"].astype(str) == transport]
                for proto, pg in sub_lo.groupby("protocol", observed=True):
                    proto = str(proto)
                    rx, ry, _ = _series(pg, "message_bytes", "rtt_us_p99")
                    if len(rx) == 0:
                        continue
                    rtt_has_data = True
                    drew_loopback = True
                    seen_protos.add(proto)
                    rtt_min = float(ry.min()) if rtt_min is None else min(rtt_min, float(ry.min()))
                    rtt_max = float(ry.max()) if rtt_max is None else max(rtt_max, float(ry.max()))
                    ax_r.plot(rx, ry, ":", color=T.protocol_color(proto), marker=".",
                              alpha=0.85, zorder=2)

        n_protos = sub["protocol"].nunique()
        T.panel_title(ax_t, f"{transport_label(transport)}  ({n_protos} proto)")
        T.byte_axis(ax_t, "x")
        if c == 0:
            ax_t.set_ylabel("throughput (Gbps)")
        if ax_l is not None:
            T.byte_axis(ax_l, "x")
            ax_l.set_yscale("log")
            ax_l.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
            # Name the blast row honestly: this is open-loop blast p99 (queue depth under
            # saturation, ms-scale at mid sizes), not service time — the honest RTT is below.
            # Once per row (first column): at panel-title size the repeated per-column
            # subtitle overruns into its neighbours; the c==0 ylabel repeats the wording.
            if c == 0:
                T.panel_title(ax_l, "p99 under saturating blast (queueing-dominated)")
                ax_l.set_ylabel("p99 under saturating blast\n(queueing-dominated, µs · log)")
        if ax_r is not None:
            # The honest number: coordinated-omission-free closed-loop ping-pong RTT. It spans
            # far less than the blast row's µs-to-hundreds-of-ms sweep (tens to a few hundred
            # µs), so a LINEAR µs axis reads cleaner than the blast row's log axis (a narrow
            # log range degrades to scientific notation). Columns with no ping-pong data yet
            # (fills after a matrix_lat_* run) get a labelled placeholder, not an empty axis.
            if rtt_has_data:
                T.byte_axis(ax_r, "x")
                ax_r.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
            else:
                ax_r.set_yticks([])
                ax_r.text(0.5, 0.5, "no closed-loop\nping-pong data\nin this run",
                          transform=ax_r.transAxes, ha="center", va="center",
                          fontsize=T.FS["annot"], color=T.GREYS["muted"], style="italic")
            if c == 0:
                T.panel_title(ax_r, "closed-loop ping-pong RTT (per-message service latency)")
                ax_r.set_ylabel("closed-loop RTT\n(service latency, µs)")
        # The bottom-most present row owns the shared x-axis label.
        axes[bottom_row][c].set_xlabel("message size (bytes, log)")

    # Shared protocol legend + chain linestyle key.
    proto_handles = [
        plt.matplotlib.patches.Patch(facecolor=T.protocol_color(p), label=protocol_label(p))
        for p in T.PROTOCOL_COLORS if p in seen_protos
    ]
    chain_handles = [
        plt.matplotlib.lines.Line2D([0], [0], color=T.GREYS["annot"], ls="-", label="2 gateways (scg-scg)"),
        plt.matplotlib.lines.Line2D([0], [0], color=T.GREYS["annot"], ls="--", label="1 gateway (scg-direct)"),
    ]
    if drew_loopback:
        chain_handles.append(plt.matplotlib.lines.Line2D(
            [0], [0], color=T.GREYS["annot"], ls=":", label="no gateway (loopback ref.)"))
    chain_handles.append(plt.matplotlib.patches.Patch(
        facecolor=T.GREYS["faint"], alpha=0.5, edgecolor="none", label="95% CI band"))
    # One legend_right call: two "outside right upper" legends overprint each other
    # (constrained layout anchors both to the same corner), so the protocol colours and
    # the path/CI key share a single sanctioned right-edge column.
    T.legend_right(fig, proto_handles + chain_handles)

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.02)
    # Footer method notes: scope + topology key, then the two latency-provenance lines. They stack
    # upward from the bottom, each on a distinct y so the chrome-on render doesn't overprint them
    # (the taller 3-row figure gives the extra room). Under --no-chrome each is recorded to
    # captions.txt (all of them — the writer no longer collapses same-kind notes to the last one).
    # Scope note: state which transports carry the 2-gateway path from the plotted slice —
    # a hardcoded restriction goes stale the moment the matrix grows a topology.
    scope = ("scope: matrix family, single-connection rows; solid = 2 gateways (scg-scg), "
             "dashed = 1 gateway (scg-direct)")
    scg_transports = [
        t for t in transports
        if "chain" in d.columns
        and t in set(d.loc[d["chain"].astype(str) == "scg", "transport"].dropna().astype(str))
    ]
    if not scg_transports:
        scope += " — no 2-gateway rows in this run"
    elif len(scg_transports) < len(transports):
        scope += (" — the 2-gateway path exists only for "
                  + "/".join(transport_label(t) for t in scg_transports))
    notes = [scope]
    if has_lat:
        notes.append(T.BLAST_LATENCY_NOTE)
    if has_rtt:
        # Quote the plotted band and the actual coverage (transports × size span) — both
        # computed from the rows drawn, never hardcoded.
        cov = d_rtt if d_rtt is not None else d_rtt_lo
        cov_transports = [t for t in transports
                          if t in set(cov["transport"].dropna().astype(str))]
        sizes = cov["message_bytes"].dropna()
        b_lo, b_hi = T.fmt_bytes(float(sizes.min())), T.fmt_bytes(float(sizes.max()))
        size_span = b_lo if b_lo == b_hi else f"{b_lo}–{b_hi}"
        cover = (f"{'/'.join(transport_label(t) for t in cov_transports)} at {size_span} "
                 "in this run" if cov_transports else "no overlap with the plotted transports")
        band = f", {_fmt_us_range(rtt_min, rtt_max)}" if rtt_min is not None else ""
        note = (
            f"bottom row = closed-loop ping-pong RTT (coordinated-omission-free{band}) — the "
            f"coordinated-omission-safe service latency, drawn only where ping-pong data exists ({cover})"
        )
        if drew_loopback:
            note += "; dotted = no-gateway loopback reference (no gateway hop measured)"
        notes.append(note)
    y0 = 0.036 + 0.015 * max(0, len(notes) - 2)
    for i, note in enumerate(notes):
        T.add_method_note(fig, note, y=y0 - 0.015 * i)
    blast_max = d["latency_p99_us_mean"].max() if has_lat else None
    blast_txt = (f" (up to {T.fmt_latency_value(float(blast_max))})"
                 if blast_max is not None and np.isfinite(blast_max) else "")
    if has_lat and has_rtt:
        # The two latency rows side by side: blast queueing vs honest service RTT at each size.
        rtt_txt = f" ({_fmt_us_range(rtt_min, rtt_max)})" if rtt_min is not None else ""
        T.add_takeaway(
            fig,
            f"Middle-row p99 is saturating-blast queueing{blast_txt}; the bottom row is the "
            f"closed-loop RTT{rtt_txt} at the same payload sizes — their gap is the "
            "coordinated-omission inflation.",
        )
    elif has_lat:
        # Bottom-row p99 is saturation queueing, not service time — point at F16's honest
        # closed-loop number instead of quoting one this run did not measure.
        T.add_takeaway(
            fig,
            f"Bottom-row p99 is saturating-blast queueing{blast_txt}, not service time — "
            "see F16 for the closed-loop RTT.",
        )
    T.add_provenance(fig, bundle.caption())
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)


# Representative protocol set for the print variant: one rung per protection family. The
# omitted variants (TLS 1.2 vs 1.3 flavours, kmTLS, DTLS 1.0, ALE framings) track their
# family's curve; the full-variant render draws every one.
_THESIS_PROTOS = ["none", "tls/1.3", "ktls/1.3", "tls/1.3+mtls", "tls/1.2+integrity", "dtls/1.2"]
_THESIS_TRANSPORTS = ["shm", "shm-slot", "unix", "tcp", "udp"]  # IPC (shm byte-stream+slot, uds) / stream / datagram


def _make_print(bundle: RunBundle, saver: T.Saver, *, d: pd.DataFrame,
                 transports: list) -> None:
    """Print variant: throughput vs payload size, 3 transports × 6 protocol families."""
    import matplotlib.pyplot as plt

    cols = [t for t in _THESIS_TRANSPORTS if t in transports] or transports[:3]
    omitted_t = [t for t in transports if t not in cols]
    dd = d[d["protocol"].astype(str).isin(_THESIS_PROTOS)].copy()
    if dd.empty:
        dd = d.copy()

    fig, axes = plt.subplots(1, len(cols), figsize=(2.55 * len(cols), 3.3),
                             sharex=True, squeeze=False)
    seen_protos: set[str] = set()
    routing_ratios: list[float] = []   # per stream transport: routing max/min scaling
    udp_routing_ratio = None
    enc_ratios: list[float] = []       # encrypted rows: scaling above 256 B

    for c, transport in enumerate(cols):
        ax = axes[0][c]
        sub = dd[dd["transport"].astype(str) == transport]
        for proto, pg in sub.groupby("protocol", observed=True):
            proto = str(proto)
            color = T.protocol_color(proto)
            for chain, cg in pg.groupby("chain", observed=True) if "chain" in pg.columns else [("scg", pg)]:
                style = "-" if chain == "scg" else "--"
                x, y, ci = _series(cg, "message_bytes", "throughput_gbps_mean", "throughput_gbps_ci95")
                if len(x) == 0:
                    continue
                seen_protos.add(proto)
                if len(x) == 1:
                    ax.scatter(x, y, color=color, marker=T.transport_marker(transport), s=42, zorder=3)
                else:
                    ax.plot(x, y, style, color=color, marker=".", zorder=3)
                    if ci is not None:
                        ci = np.nan_to_num(ci)
                        ax.fill_between(x, np.maximum(y - ci, 0.0), y + ci, color=color,
                                        alpha=0.15, zorder=1)
                # Scaling factors for the takeaway, computed from what is drawn.
                if len(x) > 1 and float(y.min()) > 0:
                    if proto == "none":
                        ratio = float(y.max()) / float(y.min())
                        if transport == "udp":
                            udp_routing_ratio = max(udp_routing_ratio or 0.0, ratio)
                        else:
                            routing_ratios.append(ratio)
                    elif transport != "udp":
                        # Stream transports only: datagram (UDP) crypto rows are message-size-
                        # bound like their transport, so they scale with size by construction.
                        big = [(xv, yv) for xv, yv in zip(x, y) if xv >= 256]
                        ys = [yv for _, yv in big]
                        if len(ys) > 1 and min(ys) > 0:
                            enc_ratios.append(max(ys) / min(ys))
        T.panel_title(ax, transport_label(transport))
        T.byte_axis(ax, "x")
        ax.set_xlabel("message size (bytes, log)")
        if c == 0:
            ax.set_ylabel("throughput (Gbps)")

    proto_handles = [
        plt.matplotlib.patches.Patch(facecolor=T.protocol_color(p), label=protocol_label(p))
        for p in _THESIS_PROTOS if p in seen_protos
    ]
    chain_handles = [
        plt.matplotlib.lines.Line2D([0], [0], color=T.GREYS["annot"], ls="-", label="2 gateways"),
        plt.matplotlib.lines.Line2D([0], [0], color=T.GREYS["annot"], ls="--", label="1 gateway"),
        plt.matplotlib.patches.Patch(
            facecolor=T.GREYS["faint"], alpha=0.5, edgecolor="none", label="95% CI band"),
    ]
    T.legend_right(fig, proto_handles + chain_handles)

    take = ""
    if routing_ratios and enc_ratios:
        r_lo, r_hi = min(routing_ratios), max(routing_ratios)
        udp_txt = (f" and {udp_routing_ratio:.0f}× on UDP"
                   if udp_routing_ratio and udp_routing_ratio > r_hi else "")
        take = (f"Plaintext routing throughput scales ~{r_lo:.1f}–{r_hi:.1f}× with payload "
                f"size on stream transports{udp_txt}, while encrypted stream paths stay "
                f"within {max(enc_ratios):.2f}× above 256 B — the per-core AEAD cost is "
                f"per-message, not per-byte (datagram crypto rows are size-bound like their "
                f"transport).")
    T.set_headline(fig, f"{THESIS_TITLE}  ·  {bundle.label}", y=1.02)
    if take:
        T.add_takeaway(fig, take)
    scope = ("scope: matrix family, single-connection sustained blast; solid = 2 gateways "
             "(scg-scg), dashed = 1 gateway (scg-direct); one representative protocol per "
             "protection family")
    if omitted_t:
        scope += (" · transports not shown: "
                  + "/".join(transport_label(t) for t in omitted_t)
                  + " (they track TCP)")
    T.add_method_note(fig, scope)
    T.add_provenance(fig, bundle.caption())
    saver.save(fig, NAME, fig_id=FIG_ID, title=THESIS_TITLE)
