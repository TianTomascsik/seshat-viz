"""
F1 — Throughput vs latency Pareto trade-off map.

The single "landscape" figure: every measured configuration placed by its p99 latency
(x, log) and throughput (y), colored by protocol and shaped by transport, sized by CPU
efficiency. The Pareto frontier (max throughput / min latency) is drawn so the dominant
configurations — and the price of climbing the security ladder — are immediate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F1"
NAME = "f01_landscape_throughput_latency"
TITLE = "Throughput–latency landscape (single-connection slice)"

_X = "latency_p99_us_mean"
_Y = "throughput_gbps_mean"


def _callout_pool(d: pd.DataFrame) -> pd.DataFrame:
    """
    Rows eligible to headline the extreme callouts: trusted (`_hstate == "trusted"`) AND
    with an SCG actually in the path (n_gateways >= 1). A harness-limited point is a
    load-generator floor and a zero-gateway loopback row is not a gateway config, so
    neither may headline "best" — loopback baselines plot but never crown.
    Degrades gracefully: trusted-any if no trusted gateway row exists (e.g. a
    loopback-only baseline run), then the full slice if nothing at all is trusted.
    """
    trust = d[d["_hstate"] == "trusted"]
    if "n_gateways" in d.columns and not trust.empty:
        gw = trust[pd.to_numeric(trust["n_gateways"], errors="coerce") >= 1]
        if not gw.empty:
            return gw
    return trust if not trust.empty else d


def make(bundle: RunBundle, saver: T.Saver) -> None:
    df = bundle.summary
    if not bundle.has(_X, _Y):
        saver.record_skip(FIG_ID, NAME, f"needs {_X} & {_Y}")
        return

    # Restrict to sustained-blast throughput scenarios BEFORE anything else: paced (lat_/
    # iface-latency), saturation, hot-reload-disrupted, connection-rate and closed-loop
    # ping-pong rows are NOT throughput-vs-blast-latency measurements and must not share this
    # axis pair (their "throughput" is a pacing rate or 0, and pp latency is closed-loop RTT).
    d = derive.throughput_scenarios_only(df)
    d = d[np.isfinite(d[_X]) & np.isfinite(d[_Y]) & (d[_X] > 0) & (d[_Y] > 0)].copy()
    if d.empty:
        saver.record_skip(FIG_ID, NAME, "no sustained-blast rows with finite latency & throughput")
        return

    # Restrict to a matched connection count so the cross-transport comparison is fair:
    # TCP multi-connection aggregate throughput would otherwise dwarf single-connection-only
    # SHM/UDS/UDP, making "TCP TLS beats SHM/UDS" a coverage artifact rather than transport
    # merit. connections==1 is the shared anchor every transport measures; concurrency scaling
    # lives in F15. NaN-connection rows (no per-conn confound) are kept.
    if "connections" in d.columns:
        d = d[d["connections"].isin([1]) | d["connections"].isna()].copy()
        if d.empty:
            saver.record_skip(FIG_ID, NAME, "no single-connection rows to compare fairly")
            return

    # Keep ALL payload sizes so every protocol/transport is represented — pinning one size would
    # structurally drop the datagram-only families (DTLS, TLS-over-UDP, ALE) that exist only at
    # small payloads. To keep payload from being a *hidden* confound (a 64 B and a 64 KB row of one
    # config sit far apart), payload size is instead made an explicit visual channel: marker area
    # ∝ log2(payload bytes). Compare within a marker-size band; payload scaling itself is F2.

    import matplotlib.pyplot as plt

    # Three columns: the full landscape (left), a zoom on the encrypted band (middle), and a
    # dedicated legend panel (right). A separate zoom panel — rather than an overlay inset —
    # guarantees it never covers main-axes points; a dedicated legend axes lets all four legends
    # share one left edge (fig.legend + constrained_layout does not align them reliably).
    fig, (ax, axz, axl) = plt.subplots(
        1, 3, figsize=(14.0, 5.6), gridspec_kw={"width_ratios": [2.3, 0.95, 1.35]}
    )
    axl.axis("off")
    axl.set_xlim(0, 1)  # stable coordinate frame for the left-anchored legends

    # Marker area ∝ payload size (log2 bytes): small payloads → small markers, bulk → large. This
    # is what lets a reader tell a point's payload apart (the whole size-confound concern) while
    # every size/protocol stays on the plot. A payload-size key is drawn below.
    _size_lb_min = _size_lb_span = None
    if "message_bytes" in d.columns and d["message_bytes"].notna().any():
        lb = np.log2(d["message_bytes"].astype(float).clip(lower=1.0))
        _size_lb_min = float(lb.min())
        _size_lb_span = (float(lb.max()) - _size_lb_min) or 1.0
        sizes = 22.0 + 190.0 * (lb - _size_lb_min) / _size_lb_span
    else:
        sizes = np.full(len(d), 70.0)
    d = d.assign(_size=np.asarray(sizes))

    def _size_area(payload_bytes: float) -> float:
        """Marker area for a payload size, matching the encoding above (for the size legend)."""
        if _size_lb_min is None:
            return 70.0
        return 22.0 + 190.0 * (np.log2(max(payload_bytes, 1.0)) - _size_lb_min) / _size_lb_span

    # Three-state credibility encoding. harness_limited is True / False / NA (unassessed —
    # e.g. rows without an SCG in the path). Mapping NA to False silently certified unassessed
    # points as "gateway-bound (trustworthy)", so distinguish them: bold dark ring = trusted
    # gateway-bound, faded = harness-limited, neutral thin grey ring = not assessed.
    def _hstate(hv) -> str:
        if pd.isna(hv):
            return "na"
        return "limited" if bool(hv) else "trusted"

    d = d.assign(_hstate=[_hstate(v) for v in d.get("harness_limited", pd.Series([pd.NA] * len(d)))])

    # Display protocol: distinguish the ETCS ALEPKT framing over UDP-TLS from the raw UDP-over-TLS
    # baseline. Both carry the same base protocol (tls/1.2 or tls/1.3) and UDP marker, so without
    # this they render identically; the `app_framing` factor (loader) is the only thing that
    # separates them. Keep the base version (`{proto}+ale`) so a TLS-1.2 ALE tunnel is not
    # mislabelled as 1.3.
    framing = d.get("app_framing", pd.Series([pd.NA] * len(d), index=d.index))
    d = d.assign(_proto=[
        f"{p}+ale" if str(f) == "ale" else str(p)
        for p, f in zip(d["protocol"].astype(str), framing)
    ])

    style = {
        "trusted": dict(edge=T.GREYS["ink"], lw=1.4, alpha=0.95, z=4),
        "limited": dict(edge="#FFFFFF", lw=0.4, alpha=0.4, z=3),
        "na": dict(edge=T.GREYS["muted"], lw=0.8, alpha=0.7, z=3),
    }
    def _scatter(target) -> None:
        """Draw the credibility-styled point cloud onto `target` (main axes or the inset)."""
        for _, r in d.iterrows():
            st = style[r["_hstate"]]
            # ALE and the raw UDP-over-TLS baseline perform near-identically (same crypto +
            # transport, only the framing differs), so their points nearly coincide. Draw ALE on
            # top at full opacity so its distinct color is never occluded by the raw point.
            is_ale = str(r["_proto"]).endswith("+ale")
            target.scatter(
                r[_X], r[_Y],
                s=r["_size"],
                c=T.protocol_color(str(r["_proto"])),
                marker=T.transport_marker(str(r["transport"])),
                edgecolors=st["edge"], linewidths=st["lw"],
                alpha=max(st["alpha"], 0.9) if is_ale else st["alpha"],
                zorder=st["z"] + (6 if is_ale else 0),
            )

    _scatter(ax)

    # CI95 whiskers (thin, behind markers). Clamp the lower ends to the physical domain:
    # degenerate n=2–3 CIs (ci95 >= mean, common on single-conn UDP blast rows) would
    # otherwise draw whiskers below 0 Gbps and, on the log-x axis, span the whole plot.
    xerr = pd.to_numeric(d.get("latency_p99_us_ci95"), errors="coerce") if "latency_p99_us_ci95" in d.columns else None
    yerr = pd.to_numeric(d.get("throughput_gbps_ci95"), errors="coerce") if "throughput_gbps_ci95" in d.columns else None
    if xerr is not None or yerr is not None:
        eps = 1e-9
        def _clamp(vals, err):
            if err is None:
                return None
            lo = np.minimum(err.to_numpy(), np.maximum(vals.to_numpy() - eps, 0.0))
            hi = err.to_numpy()
            return np.vstack([lo, hi])
        ax.errorbar(
            d[_X], d[_Y],
            xerr=_clamp(d[_X], xerr), yerr=_clamp(d[_Y], yerr),
            fmt="none", ecolor=T.GREYS["faint"], elinewidth=0.6, capsize=0, alpha=0.5, zorder=1,
        )

    # Annotate the extremes over the trusted, gateway-in-path pool only (see _callout_pool).
    ann = _callout_pool(d)
    i_tput, i_lat = ann[_Y].idxmax(), ann[_X].idxmin()
    # Annotate the max-throughput and min-latency extremes. They are usually the SAME transport/
    # protocol (e.g. TCP routing) at DIFFERENT payload sizes, so (a) the size is in each label and
    # (b) one label goes UP into the headroom and the other DOWN, so they never overlap even when
    # the two points sit close together at the top. Skip the second when both are the same row.
    callouts = [(ann.loc[i_tput], (0, 16), "bottom")]
    if i_lat != i_tput:
        callouts.append((ann.loc[i_lat], (0, -22), "top"))
    for r, dxy, va in callouts:
        size_tag = ""
        if "message_bytes" in r and pd.notna(r["message_bytes"]):
            size_tag = f" · {T.fmt_bytes(float(r['message_bytes']))}B"
        # Callout text sits ABOVE the markers (they reach zorder 10) with a translucent
        # backing: the gateway-in-path crown can land inside the dense cloud (not at a
        # visual extreme), where default-zorder text would be occluded by neighbours.
        ax.annotate(
            f"{transport_label(str(r['transport']))} · {protocol_label(str(r['_proto']))}\n"
            f"{r[_Y]:.2g} Gbps @ {T.fmt_latency_value(r[_X])}{size_tag}",
            xy=(r[_X], r[_Y]), xytext=dxy, textcoords="offset points",
            fontsize=T.FS["annot"], color=T.GREYS["ink"], ha="center", va=va, zorder=20,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65),
            arrowprops=dict(arrowstyle="-", color=T.GREYS["muted"], lw=0.6),
        )

    T.us_axis(ax, which="x", log=True)
    # Log throughput axis: the datagram families (DTLS, TLS-over-UDP, ALE) are lossy and low
    # throughput (0.05–9 Gbps), so on a linear 0–50 Gbps axis they cram invisibly against the
    # baseline under the routing/crypto points. Log spreads the ~3-decade range so every protocol
    # — ALE included — is visible and separable.
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("p99 latency (µs, log) — lower is better →")
    ax.set_ylabel("throughput (Gbps, log) — higher is better ↑")
    T.set_headline(fig, f"{TITLE} · all payloads (marker area = payload size)\n{bundle.label}", ax=ax)
    # Clamp both log axes to the real data — CI whiskers (whose lower ends touch ~0 on wide-CI
    # blast rows) would otherwise autoscale the log axes to absurd bounds (sub-µs x, ~1e-8 Gbps y).
    # Larger latency on the left so lower-latency reads toward the right.
    xv = d[_X][d[_X] > 0]
    ax.set_xlim(float(xv.max()) * 2.0, max(float(xv.min()) / 2.0, 1.0))
    yv = d[_Y][d[_Y] > 0]
    ax.set_ylim(float(yv.min()) / 1.6, float(yv.max()) * 2.2)  # top headroom for the up-callout

    # Right panel — zoom on the encrypted (non-routing) cluster. The routing outlier (e.g. TCP
    # routing ~48 Gbps) compresses the TLS/kTLS/mTLS band into a thin ribbon on the main axes, so
    # the cost of climbing the security ladder is unreadable there (Questions.md, F1 task). This
    # replays the same credibility-styled point cloud, y-limited to the encrypted band and keeping
    # the inverted log-x, in a panel of its own so it cannot overlap any main-axes point.
    enc = d[d["protocol"].astype(str) != "none"]
    ex = enc[_X][enc[_X] > 0]
    ey = enc[_Y][enc[_Y] > 0]
    if len(ex) >= 2 and len(ey) >= 2:
        _scatter(axz)
        axz.set_xscale("log")
        axz.set_xlim(float(ex.max()) * 1.35, max(float(ex.min()) / 1.35, 1.0))  # inverted, like main
        axz.set_ylim(max(float(ey.min()) * 0.85, 0.0), float(ey.max()) * 1.12)
        axz.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
        axz.tick_params(labelsize=T.FS["annot"])
        T.panel_title(axz, "encrypted configs (zoom)")
        axz.set_xlabel("p99 latency (µs, log)", fontsize=T.FS["small"])
        axz.grid(True, lw=0.4, color=T.GREYS["faint"])
    else:
        axz.set_axis_off()

    # Legends — all placed inside the dedicated `axl` axes, each anchored to the SAME left edge
    # (axes x=0) with the title left-aligned, so the four stack in one clean column. Every legend
    # uses the same base glyph (a size-8 "o"); each varies only the channel it encodes:
    # protocol = fill color, transport = marker shape, payload = marker size, credibility = edge.
    L = plt.matplotlib.lines.Line2D
    protos = [p for p in T.PROTOCOL_ORDER if p in set(d["_proto"].dropna().astype(str))]
    transports = [t for t in T.TRANSPORT_ORDER if t in set(d["transport"].dropna().astype(str))]

    def _place(handles, title, y, **kw):
        leg = axl.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=(0.0, y),
                         alignment="left", borderaxespad=0.0, handletextpad=0.8,
                         fontsize=T.FS["annot"], title_fontsize=T.FS["small"], **kw)
        leg._legend_box.align = "left"
        axl.add_artist(leg)
        return leg

    _place([L([0], [0], marker="o", color="none", markerfacecolor=T.protocol_color(p),
              markeredgecolor="none", markersize=8, label=protocol_label(p)) for p in protos],
           "protocol", 1.0, ncol=2, columnspacing=1.0)
    _place([L([0], [0], marker=T.transport_marker(t), color="none", markerfacecolor=T.GREYS["muted"],
              markeredgecolor=T.GREYS["edge"], markersize=8, label=transport_label(t)) for t in transports],
           "transport", 0.63)

    # Payload-size key (marker area ∝ log2 bytes) — CONTINUOUS over every measured size; the key
    # shows a few evenly log-spaced reference markers so the gradient is legible.
    if _size_lb_min is not None:
        present = sorted({int(s) for s in d["message_bytes"].dropna().astype(int)})
        if present:
            n = len(present)
            picks = [present[i] for i in sorted({int(round(k * (n - 1) / 3)) for k in range(4)})]
            _place([L([0], [0], marker="o", color="none", markerfacecolor=T.GREYS["faint"],
                      markeredgecolor=T.GREYS["annot"], markersize=0.58 * float(np.sqrt(_size_area(float(s)))),
                      label=f"{T.fmt_bytes(float(s))}B") for s in picks],
                   "payload (marker area, continuous)", 0.35, labelspacing=0.9)

    # Credibility encoding legend — explains the point transparency (opaque/dark-ring = gateway-
    # bound; faded = harness-limited). Shown in both modes; only the states that actually occur are
    # listed. The faded "harness-limited" swatch itself carries the transparency it describes.
    # Swatches use a solid light-grey fill for the "faded" state (rather than alpha on the legend
    # handle, which washes out the swatch and text); the plotted points keep the real transparency.
    present_states = set(d["_hstate"])
    cred_specs = [  # state, facecolor, edgecolor, edgewidth, label
        ("trusted", T.GREYS["muted"], T.GREYS["ink"], 1.4, "gateway-bound"),
        ("na", T.GREYS["faint"], T.GREYS["muted"], 0.8, "not assessed (no gateway / reload)"),
        ("limited", T.GREYS["faint"], T.GREYS["muted"], 0.8, "harness-limited (faded)"),
    ]
    cred_handles = [
        L([0], [0], marker="o", color="none", markerfacecolor=fc, markeredgecolor=ec,
          markeredgewidth=ew, markersize=8, label=lab)
        for st, fc, ec, ew, lab in cred_specs if st in present_states
    ]
    if cred_handles:
        _place(cred_handles, "credibility", 0.11)

    T.add_method_note(
        fig,
        # Document the operating point + the size encoding on the figure (and via captions.txt in
        # the LaTeX caption), so a reader knows every payload is present and how to read it.
        "operating point: 1 connection, ALL payload sizes shown — marker area scales with log2(payload "
        "bytes), so compare configs within a marker-size band (payload scaling itself is F2). "
        "Datagram-only families (DTLS, TLS-over-UDP, ALE) exist only at small payloads, hence "
        "small markers. Concurrency scaling is F15; right panel zooms the encrypted band the "
        "routing outlier compresses. Callouts crown gateway-in-path (≥1 SCG), non-harness-limited "
        "extremes — loopback baselines never headline · " + T.BLAST_LATENCY_NOTE
        # The pathological SHM point (shm 64B p99≈36ms) is the harness receive-poll stall, not a
        # transport limit — flag it so no reader reads SHM as high-latency-capable here.
        + " · " + T.SHM_STALL_NOTE,
    )
    T.add_provenance(fig, bundle.caption() + "  ·  marker area scales with payload bytes (log2)")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
