"""
F25 — SHM ring-variant comparison: byte-stream + eventfd vs fixed-slot (Vyukov) + futex.

The gateway offers two SHM ring implementations for the same local interface: the default
variable-length byte-stream ring with eventfd wakeups, and the fixed-slot Vyukov ring with
futex wakeups (`shm_ring_kind=slot`, geometry auto-sized from the message size). The faceted
figures (F2/F3/F7/F15/F16) draw them as *separate* transports — deliberately, since their
encoding channels are saturated — so this figure is the one place the two rings meet in a
single axes pair and the trade-off is read directly:

  1. **Throughput vs payload** — the slot ring loses small payloads (fixed per-segment cost)
     and wins large ones (in-place slots avoid the byte-stream ring's wrap/coalesce copies).
  2. **Closed-loop RTT vs payload** — the honest ping-pong service time of each ring
     (open-loop blast latency is queueing-dominated and deliberately not drawn, see F16).

Encoding: color+marker = ring variant (the transport identity from the theme: green ● for
byte-stream, amber ✕ for slot), linestyle = protection profile (a reduced representative set,
so 2 rings × 3 profiles = 6 lines per panel stay print-readable). Single connection,
single-gateway path — the ring, not concurrency, is the variable under test.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F25"
NAME = "f25_shm_ring_compare"
TITLE = "SHM ring variants: byte-stream + eventfd vs fixed-slot + futex"

_RINGS = ["shm", "shm-slot"]

# Representative protection profiles (plaintext / userspace TLS / kernel TLS), each with a
# fixed linestyle so the ring color+marker stays the only ring-identity channel.
_PROFILES = [("none", "-"), ("tls/1.3", "--"), ("ktls/1.3", ":")]


def _series(grp, xcol, ycol):
    """Collapse to one point per x value (mean) so duplicate scenarios don't zig-zag."""
    g = grp.dropna(subset=[xcol, ycol])
    if g.empty:
        return np.array([]), np.array([])
    out = g.groupby(xcol, observed=True)[ycol].mean().reset_index().sort_values(xcol)
    return out[xcol].values, out[ycol].values


def make(bundle: RunBundle, saver: T.Saver) -> None:
    df = bundle.summary
    if df is None or df.empty or "transport" not in df.columns:
        saver.record_skip(FIG_ID, NAME, "no summary rows")
        return

    d = df[df["transport"].astype(str).isin(_RINGS)].copy()
    # Matrix families only (throughput rows are `matrix`, ping-pong RTT rows are
    # `matrix-latency`), single connection: the ring is the variable, nothing else.
    if "family" in d.columns and (d["family"].astype(str) == "matrix").any():
        d = d[d["family"].astype(str).isin(["matrix", "matrix-latency"])]
    if "connections" in d.columns:
        d = d[(d["connections"] == 1) | d["connections"].isna()]
    if d.empty or set(d["transport"].astype(str)) != set(_RINGS):
        saver.record_skip(FIG_ID, NAME, "need both SHM ring variants in the run")
        return

    # Throughput: prefer the 1-gateway (direct) chain — the cleanest single-hop reading.
    tput = d[d["mode"].astype(str) != "pingpong"] if "mode" in d.columns else d
    if "chain" in tput.columns and (tput["chain"].astype(str) == "direct").any():
        tput = tput[tput["chain"].astype(str) == "direct"]
    # RTT: the closed-loop ping-pong rows (matrix_lat_*), rtt_us_p50.
    rtt = d[d["mode"].astype(str) == "pingpong"] if "mode" in d.columns else d.iloc[0:0]
    has_rtt = not rtt.empty and "rtt_us_p50" in rtt.columns and rtt["rtt_us_p50"].notna().any()

    ncols = 2 if has_rtt else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.5), squeeze=False)
    ax_t = axes[0][0]
    stats: dict[str, dict[str, float]] = {r: {} for r in _RINGS}

    for ring in _RINGS:
        color, marker = T.transport_color(ring), T.transport_marker(ring)
        rg_t = tput[tput["transport"].astype(str) == ring]
        for proto, style in _PROFILES:
            pg = rg_t[rg_t["protocol"].astype(str) == proto]
            x, y = _series(pg, "message_bytes", "throughput_gbps_mean")
            if len(x) == 0:
                continue
            ax_t.plot(x, y, style, color=color, marker=marker, ms=5, lw=1.7,
                      label=f"{transport_label(ring)} · {protocol_label(proto)}")
            if proto == "none":
                stats[ring]["tput_max"] = float(y[np.argmax(x)])
                stats[ring]["size_max"] = float(x.max())
    T.byte_axis(ax_t, "x")
    ax_t.set_xlabel("message size (bytes, log)")
    ax_t.set_ylabel("throughput (Gbps)")
    T.panel_title(ax_t, "Throughput — 1 connection, 1 gateway")
    ax_t.grid(True)
    # Long keylines: the dashed/dotted protocol linestyles must be readable in the legend
    # (at the default handle length a dash pattern collapses to one solid-looking segment).
    T.legend_inline(ax_t, loc="best", ncol=1, handlelength=3.4)

    if has_rtt:
        ax_r = axes[0][1]
        for ring in _RINGS:
            color, marker = T.transport_color(ring), T.transport_marker(ring)
            rg_r = rtt[rtt["transport"].astype(str) == ring]
            for proto, style in _PROFILES:
                pg = rg_r[rg_r["protocol"].astype(str) == proto]
                x, y = _series(pg, "message_bytes", "rtt_us_p50")
                if len(x) == 0:
                    continue
                ax_r.plot(x, y, style, color=color, marker=marker, ms=5, lw=1.7)
                if proto == "none":
                    stats[ring]["rtt_min_size"] = float(y[np.argmin(x)])
        T.byte_axis(ax_r, "x")
        ax_r.set_xlabel("message size (bytes, log)")
        ax_r.set_ylabel("closed-loop RTT p50 (µs)")
        T.panel_title(ax_r, "Closed-loop round-trip time — ping-pong, one message in flight")
        ax_r.grid(True)

    # Takeaway computed strictly from the drawn routing series.
    bs, sl = stats["shm"], stats["shm-slot"]
    parts = []
    if "tput_max" in bs and "tput_max" in sl and bs["tput_max"] > 0:
        parts.append(
            f"at {T.fmt_bytes(sl.get('size_max', bs['size_max']))} the slot ring moves "
            f"{sl['tput_max']:.1f} vs {bs['tput_max']:.1f} Gbps plaintext "
            f"({sl['tput_max'] / bs['tput_max']:.2f}×)"
        )
    if "rtt_min_size" in bs and "rtt_min_size" in sl:
        parts.append(
            f"smallest-payload RTT p50 {sl['rtt_min_size']:.1f} µs (slot) vs "
            f"{bs['rtt_min_size']:.1f} µs (byte-stream)"
        )
    if parts:
        T.add_takeaway(fig, "Ring trade-off: " + "; ".join(parts) +
                       " — the slot ring pays a fixed per-segment cost at small payloads and "
                       "wins once slots amortize it.")
    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.04)
    T.add_method_note(
        fig,
        "matrix family · 1 connection · throughput on the 1-gateway (direct) chain, RTT from the "
        "closed-loop ping-pong rows; slot geometry auto-sized from the message size "
        "(segment = message + header, cache-line rounded); byte-stream ring uses eventfd wakeups, "
        "slot ring futex wakeups. Reduced profile set (routing / TLS 1.3 / kTLS 1.3) so 6 lines "
        "per panel stay print-readable — the full protocol sweep lives in F2/F16.",
    )
    T.add_provenance(fig, bundle.caption())
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
