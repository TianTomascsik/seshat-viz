"""
F16 — Closed-loop round-trip latency across the full protocol × interface × payload grid.

SESHAT's ping-pong runs measure latency closed-loop (one request in flight), which is immune
to coordinated omission — the only honest absolute latency number. The `matrix_lat_*` grid
measures it for **every security protocol over every stream interface** (TCP / UDS / SHM /
TPROXY) at **every payload size** (64 B – 64 KB), so this figure now shows the whole space
rather than a TCP-only slice:

  * Top — one facet per interface: closed-loop RTT (p99) vs payload size, one line per protocol,
    so the real microsecond cost of each protocol *and* how it scales with payload is visible,
    interface by interface. Payload size is the x-axis, so the figure is self-labelling.
  * Bottom — coordinated-omission inflation: per protocol, the closed-loop p99 of one
    (protocol, interface) cell against that *same cell's* matched *open-loop blast* p99 at a
    fixed representative payload size (the cell with the largest matched ratio, interface named
    on the axis). Both endpoints of a dumbbell come from one cell, so the annotated ratio is a
    genuinely matched inflation, not a cross-interface min/max pairing. The blast baseline is
    restricted to 1-connection scg-direct rows of the throughput matrix — like-for-like with
    the `matrix_lat_*` topology — so cipher-sweep, two-gateway and non-matrix pools cannot
    blend into it. The gap is the CO inflation that makes open-loop tail latency unusable as
    an absolute number.

Datagram interfaces (UDP: DTLS, ALE, raw UDP-over-TLS, UDP routing) are absent by construction:
the one-way datagram gateway path cannot bounce a request back to the client, so it has no
closed-loop RTT (its honest per-message signal is the jitter/PDV panel of F4). `integrity_tls13`
is absent too — TLS 1.3 has no integrity-only (NULL-cipher) suite, so that combination does not
establish. Older runs without the `matrix_lat_*` grid fall back to the per-profile dumbbell view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F16"
NAME = "f16_closed_loop_rtt"
TITLE = "Closed-loop RTT across the protocol × interface × payload grid"


def make(bundle: RunBundle, saver: T.Saver) -> None:
    tbl = derive.rtt_inflation(bundle.summary)
    if tbl.empty:
        saver.record_skip(FIG_ID, NAME, "no rtt_us_p99 (run has no ping-pong scenarios)")
        return

    tbl = tbl.copy()
    # The closed-loop RTT grid: every (protocol, interface, size) ping-pong cell. Prefer it when
    # present; a run without it (only the profile/loopback ping-pong rows) uses the legacy view.
    if "family" in tbl.columns:
        grid = tbl[tbl["family"].astype(str) == "matrix-latency"].copy()
    else:
        grid = tbl.iloc[0:0].copy()

    if grid["rtt_p99"].notna().sum() >= 4 and grid["transport"].nunique() >= 1:
        _make_grid(grid, bundle, saver)
    else:
        _make_dumbbell(tbl, bundle, saver)


# --------------------------------------------------------------------------------------
# New view: the full protocol × interface × payload grid.
# --------------------------------------------------------------------------------------

def _matched_blast_baseline(summary: pd.DataFrame) -> pd.Series:
    """
    Like-for-like open-loop baseline for the `matrix_lat_*` grid: mean blast p99 keyed by
    (protocol, transport, message_bytes) over **1-connection scg-direct rows of the
    throughput matrix family only** — the same path/topology the closed-loop grid measures.

    `derive.rtt_inflation`'s generic pool spans every sustained-blast family, which blends
    cipher-sweep rows (a different cipher), two-gateway `chain=='scg'` rows and the
    iface/profile pools into the divisor; restricting family+chain here keeps
    the ratio a same-cell comparison. Rows carrying `rtt_us_p99` (closed-loop) are dropped
    as a guard even though the mode classifier already excludes them. Deliberately no
    cross-size or protocol-only fallback: a cell without an exactly-matched blast row gets
    NaN rather than a blended baseline.
    """
    need = {"scenario", "protocol", "transport", "message_bytes", "latency_p99_us_mean"}
    if not need.issubset(summary.columns):
        return pd.Series(dtype=float)
    m = derive.throughput_scenarios_only(summary)
    if "connections" in m.columns:
        m = m[(m["connections"] == 1) | m["connections"].isna()]
    if "family" in m.columns:
        m = m[m["family"].astype(str) == "matrix"]
    if "chain" in m.columns:
        m = m[m["chain"].astype(str) == "direct"]
    if "rtt_us_p99" in m.columns:
        m = m[m["rtt_us_p99"].isna()]
    m = m.dropna(subset=["latency_p99_us_mean", "message_bytes"])
    if m.empty:
        return pd.Series(dtype=float)
    return m.groupby([m["protocol"].astype(str), m["transport"].astype(str),
                      m["message_bytes"].astype(float)], observed=True)["latency_p99_us_mean"].mean()


def _make_grid(grid: pd.DataFrame, bundle: RunBundle, saver: T.Saver) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    grid = grid.dropna(subset=["rtt_p99", "message_bytes", "transport", "protocol"]).copy()
    grid["transport"] = grid["transport"].astype(str)
    grid["protocol"] = grid["protocol"].astype(str)

    # Replace derive.rtt_inflation's generic blast attachment with the strict like-for-like
    # baseline (1-conn scg-direct matrix rows, exact size+interface match) — see
    # _matched_blast_baseline. Cells without a matched blast row carry NaN, never a blend.
    base = _matched_blast_baseline(bundle.summary)
    if len(base):
        keys = pd.MultiIndex.from_arrays(
            [grid["protocol"], grid["transport"], grid["message_bytes"].astype(float)])
        grid["blast_p99"] = base.reindex(keys).to_numpy()
    else:
        grid["blast_p99"] = np.nan
    rtt = grid["rtt_p99"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        infl = grid["blast_p99"].to_numpy(dtype=float) / rtt
    grid["inflation"] = np.where(np.isfinite(infl), infl, np.nan)

    interfaces = [t for t in T.TRANSPORT_ORDER if t in set(grid["transport"])]
    interfaces += [t for t in grid["transport"].unique() if t not in interfaces]
    protocols = [p for p in T.PROTOCOL_ORDER if p in set(grid["protocol"])]
    protocols += [p for p in grid["protocol"].unique() if p not in protocols]
    sizes = sorted(grid["message_bytes"].dropna().unique())

    # Print variant: the top facets draw one representative protocol per protection family
    # (ten near-coincident lines per 4 cm facet are unreadable in print); the takeaway keeps
    # quoting the FULL grid, and the bottom inflation panel keeps every protocol. The
    # un-drawn variants track their family's curve — stated in the method note.
    in_print = T.print_variant()
    _F16_FAMILY_REPS = ["none", "tls/1.3", "ktls/1.3", "tls/1.3+mtls", "tls/1.2+integrity"]
    draw_protocols = ([p for p in _F16_FAMILY_REPS if p in protocols] or protocols) \
        if in_print else protocols

    # Shared y-range across facets so interfaces are directly comparable.
    yvals = grid["rtt_p99"].to_numpy(dtype=float)
    yvals = yvals[np.isfinite(yvals)]
    ymax = float(yvals.max()) * 1.10 if yvals.size else 1.0
    ymin = max(0.0, float(yvals.min()) * 0.85) if yvals.size else 0.0

    n_if = len(interfaces)
    # A finite matched ratio somewhere is what earns the bottom panel (blast_p99 alone is not
    # enough: the strict baseline can match a cell whose rtt is unusable).
    have_infl = bool(grid["inflation"].notna().any())
    if in_print:
        fig = plt.figure(figsize=(max(7.4, 1.85 * n_if), 8.0 if have_infl else 4.2))
    else:
        fig = plt.figure(figsize=(max(9.5, 3.3 * n_if), 8.2 if have_infl else 5.0))
    # Manual layout: the global constrained-layout engine would collapse the two explicit
    # regions on top of each other, so drive positions by hand (savefig bbox='tight' crops).
    fig.set_layout_engine("none")
    # Two explicit regions so the headline, the shared legend, and the inflation panel each
    # get their own band (a single gridspec left a large dead gap between them).
    if have_infl:
        gs_top = fig.add_gridspec(1, n_if, top=0.895, bottom=0.63, wspace=0.24)
        gs_bot = fig.add_gridspec(1, 1, top=0.45, bottom=0.16)
    else:
        gs_top = fig.add_gridspec(1, n_if, top=0.86, bottom=0.30, wspace=0.24)
        gs_bot = None

    # --- Top row: one facet per interface, RTT p99 vs payload size, line per protocol. ---
    handles: dict[str, object] = {}
    top_axes = [fig.add_subplot(gs_top[0, i]) for i in range(n_if)]
    for i, (ax, iface) in enumerate(zip(top_axes, interfaces)):
        sub = grid[grid["transport"] == iface]
        for proto in draw_protocols:
            ps = sub[sub["protocol"] == proto].sort_values("message_bytes")
            if ps.empty:
                continue
            c = T.protocol_color(proto)
            (line,) = ax.plot(
                ps["message_bytes"], ps["rtt_p99"], color=c, lw=1.7, marker="o",
                ms=4.2, markeredgecolor="white", markeredgewidth=0.5, alpha=0.95,
                solid_capstyle="round",
            )
            handles.setdefault(proto, line)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{T.fmt_bytes(v)}B"))
        ax.tick_params(axis="x", labelrotation=45, labelsize=T.FS["annot"])
        ax.set_ylim(ymin, ymax)
        T.panel_title(ax, transport_label(iface))
        ax.set_xlabel("payload size", fontsize=T.FS["small"])
        ax.grid(True, which="major", alpha=0.35)
        if i == 0:
            ax.set_ylabel("closed-loop RTT p99 (µs)", fontsize=T.FS["small"])
        else:
            ax.tick_params(axis="y", labelleft=False)

    # One shared protocol legend under the facet row (above the inflation panel).
    ordered_handles = [handles[p] for p in protocols if p in handles]
    ordered_labels = [protocol_label(p) for p in protocols if p in handles]
    if have_infl:
        # Key the inflation panel's marker shapes here too (shape = loop discipline,
        # colour = protocol): neutral-grey glyphs so the key names the shape without
        # implying any one protocol.
        from matplotlib.lines import Line2D

        ordered_handles += [
            Line2D([0], [0], ls="none", marker="o", markersize=6,
                   markerfacecolor=T.GREYS["muted"], markeredgecolor=T.GREYS["edge"]),
            Line2D([0], [0], ls="none", marker="X", markersize=7,
                   markerfacecolor=T.GREYS["muted"], markeredgecolor=T.GREYS["edge"]),
        ]
        ordered_labels += ["closed-loop RTT p99", "open-loop blast p99 (one-way)"]
    # Balance the legend grid: 7 entries as 4+3, not a stranded 6+1 second row.
    _rows = -(-len(ordered_labels) // 6)
    ncol = -(-len(ordered_labels) // _rows)
    fig.legend(
        ordered_handles, ordered_labels, loc="center",
        bbox_to_anchor=(0.5, (0.525 if in_print else 0.55) if have_infl else 0.14),
        ncol=ncol, fontsize=T.FS["small"], frameon=False, columnspacing=1.4,
        handletextpad=0.5,
        title=None if in_print else "protocol", title_fontsize=T.FS["small"],
    )

    # --- Bottom row: coordinated-omission inflation at a representative payload size. ---
    rep_size = None
    if have_infl:
        ax_infl = fig.add_subplot(gs_bot[0, 0])
        # Representative size: the measured size closest to 1 KB (typical control-message size),
        # so the closed-vs-blast comparison is at one clearly-stated payload, not a size blend.
        # Only sizes with at least one matched cell qualify (the strict baseline yields NaN,
        # never a blend, when a size has no like-for-like blast row).
        cand = sorted(grid.loc[grid["inflation"].notna(), "message_bytes"].unique())
        rep_size = min(cand, key=lambda s: abs(s - 1024)) if cand else None
        at = grid[(grid["message_bytes"] == rep_size) & grid["inflation"].notna()].copy()
        # One dumbbell per protocol, both endpoints from the SAME (protocol, interface) cell:
        # the cell with the largest matched ratio (a cross-interface min-RTT / max-blast pairing
        # is not matched and overstates the ratio). The chosen interface is named
        # on the axis so the pairing is verifiable from the figure.
        idx = at.groupby("protocol", observed=True)["inflation"].idxmax()
        per = at.loc[idx, ["protocol", "transport", "rtt_p99", "blast_p99", "inflation"]].copy()
        per["__o"] = per["protocol"].map({p: i for i, p in enumerate(protocols)}).fillna(99)
        per = per.sort_values("__o").reset_index(drop=True)
        y = np.arange(len(per))
        for i, r in per.iterrows():
            c = T.protocol_color(str(r["protocol"]))
            ax_infl.plot([r["rtt_p99"], r["blast_p99"]], [i, i], color=c, lw=2.4, alpha=0.5,
                         solid_capstyle="round", zorder=1)
            ax_infl.scatter([r["rtt_p99"]], [i], color=c, s=60, zorder=3,
                            edgecolor=T.GREYS["edge"], linewidth=0.4)
            ax_infl.scatter([r["blast_p99"]], [i], marker="X", color=c, s=80, zorder=3,
                            edgecolor=T.GREYS["edge"], linewidth=0.4)
            if np.isfinite(r["inflation"]):
                mid = float(np.sqrt(r["rtt_p99"] * r["blast_p99"]))
                ax_infl.annotate(f"{r['inflation']:.0f}×", (mid, i), xytext=(0, 7),
                                 textcoords="offset points",
                                 ha="center", fontsize=T.FS["annot"], color=T.GREYS["ink"])
        ax_infl.set_yticks(y)
        ax_infl.set_yticklabels(
            [f"{protocol_label(str(p))} · {transport_label(str(t))}"
             for p, t in zip(per["protocol"], per["transport"])], fontsize=T.FS["small"])
        ax_infl.invert_yaxis()
        ax_infl.set_xscale("log")
        ax_infl.xaxis.set_major_formatter(mticker.FuncFormatter(T.fmt_us))
        ax_infl.set_xlabel("p99 latency (µs, log)")
        T.panel_title(
            ax_infl,
            f"Coordinated-omission inflation at {T.fmt_bytes(rep_size)}B payload — worst matched "
            f"(protocol, interface) cell (one-way blast ÷ RTT, lower bound)")
        ax_infl.grid(axis="x", which="both", alpha=0.4)
        ax_infl.margins(x=0.12, y=0.06)

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=0.965)

    n_proto, n_if_lbl = len(protocols), len(interfaces)
    worst = grid.loc[grid["inflation"].idxmax()] if have_infl else None
    take = (f"Closed-loop RTT spans {T.fmt_latency_value(grid['rtt_p99'].min())}–"
            f"{T.fmt_latency_value(grid['rtt_p99'].max())} across {n_proto} protocols × "
            f"{n_if_lbl} interfaces × {len(sizes)} payload sizes — the coordinated-omission-safe absolute latency.")
    if worst is not None and np.isfinite(worst.get("inflation", np.nan)):
        # Name the grid-wide worst cell: it is usually NOT in the fixed-size bottom panel, so a
        # bare "(bottom)" pointer would cite a panel showing a smaller maximum.
        take += (f" Open-loop blast inflates the matched p99 by up to "
                 f"≥{worst['inflation']:.0f}× ({protocol_label(str(worst['protocol']))} · "
                 f"{T.fmt_bytes(worst['message_bytes'])}B over "
                 f"{transport_label(str(worst['transport']))}, grid-wide; the bottom panel shows "
                 f"the {T.fmt_bytes(rep_size)}B slice), so open-loop tail latency is not an "
                 f"absolute number.")
    T.add_takeaway(fig, take)

    note = (
        "closed-loop ping-pong RTT (one request in flight → coordinated-omission-free). Top: p99 "
        "vs payload size, one line per protocol, per stream interface (scg-direct, 1 conn). "
        "Datagram interfaces (UDP: DTLS / ALE / raw / UDP-routing) have no closed-loop RTT — the "
        "one-way datagram gateway cannot echo, see F4 jitter; integrity+TLS1.3 has no NULL-cipher "
        "suite so it is excluded."
    )
    if in_print and len(draw_protocols) < len(protocols):
        note += (
            f" Print variant: the facets draw {len(draw_protocols)} representative protection "
            f"families of the {len(protocols)} protocols measured — the un-drawn variants track "
            "their family's curve; the quoted span and the inflation panel cover all protocols."
        )
    if in_print and any(str(t) == "shm" for t in interfaces):
        note += (
            " SHM blast endpoints (✕) in the inflation panel include a harness receive-poll "
            "stall on top of coordinated omission — the SHM ratio overstates pure CO; paced "
            "SHM RTT (●) is the representative low-µs number."
        )
    if have_infl and rep_size is not None:
        # A real newline: theme.add_method_note draws one unwrapped fig.text line, and the full
        # note would stretch savefig's tight bbox far past the axes (captions.txt re-collapses
        # the whitespace).
        note += (f"\nBottom: per protocol, the (protocol, interface) cell with the largest matched "
                 f"ratio at {T.fmt_bytes(rep_size)}B — both dumbbell endpoints from that one cell, "
                 f"interface named on the axis; blast baseline = 1-conn scg-direct matrix blast, "
                 f"matched exactly per (protocol, interface, size), no cross-size/interface blending; "
                 f"one-way blast ÷ round-trip RTT keeps the ratio a conservative lower bound "
                 f"(harness-limited blast rows only understate the open-loop tail).")
    T.add_method_note(fig, note)
    T.add_provenance(fig, bundle.caption() + "  ·  ping-pong is closed-loop (CO-free); matrix blast p99 is open-loop")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)


# --------------------------------------------------------------------------------------
# Legacy fallback: per-profile dumbbell (runs without the matrix_lat_* grid).
# --------------------------------------------------------------------------------------

def _short(scenario: str, proto: str, tuning, datapath, message_bytes) -> str:
    base = protocol_label(str(proto))
    dp = str(datapath) if isinstance(datapath, str) and datapath else (
        "loopback" if "loopback" in str(scenario) else "gateway")
    tag = f" ({dp})"
    size = ""
    if pd.notna(message_bytes):
        try:
            size = f" · {T.fmt_bytes(float(message_bytes))}B"
        except (TypeError, ValueError):
            size = ""
    if isinstance(tuning, str) and tuning:
        return f"{base}{tag} · {tuning}{size}"
    if str(scenario).startswith("pp_"):
        parts = str(scenario).split("_")
        return f"{base}{tag} · {parts[1]}{size}"
    return f"{base}{tag}{size}"


def _make_dumbbell(tbl: pd.DataFrame, bundle: RunBundle, saver: T.Saver) -> None:
    import matplotlib.pyplot as plt

    tbl = tbl.copy()
    tbl["__o"] = tbl["protocol"].astype(str).map({p: i for i, p in enumerate(T.PROTOCOL_ORDER)}).fillna(99)
    tbl = tbl.sort_values(["__o", "rtt_p99"]).reset_index(drop=True)
    tbl["label"] = [_short(s, p, t, dp, mb) for s, p, t, dp, mb in
                    zip(tbl["scenario"], tbl["protocol"],
                        tbl.get("profile_tuning", pd.Series([None] * len(tbl))),
                        tbl.get("datapath", pd.Series([None] * len(tbl))),
                        tbl.get("message_bytes", pd.Series([None] * len(tbl))))]

    have_inflation = "blast_p99" in tbl.columns and tbl["blast_p99"].notna().any()
    fig, axes = plt.subplots(1, 2 if have_inflation else 1, figsize=(12.4 if have_inflation else 7.0, 6.2),
                             squeeze=False)

    axa = axes[0][0]
    for i, r in tbl.iterrows():
        c = T.protocol_color(str(r["protocol"]))
        p50, p99 = r.get("rtt_p50"), r["rtt_p99"]
        if pd.notna(p50):
            axa.plot([p50, p99], [i, i], color=c, lw=2.4, alpha=0.55, solid_capstyle="round", zorder=1)
            axa.scatter([p50], [i], color="white", edgecolor=c, s=46, zorder=3, linewidth=1.4)
        axa.scatter([p99], [i], color=c, s=62, zorder=3, edgecolor=T.GREYS["edge"], linewidth=0.4)
        axa.annotate(T.fmt_latency_value(p99), (p99, i), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=T.FS["annot"],
                     color=T.GREYS["ink"])
    axa.set_yticks(np.arange(len(tbl)))
    axa.set_yticklabels(tbl["label"], fontsize=T.FS["small"])
    axa.invert_yaxis()
    axa.set_xlabel("closed-loop latency (µs)")
    T.legend_inline(axa, T.percentile_handles(), loc="lower right")
    T.panel_title(axa, "Closed-loop RTT (coordinated-omission-free)")
    axa.grid(axis="x")
    axa.margins(x=0.18)

    if have_inflation:
        axb = axes[0][1]
        # One row per protocol, both endpoints from the SAME scenario row (the one with the
        # largest matched ratio): min-RTT/max-blast aggregation across rows would pair
        # endpoints from different scenarios and overstate the ratio.
        d = tbl.dropna(subset=["blast_p99", "rtt_p99"]).copy()
        d = d[d["rtt_p99"] > 0]
        d["__infl"] = d["blast_p99"] / d["rtt_p99"]
        d = d[np.isfinite(d["__infl"])]
        per_proto = (d.loc[d.groupby("protocol", observed=True)["__infl"].idxmax(),
                           ["protocol", "rtt_p99", "blast_p99", "__infl"]]
                     .reset_index(drop=True))
        per_proto["__o"] = per_proto["protocol"].astype(str).map({p: i for i, p in enumerate(T.PROTOCOL_ORDER)}).fillna(99)
        per_proto = per_proto.sort_values("__o").reset_index(drop=True)
        for i, r in per_proto.iterrows():
            c = T.protocol_color(str(r["protocol"]))
            axb.plot([r["rtt_p99"], r["blast_p99"]], [i, i], color=c, lw=2.4, alpha=0.5,
                     solid_capstyle="round", zorder=1)
            axb.scatter([r["rtt_p99"]], [i], color=c, s=60, zorder=3,
                        edgecolor=T.GREYS["edge"], linewidth=0.4)
            axb.scatter([r["blast_p99"]], [i], marker="X", color=c, s=80, zorder=3,
                        edgecolor=T.GREYS["edge"], linewidth=0.4)
            if np.isfinite(r["__infl"]):
                mid = np.sqrt(r["rtt_p99"] * r["blast_p99"])
                axb.annotate(f"{r['__infl']:.0f}×", (mid, i), xytext=(0, 7), textcoords="offset points",
                             ha="center", fontsize=T.FS["annot"], color=T.GREYS["ink"])
        axb.set_yticks(np.arange(len(per_proto)))
        axb.set_yticklabels([protocol_label(str(p)) for p in per_proto["protocol"]],
                            fontsize=T.FS["small"])
        axb.invert_yaxis()
        axb.set_xscale("log")
        axb.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
        axb.set_xlabel("p99 latency (µs, log)")
        from matplotlib.lines import Line2D

        T.legend_inline(axb, [
            Line2D([0], [0], ls="none", marker="o", markersize=6,
                   markerfacecolor=T.GREYS["muted"], markeredgecolor=T.GREYS["edge"],
                   label="closed-loop RTT p99"),
            Line2D([0], [0], ls="none", marker="X", markersize=7,
                   markerfacecolor=T.GREYS["muted"], markeredgecolor=T.GREYS["edge"],
                   label="open-loop blast p99 (one-way)"),
        ], loc="lower right")
        T.panel_title(axb, "Coordinated-omission inflation (one-way blast ÷ RTT, lower-bound)")
        axb.grid(axis="x", which="both")
        axb.margins(x=0.12)

    T.set_headline(fig, f"Closed-loop RTT and coordinated-omission inflation  ·  {bundle.label}", y=1.03)
    if have_inflation and len(tbl):
        worst = tbl.loc[tbl["inflation"].idxmax()] if "inflation" in tbl.columns and tbl["inflation"].notna().any() else None
        if worst is not None:
            T.add_takeaway(fig, f"Open-loop blast inflates p99 by ≥{worst['inflation']:.0f}× — "
                                f"closed-loop RTT ({T.fmt_latency_value(tbl['rtt_p99'].min())}–"
                                f"{T.fmt_latency_value(tbl['rtt_p99'].max())}) is the only coordinated-omission-safe absolute latency.")
    T.add_method_note(fig, "closed-loop ping-pong RTT (coordinated-omission-free). 'routing'=plaintext; "
                           "rows tagged (loopback)=raw no-gateway baseline, (gateway)=through SCG. Blast p99 "
                           "matched per (protocol, transport, size); the ratio divides a one-way blast p99 by a "
                           "round-trip RTT, so it is a conservative lower bound. This run carries only the "
                           "per-profile ping-pong rows; a matrix_lat_* run fills the full interface × payload grid.")
    T.add_provenance(fig, bundle.caption() + "  ·  ping-pong is closed-loop (CO-free); matrix blast p99 is open-loop")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
