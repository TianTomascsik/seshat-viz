"""
F8 — Saturation knee: how each transport BREAKS under overload (the 3×2 factorial).

Six offered-load sweeps form a deliberate 3×2 factorial — {loopback, gateway-routing,
gateway-crypto} × {TCP, UDP}, all at 1 KB / 1 connection over a *shared* offered-load
range — so the figure isolates *how* a path degrades and separates transport from crypto
(the earlier 2×2 confounded them: its gateway-TCP was plaintext but gateway-UDP was DTLS,
so "UDP sheds loss" couldn't be told apart from "the DTLS path sheds loss").
Each sweep is a dual-axis subplot: achieved goodput (left) climbs then plateaus while
loss% and p99 latency (right, log) tell the failure story.

The headline result is that there are TWO distinct degradation modes, and the figure
badges every panel with the one its data actually shows:
  * LATENCY-degrading — reliable paths (TCP, and loopback UDP the kernel never drops)
    shed ~0 packets; the sender's queue simply grows and p99 explodes into the seconds.
  * LOSS-shedding — paths whose loss genuinely crosses ~1% drop datagrams past their
    knee while latency stays bounded. On the canonical factorial BOTH gateway-UDP cells
    shed — plaintext routing and DTLS alike — so shedding is a gateway-UDP property, not
    a crypto one; the takeaway therefore names every shedding path with ITS OWN knee and
    peak loss rather than pooling them onto one path.

Because loss stays ~0 on the reliable panels, the old "max loss-free" horizontal marker
there was a *fake* knee (it merely traced the plateau); it is now drawn only on panels
whose loss genuinely crosses ~1%. A final overlay panel compares all goodput curves.

Honesty guards:
  * summary.csv's own `harness_limited` verdict is surfaced per panel: limited sweeps get
    a grey tag and a `*` on their peak / max-loss-free numbers — those plateaus are lower
    bounds on the path, not demonstrated gateway limits. The method note advertises the
    factorial cost decomposition (routing − loopback, crypto − routing) only when NO
    sweep is limited; on limited data that arithmetic yields negative "costs", so the
    figure does not claim it.
  * Panels lay out as the factorial regardless of loader order: rows {loopback,
    gateway-routing, gateway-crypto}, columns {TCP, UDP}. Sweeps that don't fit the
    factorial (older runs, extra cells) sort after it, and the figure renders however
    many sweeps a run contains.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle

FIG_ID = "F8"
NAME = "f08_saturation_knee"
TITLE = "Saturation: offered load vs achieved goodput, loss & tail latency"

_LOSS_THRESHOLD = 1.0  # percent; "loss-free" ceiling and the loss-shedding trigger

# Crypto markers in scenario names, for runs whose summary row is missing.
_CRYPTO_TAGS = ("tls", "dtls", "wireguard", "wg", "subset")


def _scenario_pretty(name: str) -> str:
    return name.replace("sat_", "").replace("_", " ")


def _col_first(g: pd.DataFrame, col: str, default: str) -> str:
    """First non-null value of a joined summary column, else `default`."""
    if col in g.columns:
        vals = g[col].dropna().unique()
        if len(vals):
            return str(vals[0])
    return default


def _panel_facts(scen: str, g: pd.DataFrame) -> tuple[bool, bool, str]:
    """(is_loopback, is_crypto, transport) for one sweep — prefers the joined summary
    columns, falls back to the scenario name so runs without a summary row still classify."""
    name = str(scen)
    loopback = "loopback" in name
    proto = _col_first(g, "protocol", "")
    if proto:
        crypto = proto not in ("none",)
    else:
        crypto = any(t in name for t in _CRYPTO_TAGS)
    transport = _col_first(g, "transport", "")
    if not transport:
        m = re.search(r"(?:^|_)(tcp|udp|uds|unix|shm)(?:_|$)", name)
        transport = m.group(1) if m else "?"
    return loopback, crypto, transport


def _factorial_key(scen: str, g: pd.DataFrame) -> tuple[int, int]:
    """Grid position (row, col) of a sweep in the factorial layout: rows {loopback,
    gateway-routing, gateway-crypto}, columns {TCP, UDP}. Sweeps outside the factorial
    (unknown transport) sort after it so the grid stays a factorial for readers."""
    loopback, crypto, transport = _panel_facts(scen, g)
    row = 0 if loopback else (2 if crypto else 1)
    col = {"tcp": 0, "udp": 1}.get(transport, 2)
    return (row, col)


def _harness_limited_by_scenario(summary: pd.DataFrame | None) -> dict:
    """summary.csv's own harness-limited verdict per scenario (the saturation table
    itself does not carry the column). Values may load as bools or 'True' strings."""
    if summary is None or summary.empty:
        return {}
    if "scenario" not in summary.columns or "harness_limited" not in summary.columns:
        return {}
    return {
        str(scen): str(val).strip().lower() in ("true", "1", "yes")
        for scen, val in zip(summary["scenario"], summary["harness_limited"])
    }


def _fmt_dur_us(us: float) -> str:
    """µs → adaptive 'N ms' / 'X.X s': a 8.6 ms floor must never print as '0.0 s'."""
    s = us / 1e6
    if s < 0.0995:  # would round to '0.0'/'0.1 s'; milliseconds carry the information
        return f"{us / 1e3:.0f} ms"
    return f"{s:.1f} s"


def _pinning_label(sat: pd.DataFrame, summary: pd.DataFrame | None, scenarios) -> str:
    """Pinned confounds for the method note, computed from the run (never asserted)."""
    parts: list[str] = []
    if "message_bytes" in sat.columns:
        u = pd.to_numeric(sat["message_bytes"], errors="coerce").dropna().unique()
        if len(u) == 1:
            parts.append(f"{T.fmt_bytes(float(u[0]))}B")
        elif len(u) > 1:
            parts.append("mixed sizes")
    if summary is not None and not summary.empty and {"scenario", "connections"} <= set(summary.columns):
        rows = summary[summary["scenario"].astype(str).isin({str(s) for s in scenarios})]
        u = pd.to_numeric(rows["connections"], errors="coerce").dropna().unique()
        if len(u) == 1:
            parts.append(f"{int(u[0])} conn")
        elif len(u) > 1:
            parts.append("mixed conns")
    return " · ".join(parts) if parts else "per-panel pinning in titles"


def _sweep_disclosure(scen: str, g: pd.DataFrame) -> str:
    """Full (datapath, protocol, transport, conn) so the viewer sees these sweeps
    are orthogonal (mixed datapath/protocol/transport), not one comparable series."""
    name = str(scen)
    # datapath / n_gateways: 'loopback' in the name == raw no-gateway baseline.
    if "loopback" in name:
        datapath = "loopback (no gateway)"
    else:
        datapath = "gateway"
    proto = _col_first(g, "protocol", "?")
    transport = _col_first(g, "transport", "?")
    conns = _col_first(g, "connections", "1")
    return f"{datapath} · {proto} · {transport} · {conns}c"


def make(bundle: RunBundle, saver: T.Saver) -> None:
    sat = bundle.saturation
    if sat is None or sat.empty or "offered_mbps" not in sat.columns:
        saver.record_skip(FIG_ID, NAME, "no saturation.csv sweeps in this run")
        return

    # Factorial grid order (rows loopback/routing/crypto × cols TCP/UDP), not loader order.
    scenarios = sorted(
        sat["scenario"].dropna().unique(),
        key=lambda s: (_factorial_key(s, sat[sat["scenario"] == s]), str(s)),
    )
    if not scenarios:
        saver.record_skip(FIG_ID, NAME, "saturation data present but no scenarios")
        return

    harness_limited = _harness_limited_by_scenario(bundle.summary)

    # Thesis variant: gateway sweeps only, one row, no overlay panel. The loopback row is a
    # harness self-baseline (it saturates the load generator first) — its content becomes a
    # method-note sentence, and the print figure carries the two degradation modes on the
    # gateway paths alone.
    thesis = T.thesis_variant()
    n_loopback_dropped = 0
    if thesis:
        gw_scenarios = [s for s in scenarios
                        if not _panel_facts(s, sat[sat["scenario"] == s])[0]]
        n_loopback_dropped = len(scenarios) - len(gw_scenarios)
        if gw_scenarios:
            scenarios = gw_scenarios

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(scenarios) + (0 if thesis else 1)  # +1 overlay panel (full variant only)
    ncol = min(2, n)
    nrow = math.ceil(n / ncol)
    figsize = (3.8 * ncol, 3.1 * nrow) if thesis else (5.6 * ncol, 3.6 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    flat = [axes[r][c] for r in range(nrow) for c in range(ncol)]

    # Accumulate per-panel degradation-mode facts for the headline takeaway. Kept per
    # panel (not pooled) so the takeaway can attribute each number to its own path.
    lat_panels: list[dict] = []   # latency-degrading: transport/loopback + exploded p99s
    shed_panels: list[dict] = []  # loss-shedding: name, knee, peak loss, harness flag
    transports_drawn: list[str] = []
    any_limited_drawn = False
    any_lossfree_drawn = False

    for ax, scen in zip(flat, scenarios):
        g = sat[sat["scenario"] == scen].sort_values("offered_mbps")
        is_loopback, is_crypto, transport = _panel_facts(scen, g)
        limited = harness_limited.get(str(scen), False)
        any_limited_drawn = any_limited_drawn or limited
        if transport not in transports_drawn:
            transports_drawn.append(transport)
        offered = g["offered_mbps"].values / 1000.0  # Gbps for a consistent x in Gbit/s
        goodput = g["throughput_gbps"].values
        # Goodput wears the panel's TRANSPORT identity (colour + marker); a
        # harness-limited sweep draws hollow markers (the plateau is a lower bound).
        color = T.transport_color(transport)
        ax.plot(
            offered, goodput, ls="-", marker=T.transport_marker(transport),
            color=color, lw=2, ms=4.5,
            markerfacecolor="none" if limited else color, markeredgecolor=color,
        )
        ax.plot([0, max(offered.max(), goodput.max())], [0, max(offered.max(), goodput.max())],
                ":", color=T.GREYS["faint"], lw=1)
        ax.set_xlabel("offered load (Gbps)")
        ax.set_ylabel("goodput (Gbps)")
        T.panel_title(ax, f"{_scenario_pretty(scen)}\n{_sweep_disclosure(scen, g)}")

        # Right axis: the two failure-story metrics, monochrome and separated by
        # linestyle + marker (colour stays reserved for the transport identity).
        ax2 = ax.twinx()
        if "loss_pct" in g.columns:
            ax2.plot(offered, g["loss_pct"].values, ls="--", marker="s",
                     color=T.GREYS["muted"], ms=3, lw=1.3)
        if "latency_p99_us" in g.columns:
            ax2.plot(offered, g["latency_p99_us"].values, ls="-.", marker="^",
                     color=T.GREYS["annot"], ms=3, lw=1.1)
            ax2.set_yscale("log")
        ax2.set_ylabel("loss % / p99 µs (log)", fontsize=T.FS["small"])
        ax2.grid(False)

        # --- Degradation mode: LATENCY-degrading vs LOSS-shedding --------------------
        # A reliable path (TCP, or loopback UDP the kernel never drops) sheds no packets:
        # loss_pct stays ~0 and it degrades purely by LATENCY (the sender's queue grows and
        # p99 climbs into the seconds). A genuinely lossy path — in this factorial, UDP
        # through the gateway, plaintext routing and DTLS alike — crosses the loss threshold
        # and sheds load by PACKET LOSS. Detect the mode from the data, badge the panel in
        # plain ink, and draw the loss-based markers ONLY where loss actually crosses ~1% —
        # otherwise "max loss-free" is a fake knee tracing the plateau.
        has_loss = "loss_pct" in g.columns
        loss = g["loss_pct"].to_numpy() if has_loss else np.zeros_like(goodput)
        crosses_loss = bool(np.any(loss > _LOSS_THRESHOLD))
        mode = "loss-shedding" if crosses_loss else "latency-degrading"
        ax.text(0.98, 0.04, mode, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=T.FS["annot"], color=T.GREYS["ink"], zorder=6)

        # Saturation PEAK (argmax goodput) — the plateau height, on every panel. On a flat
        # plateau the argmax is noise-picked, so read it as "peak", not a knee.
        peak_i = int(np.nanargmax(goodput))
        peak_right = offered[peak_i] > (offered.min() + offered.max()) / 2
        # Directly above the peak point: nothing of the goodput curve sits above its own
        # maximum, so this is the one placement the curve cannot strike through.
        ax.annotate(f"peak ≈ {goodput[peak_i]:.2f} Gbps", (offered[peak_i], goodput[peak_i]),
                    xytext=(-4 if peak_right else 4, 8), textcoords="offset points",
                    ha="right" if peak_right else "left", va="bottom",
                    fontsize=T.FS["annot"], color=T.GREYS["ink"])

        if crosses_loss:
            # Loss-shedding: the max loss-free rate and the loss knee are REAL here.
            fact = {"name": _scenario_pretty(scen), "knee": None,
                    "loss": float(np.nanmax(loss)), "limited": limited,
                    "transport": transport, "loopback": is_loopback, "crypto": is_crypto}
            lossfree = g[g["loss_pct"] <= _LOSS_THRESHOLD]
            if not lossfree.empty:
                mlf = float(lossfree["throughput_gbps"].max())
                fact["knee"] = mlf
                any_lossfree_drawn = True
                # The line is keyed in the shared legend and its value lives in the
                # takeaway — an in-plot label here always collides with the plateau,
                # which sits at the same height by definition.
                ax.axhline(mlf, color=T.GREYS["faint"], ls=":", lw=1.2)
            ki = int(np.argmax(loss > _LOSS_THRESHOLD))  # first offered load that sheds
            ax.axvline(offered[ki], color=T.ACCENT, ls="--", lw=1.1)
            shed_panels.append(fact)
        else:
            # Latency-degrading: loss never crosses ~1%, so a "max loss-free" line would just
            # trace the plateau (a fake knee) — suppress it. Mark the LATENCY knee instead: the
            # first offered load whose p99 explodes past 1 ms, and record the degraded region.
            if "latency_p99_us" in g.columns:
                p99 = g["latency_p99_us"].to_numpy()
                exploded = p99 > 1000.0
                if exploded.any():
                    ki = int(np.argmax(exploded))
                    ax.axvline(offered[ki], color=T.ACCENT, ls="--", lw=1.1)
                    lat_panels.append({"transport": transport, "loopback": is_loopback,
                                       "p99": [float(v) for v in p99[exploded]]})

    # ONE shared legend for the whole figure (replaces four identical per-panel
    # repeats): transport identities, the two right-axis metrics, the guides, and
    # the harness-limited key when it applies.
    handles = [
        Line2D([0], [0], ls="-", marker=T.transport_marker(t), color=T.transport_color(t),
               ms=6, lw=2, label=f"goodput ({t.upper()})")
        for t in transports_drawn if t in T.TRANSPORT_COLORS
    ]
    handles += [
        Line2D([0], [0], ls=":", color=T.GREYS["faint"], lw=1, label="ideal (offered = achieved)"),
        Line2D([0], [0], ls="--", marker="s", color=T.GREYS["muted"], ms=4, label="loss %"),
        Line2D([0], [0], ls="-.", marker="^", color=T.GREYS["annot"], ms=4, label="p99 latency"),
        Line2D([0], [0], ls="--", color=T.ACCENT, lw=1.1, label="knee"),
    ]
    if any_lossfree_drawn:
        handles.append(Line2D([0], [0], ls=":", color=T.GREYS["faint"], lw=1.2,
                              label="max loss-free"))
    if any_limited_drawn:
        handles.append(
            Line2D([0], [0], ls="-", marker="o", color=T.GREYS["annot"], ms=6,
                   markerfacecolor="none", label="hollow = harness-limited (lower bound)"))
    T.legend_below(fig, handles, ncol=min(4, len(handles)))

    # Overlay panel: all goodput curves together (full variant only).
    if thesis:
        for ax in flat[len(scenarios):]:
            ax.axis("off")
        _finish(fig, bundle, saver, sat=sat, scenarios=scenarios,
                harness_limited=harness_limited, lat_panels=lat_panels,
                shed_panels=shed_panels, thesis=True,
                n_loopback_dropped=n_loopback_dropped)
        return
    axo = flat[len(scenarios)]
    # Overlay encoding: colour = transport (as everywhere), linestyle = row of the
    # factorial (loopback dotted, gateway-routing dashed, gateway-crypto solid).
    _row_ls = {0: ":", 1: "--", 2: "-"}
    for scen in scenarios:
        g = sat[sat["scenario"] == scen].sort_values("offered_mbps")
        row, _ = _factorial_key(scen, g)
        _, _, transport = _panel_facts(scen, g)
        axo.plot(g["offered_mbps"].values / 1000.0, g["throughput_gbps"].values,
                 ls=_row_ls.get(row, "-"), marker=T.transport_marker(transport),
                 color=T.transport_color(transport), ms=3, lw=1.6,
                 label=_scenario_pretty(scen))
    axo.set_xlabel("offered load (Gbps)")
    axo.set_ylabel("goodput (Gbps)")
    T.panel_title(axo, f"All {len(scenarios)} offered-load sweeps overlaid")
    T.legend_inline(axo, fontsize=T.FS["annot"])

    # Hide any unused trailing axes.
    for ax in flat[len(scenarios) + 1:]:
        ax.axis("off")

    _finish(fig, bundle, saver, sat=sat, scenarios=scenarios,
            harness_limited=harness_limited, lat_panels=lat_panels,
            shed_panels=shed_panels, thesis=False, n_loopback_dropped=0)


def _finish(fig, bundle: RunBundle, saver: T.Saver, *, sat, scenarios, harness_limited,
            lat_panels, shed_panels, thesis: bool, n_loopback_dropped: int) -> None:
    # Headline the REAL conclusion as the takeaway, composed from the measured facts:
    # latency-degrading panels hold ~0% loss while p99 explodes; every loss-shedding
    # panel is named with ITS OWN knee and peak loss (no pooling across paths).
    bits = []
    starred = False
    if lat_panels:
        pool = [v for p in lat_panels for v in p["p99"]]
        lo, hi = min(pool), max(pool)
        # Attribute the range to the panels that actually produced it (loopback UDP is
        # latency-degrading too — labelling the pool "(TCP)" would misattribute the floor).
        who_parts: list[str] = []
        for p in lat_panels:
            lab = "loopback UDP" if (p["transport"] == "udp" and p["loopback"]) else str(p["transport"]).upper()
            if lab not in who_parts:
                who_parts.append(lab)
        who = " + ".join(who_parts)
        bits.append(f"{who} panels degrade by LATENCY at ~0% loss (p99 → {_fmt_dur_us(lo)}–{_fmt_dur_us(hi)})")
    if shed_panels:
        clauses = []
        for p in shed_panels:
            s = "*" if p["limited"] else ""
            starred = starred or p["limited"]
            knee_txt = f"knee ≈ {p['knee']:.2f} Gbps{s}, " if p["knee"] is not None else ""
            clauses.append(f"{p['name']} ({knee_txt}up to ~{p['loss']:.0f}% loss)")
        joined = " and ".join(clauses) if len(clauses) == 2 else "; ".join(clauses)
        shed_bit = f"load is shed by PACKET LOSS on {joined}"
        # When every shedding cell is gateway-UDP and they span plaintext + crypto, the
        # data itself shows shedding is a gateway-UDP property, not a crypto one.
        if (len(shed_panels) >= 2
                and all(p["transport"] == "udp" and not p["loopback"] for p in shed_panels)
                and any(p["crypto"] for p in shed_panels)
                and any(not p["crypto"] for p in shed_panels)):
            shed_bit += " — gateway-UDP sheds with or without crypto"
        bits.append(shed_bit)
    take = "  —  ".join(bits) if bits else "saturation sweeps recorded; no clear degradation-mode split"
    if starred:
        take += "  (* harness-limited: lower bound)"
    T.add_takeaway(fig, take)

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.01)
    n_limited = sum(1 for s in scenarios if harness_limited.get(str(s), False))
    if thesis:
        note = (
            f"{len(scenarios)} gateway offered-load sweeps at "
            f"{_pinning_label(sat, bundle.summary, scenarios)} over a shared offered-load "
            "range; "
        )
        if n_loopback_dropped:
            note += (
                f"{n_loopback_dropped} loopback (no-gateway) baseline sweeps are not drawn — "
                "they saturate the load generator itself and serve only as the harness "
                "self-ceiling. "
            )
    else:
        note = (
            f"{len(scenarios)} offered-load sweeps in factorial layout — rows loopback/gw-routing/"
            f"gw-crypto × columns TCP/UDP — at {_pinning_label(sat, bundle.summary, scenarios)} over "
            "a shared offered-load range; "
        )
    if n_limited:
        # The cost decomposition (routing − loopback, crypto − routing) is NOT claimed on
        # harness-limited data: those plateaus are lower bounds and the subtraction yields
        # negative "costs".
        note += (
            f"{n_limited}/{len(scenarios)} sweeps flagged harness-limited by the harness itself "
            "(hollow markers) — their peaks/knees are lower bounds, not demonstrated gateway "
            "limits, so plateau differences must not be read as forwarding/crypto costs. "
        )
    else:
        note += (
            "rows decompose the gateway forwarding cost (routing − loopback) and the crypto "
            "cost (crypto − routing); TCP-vs-UDP within a row isolates the transport at equal "
            "crypto. "
        )
    note += (
        "read shapes, not plateaus: a load-shedding path caps then drops packets (loss "
        "climbs) while a reliable path plateaus higher but pays it back as latency "
        "(p99 climbs at ~0% loss). "
    )
    T.add_method_note(fig, note + T.BLAST_LATENCY_NOTE)
    T.add_provenance(fig, bundle.caption())
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
