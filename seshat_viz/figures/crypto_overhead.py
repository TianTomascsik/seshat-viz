"""
F3 — Cost of security: encryption overhead vs the routing baseline, per transport.

One dumbbell panel **per transport** (at a shared payload size): each encrypted protocol's
throughput is connected to that transport's own plaintext-routing baseline, so the visual
"drop" is the marginal cost of the scheme — and every panel is a *fair, same-transport,
same-size* comparison. Protocols are shared on the y-axis (the security ladder), so a reader
can also see which transport carries which scheme.

Why not every protocol from F1? F3 is a *ratio to routing*, so a scheme needs a plaintext
baseline on its own transport, and a panel appears only where the matrix provides one. Since
the `routing_udp` profile the UDP datagram family (DTLS, TLS-over-UDP, ALE) has a real
plaintext-UDP baseline too — but that baseline is datagram-rate-bound (per-packet cost, not
AES), so the UDP panel compares crypto against a much slower plaintext path than the stream
transports and is kept out of the cross-transport AES-ceiling takeaway (`_takeaway_text`).
Harness-limited rows — where the load generator, not the gateway, set the recorded rate — are
drawn hollow — and so is a harness-limited routing baseline dot (denominator).

Bottom row (per transport): the same crypto schemes' p99 as a multiple of that transport's own
routing p99. Under a saturating blast p99 is queueing-dominated — a near-monotone restatement of
the throughput drop, not an independent signal (theme.BLAST_LATENCY_NOTE) — so read it for ranking
only; honest absolute latency is F16's closed-loop RTT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F3"
NAME = "f03_crypto_overhead"
TITLE = "Cost of security: crypto throughput vs each transport's routing baseline"


def _ale_aware_protocol(df: pd.DataFrame) -> pd.DataFrame:
    """Add `_proto`, splitting the ETCS ALEPKT framing from the raw UDP-over-TLS baseline."""
    framing = df.get("app_framing", pd.Series([pd.NA] * len(df), index=df.index))
    return df.assign(_proto=[
        f"{p}+ale" if str(f) == "ale" else str(p)
        for p, f in zip(df["protocol"].astype(str), framing)
    ])


def _fmt_base(base: float) -> str:
    """Routing-baseline Gbps with enough digits that rounding stays within a few % of truth
    (a 0.51 Gbps datagram baseline must not render as '1')."""
    if base >= 10:
        return f"{base:.0f}"
    if base >= 1:
        return f"{base:.1f}"
    return f"{base:.2f}"


def _any_limited(x: pd.Series) -> bool:
    """True if any row in the group is flagged harness-limited (NA = not assessed → False)."""
    return bool(x.map(lambda v: bool(v) if pd.notna(v) else False).any())


def _takeaway_text(panels: list) -> str | None:
    """
    Build the data-driven takeaway from the rendered panels, with two honesty guards:

    (a) The "AES-GCM ceiling" may only average transports whose encrypted throughput actually
        sits on the crypto plateau. A transport whose crypto mean falls far below the cluster
        (< 1/3 of the median — e.g. a datagram-rate-bound UDP path whose *routing baseline*
        is already below the plateau) is bound by its transport, not by AES; folding it in
        deflates the ceiling and contradicts the transport-independence claim inside its own
        average.
    (b) The fast-vs-slow retained-% contrast is computed over the schemes the two transports
        SHARE. Averaging each side's own scheme mix lets DTLS-only rows inflate one side, and
        the matched-scheme comparison can even reverse direction.
    """
    stats = []
    for tr, base, _base_lat, _base_hl, agg in panels:
        # `+integrity` is the NULL-cipher (integrity-only) profile, not AEAD — it has no
        # business in an AES-GCM ceiling.
        gcm = agg[~agg.index.astype(str).str.endswith("+integrity")]
        if gcm.empty or not np.isfinite(base) or base <= 0:
            continue
        stats.append({
            "label": transport_label(str(tr)),
            "base": float(base),
            "crypto": float(gcm["tput"].mean()),
            "schemes": {str(p): float(v) for p, v in gcm["tput"].items()},
        })
    if len(stats) < 2:
        return None

    med = float(np.median([s["crypto"] for s in stats]))
    plateau = [s for s in stats if s["crypto"] >= med / 3.0]
    off = [s for s in stats if s["crypto"] < med / 3.0]
    if len(plateau) < 2:
        # No cross-transport crypto plateau to talk about — state only the observed spread.
        pcts = [s["crypto"] / s["base"] * 100.0 for s in stats]
        return (f"crypto retains {min(pcts):.0f}–{max(pcts):.0f}% of the plaintext-routing "
                "baseline depending on transport — see the per-panel annotations.")

    ceiling = float(np.mean([s["crypto"] for s in plateau]))
    spread = max(s["crypto"] for s in plateau) / min(s["crypto"] for s in plateau)
    by_base = sorted(plateau, key=lambda s: -s["base"])
    names = "/".join(s["label"] for s in by_base)
    # Only claim transport-independence when the plateau actually clusters; otherwise report
    # the spread instead of asserting flatness the data does not show.
    indep = ("is nearly transport-independent" if spread <= 1.5 else
             f"varies ({min(s['crypto'] for s in plateau):.0f}–"
             f"{max(s['crypto'] for s in plateau):.0f} Gbps)")
    text = f"the AES-GCM ceiling (~{ceiling:.0f} Gbps, one userspace core) {indep} across {names}"

    # Fast-vs-slow contrast, matched schemes only (guard b). Claim the growth direction only
    # when the data actually shows it.
    fast, slow = by_base[0], by_base[-1]
    matched = sorted(set(fast["schemes"]) & set(slow["schemes"]))
    if matched and fast["base"] > slow["base"]:
        f_pct = float(np.mean([fast["schemes"][p] for p in matched])) / fast["base"] * 100.0
        s_pct = float(np.mean([slow["schemes"][p] for p in matched])) / slow["base"] * 100.0
        if s_pct > f_pct:
            text += (f", so relative cost grows with transport speed: matched schemes keep "
                     f"{s_pct:.0f}% of {slow['label']} routing ({_fmt_base(slow['base'])} Gbps) "
                     f"but only {f_pct:.0f}% of {fast['label']} ({_fmt_base(fast['base'])} Gbps)")
        else:
            text += (f": matched schemes keep {f_pct:.0f}% of {fast['label']} routing and "
                     f"{s_pct:.0f}% of {slow['label']}")

    # kTLS ≈ TLS only if the plateau data says so (within 10% pooled over matched families).
    kt = [v for s in plateau for p, v in s["schemes"].items() if p.startswith("ktls")]
    ut = [v for s in plateau for p, v in s["schemes"].items()
          if p.startswith("tls/") and not p.endswith("+ale")]
    if kt and ut and 0.9 <= float(np.mean(kt)) / float(np.mean(ut)) <= 1.1:
        text += " — kTLS ≈ TLS throughout (offload needs a real NIC)"

    for s in off:
        text += (f"; {s['label']} is excluded from the ceiling — its routing path itself tops "
                 f"out at {_fmt_base(s['base'])} Gbps, so its crypto rows are transport-bound, "
                 "not AES-bound")
    return text + "."


def make(bundle: RunBundle, saver: T.Saver) -> None:
    df = bundle.summary
    if not bundle.has("throughput_gbps_mean") or "protocol" not in df.columns:
        saver.record_skip(FIG_ID, NAME, "needs throughput + protocol")
        return

    # Isolate per-message crypto cost: single connection (1c is the only count every transport
    # shares), matrix family (sustained-blast, explicit topology), 1-gateway (one clean routing
    # baseline per transport), no zero-throughput rows, no 0-gateway loopback rows (they would
    # skew the baseline). See the module docstring for the DTLS/ALE baseline caveat.
    if "connections" in df.columns:
        df = df[df["connections"].isin([1]) | df["connections"].isna()]
    df = df[df["throughput_gbps_mean"] > 0]
    if "family" in df.columns and (df["family"].astype(str) == "matrix").any():
        df = df[df["family"].astype(str) == "matrix"]
    if "datapath" in df.columns:
        df = df[df["datapath"].astype(str) != "loopback"]
    if "chain" in df.columns:
        df = df[df["chain"].astype(str) == "direct"]
    if df.empty or "message_bytes" not in df.columns or "transport" not in df.columns:
        saver.record_skip(FIG_ID, NAME, "no 1c matrix routing+crypto rows")
        return
    df = _ale_aware_protocol(df)

    # Pick ONE shared payload size so the cross-transport comparison is size-matched: the size that
    # maximizes total crypto coverage across the transports that have a routing baseline there.
    size_score: dict[int, int] = {}
    for sz, gs in df.groupby("message_bytes", observed=True):
        total = 0
        for _tr, gt in gs.groupby("transport", observed=True):
            protos = set(gt["_proto"].astype(str))
            if "none" in protos:
                total += len(protos - {"none"})
        size_score[int(sz)] = total
    if not size_score or max(size_score.values()) == 0:
        saver.record_skip(FIG_ID, NAME, "no size with a routing baseline + crypto")
        return
    size = max(size_score, key=lambda s: (size_score[s], s))  # most coverage, tie → larger size
    cell = df[df["message_bytes"] == size]

    # Per transport at this size: the routing baseline (throughput + blast p99) and its crypto
    # schemes' mean throughput / CI / blast p99. Each side also carries a harness-limited flag
    # (load generator, not the gateway, set the recorded rate) so capped cells can be disclosed.
    have_ci = "throughput_gbps_ci95" in df.columns
    have_lat = "latency_p99_us_mean" in cell.columns and cell["latency_p99_us_mean"].notna().any()
    have_hl = "harness_limited" in cell.columns
    panels: list[tuple[str, float, float, bool, pd.DataFrame]] = []
    for tr, gt in cell.groupby("transport", observed=True):
        base_rows = gt[gt["_proto"].astype(str) == "none"]
        enc = gt[gt["_proto"].astype(str) != "none"]
        if base_rows.empty or enc.empty:
            continue
        base = float(base_rows["throughput_gbps_mean"].mean())
        if not np.isfinite(base) or base <= 0:
            continue
        agg_kwargs = dict(
            tput=("throughput_gbps_mean", "mean"),
            tput_ci=("throughput_gbps_ci95", "mean") if have_ci else ("throughput_gbps_mean", "mean"),
        )
        if have_lat:
            agg_kwargs["lat"] = ("latency_p99_us_mean", "mean")
        if have_hl:
            agg_kwargs["hl"] = ("harness_limited", _any_limited)
        agg = enc.groupby("_proto", observed=True).agg(**agg_kwargs)
        if not have_hl:
            agg["hl"] = False
        base_lat = float(base_rows["latency_p99_us_mean"].mean()) if have_lat else float("nan")
        base_hl = _any_limited(base_rows["harness_limited"]) if have_hl else False
        panels.append((str(tr), base, base_lat, base_hl, agg))
    if not panels:
        saver.record_skip(FIG_ID, NAME, "no transport with routing + crypto at the chosen size")
        return
    # Richest transport first, then the canonical transport order.
    panels.sort(key=lambda p: (-len(p[4]), T.TRANSPORT_ORDER.index(p[0]) if p[0] in T.TRANSPORT_ORDER else 99))
    # Keep the two SHM ring variants side by side regardless of panel richness, so the
    # byte-stream vs fixed-slot contrast reads left-to-right without hunting.
    ids = [p[0] for p in panels]
    if "shm" in ids and "shm-slot" in ids:
        slot = panels.pop(ids.index("shm-slot"))
        panels.insert([p[0] for p in panels].index("shm") + 1, slot)

    # Thesis variant: throughput dumbbells only (the blast-p99 row is CO-uncorrected and
    # banned from the thesis body), and three representative panels — the slow and fast
    # stream extremes (SHM, TCP) plus the datagram outlier (UDP) — so annotations stay
    # legible at 15 cm. The retained-% contrast and the ceiling takeaway are computed over
    # exactly the drawn panels, so the claim never outruns the figure.
    thesis = T.thesis_variant()
    omitted_panels: list[str] = []
    if thesis:
        keep = [p for p in panels if p[0] in ("shm", "shm-slot", "unix", "tcp", "udp")] or panels[:4]
        omitted_panels = [transport_label(p[0]) for p in panels if p not in keep]
        panels = keep
        have_lat = False

    # Shared y-axis = union of crypto protocols in security-ladder order (strongest-first at top).
    all_protos: set[str] = set()
    for _tr, _base, _bl, _bh, agg in panels:
        all_protos |= set(agg.index.astype(str))
    yorder = [p for p in T.PROTOCOL_ORDER if p in all_protos]
    yorder += [p for p in sorted(all_protos) if p not in yorder]
    ypos = {p: (len(yorder) - 1 - i) for i, p in enumerate(yorder)}  # first in ladder → top

    import matplotlib.pyplot as plt

    ncol = len(panels)
    nrow = 2 if have_lat else 1
    # Shared log-x per ROW so lengths are comparable across transports (and narrow-range panels
    # don't get cramped ticks); shared y so the protocol ladder lines up. Row 0 = throughput cost,
    # row 1 = blast p99 vs routing.
    if thesis:
        figsize = (2.45 * ncol + 1.4, 0.36 * len(yorder) + 1.8)
    else:
        figsize = (3.5 * ncol + 1.4, (0.42 * len(yorder) + 1.3) * nrow + 0.8)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=figsize,
        squeeze=False, sharey=True, sharex=("row" if nrow > 1 else True),
    )

    for col, (tr, base, base_lat, base_hl, agg) in enumerate(panels):
        # Row 0 — throughput dumbbell (routing baseline → encrypted), % of routing retained.
        # Harness-limited crypto rows render hollow (F15 convention); a harness-limited routing
        # baseline gets a † on the panel title — both are explained in the method note.
        ax0 = axes[0][col]
        ax0.axvline(base, color=T.GREYS["baseline"], ls=":", lw=1.3, zorder=1)
        for proto, r in agg.iterrows():
            proto = str(proto)
            y = ypos[proto]
            color = T.protocol_color(proto)
            ax0.plot([base, r["tput"]], [y, y], color=color, lw=2.4, solid_capstyle="round", zorder=2)
            if base_hl:
                # Harness-limited routing baseline: hollow, same convention as the crypto rows
                # (this replaces the old † title flag).
                ax0.scatter([base], [y], facecolors="none", edgecolors=T.GREYS["baseline"],
                            linewidths=1.2, s=26, zorder=3)
            else:
                ax0.scatter([base], [y], color=T.GREYS["baseline"], s=26, zorder=3)
            if bool(r["hl"]):
                ax0.scatter([r["tput"]], [y], facecolors="none", edgecolors=color,
                            linewidths=1.6, s=60, zorder=4)
            else:
                ax0.scatter([r["tput"]], [y], color=color, s=60, zorder=4)
            if have_ci and np.isfinite(r["tput_ci"]) and r["tput_ci"] > 0:
                ax0.errorbar(r["tput"], y, xerr=r["tput_ci"], fmt="none", ecolor=color,
                             elinewidth=1.0, capsize=2.5, zorder=3.5)
            retained = r["tput"] / base * 100.0 if base else np.nan
            # Lifted off the row's connector line (and clear of the CI caps, which sit at
            # the row centre): the old on-the-line placement struck the label through.
            ax0.annotate(f"{retained:.0f}%", (r["tput"], y), xytext=(6, 6),
                         textcoords="offset points", ha="left", va="bottom",
                         fontsize=T.FS["annot"], color=T.GREYS["ink"])
        ax0.set_xscale("log")
        T.panel_title(ax0, f"{transport_label(tr)} · routing {_fmt_base(base)} Gbps")
        ax0.set_ylim(-0.6, len(yorder) - 0.4)
        ax0.grid(axis="x")
        ax0.set_xlabel("throughput (Gbps, log)")

        # Row 1 — blast p99 as a multiple of this transport's routing p99 (queueing-dominated).
        if have_lat and np.isfinite(base_lat) and base_lat > 0:
            ax1 = axes[1][col]
            for proto, r in agg.iterrows():
                y = ypos[str(proto)]
                m = float(r["lat"]) / base_lat if base_lat else np.nan
                if np.isfinite(m):
                    ax1.barh(y, m, height=0.6, color=T.protocol_color(str(proto)),
                             edgecolor=T.GREYS["edge"], linewidth=0.5,
                             hatch=T.HARNESS_HATCH if bool(r["hl"]) else None)
                    ax1.annotate(f"{m:.0f}×" if m >= 10 else f"{m:.1f}×", (m, y), xytext=(4, 0),
                                 textcoords="offset points", va="center",
                                 fontsize=T.FS["annot"], color=T.GREYS["ink"])
            ax1.axvline(1.0, color=T.GREYS["baseline"], ls=":", lw=1.2)
            ax1.set_xscale("log")
            ax1.set_ylim(-0.6, len(yorder) - 0.4)
            ax1.grid(axis="x")
            ax1.set_xlabel("p99 ÷ routing (×, log)")

    for rrow in range(nrow):
        axes[rrow][0].set_yticks([ypos[p] for p in yorder])
        axes[rrow][0].set_yticklabels([protocol_label(p) for p in yorder])

    # One figure-level key. Protocol identity is already the y-axis ladder, so the legend
    # carries only the shared semantics: the routing-baseline guide, the harness-limited
    # hollow convention, and (full variant) the harness-limited hatch on the latency bars.
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=T.GREYS["baseline"], ls=":", lw=1.3, marker="o",
               markersize=5, markerfacecolor=T.GREYS["baseline"],
               label="plaintext-routing baseline"),
        Line2D([0], [0], ls="none", marker="o", markersize=7, markerfacecolor="none",
               markeredgecolor=T.GREYS["annot"], markeredgewidth=1.5,
               label="hollow = harness-limited"),
    ]
    if have_lat:
        handles.append(T.harness_legend_handle())
    T.legend_below(fig, handles, ncol=len(handles))

    T.set_headline(fig, f"{TITLE} — {T.fmt_bytes(size)}B payload · 1 connection · 1-gateway\n{bundle.label}")

    # Method note: what was controlled for, plus the two data-basis caveats a reader needs to
    # weigh the ratios — the datagram-bound UDP denominator and the harness-limited cells.
    note = (
        f"one panel per transport at a matched {T.fmt_bytes(size)}B payload; each crypto scheme is "
        "divided by ITS OWN transport's plaintext-routing baseline (dotted line), so the drop is a "
        "fair same-transport cost — a panel appears only for transports with a plaintext-routing "
        "row. "
    )
    if any(str(tr) == "udp" for tr, *_rest in panels):
        note += (
            "the UDP routing baseline is datagram-rate-bound (per-packet cost, not AES), so the "
            "UDP panel compares crypto against a much slower plaintext path than the stream "
            "transports — absolute rates in F1 (landscape), saturation in F8. "
        )
    n_schemes = sum(len(agg) for *_rest, agg in panels)
    n_hl = int(sum(int(agg["hl"].sum()) for *_rest, agg in panels))
    n_base_hl = sum(1 for _tr, _b, _bl, bhl, _agg in panels if bhl)
    if n_hl or n_base_hl:
        note += (
            f"hollow markers mark harness-limited measurements ({n_hl}/{n_schemes} crypto rows, "
            f"{n_base_hl}/{len(panels)} routing baselines) — the load generator, not the "
            "gateway, set the recorded rate, so the affected retained-% are ratios of capped "
            "values. "
        )
    if omitted_panels:
        note += ("panels not shown in the print variant: " + "/".join(omitted_panels)
                 + " (transparent interception sits between the drawn stream panels). ")
    T.add_method_note(fig, note + ("" if thesis else T.BLAST_LATENCY_NOTE))
    T.add_provenance(fig, bundle.caption() + f"  ·  {bundle.label}  ·  crypto cost = throughput ÷ same-transport routing")

    # Data-driven takeaway: the crypto ceiling (~one userspace/AES-NI core) is roughly transport-
    # independent across the crypto-bound (plateau) transports, so the FRACTION of routing you
    # keep FALLS as the transport gets faster — the relative cost of security is harshest on the
    # fastest paths. Selection and matched-scheme guards live in `_takeaway_text`.
    takeaway = _takeaway_text(panels)
    if takeaway:
        T.add_takeaway(fig, takeaway)
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
