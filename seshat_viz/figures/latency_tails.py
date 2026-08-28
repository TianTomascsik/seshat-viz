"""
F7 — Latency tail SHAPE, faceted per transport (p50-normalized CCDF).

One complementary-CDF (1 − percentile, log-y) subplot **per transport** — reconstructed from
the runs.csv percentile columns via :func:`_median_rep_tails` — at a single pinned payload size,
one curve per protocol. Every curve is **normalized by its own p50 (median)** so the x-axis is a
multiple of the median, not absolute µs: that makes the tail *shape* directly comparable across
transports whose absolute latency differs by orders of magnitude (µs-scale plaintext TCP vs the
ms-scale SHM harness stall). The tail is banded — ≤2× (tight) · 2–10× (moderate) · >10× (heavy) —
so "how far past the median does the p99/p999 reach?" is legible at a glance.

Fixes over the previous single-panel design:
  * (V8) the old ``_pick_cell`` chose the (transport,size) with the *most protocols* — always TCP —
    and silently discarded every non-TCP transport. We now FACET per transport so shm/unix/tcp/
    tproxy(/udp) each get their own subplot at a matched size.
  * (V11) the "best covered" cell was, by coincidence, the accidental 64 B corner (all fixed
    per-syscall overhead). We pin a load-bearing MID/LARGE payload (prefer 4096 / 16384 B) and
    surface the pinned size in the headline + method note.
  * (V5) absolute-µs CCDFs mixed the SHM harness receive-poll stall (ms) with real service tails
    (µs). We plot a p50-normalized CCDF (log-y) and annotate the latency bands instead.
  * (F7-1) the chain topology is pinned figure-wide (prefer 2-gateway scg→scg) instead of a
    per-protocol "prefer scg if any" that silently mixed a 1-gateway scg-direct curve into an
    otherwise all-scg facet; combos that exist only under another chain are dropped and disclosed.
  * (F7-2) each curve is one scenario's **median-tail repetition**, not the mean of each
    percentile across repetitions: mean-of-percentiles fabricates a curve no run produced when
    reps are bimodal, and the takeaway number quoted that artifact.
  * (F7-3) when the plaintext curves are harness-limited (sender-saturated) and the encrypted
    ones are not, the asymmetry is disclosed — it is precisely what makes plaintext's median
    µs-scale and its *relative* tail wide.

Matrix-family-only: the alphabetical per-protocol scenario picker must never reach an iface
latency-paced run (multi-second tail) or a connrate resumed-handshake run — either would invert
the ranking this figure exists to show — so we restrict to the sustained-blast matrix family first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F7"
NAME = "f07_latency_tails"
TITLE = "Open-loop blast tail shape — p50-normalized CCDF, faceted by transport"

# Tail bands on the p50-normalized x-axis: how many medians out the tail reaches. A well-behaved
# service tail stays inside ~2× its median; 2–10× is a notable tail; >10× is pathological.
_BANDS = [
    (1.0, 2.0, "≤2× tight", T.SEM["ok"]),
    (2.0, 10.0, "2–10× moderate", T.SEM["warn"]),
    (10.0, float("inf"), ">10× heavy", T.SEM["bad"]),
]


def _pick_size(runs: pd.DataFrame) -> int | None:
    """
    Pin ONE load-bearing payload for the facet grid.

    Prefer a MID/LARGE size (≥1024 B) with the widest transport coverage, so the grid shows the
    gateway's real per-message tail rather than the 64 B corner (dominated by fixed per-syscall
    overhead, the size the old `_pick_cell` fell into). Falls back to any size if only tiny
    payloads exist. Ties break toward the canonical 4096 / 16384 B thesis sizes.
    """
    if "message_bytes" not in runs.columns or "transport" not in runs.columns:
        return None
    cov: dict[int, tuple[int, int]] = {}
    for sz, g in runs.groupby("message_bytes", observed=True):
        if pd.isna(sz):
            continue
        cov[int(sz)] = (
            g["transport"].nunique(),
            g.groupby(["transport", "protocol"], observed=True).ngroups,
        )
    if not cov:
        return None
    mid = {s: c for s, c in cov.items() if s >= 1024}
    pool = mid or cov
    pref = [4096, 16384, 8192, 2048, 1024, 65536, 9000]

    def _key(sz: int) -> tuple[int, int, int]:
        n_tr, n_curves = pool[sz]
        rank = pref.index(sz) if sz in pref else len(pref)
        return (n_tr, n_curves, -rank)

    return max(pool, key=_key)


def _pin_chain(d: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Pin ONE chain topology for the whole facet grid.

    Every curve must share a topology: a 1-gateway scg-direct scenario mixed into a
    2-gateway facet measures a different path (audit F7-1 — the per-protocol "prefer scg
    if any" fallback silently drew the tproxy routing curve from the run's only tproxy
    routing scenario, a chain=direct one, next to nine scg-scg curves). Prefer ``scg``
    (the gateway chain this figure is about); if the run has no scg rows at all, fall
    back to the most common chain so a direct-only run still renders — homogeneously.
    """
    if "chain" not in d.columns:
        return d, None
    chains = d["chain"].astype(str)
    if (chains == "scg").any():
        pin = "scg"
    else:
        modes = chains.mode()
        if modes.empty:
            return d, None
        pin = str(modes.iat[0])
    return d[chains == pin], pin


def _dropped_combos(all_rows: pd.DataFrame, kept_rows: pd.DataFrame) -> list[tuple[str, str]]:
    """(transport, protocol) combos present in `all_rows` but absent after the chain pin —
    the omissions the method note must disclose (a facet quietly showing 9 of 10 protocols
    is otherwise invisible to the reader)."""
    if all_rows.empty or "transport" not in all_rows.columns or "protocol" not in all_rows.columns:
        return []

    def combos(d: pd.DataFrame) -> set[tuple[str, str]]:
        return set(zip(d["transport"].astype(str), d["protocol"].astype(str)))

    return sorted(combos(all_rows) - (combos(kept_rows) if not kept_rows.empty else set()))


def _pick_scenarios(sub: pd.DataFrame) -> list[str]:
    """One representative scenario per protocol (a single connection preferred), so each
    protocol contributes one clean monotone CCDF curve instead of many overlaid runs.
    Chain topology is already pinned figure-wide by :func:`_pin_chain` — no per-protocol
    chain fallback happens here."""
    chosen: list[str] = []
    for _proto, pg in sub.groupby("protocol", observed=True):
        scens = list(pg["scenario"].unique())
        one_c = [s for s in scens if "_1c" in s]
        chosen.append(sorted(one_c)[0] if one_c else sorted(scens)[0])
    return chosen


def _median_rep_tails(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Long CCDF tail table (same columns as ``derive.tail_table``) where each scenario
    contributes the percentile row of its **median-tail repetition** — the rep whose
    p999÷p50 (widest available percentile over its own median) is the middle value across
    reps — instead of the mean of each percentile over reps.

    Mean-of-percentiles fabricates a curve when reps are bimodal: plaintext TCP reps at
    1.7×/18×/20× rendered as a ~14× curve that matches no observed run and became the
    takeaway number (audit F7-2). The median rep is one actually-observed distribution
    and is robust to a single outlier rep. Zero-work repetitions (no messages — the
    dead-repeat stall pathology, audit D2-1) and reps without a positive finite p50 are
    excluded before selection.

    Returns ``(tail_table, spread)`` where ``spread`` maps scenario → (min, max) per-rep
    ratio, so rep heterogeneity can be disclosed rather than averaged away.
    """
    pcols = derive._PCTL_COLS  # runs.csv percentile column → distribution fraction
    if runs is None or runs.empty:
        return pd.DataFrame(), {}
    present = [c for c in pcols if c in runs.columns]
    if "latency_p50_us" not in present or len(present) < 2 or "scenario" not in runs.columns:
        return pd.DataFrame(), {}

    d = runs.copy()
    # Dead-repeat guard: a zero-work rep records no meaningful percentiles.
    if "messages" in d.columns:
        d = d[pd.to_numeric(d["messages"], errors="coerce").fillna(0) > 0]
    p50 = pd.to_numeric(d["latency_p50_us"], errors="coerce")
    d = d[np.isfinite(p50) & (p50 > 0)]
    if d.empty:
        return pd.DataFrame(), {}

    top = max(present, key=lambda c: pcols[c])  # widest percentile available (p999 normally)
    d["_tail_ratio"] = pd.to_numeric(d[top], errors="coerce") / pd.to_numeric(
        d["latency_p50_us"], errors="coerce"
    )
    d = d[np.isfinite(d["_tail_ratio"])]

    id_cols = [c for c in ("scenario", "transport", "protocol", "message_bytes") if c in d.columns]
    rows: list[dict] = []
    spread: dict[str, tuple[float, float]] = {}
    for scen, g in d.groupby("scenario", observed=True):
        g = g.sort_values("_tail_ratio", kind="stable")
        pick = g.iloc[(len(g) - 1) // 2]  # lower middle → always an actual observed rep
        spread[str(scen)] = (float(g["_tail_ratio"].iloc[0]), float(g["_tail_ratio"].iloc[-1]))
        for col in present:
            try:
                val = float(pick[col])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            row = {c: pick[c] for c in id_cols}
            row.update(percentile=pcols[col], ccdf=1.0 - pcols[col], latency_us=float(val))
            rows.append(row)
    long = pd.DataFrame(rows)
    if long.empty:
        return long, spread
    return long.sort_values(id_cols + ["percentile"]), spread


def _harness_limited_map(summary: pd.DataFrame | None) -> dict[str, bool]:
    """scenario → harness_limited from summary.csv (runs.csv carries no such column);
    NA (never evaluated) counts as False."""
    if summary is None or summary.empty:
        return {}
    if "harness_limited" not in summary.columns or "scenario" not in summary.columns:
        return {}
    return {
        str(s): (bool(v) if pd.notna(v) else False)
        for s, v in zip(summary["scenario"].astype(str), summary["harness_limited"])
    }


def _fmt_us(us: float) -> str:
    """Compact latency label for the takeaway: 116 µs / 4.6 ms."""
    return f"{us / 1000:.1f} ms" if us >= 1000 else f"{us:.0f} µs"


def _draw_bands(ax, xmax: float) -> None:
    """Shade the tail bands and draw dotted boundary lines at 2× / 10× the median (log-x)."""
    for lo, hi, _name, col in _BANDS:
        left, right = max(lo, 1.0), min(hi, xmax)
        if right <= left:
            continue
        ax.axvspan(left, right, color=col, alpha=0.05, zorder=0)
        if 1.0 < lo < xmax:  # boundary line only where it falls inside the visible range
            ax.axvline(lo, color=col, ls=":", lw=0.9, alpha=0.6, zorder=1)


def make(bundle: RunBundle, saver: T.Saver) -> None:
    runs = bundle.runs
    if runs is None or runs.empty or "latency_p99_us" not in runs.columns:
        saver.record_skip(FIG_ID, NAME, "runs.csv with latency percentiles not available")
        return

    # Matrix family only (see module docstring) — do this before size/scenario selection.
    if "family" in runs.columns and (runs["family"].astype(str) == "matrix").any():
        runs = runs[runs["family"].astype(str) == "matrix"]

    # Pin the chain topology figure-wide BEFORE picking the size, so coverage counts what
    # will actually be plotted (audit F7-1).
    pinned, chain_pin = _pin_chain(runs)

    size = _pick_size(pinned)
    if size is None:
        saver.record_skip(FIG_ID, NAME, "no (transport,size) cell with latency tails")
        return
    at_size = pinned[pinned["message_bytes"] == size].copy()
    if at_size.empty:
        saver.record_skip(FIG_ID, NAME, "pinned payload size has no rows")
        return
    # Combos that exist at this size only under another chain — disclosed, not silently mixed in.
    dropped = _dropped_combos(runs[runs["message_bytes"] == size], at_size)

    # Transports present at the pinned size, in the canonical ladder order.
    transports = [t for t in T.TRANSPORT_ORDER if t in set(at_size["transport"].astype(str))]
    transports += [t for t in at_size["transport"].astype(str).unique() if t not in transports]

    # Build one p50-normalized CCDF curve per (transport, protocol).
    facets: list[tuple[str, list[tuple[str, np.ndarray, np.ndarray]]]] = []
    seen_protos: set[str] = set()
    xmax_obs = 1.0
    # "crypto"/"none" → the widest-tail curve of that kind (multiple + where it came from),
    # so the takeaway can quote an observed value with its context instead of a bare number.
    tail_by_kind: dict[str, dict] = {}
    rep_spread: dict[str, tuple[float, float]] = {}  # scenario → (min, max) per-rep p999÷p50
    scen_kind: dict[str, str] = {}  # plotted scenario → "none"/"crypto" (for the harness note)
    for tr in transports:
        tsub = at_size[at_size["transport"].astype(str) == tr]
        chosen = _pick_scenarios(tsub)
        tsub = tsub[tsub["scenario"].isin(chosen)]
        tails, spread = _median_rep_tails(tsub)
        rep_spread.update(spread)
        curves: list[tuple[str, np.ndarray, np.ndarray]] = []
        if not tails.empty:
            for scen, g in tails.groupby("scenario", observed=True):
                g = g.sort_values("percentile")
                p50row = g[g["percentile"] == 0.5]
                if p50row.empty:
                    continue
                p50 = float(p50row["latency_us"].iloc[0])
                if not np.isfinite(p50) or p50 <= 0:
                    continue
                proto = str(g["protocol"].iloc[0])
                mult = g["latency_us"].to_numpy(dtype=float) / p50  # x = latency ÷ median
                ccdf = g["ccdf"].to_numpy(dtype=float)
                curves.append((proto, mult, ccdf))
                seen_protos.add(proto)
                m = float(np.nanmax(mult))
                xmax_obs = max(xmax_obs, m)
                kind = "none" if proto == "none" else "crypto"
                scen_kind[str(scen)] = kind
                if kind not in tail_by_kind or m > tail_by_kind[kind]["max"]:
                    tail_by_kind[kind] = {
                        "max": m, "transport": tr, "p50_us": p50, "scenario": str(scen),
                    }
        if curves:
            curves.sort(key=lambda c: T.PROTOCOL_ORDER.index(c[0])
                        if c[0] in T.PROTOCOL_ORDER else len(T.PROTOCOL_ORDER))
            facets.append((tr, curves))

    if not facets:
        saver.record_skip(FIG_ID, NAME, "no transport had a reconstructable latency tail")
        return

    # Shared x-limit: keep resolution on the tight crypto cluster, but always show the 2× boundary
    # (and cap so a lone heavy tail can't crush every other curve into the left spine).
    xmax = min(60.0, max(2.4, xmax_obs * 1.15))

    # Assemble the method-note lines BEFORE creating the figure so the bottom margin can
    # reserve exactly the space they need (they used to overprint the bottom row's ticks).
    shm_present = any(tr == "shm" for tr, _ in facets)
    chain_txt = ""
    if chain_pin is not None:
        chain_txt = f"; chain pinned to {chain_pin}" + (
            " (2-gateway scg→scg)" if chain_pin == "scg" else ""
        )
    notes: list[str] = [
        f"pinned payload = {int(size)} B ({T.fmt_bytes(size)}B), the widest-coverage mid/large "
        f"size{chain_txt}; one single-connection scenario per protocol; each curve is that "
        "scenario's median-tail repetition (median p999/p50 across reps — reps are not averaged); "
        "x = latency ÷ that rep's own p50, so tail SHAPE is comparable across transports with "
        "very different absolute latency"
    ]

    # Disclosure line: combos dropped by the chain pin, and any harness-limited asymmetry
    # between the plaintext and encrypted curve pools (audit F7-1 / F7-3).
    disclosures: list[str] = []
    if dropped:
        shown = ", ".join(f"{transport_label(t)}/{protocol_label(p)}" for t, p in dropped[:4])
        more = f" (+{len(dropped) - 4} more)" if len(dropped) > 4 else ""
        disclosures.append(f"no {chain_pin}-chain scenario at this size (omitted): {shown}{more}")
    lim_map = _harness_limited_map(bundle.summary)
    if lim_map and scen_kind:
        n_lim = {"none": 0, "crypto": 0}
        n_tot = {"none": 0, "crypto": 0}
        for scen, kind in scen_kind.items():
            n_tot[kind] += 1
            n_lim[kind] += 1 if lim_map.get(scen, False) else 0
        if n_lim["none"] + n_lim["crypto"] > 0:
            if (n_tot["none"] and n_tot["crypto"]
                    and n_lim["none"] / n_tot["none"] > n_lim["crypto"] / n_tot["crypto"]):
                disclosures.append(
                    f"plaintext curves are harness-limited (sender-saturated, "
                    f"{n_lim['none']}/{n_tot['none']}) vs {n_lim['crypto']}/{n_tot['crypto']} "
                    "encrypted — the plaintext-vs-crypto shape contrast partly reflects harness "
                    "saturation, not gateway service time"
                )
            else:
                disclosures.append(
                    f"{n_lim['none'] + n_lim['crypto']}/{n_tot['none'] + n_tot['crypto']} curves "
                    "are harness-limited (sender-saturated)"
                )
    if disclosures:
        notes.append("; ".join(disclosures))

    blast = T.BLAST_LATENCY_NOTE
    if shm_present:
        # p50-normalization already cancels the absolute inflation, but say why SHM's median is high.
        blast += "; " + T.SHM_STALL_NOTE
    notes.append(blast)

    import matplotlib.pyplot as plt

    n = len(facets)
    ncol = n if n <= 3 else (2 if n == 4 else 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol + 1.6, 3.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.ravel()

    xt_cand = [1, 1.25, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    if xmax > 8:  # on a wide log range 1.25× and 1.5× collide — keep the coarser one
        xt_cand.remove(1.25)
    xticks = [t for t in xt_cand if 1.0 <= t <= xmax]
    yt = [0.5, 0.1, 0.05, 0.01, 0.001]
    ytl = ["p50", "p90", "p95", "p99", "p999"]  # CCDF level == which tail percentile

    for i, (tr, curves) in enumerate(facets):
        ax = axes_flat[i]
        row, col = divmod(i, ncol)
        is_left = col == 0
        is_bottom = (i + ncol) >= n  # no populated subplot below this one

        _draw_bands(ax, xmax)
        n_proto = 0
        for proto, mult, ccdf in curves:
            ax.plot(mult, ccdf, marker="o", ms=3.5, lw=1.4,
                    color=T.protocol_color(proto), label=protocol_label(proto), zorder=3)
            n_proto += 1

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.95, xmax)
        ax.set_ylim(6e-4, 0.7)
        ax.minorticks_off()
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:g}×" for t in xticks], fontsize=T.FS["tick"])
        ax.set_yticks(yt)
        # sharey=True shares ONE formatter across the grid: setting [] on any inner
        # panel used to wipe the leftmost panels' labels too. Set the labels once on
        # the shared formatter and gate visibility per-axes instead.
        ax.set_yticklabels(ytl, fontsize=T.FS["tick"])
        ax.tick_params(labelleft=is_left)
        ax.grid(True, which="major", axis="both")
        T.panel_title(ax, f"{transport_label(tr)}  ({n_proto} proto)")
        if is_left:
            ax.set_ylabel("tail percentile (CCDF, log)")
        if is_bottom:
            ax.set_xlabel("latency ÷ own p50 (median), log")

    # Hide any trailing empty grid cells.
    for j in range(n, nrow * ncol):
        axes_flat[j].set_visible(False)

    # Shared protocol legend (security-ladder order) + the tail-band key, stacked to the right.
    proto_handles = [
        plt.matplotlib.lines.Line2D([0], [0], marker="o", ms=5, lw=1.4,
                                    color=T.protocol_color(p), label=protocol_label(p))
        for p in T.PROTOCOL_ORDER if p in seen_protos
    ]
    # Legend for every band that is actually PAINTED on the axes (the axis range decides,
    # not the data): _draw_bands clips to the axis xmax, so a shaded region can extend past
    # the widest curve — and a visible colour must be keyed even if no curve enters it.
    band_handles = [
        plt.matplotlib.patches.Patch(facecolor=col, alpha=0.35, edgecolor=col, label=name)
        for (lo, _hi, name, col) in _BANDS
        if lo <= 1.0 or xmax > lo
    ]
    T.legend_right(fig, proto_handles, title="protocol")
    # Second stacked key: two margin-reserving "outside right upper" legends would
    # overprint each other, so anchor this one below the protocol key inside the same
    # reserved right margin (an explicit bbox_to_anchor opts it out of reserving again).
    T.legend_right(fig, band_handles, title="tail band (× median)",
                   bbox_to_anchor=(0.998, 0.56))

    T.set_headline(fig, f"{TITLE} — {T.fmt_bytes(size)}B payload")

    # Reserve room under the bottom facet row: the constrained-layout engine (theme default)
    # ignores subplots_adjust and lays axes out over fig.text, so confine the axes to the
    # region above the note band instead (tick labels/xlabels stay inside the rect).
    y0 = 0.036 + 0.016 * max(0, len(notes) - 2)
    reserve = y0 + 0.018
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, reserve, 1.0, 1.0 - reserve))
    else:
        fig.subplots_adjust(bottom=reserve + 0.55 / (3.4 * nrow))
    for i, note_line in enumerate(notes):
        T.add_method_note(fig, note_line, y=y0 - 0.016 * i)

    note = "open-loop blast latency is CO-uncorrected; tails reflect queueing — compare SHAPES, not absolute µs"
    T.add_provenance(fig, bundle.caption() + f"  ·  {bundle.label}  ·  " + note)

    if "crypto" in tail_by_kind and "none" in tail_by_kind:
        none_best = tail_by_kind["none"]
        lo, hi = rep_spread.get(none_best["scenario"], (none_best["max"], none_best["max"]))
        # Quote the rep range when reps disagree — one number would hide the heterogeneity
        # that made the old mean-of-percentiles value an artifact (audit F7-2).
        rep_txt = f"; reps {lo:.1f}–{hi:.1f}×" if lo > 0 and hi / lo > 1.5 else ""
        lim_txt = ", harness-limited" if lim_map.get(none_best["scenario"], False) else ""
        T.add_takeaway(
            fig,
            f"Encrypted tails stay within ~{tail_by_kind['crypto']['max']:.1f}× their median on "
            "every transport — no transport adds a pathological crypto tail; the widest relative "
            f"tail is plaintext {transport_label(none_best['transport'])} at "
            f"~{none_best['max']:.1f}× its {_fmt_us(none_best['p50_us'])} median "
            f"(median rep{rep_txt}{lim_txt}).",
        )
    elif tail_by_kind:
        worst = max(v["max"] for v in tail_by_kind.values())
        T.add_takeaway(
            fig,
            f"Across transports the p999 reaches at most ~{worst:.1f}× the median — tails are "
            "tight; the gateway adds no pathological latency tail.",
        )

    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
