"""
F4 — Protocol × payload-size heatmaps.

A dense at-a-glance matrix of the whole measured space: for each transport with enough
coverage, three annotated heatmaps side by side — throughput (Gbps), latency (µs), and
jitter/PDV (µs) — with rows = protocol (security ladder) and columns = message size.
Hot/cold cells expose exactly where each protocol wins or struggles, and which cells were measured.

Why three panels: throughput and blast p99 both look near-flat across sizes for the encrypted rows
— crypto throughput because per-core AES-GCM is size-independent above 256 B (only *routing* scales
with size), and blast p99 because it is a near-monotone restatement of 1/throughput (queue backlog,
not service time). **Jitter (PDV) is the de-skewed complement**: it is not queueing-dominated (it
stays orders of magnitude below the ms-scale blast tail), yet it genuinely varies with size and
separates the protocols — the honest per-message-delivery-consistency view the flat panels hide.
Where a run carries the closed-loop ``matrix_lat_*`` RTT grid, the latency column shows that honest
ping-pong RTT instead of blast p99 (F16 remains the closed-loop deep-dive).

Honesty guards: every number a caption claims (routing's size-scaling factor, the encrypted rows'
residual growth, the jitter band and protocol separation) is recomputed from the rendered pivots at
render time — never hardcoded; cells that fell back to a 1-gateway direct measurement under the
scg-preference dedup are daggered (†) so no panel silently mixes topologies; jitter claims are
scoped to the transports that actually carry PDV data; and the RTT column's chain (typically
scg-direct, i.e. single-gateway) is disclosed next to the scg-preferred throughput column.

Thesis-review refinements retained: (V5) a takeaway + method note make explicit that ROUTING scales
with payload size while the encrypted rows stay near-flat, a fact the per-panel LogNorm color-
compresses so it is only legible in the annotated cell values; (V6) the SHM blast-p99 panel is
flagged and the figure carries ``theme.SHM_STALL_NOTE`` because its multi-ms cells are a harness
receive-poll stall, not SHM transport latency (throughput is unaffected).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F4"
NAME = "f04_protocol_size_heatmaps"
TITLE = "Protocol × size heatmaps: throughput and latency"

_MAX_TRANSPORTS = 5  # render all transports (TCP/UDP/UDS/SHM/TPROXY); blank cells show gaps

# Transports whose data plane is datagram-bound: EVERY row there (routing and encrypted
# alike) scales with datagram size, so the "encrypted rows are near-flat" claim and the
# routing size-scaling range are computed over stream transports only and the datagram
# side is reported separately (F4-4).
_DATAGRAM_TRANSPORTS = {"udp"}


def _coverage_score(grp: pd.DataFrame) -> tuple[int, int]:
    return (grp["protocol"].nunique(), grp["message_bytes"].nunique())


def _has_cells(piv: pd.DataFrame | None) -> bool:
    """True iff the pivot has at least one finite cell. seaborn.heatmap calls np.nanmin on
    the raw array and raises `zero-size array to reduction operation fmin` on an all-NaN or
    zero-size pivot (before our LogNorm can return None), so an empty per-transport panel
    must be detected and drawn blank rather than sinking the whole figure."""
    if piv is None:
        return False
    arr = piv.to_numpy(dtype=float)
    return arr.size > 0 and bool(np.isfinite(arr).any())


def _pivot(sub: pd.DataFrame, value: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """protocol (ladder-ordered rows) × size (ascending cols) pivot, scg preferred.

    Returns ``(pivot, fallback)``: ``fallback`` flags cells whose value came from a non-scg
    chain (1-gateway direct) because no 2-gateway scg row covered that (protocol, size).
    The caller daggers those cells so a panel never silently mixes topologies (F4-1 — e.g.
    a routing row measured direct sitting next to scg-measured crypto rows)."""
    s = sub.copy()
    if "chain" in s.columns:
        # Prefer the 2-gateway (scg-scg) path; fall back to 1-gateway direct where scg is
        # absent. Stable sort so ties resolve deterministically (not by frame order).
        s["__rank"] = (s["chain"] == "scg").astype(int)
        s = s.sort_values("__rank", ascending=False, kind="stable").drop_duplicates(["protocol", "message_bytes"])
        s["__fallback"] = (s["chain"].notna() & (s["chain"].astype(str) != "scg")).astype(float)
    else:
        s["__fallback"] = 0.0
    piv = s.pivot_table(index="protocol", columns="message_bytes", values=value, observed=True, aggfunc="mean")
    rows = [p for p in T.PROTOCOL_ORDER if p in piv.index]
    rows += [p for p in piv.index if p not in rows]
    piv = piv.reindex(rows)
    piv = piv.reindex(sorted(piv.columns), axis=1)
    fb = s.pivot_table(index="protocol", columns="message_bytes", values="__fallback", observed=True, aggfunc="max")
    fb = fb.reindex(index=piv.index, columns=piv.columns).fillna(0.0).astype(bool) & piv.notna()
    return piv, fb


def _annot(piv: pd.DataFrame, mask: pd.DataFrame | None, fmt: str):
    """Annotation spec for seaborn.heatmap: ``(annot, fmt, flagged)``.

    Normally numeric (``annot=True`` + a format string). When ``mask`` flags cells whose
    value fell back to a non-scg (1-gateway direct) measurement, switch to explicit string
    labels with a '†' suffix on the flagged cells — the per-cell disclosure that keeps a
    panel from presenting a mixed-topology grid as uniform (F4-1)."""
    if mask is None:
        return True, fmt, False
    flags = mask.to_numpy(dtype=bool)
    if flags.shape != piv.shape or not flags.any():
        return True, fmt, False
    vals = piv.to_numpy(dtype=float)
    labels = np.asarray(
        [
            [
                (format(vals[i, j], fmt) + ("†" if flags[i, j] else "")) if np.isfinite(vals[i, j]) else ""
                for j in range(vals.shape[1])
            ]
            for i in range(vals.shape[0])
        ],
        dtype=object,
    )
    return labels, "", True


def _fmt_x(ratio: float) -> str:
    """Compact scaling-factor text: one decimal below 10×, integer above."""
    return f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}"


def _fmt_us(value: float) -> str:
    """Compact µs text spanning sub-µs jitter to hundreds of µs."""
    if value < 1:
        return f"{value:.2f}"
    if value < 10:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _tput_clause(pivots: dict[str, pd.DataFrame]) -> str:
    """Computed throughput takeaway clause. The routing size-scaling factor and the
    encrypted rows' residual growth are derived from exactly the pivots rendered (never
    hardcoded — F4-4), and both claims are scoped to stream transports because datagram
    paths are size-bound for every row; the datagram routing factor is reported apart."""
    routing: dict[str, float] = {}
    enc_growth: list[float] = []
    for t, piv in pivots.items():
        if piv.empty:
            continue
        if "none" in piv.index:
            v = piv.loc["none"].to_numpy(dtype=float)
            v = v[np.isfinite(v) & (v > 0)]
            if v.size >= 2:
                routing[t] = float(v.max() / v.min())
        if t not in _DATAGRAM_TRANSPORTS:
            enc_rows = [p for p in piv.index if str(p) != "none"]
            enc_cols = [c for c in piv.columns if float(c) >= 256]
            if enc_rows and len(enc_cols) >= 2:
                for _, row in piv.loc[enc_rows, enc_cols].iterrows():
                    v = row.to_numpy(dtype=float)
                    v = v[np.isfinite(v) & (v > 0)]
                    if v.size >= 2:
                        enc_growth.append(float(v.max() / v.min()))
    stream = {t: r for t, r in routing.items() if t not in _DATAGRAM_TRANSPORTS}
    dgram = {t: r for t, r in routing.items() if t in _DATAGRAM_TRANSPORTS}
    segs: list[str] = []
    if stream:
        lo, hi = min(stream.values()), max(stream.values())
        span = f"~{_fmt_x(lo)}×" if _fmt_x(lo) == _fmt_x(hi) else f"~{_fmt_x(lo)}–{_fmt_x(hi)}×"
        seg = f"routing scales {span} with payload size on stream transports (see values)"
        if dgram:
            worst = max(dgram, key=lambda k: dgram[k])
            seg += f", {_fmt_x(dgram[worst])}× on {transport_label(worst)} (datagram paths are size-bound)"
        segs.append(seg)
    elif dgram:
        worst = max(dgram, key=lambda k: dgram[k])
        segs.append(f"routing scales {_fmt_x(dgram[worst])}× with datagram size on {transport_label(worst)}")
    if enc_growth:
        enc = (
            f"encrypted rows stay near-flat above 256 B (≤{max(enc_growth):.2f}× on stream transports) — "
            "per-core AES-GCM is size-independent (crypto cost is per-message, not per-byte)"
        )
        segs.append(("while " if segs else "") + enc)
    if not segs:
        return ""
    return "Throughput: " + " ".join(segs) + ". "


def _jitter_clause(pivots: dict[str, pd.DataFrame], transports: list[str]) -> str:
    """Computed jitter takeaway clause. The magnitude band, the grows-with-size claim and
    the protocol-separation extremes all come from the rendered jitter pivots, and the whole
    clause is scoped to the transports that actually carry PDV data — a run may populate
    jitter for a single transport, and generalizing that to the figure is dishonest (F4-2)."""
    recs: list[tuple[str, str, float, float]] = []
    for t, piv in pivots.items():
        for proto, row in piv.iterrows():
            for size, v in row.items():
                v = float(v)
                if np.isfinite(v):
                    recs.append((t, str(proto), float(size), v))
    if not recs:
        return ""
    jl = pd.DataFrame(recs, columns=["transport", "protocol", "size", "val"])
    jmin, jmax = float(jl["val"].min()), float(jl["val"].max())
    band = f"~{_fmt_us(jmax)} µs" if _fmt_us(jmin) == _fmt_us(jmax) else f"{_fmt_us(jmin)}–{_fmt_us(jmax)} µs"
    clause = f" Jitter (PDV, right) is the de-skewed signal: it stays {band}"
    if jmax < 1000.0:
        clause += " (not the ms queueing)"
    # Grows-with-size: largest-size / smallest-size value per (transport, protocol) row —
    # only claim growth when the median row actually at least doubles.
    growth = (
        jl.sort_values("size")
        .groupby(["transport", "protocol"])["val"]
        .agg(lambda v: v.iloc[-1] / v.iloc[0] if len(v) >= 2 and v.iloc[0] > 0 else np.nan)
        .dropna()
    )
    grows = (not growth.empty) and float(growth.median()) >= 2.0
    # Protocol separation at the largest size with ≥2 protocols measured: name the actual
    # extremes rather than asserting a fixed ordering; only claim separation when the
    # spread is at least 2×.
    sep = ""
    counts = jl.groupby("size")["protocol"].nunique()
    sizes = counts[counts >= 2].index
    if len(sizes):
        big = float(sizes.max())
        at = jl[jl["size"] == big].groupby("protocol")["val"].mean()
        hi_p, lo_p = str(at.idxmax()), str(at.idxmin())
        hi_v, lo_v = float(at.max()), float(at.min())
        if lo_v > 0 and hi_v / lo_v >= 2.0:
            sep = (
                f"separate the protocols at {T.fmt_bytes(big)}B "
                f"({protocol_label(hi_p)} {_fmt_us(hi_v)} µs vs {protocol_label(lo_p)} {_fmt_us(lo_v)} µs)"
            )
    if grows and sep:
        clause += f" yet DOES grow with size and {sep}"
    elif grows:
        clause += " yet DOES grow with size"
    elif sep:
        clause += " and does " + sep
    covered = [t for t in transports if t in pivots]
    if covered and set(covered) != set(transports):
        clause += " — measured only on " + "/".join(transport_label(t) for t in covered) + " in this run"
    return clause + "."


def make(bundle: RunBundle, saver: T.Saver) -> None:
    df = bundle.summary
    if not {"transport", "protocol", "message_bytes", "throughput_gbps_mean"}.issubset(df.columns):
        saver.record_skip(FIG_ID, NAME, "needs transport/protocol/size/throughput")
        return

    d = df[df["message_bytes"].notna() & df["protocol"].notna()].copy()
    fam = d["family"].astype(str) if "family" in d.columns else pd.Series("", index=d.index)
    # Closed-loop RTT grid (`matrix_lat_*` ping-pong) — the honest per-message latency that
    # replaces the queueing-dominated blast p99 *where it was measured*. Captured BEFORE the
    # matrix-only filter and kept in a separate frame so its rtt-only rows (near-zero throughput)
    # never pollute the throughput/jitter panels. Empty until a run carries the matrix-latency grid.
    d_lat = d[fam == "matrix-latency"].copy()
    # Restrict the main frame to the matrix (throughput) family (sustained-blast, explicit
    # topology). Otherwise guessed-chain iface/hotreload/conn rows displace real matrix cells.
    if "family" in d.columns and (fam == "matrix").any():
        d = d[fam == "matrix"]
    # Single-connection anchor so each cell is per-message, not a 1c..1024c blend (both frames).
    if "connections" in d.columns:
        d = d[d["connections"].isin([1]) | d["connections"].isna()].copy()
    if not d_lat.empty and "connections" in d_lat.columns:
        d_lat = d_lat[d_lat["connections"].isin([1]) | d_lat["connections"].isna()].copy()
    # Rank transports by coverage; keep the richest few.
    scored = sorted(
        ((str(t), _coverage_score(g)) for t, g in d.groupby("transport", observed=True)),
        key=lambda kv: kv[1], reverse=True,
    )
    transports = [t for t, sc in scored if sc[0] >= 2 and sc[1] >= 2][:_MAX_TRANSPORTS]
    if not transports:
        # Fall back: any transport with ≥2 protocols (single size is still informative).
        transports = [t for t, sc in scored if sc[0] >= 2][:_MAX_TRANSPORTS]
    if not transports:
        saver.record_skip(FIG_ID, NAME, "no transport has ≥2 protocols to compare")
        return

    have_lat = "latency_p99_us_mean" in d.columns and d["latency_p99_us_mean"].notna().any()
    have_rtt = (not d_lat.empty) and "rtt_us_p99" in d_lat.columns and d_lat["rtt_us_p99"].notna().any()
    have_latcol = have_lat or have_rtt  # the latency column shows closed-loop RTT, else blast p99
    have_jit = "jitter_us_mean" in d.columns and d["jitter_us_mean"].notna().any()
    ncol = 1 + int(have_latcol) + int(have_jit)

    # Which chains fill the RTT grid (F4-3): typically single-gateway scg-direct, sitting
    # next to a throughput column that prefers the 2-gateway scg path — that topology
    # mismatch must be disclosed, and the "scg path preferred" footer scoped away from it.
    lat_chains: set[str] = set()
    if have_rtt and "chain" in d_lat.columns:
        lat_chains = set(d_lat.loc[d_lat["rtt_us_p99"].notna(), "chain"].dropna().astype(str))

    # Per-transport slices and pivots, computed once and reused for panel sizing, rendering
    # and the takeaway — so every caption claim derives from exactly what is rendered.
    subs = {t: d[d["transport"].astype(str) == t] for t in transports}
    pivots_t = {t: _pivot(subs[t], "throughput_gbps_mean") for t in transports}
    pivots_j = {t: _pivot(subs[t], "jitter_us_mean") for t in transports} if have_jit else {}

    # scg-preference bookkeeping (F4-1): the _pivot dedup prefers the 2-gateway scg path and
    # silently falls back to 1-gateway direct cells where scg was never run. Dagger those
    # cells — but only when scg rows exist in the slice at all; with no scg anywhere nothing
    # was "preferred" and the footer drops the claim instead.
    scg_present = "chain" in d.columns and bool((d["chain"].astype(str) == "scg").any())
    any_fallback = False

    # Takeaway inputs — computed BEFORE the render loop relabels the pivots in place.
    tput_clause = _tput_clause({t: piv for t, (piv, _m) in pivots_t.items()})
    jit_pivs = {t: piv for t, (piv, _m) in pivots_j.items() if _has_cells(piv)}
    jit_clause = _jitter_clause(jit_pivs, transports) if have_jit else ""

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LogNorm

    def _lognorm(piv):
        """LogNorm spanning the positive values so a single outlier can't wash out the rest."""
        vals = piv.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return None
        return LogNorm(vmin=vals.min(), vmax=vals.max())

    def _blank(ax):
        """Empty-panel stand-in for a transport that has no finite cells for this metric."""
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=T.FS["small"],
                style="italic", color=T.GREYS["muted"], transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    nrow = len(transports)
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(5.4 * ncol, 0.5 * sum(pivots_t[t][0].shape[0] for t in transports) + 1.6 * nrow),
        squeeze=False,
    )

    for r, transport in enumerate(transports):
        sub = subs[transport]
        piv_t, mask_t = pivots_t[transport]
        piv_t.columns = [f"{T.fmt_bytes(c)}B" for c in piv_t.columns]
        piv_t.index = [protocol_label(str(p)) for p in piv_t.index]
        if _has_cells(piv_t):
            annot_t, fmt_t, flagged = _annot(piv_t, mask_t if scg_present else None, ".2f")
            any_fallback |= flagged
            sns.heatmap(
                piv_t, ax=axes[r][0], annot=annot_t, fmt=fmt_t, cmap="viridis", norm=_lognorm(piv_t),
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Gbps (log color)"},
                annot_kws={"fontsize": T.FS["annot"]},
            )
        else:
            _blank(axes[r][0])
        T.panel_title(axes[r][0], f"{transport_label(transport)} — throughput (Gbps)")
        axes[r][0].set_xlabel("message size")
        axes[r][0].set_ylabel("")

        if have_latcol:
            # Prefer the honest closed-loop RTT (matrix_lat_* ping-pong) per cell; fall back to the
            # open-loop blast p99 (queueing-dominated) only where RTT was not measured.
            piv_blast, mask_blast = _pivot(sub, "latency_p99_us_mean") if have_lat else (None, None)
            sub_lat = d_lat[d_lat["transport"].astype(str) == transport] if have_rtt else None
            piv_rtt = _pivot(sub_lat, "rtt_us_p99")[0] if (sub_lat is not None and not sub_lat.empty) else None
            mask_l = None
            if piv_rtt is not None and piv_blast is not None:
                piv_l = piv_rtt.combine_first(piv_blast)
                # Only say "blast where uncovered" if blast ACTUALLY filled a cell (rtt missing
                # where blast is present) — otherwise the panel is pure closed-loop RTT and the
                # caveat would misleadingly imply blast is mixed in.
                a_rtt = piv_rtt.reindex(index=piv_l.index, columns=piv_l.columns)
                a_blast = piv_blast.reindex(index=piv_l.index, columns=piv_l.columns)
                used_blast = bool((a_rtt.isna() & a_blast.notna()).to_numpy().any())
                lat_kind = "closed-loop RTT (blast where uncovered)" if used_blast else "closed-loop RTT"
                closed = not used_blast  # `closed` = no blast cells shown → suppress the SHM-stall note
            elif piv_rtt is not None:
                piv_l, lat_kind, closed = piv_rtt, "closed-loop RTT", True
            else:
                piv_l, lat_kind, closed = piv_blast, "open-loop blast", False
                mask_l = mask_blast  # pure-blast panel: same scg-preference dedup → same disclosure
            if not _has_cells(piv_l):
                # This transport measured neither closed-loop RTT nor blast p99 — blank panel.
                _blank(axes[r][1])
            else:
                # combine_first re-sorts on the index/columns — restore ladder rows + ascending sizes.
                rows = [p for p in T.PROTOCOL_ORDER if p in piv_l.index]
                rows += [p for p in piv_l.index if p not in rows]
                piv_l = piv_l.reindex(rows).reindex(sorted(piv_l.columns), axis=1)
                if mask_l is not None:
                    mask_l = mask_l.reindex(index=piv_l.index, columns=piv_l.columns).fillna(False).astype(bool)
                piv_l.columns = [f"{T.fmt_bytes(c)}B" for c in piv_l.columns]
                piv_l.index = [protocol_label(str(p)) for p in piv_l.index]
                annot_l, fmt_l, flagged = _annot(piv_l, mask_l if scg_present else None, ".0f")
                any_fallback |= flagged
                sns.heatmap(
                    piv_l, ax=axes[r][1], annot=annot_l, fmt=fmt_l, cmap="rocket_r", norm=_lognorm(piv_l),
                    linewidths=0.5, linecolor="white", cbar_kws={"label": "µs (log color)"},
                    annot_kws={"fontsize": T.FS["annot"]},
                )
            lat_title = f"{transport_label(transport)} — latency (µs) · {lat_kind}"
            if transport == "shm" and not closed:
                # V6: SHM's multi-ms BLAST p99 cells are a harness receive-poll stall, not SHM
                # latency; flag them (closed-loop RTT does not have this artifact).
                lat_title += " · ms cells = harness stall, not SHM"
            T.panel_title(axes[r][1], lat_title)
            axes[r][1].set_xlabel("message size")
            axes[r][1].set_ylabel("")

        if have_jit:
            # Jitter (PDV) is the de-skewed complement to the blast p99: it is NOT queueing-dominated
            # (it stays orders of magnitude below the ms-scale blast tail), and — unlike the flat
            # crypto throughput — it varies with size and separates the protocols, so it exposes
            # the per-message-delivery-consistency structure the other two panels hide. A distinct
            # colormap keeps it visually apart from throughput (viridis) and latency (rocket_r).
            col_j = 1 + int(have_latcol)
            piv_j, mask_j = pivots_j[transport]
            piv_j.columns = [f"{T.fmt_bytes(c)}B" for c in piv_j.columns]
            piv_j.index = [protocol_label(str(p)) for p in piv_j.index]
            if _has_cells(piv_j):
                # jitter (PDV) is only populated for some transports (e.g. tproxy in this run);
                # a transport with no jitter cells gets a blank panel, not a crash.
                annot_j, fmt_j, flagged = _annot(piv_j, mask_j if scg_present else None, ".2f")
                any_fallback |= flagged
                sns.heatmap(
                    piv_j, ax=axes[r][col_j], annot=annot_j, fmt=fmt_j, cmap="crest", norm=_lognorm(piv_j),
                    linewidths=0.5, linecolor="white", cbar_kws={"label": "µs (log color)"},
                    annot_kws={"fontsize": T.FS["annot"]},
                )
            else:
                _blank(axes[r][col_j])
            T.panel_title(axes[r][col_j], f"{transport_label(transport)} — jitter (µs, PDV)")
            axes[r][col_j].set_xlabel("message size")
            axes[r][col_j].set_ylabel("")

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.01)
    # V5: each panel's LogNorm compresses the row's own spread, so the fact that ROUTING
    # (protocol=none) scales strongly with payload size while the crypto rows stay near-flat is only
    # visible in the printed cell values — state it outright, with the factors recomputed from the
    # rendered pivots (never hardcoded), rather than leaving it to the color.
    if have_rtt:
        rtt_scope = ", single-gateway scg-direct" if lat_chains == {"direct"} else ""
        take = tput_clause + (
            f"The latency panel is closed-loop RTT (ping-pong{rtt_scope}) — per-message service "
            "time, not the queueing-dominated open-loop blast (blast p99 fills only cells the RTT grid "
            "did not cover)."
        )
    else:
        take = tput_clause + (
            "Blast p99 just mirrors 1/throughput (queueing, not service time — closed-loop RTT is F16)."
        )
    take += jit_clause
    T.add_takeaway(fig, take)
    # Method note: each heatmap is LogNorm'd on its own value range, so color is a within-panel
    # comparison only — the routing row's size-scaling must be read from the annotated values.
    # Append the SHM harness-stall caveat (V6) whenever an SHM latency panel is actually drawn.
    note_parts = []
    if have_rtt:
        rtt_note = (
            "latency = closed-loop ping-pong RTT (matrix_lat_* grid) — per-message service "
            "time where measured; open-loop blast p99 (queueing) only fills uncovered cells"
        )
        if lat_chains == {"direct"}:
            # F4-3: the RTT grid runs single-gateway while the throughput column prefers the
            # 2-gateway scg path — same-row panels differ in topology, say so.
            rtt_note += (
                "; the RTT grid is single-gateway (scg-direct) — the throughput column prefers the "
                "2-gateway scg path, so same-row panels differ in topology"
            )
        elif len(lat_chains) > 1:
            rtt_note += "; the RTT grid mixes chains (" + ", ".join(sorted(lat_chains)) + ")"
        note_parts.append(rtt_note)
    elif have_lat:
        note_parts.append(T.BLAST_LATENCY_NOTE)
    if have_jit:
        jit_note = (
            "jitter = packet-delay variation (PDV) — a per-message delivery-consistency signal, "
            "the de-skewed complement to the queueing-dominated blast p99"
        )
        if jit_pivs and set(jit_pivs) != set(transports):
            # F4-2: PDV coverage is partial — name the transports that carry it so the jitter
            # column (and any conclusion drawn from it) is read at its true scope.
            jit_note += (
                "; populated only for "
                + "/".join(transport_label(t) for t in transports if t in jit_pivs)
                + " in this run"
            )
        note_parts.append(jit_note)
    note_parts.append(
        "each panel is independently LogNorm-scaled (color compares within a panel, not across) — "
        "read the annotated values for the routing row's size-scaling"
    )
    if have_lat and not have_rtt and "shm" in transports:
        note_parts.append(T.SHM_STALL_NOTE)
    T.add_method_note(fig, "  ·  ".join(note_parts))
    foot = "blank cells = not measured"
    if scg_present:
        pref = "scg path preferred"
        if have_rtt and lat_chains and lat_chains != {"scg"}:
            # The RTT column has no scg cells to prefer — scope the claim to the columns it
            # actually governs (F4-3).
            pref += " (throughput/jitter panels)" if have_jit else " (throughput panels)"
        foot += " · " + pref
    if any_fallback:
        foot += " · † = 1-gateway direct (no 2-gateway scg run for that cell)"
    T.add_provenance(fig, bundle.caption() + "  ·  " + foot)
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
