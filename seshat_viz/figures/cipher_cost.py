"""
F20 — Cipher-suite (AEAD) cost: AES-128-GCM vs AES-256-GCM vs ChaCha20-Poly1305.

At a fixed protocol and payload, SESHAT sweeps the AEAD cipher suite — a clean algorithm-only
comparison that no other figure surfaces. Rather than collapse to a single cell, this figure
plots the whole cipher sweep as a grid: one **column per protocol** (TLS 1.2 ECDHE-RSA, TLS 1.3,
DTLS, kTLS) × one **row per metric** (throughput, Gbps per core, and on a perf run
cycles-per-byte), with grouped bars over the swept **message sizes** and one bar per cipher.
Reading across the size groups and across the columns shows directly that the AEAD ranking is
payload- and protocol-independent: on AES-NI hardware the GCM suites ride the instruction set
while ChaCha20-Poly1305 (the constant-time software AEAD) delivers ~30 % less throughput (a
~40-50 % GCM advantage) at every size and version. Harness-limited cells (both ciphers pinned at
the load-generator ceiling) are hatched and excluded from that median — a compressed gap there
measures the harness, not the AEAD. The header still names the single best matched cell so the
headline number has an unambiguous operating point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import PROTOCOL_ORDER, RunBundle, protocol_label, transport_label

FIG_ID = "F20"
NAME = "f20_cipher_cost"
TITLE = "Cipher-suite (AEAD) cost across protocol & payload"

# Scoped category palette for the AEAD families (fixed T.CATEGORY order, never cycled).
_CIPHER_COLOR = {
    "AES-128-GCM": T.CATEGORY[0],
    "AES-256-GCM": T.CATEGORY[1],
    "ChaCha20-Poly1305": T.CATEGORY[2],
}
# Canonical cipher order (GCM before the software AEAD) for stable bar placement.
_CIPHER_ORDER = ["AES-128-GCM", "AES-256-GCM", "ChaCha20-Poly1305"]


def _limited_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean per-row harness-limited flag; all-False when the column is absent.
    (The summary column is object-typed and may carry NA — normalize deliberately.)"""
    if "harness_limited" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["harness_limited"].fillna(False).astype(bool)


def _cell_gaps(tbl: pd.DataFrame) -> list[float]:
    """Best-GCM-over-ChaCha throughput advantage per (protocol, transport, size) cell, as
    fractions. Cells with any harness-limited row are excluded: when both ciphers sit at the
    load-generator ceiling the compressed gap measures the harness, not the AEAD (F20-1)."""
    gaps: list[float] = []
    keys = [c for c in ("protocol", "transport", "message_bytes") if c in tbl.columns]
    for _, g in tbl.groupby(keys, observed=True) if keys else [((), tbl)]:
        if _limited_mask(g).any():
            continue
        gcm = g[g["cipher_label"].str.contains("GCM", case=False)]["throughput_gbps_mean"].max()
        cha = g[g["cipher_label"].str.contains("ChaCha", case=False)]["throughput_gbps_mean"].max()
        if np.isfinite(gcm) and np.isfinite(cha) and cha > 0:
            gaps.append(float(gcm / cha - 1.0))
    return gaps


def _gcm_vs_chacha_gap(tbl: pd.DataFrame) -> float:
    """Median GCM-over-ChaCha throughput advantage (%) across the trusted (protocol, size)
    cells — the invariance headline. NaN if either family is absent everywhere trusted."""
    gaps = _cell_gaps(tbl)
    return float(np.median(gaps)) * 100.0 if gaps else float("nan")


def _metric_rows(full: pd.DataFrame) -> tuple[list[tuple[str, str, str | None, bool]], bool]:
    """Pick the metric rows: throughput always; Gbps/core when present; cycles/byte on a perf
    run else CPU%. Returns (rows, cycles_shown).

    The cycles row needs counters on ≥2 distinct ciphers — a perf sweep that counted only one
    AEAD (e.g. AES-128-only) cannot support a cipher comparison, so the CPU% stand-in stays
    until the sweep covers the ciphers being compared (F20-3 guard)."""
    rows: list[tuple[str, str, str | None, bool]] = [
        ("throughput_gbps_mean", "throughput (Gbps)", "throughput_gbps_ci95", False)
    ]
    if "gbps_per_core" in full.columns and full["gbps_per_core"].notna().any():
        rows.append(("gbps_per_core", "Gbps per core", None, False))
    cycles_ok = (
        "cycles_per_byte" in full.columns
        and full.loc[full["cycles_per_byte"].notna(), "cipher_label"].nunique() >= 2
    )
    if cycles_ok:
        rows.append(("cycles_per_byte", "cycles per wire byte (lower = better)", None, True))
    elif "cpu_pct_mean" in full.columns and full["cpu_pct_mean"].notna().any():
        rows.append(("cpu_pct_mean", "CPU utilisation (%, lower = better)", None, True))
    return rows, cycles_ok


def make(bundle: RunBundle, saver: T.Saver) -> None:
    d = derive.attach_bytes_from_runs(bundle.summary, bundle.runs)
    full = derive.cipher_table(d)
    if full.empty or full["cipher_label"].nunique() < 2:
        saver.record_skip(FIG_ID, NAME, "fewer than 2 cipher_* scenarios to compare")
        return

    # The single best matched cell still names the headline operating point (trusted first —
    # a harness-limited cell must never name the headline — then prefer TLS 1.3, the cell with
    # the most ciphers, then the largest size). The panels below show the full sweep.
    cells = list(full.groupby(["protocol", "transport", "message_bytes"], observed=True))

    def _score(item):
        (proto, _tr, size), g = item
        return (not _limited_mask(g).any(), str(proto) == "tls/1.3",
                g["cipher_label"].nunique(), int(size))

    (best_proto, best_tr, best_size), _best = max(cells, key=_score)

    # Facet columns = protocols present (canonical order); each column's bars are grouped by the
    # message sizes swept for that protocol, one bar per cipher. This container grows automatically
    # when DTLS / kTLS cipher sweeps are added (they arrive as extra protocol columns).
    protocols = [p for p in PROTOCOL_ORDER if p in set(full["protocol"])]
    protocols += [p for p in sorted(set(full["protocol"])) if p not in protocols]
    ciphers = [c for c in _CIPHER_ORDER if c in set(full["cipher_label"])]
    ciphers += [c for c in sorted(set(full["cipher_label"])) if c not in ciphers]

    rows, cycles_shown = _metric_rows(full)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    nrow, ncol = len(rows), len(protocols)
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(max(3.2 * ncol, 5.0), 2.9 * nrow),
        squeeze=False, sharey="row",
    )
    bw = 0.8 / max(len(ciphers), 1)

    for ri, (col, ylabel, errcol, lower_better) in enumerate(rows):
        for ci, proto in enumerate(protocols):
            ax = axes[ri][ci]
            sub = full[full["protocol"] == proto]
            sizes = sorted(set(sub["message_bytes"]))
            base = np.arange(len(sizes))
            for j, cipher in enumerate(ciphers):
                cg = sub[sub["cipher_label"] == cipher]
                vals = [
                    (cg[cg["message_bytes"] == s][col].mean() if not cg[cg["message_bytes"] == s].empty else np.nan)
                    for s in sizes
                ]
                err = None
                if errcol and errcol in sub.columns:
                    err = [
                        (cg[cg["message_bytes"] == s][errcol].mean() if not cg[cg["message_bytes"] == s].empty else np.nan)
                        for s in sizes
                    ]
                xpos = base + (j - (len(ciphers) - 1) / 2.0) * bw
                bars = ax.bar(
                    xpos, vals, width=bw,
                    color=_CIPHER_COLOR.get(cipher, T.GREYS["baseline"]),
                    edgecolor=T.GREYS["edge"], linewidth=0.4,
                    yerr=err,
                    error_kw=dict(ecolor=T.GREYS["annot"], elinewidth=0.8, capsize=2),
                )
                # Hatch harness-limited bars so a ceiling-pinned cell cannot be read as a
                # cipher property (the DTLS-facet trap, F20-1).
                limited = [
                    bool(_limited_mask(cg[cg["message_bytes"] == s]).any()) for s in sizes
                ]
                for patch, lim in zip(bars.patches, limited):
                    if lim:
                        patch.set_hatch(T.HARNESS_HATCH)
            ax.set_xticks(base)
            ax.set_xticklabels([T.fmt_bytes(s) for s in sizes], fontsize=T.FS["annot"])
            ax.grid(axis="y")
            if ri == 0:
                tr_txt = ""
                trs = sorted(set(sub["transport"]))
                if len(trs) == 1:
                    tr_txt = f" · {transport_label(str(trs[0]))}"
                T.panel_title(ax, f"{protocol_label(str(proto))}{tr_txt}")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=T.FS["small"])
            if ri == nrow - 1:
                ax.set_xlabel("message size", fontsize=T.FS["small"])

    # Shared cipher legend as a constrained-layout-aware row below the panels; the layout
    # engine reserves its band so it never collides with the bottom row's x-labels.
    legend = [Patch(facecolor=_CIPHER_COLOR.get(c, T.GREYS["baseline"]),
                    edgecolor=T.GREYS["edge"], label=c) for c in ciphers]
    any_limited = _limited_mask(full).any()
    if any_limited:
        legend.append(T.harness_legend_handle())
    T.legend_below(fig, legend)

    # Takeaway numbers come from the trusted cells only, and both directions are given: the
    # deficit is what ChaCha delivers, the advantage is what the fastest GCM suite gains —
    # "~44% more throughput at equal security" conflated the two and mislabeled the comparator
    # (best GCM is usually AES-128, not the equal-security AES-256 suite) (F20-2).
    gaps = _cell_gaps(full)
    if gaps:
        med = float(np.median(gaps))
        gap = med * 100.0
        deficit = (1.0 - 1.0 / (1.0 + med)) * 100.0
        lo, hi = min(gaps) * 100.0, max(gaps) * 100.0
        take = (f"AES-GCM rides AES-NI; ChaCha20-Poly1305 (software AEAD) delivers ~{deficit:.0f}% "
                f"less throughput than the fastest AES-GCM suite — a ~{gap:.0f}% GCM advantage that "
                f"holds at {lo:.0f}–{hi:.0f}% across every trusted (protocol, size) cell, so the "
                f"AEAD ranking is payload/protocol-independent.")
    else:
        take = "Same protocol and payload: the AEAD choice alone moves throughput and CPU cost."

    best_lbl = {"tls/1.3": "TLS 1.3", "tls/1.2": "TLS 1.2 (ECDHE-RSA)"}.get(str(best_proto), protocol_label(str(best_proto)))
    best_cell = f"{transport_label(str(best_tr))} · {best_lbl} · {T.fmt_bytes(best_size)}B"

    note = ("single-variable AEAD sweep: within each panel only the cipher changes; columns vary "
            "the protocol, bar groups vary the message size — so flat bars across sizes/columns "
            "are the invariance result, not filler.")
    if any_limited:
        lim_panels = sorted({protocol_label(str(p))
                             for p in full.loc[_limited_mask(full), "protocol"]})
        note += (f" Hatched bars ({', '.join(lim_panels)}) are harness-limited — both ciphers hit "
                 "the load-generator ceiling there, so the compressed gap is a harness artifact, "
                 "not a cipher property; those cells are excluded from the takeaway median.")

    prov = bundle.caption()
    if not cycles_shown:
        prov += "  ·  cycles/byte needs a perf run covering ≥2 ciphers"
    prov += " · CI95 on throughput"

    T.set_headline(fig, f"{TITLE}  ·  headline cell {best_cell}  ·  {bundle.label}", y=1.03)
    # The below-panels legend row occupies the band just under the canvas; start the
    # takeaway beneath it so the wrapped second line cannot overprint the legend.
    T.add_takeaway(fig, take, y=-0.06)
    T.add_method_note(fig, note)
    T.add_provenance(fig, prov)
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
