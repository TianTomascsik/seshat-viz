"""
F23 — Handshake-algorithm cost: server-auth signature and key-exchange group.

Where F20 isolates the *symmetric* AEAD cost, F23 isolates the *asymmetric* handshake cost — and
does it on two independent axes, one row each:

  1. **Server authentication** — RSA-2048 vs ECDSA-P256 certificate signature (default KEX group).
  2. **Key exchange** — X25519 vs P-256 ECDHE group (fixed ECDSA certificate).

SESHAT churns fresh TLS 1.3 connections (connrate mode) at a fixed cipher and transport, varying
only the one algorithm per row — TLS 1.3 is auth-agnostic, so nothing else moves. The sweep is
closed-loop: each client tears down and reopens back-to-back, so conns/sec ≈ clients ÷ handshake
latency — a churn rate at the swept client count, not a saturated-gateway capacity. The sweep runs
at more than one client count; the cells are keyed on `connections` (the replicate dimension) and
every count present is drawn, so the between-algorithm ratio is shown to hold across concurrency.
Left column: connections established per second (higher is better). Right column: per-connection
handshake latency p50 → p99 (lower is better). Each row's gap is the price of that one asymmetric
primitive: an ECDSA-P256 CertificateVerify is far cheaper than an RSA-2048 one; X25519 and P-256
are close (both ~128-bit, similar cost) so that row reads as "the KEX group barely moves it".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle

FIG_ID = "F23"
NAME = "f23_handshake_cost"
TITLE = "Handshake-algorithm cost: server auth & key exchange"

# Algorithm identity on the scoped CATEGORY palette (asymmetric primitives never share an
# axes with transport/protocol series here); the 4th simultaneous label takes baseline grey.
_LABEL_COLOR = {
    "ECDSA-P256": T.CATEGORY[0],
    "RSA-2048": T.CATEGORY[1],
    "X25519": T.CATEGORY[2],
    "P-256": T.GREYS["baseline"],
}

# Each family: (row title, name-token → label, canonical label order).
_FAMILIES = [
    (
        "Server auth — certificate signature",
        [("ecdsa", "ECDSA-P256"), ("rsa", "RSA-2048")],
        ["ECDSA-P256", "RSA-2048"],
    ),
    (
        "Key exchange — ECDHE group",
        [("x25519", "X25519"), ("p256", "P-256")],
        ["X25519", "P-256"],
    ),
]


def _label_for(scenario: str, tokens: list[tuple[str, str]]) -> str | None:
    s = scenario.lower()
    for tok, label in tokens:
        if tok in s:
            return label
    return None


def _isnan(v) -> bool:
    return isinstance(v, float) and np.isnan(v)


def _facet(hs: pd.DataFrame) -> tuple[pd.Series, str | None]:
    """
    The sweep's replicate dimension, as (numeric series, unit word). `connections` is the real
    one — the count of closed-loop clients; `conn_threads` exists in the schema but is unset on
    handshake_* rows. Cells must be keyed on whichever is actually populated, never on CSV row
    order: an unkeyed pick silently swaps 1-client rows into the wider-count bars whenever the
    summary is reordered.
    """
    for col, word in (("connections", "client"), ("conn_threads", "thread")):
        if col in hs.columns:
            vals = pd.to_numeric(hs[col], errors="coerce")
            if vals.notna().any():
                return vals, word
    return pd.Series(np.nan, index=hs.index), None


def _cell(sub: pd.DataFrame, label: str, fv) -> pd.Series | None:
    """The unique row for (algorithm label, facet value); None when absent. More than one match
    means an unkeyed dimension — refuse rather than let frame order pick a row."""
    m = sub["label"] == label
    if not _isnan(fv):
        m = m & (sub["facet"] == fv)
    g = sub[m]
    if len(g) > 1:
        names = ", ".join(sorted(g["scenario"].astype(str)))
        raise ValueError(f"ambiguous handshake rows for {label} @ {fv}: {names}")
    return g.iloc[0] if len(g) == 1 else None


def _latency_scale(vals: list[float]) -> str:
    """Log only pays over a wide span; on the narrow ranges these sweeps produce, matplotlib
    emits dense overlapping minor labels, and linear resolves the same dumbbells."""
    finite = [v for v in vals if np.isfinite(v) and v > 0]
    if len(finite) >= 2 and max(finite) / min(finite) > 8.0:
        return "log"
    return "linear"


def make(bundle: RunBundle, saver: T.Saver) -> None:
    summary = bundle.summary
    if summary.empty or "scenario" not in summary.columns or "conns_per_sec" not in summary.columns \
            or summary["conns_per_sec"].notna().sum() == 0:
        saver.record_skip(FIG_ID, NAME, "no connection-rate scenarios in this run")
        return

    # Select straight from the summary (not the connsetup projection): this figure's cell key
    # is `connections`, and the disclosure needs `message_bytes` — both dropped by connsetup.
    hs = summary[
        summary["scenario"].astype(str).str.startswith("handshake_") & summary["conns_per_sec"].notna()
    ].copy()
    if hs.empty:
        saver.record_skip(FIG_ID, NAME, "no handshake_* scenarios (run the handshake sweep)")
        return
    hs["facet"], facet_word = _facet(hs)

    # Keep only the families that have ≥2 labels present (a family with one member is no comparison).
    families = []
    for title, tokens, order in _FAMILIES:
        labelled = hs.assign(label=hs["scenario"].astype(str).map(lambda s: _label_for(s, tokens)))
        labelled = labelled[labelled["label"].notna()]
        present = [l for l in order if l in set(labelled["label"])]
        if len(present) < 2:
            continue
        # Two scenarios collapsing onto one (algorithm, client-count) cell means the sweep has
        # a dimension this figure does not key on — any pick would be arbitrary, so refuse.
        dup = labelled.duplicated(subset=["label", "facet"], keep=False)
        if dup.any():
            names = ", ".join(sorted(labelled.loc[dup, "scenario"].astype(str)))
            saver.record_skip(
                FIG_ID, NAME, f"ambiguous handshake rows (same algorithm + client count): {names}"
            )
            return
        families.append((title, labelled, present))
    if not families:
        saver.record_skip(
            FIG_ID, NAME, "need ≥2 algorithms in a handshake family to compare (run the handshake sweep)"
        )
        return

    have_hs = "conn_handshake_p99_us" in hs.columns and hs["conn_handshake_p99_us"].notna().any()

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch

    # Print variant: one operating point only (the widest client count) so each panel keeps
    # a single bar/dumbbell pair, and a 15 cm-friendly canvas. The comparison ratio already
    # comes from the widest count, so nothing in the takeaway changes.
    in_print = T.print_variant()
    ncol = 2 if have_hs else 1
    if in_print:
        figsize = (7.6 if have_hs else 4.6, 2.9 * len(families))
    else:
        figsize = (11.4 if have_hs else 6.2, 3.9 * len(families))
    fig, axes = plt.subplots(len(families), ncol, figsize=figsize, squeeze=False)

    takeaways: list[str] = []
    ratio_facets: set[int] = set()  # client counts the takeaway ratios were computed at
    facet_dropped = False

    for r, (title, sub, labels) in enumerate(families):
        facet_vals = sorted(v for v in sub["facet"].dropna().unique()) or [np.nan]
        if in_print and len(facet_vals) > 1:
            facet_dropped = True
            facet_vals = facet_vals[-1:]
        n = len(facet_vals)
        # The widest client count carries the headline comparison — draw it opaque.
        alphas = [1.0 - 0.35 * (n - 1 - j) / max(n - 1, 1) for j in range(n)]

        x = np.arange(len(labels))
        bw = 0.8 / max(n, 1)

        # Left: connections / second, one bar group per client count.
        axa = axes[r][0]
        for j, fv in enumerate(facet_vals):
            cells = [_cell(sub, lab, fv) for lab in labels]
            vals = [
                float(c["conns_per_sec"]) if c is not None and pd.notna(c.get("conns_per_sec")) else np.nan
                for c in cells
            ]
            errs = [
                float(c["conns_per_sec_ci95"]) if c is not None
                and "conns_per_sec_ci95" in sub.columns and pd.notna(c.get("conns_per_sec_ci95")) else np.nan
                for c in cells
            ]
            xpos = x + (j - (n - 1) / 2.0) * bw
            axa.bar(xpos, vals, width=bw,
                    color=[_LABEL_COLOR.get(l, T.GREYS["muted"]) for l in labels],
                    edgecolor=T.GREYS["edge"], linewidth=0.5, yerr=errs,
                    error_kw=dict(ecolor=T.GREYS["annot"], elinewidth=1, capsize=3),
                    alpha=alphas[j])
            for xi, v, e in zip(xpos, vals, errs):
                if np.isfinite(v):
                    T.annotate_value(axa, xi, v, f"{v:.0f}",
                                     yerr=e if np.isfinite(e) else 0.0)
        axa.set_xticks(x)
        axa.set_xticklabels(labels, fontsize=T.FS["small"])
        axa.set_ylabel("connections / second\n(higher = better)", fontsize=T.FS["small"])
        T.panel_title(axa, title if in_print else f"{title}  ·  establishment rate")
        axa.grid(axis="y")
        if n > 1 and facet_word:
            T.legend_inline(
                axa,
                [
                    Patch(facecolor=T.GREYS["muted"], alpha=alphas[j], edgecolor=T.GREYS["edge"],
                          label=f"{int(fv)} {facet_word}{'s' if int(fv) != 1 else ''}")
                    for j, fv in enumerate(facet_vals) if np.isfinite(fv)
                ],
                loc="best", title=f"closed-loop {facet_word}s",
                title_fontsize=T.FS["small"],
            )

        # Right: handshake latency p50 → p99, one dumbbell per (algorithm, client count).
        if have_hs:
            axb = axes[r][1]
            rows = []
            for lab in labels:
                for fv in facet_vals:
                    c = _cell(sub, lab, fv)
                    if c is None:
                        continue
                    p50 = float(c["conn_handshake_p50_us"]) if pd.notna(c.get("conn_handshake_p50_us")) else np.nan
                    p99 = float(c["conn_handshake_p99_us"]) if pd.notna(c.get("conn_handshake_p99_us")) else np.nan
                    sfx = f" · {int(fv)}{facet_word[0]}" if facet_word and not _isnan(fv) and n > 1 else ""
                    rows.append((f"{lab}{sfx}", lab, p50, p99))
            for i, (_lbl, lab, p50, p99) in enumerate(rows):
                col = _LABEL_COLOR.get(lab, T.GREYS["muted"])
                if np.isfinite(p50) and np.isfinite(p99):
                    axb.plot([p50, p99], [i, i], color=col, lw=2.0, zorder=1)
                if np.isfinite(p50):
                    # Hollow = p50 (the one percentile convention, suite-wide).
                    axb.scatter([p50], [i], facecolor="white", edgecolor=col, s=40,
                                linewidth=1.3, zorder=2)
                    axb.annotate(f"{p50:.0f}", (p50, i), xytext=(0, 6), textcoords="offset points",
                                 ha="center", fontsize=T.FS["annot"], color=T.GREYS["ink"])
                if np.isfinite(p99):
                    axb.scatter([p99], [i], color=col, s=40, zorder=2)
                    axb.annotate(f"{p99:.0f}", (p99, i), xytext=(0, -10), textcoords="offset points",
                                 ha="center", fontsize=T.FS["annot"], color=T.GREYS["ink"])
            axb.set_yticks(np.arange(len(rows)))
            axb.set_yticklabels([t[0] for t in rows], fontsize=T.FS["small"])
            axb.set_ylim(len(rows) - 0.5, -0.7)
            scale = _latency_scale([v for t in rows for v in (t[2], t[3])])
            if scale == "log":
                axb.set_xscale("log")
                # On a sub-decade span log minor labels overlap into garble — majors suffice.
                axb.xaxis.set_minor_formatter(mticker.NullFormatter())
            axb.set_xlabel(
                f"handshake latency µs (p50→p99{', log' if scale == 'log' else ''}, lower = better)",
                fontsize=T.FS["small"],
            )
            T.panel_title(axb, "handshake latency" if in_print else f"{title}  ·  handshake latency")
            axb.grid(axis="x", which="both", alpha=0.4)
            T.legend_inline(axb, T.percentile_handles(), loc="upper right")

        # Per-family takeaway ratio (best vs worst on conns/sec at the widest client count).
        fv = facet_vals[-1]
        c0, c1 = _cell(sub, labels[0], fv), _cell(sub, labels[-1], fv)
        if c0 is not None and c1 is not None and pd.notna(c0.get("conns_per_sec")) \
                and pd.notna(c1.get("conns_per_sec")) and float(c1["conns_per_sec"]) > 0:
            ratio = float(c0["conns_per_sec"]) / float(c1["conns_per_sec"])
            takeaways.append(f"{labels[0]} vs {labels[-1]}: {ratio:.2f}× the connection rate")
            if not _isnan(fv):
                ratio_facets.add(int(fv))

    if takeaways:
        trailer = ""
        if ratio_facets and facet_word == "client":
            counts = "/".join(str(c) for c in sorted(ratio_facets))
            trailer = f" (closed-loop, {counts} clients)"
        take = ("TLS 1.3 auth-agnostic, so each row varies one asymmetric primitive alone — "
                + "; ".join(takeaways) + trailer + ". The bulk-cipher cost is F20.")
    else:
        take = "Each row isolates one asymmetric handshake primitive (server-auth signature, key-exchange group)."

    # Headline context (protocol/transport) comes from the plotted rows, not assumptions.
    ctx_parts: list[str] = []
    if "protocol" in hs.columns:
        ctx_parts += [p.upper().replace("TLS/", "TLS ") for p in sorted(set(hs["protocol"].dropna().astype(str)))]
    if "transport" in hs.columns:
        ctx_parts += [t.upper() for t in sorted(set(hs["transport"].dropna().astype(str)))]
    ctx = ("  ·  " + " · ".join(ctx_parts)) if ctx_parts else ""

    T.set_headline(fig, f"{TITLE}{ctx}  ·  {bundle.label}")
    T.add_takeaway(fig, take)
    # Keep this note near the headline's width: savefig.bbox='tight' widens the canvas to fit
    # the longest fig.text line, so an over-long note pads the figure with dead whitespace.
    method = (
        "connrate churn through the gateway; each row varies ONE primitive — top: server-cert "
        "signature (RSA vs ECDSA); bottom: ECDHE group (X25519 vs P-256), pinned via the "
        "allowlist-validated `groups` parameter (only named groups are accepted). "
        "Closed-loop clients reconnect back-to-back: "
        "conns/sec ≈ clients ÷ handshake latency, a churn rate at the plotted client count — "
        "not gateway capacity."
    )
    if facet_dropped:
        method += " Print variant: only the widest client count is drawn; narrower sweeps agree."
    T.add_method_note(fig, method)
    prov = "handshake churn = fresh connections, closed-loop"
    if "connections" in hs.columns:
        clients = sorted({int(v) for v in pd.to_numeric(hs["connections"], errors="coerce").dropna()})
        if clients:
            prov += f" × {'/'.join(str(c) for c in clients)} client(s)"
    if "message_bytes" in hs.columns:
        msgs = sorted({int(v) for v in pd.to_numeric(hs["message_bytes"], errors="coerce").dropna()})
        if msgs:
            prov += f", {'/'.join(str(m) for m in msgs)} B message"
    T.add_provenance(fig, bundle.caption() + "  ·  " + prov)
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
