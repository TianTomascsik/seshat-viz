"""
F24 — Traffic-class isolation under bulk contention (the QoS-isolation campaign).

The qos_isolation suite pairs one paced safety stream (256 B every 200 µs, DSCP EF,
coordinated-omission-corrected scheduled-time stamping) with four sustained best-effort
bulk blast streams through class-aware gateway rules, for three protection modes and
three conditions each:

  * alone         — the paced stream with no contention (the baseline service latency);
  * contended     — the same stream declared traffic_class=safety next to the bulk load,
                    so the gateway provisions a dedicated safety rule pair and pool;
  * unclassified  — byte-identical contention, but the paced stream declared normal, the
                    control that isolates what the safety classification itself buys.

Panel a shows the safety stream's p99 across the nine cells. Panel b shows, for the
classified contended cells, the per-message p99 of the safety stream against the bulk
streams' p99 and aggregate rate — the three-orders-of-magnitude separation between the
paced class and the saturating class is the isolation result.

Honesty rails baked into the render: the paced stream is identified by its CO-corrected
flag (never by name), the effective protocol is read from summary.csv so a kTLS→user-space
fallback is labelled as such (only when the plotted run records one), and the method note
states the campaign's loopback boundaries (noqueue loopback, DSCP preservation not
measurable on TCP). Runs without qos_* scenarios skip this figure with a reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle

FIG_ID = "F24"
NAME = "f24_priority_isolation"
TITLE = "Traffic-class isolation under bulk contention"

_SCEN_RE = re.compile(r"^qos_(alone|contended|unclassified)_([a-z0-9]+)$")

_CONDITIONS = ["alone", "contended", "unclassified"]
_CONDITION_LABEL = {
    "alone": "alone",
    "contended": "contended · safety class",
    "unclassified": "contended · normal class",
}
# F24's condition vocabulary on the shared QoS condition palette (same colours as F27:
# alone = baseline, safety-class contended = CATEGORY[0], normal/unclassified = CATEGORY[1]).
_CONDITION_COLOR = {
    "alone": T.CONDITION_COLORS["alone"],
    "contended": T.CONDITION_COLORS["safety"],
    "unclassified": T.CONDITION_COLORS["normal"],
}
_PROTO_ORDER = ["routing", "tls", "ktls"]


def _proto_label(token: str, effective: str | None) -> str:
    base = {"routing": "routing", "tls": "TLS 1.3", "ktls": "kTLS 1.3"}.get(token, token)
    # The effective protocol reads e.g. "tls/1.3 (ktls->userspace)" after a fallback: judge
    # by its leading token, not by substring (the parenthetical mentions ktls either way).
    if token == "ktls" and effective and not effective.split(" ")[0].startswith("ktls"):
        return f"{base}\n(ran as user-space TLS)"
    return base


def _load_cells(bundle: RunBundle) -> tuple[dict, dict]:
    """(cells, effective) keyed by (proto, condition) → dict of safety/bulk stats."""
    effective: dict[str, str] = {}
    summ = bundle.summary
    if not summ.empty and {"scenario", "effective_protocol"}.issubset(summ.columns):
        for _, r in summ.iterrows():
            effective[str(r["scenario"])] = str(r.get("effective_protocol") or "")

    cells: dict[tuple[str, str], dict] = {}
    scen_root = Path(bundle.run_dir) / "scenarios"
    if not scen_root.is_dir():
        return cells, effective
    for sdir in sorted(scen_root.iterdir()):
        m = _SCEN_RE.match(sdir.name)
        if not m:
            continue
        cond, proto = m.group(1), m.group(2)
        csv = sdir / "streams.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        if df.empty or "latency_p99_us" not in df.columns:
            continue
        # The paced (measured) stream is the CO-corrected one — identified by its flag,
        # never by its role name, so the unclassified control (declared normal) is found
        # the same way as the classified stream.
        co = df.get("co_corrected")
        paced_mask = (
            co.astype(str).str.lower().isin(("true", "1")) if co is not None
            else df["stream"].astype(str).str.lower().str.startswith("safety")
        )
        paced = df[paced_mask]
        bulk = df[~paced_mask]
        if paced.empty:
            continue
        prow = paced.iloc[0]
        cell = {
            "safety_p50": float(prow.get("latency_p50_us", np.nan)),
            "safety_p99": float(prow.get("latency_p99_us", np.nan)),
            "safety_jitter": float(prow.get("jitter_us", np.nan)),
            "safety_lost": float(pd.to_numeric(paced.get("lost"), errors="coerce").fillna(0).sum()),
            "safety_gbps": float(prow.get("throughput_gbps", np.nan)),
            "send_lag_mean": float(prow.get("send_lag_mean_us", np.nan)),
        }
        if not bulk.empty:
            cell["bulk_gbps"] = float(pd.to_numeric(bulk["throughput_gbps"], errors="coerce").sum())
            cell["bulk_p99"] = float(pd.to_numeric(bulk["latency_p99_us"], errors="coerce").mean())
            cell["bulk_lost"] = float(pd.to_numeric(bulk.get("lost"), errors="coerce").fillna(0).sum())
        cells[(proto, cond)] = cell
    return cells, effective


def make(bundle: RunBundle, saver: T.Saver) -> None:
    cells, effective = _load_cells(bundle)
    if not cells:
        saver.record_skip(FIG_ID, NAME, "no qos_* multi-stream scenarios in this run")
        return

    protos = [p for p in _PROTO_ORDER if any(k[0] == p for k in cells)]
    protos += sorted({k[0] for k in cells} - set(protos))

    import matplotlib.pyplot as plt

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # --- Panel a: safety-stream p99 per protection mode × condition -----------------------
    x = np.arange(len(protos))
    bw = 0.8 / len(_CONDITIONS)
    for j, cond in enumerate(_CONDITIONS):
        vals = [cells.get((p, cond), {}).get("safety_p99", np.nan) for p in protos]
        xpos = x + (j - (len(_CONDITIONS) - 1) / 2.0) * bw
        axa.bar(xpos, vals, width=bw, color=_CONDITION_COLOR[cond],
                edgecolor=T.GREYS["edge"], linewidth=0.5, label=_CONDITION_LABEL[cond])
        for xi, v in zip(xpos, vals):
            if np.isfinite(v):
                T.annotate_value(axa, xi, v, f"{v:.0f}")
    axa.set_xticks(x)
    axa.set_xticklabels(
        [_proto_label(p, effective.get(f"qos_contended_{p}")) for p in protos],
        fontsize=T.FS["small"])
    axa.set_ylabel("safety-stream p99 (µs, closed schedule)")
    # Headroom above the tallest bar so the legend never overprints the value labels.
    finite = [c.get("safety_p99") for c in cells.values() if np.isfinite(c.get("safety_p99", np.nan))]
    if finite:
        axa.set_ylim(0, max(finite) * 1.38)
    T.legend_inline(axa, loc="upper left")
    axa.grid(axis="y")

    # --- Panel b: paced class vs saturating class, classified contended cells -------------
    rows = [(p, cells[(p, "contended")]) for p in protos if (p, "contended") in cells]
    for i, (p, c) in enumerate(rows):
        sp99, bp99 = c.get("safety_p99", np.nan), c.get("bulk_p99", np.nan)
        if not (np.isfinite(sp99) and np.isfinite(bp99)):
            continue
        axb.plot([sp99, bp99], [i, i], color=T.GREYS["muted"], lw=2.0, alpha=0.6, zorder=1)
        axb.scatter([sp99], [i], color=T.CONDITION_COLORS["safety"], s=60, zorder=3)
        axb.scatter([bp99], [i], color=T.CONDITION_COLORS["normal"], s=60, marker="s", zorder=3)
        note = f"bulk {c.get('bulk_gbps', float('nan')):.0f} Gbit/s"
        axb.annotate(note, (np.sqrt(sp99 * bp99), i), xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=T.FS["annot"], color=T.GREYS["annot"])
    axb.set_yticks(np.arange(len(rows)))
    axb.set_yticklabels([_proto_label(p, effective.get(f"qos_contended_{p}")).split("\n")[0]
                         for p, _ in rows], fontsize=T.FS["small"])
    axb.invert_yaxis()
    axb.set_xscale("log")
    axb.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
    axb.set_xlabel("p99 latency (µs, log)")
    from matplotlib.lines import Line2D

    T.legend_inline(axb, [
        Line2D([0], [0], ls="none", marker="o", markersize=7,
               markerfacecolor=T.CONDITION_COLORS["safety"],
               markeredgecolor=T.GREYS["edge"], label="paced safety class"),
        Line2D([0], [0], ls="none", marker="s", markersize=7,
               markerfacecolor=T.CONDITION_COLORS["normal"],
               markeredgecolor=T.GREYS["edge"], label="bulk class"),
    ], loc="upper right")
    axb.grid(axis="x", which="both", alpha=0.4)
    axb.margins(x=0.15, y=0.2)

    # --- Computed takeaway ----------------------------------------------------------------
    take = ""
    con = [c for (p, cond), c in cells.items() if cond == "contended"]
    alone = {p: c for (p, cond), c in cells.items() if cond == "alone"}
    if con and alone:
        worst_p99 = max(c["safety_p99"] for c in con if np.isfinite(c["safety_p99"]))
        lost = sum(c.get("safety_lost", 0) for c in con)
        bulks = [c.get("bulk_gbps", np.nan) for c in con]
        bulks = [b for b in bulks if np.isfinite(b)]
        bp99s = [c.get("bulk_p99", np.nan) for c in con]
        bp99s = [b for b in bp99s if np.isfinite(b)]
        rel = []
        for p, c in alone.items():
            cc = cells.get((p, "contended"))
            if cc and np.isfinite(c["safety_p99"]) and c["safety_p99"] > 0:
                rel.append((cc["safety_p99"] / c["safety_p99"] - 1) * 100)
        cls_gain = []
        for p in protos:
            cc, uc = cells.get((p, "contended")), cells.get((p, "unclassified"))
            if cc and uc and np.isfinite(cc["safety_p99"]) and cc["safety_p99"] > 0 \
                    and np.isfinite(uc["safety_p99"]):
                cls_gain.append((uc["safety_p99"] / cc["safety_p99"] - 1) * 100)
        take = (f"Under saturating bulk contention ({min(bulks):.0f}–{max(bulks):.0f} Gbit/s "
                f"at {min(bp99s) / 1000:.0f}–{max(bp99s) / 1000:.0f} ms p99) the paced safety "
                f"stream holds p99 ≤ {worst_p99:.0f} µs with {int(lost)} lost frames "
                f"(+{min(rel):.0f}–{max(rel):.0f}% over its uncontended baseline)")
        if cls_gain:
            take += (f"; the safety classification itself improves p99 by "
                     f"{min(cls_gain):.0f}–{max(cls_gain):.0f}% here, with the bulk of the "
                     f"isolation coming from the per-connection data plane")
        take += "."

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.03)
    if take:
        T.add_takeaway(fig, take)
    T.add_method_note(
        fig,
        "paced safety stream: 256 B every 200 µs (5000 msg/s), DSCP EF, coordinated-omission-"
        "corrected via scheduled-send stamping (send-lag recorded); contention: 4 × 64 KB "
        "sustained best-effort blast streams; TCP, single gateway (scg-direct), class-aware "
        "rule pairs, one 5 s measurement per cell. Reserved class rules and DSCP/SO_PRIORITY "
        "marking stay active; loopback has no queueing discipline for the marking to act on, "
        "so packet-level prioritisation gains need a NIC-path rerun. The effective protocol "
        "per group is recorded and shown in the group label (a kTLS fallback to user-space "
        "TLS is annotated only when it occurred on the plotted run). DSCP preservation is "
        "not observable on TCP and is reported empty, never fabricated.",
    )
    T.add_provenance(fig, bundle.caption() + "  ·  qos_isolation suite · paced stream identified by its CO-corrected flag")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
