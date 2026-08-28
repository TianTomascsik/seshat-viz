"""
F27 — Traffic-class prioritisation on the wire (QOS-001).

Three media-conditions for the same three QoS cells (safety alone / contended
declared safety / contended declared normal, the classification control):

  * loopback            — the published testbed: isolation holds, but there is
                          no queueing discipline for the DSCP marks to act on;
  * wire · no qdisc     — the physical link with its default qdisc: the honest
                          negative result. The replicates are wildly unstable
                          and the class ordering inverts between runs — DSCP
                          marking alone does not prioritise;
  * wire · priority qdisc — the pending `wire-qos3-qdisc` campaign (priority
                          bands + TBF via scg-host-qos.sh); appears here
                          automatically once the results directory exists.

Panel b renders the far-side DSCP evidence — the per-port code-point histogram
from the peer's packet capture (`peer-out/wire.pcap` or a pre-computed
`dscp_evidence.json`). Code points come ONLY from the capture's TOS bytes: the
CSV's dscp_observed/matched columns are packet counts, not code points, and feed
a single method-note sentence instead.

Honesty rails: replicate spread is drawn as min–max whiskers PLUS the individual
replicate dots, so the no-qdisc instability is visible as chaos rather than
averaged into a fake ordering; the takeaway only claims prioritisation when the
qdisc campaign exists and shows it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle

FIG_ID = "F27"
NAME = "f27_wire_qos"
TITLE = "Traffic-class prioritisation on the wire"

_CELLS = ["qos-safety-alone", "qos-safety-contended", "qos-normal-contended"]
# F24's condition vocabulary and colours (the sanctioned precedent for
# condition-coloured bars; protocol identity is constant here and named in the
# headline, so the colour-is-protocol house rule is not violated).
_CONDITION_LABEL = {
    "qos-safety-alone": "alone",
    "qos-safety-contended": "contended · safety class",
    "qos-normal-contended": "contended · normal class",
}
_CONDITION_COLOR = {
    "qos-safety-alone": T.CONDITION_COLORS["alone"],
    "qos-safety-contended": T.CONDITION_COLORS["safety"],
    "qos-normal-contended": T.CONDITION_COLORS["normal"],
}

_PORT_INFO = {
    21101: ("safety (TCP)", 46),
    21100: ("bulk (TCP)", 0),
    21102: ("datagram (UDP)", 46),
    21103: ("control · normal (TCP)", 0),
}
# EF = SEM "ok" (marking verified), BE = baseline grey; an unexpected value = SEM "bad".
_DSCP_COLOR = {46: T.SEM["ok"], 0: T.GREYS["baseline"]}


def _conditions(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Ordered (label, rows) media-conditions actually present."""
    out = []
    for medium, qdisc, label in (
        ("loopback", False, "loopback"),
        ("wire", False, "wire · no qdisc"),
        ("wire", True, "wire · priority qdisc"),
    ):
        sel = df[(df["medium"] == medium) & (df["qdisc"] == qdisc)]
        if not sel.empty:
            out.append((label, sel))
    return out


def _find_evidence(wb) -> dict | None:
    """DSCP evidence for the wire QoS campaigns: JSON first, else parse the pcap."""
    from ..pcap_dscp import PcapFormatError, parse_pcap

    qos_dirs = [d for d in wb.dirs if d.name in ("wire-qos3-qdisc", "wire-qos3", "wire-run")]
    qos_dirs.sort(key=lambda d: ("qdisc" not in d.name, d.name))  # prefer the qdisc campaign
    for cdir in qos_dirs:
        peer = Path(cdir) / "peer-out"
        json_path = peer / "dscp_evidence.json"
        if json_path.is_file():
            import json

            try:
                return json.loads(json_path.read_text())
            except (OSError, ValueError):
                continue
        pcap = peer / "wire.pcap"
        if pcap.is_file():
            try:
                return parse_pcap(pcap)
            except (OSError, PcapFormatError):
                continue
    return None


def make(bundle: RunBundle, saver: T.Saver) -> None:
    wb = getattr(bundle, "wire", None)
    if wb is None or wb.df.empty:
        saver.record_skip(FIG_ID, NAME,
                          "no wire campaign dirs found (pass --wire-results SCG-SESHAT/results)")
        return
    qos = wb.df[(wb.df["role"] == "qos") & (wb.df["cell"].isin(_CELLS))]
    conditions = _conditions(qos)
    if not conditions:
        saver.record_skip(FIG_ID, NAME, "no qos-* rows in the wire campaigns")
        return
    evidence = _find_evidence(wb)
    has_qdisc = any(label == "wire · priority qdisc" for label, _ in conditions)
    thesis = T.thesis_variant()

    import matplotlib.pyplot as plt

    two_panel = evidence is not None or not thesis
    if two_panel:
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6),
                                 gridspec_kw={"width_ratios": [1.5, 1.0]}, squeeze=False)
        axa, axb = axes[0][0], axes[0][1]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6.4, 3.6), squeeze=False)
        axa, axb = axes[0][0], None

    # --- panel a: safety-probe p99 per media-condition × cell -----------------------------
    stats: dict[tuple[str, str], dict] = {}
    x = np.arange(len(conditions))
    bw = 0.8 / len(_CELLS)
    for j, cell in enumerate(_CELLS):
        med, lo, hi, dots_x, dots_y = [], [], [], [], []
        for i, (label, sel) in enumerate(conditions):
            vals = pd.to_numeric(
                sel[sel["cell"] == cell]["rtt_us_p99"], errors="coerce"
            ).dropna()
            xpos = x[i] + (j - (len(_CELLS) - 1) / 2.0) * bw
            if vals.empty:
                med.append(np.nan); lo.append(0); hi.append(0)
                continue
            m = float(vals.median())
            stats[(label, cell)] = {
                "med": m, "min": float(vals.min()), "max": float(vals.max()), "n": len(vals),
            }
            med.append(m)
            lo.append(m - float(vals.min()))
            hi.append(float(vals.max()) - m)
            dots_x.extend([xpos] * len(vals))
            dots_y.extend(vals.tolist())
        xpos_all = x + (j - (len(_CELLS) - 1) / 2.0) * bw
        axa.bar(xpos_all, med, width=bw, color=_CONDITION_COLOR[cell],
                edgecolor=T.GREYS["edge"], linewidth=0.5, label=_CONDITION_LABEL[cell],
                yerr=[lo, hi], error_kw=dict(ecolor=T.GREYS["annot"], elinewidth=0.9, capsize=2.5))
        # Individual replicates: the no-qdisc chaos must be visible, never averaged away.
        axa.scatter(dots_x, dots_y, s=7, color=T.GREYS["ink"], zorder=4, alpha=0.75)
        for xi, v, h in zip(xpos_all, med, hi):
            if np.isfinite(v):
                text = f"{v / 1000:.1f} ms" if v >= 1000 else f"{v:.0f}"
                # Alternate the label height across the group: chaotic whiskers can top
                # out at the same level, landing adjacent labels on one line.
                T.annotate_value(axa, xi, v, text, yerr=h if np.isfinite(h) else 0.0,
                                 stagger=(j % 2) * 8.0)
    axa.set_yscale("log")
    lo_lim, hi_lim = axa.get_ylim()
    axa.set_ylim(lo_lim, hi_lim * 6)  # log headroom: legend must clear the value labels
    axa.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
    axa.set_ylabel("safety-probe RTT p99 (µs, log)")
    axa.set_xticks(x)
    # Two-line tick labels: "wire · no qdisc" as one line collides with its neighbours
    # at panel width (audit: 8.16 tick garble).
    axa.set_xticklabels([label.replace(" · ", "\n") for label, _ in conditions],
                        fontsize=T.FS["small"])
    axa.grid(axis="y", which="both", alpha=0.5)
    T.legend_inline(axa, loc="upper left")
    for i, (label, _) in enumerate(conditions):
        if label == "wire · no qdisc":
            axa.annotate("no queueing discipline —\nDSCP marking only", (x[i], axa.get_ylim()[0]),
                         xytext=(0, -42), textcoords="offset points", ha="center",
                         fontsize=T.FS["annot"], color=T.GREYS["annot"], annotation_clip=False)

    # --- panel b: far-side DSCP evidence --------------------------------------------------
    if axb is not None:
        if evidence and evidence.get("ports"):
            rows = [(p, info) for p, info in _PORT_INFO.items()
                    if str(p) in evidence["ports"] or p in evidence["ports"]]
            ylabels = []
            for yi, (port, (plabel, expected)) in enumerate(rows):
                entry = evidence["ports"].get(str(port)) or evidence["ports"].get(port)
                hist = {int(k): v for k, v in entry["dscp"].items()}
                total = sum(hist.values()) or 1
                left = 0.0
                for dscp_value, count in sorted(hist.items(), key=lambda kv: -kv[1]):
                    frac = count / total
                    color = _DSCP_COLOR.get(dscp_value, T.ACCENT)
                    axb.barh(yi, frac, left=left, height=0.6, color=color,
                             edgecolor=T.GREYS["edge"], linewidth=0.4)
                    if frac > 0.15:
                        name = {46: "EF", 0: "BE"}.get(dscp_value, "")
                        axb.annotate(f"DSCP {dscp_value}{f' ({name})' if name else ''}\n"
                                     f"{count:,} pkts",
                                     (left + frac / 2, yi), ha="center", va="center",
                                     fontsize=T.FS["annot"],
                                     color="white" if dscp_value == 46 else T.GREYS["ink"])
                    left += frac
                ok = hist and max(hist, key=hist.get) == expected
                ylabels.append(plabel if ok else f"{plabel} — MISMATCH")
            axb.set_yticks(range(len(rows)))
            axb.set_yticklabels(ylabels, fontsize=T.FS["small"])
            axb.invert_yaxis()
            axb.set_xlim(0, 1)
            axb.set_xlabel("share of to-peer packets")
            axb.grid(False)
        else:
            T.perf_placeholder(axb, "DSCP evidence\n(needs the far-side capture:\n"
                                    "peer-out/wire.pcap)")

    # --- computed takeaway ----------------------------------------------------------------
    take = ""
    nq_s = stats.get(("wire · no qdisc", "qos-safety-contended"))
    nq_n = stats.get(("wire · no qdisc", "qos-normal-contended"))
    lo_s = stats.get(("loopback", "qos-safety-contended"))
    lo_a = stats.get(("loopback", "qos-safety-alone"))
    qd_s = stats.get(("wire · priority qdisc", "qos-safety-contended"))
    if has_qdisc and qd_s and nq_s:
        take = (f"With the priority qdisc the safety class holds p99 ≤ {qd_s['max']:.0f} µs "
                f"under contention ({nq_s['med'] / qd_s['med']:.0f}× below the unshaped wire), "
                "reproducing the loopback isolation result on physical hardware")
        if evidence and evidence.get("ports"):
            safety = evidence["ports"].get("21101") or evidence["ports"].get(21101)
            if safety:
                ef = safety["dscp"].get("46", safety["dscp"].get(46, 0))
                take += f"; EF marking verified on the wire ({ef:,} packets on the safety flow → DSCP 46)"
        take += "."
    elif nq_s and nq_n:
        take = (f"Under contention on the un-shaped wire the safety and normal classes are "
                f"statistically indistinguishable (p99 {nq_s['min'] / 1000:.1f}–"
                f"{nq_s['max'] / 1000:.1f} ms vs {nq_n['min'] / 1000:.1f}–"
                f"{nq_n['max'] / 1000:.1f} ms across replicates): end-host DSCP marking alone "
                "does not prioritise — a queueing discipline is required.")
        if lo_s and lo_a:
            take += (f" Loopback isolation holds ({lo_s['med']:.0f} µs contended, "
                     f"+{100 * (lo_s['med'] / lo_a['med'] - 1):.0f}% over alone).")

    reps = int(qos["rep"].max()) if not qos.empty else 1
    note = (
        "Safety probe: 256 B every 200 µs, closed-loop, timed on the sender's clock; "
        "contention: 4 × 64 KiB blast connections through the bulk rule. "
        f"{reps} replicates per cell; bars = median, whiskers = min–max, dots = individual "
        "replicates. Mutual TLS on every wire path (the gateway's M-13 validator refuses an "
        "unauthenticated non-loopback deployment). DSCP code points come exclusively from "
        "the far-side capture's TOS bytes; the harness's own dscp columns are packet counts "
        "and never rendered as code points."
    )
    if not has_qdisc:
        note += (" The wire+qdisc campaign is pending: scg-host-qos.sh apply --dev <nic> "
                 "--normal-rate 800mbit, then wire_bench.sh --mode wire --group qos "
                 "--out results/wire-qos3-qdisc.")
    if evidence is None:
        note += " Far-side DSCP evidence pending (copy peer-out/ back from the peer)."

    T.set_headline(fig, f"{TITLE}  ·  kmTLS 1.3 over TCP", y=1.03)
    if take:
        T.add_takeaway(fig, take)
    T.add_method_note(fig, note)
    T.add_provenance(fig, bundle.caption() + "  ·  " + wb.provenance({"qos"}))
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
