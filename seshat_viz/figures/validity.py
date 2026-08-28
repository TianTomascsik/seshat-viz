"""
F11 — Measurement validity: headroom & bottleneck attribution.

SESHAT's NFR-PERF rule is "the harness must never be the bottleneck." This figure makes that
auditable in two stacked panels:

  * Panel 1 (run-wide ECDF): the headroom (harness ceiling ÷ measured throughput) distribution
    over EVERY row that carries a finite headroom, with the 3× credibility gate marked and the
    pass fraction stated. This is the honest denominator — the worst-N bar panel below is only
    the bottom tail, so on its own it would imply pervasive failure. The failing band is split
    into headroom < 1× (the measured rate beat the null-probe's own ceiling, i.e. the probe
    under-read) versus 1–3× (below the gate but physically plausible).
  * Panel 2 (worst-slice bars): per scenario, the headroom as a bar, the harness-limited flag by
    color, a reference line at the credibility threshold, and the classified bottleneck/DUT, for
    the lowest-headroom scenarios only. Lets an examiner see at a glance which numbers are shakiest.

Two honesty rules on the pool itself: rows whose ceiling probe FAILED (ceiling ≤ 0) are excluded
and counted in the headline — a dead probe is not an "under-read" and a headroom of exactly 0 is
unplottable on the log axis; and rows that never got a ceiling probe (headroom absent) are counted
in the method note, because "run-wide" must not silently mean "the probed subset".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle, protocol_label, transport_label

FIG_ID = "F11"
NAME = "f11_measurement_validity"
TITLE = "Measurement validity: headroom & bottleneck attribution"

_THRESHOLD = 3.0   # SESHAT's HEADROOM_MIN gate: below this a row is flagged unless a
                   # CPU signal proves the gateway (scg-cpu) is the limit
_MAX_BARS = 36     # keep the chart legible; show the lowest-headroom (most at-risk) scenarios

# Band alphas for the three headroom verdict zones — shared by the axvspan draw
# and the legend swatches so the two can never diverge again.
_ZONE_ALPHA = {"bad": 0.12, "warn": 0.12, "ok": 0.06}


def _zone_on_white(sem_key: str):
    """Effective on-white colour of a translucent verdict band.

    Legend swatches are drawn in this composited colour (instead of re-applying
    an unrelated alpha) so the legend matches what the band actually looks like
    on the page (audit: legend-vs-band colour mismatch).
    """
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(T.SEM[sem_key])
    a = _ZONE_ALPHA[sem_key]
    return (1 - a * (1 - r), 1 - a * (1 - g), 1 - a * (1 - b))


def _gate_handle():
    """Legend key for the ACCENT dashed trust-gate guide line."""
    from matplotlib.lines import Line2D

    return Line2D([0], [0], color=T.ACCENT, ls="--", lw=1.2,
                  label=f"{_THRESHOLD:.0f}× trust gate")


def _guide_handles():
    """Both vertical guides, keyed: the 3× trust gate and the 1× under-read line."""
    from matplotlib.lines import Line2D

    return [
        _gate_handle(),
        Line2D([0], [0], color=T.GREYS["annot"], ls=":", lw=1.0,
               label="1× (null probe under-read)"),
    ]


def make(bundle: RunBundle, saver: T.Saver) -> None:
    df = bundle.summary
    if "headroom" not in df.columns or df["headroom"].notna().sum() == 0:
        saver.record_skip(FIG_ID, NAME, "no headroom column")
        return

    n_run = len(df)
    head_all = pd.to_numeric(df["headroom"], errors="coerce")
    # Rows with no ceiling probe at all are OUTSIDE this audit — nearly all of them still
    # report a throughput, and the throughput/paced subset (hotreload above all) feeds other
    # thesis figures, so the "run-wide" claim must disclose the exclusion (audit F11-2).
    unassessed = head_all.isna()
    n_unassessed = int(unassessed.sum())
    n_unassessed_tput = (
        int(df.loc[unassessed, "mode"].isin(("throughput", "paced", "saturation")).sum())
        if "mode" in df.columns
        else None
    )

    d = df[head_all.notna()].copy()
    d["headroom"] = head_all[head_all.notna()]
    if d.empty:
        saver.record_skip(FIG_ID, NAME, "headroom all-null")
        return

    # headroom = ceiling ÷ measured, so a non-positive ceiling means the null probe FAILED
    # (returned no data), not that the row beat it. Binning such rows below 1× would smuggle
    # a probe failure into the under-read count, force the log ECDF axis down to an
    # unplottable x=0, and park a zero-length bar in the worst slice (audit F11-1) — exclude
    # them from the pool and disclose the count in the headline instead.
    probe_failed = d["headroom"] <= 0
    if "ceiling_gbps" in d.columns:
        probe_failed |= pd.to_numeric(d["ceiling_gbps"], errors="coerce") <= 0
    n_probe_failed = int(probe_failed.sum())
    d = d[~probe_failed]
    if d.empty:
        saver.record_skip(FIG_ID, NAME, "every ceiling probe failed (ceiling ≤ 0)")
        return

    # Run-wide validity summary computed over ALL finite-headroom rows, so the worst-N slice
    # below cannot be misread as the whole run. Split the failing band (headroom < gate) in two:
    # below 1× the measured rate beat the null-probe's own ceiling (probe under-read), while
    # 1–gate is below the credibility bar but physically plausible.
    n_all = len(d)
    n_pass = int((d["headroom"] >= _THRESHOLD).sum())
    n_fail = n_all - n_pass
    n_below1 = int((d["headroom"] < 1.0).sum())
    n_sub_gate = n_fail - n_below1
    all_headroom = np.sort(d["headroom"].to_numpy())  # full distribution for the ECDF panel
    pass_note = (
        f"run-wide: {n_pass}/{n_all} clear the {_THRESHOLD:.0f}× gate, {n_fail} fail "
        f"({n_below1} below 1× · {n_sub_gate} in 1–{_THRESHOLD:.0f}×); median {d['headroom'].median():.2f}×"
    )
    if n_probe_failed:
        plural = "s" if n_probe_failed != 1 else ""
        pass_note += f" · {n_probe_failed} failed ceiling probe{plural} excluded"

    # ---- Thesis variant: the run-wide ECDF panel alone -----------------------------------
    # The worst-slice bar panel is an exploration aid (36 scenario labels at 7 pt cannot
    # survive a 15 cm text column); the distribution + gate + counts are the evaluative
    # content, so the print figure is the honest single panel with everything computed.
    if T.thesis_variant():
        _make_thesis(bundle, saver, all_headroom=all_headroom,
                     n_run=n_run, n_all=n_all, n_pass=n_pass, n_fail=n_fail,
                     n_below1=n_below1, n_sub_gate=n_sub_gate,
                     median=float(d["headroom"].median()),
                     n_probe_failed=n_probe_failed, n_unassessed=n_unassessed,
                     n_unassessed_tput=n_unassessed_tput, pass_note=pass_note)
        return

    # The bar panel is deliberately the bottom tail: most at-risk first, capped for legibility.
    d = d.sort_values("headroom")
    if len(d) > _MAX_BARS:
        d = d.head(_MAX_BARS)

    def _scen_label(r) -> str:
        t = transport_label(str(r.get("transport", "")))
        p = protocol_label(str(r.get("protocol", "")))
        sz = T.fmt_bytes(r["message_bytes"]) + "B" if pd.notna(r.get("message_bytes")) else ""
        # Use the true gateway count (chain is a loader guess off the matrix family) plus the
        # connection count, so two identically-labelled bars are disambiguated.
        ng = r.get("n_gateways")
        gw = f"{int(ng)}gw" if pd.notna(ng) else str(r.get("chain", ""))
        cc = f"{int(r['connections'])}c" if pd.notna(r.get("connections")) else ""
        return f"{t}·{p}·{sz}·{gw}·{cc}".rstrip("·")

    d = d.assign(_lbl=d.apply(_scen_label, axis=1))
    y = np.arange(len(d))

    def _is_limited(v) -> bool:
        return bool(v) if isinstance(v, (bool, np.bool_)) else False

    # Four classes: baseline (grey — no gateway in the path, so NFR-PERF doesn't apply and
    # 'gateway-bound' is meaningless), trusted (blue), host-saturated (orange — still a lower
    # bound, single-host physics), and harness-io (red — suspect). Loopback baseline rows were
    # previously mislabeled the trustworthy blue class despite having no gateway to be bound by.
    limited_col = (
        d["harness_limited"] if "harness_limited" in d.columns else pd.Series([None] * len(d))
    )
    bneck_col = d["bottleneck"] if "bottleneck" in d.columns else pd.Series([""] * len(d))
    dpath_col = d["datapath"] if "datapath" in d.columns else pd.Series([""] * len(d))
    dut_col = d["dut"] if "dut" in d.columns else pd.Series([""] * len(d))

    def _bar_color(lv, bn, dp, du) -> str:
        if str(dp) == "loopback" or str(du) == "loopback":
            return T.SEM["neutral"]  # no-gateway baseline (no verdict applies)
        if _is_limited(lv) and str(bn) == "host-saturated":
            return T.SEM["warn"]
        if _is_limited(lv):
            return T.SEM["bad"]
        return T.SEM["ok"]

    bar_colors = [_bar_color(lv, bn, dp, du)
                  for lv, bn, dp, du in zip(limited_col, bneck_col, dpath_col, dut_col)]

    import matplotlib.pyplot as plt

    # Two stacked panels: (1) the honest run-wide headroom ECDF over ALL rows, so the worst-N
    # tail below cannot be read as pervasive failure; (2) the existing worst-slice bar chart.
    bar_h = 0.30 * len(d) + 2.0
    fig = plt.figure(figsize=(8.6, bar_h + 3.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, bar_h])
    ax_cdf = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    # ---- Panel 1: run-wide headroom ECDF ------------------------------------------------
    ecdf_y = np.arange(1, n_all + 1) / n_all
    lo = max(float(all_headroom[0]) * 0.97, 1e-3)
    hi = float(all_headroom[-1]) * 1.1
    # Zone shading splits the sub-gate (failing) band: below 1× the measured throughput beat the
    # null-probe's own ceiling (probe under-read — suspect), 1–3× is below the gate but plausible,
    # ≥3× is trusted. Log x so the sub-1×…10³× spread and both boundaries stay legible; `lo`
    # derives from the true minimum because probe failures (headroom 0) were excluded above.
    ax_cdf.axvspan(lo, 1.0, color=T.SEM["bad"], alpha=_ZONE_ALPHA["bad"], zorder=0)
    ax_cdf.axvspan(1.0, _THRESHOLD, color=T.SEM["warn"], alpha=_ZONE_ALPHA["warn"], zorder=0)
    ax_cdf.axvspan(_THRESHOLD, hi, color=T.SEM["ok"], alpha=_ZONE_ALPHA["ok"], zorder=0)
    ax_cdf.step(all_headroom, ecdf_y, where="post", color=T.SEM["ok"], lw=1.6, zorder=3)
    ax_cdf.axvline(1.0, color=T.GREYS["annot"], ls=":", lw=1.0, zorder=2)
    ax_cdf.axvline(_THRESHOLD, color=T.ACCENT, ls="--", lw=1.2, zorder=2)
    # Mark where the ECDF crosses the gate: that height is exactly the failing fraction.
    frac_fail = n_fail / n_all
    ax_cdf.plot([_THRESHOLD], [frac_fail], "o", color=T.GREYS["annot"], ms=4, zorder=4)
    ax_cdf.annotate(f"{frac_fail * 100:.0f}% below the {_THRESHOLD:.0f}× gate",
                    (_THRESHOLD, frac_fail), xytext=(6, -11), textcoords="offset points",
                    fontsize=T.FS["annot"], color=T.GREYS["annot"])
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlim(lo, hi)
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.set_xlabel("headroom = harness ceiling ÷ measured throughput  (log scale)")
    ax_cdf.set_ylabel("fraction of rows ≤ x")
    ax_cdf.grid(True, which="both")
    T.panel_title(ax_cdf, f"Run-wide headroom distribution — {n_all} finite-headroom rows of {n_run} in the run")
    zone_handles = [
        plt.matplotlib.patches.Patch(facecolor=_zone_on_white("bad"),
                                     edgecolor=T.GREYS["edge"], linewidth=0.5,
                                     label=f"< 1×: probe under-read ({n_below1})"),
        plt.matplotlib.patches.Patch(facecolor=_zone_on_white("warn"),
                                     edgecolor=T.GREYS["edge"], linewidth=0.5,
                                     label=f"1–{_THRESHOLD:.0f}×: below gate ({n_sub_gate})"),
        plt.matplotlib.patches.Patch(facecolor=_zone_on_white("ok"),
                                     edgecolor=T.GREYS["edge"], linewidth=0.5,
                                     label=f"≥ {_THRESHOLD:.0f}×: trusted ({n_pass})"),
    ]
    T.legend_inline(ax_cdf, zone_handles + _guide_handles(), loc="lower right")

    # ---- Panel 2: worst-slice bars (the bottom tail only) -------------------------------
    ax.barh(y, d["headroom"], color=bar_colors, edgecolor=T.GREYS["edge"], linewidth=0.4)
    ax.axvline(_THRESHOLD, color=T.ACCENT, ls="--", lw=1.2)
    ax.annotate(f"SESHAT headroom gate = {_THRESHOLD:.0f}×", (_THRESHOLD, len(d) - 0.5),
                xytext=(4, 0), textcoords="offset points", fontsize=T.FS["annot"],
                color=T.GREYS["annot"])
    ax.set_yticks(y)
    ax.set_yticklabels(d["_lbl"], fontsize=T.FS["annot"])
    ax.set_xlabel("headroom = harness ceiling ÷ measured throughput  (higher = safer)")
    T.panel_title(ax, f"Worst slice: lowest-headroom {len(d)} of {n_all} scenarios (bottom tail only)")
    ax.grid(axis="x")
    T.set_headline(fig, f"{TITLE}\n{bundle.label} · {pass_note}")

    # Annotate bottleneck/dut class at the bar tip where available. Guard NaN explicitly:
    # float('nan') is truthy, so `bottleneck or dut` short-circuited to 'nan' and the dut
    # fallback (which reveals loopback baselines) never fired.
    tip = d["headroom"].max() * 1.02
    for i, (_, r) in enumerate(d.iterrows()):
        bn = r.get("bottleneck")
        du = r.get("dut")
        tag = str(bn) if pd.notna(bn) else (str(du) if pd.notna(du) else "")
        if tag and tag.lower() != "nan":
            ax.annotate(tag, (min(r["headroom"], tip), i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=T.FS["annot"],
                        color=T.GREYS["annot"])

    # Legend patches mirror the bars exactly (SEM verdict fill + edge), so the key
    # can never drift from what the bars actually wear (audit: colour mismatch).
    handles = [
        plt.matplotlib.patches.Patch(facecolor=T.SEM["ok"], edgecolor=T.GREYS["edge"],
                                     linewidth=0.5, label="trustworthy (gateway-bound)"),
        plt.matplotlib.patches.Patch(facecolor=T.SEM["warn"], edgecolor=T.GREYS["edge"],
                                     linewidth=0.5, label="host-saturated (single-host lower bound)"),
        plt.matplotlib.patches.Patch(facecolor=T.SEM["bad"], edgecolor=T.GREYS["edge"],
                                     linewidth=0.5, label="harness-limited (interpret with care)"),
        plt.matplotlib.patches.Patch(facecolor=T.SEM["neutral"], edgecolor=T.GREYS["edge"],
                                     linewidth=0.5, label="baseline (no gateway — NFR-PERF n/a)"),
        _gate_handle(),
    ]
    T.legend_inline(ax, handles, loc="lower right")

    T.add_method_note(
        fig,
        "headroom = harness ceiling ÷ measured throughput, probed interface-true (tcp/udp/"
        "uds-null/shm-null), pinned and warmed. Blue bars are trusted: either ≥3× headroom or a "
        "CPU signal proves the gateway is the limit (scg-cpu tag: pinned pool ≥85% or hottest "
        "gateway thread ≥90% p95). Orange = whole host ≥90% busy: a lower bound imposed by "
        "single-host physics. Red = the load generator may have capped the number. At tiny "
        "messages headroom can sit slightly below 1× on passthrough (routing) and encrypted "
        "paths alike: the gateway relay pipelines across more cores than the 2-thread null "
        "probe.",
        y=0.036,
    )
    # Pool disclosure on its own line: computed from the bundle every render; the sub-count of
    # genuine throughput/paced measurements only exists when the loader's mode column does.
    if n_unassessed:
        tput_clause = (
            f" ({n_unassessed_tput} of them throughput/paced measurements)"
            if n_unassessed_tput is not None
            else ""
        )
        T.add_method_note(
            fig,
            f"{n_unassessed} rows without a ceiling probe{tput_clause} are outside this audit.",
            y=0.020,
        )
    T.add_takeaway(
        fig,
        f"Most ceiling-probed measurements are trustworthy — {n_pass}/{n_all} rows clear the "
        f"{_THRESHOLD:.0f}× gate; the bar panel is only the {len(d)}-row bottom tail.",
    )
    n_lim = int(d["harness_limited"].sum()) if "harness_limited" in d.columns else 0
    T.add_provenance(fig, bundle.caption() + f"  ·  {n_lim}/{len(d)} shown scenarios flagged harness-limited")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)


def _make_thesis(bundle: RunBundle, saver: T.Saver, *, all_headroom, n_run: int,
                 n_all: int, n_pass: int, n_fail: int, n_below1: int, n_sub_gate: int,
                 median: float, n_probe_failed: int, n_unassessed: int,
                 n_unassessed_tput, pass_note: str) -> None:
    """Print variant: the run-wide headroom ECDF alone, sized for a 15 cm text column."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ecdf_y = np.arange(1, n_all + 1) / n_all
    lo = max(float(all_headroom[0]) * 0.97, 1e-3)
    hi = float(all_headroom[-1]) * 1.1
    ax.axvspan(lo, 1.0, color=T.SEM["bad"], alpha=_ZONE_ALPHA["bad"], zorder=0)
    ax.axvspan(1.0, _THRESHOLD, color=T.SEM["warn"], alpha=_ZONE_ALPHA["warn"], zorder=0)
    ax.axvspan(_THRESHOLD, hi, color=T.SEM["ok"], alpha=_ZONE_ALPHA["ok"], zorder=0)
    ax.step(all_headroom, ecdf_y, where="post", color=T.SEM["ok"], lw=1.8, zorder=3)
    ax.axvline(1.0, color=T.GREYS["annot"], ls=":", lw=1.0, zorder=2)
    ax.axvline(_THRESHOLD, color=T.ACCENT, ls="--", lw=1.2, zorder=2)
    frac_fail = n_fail / n_all
    ax.plot([_THRESHOLD], [frac_fail], "o", color=T.GREYS["annot"], ms=4, zorder=4)
    ax.annotate(f"{frac_fail * 100:.0f}% below the {_THRESHOLD:.0f}× gate",
                (_THRESHOLD, frac_fail), xytext=(8, -12), textcoords="offset points",
                fontsize=T.FS["annot"], color=T.GREYS["annot"])
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("headroom = harness ceiling ÷ measured throughput  (log scale)")
    ax.set_ylabel("fraction of rows ≤ x")
    ax.grid(True, which="both")
    zone_handles = [
        plt.matplotlib.patches.Patch(facecolor=_zone_on_white("bad"),
                                     edgecolor=T.GREYS["edge"], linewidth=0.5,
                                     label=f"< 1×: probe under-read ({n_below1})"),
        plt.matplotlib.patches.Patch(facecolor=_zone_on_white("warn"),
                                     edgecolor=T.GREYS["edge"], linewidth=0.5,
                                     label=f"1–{_THRESHOLD:.0f}×: below gate ({n_sub_gate})"),
        plt.matplotlib.patches.Patch(facecolor=_zone_on_white("ok"),
                                     edgecolor=T.GREYS["edge"], linewidth=0.5,
                                     label=f"≥ {_THRESHOLD:.0f}×: trusted ({n_pass})"),
    ]
    T.legend_inline(ax, zone_handles + _guide_handles(), loc="center right")

    T.set_headline(fig, f"{TITLE}\n{bundle.label} · {pass_note}")
    T.add_method_note(
        fig,
        "headroom = harness ceiling ÷ measured throughput, probed interface-true "
        "(tcp/udp/uds-null/shm-null), pinned and warmed. A row below the gate is flagged "
        "harness-limited unless a CPU signal proves the gateway is the limit (pinned pool "
        "≥85% or hottest gateway thread ≥90% p95); flagged throughput is a load-generator "
        "lower bound, not a gateway capability.",
        y=0.036,
    )
    if n_unassessed:
        tput_clause = (
            f" ({n_unassessed_tput} of them throughput/paced measurements)"
            if n_unassessed_tput is not None
            else ""
        )
        T.add_method_note(
            fig,
            f"{n_unassessed} rows without a ceiling probe{tput_clause} are outside this audit.",
            y=0.020,
        )
    takeaway = (
        f"{n_pass}/{n_all} ceiling-probed rows clear the {_THRESHOLD:.0f}× headroom gate "
        f"(median {median:.2f}×); {n_fail} fall below it ({n_below1} under 1×), "
        f"{n_unassessed} rows carry no probe and sit outside this audit"
    )
    if n_probe_failed:
        takeaway += f", and {n_probe_failed} failed ceiling probe(s) are excluded"
    T.add_takeaway(fig, takeaway + ".")
    T.add_provenance(fig, bundle.caption() + f"  ·  audit pool {n_all} of {n_run} run rows")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
