"""
F18 — Hot-reload robustness: reconfiguring a live gateway under saturation.

The nightly hot-reload suite drives a gateway at saturation, then triggers a configuration
change — adding a connection or rejecting an invalid config — while traffic flows. The
harness arms exactly ONE reload per scenario, and its timer always lands inside the first of
the engine's measurement runs; the remaining runs are reload-free steady state. Left: a
heatmap of throughput *retained* — the reload run versus the same scenario's reload-free
runs when per-run data is available (undiluted, within-scenario), else the scenario mean
versus the matched matrix baseline — per protocol × concurrency × trigger (green ≈
undisturbed). Right: the integrity ledger — lost / corrupted / mis-framed frames summed
across every reload — the evidence that reconfiguration is non-disruptive, not merely fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label

FIG_ID = "F18"
NAME = "f18_hotreload_robustness"
TITLE = "Hot-reload robustness under saturation"

_TRIG_SHORT = {"add_connection": "add", "remove_connection": "rm", "invalid_config": "reject"}


def _reload_run_retention(runs: pd.DataFrame, scenarios: pd.Series) -> pd.Series:
    """Per-scenario retention of the reload run vs the same scenario's reload-free runs.

    The harness arms exactly one reload per scenario, and its timer (warmup + per-connection
    ramp allowance + fixed delay) always fires inside the FIRST measurement run; later runs
    are reload-free steady state. Comparing run 1 against the mean of the remaining runs
    isolates the reload window — no dilution with steady-state traffic, and no run-length
    mismatch against a foreign baseline (F18-2). Returns retained % indexed by scenario name;
    scenarios that cannot support the split (single run, zero-work steady runs) are omitted.
    """
    if runs is None or runs.empty or not {"scenario", "run", "throughput_gbps"}.issubset(runs.columns):
        return pd.Series(dtype=float)
    r = runs[runs["scenario"].astype(str).isin(set(scenarios.astype(str)))].copy()
    r["run"] = pd.to_numeric(r["run"], errors="coerce")
    r["throughput_gbps"] = pd.to_numeric(r["throughput_gbps"], errors="coerce")
    r = r.dropna(subset=["run", "throughput_gbps"])
    out: dict[str, float] = {}
    for scn, grp in r.groupby(r["scenario"].astype(str), observed=True):
        if grp["run"].nunique() < 2:
            continue  # a lone run cannot be split into reload vs steady state
        first = grp["run"].min()
        reload_tp = grp.loc[grp["run"] == first, "throughput_gbps"].mean()
        steady = grp.loc[grp["run"] != first, "throughput_gbps"].mean()
        if not np.isfinite(steady) or steady <= 0:
            continue  # dead / zero-work steady runs cannot serve as a baseline
        out[scn] = float(reload_tp) / float(steady) * 100.0
    return pd.Series(out, dtype=float)


def make(bundle: RunBundle, saver: T.Saver) -> None:
    tbl = derive.hotreload_retention(bundle.summary)
    if tbl.empty:
        saver.record_skip(FIG_ID, NAME, "no hot-reload scenarios (reload_trigger absent)")
        return

    # Retention is only meaningful at saturation; sub_saturation is rate-limited by design.
    sat = tbl[tbl.get("reload_load", "saturation") == "saturation"].copy() if "reload_load" in tbl.columns else tbl.copy()
    if sat.empty:
        saver.record_skip(FIG_ID, NAME, "no saturation reload rows")
        return

    # Prefer the undiluted within-scenario metric: the scenario mean averages every run but
    # the single reload fires in only one of them, diluting the reload window with steady
    # state. Only switch when every saturation scenario supports the per-run split, so all
    # heatmap cells share one metric.
    per_run = _reload_run_retention(bundle.runs, sat["scenario"])
    undiluted = (not per_run.empty
                 and sat["scenario"].astype(str).isin(per_run.index).all())
    if undiluted:
        sat["retained_pct"] = sat["scenario"].astype(str).map(per_run)
    if "retained_pct" not in sat.columns or sat["retained_pct"].notna().sum() == 0:
        saver.record_skip(FIG_ID, NAME, "no saturation reload rows with a matched baseline or per-run split")
        return

    # ---- Print variant: ledger-first, distribution instead of the 96-cell heatmap -------
    # Per-cell ±few-% in the heatmap is run-to-run noise, not reload cost; the print figure
    # leads with the zero-loss ledger over EVERY reload row (saturation + sub-saturation)
    # and shows the saturation retention values as a distribution centred on 100%.
    if T.print_variant():
        _make_print(bundle, saver, tbl=tbl, sat=sat, undiluted=undiluted)
        return

    sat["__o"] = sat["protocol"].astype(str).map({p: i for i, p in enumerate(T.PROTOCOL_ORDER)}).fillna(99)
    sat["col"] = (sat["connections"].astype("Int64").astype(str) + "c · "
                  + sat["reload_trigger"].astype(str).map(lambda t: _TRIG_SHORT.get(t, t)))
    # Order columns by connection count then trigger.
    def _colkey(c: str):
        try:
            n = int(str(c).split("c")[0])
        except ValueError:
            n = 999
        return (n, "add" not in c)
    col_order = sorted(sat["col"].unique(), key=_colkey)
    pivot = sat.pivot_table(index="protocol", columns="col", values="retained_pct", aggfunc="mean", observed=True)
    pivot = pivot.reindex(columns=col_order)
    row_order = sat.sort_values("__o")["protocol"].astype(str).drop_duplicates().tolist()
    pivot = pivot.reindex(index=[p for p in row_order if p in pivot.index])

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, (axh, axl) = plt.subplots(1, 2, figsize=(12.2, 5.4), gridspec_kw={"width_ratios": [3.1, 1.0]})

    cbar_label = ("reload run vs reload-free runs (%)" if undiluted
                  else "throughput retained vs steady-state (%)")
    sns.heatmap(pivot, ax=axh, cmap="RdYlGn", center=100.0, vmin=80, vmax=110,
                annot=True, fmt=".0f", annot_kws={"size": T.FS["annot"]}, linewidths=0.5,
                linecolor="white", cbar_kws={"label": cbar_label, "shrink": 0.7})
    axh.set_yticklabels([protocol_label(str(p)) for p in pivot.index], rotation=0,
                        fontsize=T.FS["small"])
    axh.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=T.FS["annot"])
    axh.set_xlabel("")
    axh.set_ylabel("")
    T.panel_title(axh, "Throughput retained through a live reconfiguration (TCP)")

    # --- integrity ledger (right) ---
    axl.axis("off")
    n_scn = len(sat)
    lost = int(pd.to_numeric(sat.get("total_lost"), errors="coerce").fillna(0).sum()) if "total_lost" in sat else None
    integ = int(pd.to_numeric(sat.get("integrity_failures"), errors="coerce").fillna(0).sum()) if "integrity_failures" in sat else None
    bound = int(pd.to_numeric(sat.get("boundary_violations"), errors="coerce").fillna(0).sum()) if "boundary_violations" in sat else None
    med = np.nanmedian(sat["retained_pct"].to_numpy(dtype=float))
    worst = np.nanmin(sat["retained_pct"].to_numpy(dtype=float))
    lines = [
        ("reload scenarios", f"{n_scn}"),
        ("frames lost", "—" if lost is None else f"{lost:,}"),
        ("integrity failures", "—" if integ is None else f"{integ:,}"),
        ("boundary violations", "—" if bound is None else f"{bound:,}"),
        ("median retained", f"{med:.0f}%"),
        ("worst retained", f"{worst:.0f}%"),
    ]
    T.panel_title(axl, "Integrity ledger")
    y0 = 0.92
    for i, (k, v) in enumerate(lines):
        yy = y0 - i * 0.15
        axl.text(0.04, yy, k, fontsize=T.FS["small"], va="center", transform=axl.transAxes,
                 color=T.GREYS["annot"])
        axl.text(0.96, yy, v, fontsize=T.FS["small"], va="center", ha="right",
                 transform=axl.transAxes, color=T.GREYS["ink"])

    # The headline "0 frames lost" claim needs the loss counter to actually exist and be 0;
    # a missing counter must not be presented as a measured zero (only the softer counters
    # may be absent, since the claim wording mentions frames lost alone).
    zero_loss = (lost == 0) and (integ in (0, None)) and (bound in (0, None))
    # The harness arms exactly one reload per scenario (the reload timer fires once, during
    # the first measurement run — F18-1), so events == scenarios; do not multiply by runs.
    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.03)
    if undiluted:
        ret_clause = (f"the reload run held a median {med:.0f}% of each scenario's "
                      f"reload-free throughput")
    else:
        ret_clause = (f"throughput held at a median {med:.0f}% of steady-state "
                      f"(all-runs mean — see method note)")
    if zero_loss:
        T.add_takeaway(fig, f"Reconfiguring a saturated gateway is non-disruptive: 0 frames lost across "
                            f"{n_scn} reload events (one per scenario), and {ret_clause}.")
    else:
        T.add_takeaway(fig, f"Reload retains a median {med:.0f}% throughput across {n_scn} reload events "
                            f"(worst {worst:.0f}%); see the integrity ledger for any frame loss.")
    if undiluted:
        T.add_method_note(fig, "TCP gateway path only (the hot-reload executor is TCP). Exactly one reload "
                               "fires per scenario, inside the first measurement run; each cell is that "
                               "reload run's throughput vs the mean of the same scenario's reload-free runs. "
                               "Cell-level ±% at the few-percent scale is run-to-run noise, not reload cost — "
                               "the zero-loss ledger is the load-bearing result.")
        prov_basis = "retention = reload run vs same-scenario reload-free runs"
    else:
        T.add_method_note(fig, "TCP gateway path only (the hot-reload executor is TCP). Baseline = matched "
                               "single-gateway (scg-direct) matrix throughput. Per-run data is unavailable, so "
                               "each cell averages every measurement run while the single reload fires in only "
                               "one of them — retention is diluted toward steady-state; the zero-loss ledger "
                               "is the load-bearing result.")
        prov_basis = "baseline = matched scg-direct matrix throughput"
    T.add_provenance(fig, bundle.caption() + "  ·  saturation reloads only; " + prov_basis)
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)


def _sum_counter(frame: pd.DataFrame, col: str):
    """Sum a ledger counter over a frame; None when the counter is absent entirely."""
    if col not in frame.columns:
        return None
    return int(pd.to_numeric(frame[col], errors="coerce").fillna(0).sum())


def _make_print(bundle: RunBundle, saver: T.Saver, *, tbl: pd.DataFrame,
                 sat: pd.DataFrame, undiluted: bool) -> None:
    """Print variant: integrity ledger over all reload rows + saturation retention spread."""
    import matplotlib.pyplot as plt

    n_all = len(tbl)
    n_sat = len(sat)
    lost = _sum_counter(tbl, "total_lost")
    integ = _sum_counter(tbl, "integrity_failures")
    bound = _sum_counter(tbl, "boundary_violations")
    trig_counts = (tbl["reload_trigger"].astype(str).map(lambda t: _TRIG_SHORT.get(t, t))
                   .value_counts().to_dict() if "reload_trigger" in tbl.columns else {})

    ret = sat["retained_pct"].to_numpy(dtype=float)
    ret = ret[np.isfinite(ret)]
    med = float(np.nanmedian(ret)) if len(ret) else float("nan")
    worst = float(np.nanmin(ret)) if len(ret) else float("nan")
    best = float(np.nanmax(ret)) if len(ret) else float("nan")

    fig, (axl, axd) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.0, 1.5]})

    # --- ledger (left) ---
    axl.axis("off")
    trig_txt = " · ".join(f"{k} {v}" for k, v in sorted(trig_counts.items())) if trig_counts else "—"
    lines = [
        ("reload events", f"{n_all}"),
        ("at saturation", f"{n_sat}"),
        ("triggers", trig_txt),
        ("frames lost", "—" if lost is None else f"{lost:,}"),
        ("integrity failures", "—" if integ is None else f"{integ:,}"),
        ("boundary violations", "—" if bound is None else f"{bound:,}"),
    ]
    y0 = 0.94
    for i, (k, v) in enumerate(lines):
        yy = y0 - i * 0.16
        axl.text(0.02, yy, k, fontsize=T.FS["small"], va="center", transform=axl.transAxes,
                 color=T.GREYS["annot"])
        axl.text(0.98, yy, v, fontsize=T.FS["small"], va="center", ha="right",
                 transform=axl.transAxes, color=T.GREYS["ink"])
    T.panel_title(axl, "Integrity ledger (all reload events)")

    # --- retention distribution (right, saturation rows) ---
    if len(ret):
        lo = min(worst, 90.0)
        hi = max(best, 110.0)
        bins = np.linspace(lo - 1, hi + 1, 25)
        axd.hist(ret, bins=bins, color=T.GREYS["baseline"], edgecolor=T.GREYS["edge"],
                 linewidth=0.4)
        axd.axvline(100.0, color=T.GREYS["ink"], ls=":", lw=1.2, label="100% = undisturbed")
        axd.axvline(med, color=T.ACCENT, ls="--", lw=1.2, label=f"median retained ({med:.0f}%)")
        T.legend_inline(axd, loc="upper right")
        axd.set_xlabel("throughput retained through the reload run (%)")
        axd.set_ylabel("saturation reload scenarios")
        axd.grid(axis="y")
    else:
        axd.axis("off")

    zero_loss = (lost == 0) and (integ in (0, None)) and (bound in (0, None))
    sub_clause = f" ({n_sat} at saturation, {n_all - n_sat} at sub-saturation load)" if n_all > n_sat else ""
    if zero_loss and len(ret):
        take = (f"Live reconfiguration is non-disruptive: 0 frames lost across all {n_all} "
                f"reload events{sub_clause}, and the reload run held a median {med:.0f}% "
                f"(worst {worst:.0f}%) of each scenario's reload-free throughput.")
    elif len(ret):
        take = (f"Reload retains a median {med:.0f}% throughput across {n_sat} saturation "
                f"reload events (worst {worst:.0f}%); see the ledger for frame loss.")
    else:
        take = f"{n_all} reload events executed; see the integrity ledger."
    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.03)
    T.add_takeaway(fig, take)
    ret_basis = ("each scenario's reload run vs the mean of its reload-free runs"
                 if undiluted else "scenario mean vs matched steady-state baseline (diluted)")
    T.add_method_note(
        fig,
        "TCP gateway path only (the hot-reload executor is TCP). Exactly one reload fires "
        "per scenario, inside the first measurement run; retention = " + ret_basis + ". "
        "Cell-level ±% at the few-percent scale is run-to-run noise, not reload cost — the "
        "zero-loss ledger is the load-bearing result. Certificate rotation is driven end to "
        "end as its own trigger; its rule restart severs established connections by design, "
        "so it is reported separately from this ledger.",
    )
    T.add_provenance(fig, bundle.caption() + f"  ·  ledger over {n_all} reload rows; retention over {n_sat} saturation rows")
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
