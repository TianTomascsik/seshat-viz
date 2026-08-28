"""
F13 — Code coverage (line %) by workspace/crate, with a scenario-coverage fallback.

Preferred mode (when a ``coverage.json`` artifact is present): render the SCG / SCG-SESHAT /
ale-frame line-coverage percentages against the ≥80% thesis target, per workspace and — when
the artifact carries them — per crate. This answers "how well is the code itself tested?",
which the reader cares about more than how many benchmark scenarios ran.

Fallback mode (no coverage artifact): the original suite-coverage view — executed vs
skip-logged scenarios, broken down by connection count and scenario family — so a run whose
suite was incomplete cannot masquerade as complete. The denominator is honest about its own
provenance: without a plan manifest it is only ``executed + skip-logged`` (a lower bound on
the plan, since skip records can be lost from accounting, e.g. across ``--resume``), so the
executed share is labelled an upper bound; with a plan manifest (see :func:`_load_plan`) the
true plan total is used and planned-but-unaccounted scenarios get their own bar segment.
See :func:`loader._load_coverage` for the code-coverage schema and
``scripts/llvmcov_to_json.py`` for how that artifact is produced from ``cargo llvm-cov``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import theme as T
from ..loader import RunBundle

FIG_ID = "F13"
NAME = "f13_coverage"
TITLE = "Code coverage by workspace / crate"

_EXEC_COLOR = T.SEM["ok"]
_SKIP_COLOR = T.SEM["bad"]
_LOST_COLOR = T.GREYS["faint"]  # planned but neither executed nor skip-logged (accounting hole)
_COV_OK = T.SEM["ok"]           # at/above target
_COV_LOW = T.SEM["warn"]        # below target

# Nested per-invocation dir name (`YYYYMMDD-HHMMSS`) — local copy of the loader's naming
# convention so the plan search can walk sub-run dirs without importing private names.
_TS_DIR_RE = re.compile(r"^\d{8}-\d{6}$")


def _load_plan(run_dir: Path) -> "tuple[int, set[str], Path] | None":
    """
    Best-effort suite plan for the run: how many scenarios *should* have run.

    ``executed + skip-logged`` is only a lower bound on the plan — a skip record can be
    lost from accounting (the ``--resume`` consolidation rebuilds ``skipped.csv`` from
    scenario dirs, and skips leave none behind), which would let an incomplete suite
    masquerade as ~complete — the exact failure this fallback figure exists to expose
    (audit F13-1 / D1-1). When the harness drops a plan manifest into the run dir, that
    becomes the denominator. Accepted forms, searched in the run dir, its nested
    per-invocation dirs, then the parent wrapper:

        plan.json               {"total": N} and/or {"scenarios": ["name", ...]}
        planned_scenarios.txt   one scenario name per line

    Returns ``(total, names, source_path)`` — ``names`` empty when only a count is
    known — or ``None`` when no readable manifest exists.
    """
    import json

    candidates: list[Path] = [run_dir]
    if run_dir.is_dir():
        candidates += [c for c in sorted(run_dir.iterdir())
                       if c.is_dir() and _TS_DIR_RE.match(c.name)]
    if run_dir.parent != run_dir:
        candidates.append(run_dir.parent)

    for d in candidates:
        pj = d / "plan.json"
        if pj.is_file():
            try:
                with pj.open() as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            names = {str(s) for s in data.get("scenarios", []) or []}
            try:
                total = int(data.get("total", len(names)))
            except (TypeError, ValueError):
                total = len(names)
            total = max(total, len(names))
            if total > 0:
                return total, names, pj
        pt = d / "planned_scenarios.txt"
        if pt.is_file():
            try:
                names = {ln.strip() for ln in pt.read_text().splitlines() if ln.strip()}
            except OSError:
                continue
            if names:
                return len(names), names, pt
    return None


def make(bundle: RunBundle, saver: T.Saver) -> None:
    """Draw code coverage if a coverage.json artifact is present, else scenario coverage."""
    cov = getattr(bundle, "coverage", None)
    if isinstance(cov, dict) and cov.get("workspaces"):
        _make_code_coverage(bundle, saver, cov)
    else:
        _make_scenario_coverage(bundle, saver)


def _make_code_coverage(bundle: RunBundle, saver: T.Saver, cov: dict) -> None:
    """Horizontal line-coverage bars per workspace (bold) and crate (indented) vs the target."""
    import matplotlib.pyplot as plt

    target = float(cov.get("target_pct", 80.0))
    # Flatten to bars: each workspace, then its crates indented beneath it.
    rows: list[tuple[str, float, bool]] = []
    for ws in cov.get("workspaces", []):
        pct = ws.get("line_pct")
        if pct is not None:
            rows.append((str(ws.get("name", "?")), float(pct), True))
        for cr in ws.get("crates", []) or []:
            cpct = cr.get("line_pct")
            if cpct is not None:
                rows.append(("    " + str(cr.get("name", "?")), float(cpct), False))
    if not rows:
        saver.record_skip(FIG_ID, NAME, "coverage.json carried no usable workspace line %")
        return

    labels = [r[0] for r in rows]
    pcts = [r[1] for r in rows]
    is_ws = [r[2] for r in rows]
    y = np.arange(len(rows))[::-1]  # first workspace on top

    fig_h = max(2.6, 0.42 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    colors = [_COV_OK if p >= target else _COV_LOW for p in pcts]
    ax.barh(y, pcts, color=colors,
            edgecolor=T.GREYS["edge"], linewidth=[0.7 if w else 0.4 for w in is_ws],
            height=[0.72 if w else 0.58 for w in is_ws])
    ax.axvline(target, color=T.ACCENT, ls="--", lw=1.0)
    ax.annotate(f"target {target:.0f}%", (target, y[0] + 0.6), color=T.GREYS["annot"],
                fontsize=T.FS["annot"], ha="center", va="bottom")
    for yi, p, w in zip(y, pcts, is_ws):
        ax.annotate(f"{p:.1f}%", (p, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=T.FS["small"], color=T.GREYS["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for tick, w in zip(ax.get_yticklabels(), is_ws):  # workspaces bold, crates lighter
        tick.set_fontweight("bold" if w else "normal")
        if not w:
            tick.set_color(T.GREYS["annot"])
    ax.set_xlim(0, max(100.0, max(pcts) + 6))
    ax.set_xlabel("line coverage (%)")
    ax.grid(axis="x")

    ws_names = ", ".join(str(w.get("name", "?")) for w in cov.get("workspaces", []))
    T.set_headline(fig, f"{TITLE}  —  {ws_names}", y=0.99, fontsize=12)
    if T.chrome_enabled():
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    else:
        fig.tight_layout()
    below = [labels[i].strip() for i, p in enumerate(pcts) if p < target and is_ws[i]]
    verdict = "all workspaces ≥ target" if not below else f"below target: {', '.join(below)}"
    T.add_takeaway(fig, f"Line coverage vs the ≥{target:.0f}% thesis bar — {verdict}.")
    T.add_provenance(fig, bundle.caption()
                     + (f"  ·  coverage generated {cov['generated']}" if cov.get("generated") else ""))
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)


def _make_scenario_coverage(bundle: RunBundle, saver: T.Saver) -> None:
    executed = len(bundle.summary)
    skipped_df = bundle.skipped
    n_skip = len(skipped_df)
    recorded = executed + n_skip

    # The skip ledger is as-RECORDED, not as-planned: `recorded` is only a lower bound
    # on the suite plan (skip rows can be lost from accounting, e.g. across --resume,
    # audit F13-1/D1-1). With a plan manifest we can count the hole exactly; without
    # one the figure must not claim plan coverage — only an upper bound.
    plan = _load_plan(bundle.run_dir)
    unaccounted = 0
    plan_src: "Path | None" = None
    if plan is not None:
        plan_total, plan_names, plan_src = plan
        if plan_names:
            seen: set = set()
            if "scenario" in bundle.summary.columns:
                seen |= set(bundle.summary["scenario"].astype(str))
            if "scenario" in skipped_df.columns:
                seen |= set(skipped_df["scenario"].astype(str))
            unaccounted = len(plan_names - seen)
        else:
            # Count-only manifest: clamp at 0 so an inconsistent (too-small) plan
            # never produces a negative segment.
            unaccounted = max(0, plan_total - recorded)
    total = recorded + unaccounted
    if total == 0:
        saver.record_skip(FIG_ID, NAME, "no coverage data (no executed or skipped scenarios)")
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.2, 1.2]})
    ax_head, ax_conn, ax_fam = axes

    # ── Panel 1: executed vs skip-logged (vs unaccounted, when a plan is known) ──────
    ax_head.barh([0], [executed], color=_EXEC_COLOR, edgecolor=T.GREYS["edge"], linewidth=0.5,
                 label="executed")
    ax_head.barh([0], [n_skip], left=[executed], color=_SKIP_COLOR, edgecolor=T.GREYS["edge"],
                 linewidth=0.5, label="skip-logged")
    if unaccounted:
        ax_head.barh([0], [unaccounted], left=[recorded], color=_LOST_COLOR,
                     edgecolor=T.GREYS["edge"], linewidth=0.5, hatch=T.HARNESS_HATCH,
                     label="unaccounted")
    if plan_src is not None:
        T.panel_title(ax_head,
                      f"{executed} executed · {n_skip} skip-logged\n· {unaccounted} unaccounted (plan {total})")
    else:
        T.panel_title(ax_head, f"{executed} executed · {n_skip} skip-logged\n(recorded scenarios only)")
    ax_head.set_yticks([])
    ax_head.set_xlim(0, total)
    # No xlabel here — the ticks count scenarios and the legend names the segments; an
    # xlabel at this position collided with the below-axes legend.
    T.legend_inline(ax_head, loc="lower center", ncol=3 if unaccounted else 2,
                    bbox_to_anchor=(0.5, -0.32))

    def _annotate_seg(count: int, center: float, color: str) -> None:
        # In-bar count labels only where the segment can hold them; a skinny segment's
        # label would bleed past the bar onto the background and read garbled, and its
        # exact count is already in the panel title.
        if count and count / total >= 0.05:
            ax_head.annotate(str(count), (center, 0), ha="center", va="center",
                             color=color, fontsize=T.FS["label"])

    _annotate_seg(executed, executed / 2, "white")
    _annotate_seg(n_skip, executed + n_skip / 2, "white")
    _annotate_seg(unaccounted, recorded + unaccounted / 2, T.GREYS["ink"])

    # ── Panel 2: skips by connection count (the high-concurrency wall) ───────────────
    _bar_breakdown(
        ax_conn,
        skipped_df,
        key="connections",
        title="Skip-logged by connection count",
        ylabel="connections",
        label_fmt=lambda v: f"{int(v)}c" if pd.notna(v) else "?",
        sort_numeric=True,
    )

    # ── Panel 3: skips by family (whole classes that did not run) ────────────────────
    _bar_breakdown(
        ax_fam,
        skipped_df,
        key="family",
        title="Skip-logged by scenario family",
        ylabel="family",
        label_fmt=lambda v: str(v) if pd.notna(v) else "other",
        sort_numeric=False,
    )

    # Reason breakdown in the footer (there may be several distinct reasons now that the
    # harness records a specific cause per skip rather than one generic message).
    reason_note = ""
    if n_skip and "reason" in skipped_df.columns:
        top = skipped_df["reason"].value_counts().head(3)
        reason_note = "  ·  top skip reasons: " + "; ".join(
            f"{n}× {str(r)[:48]}" for r, n in top.items()
        )
    T.set_headline(fig, f"Suite coverage: executed vs skip-logged scenarios  —  {bundle.label}",
                   y=0.99, fontsize=12)
    # Reserve headroom for the suptitle and a footer strip for provenance so the
    # per-panel titles do not collide with the figure title.
    if T.chrome_enabled():
        fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    else:
        fig.tight_layout()
    # Measurement-quality caveat over rows that actually CARRY the saturation
    # determination — matrix-lat / hotreload / connrate rows never emit one, so using
    # all executed rows as the denominator would silently count NA as not-limited
    # (audit F13-2).
    hl_note = "No harness-limited determination recorded"
    if "harness_limited" in bundle.summary.columns:
        hl = bundle.summary["harness_limited"]
        n_flag = int(hl.notna().sum())
        n_hl = int(hl.fillna(False).astype(bool).sum())
        if n_flag:
            hl_note = (f"Executed ≠ DUT-bound — {n_hl}/{n_flag} rows with a saturation "
                       f"determination are harness-limited")
            n_na = executed - n_flag
            if n_na > 0:
                hl_note += f" ({n_na} rows carry none)"

    # Family attribution is about the skip-logged rows only; "(all …)" is only honest
    # when every family fits in the parenthetical.
    fam = ""
    if n_skip and "family" in skipped_df.columns and skipped_df["family"].notna().any():
        fams = skipped_df["family"].astype(str).value_counts()
        if len(fams) <= 3:
            fam = " (all " + "/".join(fams.index) + ")"
        else:
            fam = " (top: " + "/".join(fams.index[:2]) + ", …)"

    # Data-driven takeaway: coverage vs the *honest* denominator + the quality caveat.
    if plan_src is not None:
        take = (f"{executed}/{total} planned scenarios executed ({100 * executed / total:.0f}%); "
                f"{n_skip} skip-logged{fam}"
                + (f" + {unaccounted} unaccounted (no result or skip record)" if unaccounted else "")
                + f". {hl_note}.")
        T.add_method_note(fig, f"Denominator = suite plan ({total} scenarios, {plan_src.name}); "
                               f"'unaccounted' = planned scenarios with neither a result row nor a skip record.")
        cov_note = f"  ·  coverage {executed}/{total} = {100 * executed / total:.0f}% of plan executed"
    else:
        take = (f"{executed}/{total} recorded scenarios executed "
                f"({100 * executed / total:.0f}% — upper bound on plan coverage); "
                f"{n_skip} skip-logged{fam}. {hl_note}.")
        T.add_method_note(fig, "Denominator = executed + skip-logged only (no plan manifest found); "
                               "skip records lost from accounting (e.g. across --resume) are invisible, "
                               "so the executed share is an upper bound on plan coverage.")
        cov_note = (f"  ·  coverage {executed}/{total} = {100 * executed / total:.0f}% "
                    f"of recorded executed (upper bound)")
    T.add_takeaway(fig, take)
    T.add_provenance(fig, bundle.caption() + cov_note + reason_note)
    saver.save(fig, NAME, fig_id=FIG_ID, title="Suite coverage: executed vs skip-logged scenarios")


def _bar_breakdown(ax, skipped_df: pd.DataFrame, *, key: str, title: str, ylabel: str,
                   label_fmt, sort_numeric: bool) -> None:
    """Horizontal bar chart of skip counts grouped by `key` (a column of skipped_df)."""
    T.panel_title(ax, title)
    if skipped_df.empty or key not in skipped_df.columns:
        ax.text(0.5, 0.5, "no skip records", ha="center", va="center",
                transform=ax.transAxes, color=T.GREYS["muted"])
        ax.set_yticks([])
        return

    counts = skipped_df[key].value_counts(dropna=False)
    if sort_numeric:
        counts = counts.sort_index(key=lambda idx: pd.to_numeric(idx, errors="coerce"))
    else:
        counts = counts.sort_values(ascending=True)

    labels = [label_fmt(v) for v in counts.index]
    y = np.arange(len(counts))
    # A sequential ramp so the heaviest bars read as "the deep end".
    cmax = max(int(counts.max()), 1)
    colors = [(0.16, 0.30 + 0.45 * (1 - c / cmax), 0.55 + 0.30 * (c / cmax)) for c in counts.values]
    ax.barh(y, counts.values, color=colors, edgecolor=T.GREYS["edge"], linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=T.FS["tick"])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("scenarios skip-logged")
    ax.grid(axis="x")
    for i, c in enumerate(counts.values):
        ax.annotate(str(int(c)), (c, i), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=T.FS["annot"], color=T.GREYS["annot"])
