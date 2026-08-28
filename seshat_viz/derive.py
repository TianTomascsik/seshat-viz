"""
derive.py — derived metrics computed from the raw SESHAT columns.

SESHAT already computes means, CI95, percentiles, headroom and analytical overhead, so
this module does NOT recompute statistics. It only *combines* existing columns into the
relationship metrics that the figures need to draw conclusions:

  * encryption cost relative to the routing baseline (throughput drop / latency inflation);
  * gateway insertion cost (scg vs direct);
  * normalized CPU cost: cycles-per-byte, instructions-per-byte, cache-miss-rate,
    context-switches-per-message;
  * goodput efficiency from the analytical encapsulation overhead;
  * a long-form latency-tail table reshaped from runs.csv percentile columns.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Percentile columns in runs.csv and the fraction of the distribution they represent.
_PCTL_COLS: Dict[str, float] = {
    "latency_p50_us": 0.50,
    "latency_p90_us": 0.90,
    "latency_p95_us": 0.95,
    "latency_p99_us": 0.99,
    "latency_p999_us": 0.999,
}


def add_normalized_costs(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Append normalized hardware-cost columns to a summary frame (no-op where data absent).

    Adds, when the source columns exist:
      cycles_per_byte, instr_per_byte, cache_miss_rate, ctxsw_per_byte, syscalls_per_byte.
    Per-byte (rather than per-message) keeps the metric comparable across payload sizes.
    """
    df = summary.copy()

    # Total bytes moved during the measured window, per scenario. runs.csv has exact
    # `bytes`, but the summary doesn't carry it; reconstruct from throughput × duration
    # is unreliable, so we fold it in from runs in build_summary_extras(). Here we use
    # perf_task_clock and counters directly which are already whole-run totals.
    def _safe_div(num: str, den: str) -> Optional[pd.Series]:
        if num in df.columns and den in df.columns:
            d = df[den].replace(0, np.nan)
            return df[num] / d
        return None

    # cache miss *rate* (misses per reference) is size-independent and very telling.
    rate = _safe_div("perf_cache_misses", "perf_cache_references")
    if rate is not None:
        df["cache_miss_rate"] = rate

    # IPC is already a column (perf_ipc); expose a copy under a clear name.
    if "perf_ipc" in df.columns:
        df["ipc"] = df["perf_ipc"]

    return df


def attach_bytes_from_runs(summary: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """
    Bring total `bytes` and `messages` (summed across reps) from runs.csv into the summary,
    then derive per-byte / per-message hardware costs that need a work denominator.
    """
    df = summary.copy()
    if runs is None or runs.empty or "scenario" not in runs.columns:
        return df

    agg_cols = {}
    if "bytes" in runs.columns:
        agg_cols["bytes"] = ("bytes", "sum")
    if "messages" in runs.columns:
        agg_cols["messages"] = ("messages", "sum")
    if not agg_cols:
        return df
    work = runs.groupby("scenario", observed=True).agg(**agg_cols).reset_index()
    df = df.merge(work, on="scenario", how="left")

    if "perf_cycles" in df.columns and "bytes" in df.columns:
        df["cycles_per_byte"] = df["perf_cycles"] / df["bytes"].replace(0, np.nan)
    if "perf_instructions" in df.columns and "bytes" in df.columns:
        df["instr_per_byte"] = df["perf_instructions"] / df["bytes"].replace(0, np.nan)
    if "perf_context_switches" in df.columns and "messages" in df.columns:
        df["ctxsw_per_kmsg"] = df["perf_context_switches"] / (df["messages"].replace(0, np.nan) / 1000.0)
    if "perf_syscalls" in df.columns and "messages" in df.columns:
        df["syscalls_per_msg"] = df["perf_syscalls"] / df["messages"].replace(0, np.nan)
    return df


# A core doing sustained relay work clocks well above this even at its lowest P-state;
# user-scope-demoted cycle counts on kernel-heavy rungs imply "clocks" 10–50× lower.
MIN_PLAUSIBLE_GHZ = 0.5


def perf_user_scope_only(d: pd.DataFrame) -> bool:
    """
    True when the populated hardware counters look user-scope-demoted (perf ran unprivileged
    at perf_event_paranoid>=2, so every event silently became ``:u``). summary.csv does not
    carry the event suffix, but the demotion leaves two numeric fingerprints:

      * cycles / task-clock implies an implausibly low clock rate for most rows — task-clock
        charges *all* CPU time the process used, while cycles:u misses the kernel-side share;
      * context-switches is zero on every populated row — switches occur at the kernel
        transition, definitionally invisible to a ``:u`` event.

    Either fingerprint alone marks the counters untrustworthy for cross-rung comparison: a
    cycles/byte ladder built from them measures *where* the work runs (user vs kernel), not
    what it costs (audit F9-2). Shared by F9 (withholds panels) and F30 (refuses the run).
    """
    if "perf_cycles" not in d.columns:
        return False
    cyc = pd.to_numeric(d["perf_cycles"], errors="coerce")
    if not cyc.notna().any():
        return False  # no counters at all — that is the callers' missing-data case
    if "perf_task_clock_ms" in d.columns:
        tclk = pd.to_numeric(d["perf_task_clock_ms"], errors="coerce")
        ghz = (cyc / (tclk * 1e6)).replace([np.inf, -np.inf], np.nan).dropna()
        # Median, not min: userspace-crypto rungs (integrity) legitimately show a full clock
        # even under :u; the demotion signature is *most* rows reading a sub-plausible clock.
        if len(ghz) and float(ghz.median()) < MIN_PLAUSIBLE_GHZ:
            return True
    if "perf_context_switches" in d.columns:
        ctx = pd.to_numeric(d["perf_context_switches"], errors="coerce").dropna()
        # >=3 rows so a single quiet row can't masquerade as the all-zero fingerprint.
        if len(ctx) >= 3 and not (ctx > 0).any():
            return True
    return False


def goodput_efficiency(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add `goodput_efficiency` = payload / (payload + overhead), the fraction of wire bytes
    that carry application payload. Uses the analytical encapsulation overhead column.
    """
    df = summary.copy()
    if "message_bytes" not in df.columns:
        return df
    overhead = df.get("encapsulation_overhead_bytes_analytical")
    if overhead is None:
        return df
    ovh = pd.to_numeric(overhead, errors="coerce").clip(lower=0)
    wire = df["message_bytes"].astype(float) + ovh
    df["goodput_efficiency"] = (df["message_bytes"].astype(float)) / wire.replace(0, np.nan)
    return df


def overhead_vs_baseline(
    summary: pd.DataFrame,
    *,
    metric: str,
    baseline_protocol: str = "none",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    For each (transport, message_bytes, chain) cell, express every protocol's `metric`
    relative to the routing (`none`) baseline in the same cell.

    Returns a frame with columns:
      transport, message_bytes, chain, protocol, value, baseline, delta_pct, retained_pct.
    `delta_pct` is signed cost (negative = worse than baseline when higher_is_better).
    """
    needed = {"transport", "protocol", "message_bytes", metric}
    if not needed.issubset(summary.columns):
        return pd.DataFrame()

    group_keys = ["transport", "message_bytes"]
    if "chain" in summary.columns:
        group_keys.append("chain")

    rows: List[dict] = []
    for keys, grp in summary.groupby(group_keys, observed=True):
        base = grp[grp["protocol"].astype(str) == baseline_protocol]
        if base.empty:
            continue
        base_val = float(base[metric].mean())
        if not np.isfinite(base_val) or base_val == 0:
            continue
        keymap = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        for _, r in grp.iterrows():
            proto = str(r["protocol"])
            if proto == baseline_protocol:
                continue
            val = float(r[metric])
            if not np.isfinite(val):
                continue
            retained = val / base_val * 100.0
            delta = retained - 100.0
            rows.append(
                {
                    **keymap,
                    "protocol": proto,
                    "value": val,
                    "baseline": base_val,
                    "retained_pct": retained,
                    "delta_pct": delta if higher_is_better else -delta,
                }
            )
    return pd.DataFrame(rows)


def dead_repeat_scenarios(runs: pd.DataFrame) -> set:
    """
    Scenario names where at least one repetition recorded zero work (throughput 0 or no
    messages) — the SHM/UDS multi-connection stall pathology. Their summary means silently
    average dead repeats with live ones (2026-07-07 audit D2-1: 457 rows), so comparison
    figures (F6 insertion cost, F7 tails) must exclude or disclose them. Apply to
    sustained-blast scenarios only: connrate/ping-pong reps legitimately report zero
    throughput.
    """
    if runs is None or runs.empty or "scenario" not in runs.columns:
        return set()
    checks = []
    if "throughput_gbps" in runs.columns:
        checks.append(pd.to_numeric(runs["throughput_gbps"], errors="coerce").fillna(0) <= 0)
    if "messages" in runs.columns:
        checks.append(pd.to_numeric(runs["messages"], errors="coerce").fillna(0) <= 0)
    if not checks:
        return set()
    bad = checks[0]
    for c in checks[1:]:
        bad = bad | c
    return set(runs.loc[bad, "scenario"].astype(str))


def gateway_insertion_cost(summary: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """
    Second-gateway insertion cost: pair 1-gateway (``chain=='direct'``, scg-direct) against
    2-gateway (``chain=='scg'``, scg-scg) rows and report the cost of the second hop on
    ``metric``.

    Restricted to the **matrix family**, where ``_direct``/``_scg`` are explicit topology
    tokens (elsewhere ``chain`` is a loader guess) and both sides are sustained-blast at the
    same shape. Pairs on **(transport, protocol, message_bytes, connections)** so a 1c row is
    never compared against a 1024c row, then reports the per-connection ratio. The returned
    ``direct``/``scg`` are the paired 1c values (for a representative dumbbell) while ``ratio``
    is the **median** 2gw/1gw ratio across the matched connection sweep.

    Returns: transport, protocol, message_bytes, direct, scg, ratio, delta_pct, n_pairs.
    """
    needed = {"transport", "protocol", "message_bytes", "chain", "connections", metric}
    if not needed.issubset(summary.columns):
        return pd.DataFrame()
    df = summary
    if "family" in df.columns:
        df = df[df["family"].astype(str) == "matrix"]
    if df.empty:
        return pd.DataFrame()
    keys = ["transport", "protocol", "message_bytes"]
    rows: List[dict] = []
    for kv, grp in df.groupby(keys, observed=True):
        direct = grp[grp["chain"] == "direct"]
        scg = grp[grp["chain"] == "scg"]
        if direct.empty or scg.empty:
            continue
        # Match per connection count; a pair needs both topologies at the same concurrency.
        dmap = direct.groupby("connections", observed=True)[metric].mean()
        smap = scg.groupby("connections", observed=True)[metric].mean()
        shared = [c for c in dmap.index if c in smap.index
                  and np.isfinite(dmap[c]) and np.isfinite(smap[c]) and dmap[c] != 0]
        if not shared:
            continue
        ratios = [float(smap[c] / dmap[c]) for c in shared]
        ratio = float(np.median(ratios))
        # Representative endpoints: the lowest matched connection count (usually 1c).
        c0 = min(shared)
        keymap = dict(zip(keys, kv))
        rows.append(
            {
                **keymap,
                "direct": float(dmap[c0]),
                "scg": float(smap[c0]),
                "ratio": ratio,
                "delta_pct": (ratio - 1.0) * 100.0,
                "n_pairs": len(shared),
            }
        )
    return pd.DataFrame(rows)


def tail_table(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape runs.csv percentile columns into a long tail table for CCDF plotting.

    Output columns: scenario, transport, protocol, message_bytes, percentile, ccdf,
    latency_us. `ccdf` = 1 - percentile (tail probability), suitable for a log-y CCDF.
    Percentiles are averaged across repetitions per scenario.
    """
    if runs is None or runs.empty:
        return pd.DataFrame()
    present = [c for c in _PCTL_COLS if c in runs.columns]
    if not present:
        return pd.DataFrame()

    id_cols = [c for c in ("scenario", "transport", "protocol", "message_bytes") if c in runs.columns]
    agg = runs.groupby(id_cols, observed=True)[present].mean().reset_index()

    long = agg.melt(id_vars=id_cols, value_vars=present, var_name="pctl_col", value_name="latency_us")
    long["percentile"] = long["pctl_col"].map(_PCTL_COLS)
    long["ccdf"] = 1.0 - long["percentile"]
    long = long.dropna(subset=["latency_us"])
    return long.sort_values(id_cols + ["percentile"])


def throughput_scenarios_only(d: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only sustained-blast throughput rows; drop everything whose throughput is not the
    gateway's capacity: connection-rate (``conn_*``, ``handshake_*``) and ping-pong
    (``pp_*``) runs, hot-reload rows (throughput measured *during* a config reload), paced
    ``lat_*``/``paced_*`` and saturation-sweep ``sat_*`` families, the SHM zero-copy
    microbenchmarks (``shmzc_*`` — a different measurement class than the gateway relay
    blast, whose leak crowned F1's headline callouts; audit D1-5), and the
    ``_latency_``/``_pingpong_`` *workload* sub-mode of the iface and profile families
    (a rate cap, not capacity).

    Prefers the loader-derived ``mode``/``reload_trigger`` factor columns (robust to scenario
    naming) and falls back to a name regex only when those are absent. Crucially it keys on
    the *workload* token that precedes the size (``_latency_<size>``), NOT the ``latency``
    *tuning* label — the old regex matched the tuning token and wrongly dropped genuine
    ``profile_*_latency_throughput_*`` blast rows while, on this run's naming, letting every
    ``pp_*``/``conn_*``/``hotreload_*`` row through.
    """
    if "scenario" not in d.columns:
        return d
    out = d
    name = out["scenario"].astype(str)

    if "mode" in out.columns:
        keep_mode = out["mode"].astype(str) == "throughput"
    else:
        keep_mode = ~(name.str.startswith(("pp_", "conn")) | name.str.contains("pingpong"))

    if "reload_trigger" in out.columns:
        not_reload = out["reload_trigger"].isna()
    else:
        not_reload = ~name.str.contains("hotreload")

    paced = (
        # paced_/shmzc_/handshake_ postdate the mode classifier's original vocabulary; the
        # loader now classifies them (paced/pingpong/connrate) but the name guard stays so
        # a summary loaded without _enrich_factors is still safe (audit D1-5/D2-2/D2-3).
        name.str.startswith(("lat_", "sat_", "paced_", "shmzc_", "handshake_"))
        | name.str.contains(r"_(?:latency|pingpong)_\d+[KM]?B", regex=True)
    )
    return out[keep_mode & not_reload & ~paced]


def size_match_for_protocol_compare(d: pd.DataFrame, min_protocols: int = 2) -> pd.DataFrame:
    """
    Restrict to payload sizes where at least `min_protocols` *encrypted* protocols were
    measured, so a per-byte/efficiency comparison of routing vs crypto is like-for-like.

    Crypto is sometimes probed at a couple of tiny sizes (e.g. TLS 1.3 at 64 B) where the
    per-byte cost explodes; including those would unfairly inflate routing's averaged cost.
    Keeping only well-populated sizes (typically 4096 B and 1200 B) avoids that bias.
    """
    if "message_bytes" not in d.columns or "protocol" not in d.columns:
        return d
    enc = d[d["protocol"].astype(str) != "none"]
    if enc.empty:
        return d
    counts = enc.groupby("message_bytes", observed=True)["protocol"].nunique()
    good = set(counts[counts >= min_protocols].index)
    if not good:
        good = {counts.idxmax()}  # fall back to the single best-covered size
    return d[d["message_bytes"].isin(good)]


def matched_cell(
    d: pd.DataFrame,
    vary,
    *,
    controls: Sequence[str] = ("connections", "message_bytes"),
    fixed: Optional[Dict[str, object]] = None,
    prefer: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Pin confounding dimensions to a single shared value so a comparison across ``vary``
    is like-for-like — the antidote to the TCP-centric matrix, where an unstratified
    aggregate silently blends a 1024-connection TCP row with a 1-connection SHM row.

    For each column in ``controls`` (default: ``connections`` and ``message_bytes``),
    choose the value that co-occurs with the *most distinct* ``vary`` combinations
    (maximal coverage), then filter ``d`` to it. ``fixed`` forces a control's value;
    ``prefer`` breaks coverage ties with ``"min"`` or ``"max"`` on the control value.
    The default tie-break keeps the largest value, *except* ``connections`` which keeps
    the smallest — 1 is the only connection count every transport shares, so it is the
    fair cross-transport anchor.

    ``vary`` is the column (or list of columns) the figure compares (e.g. ``"transport"``
    or ``["transport", "protocol"]``). Returns ``(filtered, chosen)`` where ``chosen``
    maps each pinned column to its value, ready for a provenance stamp
    (:func:`theme.fmt_cell`). Controls absent from ``d`` are skipped.
    """
    fixed = dict(fixed or {})
    prefer = dict(prefer or {})
    vary_cols = [vary] if isinstance(vary, str) else list(vary)
    vary_cols = [c for c in vary_cols if c in d.columns]
    chosen: Dict[str, object] = {}
    out = d
    for col in controls:
        if col not in out.columns:
            continue
        if col in fixed:
            val = fixed[col]
        else:
            sub = out.dropna(subset=[col])
            if sub.empty or not vary_cols:
                continue
            cov = sub.groupby(col, observed=True).apply(
                lambda g: g[vary_cols].drop_duplicates().shape[0]
            )
            if cov.empty:
                continue
            top = int(cov.max())
            cands = [c for c in cov.index if int(cov[c]) == top]
            tie = prefer.get(col, "min" if col == "connections" else "max")
            val = min(cands) if tie == "min" else max(cands)
        out = out[out[col] == val]
        chosen[col] = val
    return out, chosen


def scaling_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Throughput & p99 latency vs connection count, per (transport, protocol, chain).

    Restricts to the sustained-throughput matrix scenarios, and for each series picks the
    single message size with the most distinct connection counts (tie → largest size) so
    each scaling curve is drawn at one payload. Adds `tput_norm` (throughput relative to the
    series' lowest connection count) and `ideal_norm` (linear-scaling reference) so a figure
    can show scaling efficiency directly.

    Also carries the per-point `bottleneck` (SESHAT's classifier: `harness-io`, `scg-cpu`,
    `scg`, `host-saturated`) and `harness_limited` flag through the connection aggregation, so
    a figure can show *why* a curve flattens — on a single host the load generator's fixed
    sender/receiver core pools (or the serial per-connection relay) cap the curve well below
    linear, which is a testbed property, not a gateway scaling defect.
    """
    needed = {"transport", "protocol", "connections", "throughput_gbps_mean"}
    if not needed.issubset(summary.columns):
        return pd.DataFrame()
    d = throughput_scenarios_only(summary)
    # Restrict to the matrix family: it is the only one that sweeps connection count with an
    # explicit, uniform topology per series (chain token). iface/profile rows otherwise mix a
    # single-gateway iface row into a matrix scg series and confound the scaling curve.
    if "family" in d.columns and (d["family"].astype(str) == "matrix").any():
        d = d[d["family"].astype(str) == "matrix"]
    d = d[d["connections"].notna() & d["throughput_gbps_mean"].notna()].copy()
    if "chain" not in d.columns:
        d["chain"] = "scg"
    out: List[pd.DataFrame] = []
    for (tr, pr, ch), g in d.groupby(["transport", "protocol", "chain"], observed=True):
        if g["connections"].nunique() < 2:
            continue
        # pick the size carrying the most connection counts (then the largest size).
        by_size = g.groupby("message_bytes", observed=True)["connections"].nunique()
        if by_size.empty:
            continue
        best = by_size.sort_values(ascending=False).index[0]
        top = by_size.max()
        cands = [s for s in by_size.index if by_size[s] == top]
        best = max(cands)
        s = g[g["message_bytes"] == best].copy()
        aggs = dict(
            throughput_gbps_mean=("throughput_gbps_mean", "mean"),
            latency_p99_us_mean=("latency_p99_us_mean", "mean") if "latency_p99_us_mean" in g.columns else ("throughput_gbps_mean", "mean"),
        )
        # Carry the bottleneck classification through the per-connection aggregate so a figure
        # can annotate *why* the curve is flat. bottleneck → the dominant (modal) class across
        # the (usually single) rows at each connection count; harness_limited → True if any row
        # at that count was load-generator-bound.
        if "bottleneck" in g.columns:
            aggs["bottleneck"] = ("bottleneck", lambda x: (x.dropna().mode().iloc[0] if not x.dropna().mode().empty else pd.NA))
        if "harness_limited" in g.columns:
            aggs["harness_limited"] = ("harness_limited", lambda x: (True if x.dropna().any() else (False if x.notna().any() else pd.NA)))
        s = s.groupby("connections", observed=True).agg(**aggs).reset_index()
        s = s.sort_values("connections")
        base_c = s["connections"].iloc[0]
        base_t = s["throughput_gbps_mean"].iloc[0]
        s["tput_norm"] = s["throughput_gbps_mean"] / base_t if base_t else np.nan
        s["ideal_norm"] = s["connections"] / base_c
        s["transport"] = tr
        s["protocol"] = pr
        s["chain"] = ch
        s["message_bytes"] = best
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def rtt_inflation(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Closed-loop RTT vs open-loop blast p99 — the coordinated-omission story.

    For each ping-pong / profile scenario carrying `rtt_us_p99`, attach a **matched** open-loop
    blast p99 baseline and the inflation factor blast_p99 / rtt_p99.

    The baseline is matched on (protocol, transport) over sustained-blast 1-connection rows
    (``throughput_scenarios_only`` now excludes pp/conn/hotreload/paced rows, so the closed-loop
    rows can no longer contaminate their own baseline); it falls back to a protocol-only mean
    only when no same-transport blast row exists. Note the ratio divides a one-way blast p99 by
    a round-trip closed p99, so it is a **conservative lower bound** on true CO inflation.

    Columns: scenario, protocol, transport, datapath, profile_tuning, rtt_p50, rtt_p99,
    blast_p99, inflation.
    """
    if "rtt_us_p99" not in summary.columns or summary["rtt_us_p99"].notna().sum() == 0:
        return pd.DataFrame()
    closed = summary[summary["rtt_us_p99"].notna()].copy()
    keep = [c for c in ("scenario", "protocol", "transport", "message_bytes", "datapath",
                        "family", "profile_tuning", "rtt_us_p50", "rtt_us_p99") if c in closed.columns]
    closed = closed[keep].rename(columns={"rtt_us_p50": "rtt_p50", "rtt_us_p99": "rtt_p99"})

    if "latency_p99_us_mean" not in summary.columns:
        return closed
    m = throughput_scenarios_only(summary)
    # The baseline must be pure open-loop blast, like-for-like with the closed rows'
    # topology: restrict to the matrix family (drops cipher-sweep rows — a different
    # cipher config is not this row's baseline) and to the 1-gateway 'direct' chain
    # (a 2-gateway blast p99 is not a baseline for a direct-chain RTT), and drop any
    # row that itself carries an RTT percentile (closed-loop rows mislabeled as blast).
    # Each restriction applies only where the run offers it (audit F16-2).
    if "family" in m.columns and (m["family"].astype(str) == "matrix").any():
        m = m[m["family"].astype(str) == "matrix"]
    if "chain" in m.columns and (m["chain"] == "direct").any():
        m = m[m["chain"] == "direct"]
    if "rtt_us_p99" in m.columns:
        m = m[m["rtt_us_p99"].isna()]
    if "connections" in m.columns:
        m = m[(m["connections"] == 1) | (m["connections"].isna())]
    m = m.dropna(subset=["latency_p99_us_mean"])
    if m.empty:
        return closed
    # Match the blast baseline on (protocol, transport, size) so a 1 KB closed-loop RTT is not
    # divided by a blast p99 averaged over 64 B–64 KB (which polluted the F16 inflation figure,
    # e.g. ChaCha/size mixing). Fall back to (protocol, transport), then protocol-only, when a
    # size-exact blast row is unavailable.
    has_size = "message_bytes" in m.columns and "message_bytes" in closed.columns
    if has_size:
        by_pts = (m.groupby([m["protocol"].astype(str), m["transport"].astype(str),
                             m["message_bytes"]], observed=True)["latency_p99_us_mean"].mean())
    by_pt = (m.groupby([m["protocol"].astype(str), m["transport"].astype(str)], observed=True)
             ["latency_p99_us_mean"].mean())
    by_p = m.groupby(m["protocol"].astype(str), observed=True)["latency_p99_us_mean"].mean()

    def _blast(row) -> float:
        proto, tr = str(row["protocol"]), str(row.get("transport"))
        if has_size and pd.notna(row.get("message_bytes")):
            k3 = (proto, tr, row["message_bytes"])
            if k3 in by_pts.index:
                return float(by_pts[k3])
        if (proto, tr) in by_pt.index:
            return float(by_pt[(proto, tr)])
        return float(by_p[proto]) if proto in by_p.index else np.nan

    closed["blast_p99"] = closed.apply(_blast, axis=1)
    closed["inflation"] = closed["blast_p99"] / closed["rtt_p99"]
    return closed


def connsetup_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Connection-establishment scenarios: rate + handshake percentiles, tidied for a figure."""
    if "conns_per_sec" not in summary.columns or summary["conns_per_sec"].notna().sum() == 0:
        return pd.DataFrame()
    d = summary[summary["conns_per_sec"].notna()].copy()
    keep = [c for c in ("scenario", "protocol", "transport", "chain", "variant", "conn_threads",
                        "conns_per_sec", "conns_per_sec_ci95", "conn_handshake_p50_us",
                        "conn_handshake_p99_us", "resumed_fraction") if c in d.columns]
    return d[keep]


def hotreload_retention(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Hot-reload disruption ledger: throughput retained vs a matched steady-state baseline,
    plus the loss / integrity counters, for the saturation reload scenarios.

    Baseline = best matrix throughput at the same (protocol, transport, message_bytes,
    connections). Columns: scenario, protocol, connections, reload_trigger, reload_load,
    throughput_gbps_mean, baseline_gbps, retained_pct, loss_pct, total_lost,
    integrity_failures, boundary_violations.
    """
    if "reload_trigger" not in summary.columns or summary["reload_trigger"].notna().sum() == 0:
        return pd.DataFrame()
    hr = summary[summary["reload_trigger"].notna()].copy()
    # Matched steady-state baseline from the matrix family. Hot-reload runs are scg-direct
    # (one gateway), so restrict the baseline to the same 1-gateway topology (chain=='direct')
    # — pooling the 2-gateway scg rows made the baseline not like-for-like. With one matrix row
    # per (protocol,transport,size,connections) the max is then just that row's value.
    base = summary[summary["scenario"].astype(str).str.startswith("matrix")].copy()
    if "chain" in base.columns and (base["chain"] == "direct").any():
        base = base[base["chain"] == "direct"]
    bkeys = ["protocol", "transport", "message_bytes", "connections"]
    if not set(bkeys).issubset(base.columns) or base.empty:
        hr["baseline_gbps"] = np.nan
    else:
        blut = base.groupby(bkeys, observed=True)["throughput_gbps_mean"].max().reset_index()
        blut = blut.rename(columns={"throughput_gbps_mean": "baseline_gbps"})
        hr = hr.merge(blut, on=bkeys, how="left")
    if "baseline_gbps" in hr.columns:
        hr["retained_pct"] = hr["throughput_gbps_mean"] / hr["baseline_gbps"].replace(0, np.nan) * 100.0
    keep = [c for c in ("scenario", "protocol", "connections", "reload_trigger", "reload_load",
                        "throughput_gbps_mean", "baseline_gbps", "retained_pct", "loss_pct",
                        "total_lost", "integrity_failures", "boundary_violations") if c in hr.columns]
    return hr[keep]


def jitter_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Per (transport, protocol) jitter (PDV) alongside mean latency, for a determinism view."""
    if "jitter_us_mean" not in summary.columns or summary["jitter_us_mean"].notna().sum() == 0:
        return pd.DataFrame()
    d = throughput_scenarios_only(summary)
    cols = ["jitter_us_mean"]
    for c in ("latency_mean_us", "latency_p99_us_mean", "loss_pct", "throughput_gbps_mean"):
        if c in d.columns:
            cols.append(c)
    g = d.dropna(subset=["jitter_us_mean"]).groupby(["transport", "protocol"], observed=True)[cols].mean().reset_index()
    return g


def _cipher_label(raw: str) -> str:
    """Normalize an AEAD suite token to a canonical short label, protocol-independent.

    Both the TLS 1.3 form (``aes_128_gcm_sha256``) and the TLS 1.2 ECDHE-RSA form
    (``ecdhe_rsa_aes128_gcm_sha256``) must map to the SAME label so identical AEADs share a
    color and the GCM/ChaCha grouping is correct across protocols.
    """
    s = str(raw).lower()
    if "chacha20" in s:
        return "ChaCha20-Poly1305"
    # Match the AES key size on the `aes<n>` token specifically — the trailing hash (sha256
    # / sha384) also contains digits and must not be mistaken for the key size.
    m = re.search(r"aes[_-]?(128|256)", s)
    if m:
        return f"AES-{m.group(1)}-GCM"
    return str(raw).replace("_", "-")


def cipher_table(summary: pd.DataFrame) -> pd.DataFrame:
    """The AEAD cipher sweep, tidied with a canonical suite label (protocol-independent) and
    cycles-per-byte if present. Keeps the ``protocol`` column so the figure can facet TLS 1.2
    ECDHE-RSA and TLS 1.3 rather than lumping both into one 'fixed protocol' panel."""
    if "cipher" not in summary.columns or summary["cipher"].notna().sum() == 0:
        return pd.DataFrame()
    d = summary[summary["cipher"].notna()].copy()
    d["cipher_label"] = d["cipher"].map(_cipher_label)
    keep = [c for c in ("scenario", "cipher", "cipher_label", "transport", "protocol",
                        "message_bytes", "connections", "harness_limited", "throughput_gbps_mean",
                        "throughput_gbps_ci95", "cpu_pct_mean", "gbps_per_core",
                        "cycles_per_byte", "perf_cycles")
            if c in d.columns]
    return d[keep]


def representative_scenarios(summary: pd.DataFrame, n_per_transport: int = 1) -> List[str]:
    """
    Pick a small, varied set of scenario names for detail figures (timeline etc.):
    the highest-throughput scenario per transport, de-duplicated.
    """
    if summary.empty or "throughput_gbps_mean" not in summary.columns:
        return list(summary.get("scenario", pd.Series(dtype=str)).head(4))
    picks: List[str] = []
    for _, grp in summary.groupby("transport", observed=True):
        top = grp.nlargest(n_per_transport, "throughput_gbps_mean")
        picks.extend(top["scenario"].tolist())
    return picks
