"""
Loader for the two-host "wire" benchmark campaigns (figures F26–F28).

The wire campaigns are produced by ``SCG-SESHAT/scripts/wire_bench.sh`` — outside
the unified SESHAT suite (see the ``remote_two_host_wire`` matrix limitation) —
into flat directories like ``results/wire-run/`` holding a ``wire_summary.csv``
whose columns mirror SESHAT's own ``summary.csv``. The standard :mod:`.loader`
cannot discover them (no ``scenarios/`` tree, different file name), and its
scenario-name grammar does not apply, so this module owns the whole path from
directory discovery to per-replicate aggregation.

Design constraints this loader encodes (each was a real defect found while
validating the campaign — do not relax them casually):

* The CSV's ``mode`` column means *medium* (``wire``/``loopback``). It is renamed
  to ``medium`` **before anything else**, because :func:`loader._enrich_factors`
  derives its own ``mode`` (throughput/pingpong) and would silently clobber it.
  Wire rows are therefore never routed through ``_enrich_factors`` at all.
* Replicates are encoded as a ``#rN`` scenario suffix. ``derive.py`` deliberately
  refuses to recompute statistics for SESHAT runs, and no cross-replicate helper
  exists there, so :func:`aggregate` (median + t-based CI95) lives here.
* Two campaigns predate the RTT probe's desync guard, and their 64 B RTT cells
  are contaminated by the echo-reframing bug (p50 below the link's ICMP floor —
  physically impossible). Pre-guard vintage is machine-detected via the absent
  ``rtt_resyncs`` key in ``work/*.send.json``, and only the affected rows
  (message size not a multiple of the echo's 256 B frame) are dropped; the
  256-multiple RTT values in the same campaigns are valid and kept.
* ``rtt_n``/``rtt_resyncs`` (and ``sender_gbps`` etc.) exist only in the per-cell
  ``work/*.send.json`` sidecars, never in the CSVs on disk — they are merged in.
* The wire path is mutual-TLS only (the gateway's M-13 check refuses anything
  else off loopback), kTLS-preferred by default: the CSV's bare ``tls``/``dtls``
  tokens are mapped into the house protocol vocabulary accordingly so
  :func:`theme.protocol_color` works unchanged.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .loader import _read_table

# --------------------------------------------------------------------------------------
# Campaign discovery / classification
# --------------------------------------------------------------------------------------

#: Campaign-directory roles, matched against the directory basename.
_ROLE_PATTERNS: List[tuple] = [
    (re.compile(r"^ab-(wire|loopback)-ktls(true|false)-(throughput|rtt)$"), "ab"),
    (re.compile(r"^knee-(wire|loopback)$"), "knee"),
    (re.compile(r"^(wire|lo)-qos3(-qdisc)?$"), "qos"),
    (re.compile(r"^(wire|lo)-rtt3$"), "rtt3"),
    (re.compile(r"^(wire-loopback-baseline|wire-run)$"), "baseline"),
]

_REP_RE = re.compile(r"#r(\d+)$")

#: Cells that MAY carry a closed-loop RTT probe (contamination scope).
_RTT_BEARING_RE = re.compile(r"^(rtt-|qos-|sweep-tcp-)")
#: Cells whose whole point is the RTT probe: empty RTT there means "did not
#: run", never "not applicable". Deliberately narrower than _RTT_BEARING_RE —
#: a sweep-tcp row without an RTT probe is a legitimate configuration and must
#: not be silently discarded as dead.
_RTT_REQUIRED_RE = re.compile(r"^(rtt-|qos-)")


@dataclass
class WireBundle:
    """Every wire campaign found under one results root, as one tidy frame."""

    root: Path
    df: pd.DataFrame                      # one row per (campaign, scenario)
    preflight: pd.DataFrame               # one row per campaign with a preflight probe
    dirs: List[Path] = field(default_factory=list)

    def provenance(self, roles: Optional[Iterable[str]] = None) -> str:
        """Short provenance string naming the campaign dirs actually used.

        This matters because ``manifest.json`` records the SESHAT run_dir the CLI
        was pointed at, which is *not* where the wire numbers come from.
        """
        df = self.df
        if roles is not None:
            df = df[df["role"].isin(set(roles))]
        campaigns = sorted(df["campaign"].unique())
        text = "wire campaigns: " + (" · ".join(campaigns) if campaigns else "none")
        if not self.preflight.empty:
            parts = []
            for medium in ("wire", "loopback"):
                sel = self.preflight[self.preflight["medium"] == medium]
                if not sel.empty and sel["rtt_us_mean"].notna().any():
                    parts.append(f"{medium} {sel['rtt_us_mean'].median():.0f} µs")
            if parts:
                text += "  ·  preflight RTT " + " / ".join(parts)
        return text


def _classify(dirname: str) -> Dict[str, object]:
    """Role/arm/qdisc metadata for one campaign directory name."""
    out: Dict[str, object] = {"role": "other", "arm": None, "qdisc": False}
    for pattern, role in _ROLE_PATTERNS:
        m = pattern.match(dirname)
        if not m:
            continue
        out["role"] = role
        if role == "ab":
            out["arm"] = "ktls" if m.group(2) == "true" else "user"
        elif role == "qos":
            out["qdisc"] = m.group(2) is not None
        break
    return out


def _load_send_sidecars(work: Path) -> Dict[str, dict]:
    """All ``work/<scenario>.send.json`` sidecars, keyed by scenario name."""
    sidecars: Dict[str, dict] = {}
    if not work.is_dir():
        return sidecars
    for path in work.glob("*.send.json"):
        try:
            sidecars[path.name[: -len(".send.json")]] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # a torn sidecar must not sink the campaign
    return sidecars


def _campaign_pre_guard(sidecars: Dict[str, dict]) -> bool:
    """True when this campaign predates the RTT probe's desync guard.

    The guard shipped together with the ``rtt_resyncs`` field, so a campaign
    whose RTT-bearing sidecars all lack that key ran with the reframing echo
    whose 64 B round trips are physically impossible batch artefacts.
    """
    rtt_sidecars = [s for s in sidecars.values() if "rtt_n" in s]
    if not rtt_sidecars:
        return False  # nothing RTT-bearing → vintage is irrelevant
    return not any("rtt_resyncs" in s for s in rtt_sidecars)


# Sidecar fields worth carrying onto rows (CSV-invisible measurement context).
_SIDE_FIELDS = ("rtt_n", "rtt_resyncs", "sender_gbps", "sent_bytes", "sent_msgs", "measure_s")


def _load_campaign(cdir: Path) -> Optional[pd.DataFrame]:
    """One campaign directory → enriched rows, or None when unreadable/empty."""
    merged = cdir / "wire_summary_merged.csv"
    csv = merged if merged.is_file() else cdir / "wire_summary.csv"
    df = _read_table(csv)
    if df.empty or "scenario" not in df.columns:
        return None

    # 1. medium FIRST (see the module docstring — the `mode` name is a landmine).
    df = df.rename(columns={"mode": "medium"})

    # 2. replicate split.
    scen = df["scenario"].astype(str)
    df["cell"] = scen.str.replace(_REP_RE, "", regex=True)
    reps = scen.str.extract(_REP_RE.pattern, expand=False)
    df["rep"] = pd.to_numeric(reps, errors="coerce").fillna(1).astype(int)

    # 3. campaign metadata.
    meta = _classify(cdir.name)
    df["campaign"] = cdir.name
    df["source_csv"] = csv.name
    df["role"] = meta["role"]
    df["arm"] = meta["arm"]
    df["qdisc"] = meta["qdisc"]

    # 4. sidecar merge (tolerate absence — not every cell writes one).
    sidecars = _load_send_sidecars(cdir / "work")
    for fld in _SIDE_FIELDS:
        values = scen.map(lambda name, f=fld: sidecars.get(name, {}).get(f))
        col = pd.to_numeric(values, errors="coerce")
        if fld in df.columns:  # CSVs rendered after the rtt_resyncs column landed
            df[fld] = df[fld].where(df[fld].notna(), col)
        else:
            df[fld] = col

    # 5. close the gbps_per_core vintage split (older wire rows never computed it).
    basis = df.get("throughput_gbps_mean")
    if basis is not None and "sender_gbps" in df.columns:
        basis = basis.where(basis.notna(), df["sender_gbps"])
    cpu = pd.to_numeric(df.get("cpu_pct_mean"), errors="coerce")
    computed = basis / (cpu / 100.0) if basis is not None else np.nan
    raw = pd.to_numeric(df.get("gbps_per_core"), errors="coerce")
    df["gbps_per_core_filled"] = raw.where(raw.notna(), computed)

    # 6. protocol vocabulary → house keys (M-13 makes the wire path mutual-only;
    #    kTLS-preferred unless this is the user-space arm of the A/B).
    df["protocol_raw"] = df["protocol"]
    tls_key = "tls/1.3+mtls" if meta["arm"] == "user" else "ktls/1.3+mtls"
    df["protocol"] = df["protocol"].map({"tls": tls_key, "dtls": "dtls/1.2+mtls"}).fillna(
        df["protocol"]
    )

    # 7. honesty flags (dropped later unless the caller opts out).
    rtt_bearing = df["cell"].str.match(_RTT_BEARING_RE)
    rtt_p50 = pd.to_numeric(df.get("rtt_us_p50"), errors="coerce")
    msg = pd.to_numeric(df.get("message_bytes"), errors="coerce")
    pre_guard = _campaign_pre_guard(sidecars)
    df["contaminated"] = pre_guard & rtt_bearing & (msg % 256 != 0)
    rtt_required = df["cell"].str.match(_RTT_REQUIRED_RE)
    df["dead"] = rtt_required & rtt_p50.isna()
    if "rtt_n" in df.columns:
        # rtt_n == 0 is the probe's own "no samples kept" marker — precise, and
        # applies to any cell that ran a probe (including sweep rows).
        df["dead"] |= rtt_bearing & (pd.to_numeric(df["rtt_n"], errors="coerce") == 0)
    return df


def _load_preflight(cdir: Path, medium: Optional[str]) -> Optional[dict]:
    path = cdir / "work" / "preflight.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return {
        "campaign": cdir.name,
        "medium": medium,
        "rtt_n": data.get("rtt_n"),
        "rtt_us_mean": data.get("rtt_us_mean"),
        "rtt_us_p50": data.get("rtt_us_p50"),
        "rtt_us_p99": data.get("rtt_us_p99"),
        "rtt_resyncs": data.get("rtt_resyncs"),
    }


def load_wire(
    root: Path | str,
    *,
    drop_contaminated: bool = True,
    drop_dead: bool = True,
) -> Optional[WireBundle]:
    """Discover and load every wire campaign under ``root``.

    Returns ``None`` when no campaign directory exists — the figures turn that
    into a ``record_skip`` rather than an error, since most SESHAT runs simply
    have no wire campaign next to them.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    frames: List[pd.DataFrame] = []
    preflights: List[dict] = []
    dirs: List[Path] = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (cdir / "wire_summary.csv").is_file() and not (
            cdir / "wire_summary_merged.csv"
        ).is_file():
            continue
        df = _load_campaign(cdir)
        if df is None:
            continue
        dirs.append(cdir)
        frames.append(df)
        medium = str(df["medium"].iloc[0]) if "medium" in df.columns and len(df) else None
        pf = _load_preflight(cdir, medium)
        if pf is not None:
            preflights.append(pf)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if drop_contaminated:
        df = df[~df["contaminated"]]
    if drop_dead:
        df = df[~df["dead"]]
    return WireBundle(
        root=root,
        df=df.reset_index(drop=True),
        preflight=pd.DataFrame(preflights),
        dirs=dirs,
    )


# --------------------------------------------------------------------------------------
# Replicate aggregation (median + t-based CI95)
# --------------------------------------------------------------------------------------

# Two-sided 97.5 % Student-t quantiles by degrees of freedom. Hardcoded because the
# viz venv deliberately carries no scipy; beyond df=30 the normal 1.96 is within 2 %.
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def ci95(values: Sequence[float] | np.ndarray) -> float:
    """Half-width of the two-sided 95 % CI on the mean (t-based; 0.0 for n < 2)."""
    arr = np.asarray([v for v in np.asarray(values, dtype=float) if np.isfinite(v)])
    n = arr.size
    if n < 2:
        return 0.0
    sd = float(np.std(arr, ddof=1))
    t = _T_TABLE.get(n - 1, 1.96)
    return t * sd / math.sqrt(n)


def aggregate(df: pd.DataFrame, by: List[str], cols: List[str]) -> pd.DataFrame:
    """Per-group replicate summary: ``<col>_med/_ci95/_min/_max`` plus ``n``.

    Lives here (not in ``derive.py``) on purpose: derive's contract is to never
    recompute statistics for SESHAT runs, whose CSVs already carry means and CIs.
    The wire campaigns encode replicates as ``#rN`` rows with no aggregate row,
    so the statistics genuinely do not exist until computed.
    """
    records = []
    for key, group in df.groupby(by, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(by, key))
        rec["n"] = len(group)
        for col in cols:
            vals = pd.to_numeric(group.get(col), errors="coerce").dropna().to_numpy()
            if vals.size:
                rec[f"{col}_med"] = float(np.median(vals))
                rec[f"{col}_ci95"] = ci95(vals)
                rec[f"{col}_min"] = float(np.min(vals))
                rec[f"{col}_max"] = float(np.max(vals))
            else:
                rec[f"{col}_med"] = np.nan
                rec[f"{col}_ci95"] = np.nan
                rec[f"{col}_min"] = np.nan
                rec[f"{col}_max"] = np.nan
        records.append(rec)
    return pd.DataFrame(records)
