"""
loader.py — turn a SESHAT results directory into tidy pandas DataFrames.

SESHAT (SCG-SESHAT) writes a tree of RFC-4180 CSV files per run::

    results/<YYYYMMDD-HHMMSS>/
        combined_summary.csv                       # one row per scenario (top-level)
        meta.csv                                   # suite metadata (key,value)
        sysinfo.csv                                # host fingerprint (key,value)
        <YYYYMMDD-HHMMSS>/                          # nested per-run dir
            summary.csv                            # one row per scenario (fallback)
            meta.csv / sysinfo.csv
            scenarios/<name>/
                config.csv                         # resolved scenario config (key,value)
                summary.csv                        # cross-run aggregate (key,value)
                runs.csv                           # one row per repetition
                saturation.csv                     # optional offered-load sweep
                system_metrics/gateway_pid_*.csv   # /proc timeseries during the run

Column sets vary between runs (e.g. perf_* counters are only present on perf runs;
runs.csv sometimes omits integrity_failures/boundary_violations; system_metrics
sometimes omits pss_kib). Everything here is therefore keyed by *column name* with
graceful fallbacks — never by position — so the same code ingests "normal" (procfs)
and "perf" runs alike.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Canonical orderings / labels (shared by every figure so colors & sort order are stable)
# --------------------------------------------------------------------------------------

# Ascending "security ladder" — plaintext routing first, mutual-auth / datagram later.
PROTOCOL_ORDER: List[str] = [
    "none",
    "tls/1.2",
    "tls/1.2+ale",
    "tls/1.3",
    "tls/1.3+ale",
    "tls/1.3+resume",
    "ktls/1.2",
    "ktls/1.3",
    "ktls/1.2+mtls",
    "ktls/1.3+mtls",
    "tls/1.2+mtls",
    "tls/1.3+mtls",
    "tls/1.2+integrity",
    "tls/1.3+integrity",
    "dtls/1.0",
    "dtls/1.2",
    "dtls/1.2+mtls",
]

# Order transports left→right for ladders/heatmaps; `tproxy` was added when the
# transparent-proxy interface entered the matrix (a run without it just omits it).
TRANSPORT_ORDER: List[str] = ["shm", "shm-slot", "unix", "tcp", "tproxy", "udp"]

# Human-friendly display names (kept short for legends / tick labels).
PROTOCOL_LABELS: Dict[str, str] = {
    "none": "routing",
    "tls/1.2": "TLS 1.2",
    "tls/1.3": "TLS 1.3",
    "tls/1.2+ale": "TLS 1.2 · ALE",
    "tls/1.3+ale": "TLS 1.3 · ALE",
    "tls/1.3+resume": "TLS 1.3 (resumed)",
    "ktls/1.2": "kTLS 1.2",
    "ktls/1.3": "kTLS 1.3",
    "ktls/1.2+mtls": "kmTLS 1.2",
    "ktls/1.3+mtls": "kmTLS 1.3",
    "tls/1.2+mtls": "mTLS 1.2",
    "tls/1.3+mtls": "mTLS 1.3",
    "tls/1.2+integrity": "TLS 1.2 (integrity)",
    "tls/1.3+integrity": "TLS 1.3 (integrity)",
    "dtls/1.0": "DTLS 1.0",
    "dtls/1.2": "DTLS 1.2",
    "dtls/1.2+mtls": "mDTLS 1.2",
}

TRANSPORT_LABELS: Dict[str, str] = {
    "shm": "SHM",
    "shm-slot": "SHM (slot)",
    "unix": "UDS",
    "tcp": "TCP",
    "tproxy": "TPROXY",
    "udp": "UDP",
}


def protocol_label(proto: str) -> str:
    """Display label for a raw protocol string (falls back to the raw value)."""
    return PROTOCOL_LABELS.get(proto, proto)


def transport_label(transport: str) -> str:
    """Display label for a raw transport string (falls back to upper-cased value)."""
    return TRANSPORT_LABELS.get(transport, str(transport).upper())


# --------------------------------------------------------------------------------------
# Bundle returned by load_run()
# --------------------------------------------------------------------------------------


@dataclass
class RunBundle:
    """Everything one SESHAT run directory contains, as tidy DataFrames + metadata."""

    run_dir: Path
    summary: pd.DataFrame                      # one row per executed scenario
    runs: pd.DataFrame                         # one row per (scenario, repetition)
    sysmetrics: pd.DataFrame                   # /proc timeseries, long form
    saturation: pd.DataFrame                   # offered-load sweep points, long form
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)  # scenario,reason (+ factors)
    meta: Dict[str, str] = field(default_factory=dict)
    sysinfo: Dict[str, str] = field(default_factory=dict)
    coverage: Optional[dict] = None            # code-coverage summary (coverage.json), if present
    wire: Optional[object] = None              # wire.WireBundle for the two-host campaigns (F26–F28)

    # -- convenience predicates used by figures to decide what they can draw ----------

    def has(self, *columns: str) -> bool:
        """True if `summary` carries every named column with at least one non-null value."""
        for col in columns:
            if col not in self.summary.columns or self.summary[col].notna().sum() == 0:
                return False
        return True

    def has_skips(self) -> bool:
        """True if any scenario was skipped (so a coverage figure has something to show)."""
        return not self.skipped.empty

    @property
    def label(self) -> str:
        """Short host/run label for figure captions, e.g. 'bench-host · 20260626-104017'."""
        host = self.sysinfo.get("hostname", "host")
        return f"{host} · {self.run_dir.name}"

    def caption(self) -> str:
        """One-line provenance string with host + CPU + kernel, for figure footers."""
        cpu = self.sysinfo.get("cpu_model", "?")
        kernel = self.sysinfo.get("kernel", "?")
        gov = self.sysinfo.get("governor", "?")
        return f"{self.sysinfo.get('hostname', 'host')} · {cpu} · kernel {kernel} · governor {gov}"


# --------------------------------------------------------------------------------------
# Low-level CSV helpers
# --------------------------------------------------------------------------------------


def _read_kv(path: Path) -> Dict[str, str]:
    """Read a two-column key,value CSV into a dict (empty dict if missing)."""
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        # Some key/value files have a 'key,value' header; others don't. Detect & keep.
        if header and not (len(header) == 2 and header[0] == "key" and header[1] == "value"):
            if len(header) >= 2:
                out[header[0]] = header[1]
        for row in reader:
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def _is_boollike(series: pd.Series) -> bool:
    """True if the column is real booleans or only the tokens true/false (case-insensitive).

    pandas read_csv parses ``true``/``false`` into Python ``bool`` objects, which
    ``pd.to_numeric`` would happily turn into 1.0/0.0 — so these columns must be excluded
    from numeric coercion and handled as booleans instead.
    """
    non_null = series.dropna()
    if non_null.empty:
        return False
    tokens = {str(v).strip().lower() for v in non_null.unique()}
    return tokens <= {"true", "false"}


def _read_table(path: Path) -> pd.DataFrame:
    """Read a columnar CSV, coercing numeric-looking columns to numbers."""
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # SESHAT leaves not-applicable cells empty (e.g. rtt_* on throughput rows). Coerce
    # every object column that is actually numeric; leave true strings & booleans alone.
    for col in df.columns:
        if df[col].dtype == object:
            if _is_boollike(df[col]):
                continue  # boolean column — normalized later in _enrich_factors
            coerced = pd.to_numeric(df[col], errors="coerce")
            # Only adopt the numeric view if it didn't wipe out real string content.
            non_null_orig = df[col].notna() & (df[col].astype(str).str.len() > 0)
            if non_null_orig.sum() == 0 or coerced.notna().sum() >= non_null_orig.sum():
                df[col] = coerced
    return df


# --------------------------------------------------------------------------------------
# Scenario-name parsing → analysis factors
# --------------------------------------------------------------------------------------

_CONN_RE = re.compile(r"_(\d+)c(?:_|$)")
# Accept both the raw-byte token (`_4096B_`) and the K/M-suffixed form (`_1KB_`, `_64KB_`).
_SIZE_RE = re.compile(r"_(\d+)([KM]?)B(?:_|$)")
_THREAD_RE = re.compile(r"_(\d+)thread")
_TUNING_RE = re.compile(r"_(latency|balanced|throughput)(?:_|$)")
_CIPHER_RE = re.compile(r"^cipher_(?:tls\d+_)?(?:tls_)?(.+)$")
_SIZE_MULT = {"": 1, "K": 1024, "M": 1024 * 1024}


def _parse_size(name: str):
    """Message size in bytes from a `_<n>[K|M]B_` token, or NA if none present."""
    m = _SIZE_RE.search(name)
    if not m:
        return pd.NA
    return int(m.group(1)) * _SIZE_MULT.get(m.group(2), 1)


def _scenario_family(name: str) -> str:
    """Coarse family from the scenario-name prefix (matrix/iface/hotreload/...)."""
    # The closed-loop ping-pong RTT grid (`matrix_lat_*`) is a DISTINCT family from the
    # throughput matrix — otherwise its rtt-only rows (near-zero throughput) would pollute
    # every `family=="matrix"` throughput comparison (F1/F3/F4). Match it before "matrix".
    if name.startswith("matrix_lat_"):
        return "matrix-latency"
    for prefix in (
        "matrix", "iface", "hotreload", "cipher", "profile",
        # paced_* (rate-capped encrypted datagram runs), shmzc_* (SHM zero-copy
        # microbenchmarks) and handshake_* (cert/kex connection-rate sweeps) landed in
        # 'other' before 2026-07-07, which let them leak into blast pools (audit D1-5).
        "paced", "shmzc", "handshake",
        "conn", "pp", "sat", "lat", "baseline", "scg", "direct",
    ):
        if name.startswith(prefix):
            return prefix
    return "other"


def _scenario_chain(name: str) -> str:
    """
    Topology token carried explicitly in the scenario name: 'scg' (scg-scg, two gateway
    hops in the matrix family), 'direct' (scg-direct / a loopback baseline), or 'n/a' when
    the name carries no topology token.

    Only the matrix family tags every scenario with `_scg`/`_direct`. Other families
    (hotreload, cipher, non-`_direct` profile, iface uds/shm/tproxy) leave it unmarked, so
    returning a *guessed* 'scg' here silently equated them with the two-gateway matrix
    topology and corrupted every chain-keyed comparison — return 'n/a' and let figures scope
    to the matrix family (or read `n_gateways`) instead.
    """
    if "_scg" in name or name.startswith("scg"):
        return "scg"
    if "_direct" in name or name.startswith("direct") or "loopback" in name:
        return "direct"
    return "n/a"


def _enrich_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add family/chain/connections factor columns + ordered categoricals to a summary."""
    if df.empty:
        return df
    df = df.copy()
    names = df["scenario"].astype(str)
    df["family"] = names.map(_scenario_family)
    df["chain"] = names.map(_scenario_chain)

    # connections: prefer the real column, else recover from the scenario name.
    if "connections" not in df.columns:
        df["connections"] = pd.NA
    parsed_conn = names.str.extract(_CONN_RE.pattern, expand=False)
    df["connections"] = df["connections"].fillna(pd.to_numeric(parsed_conn, errors="coerce"))

    # message_bytes: same idea (recover from the _<n>[K|M]B token if the column is absent —
    # needed for skipped.csv rows, whose names use the KB form, e.g. `_1KB_`).
    if "message_bytes" not in df.columns:
        df["message_bytes"] = pd.NA
    parsed_size = pd.to_numeric(names.map(_parse_size), errors="coerce")
    df["message_bytes"] = df["message_bytes"].fillna(parsed_size)

    # Workload factors derived from the scenario name (all additive — existing figures that
    # never reference these columns are unaffected; new figures key on them).
    #   reload_trigger / reload_load : the hot-reload suite (hotreload_*).
    #   conn_threads                 : connrate scenarios (conn_*_<n>thread).
    #   cipher                       : the cipher_tls13_<suite> sweep.
    #   profile_tuning               : profile_* tuning target (latency|balanced|throughput).
    #   mode                         : 'pingpong' for closed-loop RTT scenarios, else 'throughput'.
    def _reload_trigger(n: str):
        for t in ("add_connection", "remove_connection", "invalid_config"):
            if t in n:
                return t
        return pd.NA

    def _reload_load(n: str):
        if "hotreload" not in n:
            return pd.NA
        return "sub_saturation" if "sub_saturation" in n else ("saturation" if "saturation" in n else pd.NA)

    def _cipher(n: str):
        m = _CIPHER_RE.match(n)
        if not m:
            return pd.NA
        # Cipher scenarios may carry a trailing `_<n>B` size token (the cipher size grid);
        # strip it so the cipher factor is the suite alone (size lives in message_bytes).
        return re.sub(r"_\d+B$", "", m.group(1))

    def _framing(n: str):
        # Application framing over the (UDP) datagram path — the ETCS ALEPKT framing
        # (`matrix_ale_*`) vs the raw length-prefixed UDP-over-TLS (`matrix_raw_*`). Both carry
        # `protocol=tls/1.3`, so this is the only column that tells them apart.
        if "_ale_" in n:
            return "ale"
        if "_raw_" in n:
            return "raw"
        return pd.NA

    def _mode(n: str):
        # Closed-loop ping-pong (honest RTT) and connection-rate runs are not throughput.
        # shmzc_*_rtt_* are closed-loop RTT probes of the zero-copy path (their
        # latency_p99 IS an rtt percentile — audit D2-3/F16-2).
        if n.startswith("pp_") or "pingpong" in n or n.startswith("matrix_lat_"):
            return "pingpong"
        if n.startswith("shmzc_") and "_rtt" in n:
            return "pingpong"
        # handshake_* (cert/kex sweeps) are connection-rate benchmarks exactly like
        # conn_* — conns_per_sec populated, throughput 0.0 (audit D2-2).
        if n.startswith(("conn", "handshake_")):
            return "connrate"
        # Paced / saturation-sweep families and the `_latency_`/`_pingpong_` *workload*
        # sub-mode of iface & profile scenarios rate-cap the offered load, so their
        # throughput is not capacity. The workload token sits immediately before the size
        # token — `_latency_<size>` is the paced workload, while `_latency_<word>` is a
        # tuning label that must stay classified 'throughput'. paced_* is the encrypted
        # datagram sub-suite paced below the loss knee (audit D1-5).
        if n.startswith(("lat_", "paced_")):
            return "paced"
        if n.startswith("sat_"):
            return "saturation"
        if re.search(r"_(?:latency|pingpong)_\d+[KM]?B", n):
            return "paced"
        return "throughput"

    #   datapath   : 'loopback' for the raw no-gateway baseline, else 'gateway'.
    #   n_gateways : hops on the secured path — 0 (loopback / profile_direct), 1 (scg-direct,
    #                SingleGateway), 2 (scg-scg, ScgToScg). scg-scg (two gateways) exists ONLY
    #                in the matrix family; `_scg` in the iface/conn/lat/sat families is a
    #                single gateway, and profile_direct_* is a zero-gateway loopback baseline
    #                whose name lacks the 'loopback' token — special-case both so the count is
    #                true (this is exactly the disambiguation `chain` alone cannot make).
    def _datapath(n: str):
        return "loopback" if ("loopback" in n or n.startswith("profile_direct")) else "gateway"

    def _n_gateways(n: str):
        if "loopback" in n or n.startswith("profile_direct"):
            return 0
        if "_scg" in n or n.startswith("scg"):
            return 2 if n.startswith("matrix") else 1
        return 1

    #   variant : a name-only handshake sub-scheme the protocol column doesn't capture —
    #             'psk' (pre-shared key) or 'resumed' (session resumption) on connrate
    #             scenarios — so two rows sharing a protocol string stay distinguishable.
    def _variant(n: str):
        if "_psk" in n:
            return "psk"
        if "resume" in n:
            return "resumed"
        return pd.NA

    df["reload_trigger"] = names.map(_reload_trigger)
    df["reload_load"] = names.map(_reload_load)
    df["conn_threads"] = pd.to_numeric(names.str.extract(_THREAD_RE.pattern, expand=False), errors="coerce")
    df["cipher"] = names.map(_cipher)
    df["app_framing"] = names.map(_framing)
    df["profile_tuning"] = names.str.extract(_TUNING_RE.pattern, expand=False)
    df["mode"] = names.map(_mode)
    df["datapath"] = names.map(_datapath)
    df["n_gateways"] = names.map(_n_gateways)
    df["variant"] = names.map(_variant)

    # Ordered categoricals so groupby/sort/hue ordering is stable across every figure.
    if "protocol" in df.columns:
        present = [p for p in PROTOCOL_ORDER if p in set(df["protocol"])]
        extra = [p for p in df["protocol"].dropna().unique() if p not in present]
        df["protocol"] = pd.Categorical(df["protocol"], categories=present + extra, ordered=True)
        df["protocol_label"] = df["protocol"].astype(str).map(protocol_label)
    if "transport" in df.columns:
        # The fixed-slot SHM ring shares transport="shm" with the byte-stream ring
        # (SESHAT records no ring-kind column), so figures would silently average the
        # two. Split it into its own "shm-slot" series off the only surviving signal —
        # the 'shmslot' scenario-name token — before the transport column is frozen
        # into an ordered categorical below.
        slot = names.str.contains("shmslot", na=False)
        if slot.any():
            df.loc[slot, "transport"] = "shm-slot"
        present = [t for t in TRANSPORT_ORDER if t in set(df["transport"])]
        extra = [t for t in df["transport"].dropna().unique() if t not in present]
        df["transport"] = pd.Categorical(df["transport"], categories=present + extra, ordered=True)
        df["transport_label"] = df["transport"].astype(str).map(transport_label)

    # harness_limited / overloaded / capture-verified arrive either as real Python bools
    # (pandas parsed "true"/"false") or as those strings — normalize both to bool/NA.
    def _to_bool(value) -> object:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return pd.NA
        token = str(value).strip().lower()
        if token in ("true", "1", "yes"):
            return True
        if token in ("false", "0", "no"):
            return False
        return pd.NA

    for boolcol in ("harness_limited", "overloaded", "encapsulation_overhead_capture_verified"):
        if boolcol in df.columns:
            df[boolcol] = df[boolcol].map(_to_bool)
    return df


# --------------------------------------------------------------------------------------
# Run-directory discovery
# --------------------------------------------------------------------------------------

_TS_RE = re.compile(r"^\d{8}-\d{6}$")
# Top-level run dirs may carry a suffix naming the invocation (`-procfs`, `-perf`,
# `-rerun`, …). The old pure-timestamp `_TS_RE` made them invisible to auto-pick, so
# pointing the CLI at `results/` silently resolved to an OLD unsuffixed run while the
# newest data sat in a suffixed wrapper (audit D4-1). Nested per-invocation dirs stay
# pure timestamps, so `_TS_RE` remains correct for nested discovery.
_TOP_RUN_RE = re.compile(r"^\d{8}-\d{6}(?:[-_][A-Za-z0-9_-]+)?$")


def _is_run_dir(path: Path) -> bool:
    """A directory that directly contains scenario results (combined or nested summary)."""
    if not path.is_dir():
        return False
    if (path / "combined_summary.csv").is_file():
        return True
    if (path / "scenarios").is_dir():
        return True
    # A timestamped wrapper whose nested timestamp dir holds the data.
    return any(_TS_RE.match(c.name) and c.is_dir() for c in path.iterdir())


def resolve_run_dir(results_root: Path) -> Path:
    """
    Accept either a specific run dir or a `results/` root and return the run dir to load.

    If given the `results/` root, picks the newest timestamped subdirectory that actually
    contains scenario data — including suffixed wrappers like `20260705-011302-procfs`
    (ordered by their leading timestamp).
    """
    results_root = Path(results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results path does not exist: {results_root}")

    # Given a concrete run dir already?
    if (results_root / "combined_summary.csv").is_file() or (results_root / "scenarios").is_dir():
        return results_root

    def _has_summary(p: Path) -> bool:
        # "Newest run WITH DATA": an aborted invocation leaves a wrapper with scenario
        # skip stubs but no consolidated summary — auto-pick must skip it.
        if (p / "combined_summary.csv").is_file() or (p / "summary.csv").is_file():
            return True
        return any(
            (c / "summary.csv").is_file() or (c / "combined_summary.csv").is_file()
            for c in p.iterdir() if c.is_dir() and _TS_RE.match(c.name)
        )

    candidates = sorted(
        (c for c in results_root.iterdir()
         if _TOP_RUN_RE.match(c.name) and _is_run_dir(c) and _has_summary(c)),
        key=lambda p: p.name[:15],  # order by the leading YYYYMMDD-HHMMSS timestamp
    )
    if not candidates:
        # Maybe results_root *is* a timestamped wrapper with a single nested run.
        if _is_run_dir(results_root):
            return results_root
        raise FileNotFoundError(f"no run directories with data found under {results_root}")
    return candidates[-1]


def _find_scenarios_roots(run_dir: Path) -> List[Path]:
    """
    Locate every `scenarios/` directory belonging to this run.

    A SESHAT top-level results dir often holds *several* nested per-invocation run dirs
    (one `seshat run` each, e.g. the routing/crypto/saturation passes of a single
    `collect_perf_data.sh` call), whose union is `combined_summary.csv`. We therefore
    gather the direct `scenarios/` plus every nested timestamp dir's `scenarios/` so the
    per-scenario detail (runs.csv, system_metrics, saturation.csv) matches the summary.
    """
    roots: List[Path] = []
    direct = run_dir / "scenarios"
    if direct.is_dir():
        roots.append(direct)
    for c in sorted(run_dir.iterdir()):
        if c.is_dir() and _TS_RE.match(c.name) and (c / "scenarios").is_dir():
            roots.append(c / "scenarios")
    return roots


def _load_skipped(run_dir: Path) -> pd.DataFrame:
    """
    Concatenate every per-sub-run ``skipped.csv`` (columns: ``scenario,reason``).

    The skip records live in each nested per-invocation run dir (never the top-level
    combined dir), mirroring the layout :func:`_find_scenarios_roots` walks. Returns an
    empty frame (with the expected columns) when nothing was skipped, so callers can rely
    on the columns existing. Each row is tagged with the ``sub_run`` it came from and
    enriched with the same name-derived factors as the summary (family / connections / …).
    """
    candidates = [run_dir] + [
        c for c in sorted(run_dir.iterdir()) if c.is_dir() and _TS_RE.match(c.name)
    ]
    frames: List[pd.DataFrame] = []
    for d in candidates:
        sk = d / "skipped.csv"
        if not sk.is_file():
            continue
        df = pd.read_csv(sk)
        if df.empty or "scenario" not in df.columns:
            continue
        df["sub_run"] = d.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["scenario", "reason"])
    out = pd.concat(frames, ignore_index=True)
    if "reason" not in out.columns:
        out["reason"] = "unknown"
    return _enrich_factors(out)


def _find_summary_csv(run_dir: Path) -> Optional[Path]:
    """Top-level combined_summary.csv, else the nested per-run summary.csv."""
    top = run_dir / "combined_summary.csv"
    if top.is_file():
        return top
    nested = sorted(
        (c / "summary.csv")
        for c in run_dir.iterdir()
        if c.is_dir() and _TS_RE.match(c.name) and (c / "summary.csv").is_file()
    )
    if nested:
        return nested[-1]
    # Last resort: a summary.csv directly in the run dir.
    if (run_dir / "summary.csv").is_file():
        return run_dir / "summary.csv"
    return None


def _load_coverage(run_dir: Path, override: Optional[Path] = None) -> Optional[dict]:
    """
    Load a code-coverage summary (``coverage.json``) if one is available.

    Searched in order: an explicit ``override`` path, ``<run_dir>/coverage.json``, then
    ``<results-root>/coverage.json`` (so a single CI-emitted artifact can serve every run).
    The expected schema is the compact form produced by ``scripts/llvmcov_to_json.py``::

        {"target_pct": 80.0,
         "workspaces": [{"name": "SCG", "line_pct": 82.1, "lines_covered": .., "lines_total": ..,
                         "crates": [{"name": "gateway", "line_pct": 85.0}, ...]}, ...]}

    Returns the parsed dict, or ``None`` when no readable coverage file is found (F13 then
    falls back to scenario-execution coverage).
    """
    import json

    candidates: List[Path] = []
    if override is not None:
        candidates.append(Path(override))
    candidates.append(run_dir / "coverage.json")
    candidates.append(run_dir.parent / "coverage.json")
    for path in candidates:
        if path and path.is_file():
            try:
                with path.open() as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(data, dict) and data.get("workspaces"):
                return data
    return None


def _find_meta(run_dir: Path, name: str) -> Dict[str, str]:
    """Read a key/value file (meta.csv / sysinfo.csv) from run_dir or its nested run dir."""
    here = _read_kv(run_dir / name)
    if here:
        return here
    for c in sorted(run_dir.iterdir()):
        if c.is_dir() and _TS_RE.match(c.name):
            nested = _read_kv(c / name)
            if nested:
                return nested
    return {}


# --------------------------------------------------------------------------------------
# Per-scenario detail loading (runs.csv, system_metrics, saturation.csv, config.csv)
# --------------------------------------------------------------------------------------


def _load_runs(scenarios_roots: List[Path], summary: pd.DataFrame) -> pd.DataFrame:
    """Concatenate every scenarios/*/runs.csv, tagged with scenario + joined factors."""
    frames: List[pd.DataFrame] = []
    for scenarios_root in scenarios_roots:
        for scen_dir in sorted(scenarios_root.iterdir()):
            runs_csv = scen_dir / "runs.csv"
            if not runs_csv.is_file():
                continue
            df = _read_table(runs_csv)
            if df.empty:
                continue
            df["scenario"] = scen_dir.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    # Join transport/protocol/message_bytes from the summary for grouping.
    join_cols = [c for c in ("scenario", "transport", "protocol", "message_bytes", "family", "chain") if c in summary.columns]
    if "scenario" in join_cols and not summary.empty:
        runs = runs.merge(summary[join_cols].drop_duplicates("scenario"), on="scenario", how="left")
    return runs


def _load_sysmetrics(scenarios_roots: List[Path], summary: pd.DataFrame) -> pd.DataFrame:
    """Concatenate every scenarios/*/system_metrics/gateway_pid_*.csv as a long timeseries."""
    frames: List[pd.DataFrame] = []
    for scenarios_root in scenarios_roots:
        for scen_dir in sorted(scenarios_root.iterdir()):
            sysdir = scen_dir / "system_metrics"
            if not sysdir.is_dir():
                continue
            for csv_path in sorted(sysdir.glob("gateway_pid_*.csv")):
                df = _read_table(csv_path)
                if df.empty:
                    continue
                df["scenario"] = scen_dir.name
                df["pid"] = csv_path.stem.replace("gateway_pid_", "")
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    sysm = pd.concat(frames, ignore_index=True)
    join_cols = [c for c in ("scenario", "transport", "protocol", "message_bytes") if c in summary.columns]
    if "scenario" in join_cols and not summary.empty:
        sysm = sysm.merge(summary[join_cols].drop_duplicates("scenario"), on="scenario", how="left")
    return sysm


def _load_saturation(scenarios_roots: List[Path], summary: pd.DataFrame) -> pd.DataFrame:
    """Concatenate every scenarios/*/saturation.csv as a long sweep table."""
    frames: List[pd.DataFrame] = []
    for scenarios_root in scenarios_roots:
        for scen_dir in sorted(scenarios_root.iterdir()):
            sat_csv = scen_dir / "saturation.csv"
            if not sat_csv.is_file():
                continue
            df = _read_table(sat_csv)
            if df.empty:
                continue
            df["scenario"] = scen_dir.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    sat = pd.concat(frames, ignore_index=True)
    join_cols = [c for c in ("scenario", "transport", "protocol", "message_bytes") if c in summary.columns]
    if "scenario" in join_cols and not summary.empty:
        sat = sat.merge(summary[join_cols].drop_duplicates("scenario"), on="scenario", how="left")
    # Recover transport/protocol from the scenario name where the summary lacked the row.
    if "transport" not in sat.columns or sat["transport"].isna().all():
        sat["transport"] = sat["scenario"].astype(str).str.extract(r"_(tcp|udp|unix|shm|uds)_", expand=False)
    return sat


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def load_run(
    results_path: Path | str,
    *,
    coverage_path: Path | str | None = None,
    wire_results: Path | str | None = None,
) -> RunBundle:
    """
    Load one SESHAT run directory into a :class:`RunBundle`.

    `results_path` may be a specific `results/<ts>/` directory or the `results/` root
    (in which case the newest run with data is selected). `coverage_path` optionally points
    at a code-coverage summary (``coverage.json``) for F13; when omitted it is auto-discovered
    next to the run. `wire_results` optionally points at the results root holding the
    two-host wire campaign dirs (F26–F28); when omitted the run's ancestors are probed,
    since the wire dirs sit directly under ``SCG-SESHAT/results/``.
    """
    run_dir = resolve_run_dir(Path(results_path))

    summary_csv = _find_summary_csv(run_dir)
    summary = _read_table(summary_csv) if summary_csv else pd.DataFrame()
    if "scenario" not in summary.columns and not summary.empty:
        # A key/value summary.csv was picked up by mistake; ignore it.
        summary = pd.DataFrame()
    summary = _enrich_factors(summary)

    scenarios_roots = _find_scenarios_roots(run_dir)
    runs = _load_runs(scenarios_roots, summary)
    sysmetrics = _load_sysmetrics(scenarios_roots, summary)
    saturation = _load_saturation(scenarios_roots, summary)
    skipped = _load_skipped(run_dir)

    return RunBundle(
        run_dir=run_dir,
        summary=summary,
        runs=runs,
        sysmetrics=sysmetrics,
        saturation=saturation,
        skipped=skipped,
        meta=_find_meta(run_dir, "meta.csv"),
        sysinfo=_find_meta(run_dir, "sysinfo.csv"),
        coverage=_load_coverage(run_dir, Path(coverage_path) if coverage_path else None),
        wire=_load_wire_campaigns(run_dir, Path(wire_results) if wire_results else None),
    )


def _load_wire_campaigns(run_dir: Path, override: Optional[Path]):
    """Locate the two-host wire campaign dirs (F26–F28), or None.

    Explicit ``--wire-results`` wins; otherwise probe the run's ancestors, since
    `wire_bench.sh` writes its campaign dirs (``wire-run/``, ``ab-*/``, ``knee-*/``,
    …) directly into ``SCG-SESHAT/results/`` next to the suite runs. Imported
    lazily: :mod:`.wire` builds on this module's `_read_table`.
    """
    from . import wire as wire_mod

    candidates = [override] if override else [run_dir.parent, run_dir.parent.parent]
    for root in candidates:
        if root is None or not root.is_dir():
            continue
        bundle = wire_mod.load_wire(root)
        if bundle is not None:
            return bundle
    return None
