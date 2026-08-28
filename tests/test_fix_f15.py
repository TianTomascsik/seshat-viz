"""
Regression tests for F15 (concurrency scaling).

Guards the two P0 conclusion defects plus the stale placeholder note:

* F15-1 — the takeaway crowned the series with the SHALLOWEST sweep: `max` over per-series
  endpoint efficiencies mechanically hands the crown to a series whose high-concurrency
  scenarios were skipped (its endpoint sits at a lower connection count, where efficiency is
  higher). The callout must compare all series at the deepest connection count they ALL
  reach, and a truncated / single-gateway-fallback series must be daggered with its reason.
* F15-2 — the "host stays <30% busy … not core count" causal prose was hardcoded from an
  older run. It must be computed from host_busy_frac_p95 over the plotted rows, and the
  "not core count" clause must only survive when the measured peak stays low.
* F15-3 — the placeholder explanation hardcoded "TPROXY pinned to 1c" even when TPROXY
  sweeps connections in the same figure; the reason must be assembled per transport.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f15.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import concurrency_scaling  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402


def _bundle(summary: pd.DataFrame, skipped: pd.DataFrame | None = None) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=summary,
        runs=empty,
        sysmetrics=empty,
        saturation=empty,
        skipped=skipped if skipped is not None else empty,
        sysinfo={"hostname": "test"},
    )


def _row(transport: str, chain: str, conns: int, tput: float, *, busy: float,
         bottleneck: str, harness_limited: bool) -> dict:
    return {
        "scenario": f"matrix_none_{transport}_{transport}_64KB_{chain}_{conns}c",
        "family": "matrix",
        "transport": transport,
        "protocol": "none",
        "chain": chain,
        "message_bytes": 65536,
        "connections": conns,
        "throughput_gbps_mean": tput,
        "latency_p99_us_mean": 100.0 * conns,
        "bottleneck": bottleneck,
        "harness_limited": harness_limited,
        "host_busy_frac_p95": busy,
        "cpu_hot_thread_pct_p95": 90.0,
    }


def _scaling_summary(*, busy_by_conns: dict[int, float], with_udp: bool = False) -> pd.DataFrame:
    """
    The audit F15-1 shape, in miniature. Series A (TCP · routing, 2-gateway scg chain) sweeps
    1→64c: eff = 100/50/25/5 %. Series B (TPROXY · routing, single-gateway direct chain) is
    TRUNCATED at 16c with eff = 100/40/14 % — under the old max-over-endpoints rule B's 14%
    endpoint beat A's 5% endpoint, crowning the truncated, load-generator-bound series.
    """
    rows = []
    for conns, tput in ((1, 10.0), (4, 20.0), (16, 40.0), (64, 32.0)):
        rows.append(_row("tcp", "scg", conns, tput, busy=busy_by_conns[conns],
                         bottleneck="scg-cpu", harness_limited=False))
    for conns, tput in ((1, 10.0), (4, 16.0), (16, 22.4)):
        rows.append(_row("tproxy", "direct", conns, tput, busy=busy_by_conns[conns],
                         bottleneck="harness-io", harness_limited=True))
    if with_udp:
        rows.append(_row("udp", "scg", 1, 5.0, busy=busy_by_conns[1],
                         bottleneck="harness-io", harness_limited=True))
    return pd.DataFrame(rows)


def _skipped_tproxy_64c() -> pd.DataFrame:
    """The skip register entry that truncated series B: its 64c scenario failed to run."""
    return pd.DataFrame([{
        "scenario": "matrix_none_tproxy_tproxy_64KB_direct_64c",
        "reason": "TPROXY gateway did not forward connection to backend",
        "family": "matrix",
        "chain": "direct",
        "connections": 64,
    }])


_BUSY_HIGH = {1: 0.10, 4: 0.30, 16: 0.85, 64: 0.91}
_BUSY_LOW = {1: 0.10, 4: 0.12, 16: 0.15, 64: 0.18}


def _render(summary: pd.DataFrame, skipped: pd.DataFrame | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp), formats=("png",))
        concurrency_scaling.make(_bundle(summary, skipped), saver)
        entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F15 unexpectedly skipped: {entry.get('skipped')}"
    return entry


def _chrome(entry: dict, kind: str) -> str:
    return " ".join(c["text"] for c in entry.get("chrome", []) if c["kind"] == kind)


def test_takeaway_compares_at_common_depth_not_endpoint_max():
    """F15-1: the crown must go to the best series at the deepest COMMON connection count
    (A: 25% at 16c), not to the truncated series' shallow endpoint (B: 14% at 16c)."""
    entry = _render(_scaling_summary(busy_by_conns=_BUSY_HIGH), _skipped_tproxy_64c())
    take = _chrome(entry, "takeaway")
    assert "16 connections" in take, take
    assert "25% of ideal-linear" in take, take
    assert "routing · TCP" in take, take
    assert "gateway-bound" in take, take
    # The old rule's winner ("… 14% … (routing · TPROXY)") must be gone.
    assert "14%" not in take, take
    assert "routing · TPROXY" not in take, take
    # Series swept beyond the common depth are summarized at their true endpoint (A: 5% @64c).
    assert "5.0%" in take and "64 connections" in take, take


def test_truncated_fallback_series_is_disclosed_with_reason():
    """F15-1 disclosure: the lone direct-chain series whose 64c run was skipped must be
    flagged and explained, not passed off as a converged 2-gateway curve. (The old †
    glyph is a retired encoding — the disclosure is now plain prose in the method note.)"""
    entry = _render(_scaling_summary(busy_by_conns=_BUSY_HIGH), _skipped_tproxy_64c())
    method = _chrome(entry, "method")
    assert "routing · TPROXY" in method, method
    assert "†" not in method, method
    assert "single-gateway 'direct' chain" in method, method
    assert "ends at 16c" in method, method
    assert "1 higher-concurrency scenario(s) were skipped" in method, method


def test_no_skip_and_scg_chain_means_no_dagger():
    """A series that genuinely ends where its sweep ends (no skips, scg chain) is not flagged."""
    summary = _scaling_summary(busy_by_conns=_BUSY_HIGH)
    summary = summary[summary["transport"] == "tcp"]  # single, well-covered series
    entry = _render(summary)
    assert "†" not in _chrome(entry, "method")


def test_host_busy_is_measured_not_hardcoded():
    """F15-2: with the host ~91% busy at 64c, the '<30% busy … not core count' prose must be
    replaced by the measured profile and a saturation co-limit statement."""
    entry = _render(_scaling_summary(busy_by_conns=_BUSY_HIGH), _skipped_tproxy_64c())
    method, take = _chrome(entry, "method"), _chrome(entry, "takeaway")
    for text in (method, take):
        assert "<30% busy" not in text, text
        assert "not core count" not in text, text
    assert "10% at 1c" in method and "91% at 64c" in method, method
    assert "91%" in take and "co-limit" in take, take


def test_idle_host_keeps_not_core_count_claim():
    """F15-2 counterpart: when the measured peak stays low, the serial-data-plane / 'not core
    count' reading is defensible and must be stated with the measured peak."""
    entry = _render(_scaling_summary(busy_by_conns=_BUSY_LOW), _skipped_tproxy_64c())
    method, take = _chrome(entry, "method"), _chrome(entry, "takeaway")
    assert "not core count" in method, method
    assert "not core count" in take, take
    assert "18%" in take, take  # the measured peak, not a hardcoded 30%


def test_single_only_reason_is_per_transport():
    """F15-3: a UDP-only placeholder must cite the UDP/DTLS reason and must NOT claim
    'TPROXY pinned to 1c' while the TPROXY panel sweeps connections."""
    entry = _render(_scaling_summary(busy_by_conns=_BUSY_HIGH, with_udp=True),
                    _skipped_tproxy_64c())
    method = _chrome(entry, "method")
    assert "UDP — the 'dtls_multi_connection' limitation" in method, method
    assert "TPROXY pinned to 1c" not in method, method
    assert "were run" not in method, method  # old subject-verb artifact


def test_is_loadgen_accepts_numpy_bool():
    """derive aggregates harness_limited with .any() (numpy bool); an identity check against
    Python True silently dropped the flag."""
    assert concurrency_scaling._is_loadgen("scg-cpu", np.True_) is True
    assert concurrency_scaling._is_loadgen("scg-cpu", np.False_) is False
    assert concurrency_scaling._is_loadgen("harness-io", pd.NA) is True
    assert concurrency_scaling._is_loadgen("scg", None) is False


def test_series_stem_strips_size_chain_conns():
    stem = concurrency_scaling._series_stem
    assert stem("matrix_routing_tproxy_tproxy_64B_direct_64c") == "matrix_routing_tproxy_tproxy"
    assert stem("matrix_integrity_tls13_shm_shm_16KB_scg_16c") == "matrix_integrity_tls13_shm_shm"
    assert stem("matrix_none_tcp_tcp_65536B_scg_1024c") == "matrix_none_tcp_tcp"


if __name__ == "__main__":
    test_takeaway_compares_at_common_depth_not_endpoint_max()
    test_truncated_fallback_series_is_disclosed_with_reason()
    test_no_skip_and_scg_chain_means_no_dagger()
    test_host_busy_is_measured_not_hardcoded()
    test_idle_host_keeps_not_core_count_claim()
    test_single_only_reason_is_per_transport()
    test_is_loadgen_accepts_numpy_bool()
    test_series_stem_strips_size_chain_conns()
    print("ok")
