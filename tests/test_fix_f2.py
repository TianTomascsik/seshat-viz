"""
Regression tests for F2 (payload-size scaling) — the honest closed-loop RTT row.

Guards the audited contaminations of the bottom row (audit F2-1/F2-2/F2-3/F2-4):
the RTT pool once took ANY row carrying `rtt_us_p99`, so rate-capped paced iface_* rows,
QoS-profile-tuned ping-pongs and zero-copy shmzc variants blended into the gateway means,
and 0-gateway loopback probes (named `*_direct`/`*_loopback`) rendered under the
'1 gateway (scg-direct)' legend — presenting a no-gateway baseline as the fastest gateway
path. The quoted RTT band and the scope sentences were hardcoded ('~11–60 µs',
'TCP/UDP at ~1 KB today') and drifted from the data; they must be computed at render time.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f2.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import payload_scaling  # noqa: E402
from seshat_viz.loader import RunBundle, _enrich_factors  # noqa: E402


def _bundle(summary: pd.DataFrame) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=_enrich_factors(summary),
        runs=empty,
        sysmetrics=empty,
        saturation=empty,
        skipped=pd.DataFrame(columns=["scenario", "reason"]),
        sysinfo={"hostname": "test"},
    )


def _row(scenario, transport, protocol, size, *, tput=np.nan, blast=np.nan, rtt=np.nan):
    return {
        "scenario": scenario, "transport": transport, "protocol": protocol,
        "message_bytes": size, "connections": 1,
        "throughput_gbps_mean": tput, "latency_p99_us_mean": blast, "rtt_us_p99": rtt,
    }


def _mixed_summary() -> pd.DataFrame:
    """The real-run shape in miniature: a matrix blast sweep, the matrix_lat_* closed-loop
    grid, plus every contaminant class the audit found leaking into the RTT row."""
    rows = []
    # Blast rows (top/middle rows' pool) — both chains on tcp, direct-only on udp.
    for size in (64, 1024):
        rows.append(_row(f"matrix_routing_tcp_tcp_{size}B_direct_1c", "tcp", "none", size,
                         tput=10.0, blast=5000.0))
        rows.append(_row(f"matrix_routing_tcp_tcp_{size}B_scg_1c", "tcp", "none", size,
                         tput=9.0, blast=6000.0))
        rows.append(_row(f"matrix_routing_udp_udp_{size}B_direct_1c", "udp", "none", size,
                         tput=8.0, blast=4000.0))
    # Honest 1-gateway closed-loop grid (the ONLY legitimate gateway RTT source here).
    for size, rtt in ((64, 44.0), (1024, 46.0), (65536, 300.0)):
        rows.append(_row(f"matrix_lat_routing_tcp_tcp_{size}B_direct_1c", "tcp", "none",
                         size, rtt=rtt))
    # Contaminants (audit F2-1): paced open-loop iface rows carrying rtt percentiles,
    # a QoS-tuned profile ping-pong, and a zero-copy shmzc rtt variant.
    rows.append(_row("iface_tcp_loopback_latency_1KB", "tcp", "none", 1024, rtt=12.0))
    rows.append(_row("iface_tcp_scg_latency_1KB", "tcp", "none", 1024, rtt=48.0))
    rows.append(_row("profile_routing_latency_pingpong_1KB", "tcp", "none", 1024, rtt=31.0))
    rows.append(_row("shmzc_tcp_rtt_1KB", "tcp", "none", 1024, rtt=44.6))
    # Loopback ping-pong probes (audit F2-2): 0 gateways, `direct`-chain names — UDP has
    # ONLY these (no gateway ping-pong data at all).
    rows.append(_row("pp_tcp_loopback_1KB", "tcp", "none", 1024, rtt=12.6))
    rows.append(_row("pp_udp_loopback_64B", "udp", "none", 64, rtt=10.8))
    rows.append(_row("pp_udp_loopback_1KB", "udp", "none", 1024, rtt=11.4))
    return pd.DataFrame(rows)


def _render(summary: pd.DataFrame):
    """Render F2 chrome-off; return the manifest entry (with recorded chrome texts)."""
    prev = theme.chrome_enabled()
    theme.set_chrome(False)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            saver = theme.Saver(Path(tmp), formats=("png",))
            payload_scaling.make(_bundle(summary), saver)
            return saver.manifest[-1]
    finally:
        theme.set_chrome(prev)


def _chrome(entry, kind):
    return " | ".join(r["text"] for r in entry.get("chrome", []) if r["kind"] == kind)


# ---------------------------------------------------------------------------- F2-1 ----

def test_rtt_gateway_pool_excludes_paced_profile_and_shmzc_rows():
    """The gateway RTT pool is matrix_lat_* only when that grid exists: paced iface_*
    (mode!='pingpong'), QoS-profile-tuned and shmzc-variant ping-pongs must not blend in."""
    df = _enrich_factors(_mixed_summary())
    gw, lo = payload_scaling._rtt_pools(df)
    assert gw is not None
    names = set(gw["scenario"])
    assert names == {f"matrix_lat_routing_tcp_tcp_{s}B_direct_1c" for s in (64, 1024, 65536)}
    # The blended 1 KB mean was the audit's smoking gun (46 µs rendered as ~21 µs): with a
    # clean pool the per-size mean equals the matrix_lat value exactly.
    assert float(gw.loc[gw["message_bytes"] == 1024, "rtt_us_p99"].mean()) == 46.0


def test_rtt_pool_falls_back_to_gateway_pingpong_without_matrix_lat():
    """Without the matrix_lat_* grid, gateway ping-pong rows (e.g. profile_*_pingpong)
    still feed the row — the scoping prefers the canonical grid, it does not require it."""
    df = _mixed_summary()
    df = df[~df["scenario"].str.startswith("matrix_lat_")]
    gw, _lo = payload_scaling._rtt_pools(_enrich_factors(df))
    assert gw is not None
    assert set(gw["scenario"]) == {"profile_routing_latency_pingpong_1KB", "shmzc_tcp_rtt_1KB"}
    # Paced rows stay out even in the fallback.
    assert not any(gw["scenario"].str.startswith("iface_"))


# ---------------------------------------------------------------------------- F2-2 ----

def test_loopback_rows_split_out_of_the_gateway_pool():
    """0-gateway ping-pong probes are `direct`-chain-named; they must land in the loopback
    pool, never in the gateway pool that wears the '1 gateway (scg-direct)' style."""
    df = _enrich_factors(_mixed_summary())
    gw, lo = payload_scaling._rtt_pools(df)
    assert lo is not None
    assert set(lo["scenario"]) == {"pp_tcp_loopback_1KB", "pp_udp_loopback_64B",
                                   "pp_udp_loopback_1KB"}
    assert not set(lo["scenario"]) & set(gw["scenario"])


def test_loopback_disclosed_in_note_and_udp_never_claims_a_gateway():
    """Rendering the mixed run must disclose the dotted loopback reference in the method
    note — the UDP panel's only ping-pong data has no gateway hop."""
    entry = _render(_mixed_summary())
    assert "skipped" not in entry, entry
    methods = _chrome(entry, "method")
    assert "no-gateway loopback reference" in methods, methods
    # Coverage names only transports with GATEWAY ping-pong data (tcp), not udp.
    assert "ping-pong data exists (TCP at" in methods, methods


# ---------------------------------------------------------------------------- F2-3 ----

def test_takeaway_band_is_computed_not_hardcoded():
    """The quoted RTT band must come from the plotted rows (44–300 µs here), never the
    stale '~11–60 µs' constant; the blast clause quotes the computed maximum."""
    entry = _render(_mixed_summary())
    takeaway = _chrome(entry, "takeaway")
    methods = _chrome(entry, "method")
    assert "11–60" not in takeaway and "11–60" not in methods
    # min plotted RTT = 10.8 (udp loopback), max = 300 (matrix_lat 64K row).
    assert "~11–300 µs" in takeaway, takeaway
    assert "~11–300 µs" in methods, methods
    assert "up to 6 ms" in takeaway, takeaway  # blast max 6000 µs, computed


def test_blast_only_takeaway_quotes_no_uncomputable_rtt():
    """With no ping-pong data at all, the takeaway may not quote an RTT number — it can
    only point at F16."""
    df = _mixed_summary()
    df = df[df["rtt_us_p99"].isna()]
    entry = _render(df)
    assert "skipped" not in entry, entry
    takeaway = _chrome(entry, "takeaway")
    assert "F16" in takeaway, takeaway
    assert "µs (" not in takeaway and "~11" not in takeaway, takeaway


# ---------------------------------------------------------------------------- F2-4 ----

def test_scope_note_derives_two_gateway_coverage():
    """The 2-gateway clause reflects the data: named when only a subset of transports has
    scg-scg rows, absent when every plotted transport has them."""
    # tcp has scg rows, udp does not → clause must name TCP only.
    entry = _render(_mixed_summary())
    methods = _chrome(entry, "method")
    assert "2-gateway path exists only for TCP" in methods, methods
    # Give udp an scg row too → the restriction clause must disappear.
    df = _mixed_summary()
    extra = pd.DataFrame([_row("matrix_routing_udp_udp_64B_scg_1c", "udp", "none", 64,
                               tput=7.0, blast=4500.0)])
    entry = _render(pd.concat([df, extra], ignore_index=True))
    methods = _chrome(entry, "method")
    assert "exists only for" not in methods, methods


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")
