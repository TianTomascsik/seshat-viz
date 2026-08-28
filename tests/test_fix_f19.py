"""
Regression tests for F19 (jitter & determinism).

Guards the two audited comparison defects:

  * F19-1 — the matched cell was voted on the full measurement pool, not the jitter-BEARING
    rows: it pinned a payload size where whole transports carry NaN jitter for every
    encrypted row, so the "crypto adds ~0.3 µs" story silently came from one transport at
    the jitter-pessimal size. The vote must run after the jitter dropna.
  * F19-2 — topology was unpinned: a 0-gateway loopback baseline (`baseline_udp_loopback_*`)
    stood in for the gateway's UDP-routing bar, and 1-gw/2-gw rows blended inside one bar.
    The pool must exclude 0-gateway rows and the gateway count must join the matched cell.

Plus the F19-3 disclosures: no hardcoded "datagram points are DTLS at datagram sizes"
clause, a computed jitter-coverage note, and a computed harness-limited note.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f19.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import jitter  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402


def _bundle(summary: pd.DataFrame) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=summary,
        runs=empty,
        sysmetrics=empty,
        saturation=empty,
        skipped=empty,
        sysinfo={"hostname": "test"},
    )


def _row(scenario, transport, protocol, size, *, n_gw=1, jit=np.nan, tput=10.0, hl=False):
    return {
        "scenario": scenario,
        "family": "matrix",
        "transport": transport,
        "protocol": protocol,
        "message_bytes": size,
        "connections": 1,
        "chain": "scg",
        "n_gateways": n_gw,
        "jitter_us_mean": jit,
        "throughput_gbps_mean": tput,
        "harness_limited": hl,
    }


def _audit_shape_summary() -> pd.DataFrame:
    """The nightly's shape in miniature: at 256B only TPROXY carries encrypted jitter (and
    a 0-gw UDP loopback baseline is the only jitter-bearing 'UDP · routing' row), while at
    16KB tcp/tproxy/unix routing AND tcp/tproxy crypto all carry jitter. The pre-fix vote
    picked 256B (broadest measured coverage); the honest vote picks 16KB (broadest jitter
    coverage) and drops the baseline."""
    rows = [
        # 256B: everything measured, jitter only on tproxy/tls + the 0-gw udp baseline
        _row("matrix_routing_tcp_256B_scg_1c", "tcp", "none", 256, tput=40.0, hl=True),
        _row("matrix_tls13_tcp_256B_scg_1c", "tcp", "tls/1.3", 256, tput=8.0),
        _row("matrix_routing_tproxy_256B_scg_1c", "tproxy", "none", 256, tput=41.0, hl=True),
        _row("matrix_tls13_tproxy_256B_scg_1c", "tproxy", "tls/1.3", 256, jit=0.4, tput=8.0),
        _row("matrix_routing_uds_256B_scg_1c", "unix", "none", 256, tput=30.0, hl=True),
        _row("matrix_tls13_uds_256B_scg_1c", "unix", "tls/1.3", 256, tput=7.0),
        _row("matrix_routing_shm_256B_scg_1c", "shm", "none", 256, tput=15.0, hl=True),
        _row("baseline_udp_loopback_256B", "udp", "none", 256, n_gw=0, jit=5.7, tput=24.0),
        # 16KB: fewer combos measured, but jitter everywhere except encrypted UDS
        _row("matrix_routing_tcp_16KB_scg_1c", "tcp", "none", 16384, jit=4.0, tput=46.0, hl=True),
        _row("matrix_tls13_tcp_16KB_scg_1c", "tcp", "tls/1.3", 16384, jit=26.0, tput=9.3),
        _row("matrix_routing_tproxy_16KB_scg_1c", "tproxy", "none", 16384, jit=4.5, tput=43.0, hl=True),
        _row("matrix_tls13_tproxy_16KB_scg_1c", "tproxy", "tls/1.3", 16384, jit=25.0, tput=9.6),
        _row("matrix_routing_uds_16KB_scg_1c", "unix", "none", 16384, jit=6.7, tput=37.0, hl=True),
        _row("matrix_tls13_uds_16KB_scg_1c", "unix", "tls/1.3", 16384, tput=9.0),
    ]
    return pd.DataFrame(rows)


def _chrome(saver: theme.Saver, kind: str) -> str:
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F19 unexpectedly skipped: {entry.get('skipped')}"
    return " ".join(c["text"] for c in entry.get("chrome", []) if c["kind"] == kind)


def test_pool_excludes_zero_gateway_rows():
    """F19-2: a 0-gw loopback baseline must never enter the per-configuration pool —
    with `n_gateways` present, and via the naming fallback when it is absent."""
    s = _audit_shape_summary()
    pool = jitter._gateway_blast_pool(s)
    assert not (pool["n_gateways"] == 0).any()
    assert "baseline_udp_loopback_256B" not in set(pool["scenario"])
    # naming fallback: same exclusion without the loader's n_gateways column
    pool2 = jitter._gateway_blast_pool(s.drop(columns=["n_gateways"]))
    assert "baseline_udp_loopback_256B" not in set(pool2["scenario"])


def test_size_vote_runs_on_jitter_bearing_rows():
    """F19-1: measured coverage favors 256B (7 gateway combos vs 6) but jitter coverage
    favors 16KB (5 combos vs 1) — the cell must pin 16KB, and the plotted set must span
    crypto on more than one transport (the 256B cell's crypto story was TPROXY-only)."""
    df, chosen, _cell = jitter._matched_jitter(_audit_shape_summary())
    assert chosen["message_bytes"] == 16384
    assert chosen["connections"] == 1
    enc_transports = set(df[df["protocol"] != "none"]["transport"].astype(str))
    assert enc_transports == {"tcp", "tproxy"}
    assert "udp" not in set(df["transport"].astype(str))  # the baseline stayed out


def test_gateway_count_is_pinned():
    """F19-2: identical combos at 1 and 2 gateways must not blend — the cell pins the
    (tie-broken minimal) gateway count."""
    rows = []
    for n_gw, tag in ((1, "direct"), (2, "scg")):
        for proto in ("none", "tls/1.3"):
            rows.append(_row(f"matrix_{proto}_tproxy_16KB_{tag}_1c", "tproxy", proto,
                             16384, n_gw=n_gw, jit=5.0 + n_gw, tput=10.0))
    df, chosen, _cell = jitter._matched_jitter(pd.DataFrame(rows))
    assert chosen["n_gateways"] == 1
    assert set(df["n_gateways"]) == {1}


def test_render_notes_and_takeaway_are_computed():
    """End-to-end: the figure renders, the false DTLS-sizes clause is gone, coverage and
    harness-limited notes carry computed counts, and the takeaway states the computed
    encryption multiplier (6.5x here) instead of the hardcoded 'adds little'."""
    bundle = _bundle(_audit_shape_summary())
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        jitter.make(bundle, saver)
    method = _chrome(saver, "method")
    assert "datagram points are DTLS at datagram sizes" not in method
    assert "1 gw" in method  # topology stamped on the matched cell
    # jitter coverage: 5 jitter-bearing of 6 measured combos; the gap is encrypted UDS
    assert "5/6 configs measured in this cell carry it" in method
    assert "encrypted UDS" in method
    assert "gaps are missing data, not zeros" in method
    # harness-limited: the three routing rows out of five plotted configs
    assert "3/5 plotted configs are harness-limited" in method
    take = _chrome(saver, "takeaway")
    assert "adds little" not in take
    assert "multiplies PDV ≈ 6.5×" in take
    assert "worst encrypted (TCP · TLS 1.3) ≈ 26.0 µs" in take


def test_takeaway_keeps_adds_little_when_true():
    """The 'adds little' phrasing must survive where it is computed to be true
    (best-encrypted / routing ratio under the threshold)."""
    rows = [
        _row("matrix_routing_tcp_16KB_scg_1c", "tcp", "none", 16384, jit=4.0, tput=40.0),
        _row("matrix_tls13_tcp_16KB_scg_1c", "tcp", "tls/1.3", 16384, jit=4.4, tput=10.0),
    ]
    bundle = _bundle(pd.DataFrame(rows))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        jitter.make(bundle, saver)
    take = _chrome(saver, "takeaway")
    assert "adds little" in take
    assert "multiplies" not in take


if __name__ == "__main__":
    test_pool_excludes_zero_gateway_rows()
    test_size_vote_runs_on_jitter_bearing_rows()
    test_gateway_count_is_pinned()
    test_render_notes_and_takeaway_are_computed()
    test_takeaway_keeps_adds_little_when_true()
    print("ok")
