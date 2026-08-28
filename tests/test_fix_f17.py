"""
Regression tests for F17 (connection setup) — audit findings F17-1 and F17-2.

F17-1: the handshake-auth sweep (handshake_*) varies cert_key_type / kex_group, but those
factors never reach summary.csv and conn_threads is name-regex-derived (NaN for handshake_*
names), so all sweep rows collapsed to identical bare "TLS 1.3" labels — RSA's genuine
handshake cost read as unexplained variance within one config. The module must recover the
varied factor from the scenario name and the thread count from `connections` (the connrate
engine spawns one connector thread per configured connection), and the 1-thread takeaway
ratio must stay pinned to the default TLS 1.3 config (no auth-sweep contamination).

F17-2: resumed_fraction=0 is a harness measurement limit (the plaintext connrate probe never
presents a session ticket), not an SCG defect — the takeaway must not blame the gateway.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f17.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import connection_setup  # noqa: E402
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


def _connrate_summary() -> pd.DataFrame:
    """A minimal connrate slice: bare-accept + plain TLS 1.3 baselines with the name-derived
    conn_threads populated, plus handshake-auth sweep rows where conn_threads is NaN (their
    names lack the `_<n>thread` token) and only `connections` carries the thread count."""
    def row(scenario, protocol, threads, connections, cps, p50, p99,
            variant=np.nan, resumed=np.nan):
        return {
            "scenario": scenario,
            "protocol": protocol,
            "transport": "tcp",
            "chain": "scg",
            "variant": variant,
            "conn_threads": threads,
            "connections": connections,
            "conns_per_sec": cps,
            "conn_handshake_p50_us": p50,
            "conn_handshake_p99_us": p99,
            "resumed_fraction": resumed,
        }

    return pd.DataFrame([
        row("conn_tcp_loopback_1thread", "none", 1.0, 1, 4000.0, 20.0, 30.0),
        row("conn_scg_tls13_1thread", "tls/1.3", 1.0, 1, 1000.0, 1000.0, 1100.0,
            resumed=0.0),
        # Auth sweep: RSA cert, 5x slower than the default config — must not blend into
        # either the plain "TLS 1.3" label or the 1-thread takeaway ratio.
        row("handshake_tls13_rsa_1c", "tls/1.3", np.nan, 1, 700.0, 5000.0, 6000.0,
            resumed=0.0),
        row("handshake_tls13_rsa", "tls/1.3", np.nan, 4, 2800.0, 4800.0, 5900.0,
            resumed=0.0),
        row("handshake_kex_p256_1c", "tls/1.3", np.nan, 1, 990.0, 1020.0, 1120.0,
            resumed=0.0),
        # Resume-intent row whose telemetry shows resumption never engaged (harness probe).
        row("conn_scg_tls13_resumed_1thread", "tls/1.3+resume", 1.0, 1, 1100.0, 900.0,
            1000.0, variant="resumed", resumed=0.0),
    ])


def _chrome(saver: theme.Saver, kind: str) -> str:
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F17 unexpectedly skipped: {entry.get('skipped')}"
    texts = [c["text"] for c in entry.get("chrome", []) if c["kind"] == kind]
    assert texts, f"no '{kind}' chrome recorded"
    return " ".join(texts)


def _render(summary: pd.DataFrame) -> theme.Saver:
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        connection_setup.make(_bundle(summary), saver)
    return saver


def test_hs_variant_from_scenario_name():
    """The auth-sweep factor lives only in the scenario name — the mapping must be exact."""
    assert connection_setup._hs_variant("handshake_tls13_rsa") == "RSA cert"
    assert connection_setup._hs_variant("handshake_tls13_rsa_1c") == "RSA cert"
    assert connection_setup._hs_variant("handshake_tls13_ecdsa_1c") == "ECDSA cert"
    assert connection_setup._hs_variant("handshake_kex_x25519") == "X25519 kex"
    assert connection_setup._hs_variant("handshake_kex_p256_1c") == "P-256 kex"
    assert pd.isna(connection_setup._hs_variant("conn_scg_tls13_1thread"))
    assert pd.isna(connection_setup._hs_variant("matrix_tls13_tcp_64B"))


def test_auth_sweep_rows_render_distinct_labels():
    """F17-1: sweep rows must not collapse to bare 'TLS 1.3' — the label carries the varied
    factor and the thread count recovered from `connections`."""
    label = connection_setup._label
    rsa = pd.Series({"scenario": "handshake_tls13_rsa", "protocol": "tls/1.3",
                     "conn_threads": 4.0, "variant": np.nan})
    p256 = pd.Series({"scenario": "handshake_kex_p256_1c", "protocol": "tls/1.3",
                      "conn_threads": 1.0, "variant": np.nan})
    plain = pd.Series({"scenario": "conn_scg_tls13_1thread", "protocol": "tls/1.3",
                       "conn_threads": 1.0, "variant": np.nan})
    assert label(rsa) == "TLS 1.3 (RSA cert) · 4-thread"
    assert label(p256) == "TLS 1.3 (P-256 kex) · 1-thread"
    assert label(plain) == "TLS 1.3 · 1-thread"


def test_method_note_covers_sweep_variants_and_threads():
    """End-to-end through make(): the method note's protocol list names the sweep variants and
    the thread set includes the handshake rows' counts (recovered from `connections`)."""
    saver = _render(_connrate_summary())
    method = _chrome(saver, "method")
    assert "TLS 1.3 (RSA cert)" in method
    assert "TLS 1.3 (P-256 kex)" in method
    # handshake_tls13_rsa ran at 4 threads; with conn_threads NaN pre-fix the set was {1}.
    assert "{1, 4}" in method


def test_takeaway_ratio_excludes_auth_sweep_rows():
    """F17-1: the 1-thread TLS-vs-bare ratio must stay pinned to the default config — pooling
    the RSA/P-256 sweep rows (p50 5000/1020 vs plain 1000) would inflate ~50× to ~117×."""
    saver = _render(_connrate_summary())
    take = _chrome(saver, "takeaway")
    assert "~50×" in take, f"takeaway ratio contaminated: {take}"


def test_takeaway_discloses_dead_resumption_root_cause():
    """F17-2 (superseded wording): resumed_fraction=0 rows are omitted and the omission is
    disclosed with its post-campaign root cause — the gateway's encrypt connector never
    presented its cached upstream ticket, a defect since fixed and regression-tested. (The
    earlier 'probe limitation' explanation was wrong and must not come back.)"""
    saver = _render(_connrate_summary())
    take = _chrome(saver, "takeaway")
    assert "resumed_fraction=0" in take
    assert "encrypt connector" in take
    assert "since fixed" in take
    assert "harness limit" not in take
    method = _chrome(saver, "method")
    assert "encrypt connector" in method
    assert "since fixed" in method


def test_conn_threads_absent_column_still_renders():
    """If the summary carries no conn_threads column at all (connsetup_table drops it), the
    module must synthesize it from `connections` and still render."""
    summary = _connrate_summary().drop(columns=["conn_threads"])
    saver = _render(summary)
    method = _chrome(saver, "method")
    assert "{1, 4}" in method


if __name__ == "__main__":
    test_hs_variant_from_scenario_name()
    test_auth_sweep_rows_render_distinct_labels()
    test_method_note_covers_sweep_variants_and_threads()
    test_takeaway_ratio_excludes_auth_sweep_rows()
    test_takeaway_blames_probe_not_gateway_for_dead_resumption()
    test_conn_threads_absent_column_still_renders()
    print("ok")
