"""
Unit tests for the confound-control helpers added for the 2026-07 figure overhaul.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_derive.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import derive, loader, theme  # noqa: E402


def _frame() -> pd.DataFrame:
    # TCP measured at 1c and 1024c; SHM/UDS only at 1c — the real asymmetry.
    rows = [
        ("tcp", "none", 1, 4096, 5.0),
        ("tcp", "none", 1024, 4096, 80.0),
        ("tcp", "tls/1.3", 1, 4096, 3.0),
        ("tcp", "tls/1.3", 1024, 4096, 35.0),
        ("shm", "none", 1, 4096, 16.0),
        ("unix", "none", 1, 4096, 4.0),
    ]
    return pd.DataFrame(rows, columns=["transport", "protocol", "connections",
                                       "message_bytes", "throughput_gbps_mean"])


def test_matched_cell_pins_single_connection():
    df = _frame()
    out, chosen = derive.matched_cell(df, "transport")
    # connections tie-breaks to the smallest shared value (1c), never blends 1024c.
    assert chosen["connections"] == 1
    assert chosen["message_bytes"] == 4096
    assert set(out["connections"].unique()) == {1}
    # the 80 Gbps TCP-1024c row must be gone, so SHM (16) now leads throughput.
    assert out["throughput_gbps_mean"].max() == 16.0
    assert set(out["transport"]) == {"tcp", "shm", "unix"}


def test_matched_cell_respects_fixed():
    df = _frame()
    out, chosen = derive.matched_cell(df, "transport", fixed={"connections": 1024})
    assert chosen["connections"] == 1024
    assert set(out["connections"].unique()) == {1024}


def test_fmt_cell_formats_units():
    assert theme.fmt_cell({"connections": 1, "message_bytes": 4096}) == "1c · 4KB"


def test_loader_factors_datapath_and_cipher():
    df = pd.DataFrame({"scenario": [
        "pp_tcp_loopback",
        "matrix_routing_tcp_tcp_4096B_scg_1c",
        "matrix_routing_tcp_tcp_4096B_direct_1c",
        "cipher_tls13_tls_aes_128_gcm_sha256_1024B",
        "cipher_tls13_tls_aes_128_gcm_sha256",
    ]})
    out = loader._enrich_factors(df)
    dp = dict(zip(out["scenario"], out["datapath"]))
    ng = dict(zip(out["scenario"], out["n_gateways"]))
    assert dp["pp_tcp_loopback"] == "loopback" and ng["pp_tcp_loopback"] == 0
    assert ng["matrix_routing_tcp_tcp_4096B_scg_1c"] == 2      # scg-scg
    assert ng["matrix_routing_tcp_tcp_4096B_direct_1c"] == 1   # scg-direct
    # the cipher factor strips the trailing size token so both sizes share one suite label.
    ciph = dict(zip(out["scenario"], out["cipher"].astype(str)))
    assert ciph["cipher_tls13_tls_aes_128_gcm_sha256_1024B"] == "aes_128_gcm_sha256"
    assert ciph["cipher_tls13_tls_aes_128_gcm_sha256"] == "aes_128_gcm_sha256"


def test_loader_topology_and_chain_are_family_aware():
    df = pd.DataFrame({"scenario": [
        "iface_tcp_scg_throughput_1KB_1c",      # single-gateway despite _scg
        "conn_scg_tls13_1thread",               # single-gateway despite _scg
        "profile_direct_pingpong_1KB",          # zero-gateway loopback (no 'loopback' token)
        "hotreload_routing_tcp_add_connection_saturation_1c",  # unmarked topology
    ]})
    out = loader._enrich_factors(df)
    ng = dict(zip(out["scenario"], out["n_gateways"]))
    chain = dict(zip(out["scenario"], out["chain"].astype(str)))
    dp = dict(zip(out["scenario"], out["datapath"]))
    assert ng["iface_tcp_scg_throughput_1KB_1c"] == 1     # NOT 2 — only matrix _scg is scg-scg
    assert ng["conn_scg_tls13_1thread"] == 1
    assert ng["profile_direct_pingpong_1KB"] == 0 and dp["profile_direct_pingpong_1KB"] == "loopback"
    # unmarked names get chain 'n/a', not a guessed 'scg' that fakes the 2-gateway topology.
    assert chain["hotreload_routing_tcp_add_connection_saturation_1c"] == "n/a"


def test_loader_size_re_parses_kb_tokens():
    df = pd.DataFrame({"scenario": [
        "matrix_routing_uds_unix_1KB_direct_64c",
        "matrix_ktls_shm_shm_64KB_direct_64c",
        "matrix_tls13_tcp_tcp_4096B_scg_1c",
    ]})
    out = loader._enrich_factors(df)
    mb = dict(zip(out["scenario"], out["message_bytes"]))
    assert mb["matrix_routing_uds_unix_1KB_direct_64c"] == 1024
    assert mb["matrix_ktls_shm_shm_64KB_direct_64c"] == 65536
    assert mb["matrix_tls13_tcp_tcp_4096B_scg_1c"] == 4096


def test_throughput_scenarios_only_excludes_nonblast():
    df = loader._enrich_factors(pd.DataFrame({"scenario": [
        "matrix_routing_tcp_tcp_1KB_scg_1c",              # keep: sustained blast
        "profile_routing_latency_throughput_1KB",         # keep: throughput workload, latency-TUNED
        "pp_tcp_loopback_1KB",                            # drop: closed-loop ping-pong
        "conn_scg_tls13_1thread",                        # drop: connection-rate
        "hotreload_routing_tcp_add_connection_saturation_1c",  # drop: reload-disrupted
        "lat_scg_routing_tcp_1KB",                       # drop: paced
        "iface_tcp_scg_latency_64B_1c",                  # drop: latency workload sub-mode
        "iface_tcp_scg_throughput_64B_1c",               # keep: throughput workload
    ]}))
    kept = set(derive.throughput_scenarios_only(df)["scenario"])
    assert "matrix_routing_tcp_tcp_1KB_scg_1c" in kept
    assert "profile_routing_latency_throughput_1KB" in kept  # tuning token must NOT exclude it
    assert "iface_tcp_scg_throughput_64B_1c" in kept
    for dropped in ("pp_tcp_loopback_1KB", "conn_scg_tls13_1thread",
                    "hotreload_routing_tcp_add_connection_saturation_1c",
                    "lat_scg_routing_tcp_1KB", "iface_tcp_scg_latency_64B_1c"):
        assert dropped not in kept, dropped


def test_gateway_insertion_cost_matrix_only_and_conn_matched():
    # Matrix pair at 1c and 4c (should pair per connection), plus a paced lat_ row that must
    # NOT leak into the scg mean, and an iface loopback row that must be excluded (non-matrix).
    rows = [
        ("matrix_r_tcp_tcp_1KB_direct_1c", "matrix", "tcp", "none", 1024, 1, "direct", 10.0),
        ("matrix_r_tcp_tcp_1KB_direct_4c", "matrix", "tcp", "none", 1024, 4, "direct", 40.0),
        ("matrix_r_tcp_tcp_1KB_scg_1c",    "matrix", "tcp", "none", 1024, 1, "scg", 10.1),
        ("matrix_r_tcp_tcp_1KB_scg_4c",    "matrix", "tcp", "none", 1024, 4, "scg", 39.0),
        ("lat_scg_routing_tcp_1KB",        "lat",    "tcp", "none", 1024, 1, "scg", 0.016),
        ("iface_tcp_loopback_1KB_1c",      "iface",  "tcp", "none", 1024, 1, "direct", 44.0),
    ]
    df = pd.DataFrame(rows, columns=["scenario", "family", "transport", "protocol",
                                     "message_bytes", "connections", "chain", "throughput_gbps_mean"])
    gc = derive.gateway_insertion_cost(df, metric="throughput_gbps_mean")
    assert len(gc) == 1
    r = gc.iloc[0]
    assert r["n_pairs"] == 2                       # paired at 1c and 4c, not the paced/iface rows
    # median of {10.1/10.0, 39.0/40.0} = median{1.01, 0.975} — near 1.0, not the fake -65%.
    assert 0.97 <= r["ratio"] <= 1.02


def test_scaling_table_propagates_bottleneck_and_harness_limited():
    # SHM/UDS routing swept at 1/4/16c, mirroring the real run: 1c is load-generator bound
    # (harness-io / harness_limited), 4c+ flip to the serial gateway relay (scg / scg-cpu).
    # scaling_table must carry these per-connection so F15 can encode why a curve is flat.
    rows = [
        ("matrix_r_shm_shm_4KB_direct_1c",  "shm",  "none", 1,  4096, 31.1, "harness-io", True),
        ("matrix_r_shm_shm_4KB_direct_4c",  "shm",  "none", 4,  4096, 33.2, "scg-cpu",   False),
        ("matrix_r_shm_shm_4KB_direct_16c", "shm",  "none", 16, 4096, 36.7, "scg",       False),
        ("matrix_r_uds_uds_4KB_direct_1c",  "unix", "none", 1,  4096, 38.4, "harness-io", True),
        ("matrix_r_uds_uds_4KB_direct_4c",  "unix", "none", 4,  4096, 37.6, "scg",       False),
        ("matrix_r_uds_uds_4KB_direct_16c", "unix", "none", 16, 4096, 37.5, "scg",       False),
    ]
    df = pd.DataFrame(rows, columns=["scenario", "transport", "protocol", "connections",
                                     "message_bytes", "throughput_gbps_mean", "bottleneck",
                                     "harness_limited"])
    tbl = derive.scaling_table(df)
    assert not tbl.empty
    assert "bottleneck" in tbl.columns and "harness_limited" in tbl.columns
    shm1 = tbl[(tbl["transport"] == "shm") & (tbl["connections"] == 1)].iloc[0]
    assert shm1["bottleneck"] == "harness-io"
    assert bool(shm1["harness_limited"]) is True
    shm16 = tbl[(tbl["transport"] == "shm") & (tbl["connections"] == 16)].iloc[0]
    assert shm16["bottleneck"] == "scg"
    assert bool(shm16["harness_limited"]) is False
    # And the existing scaling metrics still compute (16c ≈ 36.7/31.1 ≈ 1.18× vs ideal 16×).
    assert abs(float(shm16["tput_norm"]) - 36.7 / 31.1) < 1e-6
    assert float(shm16["ideal_norm"]) == 16.0


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
