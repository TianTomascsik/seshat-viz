"""
Regression tests for F22 (protocol timeline: pin transport, vary protection mode).

Guards the selection invariants of F22:

  * F22-1 (P0) — the plaintext-baseline panel must be the crypto panels' like-for-like
    sibling (same scenario family + chain), not a faster cross-family row that happens to
    share the transport (the run's iface_* / shmzc_* rows used to outsort the matched
    matrix_routing_* baseline on raw throughput, so the routing panel showed a different
    workload, not "the same workload minus crypto").
  * F22-2 (P1) — on full runs shm/unix/tcp/tproxy tie on protocol coverage; the transport
    pick must break the tie deterministically toward TCP (the documented intent), not by
    groupby order (which crowned SHM, whose busy-poll CPU floor flattens the crypto deltas).
  * F22-3 (P2) — the panel cap must not split a userspace/kernel TLS pair: a partnerless
    tls/X panel invites reading its delta against the wrong kTLS version.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f22_timeline_protocol.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import timeline, timeline_protocol as tp  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402


def _bundle(summary: pd.DataFrame, sysmetrics: pd.DataFrame) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=summary,
        runs=empty,
        sysmetrics=sysmetrics,
        saturation=empty,
        skipped=empty,
        sysinfo={"hostname": "test"},
    )


def _row(scenario, transport, protocol, family, chain, gbps, *, harness_limited=False):
    return {
        "scenario": scenario,
        "transport": transport,
        "protocol": protocol,
        "family": family,
        "chain": chain,
        "message_bytes": 65536,
        "connections": 1,
        "n_gateways": 1,
        "mode": "throughput",
        "throughput_gbps_mean": gbps,
        "harness_limited": harness_limited,
    }


def _matrix_ladder(transport: str) -> list:
    """A like-for-like matrix/direct cell: plaintext + both TLS-version pairs."""
    return [
        _row(f"matrix_routing_{transport}_64KB_1c", transport, "none", "matrix", "direct", 10.0),
        _row(f"matrix_tls12_{transport}_64KB_1c", transport, "tls/1.2", "matrix", "direct", 5.0),
        _row(f"matrix_ktls12_{transport}_64KB_1c", transport, "ktls/1.2", "matrix", "direct", 5.5),
        _row(f"matrix_tls13_{transport}_64KB_1c", transport, "tls/1.3", "matrix", "direct", 4.8),
        _row(f"matrix_ktls13_{transport}_64KB_1c", transport, "ktls/1.3", "matrix", "direct", 5.2),
    ]


def _decoys(transport: str) -> list:
    """Faster plaintext rows from OTHER measurement families (the F22-1 contamination):
    a zero-copy shmzc microbenchmark and an iface_* baseline. Both carry sysmetrics, so
    only family/chain matching — not data availability — can keep them out."""
    return [
        _row("shmzc_shm_slot_tput_64KB", transport, "none", "shmzc", "n/a", 60.0),
        _row(f"iface_{transport}_throughput_64KB_1c", transport, "none", "iface", "n/a", 30.0),
    ]


def _sysmetrics_for(scenarios: list) -> pd.DataFrame:
    """A plausible /proc timeseries for every scenario (ramp → plateau → teardown)."""
    frames = []
    t = np.arange(0, 30_000, 1000)
    cpu = np.where((t > 3_000) & (t < 26_000), 150.0, 20.0)
    for scen in scenarios:
        frames.append(pd.DataFrame({
            "scenario": scen,
            "pid": 1234,
            "elapsed_ms": t,
            "cpu_pct": cpu,
            "rss_kib": 30_000 + t / 100.0,
            "voluntary_ctxt_switches": np.cumsum(np.full(len(t), 100)),
            "nonvoluntary_ctxt_switches": np.cumsum(np.full(len(t), 10)),
        }))
    return pd.concat(frames, ignore_index=True)


def test_baseline_pick_rejects_cross_family_rows():
    """F22-1: the 'none' representative must be the matrix/direct sibling even when
    higher-throughput shmzc/iface plaintext rows (with sysmetrics) exist on the transport."""
    summ = pd.DataFrame(_matrix_ladder("tcp") + _decoys("tcp"))
    have_sys = set(summ["scenario"])

    pin = tp._best_slice(summ, have_sys)
    assert pin is not None
    assert pin["family"] == "matrix" and pin["chain"] == "direct"

    scenarios, chosen, ok = timeline._pin_and_pick(
        summ, have_sys, vary="protocol", pin=pin, order=tp._LADDER, max_scen=len(tp._LADDER)
    )
    assert ok
    assert scenarios[0] == "matrix_routing_tcp_64KB_1c"
    assert "shmzc_shm_slot_tput_64KB" not in scenarios
    assert "iface_tcp_throughput_64KB_1c" not in scenarios


def test_transport_tie_prefers_tcp():
    """F22-2: a 4-way coverage tie resolves to TCP, not to whichever transport groupby
    order surfaces first; without TCP the tie-break is still deterministic (TRANSPORT_ORDER)."""
    rows = []
    for transport in ("shm", "unix", "tcp", "tproxy"):
        rows += _matrix_ladder(transport)
    summ = pd.DataFrame(rows)
    have_sys = set(summ["scenario"])
    assert tp._best_slice(summ, have_sys)["transport"] == "tcp"

    no_tcp = summ[summ["transport"] != "tcp"]
    assert tp._best_slice(no_tcp, have_sys)["transport"] == "shm"


def test_transport_pick_ignores_non_blast_rows():
    """F22-2 counting basis: paced/handshake rows carry sysmetrics too but cannot render as
    panels, so they must not inflate a transport's protocol coverage."""
    rows = _matrix_ladder("tcp")
    # udp "wins" on raw row count only via non-blast families.
    rows.append(_row("matrix_routing_udp_64KB_1c", "udp", "none", "matrix", "direct", 8.0))
    for i, proto in enumerate(["dtls/1.0", "dtls/1.2", "dtls/1.2+mtls", "tls/1.2", "tls/1.3", "ktls/1.2"]):
        r = _row(f"paced_udp_{i}_64KB_1c", "udp", proto, "paced", "direct", 1.0)
        r["mode"] = "paced"
        rows.append(r)
    summ = pd.DataFrame(rows)
    assert tp._best_slice(summ, set(summ["scenario"]))["transport"] == "tcp"


def test_trim_never_splits_tls_ktls_pair():
    """F22-3: a cap landing mid-pair drops the partnerless tls/X panel; a cap landing on a
    pair boundary keeps everything; short lists are never trimmed below two panels."""
    scenarios = ["s_none", "s_t12", "s_k12", "s_t13", "s_k13", "s_tm12", "s_km12"]
    proto_of = {
        "s_none": "none",
        "s_t12": "tls/1.2", "s_k12": "ktls/1.2",
        "s_t13": "tls/1.3", "s_k13": "ktls/1.3",
        "s_tm12": "tls/1.2+mtls", "s_km12": "ktls/1.2+mtls",
    }
    # Cap=6 would cut ktls/1.2+mtls away from tls/1.2+mtls → the tls panel goes too.
    assert tp._trim_to_pairs(scenarios, proto_of, 6) == scenarios[:5]
    # Cap=5 lands exactly on the 1.3-pair boundary → keep all five.
    assert tp._trim_to_pairs(scenarios, proto_of, 5) == scenarios[:5]
    # A tls/X panel whose partner does not exist anywhere is legitimate and kept.
    assert tp._trim_to_pairs(["s_none", "s_t12", "s_t13"], proto_of, 2) == ["s_none", "s_t12"]
    # Never trim below two panels — a single panel is no comparison.
    assert tp._trim_to_pairs(["s_t12", "s_k12"], proto_of, 1) == ["s_t12"]


def test_make_end_to_end_pins_family_and_discloses_load():
    """Full make(): decoys present, all with sysmetrics → figure renders, the method note
    stamps the matched family/chain, claims tls/ktls adjacency (both pairs render), and
    disclosures (achieved-load spread) are computed from the plotted rows."""
    summ = pd.DataFrame(_matrix_ladder("tcp") + _decoys("tcp"))
    bundle = _bundle(summ, _sysmetrics_for(sorted(summ["scenario"])))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        tp.make(bundle, saver)
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F22 unexpectedly skipped: {entry.get('skipped')}"
    assert entry["id"] == "F22"
    # The note is wrapped for rendering; normalize like captions.txt does before matching.
    method = next(c["text"] for c in entry["chrome"] if c["kind"] == "method")
    method = " ".join(method.split())
    assert "family=matrix" in method and "chain=direct" in method
    assert "adjacent panels" in method
    assert "achieved load differs per panel" in method


def test_make_softens_adjacency_claim_without_a_pair():
    """When no full tls/ktls version pair exists, the method note must not promise the
    adjacent-panel offload comparison."""
    rows = [
        _row("matrix_routing_tcp_64KB_1c", "tcp", "none", "matrix", "direct", 10.0),
        _row("matrix_tls12_tcp_64KB_1c", "tcp", "tls/1.2", "matrix", "direct", 5.0),
        _row("matrix_tls13_tcp_64KB_1c", "tcp", "tls/1.3", "matrix", "direct", 4.8),
    ]
    summ = pd.DataFrame(rows)
    bundle = _bundle(summ, _sysmetrics_for(sorted(summ["scenario"])))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        tp.make(bundle, saver)
    entry = saver.manifest[-1]
    assert "skipped" not in entry
    method = next(c["text"] for c in entry["chrome"] if c["kind"] == "method")
    assert "adjacent panels" not in " ".join(method.split())


if __name__ == "__main__":
    test_baseline_pick_rejects_cross_family_rows()
    test_transport_tie_prefers_tcp()
    test_transport_pick_ignores_non_blast_rows()
    test_trim_never_splits_tls_ktls_pair()
    test_make_end_to_end_pins_family_and_discloses_load()
    test_make_softens_adjacency_claim_without_a_pair()
    print("ok")
