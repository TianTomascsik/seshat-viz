"""
Regression tests for F3 (crypto overhead vs same-transport routing baseline).

Guards the audited takeaway/labelling defects:
  * the "AES-GCM ceiling" mean must exclude transports whose crypto rows are transport-bound,
    not AES-bound (a datagram-rate-bound UDP path 30x below the plateau deflated the ceiling
    and contradicted the transport-independence claim inside its own average);
  * the fast-vs-slow retained-% contrast must be matched-scheme, not a protocol-mix blend
    (blending DTLS-only rows into one side could even reverse the direction), and the "cost
    grows with transport speed" claim may only render when the data shows it;
  * panel titles must not round a sub-1 Gbps routing baseline up to "1 Gbps";
  * harness-limited rows/baselines must be disclosed (hollow markers / dagger titles /
    method-note counts), and the method note must not deny the UDP routing baseline the
    figure divides by.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_fix_f3.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import crypto_overhead  # noqa: E402
from seshat_viz.loader import RunBundle, _enrich_factors  # noqa: E402


def _panel(tr: str, base: float, schemes: dict[str, float], *,
           base_hl: bool = False, hl: dict[str, bool] | None = None):
    """One `panels` entry shaped exactly as make() builds it:
    (transport, base, base_lat, base_hl, agg with a 'tput' + 'hl' column per scheme)."""
    agg = pd.DataFrame({"tput": pd.Series(schemes, dtype=float)})
    agg.index.name = "_proto"
    agg["hl"] = [bool((hl or {}).get(p, False)) for p in agg.index]
    return (tr, base, float("nan"), base_hl, agg)


# ------------------------------------------------------------------------------------------
# _takeaway_text: ceiling pool + matched-scheme contrast (audit F3-1 / F3-2)
# ------------------------------------------------------------------------------------------

def test_takeaway_ceiling_excludes_datagram_bound_transport():
    """A transport whose crypto mean sits far below the plateau (UDP at ~0.3 Gbps vs ~10.6)
    must be excluded from the ceiling mean and named as excluded — not silently averaged in."""
    panels = [
        _panel("tcp", 22.0, {"tls/1.2": 10.0, "ktls/1.2": 10.4}),
        _panel("shm", 15.0, {"tls/1.2": 11.0, "ktls/1.2": 11.0}),
        _panel("udp", 0.5, {"tls/1.2": 0.21, "dtls/1.2": 0.42}),
    ]
    text = crypto_overhead._takeaway_text(panels)
    assert text is not None
    assert "~11 Gbps" in text, text          # plateau mean (10.2 + 11.0)/2 -> ~11
    assert "~7 Gbps" not in text, text       # deflated all-transport mean (~7.2) must not render
    assert "UDP is excluded from the ceiling" in text, text
    assert "0.50 Gbps" in text, text         # the excluded clause states the true baseline
    # UDP never supplies a side of the fast/slow contrast (the old 47%-vs-59% artifact).
    assert "% of UDP" not in text, text


def test_takeaway_contrast_is_matched_scheme_not_protocol_mix():
    """The slow side carries an extra DTLS-only row; its own-mix mean would read 50% but the
    matched-scheme (tls/1.2) value is 70% — the takeaway must use the matched number."""
    panels = [
        _panel("tcp", 20.0, {"tls/1.2": 9.0}),                    # keeps 45%
        _panel("shm", 10.0, {"tls/1.2": 7.0, "dtls/1.2": 3.0}),   # matched keeps 70%, blend 50%
    ]
    text = crypto_overhead._takeaway_text(panels)
    assert text is not None
    assert "70% of SHM" in text, text
    assert "45% of TCP" in text, text
    assert "50%" not in text, text
    assert "grows with transport speed" in text, text


def test_takeaway_direction_claim_only_when_supported():
    """When the faster transport retains MORE, the 'cost grows with transport speed' claim
    must not render; the contrast is stated neutrally instead."""
    panels = [
        _panel("tcp", 20.0, {"tls/1.2": 12.0}),   # keeps 60%
        _panel("shm", 10.0, {"tls/1.2": 5.0}),    # keeps 50%
    ]
    text = crypto_overhead._takeaway_text(panels)
    assert text is not None
    assert "grows with transport speed" not in text, text
    assert "60% of TCP" in text, text
    assert "50% of SHM" in text, text


def test_takeaway_integrity_profile_stays_out_of_the_aes_ceiling():
    """+integrity is the NULL-cipher profile — it must not drag the AES-GCM ceiling down."""
    panels = [
        _panel("tcp", 20.0, {"tls/1.2": 10.0, "tls/1.2+integrity": 2.0}),
        _panel("shm", 15.0, {"tls/1.2": 11.6, "tls/1.2+integrity": 2.0}),
    ]
    text = crypto_overhead._takeaway_text(panels)
    assert text is not None
    assert "~11 Gbps" in text, text   # mean(10, 11.6), NOT the integrity-diluted ~6


# ------------------------------------------------------------------------------------------
# Panel-title baseline precision (audit F3-4)
# ------------------------------------------------------------------------------------------

def test_panel_title_base_precision():
    fmt = crypto_overhead._fmt_base
    assert fmt(0.5147) == "0.51"   # the old f'{:.0f}' rendered this as '1' (+94%)
    assert fmt(1.5) == "1.5"
    assert fmt(22.4) == "22"
    assert fmt(14.977) == "15"


# ------------------------------------------------------------------------------------------
# End-to-end make(): titles, harness-limited disclosure, method-note honesty (F3-3 / F3-5)
# ------------------------------------------------------------------------------------------

class _CaptureSaver(theme.Saver):
    """Saver that snapshots the axes titles before the figure is closed."""
    titles: list[str] | None = None

    def save(self, fig, name, **kw):
        self.titles = [ax.get_title() for ax in fig.get_axes()]
        return super().save(fig, name, **kw)


def _bundle(summary: pd.DataFrame) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(
        run_dir=Path("20260101-000000"),
        summary=summary,
        runs=empty,
        sysmetrics=empty,
        saturation=empty,
        skipped=pd.DataFrame(columns=["scenario", "reason"]),
        sysinfo={"hostname": "test"},
    )


def _f3_summary() -> pd.DataFrame:
    """Three transports x (routing + crypto) at one 256B/1c/direct cell, with a sub-1 Gbps
    datagram-bound UDP baseline and a mix of harness-limited flags — the audited shape."""
    rows = [
        # scenario, transport, protocol, tput, harness_limited
        ("matrix_routing_tcp_tcp_256B_direct_1c", "tcp", "none", 22.3997, True),
        ("matrix_tls13_tcp_tcp_256B_direct_1c", "tcp", "tls/1.3", 10.0, False),
        ("matrix_ktls13_tcp_tcp_256B_direct_1c", "tcp", "ktls/1.3", 10.3, True),
        ("matrix_routing_shm_shm_256B_direct_1c", "shm", "none", 15.0, False),
        ("matrix_tls13_shm_shm_256B_direct_1c", "shm", "tls/1.3", 11.0, False),
        ("matrix_ktls13_shm_shm_256B_direct_1c", "shm", "ktls/1.3", 11.2, False),
        ("matrix_routing_udp_udp_256B_direct_1c", "udp", "none", 0.5147, True),
        ("matrix_dtls12_udp_udp_256B_direct_1c", "udp", "dtls/1.2", 0.42, False),
        ("matrix_tls13_udp_udp_256B_direct_1c", "udp", "tls/1.3", 0.21, False),
    ]
    return _enrich_factors(pd.DataFrame({
        "scenario": [r[0] for r in rows],
        "transport": [r[1] for r in rows],
        "protocol": [r[2] for r in rows],
        "message_bytes": [256] * len(rows),
        "connections": [1] * len(rows),
        "throughput_gbps_mean": [r[3] for r in rows],
        "latency_p99_us_mean": [100.0 if r[2] == "none" else 5000.0 for r in rows],
        "harness_limited": [r[4] for r in rows],
    }))


def test_f3_end_to_end_titles_and_disclosures():
    bundle = _bundle(_f3_summary())
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CaptureSaver(Path(tmp), formats=("png",))
        crypto_overhead.make(bundle, saver)
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F3 unexpectedly skipped: {entry.get('skipped')}"
    assert entry["id"] == "F3"

    # Titles: adaptive precision; the old † flag is retired — a harness-limited baseline
    # now renders as a hollow baseline dot, keyed in the figure legend.
    titles = saver.titles or []
    assert "UDP · routing 0.51 Gbps" in titles, titles    # not '1 Gbps' (F3-4), capped (F3-5)
    assert "TCP · routing 22 Gbps" in titles, titles
    assert "SHM · routing 15 Gbps" in titles, titles
    assert not any("†" in t for t in titles), titles
    assert not any("routing 1 Gbps" in t for t in titles), titles

    chrome = {r["kind"]: r["text"] for r in entry.get("chrome", [])}
    method = chrome.get("method", "")
    # The stale denial of the UDP baseline is gone; the datagram-bound caveat replaces it.
    assert "no plaintext-UDP routing baseline" not in method, method
    assert "datagram-rate-bound" in method, method
    # Harness-limited disclosure with computed counts: 1 of 6 crypto rows, 2 of 3 baselines.
    assert "harness-limited" in method, method
    assert "1/6 crypto rows" in method, method
    assert "2/3 routing baselines" in method, method

    takeaway = chrome.get("takeaway", "")
    assert "UDP is excluded from the ceiling" in takeaway, takeaway
    assert "% of UDP" not in takeaway, takeaway


def test_f3_no_harness_column_still_renders_without_disclosure():
    """Without a harness_limited column the figure renders and the note omits the clause."""
    summary = _f3_summary().drop(columns=["harness_limited"])
    bundle = _bundle(summary)
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CaptureSaver(Path(tmp), formats=("png",))
        crypto_overhead.make(bundle, saver)
    entry = saver.manifest[-1]
    assert "skipped" not in entry, entry
    titles = saver.titles or []
    assert "UDP · routing 0.51 Gbps" in titles, titles    # no dagger anywhere
    assert not any("†" in t for t in titles), titles
    chrome = {r["kind"]: r["text"] for r in entry.get("chrome", [])}
    assert "harness-limited" not in chrome.get("method", "")


if __name__ == "__main__":
    test_takeaway_ceiling_excludes_datagram_bound_transport()
    test_takeaway_contrast_is_matched_scheme_not_protocol_mix()
    test_takeaway_direction_claim_only_when_supported()
    test_takeaway_integrity_profile_stays_out_of_the_aes_ceiling()
    test_panel_title_base_precision()
    test_f3_end_to_end_titles_and_disclosures()
    test_f3_no_harness_column_still_renders_without_disclosure()
    print("ok")
