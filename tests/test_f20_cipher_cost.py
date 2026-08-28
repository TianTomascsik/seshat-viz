"""
Regression tests for F20 (cipher-suite AEAD cost grid).

Guards three failure modes:
  * F20-1 — harness-limited cells (both ciphers pinned at the load-generator ceiling, the
    whole DTLS facet in the nightly) rendered indistinguishably from trusted cells and fed
    the takeaway median; they must be hatched, disclosed in the method note, and excluded
    from the per-cell gap pool. The headline cell must never be a harness-limited cell.
  * F20-2 — the takeaway claimed ChaCha "costs ~44% more throughput at equal security":
    the comparator is the *fastest* GCM suite (usually AES-128, not equal-security AES-256)
    and the direction was inverted (ChaCha delivers ~31% LESS). The wording must state the
    computed deficit against the fastest GCM suite and drop the equal-security framing.
  * F20-3 guard — a perf run whose cycles counters cover only ONE cipher (the existing
    AES-128-only sweep) cannot support a cipher comparison; the cycles/byte row must stay
    on the CPU% stand-in until ≥2 ciphers carry counters.

Runnable either under pytest (`pytest tests/`) or as a plain script
(`python tests/test_f20_cipher_cost.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import cipher_cost  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402

_RAW_CIPHER = {
    "AES-128-GCM": "aes_128_gcm_sha256",
    "AES-256-GCM": "aes_256_gcm_sha384",
    "ChaCha20-Poly1305": "chacha20_poly1305_sha256",
}


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


def _cipher_summary(
    *,
    trusted_proto: str = "tls/1.3",
    limited_proto: str | None = "dtls/1.2",
    with_limited_column: bool = True,
) -> pd.DataFrame:
    """A minimal cipher sweep: the trusted protocol has a 100% best-GCM-over-ChaCha gap per
    cell; the harness-limited protocol has a compressed 5% gap (the ceiling artifact that
    must not dilute the median)."""
    rows = []
    specs = [(trusted_proto, "tcp", {"AES-128-GCM": 10.0, "AES-256-GCM": 9.5,
                                     "ChaCha20-Poly1305": 5.0}, False)]
    if limited_proto is not None:
        specs.append((limited_proto, "udp", {"AES-128-GCM": 2.1, "AES-256-GCM": 2.05,
                                             "ChaCha20-Poly1305": 2.0}, True))
    for proto, transport, gbps, limited in specs:
        for size in (1024, 16384):
            for label, tput in gbps.items():
                row = {
                    "scenario": f"cipher_{proto.replace('/', '')}_{_RAW_CIPHER[label]}_{size}",
                    "cipher": _RAW_CIPHER[label],
                    "transport": transport,
                    "protocol": proto,
                    "message_bytes": size,
                    "connections": 1,
                    "throughput_gbps_mean": tput,
                    "throughput_gbps_ci95": 0.1,
                    "cpu_pct_mean": 110.0,
                }
                if with_limited_column:
                    row["harness_limited"] = limited
                rows.append(row)
    return pd.DataFrame(rows)


class _CapturingSaver(theme.Saver):
    """Counts hatched bar patches per axes-column before the figure is closed."""

    def save(self, fig, name, *, fig_id="", title=""):  # noqa: D102 - see class docstring
        self.hatched = sum(
            1 for ax in fig.axes for p in ax.patches if p.get_hatch()
        )
        self.total_bars = sum(len(ax.patches) for ax in fig.axes)
        return super().save(fig, name, fig_id=fig_id, title=title)


def _chrome(saver: theme.Saver, kind: str) -> str:
    entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F20 unexpectedly skipped: {entry.get('skipped')}"
    return " ".join(c["text"] for c in entry.get("chrome", []) if c["kind"] == kind)


# ------------------------------------------------------------------ gap pool (F20-1)

def test_cell_gaps_exclude_harness_limited_cells():
    """The 5%-gap harness-limited cells must not enter the pool: only the two trusted
    100%-gap cells survive, so the median is 100% (not diluted toward ~52%)."""
    d = _cipher_summary()
    d["cipher_label"] = d["cipher"].map(
        {v: k for k, v in _RAW_CIPHER.items()}
    )
    gaps = cipher_cost._cell_gaps(d)
    assert len(gaps) == 2
    assert all(abs(g - 1.0) < 1e-9 for g in gaps)
    assert abs(cipher_cost._gcm_vs_chacha_gap(d) - 100.0) < 1e-6


def test_cell_gaps_without_limited_column_keep_everything():
    """No harness_limited column → nothing is excluded (all 4 cells pool; median 52.5%)."""
    d = _cipher_summary(with_limited_column=False)
    d["cipher_label"] = d["cipher"].map({v: k for k, v in _RAW_CIPHER.items()})
    gaps = cipher_cost._cell_gaps(d)
    assert len(gaps) == 4
    assert abs(cipher_cost._gcm_vs_chacha_gap(d) - 52.5) < 0.01


# ------------------------------------------------------------- cycles-row guard (F20-3)

def test_metric_rows_reject_single_cipher_cycles():
    """cycles/byte on AES-128 only (the real perf run's shape) → CPU% stand-in stays."""
    d = _cipher_summary(limited_proto=None)
    d["cipher_label"] = d["cipher"].map({v: k for k, v in _RAW_CIPHER.items()})
    d["cycles_per_byte"] = np.where(d["cipher_label"] == "AES-128-GCM", 0.07, np.nan)
    rows, cycles_shown = cipher_cost._metric_rows(d)
    assert cycles_shown is False
    assert [r[0] for r in rows] == ["throughput_gbps_mean", "cpu_pct_mean"]


def test_metric_rows_accept_two_cipher_cycles():
    d = _cipher_summary(limited_proto=None)
    d["cipher_label"] = d["cipher"].map({v: k for k, v in _RAW_CIPHER.items()})
    d["cycles_per_byte"] = np.where(
        d["cipher_label"] == "AES-256-GCM", np.nan, 0.07
    )  # AES-128 + ChaCha counted
    rows, cycles_shown = cipher_cost._metric_rows(d)
    assert cycles_shown is True
    assert "cycles_per_byte" in [r[0] for r in rows]
    assert "cpu_pct_mean" not in [r[0] for r in rows]


# --------------------------------------------------- full render: hatching + wording

def test_make_hatches_limited_bars_and_scopes_claims():
    bundle = _bundle(_cipher_summary())
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CapturingSaver(Path(tmp))
        cipher_cost.make(bundle, saver)

    # 2 protocols × 3 ciphers × 2 sizes × 2 metric rows = 24 bars; the DTLS half is hatched.
    assert saver.total_bars == 24
    assert saver.hatched == 12

    take = _chrome(saver, "takeaway")
    # F20-2: computed from the trusted pool (gap 100% → deficit 50%), correct direction,
    # no equal-security framing (the comparator is the fastest GCM suite).
    assert "~50% less throughput" in take
    assert "~100% GCM advantage" in take
    assert "equal security" not in take

    note = _chrome(saver, "method")
    assert "harness-limited" in note
    assert "DTLS 1.2" in note
    assert "excluded from the takeaway median" in note

    # F20-1: the harness-limited protocol never names the headline cell.
    headline = _chrome(saver, "headline")
    assert "TLS 1.3" in headline
    assert "DTLS" not in headline


def test_make_headline_rejects_harness_limited_cell():
    """Even the preferred TLS 1.3 cell loses the headline when it is harness-limited."""
    bundle = _bundle(_cipher_summary(trusted_proto="tls/1.2", limited_proto="tls/1.3"))
    with tempfile.TemporaryDirectory() as tmp:
        saver = theme.Saver(Path(tmp))
        cipher_cost.make(bundle, saver)
    headline = _chrome(saver, "headline")
    assert "TLS 1.2" in headline
    assert "TLS 1.3" not in headline


def test_make_no_limited_rows_keeps_clean_chrome():
    """A fully trusted sweep draws no hatching and adds no harness-limited clause."""
    bundle = _bundle(_cipher_summary(limited_proto=None))
    with tempfile.TemporaryDirectory() as tmp:
        saver = _CapturingSaver(Path(tmp))
        cipher_cost.make(bundle, saver)
    assert saver.hatched == 0
    assert "harness-limited" not in _chrome(saver, "method")


if __name__ == "__main__":
    test_cell_gaps_exclude_harness_limited_cells()
    test_cell_gaps_without_limited_column_keep_everything()
    test_metric_rows_reject_single_cipher_cycles()
    test_metric_rows_accept_two_cipher_cycles()
    test_make_hatches_limited_bars_and_scopes_claims()
    test_make_headline_rejects_harness_limited_cell()
    test_make_no_limited_rows_keeps_clean_chrome()
    print("ok")
