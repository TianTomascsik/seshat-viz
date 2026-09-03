"""
Regression tests for F15 (concurrency scaling).

Guards two conclusion invariants plus the placeholder note:

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
(`python tests/test_f15_concurrency_scaling.py`) so it needs no extra dev dependency.
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
         bottleneck: object, harness_limited: bool, protocol: str = "none",
         stem: str = "none") -> dict:
    return {
        "scenario": f"matrix_{stem}_{transport}_{transport}_64KB_{chain}_{conns}c",
        "family": "matrix",
        "transport": transport,
        "protocol": protocol,
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
    The crown-hazard shape, in miniature. Series A (TCP · routing, 2-gateway scg chain) sweeps
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


@contextlib.contextmanager
def _variant(name: str):
    """Switch the render variant for one render and restore it afterwards."""
    before = theme.variant()
    theme.set_variant(name)
    try:
        yield
    finally:
        theme.set_variant(before)


class _CaptureSaver(theme.Saver):
    """Record the panel grid (labels, titles, limits, line colours) before the figure closes."""

    panels: list[dict] = []
    height: float = 0.0

    def save(self, fig, name, **kw):
        fig.canvas.draw()  # materialise tick labels so their visibility can be read
        panels = []
        for ax in fig.axes:
            if not ax.patch.get_visible():  # twinx host-busy axes carry an invisible patch
                continue
            ss = ax.get_subplotspec()
            panels.append({
                "row": ss.rowspan.start, "col": ss.colspan.start,
                "ylabel": ax.get_ylabel(), "xlabel": ax.get_xlabel(), "title": ax.get_title(),
                "xlim": ax.get_xlim(),
                "colors": {ln.get_color() for ln in ax.get_lines()},
                "ticklabels": [t.get_text() for t in ax.get_xticklabels()
                               if t.get_visible() and t.get_text()],
                "texts": [t.get_text() for t in ax.texts],
            })
        self.panels = panels
        self.height = float(fig.get_size_inches()[1])
        return super().save(fig, name, **kw)


def _render_capture(summary: pd.DataFrame, skipped: pd.DataFrame | None = None, *,
                    variant: str = "full") -> tuple[dict, _CaptureSaver]:
    with tempfile.TemporaryDirectory() as tmp, _variant(variant):
        saver = _CaptureSaver(Path(tmp), formats=("png",))
        concurrency_scaling.make(_bundle(summary, skipped), saver)
        entry = saver.manifest[-1]
    assert "skipped" not in entry, f"F15 unexpectedly skipped: {entry.get('skipped')}"
    return entry, saver


def _render(summary: pd.DataFrame, skipped: pd.DataFrame | None = None, *,
            variant: str = "full") -> dict:
    return _render_capture(summary, skipped, variant=variant)[0]


def _three_group_summary(busy_by_conns: dict[int, float], *,
                         tproxy_kernel: bool = True) -> pd.DataFrame:
    """
    One series per print row group on TCP and TPROXY: plaintext routing, a user-space pair
    (TLS 1.3 + integrity) and kernel TLS 1.3. The TPROXY routing sweep is truncated at 16c so a
    column's rows have different reaches (the per-column shared x range case).
    """
    series = {
        ("none", "none"): (10.0, 20.0, 40.0, 32.0),
        ("tls/1.3", "tls13"): (5.0, 14.0, 30.0, 28.0),
        ("tls/1.2+integrity", "integrity_tls12"): (4.0, 13.0, 34.0, 31.0),
        ("ktls/1.3", "ktls13"): (5.5, 15.0, 32.0, 29.0),
    }
    rows = []
    for (proto, stem), tputs in series.items():
        for conns, tput in zip((1, 4, 16, 64), tputs):
            rows.append(_row("tcp", "scg", conns, tput, busy=busy_by_conns[conns],
                             bottleneck="scg-cpu", harness_limited=False,
                             protocol=proto, stem=stem))
            if proto == "none":
                if conns == 64:
                    continue  # truncated routing sweep on TPROXY
                rows.append(_row("tproxy", "direct", conns, tput * 0.8,
                                 busy=busy_by_conns[conns], bottleneck="harness-io",
                                 harness_limited=True, protocol=proto, stem=stem))
            elif proto.startswith("ktls/") and not tproxy_kernel:
                continue
            else:
                rows.append(_row("tproxy", "scg", conns, tput * 0.9, busy=busy_by_conns[conns],
                                 bottleneck="scg-cpu", harness_limited=False,
                                 protocol=proto, stem=stem))
    return pd.DataFrame(rows)


_PLAIN_YLABEL = "scaling efficiency\n(% of ideal-linear)"


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


def _by_cell(saver: _CaptureSaver) -> dict[tuple[int, int], dict]:
    return {(p["row"], p["col"]): p for p in saver.panels}


def test_row_group_is_decided_by_protocol_prefix():
    rg = concurrency_scaling._row_group
    assert rg("none") == "routing"
    assert rg("ktls/1.3") == "kernel" and rg("ktls/1.2+mtls") == "kernel"
    assert rg("tls/1.3") == "user" and rg("tls/1.2+integrity") == "user"
    assert rg("tls/1.3+mtls") == "user" and rg("dtls/1.2") == "user"


def test_print_variant_splits_rows_by_protection_group():
    """Print: one row per group (routing / user-space / kernel), columns = transports."""
    _entry, saver = _render_capture(_three_group_summary(_BUSY_HIGH), variant="print")
    cells = _by_cell(saver)
    assert set(cells) == {(r, c) for r in range(3) for c in range(2)}, sorted(cells)
    grey, faint = theme.protocol_color("none"), theme.GREYS["faint"]
    user = {theme.protocol_color("tls/1.3"), theme.protocol_color("tls/1.2+integrity")}
    kernel = {theme.protocol_color("ktls/1.3")}
    for col in range(2):
        assert grey in cells[(0, col)]["colors"] <= {grey, faint}, cells[(0, col)]["colors"]
        assert cells[(1, col)]["colors"] == user | {faint}, cells[(1, col)]["colors"]
        assert cells[(2, col)]["colors"] == kernel | {faint}, cells[(2, col)]["colors"]
        # Per-column shared x range (the truncated TPROXY routing row spans its siblings' 64c).
        assert cells[(0, col)]["xlim"] == cells[(1, col)]["xlim"] == cells[(2, col)]["xlim"]
        # Transport title on the top row only; tick labels + axis label on the bottom row only.
        assert cells[(0, col)]["title"] in ("TCP", "TPROXY")
        assert cells[(1, col)]["title"] == "" and cells[(2, col)]["title"] == ""
        assert cells[(0, col)]["ticklabels"] == [] and cells[(1, col)]["ticklabels"] == []
        assert cells[(2, col)]["ticklabels"], "bottom row must carry the connection ticks"
        assert cells[(0, col)]["xlabel"] == "" and cells[(1, col)]["xlabel"] == ""
        assert cells[(2, col)]["xlabel"] == "connections (log)"
    assert saver.height == pytest.approx(3.4 + 2.8 * 2)


def test_print_rows_carry_identity_ylabels():
    _entry, saver = _render_capture(_three_group_summary(_BUSY_HIGH), variant="print")
    cells = _by_cell(saver)
    assert cells[(0, 0)]["ylabel"] == "plaintext routing\n" + _PLAIN_YLABEL
    assert cells[(1, 0)]["ylabel"] == "user-space TLS\n" + _PLAIN_YLABEL
    assert cells[(2, 0)]["ylabel"] == "kernel TLS (kTLS)\n" + _PLAIN_YLABEL
    assert all(cells[(r, 1)]["ylabel"] == "" for r in range(3))


def test_full_variant_keeps_single_efficiency_row():
    """Full: every protocol in one efficiency row (plus absolute + latency rows), no row labels."""
    entry, saver = _render_capture(_three_group_summary(_BUSY_HIGH), variant="full")
    cells = _by_cell(saver)
    assert {r for r, _c in cells} == {0, 1, 2}  # efficiency / absolute / blast latency
    assert cells[(0, 0)]["ylabel"] == _PLAIN_YLABEL
    assert not any("plaintext routing" in p["ylabel"] for p in saver.panels)
    top = cells[(0, 0)]["colors"]
    for proto in ("none", "tls/1.3", "tls/1.2+integrity", "ktls/1.3"):
        assert theme.protocol_color(proto) in top, (proto, top)
    assert "Rows per transport" not in _chrome(entry, "method")


def test_row_sentence_only_in_print_and_takeaway_invariant():
    summary = _three_group_summary(_BUSY_HIGH)
    full = _render(summary, variant="full")
    prnt = _render(summary, variant="print")
    method = _chrome(prnt, "method")
    assert ("Rows per transport, top to bottom: plaintext routing, user-space TLS, "
            "kernel TLS (kTLS)") in method, method
    assert "Rows per transport" not in _chrome(full, "method")
    # The split is presentation only: the load-bearing numbers must not move.
    assert _chrome(prnt, "takeaway") == _chrome(full, "takeaway")


def test_missing_group_drops_its_row():
    # Routing-only data (the existing fixtures) keeps the historical single-row print figure.
    entry, saver = _render_capture(_scaling_summary(busy_by_conns=_BUSY_HIGH),
                                   _skipped_tproxy_64c(), variant="print")
    assert {r for r, _c in _by_cell(saver)} == {0}
    assert _by_cell(saver)[(0, 0)]["ylabel"] == _PLAIN_YLABEL
    assert "Rows per transport" not in _chrome(entry, "method")
    assert saver.height == pytest.approx(3.4)
    # No kernel series anywhere → two rows, and the sentence names only the rows present.
    summary = _three_group_summary(_BUSY_HIGH)
    summary = summary[~summary["protocol"].str.startswith("ktls/")]
    entry, saver = _render_capture(summary, variant="print")
    assert {r for r, _c in _by_cell(saver)} == {0, 1}
    method = _chrome(entry, "method")
    assert "top to bottom: plaintext routing, user-space TLS," in method, method
    assert "kernel TLS" not in method, method


def test_transport_missing_a_group_gets_a_placeholder_panel():
    _entry, saver = _render_capture(_three_group_summary(_BUSY_HIGH, tproxy_kernel=False),
                                    variant="print")
    cells = _by_cell(saver)
    assert set(cells) == {(r, c) for r in range(3) for c in range(2)}
    tproxy_kernel = cells[(2, 1)]
    assert tproxy_kernel["colors"] == set(), tproxy_kernel["colors"]
    assert any("no kernel TLS (kTLS) series" in t for t in tproxy_kernel["texts"])
    assert cells[(2, 0)]["colors"] == {theme.protocol_color("ktls/1.3"), theme.GREYS["faint"]}


def test_is_loadgen_tolerates_missing_bottleneck():
    """derive.scaling_table aggregates an all-NaN bottleneck to pd.NA; its truth value raised
    'boolean value of NA is ambiguous' and sank the whole figure."""
    f = concurrency_scaling._is_loadgen
    assert f(pd.NA, False) is False
    assert f(pd.NA, pd.NA) is False
    assert f(np.nan, None) is False
    assert f(None, np.True_) is True
    assert f("harness-io", pd.NA) is True


def test_render_survives_na_bottleneck_series():
    summary = _scaling_summary(busy_by_conns=_BUSY_HIGH)
    tcp = summary["transport"] == "tcp"
    summary.loc[tcp, "bottleneck"] = np.nan  # → pd.NA after aggregation
    summary.loc[tcp, "harness_limited"] = False
    for variant in ("full", "print"):
        entry = _render(summary, _skipped_tproxy_64c(), variant=variant)
        assert "gateway-bound" in _chrome(entry, "takeaway")


if __name__ == "__main__":
    test_takeaway_compares_at_common_depth_not_endpoint_max()
    test_truncated_fallback_series_is_disclosed_with_reason()
    test_no_skip_and_scg_chain_means_no_dagger()
    test_host_busy_is_measured_not_hardcoded()
    test_idle_host_keeps_not_core_count_claim()
    test_single_only_reason_is_per_transport()
    test_is_loadgen_accepts_numpy_bool()
    test_series_stem_strips_size_chain_conns()
    test_row_group_is_decided_by_protocol_prefix()
    test_print_variant_splits_rows_by_protection_group()
    test_print_rows_carry_identity_ylabels()
    test_full_variant_keeps_single_efficiency_row()
    test_row_sentence_only_in_print_and_takeaway_invariant()
    test_missing_group_drops_its_row()
    test_transport_missing_a_group_gets_a_placeholder_panel()
    test_is_loadgen_tolerates_missing_bottleneck()
    test_render_survives_na_bottleneck_series()
    print("ok")
