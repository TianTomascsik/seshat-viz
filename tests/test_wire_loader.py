"""
Tests for seshat_viz.wire — the wire-campaign loader.

Guards the data-hygiene rules the wire campaign forced into existence:

  * the CSV's `mode` column is the MEDIUM and must survive verbatim as `medium`
    (loader._enrich_factors would clobber a column named `mode`);
  * `#rN` replicate suffixes split into cell/rep;
  * pre-guard campaigns (send.json without `rtt_resyncs`) drop ONLY their
    non-256-multiple RTT rows — the contaminated 64 B cells — keeping the valid
    1024/16384 values from the same directory;
  * dead cells (rtt_n == 0 / empty RTT on an RTT-bearing cell) are dropped;
  * the ktlsfalse A/B arm maps to tls/1.3+mtls, everything else to ktls/1.3+mtls,
    dtls to dtls/1.2+mtls;
  * wire_summary_merged.csv is preferred over the raw CSV;
  * aggregate() computes the t-based CI95 (no scipy in the venv).

Runnable under pytest or as a plain script.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wire_fixtures import make_campaign, row  # noqa: E402

from seshat_viz.wire import aggregate, ci95, load_wire  # noqa: E402


def _rtt_row(scenario, medium, msg, p50, **kw):
    return row(scenario, medium, message_bytes=msg, connections="", rtt_us_mean=p50,
               rtt_us_ci95=0.5, rtt_us_p50=p50, rtt_us_p99=p50 * 1.5, cpu_pct_mean=5.0, **kw)


def _sidecar(p50=None, resyncs=None, n=1000, sender_gbps=None):
    out = {"measure_s": 20.0, "measurement_side": "sender", "proto": "tcp", "rtt_n": n}
    if p50 is not None:
        out.update({"rtt_us_mean": p50, "rtt_us_p50": p50, "rtt_us_p99": p50 * 1.5})
    if resyncs is not None:
        out["rtt_resyncs"] = resyncs
    if sender_gbps is not None:
        out["sender_gbps"] = sender_gbps
    return out


def test_medium_rename_and_rep_split():
    with tempfile.TemporaryDirectory() as tmp:
        make_campaign(Path(tmp), "wire-qos3", [
            _rtt_row("qos-safety-alone#r1", "wire", 256, 300.0),
            _rtt_row("qos-safety-alone#r2", "wire", 256, 310.0),
        ], sidecars={"qos-safety-alone#r1": _sidecar(300, resyncs=0),
                     "qos-safety-alone#r2": _sidecar(310, resyncs=0)})
        wb = load_wire(tmp)
        assert wb is not None
        # medium survives verbatim; the loader-derived `mode` never touches it
        assert sorted(wb.df["medium"].unique()) == ["wire"]
        assert "mode" not in wb.df.columns
        assert sorted(wb.df["cell"].unique()) == ["qos-safety-alone"]
        assert sorted(wb.df["rep"].tolist()) == [1, 2]
        assert wb.df["role"].iloc[0] == "qos"
        assert not wb.df["qdisc"].iloc[0]


def test_preguard_drops_only_the_64b_rtt_rows():
    with tempfile.TemporaryDirectory() as tmp:
        # pre-guard: NO rtt_resyncs key anywhere → 64 B contaminated, 1024 valid
        make_campaign(Path(tmp), "wire-run", [
            _rtt_row("rtt-tls-64", "wire", 64, 7.7),
            _rtt_row("rtt-tls-1024", "wire", 1024, 345.4),
            row("sweep-tcp-950", "wire", offered_mbps=950, throughput_gbps_mean=0.95,
                rtt_us_p50=300, rtt_us_p99=600, message_bytes=65536, cpu_pct_mean=10),
        ], sidecars={"rtt-tls-64": _sidecar(7.7), "rtt-tls-1024": _sidecar(345.4)})
        # post-guard: rtt_resyncs present → the clean 64 B point is kept
        make_campaign(Path(tmp), "ab-loopback-ktlstrue-rtt", [
            _rtt_row("rtt-tls-64#r1", "loopback", 64, 54.75),
        ], sidecars={"rtt-tls-64#r1": _sidecar(54.75, resyncs=0)})
        wb = load_wire(tmp)
        cells = wb.df.groupby("campaign")["cell"].apply(list).to_dict()
        assert "rtt-tls-64" not in cells["wire-run"], "contaminated 64 B row survived"
        assert "rtt-tls-1024" in cells["wire-run"], "valid 256-multiple RTT was dropped"
        assert "sweep-tcp-950" in cells["wire-run"]
        assert cells["ab-loopback-ktlstrue-rtt"] == ["rtt-tls-64"], "clean 64 B point lost"
        # opting out keeps the row but flags it
        wb_all = load_wire(tmp, drop_contaminated=False)
        flagged = wb_all.df[wb_all.df["contaminated"].fillna(False)]
        assert set(flagged["cell"]) == {"rtt-tls-64"}
        assert set(flagged["campaign"]) == {"wire-run"}


def test_dead_cells_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        make_campaign(Path(tmp), "lo-rtt3", [
            row("rtt-tls-1024#r1", "loopback", message_bytes=1024, connections="",
                cpu_pct_mean=0.3),  # empty RTT on an RTT-bearing cell
        ], sidecars={"rtt-tls-1024#r1": _sidecar(resyncs=22, n=0)})
        make_campaign(Path(tmp), "ab-wire-ktlstrue-throughput", [
            row("tput-tls-c1#r1", "wire", throughput_gbps_mean=0.948, cpu_pct_mean=4.8),
        ])
        wb = load_wire(tmp)
        assert "lo-rtt3" not in set(wb.df["campaign"]), "dead campaign survived"
        assert "ab-wire-ktlstrue-throughput" in set(wb.df["campaign"])
        # tput rows have no RTT by design — they must never be treated as dead
        assert len(wb.df[wb.df["campaign"] == "ab-wire-ktlstrue-throughput"]) == 1


def test_protocol_mapping_and_arm():
    with tempfile.TemporaryDirectory() as tmp:
        make_campaign(Path(tmp), "ab-wire-ktlsfalse-throughput", [
            row("tput-tls-c1#r1", "wire", throughput_gbps_mean=0.947, cpu_pct_mean=5.7),
        ])
        make_campaign(Path(tmp), "ab-wire-ktlstrue-throughput", [
            row("tput-tls-c1#r1", "wire", throughput_gbps_mean=0.948, cpu_pct_mean=4.8),
        ])
        make_campaign(Path(tmp), "wire-run", [
            row("dtls-dgram", "wire", transport="udp", protocol="dtls",
                throughput_gbps_mean=0.9, cpu_pct_mean=8.0),
        ])
        wb = load_wire(tmp)
        by_campaign = wb.df.set_index("campaign")
        assert by_campaign.loc["ab-wire-ktlsfalse-throughput", "protocol"] == "tls/1.3+mtls"
        assert by_campaign.loc["ab-wire-ktlsfalse-throughput", "arm"] == "user"
        assert by_campaign.loc["ab-wire-ktlstrue-throughput", "protocol"] == "ktls/1.3+mtls"
        assert by_campaign.loc["wire-run", "protocol"] == "dtls/1.2+mtls"
        assert (wb.df["protocol_raw"].isin(["tls", "dtls"])).all()


def test_merged_csv_preferred_and_gbps_per_core_filled():
    with tempfile.TemporaryDirectory() as tmp:
        raw = [row("tput-tls-c1", "wire", throughput_gbps_mean=0.948, cpu_pct_mean=4.8)]
        merged = [row("tput-tls-c1", "wire", throughput_gbps_mean=0.948, cpu_pct_mean=4.8,
                      delivered_gbps=0.947, link_limited="true", bottleneck="link")]
        make_campaign(Path(tmp), "wire-run", raw, merged_rows=merged)
        wb = load_wire(tmp)
        r = wb.df.iloc[0]
        assert r["source_csv"] == "wire_summary_merged.csv"
        assert float(r["delivered_gbps"]) == 0.947
        # vintage split closed: gbps_per_core recomputed from sender-side achieved
        assert abs(float(r["gbps_per_core_filled"]) - 0.948 / 0.048) < 1e-6


def test_aggregate_t_ci():
    import pandas as pd

    df = pd.DataFrame({"arm": ["k"] * 3, "v": [4.0, 5.0, 6.0]})
    out = aggregate(df, ["arm"], ["v"])
    assert out["n"].iloc[0] == 3
    assert out["v_med"].iloc[0] == 5.0
    # t(0.975, df=2)=4.303, sd=1.0, n=3 → 4.303/sqrt(3)
    assert abs(out["v_ci95"].iloc[0] - 4.303 / np.sqrt(3)) < 1e-9
    assert ci95([1.0]) == 0.0


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_wire_loader: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
