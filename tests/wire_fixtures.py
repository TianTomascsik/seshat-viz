"""Shared builders for synthetic wire-campaign directories (F26–F28 tests)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

HEADER = (
    "scenario,mode,transport,protocol,traffic_class,message_bytes,connections,runs,"
    "offered_mbps,throughput_gbps_mean,delivered_gbps,rtt_us_mean,rtt_us_ci95,"
    "rtt_us_p50,rtt_us_p99,latency_mean_us,latency_p99_us_mean,jitter_us_mean,"
    "loss_pct,total_lost,send_lag_mean_us,send_lag_max_us,cpu_pct_mean,gbps_per_core,"
    "dscp_observed,dscp_matched,dscp_preserved,ceiling_gbps,link_limited,bottleneck,"
    "measurement_side,dut"
).split(",")


def row(scenario: str, medium: str, **kw) -> dict:
    """One wire_summary.csv row; unset cells stay empty like the real harness."""
    base = {name: "" for name in HEADER}
    base.update(
        scenario=scenario, mode=medium, transport=kw.pop("transport", "tcp"),
        protocol=kw.pop("protocol", "tls"), traffic_class=kw.pop("traffic_class", "normal"),
        message_bytes=kw.pop("message_bytes", 65536), connections=kw.pop("connections", 1),
        runs=1, ceiling_gbps=kw.pop("ceiling_gbps", 0.9493),
        measurement_side="sender", dut="scg-over-wire",
    )
    base.update(kw)
    return base


def make_campaign(root: Path, name: str, rows: list[dict],
                  sidecars: dict[str, dict] | None = None,
                  preflight: dict | None = None,
                  merged_rows: list[dict] | None = None) -> Path:
    """Materialise one campaign dir: CSV (+ optional merged CSV, sidecars, preflight)."""
    cdir = Path(root) / name
    work = cdir / "work"
    work.mkdir(parents=True, exist_ok=True)
    for fname, data in (("wire_summary.csv", rows), ("wire_summary_merged.csv", merged_rows)):
        if data is None:
            continue
        with open(cdir / fname, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(data)
    for scenario, payload in (sidecars or {}).items():
        (work / f"{scenario}.send.json").write_text(json.dumps(payload))
    if preflight is not None:
        (work / "preflight.json").write_text(json.dumps(preflight))
    return cdir
