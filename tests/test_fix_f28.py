"""
Tests for F28 (kTLS vs user-space TLS on a physical NIC).

Guards: the CPU-delta takeaway formula against known synthetic numbers; the
wire-RTT-starts-at-1024 disclosure (64 B cells died under the desync guard);
both variants render; skip without ab campaigns. Runnable under pytest or as a
plain script.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wire_fixtures import make_campaign, row  # noqa: E402

from seshat_viz import theme  # noqa: E402
from seshat_viz.figures import wire_ktls_ab as f28  # noqa: E402
from seshat_viz.loader import RunBundle  # noqa: E402
from seshat_viz.wire import load_wire  # noqa: E402


def _bundle(wire) -> RunBundle:
    empty = pd.DataFrame()
    return RunBundle(run_dir=Path("20260101-000000"), summary=empty, runs=empty,
                     sysmetrics=empty, saturation=empty, skipped=empty,
                     sysinfo={"hostname": "test"}, wire=wire)


def _tput(cell, medium, gbps, cpu):
    return [row(f"{cell}#r{i}", medium, throughput_gbps_mean=g, cpu_pct_mean=cpu)
            for i, g in enumerate(gbps, 1)]


def _rtt(medium, size, p50, dead=False):
    rows, sidecars = [], {}
    for i in range(1, 4):
        name = f"rtt-tls-{size}#r{i}"
        if dead:
            rows.append(row(name, medium, message_bytes=size, connections="",
                            cpu_pct_mean=0.3))
            sidecars[name] = {"rtt_n": 0, "rtt_resyncs": 22}
        else:
            rows.append(row(name, medium, message_bytes=size, connections="",
                            rtt_us_p50=p50 + i, rtt_us_p99=(p50 + i) * 1.5,
                            cpu_pct_mean=3.0))
            sidecars[name] = {"rtt_n": 5000, "rtt_resyncs": 0,
                              "rtt_us_p50": p50 + i}
    return rows, sidecars


def _campaigns(tmp: Path):
    make_campaign(tmp, "ab-loopback-ktlstrue-throughput",
                  _tput("tput-tls-c1", "loopback", [12.0, 12.1, 12.05], 49) +
                  _tput("tput-tls-c4", "loopback", [34.5, 34.6, 34.57], 250))
    make_campaign(tmp, "ab-loopback-ktlsfalse-throughput",
                  _tput("tput-tls-c1", "loopback", [10.2, 10.25, 10.22], 49) +
                  _tput("tput-tls-c4", "loopback", [31.1, 31.2, 31.16], 237))
    make_campaign(tmp, "ab-wire-ktlstrue-throughput",
                  _tput("tput-tls-c1", "wire", [0.948, 0.948, 0.948], 4.0) +
                  _tput("tput-tls-c4", "wire", [0.948, 0.948, 0.948], 4.8))
    make_campaign(tmp, "ab-wire-ktlsfalse-throughput",
                  _tput("tput-tls-c1", "wire", [0.948, 0.947, 0.948], 6.0) +
                  _tput("tput-tls-c4", "wire", [0.947, 0.948, 0.947], 6.0))
    for medium, arm in (("loopback", "true"), ("loopback", "false"),
                        ("wire", "true"), ("wire", "false")):
        rows, sidecars = [], {}
        for size, p50 in ((64, 55), (1024, 350), (16384, 1000)):
            dead = medium == "wire" and size == 64
            r, s = _rtt(medium, size, p50, dead=dead)
            rows += r
            sidecars.update(s)
        make_campaign(tmp, f"ab-{medium}-ktls{arm}-rtt", rows, sidecars=sidecars)


def _chrome(saver):
    return {c["kind"]: c["text"] for c in saver.manifest[-1]["chrome"]}


def test_takeaway_formula_and_64b_disclosure():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
        _campaigns(Path(tmp))
        wb = load_wire(tmp)
        saver = theme.Saver(Path(out), formats=("png",))
        f28.make(_bundle(wb), saver)
        chrome = _chrome(saver)
        # cpu medians 4.0 (ktls) vs 6.0 (user) at c1 → 33% less CPU
        assert "33% less CPU" in chrome["takeaway"]
        assert "4.0% vs 6.0%" in chrome["takeaway"]
        # loopback delta: 12.05/10.22 - 1 = +18%
        assert "+18% (1 conn)" in chrome["takeaway"]
        # wire 64 B cells are dead → the disclosure names the real series start
        assert "starts at 1024 B" in chrome["method"]
        assert "hardware offload remains unmeasured" in chrome["method"]
        assert "wire campaigns:" in chrome["provenance"]


def test_both_variants_render_and_skip_path():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
        _campaigns(Path(tmp))
        wb = load_wire(tmp)
        for variant in ("thesis", "full"):
            theme.set_variant(variant)
            try:
                saver = theme.Saver(Path(out) / variant, formats=("png",))
                f28.make(_bundle(wb), saver)
            finally:
                theme.set_variant("full")
            assert "chrome" in saver.manifest[-1], f"{variant} render failed"
        saver = theme.Saver(Path(out) / "skip", formats=("png",))
        f28.make(_bundle(None), saver)
        assert "skipped" in saver.manifest[-1]


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_fix_f28: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
