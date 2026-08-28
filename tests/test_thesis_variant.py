"""
Tests for the thesis render variant (--variant thesis): the theme switch itself, the nine
per-module thesis branches, and the F12 computed context-switch takeaway.

The render tests need a reference SESHAT run on disk: point SESHAT_THESIS_RUN (matrix
run) and SESHAT_QOS_RUN (qos_isolation run) at results/<campaign>/<timestamp> dirs.
They skip (not fail) when the variables are unset or the dirs are missing.
Runnable under pytest or as a plain script, like the other test files.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz import theme  # noqa: E402

_THESIS_RUN = os.environ.get("SESHAT_THESIS_RUN")
RUN_DIR = Path(_THESIS_RUN) if _THESIS_RUN else None

# Figure modules that grew a thesis branch, and a fragment their computed thesis takeaway
# must contain (all fragments are structural wording, not data values).
THESIS_MODULES = {
    "F11": ("validity", "headroom gate"),
    "F2": ("payload_scaling", "per-message"),
    "F3": ("crypto_overhead", "ceiling"),
    "F8": ("saturation", "loss"),
    "F9": ("resource_cost", "peak RSS"),
    "F15": ("concurrency_scaling", "ideal-linear"),
    "F16": ("closed_loop_rtt", "Closed-loop RTT"),
    "F18": ("hotreload", "reload events"),
    "F23": ("handshake_cost", "connection rate"),
}


class _restore_theme:
    """Restore the module-global variant + chrome switches around a test."""

    def __enter__(self):
        self.variant = theme.variant()
        self.chrome = theme.chrome_enabled()
        return self

    def __exit__(self, *exc):
        theme.set_variant(self.variant)
        theme.set_chrome(self.chrome)


def test_variant_switch_roundtrip_and_validation():
    with _restore_theme():
        assert theme.variant() in theme.VARIANTS
        theme.set_variant("thesis")
        assert theme.variant() == "thesis"
        assert theme.thesis_variant()
        theme.set_variant("full")
        assert not theme.thesis_variant()
        with pytest.raises(ValueError):
            theme.set_variant("poster")


def _load_bundle():
    from seshat_viz.loader import load_run

    return load_run(str(RUN_DIR))


@pytest.fixture(scope="module")
def bundle():
    if RUN_DIR is None or not RUN_DIR.is_dir():
        pytest.skip(f"reference run not on disk (set SESHAT_THESIS_RUN): {RUN_DIR}")
    return _load_bundle()


@pytest.mark.parametrize("fig_id", sorted(THESIS_MODULES))
def test_thesis_render_produces_computed_takeaway(bundle, fig_id):
    import importlib

    mod_name, fragment = THESIS_MODULES[fig_id]
    mod = importlib.import_module(f"seshat_viz.figures.{mod_name}")
    with _restore_theme(), tempfile.TemporaryDirectory() as td:
        theme.apply_thesis_style()
        theme.set_chrome(False)
        theme.set_variant("thesis")
        saver = theme.Saver(Path(td), formats=("png",))
        mod.make(bundle, saver)
        assert saver.manifest, f"{fig_id}: nothing rendered"
        entry = saver.manifest[-1]
        assert "skipped" not in entry, f"{fig_id}: skipped — {entry.get('skipped')}"
        chrome = entry.get("chrome", [])
        takeaways = [r["text"] for r in chrome if r["kind"] == "takeaway"]
        assert takeaways, f"{fig_id}: thesis render recorded no takeaway"
        joined = " ".join(takeaways)
        assert fragment.lower() in joined.lower(), (
            f"{fig_id}: takeaway lacks expected wording {fragment!r}: {joined}"
        )
        # Every thesis takeaway must carry computed numbers, never be pure prose.
        assert any(ch.isdigit() for ch in joined), f"{fig_id}: takeaway carries no numbers"


_QOS_RUN = os.environ.get("SESHAT_QOS_RUN")
QOS_RUN_DIR = Path(_QOS_RUN) if _QOS_RUN else None


def test_f24_renders_from_qos_run_with_computed_takeaway():
    if QOS_RUN_DIR is None or not QOS_RUN_DIR.is_dir():
        pytest.skip(f"qos-isolation run not on disk (set SESHAT_QOS_RUN): {QOS_RUN_DIR}")
    from seshat_viz.figures import priority_isolation
    from seshat_viz.loader import load_run

    qos_bundle = load_run(str(QOS_RUN_DIR))
    with _restore_theme(), tempfile.TemporaryDirectory() as td:
        theme.apply_thesis_style()
        theme.set_chrome(False)
        theme.set_variant("thesis")
        saver = theme.Saver(Path(td), formats=("png",))
        priority_isolation.make(qos_bundle, saver)
        entry = saver.manifest[-1]
        assert "skipped" not in entry, f"F24 skipped: {entry.get('skipped')}"
        takeaways = [r["text"] for r in entry.get("chrome", []) if r["kind"] == "takeaway"]
        assert takeaways and "bulk contention" in takeaways[0]
        assert "lost frames" in takeaways[0]


def test_f24_skips_cleanly_without_qos_scenarios(bundle):
    from seshat_viz.figures import priority_isolation

    with _restore_theme(), tempfile.TemporaryDirectory() as td:
        theme.set_chrome(False)
        saver = theme.Saver(Path(td), formats=("png",))
        priority_isolation.make(bundle, saver)
        entry = saver.manifest[-1]
        assert entry.get("skipped"), "F24 must skip on runs without qos_* scenarios"


def test_f12_full_render_records_ctxsw_takeaway(bundle):
    from seshat_viz.figures import timeline

    with _restore_theme(), tempfile.TemporaryDirectory() as td:
        theme.apply_thesis_style()
        theme.set_chrome(False)
        theme.set_variant("full")
        saver = theme.Saver(Path(td), formats=("png",))
        timeline.make(bundle, saver)
        assert saver.manifest and "skipped" not in saver.manifest[-1]
        chrome = saver.manifest[-1].get("chrome", [])
        takeaways = [r["text"] for r in chrome if r["kind"] == "takeaway"]
        assert takeaways, "F12 full render records no takeaway"
        assert "context-switch" in takeaways[0]


if __name__ == "__main__":
    test_variant_switch_roundtrip_and_validation()
    if RUN_DIR is not None and RUN_DIR.is_dir():
        b = _load_bundle()
        for fid in sorted(THESIS_MODULES):
            test_thesis_render_produces_computed_takeaway(b, fid)
        test_f12_full_render_records_ctxsw_takeaway(b)
    print("ok")
