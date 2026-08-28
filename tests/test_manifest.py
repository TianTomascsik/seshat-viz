"""Provenance manifest: every render records which run produced which figure, and
mixed-run figure directories are flagged incoherent — otherwise a figure silently
overwritten from the wrong run would be indistinguishable from a fresh one."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seshat_viz.cli import _write_manifest  # noqa: E402
from seshat_viz.theme import Saver  # noqa: E402


def _bundle(run_dir: str) -> SimpleNamespace:
    return SimpleNamespace(run_dir=Path(run_dir), caption=lambda: f"host · {Path(run_dir).name}")


def _saver_with(out: Path, fig_id: str) -> Saver:
    saver = Saver(out, formats=("png",))
    saver.manifest.append({"id": fig_id, "name": f"f_{fig_id}", "title": "", "files": f"f_{fig_id}.png"})
    return saver


def test_manifest_records_run_dir_per_figure():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        path = _write_manifest(_saver_with(out, "F1"), _bundle("/data/results/run-a"))
        payload = json.loads(path.read_text())
        assert payload["coherent"] is True
        assert payload["figures"]["F1"]["run_dir"] == "/data/results/run-a"


def test_manifest_flags_mixed_run_dirs_incoherent():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _write_manifest(_saver_with(out, "F1"), _bundle("/data/results/run-a"))
        # A later partial re-render (--only F9) against a DIFFERENT run must be visible:
        path = _write_manifest(_saver_with(out, "F9"), _bundle("/data/results/run-b"))
        payload = json.loads(path.read_text())
        assert payload["coherent"] is False
        assert payload["run_dirs"] == ["/data/results/run-a", "/data/results/run-b"]
        # the untouched figure keeps its original provenance
        assert payload["figures"]["F1"]["run_dir"] == "/data/results/run-a"
        assert payload["figures"]["F9"]["run_dir"] == "/data/results/run-b"


def test_manifest_skip_entries_carry_reason():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        saver = Saver(Path(tmp), formats=("png",))
        saver.record_skip("F4", "f_f4", "no data")
        payload = json.loads(_write_manifest(saver, _bundle("/data/results/run-a")).read_text())
        assert payload["figures"]["F4"]["skipped"] == "no data"


if __name__ == "__main__":
    test_manifest_records_run_dir_per_figure()
    test_manifest_flags_mixed_run_dirs_incoherent()
    test_manifest_skip_entries_carry_reason()
    print("ok")


def test_captions_partial_rerender_merges_instead_of_clobbering():
    from seshat_viz.cli import _write_captions

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        s1 = Saver(out, formats=("png",))
        s1.manifest.append({"id": "F1", "name": "f01", "title": "",
                            "chrome": [{"kind": "headline", "text": "one"}]})
        s1.manifest.append({"id": "F2", "name": "f02", "title": "",
                            "chrome": [{"kind": "headline", "text": "two"}]})
        _write_captions(s1)
        # a partial --only F2 re-render must update F2's block and keep F1's
        s2 = Saver(out, formats=("png",))
        s2.manifest.append({"id": "F2", "name": "f02", "title": "",
                            "chrome": [{"kind": "headline", "text": "two-updated"}]})
        text = _write_captions(s2).read_text()
        assert "one" in text and "two-updated" in text and "headline:   two\n" not in text
