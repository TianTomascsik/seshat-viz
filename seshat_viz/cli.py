"""
Command-line driver for the SESHAT visualization suite.

    python -m seshat_viz <results_dir> [--out figures/] [--only F1,F4] [--format pdf,png]

`<results_dir>` may be a specific run directory (e.g. .../results/20260626-104017) or a
`results/` root, in which case the newest run with data is used. Each figure is built
independently; one that lacks the data it needs is skipped with a logged reason rather than
aborting the whole run. A manifest of what was written (or skipped) is printed at the end.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

from . import figures as figpkg
from .loader import load_run
from .theme import VARIANTS, Saver, apply_print_style, set_chrome, set_variant, variant

# Default results root, relative to the repo layout (seshat-viz/ sits beside SCG-SESHAT/).
_DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "SCG-SESHAT" / "results"


def _parse_only(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    return {tok.strip().upper() for tok in value.split(",") if tok.strip()}


def _write_captions(saver: Saver) -> Path:
    """
    Persist each figure's chrome text (headline, method note, provenance, takeaway) to
    <out>/captions.txt, one block per figure. Written on EVERY render so there is always
    a machine-readable record of which run produced which figure — a chrome-on render
    otherwise leaves no on-disk provenance, and a stale or wrong-run figure would be
    indistinguishable from a fresh one. Under --no-chrome the same text doubles as the
    LaTeX caption source.
    """
    path = saver.out_dir / "captions.txt"
    order = ("headline", "takeaway", "method", "provenance")

    def _block(entry) -> List[str]:
        lines = [f"{entry.get('id', '')}  {entry.get('name', '')}"]
        # A figure may record several notes of the same kind (e.g. multiple method notes);
        # keep them all — collapsing by kind silently dropped every note but the last.
        chrome: dict = {}
        for r in entry.get("chrome", []):
            chrome.setdefault(r["kind"], []).append(" ".join(str(r["text"]).split()))
        for kind in order:
            texts = chrome.get(kind)
            lines.append(f"  {kind + ':':<12}{' · '.join(texts) if texts else '(none)'}")
        lines.append("")
        return lines

    # Merge with existing blocks so a partial re-render (--only F13) updates only its own
    # figures instead of clobbering every other block (same rule as manifest.json).
    rendered = {str(e.get("id", "")): _block(e) for e in saver.manifest if "skipped" not in e}
    blocks: List[Tuple[str, List[str]]] = []
    if path.is_file():
        cur_id, cur_lines = "", []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^(F\d+[A-Z]?)\s", line)
            if m:
                if cur_id:
                    blocks.append((cur_id, cur_lines))
                cur_id, cur_lines = m.group(1), [line]
            elif cur_id:
                cur_lines.append(line)
        if cur_id:
            blocks.append((cur_id, cur_lines))
    merged: List[str] = []
    seen = set()
    for bid, lines in blocks:
        merged.extend(rendered.get(bid, lines) if bid in rendered else lines)
        seen.add(bid)
    for bid, lines in rendered.items():
        if bid not in seen:
            merged.extend(lines)
    path.write_text("\n".join(merged), encoding="utf-8")
    return path


def _write_manifest(saver: Saver, bundle) -> Path:
    """
    Persist a machine-readable provenance manifest to <out>/manifest.json: the run
    directory and label every figure in this render came from, plus per-figure status.
    A partial re-render (--only) against a different run then leaves an inspectable
    trail instead of silently mixing runs in one figures/ directory.
    """
    import datetime as _dt
    import json as _json

    path = saver.out_dir / "manifest.json"
    prior: dict = {}
    if path.is_file():
        try:
            prior = _json.loads(path.read_text(encoding="utf-8")).get("figures", {})
        except Exception:
            prior = {}
    stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    figures = dict(prior)
    for entry in saver.manifest:
        rec = {
            "run_dir": str(bundle.run_dir),
            "run_label": bundle.caption(),
            "rendered_at": stamp,
            "variant": variant(),
        }
        if "skipped" in entry:
            rec["skipped"] = entry["skipped"]
        else:
            rec["files"] = entry.get("files", "")
        figures[str(entry.get("id", entry.get("name", "?")))] = rec
    runs = sorted({v.get("run_dir", "?") for v in figures.values()})
    payload = {
        "note": "one entry per figure id; MIXED run_dirs here mean the figures dir "
                "holds renders from different runs — regenerate before publishing",
        "coherent": len(runs) <= 1,
        "run_dirs": runs,
        "figures": figures,
    }
    path.write_text(_json.dumps(payload, indent=1), encoding="utf-8")
    return path


def _print_manifest(saver: Saver) -> None:
    print("\n" + "=" * 72)
    print(f"  Figures written to: {saver.out_dir}")
    print("=" * 72)
    for entry in saver.manifest:
        fid = entry.get("id", "")
        if "skipped" in entry:
            print(f"  [SKIP] {fid:<4} {entry['name']:<34} — {entry['skipped']}")
        else:
            print(f"  [ OK ] {fid:<4} {entry['name']:<34} {entry.get('files', '')}")
    n_ok = sum(1 for e in saver.manifest if "skipped" not in e)
    n_skip = sum(1 for e in saver.manifest if "skipped" in e)
    print("-" * 72)
    print(f"  {n_ok} figure(s) generated, {n_skip} skipped.")
    print("=" * 72)


def _variant_arg(value: str) -> str:
    """Validate --variant; "thesis" stays accepted as a silent alias of "print"."""
    if value == "thesis":
        return "print"
    if value not in VARIANTS:
        raise argparse.ArgumentTypeError(f"invalid choice: {value!r} (choose from {', '.join(VARIANTS)})")
    return value


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seshat-viz",
        description="Generate publication-grade figures from a SESHAT results directory.",
    )
    parser.add_argument(
        "results_dir", nargs="?", default=str(_DEFAULT_RESULTS),
        help="A specific run dir or a results/ root (default: newest under SCG-SESHAT/results).",
    )
    parser.add_argument("--out", default="figures", help="Output directory for figures (default: figures/).")
    parser.add_argument("--only", default=None, help="Comma-separated figure IDs to build, e.g. F1,F4,F8.")
    parser.add_argument("--format", default="pdf,png", help="Comma-separated formats (default: pdf,png).")
    parser.add_argument("--coverage", default=None,
                        help="Path to a code-coverage summary (coverage.json) for F13; "
                             "defaults to <run_dir>/coverage.json or <results-root>/coverage.json.")
    parser.add_argument("--wire-results", default=None,
                        help="Results root holding the two-host wire campaign dirs "
                             "(wire-run/, ab-*/, knee-*/, ...) for F26–F28; defaults to "
                             "probing the run's parent directories.")
    parser.add_argument("--variant", default="full", type=_variant_arg,
                        metavar="{full,print}",
                        help="Render variant: 'full' draws every panel/series (exploratory "
                             "dashboard); 'print' lets each figure subset its panels to a "
                             "print-legible layout and recompute its takeaway accordingly. "
                             "Pair with a dedicated --out dir (e.g. figures-print/).")
    parser.add_argument("--no-chrome", action="store_true",
                        help="Strip figure chrome for LaTeX embedding: the bold headline "
                             "(title + run label), the small grey provenance/method footer "
                             "lines, and the red takeaway banner. The suppressed text is "
                             "written per figure to <out>/captions.txt so it can be moved "
                             "into a document's captions.")
    parser.add_argument("--list", action="store_true", help="List available figures and exit.")
    args = parser.parse_args(argv)

    if args.list:
        print("Available figures:")
        for mod in figpkg.REGISTRY:
            print(f"  {mod.FIG_ID:<4} {mod.NAME:<34} {mod.TITLE}")
        return 0

    only = _parse_only(args.only)
    formats = tuple(f.strip().lower() for f in args.format.split(",") if f.strip())
    if not formats:
        formats = ("pdf", "png")

    try:
        bundle = load_run(args.results_dir, coverage_path=args.coverage,
                          wire_results=args.wire_results)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Loaded run: {bundle.run_dir}")
    executed = len(bundle.summary)
    skipped = len(bundle.skipped)
    total = executed + skipped
    coverage = f"  ({100 * executed / total:.0f}% executed)" if total else ""
    print(f"  scenarios: {executed}   runs rows: {len(bundle.runs)}   "
          f"sysmetrics rows: {len(bundle.sysmetrics)}   saturation rows: {len(bundle.saturation)}")
    print(f"  skipped rows: {skipped}{coverage}")
    if bundle.wire is not None:
        print(f"  wire rows: {len(bundle.wire.df)} ({len(bundle.wire.dirs)} campaign dirs)")
    else:
        print("  wire: none")
    print(f"  host: {bundle.caption()}")

    apply_print_style()
    set_chrome(not args.no_chrome)
    set_variant(args.variant)
    saver = Saver(Path(args.out), formats=formats)

    for mod in figpkg.REGISTRY:
        if only is not None and mod.FIG_ID.upper() not in only:
            continue
        try:
            mod.make(bundle, saver)
        except Exception as exc:  # one bad figure must not sink the rest
            saver.record_skip(mod.FIG_ID, mod.NAME, f"error: {exc}")
            print(f"  [ERR ] {mod.FIG_ID} {mod.NAME}: {exc}", file=sys.stderr)
            traceback.print_exc()

    _print_manifest(saver)
    captions = _write_captions(saver)
    manifest = _write_manifest(saver, bundle)
    if args.no_chrome:
        print(f"  chrome suppressed — caption text written to {captions}")
    else:
        print(f"  caption text + provenance manifest written to {captions} / {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
