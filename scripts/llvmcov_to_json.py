#!/usr/bin/env python3
"""
llvmcov_to_json.py — turn ``cargo llvm-cov --json`` exports into the compact ``coverage.json``
that seshat-viz F13 consumes.

The quality gate already runs ``cargo llvm-cov`` per workspace (the ≥80 % line-coverage bar).
Emit a machine-readable export from each, then fold them into one artifact::

    # in SCG/            (repeat for SCG-SESHAT/ and ale-frame/)
    cargo llvm-cov --workspace --json --output-path /tmp/scg.cov.json
    cargo llvm-cov --json --output-path /tmp/seshat.cov.json   # in SCG-SESHAT/
    cargo llvm-cov --json --output-path /tmp/ale.cov.json      # in ale-frame/

    python seshat-viz/scripts/llvmcov_to_json.py \
        SCG /tmp/scg.cov.json \
        SCG-SESHAT /tmp/seshat.cov.json \
        ale-frame /tmp/ale.cov.json \
        --target 80 --generated "$(date -u +%FT%TZ)" \
        > SCG-SESHAT/results/<run>/coverage.json

F13 then renders per-workspace line % (and per-crate, grouped by path) against the target line.
Input is the standard ``llvm.coverage.json.export`` format; missing keys degrade gracefully.
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List, Tuple


def _crate_of(path: str) -> str:
    """Best-effort crate name from a source path (``crates/<name>/`` or the dir above ``src/``)."""
    parts = path.replace("\\", "/").split("/")
    if "crates" in parts:
        i = parts.index("crates")
        if i + 1 < len(parts):
            return parts[i + 1]
    if "src" in parts:
        i = parts.index("src")
        if i > 0:
            return parts[i - 1]
    return parts[-2] if len(parts) >= 2 else "?"


def _pct(section: dict) -> Tuple[float, int, int]:
    """Return (percent, covered, count) from an llvm-cov summary section, defensively."""
    covered = int(section.get("covered", 0))
    count = int(section.get("count", 0))
    percent = float(section.get("percent", 100.0 * covered / count if count else 0.0))
    return percent, covered, count


def workspace_from_export(name: str, export: dict) -> dict:
    data = (export.get("data") or [{}])[0]
    totals = data.get("totals", {})
    line_pct, line_cov, line_tot = _pct(totals.get("lines", {}))
    ws: dict = {"name": name, "line_pct": round(line_pct, 2),
                "lines_covered": line_cov, "lines_total": line_tot}
    if "functions" in totals:
        ws["function_pct"] = round(_pct(totals["functions"])[0], 2)
    if "regions" in totals:
        ws["region_pct"] = round(_pct(totals["regions"])[0], 2)

    agg: Dict[str, List[int]] = {}
    for f in data.get("files", []):
        _, cov, tot = _pct((f.get("summary") or {}).get("lines", {}))
        bucket = agg.setdefault(_crate_of(f.get("filename", "?")), [0, 0])
        bucket[0] += cov
        bucket[1] += tot
    crates = [
        {"name": c, "line_pct": round(100.0 * cov / tot, 2), "lines_total": tot}
        for c, (cov, tot) in sorted(agg.items()) if tot > 0
    ]
    if len(crates) > 1:  # only worth showing when a workspace actually has several crates
        ws["crates"] = crates
    return ws


def main(argv: List[str]) -> int:
    target = 80.0
    generated = ""
    pairs: List[str] = []
    it = iter(argv)
    for tok in it:
        if tok == "--target":
            target = float(next(it))
        elif tok == "--generated":
            generated = next(it)
        else:
            pairs.append(tok)
    if not pairs or len(pairs) % 2 != 0:
        sys.stderr.write("usage: NAME EXPORT.json [NAME EXPORT.json ...] [--target N] [--generated ISO]\n")
        return 2

    workspaces = []
    for i in range(0, len(pairs), 2):
        name, path = pairs[i], pairs[i + 1]
        with open(path) as fh:
            workspaces.append(workspace_from_export(name, json.load(fh)))

    out = {"target_pct": target, "workspaces": workspaces}
    if generated:
        out["generated"] = generated
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
