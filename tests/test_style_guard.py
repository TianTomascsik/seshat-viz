"""
Style guard: the theme invariants, enforced at the source level.

Every colour, fontsize, and legend decision lives in seshat_viz/theme.py
(GREYS/SEM/CATEGORY/METRIC/FS + panel_title/annotate_value/legend_*). These tests fail
the build if a figure module reintroduces a module-local literal — a local hex, a
fontsize literal, an ad-hoc font family, a home-grown harness-limited encoding, or
ACCENT used as a text colour.

Sanctioned exceptions, encoded below rather than waived ad hoc:
- "#FFFFFF"/"white" as a hollow-marker face or in-dark-bar text colour;
- numeric fontsize inside T.set_headline(...) calls (figure chrome, not in-axes text).
"""

from __future__ import annotations

import re
from pathlib import Path

from seshat_viz import theme as T

FIGURES_DIR = Path(__file__).resolve().parents[1] / "seshat_viz" / "figures"
MODULES = sorted(p for p in FIGURES_DIR.glob("*.py") if p.name != "__init__.py")

_HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
_FONTSIZE_LIT = re.compile(r"fontsize\s*=\s*[0-9]")
_FAMILY = re.compile(r"""(?:font)?family\s*=\s*["']""")
_SET_TITLE = re.compile(r"\.set_title\(")
_SLASH_HATCH = re.compile(r"""hatch\s*=\s*["']/+["']""")
_BOLD = re.compile(r"""fontweight\s*=\s*["']bold["']""")
_ACCENT_TEXT = re.compile(r"(?:\.text\(|\.annotate\().*color\s*=\s*T\.ACCENT")


def _hits(pattern: re.Pattern, *, allow=None) -> list[str]:
    out = []
    for mod in MODULES:
        lines = mod.read_text().splitlines()
        for lineno, line in enumerate(lines, 1):
            stripped = line.split("#", 1)[0] if not pattern.pattern.startswith("#") else line
            if not pattern.search(stripped if pattern is not _HEX else line):
                continue
            # A statement's kwargs may wrap: give the allow-check the two lines above
            # too, so e.g. a set_headline(...) call spanning lines stays sanctioned.
            context = "\n".join(lines[max(0, lineno - 3):lineno])
            if allow and allow(mod, context):
                continue
            out.append(f"{mod.name}:{lineno}: {line.strip()}")
    return out


def test_no_hex_colour_literals():
    """Every colour routes through theme constants; #FFFFFF hollow faces are the one out."""
    def allowed(_mod: Path, line: str) -> bool:
        return all(m.group(0).upper() == "#FFFFFF" for m in _HEX.finditer(line))

    hits = _hits(_HEX, allow=allowed)
    assert not hits, "hex colour literals outside theme.py:\n" + "\n".join(hits)


def test_no_fontsize_literals():
    """Every in-axes fontsize is a T.FS role; chrome headlines may size themselves."""
    def allowed(_mod: Path, line: str) -> bool:
        return "set_headline" in line

    hits = _hits(_FONTSIZE_LIT, allow=allowed)
    assert not hits, "numeric fontsize= literals:\n" + "\n".join(hits)


def test_no_font_family_overrides():
    hits = _hits(_FAMILY)
    assert not hits, "font family overrides:\n" + "\n".join(hits)


def test_no_raw_set_title():
    """Panel titles go through T.panel_title (one size, one weight, everywhere)."""
    hits = _hits(_SET_TITLE)
    assert not hits, "raw ax.set_title calls:\n" + "\n".join(hits)


def test_no_raw_slash_hatch():
    """Harness-limited bars hatch via T.HARNESS_HATCH, never a literal '///'."""
    hits = _hits(_SLASH_HATCH)
    assert not hits, "raw hatch strings:\n" + "\n".join(hits)


def test_no_bold_in_axes_text():
    """In-plot text is regular weight, everywhere — no exceptions."""
    hits = _hits(_BOLD)
    assert not hits, 'fontweight="bold" in figure modules:\n' + "\n".join(hits)


def test_accent_never_colours_text():
    """T.ACCENT is a dashed guide-line colour, never a text colour."""
    hits = _hits(_ACCENT_TEXT)
    assert not hits, "ACCENT-coloured text calls:\n" + "\n".join(hits)


def test_theme_contract():
    """The constants the recipes build on exist with their agreed keys and semantics."""
    assert set(T.GREYS) == {"ink", "annot", "muted", "faint", "edge", "baseline"}
    assert set(T.SEM) == {"ok", "warn", "bad", "neutral"}
    assert len(T.CATEGORY) == 3 and len(set(T.CATEGORY)) == 3
    assert set(T.FS) == {"annot", "small", "label", "panel", "tick"}
    assert T.SEM["bad"] == T.ACCENT  # one red, thesis-wide
    for m in ("cpu", "rss", "pss", "ctxsw"):
        assert m in T.METRIC
    for c in ("alone", "safety", "normal"):
        assert c in T.CONDITION_COLORS

    # Percentile convention: hollow = p50, filled = p99 — encoded in the shared handles.
    h50, h99 = T.percentile_handles()
    assert "p50" in h50.get_label() and h50.get_markerfacecolor() in ("white", "#FFFFFF")
    assert "p99" in h99.get_label() and h99.get_markerfacecolor() not in ("white", "none")


def _main() -> int:
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    return failed


if __name__ == "__main__":
    raise SystemExit(_main())
