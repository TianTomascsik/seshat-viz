"""
theme.py — one consistent visual identity for every thesis figure.

Provides:
  * stable, colorblind-safe colors keyed by transport and by protocol, so a given
    color means the same thing in *every* figure;
  * `apply_thesis_style()` — print-friendly matplotlib rcParams (serif body font to
    match a LaTeX thesis, restrained grid/spines);
  * `Saver` — writes each figure as BOTH a vector PDF (for \\includegraphics) and a
    raster PNG preview in one call, and records a manifest;
  * figure chrome — `set_headline()` (the ONLY way figures may set their headline;
    never call `fig.suptitle`/headline `ax.set_title` directly) plus the
    `add_provenance`/`add_method_note`/`add_takeaway` footer helpers. All of them
    honor `set_chrome(False)` (the CLI's `--no-chrome`): the text is then recorded
    on the figure instead of drawn, and lands in the Saver manifest / captions.txt
    so a LaTeX caption can carry it;
  * small shared helpers (Pareto frontier, unit formatters, harness-limited flagging).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: never needs an X server
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .loader import PROTOCOL_ORDER, TRANSPORT_ORDER, protocol_label, transport_label

# --------------------------------------------------------------------------------------
# Palettes (Okabe–Ito colorblind-safe base, extended where needed)
# --------------------------------------------------------------------------------------

# Okabe & Ito colorblind-safe qualitative palette.
_OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Transport → color (distinct hues; tproxy added with the transparent-proxy interface).
TRANSPORT_COLORS: Dict[str, str] = {
    "shm": "#009E73",    # green   — fastest IPC path (byte-stream ring)
    "shm-slot": "#E69F00", # amber — SHM fixed-slot (Vyukov) ring variant
    "unix": "#56B4E9",   # sky     — UDS
    "tcp": "#0072B2",    # blue    — the workhorse
    "tproxy": "#CC79A7", # purple  — transparent proxy
    "udp": "#D55E00",    # orange  — datagram / lossy
}

# Transport → marker (so transports are distinguishable in greyscale prints too).
TRANSPORT_MARKERS: Dict[str, str] = {
    "shm": "o",
    "shm-slot": "X",
    "unix": "s",
    "tcp": "^",
    "tproxy": "P",
    "udp": "D",
}

# Protocol → color. A sequential-ish ramp up the "security ladder": plaintext is grey,
# TLS family in blues, kTLS in greens, mTLS in purples, DTLS in oranges/reds.
PROTOCOL_COLORS: Dict[str, str] = {
    "none": "#7F7F7F",
    "tls/1.2": "#9ECAE1",
    "tls/1.3": "#3182BD",
    "tls/1.2+ale": "#7FCDCD",      # TLS 1.2 over UDP with ETCS ALEPKT framing (lighter teal)
    "tls/1.3+ale": "#17BECF",      # TLS 1.3 over UDP with ETCS ALEPKT framing (distinct teal)
    "tls/1.3+resume": "#6BAED6",   # TLS 1.3 family, lighter (session resumption)
    "ktls/1.2": "#A1D99B",
    "ktls/1.3": "#31A354",
    "ktls/1.2+mtls": "#74C476",  # kernel-offloaded mutual TLS (green family, mtls variant)
    "ktls/1.3+mtls": "#006D2C",
    "tls/1.2+mtls": "#BCBDDC",
    "tls/1.3+mtls": "#756BB1",
    "tls/1.2+integrity": "#FDD0A2",
    "tls/1.3+integrity": "#FD8D3C",  # integrity family (orange), darker for the 1.3 variant
    "dtls/1.0": "#FDAE6B",
    "dtls/1.2": "#E6550D",
    "dtls/1.2+mtls": "#A63603",
}


def transport_color(transport: str) -> str:
    return TRANSPORT_COLORS.get(str(transport), "#444444")


def transport_marker(transport: str) -> str:
    return TRANSPORT_MARKERS.get(str(transport), "o")


def protocol_color(protocol: str) -> str:
    return PROTOCOL_COLORS.get(str(protocol), "#444444")


def palette_for(values: Sequence[str]) -> List[str]:
    """A stable color list for an arbitrary categorical sequence (cycles Okabe–Ito)."""
    return [_OKABE_ITO[i % len(_OKABE_ITO)] for i in range(len(values))]


# --------------------------------------------------------------------------------------
# Global style
# --------------------------------------------------------------------------------------

# A single accent for threshold/reference guides.
#
# ACCENT POLICY — #B2182B in the accent role may be used ONLY as a threshold or
# reference guide LINE: always dashed (or dotted), always with a legend entry
# naming the threshold. Never for a data series, never for text colour, never
# for box/badge edges, never as an unkeyed ring. In-plot annotation text is
# always GREYS["ink"] or GREYS["annot"] at regular weight (white is permitted
# only for labels inside dark filled bars). Badges are plain ink text with no
# box. The chrome takeaway banner keeps ACCENT — it is chrome, stripped from
# thesis renders.
ACCENT = "#B2182B"
GRID = "#D9D9D9"
HARNESS_HATCH = "////"

# The only permitted neutral inks. Every grey drawn by a figure module comes
# from here — ad-hoc grey literals in figures/ are a style-guard test failure.
GREYS = {
    "ink": "#222222",       # value labels, primary in-plot text
    "annot": "#555555",     # secondary annotations, neutral legend glyphs
    "muted": "#8A8A8A",     # de-emphasised notes
    "faint": "#BBBBBB",     # reference/ideal lines, range rails
    "edge": "#333333",      # bar/patch edge colour, everywhere
    "baseline": "#9A9A9A",  # THE baseline-series grey: plaintext reference
                            # dots, "alone" bars, copy-baseline series
}

# Semantic verdict colours — fills and bands that pass judgement (headroom
# classes, expected-vs-unexpected DSCP marks, coverage states). One meaning per
# hex, thesis-wide; never reused as "series 4". `bad` deliberately shares the
# ACCENT hex: one red in the whole document, role-constrained by the policy
# above. `neutral` is grey by design (the no-verdict state) and `warn` sits
# below 3:1 contrast on white, so every SEM fill carries an adjacent label or
# legend entry — colour is never the only carrier.
SEM = {
    "ok": "#2C7FB8",
    "warn": "#E08214",
    "bad": "#B2182B",
    "neutral": "#9E9E9E",
}

# Scoped category palette for figure-local categoricals: ciphers, handshake
# primitives, relay backends, QoS conditions. Fixed order, never cycled. The
# transport + protocol palettes exhaust the CVD-safe hue space (validated:
# worst internal pair ΔE 14.5 deutan / 22.0 normal), so this palette is legal
# ONLY where its entities are legend-keyed in the same figure and never drawn
# in the same axes as transport- or protocol-coloured series. A fourth
# simultaneous category is always a baseline and wears GREYS["baseline"].
CATEGORY = [
    "#8465DB",  # violet
    "#6F6A00",  # olive
    "#85336A",  # plum
]

# Metric channels for multi-signal panels (F12 timelines, F9 memory).
METRIC = {
    "cpu": CATEGORY[0],
    "rss": CATEGORY[2],
    "pss": "#C490B1",   # light-plum sibling of rss
    "ctxsw": GREYS["muted"],
}

# QoS condition colours, shared by F24 and F27 (previously two duplicated
# module-local dicts). "alone" is the uncontended baseline, hence grey.
CONDITION_COLORS = {
    "alone": GREYS["baseline"],
    "safety": CATEGORY[0],
    "normal": CATEGORY[1],
}

# The five figure font sizes. Every fontsize= in figures/ uses one of these —
# numeric literals there are a style-guard failure. Panel titles are always
# FS["panel"] at regular weight via panel_title(); the bold rcParams title
# path is reserved for the chrome headline.
FS = {
    "annot": 7.5,   # in-plot value labels and annotations
    "small": 8.5,   # legend text, secondary labels
    "label": 10.5,  # axis labels (matches rcParams axes.labelsize)
    "panel": 10.5,  # per-panel titles, regular weight
    "tick": 9.0,    # tick labels (matches rcParams)
}

# MARKER CONVENTION — one global rule set:
#   * Percentile pairs: hollow marker = p50, filled marker = p99.
#   * Boundness (F15/F1 only): hollow = load-generator/host-bound,
#     filled = gateway-bound. Never both meanings in one figure; the legend
#     states which applies.
#   * Harness-limited has exactly two encodings: HARNESS_HATCH on bars
#     (keyed via harness_legend_handle) and hollow markers on points/lines
#     (keyed via an explicit legend entry). The historical "///", "*", "†",
#     alpha-fade, grey-out and red-patch variants are retired.

# Figure "chrome" = the self-describing text layers (headline, grey footer lines, the
# ▸ takeaway banner). Enabled by default; the CLI's --no-chrome disables it for thesis
# embedding, where the LaTeX caption carries that information instead.
_CHROME_ENABLED = True


def set_chrome(enabled: bool) -> None:
    """Globally enable/disable figure chrome (headline, footers, takeaway banner)."""
    global _CHROME_ENABLED
    _CHROME_ENABLED = bool(enabled)


def chrome_enabled() -> bool:
    return _CHROME_ENABLED


# Render variant. "full" is the exploratory dashboard render (every panel, every series);
# "thesis" is the print variant: each figure module may subset its panels/series to what a
# 15 cm text column can carry and recompute its takeaway over the reduced set. The variant
# NEVER changes how numbers are computed — only which panels are drawn and which takeaway
# text is recorded. Mirrors the chrome switch above.
_VARIANT = "full"
VARIANTS = ("full", "thesis")


def set_variant(variant: str) -> None:
    """Globally select the render variant ("full" or "thesis")."""
    global _VARIANT
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    _VARIANT = variant


def variant() -> str:
    return _VARIANT


def thesis_variant() -> bool:
    """True when rendering the print (thesis) variant."""
    return _VARIANT == "thesis"


def _record_chrome(fig: "plt.Figure", kind: str, text: str) -> None:
    """Remember chrome text on the figure; Saver copies it to the manifest.

    Recorded UNCONDITIONALLY (chrome on or off) so every render — including the default
    chrome-on one — leaves a machine-readable captions.txt/manifest trail. Without this,
    nothing on disk records which run produced a PNG, and a figure silently rendered
    from the wrong run is indistinguishable from a fresh one.
    """
    records = getattr(fig, "_seshat_chrome", None)
    if records is None:
        records = []
        fig._seshat_chrome = records
    records.append({"kind": kind, "text": text})


def set_headline(fig: "plt.Figure", text: str, *, ax: "plt.Axes | None" = None, **kwargs) -> None:
    """
    Set a figure's headline (bold title + run-label subtext). The one sanctioned way to
    title a figure: with chrome enabled it forwards to ``fig.suptitle`` (or
    ``ax.set_title`` when the figure hangs its headline on a single axes); with chrome
    disabled the text is only recorded for the captions manifest (it is recorded either
    way). Per-panel subplot titles are NOT chrome — keep calling ``ax.set_title`` for
    those directly.
    """
    _record_chrome(fig, "headline", text)
    if not _CHROME_ENABLED:
        return
    if ax is not None:
        ax.set_title(text, **kwargs)
    else:
        fig.suptitle(text, **kwargs)


def apply_thesis_style() -> None:
    """Install print-friendly rcParams. Call once before building figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,
            # Serif to sit naturally next to LaTeX body text.
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelweight": "normal",
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # Restrained chrome.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "pdf.fonttype": 42,   # embed TrueType (editable text in the PDF, not paths)
            "ps.fonttype": 42,
        }
    )


# --------------------------------------------------------------------------------------
# Saving: one call -> PDF + PNG, plus a manifest
# --------------------------------------------------------------------------------------


# Editorial "— higher/lower is better ↑/→" cue appended to some axis labels. It is reader
# guidance (chrome), not data, so `--no-chrome` strips it — the LaTeX caption says which way
# is better. Matches an em/en/hyphen separator + "higher|lower is better" + any trailing arrows.
_EDITORIAL_AXIS_RE = re.compile(
    r"\s*[—–-]\s*(?:higher|lower)\s+is\s+better\s*[↑↓→←]*\s*$", re.IGNORECASE
)


def strip_editorial_axis_label(label: str) -> str:
    """Remove a trailing '— higher/lower is better ↑' editorial cue from an axis label."""
    return _EDITORIAL_AXIS_RE.sub("", label).rstrip()


@dataclass
class Saver:
    """Writes figures to `out_dir` in the requested formats and tracks what was written."""

    out_dir: Path
    formats: Tuple[str, ...] = ("pdf", "png")
    manifest: List[Dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, name: str, *, fig_id: str = "", title: str = "") -> List[Path]:
        """Save `fig` as `<name>.<fmt>` for each configured format; record + close."""
        # In --no-chrome (thesis embedding) drop the "higher/lower is better" editorial cue from
        # every axis label — it is reader guidance, not data, and the caption carries it.
        if not _CHROME_ENABLED:
            for a in fig.get_axes():
                xl = a.get_xlabel()
                if xl:
                    a.set_xlabel(strip_editorial_axis_label(xl))
                yl = a.get_ylabel()
                if yl:
                    a.set_ylabel(strip_editorial_axis_label(yl))
        written: List[Path] = []
        for fmt in self.formats:
            path = self.out_dir / f"{name}.{fmt}"
            fig.savefig(path, format=fmt)
            written.append(path)
        chrome = list(getattr(fig, "_seshat_chrome", []))
        plt.close(fig)
        entry: Dict[str, object] = {
            "id": fig_id,
            "name": name,
            "title": title,
            "files": ", ".join(p.name for p in written),
        }
        if chrome:
            entry["chrome"] = chrome
        self.manifest.append(entry)
        return written

    def record_skip(self, fig_id: str, name: str, reason: str) -> None:
        self.manifest.append({"id": fig_id, "name": name, "title": "", "skipped": reason})


# --------------------------------------------------------------------------------------
# Shared plotting helpers
# --------------------------------------------------------------------------------------


def fmt_gbps(value: float, _pos: int = 0) -> str:
    return f"{value:g}"


def fmt_us(value: float, _pos: int = 0) -> str:
    """Microsecond tick label that switches to ms above 1000 µs for readability."""
    if value >= 1000:
        return f"{value / 1000:g} ms"
    return f"{value:g}"


def fmt_latency_value(us: float) -> str:
    """A complete latency string with units for annotations: '75.8 ms' / '10.9 µs'."""
    if us is None or not np.isfinite(us):
        return "—"
    if us >= 1000:
        return f"{us / 1000:.3g} ms"
    return f"{us:.3g} µs"


def fmt_bytes(value: float, _pos: int = 0) -> str:
    """Byte tick label: 64, 1K, 4K, 64K …"""
    if value <= 0:
        return "0"
    for unit, factor in (("M", 1 << 20), ("K", 1 << 10)):
        if value >= factor and value % factor == 0:
            return f"{int(value // factor)}{unit}"
    return f"{int(value)}"


def byte_axis(ax: plt.Axes, which: str = "x") -> None:
    """Format an axis as log2 byte sizes with friendly K/M tick labels."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    if which == "x":
        ax.set_xscale("log", base=2)
    else:
        ax.set_yscale("log", base=2)
    axis.set_major_formatter(FuncFormatter(fmt_bytes))


def us_axis(ax: plt.Axes, which: str = "x", log: bool = True) -> None:
    """Format an axis as latency in µs (log by default, ms labels above 1000 µs)."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    if log:
        (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis.set_major_formatter(FuncFormatter(fmt_us))


def pareto_front(
    points: Iterable[Tuple[float, float]],
    *,
    x_lower_better: bool = True,
    y_higher_better: bool = True,
) -> List[int]:
    """
    Indices of the non-dominated (Pareto-optimal) points.

    Default: minimize x (e.g. latency), maximize y (e.g. throughput). A point is on
    the frontier if no other point is at least as good on both axes and strictly
    better on one.
    """
    pts = list(points)
    keep: List[int] = []
    for i, (xi, yi) in enumerate(pts):
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        dominated = False
        for j, (xj, yj) in enumerate(pts):
            if i == j or not (np.isfinite(xj) and np.isfinite(yj)):
                continue
            better_x = (xj <= xi) if x_lower_better else (xj >= xi)
            better_y = (yj >= yi) if y_higher_better else (yj <= yi)
            strict = (xj < xi if x_lower_better else xj > xi) or (
                yj > yi if y_higher_better else yj < yi
            )
            if better_x and better_y and strict:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    # Sort frontier by x for a clean connecting line.
    keep.sort(key=lambda k: pts[k][0])
    return keep


# Standard caption for any axis/panel whose latency is open-loop *blast* latency
# (F1/F2/F3/F4/F7/F8): queue depth under an unthrottled sender, not service time. It is
# coordinated-omission-uncorrected, so only the *ranking* is meaningful — the honest
# absolute latency is F16's closed-loop ping-pong RTT. State this wherever blast p99 shows.
BLAST_LATENCY_NOTE = (
    "latency = open-loop blast (coordinated-omission-uncorrected) — relative ranking only; "
    "see F16 for honest closed-loop RTT"
)

# Standard caption for any SHM panel: the multi-ms SHM latency is a SESHAT harness
# receive-poll stall (a fixed RECV_POLL_TIMEOUT blocking recv on the client), NOT the SHM
# transport's capability. The paced SHM rows land at low-µs; the SCG relay + scg-client are
# already eventfd/futex signal-driven. State this wherever an SHM latency point looks pathological.
SHM_STALL_NOTE = (
    "SHM latency here is a harness receive-poll stall (fixed-timeout recv), not SHM capability — "
    "paced SHM is low-µs (see F16); throughput is unaffected"
)


def perf_placeholder(ax: plt.Axes, metric_label: str) -> None:
    """
    Render an empty hardware-counter panel as a *labelled placeholder* rather than a blank
    or hidden axes. On a procfs run the `perf_*` columns are all-NaN, so cycles/byte,
    cache-miss-rate and IPC panels have no data; instead of silently dropping them (which
    reads as "measured & zero"), draw a centered note so the figure honestly shows the panel
    exists but needs a `--metrics-backend perf` run.
    """
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#DDDDDD")
    ax.set_title(metric_label, fontsize=10.5)
    ax.text(
        0.5, 0.5,
        f"{metric_label}\nrequires a --metrics-backend perf run\n(procfs run: no hardware counters)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8.0, color="#9E9E9E", style="italic",
    )


def fmt_cell(chosen: Dict[str, object]) -> str:
    """
    Render a matched-cell dict (from :func:`derive.matched_cell`) as a compact provenance
    string, e.g. ``{'connections': 1, 'message_bytes': 4096}`` → ``"1c · 4K"``. Documents
    exactly which confound-controlled slice a cross-transport/-protocol figure plotted.
    """
    parts: List[str] = []
    for key, val in chosen.items():
        if val is None:
            continue
        if key == "connections":
            try:
                parts.append(f"{int(val)}c")
            except (TypeError, ValueError):
                parts.append(f"{val}c")
        elif key == "message_bytes":
            parts.append(fmt_bytes(float(val)) + "B")
        else:
            parts.append(f"{key}={val}")
    return " · ".join(parts)


def add_provenance(fig: plt.Figure, text: str) -> None:
    """Tiny grey provenance line at the bottom of a figure (host/CPU/kernel)."""
    _record_chrome(fig, "provenance", text)
    if not _CHROME_ENABLED:
        return
    fig.text(0.005, 0.002, text, fontsize=6.0, color="#9E9E9E", ha="left", va="bottom")


def add_method_note(fig: plt.Figure, text: str, *, y: float = 0.020) -> None:
    """
    A second small grey line (sitting just above :func:`add_provenance`) that documents the
    *measurement* a figure made — the matched cell it controlled for, the latency source, or
    a coverage caveat. Keeps "what interface / size / connection-count was this?" answerable
    from the figure itself rather than only the README.
    """
    _record_chrome(fig, "method", text)
    if not _CHROME_ENABLED:
        return
    fig.text(0.005, y, text, fontsize=6.0, color="#8A8A8A", ha="left", va="bottom")


def add_takeaway(fig: plt.Figure, text: str, *, y: float = -0.02) -> None:
    """
    One bold conclusion line under the figure, so every thesis figure states *the* point.

    Rendered centered just below the axes in the accent color. Keep it short — a single
    declarative sentence ("kTLS reaches 95% of routing throughput; userspace TLS pays 2×").
    """
    _record_chrome(fig, "takeaway", text)
    if not _CHROME_ENABLED:
        return
    fig.text(
        0.5, y, f"▸  {text}", fontsize=9.0, fontweight="bold", color=ACCENT,
        ha="center", va="top", wrap=True,
    )


def harness_legend_handle():
    """A proxy artist explaining the harness-limited hatch, for figure legends."""
    from matplotlib.patches import Patch

    return Patch(facecolor="white", edgecolor="#666666", hatch=HARNESS_HATCH, label="harness-limited")


def transport_legend(ax: plt.Axes, transports: Sequence[str], *, loc: str = "best") -> None:
    """Add a marker/color legend mapping transports → symbols."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0],
            marker=transport_marker(t), color="none",
            markerfacecolor=transport_color(t), markeredgecolor=transport_color(t),
            markersize=8, label=transport_label(t),
        )
        for t in transports
    ]
    ax.legend(handles=handles, title="transport", loc=loc)


def protocol_legend(ax: plt.Axes, protocols: Sequence[str], *, loc: str = "best", ncol: int = 1) -> None:
    """Add a color legend mapping protocols → swatches (ordered by the security ladder)."""
    from matplotlib.patches import Patch

    ordered = [p for p in PROTOCOL_ORDER if p in protocols]
    ordered += [p for p in protocols if p not in ordered]
    handles = [Patch(facecolor=protocol_color(p), label=protocol_label(p)) for p in ordered]
    ax.legend(handles=handles, title="protocol", loc=loc, ncol=ncol)


# --------------------------------------------------------------------------------------
# The legend system — three sanctioned forms + handle builders
# --------------------------------------------------------------------------------------
#
# Every figure legend uses exactly one of these three forms. Keys smuggled into
# axis labels or panel titles, and framed legends, are retired: legend glyphs
# always carry the true mark (colour AND fill state), never black stand-ins.


def legend_inline(ax: plt.Axes, handles=None, *, loc: str = "upper left", **kw) -> None:
    """In-axes frameless legend for single-panel figures with <= 4 entries."""
    kw.setdefault("fontsize", FS["small"])
    if handles is not None:
        ax.legend(handles=handles, loc=loc, **kw)
    else:
        ax.legend(loc=loc, **kw)


def legend_right(fig: plt.Figure, handles, *, title: str | None = None, **kw) -> None:
    """Figure-level titled column at the right edge, for multi-panel shared series."""
    kw.setdefault("fontsize", FS["small"])
    kw.setdefault("title_fontsize", FS["small"])
    fig.legend(handles=handles, loc="outside right upper", title=title, **kw)


def legend_below(fig: plt.Figure, handles, *, ncol: int | None = None, **kw) -> None:
    """Figure-level centred row below the panels, for wide/banner layouts.

    Uses the constrained-layout-aware "outside" location so the row reserves its
    own space instead of overprinting the bottom panels' axis labels.
    """
    kw.setdefault("fontsize", FS["small"])
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=ncol or len(handles),
        **kw,
    )


def percentile_handles(color: str | None = None):
    """Hollow = p50, filled = p99 (the one percentile convention, thesis-wide)."""
    from matplotlib.lines import Line2D

    c = color or GREYS["annot"]
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=c, markeredgewidth=1.3, markersize=8, label="p50 (hollow)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
               markeredgecolor=c, markersize=8, label="p99 (filled)"),
    ]


def condition_handles(conditions: Sequence[str], labels: Dict[str, str] | None = None):
    """Patch swatches for the shared QoS condition palette (F24/F27)."""
    from matplotlib.patches import Patch

    labels = labels or {}
    return [
        Patch(facecolor=CONDITION_COLORS[c], edgecolor=GREYS["edge"], linewidth=0.5,
              label=labels.get(c, c))
        for c in conditions
    ]


def category_handles(labels: Sequence[str], *, baseline: str | None = None):
    """Patch swatches in fixed CATEGORY order; optional trailing baseline-grey entry."""
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=CATEGORY[i], edgecolor=GREYS["edge"], linewidth=0.5, label=lab)
        for i, lab in enumerate(labels)
    ]
    if baseline is not None:
        handles.append(
            Patch(facecolor=GREYS["baseline"], edgecolor=GREYS["edge"], linewidth=0.5,
                  label=baseline)
        )
    return handles


def metric_handles(metrics: Sequence[str], labels: Dict[str, str] | None = None):
    """Line handles for the METRIC channels (F12 timelines, F9 memory)."""
    from matplotlib.lines import Line2D

    labels = labels or {}
    styles = {"cpu": "-", "rss": "--", "pss": "--", "ctxsw": "-"}
    return [
        Line2D([0], [0], color=METRIC[m], ls=styles.get(m, "-"), lw=1.8,
               label=labels.get(m, m))
        for m in metrics
    ]


def panel_title(ax: plt.Axes, text: str) -> None:
    """The one way to title a panel: FS["panel"], regular weight.

    Replaces the 32 historical per-module set_title(fontsize=...) overrides.
    The bold rcParams title path stays reserved for the chrome headline.
    """
    ax.set_title(text, fontsize=FS["panel"], fontweight="normal")


def annotate_value(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    yerr: float = 0.0,
    xerr: float = 0.0,
    horizontal: bool = False,
    fontsize: float | None = None,
    stagger: float = 0.0,
) -> None:
    """Value label offset past the error-bar cap, in ink, regular weight.

    Fixes the historical strike-through defect where labels anchored at the
    bar/point centre were crossed by their own error bars. Value labels are
    never tinted in the series colour: text wears ink, the mark carries
    identity. ``stagger`` adds extra points along the offset direction so
    adjacent labels whose anchors land at the same height can alternate
    (e.g. ``stagger=(j % 2) * 8`` across a bar group).
    """
    if horizontal:
        ax.annotate(
            text, (x + (xerr or 0.0), y), xytext=(4 + stagger, 0),
            textcoords="offset points",
            va="center", ha="left", fontsize=fontsize or FS["annot"], color=GREYS["ink"],
        )
    else:
        ax.annotate(
            text, (x, y + (yerr or 0.0)), xytext=(0, 3 + stagger),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=fontsize or FS["annot"], color=GREYS["ink"],
        )
