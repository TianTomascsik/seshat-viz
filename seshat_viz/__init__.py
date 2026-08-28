"""
seshat_viz — thesis-grade visualizations of SESHAT (SCG benchmark harness) measurements.

Public surface:
    from seshat_viz.loader import load_run
    from seshat_viz.theme import apply_thesis_style, Saver
"""

from __future__ import annotations

__version__ = "0.1.0"

from .loader import RunBundle, load_run  # noqa: F401
from .theme import Saver, apply_thesis_style  # noqa: F401
