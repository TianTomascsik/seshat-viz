"""
F22 — System-metrics timeline, protection modes compared.

Sibling of F12. Where F12 holds the protocol fixed and varies the transport, F22 holds the
*transport* fixed and varies the **protocol**: routing (plaintext) → userspace TLS → kernel
kTLS. Everything else is pinned — single gateway, one shared connection count and message
size, and a single scenario family + chain, so the plaintext baseline is the crypto panels'
own like-for-like sibling (same client API, same phase schedule) rather than a faster
cross-family scenario that happens to share the transport. With all of that fixed, the
CPU / RSS / context-switch differences are the *crypto* cost over time — most directly,
userspace OpenSSL TLS vs kernel-offloaded kTLS at the same TLS version (rendered as
adjacent panels, and never truncated mid-pair).

The transport is auto-picked as the one carrying the richest like-for-like protocol ladder;
ties are broken toward TCP, the kernel-socket path that runs every protocol and whose CPU
trace does not carry a busy-poll floor that would flatten the very crypto deltas the figure
exists to show.

Panel mechanics (steady-state window recovery, per-PID CPU/RSS summation, context-switch
counter-reset masking) are shared verbatim with F12 via :func:`timeline._render`; the
comparable-slice selection is shared via :func:`timeline._pin_and_pick`.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import TRANSPORT_ORDER, RunBundle
from . import timeline

FIG_ID = "F22"
NAME = "f22_protocol_metrics_timeline"
TITLE = "System-metrics timeline — protection modes compared"

# Curated crypto ladder for panel order. Grouped by TLS version with the userspace/kernel
# pair (tls / ktls) adjacent so the offload comparison sits side by side, then mutual-auth
# and integrity-only, then DTLS. Values not present in a given run are simply skipped.
_LADDER = [
    "none",
    "tls/1.2",
    "ktls/1.2",
    "tls/1.3",
    "ktls/1.3",
    "tls/1.2+mtls",
    "ktls/1.2+mtls",
    "tls/1.3+mtls",
    "ktls/1.3+mtls",
    "tls/1.2+integrity",
    "tls/1.3+integrity",
    "dtls/1.0",
    "dtls/1.2",
    "dtls/1.2+mtls",
]

# Five panels: the plaintext baseline plus BOTH userspace/kernel TLS pairs (1.2 and 1.3) —
# the full offload story. A cap of 4 used to cut the ladder mid-pair, leaving TLS 1.3
# without its kTLS partner and inviting a read against kTLS 1.2 instead;
# :func:`_trim_to_pairs` additionally guarantees no cap ever splits a pair.
_MAX_SCEN = 5


def _best_slice(summ: pd.DataFrame, have_sys: set) -> dict | None:
    """
    The (transport, family, chain) slice carrying the most distinct protection modes with a
    /proc timeseries at a single gateway, connections==1 — i.e. the richest *like-for-like*
    protocol ladder, returned as a pin dict for :func:`timeline._pin_and_pick`.

    Counting per (family, chain) rather than per transport alone matters: a transport's raw
    protocol count can span measurement families with different client APIs and phase
    schedules (e.g. an ``iface_*`` plaintext row vs the ``matrix_*`` crypto rows), and a
    cross-family "baseline" panel is not the crypto panels' workload minus crypto.
    Eligibility mirrors ``_pin_and_pick`` (sustained-blast default-path rows only),
    so the count reflects what can actually render. Ties are broken toward TCP — it runs
    every protocol, and its CPU trace has no busy-poll floor to flatten the crypto deltas
    (shm/unix/tcp/tproxy all tie on full runs, so groupby order alone would
    crown SHM) — then by ``TRANSPORT_ORDER``, then lexically, so the pick is deterministic.
    """
    if summ is None or summ.empty or not {"scenario", "transport", "protocol"}.issubset(summ.columns):
        return None
    d = summ[summ["scenario"].isin(have_sys)].copy()
    d = derive.throughput_scenarios_only(d)
    for col, val in (("connections", 1), ("n_gateways", 1)):
        if col in d.columns:
            d = d[d[col] == val]
    if d.empty:
        return None
    group_cols = ["transport"] + [c for c in ("family", "chain") if c in d.columns]
    per = d.groupby(group_cols, observed=True)["protocol"].nunique()
    per = per[per >= 2]  # need at least two protection modes to compare
    if per.empty:
        return None

    def _key(item: tuple) -> tuple:
        idx, n = item
        key = idx if isinstance(idx, tuple) else (idx,)
        tr = str(key[0])
        in_order = TRANSPORT_ORDER.index(tr) if tr in TRANSPORT_ORDER else len(TRANSPORT_ORDER)
        return (-int(n), 0 if tr == "tcp" else 1, in_order, tuple(str(k) for k in key))

    best = min(per.items(), key=_key)[0]
    if not isinstance(best, tuple):
        best = (best,)
    pin: dict = {"transport": str(best[0]), "n_gateways": 1}
    for col, val in zip(group_cols[1:], best[1:]):
        pin[col] = val
    return pin


def _trim_to_pairs(scenarios: list, proto_of: dict, max_scen: int) -> list:
    """
    Truncate the ladder-ordered panel list to ``max_scen`` without splitting a
    userspace/kernel pair: when the cut would keep a ``tls/X`` panel whose ``ktls/X``
    partner exists further down the ladder but no longer fits, drop the partnerless panel
    too. Never trims below two panels — a single panel is no comparison.
    """
    kept = scenarios[:max_scen]
    cut = scenarios[max_scen:]
    if not cut or len(kept) < 3:
        return kept
    last = proto_of.get(kept[-1])
    cut_protos = {proto_of.get(s) for s in cut}
    if isinstance(last, str) and last.startswith("tls/") and f"k{last}" in cut_protos:
        kept = kept[:-1]
    return kept


def _load_disclosure(summ: pd.DataFrame, scenarios: list) -> str:
    """
    Method-note clause disclosing the panels' *achieved* operating points (F22 wording of
    the F12 helper: the panels differ by protection mode, not transport). Pinning the
    workload cell does not pin the achieved throughput — under a sustained blast each
    protection mode runs at its own crypto-bound rate, so the plaintext baseline moves
    several times the crypto panels' byte rate, and rows flagged harness-limited hit the
    load generator's ceiling, not the gateway's. Every number is computed from the summary
    rows actually plotted; empty when the columns are absent.
    """
    if summ is None or summ.empty or "scenario" not in summ.columns:
        return ""
    rows = summ[summ["scenario"].isin(scenarios)].drop_duplicates("scenario")
    if rows.empty:
        return ""
    parts: list[str] = []
    if "throughput_gbps_mean" in rows.columns:
        tput = pd.to_numeric(rows["throughput_gbps_mean"], errors="coerce").dropna()
        tput = tput[tput > 0]
        if len(tput) > 1 and float(tput.max()) - float(tput.min()) >= 0.5:
            parts.append(f"achieved load differs per panel ({tput.min():.0f}–{tput.max():.0f} Gbps)")
    if "harness_limited" in rows.columns:
        n_hl = int(sum(1 for v in rows["harness_limited"] if isinstance(v, (bool, np.bool_)) and v))
        if n_hl:
            parts.append(
                f"{n_hl}/{len(rows)} panels are harness-limited (the load generator, not the "
                "gateway, was the bottleneck)"
            )
    if not parts:
        return ""
    return (
        " · " + " · ".join(parts)
        + " — per-panel CPU reflects each protection mode at its own achievable operating point"
    )


def make(bundle: RunBundle, saver: T.Saver) -> None:
    sysm = bundle.sysmetrics
    if sysm is None or sysm.empty or "elapsed_ms" not in sysm.columns:
        saver.record_skip(FIG_ID, NAME, "no system_metrics timeseries in this run")
        return

    have_sys = set(sysm["scenario"])
    summ = bundle.summary
    pin = _best_slice(summ, have_sys)
    if pin is None:
        saver.record_skip(
            FIG_ID, NAME, "no transport carries ≥2 protection modes with system metrics"
        )
        return
    transport = pin["transport"]

    # Ask for the full ladder, then truncate locally so the cap can respect pair boundaries.
    scenarios, chosen, comparable = timeline._pin_and_pick(
        summ,
        have_sys,
        vary="protocol",
        pin=pin,
        order=_LADDER,
        max_scen=len(_LADDER),
    )
    if not comparable:
        saver.record_skip(
            FIG_ID,
            NAME,
            f"fewer than two protection modes share {transport}/single-gateway with system metrics",
        )
        return

    proto_of: dict = {}
    if "scenario" in summ.columns and "protocol" in summ.columns:
        rows = summ[summ["scenario"].isin(scenarios)].drop_duplicates("scenario")
        proto_of = {str(s): str(p) for s, p in zip(rows["scenario"], rows["protocol"])}
    scenarios = _trim_to_pairs(scenarios, proto_of, _MAX_SCEN)

    # Claim the adjacent-panel offload comparison only when at least one full tls/ktls
    # version pair actually renders side by side (a partial run may lack the partner).
    protos = [proto_of.get(s, "") for s in scenarios]
    has_pair = any(
        p.startswith("tls/") and protos[i + 1] == f"k{p}" for i, p in enumerate(protos[:-1])
    )

    tlabel = T.transport_label(transport)
    method_note = (
        f"panels ARE comparable — identical transport ({tlabel}) · single gateway · matched "
        + T.fmt_cell(chosen)
        + "; only the protection mode varies, so the CPU/ctxsw deltas are the crypto cost over "
        "time"
        + (
            " — userspace TLS vs kernel-offloaded kTLS render as adjacent panels"
            if has_pair
            else ""
        )
        + " · sustained-blast default-path rows only · each panel sums CPU/RSS over its gateway "
        "PIDs · shaded bands = steady-state windows recovered from the CPU trace (no phase "
        "marker in /proc) · per-rep context-switch counter resets masked"
        + _load_disclosure(summ, scenarios)
        + "."
    )
    # fig.text does not wrap and savefig uses a tight bbox, so a single overlong footer line
    # would stretch the whole canvas to its width; wrap for rendering (captions.txt
    # re-normalizes whitespace, so the machine-readable record is unaffected). Two lines at
    # most: the note is bottom-anchored, and further lines grow up into the x-tick row.
    method_note = textwrap.fill(
        method_note,
        width=max(340, len(method_note) // 2 + 40),
        break_on_hyphens=False,  # a mid-word break would survive into captions.txt as "steady- state"
        break_long_words=False,
    )
    timeline._render(
        bundle, saver, scenarios, fig_id=FIG_ID, name=NAME, title=TITLE, method_note=method_note
    )
