"""
F17 — Connection establishment: rate and the TLS handshake tax.

SESHAT's connrate runs measure how fast new connections can be set up and how long each
handshake takes — data no figure used. Left: connections-per-second (with 95% CI), routing
vs TLS variants through the gateway, across client-thread counts. Right: the per-connection
handshake latency (p50 → p99) on a log axis — the gap between a bare TCP accept and a full
TLS 1.3 handshake is the price of the first byte of a secure session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import derive
from .. import theme as T
from ..loader import RunBundle, protocol_label

FIG_ID = "F17"
NAME = "f17_connection_setup"
TITLE = "Connection establishment rate & handshake latency"

# The handshake-auth sweep (handshake_*) varies TLS parameters that never reach summary.csv
# — cert_key_type / kex_group exist only in the scenario config — so the varied factor must
# be recovered from the scenario name. Without it, every sweep row collapses to a bare
# "TLS 1.3" label and RSA's genuine ~1.5× handshake cost reads as unexplained variance
# within a single config. Prefix-keyed, longest first (rsa before kex).
_HS_VARIANTS = (
    ("handshake_tls13_rsa", "RSA cert"),
    ("handshake_tls13_ecdsa", "ECDSA cert"),
    ("handshake_kex_x25519", "X25519 kex"),
    ("handshake_kex_p256", "P-256 kex"),
)


def _hs_variant(name: object) -> object:
    """Auth-sweep factor a handshake_* scenario varies, from its name; NA otherwise."""
    n = str(name)
    for prefix, label in _HS_VARIANTS:
        if n.startswith(prefix):
            return label
    return pd.NA


def _label(r: pd.Series) -> str:
    proto = protocol_label(str(r.get("protocol", "?")))
    # Fold the name-only handshake variant into the label so two rows with the same protocol
    # string (e.g. certificate TLS 1.2 vs PSK TLS 1.2) stay distinguishable — but only when the
    # protocol label doesn't already convey it (tls/1.3+resume already reads "TLS 1.3 (resumed)").
    var = r.get("variant")
    if isinstance(var, str) and var == "psk" and "PSK" not in proto:
        proto = f"{proto} (PSK)"
    elif isinstance(var, str) and var == "resumed" and "resum" not in proto.lower():
        proto = f"{proto} (resumed)"
    hs = _hs_variant(r.get("scenario", ""))
    if not pd.isna(hs):
        proto = f"{proto} ({hs})"
    th = r.get("conn_threads")
    thtxt = f"{int(th)}-thread" if pd.notna(th) else ""
    return f"{proto} · {thtxt}".strip(" ·")


def make(bundle: RunBundle, saver: T.Saver) -> None:
    tbl = derive.connsetup_table(bundle.summary)
    if tbl.empty:
        saver.record_skip(FIG_ID, NAME, "no conns_per_sec (run has no connrate scenarios)")
        return
    tbl = tbl.copy()
    if "conn_threads" not in tbl.columns:
        tbl["conn_threads"] = np.nan
    # The connrate engine spawns one connector thread per configured connection, so for
    # scenarios whose name lacks the `_<n>thread` token (the handshake_* auth sweep, where
    # conn_threads is name-derived and thus NaN) `connections` IS the client-thread count.
    # connsetup_table doesn't carry `connections`; recover it from the summary by scenario.
    if tbl["conn_threads"].isna().any() and "connections" in bundle.summary.columns:
        cmap = bundle.summary.drop_duplicates("scenario").set_index("scenario")["connections"]
        tbl["conn_threads"] = tbl["conn_threads"].fillna(
            pd.to_numeric(tbl["scenario"].map(cmap), errors="coerce"))
    tbl["__o"] = tbl["protocol"].astype(str).map({p: i for i, p in enumerate(T.PROTOCOL_ORDER)}).fillna(99)
    tbl = tbl.sort_values(["__o", "conn_threads", "scenario"]).reset_index(drop=True)
    tbl["label"] = tbl.apply(_label, axis=1)

    # Detect resume-intent rows whose telemetry shows resumption never engaged, so the
    # "resumed" run measured a FULL handshake, not a cheap resume. Root cause (located after
    # the campaign): the gateway's encrypt connector cached upstream session tickets but never
    # presented one on reconnect — a gateway defect, since fixed and regression-tested. Gate on
    # resume-intent because resumed_fraction=0 is also recorded on plain TLS rows.
    _rf = tbl.get("resumed_fraction")
    _resume_intent = tbl["protocol"].astype(str).str.lower().str.contains("resume", regex=False)
    resumed_dead = (_resume_intent & (_rf.fillna(1.0) == 0.0)) if _rf is not None \
        else pd.Series(False, index=tbl.index)
    # Rows whose telemetry shows no resumption ever occurred measured only full handshakes,
    # so drawing them as "(resumed)" would misrepresent the data — omit them from the figure
    # and disclose the omission (and its probe-side cause) in the method note and takeaway.
    resume_omitted = bool(resumed_dead.any())
    if resume_omitted:
        tbl = tbl[~resumed_dead.to_numpy()].reset_index(drop=True)

    have_hs = "conn_handshake_p99_us" in tbl.columns and tbl["conn_handshake_p99_us"].notna().any()

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if have_hs else 1, figsize=(11.6 if have_hs else 6.4, 4.8), squeeze=False)

    # --- Panel A: connections per second ---
    axa = axes[0][0]
    y = np.arange(len(tbl))
    colors = [T.protocol_color(str(p)) for p in tbl["protocol"]]
    err = tbl["conns_per_sec_ci95"] if "conns_per_sec_ci95" in tbl.columns else None
    axa.barh(y, tbl["conns_per_sec"], color=colors, edgecolor=T.GREYS["edge"], linewidth=0.5,
             xerr=err, error_kw=dict(ecolor=T.GREYS["annot"], elinewidth=1, capsize=3))
    axa.set_yticks(y)
    axa.set_yticklabels(tbl["label"], fontsize=T.FS["small"])
    axa.invert_yaxis()
    axa.set_xlabel("connections / second (higher = better)")
    T.panel_title(axa, "Establishment rate")
    for i, v in enumerate(tbl["conns_per_sec"]):
        e = float(err.iloc[i]) if err is not None and pd.notna(err.iloc[i]) else 0.0
        T.annotate_value(axa, v, i, f"{v:,.0f}", xerr=e, horizontal=True)
    axa.grid(axis="x")
    axa.margins(x=0.16)

    # --- Panel B: handshake latency p50 -> p99 ---
    if have_hs:
        axb = axes[0][1]
        for i, r in tbl.iterrows():
            c = T.protocol_color(str(r["protocol"]))
            p50, p99 = r.get("conn_handshake_p50_us"), r["conn_handshake_p99_us"]
            if pd.notna(p50):
                axb.plot([p50, p99], [i, i], color=c, lw=2.4, alpha=0.55, solid_capstyle="round", zorder=1)
                axb.scatter([p50], [i], color="white", edgecolor=c, s=46, zorder=3, linewidth=1.4)
            axb.scatter([p99], [i], color=c, s=62, zorder=3, edgecolor=T.GREYS["edge"], linewidth=0.4)
            axb.annotate(T.fmt_latency_value(p99), (p99, i), xytext=(6, 0),
                         textcoords="offset points", va="center", fontsize=T.FS["annot"],
                         color=T.GREYS["ink"])
        axb.set_yticks(y)
        axb.set_yticklabels(tbl["label"], fontsize=T.FS["small"])
        axb.invert_yaxis()
        axb.set_xscale("log")
        axb.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(T.fmt_us))
        axb.set_xlabel("handshake latency (µs, log)")
        T.legend_inline(axb, T.percentile_handles(), loc="best")
        T.panel_title(axb, "Per-connection handshake cost")
        axb.grid(axis="x", which="both")
        axb.margins(x=0.18)

    # Method note derived from the table (protocols and thread counts actually present), not a
    # hard-coded literal that drifts as the connrate matrix grows.
    protos_present = ", ".join(dict.fromkeys(
        [_label(r).split(" · ")[0] for _, r in tbl.iterrows()]))
    threads_present = ", ".join(str(int(t)) for t in sorted(tbl["conn_threads"].dropna().unique())) \
        if "conn_threads" in tbl.columns else "?"
    method = (f"'routing' baseline = bare TCP accept (loopback, kernel-level); encrypted rows go "
              f"through the SCG gateway, so the tax bundles gateway accept-path overhead with crypto. "
              f"Measured: {protos_present} at {{{threads_present}}} client threads.")
    # Resumption note: telemetry says no resumption actually occurred, so resume-intent rows
    # measured only full handshakes and are omitted from the figure. Disclose the omission with
    # its root cause rather than implying a speed-up.
    if resume_omitted:
        method += ("  ·  resume-intent rows are omitted, resumed_fraction=0 on every attempt: "
                   "root-caused post-campaign to the gateway's encrypt connector never "
                   "presenting its cached upstream ticket (a defect since fixed and "
                   "regression-tested).")
    T.add_method_note(fig, method)

    # Data-driven takeaway: like-for-like handshake tax at 1 thread (TLS 1.3 vs bare accept).
    take = "Connection setup measured directly: routing accepts dominate, TLS pays a per-handshake tax."
    if have_hs:
        one = tbl[tbl["conn_threads"] == 1] if "conn_threads" in tbl.columns else tbl
        # Pin the encrypted side to the default TLS 1.3 config: no PSK/resumed variant and no
        # handshake-auth sweep row — a non-default cert (RSA) or kex group would contaminate
        # the like-for-like ratio now that the sweep rows carry a thread count.
        plain = one["scenario"].astype(str).map(_hs_variant).isna() \
            if "scenario" in one.columns else True
        enc_rows = one[(one["protocol"].astype(str) == "tls/1.3") & (one.get("variant").isna()
                       if "variant" in one.columns else True) & plain]
        base_rows = one[one["protocol"].astype(str) == "none"]
        enc = enc_rows["conn_handshake_p50_us"].mean() if not enc_rows.empty else np.nan
        base = base_rows["conn_handshake_p50_us"].mean() if not base_rows.empty else np.nan
        if np.isfinite(enc) and np.isfinite(base) and base > 0:
            take = (f"A TLS 1.3 handshake costs ~{enc/base:.0f}× a bare loopback accept at 1 thread "
                    f"({T.fmt_latency_value(base)} → {T.fmt_latency_value(enc)} at p50; the tax bundles "
                    f"gateway accept-path overhead with crypto).")

    # Surface the resumption omission in the takeaway itself. Root cause (located after the
    # campaign): the gateway's TCP encrypt connector cached upstream session tickets but never
    # presented one on reconnect — a gateway defect, since fixed and regression-tested.
    # Resumption also stays disabled by default as a security posture.
    if resume_omitted:
        take += ("  Resume-intent configurations are omitted: resumption never engaged "
                 "(resumed_fraction=0 on every attempt) — root-caused post-campaign to the "
                 "gateway's encrypt connector never presenting its cached upstream ticket, "
                 "a defect since fixed and regression-tested.")

    T.set_headline(fig, f"{TITLE}  ·  {bundle.label}", y=1.04)
    T.add_takeaway(fig, take)
    T.add_provenance(fig, bundle.caption())
    saver.save(fig, NAME, fig_id=FIG_ID, title=TITLE)
