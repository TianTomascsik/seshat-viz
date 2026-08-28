"""
Figure registry.

Each figure module exposes:
    FIG_ID : str       e.g. "F1"
    NAME   : str       output-file stem, e.g. "f01_landscape"
    TITLE  : str       human title for the manifest
    make(bundle, saver) -> None    draws & saves, or calls saver.record_skip(...)

`REGISTRY` is the ordered list the CLI iterates over. Order follows the thesis narrative:
the trade-off landscape, then transport/structural cost, the latency/determinism family,
load & operational robustness, the resource-cost composite, and finally the
measurement-validity / provenance figures.
"""

from __future__ import annotations

from . import (
    cipher_cost,
    closed_loop_rtt,
    concurrency_scaling,
    connection_setup,
    coverage,
    crypto_overhead,
    gateway_cost,
    handshake_cost,
    heatmaps,
    hotreload,
    hw_counters,
    jitter,
    landscape,
    latency_tails,
    parallel_coords,
    payload_scaling,
    priority_isolation,
    relay_backend,
    resource_cost,
    saturation,
    shm_ring_compare,
    timeline,
    timeline_protocol,
    transport_compare,
    validity,
    wire_ktls_ab,
    wire_loopback_sweep,
    wire_qos,
)

REGISTRY = [
    # trade-off landscape & structural cost
    landscape,            # F1
    payload_scaling,      # F2
    crypto_overhead,      # F3
    heatmaps,             # F4
    transport_compare,    # F5
    gateway_cost,         # F6
    concurrency_scaling,  # F15
    shm_ring_compare,     # F25  (SHM byte-stream vs fixed-slot ring trade-off)
    # latency / determinism family
    latency_tails,        # F7
    closed_loop_rtt,      # F16
    jitter,               # F19
    # load & operational robustness
    saturation,           # F8
    connection_setup,     # F17
    hotreload,            # F18
    priority_isolation,   # F24  (traffic-class isolation, qos_isolation campaign)
    # physical-path validation (two-host wire campaign; needs --wire-results)
    wire_loopback_sweep,  # F26  (loopback realism: sweep + knee on a real 1 GbE link)
    wire_qos,             # F27  (QOS-001: prioritisation + far-side DSCP evidence)
    wire_ktls_ab,         # F28  (kTLS vs user-space TLS on a physical NIC)
    # resource cost
    resource_cost,        # F9 (consolidates old F9/F10/F14)
    relay_backend,        # F29  (relay-backend A/B: splice vs io_uring)
    hw_counters,          # F30  (kernel-scope perf: relay copy cost + protection ladder)
    cipher_cost,          # F20  (symmetric AEAD cost)
    handshake_cost,       # F23  (asymmetric handshake / cert-sig cost)
    parallel_coords,      # F21
    # measurement validity & provenance
    validity,             # F11
    timeline,             # F12  (interface view: pin protocol, vary transport)
    timeline_protocol,    # F22  (protection-mode view: pin transport, vary protocol)
    coverage,             # F13
]

__all__ = ["REGISTRY"]
