#!/usr/bin/env bash
# Print-figure batch export, kept as a worked example of driving seshat-viz for a
# multi-campaign print export: per-figure source runs, the print variant, a manifest
# provenance check, and staging the vector PDFs under stable names for a LaTeX build.
#
#   scripts/export_print_figures.sh [RUN_PROCFS] [RUN_EBPF] [RUN_QOS] [RUN_PERF]
#
# The print figure set is fed by FOUR campaign runs (the manifest records which
# figure came from which — the post-render check below fails loud on a mismatch):
#   RUN_PROCFS  — the unperturbed procfs pass: every figure except F9/F24/F30
#   RUN_EBPF    — the eBPF pass: F9 only (per-syscall counters)
#   RUN_QOS     — the qos_isolation campaign: F24 only
#   RUN_PERF    — the kernel-scope perf ladder slice: F30 only (which also reads the
#                 relaybackend-perf-* relay pass it finds beside RUN_PERF)
# The two-host wire figures (F26–F28) come from the wire campaign dirs under
# WIRE_RESULTS, and F29 finds its relay-backend-ab-* dirs next to RUN_PROCFS.
# The renders land in figures-print/ (captions.txt + manifest.json are the citation
# source) and the vector PDFs are copied to $IMG (default figures-print/export;
# point IMG at a LaTeX img/ tree) under the stable eval_* names a document includes.
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_PROCFS="${1:-${SESHAT_RUN_PROCFS:?pass the procfs matrix run dir as arg 1 or set SESHAT_RUN_PROCFS (e.g. .../SCG-SESHAT/results/<campaign>/<timestamp>)}}"
RUN_EBPF="${2:-${SESHAT_RUN_EBPF:?pass the eBPF run dir (F9) as arg 2 or set SESHAT_RUN_EBPF}}"
RUN_QOS="${3:-${SESHAT_RUN_QOS:?pass the qos_isolation run dir (F24) as arg 3 or set SESHAT_RUN_QOS}}"
RUN_PERF="${4:-${SESHAT_RUN_PERF:?pass the kernel-scope perf ladder run dir (F30) as arg 4 or set SESHAT_RUN_PERF}}"
OUT="${OUT:-figures-print}"
# Where the eval_* vector PDFs are staged; override IMG to point at a LaTeX img/ tree.
IMG="${IMG:-$OUT/export}"
FIGS_PROCFS="F11,F2,F3,F7,F8,F12,F15,F16,F17,F18,F19,F20,F23,F25,F29"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

"$PY" -m seshat_viz "$RUN_PROCFS" --variant print --no-chrome --only "$FIGS_PROCFS" \
    --out "$OUT" --format pdf,png
"$PY" -m seshat_viz "$RUN_EBPF" --variant print --no-chrome --only F9 \
    --out "$OUT" --format pdf,png
"$PY" -m seshat_viz "$RUN_QOS" --variant print --no-chrome --only F24 \
    --out "$OUT" --format pdf,png
"$PY" -m seshat_viz "$RUN_PERF" --variant print --no-chrome --only F30 \
    --out "$OUT" --format pdf,png

# The two-host wire figures (F26–F28) come from the wire campaign dirs, not from a
# matrix run — rendered only when WIRE_RESULTS points at a results/ root holding them.
WIRE_RESULTS="${WIRE_RESULTS:-}"
if [ -n "$WIRE_RESULTS" ] && [ -d "$WIRE_RESULTS/wire-run" ]; then
    "$PY" -m seshat_viz "$RUN_PROCFS" --variant print --no-chrome --only F26,F27,F28 \
        --wire-results "$WIRE_RESULTS" --out "$OUT" --format pdf,png
else
    echo "skip: wire figures F26-F28 (set WIRE_RESULTS to a SESHAT results root containing wire-run/)"
fi

declare -A MAP=(
  [f11_measurement_validity]=eval_validity
  [f02_payload_scaling]=eval_payload_scaling
  [f03_crypto_overhead]=eval_crypto_retained
  [f16_closed_loop_rtt]=eval_latency
  [f15_concurrency_scaling]=eval_concurrency
  [f08_saturation_knee]=eval_saturation
  [f09_resource_cost_of_security]=eval_memory
  [f23_handshake_cost]=eval_handshake
  [f18_hotreload_robustness]=eval_hotreload
  [f24_priority_isolation]=eval_priority_isolation
  [f07_latency_tails]=eval_latency_tails
  [f19_jitter_determinism]=eval_jitter
  [f25_shm_ring_compare]=eval_shm_ring_compare
  [f12_system_metrics_timeline]=eval_sysmetrics_timeline
  [f17_connection_setup]=eval_connection_setup
  [f20_cipher_cost]=eval_cipher_cost
  [f26_wire_loopback_sweep]=eval_wire_sweep
  [f27_wire_qos]=eval_wire_qos
  [f28_wire_ktls_ab]=eval_wire_ktls
  [f29_relay_backend_ab]=eval_relay_backend
  [f30_hw_counters]=eval_hw_counters
)

# Post-render check: every mapped figure must exist, and the manifest must attribute
# each figure to the expected campaign — a silent re-render from a stale or wrong run
# is exactly the failure mode this guards against.
"$PY" - "$OUT" "$RUN_PROCFS" "$RUN_EBPF" "$RUN_QOS" "$RUN_PERF" <<'EOF'
import json, sys
from pathlib import Path

out, run_procfs, run_ebpf, run_qos, run_perf = sys.argv[1:6]
manifest = json.loads((Path(out) / "manifest.json").read_text())
figs = manifest["figures"]  # dict: FIG_ID -> {files, run_dir, run_label, variant, ...}
expected = {  # FIG_ID -> (output stem, expected source run)
    "F11": ("f11_measurement_validity", run_procfs),
    "F2": ("f02_payload_scaling", run_procfs),
    "F3": ("f03_crypto_overhead", run_procfs),
    "F16": ("f16_closed_loop_rtt", run_procfs),
    "F15": ("f15_concurrency_scaling", run_procfs),
    "F8": ("f08_saturation_knee", run_procfs),
    "F9": ("f09_resource_cost_of_security", run_ebpf),
    "F23": ("f23_handshake_cost", run_procfs),
    "F18": ("f18_hotreload_robustness", run_procfs),
    "F24": ("f24_priority_isolation", run_qos),
    "F7": ("f07_latency_tails", run_procfs),
    "F19": ("f19_jitter_determinism", run_procfs),
    "F25": ("f25_shm_ring_compare", run_procfs),
    "F12": ("f12_system_metrics_timeline", run_procfs),
    "F17": ("f17_connection_setup", run_procfs),
    "F20": ("f20_cipher_cost", run_procfs),
    "F26": ("f26_wire_loopback_sweep", run_procfs),
    "F27": ("f27_wire_qos", run_procfs),
    "F28": ("f28_wire_ktls_ab", run_procfs),
    "F29": ("f29_relay_backend_ab", run_procfs),
    "F30": ("f30_hw_counters", run_perf),
}
errors = []
for fig_id, (stem, run) in expected.items():
    entry = figs.get(fig_id)
    if entry is None:
        errors.append(f"MISSING from manifest: {fig_id} ({stem})")
        continue
    if not (Path(out) / f"{stem}.pdf").is_file():
        errors.append(f"MISSING PDF: {stem}.pdf")
    rd = entry.get("run_dir", "")
    if rd != run:
        errors.append(f"WRONG RUN for {fig_id}: {rd} (expected {run})")
if errors:
    print("manifest check FAILED:", file=sys.stderr)
    for e in errors:
        print("  " + e, file=sys.stderr)
    sys.exit(1)
print(f"manifest check OK: {len(expected)} print figures, correctly attributed")
EOF

mkdir -p "$IMG"
copied=0
for stem in "${!MAP[@]}"; do
    src="$OUT/$stem.pdf"
    if [ -f "$src" ]; then
        cp "$src" "$IMG/${MAP[$stem]}.pdf"
        copied=$((copied + 1))
        echo "  $src -> $IMG/${MAP[$stem]}.pdf"
    fi
done
echo "exported $copied figure PDF(s) to $IMG"
