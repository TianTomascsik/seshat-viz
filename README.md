# seshat-viz — publication-grade visualizations of SESHAT benchmark data

`seshat-viz` turns the CSV measurement tree produced by **SESHAT** (the SCG benchmark
harness, <https://github.com/TianTomascsik/SCG-SESHAT>, conventionally checked out as a
sibling `../SCG-SESHAT/` of this repo) into a set of publication-quality figures for the
Secure Communication Gateway evaluation. It reads a results directory and emits **vector PDF
(for LaTeX `\includegraphics`) + PNG (preview)** figures that go well beyond plain
bar/line charts: a throughput–latency Pareto map, payload-size scaling small-multiples,
protocol×size heatmaps, latency-tail CCDFs + box-and-whisker, saturation knees, a
closed-loop-RTT / coordinated-omission-inflation map, hot-reload robustness, concurrency
scaling, jitter/determinism, a consolidated resource-cost composite, cipher-suite cost, and a
parallel-coordinates multi-objective view.

Every figure also carries a one-line **conclusion banner** (`▸ …`) stating the point it makes,
so each can stand alone in a report or paper.

Every figure is built to *make a measurement mean something* — compare transports and
protocols on latency **and** throughput, expose how payload size changes the picture, and
point at where the bottlenecks and the biggest improvement opportunities are.

This folder is self-contained and read-only with respect to SESHAT: it never modifies the
harness or the `results/` data.

## Setup

```bash
cd seshat-viz
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]                  # matplotlib, numpy, pandas, seaborn (+ pytest)
```

Dependencies are declared in `pyproject.toml`.

## Usage

```bash
# Newest run under ../SCG-SESHAT/results, all figures -> ./figures/{pdf,png}
python -m seshat_viz

# A specific run directory
python -m seshat_viz ../SCG-SESHAT/results/20260626-104017 --out figures

# Just a couple of figures, PDF only
python -m seshat_viz <run_dir> --only F1,F4,F8 --format pdf

# Document embedding: strip the headline / grey footers / red takeaway banner so the
# LaTeX caption carries that text. The suppressed text lands in <out>/captions.txt.
python -m seshat_viz <run_dir> --no-chrome

# Code-coverage mode for F13 (else it renders scenario-execution coverage)
python -m seshat_viz <run_dir> --coverage path/to/coverage.json

# Print (paper/report) variant: each figure may subset its panels to a print-legible
# layout and recompute its takeaway accordingly. Pair with a dedicated --out dir.
python -m seshat_viz <run_dir> --variant print --out figures-print

# Two-host wire campaign figures (F26-F28): point --wire-results at the results root
# holding the wire campaign dirs (wire-run/, ab-*/, knee-*/, ...). Without it, the
# run's parent directories are probed; if nothing is found the wire figures skip.
python -m seshat_viz <run_dir> --wire-results ../SCG-SESHAT/results

# List the figure catalogue and exit
python -m seshat_viz --list
```

`--variant` selects the render variant: `full` (default) draws every panel and series —
the exploratory dashboard; `print` is the print variant. `scripts/export_print_figures.sh`
is a worked example of a full batch export: it renders the print variant of every figure
from its correct source campaign, verifies the provenance manifest, and stages the vector
PDFs under stable names for a LaTeX build.

`--no-chrome` removes only the *figure-level* chrome (title + run label, the two grey
footer lines, the ▸ takeaway banner) — per-panel subplot titles, axes, legends and
in-plot annotations stay. Every suppressed line is written per figure to
`<out>/captions.txt` so the data-bearing bits (e.g. F3/F5/F7/F20's matched
transport·size·topology cell) can be moved into the caption rather than lost.

`<run_dir>` may be either a specific `results/<timestamp>/` directory **or** the
`results/` root (the newest run with data is selected). A run directory often contains
several nested per-invocation sub-runs (the routing / crypto / saturation passes of one
`collect_perf_data.sh` call); the loader unions all of their `scenarios/*/` detail so the
figures see the same scenarios as `combined_summary.csv`.

Each figure is built independently. If a figure needs data the run doesn't contain (e.g.
perf counters on a procfs-only run, or `saturation.csv` on a run without a sweep), it is
**skipped with a logged reason** rather than crashing the batch. A manifest of what was
written or skipped prints at the end.

## Figure catalogue

Listed in build order (registry order in `seshat_viz/figures/__init__.py`: trade-off
landscape → structural cost → latency/determinism → load & operational robustness →
physical-path validation → resource cost → measurement validity). Each figure is a module
under `seshat_viz/figures/`; `python -m seshat_viz --list` prints the same catalogue with
each figure's output-file stem. Every figure carries its own method note and provenance
footer, and skips with a logged reason when the run lacks the data it needs.

| ID | Module | What it shows |
|----|--------|---------------|
| **F1** | `landscape` | Throughput vs p99-latency Pareto trade-off map at a matched single-connection slice |
| **F2** | `payload_scaling` | Payload-size scaling small-multiples: throughput, blast latency & closed-loop RTT vs message size |
| **F3** | `crypto_overhead` | Cost of security: encryption overhead vs each transport's routing baseline |
| **F4** | `heatmaps` | Protocol × payload-size heatmaps of throughput and latency, per transport |
| **F5** | `transport_compare` | Transport comparison: multi-metric radar + absolute bars |
| **F6** | `gateway_cost` | Second-gateway insertion cost: 1 gateway (scg-direct) vs 2 gateways (scg-scg) |
| **F15** | `concurrency_scaling` | Concurrency scaling: throughput speedup & tail latency as connections multiply |
| **F25** | `shm_ring_compare` | SHM ring-variant trade-off: byte-stream + eventfd vs fixed-slot (Vyukov) + futex |
| **F7** | `latency_tails` | Latency tail shape: p50-normalized CCDF, faceted per transport |
| **F16** | `closed_loop_rtt` | Closed-loop RTT across the protocol × interface × payload grid, plus coordinated-omission inflation |
| **F19** | `jitter` | Jitter (packet delay variation) & determinism — the metric an ETCS/EuroRadio control loop must tolerate |
| **F8** | `saturation` | Saturation knee: offered load vs achieved goodput, loss & tail latency |
| **F17** | `connection_setup` | Connection-establishment rate and the TLS handshake tax |
| **F18** | `hotreload` | Hot-reload robustness: reconfiguring a live gateway under saturation |
| **F24** | `priority_isolation` | Traffic-class isolation under bulk contention (QoS-isolation campaign) |
| **F26** | `wire_loopback_sweep` | Loopback-testbed realism on a physical 1 GbE path (wire campaign) |
| **F27** | `wire_qos` | Traffic-class prioritisation on the wire, with far-side DSCP evidence |
| **F28** | `wire_ktls_ab` | kTLS vs user-space TLS on a physical NIC (wire A/B) |
| **F9** | `resource_cost` | The resource cost of security in one place: throughput, CPU efficiency, memory |
| **F29** | `relay_backend` | Relay-backend A/B on the routing path: splice vs read/write vs io_uring |
| **F30** | `hw_counters` | Hardware-counter evidence: relay copy cost and the protection ladder |
| **F20** | `cipher_cost` | Cipher-suite (AEAD) cost: AES-128-GCM vs AES-256-GCM vs ChaCha20-Poly1305 |
| **F23** | `handshake_cost` | Handshake-algorithm cost: server-auth signature and key-exchange group |
| **F21** | `parallel_coords` | Every configuration at once: a parallel-coordinates multi-objective trade-off map |
| **F11** | `validity` | Measurement validity: harness headroom & bottleneck attribution |
| **F12** | `timeline` | System-metrics (/proc) timeline, transports compared: pin protocol, vary transport |
| **F22** | `timeline_protocol` | System-metrics (/proc) timeline, protection modes compared: pin transport, vary protocol |
| **F13** | `coverage` | Code coverage by workspace/crate, with a scenario-coverage fallback |

Some figures need a dedicated campaign beyond the standard matrix run: F24 the
QoS-isolation campaign, F26–F28 the two-host wire campaign (`--wire-results`), F29 the
relay-backend A/B trees, F30 two kernel-scope `perf stat` campaigns, and F13's code-mode
a `coverage.json` (`--coverage`). Generated figure sets (`figures/`, `figures-nochrome/`,
`figures-print/`) are **not committed** — regenerate them from a results directory.

## How the data is read

- `combined_summary.csv` → `summary` (one row per scenario, ~64 columns).
- `scenarios/*/runs.csv` → `runs` (per-repetition rows: full latency percentiles, jitter, loss).
- `scenarios/*/system_metrics/gateway_pid_*.csv` → `sysmetrics` (CPU%/RSS/ctxsw timeseries).
- `scenarios/*/saturation.csv` → `saturation` (offered-load sweep points).
- `skipped.csv` (per sub-run) → `skipped` (scenario + reason; drives F13 coverage and the
  F15 coverage-wall caption).
- `meta.csv` / `sysinfo.csv` → host/CPU/kernel provenance shown in figure footers.

`loader._enrich_factors` also derives **analysis factors from the scenario name** so figures
can group without extra columns from SESHAT: `family` (matrix / iface / hotreload / cipher /
profile / conn / pp / …), `chain` (`scg` vs `direct`), `connections`, `message_bytes`,
`reload_trigger` / `reload_load` (hot-reload), `conn_threads` (connrate), `cipher`,
`profile_tuning` (latency / balanced / throughput), `mode` (`pingpong` / `connrate` /
`throughput`), `datapath` (`loopback` = raw, no gateway · `gateway` = through SCG), and
`n_gateways` (`0` loopback · `1` scg-direct/SingleGateway · `2` scg-scg/ScgToScg — this
disambiguates what `chain` alone cannot, since loopback and scg-direct both read `chain=direct`).
These are additive — figures that don't reference them are unaffected.

## Confound control & honest comparison

The measured space is asymmetric (historically only TCP got a multi-connection sweep and the
full protocol ladder), so any figure that aggregates across transports/protocols must **control
for the confounds** or it compares apples to oranges. Two shared mechanisms enforce this:

- **`derive.matched_cell(df, vary, controls=("connections","message_bytes"), fixed=…)`** — pins
  the confounding dimensions to a single shared value before a cross-`vary` comparison, so a
  1024-connection TCP row is never blended with a 1-connection SHM row. `connections` tie-breaks
  to the smallest present (`1c`, the only count every transport shares → the fair anchor). Used
  by F1/F3/F5/F9/F12/F19/F21/F22; each stamps the pinned cell via `theme.fmt_cell(chosen)` in a
  `theme.add_method_note(...)` line so the slice is readable from the figure itself. F12/F22
  extend the idea to the /proc timelines — pinning *all* confounders but one (transport for F12,
  protocol for F22) so their panels are mutually comparable rather than merely co-plotted.
- **Honest latency labelling** — any axis plotting `latency_p99_us_mean` is open-loop *blast*
  latency (queue depth, coordinated-omission-uncorrected); those panels carry
  `theme.BLAST_LATENCY_NOTE` pointing at F16's closed-loop RTT for the honest absolute. New
  paced-latency scenarios (`sender.rate_limit_mbps`, `co_corrected=true`) give per-message
  latency at a real throughput target below saturation.

**New SESHAT capabilities these figures consume when present** (all in the SESHAT repo,
<https://github.com/TianTomascsik/SCG-SESHAT>):
fair-tier multi-connection UDS/SHM (`configs/matrix_spec.json` → nightly `[1,4,16,64]`, so F15
and the cross-transport figures cover more than TCP); paced latency-at-target
(`configs/latency.json`); a cipher × size grid (F20); TLS resumption / PSK / more protocols &
thread counts (`configs/connrate.json`, F17); and the perf backend
(`seshat suite --tier nightly --metrics-backend perf`) which fills the cycles/byte, cache-miss
and IPC panels of F9/F20/F21. For code coverage in F13, produce a `coverage.json` with
`scripts/llvmcov_to_json.py` and pass `--coverage <path>` (or drop it beside the run).

Columns are referenced **by name with graceful fallbacks** (never by position), so the same
code ingests both "normal" (procfs) and "perf" (hardware-counter) runs, and tolerates the
schema differences between SESHAT versions. SESHAT already computes the statistics (means,
CI95, percentiles, headroom, analytical overhead); this tool only reads and visualizes them
— it does **not** recompute confidence intervals. `derive.py` only *combines* existing columns
into relationship metrics (scaling speedup, RTT-vs-blast inflation, hot-reload retention,
cycles-per-byte, …); it never recomputes statistics.

## Methodology notes to cite in captions

These reflect SESHAT's `docs/methodology.md`
(<https://github.com/TianTomascsik/SCG-SESHAT>) and are honored in the figures:

- **Throughput is wire-bytes**, not goodput (includes the 24 B SESHAT header + protocol
  overhead). Prefer relative comparisons.
- **Open-loop blast latency is coordinated-omission-uncorrected.** F1/F3/F7 latency is
  blast latency — use it for *relative* ranking, not absolute values; cite ping-pong runs
  for absolute latency claims.
- **UDP loss is real** (DTLS sweeps show 30–65 % loss past the knee) and is shown, not hidden.
- **Harness-limited** scenarios (most loopback points on a 32-thread host) are faded in F1
  and flagged in F11; headroom and bottleneck class are reported for credibility.
- **Size-matching:** F9/F20/F21 compare routing to crypto only at well-populated payload sizes
  (so routing isn't unfairly averaged over tiny messages) and exclude paced/saturation runs.
- **Closed-loop RTT is the honest absolute latency.** F16 uses the ping-pong `rtt_us_*`
  (one request in flight → coordinated-omission-free). Its right panel shows the inflation
  factor against open-loop blast p99 — often 40×–3000× — which is *why* open-loop tail latency
  (F1/F3/F7) must be read as relative ranking only.
- **F7 boxes are built from per-run percentiles, not raw samples.** SESHAT exports
  p50/p90/p99/p999/min/max, so the box is an *inter-percentile* box (p50–p99, p90 line),
  clearly labeled — a true Tukey/violin distribution is not reconstructable.
- **Hot-reload (F18) measures non-disruption, not damage.** Across the saturation reloads the
  gateway drops 0 frames; the figure shows throughput retained vs a matched matrix baseline,
  and an integrity ledger of (lost / integrity / boundary) counts. Sub-saturation reloads are
  rate-limited by design and excluded from the retention view.
- **Session resumption (F17 caption).** `resumed_fraction` is recorded but is 0 in these runs
  (all full handshakes); F17 reports the cold-handshake cost and flags the absence of resumption
  rather than implying it occurred.
- **Concurrency scaling (F15)** is plotted per transport only where the matrix sweeps
  connections — SHM/UDS to 16c, TCP to 1024c; UDP/TPROXY run a single connection by design. The
  ≥64c gaps for SHM/UDS (they do not opt into the scalability tier) are the coverage wall,
  reported in the caption, not hidden. Each point is tagged with its bottleneck class, since on a
  single loopback host the flat curves are the serial per-connection data plane, not a gateway
  fan-out defect.

## Extending

- A new figure is a module under `seshat_viz/figures/` exposing `FIG_ID`, `NAME`, `TITLE`,
  and `make(bundle, saver)`; add it to `REGISTRY` in `figures/__init__.py` (registry order is
  the build/display order — keep it in the evaluation narrative).
- Guard on data with `bundle.has(*cols)` and bail via `saver.record_skip(FIG_ID, NAME, reason)`
  so a missing-data figure is skipped, never a crash.
- Colors/markers/fonts live in `seshat_viz/theme.py` (one stable color per transport and
  per protocol, used by every figure). Derived/relationship metrics live in
  `seshat_viz/derive.py`; name-derived grouping factors live in `loader._enrich_factors`.
- End every figure with `theme.add_takeaway(fig, "…")` — a one-line, ideally data-driven
  conclusion — and `theme.add_provenance(fig, bundle.caption())`.
- For legends that sit outside the axes, use constrained-layout locations
  (`fig.legend(..., loc="outside right upper")`); a plain `bbox_to_anchor` outside the axes gets
  clipped at the figure edge because constrained layout won't reserve room for it.

## License

Licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
