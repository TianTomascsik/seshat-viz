# seshat-viz — thesis-grade visualizations of SESHAT benchmark data

`seshat-viz` turns the CSV measurement tree produced by **SESHAT** (the SCG benchmark
harness in `../SCG-SESHAT/`) into a set of publication-quality figures for the Secure
Communication Gateway master's thesis. It reads a results directory and emits **vector PDF
(for LaTeX `\includegraphics`) + PNG (preview)** figures that go well beyond plain
bar/line charts: a throughput–latency Pareto map, payload-size scaling small-multiples,
protocol×size heatmaps, latency-tail CCDFs + box-and-whisker, saturation knees, a
closed-loop-RTT / coordinated-omission-inflation map, hot-reload robustness, concurrency
scaling, jitter/determinism, a consolidated resource-cost composite, cipher-suite cost, and a
parallel-coordinates multi-objective view.

Every figure also carries a one-line **conclusion banner** (`▸ …`) stating the point it makes,
so each can stand alone in a thesis chapter.

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

Dependencies are declared in `pyproject.toml`; `requirements.txt` is a thin pointer at it.

## Usage

```bash
# Newest run under ../SCG-SESHAT/results, all figures -> ./figures/{pdf,png}
python -m seshat_viz

# A specific run directory
python -m seshat_viz ../SCG-SESHAT/results/20260626-104017 --out figures

# Just a couple of figures, PDF only
python -m seshat_viz <run_dir> --only F1,F4,F8 --format pdf

# Thesis embedding: strip the headline / grey footers / red takeaway banner so the
# LaTeX caption carries that text. The suppressed text lands in <out>/captions.txt.
python -m seshat_viz <run_dir> --no-chrome

# Code-coverage mode for F13 (else it renders scenario-execution coverage)
python -m seshat_viz <run_dir> --coverage path/to/coverage.json

# List the figure catalogue and exit
python -m seshat_viz --list
```

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

Listed in build order (the thesis narrative: trade-off landscape → structural cost →
latency/determinism → load & robustness → resource cost → measurement validity).

| ID | File | What it shows | The conclusion it supports |
|----|------|---------------|----------------------------|
| **F1** | `f01_landscape_*` | Throughput vs p99-latency scatter **at a matched single-connection (1c) slice** (so TCP's multi-conn aggregate no longer dominates); color=protocol, marker=transport, size∝Gbps/core, **Pareto frontier**; harness-limited points faded | Which (transport, protocol) configurations dominate *at equal concurrency*, and the price of climbing the security ladder (concurrency scaling is F15) |
| **F2** | `f02_payload_scaling` | Faceted size-scaling curves per transport (throughput + p99 latency vs message size), gateway vs direct path | Where per-message overhead dominates (small msgs) vs link/crypto (large msgs); the saturation size |
| **F3** | `f03_crypto_overhead` | Dumbbell routing→protocol throughput + latency-inflation bars vs the routing baseline | The marginal cost of each security scheme; the kTLS-vs-userspace-TLS gap |
| **F4** | `f04_protocol_size_heatmaps` | Protocol×size **heatmaps** (throughput + p99 latency, **log color**), per transport | Dense at-a-glance map of the whole measured space; hot/cold spots; what was measured |
| **F5** | `f05_transport_radar` | Multi-metric **radar** (throughput, low latency, CPU eff., low CPU, low loss) + absolute bars | TCP vs UDS vs SHM vs UDP holistic profile; the shared-memory IPC advantage |
| **F6** | `f06_gateway_insertion_cost` | **1-gateway (scg-direct) vs 2-gateway (scg-scg)** dumbbells for throughput & p99 latency on matched scenarios | The cost of the *second* gateway hop, isolated from transport & crypto (both `chain`s traverse the gateway; the only no-gateway baseline is the `loopback` family) |
| **F15** | `f15_concurrency_scaling` | Scaling **efficiency** (achieved ÷ ideal-linear %) vs connections + absolute p99 latency, per transport, each point tagged with its bottleneck class | Why SHM/UDS/TCP all fall below linear on a single loopback host: the serial per-connection data plane (host stays <30% busy), not the gateway's fan-out |
| **F7** | `f07_latency_tails` | Latency **CCDF** + per-protocol **box-and-whisker** (box p50–p99 · p90 line · min–max whiskers · ◇ mean) | Tail behaviour that means/medians hide; which protocols have fat tails |
| **F16** | `f16_closed_loop_rtt` | Closed-loop **RTT** vs payload size across every protocol × stream interface (TCP/UDS/SHM/TPROXY) + closed-vs-open-loop **coordinated-omission inflation** | The only honest absolute latency, and how it scales with payload for every protocol on every interface; why open-loop blast tail latency is unusable as an absolute number |
| **F19** | `f19_jitter_determinism` | Per-config **jitter (PDV)** ranking + a throughput-vs-jitter **determinism quadrant** | Delivery predictability — the determinism an ETCS/EuroRadio control loop must tolerate |
| **F8** | `f08_saturation_knee` | Offered-load sweep: goodput + loss% + p99 (dual axis), **knee** & max-loss-free marked, overlay panel | Where each configuration breaks and how gracefully — the headline capacity result |
| **F17** | `f17_connection_setup` | **Connections/sec** (with CI) + per-connection **handshake** p50→p99 dumbbells | Connection-establishment rate and the TLS handshake tax (≈50× a bare TCP accept) |
| **F18** | `f18_hotreload_robustness` | Throughput **retained** heatmap (protocol×concurrency×trigger) + **integrity ledger** | Reconfiguring a saturated gateway is non-disruptive: 0 frames lost, throughput held near steady-state |
| **F9** | `f09_resource_cost_of_security` | Consolidated grid: throughput, Gbps/core, cycles/byte, cache-miss rate, ctxsw/1k-msg, peak RSS/PSS + memory-pressure scatter | The full CPU & memory cost of each security tier in one place; kTLS's kernel-offload payoff (**replaces old F9+F10+F14**) |
| **F20** | `f20_cipher_cost` | AES-128-GCM vs AES-256-GCM vs ChaCha20-Poly1305 across the whole sweep: metric rows × protocol columns, grouped by payload size | The AEAD choice alone — and that the AES-NI advantage over software ChaCha20 is invariant across payload size and TLS version (DTLS/kTLS columns appear once those sweeps run) |
| **F23** | `f23_handshake_cost` | **Asymmetric** handshake cost (sibling of F20's symmetric AEAD), two rows: **server auth** (RSA-2048 vs ECDSA-P256 cert) and **key exchange** (X25519 vs P-256 ECDHE group) — conns/sec + handshake p50→p99 | The cost of each asymmetric primitive alone — ECDSA-P256 sustains ~1.5× RSA-2048's connection rate, while the KEX group barely moves it (X25519 ≈ P-256). Needs the handshake sweep run |
| **F21** | `f21_parallel_coordinates` | **Parallel-coordinates**: every (transport, protocol) across normalized axes (throughput, Gbps/core, latency, jitter, cycles/byte, RSS, loss), up = better | The whole multi-objective trade-off at once; a dip marks exactly where a config pays |
| **F11** | `f11_measurement_validity` | Per-scenario headroom bars, harness-limited flag, bottleneck/DUT class | Proves NFR-PERF (harness isn't the limit) so the other figures are trustworthy |
| **F12** | `f12_system_metrics_timeline` | `/proc` timeseries (CPU% + RSS + ctxsw) per **transport**, comparable slice — routing · 1 gateway · matched size/conn, only the transport varies | The transport mechanism's own CPU/memory/scheduling cost over time: SHM vs UDS vs TCP vs TPROXY (vs UDP) at one identical workload |
| **F22** | `f22_protocol_metrics_timeline` | Same `/proc` timeseries per **protection mode** at a fixed transport (routing → TLS → kTLS → mTLS), only the protocol varies | The crypto cost over time — userspace TLS vs kernel-offloaded kTLS render as adjacent panels |
| **F13** | `f13_coverage` | **Code coverage** (line %) per workspace/crate vs the ≥80% target, when a `coverage.json` artifact is present; otherwise falls back to the executed-vs-skipped scenario split | How well the code itself is tested (or, in fallback, what fraction of the suite ran) |

## Figure reference — how to read each figure

Generated figure sets (`figures/`, `figures-nochrome/`, `figures-thesis/`) are **not
committed** to this repository — they are regenerated from a SESHAT results directory with
`python -m seshat_viz <run_dir> --out figures`. The reference set described below was
rendered from SESHAT run
**`20260702-191908`** (inside `results/20260702-211814-procfs`, `configs/full_matrix.json`,
882 executed / 36 skipped of 918 planned) on **the benchmark host** (AMD Ryzen 9 5950X,
16C/32T, kernel 7.0.9-arch2-1, governor `performance`, 32 GiB), `procfs` metrics backend
(no `perf` hardware counters this run).

> **⚠ The per-figure prose below is stale — quote figures ONLY from the regenerated PNGs and
> `figures/captions.txt` / `figures-nochrome/captions.txt`, never from this section.** It was
> written against the older `20260701-063358` run and predates BOTH figure audits
> (2026-07-03: 78 findings, fixed; 2026-07-07: 99 findings against the
> `20260704-231345` nightly, all viz-side fixes applied + full regeneration the same
> day). Load-bearing 2026-07-07 corrections the old prose
> (and the 07-03 numbers) get wrong: the blast pools now exclude the
> `paced_*`/`shmzc_*`/`handshake_*` families (F1's old 60 Gbps SHM / 12.5 µs crowns were
> leaked microbenchmark rows; honest crowns are UDS routing ~40 Gbps / 167 µs); F13 uses the
> **plan denominator** (2696/2969 = 91 % executed, 132 unaccounted — not 95 %, not 96 %);
> F15's best series reaches **25 % of ideal-linear at the shared 16c point** and host-busy
> climbs 11 %→91 % (the "host stays <30 % busy" claim was false); F8: **gateway routing-UDP
> also sheds by loss** (~36 %), not only DTLS; F5/F21 rankings are now computed on measured
> axes only (no fabricated jitter fill; F21 winner kTLS 1.2 · UDS at 57 % min, tie disclosed);
> F2's honest RTT band is the computed **7.7–326 µs** (not "11–60 µs"); F18 counts **96 reload
> events** (one per scenario). `figures/manifest.json` now records per-figure run provenance —
> check `"coherent": true` before publishing any figure set.

Each figure below is documented with the same four-part structure — **What it shows** (panels,
axes, encodings), **How to read it** (what a mark/line/reference means and which direction is
better), **How to interpret it** (the conclusion it supports, with this run's actual numbers),
and **Caveats** (the measurement-validity limits that apply to that specific figure). The
order matches the build/display order (`--list`): trade-off landscape → structural cost →
latency/determinism → load & robustness → resource cost → measurement validity.

Read the whole-run measurement caveats once and they apply throughout: **416/800 throughput
rows are harness-limited** on this 32-thread loopback host (faded in F1, flagged in F11);
**throughput is wire-bytes**, not goodput; **open-loop latency** (F1/F3/F7) is
coordinated-omission-uncorrected and must be read as *relative* ranking, with F16's ping-pong
RTT the only honest absolute; **UDP/DTLS loss is real** and shown, not hidden; and the
single-connection UDP `raw_tls`/`ale_tls` rows carry **degenerate CIs** (ci95 ≥ mean) so their
point estimates are lower bounds only.

### F1 · `f01_landscape_throughput_latency`
> _Routing TCP anchors the Pareto frontier at 83.8 Gbps @ 72.1 µs p99; every TLS/kTLS variant clusters near 35 Gbps, so climbing the security ladder costs latency, not a throughput cliff._

![F1](figures/f01_landscape_throughput_latency.png)

**What it shows.** A single scatter of every measured configuration: p99 latency on the x-axis (µs, log, **inverted** so lower/better is toward the right) versus mean throughput on the y-axis (Gbps, higher is better ↑). Color encodes protocol (grey routing through the TLS/kTLS/mTLS blues-greens to the DTLS oranges/reds), marker shape encodes transport (triangle TCP, diamond UDP, square UDS, circles SHM/TPROXY), and marker size scales with CPU efficiency (Gbps/core). A dashed red Pareto frontier connects the non-dominated points.

**How to read it.** Each marker is one scenario; up-and-right is strictly better (more throughput, less latency). Bold dark-ringed markers are gateway-bound (trustworthy DUT measurements); faded white-ringed markers are harness-limited. Thin grey whiskers are CI95. The red frontier traces the best achievable throughput/latency trade-off; the two annotations flag the extremes.

**How to interpret it (this run).** The frontier is anchored by TCP · routing at 83.8 Gbps @ 72.1 µs and descends through the TLS band, which saturates near 35–36 Gbps (kTLS 1.3 35.71, TLS 1.3 36.29, mTLS 1.2 36.48). DTLS/UDP sits far lower (DTLS 1.2 peak 8.85 Gbps), reflecting datagram single-connection design. The gateway's insertion cost is effectively nil — matched scg/direct median ratio 0.998 with no CI-disjoint regression — so encryption tier, not the proxy, sets the position.

**Caveats.** 416/800 throughput rows are harness-limited (faded) on this 32-thread loopback host, so absolute ceilings understate real-link capacity; throughput is wire-bytes. Latency is open-loop p99 and relative-only (coordinated omission). Single-conn UDP raw_tls/ale_tls rows have degenerate CIs (ci95 ≥ mean) — treat as lower bounds. procfs-only run: no perf counters, so no cycles/byte encoding here.

### F2 · `f02_payload_scaling`
> _Bigger messages amortize per-message syscall cost: throughput climbs from ~0.2 Gbps at 64 B to ~12-16 Gbps at 64 KB, and inserting the gateway costs essentially nothing (median scg/direct ratio 0.998)._

![F2](figures/f02_payload_scaling.png)

**What it shows.** A 2-row grid of small-multiples, one column per transport (SHM, UDS, TCP, UDP, TPROXY). Top row plots throughput (Gbps, linear y) against message size (64 B-64 KB, log x); bottom row plots p99 latency (µs, log y) against the same log-x size. Each line is one protocol (color: routing grey, TLS 1.3, kTLS 1.3, mTLS, DTLS 1.2, integrity, etc.); solid = through-gateway (scg), dashed = direct baseline, with CI95 throughput bands.

**How to read it.** A point is the mean over single-connection rows at that size. Up-and-right is better for throughput; down is better for latency. Solid-vs-dashed separation is the gateway's insertion cost; where solid and dashed overlap the gateway is transparent. Shaded bands are CI95 — wide bands (notably UDP) mean unreliable point estimates.

**How to interpret it (this run).** On the benchmark host (Ryzen 9 5950X, procfs backend), throughput scales monotonically as syscall overhead amortizes: TCP protocols climb from near-0 at 64 B to ~11-13 Gbps at 64 KB, while p99 latency falls from hundreds of ms to ~10 ms. Userspace crypto is the ceiling — DTLS 1.2 peaks 8.85 Gbps versus kTLS 1.3's ~35 Gbps at 4 KB. Solid and dashed curves nearly coincide, confirming the gateway's ~0% insertion cost (51/55 matched pairs ≥0.95). SHM tops out ~13-16 Gbps and TPROXY reaches ~12.6 Gbps at 64 KB.

**Caveats.** 416/800 throughput rows are harness-limited on loopback, so many high points are I/O-bound not DUT-bound. UDP's wide CI95 band reflects the degenerate single-conn raw/ALE-TLS CIs (ci95≥mean) — treat those as lower bounds. Datagram/UDS/SHM are single-connection by design; throughput is wire-bytes.

### F3 · `f03_crypto_overhead`
> _On the TCP · 16KB · direct cell, every crypto scheme roughly halves routing's 50.01 Gbps: TLS 1.3 and kTLS 1.3 retain ~53% (26.64 Gbps), TLS 1.2 (integrity) only 40% (20.03 Gbps)._

![F3](figures/f03_crypto_overhead.png)

**What it shows.** Two panels for the single (transport, size, chain) cell with the strongest routing baseline and most encrypted protocols — here TCP · 16KB · direct. Left ("Throughput vs routing") is a dumbbell per protocol: a grey dot at the routing baseline (50.01 Gbps, dotted line) joined by a protocol-colored line to that scheme's throughput (Gbps), labelled with absolute value and % of routing retained. Right ("Latency inflation vs routing") shows p99-latency multiplier (protocol ÷ routing) as colored horizontal bars on a log-x axis, with a dotted 1× reference.

**How to read it.** Left: a longer dumbbell means a bigger throughput drop; further right (closer to the baseline) is better. Right: bars past 1× are slower than routing, left of 1× are faster; the log axis compresses ratios. Rows are stacked down the security ladder, from TLS 1.2 at top to TLS 1.2 (integrity) at bottom.

**How to interpret it (this run).** Against routing's 50.01 Gbps baseline, TLS 1.3 and kTLS 1.3 both land at 26.64 Gbps (53% retained); TLS 1.2 at 26.17 (52%), kTLS 1.2 at 26.07 (52%), mTLS 1.2/1.3 at 25.64/25.17 (51/50%). TLS 1.2 (integrity) is worst at 20.03 Gbps (40%) — its extra ALEPKT integrity framing is userspace-crypto-bound. kTLS 1.3 matches userspace TLS 1.3 exactly here, so the kernel-offload gap is negligible at 16KB. The latency panel shows all multipliers near 1× (0.8-1.0×) except TLS 1.2 at 2.1×. Separately, the gateway's own insertion cost is ~0% (matched scg/direct median ratio 0.998).

**Caveats.** The latency panel is open-loop blast (coordinated-omission-uncorrected) — use for relative ranking only, not absolute microseconds. Throughput is wire-bytes; 416/800 rows this run are harness-limited, and this cell's high routing baseline still sits below the loopback ceiling. This is a procfs-only run with no perf hardware counters, so no cycles/byte panel exists to attribute the crypto cost.

### F4 · `f04_protocol_size_heatmaps`
> _kTLS 1.3 and userspace TLS 1.3 both peak near 36 Gbit/s at 4 KB on this 5950X, while DTLS 1.2 tops out at ~8.9 Gbit/s at 9000 B — the whole protocol × size space at a glance._

![F4](figures/f04_protocol_size_heatmaps.png)

**What it shows.** Four annotated heatmaps for the two best-covered transports (TCP, top; UDP, bottom), each split into throughput (Gbps, viridis) and p99 latency (µs, rocket_r). Rows are the security ladder (routing, TLS 1.2/1.3, kTLS 1.2/1.3, mTLS 1.2/1.3, TLS 1.2 integrity for TCP; routing, TLS 1.3, DTLS 1.0/1.2, mDTLS 1.2 for UDP); columns are message size (64 B→64 KB for TCP, 64 B→9000 B for UDP). Cell color is on a log scale; each cell is annotated with its numeric value.

**How to read it.** Each cell is one protocol×size point (scg gateway path preferred over direct). On throughput panels brighter/yellow = faster (better); on latency panels lighter = lower µs (better). Blank white cells were not measured. Reading a row across shows size scaling; reading a column down shows the cost of climbing the security ladder.

**How to interpret it (this run).** kTLS 1.3, TLS 1.3, and mTLS all converge at ~32–36 Gbit/s at 4 KB, showing kTLS's kernel offload matches userspace TLS at large messages while integrity-mode caps at 22–29 Gbit/s. Routing peaks highest (63.85 Gbit/s at 64 KB), confirming the crypto tax. DTLS 1.2 datagram throughput climbs to 8.81 Gbit/s at 9000 B — datagram-bound, an order of magnitude below TCP. TCP p99 latencies balloon to 10^5–10^6 µs at 16–64 KB under open-loop saturation, while UDP p99 stays in the 6–240 ms band.

**Caveats.** Throughput is wire-bytes; 416/800 rows are harness-limited on loopback, so many bright TCP cells reflect the harness ceiling, not the gateway (median scg/direct insertion ratio 0.998). The huge TCP p99 values are open-loop coordinated-omission artifacts — relative only; honest closed-loop RTT is ~12.5 µs (see F-latency figures). Datagram/UDS/SHM rows are single-connection by design, and single-conn UDP raw/ALE-TLS rows have degenerate CIs. This procfs-only run has no perf counters, so no cycles/byte overlay is possible.

### F5 · `f05_transport_radar`
> _At routing · 1KB, TCP loopback sustains ~4.2 Gbps at sub-millisecond p99, with the gateway adding essentially no cost (median scg/direct throughput ratio 0.998)._

![F5](figures/f05_transport_radar.png)

**What it shows.** Two panels for a matched workload (routing, protocol `none`, at the payload size shared by the most transports — 1KB here). Left: a normalized radar with up to six axes — throughput, low latency, low jitter, CPU efficiency, low CPU use, low loss — where each transport (SHM green, UDS light blue, TCP dark blue, TPROXY grey) is min-max scaled so the outer rim is best. Right: absolute throughput bars in Gbps (left axis, with CI95 caps) overlaid with a dashed p99-latency line in µs on a log right axis.

**How to read it.** On the radar, a vertex pushed outward is better on that axis (loss/latency/jitter/CPU are inverted so "outer = lower"); a larger enclosed area is a broadly stronger transport. On the bars, taller is more throughput and a lower dashed point is lower latency — the two need not agree.

**How to interpret it (this run).** SHM leads throughput (~8.2 Gbps) and CPU efficiency but not latency; TCP delivers ~4.2 Gbps at sub-millisecond p99, while UDS sits at ~2.2 Gbps and TPROXY lowest (~1.3 Gbps) but with the best (lowest, ~100 µs) p99. Loopback routing is largely harness-bound rather than gateway-bound, consistent with the gateway's near-zero insertion cost (median scg/direct ratio 0.998, 51/55 matched pairs ≥0.95, no CI-disjoint regression).

**Caveats.** Throughput is wire-bytes and most routing rows here are harness-limited (416/800), so these bars reflect the loopback I/O ceiling, not a gateway limit. This is a procfs-only run with no perf hardware counters, so `gbps_per_core` CPU-efficiency is procfs-derived; datagram/UDS/SHM are single-connection by design, so their radar and CI figures should be read as such.

### F6 · `f06_gateway_insertion_cost`
> _On the benchmark host, the gateway's own insertion cost is negligible: the median through-SCG/direct throughput ratio is 0.998 (51/55 matched pairs ≥0.95), with no CI-disjoint regression._

![F6](figures/f06_gateway_insertion_cost.png)

**What it shows.** Two dumbbell panels pairing scenarios that share (transport, protocol, message size) and differ only in whether traffic crosses the gateway. Left panel: throughput (Gbps, linear x). Right panel: p99 latency (µs, log x). Each row is one pair; the hollow marker is the direct baseline, the filled marker is through-SCG, joined by a line colored by transport (routing, TLS 1.3, kTLS 1.3, mTLS, DTLS 1.2, integrity). Rows are sorted by throughput delta, largest gateway penalty at top; the left panel annotates each pair's signed percent change.

**How to read it.** A short line means the gateway is cheap; a long line pulling the filled dot left (throughput) or right (latency) means real insertion cost. Rightward is better for throughput, leftward is better for latency. Percent labels near the closer endpoint quantify the throughput gap per pair.

**How to interpret it (this run).** Most pairs sit within ±5%, matching the run-facts median ratio of 0.998 and 51/55 ≥0.95. The visible large penalties are the small-message routing/UDS/SHM and TPROXY rows at the bottom (e.g. −63% and −66% on TCP · TLS 1.3 · 1KB, SHM/UDS routing −38% to −49%), where per-syscall overhead dominates and the harness, not the SCG, bounds the direct baseline. Latency shifts are modest and mostly track the throughput story.

**Caveats.** 416/800 throughput rows are harness-limited, so many direct baselines are I/O-bound rather than true ceilings, inflating apparent gateway penalties on small-message rows. Latency is open-loop p99 (relative, coordinate-omission-affected), throughput is wire-bytes. Single-conn UDP raw/ALE-TLS rows have degenerate CIs (use as lower bounds); datagram/UDS/SHM pairs are single-connection by design.

### F15 · `f15_concurrency_scaling`
> _SHM/UDS and TCP-crypto all fall far below ideal-linear scaling — but the host stays <30% busy the whole time: the ceiling is the serial single-thread-per-connection data plane on a no-NIC loopback host, not the gateway's fan-out. Hollow points are load-generator bound, filled are gateway-relay bound._

![F15](figures/f15_concurrency_scaling.png)

**What it shows.** One column per transport that actually sweeps connections (SHM, UDS, TCP), each with two stacked panels. Top: scaling *efficiency* (achieved throughput ÷ ideal-linear, %) versus connection count on a **per-transport** log2 x-axis (SHM/UDS to 16c, TCP to 1024c), one colored line per protocol against a 100% "ideal (linear)" guide. Each point is drawn **hollow** when its SESHAT bottleneck is the load generator / host (`harness-io` / `host-saturated` / flagged `harness_limited`) and **filled** when it is the gateway path (`scg` / `scg-cpu`). Bottom: absolute p99 latency (µs, log). Color encodes protocol per the shared legend. UDP/TPROXY are single-connection by design and so absent.

**How to read it.** Each point is one connection-count measurement. Top panel: 100% is perfect linear scaling; falling toward 0 means added connections buy no proportional throughput. A hollow point means the load generator / host — not the gateway — was the ceiling there; a filled point means a gateway relay thread was. The percentage annotation at the rightmost point is efficiency at that transport's peak measured concurrency. Bottom panel: lower and flatter is better.

**How to interpret it (this run).** No configuration tracks ideal-linear — but the box stays 70–90% idle throughout (`host_busy_frac_p95` 0.09–0.30) while a single relay thread pegs a core (`cpu_hot_thread_pct_p95` 85–100%). The gateway data plane is deliberately serial per connection and loopback has no NIC to bypass, so extra connections add threads but no aggregate throughput: routing already sits at the single-stream loopback ceiling (SHM ~28→37, UDS flat ~38 Gbit/s across 1–16c), and encrypted paths actually *decline* (UDS kTLS 14.5→4.8 Gbit/s, per-thread CPU 147%→44%) as they stall on per-connection poll/futex wakeups. TCP crypto does fan out (~10→39 Gbit/s at 1→16c) until the host-CPU wall at 64c. This is a serial-data-plane + single-host property, not a gateway fan-out defect (median scg/direct ratio 0.998) — the "add a thread per interface" fix is *already* implemented (the gateway spawns a relay thread per endpoint); it needs a bandwidth-bound / real-NIC tier to show gains.

**Caveats.** Open-loop p99 here carries coordinated-omission inflation, so treat latency as relative, not absolute (closed-loop RTT was ~12.8 µs). Throughput is wire-bytes; the hollow points are harness-limited, so those efficiencies reflect the load generator, not the gateway ceiling. SHM/UDS stop at 16c (nightly ladder, no scalability-tier opt-in); UDP/TPROXY are single-connection by design.

### F7 · `f07_latency_tails`
> _Under open-loop blast, routing holds tens-of-µs latency while every crypto path sits at 100 ms–1 s p50 — TLS 1.2 (integrity) owns the worst tail (mean ≈ 600 ms), but this is harness queueing, not gateway overhead._

![F7](figures/f07_latency_tails.png)

**What it shows.** Two stacked panels for the TCP · 64B cell. Top: a complementary-CDF, `P(latency > x)` (log-y) versus latency (µs, log-x), one marked line per protocol — routing, TLS 1.2, TLS 1.3, kTLS 1.2, kTLS 1.3, mTLS 1.2, mTLS 1.3, TLS 1.2 (integrity) — reconstructed from the runs.csv percentile columns. Bottom: horizontal per-protocol boxes where the box spans p50→p99, the interior line is p90, whiskers run min→max, the diamond flier is p999, and the hollow ◇ is the arithmetic mean.

**How to read it.** In the CCDF, further right / higher = a worse (heavier) tail; a curve hugging the left is best. In the box panel, a box farther right means slower, a wide box means high tail spread, and a ◇ sitting well right of the box signals a heavy, mean-inflating tail.

**How to interpret it (this run).** Routing is isolated near the left at tens of µs (consistent with the 48.6 µs one-way `lat_scg_routing_tcp_1KB` mean), while all crypto protocols cluster at 100 ms–1 s. TLS 1.2 (integrity) has the rightmost box and largest mean (~600 ms), followed by mTLS 1.2; kTLS 1.2/1.3 and TLS 1.3 tails are somewhat tighter. The gateway itself is not the cause — matched scg/direct insertion cost is a 0.998 median ratio with no CI-disjoint regression — so the spread is open-loop backlog, not per-message crypto cost.

**Caveats.** Latency here is open-loop, coordinated-omission-uncorrected blast: the banner warns to compare tail shapes, not absolute µs (honest absolute RTTs live in the closed-loop ping-pong, e.g. 12.5 µs TCP p50). Boxes are reconstructed from exported percentiles, not raw samples, so they are not true Tukey distributions. This is a procfs run with no perf counters, but that does not affect this latency figure.

### F16 · `f16_closed_loop_rtt`
> _Open-loop blast inflates p99 by up to 1703× — closed-loop RTT (11.1 µs–57.6 µs) is the only coordinated-omission-free absolute latency on the benchmark host._

![F16](figures/f16_closed_loop_rtt.png)

**What it shows.** The full closed-loop RTT grid. **Top:** one facet per stream interface (SHM, UDS, TCP, TPROXY); within each, closed-loop ping-pong RTT p99 (µs, linear y) versus payload size (64 B–64 KB, log₂ x), one line per security protocol, so every protocol on every interface is visible and its payload-size scaling is explicit. **Bottom:** "Coordinated-omission inflation": for each protocol at a fixed representative payload (1 KB), closed-loop p99 (●) versus the matched open-loop blast p99 (✕) on a log µs x-axis, with the inflation ratio annotated in red at the geometric midpoint. When a run has no `matrix_lat_*` grid the figure falls back to the legacy per-profile p50→p99 dumbbell.

**How to read it.** Top, lower is better — smaller RTT; the slope of each line is how that protocol's latency grows with payload. Bottom, the horizontal gap between ● and ✕ is the coordinated-omission blowup; the red "N×" is blast_p99 ÷ closed_p99 (a one-way blast p99 over a round-trip RTT, so a conservative lower bound). Only the closed-loop points are trustworthy absolute numbers.

**Coverage.** The grid covers the stream protocols (routing, TLS 1.2/1.3, kTLS 1.2/1.3, mTLS 1.2/1.3, kmTLS 1.2/1.3, TLS 1.2 integrity) over TCP/UDS/SHM/TPROXY at all six payload sizes. Datagram interfaces (UDP: DTLS, ALE, raw UDP-over-TLS, UDP routing) are absent by construction — the one-way datagram gateway path cannot echo a request, so it has no closed-loop RTT (its per-message determinism signal is F4's jitter/PDV panel). `integrity`+TLS 1.3 is absent too: TLS 1.3 has no integrity-only (NULL-cipher) suite, so that combination does not establish.

**Caveats.** The ✕ values come from the open-loop matrix and are coordinated-omission-inflated by design (relative only). Blast p99 is matched per (protocol, interface) at the representative payload; where a size-exact blast row is unavailable it falls back to a (protocol, interface) then protocol-only mean.

### F19 · `f19_jitter_determinism`
> _Plaintext routing is the most deterministic (jitter from 0.3 µs); encryption adds delay variation — best encrypted ≈ 3 µs (UDP · TLS 1.3), worst 1483 µs (TCP · TLS 1.2)._

![F19](figures/f19_jitter_determinism.png)

**What it shows.** Two panels of packet-delay-variation (PDV), defined as the mean absolute difference of consecutive one-way latencies — not stddev. Left ("Per-configuration jitter"): a horizontal bar per transport·protocol pair, x-axis jitter in µs, sorted ascending (lowest/most deterministic on top), bars colored by protocol and value-labelled. Right ("Determinism quadrant"): a scatter of throughput (Gbps, x) against jitter (µs, y); marker shape encodes transport (circle SHM, square UDS, triangle TCP, diamond UDP), color encodes protocol.

**How to read it.** In the quadrant, right is faster and down is steadier, so the ideal corner is bottom-right; the dashed horizontal line is the median jitter ("more deterministic ↓" below it). Shorter left-panel bars mean tighter, more predictable delivery — the property an ETCS/EuroRadio control loop must tolerate.

**How to interpret it (this run).** On the benchmark host (Ryzen 9 5950X), UDP · routing is essentially jitter-free at 0.3 µs, and UDP · TLS 1.3 is the steadiest encrypted path at 2.9 µs, with DTLS 1.2/mDTLS 1.2 at 9.6 µs. Datagram and shared-memory transports dominate the low-jitter band (SHM/UDS routing ≈14–17 µs); TCP TLS variants cluster at 116–132 µs despite peaking ~36 Gbps. The two outliers — TCP · routing (527 µs) and TCP · TLS 1.2 (1483 µs) — are high-throughput but poorly deterministic, reflecting loopback TCP coalescing rather than crypto cost.

**Caveats.** PDV here is derived from open-loop one-way latencies (subject to coordinated omission — treat as relative, not absolute); honest closed-loop RTT was 10.7–12.5 µs. 416/800 throughput rows were harness-limited on loopback, so the quadrant's x-position for those points is a floor, not a ceiling. This is a procfs run with no perf counters, but this figure uses neither.

### F8 · `f08_saturation_knee`
> _SCG DTLS 1.2 saturates loss-free only to ~1.50 Gbps and then sheds 30-75% of offered load past its ~1.75 Gbps knee, whereas raw UDP loopback holds the highest clean plateau at ~2.22 Gbps._

![F8](figures/f08_saturation_knee.png)

**What it shows.** Five panels, one per saturation sweep plus an overlay. Each per-sweep panel is dual-axis: offered load (Gbps, x) versus achieved goodput (Gbps, left y, colored line+markers) and, on a shared log right y-axis, loss % (red dashed squares) and p99 latency in µs (grey dash-dot triangles). The four sweeps are `scg dtls12 udp 1KB`, `scg routing tcp 1KB`, `tcp loopback 1KB`, and `udp loopback 1KB`; the final "All sweeps compared" panel overlays their goodput curves.

**How to read it.** The dotted grey diagonal is ideal (offered=achieved); a point falling below it means the config cannot absorb the offered rate. The green dotted horizontal is the max loss-free rate (≤1% loss); the red circle marks the knee (peak achieved goodput). Higher/further-right goodput is better; a loss% or p99 curve rocketing upward marks graceless collapse.

**How to interpret it (this run).** DTLS 1.2 tracks the diagonal to 1.5 Gbps, keeps loss ≈0 up to `max loss-free ≈ 1.50 Gbps`, then its knee lands at ~1.75 Gbps as loss climbs into the 30-75% band. Raw UDP loopback holds `≈2.22 Gbps` loss-free, TCP loopback `≈1.70 Gbps`, and SCG routing TCP `≈2.07 Gbps`. So the gateway's crypto/datagram path costs only a modest cap versus raw transport, consistent with the median scg/direct insertion ratio of 0.998.

**Caveats.** These are single-connection-by-design loopback datagram/TCP sweeps, so plateaus reflect one-flow harness limits, not the gateway's ~35 Gbps multicore ceiling; 416/800 rows this run are harness-limited. Throughput is wire-bytes and the p99 latency is open-loop (coordinated-omission), so treat it as relative shape, not absolute µs.

### F17 · `f17_connection_setup`
> _A TLS 1.3 handshake costs ~53× a bare TCP accept (18.1 µs → 1.03 ms at p50) — the price of the first byte of a secure session._

![F17](figures/f17_connection_setup.png)

**What it shows.** Two panels over four connrate rows (routing and TLS 1.3, each at 1 and 4 client threads). Left, "Establishment rate": horizontal bars of connections/second (higher = better) with 95% CI whiskers, colored by protocol (grey = routing, blue = TLS 1.3). Right, "Per-connection handshake cost": a dumbbell on a log-µs axis where the hollow marker is the p50 handshake latency and the filled marker the p99, joined by a colored bar.

**How to read it.** Each left bar is one protocol × thread-count; a longer bar means more new secure sessions per second. On the right, points further left are faster handshakes; the hollow-to-filled span is the p50→p99 tail. Note the log x-axis: routing sits near ~20 µs, TLS 1.3 near ~1 ms — roughly two decades apart.

**How to interpret it (this run).** On the benchmark host (Ryzen 9 5950X), bare TCP accept runs 3,753 conn/s single-thread (18.1 µs p50 / 25.8 µs p99) and 14,044 conn/s at 4 threads. Full TLS 1.3 through the gateway drops to 988 conn/s single-thread (1,025 µs p50 / 1,063 µs p99) and 4,025 conn/s at 4 threads (1,036 µs / 1,166 µs). The ~53× p50 tax is the asymmetric key exchange and certificate work; it parallelizes well (~4× from 1→4 threads) but each session still pays ~1 ms before its first secure byte.

**Caveats.** `resumed_fraction = 0`: every handshake here is full, with no session resumption, so this is the worst-case cost. Handshake latencies are per-connection setup times, not steady-state data-plane RTT (F17 is orthogonal to the procfs-only throughput rows). No perf hardware counters were captured this run, so no cycles/handshake breakdown is available.

### F18 · `f18_hotreload_robustness`
> _Reconfiguring a saturated gateway is non-disruptive: 0 frames lost across 96 live reloads, throughput held at a median 95% of steady-state._

![F18](figures/f18_hotreload_robustness.png)

**What it shows.** Two panels. Left is a heatmap of *throughput retained versus the matched steady-state baseline* (%), rows = protocol (routing, TLS 1.2/1.3, kTLS 1.2/1.3, mTLS 1.2/1.3, TLS 1.2 integrity), columns = concurrency × reload trigger (1c/4c/16c/64c crossed with add / rm / reject). Cell color is the RdYlGn scale centered on 100% (vmin 80, vmax 110): green ≈ undisturbed, red ≈ throughput dip during reload. Right is the "integrity ledger" tallying reload scenarios, frames lost, integrity failures, boundary violations, and median/worst retained.

**How to read it.** Each cell is one reload scenario's throughput while a live config change flows through a saturated gateway; higher (greener, ≥100) is better. Ledger counts colored green mean zero (good), red means nonzero; retained percentages are green at ≥90%. Values above 100% are steady-state measurement noise, not gains.

**How to interpret it (this run).** On the benchmark host (Ryzen 9 5950X, governor performance), 96 saturation reloads ran with **0 frames lost, 0 integrity failures, 0 boundary violations** — reconfiguration is non-disruptive, not merely fast. Median retained throughput is **95%**, worst **81%** (mTLS 1.2, 64c·rm); saturation-trigger throughput held ~34–36 Gbit/s near the matrix baseline. The reddest cells cluster at 1c·add and 64c·rm/reject, where a single-connection start-up or a high-fanout teardown briefly perturbs the pipe; routing and the TLS 1.2 integrity row stay ≥96–104%, confirming the reload path costs nothing structurally.

**Caveats.** Retention is only computed for saturation reloads (sub-saturation is rate-limited by design and excluded); the baseline is the best matched matrix throughput, so >100% cells reflect run-to-run variance, not real speed-up. Throughput is wire-bytes on loopback where 416/800 rows are harness-io-limited, and this procfs-only run carries no perf hardware counters — no cycles/byte here.

### F9 · `f09_resource_cost_of_security`
> _kTLS 1.3 delivers 2.84 Gbps/core vs userspace TLS 1.3's 2.41 (+18%) — the kernel offload's payoff, visible across CPU and memory._

![F9](figures/f09_resource_cost_of_security.png)

**What it shows.** Three per-protocol bar panels read along the security ladder (routing → TLS 1.2/1.3 → kTLS 1.2/1.3 → mTLS 1.2/1.3 → TLS 1.2 integrity → DTLS 1.0/1.2 → mDTLS 1.2), each bar colored by protocol. Left: mean throughput (Gbps, ↑ better). Center: CPU efficiency, Gbps/core (↑ better). Right: peak memory (MiB, ↓ better) with grouped RSS (blue) and PSS (green) bars. Values are annotated above each bar.

**How to read it.** Taller is better in the first two panels; shorter is better for memory. A bar's Gbps/core is throughput normalized by CPU cores consumed, so it isolates efficiency from raw speed. Comparing adjacent kernel vs userspace TLS bars exposes the kTLS offload gain.

**How to interpret it (this run).** routing tops throughput at 17.73 Gbps and efficiency at 5.07 Gbps/core. kTLS 1.3 reaches 2.84 Gbps/core against userspace TLS 1.3's 2.41 (+18%), and kTLS 1.2's 2.56 vs TLS 1.2's 2.28, confirming the kernel offload's payoff. DTLS 1.0 is the efficiency floor at 0.80 Gbps/core. Memory is modest (RSS ~30–170 MiB) except the TLS 1.2 integrity tier at ~600 MiB RSS/PSS, the outlier resource cost.

**Caveats.** This is a procfs run with no perf hardware counters, so the module's cycles/byte, cache-miss, ctx-switch, and throughput-vs-memory-pressure panels are absent (see the provenance note "cycles/cache/ctxsw need a perf run"). Throughput is wire-bytes, size-matched across protocols, and 416/800 rows were harness-limited, so these are efficiency-ordered comparisons rather than absolute per-tier ceilings.

### F20 · `f20_cipher_cost`
> _The AEAD choice alone: AES-GCM rides AES-NI to ~10.4 Gbps, ChaCha20-Poly1305 (software AEAD) to ~7.2 Gbps — a stable ~44% gap that holds at every payload size and both TLS versions, so the ranking is payload/protocol-independent._

![F20](figures/f20_cipher_cost.png)

**What it shows.** The whole AEAD cipher sweep as a grid, not a single cell: one **column per protocol** (here TLS 1.2 ECDHE-RSA and TLS 1.3, both on TCP) × one **row per metric** (throughput with 95% CI whiskers, Gbps per core, and on a perf run cycles-per-byte — CPU% stands in here). Within each panel the bars are grouped by the swept **message sizes** (1 K / 4 K / 16 K), three bars per group — one per cipher (GCM blue, ChaCha green). The header still names the single best matched cell (TCP · TLS 1.3 · 16 KB) so the headline number has an unambiguous operating point. The grid is the container the DTLS and kTLS cipher sweeps drop into as extra columns once those scenarios run.

**How to read it.** Only the cipher changes *within* a panel, so read the three-bar shape — GCM tall, ChaCha short. Then read *across* the size groups and *across* the protocol columns: the shape not changing is the point — it is the evidence that the AES-NI advantage is independent of payload size and TLS version. Taller is better in the top two rows, shorter in the CPU row.

**How to interpret it (this run).** On the benchmark host's Ryzen 9 5950X (AES-NI present), AES-128-GCM and AES-256-GCM both sit at ~10.3–10.6 Gbps at every size and both versions; ChaCha20-Poly1305 (constant-time software AEAD) holds ~7.0–7.4 Gbps — a **~44% throughput gap that is flat across all six (protocol × size) cells** (TLS 1.3: +50/45/49 % at 1/4/16 K; TLS 1.2: +43/39/41 %). The CPU row mirrors it (GCM ~105–113 %, ChaCha ~135–140 %). So the cipher choice moves throughput by ~a third-to-half and CPU by ~30 points, and — the headline result — that cost is a property of the algorithm, not the payload or the protocol version.

**Caveats.** This is a procfs-only run (no perf hardware counters), so the intended cycles-per-wire-byte row is replaced by CPU utilisation. Throughput is wire-bytes and single-connection; the CI whiskers on the throughput row show the run-to-run variance. Only userspace-TLS-over-TCP cipher rows exist in this run — DTLS (UDP) and kTLS (kernel-offload) cipher columns require their sweeps to be run (both are harness-only additions; the gateway already honors the cipher on those paths).

### F21 · `f21_parallel_coordinates`
> _kTLS 1.3 keeps a high line across CPU-cost axes where userspace TLS and TLS 1.2 (integrity) dip; DTLS/mDTLS plunge only on the loss axis — each config's one weak point is visible at a glance._

![F21](figures/f21_parallel_coordinates.png)

**What it shows.** One polyline per (transport, protocol) configuration crossing six normalized axes: throughput (Gbit/s), Gbps/core, p99 latency (µs), jitter (µs), peak RSS (KiB), and loss (%). Every axis is min-max normalized so up = best, down = worst; the `↑/↓ raw` tick shows the raw-metric direction. Line color encodes protocol (routing, TLS 1.2/1.3, kTLS 1.2/1.3, mTLS 1.2/1.3, TLS 1.2 integrity, DTLS 1.0/1.2, mDTLS 1.2); marker shape encodes transport (SHM circle, UDS square, TCP triangle, UDP diamond).

**How to read it.** Follow a single colored line: a point high on an axis means that config is best-in-class there, low means worst. A flat high line has no weak point; a sharp dive marks exactly the axis where the configuration pays its cost. The cycles/byte axis listed in the source is absent here because this run carries no perf counters.

**How to interpret it (this run).** routing (grey) tops throughput and Gbps/core (76.7 Gbit/s TCP) but its userspace-crypto siblings collapse there. kTLS 1.3 rides mid-to-high on throughput and Gbps/core (35.7 Gbit/s peak) where TLS 1.2 and TLS 1.2 (integrity) sag, confirming the kernel-offload CPU win. DTLS 1.2/mDTLS 1.2 (diamonds) sit worst on throughput yet best on latency/jitter, then crater on loss — the 30–75% DTLS drop past the saturation knee. TLS 1.2 (integrity) is the lone line worst on both peak RSS and loss.

**Caveats.** Axes are size-matched throughput rows only, and 416/800 rows were harness-limited, so several throughput/Gbps-core positions are wire-bytes ceilings not gateway limits. The cycles/byte axis is dropped this procfs-only run (no hardware counters). Single-conn UDP raw/ALE-TLS rows have degenerate CIs, so their exact positions are lower bounds; datagram/UDS/SHM lines are single-connection by design.

### F11 · `f11_measurement_validity`
> _Only 1 of the 36 lowest-headroom scenarios is gateway-bound and trustworthy; the rest sit under the 1.05x credibility line — 416/800 throughput rows this run are harness-limited._

![F11](figures/f11_measurement_validity.png)

**What it shows.** A single horizontal-bar panel of the 36 lowest-headroom (most at-risk) scenarios from this run, sorted ascending. The x-axis is `headroom = harness ceiling ÷ measured throughput` (dimensionless; higher = safer); each y-tick is a scenario labelled `transport·protocol·size·chain` (e.g. `TCP·kTLS 1.3·64B·scg`). Bar color encodes SESHAT's `harness_limited` flag — red = harness-limited, blue = trustworthy/gateway-bound — and each bar tip is annotated with its classified bottleneck (`harness-io`, `scg-cpu`).

**How to read it.** Each bar's length is a scenario's headroom; a bar reaching past the dashed vertical `credibility threshold = 1.05×` line means the harness had ≥5% spare capacity, so the measured throughput reflects the gateway rather than the load generator. Bars left of the line are harness-limited: the load tool, not the DUT, capped the number. Longer (further right) is safer.

**How to interpret it (this run).** On the benchmark host (Ryzen 9 5950X, procfs backend), only `TCP·routing·64KB·direct` renders blue with a `scg-cpu` tag — the sole trustworthy, gateway-bound row in this bottom slice, at ~0.74 headroom yet correctly attributed to gateway CPU. Every other of the 35 bars is red and `harness-io`-bound, and none crosses 1.05×, consistent with the 416/800 rows flagged harness-limited fleet-wide and the bottleneck census {harness-io: 416, scg: 120, scg-cpu: 27}. The SHM small-message rows (`SHM·routing·64B`, `SHM·kTLS 1.3·64B`) bottom out near ~0.03 headroom — the load generator dominates almost entirely there.

**Caveats.** This panel intentionally shows only the 36 lowest-headroom scenarios, so most trustworthy high-headroom rows are off-chart; a red bar means interpret-with-care, not that the gateway is slow. Throughput is wire-bytes; datagram/UDS/SHM rows are single-connection by design, and this is a procfs-only run with no perf hardware counters, so no cycles/byte attribution is available to refine the `harness-io` vs `scg-cpu` split.

### F12 · `f12_system_metrics_timeline`
> _The transport mechanism's own CPU/memory/scheduling cost over time, at one identical routing workload: UDS is the hungriest (~1.8 cores, +8 MiB RSS), SHM ~1.3, TCP ~1.2, TPROXY lowest (~0.6, kernel-side path)._

![F12](figures/f12_system_metrics_timeline.png)

**What it shows.** A **comparable** cross-transport slice: the selection pins protocol (routing/plaintext), a single gateway, connection count and message size to one shared value and varies **only the transport**, so the panels line up the same workload across transports (here 4: `iface_shm_throughput_64KB_1c`, `iface_uds_throughput_64KB_1c`, `matrix_routing_tcp_tcp_64KB_direct_1c`, `matrix_routing_tproxy_tproxy_64KB_direct_1c` — all routing · 1 gateway · 64 KB · 1c). Each panel plots the gateway's /proc timeseries vs. elapsed time (s): a blue solid line for CPU% (100 = one core, summed across gateway PIDs) on the left axis, an orange dashed line for resident RSS (MiB) on the first right axis, and a faint grey filled area for context-switch rate (ctxsw/s) on a second offset right axis. Shaded bands are the steady-state measurement window(s), recovered from the CPU trace itself (the /proc stream carries no phase marker). Its sibling **F22** holds the transport fixed and varies the protection mode instead.

**How to read it.** Because every dimension except the transport is held fixed, any difference **between** panels is the transport mechanism's own — kernel TCP splice vs UDS vs the SHM ring vs TPROXY. Within a panel, follow left-to-right in time: the initial blue rise is warm-up, the shaded band is steady-state, and the orange dashed line staying flat means no memory growth (up on RSS is worse). The grey ctxsw area appears only when the rate exceeds 5/s, so a missing grey axis means negligible scheduling churn, not missing data.

**How to interpret it (this run).** On the benchmark host (Ryzen 9 5950X, governor performance), at the identical routing · 64 KB · 1c workload the transports separate cleanly: **UDS** peaks near ~1.8 cores and grows RSS ~7 → 15 MiB, **SHM** ~1.3 cores with a ~30 MiB working-set RSS during load (its ring buffers — RSS returns to ~7 MiB at each per-rep restart, so end-to-end drift is ~0, and its RC4 busy-poll keeps the steady-state band unbroken between reps), **TCP** ~1.2 cores (RSS flat 7 MiB), and **TPROXY** the lowest at ~0.6 cores (RSS flat 7 MiB), consistent with its kernel-side data path. No panel leaks over the ~25 s window — the flat footprint underpins the run's near-transparent gateway cost (median scg/direct 0.998). When no comparable cross-transport routing slice exists in a run, F12 falls back to one highest-throughput scenario per transport and labels itself **NOT comparable**.

**Caveats.** This is a single-connection slice (the only count every transport shares), so CPU reflects one flow, not saturation — the saturated per-tier cost is F9/F3. This is a procfs-only run (no perf hardware counters), so no cycles/byte panel exists here. The mid-run CPU/RSS dips are phase boundaries (warmup→measure→cooldown), not stalls. UDP joins as a fifth panel once a run includes the plaintext `routing_udp` profile.

### F22 · `f22_protocol_metrics_timeline`
> _Sibling of F12 with the axes swapped: transport fixed (TCP), protection mode varied — routing → TLS → kTLS → mTLS — so the CPU/RSS/ctxsw deltas are the crypto cost over time, with userspace TLS and kernel-offloaded kTLS as adjacent panels._

![F22](figures/f22_protocol_metrics_timeline.png)

**What it shows.** The comparable slice of F12 with transport and protocol roles swapped: it auto-picks the transport carrying the most protection modes with a /proc timeseries (TCP in practice), pins a single gateway plus one shared connection count and message size, and varies **only the protocol** along a curated crypto ladder (routing → userspace TLS → kernel kTLS → mutual-auth / integrity), grouped by TLS version so the userspace-vs-kernel pair renders as adjacent panels. Same three traces as F12 (CPU%, RSS, ctxsw) with the same steady-state shading. Here: `matrix_routing_tcp_tcp_64KB_direct_1c`, `matrix_tls12_tcp_tcp_64KB_direct_1c`, `matrix_ktls12_tcp_tcp_64KB_direct_1c`, `matrix_tls13_tcp_tcp_64KB_direct_1c` — all TCP · 1 gateway · 64 KB · 1c.

**How to read it.** With the transport fixed, differences between panels are the crypto cost. Compare the adjacent **TLS 1.2** and **kTLS 1.2** panels directly: userspace OpenSSL vs the kernel-offloaded data path at the identical AEAD and workload — watch CPU% and the grey context-switch trace, where the offload's different syscall/copy profile shows even when peak cores are similar.

**How to interpret it (this run).** At TCP · 64 KB · 1c the crypto tiers all peak near ~1.2 cores — a single loopback flow at this size is not AES-GCM-saturated, so the peak-core separation is small (kTLS ≈ userspace TLS here; kTLS's offload payoff needs a real NIC or higher load, matching F3/F9). The visible, consistent delta is memory: routing sits at ~7 MiB RSS while every TLS/kTLS tier carries ~12–13 MiB (session state + record buffers). The value of this figure is the *temporal + scheduling* view — the ctxsw shape differs between userspace TLS and kTLS even where the CPU plateaus coincide; the saturated throughput/cycles-per-byte cost lives in F3, F9 and F20.

**Caveats.** Single-connection at one size, so this is the shape of the cost over time, not its saturated magnitude. Procfs-only run, so no cycles/byte here. The ladder shows at most four modes to stay under a full-page stack; which minor versions appear depends on what the pinned (max-coverage) message size carries in the run.

### F13 · `f13_coverage`
> _882/918 scenarios executed, 36 skipped (~4%) — the skips are the high-concurrency coverage wall (≥64c on paths not in the scalability tier), which is exactly why F15's SHM/UDS columns stop at 16c._

![F13](figures/f13_coverage.png)

**What it shows.** Three horizontal-bar panels over scenario counts. Left: a single stacked bar splitting the suite into `executed` (blue, `#2C7FB8`) vs `skipped` (red accent), x-axis in scenarios (0–total). Middle: skips broken down by connection count (`1c`/`16c`/`64c`/`1024c`…), exposing a high-concurrency wall. Right: skips broken down by scenario family (transport, hot-reload, cipher…), exposing whole classes that did not run. Bar color in the two breakdown panels ramps sequentially so the heaviest skip bucket reads as "the deep end."

**How to read it.** In the left bar, more blue and less red is better; the annotated number is the executed count. In the breakdown panels a longer bar means more scenarios skipped for that connection count or family — shorter (or absent) is better. When nothing is skipped, both breakdown panels print "no skipped scenarios" rather than empty axes.

**How to interpret it (this run).** On the benchmark host (AMD Ryzen 9 5950X, 16C/32T, kernel 7.0.9-arch2-1, governor performance), 882 of 918 scenarios executed and 36 were skipped (~96% executed). The middle panel shows the skips concentrate at high connection counts (≥64c): SHM/UDS and the datagram/TPROXY paths do not opt into the 256/1024-connection scalability tier, so those cells are absent by design rather than failed. No whole scenario family is dropped — the DTLS 1.2, kTLS 1.3, mTLS, integrity, and hot-reload matrices all ran — but the high-concurrency wall is real and is exactly why F15's SHM/UDS columns stop at 16c.

**Caveats.** This figure counts scenario execution, not measurement quality: 416/800 throughput rows are harness-limited (bottleneck `harness-io`), and single-conn UDP raw/ALE-TLS rows carry degenerate CIs (ci95 ≥ mean), so "executed" does not imply DUT-bound. The run is procfs-only with no perf hardware counters, but this figure needs none, so it is unaffected.

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

**New SESHAT capabilities these figures consume when present** (all in `../SCG-SESHAT/`):
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

These reflect `../SCG-SESHAT/docs/methodology.md` and are honored in the figures:

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
  single loopback host the flat curves are the serial per-connection data plane (host <30% busy),
  not a gateway fan-out defect.

## Extending

- A new figure is a module under `seshat_viz/figures/` exposing `FIG_ID`, `NAME`, `TITLE`,
  and `make(bundle, saver)`; add it to `REGISTRY` in `figures/__init__.py` (registry order is
  the build/display order — keep it in the thesis narrative).
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
