# Sparse HNSW — Profiling Guide (the experimental ladder, explained)

This document explains the **profiling toolkit** in `sparse/profiling/`: what
question it exists to answer, what the "experimental ladder" is, what every file
and function does, and exactly how to run it on the **msmarco_full** dataset. It
is the measurement companion to [`../CODE_GUIDE.md`](../CODE_GUIDE.md) (which
explains the algorithm being measured) and is written for someone new to
performance profiling — every tool and term is introduced before it's used.

> **This guide documents the apparatus, not the findings.** It deliberately
> contains **no measured numbers**. Every "read-out" below is a *formula* or a
> *rule for interpreting* an experiment — you fill in the values by running it.
> A separate manual, [`README.md`](README.md), is the working lab notebook and
> does quote observed values; this guide is the clean, results-free walkthrough.

---

## Table of contents

1. [Part 1 — The big picture: what the profiling answers](#part-1--the-big-picture)
2. [Part 2 — Profiling concepts & tools glossary](#part-2--profiling-concepts--tools-glossary)
3. [Part 3 — File-by-file, function-by-function](#part-3--file-by-file-function-by-function)
   - [`bench_distance.cpp`](#31-bench_distancecpp--the-single-call-ablation-e3--e4)
   - [`bench_scale.cpp`](#32-bench_scalecpp--thread-scaling-e5)
   - [`profile_search.cpp`](#33-profile_searchcpp--the-instrumented-real-driver-e1e3be6)
   - [`perf_ctl.h`](#34-perf_ctlh--the-perf-gate)
   - [`perf_groups.sh`](#35-perf_groupssh--hardware-counter-groups-e2)
   - [`perf_hotspots.sh`](#36-perf_hotspotssh--where-the-cycles-go-e6b)
   - [`slurm_profile.sh`](#37-slurm_profilesh--run-the-whole-ladder)
   - [`plot_profile.py`](#38-plot_profilepy--turn-the-log-into-figures)
4. [Part 4 — The experimental ladder, rung by rung (with msmarco_full commands)](#part-4--the-experimental-ladder-rung-by-rung)
5. [Part 5 — Running it end-to-end on msmarco_full](#part-5--running-it-end-to-end-on-msmarco_full)
6. [Appendix — the Perlmutter constraints that shaped the design](#appendix--the-perlmutter-constraints-that-shaped-the-design)

---

## Part 1 — The big picture

### The question this toolkit exists to answer

From the algorithm walkthrough we already know one thing: almost all of a query's
time is spent inside `distance()`, streaming document vectors through memory. The
code is **memory-bound** — limited by moving bytes, not by doing arithmetic.

But **"memory-bound" is not a single diagnosis**, and each possible cause needs a
*different* fix. If you guess wrong you optimize the wrong thing. There are four
rival explanations:

| Hypothesis (why memory is the limit) | The fix *if this one is true* |
|---|---|
| DRAM **bandwidth** is saturated — the memory channels are simply full | move **fewer bytes** per vector (compression, fp16, better layout) |
| DRAM **latency** is exposed — we stall waiting for each load, not enough requests in flight | expose **more parallelism** (deeper prefetch, batching, gather pipelines) |
| **TLB / page-walk** overhead — the CPU keeps re-translating addresses | **huge pages** (bigger translation reach) |
| Not memory at all — the merge loop's **serial dependency chain** is the limit | **SIMD** set intersection / galloping search |

The whole point of the profiling ladder is to **decide between these four** with
evidence instead of intuition.

### Why we can't just use two stopwatches

The obvious approach — time the "load" and time the "compute" separately — does
not work for this kernel, and understanding *why* motivates the whole design:

- The sparse distance kernel is a **fused merge loop**: it loads a byte and uses
  it in the same step, and the CPU deliberately overlaps loading the *next* bytes
  with computing on the current ones. There is no separate "load phase" to time —
  "load time" only exists as a *counterfactual* ("how much faster would it be if
  the bytes were already in cache?").
- A single stopwatch read is itself expensive (tens of nanoseconds) and, worse,
  **serializes** the CPU pipeline — it forces all outstanding memory requests to
  finish. Putting one *inside* a sub-microsecond distance call would destroy the
  very overlap we are trying to measure.

So instead of timers-inside-the-kernel, the ladder uses **ablation**: run the
kernel several times, each time *removing one cost while keeping the other*, and
compare. The difference between two variants tells you what the removed cost was.

### What "the experimental ladder" means

The ladder is a **sequence of small, self-contained experiments** — the "rungs" —
each isolating one of the four hypotheses. They are numbered **E1–E6**. You don't
run them blindly top to bottom; there's a natural order:

1. **First, check the premise (E6).** Every later rung reports time *per distance
   call*, which silently assumes distance dominates. E6 proves that assumption
   before you trust anything built on it.
2. **Then decompose a single call (E3, E4)** with the standalone kernel benchmark:
   split it into compute vs memory, and find the cache/TLB cliffs.
3. **Then measure the *real* code path (E1, E2, E3b)** — the actual index, not a
   proxy — for achieved bandwidth, hardware counters, and cold/warm split.
4. **Finally, the scaling verdict (E5):** run on 1→128 threads to see whether
   throughput is capped by latency or by bandwidth — the decision that picks the
   fix.

Here is the whole ladder at a glance (each rung is detailed in Part 4):

| Rung | The question it answers | Tool it uses |
|---|---|---|
| **E6** | Does `distance()` *actually* own the search time? **(check first)** | `sparse_profile … replay` |
| **E6b** | If not distance, *what* is the rest of the time? | `perf_hotspots.sh … batch` |
| **E3** | Split one distance call into compute vs memory-exposed cost | `bench_distance` (5 variants) |
| **E4** | Where are the cache / TLB cliffs as the data grows? | `bench_distance` (working-set sweep) |
| **E1** | What memory bandwidth does the *real* search achieve? | `sparse_profile … batch` |
| **E2** | Which hardware counters confirm/deny each hypothesis? | `perf_groups.sh … batch` |
| **E3b** | Memory share on the *real* traversal (with real locality)? | `sparse_profile … repeat` |
| **E5** | Latency-bound or bandwidth-bound? **(the verdict)** | `bench_scale` |

Two "families" of tool run this ladder: **standalone micro-benchmarks**
(`bench_distance`, `bench_scale`) that hammer just the merge kernel over random
rows, and the **instrumented real driver** (`sparse_profile`) that builds an
actual index and measures the true search path. The micro-benchmarks are clean
and controllable; the real driver is authoritative. Cross-checking the two is how
you gain confidence.

---

## Part 2 — Profiling concepts & tools glossary

Everything measurement-specific you need, explained once. (For C++ basics —
`std::vector`, pointers, the CSR format, the merge kernel itself — see
[`../CODE_GUIDE.md`](../CODE_GUIDE.md); they are not repeated here.)

### Profiling, wall-clock time, and throughput

- **Profiling** = measuring *where* a program spends its time/resources, to find
  what's worth optimizing.
- **Wall-clock time** = a real stopwatch (`std::chrono::steady_clock`). The
  benchmarks wrap a loop in two clock reads and divide by the number of calls to
  get **ns/call** (nanoseconds per distance call), the common unit everything is
  normalized to.
- **Throughput** = work per second, here **GB/s** (bytes of vectors streamed per
  second) or **Mcall/s** (million distance calls per second).

### The memory hierarchy (why any of this matters)

A CPU can't compute on data in main memory (**DRAM**) directly; data travels
through progressively smaller, faster **caches** first:

```
registers  →  L1 (~32 KB, ~4 cycles)  →  L2 (~1 MB)  →  L3 (~32 MB shared)  →  DRAM (~100+ ns)
   fastest  ←──────────────────────────────────────────────────────────────→  slowest
```

Data moves in 64-byte chunks called **cache lines**. If the byte you need is
already in L1, it's nearly free; if it's only in DRAM, you wait ~100+ ns — a
**cache miss**. This kernel misses a lot (documents are large and accessed in a
scattered order), which is why it's memory-bound.

### Bandwidth vs latency (the two ways memory limits you)

These are different limits with different fixes — the ladder's central
distinction. An analogy: a highway between the CPU and DRAM.

- **Bandwidth** = how many lanes the highway has (bytes/second it can carry). You
  hit the **bandwidth** wall when the road is *full* — every lane busy. Fix: send
  fewer cars (fewer bytes: compression, fp16).
- **Latency** = how long one car takes to drive the road, end to end. You hit the
  **latency** wall when the road is *empty but slow* — you sent one car, waited
  for it to arrive, then sent the next. Fix: send **many cars at once** so their
  travel times overlap.

### Memory-level parallelism (MLP)

**MLP** = how many memory requests are "in flight" at the same time. High MLP
hides latency (many cars driving at once). **Prefetching** (`__builtin_prefetch`,
see CODE_GUIDE) and **batching** raise MLP. A key subtlety this project chases:
the merge loop advances its pointers based on the data it just read
(*data-dependent*), which can *lower* MLP because the CPU can't run ahead until
the previous read lands. That's the "fusion penalty" E3 measures.

### Cold vs warm cache

- **Cold** = the data is *not* in cache yet; the access pays full DRAM cost. A
  first, from-scratch access is cold.
- **Warm** = the data was *just* accessed and is still in cache; a repeat access
  is cheap. Running the same query twice back-to-back makes pass 2 warm — the
  difference between the two passes is the memory cost (rung E3b).

### TLB, page walks, and huge pages

The CPU addresses memory in **pages** (4 KB by default). Translating a program
address to a physical one uses a small cache called the **TLB**. A TLB miss
triggers a slow **page walk**. With 4 KB pages the TLB only "covers" a few MB, so
a big scattered dataset causes constant page walks. **Huge pages** (2 MB each)
let one TLB entry cover 512× more memory, often removing the walks. Rung E2's G3
group and rung E4 look for this.

### `perf`: `perf stat` vs `perf record`

`perf` is Linux's profiling tool that reads the CPU's built-in
**hardware performance counters (PMCs)** — special registers that tally low-level
**events** (cache misses, cycles, instructions, TLB misses…).

- **`perf stat -e <events>`** — *counts* events over a region and prints totals.
  Used for the counter **groups** (E2). The CPU (AMD Zen3 here) has only **6**
  counters, and asking for more than ~5 forces perf to time-share them and
  *estimate* — so events are grouped into sets of ≤5 (`perf_groups.sh`).
- **`perf record` + `perf report`** — *samples* the program periodically and
  builds a ranked list of which functions the CPU was in. Used to find hot spots
  (E6b). Sampling has some "skid" (the blamed instruction is *near*, not exactly,
  the guilty one), so it's trusted at function granularity, not instruction.
- `:u` on every event = "count **u**ser-space only," required by this cluster's
  security setting (`perf_event_paranoid=2`).

### Gating counters to a region (`perf_ctl` + FIFOs)

**Problem:** building the index and searching it both run the *same*
`searchLayer` code, and the build dominates wall time. If perf counted the whole
program, build traffic would swamp the search numbers. **Solution:** a *gate*.
Perf can be told "start counting disabled, and only count when I tell you," via a
pair of named pipes (**FIFOs** — a tiny inter-process mailbox). The program sends
`enable` right before the search and `disable` right after. That's what
[`perf_ctl.h`](perf_ctl.h) does, and `perf_groups.sh`/`perf_hotspots.sh` wire up
the FIFOs. If the FIFOs aren't set, the calls are silent no-ops — so a normal run
is unaffected.

### Ablation and record/replay (the two clever measurements)

- **Ablation** = measure a cost by *removing* it and seeing what changes. E3 runs
  the kernel in five regimes that each add or remove one cost; subtracting two
  regimes isolates the cost in between.
- **Record/replay** = run the search once *recording* every distance value it
  computed, then run it again *replaying* those recorded values instead of
  computing them. Because the traversal is fully determined by the distance
  *values*, replay reproduces the identical path, heaps, and visited-set — but
  touches **no document data**. The time difference is exactly `distance()`'s
  cost. This is E6, and it perturbs nothing (unlike a stopwatch inside the loop).

### Pinning: `taskset`, `OMP_PLACES`, NUMA

Measurements are only stable if threads stay put on their cores.

- **`taskset -c 8 <cmd>`** pins a single-threaded program to core 8.
- **`OMP_PLACES=cores OMP_PROC_BIND=spread`** tells OpenMP to pin its threads to
  distinct cores, spread out — used for the multi-threaded rungs.
- **NUMA** (Non-Uniform Memory Access): on these two-socket nodes, memory is
  split into domains and reaching a *remote* domain is slower. The run scripts use
  `numactl --interleave` to spread data across domains; E2's G2 group diagnoses
  whether remote access is hurting. (One caveat matters: never `taskset` the
  `sparse_profile` binary when it builds with many threads — it would jam all the
  build threads onto one core. Details in the Appendix.)

---

## Part 3 — File-by-file, function-by-function

### 3.1 `bench_distance.cpp` — the single-call ablation (E3 + E4)

A standalone program that measures **one distance call in isolation**, over
random document rows, under five access regimes. It only needs `csr_matrix.h` —
it does *not* build an index. It is the workhorse of the ablation (E3) and the
working-set sweep (E4).

**`merge(q, qn, p, pn)`** — an exact copy of the production distance kernel (the
branchless sorted-list merge that returns `1 − dot`), marked `always_inline` so
the compiler pastes it in and the benchmark measures the real inner loop, not a
function call.

**`stream_row(p, pn)`** — reads all the bytes of one row and sums them, with *no*
merge logic. This isolates the **memory** cost: it touches the exact same bytes a
distance call would, but does none of the comparison/branch work, and the loop
has high instruction-level parallelism so memory is the only thing that can
bottleneck it.

**`timeit(name, ncalls, bytes, f)`** — the measurement harness. Opens the perf
gate, reads the clock, runs the lambda `f` (one of the five regimes), reads the
clock again, closes the gate, and prints `ns/call` and `GB/s`. The `volatile
sink` variable holds `f`'s result so the optimizer can't delete the whole loop as
"unused."

**`main(...)`** — CLI:
`bench_distance <base.csr> <queries.csr> [ncalls=2000000] [variant=0..5] [maxrow=0] [reps=1]`.
It loads the two matrices, picks one fixed query row (kept resident, like a real
query), generates `ncalls` random document ids (optionally restricted to the
first `maxrow` rows — that's the E4 knob), and runs the five regimes:

| Variant | Name | What it isolates |
|---|---|---|
| **1** | HOT merge | Rows are forced L1-resident (only 16 of them), so there's **no** memory cost — pure compute. Call this **T_compute** (V1). |
| **2** | COLD stream | Streams the same cold bytes with **no** merge — pure memory at maximum MLP (addresses known up front). Call this **T_load** (V2). |
| **3** | COLD merge | The **actual production kernel** on cold rows: load + compute fused. **T_total** (V3). |
| **4** | COLD merge + batch prefetch | Same as 3 but prefetches a whole batch of upcoming rows first, mimicking `searchLayer`'s two-pass prefetch — tests whether more MLP helps. |
| **5** | COLD gather→merge | Explicitly `memcpy`s a batch of rows into a small scratch buffer, *then* merges from it — the "dense-style" two-phase load-then-compute pipeline, for comparison. |

The **read-outs** (printed as formulas over the variant times, not baked in
here):
- memory-attributable share `= (V3 − V1) / V3`
- **fusion penalty** `= V3 − (V1 + V2)` — if positive, the merge's data-dependent
  pointer advance is destroying MLP, and restructuring (variants 4/5) should help.
- pipelining bound `= max(V1, V2)` — the best you could do if load and compute
  overlapped perfectly.

**Deriving each variant's memory-exposed time (the stacked bar in `fig1`).**
`bench_distance` prints only *one* number per variant — its **total** ns/call.
The compute-vs-memory split you see in `fig1_ablation.png` is therefore *derived*
from those totals, not measured directly, and it rests on a single fact:
**variants 1, 3, 4 and 5 all run the identical merge arithmetic** — only their
memory conditions differ. So V1 (HOT merge, rows already in L1) is the shared,
pure-ALU **compute floor**, and whatever a cold variant spends *above* that floor
is memory latency it could not hide:

```
memory-exposed(Vk) = total(Vk) − compute_floor        (compute_floor = V1)

  V3  production kernel      mem-exposed = V3 − V1
  V4  + batched prefetch     mem-exposed = V4 − V1
  V5  gather → merge         mem-exposed = V5 − V1
```

Each bar in `fig1` is thus stacked as `[ memory-exposed | compute_floor ]` and
sums back to that variant's total. Two special cases: **V2** does *no* merge, so
its compute floor is 0 and its *entire* total is the memory bar; and the plotter
uses `min(V1, Vk)` as the floor, so a variant that dips *below* V1 through
measurement noise is charged all-compute (a zero-height, never negative, memory
bar). Comparing the three cold-merge bars is the whole point of the rung — V4 and
V5 keep the **same** compute floor as V3, so a **shorter** memory segment on them
is direct evidence that their prefetch/staging overlapped the loads V3 left
exposed. (The `(V3 − V1) / V3` "share" above is just this same subtraction for the
production variant, expressed as a fraction.)

`variant=0` runs all five; passing `1..5` runs a single one (used so
`perf_groups.sh` can attribute hardware counters to exactly one regime). `maxrow`
shrinks the pool of rows touched — sweeping it (e.g. 4000 → 1,000,000) walks the
working set through the L2/L3/TLB sizes so you can see exactly where each cliff
is (E4). `reps` repeats the whole thing so you can take the best run on a noisy
shared node.

### 3.2 `bench_scale.cpp` — thread scaling (E5)

The final-verdict benchmark: run the kernel on **1, 2, 4, … 128 threads** and
watch how throughput grows. This is what distinguishes latency- from
bandwidth-bound.

**`merge(...)`** and **`stream_row(...)`** — the same two kernels as
`bench_distance` (production merge; pure memory stream).

**`main(...)`** — CLI:
`bench_scale <base.csr> <queries.csr> [ncalls_per_thread=400000] [threads=1,2,4,8,16,32,64,128]`.
For each thread count `T` it runs two OpenMP-parallel loops over random rows:
first the **stream** kernel (the achievable **bandwidth ceiling** — what the
memory system can deliver), then the **merge** kernel (what the real distance
achieves), with the perf gate opened only around the merge. It prints, per `T`:
stream GB/s, merge ns/call, merge Mcall/s, merge GB/s, and **GB/s per core**. The
interpretation (a rule, not a result):

- If **per-core** merge throughput stays roughly *flat* as `T` grows → you're
  **latency-bound** → the fix is more MLP (prefetch depth, batching).
- If **aggregate** merge GB/s *plateaus into the stream ceiling* → you're
  **bandwidth-bound** → the fix is fewer bytes (compression, layout, locality).

Pinning must come from OpenMP (`OMP_PLACES=cores OMP_PROC_BIND=spread`), and this
rung is only meaningful on an **exclusive compute node** — a shared login node
gives noise, not signal.

### 3.3 `profile_search.cpp` — the instrumented *real* driver (E1/E3b/E6)

This is the production path under a microscope. It builds a **real**
`SPARSE_HNSW` index and runs **real** queries; the difference from
`sparse_hnsw_demo` is that it's compiled with `-DSPARSE_HNSW_PROFILE`, which turns
on the distance-call and byte counters inside `searchLayer` (see the
`#ifdef SPARSE_HNSW_PROFILE` blocks in the algorithm). It has three **modes**,
each a different rung.

**`get_gt(...)` / `calculate_recall(...)`** — load the ground truth and score the
results, identical in spirit to the demo driver; recall is printed as a sanity
check that you profiled the same configuration you actually race.

**`main(...)`** — CLI (first 11 args identical to `sparse_hnsw_demo`):
`sparse_profile <M> <efC> <ef> <use_heuristic> <extend> <keep_pruned> <use_mkl> <mklThreshold> <base> <queries> <gt> [mode=batch|repeat|replay] [num_queries=all] [gate=cold|warm|all]`.
It builds the index (using `PROF_BUILD_THREADS` cores, *not* profiled — the gate
is closed during build), then dispatches on `mode`:

- **`batch` mode (E1, the default).** Runs `searchKNNBatch` once with the perf
  gate open around it, then prints exact totals from the instrumentation: number
  of distance calls, bytes of document rows streamed, and the derived
  **`ns_per_dist`** and **`achieved_GBps`**. This places the *real* search on the
  same ns/call scale as the micro-benchmark and gives the achieved-vs-ceiling
  bandwidth comparison. Run it under `perf_groups.sh` to attach the E2 hardware
  counters.

- **`repeat` mode (E3b).** Single-threaded on purpose. For each query it runs the
  search **twice back-to-back** on a reused scratch: pass 1 is **cold**, pass 2 is
  **warm** (same deterministic path, now L3-resident). `mem_share = (cold −
  warm) / cold` is the memory-attributable share of *real* search time — with the
  index's real hub locality, which the random-id micro-benchmark can't capture. It
  prints **validity checks** you must confirm: `path_mismatches` must be 0 (both
  passes walked the identical node sequence) and the per-query footprint must fit
  in one cache slice (else pass 2 wasn't truly warm and the share is an
  underestimate — it prints a warning). The `gate` argument selects which pass the
  perf FIFO counts (`cold`, `warm`, or `all`).

- **`replay` mode (E6, run this first).** The premise check. It does three sweeps
  over the queries: **A** records every query's distance sequence into a flat
  array; **B** runs a normal (timed) search → `T_full`; **C** runs again but
  *replays* the recorded values instead of computing them → `T_overhead`. Sweep C
  reproduces the identical traversal/heaps/visited-set while touching no document
  data, so `distance_share = (T_full − T_overhead) / T_full`. It prints
  `ndist_match` — a **validity gate**: if the replay didn't reproduce the exact
  same number of distance calls the whole number is meaningless and a warning is
  printed. It also splits traffic into document-row bytes vs graph
  (neighbor-list) bytes, to confirm which one is the real traffic.

Environment knobs it reads: `PROF_BUILD_THREADS` (cores for the build),
`OMP_NUM_THREADS` (cores for the batch search), and the `PERF_CTL_FIFO` /
`PERF_ACK_FIFO` pair (set by the perf wrapper scripts to enable gating).

### 3.4 `perf_ctl.h` — the perf gate

A tiny header (a namespace of `inline` functions, no `.cpp`) that lets the
program tell `perf` *when* to count, so index-build traffic stays out of the
search-phase counters. Functions:

- **`init()`** — reads the `PERF_CTL_FIFO` / `PERF_ACK_FIFO` environment variables
  and opens those two named pipes. If they're unset, it does nothing and every
  later call becomes a no-op (so unprofiled runs are completely unaffected).
- **`send(cmd)`** — writes a command to perf's control FIFO and blocks until perf
  writes back its `ack` on the acknowledgement FIFO (so timing is exact).
- **`enable()` / `disable()`** — send `"enable\n"` / `"disable\n"`; call them
  around the region of interest.

It's paired with `perf stat -D -1 --control=fifo:...`, where `-D -1` means "start
with counters disabled" and `--control=fifo` means "listen on these pipes."

### 3.5 `perf_groups.sh` — hardware counter groups (E2)

Runs `perf stat` against any command, in **three passes** (because only ~5
counters fit at once on Zen3), each pass a themed group of events, all gated to
the region of interest via the FIFOs. Usage:
`./perf_groups.sh <command> [args…]` — e.g. wrap `sparse_profile … batch`, or
`taskset -c 8 bench_distance … <variant>`. The three groups:

- **G1 — "is it memory at all?"** IPC (instructions per cycle), L1 miss activity,
  and DRAM/L2 fill rates.
- **G2 — "where do the fills come from?"** Local vs remote DRAM and same- vs
  other-cache-cluster fills — the **NUMA** diagnosis (does remote memory hurt?).
- **G3 — "translation + speculation."** Page walks split by page size (the
  **huge-page** evidence) plus branch mispredictions.

For each group it sets up fresh FIFOs, runs perf, then post-processes the CSV: if
the program printed a `PROF ndist=<N>` line, every count is also divided by the
distance-call count so the groups are all in the **same comparable unit
(per-distance-call)**.

### 3.6 `perf_hotspots.sh` — where the cycles go (E6b)

The companion to E6. Where `replay` tells you *how much* of the time isn't
`distance()`, this tells you *what* that remainder is. Usage:
`./perf_hotspots.sh <command> [args…]`. It runs `perf record` (gated to the
search phase), then `perf report` to print the top self-cycle functions. Because
the profile build keeps `distance()` **out-of-line** (its own symbol), the
remainder — priority-queue sifting, visited-bit tests, neighbor-list walks — shows
up separately inside `searchLayer`. Trust the E6 replay number for the *split*;
trust this for the *ranking* of what's left.

### 3.7 `slurm_profile.sh` — run the whole ladder

The **orchestrator**: a SLURM batch script (`#SBATCH` headers request one
exclusive CPU node for 20 hours, 128 cores) that runs **every rung in sequence on
the msmarco_full dataset** and, at the end, calls the plotter on its own log. It
defines the dataset paths and the shared `HNSW_ARGS` once at the top, prints the
node's huge-page and paranoid settings for the record, then runs (in this order):
E3 ablation → E4 sweep → E3 per-variant counters → E5 scaling → E1/E2 batch → E1
at 64 threads → E3b repeat → E6 replay → E6b hotspots → plot. This is the
one-command way to get everything; see Part 5.

### 3.8 `plot_profile.py` — turn the log into figures

A Python script (needs `matplotlib`, so `module load python` first) that **parses
whatever the ladder printed** — a full SLURM log or the output of a single tool
piped in — and draws the figures. Sections absent from the input are simply
skipped, so partial runs still produce whatever they can. Structure:

- **`parse(text)`** — scans the log text and pulls each rung's numbers into a
  structured dictionary using regular expressions. The single point that knows the
  log format.
- **`style(ax)`, `fnum(d, key)`, `short_ctx(c)`** — small formatting/lookup
  helpers.
- **`fig_time_split(data)`** → `fig6` — the **premise check** (E6): the share of
  time `distance()` owns.
- **`fig_ablation(data)`** → `fig1` — the ablation (E3): each variant's ns/call
  split into compute vs memory, with the `max(V1,V2)` and `V1+V2` reference lines.
- **`fig_working_set(data)`** → `fig2` — ns/call vs working set (E4), cache/TLB
  knees marked.
- **`fig_scaling(data)`** → `fig3` — achieved vs ceiling GB/s and per-core
  throughput (E5), with the latency-vs-bandwidth verdict.
- **`fig_counters(data)`** → `fig4` — hardware counters per distance call (E2).
- **`fig_cold_warm(data)`** → `fig5` — cold vs warm and the real-path memory
  share (E3b).
- **`write_csv(data, path)`** — dumps everything as tidy long-form CSV
  (`section, series, working_set_mb, metric, value`) to replot elsewhere.
- **`main()`** — argument parsing (input file or stdin, `-o` output dir, `--pdf`),
  then calls each `fig_*`, writes a combined `summary.png`, and the CSV.

Invocation:
`python3 plot_profile.py <log> -o <outdir> [--pdf]`, or pipe a single tool:
`bin/bench_distance base.csr q.csr 2000000 | python3 plot_profile.py -o plots/`.

---

## Part 4 — The experimental ladder, rung by rung

Each rung below gives the **question**, the **mechanism**, and the **exact
command on msmarco_full**. Define these once (matching `slurm_profile.sh`) and
reuse them in every command:

```bash
module load intel                     # Perlmutter: the toolchain the binaries were built with
BIN=$SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/bin
PROF=$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/profiling
DATA=$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_full
BASE=$DATA/base_full.csr
QUERIES=$DATA/queries.dev.csr
GT=$DATA/base_full.dev.gt
# The 11 demo-compatible args (M=16, efC=200, ef=150, heuristic on, MKL off):
HNSW_ARGS="16 200 150 1 0 0 1 0 $BASE $QUERIES $GT"
```

> `bench_distance` and `bench_scale` take only `<base> <queries>` (they don't
> build an index, so no ground truth). `sparse_profile` takes the full 11-arg
> demo signature plus a mode.

### E6 — does `distance()` actually own the time? *(run/read first)*

- **Question:** every ns/call number assumes distance dominates. Is that true?
- **Mechanism:** record each query's distance sequence, then replay it; the gap
  between a full search and a replay-only search is `distance()`'s true cost
  (`distance_share = (T_full − T_replay) / T_full`). Confirm `ndist_match=1`.

```bash
OMP_NUM_THREADS=1 $BIN/sparse_profile $HNSW_ARGS replay 2000
```

(`2000` = number of queries to use; keep it small since this is single-threaded.)

### E6b — what the non-distance time *is*

- **Question:** whatever E6 says *isn't* distance — what is it?
- **Mechanism:** gated `perf record` over the search phase, ranked by self-cycles.

```bash
OMP_NUM_THREADS=1 $PROF/perf_hotspots.sh $BIN/sparse_profile $HNSW_ARGS batch
```

### E3 — hot/cold ablation (the sparse "two timers")

- **Question:** of one distance call's time, how much is compute vs
  memory-exposed, and is the fused merge hurting MLP?
- **Mechanism:** the five variants of `bench_distance` (Part 3.1). Pin to one core;
  take the best of 3 reps on a shared node.

```bash
taskset -c 8 $BIN/bench_distance $BASE $QUERIES 2000000 0 0 3
#                                              ncalls  variant  maxrow  reps
```

### E4 — working-set sweep (find the cache/TLB cliffs)

- **Question:** at what data size does each cache/TLB level stop helping?
- **Mechanism:** run variant 3 (cold merge) while growing `maxrow` through the
  L2 / L3 / TLB-reach sizes.

```bash
for N in 4000 16000 64000 250000 1000000; do
  taskset -c 8 $BIN/bench_distance $BASE $QUERIES 1000000 3 $N
done
```

### E1 — achieved bandwidth of the *real* search

- **Question:** what bandwidth and ns/call does the actual index achieve
  (with real hub locality, not random rows)?
- **Mechanism:** `batch` mode reports exact bytes and distance calls from the
  in-code counters → `achieved_GBps` and `ns_per_dist`.

```bash
# build with all cores, search single-threaded (see the taskset caveat below):
PROF_BUILD_THREADS=128 OMP_NUM_THREADS=1 $BIN/sparse_profile $HNSW_ARGS batch
# and at scale:
OMP_NUM_THREADS=64 $BIN/sparse_profile $HNSW_ARGS batch
```

### E2 — hardware counter groups (gated to search)

- **Question:** which hypothesis do the CPU counters support — memory-bound,
  NUMA-remote, page-walks?
- **Mechanism:** the three ≤5-event groups of `perf_groups.sh` (Part 3.5),
  normalized per distance call.

```bash
PROF_BUILD_THREADS=128 OMP_NUM_THREADS=1 \
  $PROF/perf_groups.sh $BIN/sparse_profile $HNSW_ARGS batch
```

> **Do not `taskset` `sparse_profile`** when `PROF_BUILD_THREADS > 1` — it would
> pin all build threads to one core. `OMP_PROC_BIND` handles the single search
> thread. `perf_groups.sh` also wraps a single micro-benchmark variant, e.g.
> `perf_groups.sh taskset -c 8 $BIN/bench_distance $BASE $QUERIES 2000000 3`.

### E3b — repeat-query cold/warm on the real path

- **Question:** on the *real* traversal (real locality), what share of time is
  memory?
- **Mechanism:** run each query twice; `mem_share = (cold − warm) / cold`. Check
  `path_mismatches=0` and the footprint warning.

```bash
OMP_NUM_THREADS=1 $PROF/perf_groups.sh $BIN/sparse_profile $HNSW_ARGS repeat 2000 cold
#                                                                        mode  nq  gate
```

### E5 — thread scaling (the verdict; exclusive compute node only)

- **Question:** latency-bound or bandwidth-bound?
- **Mechanism:** `bench_scale` across 1→128 threads; compare per-core throughput
  (flat ⇒ latency) and aggregate GB/s vs the stream ceiling (plateau ⇒ bandwidth).

```bash
export OMP_PLACES=cores OMP_PROC_BIND=spread
$BIN/bench_scale $BASE $QUERIES 400000 1,2,4,8,16,32,64,128
```

### Interpreting the ladder (a decision guide, not a result)

Once the rungs are run, the *logic* for choosing a fix is:

- E3 shows a positive **fusion penalty** *and* E5 scales near-linearly →
  **latency-bound** → deepen the prefetch pipeline in `searchLayer` (variant 4/5
  are the proof-of-concept); the target is `max(V1, V2)`.
- E5 **saturates** into the stream ceiling *and* E1's achieved/ceiling ratio is
  high → **bandwidth-bound** → move fewer bytes (fp16 was one such lever) or
  improve visit locality (reordering).
- E2's G3 shows page walks per call → stack the **huge-page** experiment on top
  (an independent win): `module load craype-hugepages2M`, relink, re-run G3.
- Once memory is fully hidden you're left sitting on variant 1's compute floor →
  the next lever is **SIMD** set intersection / galloping in the merge.

---

## Part 5 — Running it end-to-end on msmarco_full

### 1. Build the profiling targets

```bash
cd $SCRATCH/repos/sparse_hnsw/minimal_hnsw/build
module load intel
cmake .. && make bench_distance bench_scale sparse_profile
```

All three land in `build/bin/`. `sparse_profile` compiles its **own** copy of
`sparse_hnsw.cpp` with `-DSPARSE_HNSW_PROFILE`, so the normal `sparse_hnsw_demo`
library stays bit-for-bit unchanged — the instrumentation exists only in this
build.

### 2. Run the whole ladder in one job (recommended)

`slurm_profile.sh` already points at msmarco_full and runs every rung, then
plots. Submit it from `sparse/scripts/` (so its `logs/` path resolves):

```bash
cd $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts
sbatch ../profiling/slurm_profile.sh
```

When the job finishes you get `logs/profile_<jobid>.out` (the full log) and
`logs/plots_<jobid>/` (the figures + `profile_data.csv`). **Read the log starting
from the E6 section** — it validates the premise the rest depends on.

### 3. Or run rungs by hand

Use the commands in Part 4 (they all target msmarco_full via the `BASE` /
`QUERIES` / `GT` variables). To plot a single hand-run tool while iterating, pipe
it straight into the plotter:

```bash
module load python
$BIN/bench_distance $BASE $QUERIES 2000000 | python3 $PROF/plot_profile.py -o plots/
```

### 4. Where to run what

- **E3, E4, E3b, E6, E6b, E1-single-thread, E2** are single-core or single-thread
  and can be *sanity-checked* on a login node, but login nodes are heavily shared
  (only best-of-N and within-run comparisons mean anything there).
- **E5 and any real bandwidth/scaling claim require an exclusive compute node**
  (`salloc`/`sbatch` with `--constraint=cpu`), because they depend on having the
  whole memory system to yourself. `slurm_profile.sh` requests exactly that.

---

## Appendix — the Perlmutter constraints that shaped the design

Several design choices only make sense once you know the machine's limits. These
are environment facts (not measurements), captured so the toolkit's shape is
understandable:

- **`perf_event_paranoid=2`** — only user-space counting is allowed, so every
  perf event carries a `:u` suffix.
- **No uncore counters** — the direct DRAM-bandwidth PMUs (`amd_df` / `amd_l3`)
  aren't permitted, which is *why* bandwidth is derived analytically (exact bytes
  ÷ time from the in-code counters, rung E1) instead of read from a register.
- **No precise per-instruction attribution** — AMD's IBS/PEBS sampling needs
  system-wide mode, which the paranoid setting denies; that's why the split comes
  from **replay** (E6) and sampling (E6b) is only trusted at function granularity.
- **≤ 5 counters at once** — Zen3 has 6 core counters and starts time-sharing
  (estimating) past ~5, which is why `perf_groups.sh` uses three separate groups.
- **Build vs search share `searchLayer`** — and the build dominates wall time, so
  counters are FIFO-gated to the search phase (`perf_ctl.h`).
- **Login nodes are shared and use small pages** — real numbers need a compute
  node; the huge-page experiment needs the `craype-hugepages2M` module and a full
  relink before re-running E2's G3 group.
- **A known gap:** every `sparse_profile` run pays a full index build first
  (minutes at 128 threads). Adding index save/load would cut iteration time
  dramatically before a long msmarco_full campaign.
```
