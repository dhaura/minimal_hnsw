# Final method sweep (cleanup Step 6b), msmarco_full and nq_splade

Jobs 60871988, 60871990, 60871991, 60880834 (msmarco_full) and 60871992, 60871993, 60871994, 60880841 (nq_splade),
2026-11-05; work counts from 60871995 / 60871996. Frozen Intel binaries in `/pscratch/sd/d/dhaura/cleanup_6b/bin`
(`sparse_hnsw_sweep` md5 6db9bc10, `sparse_profile` d13a0bc7), 64 threads, 5 timed runs + 1 warmup.

Every series uses the best options: u8 slots only (320 B msmarco_full, 448 B nq_splade), inverted seeds
top_k 8, 8 terms x 4 docs, alpha / M / efC grids as the inline sweep (alpha 0.75-0.9 or 0.65-0.8 at
M32 efC200; alpha 0.85 at M32 efC200-1600, M48 efC800-1600 / M64 efC400-800), beta 2-8. Four series per graph:
off (no gate, no stop: the same-job baseline), Gap (score-gap stop, gap 0.05-0.3), inline16, inline16+Gap.
Low-recall jobs (60880834 / 60880841, `final_<ds>_lowrec_*.csv`): M32 efC200, alpha 0.65-0.85 / 0.6-0.8, beta 2-4,
seeds off and 8:4. Seeded runs bottom out at recall 0.873 / 0.823, so the unseeded rows (down to 0.785 / 0.754)
are the only points below that; seeded and unseeded rows are separate ef curves in the analysis.

Files: `final_sweep_<ds>.md` (matched-recall and same-ef tables), `final_profile_<ds>.md` (work counts and
per-query recall), `figures_<ds>/` (fronts, Pareto, speedup tables, figures vs baselines and vs off).
Scripts: `sparse/scripts/final_sweep/` (run_final_sweep / run_final_profile, analyze_final_sweep.py,
analyze_final_profile.py, make_final_figures.sh).

## Same-job speedup at matched recall (QPS / off QPS, log-QPS interpolation per fixed ef curve)

| recall | msmarco Gap | msmarco inline16 | msmarco i16+Gap | nq Gap | nq inline16 | nq i16+Gap |
|---|---|---|---|---|---|---|
| 0.80 | 0.98 | 1.06 | 1.03 | 0.98 | 1.15 | 1.08 |
| 0.85 | 0.99 | 1.02 | 1.03 | 1.01 | 1.07 | 1.12 |
| 0.90 | 1.05 | 1.04 | 1.08 | 1.04 | 1.10 | 1.13 |
| 0.95 | 1.07 | 1.02 | 1.07 | 1.06 | 1.07 | 1.09 |
| 0.97 | 1.09 | 1.03 | 1.09 | 1.06 | 1.11 | 1.22 |
| 0.98 | 1.09 | 1.03 | 1.11 | 1.11 | 1.09 | 1.23 |
| 0.99 | 1.21 | 1.05 | 1.23 | 1.24 | 0.92 | 1.26 |
| 0.995 | 1.33 | 1.09 | 1.41 | 1.13 | 0.73 | 1.03 |

At 0.80-0.85 Gap alone gives nothing (0.98-1.01x), while the gate gives
1.02-1.06x on msmarco_full and 1.07-1.15x on nq_splade. Consistent with the earlier inline sweep
(sketch_theory/inline_sweep_<ds>.md): the gate alone is small on msmarco_full and 1.07-1.14x on nq_splade
up to 0.98, and it loses at 0.99+ on nq_splade unless combined with the stop.

## Against the baselines (method front = best of all four series; `figures_<ds>/speedup_vs_baselines_SparseHNSW.csv`)

| recall | msmarco: SEISMIC | kANNolo | SINDI | nq: SEISMIC | kANNolo | SINDI |
|---|---|---|---|---|---|---|
| 0.80 | 3.41 | 3.51 | 8.25 | 2.89 | 5.07 | 3.96 |
| 0.85 | 2.99 | 3.22 | 8.48 | 2.71 | 4.56 | 4.27 |
| 0.90 | 3.16 | 3.61 | 11.90 | 2.54 | 5.12 | 5.24 |
| 0.95 | 2.47 | 3.69 | 12.85 | 1.52 | 3.90 | 3.94 |
| 0.97 | 2.15 | 3.02 | 14.34 | 1.14 | 3.68 | 3.48 |
| 0.98 | 1.75 | 2.96 | 10.07 | 0.97 | 3.58 | 2.75 |
| 0.99 | 1.44 | 2.69 | 8.04 | 0.61 | 3.13 | 1.71 |

GrassRMA and PyANNS are 6-29x slower at every recall from 0.80 to 0.99. On nq_splade SEISMIC wins at 0.98-0.99, as in the
earlier comparisons (0.56-0.59x at 0.99).

## Work per query (headline graph M32 efC200 alpha 0.85 beta 3, gap 0.1; `final_profile_<ds>.md`)

- The gate cuts 64 B row lines by 9-17% on msmarco_full (rejecting 15-24% of checked neighbors) and by
  24-34% on nq_splade (31-42% rejected), at the same ef, for a recall change of <= 0.0002 and 0.0025-0.004.
- Gap 0.1 stops searching once ef >= 100: recall plateaus at 0.963-0.965 (msmarco_full) / 0.960-0.963 (nq_splade), and lines per query stay at
  ~7.3k (msmarco_full) / ~10.7k (nq_splade) however large ef is; with the gate, 6.5-6.7k / 8.1-8.3k.
- Per-query recall: at ef 400, off has 96.2% / 97.3% of queries at recall >= 0.9; inline16 96.2% / 96.7%;
  inline16+Gap 93.4% / 92.8%. p5 rises with ef from 0.6-0.7 (ef 50) to 0.8-0.9 (ef 400), p10 0.8-0.9; min recall 0 (0.1 for off and inline16
on nq_splade at ef 400).
