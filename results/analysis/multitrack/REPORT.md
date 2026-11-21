# Multi-track training, leave one track out (Spielberg held out)

Plan, design and decision rule: [`docs/multitrack_plan.md`](../../../docs/multitrack_plan.md), fixed before any run.

**Question:** does training the wall-aware dual residual policy on Austin + Catalunya + Silverstone make it better on Spielberg, a track it never sees in training or checkpoint selection, than training on Austin alone?

**Short answer:**
- Multi-track training makes Spielberg results **reliable across seeds**. Spielberg qualifying is 29–32 of 36 for all five multi-track seeds, against 0–32 for the Austin-only policy.
- It does **not raise the best case**. The best Austin-only seed reaches 32.
- Under the pre-registered rule the result is **"ambiguous"**, even with 5 seeds. That rule needed the multi-track median (30) to be above the *best* baseline seed (32). With a ceiling of 34 qualifying Spielberg cells, one lucky baseline seed is enough to block it.

## What was run

| Variant | Trained on | Checkpoint selected on | Runs |
| --- | --- | --- | --- |
| base | Austin | Austin (published protocol) | existing s0–s2, new s3–s4 |
| A3 | Austin | Austin + Catalunya + Silverstone | same runs as base, re-selected |
| **MT** | Austin + Catalunya + Silverstone | Austin + Catalunya + Silverstone | new s0–s4 |
| MT-A | Austin + Catalunya + Silverstone | Austin | same runs as MT, re-selected |

- **Training settings:** all runs are 3M PPO steps with identical settings. MT draws one training track per episode, so Austin gets about 1/3 of MT's samples.
- **Evaluation:** the 144-lap reporting grid from the multi-seed study (4 tracks × a_lat {4, 6, 8} × 12 cells) and the unchanged qualifying rule.
- **Main comparison:** MT vs A3, which isolates the training effect.
- **Runtime:** about 13–17 min per run (training plus protocol) on the Rust kernel; about 1 h 50 min in total.

**Commands:**

```bash
results/analysis/multitrack/run.sh          # stages 1-4: MT s0-s2, both selections, A3 re-selection of base s0-s2
results/analysis/multitrack/run_stage6.sh   # stage 6: base and MT seeds 3-4, five-seed comparison -> five_seeds/
```

**Outputs:**
- `per_run.csv`, `per_variant.csv`, `report_laps.csv`, `cell_qualifying_share.csv`, `selection_curves.csv`, in `five_seeds/` (5 seeds) and here (3 seeds);
- under each run folder: `protocol_*_sel3.*` for 3-track selection and `protocol_*` (no suffix) for Austin selection.

## Results, selected checkpoints (5 seeds)

Qualifying laps out of 36 per track (144 in total), median [min, max] over seeds:

| Variant | Total | Austin | Catalunya | Silverstone | **Spielberg** | Crashes | Mean lap-time ratio* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 91 [58, 119] | 30 [28, 31] | 28 [28, 32] | 13 [0, 30] | **21 [0, 30]** | 51 | 1.008 |
| A3 | 114 [67, 124] | 30 [27, 31] | 30 [28, 32] | 28 [12, 32] | **21 [0, 32]** | 30 | 1.026 |
| **MT** | **120 [118, 124]** | 30 [20, 30] | 32 [30, 32] | 32 [28, 34] | **30 [29, 32]** | 22 | 1.032 |
| MT-A | 120 [89, 120] | 30 [24, 30] | 32 [26, 32] | 28 [17, 30] | 30 [22, 30] | 24 | 1.028 |

\* On the 75 cells that every 3-seed run completes (from `per_run.csv`).

Spielberg per seed (s0–s4):

| Variant | s0 | s1 | s2 | s3 | s4 |
| --- | --- | --- | --- | --- | --- |
| A3 | 9 | 21 | 30 | **0** | 32 |
| MT | 30 | 32 | 30 | 30 | 29 |

**Ceiling:** over every checkpoint of every run, only 34 of the 36 Spielberg cells ever qualify; the two that never do are at a_lat 8. Across all four tracks it is 132 of 144. MT's Spielberg results are within 2–5 laps of that ceiling for every seed.

### Decision rule (pre-registered)

| Condition | Value | Met? |
| --- | --- | --- |
| Helps: MT Spielberg median > A3 maximum | 30 vs 32 | no |
| …and MT Austin median ≥ base Austin median − 3 | 30 vs 30 | yes |
| Hurts: MT Spielberg median < A3 minimum | 30 vs 0 | no |

**Verdict: ambiguous**, after the stage-6 extension that the plan prescribed.

The rule was too strict for a metric this close to its ceiling. I am reporting it as written rather than changing it after seeing the data. A post-hoc one-sided Mann-Whitney test on the 5 vs 5 Spielberg counts (30, 32, 30, 30, 29 against 9, 21, 30, 0, 32) gives p = 0.14, so it is **not significant** either. MT's median is 9 laps higher, but A3/s4 (32) beats four of the five MT seeds.

## Findings

### 1. Multi-track training removes the seed lottery on unseen tracks (observed)

- **Every MT seed does well on Spielberg.** All five are at 29–32 qualifying laps.
- **The Austin-only policy's Spielberg result is a coin flip on the seed:**
  - three of five seeds score ≤ 21;
  - s3 scores **0** (and 12 on Silverstone even after re-selection);
  - two seeds reach 30–32.
- **This holds across all checkpoints, not just the selected ones.** The diagnostic below covers every checkpoint from 0.75M on, which selection never sees: 55 checkpoints per variant.

| Variant | Austin | Catalunya | Silverstone | Spielberg |
| --- | --- | --- | --- | --- |
| base, median per checkpoint | 29 | 28 | 12 | **7** |
| MT, median per checkpoint | 25 | 30 | 30 | **30** |

### 2. Austin is unchanged at the selected checkpoint but weaker across training (observed)

- **At the selected checkpoint:** the Austin median is 30 for both variants.
- **Over all checkpoints:** the MT Austin median is 25, against 29 for base.
- **MT/s1's selected checkpoint loses Austin:** 20 of 36, with 10 too-slow laps. MT/s4 also scores 26.
- The cost of giving Austin a third of the samples shows up as more variable Austin behaviour, not as a lower median for the selected checkpoint.

### 3. The late-training slowdown is more common with multi-track training (observed)

- **What it looks like:** checkpoints where ≥ 10 of 144 laps fail the 5% lap-time rule. This is the same failure as base/s0 in the multi-seed study: a nearly constant negative acceleration residual.
- **How often:** 2 of 5 MT seeds (s1: 7 checkpoints, s2: 8), against 1 of 5 base seeds (s0: 7).
- **Where:** in MT/s2 the slowdown takes out Austin, Catalunya and Silverstone completely from 2.25M on, while **Spielberg stays at 31–32**. It is track-dependent, not a global loss of speed.
- **Selection avoided it in MT/s2** (it picked 1.25M) but **not fully in MT/s1** (it picked 2.75M, with 10 too-slow Austin laps).
- A fix such as the speed-shortfall hinge, or early stopping on the training tracks, would be a separate experiment. In the multi-seed study the hinge hurt held-out tracks.

### 4. For multi-track policies, selecting on Austin alone is nearly as good (observed)

- **MT-A and MT are almost tied:** MT-A's Spielberg median is also 30. MT-A has a worse minimum (22, from s4 picking 1.0M) and a worse total minimum (89 vs 118).
- **Compared with the Austin-only policies:** there, Austin selection predicted held-out results poorly. For an Austin-trained policy, re-selecting on three tracks (A3 vs base) helps the median total (114 vs 91) but not the Spielberg median (21 vs 21).
- **So the gain comes from training, not from selection.**

### 5. Multi-track policies are a little slower (observed)

- **Mean lap-time ratio on common completed cells:** 1.032 for MT against 1.008 for base and 1.026 for A3.
- Part of MT's robustness may be bought with speed; the ratios are still well inside 1.05.
- No selected MT lap fails the 5% rule except MT/s1's 10 Austin laps.

## Caveats

- **Only one held-out track.** "Generalisation" here means Spielberg. Spielberg is the shortest of the four tracks (338 m of reference line, against 404–446 m), and its two known crash hotspots (s ≈ 42, 304) are largely avoided by MT (2–3 crashes at s ≈ 42, none at s ≈ 304).
- **Catalunya and Silverstone are training tracks for MT** and held-out tracks for base/A3, so their gains are not evidence of generalisation.
- **Equal total samples (3M).** MT's Austin share is 1/3 of base's. Stage 5 (9M steps) was not run: the training-track curves were not still rising at 3M (MT/s2 collapsed instead), which was the plan's condition for it.
- **The rule-limited verdict.** The plan's decision rule cannot separate "reliably near the ceiling" from "sometimes at the ceiling". A future rule should compare a lower quantile, such as the worst seed or the number of seeds ≥ 24 of 36, or use a test fixed in advance.
