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

**Commands:** one script runs a whole fold. The held-out track is its argument, and `pilot.train_residual --holdout` sets the training tracks.

```bash
results/analysis/multitrack/run_fold.sh Spielberg     # this fold
results/analysis/multitrack/run_fold.sh Silverstone   # fold 2 below
```

The script skips training runs and protocol steps whose outputs exist, so re-running it only rebuilds the comparison. This fold was originally run in two stages (seeds 0–2, then 3–4); the logs are in `logs/`.

**Outputs:**
- `heldout_spielberg/`: `per_run.csv`, `per_variant.csv`, `report_laps.csv`, `cell_qualifying_share.csv`, `selection_curves.csv`;
- MT runs in `results_st_pacejka/multitrack_heldout_spielberg/s0–s4`;
- under each run folder: `protocol_*_noSpielberg.*` for selection on this fold's training tracks, and `protocol_*` (no suffix) for Austin selection.

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

---

# Fold 2: Silverstone held out

The same design with Spielberg and Silverstone swapped. The plan and a **revised** decision rule were fixed before any run (`docs/multitrack_plan.md`, "Second fold").

| Variant | Trained on | Selected on |
| --- | --- | --- |
| base | Austin | Austin |
| A3S | Austin | Austin + Catalunya + Spielberg |
| **MTS** | Austin + Catalunya + Spielberg | Austin + Catalunya + Spielberg |
| MTS-A | Austin + Catalunya + Spielberg | Austin |

- **Runs:** 5 seeds per variant. base and A3S reuse the 5 Austin-only runs.
- **Runtime:** about 1 h 37 min (`run_fold.sh Silverstone`).
- **Outputs:**
  - `heldout_silverstone/`, where A3S and MTS are labelled `A3` and `MT`, as in every fold;
  - runs in `results_st_pacejka/multitrack_heldout_silverstone/`;
  - selection files tagged `_noSilverstone`.

**Short answer: no measurable benefit on Silverstone.**
- MTS and A3S have the same Silverstone median, and both have one seed that collapses.
- The verdict under the pre-registered rule is **ambiguous**, and the Mann-Whitney p is 0.67.
- Unlike fold 1, the gain over the published Austin-only policy (base) comes almost entirely from **selecting on more tracks**, not from training on them.

## Results, selected checkpoints

Qualifying laps out of 36 per track, median [min, max] over 5 seeds:

| Variant | Total | Austin | Catalunya | **Silverstone** | Spielberg | Crashes | Mean lap-time ratio* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 91 [58, 119] | 30 [28, 31] | 28 [28, 32] | **13 [0, 30]** | 21 [0, 30] | 51 | 1.009 |
| A3S | 114 [58, 120] | 30 [30, 31] | 30 [28, 32] | **28 [0, 30]** | 21 [0, 30] | 30 | 1.027 |
| **MTS** | 120 [94, 124] | 30 [26, 30] | 32 [28, 32] | **27 [2, 28]** | 30 [27, 34] | 24 | 1.036 |
| MTS-A | 118 [94, 124] | 30 [22, 30] | 30 [26, 32] | 28 [22, 30] | 30 [24, 34] | 22 | 1.034 |

\* On the cells that every run in this comparison completes.

Silverstone per seed (s0–s4):

| Variant | s0 | s1 | s2 | s3 | s4 |
| --- | --- | --- | --- | --- | --- |
| A3S | 20 | 30 | 28 | **0** | 28 |
| MTS | 22 | **2** | 28 | 27 | 28 |

### Decision rule (pre-registered for this fold)

| Condition | Value | Met? |
| --- | --- | --- |
| Helps: MTS median ≥ A3S median + 6 | 27 vs 34 | no |
| …and MTS worst seed ≥ A3S median | 2 vs 28 | no |
| …and MTS Austin median ≥ base Austin median − 3 | 30 vs 27 | yes |
| Hurts: MTS median ≤ A3S median − 6 | 27 vs 22 | no |

**Verdict: ambiguous.** One-sided Mann-Whitney: p = 0.67.

## Findings

### 1. Fold 2 does not reproduce fold 1's reliability gain (observed)

- **The result does not carry over.** In fold 1 every multi-track seed scored 29–32 on the held-out track. Here, MTS/s1's selected checkpoint (3M) scores **2 of 36** on Silverstone:
  - 31 wall collisions, 30 of them at s ≈ 240–260 m, at every speed level;
  - this is not a known hotspot.
- **One checkpoint earlier it was fine.** The same run's 2.75M checkpoint qualifies 30 of 36 there, so Silverstone performance can flip between neighbouring checkpoints.
- **Selection cannot see this by design.** The 3M checkpoint is at least as good on all three training tracks.
- **Without that seed, MTS and A3S are indistinguishable:** 22–28 against 20–30.
- **What happens at s ≈ 246 [observed, replay of a_lat 6, μ/μ₀ 1.0, start s 200]:**
  - It is a corner exit: a right-hander (κ ≈ −0.05 to −0.08 1/m over s ≈ 222–245) runs into a wall on the outside (left).
  - The reference line passes about 14 cm from that wall.
  - Fed the same states, the 3M policy steers about 0.012 rad less to the right than the 2.75M policy through the corner (mean residual −0.040 against −0.052 rad), and brakes slightly less (a_res −0.45 against −0.53 m/s²).
  - Result: the 3M lap reaches e_y +0.09 m at the exit and touches the wall. The 2.75M lap peaks at +0.05 m, with a minimum gap of 0.077 m.
  - The wall-gap observation shows the pinch from s ≈ 235 on (gap 1 s ahead ≈ 0.1 m), but the 3M policy does not respond enough.
- **Policies trained on Silverstone never crash there.** Fold 1's MT has 0 crashes at s 240–252 over all 65 checkpoints.
- **Policies not trained on Silverstone fail there now and then:** MTS s0 at 1.5M (28 laps), s1 at 3M (30), s2 at 2.5M (2); the Austin-only runs fail there on 1–9 laps at some checkpoints. It is a fifth corner-exit wall pinch, of the same kind as the known hotspots.

### 2. Across checkpoints, multi-track training is still much more robust on the held-out track (observed, diagnostic)

Median qualifying per checkpoint, over every checkpoint from 0.75M on (55 per variant), which selection never sees:

| Variant | Austin | Catalunya | Silverstone | Spielberg |
| --- | --- | --- | --- | --- |
| base | 29 | 28 | **12** | 7 |
| MTS (Silverstone held out) | 28 | 28 | **28** | 30 (trained) |
| MT, fold 1 (Spielberg held out) | 25 | 30 | 30 (trained) | **30** |

- A typical multi-track checkpoint does well on the held-out track in both folds.
- A typical Austin-only checkpoint does not: its median is 12 on Silverstone and 7 on Spielberg.
- But in fold 2, **three-track selection already finds such a checkpoint for most Austin-only seeds** (A3S median 28). So the extra reliability from multi-track training does not show up at the selected checkpoint.

### 3. Selection on more tracks helps asymmetrically (observed)

| Comparison | Held-out median: Austin selection → 3-track selection |
| --- | --- |
| Fold 2: base → A3S, Silverstone | 13 → 28 |
| Fold 1: base → A3, Spielberg | 21 → 21 |

- In fold 2, selecting the Austin-only runs with Spielberg in the grid picks checkpoints that also do well on Silverstone.
- In fold 1, Silverstone in the grid did not help on Spielberg.
- One reading [hypothesis]: Spielberg-good checkpoints tend to be Silverstone-good, but not the other way round. With 5 seeds and 2 folds this is not established.

### 4. The slowdown again appears in some multi-track seeds (observed)

- **MTS/s0** has ≥ 10 too-slow laps (out of 144) at 9 of its 11 checkpoints from 0.75M on. It falls to 0 qualifying on all tracks from 1.5–1.75M; selection picked its 1.0M checkpoint.
- **MTS/s4** dips at 1.75–2.25M and recovers.
- **Counts:** 2 of 5 seeds with ≥ 3 such checkpoints, as in fold 1 (2 of 5). Base has 1 of 5.

### 5. Training on Spielberg costs nothing on the other tracks (observed)

- **Austin and Catalunya** are the same as base or better: medians 30 and 32.
- **Spielberg (now a training track)** has a median of 30 [27, 34].
- **Speed:** MTS laps are slightly slower again, 1.036 against 1.009 for base on common cells.

## Both folds together

| Fold | Held-out track | Austin-only, 3-track selection | Multi-track, 3-track selection | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Spielberg | 21 [0, 32] | **30 [29, 32]** | ambiguous (rule 1) |
| 2 | Silverstone | 28 [0, 30] | 27 [2, 28] | ambiguous (rule 2) |

**What holds in both folds:**
- Multi-track checkpoints are typically good on the held-out track (median 28–30 per checkpoint, against 7–12 for Austin-only).
- At least one Austin-only seed collapses on the held-out track (0 of 36).
- Multi-track training costs nothing at the selected checkpoint on Austin, though it is slightly slower overall.

**What does not hold:**
- The selected multi-track checkpoint is not reliably good on the held-out track. In fold 2 one seed's selected checkpoint fails a held-out corner that the training tracks cannot reveal.
- Whether multi-track training beats three-track selection alone at the selected checkpoint depends on the fold: yes for Spielberg, no for Silverstone.

**Overall:** multi-track training makes good held-out behaviour the *typical* checkpoint, but does not guarantee it for the *selected* one. With four tracks there is no independent track to select on. More tracks, or a selection signal that measures robustness rather than training-track qualifying, would be needed to turn the per-checkpoint robustness into a guarantee.
