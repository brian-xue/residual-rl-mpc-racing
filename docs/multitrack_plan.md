# Plan: multi-track training, leave one track out (Spielberg held out)

Branch `explore/multitrack-loto`. **Status: both folds done; results in [`results/analysis/multitrack/REPORT.md`](../results/analysis/multitrack/REPORT.md).**

## Question

Does training the wall-aware dual residual policy (δ, a) on **Austin + Catalunya + Silverstone** at the same time make it better on **Spielberg**, a track it never sees in training or checkpoint selection, than the Austin-only policy?

Secondary questions:
- Does it lose anything on Austin, which now gets only ~1/3 of the training samples?
- How much of any gain comes from **training** on more tracks, and how much from **selecting** the checkpoint on more tracks? The multi-seed study found that the Austin-only selection grid does not predict held-out performance (selection vs reporting correlation as low as −0.08).

## Baseline: what we compare against

The existing 3 seeds of the wall-aware dual config (`configs/pilot_st_pacejka_residual_dual_wall.yaml`), from `results/analysis/multiseed/report_laps.csv`. Qualifying laps out of 36 per track, selected checkpoints:

| Seed | Austin | Catalunya | Silverstone | **Spielberg** |
| --- | --- | --- | --- | --- |
| base/s0 | 28 | 28 | 13 | **12** |
| base/s1 | 31 | 32 | 30 | **21** |
| base/s2 | 28 | 30 | 12 | **21** |

Spielberg is 12–21 of 36. That spread over seeds (9 laps) is the noise floor any effect has to clear.

## Design: a 2 × 2 comparison

|  | Selected on Austin (current protocol) | Selected on Austin + Catalunya + Silverstone |
| --- | --- | --- |
| **Trained on Austin** | existing base s0–s2 (no new training) | A3: base s0–s2, re-selected (lapping only) |
| **Trained on Austin + Catalunya + Silverstone** | MT-A: diagnostic only | **MT: main variant** |

- **MT vs A3** isolates the effect of multi-track **training**: same selection rule, both have never seen Spielberg.
- **A3 vs base** isolates the effect of multi-track **selection**.
- Spielberg is used by nothing except the final report in every cell of the table.

## Changes to the code (all opt-in; existing configs give exactly the same results)

1. **`pilot/residual_env.py`**: new key `residual.tracks` (a list; default `[cfg["track"]]`).
   - Each env worker builds one `Track` and one Gym-Khana env per training track at start-up (per-track copy of `cfg` with `track` set).
   - On each `reset`, the track is drawn uniformly per episode (`options["track"]` can force one). Episodes are a fixed 600 steps, so uniform per episode means about equal samples per track.
   - The observation does not change. It is already track-independent, and the wall gaps are computed from the current track.
   - `info` gets the track name.
2. **`pilot/train_residual.py`**: per-track episode return and crash rate in `train_log.csv`, so we can see if one track dominates or is ignored.
3. **`pilot/protocol.py`**:
   - `--select-tracks` (default `Austin`, so existing runs reproduce): the selection grid becomes the same cells (a_lat {3.5, 5, 7, 8.5} × μ/μ₀ {1.0, 0.85, 0.65} × s {50, 250}) on each listed track: 24 laps per track, 72 for three tracks.
   - `select` subcommand that re-runs selection on an existing run without retraining (needed for A3 and MT-A).
   - `summarize` reports qualifying per track instead of the hard-coded "Austin vs held-out" split, and takes the training-track set from the config.
4. **New config** `configs/pilot_st_pacejka_residual_dual_wall_mt.yaml` (later: the fold is set with `train_residual --holdout`): the wall-aware dual config plus `residual.tracks: [Austin, Catalunya, Silverstone]` and a new `out`. Nothing else changes: same reward, PPO settings, 3M steps, μ and a_lat ranges.
5. **Run script and report** in `results/analysis/multitrack/` (later merged into `run_fold.sh <Holdout>`; `REPORT.md`), the same layout as `results/analysis/multiseed/`.

Before training, a check: an env with `tracks: [Austin]` must give bit-identical episodes to the current env with the same seed, and a short smoke run (≈50k steps) must show episodes on all three tracks.

## Runs

| Stage | What | Cost (from multiseed timings: ~6 min training, ~5 min protocol per run) |
| --- | --- | --- |
| 1 | MT seeds 0, 1, 2 at 3M steps (same total samples as base) | ~3 × 6–8 min training |
| 2 | Protocol on MT seeds with 3-track selection, plus `--all-ckpts` diagnostic | ~3 × 10 min (selection grid is 3× larger) |
| 3 | A3: re-select base s0–s2 with 3-track selection, report the chosen checkpoint | lapping only (reuses `protocol_report_all_ckpts.csv` where possible) |
| 4 | MT-A: re-select MT seeds on Austin only | lapping only |
| 5 (conditional) | MT at 9M steps (same **per-track** samples as base) | only if MT's checkpoint curves are still rising at 3M on Catalunya/Silverstone |
| 6 (conditional) | 2 more seeds for MT and base | only if the result falls in the "ambiguous" band below |

## Reporting

Same 144-lap reporting grid as the multi-seed study (4 tracks × a_lat {4, 6, 8} × sweep + fresh cells), unchanged qualifying rule.

**Primary metric:** Spielberg qualifying laps (of 36), per seed and median [range], MT vs A3.

**Secondary:**
- Spielberg crashes and where they happen (known hotspots s ≈ 42 and 304);
- Austin qualifying (does MT give up Austin?);
- Catalunya / Silverstone (in-distribution for MT, held out for base; expected to improve, so not evidence of generalisation on its own);
- mean lap-time ratio on cells completed by every run (MT shouldn't buy safety by driving slower);
- selection vs reporting correlation across checkpoints: does 3-track selection predict Spielberg better than Austin-only selection did?

## Decision rule (fixed before any run)

With 3 seeds and a 9-lap seed spread, we can only call large effects:

- **Helps:** MT's median Spielberg qualifying exceeds A3's **maximum**, and MT's Austin median is no more than 3 laps below base's.
- **Hurts:** MT's median is below A3's minimum.
- **Ambiguous** (anything else): run stage 6 (5 seeds each) before writing a conclusion.

## Risks and things to watch

- **One track dominating training:** if Silverstone episodes crash much more often, the value function and the advantages may be dominated by it. The per-track training log will show this. A fix (per-track reward normalisation) is out of scope unless it happens.
- **Track-specific tuning carried over from Austin:** `wall_margin_m` and the reference settings were chosen on Austin. The protocol already laps all four tracks with the same config, so this is unchanged, but it is a shared confound for every variant.
- **Only four tracks on disk:** with three for training and one for test there is no independent selection track. Spielberg must stay out of every choice, including the stage 5/6 decisions (those are based on training tracks only).

---

# Second fold: Silverstone held out (pre-registered 2026-11-20, before any run of this fold)

The same study with the roles of Spielberg and Silverstone swapped. It tests whether the first fold's result (multi-track training makes held-out results reliable from seed to seed) holds on a second unseen track. Silverstone is the longest track (446 m) and the one where Austin-only policies did worst (median 13 of 36 over 5 seeds, with Austin selection).

**Variants** (5 seeds each from the start; no stage-6 extension):

| Variant | Trained on | Selected on | Runs |
| --- | --- | --- | --- |
| base | Austin | Austin | existing base s0–s4 (no new laps) |
| A3S | Austin | Austin + Catalunya + Spielberg | base s0–s4, re-selected (tag `selACS`, later renamed `noSilverstone`) |
| **MTS** | Austin + Catalunya + Spielberg | Austin + Catalunya + Spielberg | new, `results_st_pacejka/multitrack_heldout_silverstone/s0–s4` |
| MTS-A | Austin + Catalunya + Spielberg | Austin | MTS runs re-selected (diagnostic) |

- **Everything else is unchanged:** reward, PPO settings, 3M steps, μ and a_lat ranges, selection cells, reporting grid and qualifying rule.
- **Silverstone is used by nothing** except the final report: not training, selection or any stage decision.
- **Main comparison:** MTS vs A3S on Silverstone qualifying (of 36).

**Decision rule.** This revises the first fold's rule, which could not separate "reliably near the ceiling" from "sometimes at the ceiling". It is fixed before any lap of this fold.
- **Helps:**
  - the MTS median is at least 6 laps above the A3S median,
  - **and** MTS's worst seed is at or above the A3S median,
  - **and** the MTS Austin median is no more than 3 laps below base's.
- **Hurts:** the MTS median is at least 6 laps below the A3S median.
- **Ambiguous:** anything else. There is no further extension; it gets reported as is.
- A one-sided Mann-Whitney p-value on the 5 vs 5 Silverstone counts is reported alongside, but it does not decide.

**Secondary questions:**
- Does including Spielberg instead of Silverstone in training change Austin and Catalunya?
- Does the late-training slowdown appear as often (2 of 5 in the first fold)?
- Compare with the first fold: is the Spielberg result of fold 1's MT (trained with Silverstone) matched by MTS's Silverstone result?
