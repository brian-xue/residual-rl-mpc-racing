# Plan: arm R with a short history window, so the policy can infer μ online

Branch `explore/history-window-mu`. **Status: Stages 0–2 done. The gate passed with F2, N = 5 ([Stage 0](#stage-0-results)); checks pass ([Stage 1](#stage-1-results)). In Stage 2, R-H is worse than R and R-H-zero on low-μ failures (median 24 vs 14 vs 14) and on qualifying laps, so the decision rule's verdict is no gain ([Stage 2](#stage-2-results)). Stages 3 (stress grid) and 4 (diagnostics) not run.** The design and decision rule below are fixed before any run, as in [`slip_rl_plan.md`](slip_rl_plan.md).

## Why

- Arm R (dual residual on SlipMPC) is the best controller so far: 128 of 144 qualifying laps, 127–128 over seeds ([`slip_rl_plan.md`](slip_rl_plan.md), Stages 2–4).
- **Every failure it has left is at μ/μ₀ 0.6–0.7** (Stage 5 §2: 9 spins and 35 walls over three seeds, none at μ/μ₀ ≥ 0.8). Neither SlipMPC nor the policy observes friction.
- The offline probe (`results/analysis/followup/REPORT.md` §3.2) found μ **partly inferable from 10 frames of history under high excitation** (MLP R² 0.52 on Austin), but:
  - the current observation alone has no friction signal (R² 0.01);
  - it did **not transfer** to Spielberg (R² ≤ 0);
  - its inputs were the policy observation and action only. It had no realized lateral acceleration and no model-mismatch signal.
- P′ already had `obs_history: 10` (r, β, r − vκ, δ, a_cmd, v) and its speed scale was flat in μ (Stage 5 §3). That failure was blamed on the reward, so it does not tell us whether history helps arm R.

**Main question:** does adding a short history window (and the right per-frame signals) to arm R reduce its low-friction failures without costing laps elsewhere, and does the policy's behaviour actually change with μ?

**Secondary questions:**
- Which per-frame signals carry μ: raw state (r, β), realized a_y, or the SlipMPC model's own one-step prediction error?
- Does the inference transfer to tracks the policy was not trained on?

## The saturation problem (has to be solved first)

Over every checkpoint of every run, only 136 of the 144 reporting-grid cells ever qualify, and Austin-only R already gets 128 (slip_rl_plan.md, "How much room there is"). The remaining 8 are μ/μ₀ 0.6 at a_lat 8, which Stage 0 argued no controller can drive within the 5% rule. **A history arm cannot show a win on qualifying laps on this grid.** So:

- **Primary metric moves to failures (spins + walls) at μ/μ₀ 0.6–0.7** on the existing 144-lap grid. R has 44 over three seeds; this is not saturated.
- Qualifying laps on the 144-lap grid become a **non-inferiority** check.
- A **low-μ stress grid**, fixed now and not used for any selection: 4 tracks × a_lat {5, 6, 7} × μ/μ₀ {0.5, 0.55, 0.6, 0.65, 0.7} × 2 start poses = 120 laps. Metric: safe laps (completed, |e_y| ≤ 0.5 m, lap-time rule ignored) and spins. μ/μ₀ 0.5–0.55 is outside the training range [0.6, 1.0], so this also tests extrapolation.

## Stage 0: probe before training (cheap, decides the feature set)

Re-run the offline probe, but on the thing we will actually train on, with better features, before spending any training compute.

- **Data:** R/s0–s2 selected checkpoints, stochastic actions, training-distribution episodes on **all four tracks** (≈ 200 episodes per track). Austin-only R has never trained on three of them, so this measures transfer directly.
- **Candidate per-frame signals** (all non-privileged):
  - **F0** (existing `obs_history`): r, β, r − vκ, δ, a_cmd, v.
  - **F1** = F0 + realized lateral acceleration a_y over the last control interval. Computed in the env from consecutive control-instant states, a_y ≈ v̄ (r̄ + Δβ/Ts), so neither kernel changes.
  - **F2** = F1 + **SlipMPC one-step innovation**: realized (r, β, v) minus what SlipMPC's nominal μ₀ model (`SlipMPC.predict`) predicts from the previous state and the command actually applied (MPC + residual). This is the model mismatch that μ causes, so it should be the most direct signal, and it is track-independent by construction.
- **Window lengths:** N ∈ {5, 10}.
- **Probe:** ridge and the same 2×64 MLP, episode-grouped CV. Report R² overall and by excitation (|v·r| < 2, > 4), **leave-one-track-out** (train on three tracks, test on the fourth), and Austin-only → other three.
- **Gate (fixed now):**
  - Pick the smallest feature set and window whose leave-one-track-out high-excitation R² is ≥ 0.3 on at least 3 of 4 held-out tracks, preferring F0 < F1 < F2 and 5 < 10 on ties (within 0.05).
  - **If no set passes, stop here** and write it up: history alone cannot carry μ across tracks for this car, and the next step is a model-based estimator, not RL observation design.
- **Cost:** lapping only, ≈ 30–45 min.

## Arms (3 seeds each, Austin training, Austin selection, as for R)

| Arm | Observation | What it tests |
| --- | --- | --- |
| R (existing) | arm-R observation | baseline; R/s0–s2 reused, no retraining |
| **R-H** | arm R + N frames of the Stage-0 feature set | the main question |
| **R-H-zero** | as R-H, but the history frames hold the *current* frame repeated N times | controls for the larger network input / extra dims, not the information in the past frames |

- Unchanged: arm-R config, PPO settings (MLP 128×128), 3M steps, μ range [0.6, 1.0], reward. True μ is never observed and never in the reward.
- **Not in this plan** (each is its own follow-up if R-H is "no gain"):
  - recurrent policy (`sb3_contrib.RecurrentPPO`), or a 1-D conv encoder over the window;
  - asymmetric critic that sees true μ (privileged value only);
  - a teacher–student / RMA-style setup with a supervised μ head;
  - multi-track training of R-H (only if R-H wins on Austin but fails to transfer).

## Code changes (all opt-in; existing configs give exactly the same results)

1. **`pilot/residual_env.py` `ObsExtras`:**
   - `residual.obs_history_features: [f0 | f1 | f2]` (default `f0`, the current 6 signals, so P′'s config is unchanged).
   - `f1` adds interval a_y; `f2` adds the SlipMPC innovation (needs `longitudinal.mode: slip`; raise otherwise).
   - `residual.obs_history_mode: past | repeat` for the R-H-zero control.
   - State needed across steps (previous state, previous applied command) lives in `ObsExtras` and is reset with it, so `_LapResidual` (eval) and the env (train) stay on the same code path.
2. **`pilot/mpc.py` `SlipMPC`:** a side-effect-free one-interval prediction helper for the innovation, built on `predict`. It must not touch `guess`, `v_lin` or the slack state.
3. **New configs:** `pilot_st_pacejka_slip_res_hist.yaml` (R-H) and `..._hist_zero.yaml` (R-H-zero), each derived from `pilot_st_pacejka_slip_res.yaml`.
4. **Stress grid:** a `stress` entry for `pilot.protocol` (or a small script under `results/analysis/history_mu/`), with the cells fixed above.
5. **Probe script:** `results/analysis/history_mu/probe.py`, reusing `followup/feasibility.py`'s ridge/MLP fits.
6. **Checks before training:**
   - `pilot.kernel_check` passes.
   - With the history on, env trajectories under a fixed action sequence are identical to those with it off (the observation is passive).
   - Two Austin laps through `ResidualPolicy.from_rc` reproduce the env's observations exactly (train / eval parity, as for P′).

## Runs

| Stage | What | Cost estimate |
| --- | --- | --- |
| 0 | Probe on R/s0–s2 rollouts, 4 tracks; pick features and N by the gate | ≈ 45 min |
| 1 | Code + checks; 300k-step smoke run of R-H (does crash rate fall as fast as R's 0.80 → 0.25?) | ≈ 5 min |
| 2 | R-H and R-H-zero, seeds 0–2, 3M steps | 6 × ≈ 41 min ≈ 4.1 h |
| 3 | Protocol report (selection + 144-lap grid) and the stress grid for R, R-H, R-H-zero, S-E | ≈ 1 h |
| 4 | Diagnostics: residual vs μ (does behaviour depend on μ?), in-policy μ probe on the policy's hidden layer, failure locations, Stage-5 slack columns | lapping only, ≈ 30 min |

Total ≈ 6.5 h; Stage 2 needs `caffeinate -s` on AC power.

## Decision rule (fixed now)

Primary metric: **spins + walls at μ/μ₀ 0.6–0.7 on the 144-lap grid**, median over 3 seeds. R's per-seed values are taken from the existing Stage 5 laps.

- **R-H wins** if all hold:
  1. its median low-μ failures are ≤ 0.7 × R's median (a 30% cut), and below R-H-zero's median;
  2. its median qualifying laps on the 144-lap grid are ≥ R's median − 2 (non-inferiority; the ceiling is 136);
  3. no held-out track (Catalunya, Silverstone, Spielberg) loses more than 2 qualifying laps against R's median on that track;
  4. **the mechanism shows:** in Stage 4, the policy's mean acceleration residual in corners differs between μ/μ₀ 0.6 and 1.0 by more than the seed-to-seed spread of R on the same quantity. Without this, a failure reduction is reported as "helps, cause unclear".
- **Information, not capacity:** if R-H and R-H-zero are within each other's seed range, the gain (if any) is not from inferring μ.
- **No gain:** R-H's median is within R's seed spread on the primary metric.
- **Secondary, no verdict:** stress-grid safe laps and spins (especially μ/μ₀ 0.5–0.55, outside the training range), lap-time ratio on cells all variants qualify, Stage-5 slack / slip-angle columns.

## Risks

- **Transfer.** The old probe overfitted Austin. Stage 0 measures this before training; Austin-only training of R-H may still learn Austin-specific μ cues. Multi-track R-H is the follow-up if so.
- **Excitation.** μ is only visible near the grip limit. The policy may learn it too late in a corner to help: it sees μ at turn-in, but must brake before. Stage 4 will look at *when* in a corner the residual starts to depend on μ.
- **Per-episode μ is constant.** Within-lap μ changes (patches) are not trained or tested. That is a separate study.
- **Compute.** The history adds 6–9 × N inputs to a 128×128 MLP, which is negligible next to SlipMPC's 0.74 ms per step.

## Stage 0 results

Script: [`results/analysis/history_mu/probe.py`](../results/analysis/history_mu/probe.py). Scores are in `results/analysis/history_mu/probe/probe_scores.csv`, the gate in `probe/gate.json` and the full printout in `fit.log`. The rollouts (`raw/`, 81 MB) are not committed.

**Two versions.**
- The first run (v1, kept in `probe_v1/`) computed the innovation with the *next* solve's acceleration command, not the one applied over the interval.
- v2 takes every signal from the environment's own `ObsExtras` frames, exactly as an R-H policy sees them. Its innovation uses the applied command: the previous solve's acceleration plus the residual.
- The gate verdict and the leave-one-track-out scores are the same in both, within 0.02 except Spielberg F1. The numbers below are v2.
- **v2 shows the v-innovation is identically 0.** The plant's v′ equals the applied acceleration, so the speed channel of the innovation carries no friction information. v1's v-innovation was only the change in the MPC's command. The channel stays in F2 as pre-registered; it is a constant input.

**Data:**
- 864 episodes, 470k control steps: the selected checkpoints of R/s0–s2, stochastic actions, 72 episodes per seed per track, on all four tracks.
- High excitation (|v·r| > 4 m/s²) is 33% of steps.
- Collection took 5 min with 8 workers. The fits (ridge and 2×64 MLP, 7 variants, 12 splits each) took 8 min.

**Probe inputs:** the current arm-R observation (38 dims), plus N frames of the signal set and their first differences. The "obs" row has no history.

### Leave-one-track-out, high excitation (the gate metric; MLP R²)

| Signals | N | Austin | Catalunya | Silverstone | Spielberg | Mean | Tracks ≥ 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| obs only | 0 | 0.25 | 0.29 | 0.24 | 0.25 | 0.26 | 0 |
| F0 (existing history) | 5 | 0.56 | 0.78 | 0.73 | 0.72 | 0.70 | 4 |
| F0 | 10 | 0.69 | 0.77 | 0.71 | 0.75 | 0.73 | 4 |
| F1 (+ a_y) | 5 | 0.79 | 0.82 | 0.76 | 0.70 | 0.77 | 4 |
| F1 | 10 | 0.78 | 0.76 | 0.78 | 0.76 | 0.77 | 4 |
| **F2 (+ innovation)** | **5** | 0.80 | 0.87 | 0.82 | 0.83 | **0.83** | 4 |
| F2 | 10 | 0.85 | 0.83 | 0.85 | 0.83 | 0.84 | 4 |

**Gate: PASS, F2 with N = 5** (v1: the same). All six variants pass the threshold. The tie band is 0.05 below the best mean (0.840), so a variant needs ≥ 0.790 to count as tied. F2 N=5 is the first variant in preference order inside the band.

### Austin-only training → unseen track, high excitation (MLP R²)

This is the setting R-H will train in (Austin only).

| Signals | N | Catalunya | Silverstone | Spielberg |
| --- | --- | --- | --- | --- |
| obs only | 0 | 0.11 | −0.13 | −0.29 |
| F0 | 5 / 10 | 0.44 / 0.17 | 0.19 / −0.19 | 0.01 / −0.30 |
| F1 | 5 / 10 | 0.58 / 0.52 | 0.45 / 0.42 | 0.40 / 0.41 |
| F2 | 5 / 10 | **0.70** / 0.59 | **0.57** / 0.37 | **0.47** / 0.43 |

In v1, F2 with N = 10 transferred better than N = 5 (0.57–0.68). In v2 it is the other way round. Single MLP fits differ by about 0.1 in this split, so the window length is not resolved by it. The pre-registered N = 5 stands (decided after v1, before v2).

### Findings

1. **A short history makes μ well inferable near the grip limit, on every track, once the probe trains on several tracks.**
   - Leave-one-track-out R² at high excitation is 0.70–0.84 for every signal set, against 0.26 with no history.
   - The earlier probe (`followup/REPORT.md` §3.2: 0.52 on Austin, ≤ 0 on Spielberg) trained on Austin only, used the kinematic-MPC wall policy, and had no a_y or innovation inputs.
2. **The existing signals (F0) do not transfer from Austin alone.** Trained on Austin, F0 is at or below the constant predictor on Spielberg (0.01 / −0.30), which repeats the old probe's failure. **F2 transfers from Austin to all three unseen tracks** (0.47–0.70 with N = 5); F1 is second (0.40–0.58).
3. **The innovation in β and r is the most informative signal.**
   - F2 gives the best score in every split.
   - At low excitation it is also the best set (leave-one-track-out R² 0.13–0.30), although every set is weak there.
4. **The relation is nonlinear.** Ridge is at R² ≈ 0 for every variant and split. A policy MLP can represent it; a linear μ estimator could not.
5. **At low excitation μ stays mostly hidden,** and Austin-only probes are below the constant there on unseen tracks. This was expected: friction shows only near the limit. It is the plan's "excitation" risk: the policy may learn μ at turn-in, which is late for braking.

## Stage 1 results

Script: [`results/analysis/history_mu/stage1.py`](../results/analysis/history_mu/stage1.py). Outputs: `stage1/checks.json` and `stage1/smoke_summary.csv`. The training log is in `results_st_pacejka/history_mu_smoke/RH/`.

### Code added (all opt-in; existing configs unchanged)

- **`pilot/residual_env.py` `ObsExtras`:**
  - `residual.obs_history_features: f0 | f1 | f2` (default `f0`);
  - `residual.obs_history_mode: past | repeat` (default `past`);
  - `f2` raises unless `longitudinal.mode: slip`.
- **The innovation** uses `SlipMPC.predict` directly. It is a pure function of (x, u, κ, T) and touches no solver state, so no new helper was needed (plan item 2).
  - Inputs: the state at the previous control instant, the mean realized steering angle over the interval, the applied acceleration (previous solve + residual, clipped to a_max) and κ at the previous instant.
  - At the first instant of an episode or lap there is no previous state, so a_y = v·r and the innovation is 0.
- **Scales:** a_y / 8 m/s²; innovation / (0.01 rad, 0.1 rad/s, 0.01 m/s). On the probe rollouts, the 1st–99th percentiles of the β and r channels are about ±1.1 and ±1.4–1.6.
- **Configs:** `configs/pilot_st_pacejka_slip_res_hist.yaml` (R-H) and `..._hist_zero.yaml` (R-H-zero). The observation has 38 + 5 × 10 = 88 dims.

### Checks (all pass)

| Check | Result |
| --- | --- |
| 1. Passive: R vs R-H and R vs R-H-zero, same seed and actions | identical simulator states and identical first 38 obs entries (359 steps, 3 episodes) |
| 2. P′'s `f0` history matches the pre-change formula | 0 mismatches in 286 instants |
| 3. History invariants, env (training) | `past` frames shift by one each instant; `repeat` frames all equal; first-instant innovation 0 |
| 3. History invariants, `run_lap` + `ResidualPolicy.from_rc` (evaluation) | same, on 2 Austin laps per variant (422 and 1,224 instants) |
| `pilot.kernel_check` | kernels identical |

### Smoke run (300k steps, seed 0; arm R's Stage 1 run for comparison)

Episode-weighted means over training windows, with exploration noise:

| | Crash rate: first 50k → 100–200k → last 100k | Return: first 50k → last 100k |
| --- | --- | --- |
| R (slip_rl Stage 1) | 0.80 → 0.35 → **0.26** | 157 → **420** |
| **R-H** | 0.84 → 0.50 → **0.34** | 135 → **397** |
| S-E, zero action (reference) | 0.50 | 319 |

- **R-H learns and goes below S-E within 300k steps,** but it is behind R at every window: 0.34 against 0.26 at the end.
- That difference is close to the window-to-window noise (±0.1) of a single seed. A plausible cause is the input growing from 38 to 88 dims, with no information yet useful early in training. R-H-zero (Stage 2) has the same input size, so it controls for this.
- **As with P′, no tuning on the smoke run.** Stage 2 runs R-H as designed.
- **Cost:** 260 s for 300k steps, against 194 s for R's smoke run (+34%). The step-cost check measured +8%, but under load from the probe fit, so the smoke time is the better estimate. Projected ≈ 55 min per 3M-step run, so **≈ 5.5 h for Stage 2's six runs** (the plan said 4.1 h).

## Stage 2 results

Script: [`results/analysis/history_mu/run_stage2.sh`](../results/analysis/history_mu/run_stage2.sh). Six runs, 3M steps each, each followed by `pilot.protocol run` (Austin selection, then the 144-lap grid). Run directories: `results_st_pacejka/history_mu/{RH,RH0}/s{0,1,2}`. Wall time 4.5 h (22:10–02:42), about 32–45 min of training and 6 min of protocol per run.

### Primary metric and the decision rule (144-lap grid, selected checkpoint)

| Arm | Seed | Selected | Spins + walls, μ/μ₀ 0.6–0.7 | Spins + walls, μ/μ₀ ≥ 0.8 | Qualifying (/144) | Aus / Cat / Sil / Spi | Train crash rate, last 300k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R | 0 | 2.25M | 14 | 0 | 128 | 30 / 32 / 34 / 32 | 0.12 |
| R | 1 | 3.0M | 16 | 0 | 128 | 32 / 32 / 32 / 32 | 0.15 |
| R | 2 | 2.5M | 14 | 0 | 127 | 32 / 32 / 31 / 32 | 0.12 |
| R-H | 0 | 1.75M | 27 | 6 | 111 | 30 / 28 / 26 / 27 | 0.23 |
| R-H | 1 | final | 24 | 0 | 120 | 30 / 30 / 30 / 30 | 0.17 |
| R-H | 2 | 0.75M | 14 | 0 | 124 | 32 / 30 / 30 / 32 | 0.22 |
| R-H-zero | 0 | 0.75M | 18 | 0 | 126 | 32 / 32 / 30 / 32 | 0.17 |
| R-H-zero | 1 | final | 12 | 0 | 122 | 30 / 30 / 33 / 29 | 0.14 |
| R-H-zero | 2 | 2.0M | 14 | 0 | 129 | 33 / 32 / 32 / 32 | 0.14 |
| **Median** | R / R-H / R-H-zero | | **14 / 24 / 14** | | **128 / 120 / 126** | | |

Against the rule:

1. Low-μ failures ≤ 0.7 × R's median (≤ 9.8) and below R-H-zero's median: **fails.** R-H's median is 24, against 14 for both R and R-H-zero.
2. Qualifying ≥ R's median − 2 (≥ 126): **fails** (120).
3. No held-out track loses more than 2 qualifying laps against R: **fails** (median 30 on each held-out track, against 32; seed 0 loses 4–6 on each).
4. Mechanism: not tested. Stage 4 cannot turn a fail on 1–3 into a win.

**Verdict: R-H does not help; it is worse than R on the primary metric, and outside R's seed range (14–16) on 2 of 3 seeds.** R-H-zero, with the same 88-dim input but no past frames, matches R (median 14 and 126). So the extra input size alone does not hurt; it is the past frames that do.

### Observations (no verdict)

- **R-H trains worse, not just generalises worse.** Its crash rate in the last 300k training steps (0.17–0.23, with exploration noise) is above R's (0.12–0.15) and R-H-zero's (0.14–0.17), on Austin, the training track. The smoke run showed the same gap early. So the policy did not learn to use the window within 3M steps, rather than learning a μ estimate that fails to transfer.
- **Selection lands early for some runs:** R-H seed 2 and R-H-zero seed 0 both select the 0.75M checkpoint; R's three seeds all select ≥ 2.25M.
- **R-H seed 0 has the only failures at μ/μ₀ ≥ 0.8** in any arm (6), i.e. a policy degraded in general, not one that is miscalibrated at low μ.
- The Stage 0 probe showed μ is decodable from F2 frames by a supervised MLP (held-out R² 0.83 at high excitation). PPO's 128×128 policy, trained on the race reward alone, does not get there in 3M steps. That gap (supervised decodability vs RL use) is the main lesson.
