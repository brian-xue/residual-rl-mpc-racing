# Plan: RL on top of the slip-aware MPC (SlipMPC + RL)

Branch `explore/slip-mpc-rl`. **Status: Stages 0–5 done. Arm R wins under the decision rule (median 128 of 144 qualifying laps, against 79–91 for the baselines) without driving past SlipMPC's slip limits. Arm P′ (learned speed reference) is no gain. See [Stages 2–4 results](#stages-24-results-3-seeds-per-arm-144-lap-reporting-grid) and [Stage 5 results](#stage-5-results-diagnostics-lapping-only).** The design and decision rule below are fixed before any run, as in [`multitrack_plan.md`](multitrack_plan.md).

## Why

The two best approaches so far are good at different things (README, four-track comparison; multiseed study):

| | Strengths | Weaknesses |
| --- | --- | --- |
| **S-E** (SlipMPC + friction-ellipse plan, no learning) | Most consistent across tracks: 10–11 of 18 sweep laps on every held-out track. Fastest on Spielberg and Silverstone. Few spins (8 of 72). | Assumes μ₀: its 0.12 rad rear limit sits close to the real peak at μ/μ₀ 0.6 (finding 4). Understeers into the wall at a_lat ≥ 6, because the 1 s horizon sees the needed braking too late (finding 3). |
| **Wall-aware dual residual** (kinematic MPC + RL (δ, a)) | Fastest on Austin and Catalunya. 40 of 72 sweep laps in total. | Results vary a lot by seed and track (Spielberg 0–32 of 36). Learned steering biases. It has to learn the turn-in spin from scratch. |

Feasibility check (`results/analysis/followup/REPORT.md` §3.1, same 72 cells):
- S-E **recovers 8** wall-policy failures, exactly the two corner-exit hotspots (Spielberg s ≈ 304, Silverstone s ≈ 176).
- S-E **loses 9** cells the policy completes: Austin a_lat 6–8 at s ≈ 55 / 151 / 286, and a_lat 8 on Spielberg and Catalunya.
- The two failure sets barely overlap, so a combination could do better than either one.
- Friction is partly inferable from 10 frames of history at high excitation (MLP R² 0.52; §3.2). This is the information S-E lacks.

**Main question:** does learning on top of SlipMPC keep S-E's held-out robustness while gaining the learned policy's speed and grip-limit performance?

**Secondary questions:**
- Where should learning act: on the MPC's *outputs* (residual) or on its *parameters* (μ estimate / slip limits)?
- Does the slip-aware base make RL less sensitive to seed (smaller spread than 0–32 of 36)?

## Design: two arms on the same base

Both arms use S-E as the base: `longitudinal.mode: slip`, friction-ellipse plan, the same SlipMPC weights and slip limits as published, and the Rust physics kernel (SlipMPC itself stays in Python).

### Arm R: output residual (the direct analogue)
- The policy adds to the SlipMPC commands: `δ = clip(δ_MPC + r_max·π_δ)`, `a = a_MPC + ra_max·π_a`. The plumbing already exists: `LapSim.apply(d_res, a_res)` adds `a_res` to `a_mpc`.
- **Risk, from README finding 5:** anything added after the MPC bypasses its slip limits. K·v²·κ on top of S made spins much worse (S-E 3 spins → S-E+ff 11).
- **Mitigations,** opt-in config keys:
  - smaller authority than on the kinematic base. Candidates are r_max 0.1 (vs 0.2) and ra_max 1.0 (vs 2.0); pick from a short smoke run, not the sweep.
  - keep the sideslip penalty from the dual config.
  - **R-proj variant, only if R spins:** project the residual back into the MPC's slip-angle limits using the predicted α_f and α_r at k = 1. This is a cheap linear check, with no second QP.

### Arm P′: learned speed reference (learning inside the MPC's constraints)

*Redesigned after the Stage 0 oracle checks. The original arm P put a friction estimate into the slip limits, and that knob turned out to hurt.*

- **What it does:** the policy does not touch the actuators. At each 20 Hz solve it sets one number, a **speed-reference scale** s ∈ [0.85, 1.05]. SlipMPC then tracks s·v_plan, with s²·a_plan as the acceleration reference (`SlipMPC.set_speed_scale`).
  - The slip limits stay at their published values. Every oracle setting with tighter limits was worse.
  - **Why it must be state-dependent:** the slowdown has to happen locally, in the corners, and be won back on the straights. A constant scale can only go 5% slower before the lap-time rule rejects the lap, and the oracle shows that buys at most +5 cells (below).
- **Constraints hold by construction:** every command still satisfies the MPC's constraints, so README finding 5 (added terms bypassing the slip limits) cannot apply.
- **Targets:** the μ mismatch (finding 4) and the late braking before high-a_lat corner exits (finding 3).
- **Observation:** the current dual + wall observation plus a short history (5–10 frames of r, β, a_y and the commands). The friction probe needed history to see μ.
- **Reward:** the same progress and crash terms. The true μ is never observed and never used in the reward.
- **Per-solve cost:** the scale changes only the QP's linear cost and the linearisation guess. There is no solver rebuild, so it costs the same as arm R.
- **How it differs from arm R's acceleration channel:** arm R adds the acceleration *after* the MPC. P′ changes what the MPC *aims for*, so steering and speed stay jointly optimal and within the slip limits.

### Not in scope (noted for later)
- Learning a dynamics correction inside the MPC model (learned tire residual or GP). This is a different method with its own study.
- Multi-track training. If an arm wins on Austin-only training, a multi-track fold (`run_fold.sh`) is the obvious follow-up.

## Code changes (all opt-in; existing configs give exactly the same results)

1. **`pilot/residual_env.py`**
   - Allow `longitudinal.mode: slip` as the training base. The feasibility check already did this in memory.
   - Add `a_mpc` (and, for R-proj and P, predicted α_f and α_r margins) to the observation when the base is slip.
   - New `residual.kind: output | speed` (default `output`). `speed` maps the action to the speed-reference scale s (arm P′).
   - Optional observation history (`residual.obs_history: N`).
2. **`pilot/mpc.py` `SlipMPC`**: `set_limit_scale(front, rear)` and `set_speed_scale(s)`, neither of which rebuilds the solver. **Both done in Stage 0;** at their defaults they reproduce the published S-E laps.
3. **`pilot/closedloop.py`**: per-solve speed-scale hook; log s and the slack-active flags per step.
4. **`pilot/eval_residual.py` / `protocol.py`**: run the policy with the slip base, so evaluation uses the same path as training.
5. **New configs:** `pilot_st_pacejka_slip_res.yaml` (arm R), `pilot_st_pacejka_slip_speed.yaml` (arm P′), each derived from `..._residual_dual_wall.yaml`.
6. **Check before training:** the env with a zero residual (arm R) or s = 1 (arm P′) must give S-E's published laps on the 72 sweep cells, bit-for-bit or to rounding.

## Compute (the main practical risk)

- Measured: SlipMPC adds **7.5× CPU per million env steps** (1.5 ms vs 0.17 ms per step). Projected **≈ 30 min per 3M-step run**, vs ≈ 6 min now.
- **Stage 0 measures the real `SubprocVecEnv` scaling** before anything else.
- **Decision point:**
  - ≤ 45 min per run: stay in Python.
  - Worse than that: either cut to 2M steps (checkpoints from 1.25M already did best on the kinematic base), or port SlipMPC to `pilot_core`. The port is a separate decision; it is several days of work and needs its own bit-identity check.
- Arm P′ can also update less often than every step (hold s for 4 steps) without changing the MPC rate.

## Runs

| Stage | What | Cost estimate |
| --- | --- | --- |
| 0 | Zero-residual reproduction check; `SubprocVecEnv` throughput with the slip base | < 30 min |
| 1 | Smoke runs (300k steps) for R and P: does the training crash rate fall? Pick R's authority | ~2 × 5 min |
| 2 | Arm R, seeds 0–2, 3M steps, Austin training and Austin selection (published protocol) | ~3 × 37 min |
| 3 | Arm P′, seeds 0–2, same protocol | ~3 × 37 min |
| 4 | Protocol report on all checkpoints: 144-lap reporting grid (4 tracks) and the 72-cell sweep | ~6 × 15 min |
| 5 | Diagnostics: the 8 + 9 feasibility cells; spin/wall location; s vs true μ (P′); residual size vs slack activity (R) | lapping only |

Total ≈ 4–5 h wall time, about 3× the multi-track study.

## Baselines

- **S-E**, deterministic, no seed: 72-cell sweep and the 144-lap grid. The grid needs S-E on the reporting grid; check whether it is already in `results/analysis/multiseed/`, and run it otherwise.
- **Wall-aware dual residual, base s0–s4** (`results/analysis/multiseed/report_laps.csv`), same protocol.
- **S-E at 0.97 × plan speed** (added after Stage 0). A constant 3% slower reference is a free, no-learning improvement: 44 of 72 sweep cells, against 39 for S-E. Any learned arm has to beat this, not just S-E. The 0.97 was picked on the 72 sweep cells themselves, so its 44 is optimistic; it will be re-measured on the 144-lap grid with no further tuning.

## Decision rule (fixed now; amended once after Stage 0, before any training)

*Amendment:* the S-E at 0.97 × speed baseline was added, because the Stage 0 oracle showed it beats S-E for free. Safe laps (laps failing only the 5% lap-time rule) are reported next to qualifying laps.

Primary metric: qualifying laps on the 144-lap reporting grid. For each arm, take the **median over 3 seeds**.

- **Arm wins** if all three hold:
  1. the median is above S-E, **S-E at 0.97 × speed** and the median of the dual-residual base;
  2. no held-out track (Spielberg, Catalunya, Silverstone) is more than 3 laps (of 36) below S-E on that track;
  3. spins summed over the grid are ≤ S-E's.
- **Faster but less robust:** the median of fastest-lap time over qualifying cells improves by ≥ 1 s on Austin, but (2) fails. Report it as a trade-off, not a win.
- **No gain:** the median is within the seed spread of S-E. This is still informative, because it means the slip-aware base already captures what the residual learned.
- **R vs P′:** compare their medians directly. If R wins only by bypassing the constraints (more cells with active slack than S-E), say so.

## Decisions

1. **Both arms run** (R and P′, which replaces P after Stage 0), 3 seeds each.
2. **Austin-only training and Austin selection first**, matching the published protocol. A multi-track fold is a follow-up only if an arm wins.
3. **Python vs Rust SlipMPC:** decided by the Stage 0 throughput measurement, using the rule in [Compute](#compute-the-main-practical-risk) (≤ 45 min per run: stay in Python).

## Stage 0 results

Script: [`results/analysis/slip_rl/stage0.py`](../results/analysis/slip_rl/stage0.py); outputs in `results/analysis/slip_rl/stage0/`.

**Code added (all opt-in; existing configs unchanged):**
- `SlipMPC.set_limit_scale(front, rear)`;
- `longitudinal.slip_mpc.oracle_mu_limits`, a **privileged** diagnostic that scales the limits by the true μ/μ₀;
- `configs/pilot_st_pacejka_slip_res.yaml`: the arm R config, i.e. the wall-aware dual config with `longitudinal.mode: slip` and a new `out`.

`pilot.kernel_check` still passes.

### 1. Reproduction: the slip base in the residual path is exactly S-E

The 72 published S-E sweep cells (4 tracks × a_lat {4, 6, 8} × μ/μ₀ {1.0, 0.8, 0.6} × 2 start poses) were run through `run_lap` with a residual that always returns (0, 0):

| Kernel | Status mismatches | Max \|Δ\| (progress, e_y, a_y) | Max \|Δ lap time\| | Qualifying / spin / wall |
| --- | --- | --- | --- | --- |
| python | 0 of 72 | 5e-5 (the published CSV's 4-decimal rounding) | 4e-12 s | 39 / 8 / 24 (published: 39 / 8 / 24) |
| rust (training config) | 0 of 72 | 5e-5 | 4e-12 s | 39 / 8 / 24 |

### 2. Training throughput: stay in Python

`pilot.train_residual` with 8 `SubprocVecEnv` workers, timed at 82k and 246k steps. The slope removes the ≈ 18 s start-up cost.

| Base | Marginal time per env step | Projected 3M-step run |
| --- | --- | --- |
| kinematic MPC (Rust) | 96 µs | 5.1 min (matches the ≈ 6 min of earlier runs) |
| **SlipMPC (Python)** | 743 µs (7.7×) | **37 min** |

This is under the 45 min threshold, so SlipMPC stays in Python and needs no Rust port. Six runs (2 arms × 3 seeds) need ≈ 3.7 h of training, plus the protocol runs.

### 3. Oracle μ in the slip limits: fewer spins, more walls

S-E with both slip limits multiplied by the **true** μ/μ₀, same 72 cells:

| | Qualifying | Spin | Wall | Other |
| --- | --- | --- | --- | --- |
| S-E (published) | **39** | 8 | 24 | 1 |
| S-E, oracle-μ limits | 27 | **0** | 42 | 3 |

- Every spin is gone, and every one became a wall collision.
- All 12 cells lost are qualifying S-E laps at μ/μ₀ 0.6 (a_lat 4) or 0.8 (a_lat 6), turned into walls or lateral-error failures.
- No cell improved. By friction level, oracle qualifying is 17 / 8 / 2 of 24 at μ/μ₀ 1.0 / 0.8 / 0.6.

**Why:** scaling the limits by μ keeps the linear model as accurate as at μ₀ (87% of the linear force at the limit). But the limits are then far below the real grip: the Pacejka rear peak is at ≈ 0.38 rad at μ/μ₀ 0.6, against a scaled limit of 0.072 rad. The speed plan does not slow down to match, so the MPC holds the planned speed, cannot turn hard enough, and runs wide.

**Consequences for the plan:**
- **Arm P as first designed (μ̂ in the limits only) would fail even with a perfect estimate.** §4 tested limits and speed together.
- **README finding 4 needs a correction** (computed from the tire formula, not yet measured in the simulator): the published 0.12 rad rear limit is not close to the real peak at μ/μ₀ 0.6: it is at 73% of the linear force, against a peak at ≈ 0.38 rad. S-E's low-μ spins come from the linear model *over-predicting* force inside the limit, not from the limit sitting at the peak. I have not edited the README; that is for you to decide.
- **Arm R is not affected** and is ready for its Stage 1 smoke run, once the `a_mpc` observation is added (code change 1).

### 4. Oracle μ in the slip limits *and* the speed reference

Each lap holds a constant slip-limit scale L and speed-reference scale S, set from the **true** μ/μ₀ (privileged):
- L ∈ {1, √μ, μ}, S ∈ {1, 0.97, 0.94, √μ};
- 648 laps, after removing combinations that coincide at μ = 1;
- `stage0.py oracle2`; outputs `oracle2_*.csv`.

The lap-time rule still compares against the unscaled plan, so any S < 0.95 fails it. **Safe** counts laps that fail nothing except that rule.

| L (limits) | S (speed) | Qualifying | Spin | Wall | Safe | Safe at μ/μ₀ 0.6 (of 24) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 (= S-E) | 39 | 8 | 24 | 39 | 8 |
| **1** | **0.97** | **44** | 5 | 23 | 44 | 8 |
| 1 | 0.94 | 0 | 1 | 19 | 50 | 12 |
| **1** | **√μ** | 17 | **0** | **6** | **63** | **24** |
| √μ | 1 | 35 | 0 | 36 | 35 | 8 |
| √μ | 0.97 | 42 | 0 | 30 | 42 | 8 |
| √μ | √μ | 17 | 0 | 10 | 60 | 24 |
| μ | 1 | 27 | 0 | 42 | 27 | 2 |
| μ | 0.97 | 38 | 0 | 34 | 38 | 8 |
| μ | √μ ("both") | 17 | 0 | 16 | 55 | 20 |

The first row matches the published S-E (39 / 8 / 24), so the new keys change nothing at their defaults.

**Findings**
1. **"μ in both" does not beat S-E on qualifying.** The doubly scaled run (L = μ, S = √μ) qualifies 17 of 72, all of them at μ/μ₀ 1.0, where the run is just S-E. At 0.6 and 0.8 a √μ slowdown is 11–23%, and the lap-time rule rejects every such lap. It is safer, though: 55 safe laps against S-E's 39, and no spins.
2. **The slip limits are the wrong knob.** At every speed scale, tightening the limits lowers the safe count. With S = √μ, it goes 63 → 60 → 55 as L goes 1 → √μ → μ. The published limits are already conservative enough; tighter ones only take away cornering force.
3. **Speed is the right knob.**
   - With the published limits and the true-μ speed scale (L = 1, S = √μ), there are no spins, only 6 walls, and **every μ/μ₀ 0.6 cell is safe** (24 of 24, against 8 for S-E).
   - But that slowdown is constant over the lap, so it is too slow to qualify.
4. **A perfect constant-per-lap choice is capped at 44 of 72 qualifying.**
   - That is only +5 over S-E: 3 cells at μ/μ₀ 1.0 and 2 at 0.8.
   - All 5 come from the plain S = 0.97 setting, which loses no cell.
   - No constant setting recovers a μ/μ₀ 0.6 cell under the rule.
   - 70 of 72 cells can be driven *safely* by some constant setting, but only by going slower than the rule allows.

**Consequences**
- **Arm P becomes P′:** a state-dependent speed-reference scale, with the limits left alone (see the arm P′ section above).
  - The gap P′ is after: 63–70 of 72 cells *can* be driven safely, but only 44 within the 5% rule with a constant scale.
  - Only a policy that slows down locally can close it.
  - This is a real learning problem, with a clear oracle ceiling.
- **New baseline:** S-E at 0.97 × plan speed qualifies 44 of 72, more than any controller so far (the best learned policy has 40). The decision rule now requires beating it.
- **Low friction is largely capped by the rule itself.** At μ/μ₀ 0.6 the grip limit is about 6.2 m/s². The a_lat 8 cells, and probably a_lat 6, cannot be driven within 5% of the plan by any controller. The sweep's μ/μ₀ 0.6 row will measure mostly safety, not speed. I'll report safe laps alongside qualifying for every arm.

## Stage 1 results

Script: [`results/analysis/slip_rl/stage1.py`](../results/analysis/slip_rl/stage1.py). Training logs are in `results_st_pacejka/slip_smoke/<variant>/`, and the summary in `results/analysis/slip_rl/stage1/`.

### Code added (all opt-in)

- **`pilot/residual_env.py`:**
  - `residual.obs_a_mpc` (the base MPC's acceleration command);
  - `residual.obs_history: N` (last N frames of r, β, r − vκ, δ, a_cmd, v);
  - `residual.kind: speed` (arm P′: one action → speed scale s, with 0 → 1.0, −1 → 0.85, +1 → 1.05);
  - an optional `reward.dspeed` chatter penalty.
- **`ResidualPolicy.from_rc`:** evaluation (`eval_residual`, `protocol`) uses the same observation and action mapping as training.
- **Configs:**
  - `configs/pilot_st_pacejka_slip_res.yaml` (arm R) gains `obs_a_mpc: true`;
  - new `configs/pilot_st_pacejka_slip_speed.yaml` (arm P′: `obs_history: 10`, `dspeed: 5`, untuned).
- **Timing:** P′'s action sets the scale for the *next* solve, a 50 ms lag. Training and evaluation use the same timing.

### Checks

- `pilot.kernel_check` passes.
- With `obs_a_mpc` on, env trajectories are identical to those with it off (max |Δ| 0 over 700 steps).
- P′ with action 0 gives trajectories identical to the zero-residual slip base, in the env and in the lap path. Two Austin S-E sweep laps reproduce exactly through `ResidualPolicy.from_rc`.

### Zero-action reference: S-E on the training distribution

These are 240 episodes of a_lat 3–9, μ/μ₀ 0.6–1.0, random start, with action 0 and no exploration noise:
- **crash rate 0.50**, episode return 319, mean length 353 steps;
- the crashes are 61 lateral-error and 58 wall terminations, and only 1 spin.

The policies start from this reference, plus exploration noise.

### Smoke runs (300k steps, seed 0, about 3.5 min each)

Episode-weighted means over training windows. Training episodes include exploration noise (std ≈ 0.2 in action units).

| Variant | Crash rate: first 50k → 100–200k → last 100k | Return: first 50k → last 100k |
| --- | --- | --- |
| S-E, zero action (reference) | 0.50 | 319 |
| **R** (r_max 0.2, ra_max 2.0) | 0.80 → 0.35 → **0.25** | 157 → **423** |
| R-low (r_max 0.1, ra_max 1.0) | 0.56 → 0.51 → 0.42 | 277 → 363 |
| P′ (speed scale) | 0.54 → 0.53 → 0.48 | 293 → 322 |

For comparison, the kinematic wall-aware dual policy logged crash rates of 0.31–0.67 over the same 250–400k steps.

**Findings**
1. **Arm R learns fast and goes below S-E.** Its crash rate halves relative to S-E within 300k steps (0.25 vs 0.50, *with* exploration noise). At first the exploration noise at full authority makes it much worse (0.80): the noise bypasses the slip limits, as finding 5 predicts, and training then removes it.
2. **R-low learns more slowly and is not safer early on.** Its first-50k crash rate is still 0.56, so the smaller authority buys little safety and costs learning speed. **Decision: arm R keeps r_max 0.2 / ra_max 2.0** (the config as is). R-proj is not needed yet, because training removes the spins. Whether R's laps bypass the slip limits will be checked in the Stage 5 diagnostics (slack activity).
3. **P′ barely moves in 300k steps** (0.54 → 0.48, within the window-to-window noise of ±0.1).
   - Exploration noise does not hurt it (0.54 vs 0.50), as expected when everything stays inside the MPC's constraints.
   - It also gets a weak learning signal: a speed change acts only through the next solve and the speed dynamics.
   - This is a warning, not yet a failure: the kinematic policies took about 1M steps to settle.
   - **Decision: run P′ as designed for 3M steps** (no tuning on the smoke run, which would be tuning on seed 0). If its 3-seed result is "no gain", a P′ variant with a larger range or a smaller `dspeed` is the follow-up. It would be reported separately, not swapped in.

**Cost of Stages 2–3:** 300k steps took 3.2–3.7 min, consistent with the ≈ 37 min per 3M-step run measured in Stage 0. Six runs need ≈ 3.7 h sequentially. The machine has 10 cores and each run uses 8 workers, so running two at once would not help.


## Stages 2–4 results (3 seeds per arm, 144-lap reporting grid)

Each run was trained on Austin for 3M steps, with the checkpoint chosen on the Austin selection grid (`pilot.protocol`). The tables below come from `results_st_pacejka/slip_rl/compare/`. The dual-residual base comes from `results/analysis/multiseed/report_laps.csv` and is on the same 144 cells. That file has seeds s0–s2 only, not s0–s4 as the plan said, so its median is over 3 seeds.

The Mac slept for about 7 h during R/s1 and P/s0 training. Sleep pauses training without changing what it computes, so the results are not affected; it only made the wall-clock times longer.

| | Qualifying (median; seeds) | Austin | Catalunya | Silverstone | Spielberg | Spin | Wall | Safe (completed, \|e_y\| ≤ 0.5 m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-E | 79 | 18 | 21 | 20 | 20 | 9 | 49 | 79 |
| S-E at 0.97 × speed | 89 | 20 | 22 | 26 | 21 | 5 | 44 | 90 |
| Dual-residual base | 91 (81, 114, 91) | 28 | 30 | 13 | 21 | 13 per seed | — | 81–114 |
| **R** (residual on SlipMPC) | **128** (128, 128, 127) | 32 | 32 | 32 | 32 | **3, 2, 4** | 11, 14, 10 | 127–128 |
| P′ (speed scale) | 46 (52, 32, 46) | 20 | 12 | 12 | 0 | 8, 8, 5 | 16, 21, 50 | 87–112 |

(Track columns are medians over seeds.)

**Decision rule**
- **Arm R wins.** It passes all three conditions:
  1. Its median of 128 is above S-E (79), S-E at 0.97 (89) and the dual base (91). Its worst seed (127) beats every baseline.
  2. Every held-out track is at 31–34 of 36, against S-E's 20–21.
  3. Its spins (2–4) are fewer than S-E's 9.
- The seed spread is 1 lap (127–128), against 33 for the dual base. R gets the dual residual's speed and SlipMPC's consistency at once, which is what the plan was looking for.
- **P′ is a no gain: it is worse than S-E** (median 46). It does not crash more. It fails by being **too slow**: 41–73 laps fail only the 5% lap-time rule, so it has many safe laps (87–112) but few qualifying ones.

**Where R's gain comes from** (qualifying laps by cell, mean over seeds, out of 8 laps per cell, 16 at μ/μ₀ 1.0):

| μ/μ₀ | a_lat 4: S-E / R | a_lat 6: S-E / R | a_lat 8: S-E / R |
| --- | --- | --- | --- |
| 1.0 | 16 / 16 | 16 / 16 | **1 / 16** |
| 0.9 | 8 / 8 | 8 / 8 | **0 / 8** |
| 0.8 | 8 / 8 | 6 / 8 | **0 / 8** |
| 0.7 | 8 / 8 | **0 / 8** | 0 / 2.3 |
| 0.6 | 8 / 8 | **0 / 5.3** | 0 / 0 |

- **Most of the gain is at a_lat 8** (+40 laps). S-E understeers into the wall there because its 1 s horizon sees the braking too late (README finding 3). R fixes this at μ/μ₀ ≥ 0.8.
- **The rest is at low friction with a_lat 6** (+13).
- R still fails μ/μ₀ 0.6 at a_lat 8, as do all the baselines. This matches the Stage 0 argument that those cells cannot be driven within 5% of the plan.
- R laps at a mean of 0.98 × plan time on completed laps (S-E: 1.00). So the policy learned to go **faster** than the plan where grip allows, not slower. Two seeds get there by accelerating harder on the straights; the third brakes more in the corners (Stage 5, below).

**Checkpoint robustness (all-checkpoint diagnostic; selection never saw it)**
- **R:** every checkpoint from 250k to 3M steps, on all three seeds, scores 112–134 on the reporting grid. Every one is above all the baselines, so R's win does not depend on checkpoint selection. The 250k checkpoints already score 112–117.
- **P′ gets worse as it trains:** s0 goes 87 → 95 (noisy), s1 goes 90 → 45, and s2 goes 82 → 0 (it is at 0 from 1.25M on). Its early checkpoints (82–90) are about level with S-E at 0.97 × speed. Later the policy drifts toward slowing down (most failures are too-slow laps), which suggests the reward values safety over the lap-time rule. **The likely problem is the reward, not the action space**; Stage 5 §3 confirms it. A follow-up would add a lap-time term to the reward, or tie the lower end of the range to 0.95. As the plan says, that variant would be reported separately.

See Stage 5 for the slack, speed-scale and failure-location checks.

## Stage 5 results (diagnostics, lapping only)

`results/analysis/slip_rl/stage5.py` re-laps the 144-lap grid with S-E, S-E at 0.97 and the **selected** checkpoint of each of the six runs, with two new passive log columns: the speed scale each solve tracked, and SlipMPC's soft-limit slacks (`SlipMPC.last_slack`). **Every one of the 1,152 laps has the same outcome as in the protocol report**, so the logging changes nothing. The per-lap table is `stage5/laps.csv` and the full printout is `stage5/analysis.txt`. The 20 Hz traces are in `results_st_pacejka/slip_rl/stage5/traces/` and are not committed.

Slip angles are the plant's, measured at the control instants: α_f = δ − β − l_f r/v and α_r = −β + l_r r/v. They are compared against the MPC's soft limits (0.25 / 0.12 rad). Those limits are not the tire peaks, which are higher.

### 1. The 8 + 9 feasibility cells

| Cells | S-E | S-E at 0.97 | R (s0 / s1 / s2) | P′ (s0 / s1 / s2) |
| --- | --- | --- | --- | --- |
| 8 that S-E fixed and the wall policy failed (corner-exit hotspots) | 8 | 8 | **8 / 8 / 8** | 2 / 0 / 0 |
| 9 that the wall policy fixed and S-E failed (Austin a_lat 6–8, a_lat 8 elsewhere) | 0 | 3 | **9 / 9 / 9** | 2 / 2 / 2 |

**R gets the union of the two failure sets on every seed.** This is the outcome the feasibility check said a combination could reach. P′ loses even the cells S-E already had.

### 2. Where the spins and walls are

- **R has no spin or wall at μ/μ₀ ≥ 0.8** on any seed. There, S-E has 30 walls and S-E at 0.97 has 22.
- All of R's failures are at μ/μ₀ 0.6–0.7: 9 spins over three seeds, all at 0.6, and 35 walls.
- **The locations are spread out.** The most common is Austin s ≈ 152 (8 of 44 over three seeds), and the old corner-exit hotspots are nearly gone: Spielberg 304 has 0, against 2 for S-E.
- P′ fails all over the place, most of all at Austin 152 (24).

Failures by friction, summed over the runs in each variant (S-E and S-E at 0.97 have one run each; R and P′ have three):

| μ/μ₀ | S-E spin / wall | S-E at 0.97 | R | P′ |
| --- | --- | --- | --- | --- |
| 0.6 | 8 / 8 | 5 / 11 | 9 / 21 | 20 / 28 |
| 0.7 | 1 / 11 | 0 / 11 | 0 / 14 | 1 / 25 |
| 0.8–1.0 | 0 / 30 | 0 / 22 | **0 / 0** | 0 / 34 |

### 3. P′'s speed scale does not track friction

| Mean scale | μ/μ₀ 0.6 | 0.8 | 1.0 | a_lat 4 | a_lat 8 | Corners (\|κ\| > 0.05) | Straights (\|κ\| < 0.01) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P′ s0 | 0.952 | 0.952 | 0.954 | 0.975 | 0.933 | 0.954 | 0.943 |
| P′ s1 | 0.948 | 0.942 | 0.943 | 0.959 | 0.931 | 0.950 | 0.932 |
| P′ s2 | 0.949 | 0.951 | 0.953 | 0.960 | 0.947 | 0.963 | 0.939 |

- **The scale is flat in μ** (spread ≤ 0.01 between μ/μ₀ 0.6 and 1.0 on every seed). The policy does not infer friction, although the friction probe found μ partly inferable from 10 frames of history.
- It responds to the **speed level** instead: it slows down more at a_lat 8 than at 4. That is information it already gets from the observation.
- **It slows down more on the straights than in the corners**, the opposite of the design idea. Slowing on a straight costs lap time and buys no grip.
- Its mean of 0.94–0.95 sits right at the 5% lap-time limit, so most laps fail as too slow.
- **This confirms the Stage 2–4 reading:** P′ found a "slow down everywhere a bit" optimum. The reward does not pay for lap time against the plan, and there is no friction signal to make a local slowdown worth learning. As noted, a follow-up variant would need a lap-time term.

### 4. R does not bypass SlipMPC's slip limits

Mean fraction of control instants:

| | Stage-1 slack active | Plant \|α_r\| > 0.12 | Plant \|α_f\| > 0.25 | v / v_plan | Same three, on the 79 cells S-E qualifies |
| --- | --- | --- | --- | --- | --- |
| S-E | 0.035 | 0.20 | 0.015 | 1.00 | 0.001 / 0.06 / 0 |
| S-E at 0.97 | 0.020 | 0.17 | 0.007 | 0.97 | 0 / 0.03 / 0 |
| R s0 / s1 / s2 | **0.008 / 0.007 / 0.011** | 0.15 / 0.13 / 0.15 | 0.024 / 0.022 / 0.014 | 1.01 / 0.98 / 1.02 | 0 / 0.03–0.06 / 0 |
| P′ s0 / s1 / s2 | 0.007 / 0.009 / 0.017 | 0.14 / 0.15 / 0.14 | ≤ 0.008 | 0.94–0.95 | 0 / 0.01–0.03 / 0 |

- **Over all 144 laps,** R's laps spend *less* time with an active slack, or beyond the rear limit, than S-E's, while going as fast as the plan or faster. README finding 5's risk, that terms added after the MPC drive past its limits, does not show up here.
- **On equal laps** (the cells S-E qualifies), R's rear slip matches S-E's (0.03–0.06 vs 0.06), and neither touches the slack.
- **The front axle is the one exception.** R is beyond 0.25 rad at 1.4–2.4% of instants, against S-E's 1.5%, with a higher peak (0.22–0.25 vs 0.21). This is small, and it is in the failing low-friction laps, not the qualifying ones.
- **The steering residual is bigger where the MPC is at its limits:** |δ_res| averages 0.025–0.057 rad when the stage-1 slack is active, against 0.017–0.033 elsewhere. It acts mostly near the grip limit, as a correction, not as a constant bias.
- **The acceleration residual differs by seed:**
  - s0 and s2 add +0.42 and +0.28 m/s² on the straights; this is where their speed above the plan comes from.
  - s1 brakes in the corners (−0.48 m/s²) and runs at 0.98 × plan speed.
  - All three seeds score 127–128, so there are at least two strategies that work.
- **Before a spin**, the stage-1 slack is active 73% of the last second for R, against 48% for S-E. R's remaining spins happen where the MPC itself already predicts a violation: R is not causing them by going past an MPC that saw no problem.

### What Stage 5 adds to the decision

R's win is not bought by breaking the constraints. It recovers both feasibility failure sets on every seed and removes every failure at μ/μ₀ ≥ 0.8. What is left is low-friction driving (μ/μ₀ 0.6–0.7), where no controller here has a friction estimate. The natural next steps are:
- the multi-track fold (`run_fold.sh`) for arm R, to test held-out robustness under multi-track training;
- an R variant with a friction-informative observation, for the μ/μ₀ 0.6 cells.

# Next: arm R in a leave-one-track-out fold (plan, fixed before any run)

*Written 2026-11-22, after Stage 5. Nothing has been run for it, and no lap of the held-out track under multi-track training has been seen.*

## Question

Arm R was trained on Austin alone. Does training it on three tracks change what it does on the fourth, unseen one? For the kinematic dual residual, multi-track training made held-out results reliable from seed to seed (Spielberg 29–32 of 36 against 0–32; `results/analysis/multitrack/REPORT.md`). Austin-only R is already reliable (31–34 per held-out track, 127–128 in total), so the question here is different from that study's: **does R lose anything, or gain lap time, when Austin is no longer the only training track?**

## How much room there is: the grid is nearly saturated

Over every checkpoint of every run in this study and the multi-seed study, **only 136 of the 144 cells ever qualify, exactly 34 per track**. The 8 that never qualify are μ/μ₀ 0.6 at a_lat 8, two per track, and Stage 0 already argued they cannot be driven within 5% of the plan.

Austin-only R's selected checkpoints score 32 per track (median). That leaves **about 2 laps per held-out track**, within its checkpoint-to-checkpoint range of 28–34. So:
- **A "helps" verdict on qualifying laps is not reachable.** A 2-lap gain cannot be told apart from checkpoint noise with 3 seeds. The rule below therefore tests *non-inferiority* on qualifying laps and puts any gain into a secondary, continuous metric (lap time) that is not saturated.
- **The same saturation limits any further arm-R study on this grid**, including the friction-aware variant: Austin-only R already gets 128 of 136. A harder reporting grid (a_lat 9–10, μ/μ₀ 0.5, or a 2% lap-time rule) is the alternative to this fold. It would be a new grid, fixed before use, and it is not part of this plan.

## Design

Held-out track: **Spielberg** (the original fold).
- It is R's weakest track: 28–32 over its checkpoints, and it loses all 6 μ/μ₀ 0.7, a_lat 8 laps.
- It keeps the comparison with the dual residual's MT result (29–32) direct.
- **Silverstone** (fold 2) is run only if the Spielberg result is "hurts" or borderline (see the rule). Then it checks whether that holds on a second track.

| Variant | Trained on | Checkpoint selected on | Runs |
| --- | --- | --- | --- |
| R-base | Austin | Austin | the existing R/s0–s2 |
| **R-A3** | Austin | Austin + Catalunya + Silverstone | R/s0–s2 re-selected (lapping only) |
| **R-MT** | Austin + Catalunya + Silverstone | Austin + Catalunya + Silverstone | new, seeds 0–2 |
| R-MT-A | Austin + Catalunya + Silverstone | Austin | the R-MT runs re-selected |

- The main comparison is **R-MT vs R-A3**, which separates the training effect from the selection effect, as in the multi-track study.
- **3 seeds**, as in Stages 2–4. The multi-track study needed 5 because its baseline varied by 32 laps; R's varies by 1.
- **Unchanged:** the arm-R config, PPO settings, 3M steps, the 144-lap grid and the qualifying rule. MT draws one training track per episode, so Austin gets about 1/3 of R-MT's samples.
- **Spielberg is used by nothing** except the final report: no training, no selection, no stage decision.

## Code (opt-in, no change to existing results)

1. `configs/pilot_st_pacejka_slip_res_mt.yaml`: `pilot_st_pacejka_slip_res.yaml` plus `residual.tracks` and `out`, in the same way that `..._residual_dual_wall_mt.yaml` derives from its base. `train_residual --holdout` already sets the track list.
2. `results/analysis/multitrack/run_fold.sh`: make the base config, the MT config, the base run directories and the output directory overridable by environment variables. The defaults stay as they are, so the dual folds rerun unchanged. A thin wrapper, `results/analysis/slip_rl/run_fold_R.sh Spielberg`, sets them for arm R and seeds 0–2.
3. **A check before training:** a 100k-step smoke run of R-MT, to confirm that multi-track sampling works with the slip base and to see that the crash rate falls on all three tracks (per-track columns in `train_log.csv`). Nothing in `residual_env` blocks it; the check is for runtime surprises only.

## Compute

Rates measured in Stages 2–5: about 41 min per 3M-step R run, and about 53 laps per minute when lapping with 8 workers.

| Step | Laps | Time |
| --- | --- | --- |
| R-A3: re-select R/s0–s2 on 3 tracks (reports taken from the existing all-checkpoint files) | 3 × 936 | ≈ 55 min |
| R-MT: train seeds 0–2 | — | ≈ 3 × 41 min |
| R-MT: 3-track selection + report | 3 × (936 + 144) | ≈ 60 min |
| R-MT-A: Austin selection | 3 × 312 | ≈ 18 min |
| R-MT: all-checkpoint diagnostic, last; selection never sees it | 3 × 1,872 | ≈ 1 h 45 min |

That is **≈ 4.2 h to the primary result** and ≈ 6 h with diagnostics. Silverstone, if it is needed, takes about the same again. The Mac needs `caffeinate -s` on AC power; `-i` alone let it sleep for 7 h in Stages 2–4.

## Decision rule (fixed now)

**Primary metric:** Spielberg qualifying laps (of 36; ceiling 34), median of 3 seeds, R-MT vs R-A3.

- **No loss** (expected) if all three hold:
  1. the R-MT Spielberg median is ≥ the R-A3 median − 2;
  2. R-MT's worst seed is ≥ 28 on Spielberg, the low end of Austin-only R's checkpoint range;
  3. the R-MT Austin median is ≥ the R-base Austin median − 3 (as in the multi-track study).
- **Hurts** if the R-MT Spielberg median is ≤ the R-A3 median − 4. Then run Silverstone.
- **Borderline** is anything between those two. Then run Silverstone, with the same thresholds.
- **There is no "helps" verdict on qualifying laps** (see the ceiling above).

**Secondary metric, where a gain could show:** the Spielberg mean lap-time ratio on the cells that every R-A3 and R-MT run qualifies. R-MT is "faster" if its median is ≥ 1 percentage point below R-A3's, and every R-MT seed is below the R-A3 median. This is the only part of the question the grid has room for.

**Also reported, with no verdict:**
- totals over all 4 tracks;
- spins;
- μ/μ₀ 0.6–0.7 failures by track;
- safe laps;
- R-MT against the dual residual's MT on Spielberg (29–32 of 36);
- the Stage 5 slack and slip-angle columns for R-MT's selected checkpoints, to check that multi-track training does not push R past the slip limits.

## Fold results: Spielberg held out (3 seeds)

Run with `results/analysis/slip_rl/run_fold_R.sh Spielberg` (2026-11-22, 20:35 → 00:47). The comparison is in `results/analysis/slip_rl/heldout_spielberg/`, and the R-MT runs are in `results_st_pacejka/slip_rl/heldout_spielberg/s0–s2`. A 100k-step smoke run came first: the crash rate fell on all three training tracks. One deviation from the plan's compute table: the fold script runs each MT run's all-checkpoint pass right after its selection, not at the very end. Selection is made first and never reads it.

Qualifying laps of 36 per track (144 in total). Medians over 3 seeds; seeds in brackets.

| Variant | Total | Austin | Catalunya | Silverstone | **Spielberg** | Spins (sum) |
| --- | --- | --- | --- | --- | --- | --- |
| R-base (Austin / Austin) | 128 (128, 128, 127) | 32 | 32 | 32 | **32** (32, 32, 32) | 9 |
| R-A3 (Austin / 3 tracks) | 133 (133, 128, 134) | 32 | 34 | 34 | **32** (32, 32, 34) | 5 |
| **R-MT** (3 tracks / 3 tracks) | 125 (125, 127, 125) | 30 | 33 | 32 | **32** (26, 32, 32) | 9 |
| R-MT-A (3 tracks / Austin) | 121 (114, 121, 126) | 30 | 32 | 32 | 30 (18, 30, 32) | 9 |

### Decision rule (as fixed in the plan)

| Condition | Value | Met? |
| --- | --- | --- |
| 1. R-MT Spielberg median ≥ R-A3 median − 2 | 32 vs 30 | yes |
| 2. R-MT worst seed ≥ 28 on Spielberg | **26** (s0) | **no** |
| 3. R-MT Austin median ≥ R-base Austin median − 3 | 30 vs 29 | yes |
| Hurts: R-MT Spielberg median ≤ R-A3 median − 4 | 32 vs 28 | no |

**Primary verdict: borderline.** The median shows no loss, but one seed (s0) falls 2 laps below the floor. The plan says to run the Silverstone fold next, with the same thresholds.

**Secondary metric: R-MT is faster on Spielberg.** On the 26 Spielberg cells that every R-A3 and R-MT run qualifies, the mean lap-time ratio is:
- R-MT: 0.985 / 0.982 / 0.957 (median 0.982);
- R-A3: 1.018 / 1.018 / 0.996 (median 1.018).

That is 3.6 percentage points lower, and every R-MT seed is below the R-A3 median, so the "faster" condition holds (≥ 1 point, every seed below).

### What else the fold shows

- **The cost of multi-track training is consistency, not the median.**
  - R-MT/s0's 10 Spielberg failures are all wall collisions, and they are at the same places as Austin-only R's (s ≈ 42, 114, 224). Austin-only R fails at s ≈ 114 only at μ/μ₀ 0.7 with a_lat 8. R-MT/s0 also fails there at μ/μ₀ 0.7 with a_lat 6, at μ/μ₀ 0.8 with a_lat 8 (where Austin-only R had no failure at all; Stage 5 §2), and at μ/μ₀ 0.6 with a_lat 6. So the place is the same, but the hard corner fails over a wider range of conditions.
  - Over checkpoints (diagnostic only), R-MT's Spielberg score ranges **15–34** (s0), 20–32 (s1) and 26–32 (s2), against 28–34 for Austin-only R.
  - Multi-track training spreads the residual over three tracks' corners. It drives faster, but the selected checkpoint is less certain to be a safe one on the unseen track.
- **This is the opposite of the dual residual's fold.** There, MT removed a 0–32 seed lottery. Here the Austin-only base had no lottery to remove (32 on every seed), and MT adds a small one.
- **Selection on three tracks helps the Austin-only runs:** R-A3's 133 / 128 / 134 are the highest totals in the study, with 5 spins over three seeds. That re-selection uses no Spielberg laps, so it is a fair, free improvement on the Stage 2–4 result. It was not pre-registered as a variant to report as the headline, though, and it is not what the decision rule compares.
- **Austin selection of the MT runs (R-MT-A) is worst** (s0: 18 on Spielberg), as for the dual residual: one track is not enough to choose among multi-track checkpoints.

**Not done:** the Stage 5 slack and slip-angle check for R-MT's selected checkpoints. `stage5.py` reads only the Stage 2–4 run folders; it needs a small change to take the `_noSpielberg` selections.

**Next, per the rule:** the Silverstone fold (`results/analysis/slip_rl/run_fold_R.sh Silverstone`, ≈ 6 h). It reuses the R-base runs, and nothing from this fold's result changes its design or thresholds.
