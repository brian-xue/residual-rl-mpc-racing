# Plan: RL on top of the slip-aware MPC (SlipMPC + RL)

Branch `explore/slip-mpc-rl`. **Status: Stage 0 done, including both oracle checks (see [Stage 0 results](#stage-0-results)). Arm P has been redesigned as a learned speed reference (P′). No training yet.** The design and decision rule below are fixed before any run, as in [`multitrack_plan.md`](multitrack_plan.md).

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

