"""Shared helpers for the RL analysis (results/analysis/). Analysis-only: imports the repository code unchanged.

Experiment registry, saved-result loaders, a deterministic lap runner (process pool, one Track per track per
worker) and a residual-policy wrapper that records observations/actions and supports counterfactual ablations.

Ablations (analysis only, never reported as controllers):
    none      exactly pilot.residual_env._LapResidual (bit-identical laps)
    zero_d    steering residual forced to 0 (acceleration residual kept)
    zero_a    acceleration residual forced to 0 (steering residual kept)
    wall_cap  the 8 wall-gap features set to their "no wall in range" value (4.0 = clip max)
    debias    a constant ``bias`` [rad] is subtracted from the steering residual where |kappa| < 0.01
The previous-residual observation always reports the residual that was actually applied.
"""

from __future__ import annotations

import copy
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "analysis"
TRACKS = ["Austin", "Spielberg", "Catalunya", "Silverstone"]
P = ROOT / "results_st_pacejka"

EXPS = {  # key: (config, run folder, label)
    "delta": ("configs/pilot_st_pacejka_residual.yaml", P / "residual", "MPC + RL (δ)"),
    "dual": ("configs/pilot_st_pacejka_residual_dual.yaml", P / "residual_dual", "MPC + RL (δ, a)"),
    "wall": ("configs/pilot_st_pacejka_residual_dual_wall.yaml", P / "residual_dual_wall", "MPC + RL (δ, a) + wall"),
}
SELECTED = "ckpt_1250000.zip"
LABEL = {"delta": "RL δ", "dual": "RL δa", "wall": "RL δa+wall", "SE": "S-E", "nominal_std": "P (std plan)",
         "ff_std": "P+ff (std plan)", "nominal_ell": "P-E", "ff_ell": "P-E+ff"}
K_FF = 0.03
STATUSES = ["qualifying", "speed_tolerance", "lateral_error", "off_track", "solver_error", "wall_collision", "spin",
            "timeout"]


def load_cfg(exp: str, track: str = "Austin") -> dict:
    """Config for an experiment key: delta / dual / wall (residual configs) or SE / P-E / ... (base Pacejka config)."""
    if exp in EXPS:
        cfg = yaml.safe_load(open(ROOT / EXPS[exp][0]))
    else:
        from pilot.longitudinal import variant_cfg

        cfg = variant_cfg(yaml.safe_load(open(ROOT / "configs/pilot_st_pacejka.yaml")), exp.replace("+ff", ""))
    cfg = copy.deepcopy(cfg)
    cfg["track"] = track
    return cfg


def saved_sweep(exp: str, track: str) -> pd.DataFrame:
    """The saved sweep.csv rows of an experiment on a track (residual runs: nominal/ff/residual; SE: S-E only)."""
    if exp == "SE":
        df = pd.read_csv(P / "longitudinal" / track / "sweep.csv")
        return df[df.controller == "S-E"].reset_index(drop=True)
    run = EXPS[exp][1]
    return pd.read_csv(run / "sweep.csv" if track == "Austin" else run / "heldout" / track / "sweep.csv")


def saved_grid(exp: str, track: str, from3: bool = False) -> pd.DataFrame:
    if exp == "SE":
        df = pd.read_csv(P / "longitudinal" / track / "lap_time_grid.csv")
        return df[df.controller == "S-E"].reset_index(drop=True)
    run = EXPS[exp][1]
    if from3:
        return pd.read_csv(run / "grid_from3" / track / "lap_time_grid.csv")
    return pd.read_csv(run / "lap_time_grid.csv" if track == "Austin" else run / "heldout" / track / "lap_time_grid.csv")


# --------------------------------------------------------------------------------------- recording residual
_MODELS: dict = {}


def _model(path):
    if path not in _MODELS:
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        _MODELS[path] = PPO.load(path, device="cpu")
    return _MODELS[path]


class RecResidual:
    """Picklable factory for run_lap(residual=...). After the lap, ``self.last`` holds the per-lap recorder."""

    def __init__(self, path, r_max, ra_max=0.0, wall=False, ablation="none", bias=0.0, record=False):
        self.path, self.r_max, self.ra_max, self.wall = str(path), r_max, float(ra_max), bool(wall)
        self.ablation, self.bias, self.record = ablation, float(bias), record
        self.last = None

    def __call__(self):
        self.last = _Rec(self)
        return self.last


class _Rec:
    def __init__(self, f: RecResidual):
        self.f, self.model = f, _model(f.path)
        self.prev = self.prev_a = 0.0
        self.obs, self.act = [], []

    def reset(self):
        self.prev = self.prev_a = 0.0

    def __call__(self, sim):
        from pilot.residual_env import residual_obs

        f = self.f
        obs = residual_obs(sim, self.prev, f.r_max, self.prev_a, f.ra_max, f.wall)
        if f.ablation == "wall_cap":
            obs = obs.copy()
            obs[-8:] = 4.0
        a, _ = self.model.predict(obs, deterministic=True)
        d = float(np.clip(a[0], -1, 1)) * f.r_max
        da = float(np.clip(a[1], -1, 1)) * f.ra_max if f.ra_max > 0 else 0.0
        if f.record:
            self.obs.append(obs)
            self.act.append((d, da))
        if f.ablation == "zero_d":
            d = 0.0
        elif f.ablation == "zero_a":
            da = 0.0
        elif f.ablation == "debias" and abs(sim.ref.kappa[sim.i_g]) < 0.01:
            d -= f.bias
        self.prev, self.prev_a = d, da
        return (d, da) if f.ra_max > 0 else d


def residual_for(exp, ckpt=SELECTED, ablation="none", bias=0.0, record=False):
    cfg = load_cfg(exp)
    rc = cfg["residual"]
    return RecResidual(EXPS[exp][1] / ckpt, rc["r_max"], rc.get("accel_r_max", 0.0), rc.get("obs_wall_clearance", False),
                       ablation, bias, record)


# --------------------------------------------------------------------------------------------- lap runner
_W: dict = {}


def _track(track, cfg):
    from pilot.closedloop import Track

    if track not in _W:
        _W[track] = Track(cfg)
    return _W[track]


def _job(job):
    """job = dict(exp, track, a_lat, mu, start_s, K, residual (RecResidual|None), log, tags)."""
    from pilot.closedloop import Scenario, qualifies, run_lap

    cfg = load_cfg(job["exp"], job["track"])
    tr = _track(job["track"], cfg)
    res = job.get("residual")
    sm, trace = run_lap(tr, Scenario(job["a_lat"], job["mu"], job["start_s"]), job["K"], cfg, log=job.get("log", False),
                        residual=res)
    ok, why = qualifies(sm, cfg["qualifying"])
    sm = {k: (float(v) if isinstance(v, (np.floating, np.integer, np.bool_)) else v) for k, v in sm.items()}
    sm.update(qualifying=ok, status=why, **job.get("tags", {}))
    obs = act = None
    if res is not None and res.record and res.last is not None:
        obs, act = np.asarray(res.last.obs, np.float32), np.asarray(res.last.act, np.float32)
    return sm, (trace if job.get("log") else None), obs, act


def run_jobs(jobs, workers=8):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_job, jobs, chunksize=1))


def sweep_cells(cfg):
    sw = cfg["sweep"]
    return [(a, mu, s0) for a in sw["a_lat_levels"] for mu in sw["mu_scales"] for s0 in (sw["start_s"], sw["recheck_start_s"])]


# --------------------------------------------------------------------------------------- post-hoc quantities
def add_posthoc(trace: pd.DataFrame, track_obj, mu_scale: float, Ts=0.05) -> pd.DataFrame:
    """Wall gaps of the body (left/right, now), approximate axle slip angles and lateral-force utilisation.
    Acceleration for the load transfer is estimated from the 20 Hz speed trace (approximate)."""
    from pilot.sim import NOMINAL_PARAMS, set_plant, vehicle_params
    from pilot.tires import axle_forces

    t = trace.copy()
    s_ext, left, right = track_obj.wall_side_clearance(1.5)
    half_w = NOMINAL_PARAMS["width"] / 2
    L = np.mod(t.s.values, track_obj.ref.length)
    t["gap_left"] = np.interp(L, s_ext, left) - t.e_y - half_w
    t["gap_right"] = np.interp(L, s_ext, right) + t.e_y - half_w
    t["clear_left"], t["clear_right"] = np.interp(L, s_ext, left), np.interp(L, s_ext, right)
    t["gap_min"] = np.minimum(t.gap_left, t.gap_right)
    set_plant("st", {"model": "pacejka", "C": 1.3, "stiffness_scales_with_mu": False, "drive_split_front": None,
                     "brake_split_front": None, "derate": "peak"})
    p = vehicle_params(mu_scale)
    ax = np.gradient(t.v.values, Ts) if len(t) > 2 else np.zeros(len(t))
    uf, ur, af, ar = [], [], [], []
    for (dl, v, r, b), a in zip(t[["delta", "v", "yaw_rate", "slip"]].values, ax):
        if v < 0.5:
            uf.append(np.nan), ur.append(np.nan), af.append(np.nan), ar.append(np.nan)
            continue
        f = axle_forces(np.array([0, 0, dl, v, 0, r, b]), float(np.clip(a, -p["a_max"], p["a_max"])), p, "pacejka")
        Df, Dr = f["xif"] * p["mu"] * f["fzf"], f["xir"] * p["mu"] * f["fzr"]
        uf.append(abs(f["fyf"]) / max(Df, 1e-9)), ur.append(abs(f["fyr"]) / max(Dr, 1e-9))
        af.append(f["af"]), ar.append(f["ar"])
    t["ax_est"], t["util_f"], t["util_r"], t["alpha_f"], t["alpha_r"] = ax, uf, ur, af, ar
    return t
