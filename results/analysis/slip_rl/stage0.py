"""SlipMPC + RL, Stage 0 (docs/slip_rl_plan.md): checks before any training.

    uv run python results/analysis/slip_rl/stage0.py repro        # zero residual on the slip base == published S-E?
    uv run python results/analysis/slip_rl/stage0.py oracle       # S-E with slip limits scaled by the TRUE mu/mu0
    uv run python results/analysis/slip_rl/stage0.py throughput   # short PPO runs: kinematic base vs slip base
    uv run python results/analysis/slip_rl/stage0.py oracle2      # true mu in slip limits AND speed reference (grid)

repro: the 18 published S-E sweep cells per track (a_lat {4, 6, 8} x mu/mu0 {1.0, 0.8, 0.6} x start s {0, 200}) on
all four tracks, run with configs/pilot_st_pacejka_slip_res.yaml through the residual path (run_lap with a residual
that always returns (0, 0)), compared with results_st_pacejka/longitudinal/<track>/sweep.csv (controller S-E).
The published laps used kernel: python; this config uses kernel: rust (physics only; SlipMPC stays in Python), so
agreement is expected to rounding, not bit for bit. --kernel python checks the bit-for-bit case.

oracle: same cells with longitudinal.slip_mpc.oracle_mu_limits (PRIVILEGED: alpha limits x true mu/mu0). This is
an upper bound for what arm P's friction estimate can buy through the slip limits alone.

oracle2: S-E with a constant slip-limit scale L in {1, sqrt(mu), mu} and speed-reference scale S in {1, 0.97, 0.94,
sqrt(mu)} per lap, mu = TRUE mu/mu0 (PRIVILEGED). "Both" = (L, S) = (mu, sqrt(mu)): limits and speed as the grip
scales. Best-per-cell over the grid bounds what a perfect constant-per-lap (limit, speed) choice can buy. The lap-time
rule still compares against the unscaled plan, so S < 0.95 can only give "speed_tolerance"; "safe" counts laps that
fail nothing but that rule.

throughput: pilot.train_residual for --steps on both configs (8 SubprocVecEnv workers, as in training), wall time.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pilot.closedloop import Scenario, Track, qualifies, run_lap, run_many  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "stage0"
SLIP_CFG = ROOT / "configs/pilot_st_pacejka_slip_res.yaml"
KIN_CFG = ROOT / "configs/pilot_st_pacejka_residual_dual_wall.yaml"
TRACKS = ["Austin", "Spielberg", "Catalunya", "Silverstone"]
CMP = ["progress_m", "lap_time", "rms_ey", "max_ey", "max_abs_ay"]


class ZeroResidual:
    """Picklable residual factory: every control instant returns (d_delta, d_a) = (0, 0)."""

    def __call__(self):
        return lambda sim: (0.0, 0.0)


def load(path, **over):
    cfg = yaml.safe_load(open(path))
    for k, v in over.items():
        cfg[k] = v
    return cfg


def sweep(cfg, track, residual=None):
    c = copy.deepcopy(cfg)
    c["track"] = track
    sw = c["sweep"]
    cells = [(a, mu, s0) for a in sw["a_lat_levels"] for mu in sw["mu_scales"]
             for s0 in (sw["start_s"], sw["recheck_start_s"])]
    jobs = [(Scenario(a, mu, s0), 0.0, False, None, residual) for a, mu, s0 in cells]
    rows = []
    for (a, mu, s0), (sm, _) in zip(cells, run_many(c, jobs)):
        ok, why = qualifies(sm, c["qualifying"])
        rows.append({"a_lat": a, "mu_scale": mu, "start_s": s0, **sm, "track": track, "qualifying": ok, "status": why})
    return pd.DataFrame(rows)


def published(track):
    df = pd.read_csv(ROOT / f"results_st_pacejka/longitudinal/{track}/sweep.csv")
    return df[df.controller == "S-E"]


def counts(df):
    return dict(qualifying=int(df.qualifying.sum()), spin=int((df.status == "spin").sum()),
                wall=int((df.status == "wall_collision").sum()), n=len(df))


def repro(args):
    cfg = load(SLIP_CFG, **({"kernel": args.kernel} if args.kernel else {}))
    tag = f"repro_{cfg['kernel']}"
    res = pd.concat([sweep(cfg, t, ZeroResidual()) for t in TRACKS], ignore_index=True)
    pub = pd.concat([published(t).assign(track=t) for t in TRACKS], ignore_index=True)
    key = ["track", "a_lat", "mu_scale", "start_s"]
    for c in key[1:]:
        res[c], pub[c] = res[c].round(4), pub[c].round(4)
    j = res.merge(pub[key + ["status"] + CMP], on=key, suffixes=("", "_pub"), how="outer", indicator=True)
    j["status_match"] = j.status == j.status_pub
    for c in CMP:
        j[f"d_{c}"] = (j[c] - j[f"{c}_pub"]).abs()
    j.to_csv(OUT / f"{tag}.csv", index=False, float_format="%.6g")
    summ = dict(kernel=cfg["kernel"], cells=len(j), unmatched_rows=int((j._merge != "both").sum()),
                status_mismatch=int((~j.status_match).sum()),
                max_abs_diff={c: float(j[f"d_{c}"].max(skipna=True)) for c in CMP},
                ours=counts(res), published=counts(pub.assign(qualifying=pub.qualifying.astype(bool))))
    (OUT / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    if summ["status_mismatch"]:
        print(j.loc[~j.status_match, key + ["status", "status_pub", "progress_m", "progress_m_pub"]].to_string())


def oracle(args):
    cfg = load(SLIP_CFG)
    cfg["longitudinal"]["slip_mpc"]["oracle_mu_limits"] = True
    res = pd.concat([sweep(cfg, t) for t in TRACKS], ignore_index=True)
    res.to_csv(OUT / "oracle_sweep.csv", index=False, float_format="%.6g")
    base_path = OUT / f"repro_{cfg['kernel']}.csv"
    rows = []
    for t in TRACKS:
        r = res[res.track == t]
        p = published(t).assign(qualifying=lambda d: d.qualifying.astype(bool))
        rows += [dict(track=t, ctrl="S-E (published)", **counts(p)), dict(track=t, ctrl="S-E oracle-mu limits", **counts(r))]
    for ctrl in ("S-E (published)", "S-E oracle-mu limits"):
        sub = [r for r in rows if r["ctrl"] == ctrl]
        rows.append(dict(track="All", ctrl=ctrl, **{k: sum(r[k] for r in sub) for k in ("qualifying", "spin", "wall", "n")}))
    tab = pd.DataFrame(rows)
    by_mu = res.groupby("mu_scale").apply(lambda d: pd.Series(counts(d)), include_groups=False)
    tab.to_csv(OUT / "oracle_summary.csv", index=False)
    by_mu.to_csv(OUT / "oracle_by_mu.csv")
    print(tab.to_string(index=False))
    print(by_mu.to_string())
    if base_path.exists():  # cell-level flips against the zero-residual reproduction
        b = pd.read_csv(base_path)
        key = ["track", "a_lat", "mu_scale", "start_s"]
        f = res[key + ["status"]].merge(b[key + ["status"]], on=key, suffixes=("_oracle", "_base"))
        f = f[f.status_oracle != f.status_base]
        f.to_csv(OUT / "oracle_flips.csv", index=False)
        print(f.to_string(index=False))


_TR: dict = {}


def _grid_lap(job):
    cfg, scn, tag = job
    if cfg["track"] not in _TR:
        _TR[cfg["track"]] = Track(cfg)
    sm, _ = run_lap(_TR[cfg["track"]], scn, 0.0, cfg, log=False)
    ok, why = qualifies(sm, cfg["qualifying"])
    return {**sm, **tag, "qualifying": ok, "status": why}


L_RULES = {"1": lambda mu: 1.0, "sqrt_mu": np.sqrt, "mu": lambda mu: mu}
S_RULES = {"1": lambda mu: 1.0, "0.97": lambda mu: 0.97, "0.94": lambda mu: 0.94, "sqrt_mu": np.sqrt}


def oracle2(args):
    from concurrent.futures import ProcessPoolExecutor

    base = load(SLIP_CFG)
    sw = base["sweep"]
    jobs, seen = [], set()
    for t in TRACKS:
        for a in sw["a_lat_levels"]:
            for mu in sw["mu_scales"]:
                for s0 in (sw["start_s"], sw["recheck_start_s"]):
                    for ln, lf in L_RULES.items():
                        for sn, sf in S_RULES.items():
                            lv, sv = round(float(lf(mu)), 6), round(float(sf(mu)), 6)
                            if (t, a, mu, s0, lv, sv) in seen:  # at mu = 1 several rules coincide
                                continue
                            seen.add((t, a, mu, s0, lv, sv))
                            c = copy.deepcopy(base)
                            c["track"] = t
                            c["longitudinal"]["slip_mpc"].update(limit_scale=lv, speed_scale=sv)
                            tag = dict(track=t, a_lat=a, mu_scale=mu, start_s=s0, L=lv, S=sv)
                            jobs.append((c, Scenario(a, mu, s0), tag))
    print(f"{len(jobs)} laps", flush=True)
    with ProcessPoolExecutor(base.get("workers") or 8) as ex:
        df = pd.DataFrame(list(ex.map(_grid_lap, jobs, chunksize=4)))
    df["safe"] = df.status.isin(["qualifying", "speed_tolerance"])
    df.to_csv(OUT / "oracle2_laps.csv", index=False, float_format="%.6g")
    key = ["track", "a_lat", "mu_scale", "start_s"]
    cells = df[key].drop_duplicates()

    def rule_rows(ln, sn):  # one lap per cell for this (L rule, S rule)
        m = df.apply(lambda r: np.isclose(r.L, L_RULES[ln](r.mu_scale)) and np.isclose(r.S, S_RULES[sn](r.mu_scale)), axis=1)
        return df[m].drop_duplicates(key)

    summ = []
    for ln in L_RULES:
        for sn in S_RULES:
            r = rule_rows(ln, sn)
            assert len(r) == len(cells), (ln, sn, len(r))
            q = r[r.qualifying]
            summ.append(dict(L=ln, S=sn, **counts(r), safe=int(r.safe.sum()),
                             q_mu06=int(r[r.mu_scale == 0.6].qualifying.sum()),
                             safe_mu06=int(r[r.mu_scale == 0.6].safe.sum()),
                             mean_ratio_safe=float(r[r.safe].lap_time_ratio.mean()),
                             mean_ratio_q=float(q.lap_time_ratio.mean()) if len(q) else np.nan))
    tab = pd.DataFrame(summ)
    best = df.groupby(key).agg(any_q=("qualifying", "any"), any_safe=("safe", "any")).reset_index()
    base_q = rule_rows("1", "1").set_index(key).qualifying
    best["base_q"] = best.set_index(key).index.map(base_q).values
    extra = dict(best_per_cell_qualifying=int(best.any_q.sum()), best_per_cell_safe=int(best.any_safe.sum()),
                 cells_gained_over_SE=int((best.any_q & ~best.base_q).sum()),
                 gained_by_mu=best[best.any_q & ~best.base_q].mu_scale.value_counts().sort_index().to_dict(),
                 gained_by_track=best[best.any_q & ~best.base_q].track.value_counts().to_dict())
    tab.to_csv(OUT / "oracle2_summary.csv", index=False)
    best.to_csv(OUT / "oracle2_best_per_cell.csv", index=False)
    (OUT / "oracle2_best.json").write_text(json.dumps(extra, indent=1, default=float))
    print(tab.to_string(index=False))
    print(json.dumps(extra, indent=1, default=float))
    both = rule_rows("mu", "sqrt_mu")
    print("both (L=mu, S=sqrt mu) by mu:")
    print(both.groupby("mu_scale").apply(lambda d: pd.Series({**counts(d), "safe": int(d.safe.sum())}),
                                         include_groups=False).to_string())


def throughput(args):
    """Wall time of pilot.train_residual at two run lengths per base; the slope (marginal time per step) removes
    the fixed start-up cost (spawning 8 workers, building tracks and simulators, first solver setups)."""
    out = {}
    for name, path in (("kinematic", KIN_CFG), ("slip", SLIP_CFG)):
        walls = []
        for n in args.steps:
            run_out = pathlib.Path(args.scratch) / f"tp_{name}_{n}"
            cmd = [sys.executable, "-m", "pilot.train_residual", "--config", str(path), "--steps", str(n),
                   "--out", str(run_out)]
            t0 = time.perf_counter()
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            walls.append(time.perf_counter() - t0)
            print(name, n, f"{walls[-1]:.1f} s", flush=True)
        slope, fixed = np.polyfit(args.steps, walls, 1)
        out[name] = dict(steps=args.steps, wall_s=walls, fixed_s=fixed, marginal_us_per_step=slope * 1e6,
                         min_per_3M=(fixed + slope * 3e6) / 60)
    out["marginal_ratio_slip_over_kinematic"] = out["slip"]["marginal_us_per_step"] / out["kinematic"]["marginal_us_per_step"]
    (OUT / "throughput.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["repro", "oracle", "oracle2", "throughput"])
    ap.add_argument("--kernel", choices=["python", "rust"], default=None, help="repro: override the config's kernel")
    ap.add_argument("--steps", type=int, nargs="+", default=[81920, 245760],
                    help="throughput: run lengths in env steps (10 and 30 PPO updates)")
    ap.add_argument("--scratch", required="throughput" in sys.argv, help="throughput: checkpoint folder (discarded)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"repro": repro, "oracle": oracle, "oracle2": oracle2, "throughput": throughput}[args.what](args)
