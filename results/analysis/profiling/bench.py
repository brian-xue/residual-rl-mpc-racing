"""Part 4: performance benchmarks (measurement only; the repository code is wrapped, never modified).

    uv run python results/analysis/profiling/bench.py <cmd>
cmds:
    cold     cold-start costs (run in a fresh process; called 3x by `all`)
    laps     whole-lap wall time per controller, single process (warm-up 2, repeats 7)
    steps    per-control-step breakdown (plan / obs / policy / apply / physics) inside laps and inside env.step
    infer    policy inference micro-benchmarks (SB3 predict vs torch forward vs numpy)
    mpc      MPC latency distributions and SlipMPC phase breakdown (disc/expm, QP build, OSQP update/solve, iters)
    wallobs  wall-clearance observation cost (precompute, lookup, obs assembly)
    ppo      PPO rollout vs update phases, n_envs scaling, worker CPU vs wall, torch threads in the update
    eval     eval_residual-equivalent pipeline timing (selection, sweep, chains), outputs to a scratch dir
    native   /usr/bin/sample native stacks + cProfile of an RL rollout loop and a SlipMPC lap loop
Every command writes JSON/CSV to results/analysis/profiling/out/ with a system snapshot before and after.
"""

from __future__ import annotations

import cProfile
import functools
import gc
import json
import os
import pathlib
import pstats
import resource
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rlana as R  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
pc = time.perf_counter


# ---------------------------------------------------------------------------------------------- system state
def snapshot(tag=""):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except Exception as e:  # noqa: BLE001
            return f"ERR {e}"

    load = os.getloadavg()
    top = sh(["top", "-l", "2", "-n", "0", "-s", "1"])
    cpu_line = [ln for ln in top.splitlines() if ln.startswith("CPU usage")]
    ps = sh(["ps", "-Ao", "pid,%cpu,rss,comm"]).splitlines()[1:]
    me = {os.getpid(), os.getppid()}
    procs = []
    for ln in ps:
        parts = ln.split(None, 3)
        if len(parts) == 4:
            try:
                procs.append((int(parts[0]), float(parts[1]), int(parts[2]), parts[3]))
            except ValueError:
                pass
    others = [p for p in procs if p[0] not in me and "python" not in p[3].lower()]
    vm = sh(["vm_stat"])
    pages = {ln.split(":")[0]: ln.split(":")[1].strip().rstrip(".") for ln in vm.splitlines()[1:] if ":" in ln}
    free_mb = (int(pages.get("Pages free", 0)) + int(pages.get("Pages inactive", 0))) * 16384 / 2**20
    return dict(tag=tag, time=time.strftime("%H:%M:%S"), load1=load[0], load5=load[1], load15=load[2],
                cpu_usage=cpu_line[-1] if cpu_line else None,
                other_cpu_pct=round(sum(p[1] for p in others), 1),
                top_others=[(p[3].split("/")[-1][:40], p[1]) for p in sorted(others, key=lambda p: -p[1])[:5]],
                free_inactive_mb=round(free_mb))


def stats(x):
    x = np.asarray(x, float)
    if not len(x):
        return {}
    q = np.percentile(x, [25, 50, 75, 95, 99])
    return dict(n=len(x), mean=x.mean(), median=q[1], p25=q[0], p75=q[2], iqr=q[2] - q[0], p95=q[3], p99=q[4],
                max=x.max(), min=x.min(), total=x.sum())


def save(name, obj):
    obj = json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(json.dumps(obj, indent=1)[:6000])


def env_info():
    import platform

    import osqp
    import stable_baselines3
    import torch

    def sysctl(k):
        return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()

    return dict(cpu=sysctl("machdep.cpu.brand_string"), ncpu=sysctl("hw.ncpu"), p_cores=sysctl("hw.perflevel0.physicalcpu"),
                e_cores=sysctl("hw.perflevel1.physicalcpu"), mem_gb=int(sysctl("hw.memsize")) / 2**30,
                macos=platform.mac_ver()[0], python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
                torch_threads=torch.get_num_threads(), sb3=stable_baselines3.__version__, osqp=osqp.__version__)


# ----------------------------------------------------------------------------------------------- workloads
COND = dict(track="Austin", a_lat=4.0, mu=1.0, start_s=0.0)  # every benchmarked controller completes this lap
WORKLOADS = {  # name: (experiment key for the config, K, residual?, kernel override)
    "kin_py (P-E+ff)": ("dual", 0.03, False, "python"),
    "kin_rust (P-E+ff)": ("dual", 0.03, False, "rust"),
    "S-E (SlipMPC, py)": ("S-E", 0.0, False, None),
    "RL δ (rust)": ("delta", 0.0, True, "rust"),
    "RL δa (rust)": ("dual", 0.0, True, "rust"),
    "RL δa+wall (rust)": ("wall", 0.0, True, "rust"),
    "RL δa+wall (py)": ("wall", 0.0, True, "python"),
}


def lap_setup(name, track="Austin"):
    from pilot.closedloop import Track

    exp, K, res, kern = WORKLOADS[name]
    cfg = R.load_cfg(exp, track)
    if kern:
        cfg["kernel"] = kern
    tr = Track(cfg)
    resf = None
    if res:
        from pilot.residual_env import ResidualPolicy

        rc = cfg["residual"]
        resf = ResidualPolicy(R.EXPS[exp][1] / R.SELECTED, rc["r_max"], rc.get("accel_r_max", 0.0),
                              rc.get("obs_wall_clearance", False))
    return cfg, tr, K, resf


def one_lap(cfg, tr, K, resf, a_lat=COND["a_lat"], mu=COND["mu"], s0=COND["start_s"]):
    from pilot.closedloop import Scenario, run_lap

    t0 = pc()
    sm, _ = run_lap(tr, Scenario(a_lat, mu, s0), K, cfg, log=False, residual=resf)
    return pc() - t0, sm


# ------------------------------------------------------------------------------------------------- cold
def cmd_cold():
    t0 = pc()
    import pilot.closedloop as cl  # noqa: F401
    import pilot.residual_env  # noqa: F401
    t_import_pilot = pc() - t0
    t0 = pc()
    import stable_baselines3  # noqa: F401
    import torch  # noqa: F401
    t_import_sb3 = pc() - t0
    from pilot.closedloop import Track
    from pilot.sim import make_env

    cfg = R.load_cfg("wall")
    out = dict(import_pilot=t_import_pilot, import_sb3_torch=t_import_sb3)
    t0 = pc()
    tr = Track(cfg)
    out["track_first"] = pc() - t0
    t0 = pc()
    Track(cfg)
    out["track_second"] = pc() - t0
    ts = []
    for _ in range(3):
        t0 = pc()
        make_env("Austin", 1.0, cfg["steer_mode"], cfg["dt"])
        ts.append(pc() - t0)
    out["make_env"] = ts
    ts = []
    for _ in range(5):
        tr._plans.clear()
        t0 = pc()
        tr.plan(4.0, None, True)
        ts.append(pc() - t0)
    out["speed_plan"] = ts
    ts = []
    for _ in range(3):
        tr._wall_side = None
        t0 = pc()
        tr.wall_side_clearance(1.5)
        ts.append(pc() - t0)
    out["wall_side_clearance"] = ts
    from stable_baselines3 import PPO

    ts = []
    for _ in range(3):
        t0 = pc()
        PPO.load(str(R.EXPS["wall"][1] / R.SELECTED), device="cpu")
        ts.append(pc() - t0)
    out["ppo_load"] = ts
    cfg2, tr2, K, resf = lap_setup("RL δa+wall (rust)")
    laps = [one_lap(cfg2, tr2, K, resf)[0] for _ in range(4)]
    out["lap_1st_to_4th"] = laps
    print(json.dumps(out))
    return out


def cmd_cold_all():
    runs = []
    for _ in range(3):
        r = subprocess.run([sys.executable, __file__, "cold"], capture_output=True, text=True)
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    save("cold", dict(env=env_info(), before=snapshot("cold"), runs=runs))


# ------------------------------------------------------------------------------------------------- laps
def cmd_laps():
    rows, snaps = [], [snapshot("before")]
    for name in WORKLOADS:
        cfg, tr, K, resf = lap_setup(name)
        for _ in range(2):
            one_lap(cfg, tr, K, resf)
        ts, cpus = [], []
        reps = 5 if "py" in name and "RL" in name else 7
        for _ in range(reps):
            c0 = time.process_time()
            dt, sm = one_lap(cfg, tr, K, resf)
            ts.append(dt)
            cpus.append(time.process_time() - c0)
        n_ctrl = sm["lap_time"] / 0.05 if sm["lap_time"] else None
        st = stats(ts)
        rows.append(dict(workload=name, status=sm["outcome"], sim_lap_s=sm["lap_time"], n_ctrl_steps=n_ctrl,
                         median_s=st["median"], iqr_s=st["iqr"], min_s=st["min"], max_s=st["max"], n=st["n"],
                         cpu_over_wall=float(np.sum(cpus) / np.sum(ts)),
                         ms_per_ctrl_step=st["median"] / n_ctrl * 1e3 if n_ctrl else None,
                         realtime_factor=sm["lap_time"] / st["median"] if sm["lap_time"] else None))
        print(rows[-1], flush=True)
    snaps.append(snapshot("after"))
    pd.DataFrame(rows).to_csv(OUT / "laps.csv", index=False, float_format="%.5f")
    save("laps", dict(env=env_info(), cond=COND, snapshots=snaps, rows=rows))


# ------------------------------------------------------------------------------------------------ steps
class Timers:
    def __init__(self):
        self.t = {}

    def wrap(self, owner, attr, key=None):
        f = getattr(owner, attr)
        key = key or attr
        rec = self.t.setdefault(key, [])

        @functools.wraps(f)
        def w(*a, **k):
            t0 = pc()
            try:
                return f(*a, **k)
            finally:
                rec.append(pc() - t0)

        setattr(owner, attr, w)
        return f


def cmd_steps():
    import pilot.closedloop as cl
    import pilot.residual_env as re_

    snaps = [snapshot("before")]
    result = {}
    for name in ("kin_py (P-E+ff)", "kin_rust (P-E+ff)", "S-E (SlipMPC, py)", "RL δ (rust)", "RL δa (rust)",
                 "RL δa+wall (rust)", "RL δa+wall (py)"):
        cfg, tr, K, resf = lap_setup(name)
        one_lap(cfg, tr, K, resf)  # warm
        T = Timers()
        orig = {}
        for attr in ("plan", "apply", "advance_interval"):
            orig[("LapSim", attr)] = T.wrap(cl.LapSim, attr)
        orig[("re", "residual_obs")] = T.wrap(re_, "residual_obs")
        orig[("re", "wall_gaps")] = T.wrap(re_, "wall_gaps")
        orig[("re", "_LapResidual.__call__")] = T.wrap(re_._LapResidual, "__call__", "policy_total")
        model = re_._MODELS.get(str(R.EXPS[WORKLOADS[name][0]][1] / R.SELECTED)) if WORKLOADS[name][2] else None
        if model is not None:
            T.wrap(model, "predict", "sb3_predict")
        tot = []
        for _ in range(3):
            dt, sm = one_lap(cfg, tr, K, resf)
            tot.append(dt)
        for (o, attr), f in orig.items():
            setattr(cl.LapSim if o == "LapSim" else (re_._LapResidual if "LapResidual" in attr else re_),
                    attr.split(".")[-1], f)
        if model is not None:
            del model.predict  # restore the class method
        n_steps = len(T.t["plan"])
        lap_total = float(np.sum(tot))
        comp = {k: dict(**stats(np.asarray(v) * 1e6), share=float(np.sum(v) / lap_total)) for k, v in T.t.items() if v}
        result[name] = dict(lap_total_s=lap_total, ctrl_steps=n_steps, us_per_step_total=lap_total / n_steps * 1e6,
                            components_us=comp)
        print(name, {k: (round(v["median"], 1), round(v["share"], 3)) for k, v in comp.items()}, flush=True)
    # training path: ResidualRacingEnv.step components (random actions, includes resets)
    for exp in ("delta", "dual", "wall"):
        cfg = R.load_cfg(exp)
        env = re_.ResidualRacingEnv(cfg, seed=1)
        T = Timers()
        o1 = T.wrap(cl.LapSim, "plan")
        o2 = T.wrap(cl.LapSim, "advance_interval")
        o3 = T.wrap(re_, "residual_obs")
        o4 = T.wrap(cl.LapSim, "__init__", "LapSim_init")
        rng = np.random.default_rng(0)
        env.reset()
        for _ in range(200):
            _, _, te, tu, _ = env.step(rng.uniform(-1, 1, env.action_space.shape) * 0.3)
            if te or tu:
                env.reset()
        for v in T.t.values():
            v.clear()
        step_t, reset_t = [], []
        for _ in range(4000):
            t0 = pc()
            _, _, te, tu, _ = env.step(rng.uniform(-1, 1, env.action_space.shape) * 0.3)
            step_t.append(pc() - t0)
            if te or tu:
                t0 = pc()
                env.reset()
                reset_t.append(pc() - t0)
        cl.LapSim.plan, cl.LapSim.advance_interval, re_.residual_obs, cl.LapSim.__init__ = o1, o2, o3, o4
        tot = np.sum(step_t) + np.sum(reset_t)
        result[f"env.step {exp}"] = dict(
            step=stats(np.asarray(step_t) * 1e6), reset=stats(np.asarray(reset_t) * 1e6), n_resets=len(reset_t),
            reset_share=float(np.sum(reset_t) / tot),
            components_us={k: dict(**stats(np.asarray(v) * 1e6), share=float(np.sum(v) / tot)) for k, v in T.t.items() if v})
        print(exp, result[f"env.step {exp}"]["step"]["median"], result[f"env.step {exp}"]["reset"].get("median"), flush=True)
    snaps.append(snapshot("after"))
    save("steps", dict(env=env_info(), snapshots=snaps, result=result))


# ------------------------------------------------------------------------------------------------ infer
def cmd_infer():
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    res = {}
    for exp in R.EXPS:
        m = PPO.load(str(R.EXPS[exp][1] / R.SELECTED), device="cpu")
        d = m.observation_space.shape[0]
        obs = np.random.default_rng(0).uniform(-1, 1, d).astype(np.float32)
        pol = m.policy

        def sb3():
            m.predict(obs, deterministic=True)

        def torch_fwd():
            with torch.no_grad():
                pol._predict(torch.as_tensor(obs)[None], deterministic=True)

        Ws = [(l_.weight.detach().numpy().astype(np.float64), l_.bias.detach().numpy().astype(np.float64))
              for l_ in pol.mlp_extractor.policy_net if isinstance(l_, torch.nn.Linear)]
        Wa, ba = pol.action_net.weight.detach().numpy().astype(np.float64), pol.action_net.bias.detach().numpy().astype(np.float64)

        def numpy_mlp():
            h = obs.astype(np.float64)
            for W, b in Ws:
                h = np.tanh(W @ h + b)
            np.clip(Wa @ h + ba, -1, 1)

        # check numpy reproduces SB3's action
        h = obs.astype(np.float64)
        for W, b in Ws:
            h = np.tanh(W @ h + b)
        ref = m.predict(obs, deterministic=True)[0]
        r = dict(obs_dim=d, numpy_matches_sb3=float(np.abs(np.clip(Wa @ h + ba, -1, 1) - ref).max()))
        for nm, f in (("sb3_predict", sb3), ("torch_forward_no_grad", torch_fwd), ("numpy_mlp", numpy_mlp)):
            for _ in range(500):
                f()
            reps = []
            for _ in range(7):
                t0 = pc()
                for _ in range(2000):
                    f()
                reps.append((pc() - t0) / 2000 * 1e6)
            r[nm + "_us"] = stats(reps)
        # batched forward (what PPO's rollout does with n_envs observations)
        for n in (1, 8, 64):
            ob = torch.as_tensor(np.tile(obs, (n, 1)))
            reps = []
            for _ in range(7):
                t0 = pc()
                for _ in range(500):
                    with torch.no_grad():
                        pol.forward(ob)
                reps.append((pc() - t0) / 500 * 1e6)
            r[f"policy.forward_batch{n}_us"] = stats(reps)
        res[exp] = r
        print(exp, {k: (round(v["median"], 1) if isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
    save("infer", dict(env=env_info(), snapshot=snapshot("infer"), result=res))


# -------------------------------------------------------------------------------------------------- mpc
def cmd_mpc():
    import osqp

    import pilot.closedloop as cl
    import pilot.mpc as M

    snaps = [snapshot("before")]
    recs = []
    cur = {}

    class TOSQP(osqp.OSQP):
        def setup(self, *a, **k):
            t0 = pc()
            r = super().setup(*a, **k)
            cur["setup"] = cur.get("setup", 0) + pc() - t0
            return r

        def update(self, *a, **k):
            t0 = pc()
            r = super().update(*a, **k)
            cur["update"] = cur.get("update", 0) + pc() - t0
            return r

        def solve(self, *a, **k):
            t0 = pc()
            r = super().solve(*a, **k)
            cur["osqp_solve"] = cur.get("osqp_solve", 0) + pc() - t0
            cur["iter"], cur["status"] = r.info.iter, r.info.status_val
            cur["osqp_internal_solve"] = r.info.solve_time
            cur["polish"] = getattr(r.info, "polish_time", np.nan)
            return r

    M.osqp.OSQP = TOSQP
    orig_expm = M.expm

    def texpm(A):
        t0 = pc()
        r = orig_expm(A)
        cur["expm"] = cur.get("expm", 0) + pc() - t0
        cur["n_expm"] = cur.get("n_expm", 0) + 1
        return r

    M.expm = texpm

    def timed(cls, attr, key):
        f = getattr(cls, attr)

        def w(*a, **k):
            t0 = pc()
            r = f(*a, **k)
            cur[key] = cur.get(key, 0) + pc() - t0
            return r

        setattr(cls, attr, w)

    timed(M.SlipMPC, "_disc", "disc")
    timed(M.SlipMPC, "_build", "build")
    timed(M.SlipMPC, "steady", "steady")
    timed(M.KinematicMPC, "assemble", "assemble")
    f_slip = M.SlipMPC.solve
    f_kin = M.KinematicMPC.solve

    def slip_solve(self, *a, **k):
        cur.clear()
        t0 = pc()
        r = f_slip(self, *a, **k)
        cur["total"] = pc() - t0
        cur["first"] = not hasattr(self, "_bench_seen")
        self._bench_seen = True
        recs.append(dict(kind="slip", **cur))
        return r

    def kin_solve(self, *a, **k):
        cur.clear()
        t0 = pc()
        r = f_kin(self, *a, **k)
        cur["total"] = pc() - t0
        cur["first"] = not hasattr(self, "_bench_seen")
        self._bench_seen = True
        recs.append(dict(kind="kin_py", **cur))
        return r

    M.SlipMPC.solve, M.KinematicMPC.solve = slip_solve, kin_solve
    # rust kinematic MPC: total of the Rust plan call (QP assembly + OSQP + previews)
    f_plan = cl.LapSim.plan

    def plan_rust(self):
        if self.rqp is None:
            return f_plan(self)
        t0 = pc()
        r = f_plan(self)
        recs.append(dict(kind="kin_rust", total=pc() - t0, osqp_solve=self.solve_times[-1], first=len(self.solve_times) == 1))
        return r

    cl.LapSim.plan = plan_rust
    lapsums = []
    for name in ("S-E (SlipMPC, py)", "kin_py (P-E+ff)", "kin_rust (P-E+ff)"):
        for track in ("Austin", "Spielberg"):
            cfg, tr, K, resf = lap_setup(name, track)
            for a_lat in (4.0, 6.0):
                for mu in (1.0, 0.6):
                    gc.collect()
                    n0 = len(recs)
                    dt, sm = one_lap(cfg, tr, K, resf, a_lat, mu, 0.0)
                    for r in recs[n0:]:
                        r.update(workload=name, track=track, a_lat=a_lat, mu=mu)
                    lapsums.append(dict(workload=name, track=track, a_lat=a_lat, mu=mu, outcome=sm["outcome"], wall_s=dt,
                                        reported_solve_ms_p99=sm["solve_ms_p99"], reported_solve_ms_max=sm["solve_ms_max"]))
                    print(lapsums[-1], flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "mpc_solves.csv.gz", index=False, float_format="%.7g")
    pd.DataFrame(lapsums).to_csv(OUT / "mpc_laps.csv", index=False)
    summ = {}
    for kind, g in df.groupby("kind"):
        gw = g[~g["first"].astype(bool)]
        cols = [c for c in ("total", "disc", "expm", "build", "steady", "assemble", "update", "osqp_solve", "setup") if c in g]
        summ[kind] = {c: stats(gw[c].dropna() * 1e3) for c in cols}
        summ[kind]["share_of_total"] = {c: float(gw[c].fillna(0).sum() / gw.total.sum()) for c in cols if c != "total"}
        summ[kind]["first_solve_ms"] = stats(g[g["first"].astype(bool)].total * 1e3)
        if "iter" in g:
            summ[kind]["iter"] = stats(gw["iter"].dropna())
            tail = gw[gw.total >= gw.total.quantile(0.99)]
            summ[kind]["p99_tail"] = dict(n=len(tail), iter_median=float(tail["iter"].median()),
                                          iter_median_all=float(gw["iter"].median()),
                                          osqp_share=float(tail.osqp_solve.sum() / tail.total.sum()),
                                          corr_total_iter=float(gw[["total", "iter"]].corr().iloc[0, 1]))
    snaps.append(snapshot("after"))
    save("mpc", dict(env=env_info(), snapshots=snaps, summary=summ))


# ---------------------------------------------------------------------------------------------- wallobs
def cmd_wallobs():
    from pilot.closedloop import LapSim, Scenario, Track
    from pilot.residual_env import residual_obs, wall_gaps

    res = {}
    for track in R.TRACKS:
        cfg = R.load_cfg("wall", track)
        tr = Track(cfg)
        ts = []
        for _ in range(3):
            tr._wall_side = None
            t0 = pc()
            tr.wall_side_clearance(1.5)
            ts.append(pc() - t0)
        res[f"precompute_{track}_s"] = stats(ts)
    sim = LapSim(tr, Scenario(4.0, 1.0, 0.0), 0.0, cfg, log=False)
    sim.plan()
    sa = sim.s + np.array([0.0, 0.25, 0.5, 1.0]) * sim.v
    for nm, f in (("wall_gaps", lambda: wall_gaps(sim, sa)),
                  ("residual_obs_dual", lambda: residual_obs(sim, 0.0, 0.2, 0.0, 2.0, False)),
                  ("residual_obs_dual_wall", lambda: residual_obs(sim, 0.0, 0.2, 0.0, 2.0, True)),
                  ("residual_obs_single", lambda: residual_obs(sim, 0.0, 0.2))):
        for _ in range(500):
            f()
        reps = []
        for _ in range(7):
            t0 = pc()
            for _ in range(3000):
                f()
            reps.append((pc() - t0) / 3000 * 1e6)
        res[nm + "_us"] = stats(reps)
    print({k: round(v["median"], 2) for k, v in res.items()})
    save("wallobs", dict(env=env_info(), snapshot=snapshot("wallobs"), result=res))


# -------------------------------------------------------------------------------------------------- ppo
class TimedEnv:
    """Picklable env factory: ResidualRacingEnv with per-worker wall/CPU accounting of step() and reset()."""

    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed

    def __call__(self):
        import gymnasium as gym
        import torch

        from pilot.residual_env import ResidualRacingEnv

        torch.set_num_threads(1)

        class W(gym.Wrapper):
            def __init__(s, env):
                super().__init__(env)
                s.acc = dict(step_wall=0.0, step_cpu=0.0, n_step=0, reset_wall=0.0, n_reset=0)

            def step(s, a):
                t0, c0 = pc(), time.process_time()
                r = s.env.step(a)
                s.acc["step_wall"] += pc() - t0
                s.acc["step_cpu"] += time.process_time() - c0
                s.acc["n_step"] += 1
                return r

            def reset(s, **k):
                t0 = pc()
                r = s.env.reset(**k)
                s.acc["reset_wall"] += pc() - t0
                s.acc["n_reset"] += 1
                return r

            def timing(s):
                return dict(s.acc, proc_cpu=time.process_time())

        return W(ResidualRacingEnv(self.cfg, seed=self.seed))


def ppo_run(exp, n_envs, rollouts=4, threads=None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    if threads:
        torch.set_num_threads(threads)
    cfg = R.load_cfg(exp)
    pcf = cfg["residual"]["ppo"]
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = pc()
    venv = SubprocVecEnv([TimedEnv(cfg, 1000 + i) for i in range(n_envs)], start_method="spawn")
    venv = VecNormalize(VecMonitor(venv), norm_obs=False, norm_reward=True, gamma=pcf["gamma"])
    t_spawn = pc() - t0
    model = PPO("MlpPolicy", venv, n_steps=pcf["n_steps"], batch_size=pcf["batch_size"], n_epochs=pcf["n_epochs"],
                learning_rate=pcf["learning_rate"], gamma=pcf["gamma"], gae_lambda=pcf["gae_lambda"],
                clip_range=pcf["clip_range"], ent_coef=0.0, seed=0, device="cpu", verbose=0,
                policy_kwargs=dict(log_std_init=pcf["log_std_init"], net_arch=list(pcf["net_arch"]),
                                   activation_fn=torch.nn.Tanh))
    marks = []

    class CB(BaseCallback):
        def _on_rollout_start(self):
            marks.append(("rs", pc(), time.process_time()))

        def _on_rollout_end(self):
            marks.append(("re", pc(), time.process_time()))
            if len([m for m in marks if m[0] == "re"]) == 1:
                tm = self.training_env.env_method("timing")
                marks.append(("w1", tm))

        def _on_step(self):
            return True

    t0 = pc()
    model.learn(pcf["n_steps"] * n_envs * rollouts, callback=CB())
    t_learn = pc() - t0
    marks.append(("end", pc(), time.process_time()))
    tm = venv.env_method("timing")
    venv.close()
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    ev = [m for m in marks if m[0] in ("rs", "re", "end")]
    roll, upd, roll_cpu, upd_cpu = [], [], [], []
    for a, b in zip(ev, ev[1:]):
        if a[0] == "rs" and b[0] == "re":
            roll.append(b[1] - a[1])
            roll_cpu.append(b[2] - a[2])
        elif a[0] == "re" and b[0] in ("rs", "end"):
            upd.append(b[1] - a[1])
            upd_cpu.append(b[2] - a[2])
    w1 = [m for m in marks if m[0] == "w1"][0][1]
    # worker accounting excluding the first rollout (warm-up)
    sw = sum(t["step_wall"] - a["step_wall"] for t, a in zip(tm, w1))
    sc = sum(t["step_cpu"] - a["step_cpu"] for t, a in zip(tm, w1))
    rw = sum(t["reset_wall"] - a["reset_wall"] for t, a in zip(tm, w1))
    ns = sum(t["n_step"] - a["n_step"] for t, a in zip(tm, w1))
    steady_roll = float(np.sum(roll[1:]))
    steps_per_roll = pcf["n_steps"] * n_envs
    return dict(exp=exp, n_envs=n_envs, threads=threads or torch.get_num_threads(), spawn_s=t_spawn, learn_s=t_learn,
                rollout_s_first=roll[0], rollout_s_steady=float(np.median(roll[1:])),
                update_s_median=float(np.median(upd[:-1] if len(upd) > 1 else upd)),
                rollout_main_cpu_s=float(np.median(roll_cpu[1:])), update_main_cpu_s=float(np.median(upd_cpu)),
                steady_steps_per_s=steps_per_roll / float(np.median(roll[1:])),
                end_to_end_steps_per_s=steps_per_roll * (rollouts - 1) / (steady_roll + float(np.sum(upd[:-1] if len(upd) > 1 else upd))),
                worker_step_us_wall=sw / ns * 1e6, worker_step_us_cpu=sc / ns * 1e6, worker_reset_s=rw,
                worker_busy_frac=(sw + rw) / (steady_roll * n_envs),
                children_cpu_s=(ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime))


def cmd_ppo():
    rows, snaps = [], [snapshot("before")]
    for exp in ("delta", "dual", "wall"):
        for n in (1, 2, 4, 6, 8, 10):
            r = ppo_run(exp, n)
            r["load1"] = os.getloadavg()[0]
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "ppo_scaling.csv", index=False, float_format="%.5f")
    for th in (1, 4):
        r = ppo_run("wall", 8, threads=th)
        r["load1"] = os.getloadavg()[0]
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    snaps.append(snapshot("after"))
    pd.DataFrame(rows).to_csv(OUT / "ppo_scaling.csv", index=False, float_format="%.5f")
    save("ppo", dict(env=env_info(), snapshots=snaps, rows=rows))


def cmd_ppo_repeat():
    """Repeat the n_envs = 8 wall configuration 3x and n_envs 4 / 10 2x (run-to-run variability)."""
    rows = []
    for n, k in ((8, 3), (4, 2), (10, 2)):
        for _ in range(k):
            r = ppo_run("wall", n)
            r["load1"] = os.getloadavg()[0]
            rows.append(r)
            print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ppo_repeat.csv", index=False, float_format="%.5f")


# ------------------------------------------------------------------------------------------------- eval
def cmd_eval():
    import pilot.closedloop as cl
    import pilot.eval_residual as ev

    snaps = [snapshot("before")]
    times = {}

    def tw(mod, attr):
        f = getattr(mod, attr)

        def w(*a, **k):
            t0 = pc()
            r = f(*a, **k)
            times.setdefault(attr, []).append(pc() - t0)
            return r

        setattr(mod, attr, w)

    tw(ev, "run_many")
    tw(ev, "run_chains")
    cfg = R.load_cfg("wall")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        for p in R.EXPS["wall"][1].glob("ckpt_*.zip"):
            (td / p.name).symlink_to(p)
        (td / "final.zip").symlink_to(R.EXPS["wall"][1] / "final.zip")
        t0 = pc()
        ev.select_checkpoint(cfg, td, cfg["residual"]["r_max"])
        times["select_checkpoint_total"] = pc() - t0
        sel_runs = list(times["run_many"])
        times["run_many"] = []
        out = td / "eval"
        sys.argv = ["eval_residual", "--config", R.EXPS["wall"][0], "--model", str(R.EXPS["wall"][1] / R.SELECTED),
                    "--out", str(out), "--no-gif"]
        cwd = os.getcwd()
        os.chdir(R.ROOT)
        t0 = pc()
        ev.main()
        times["eval_main_total"] = pc() - t0
        os.chdir(cwd)
        times["selection_run_many"] = sel_runs
        n_grid = len(pd.read_csv(out / "lap_time_grid.csv"))
        times["n_laps_selection"] = 13 * 4
        times["n_laps_sweep"] = 54
        times["n_laps_chains"] = n_grid
    # pool overhead: empty map with 8 workers (spawn + Track init per worker)
    t0 = pc()
    with cl.ProcessPoolExecutor(8, initializer=cl._init_worker, initargs=(cfg,)) as ex:
        list(ex.map(abs, range(8)))
    times["pool_startup_with_track_init"] = pc() - t0
    snaps.append(snapshot("after"))
    save("eval", dict(env=env_info(), snapshots=snaps, times=times))


# ------------------------------------------------------------------------------------- concurrency vs tail
def _pool_lap(args):
    name, reps = args
    cfg, tr, K, resf = lap_setup(name)
    one_lap(cfg, tr, K, resf)  # warm
    out = []
    for _ in range(reps):
        t0 = pc()
        from pilot.closedloop import LapSim, Scenario  # noqa: F401

        dt, sm = one_lap(cfg, tr, K, resf)
        out.append(dict(lap_s=dt, solve_med=sm["solve_ms_median"], solve_p99=sm["solve_ms_p99"],
                        solve_max=sm["solve_ms_max"], cpu=time.process_time()))
    return out


def cmd_pool():
    """Same lap run concurrently by n worker processes: lap time and MPC solve-time tail vs n (P/E-core topology)."""
    from concurrent.futures import ProcessPoolExecutor

    rows, snaps = [], [snapshot("before")]
    for name, reps in (("S-E (SlipMPC, py)", 3), ("RL δa+wall (rust)", 20)):
        for n in (1, 2, 4, 6, 8, 10):
            with ProcessPoolExecutor(n) as ex:
                t0 = pc()
                res = list(ex.map(_pool_lap, [(name, reps)] * n))
                wall = pc() - t0
            laps = [r for w in res for r in w]
            rows.append(dict(workload=name, n_workers=n, load1=os.getloadavg()[0], wall_s=wall,
                             lap_s_median=np.median([r["lap_s"] for r in laps]),
                             lap_s_p90=np.percentile([r["lap_s"] for r in laps], 90),
                             laps_per_s=len(laps) / sum(r["lap_s"] for r in laps) * n,
                             solve_ms_median=np.median([r["solve_med"] for r in laps]),
                             solve_ms_p99_median=np.median([r["solve_p99"] for r in laps]),
                             solve_ms_p99_max=np.max([r["solve_p99"] for r in laps]),
                             solve_ms_max=np.max([r["solve_max"] for r in laps])))
            print(rows[-1], flush=True)
    snaps.append(snapshot("after"))
    pd.DataFrame(rows).to_csv(OUT / "pool_scaling.csv", index=False, float_format="%.4f")
    save("pool", dict(env=env_info(), snapshots=snaps, rows=rows))


# ----------------------------------------------------------------------------------------------- native
def loop_rl(seconds=12.0):
    cfg, tr, K, resf = lap_setup("RL δa+wall (rust)")
    t_end = time.time() + seconds
    n = 0
    while time.time() < t_end:
        one_lap(cfg, tr, K, resf)
        n += 1
    return n


def loop_se(seconds=12.0):
    cfg, tr, K, resf = lap_setup("S-E (SlipMPC, py)")
    t_end = time.time() + seconds
    n = 0
    while time.time() < t_end:
        one_lap(cfg, tr, K, resf)
        n += 1
    return n


def cmd_native():
    nd = OUT / "native"
    nd.mkdir(exist_ok=True)
    for nm in ("rl", "se"):
        # cProfile (Python-level attribution)
        pr = cProfile.Profile()
        pr.enable()
        (loop_rl if nm == "rl" else loop_se)(8.0)
        pr.disable()
        with open(nd / f"cprofile_{nm}.txt", "w") as f:
            pstats.Stats(pr, stream=f).sort_stats("tottime").print_stats(30)
        with open(nd / f"cprofile_{nm}_cum.txt", "w") as f:
            pstats.Stats(pr, stream=f).sort_stats("cumulative").print_stats(40)
        # native sampling of a separate process (macOS /usr/bin/sample, 1 ms interval)
        p = subprocess.Popen([sys.executable, __file__, f"loop_{nm}"])
        time.sleep(4.0)  # past imports / warm-up
        subprocess.run(["/usr/bin/sample", str(p.pid), "8", "1", "-file", str(nd / f"sample_{nm}.txt")],
                       capture_output=True)
        p.wait()
    print("native profiles written to", nd)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cold":
        cmd_cold()
    elif cmd == "loop_rl":
        loop_rl(14.0)
    elif cmd == "loop_se":
        loop_se(14.0)
    else:
        {"cold_all": cmd_cold_all, "laps": cmd_laps, "steps": cmd_steps, "infer": cmd_infer, "mpc": cmd_mpc,
         "wallobs": cmd_wallobs, "ppo": cmd_ppo, "ppo_repeat": cmd_ppo_repeat, "eval": cmd_eval,
         "native": cmd_native, "pool": cmd_pool}[cmd]()
