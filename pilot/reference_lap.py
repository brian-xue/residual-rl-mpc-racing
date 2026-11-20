"""Reference lap time with a privileged dynamic-model MPC.

The steering MPC predicts with the simulator's own plant model and the TRUE friction of the scenario, and
it measures the plant's sideslip and yaw rate.

* plant ``st``  -> exact linear ST equations below (``vehicle_dynamics_st``).
* plant ``std`` -> ``STDMPC``: the true PAC2002 STD model (``vehicle_dynamics_std``, true tire friction) is
  linearised at its steady-state cornering point for every (speed, curvature) of the horizon (cached on a
  0.25 m/s x 0.01 1/m grid; finite-difference Jacobians of beta' and r' w.r.t. beta, r, delta at free-rolling
  wheels and zero acceleration). Beyond the grip limit, where no steady state exists, the saturated steady
  state at the largest feasible curvature is used, so the MPC predicts that the car cannot hold the line.
 It is the proposal's "ground-truth-model reference": it shows what lap is
achievable under the same reference line, speed plans, longitudinal controller, steering limits, servo,
start poses and lap-validity checks (four-tire track limits, body-wall contact). It is not a competitor.

Model (path frame, speed v and acceleration a taken from the speed plan along the horizon):
    e_y'   = v (e_psi + beta)
    e_psi' = r - v kappa_ref
    r'     = mu m / (I L) [ lf Cf glr delta + (lr Cr glf - lf Cf glr) beta - (lf^2 Cf glr + lr^2 Cr glf) r / v ]
    beta'  = mu / (v L) [ Cf glr delta - (Cr glf + Cf glr) beta + (Cr glf lr - Cf glr lf) r / v ] - r
    glr = g lr - a h,  glf = g lf + a h            (identical to single_track.py for v >= 0.5 m/s)
Cost: q_y e_y^2 + q_psi (e_psi + beta)^2 + r_du du^2 (course error instead of heading error), same weights,
horizon and rates as the kinematic MPC; hard |delta| <= s_max, |du| <= sv_max Ts. Condensed QP, OSQP.

Usage:  uv run python -m pilot.reference_lap [--config configs/pilot.yaml] [--out DIR]   (DIR defaults to the config's `out`)
(run ``pilot.run_pilot`` first: the comparison uses its K and lap_time_comparison.csv)
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp
from gymkhana.envs.dynamic_models.single_track_drift.single_track_drift import vehicle_dynamics_std
from scipy.linalg import expm
from scipy.optimize import fsolve

from . import animate, cli
from .closedloop import Scenario, Track, qualifies, run_chains, run_lap
from .sim import plant_model, vehicle_params

G = 9.81


class DynamicMPC:
    def __init__(self, params, Ts, N, q_y, q_psi, r_du, q_terminal, s_max, du_max, tau):
        self.p, self.Ts, self.N = params, Ts, N
        self.q_y, self.q_psi, self.r_du, self.qT = q_y, q_psi, r_du, q_terminal
        self.s_max, self.du_max, self.tau = s_max, du_max, tau
        self.L = params["lf"] + params["lr"]
        # constant constraint matrix: delta_k = delta_0 + sum_{j<k} u_j (k = 1..N), and u itself
        Lmat = np.tril(np.ones((N, N)))
        self.A = sp.csc_matrix(np.vstack([Lmat, np.eye(N)]))
        self.iu, self.ju = [], []
        for j in range(N):
            for i in range(j + 1):
                self.iu.append(i), self.ju.append(j)
        self.iu, self.ju = np.array(self.iu), np.array(self.ju)
        self.solver = None
        H = np.zeros((2, 5))
        H[0, 0] = 1.0
        H[1, 1] = H[1, 2] = 1.0  # course error e_psi + beta
        self.H = H

    def _cont(self, v, a, kappa):
        p = self.p
        v = max(v, 0.5)
        lf, lr, Cf, Cr, mu, m, I, h, L = p["lf"], p["lr"], p["C_Sf"], p["C_Sr"], p["mu"], p["m"], p["I"], p["h"], self.L
        glr, glf = G * lr - a * h, G * lf + a * h
        c1, c2 = mu * m / (I * L), mu / (v * L)
        A = np.zeros((4, 4))  # [e_y, e_psi, beta, r]
        A[0, 1] = A[0, 2] = v
        A[1, 3] = 1.0
        A[2, 2] = -c2 * (Cr * glf + Cf * glr)
        A[2, 3] = c2 * (Cr * glf * lr - Cf * glr * lf) / v - 1.0
        A[3, 2] = c1 * (lr * Cr * glf - lf * Cf * glr)
        A[3, 3] = -c1 * (lf * lf * Cf * glr + lr * lr * Cr * glf) / v
        B = np.array([0.0, 0.0, c2 * Cf * glr, c1 * lf * Cf * glr])
        c = np.array([0.0, -v * kappa, 0.0, 0.0])
        return A, B, c

    def _disc(self, v, a, kappa, T):
        A, B, c = self._cont(v, a, kappa)
        M = np.zeros((6, 6))
        M[:4, :4], M[:4, 4], M[:4, 5] = A, B, c
        E = expm(M * T)
        return E[:4, :4], E[:4, 4], E[:4, 5]

    def solve(self, st, e_y, e_psi, s, delta_prev, vplan_at, ref):
        t0 = time.perf_counter()
        N, Ts = self.N, self.Ts
        v, beta, r, delta = st[3], st[6], st[5], st[2]
        dvds = lambda s_: (vplan_at(s_ + 0.2) - vplan_at(s_ - 0.2)) / 0.4  # noqa: E731
        # delay compensation with the true dynamic model and the current steering angle
        Phi, Gam, w = self._disc(v, v * dvds(s), ref.kappa_at(s), self.tau)
        x0 = Phi @ np.array([e_y, e_psi, beta, r]) + Gam * delta + w
        s_p = s + self.tau * v
        s_k = s_p + np.r_[0.0, np.cumsum(Ts * np.maximum(vplan_at(s_p + np.arange(N - 1) * Ts * v), 0.5))]
        v_seq = vplan_at(s_k)
        v_seq[0] = v
        kap = ref.kappa_at(s_k)
        z0 = np.r_[x0, delta_prev]
        Sx, Su, Sc = np.eye(5), np.zeros((5, N)), np.zeros(5)
        P = np.zeros((N, N))
        q = np.zeros(N)
        Q = np.diag([self.q_y, self.q_psi])
        for k in range(N):
            Phi, Gam, w = self._disc(v_seq[k], v_seq[k] * dvds(s_k[k]), kap[k], Ts)
            Ad = np.zeros((5, 5))
            Ad[:4, :4], Ad[:4, 4], Ad[4, 4] = Phi, Gam, 1.0
            Bd = np.r_[Gam, 1.0]
            cd = np.r_[w, 0.0]
            Sx, Su, Sc = Ad @ Sx, Ad @ Su, Ad @ Sc + cd
            Su[:, k] += Bd
            wk = self.qT if k == N - 1 else 1.0
            HS = self.H @ Su
            HQ = wk * HS.T @ Q
            P += HQ @ HS
            q += HQ @ (self.H @ (Sx @ z0 + Sc))
        P += self.r_du * np.eye(N)
        P = 2 * P
        q = 2 * q
        lo = np.r_[np.full(N, -self.s_max - delta_prev), np.full(N, -self.du_max)]
        hi = np.r_[np.full(N, self.s_max - delta_prev), np.full(N, self.du_max)]
        Px = P[self.iu, self.ju]
        if self.solver is None:
            self.solver = osqp.OSQP()
            self.solver.setup(sp.csc_matrix((Px, (self.iu, self.ju)), shape=(N, N)), q, self.A, lo, hi,
                              verbose=False, eps_abs=1e-6, eps_rel=1e-6, polishing=False, warm_starting=True,
                              max_iter=4000)
        else:
            self.solver.update(Px=Px, q=q, l=lo, u=hi)
        res = self.solver.solve()
        ok = res.info.status_val in (1, 2) and res.x is not None and np.all(np.isfinite(res.x))
        dt = time.perf_counter() - t0
        if not ok:
            return float(delta_prev), False, dt
        return float(delta_prev + res.x[0]), True, dt


class STDMPC(DynamicMPC):
    """Same QP; the lateral model is the true STD model linearised at steady-state cornering."""

    DV, DK = 0.25, 0.01

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._table, self._kmax = {}, {}

    def _lat(self, beta, r, delta, v):
        p = self.p
        u_wf = max(0.0, v * np.cos(beta) * np.cos(delta) + (v * np.sin(beta) + p["lf"] * r) * np.sin(delta))
        u_wr = max(0.0, v * np.cos(beta))
        x = np.array([0.0, 0.0, delta, v, 0.0, r, beta, u_wf / p["R_w"], u_wr / p["R_w"]])
        f = vehicle_dynamics_std(x, np.array([0.0, 0.0]), p)
        return np.array([f[6], f[5]])  # beta', r'

    def _steady(self, v, kappa):
        """(beta, delta) with r = v kappa at steady state, or None if beyond the grip limit."""
        r = v * kappa
        guess = np.array([self.p["lr"] * kappa, np.arctan(self.L * kappa)])
        sol, info, ier, _ = fsolve(lambda z: self._lat(z[0], r, z[1], v), guess, full_output=True, xtol=1e-10)
        if ier != 1 or np.max(np.abs(info["fvec"])) > 1e-6 or abs(sol[1]) > 0.6 or abs(sol[0]) > 0.5:
            return None
        return sol

    def _entry(self, iv, ik):
        key = (iv, ik)
        if key in self._table:
            return self._table[key]
        v, kappa = max(iv * self.DV, 0.5), ik * self.DK
        ss = self._steady(v, kappa)
        if ss is None:  # beyond grip: largest feasible curvature at this speed (bisection on the grid)
            if iv not in self._kmax:
                lo, hi = 0, ik
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    lo, hi = (mid, hi) if self._steady(v, mid * self.DK) is not None else (lo, mid)
                self._kmax[iv] = lo
            ent = self._entry(iv, self._kmax[iv])
            self._table[key] = ent
            return ent
        beta, delta = ss
        r = v * kappa
        x0 = np.array([beta, r, delta])
        J = np.zeros((2, 3))
        h = 1e-5
        for j in range(3):
            xp, xm = x0.copy(), x0.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (self._lat(*xp, v) - self._lat(*xm, v)) / (2 * h)
        c = self._lat(beta, r, delta, v) - J @ x0
        self._table[key] = (J, c)
        return self._table[key]

    def _cont(self, v, a, kappa):
        v = max(v, 0.5)
        sgn = 1.0 if kappa >= 0 else -1.0
        J, c = self._entry(int(round(v / self.DV)), int(round(abs(kappa) / self.DK)))
        A = np.zeros((4, 4))  # [e_y, e_psi, beta, r]
        A[0, 1] = A[0, 2] = v
        A[1, 3] = 1.0
        A[2:, 2:] = J[:, :2]
        B = np.array([0.0, 0.0, J[0, 2], J[1, 2]])
        cc = np.array([0.0, -v * kappa, sgn * c[0], sgn * c[1]])  # odd symmetry of the lateral dynamics
        return A, B, cc


def make_planner(cfg, mu_scale):
    mc, lim = cfg["mpc"], cfg["limits"]
    cls = STDMPC if plant_model() == "std" else DynamicMPC
    return cls(vehicle_params(mu_scale), mc["Ts"], mc["N"], mc["q_y"], mc["q_psi"], mc["r_du"],
               mc["q_terminal"], lim["s_max"], lim["sv_max"] * mc["Ts"], mc["delay_comp_s"])


def main():
    cfg, _, out = cli.parse(__doc__)
    rout = out / "reference"
    rout.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    track = Track(cfg)
    lc, sw = cfg["lap_time"], cfg["sweep"]
    starts = (sw["start_s"], sw["recheck_start_s"])
    rc = cfg.get("reference_lap", {})
    grid = list(lc["a_lat_grid"]) + [a for a in rc.get("extra_a_lat", []) if a > max(lc["a_lat_grid"])]

    # fastest qualifying level of the reference controller (same contiguous rule as pilot.lap_time)
    rows, traces, summs, best = [], {}, {}, []
    chains = run_chains(cfg, [(0.0, mu, grid, starts, make_planner, "reference") for mu in lc["mu_scales"]])
    for mu, chain in zip(lc["mu_scales"], chains):
        for a_lat, s0, sm, tr in chain:
            rows.append({c: sm[c] for c in ("controller", "mu_scale", "a_lat", "start_s", "status", "qualifying",
                                            "progress_frac", "fail_s", "lap_time", "plan_lap_time", "lap_time_ratio",
                                            "max_ey", "rms_ey", "solver_failures", "solve_ms_median", "solve_ms_p99")})
            traces[(mu, a_lat, s0)], summs[(mu, a_lat, s0)] = tr, sm
        ok_levels = []
        for a in grid:
            if not all((mu, a, s0) in summs and summs[(mu, a, s0)]["qualifying"] for s0 in starts):
                break
            ok_levels.append(a)
        fails = [f"a_lat {a_} {sm_['status']} @ s={sm_['fail_s']}" for a_, _, sm_, _ in chain if not sm_["qualifying"]]
        print(f"   mu={mu}: qualifying levels {ok_levels}; first failure: {fails[0] if fails else 'none'}", flush=True)
        a = max(ok_levels) if ok_levels else None
        best.append(dict(mu_scale=mu, controller="reference", best_a_lat=a,
                         lap_time_s0=summs[(mu, a, starts[0])]["lap_time"] if a else None,
                         lap_time_s1=summs[(mu, a, starts[1])]["lap_time"] if a else None,
                         plan_lap_time=summs[(mu, a, starts[0])]["plan_lap_time"] if a else None,
                         max_ey=max(summs[(mu, a, s)]["max_ey"] for s in starts) if a else None,
                         beyond_shared_grid=bool(a is not None and a > max(lc["a_lat_grid"])),
                         grid_exhausted=bool(a == grid[-1])))
    pd.DataFrame(rows).to_csv(rout / "reference_grid.csv", index=False, float_format="%.4f")
    rdf = pd.DataFrame(best)

    # combine with the pilot's nominal / feedforward fastest qualifying laps
    comp_path = out / "lap_time_comparison.csv"
    if not comp_path.exists():
        raise SystemExit("results/lap_time_comparison.csv missing: run `uv run python -m pilot.run_pilot` first")
    comp = pd.read_csv(comp_path)
    allc = pd.concat([comp[["mu_scale", "controller", "best_a_lat", "lap_time_s0", "lap_time_s1", "plan_lap_time",
                            "max_ey"]], rdf.drop(columns=["grid_exhausted", "beyond_shared_grid"])], ignore_index=True)
    ref_t = rdf.set_index("mu_scale").lap_time_s0
    allc["gap_to_reference_s"] = [r.lap_time_s0 - ref_t.get(r.mu_scale, np.nan) for r in allc.itertuples()]
    allc.to_csv(rout / "reference_lap_time.csv", index=False, float_format="%.3f")
    print(allc.to_string(), flush=True)

    # bar chart: three controllers per friction
    names = animate.NAMES
    colors = {"nominal": "#c0392b", "ff": "#1f6fb4", "reference": "#8e44ad"}
    mus = list(lc["mu_scales"])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    w = 0.27
    for j, c in enumerate(("nominal", "ff", "reference")):
        for i, mu in enumerate(mus):
            r = allc[(allc.mu_scale == mu) & (allc.controller == c)].iloc[0]
            x = i + (j - 1) * w
            if pd.isna(r.lap_time_s0):
                ax.text(x, 50, "none", ha="center", fontsize=7)
                continue
            ax.bar(x, r.lap_time_s0, w, color=colors[c], label=names[c] if i == 0 else None)
            ax.text(x, r.lap_time_s0 + 0.3, f"{r.lap_time_s0:.1f}\na_lat {r.best_a_lat:g}", ha="center", va="bottom",
                    fontsize=7)
    ax.set_ylim(np.nanmin(allc.lap_time_s0) - 8, np.nanmax(allc.lap_time_s0) + 6)
    ax.set_xticks(range(len(mus)), [f"μ/μ₀ = {m}" for m in mus])
    ax.set_ylabel("fastest qualifying lap [s]")
    ax.set_title(f"{cfg['track']}: fastest qualifying lap, with the dynamic-model reference (true μ)\n"
                 "speed level raised until the lap stops qualifying from both start poses", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(rout / "reference_lap_time.png", dpi=130)
    plt.close(fig)

    # GIF: the three controllers at their own fastest qualifying level, same friction and start
    mu_g = rc.get("gif_mu_scale", 0.6)
    K = json.loads((out / "report.json").read_text())["K_frozen"]
    gtr, gsm = {}, {}
    for c in ("nominal", "ff", "reference"):
        r = allc[(allc.mu_scale == mu_g) & (allc.controller == c)].iloc[0]
        if pd.isna(r.best_a_lat):
            continue
        if c == "reference":
            gtr[c], gsm[c] = traces[(mu_g, r.best_a_lat, starts[0])], summs[(mu_g, r.best_a_lat, starts[0])]
        else:
            sm, tr = run_lap(track, Scenario(float(r.best_a_lat), mu_g, starts[0]), K if c == "ff" else 0.0, cfg)
            sm.update(qualifying=qualifies(sm, cfg["qualifying"])[0], status=qualifies(sm, cfg["qualifying"])[1])
            gtr[c], gsm[c] = tr, sm
    for c, tr in gtr.items():
        tr.to_csv(rout / f"trace_mu{mu_g}_{c}.csv", index=False, float_format="%.5f")
    gif = animate.render(track, cfg, gtr, gsm, rout / "reference_lap.gif")
    report = dict(K=K, a_lat_grid=grid, gif=gif, gif_mu_scale=mu_g, reference=rdf.to_dict(orient="records"),
                  comparison=allc.to_dict(orient="records"), runtime_s=time.time() - t0)
    (rout / "reference_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"done in {time.time() - t0:.0f} s -> {rout}/", flush=True)


if __name__ == "__main__":
    main()
