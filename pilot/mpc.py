"""Steering-only kinematic-bicycle MPC, linearised along the reference and solved with OSQP.

Model (CG reference point, matching the ST plant's (x, y)):
    beta(d)   = atan(lr/L * tan d)
    e_y'      = v sin(e_psi + beta(d))
    e_psi'    = v cos(beta(d)) tan(d) / L - v kappa_ref
State z = [e_y, e_psi, delta_prev]; input u = steering change per MPC step.
The steering applied over interval k is delta_k + u_k (= delta_{k+1}).
Cost: q_y e_y^2 + q_psi (e_psi - e_psi_ref)^2 + r_du u^2, with e_psi_ref = -beta(delta_ref)
(the kinematic body-slip that a car exactly on the path would have).
Hard constraints: |delta| <= s_max, |u| <= sv_max * Ts.
"""

from __future__ import annotations

import time

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import expm


class KinematicMPC:
    def __init__(self, lf, lr, Ts, N, q_y, q_psi, r_du, q_terminal, s_max, du_max):
        self.lf, self.lr, self.L = lf, lr, lf + lr
        self.Ts, self.N = Ts, N
        self.q_y, self.q_psi, self.r_du, self.qT = q_y, q_psi, r_du, q_terminal
        self.s_max, self.du_max = s_max, du_max
        self._build()

    # ---------------------------------------------------------------- model
    def _beta(self, d):
        return np.arctan(self.lr / self.L * np.tan(d))

    def _f(self, ey, epsi, d, v, kappa):
        b = self._beta(d)
        return np.array([v * np.sin(epsi + b), v * np.cos(b) * np.tan(d) / self.L - v * kappa])

    def _discretize(self, v, kappa):
        """Exact ZOH of the Jacobian linearisation at (0, -beta_r, delta_r)."""
        d_r = np.arctan(self.L * kappa)
        e0 = np.array([0.0, -self._beta(d_r)])
        f0 = self._f(e0[0], e0[1], d_r, v, kappa)
        h = 1e-6
        Ac = np.array(
            [
                (self._f(h, e0[1], d_r, v, kappa) - f0) / h,
                (self._f(0.0, e0[1] + h, d_r, v, kappa) - f0) / h,
            ]
        ).T
        Bc = (self._f(0.0, e0[1], d_r + h, v, kappa) - f0) / h
        cc = f0 - Ac @ e0 - Bc * d_r
        M = np.zeros((4, 4))
        M[:2, :2], M[:2, 2], M[:2, 3] = Ac, Bc, cc
        E = expm(M * self.Ts)
        Phi, Gam, w = E[:2, :2], E[:2, 2], E[:2, 3]
        Ad = np.zeros((3, 3))
        Ad[:2, :2], Ad[:2, 2], Ad[2, 2] = Phi, Gam, 1.0
        Bd = np.r_[Gam, 1.0]
        cd = np.r_[w, 0.0]
        return Ad, Bd, cd, e0[1]

    # ---------------------------------------------------------------- QP
    def _build(self):
        N, nz = self.N, 3
        self.nx = nz * N + N  # z_1..z_N, u_0..u_{N-1}
        zi = lambda k: nz * (k - 1)  # index of z_k, k = 1..N  # noqa: E731
        ui = lambda k: nz * N + k  # index of u_k  # noqa: E731
        self._zi, self._ui = zi, ui
        # Equality rows: z_{k+1} - Ad_k z_k - Bd_k u_k = cd_k  (z_0 term moved to rhs)
        rows, cols, self._slots = [], [], []
        r = 0
        for k in range(N):
            for i in range(nz):
                rows.append(r + i), cols.append(zi(k + 1) + i), self._slots.append(("I", k, i, i))
                if k > 0:
                    for j in range(nz):
                        rows.append(r + i), cols.append(zi(k) + j), self._slots.append(("A", k, i, j))
                rows.append(r + i), cols.append(ui(k)), self._slots.append(("B", k, i, 0))
            r += nz
        n_eq = r
        # inequality rows: delta_k (k=1..N) and u_k bounds
        for k in range(1, N + 1):
            rows.append(r), cols.append(zi(k) + 2), self._slots.append(("one", 0, 0, 0))
            r += 1
        for k in range(N):
            rows.append(r), cols.append(ui(k)), self._slots.append(("one", 0, 0, 0))
            r += 1
        self.n_eq, self.m = n_eq, r
        marker = np.arange(1, len(rows) + 1, dtype=float)
        Am = sp.csc_matrix((marker, (rows, cols)), shape=(self.m, self.nx))
        self._perm = (Am.data - 1).astype(int)  # csc data position -> slot index
        self._A_struct = Am
        # cost: diagonal P
        P = np.zeros(self.nx)
        for k in range(1, N + 1):
            w = self.qT if k == N else 1.0
            P[zi(k)] = self.q_y * w
            P[zi(k) + 1] = self.q_psi * w
        for k in range(N):
            P[ui(k)] = self.r_du
        self._Pdiag = P
        self.l = np.zeros(self.m)
        self.u = np.zeros(self.m)
        self.l[n_eq : n_eq + N], self.u[n_eq : n_eq + N] = -self.s_max, self.s_max
        self.l[n_eq + N :], self.u[n_eq + N :] = -self.du_max, self.du_max
        self.solver = None

    def _A_values(self, Ads, Bds):
        vals = np.empty(len(self._slots))
        for n, (kind, k, i, j) in enumerate(self._slots):
            if kind == "I" or kind == "one":
                vals[n] = 1.0
            elif kind == "A":
                vals[n] = -Ads[k][i, j]
            else:
                vals[n] = -Bds[k][i]
        return vals[self._perm]

    def solve(self, e_y, e_psi, delta_prev, v_seq, kappa_seq):
        """v_seq, kappa_seq: length-N previews. Returns (delta_cmd, status_ok, solve_time_s, pred)."""
        N = self.N
        Ads, Bds, cds, epr = [], [], [], []
        for k in range(N):
            Ad, Bd, cd, er = self._discretize(max(v_seq[k], 0.5), kappa_seq[k])
            Ads.append(Ad), Bds.append(Bd), cds.append(cd), epr.append(er)
        z0 = np.array([e_y, e_psi, delta_prev])
        beq = np.concatenate([cds[k] + (Ads[0] @ z0 if k == 0 else 0.0) for k in range(N)])
        self.l[: self.n_eq] = beq
        self.u[: self.n_eq] = beq
        q = np.zeros(self.nx)
        for k in range(1, N + 1):
            w = self.qT if k == N else 1.0
            ref = epr[min(k, N - 1)]
            q[self._zi(k) + 1] = -self.q_psi * w * ref
        Ax = self._A_values(Ads, Bds)
        t0 = time.perf_counter()
        if self.solver is None:
            A = self._A_struct.copy()
            A.data = Ax
            self.solver = osqp.OSQP()
            self.solver.setup(
                sp.diags(self._Pdiag).tocsc(), q, A, self.l, self.u,
                verbose=False, eps_abs=1e-5, eps_rel=1e-5, polishing=True, warm_starting=True, max_iter=4000,
            )
        else:
            self.solver.update(q=q, l=self.l, u=self.u, Ax=Ax)
        res = self.solver.solve()
        dt = time.perf_counter() - t0
        ok = res.info.status_val in (1, 2)  # solved / solved inaccurate
        if not ok or res.x is None or not np.all(np.isfinite(res.x)):
            return delta_prev, False, dt, None
        x = res.x
        delta_cmd = delta_prev + x[self._ui(0)]
        pred = x[: 3 * N].reshape(N, 3)
        return float(delta_cmd), True, dt, pred


class CoupledMPC:
    """Steering + acceleration MPC: the lateral model of ``KinematicMPC`` extended with speed.

    State z = [e_y, e_psi, delta, v, a]; input u = [d_delta, d_a] per MPC step. As for the steering, the
    acceleration applied over interval k is a_k + d_a_k (= a_{k+1}), so both channels are rate-penalised.
        lateral   : exact ZOH of the kinematic error model, linearised along the reference and scheduled on
                    the MPC's own predicted speed (previous solution, shifted)
        speed     : v_{k+1} = v_k + Ts a_{k+1}      (the ST plant integrates the command exactly)
    Cost: q_y e_y^2 + q_psi (e_psi - e_psi_ref)^2 + q_v (v - v_ref)^2 + q_a (a - a_ref)^2
          + r_du d_delta^2 + r_da d_a^2 (terminal x q_terminal on e_y, e_psi, v) + slack penalty.
    Hard constraints: |delta| <= s_max, |d_delta| <= du_max, -a_max <= a <= a_max min(1, v_switch / v),
    |d_a| <= da_max, v >= v_min.
    Optional friction ellipse (a / ax_max)^2 + (v^2 kappa / ay_max)^2 <= 1, as an n-facet outer polygon,
    linearised in v around the predicted speed and softened with one slack per stage (keeps the QP feasible
    when the current speed is already too high for the corner ahead).
    """

    NZ, NU = 5, 2

    def __init__(self, lf, lr, Ts, N, lat_w, lon_w, s_max, du_max, a_max, v_switch, ellipse=None):
        self.lat = KinematicMPC(lf, lr, Ts, N, lat_w["q_y"], lat_w["q_psi"], lat_w["r_du"], lat_w["q_terminal"],
                                s_max, du_max)  # used for its lateral model only
        self.Ts, self.N = Ts, N
        self.w, self.lw = lat_w, lon_w
        self.s_max, self.du_max = s_max, du_max
        self.a_max, self.v_switch = a_max, v_switch
        self.ell = ellipse  # None or dict(ax_max, ay_max, n_facets, slack_l1, slack_l2)
        n_f = ellipse["n_facets"] if ellipse else 0
        th = np.arange(n_f) * 2 * np.pi / max(n_f, 1)
        # outer polygon: facet j touches the unit circle at angle th_j; scale 1/cos(pi/n) would make it inner
        self._fc, self._fs = np.cos(th), np.sin(th)
        nz, nu = self.NZ, self.NU
        self.n_ns = N if ellipse else 0  # slack variables
        self.nx = nz * N + nu * N + self.n_ns
        self.n_eq = nz * N
        self.n_in = N * 3 + N * 2 + N * n_f + self.n_ns  # delta, a, v | d_delta, d_a | facets | slack >= 0
        self.m = self.n_eq + self.n_in
        self.solver = None
        self.v_lin = None
        P = np.zeros(self.nx)
        for k in range(1, N + 1):
            wT = self.w["q_terminal"] if k == N else 1.0
            z = self._z(k)
            P[z], P[z + 1] = self.w["q_y"] * wT, self.w["q_psi"] * wT
            P[z + 3], P[z + 4] = self.lw["q_v"] * wT, self.lw["q_a"]
        for k in range(N):
            P[self._u(k)], P[self._u(k) + 1] = self.w["r_du"], self.lw["r_da"]
        if ellipse:
            P[nz * N + nu * N:] = ellipse["slack_l2"]
        self._P = sp.diags(P).tocsc()
        self._mask = None

    def _z(self, k):  # index of z_k, k = 1..N
        return self.NZ * (k - 1)

    def _u(self, k):  # index of u_k, k = 0..N-1
        return self.NZ * self.N + self.NU * k

    def _a_upper(self, v):
        return self.a_max * min(1.0, self.v_switch / max(v, 1e-3))

    def _build(self, Ads, Bds, kap, v_lin):
        """Dense constraint matrix (fixed structure: every structural entry is written, zero or not)."""
        N, nz = self.N, self.NZ
        A = np.zeros((self.m, self.nx))
        for k in range(N):
            r = nz * k
            A[r:r + nz, self._z(k + 1):self._z(k + 1) + nz] = np.eye(nz)
            if k > 0:
                A[r:r + nz, self._z(k):self._z(k) + nz] = -Ads[k]
            A[r:r + nz, self._u(k):self._u(k) + 2] = -Bds[k]
        r = self.n_eq
        for k in range(1, N + 1):
            z = self._z(k)
            A[r, z + 2] = 1.0  # delta
            A[r + 1, z + 4] = 1.0  # a
            A[r + 2, z + 3] = 1.0  # v
            r += 3
        for k in range(N):
            A[r, self._u(k)] = 1.0
            A[r + 1, self._u(k) + 1] = 1.0
            r += 2
        if self.ell:
            ax, ay = self.ell["ax_max"], self.ell["ay_max"]
            for k in range(1, N + 1):
                z, j = self._z(k), self.NZ * N + self.NU * N + (k - 1)
                for c, s in zip(self._fc, self._fs):
                    A[r, z + 4] = c / ax  # a
                    A[r, z + 3] = s / ay * 2.0 * v_lin[k - 1] * kap[k - 1]  # a_y ~ kappa (2 v_lin v - v_lin^2)
                    A[r, j] = -1.0
                    r += 1
            for k in range(N):
                A[r, self.NZ * N + self.NU * N + k] = 1.0
                r += 1
        return A

    def _discretize(self, v, kappa):
        Ad3, Bd3, cd3, e_psi_ref = self.lat._discretize(max(v, 0.5), kappa)  # z3 = [e_y, e_psi, delta]
        Ad, Bd, cd = np.zeros((5, 5)), np.zeros((5, 2)), np.zeros(5)
        Ad[:3, :3], Bd[:3, 0], cd[:3] = Ad3, Bd3, cd3
        Ad[3, 3], Ad[3, 4], Bd[3, 1] = 1.0, self.Ts, self.Ts
        Ad[4, 4], Bd[4, 1] = 1.0, 1.0
        return Ad, Bd, cd, e_psi_ref

    def solve(self, e_y, e_psi, delta_prev, v, a_prev, s0, v_ref_at, a_ref_at, kappa_at, v_min=0.5):
        """One MPC step from the (delay-compensated) state at arc length s0.
        ``v_ref_at``/``a_ref_at``/``kappa_at``: callables of arc length. Returns
        (delta_cmd, a_cmd, ok, solve_time_s, pred) with pred the (N, 5) predicted states."""
        N, Ts = self.N, self.Ts
        # v_lin[k-1] = speed of z_k used for linearisation: previous prediction shifted by one step, else the plan.
        # Positions follow KinematicMPC: p_0 = s0, p_k = p_{k-1} + Ts * (speed at the start of interval k-1);
        # interval k is discretised at (speed, kappa) of its start, z_k is referenced at p_k.
        if self.v_lin is None:
            v_lin, s = np.empty(N), s0 + Ts * v
            for k in range(N):
                v_lin[k] = max(float(v_ref_at(s)), 0.5)
                s += Ts * v_lin[k]
        else:
            v_lin = np.r_[self.v_lin[1:], self.v_lin[-1]]
        v_lin = np.maximum(v_lin, 0.5)
        v_start = np.r_[v, v_lin[:-1]]  # speed at the start of interval k
        p = s0 + Ts * np.r_[0.0, np.cumsum(v_start)]  # p[k], k = 0..N
        kap_p = kappa_at(p)
        kap = kap_p[1:]  # at z_1..z_N (friction ellipse)
        v_ref, a_ref = v_ref_at(p[1:]), a_ref_at(p[1:])
        Ads, Bds, cds, epr = [], [], [], []
        for k in range(N):
            Ad, Bd, cd, er = self._discretize(v_start[k], kap_p[k])
            Ads.append(Ad), Bds.append(Bd), cds.append(cd), epr.append(er)
        z0 = np.array([e_y, e_psi, delta_prev, v, a_prev])
        beq = np.concatenate([cds[k] + (Ads[0] @ z0 if k == 0 else 0.0) for k in range(N)])
        lo, hi = [beq], [beq]
        lo.append(np.ravel([[-self.s_max, -self.a_max, v_min] for _ in range(N)]))
        hi.append(np.ravel([[self.s_max, self._a_upper(v_lin[k]), 1e9] for k in range(N)]))
        lo.append(np.tile([-self.du_max, -self.lw["da_max"]], N))
        hi.append(np.tile([self.du_max, self.lw["da_max"]], N))
        q = np.zeros(self.nx)
        for k in range(1, N + 1):
            wT = self.w["q_terminal"] if k == N else 1.0
            z = self._z(k)
            q[z + 1] = -self.w["q_psi"] * wT * epr[min(k, N - 1)]
            q[z + 3] = -self.lw["q_v"] * wT * v_ref[k - 1]
            q[z + 4] = -self.lw["q_a"] * a_ref[k - 1]
        if self.ell:
            ay = self.ell["ay_max"]
            rhs = [1.0 + s / ay * kap[k] * v_lin[k] ** 2 for k in range(N) for s in self._fs]
            lo.append(np.full(len(rhs), -1e9)), hi.append(np.array(rhs))
            lo.append(np.zeros(N)), hi.append(np.full(N, 1e9))
            q[self.NZ * N + self.NU * N:] = self.ell["slack_l1"]
        l, u = np.concatenate(lo), np.concatenate(hi)
        A = self._build(Ads, Bds, kap, v_lin)
        t0 = time.perf_counter()
        if self.solver is None:
            self._mask = self._build([np.ones((5, 5))] * N, [np.ones((5, 2))] * N, np.ones(N), np.ones(N)) != 0
            Acsc = sp.csc_matrix(self._mask.astype(float))
            Acsc.data = A.T[self._mask.T]
            self.solver = osqp.OSQP()
            self.solver.setup(self._P, q, Acsc, l, u, verbose=False, eps_abs=1e-5, eps_rel=1e-5, polishing=True,
                              warm_starting=True, max_iter=4000)
        else:
            self.solver.update(q=q, l=l, u=u, Ax=A.T[self._mask.T])
        res = self.solver.solve()
        dt = time.perf_counter() - t0
        ok = res.info.status_val in (1, 2)
        if not ok or res.x is None or not np.all(np.isfinite(res.x)):
            self.v_lin = None
            return delta_prev, a_prev, False, dt, None
        x = res.x
        pred = x[: self.NZ * N].reshape(N, self.NZ)
        self.v_lin = np.maximum(pred[:, 3], 0.5)
        u0 = x[self._u(0):self._u(0) + 2]
        return float(delta_prev + u0[0]), float(a_prev + u0[1]), True, dt, pred
