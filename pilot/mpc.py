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

    def _discretize_many(self, v, kappa):
        """Exact ZOH of the Jacobian linearisation at (0, -beta_r, delta_r) for every stage at once.

        v, kappa: (n,) arrays. Returns Ad (n, 3, 3), Bd (n, 3), cd (n, 3), e_psi_ref (n,). Forward differences
        (h = 1e-6) and one stacked expm; bit-identical to evaluating the stages one by one."""
        v, kappa = np.asarray(v, dtype=float), np.asarray(kappa, dtype=float)
        n, L, h = len(v), self.L, 1e-6

        def f(epsi, d):
            b = self._beta(d)
            return v * np.sin(epsi + b), v * np.cos(b) * np.tan(d) / L - v * kappa

        d_r = np.arctan(L * kappa)
        e1 = -self._beta(d_r)  # e0 = (0, e1)
        f0, fp, fd = f(e1, d_r), f(e1 + h, d_r), f(e1, d_r + h)
        M = np.zeros((n, 4, 4))
        for i in range(2):
            # column 0 (d/d e_y) stays 0: the error model does not depend on e_y
            M[:, i, 1] = (fp[i] - f0[i]) / h
            M[:, i, 2] = (fd[i] - f0[i]) / h
            M[:, i, 3] = f0[i] - (M[:, i, 0] * 0.0 + M[:, i, 1] * e1) - M[:, i, 2] * d_r  # f0 - Ac e0 - Bc d_r
        E = expm(M * self.Ts)
        Ad = np.zeros((n, 3, 3))
        Ad[:, :2, :2], Ad[:, :2, 2], Ad[:, 2, 2] = E[:, :2, :2], E[:, :2, 2], 1.0
        Bd = np.zeros((n, 3))
        Bd[:, :2], Bd[:, 2] = E[:, :2, 2], 1.0
        cd = np.zeros((n, 3))
        cd[:, :2] = E[:, :2, 3]
        return Ad, Bd, cd, e1

    def _discretize(self, v, kappa):
        """Single stage of ``_discretize_many``: (Ad, Bd, cd, e_psi_ref)."""
        Ad, Bd, cd, e1 = self._discretize_many([v], [kappa])
        return Ad[0], Bd[0], cd[0], e1[0]

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
        # gather indices for _A_values: slot position <- flat index into the stacked Ad (N, 3, 3) / Bd (N, 3)
        self._A_pos = np.array([n for n, s in enumerate(self._slots) if s[0] == "A"], dtype=int)
        self._A_src = np.array([k * 9 + i * 3 + j for kind, k, i, j in self._slots if kind == "A"], dtype=int)
        self._B_pos = np.array([n for n, s in enumerate(self._slots) if s[0] == "B"], dtype=int)
        self._B_src = np.array([k * 3 + i for kind, k, i, j in self._slots if kind == "B"], dtype=int)
        # linear cost on e_psi: q[z_k + 1] = -q_psi w_k e_psi_ref[min(k, N-1)], k = 1..N
        ks = np.arange(1, N + 1)
        self._q_idx = nz * (ks - 1) + 1
        self._q_coef = -self.q_psi * np.where(ks == N, self.qT, 1.0)
        self._q_ref = np.minimum(ks, N - 1)
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

    def _A_values(self, Ad, Bd):
        """CSC data of the constraint matrix for stacked Ad (N, 3, 3), Bd (N, 3)."""
        vals = np.ones(len(self._slots))
        vals[self._A_pos] = -Ad.ravel()[self._A_src]
        vals[self._B_pos] = -Bd.ravel()[self._B_src]
        return vals[self._perm]

    def solve(self, e_y, e_psi, delta_prev, v_seq, kappa_seq):
        """v_seq, kappa_seq: length-N previews. Returns (delta_cmd, status_ok, solve_time_s, pred)."""
        N = self.N
        Ad, Bd, cd, epr = self._discretize_many(np.maximum(np.asarray(v_seq[:N], dtype=float), 0.5), kappa_seq[:N])
        z0 = np.array([e_y, e_psi, delta_prev])
        beq = cd.ravel()  # view of cd, which is not used after this
        beq[:3] = cd[0] + Ad[0] @ z0
        self.l[: self.n_eq] = beq
        self.u[: self.n_eq] = beq
        q = np.zeros(self.nx)
        q[self._q_idx] = self._q_coef * epr[self._q_ref]
        Ax = self._A_values(Ad, Bd)
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

    def _discretize_many(self, v, kappa):
        """All stages at once: Ad (n, 5, 5), Bd (n, 5, 2), cd (n, 5), e_psi_ref (n,)."""
        Ad3, Bd3, cd3, e_psi_ref = self.lat._discretize_many(np.maximum(v, 0.5), kappa)  # z3 = [e_y, e_psi, delta]
        n = len(e_psi_ref)
        Ad, Bd, cd = np.zeros((n, 5, 5)), np.zeros((n, 5, 2)), np.zeros((n, 5))
        Ad[:, :3, :3], Bd[:, :3, 0], cd[:, :3] = Ad3, Bd3, cd3
        Ad[:, 3, 3], Ad[:, 3, 4], Bd[:, 3, 1] = 1.0, self.Ts, self.Ts
        Ad[:, 4, 4], Bd[:, 4, 1] = 1.0, 1.0
        return Ad, Bd, cd, e_psi_ref

    def _discretize(self, v, kappa):
        Ad, Bd, cd, e_psi_ref = self._discretize_many(np.array([v], dtype=float), [kappa])
        return Ad[0], Bd[0], cd[0], e_psi_ref[0]

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
        Ads, Bds, cds, epr = self._discretize_many(v_start, kap_p[:N])
        z0 = np.array([e_y, e_psi, delta_prev, v, a_prev])
        beq = cds.ravel()  # view of cds, which is not used after this
        beq[:5] = cds[0] + Ads[0] @ z0
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


class SlipMPC:
    """Slip-aware coupled MPC: dynamic single-track error model with speed, rear/front slip-angle limits.

    Not privileged: predicts with the NOMINAL friction mu0 and the nominal (linear) cornering stiffness; it
    measures sideslip beta and yaw rate r, as pilot.reference_lap's DynamicMPC does.

    Continuous model (path frame), x = [e_y, e_psi, beta, r, v], held inputs u = [delta, a]:
        e_y'   = v sin(e_psi + beta)          e_psi' = r - v kappa
        beta'  = (Fyf + Fyr) / (m v) - r      r'     = (lf Fyf - lr Fyr) / I        v' = a
        Fyf = mu0 C_Sf Fzf alpha_f,  Fyr = mu0 C_Sr Fzr alpha_r     (ST's linear tires, axle-force form)
        alpha_f = delta - beta - lf r / v,  alpha_r = -beta + lr r / v
        Fzf = m (g lr - a h) / L,  Fzr = m (g lf + a h) / L      (longitudinal load transfer)
    Linearised (finite differences) along the previous prediction, shifted by one step (first call: the
    steady-state cornering point of the plan), exact ZOH per interval. Discrete state z = [x, delta, a],
    input [d_delta, d_a] (the commands applied over interval k are delta_{k+1}, a_{k+1}, as in CoupledMPC).
    Cost: q_y e_y^2 + q_psi (e_psi + beta)^2 (course error) + q_v (v - v_ref)^2 + q_a (a - a_ref)^2
          + r_du d_delta^2 + r_da d_a^2 (terminal x q_terminal on e_y, course, v) + slack penalty.
    Hard: |delta| <= s_max, |d_delta| <= du_max, -a_max <= a <= a_max min(1, v_switch/v), |d_a| <= da_max,
    v >= v_min. Soft (one slack per stage): |alpha_f| <= alpha_f_max, |alpha_r| <= alpha_r_max, and the
    optional friction-ellipse polygon of CoupledMPC.
    """

    NX, NZ, NU = 5, 7, 2
    G = 9.81

    def __init__(self, params, Ts, N, lat_w, w, s_max, du_max, ellipse=None):
        self.p = params  # NOMINAL vehicle parameters (mu0)
        self.Ts, self.N = Ts, N
        self.lw, self.w = lat_w, w
        self.s_max, self.du_max = s_max, du_max
        self.a_max, self.v_switch = params["a_max"], params["v_switch"]
        self.ell = ellipse
        n_f = ellipse["n_facets"] if ellipse else 0
        th = np.arange(n_f) * 2 * np.pi / max(n_f, 1)
        self._fc, self._fs = np.cos(th), np.sin(th)
        nz, nu = self.NZ, self.NU
        self.nx = nz * N + nu * N + N  # states, inputs, slacks
        self.n_eq = nz * N
        self.n_soft = 4 + n_f  # +-alpha_f, +-alpha_r, facets
        self.m = self.n_eq + 3 * N + 2 * N + self.n_soft * N + N
        self.solver, self.guess, self._mask = None, None, None
        # cost (constant): block per stage, course error couples e_psi and beta
        rows, cols, vals = [], [], []
        for k in range(1, N + 1):
            wT = lat_w["q_terminal"] if k == N else 1.0
            z = self._z(k)
            qy, qp = lat_w["q_y"] * wT, lat_w["q_psi"] * wT
            for (i, j, v) in ((0, 0, qy), (1, 1, qp), (2, 2, qp), (1, 2, qp), (2, 1, qp),
                              (4, 4, w["q_v"] * wT), (6, 6, w["q_a"])):
                rows.append(z + i), cols.append(z + j), vals.append(v)
        for k in range(N):
            u = self._u(k)
            rows += [u, u + 1]
            cols += [u, u + 1]
            vals += [lat_w["r_du"], w["r_da"]]
        for k in range(N):
            j = self._e(k)
            rows.append(j), cols.append(j), vals.append(w["slack_l2"])
        self._P = sp.triu(sp.csc_matrix((vals, (rows, cols)), shape=(self.nx, self.nx))).tocsc()

    def _z(self, k):  # z_k, k = 1..N
        return self.NZ * (k - 1)

    def _u(self, k):  # u_k, k = 0..N-1
        return self.NZ * self.N + self.NU * k

    def _e(self, k):  # slack of z_{k+1}, k = 0..N-1
        return self.NZ * self.N + self.NU * self.N + k

    # ------------------------------------------------------------------ model
    def _loads(self, a):
        p = self.p
        L = p["lf"] + p["lr"]
        return p["m"] * (self.G * p["lr"] - a * p["h"]) / L, p["m"] * (self.G * p["lf"] + a * p["h"]) / L

    def f(self, x, u, kappa):
        p = self.p
        e_y, e_psi, beta, r, v = x
        delta, a = u
        v = max(v, 0.5)
        fzf, fzr = self._loads(a)
        af = delta - beta - p["lf"] * r / v
        ar = -beta + p["lr"] * r / v
        fyf, fyr = p["mu"] * p["C_Sf"] * fzf * af, p["mu"] * p["C_Sr"] * fzr * ar
        return np.array([v * np.sin(e_psi + beta), r - v * kappa, (fyf + fyr) / (p["m"] * v) - r,
                         (p["lf"] * fyf - p["lr"] * fyr) / p["I"], a])

    def steady(self, v, a, kappa):
        """Steady cornering on the path at speed v: returns x = [0, -beta, beta, r, v], delta."""
        p = self.p
        v = max(v, 0.5)
        L, r = p["lf"] + p["lr"], v * kappa
        fzf, fzr = self._loads(a)
        fyf, fyr = p["m"] * v * r * p["lr"] / L, p["m"] * v * r * p["lf"] / L
        ar, af = fyr / (p["mu"] * p["C_Sr"] * fzr), fyf / (p["mu"] * p["C_Sf"] * fzf)
        beta = p["lr"] * r / v - ar
        delta = af + beta + p["lf"] * r / v
        return np.array([0.0, -beta, beta, r, v]), delta

    def _disc(self, x, u, kappa, T):
        """Exact ZOH of the Jacobian linearisation at (x, u): x+ = Phi x + Gam u + w."""
        f0 = self.f(x, u, kappa)
        J = np.zeros((self.NX, self.NX + 2))
        for i in range(self.NX + 2):
            h = 1e-6
            xp, up = x.copy(), np.array(u, float)
            if i < self.NX:
                xp[i] += h
            else:
                up[i - self.NX] += h
            J[:, i] = (self.f(xp, up, kappa) - f0) / h
        A, B = J[:, :self.NX], J[:, self.NX:]
        c = f0 - A @ x - B @ np.asarray(u, float)
        M = np.zeros((self.NX + 3, self.NX + 3))
        M[:self.NX, :self.NX], M[:self.NX, self.NX:self.NX + 2], M[:self.NX, -1] = A, B, c
        E = expm(M * T)
        return E[:self.NX, :self.NX], E[:self.NX, self.NX:self.NX + 2], E[:self.NX, -1]

    def predict(self, x, u, kappa, T):
        Phi, Gam, w = self._disc(x, u, kappa, T)
        return Phi @ x + Gam @ np.asarray(u, float) + w

    def _a_upper(self, v):
        return self.a_max * min(1.0, self.v_switch / max(v, 1e-3))

    # ------------------------------------------------------------------ QP
    def _build(self, Ads, Bds, xbar, kap):
        N, nz, p = self.N, self.NZ, self.p
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
            A[r, z + 5], A[r + 1, z + 6], A[r + 2, z + 4] = 1.0, 1.0, 1.0  # delta, a, v
            r += 3
        for k in range(N):
            A[r, self._u(k)], A[r + 1, self._u(k) + 1] = 1.0, 1.0
            r += 2
        for k in range(1, N + 1):  # soft rows <= bound + slack, linear in z_k around xbar (v-dependence of r/v)
            z, j = self._z(k), self._e(k - 1)
            vb, rb = max(xbar[k - 1][4], 0.5), xbar[k - 1][3]
            for sgn in (1.0, -1.0):
                A[r, z + 5], A[r, z + 2], A[r, z + 3] = sgn, -sgn, -sgn * p["lf"] / vb  # alpha_f
                A[r, z + 4] = sgn * p["lf"] * rb / vb**2
                A[r, j] = -1.0
                A[r + 1, z + 2], A[r + 1, z + 3] = -sgn, sgn * p["lr"] / vb  # alpha_r
                A[r + 1, z + 4] = -sgn * p["lr"] * rb / vb**2
                A[r + 1, j] = -1.0
                r += 2
            if self.ell:
                ax, ay = self.ell["ax_max"], self.ell["ay_max"]
                for c, s in zip(self._fc, self._fs):
                    A[r, z + 6] = c / ax
                    A[r, z + 4] = s / ay * 2.0 * vb * kap[k - 1]
                    A[r, j] = -1.0
                    r += 1
        for k in range(N):
            A[r, self._e(k)] = 1.0
            r += 1
        return A

    def solve(self, e_y, e_psi, beta, r, v, delta_prev, a_prev, s0, v_ref_at, a_ref_at, kappa_at, tau=0.0,
              v_min=0.5):
        """One MPC step. The measured state is propagated by ``tau`` (actuator delay) with the model.
        Returns (delta_cmd, a_cmd, ok, solve_time_s, pred) with pred the (N, 7) predicted z."""
        N, Ts, nz, w = self.N, self.Ts, self.NZ, self.w
        x0 = np.array([e_y, e_psi, beta, r, v], float)
        if tau > 0:
            x0 = self.predict(x0, (delta_prev, a_prev), float(kappa_at(s0)), tau)
            s0 = s0 + tau * v
        # linearisation trajectory: guesses for z_1..z_N
        if self.guess is None:
            g = np.zeros((N, nz))
            s, vv = s0, x0[4]
            for k in range(N):
                s += Ts * max(vv, 0.5)
                vv = max(float(v_ref_at(s)), 0.5)
                xs, ds = self.steady(vv, float(a_ref_at(s)), float(kappa_at(s)))
                g[k, :5], g[k, 5], g[k, 6] = xs, ds, float(a_ref_at(s))
        else:
            g = np.r_[self.guess[1:], self.guess[-1:]]
        x_start = np.r_[[x0], g[:-1, :5]]  # linearisation state at the start of each interval
        u_lin = g[:, 5:7]  # commands applied over each interval
        v_start = np.maximum(x_start[:, 4], 0.5)
        pos = s0 + Ts * np.r_[0.0, np.cumsum(v_start)]  # pos[k] = location of z_k
        kap_p = kappa_at(pos)
        v_ref, a_ref = v_ref_at(pos[1:]), a_ref_at(pos[1:])
        Ads, Bds, cds = [], [], []
        for k in range(N):
            Phi, Gam, wv = self._disc(x_start[k], u_lin[k], kap_p[k], Ts)
            Ad, Bd, cd = np.zeros((nz, nz)), np.zeros((nz, 2)), np.zeros(nz)
            Ad[:5, :5], Ad[:5, 5:7], Ad[5, 5], Ad[6, 6] = Phi, Gam, 1.0, 1.0
            Bd[:5, :], Bd[5, 0], Bd[6, 1] = Gam, 1.0, 1.0
            cd[:5] = wv
            Ads.append(Ad), Bds.append(Bd), cds.append(cd)
        z0 = np.r_[x0, delta_prev, a_prev]
        beq = np.concatenate([cds[k] + (Ads[0] @ z0 if k == 0 else 0.0) for k in range(N)])
        lo = [beq, np.ravel([[-self.s_max, -self.a_max, v_min] for _ in range(N)]),
              np.tile([-self.du_max, -w["da_max"]], N)]
        hi = [beq, np.ravel([[self.s_max, self._a_upper(g[k, 4]), 1e9] for k in range(N)]),
              np.tile([self.du_max, w["da_max"]], N)]
        soft_hi = []
        for k in range(N):
            vb, rb = max(g[k, 4], 0.5), g[k, 3]
            # r/v ~ r/vb - rb v/vb^2 + rb/vb: constant parts alpha_f -= lf rb/vb, alpha_r += lr rb/vb, so
            # sgn alpha <= alpha_max becomes sgn (row z) <= alpha_max - sgn * const
            cf, cr = self.p["lf"] * rb / vb, self.p["lr"] * rb / vb
            for sgn in (1.0, -1.0):
                soft_hi += [w["alpha_f_max"] + sgn * cf, w["alpha_r_max"] - sgn * cr]
            if self.ell:
                ay = self.ell["ay_max"]
                soft_hi += [1.0 + s / ay * kap_p[k + 1] * vb**2 for s in self._fs]
        lo += [np.full(len(soft_hi), -1e9), np.zeros(N)]
        hi += [np.array(soft_hi), np.full(N, 1e9)]
        l, u = np.concatenate(lo), np.concatenate(hi)
        q = np.zeros(self.nx)
        for k in range(1, N + 1):
            wT = self.lw["q_terminal"] if k == N else 1.0
            z = self._z(k)
            q[z + 4] = -w["q_v"] * wT * v_ref[k - 1]
            q[z + 6] = -w["q_a"] * a_ref[k - 1]
        q[self._e(0):self._e(0) + N] = w["slack_l1"]
        A = self._build(Ads, Bds, g, kap_p[1:])
        t0 = time.perf_counter()
        if self.solver is None:
            ones = [np.ones((nz, nz))] * N, [np.ones((nz, 2))] * N
            gm = np.ones((N, nz))
            self._mask = self._build(ones[0], ones[1], gm, np.ones(N)) != 0
            Acsc = sp.csc_matrix(self._mask.astype(float))
            Acsc.data = A.T[self._mask.T]
            self.solver = osqp.OSQP()
            self.solver.setup(self._P, q, Acsc, l, u, verbose=False, eps_abs=1e-5, eps_rel=1e-5, polishing=True,
                              warm_starting=True, max_iter=4000)
        else:
            self.solver.update(q=q, l=l, u=u, Ax=A.T[self._mask.T])
        res = self.solver.solve()
        dt = time.perf_counter() - t0
        ok = res.info.status_val in (1, 2) and res.x is not None and np.all(np.isfinite(res.x))
        if not ok:
            self.guess = None
            return delta_prev, a_prev, False, dt, None
        x = res.x
        pred = x[: nz * N].reshape(N, nz)
        self.guess = pred.copy()
        u0 = x[self._u(0):self._u(0) + 2]
        return float(delta_prev + u0[0]), float(a_prev + u0[1]), True, dt, pred
