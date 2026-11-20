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
