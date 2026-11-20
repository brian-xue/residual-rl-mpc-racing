//! Steering-only kinematic MPC of ``pilot.mpc.KinematicMPC``: closed-form ZOH discretisation, QP assembly
//! (same CSC structure, built by Python) and OSQP through its C API with the calls and settings the Python
//! ``osqp`` wrapper uses (``osqp_set_default_settings`` + the overrides, ``osqp_update_data_vec(q, l, u)``
//! then ``osqp_update_data_mat(A_x)``).

use std::ptr;
use std::time::Instant;

use osqp_sys as ffi;

use crate::numerics::sin;

/// Kinematic error model of ``KinematicMPC._f``: ``(e_y', e_psi')`` at ``(e_y, e_psi, steering d)``.
///
/// Closed-form ZOH requirement (checked by [`zoh_stage`] on every stage): e_y' and e_psi' must not depend on
/// e_y, and e_psi' must not depend on e_psi.
pub fn kinematic_error_model(
    lr: f64,
    l: f64,
    v: f64,
    kappa: f64,
) -> impl Fn(f64, f64, f64) -> (f64, f64) {
    move |_ey: f64, epsi: f64, d: f64| {
        let b = (lr / l * d.tan()).atan();
        (v * sin(epsi + b), v * b.cos() * d.tan() / l - v * kappa)
    }
}

/// One stage of the closed-form ZOH: the entries of ``I + M Ts + (M Ts)^2 / 2`` that are not 0 or 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZohStage {
    /// ``Ad[0, 1]``
    pub a_t: f64,
    /// ``Ad[0, 2] = Bd[0]``
    pub g0: f64,
    /// ``Ad[1, 2] = Bd[1]``
    pub b1_t: f64,
    /// ``cd[0]``
    pub w0: f64,
    /// ``cd[1]``
    pub c1_t: f64,
}

#[inline]
fn same(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan()) // np.array_equal(..., equal_nan=True)
}

/// Closed-form ZOH of the model ``f`` linearised at ``(0, e1, d_r)`` (forward differences, step ``h``) over ``t``,
/// as ``KinematicMPC._discretize_many``.
///
/// With ``M = [[Ac, Bc, cc], [0, 0, 0, 0], [0, 0, 0, 0]]``, ``expm(M t) = I + M t + (M t)^2 / 2`` holds exactly
/// only when ``M^3 = 0``, i.e. when ``Ac`` is strictly upper triangular: d e_y'/d e_y = d e_psi'/d e_y =
/// d e_psi'/d e_psi = 0. Those three derivatives are evaluated here and the function panics if any is not
/// exactly zero (a changed model needs a general matrix exponential instead).
#[inline]
pub fn zoh_stage(
    f: impl Fn(f64, f64, f64) -> (f64, f64),
    e1: f64,
    d_r: f64,
    h: f64,
    t: f64,
) -> ZohStage {
    let f0 = f(0.0, e1, d_r);
    let fe = f(h, e1, d_r);
    let fp = f(0.0, e1 + h, d_r);
    let fd = f(0.0, e1, d_r + h);
    assert!(
        same(fe.0, f0.0) && same(fe.1, f0.1) && same(fp.1, f0.1),
        "kinematic error model depends on e_y, or e_psi' on e_psi: M is not nilpotent, the closed-form ZOH is invalid"
    );
    let a = (fp.0 - f0.0) / h;
    let (b0, b1) = ((fd.0 - f0.0) / h, (fd.1 - f0.1) / h);
    let (c0, c1) = (f0.0 - a * e1 - b0 * d_r, f0.1 - b1 * d_r);
    let (a_t, b0_t, b1_t, c0_t, c1_t) = (a * t, b0 * t, b1 * t, c0 * t, c1 * t);
    ZohStage {
        a_t,
        g0: b0_t + a_t * b1_t / 2.0,
        b1_t,
        w0: c0_t + a_t * c1_t / 2.0,
        c1_t,
    }
}

/// ``np.maximum(a, b)`` for float64: ``a`` if ``a >= b`` or ``a`` is NaN, else ``b``.
#[inline]
fn np_maximum(a: f64, b: f64) -> f64 {
    if a >= b || a.is_nan() { a } else { b }
}

#[derive(Clone, Debug)]
pub struct Csc {
    pub m: usize,
    pub n: usize,
    pub indptr: Vec<ffi::osqp_int>,
    pub indices: Vec<ffi::osqp_int>,
    pub data: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct OsqpSettings {
    pub eps_abs: f64,
    pub eps_rel: f64,
    pub max_iter: i64,
    pub polishing: bool,
    pub warm_starting: bool,
}

struct Solver(*mut ffi::OSQPSolver);

// The solver is only touched through &mut KinematicQp.
unsafe impl Send for Solver {}
unsafe impl Sync for Solver {}

impl Drop for Solver {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                ffi::osqp_cleanup(self.0);
            }
        }
    }
}

pub struct KinematicQp {
    pub lr: f64,
    pub l: f64,
    pub ts: f64,
    pub n: usize,
    pub nx: usize,
    pub n_eq: usize,
    pub n_slots: usize,
    pub a_pos: Vec<usize>,
    pub a_src: Vec<usize>,
    pub b_pos: Vec<usize>,
    pub b_src: Vec<usize>,
    pub perm: Vec<usize>,
    pub q_idx: Vec<usize>,
    pub q_coef: Vec<f64>,
    pub q_ref: Vec<usize>,
    pub p: Csc,
    /// constraint matrix structure (values are replaced by ``assemble``'s A values)
    pub a: Csc,
    /// bounds; rows ``0..n_eq`` are overwritten with b_eq on every solve
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    pub settings: OsqpSettings,
    solver: Option<Solver>,
}

pub struct Assembled {
    pub q: Vec<f64>,
    pub beq: Vec<f64>,
    pub ax: Vec<f64>,
}

impl KinematicQp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lr: f64,
        l: f64,
        ts: f64,
        n: usize,
        nx: usize,
        n_eq: usize,
        n_slots: usize,
        a_pos: Vec<usize>,
        a_src: Vec<usize>,
        b_pos: Vec<usize>,
        b_src: Vec<usize>,
        perm: Vec<usize>,
        q_idx: Vec<usize>,
        q_coef: Vec<f64>,
        q_ref: Vec<usize>,
        p: Csc,
        a: Csc,
        lo: Vec<f64>,
        hi: Vec<f64>,
        settings: OsqpSettings,
    ) -> Self {
        Self {
            lr,
            l,
            ts,
            n,
            nx,
            n_eq,
            n_slots,
            a_pos,
            a_src,
            b_pos,
            b_src,
            perm,
            q_idx,
            q_coef,
            q_ref,
            p,
            a,
            lo,
            hi,
            settings,
            solver: None,
        }
    }

    #[inline]
    fn beta(&self, d: f64) -> f64 {
        (self.lr / self.l * d.tan()).atan()
    }

    /// ``KinematicMPC.assemble``: (q, b_eq, A values in CSC order).
    pub fn assemble(
        &self,
        e_y: f64,
        e_psi: f64,
        delta_prev: f64,
        v_seq: &[f64],
        kappa_seq: &[f64],
    ) -> Assembled {
        let (n, l, h, t) = (self.n, self.l, 1e-6, self.ts);
        let mut ad = vec![0.0; n * 9];
        let mut bd = vec![0.0; n * 3];
        let mut cd = vec![0.0; n * 3];
        let mut epr = vec![0.0; n];
        for k in 0..n {
            let v = np_maximum(v_seq[k], 0.5);
            let kappa = kappa_seq[k];
            let f = kinematic_error_model(self.lr, l, v, kappa);
            let d_r = (l * kappa).atan();
            let e1 = -self.beta(d_r);
            let ZohStage {
                a_t,
                g0,
                b1_t,
                w0,
                c1_t,
            } = zoh_stage(&f, e1, d_r, h, t);
            ad[k * 9..k * 9 + 9].copy_from_slice(&[1.0, a_t, g0, 0.0, 1.0, b1_t, 0.0, 0.0, 1.0]);
            bd[k * 3..k * 3 + 3].copy_from_slice(&[g0, b1_t, 1.0]);
            cd[k * 3..k * 3 + 3].copy_from_slice(&[w0, c1_t, 0.0]);
            epr[k] = e1;
        }
        let mut beq = cd.clone();
        for i in 0..3 {
            beq[i] = cd[i] + (ad[i * 3] * e_y + ad[i * 3 + 1] * e_psi + ad[i * 3 + 2] * delta_prev);
        }
        let mut q = vec![0.0; self.nx];
        for j in 0..self.q_idx.len() {
            q[self.q_idx[j]] = self.q_coef[j] * epr[self.q_ref[j]];
        }
        let mut vals = vec![1.0; self.n_slots];
        for j in 0..self.a_pos.len() {
            vals[self.a_pos[j]] = -ad[self.a_src[j]];
        }
        for j in 0..self.b_pos.len() {
            vals[self.b_pos[j]] = -bd[self.b_src[j]];
        }
        let ax = self.perm.iter().map(|&p| vals[p]).collect();
        Assembled { q, beq, ax }
    }

    /// ``KinematicMPC.solve``: (delta_cmd, ok, solve_time_s). On failure delta_cmd = delta_prev.
    pub fn solve(
        &mut self,
        e_y: f64,
        e_psi: f64,
        delta_prev: f64,
        v_seq: &[f64],
        kappa_seq: &[f64],
    ) -> (f64, bool, f64) {
        let asm = self.assemble(e_y, e_psi, delta_prev, v_seq, kappa_seq);
        self.lo[..self.n_eq].copy_from_slice(&asm.beq);
        self.hi[..self.n_eq].copy_from_slice(&asm.beq);
        let t0 = Instant::now();
        let solver = unsafe {
            match &self.solver {
                Some(s) => {
                    // Python wrapper: l / u clipped to +-OSQP_INFTY (no-op here), vectors first, then A values
                    ffi::osqp_update_data_vec(
                        s.0,
                        asm.q.as_ptr(),
                        self.lo.as_ptr(),
                        self.hi.as_ptr(),
                    );
                    ffi::osqp_update_data_mat(
                        s.0,
                        ptr::null(),
                        ptr::null(),
                        0,
                        asm.ax.as_ptr(),
                        ptr::null(),
                        asm.ax.len() as ffi::osqp_int,
                    );
                    s.0
                }
                None => {
                    let s = self.setup(&asm);
                    self.solver = Some(Solver(s));
                    s
                }
            }
        };
        let (x0, status) = unsafe {
            ffi::osqp_solve(solver);
            let info = &*(*solver).info;
            let x = (*(*solver).solution).x;
            let xs = std::slice::from_raw_parts(x, self.nx);
            let finite = xs.iter().all(|v| v.is_finite());
            (
                if finite { Some(xs[3 * self.n]) } else { None },
                info.status_val,
            )
        };
        let dt = t0.elapsed().as_secs_f64();
        let ok = status == 1 || status == 2; // solved / solved inaccurate
        match (ok, x0) {
            (true, Some(u0)) => (delta_prev + u0, true, dt),
            _ => (delta_prev, false, dt),
        }
    }

    unsafe fn setup(&mut self, asm: &Assembled) -> *mut ffi::OSQPSolver {
        unsafe {
            let mut settings: ffi::OSQPSettings = std::mem::zeroed();
            ffi::osqp_set_default_settings(&mut settings);
            settings.verbose = 0;
            settings.eps_abs = self.settings.eps_abs;
            settings.eps_rel = self.settings.eps_rel;
            settings.polishing = self.settings.polishing as ffi::osqp_int;
            settings.warm_starting = self.settings.warm_starting as ffi::osqp_int;
            settings.max_iter = self.settings.max_iter as ffi::osqp_int;
            let mut a_data = asm.ax.clone();
            let p = &mut self.p;
            let pm = ffi::OSQPCscMatrix_new(
                p.m as ffi::osqp_int,
                p.n as ffi::osqp_int,
                p.data.len() as ffi::osqp_int,
                p.data.as_mut_ptr(),
                p.indices.as_mut_ptr(),
                p.indptr.as_mut_ptr(),
            );
            let a = &mut self.a;
            let am = ffi::OSQPCscMatrix_new(
                a.m as ffi::osqp_int,
                a.n as ffi::osqp_int,
                a_data.len() as ffi::osqp_int,
                a_data.as_mut_ptr(),
                a.indices.as_mut_ptr(),
                a.indptr.as_mut_ptr(),
            );
            let mut solver: *mut ffi::OSQPSolver = ptr::null_mut();
            let err = ffi::osqp_setup(
                &mut solver,
                pm,
                asm.q.as_ptr(),
                am,
                self.lo.as_ptr(),
                self.hi.as_ptr(),
                self.a.m as ffi::osqp_int,
                self.a.n as ffi::osqp_int,
                &settings,
            );
            ffi::OSQPCscMatrix_free(pm);
            ffi::OSQPCscMatrix_free(am);
            assert!(
                err == 0 && !solver.is_null(),
                "osqp_setup failed with code {err}"
            );
            solver
        }
    }
}

/// OSQP C library version string.
pub fn osqp_version() -> String {
    unsafe {
        std::ffi::CStr::from_ptr(ffi::osqp_version())
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type M4 = [[f64; 4]; 4];

    fn mul(a: &M4, b: &M4) -> M4 {
        let mut c = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        c
    }

    /// exp(M) by its Taylor series (40 terms; ||M|| is small here).
    fn expm_taylor(m: &M4) -> M4 {
        let mut e = [[0.0; 4]; 4];
        let mut term = [[0.0; 4]; 4];
        for i in 0..4 {
            e[i][i] = 1.0;
            term[i][i] = 1.0;
        }
        for k in 1..40 {
            term = mul(&term, m);
            for row in term.iter_mut() {
                for x in row.iter_mut() {
                    *x /= k as f64;
                }
            }
            for i in 0..4 {
                for j in 0..4 {
                    e[i][j] += term[i][j];
                }
            }
        }
        e
    }

    /// Full augmented matrix M t from forward differences of ``f``, with every Jacobian column (including the
    /// ones the closed form assumes zero).
    fn full_m(f: &impl Fn(f64, f64, f64) -> (f64, f64), e1: f64, d_r: f64, h: f64, t: f64) -> M4 {
        let f0 = f(0.0, e1, d_r);
        let col = |g: (f64, f64)| [(g.0 - f0.0) / h, (g.1 - f0.1) / h];
        let (ce, cp, cd) = (
            col(f(h, e1, d_r)),
            col(f(0.0, e1 + h, d_r)),
            col(f(0.0, e1, d_r + h)),
        );
        let mut m = [[0.0; 4]; 4];
        let f0v = [f0.0, f0.1];
        for i in 0..2 {
            m[i] = [
                ce[i] * t,
                cp[i] * t,
                cd[i] * t,
                (f0v[i] - cp[i] * e1 - cd[i] * d_r) * t,
            ];
        }
        m
    }

    fn cases() -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        for &v in &[0.5, 2.0, 5.0, 8.0, 15.0] {
            for &kappa in &[-1.5, -0.3, -0.01, 0.0, 0.004, 0.2, 1.2] {
                out.push((v, kappa));
            }
        }
        out
    }

    #[test]
    fn model_gives_nilpotent_m_and_closed_form_matches_exponential() {
        let (lr, l, h, t) = (0.17145, 0.3302, 1e-6, 0.05);
        for (v, kappa) in cases() {
            let f = kinematic_error_model(lr, l, v, kappa);
            let d_r = (l * kappa).atan();
            let e1 = -(lr / l * d_r.tan()).atan();
            let m = full_m(&f, e1, d_r, h, t);
            let m3 = mul(&mul(&m, &m), &m);
            assert!(
                m3.iter().flatten().all(|&x| x == 0.0),
                "M^3 != 0 for v={v}, kappa={kappa}"
            );
            let e = expm_taylor(&m);
            let z = zoh_stage(&f, e1, d_r, h, t);
            let got = [z.a_t, z.g0, z.b1_t, z.w0, z.c1_t];
            let want = [e[0][1], e[0][2], e[1][2], e[0][3], e[1][3]];
            for (g, w) in got.iter().zip(want) {
                assert!(
                    (g - w).abs() <= 1e-14 * w.abs().max(1.0),
                    "v={v} kappa={kappa}: {g} vs {w}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "not nilpotent")]
    fn guard_rejects_e_psi_dependence_of_e_psi_rate() {
        let base = kinematic_error_model(0.17145, 0.3302, 5.0, 0.1);
        let f = move |ey: f64, epsi: f64, d: f64| {
            let (a, b) = base(ey, epsi, d);
            (a, b - 0.5 * epsi) // e.g. a heading-error damping term
        };
        zoh_stage(&f, -0.05, 0.03, 1e-6, 0.05);
    }

    #[test]
    #[should_panic(expected = "not nilpotent")]
    fn guard_rejects_e_y_dependence() {
        let base = kinematic_error_model(0.17145, 0.3302, 5.0, 0.1);
        let f = move |ey: f64, epsi: f64, d: f64| {
            let (a, b) = base(ey, epsi, d);
            (a - 0.2 * ey, b)
        };
        zoh_stage(&f, -0.05, 0.03, 1e-6, 0.05);
    }
}
