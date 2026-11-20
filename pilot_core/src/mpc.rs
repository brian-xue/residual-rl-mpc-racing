//! Steering-only kinematic MPC of ``pilot.mpc.KinematicMPC``: closed-form ZOH discretisation, QP assembly
//! (same CSC structure, built by Python) and OSQP through its C API with the calls and settings the Python
//! ``osqp`` wrapper uses (``osqp_set_default_settings`` + the overrides, ``osqp_update_data_vec(q, l, u)``
//! then ``osqp_update_data_mat(A_x)``).

use std::ptr;
use std::time::Instant;

use osqp_sys as ffi;

use crate::numerics::sin;

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
            let f = |epsi: f64, d: f64| -> (f64, f64) {
                let b = self.beta(d);
                (v * sin(epsi + b), v * b.cos() * d.tan() / l - v * kappa)
            };
            let d_r = (l * kappa).atan();
            let e1 = -self.beta(d_r);
            let f0 = f(e1, d_r);
            let fp = f(e1 + h, d_r);
            let fd = f(e1, d_r + h);
            let a = (fp.0 - f0.0) / h;
            let (b0, b1) = ((fd.0 - f0.0) / h, (fd.1 - f0.1) / h);
            let (c0, c1) = (f0.0 - a * e1 - b0 * d_r, f0.1 - b1 * d_r);
            let (a_t, b0_t, b1_t, c0_t, c1_t) = (a * t, b0 * t, b1 * t, c0 * t, c1 * t);
            let g0 = b0_t + a_t * b1_t / 2.0;
            let w0 = c0_t + a_t * c1_t / 2.0;
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
