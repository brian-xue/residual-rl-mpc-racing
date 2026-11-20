//! The per-physics-step part of ``pilot.closedloop.LapSim`` (``_physics``, ``_pre``, ``advance_interval``):
//! servo, longitudinal controller, plant step, acceleration metrics, track-limit / wall / spin checks,
//! Frenet projection, progress and lap metrics. Python keeps the 20 Hz part (MPC, residual, logging).
//!
//! Field names follow the ``LapSim`` attributes they mirror.

use std::sync::Arc;

use crate::dynamics::{Plant, State};
use crate::geometry::Geometry;
use crate::mpc::KinematicQp;
use crate::numerics::{np_clip, pow2, py_max, py_min, py_mod, sin};
use crate::track::{Periodic, RefLine};

const PI: f64 = std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct LapConfig {
    pub dt: f64,
    pub n_sub: u64,
    pub t_max: f64,
    pub kv: f64,
    pub a_max: f64,
    pub ell_ax: f64,
    pub ell_ay: f64,
    /// servo rate limit and the saturation count threshold (``LapSim.sv_max``)
    pub sv_max: f64,
    pub spin_beta_max: Option<f64>,
    /// ``LapSim._own_env``: only an own env truncates at ``max_episode_steps``
    pub own_env: bool,
    pub max_episode_steps: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Timeout,
    WallCollision,
    Spin,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Completed => "completed",
            Outcome::Timeout => "timeout",
            Outcome::WallCollision => "wall_collision",
            Outcome::Spin => "spin",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LapMetrics {
    pub max_ey: f64,
    pub sum_ey2: f64,
    pub n_ey: u64,
    pub sum_dv: f64,
    pub ell_max: f64,
    pub n_ell_over: u64,
    pub max_ay: f64,
    pub min_ax: f64,
    pub n_off: u64,
    pub first_off_s: Option<f64>,
    pub first_off_progress: Option<f64>,
    pub n_sat_rate: u64,
}

#[derive(Clone, Debug)]
pub struct Lap {
    pub cfg: LapConfig,
    pub plant: Plant,
    pub geo: Arc<Geometry>,
    pub refl: Arc<RefLine>,
    pub vplan: Periodic,
    /// ``SteeringServo.pending`` (steering-speed mode, 2-step delay)
    pub servo_pending: [f64; 2],
    /// ``LapSim.st``: the state read by the last ``_pre`` (not refreshed when a step ends the lap)
    pub st: State,
    pub t: f64,
    pub step: u64,
    /// ``env.current_step``
    pub env_step: i64,
    pub s: f64,
    pub e_y: f64,
    pub e_psi: f64,
    pub v: f64,
    pub vp: f64,
    pub i_g: usize,
    pub s_prev: f64,
    pub progress: f64,
    pub off_in_interval: u64,
    pub tires_in: [bool; 4],
    pub metrics: LapMetrics,
    pub outcome: Option<Outcome>,
    pub done: bool,
    pub delta_cmd: f64,
    pub a_mpc: Option<f64>,
    /// Longitudinal residual [m/s^2] added to the speed command before the clip (dual-channel residual RL);
    /// only applied when nonzero, so laps without it are bit-identical to before.
    pub a_res: f64,
}

impl Lap {
    /// ``SteeringServo.command`` in steering-speed mode.
    fn servo(&mut self, target: f64, delta_now: f64) -> f64 {
        let dt = self.cfg.dt;
        let pending_sum = (0.0 + self.servo_pending[0]) + self.servo_pending[1]; // sum(list) starts at 0
        let delta_pred = delta_now + dt * pending_sum;
        let sv = np_clip(
            (target - delta_pred) / dt,
            -self.cfg.sv_max,
            self.cfg.sv_max,
        );
        self.servo_pending = [self.servo_pending[1], sv];
        sv
    }

    /// ``LapSim._physics``. Returns true when the lap ended.
    fn physics(&mut self) -> bool {
        let c = self.cfg.clone();
        let st = self.st;
        let u_steer = self.servo(self.delta_cmd, st[2]);
        self.metrics.n_sat_rate += (u_steer.abs() >= c.sv_max - 1e-9) as u64;
        let (s, v) = (self.s, self.v);
        let mut a_req = match self.a_mpc {
            Some(am) => am,
            None => {
                let dvds = (self.vplan.at(s + 0.2) - self.vplan.at(s - 0.2)) / 0.4;
                v * dvds + c.kv * (self.vp - v)
            }
        };
        if self.a_res != 0.0 {
            a_req += self.a_res;
        }
        let a_cmd = np_clip(a_req, -c.a_max, c.a_max);
        self.plant.step(u_steer, a_cmd);
        self.env_step += 1;
        let trunc = self.env_step > c.max_episode_steps;
        self.t += c.dt;
        self.step += 1;
        let x = self.plant.x;
        let ax = (x[3] - v) / c.dt;
        let ay = x[3] * (x[5] + (x[6] - st[6]) / c.dt);
        let e = (ax / c.ell_ax).hypot(ay / c.ell_ay);
        let m = &mut self.metrics;
        m.ell_max = py_max(m.ell_max, e);
        m.n_ell_over += (e > 1.02) as u64;
        m.max_ay = py_max(m.max_ay, ay.abs());
        m.min_ax = py_min(m.min_ax, ax);
        let (off, wall, tires_in) = self.geo.check(x[0], x[1], x[4], x[2]);
        self.tires_in = tires_in;
        if off {
            self.metrics.n_off += 1;
            self.off_in_interval += 1;
            if self.metrics.first_off_s.is_none() {
                self.metrics.first_off_s =
                    Some(self.refl.project(x[0], x[1], Some(self.i_g as i64)).0);
                self.metrics.first_off_progress = Some(self.progress);
            }
        }
        let ended = if wall {
            Some(Outcome::WallCollision)
        } else if c.spin_beta_max.is_some_and(|b| x[6].abs() > b) {
            Some(Outcome::Spin)
        } else {
            None
        };
        if let Some(o) = ended {
            self.outcome = Some(o);
            let (s, e_y, i) = self.refl.project(x[0], x[1], Some(self.i_g as i64));
            (self.s, self.e_y, self.i_g) = (s, e_y, i);
            return true;
        }
        if trunc && c.own_env {
            self.outcome = Some(Outcome::Timeout);
            return true;
        }
        false
    }

    /// ``LapSim._pre``. Returns true when the lap ended.
    pub fn pre(&mut self) -> bool {
        let x = self.plant.x;
        self.st = x;
        let (px, py, v, yaw) = (x[0], x[1], x[3], x[4]);
        let len = self.refl.length;
        let (s, e_y, i) = self.refl.project(px, py, Some(self.i_g as i64));
        self.i_g = i;
        let ds = py_mod(s - self.s_prev + len / 2.0, len) - len / 2.0;
        self.progress += ds;
        self.s_prev = s;
        (self.s, self.e_y, self.v) = (s, e_y, v);
        self.e_psi = py_mod(yaw - self.refl.psi[i] + PI, 2.0 * PI) - PI;
        let m = &mut self.metrics;
        m.max_ey = py_max(m.max_ey, e_y.abs());
        m.sum_ey2 += pow2(e_y);
        m.n_ey += 1;
        self.vp = self.vplan.at(s);
        self.metrics.sum_dv += (v - self.vp).abs();
        if self.progress >= len {
            self.outcome = Some(Outcome::Completed);
            return true;
        }
        if self.t >= self.cfg.t_max {
            self.outcome = Some(Outcome::Timeout);
            return true;
        }
        false
    }

    /// Kinematic-MPC branch of ``LapSim.plan``: delay compensation (kinematic error model over ``tau`` with the
    /// current steering), speed / curvature previews along the plan, the MPC solve and the ``K v^2 kappa``
    /// feedforward. Returns (d_mpc, ok, solve_time_s, s_p, d_ff).
    pub fn plan_kinematic(
        &self,
        qp: &mut KinematicQp,
        delta_mpc_prev: f64,
        tau: f64,
        k_ff: f64,
    ) -> (f64, bool, f64, f64, f64) {
        let (lr, l, ts, n) = (qp.lr, qp.l, qp.ts, qp.n);
        let (delta, v, e_y, e_psi, s) = (self.st[2], self.v, self.e_y, self.e_psi, self.s);
        let refl = &self.refl;
        let b = (lr / l * delta.tan()).atan();
        let e_y_p = e_y + tau * v * sin(e_psi + b);
        let e_psi_p = e_psi + tau * (v * b.cos() * delta.tan() / l - v * refl.kappa[self.i_g]);
        let s_p = s + tau * v;
        // s_k = s_p + cumsum([0, c, ..., c]),  c = Ts * max(v_plan(s_p), 0.5)
        let c = ts * py_max(self.vplan.at(s_p), 0.5);
        let mut s1 = vec![0.0; n];
        let mut acc = 0.0;
        for (j, sj) in s1.iter_mut().enumerate() {
            acc = if j == 0 { 0.0 } else { acc + c };
            *sj = s_p + acc;
        }
        // s_k = s_p + [0, cumsum(Ts * v_plan(s_k[:-1]))]
        let mut s_k = vec![0.0; n];
        s_k[0] = s_p + 0.0;
        let mut acc = 0.0;
        for j in 0..n - 1 {
            let w = ts * self.vplan.at(s1[j]);
            acc = if j == 0 { w } else { acc + w };
            s_k[j + 1] = s_p + acc;
        }
        let mut v_seq: Vec<f64> = s_k.iter().map(|&x| self.vplan.at(x)).collect();
        v_seq[0] = v;
        let kap: Vec<f64> = s_k.iter().map(|&x| refl.kappa_at.at(x)).collect();
        let (d_mpc, ok, dt) = qp.solve(e_y_p, e_psi_p, delta_mpc_prev, &v_seq, &kap);
        let d_ff = k_ff * pow2(v) * refl.kappa_at.at(s_p);
        (d_mpc, ok, dt, s_p, d_ff)
    }

    /// ``LapSim.apply``'s interval reset + ``LapSim.advance_interval``. Returns ``done``.
    pub fn advance(&mut self, delta_cmd: f64, a_mpc: Option<f64>, a_res: f64) -> bool {
        self.delta_cmd = delta_cmd;
        self.a_mpc = a_mpc;
        self.a_res = a_res;
        self.off_in_interval = 0;
        while !self.done {
            if self.physics() || self.pre() {
                self.done = true;
            } else if self.step.is_multiple_of(self.cfg.n_sub) {
                break;
            }
        }
        self.done
    }
}
