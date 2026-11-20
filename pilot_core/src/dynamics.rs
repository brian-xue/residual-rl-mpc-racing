//! ST plant: Gym-Khana 1.2.0 ``vehicle_dynamics_st`` (linear tires), the saturating-tire axle-force form of
//! ``pilot.tires``, RK4, and one ``RaceCar.update_pose`` step (2-step steering FIFO, yaw wrap).
//!
//! Every expression keeps the Python evaluation order (left to right, same parenthesisation) so that results
//! are bit-identical to the Python plant.

use crate::numerics::{pow2, py_max, py_min, py_mod, sin};
use crate::params::{Params, TireLaw};

pub const G: f64 = 9.81;
const TIRE_EPS: f64 = 1e-9;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI; // == 2 * np.pi

pub type State = [f64; 7];

/// ``gymkhana.envs.dynamic_models.utils.steering_constraint``
#[inline]
pub fn steering_constraint(
    angle: f64,
    sv: f64,
    s_min: f64,
    s_max: f64,
    sv_min: f64,
    sv_max: f64,
) -> f64 {
    if (angle <= s_min && sv <= 0.0) || (angle >= s_max && sv >= 0.0) {
        0.0
    } else if sv <= sv_min {
        sv_min
    } else if sv >= sv_max {
        sv_max
    } else {
        sv
    }
}

/// ``gymkhana.envs.dynamic_models.utils.accl_constraints``
#[inline]
pub fn accl_constraints(
    vel: f64,
    a: f64,
    v_switch: f64,
    a_max: f64,
    v_min: f64,
    v_max: f64,
) -> f64 {
    let uac = if vel > v_switch {
        a_max * (v_switch / vel)
    } else {
        a_max
    };
    if (vel <= v_min && a <= 0.0) || (vel >= v_max && a >= 0.0) {
        0.0
    } else if a <= -a_max {
        -a_max
    } else if a >= uac {
        uac
    } else {
        a
    }
}

/// ``vehicle_dynamics_st`` (Gym-Khana 1.2.0, single_track.py).
pub fn st_linear(x: &State, u: [f64; 2], p: &Params) -> State {
    let (delta, v, psi, psi_dot, beta) = (x[2], x[3], x[4], x[5], x[6]);
    let sv = steering_constraint(delta, u[0], p.s_min, p.s_max, p.sv_min, p.sv_max);
    let acc = accl_constraints(v, u[1], p.v_switch, p.a_max, p.v_min, p.v_max);
    if v < 0.5 {
        let lwb = p.lf + p.lr;
        let beta_hat = (delta.tan() * p.lr / lwb).atan();
        let beta_dot = (1.0 / (1.0 + pow2(delta.tan() * (p.lr / lwb))))
            * (p.lr / (lwb * pow2(delta.cos())))
            * sv;
        [
            v * (psi + beta_hat).cos(),
            v * sin(psi + beta_hat),
            sv,
            acc,
            v * beta_hat.cos() * delta.tan() / lwb,
            (1.0 / lwb)
                * (acc * beta.cos() * delta.tan() - v * sin(beta) * delta.tan() * beta_dot
                    + ((v * beta.cos() * sv) / pow2(delta.cos()))),
            beta_dot,
        ]
    } else {
        let (lf, lr, c_sf, c_sr, mu, m, i_z) = (p.lf, p.lr, p.c_sf, p.c_sr, p.mu, p.m, p.i_z);
        let glr = G * lr - acc * p.h;
        let glf = G * lf + acc * p.h;
        let psi_ddot = ((mu * m) / (i_z * (lf + lr)))
            * (lf * c_sf * (glr) * delta + (lr * c_sr * (glf) - lf * c_sf * (glr)) * beta
                - (lf * lf * c_sf * (glr) + lr * lr * c_sr * (glf)) * (psi_dot / v));
        let beta_dot = (mu / (v * (lr + lf)))
            * (c_sf * (glr) * delta - (c_sr * (glf) + c_sf * (glr)) * beta
                + (c_sr * (glf) * lr - c_sf * (glr) * lf) * (psi_dot / v))
            - psi_dot;
        [
            v * (psi + beta).cos(),
            v * sin(psi + beta),
            sv,
            acc,
            psi_dot,
            psi_ddot,
            beta_dot,
        ]
    }
}

// ------------------------------------------------------------------ pilot.tires (saturating tires)

#[inline]
fn pacejka(alpha: f64, c_alpha: f64, d: f64, c: f64) -> f64 {
    if d <= TIRE_EPS {
        return 0.0;
    }
    d * sin(c * (c_alpha / (c * d) * alpha).atan())
}

#[inline]
fn fiala(alpha: f64, c_alpha: f64, d: f64) -> f64 {
    if d <= TIRE_EPS {
        return 0.0;
    }
    let z = c_alpha * alpha.abs() / (3.0 * d);
    if z >= 1.0 {
        return d.copysign(alpha);
    }
    c_alpha * alpha * (1.0 - z + z * z / 3.0)
}

#[inline]
fn loads(a: f64, m: f64, lf: f64, lr: f64, h: f64, l: f64) -> (f64, f64) {
    (
        py_max(m * (G * lr - a * h) / l, 0.0),
        py_max(m * (G * lf + a * h) / l, 0.0),
    )
}

#[inline]
fn long_forces(a_cmd: f64, fzf: f64, fzr: f64, mu: f64, m: f64, split: Option<f64>) -> (f64, f64) {
    let lam = match split {
        None => fzf / (fzf + fzr),
        Some(s) => s,
    };
    let (cf, cr) = (mu * fzf, mu * fzr);
    let fxf = py_min(py_max(lam * m * a_cmd, -cf), cf);
    let fxr = py_min(py_max((1.0 - lam) * m * a_cmd, -cr), cr);
    (fxf, fxr)
}

#[derive(Clone, Copy, Debug)]
pub struct AxleForces {
    pub a: f64,
    pub fyf: f64,
    pub fyr: f64,
}

/// ``pilot.tires.axle_forces`` for the nonlinear laws (V >= 0.5).
pub fn axle_forces(x: &State, a_cmd: f64, p: &Params) -> AxleForces {
    let (v, delta, r, beta) = (x[3], x[2], x[5], x[6]);
    let (m, lf, lr, h, mu) = (p.m, p.lf, p.lr, p.h, p.mu);
    let l = lf + lr;
    let af = delta - beta - lf * r / v;
    let ar = -beta + lr * r / v;
    let mu_c = if p.stiffness_scales_with_mu {
        mu
    } else {
        p.mu0
    };
    let split = if a_cmd >= 0.0 {
        p.drive_split_front
    } else {
        p.brake_split_front
    };
    let (mut fzf, mut fzr) = loads(a_cmd, m, lf, lr, h, l);
    let (mut fxf, mut fxr) = long_forces(a_cmd, fzf, fzr, mu, m, split);
    let mut a = (fxf + fxr) / m;
    if a != a_cmd {
        (fzf, fzr) = loads(a, m, lf, lr, h, l);
        (fxf, fxr) = long_forces(a_cmd, fzf, fzr, mu, m, split);
        a = (fxf + fxr) / m;
    }
    let xif = if fzf > 0.0 {
        py_max(1.0 - pow2(fxf / (mu * fzf)), 0.0).sqrt()
    } else {
        0.0
    };
    let xir = if fzr > 0.0 {
        py_max(1.0 - pow2(fxr / (mu * fzr)), 0.0).sqrt()
    } else {
        0.0
    };
    let (caf, car) = (mu_c * p.c_sf * fzf, mu_c * p.c_sr * fzr);
    let peak = p.derate_peak;
    let df = (if peak { xif } else { 1.0 }) * mu * fzf;
    let dr = (if peak { xir } else { 1.0 }) * mu * fzr;
    let (mut fyf, mut fyr) = match p.tire {
        TireLaw::Pacejka => (
            pacejka(af, caf, df, p.shape_c),
            pacejka(ar, car, dr, p.shape_c),
        ),
        TireLaw::Fiala => (fiala(af, caf, df), fiala(ar, car, dr)),
        TireLaw::Linear => unreachable!("linear tires use st_linear"),
    };
    if !peak {
        fyf *= xif; // xif * fyf in Python; IEEE multiplication is commutative
        fyr *= xir;
    }
    AxleForces { a, fyf, fyr }
}

/// ``pilot.tires.st_force_dynamics`` (nonlinear laws).
pub fn st_force(x: &State, u: [f64; 2], p: &Params) -> State {
    let v = x[3];
    if v < 0.5 {
        return st_linear(x, u, p);
    }
    let sv = steering_constraint(x[2], u[0], p.s_min, p.s_max, p.sv_min, p.sv_max);
    let a_cmd = accl_constraints(v, u[1], p.v_switch, p.a_max, p.v_min, p.v_max);
    let f = axle_forces(x, a_cmd, p);
    let (psi, r, beta) = (x[4], x[5], x[6]);
    [
        v * (psi + beta).cos(),
        v * sin(psi + beta),
        sv,
        f.a,
        r,
        (p.lf * f.fyf - p.lr * f.fyr) / p.i_z,
        (f.fyf + f.fyr) / (p.m * v) - r,
    ]
}

/// ``pilot.tires.vehicle_dynamics_st_tire``: dispatch on the tire law.
#[inline]
pub fn st_dynamics(x: &State, u: [f64; 2], p: &Params) -> State {
    match p.tire {
        TireLaw::Linear => st_linear(x, u, p),
        _ => st_force(x, u, p),
    }
}

/// ``RK4Integrator.integrate`` (numpy element-wise arithmetic, same order).
pub fn rk4(x: &State, u: [f64; 2], dt: f64, p: &Params) -> State {
    let k1 = st_dynamics(x, u, p);
    let mut xs = [0.0; 7];
    for i in 0..7 {
        xs[i] = x[i] + dt * (k1[i] / 2.0);
    }
    let k2 = st_dynamics(&xs, u, p);
    for i in 0..7 {
        xs[i] = x[i] + dt * (k2[i] / 2.0);
    }
    let k3 = st_dynamics(&xs, u, p);
    for i in 0..7 {
        xs[i] = x[i] + dt * k3[i];
    }
    let k4 = st_dynamics(&xs, u, p);
    let c = dt * (1.0 / 6.0);
    let mut out = [0.0; 7];
    for i in 0..7 {
        out[i] = x[i] + c * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

/// One ``RaceCar.update_pose`` with ``control_input=["accl", "steering_speed"]``, ``normalize_act=False``:
/// the steering command goes through the 2-step FIFO (``steer_buffer``, newest first), both commands are
/// passed through unscaled, RK4, then ``state[4] %= 2 * np.pi``.
#[derive(Clone, Debug)]
pub struct Plant {
    pub x: State,
    /// ``RaceCar.steer_buffer`` in NumPy order (index 0 = newest command)
    pub steer_buffer: Vec<f64>,
    pub dt: f64,
    pub params: Params,
}

impl Plant {
    pub fn step(&mut self, raw_steer: f64, raw_accl: f64) {
        const SIZE: usize = 2;
        let steer = if self.steer_buffer.len() < SIZE {
            self.steer_buffer.insert(0, raw_steer);
            0.0
        } else {
            let s = self.steer_buffer.pop().unwrap();
            self.steer_buffer.insert(0, raw_steer);
            s
        };
        // SteeringSpeedAction / AcclAction with normalize=False: action * 1.0
        let u = [steer * 1.0, raw_accl * 1.0];
        let mut x = rk4(&self.x, u, self.dt, &self.params);
        x[4] = py_mod(x[4], TWO_PI);
        self.x = x;
    }
}
