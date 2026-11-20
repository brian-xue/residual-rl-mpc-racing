//! Reference line projection (``pilot.reference.Reference.project``) and periodic speed-plan lookup.

use crate::numerics::{np_interp, py_imod, py_mod, sin};

#[derive(Clone, Debug)]
pub struct RefLine {
    pub s: Vec<f64>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub psi: Vec<f64>,
    pub length: f64,
    /// ``Reference.kappa`` and its periodic lookup ``Reference.kappa_at``
    pub kappa: Vec<f64>,
    pub kappa_at: Periodic,
}

pub const PROJECT_WINDOW: i64 = 60;

impl RefLine {
    pub fn n(&self) -> usize {
        self.s.len()
    }

    /// ``Reference.project(x, y, i_guess, window=60)``: nearest sample (first minimum of the squared
    /// distance over the window, NaN-first like ``np.argmin``), then (s, e_y, i).
    pub fn project(&self, x: f64, y: f64, i_guess: Option<i64>) -> (f64, f64, usize) {
        let n = self.n() as i64;
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        let mut first = true;
        let mut visit = |i: usize| -> bool {
            let dx = self.x[i] - x;
            let dy = self.y[i] - y;
            let d = dx * dx + dy * dy; // array ** 2 is an exact square in NumPy
            if d.is_nan() {
                best = i;
                return true; // np.argmin returns the first NaN
            }
            if first || d < best_d {
                best = i;
                best_d = d;
                first = false;
            }
            false
        };
        match i_guess {
            None => {
                for i in 0..self.n() {
                    if visit(i) {
                        break;
                    }
                }
            }
            Some(g) => {
                for k in (g - PROJECT_WINDOW)..(g + PROJECT_WINDOW) {
                    if visit(py_imod(k, n) as usize) {
                        break;
                    }
                }
            }
        }
        let i = best;
        let (tx, ty) = (self.psi[i].cos(), sin(self.psi[i]));
        let (rx, ry) = (x - self.x[i], y - self.y[i]);
        let along = rx * tx + ry * ty;
        let e_y = -rx * ty + ry * tx;
        let s = py_mod(self.s[i] + along, self.length);
        (s, e_y, i)
    }
}

/// ``lambda s: np.interp(np.mod(s, L), s_ext, v_ext)`` from ``LapSim.__init__``.
#[derive(Clone, Debug)]
pub struct Periodic {
    pub length: f64,
    pub xp: Vec<f64>,
    pub fp: Vec<f64>,
}

impl Periodic {
    #[inline]
    pub fn at(&self, s: f64) -> f64 {
        np_interp(py_mod(s, self.length), &self.xp, &self.fp)
    }
}
