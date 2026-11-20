//! ``pilot_core``: Rust core of the residual-RL racing pilot, exposed to Python with PyO3.
//!
//! * ``Geometry``  track-limit / wall checks (``pilot.geometry.TrackGeometry.check``)
//! * ``RefLine``   Frenet projection (``pilot.reference.Reference.project``)
//! * ``LapKernel`` the per-physics-step loop of ``pilot.closedloop.LapSim``
//! * ``st_dynamics``, ``rk4_step``  the ST plant (Gym-Khana linear tires, ``pilot.tires`` nonlinear tires)
//!
//! Python crosses into Rust once per control interval; all physics-rate work stays here.

pub mod dynamics;
pub mod geometry;
pub mod kernel;
pub mod numerics;
pub mod params;
pub mod track;

use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use dynamics::{Plant, State};
use kernel::{Lap, LapConfig, LapMetrics};
use params::Params;

fn state_from(a: &PyReadonlyArray1<'_, f64>) -> PyResult<State> {
    let s = a.as_slice()?;
    if s.len() < 7 {
        return Err(PyValueError::new_err("state must have 7 entries"));
    }
    let mut x = [0.0; 7];
    x.copy_from_slice(&s[..7]);
    Ok(x)
}

fn rows2(a: &PyReadonlyArray2<'_, f64>, cols: usize) -> PyResult<Vec<Vec<f64>>> {
    let v = a.as_array();
    if v.ncols() != cols {
        return Err(PyValueError::new_err(format!(
            "expected {cols} columns, got {}",
            v.ncols()
        )));
    }
    Ok(v.rows().into_iter().map(|r| r.to_vec()).collect())
}

fn get<'py, T: for<'a> FromPyObject<'a, 'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<T>
where
    for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
{
    match d.get_item(k)? {
        Some(v) => v.extract::<T>().map_err(Into::into),
        None => Err(PyValueError::new_err(format!("missing key {k:?}"))),
    }
}

/// ``Geometry(legal_rings, wall_bounds, tire_offsets, patch_len, patch_w, length, width, legal_cell, wall_cell)``
#[pyclass(name = "Geometry", frozen, module = "pilot_core")]
struct PyGeometry {
    inner: Arc<geometry::Geometry>,
}

#[pymethods]
impl PyGeometry {
    #[new]
    #[pyo3(signature = (legal_rings, wall_bounds, tire_offsets, patch_len, patch_w, length, width, legal_cell=1.0, wall_cell=0.5))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        legal_rings: Vec<PyReadonlyArray2<'_, f64>>,
        wall_bounds: PyReadonlyArray2<'_, f64>,
        tire_offsets: PyReadonlyArray2<'_, f64>,
        patch_len: f64,
        patch_w: f64,
        length: f64,
        width: f64,
        legal_cell: f64,
        wall_cell: f64,
    ) -> PyResult<Self> {
        let mut rings = Vec::new();
        for r in &legal_rings {
            rings.push(
                rows2(r, 2)?
                    .into_iter()
                    .map(|p| [p[0], p[1]])
                    .collect::<Vec<_>>(),
            );
        }
        let walls = rows2(&wall_bounds, 4)?
            .into_iter()
            .map(|b| [b[0], b[1], b[2], b[3]])
            .collect();
        let to = rows2(&tire_offsets, 2)?;
        if to.len() != 4 {
            return Err(PyValueError::new_err("tire_offsets must be (4, 2)"));
        }
        let offsets = [
            [to[0][0], to[0][1]],
            [to[1][0], to[1][1]],
            [to[2][0], to[2][1]],
            [to[3][0], to[3][1]],
        ];
        Ok(Self {
            inner: Arc::new(geometry::Geometry {
                legal: geometry::Region::new(&rings, legal_cell),
                walls: geometry::Boxes::new(walls, wall_cell),
                footprint: geometry::Footprint::new(offsets, patch_len, patch_w, length, width),
            }),
        })
    }

    /// ``(off_track, wall_collision, tires_in)`` as ``TrackGeometry.check``.
    fn check(
        &self,
        x: f64,
        y: f64,
        yaw: f64,
        delta: f64,
    ) -> (bool, bool, (bool, bool, bool, bool)) {
        let (off, wall, t) = self.inner.check(x, y, yaw, delta);
        (off, wall, (t[0], t[1], t[2], t[3]))
    }

    /// Footprint corners, flat (5 * 4 * 2): reshape to (5, 4, 2) for ``TrackGeometry.footprints``.
    fn footprints<'py>(
        &self,
        py: Python<'py>,
        x: f64,
        y: f64,
        yaw: f64,
        delta: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let c = self.inner.footprint.corners(x, y, yaw, delta);
        let flat: Vec<f64> = c
            .iter()
            .flat_map(|r| r.iter().flat_map(|p| p.iter().copied()))
            .collect();
        PyArray1::from_vec(py, flat)
    }
}

/// ``RefLine(s, x, y, psi, length)``
#[pyclass(name = "RefLine", frozen, module = "pilot_core")]
struct PyRefLine {
    inner: Arc<track::RefLine>,
}

#[pymethods]
impl PyRefLine {
    #[new]
    fn new(
        s: PyReadonlyArray1<'_, f64>,
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        psi: PyReadonlyArray1<'_, f64>,
        length: f64,
    ) -> PyResult<Self> {
        let (s, x, y, psi) = (
            s.as_slice()?.to_vec(),
            x.as_slice()?.to_vec(),
            y.as_slice()?.to_vec(),
            psi.as_slice()?.to_vec(),
        );
        if [x.len(), y.len(), psi.len()].iter().any(|&n| n != s.len()) || s.is_empty() {
            return Err(PyValueError::new_err(
                "s, x, y, psi must have the same non-zero length",
            ));
        }
        Ok(Self {
            inner: Arc::new(track::RefLine {
                s,
                x,
                y,
                psi,
                length,
            }),
        })
    }

    /// ``Reference.project(x, y, i_guess)`` -> (s, e_y, i)
    #[pyo3(signature = (x, y, i_guess=None))]
    fn project(&self, x: f64, y: f64, i_guess: Option<i64>) -> (f64, f64, usize) {
        self.inner.project(x, y, i_guess)
    }
}

/// ``vehicle_dynamics_st_tire(x, u, params)`` (tire law from ``params["tire_model"]``).
#[pyfunction]
fn st_dynamics<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray1<'py, f64>,
    params: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = Params::from_dict(params)?;
    let u = u.as_slice()?;
    Ok(PyArray1::from_slice(
        py,
        &dynamics::st_dynamics(&state_from(&x)?, [u[0], u[1]], &p),
    ))
}

/// ``RK4Integrator.integrate(vehicle_dynamics_st_tire, x, u, dt, params)``
#[pyfunction]
fn rk4_step<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray1<'py, f64>,
    dt: f64,
    params: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = Params::from_dict(params)?;
    let u = u.as_slice()?;
    Ok(PyArray1::from_slice(
        py,
        &dynamics::rk4(&state_from(&x)?, [u[0], u[1]], dt, &p),
    ))
}

type Sync<'py> = (
    bool,
    Option<&'static str>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    (f64, f64, f64, f64, f64, f64, f64),
    (u64, usize, u64, i64),
    (bool, bool, bool, bool),
);

/// ``LapKernel(geometry, refline, params, init)``: see ``pilot.kernel.make_kernel`` for ``init``.
#[pyclass(name = "LapKernel", module = "pilot_core")]
struct PyLapKernel {
    lap: Lap,
}

#[pymethods]
impl PyLapKernel {
    #[new]
    fn new(
        geometry: &Bound<'_, PyGeometry>,
        refline: &Bound<'_, PyRefLine>,
        params: &Bound<'_, PyDict>,
        init: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let p = Params::from_dict(params)?;
        let state: PyReadonlyArray1<'_, f64> = get(init, "state")?;
        let st: PyReadonlyArray1<'_, f64> = get(init, "st")?;
        let steer_buffer: Vec<f64> = get(init, "steer_buffer")?;
        let pending: Vec<f64> = get(init, "servo_pending")?;
        if pending.len() != 2 {
            return Err(PyValueError::new_err(
                "servo_pending must have 2 entries (steering_speed servo)",
            ));
        }
        let s_ext: PyReadonlyArray1<'_, f64> = get(init, "plan_s")?;
        let v_ext: PyReadonlyArray1<'_, f64> = get(init, "plan_v")?;
        let cfg = LapConfig {
            dt: get(init, "dt")?,
            n_sub: get(init, "n_sub")?,
            t_max: get(init, "t_max")?,
            kv: get(init, "kv")?,
            a_max: get(init, "a_max")?,
            ell_ax: get(init, "ell_ax")?,
            ell_ay: get(init, "ell_ay")?,
            sv_max: get(init, "sv_max")?,
            spin_beta_max: get(init, "spin_beta_max")?,
            own_env: get(init, "own_env")?,
            max_episode_steps: get(init, "max_episode_steps")?,
        };
        let refl = refline.get().inner.clone();
        let lap = Lap {
            plant: Plant {
                x: state_from(&state)?,
                steer_buffer,
                dt: cfg.dt,
                params: p,
            },
            geo: geometry.get().inner.clone(),
            vplan: track::Periodic {
                length: get(init, "plan_length")?,
                xp: s_ext.as_slice()?.to_vec(),
                fp: v_ext.as_slice()?.to_vec(),
            },
            refl,
            servo_pending: [pending[0], pending[1]],
            st: state_from(&st)?,
            t: get(init, "t")?,
            step: get(init, "step")?,
            env_step: get(init, "env_step")?,
            s: get(init, "s")?,
            e_y: get(init, "e_y")?,
            e_psi: get(init, "e_psi")?,
            v: get(init, "v")?,
            vp: get(init, "vp")?,
            i_g: get(init, "i_g")?,
            s_prev: get(init, "s_prev")?,
            progress: get(init, "progress")?,
            off_in_interval: get(init, "off_in_interval")?,
            tires_in: get(init, "tires_in")?,
            metrics: LapMetrics {
                max_ey: get(init, "max_ey")?,
                sum_ey2: get(init, "sum_ey2")?,
                n_ey: get(init, "n_ey")?,
                sum_dv: get(init, "sum_dv")?,
                ell_max: get(init, "ell_max")?,
                n_ell_over: get(init, "n_ell_over")?,
                max_ay: get(init, "max_ay")?,
                min_ax: get(init, "min_ax")?,
                n_off: get(init, "n_off")?,
                first_off_s: get(init, "first_off_s")?,
                first_off_progress: get(init, "first_off_progress")?,
                n_sat_rate: get(init, "n_sat_rate")?,
            },
            outcome: None,
            done: get(init, "done")?,
            delta_cmd: 0.0,
            a_mpc: None,
            cfg,
        };
        Ok(Self { lap })
    }

    /// Run physics steps up to the next control instant or the end of the lap.
    /// Returns (done, outcome, st, plant_state, (t, s, e_y, e_psi, v, vp, progress),
    /// (step, i_g, off_in_interval, env_step), tires_in).
    #[pyo3(signature = (delta_cmd, a_mpc=None))]
    fn advance<'py>(&mut self, py: Python<'py>, delta_cmd: f64, a_mpc: Option<f64>) -> Sync<'py> {
        let l = &mut self.lap;
        l.advance(delta_cmd, a_mpc);
        let t = l.tires_in;
        (
            l.done,
            l.outcome.map(|o| o.as_str()),
            PyArray1::from_slice(py, &l.st),
            PyArray1::from_slice(py, &l.plant.x),
            (l.t, l.s, l.e_y, l.e_psi, l.v, l.vp, l.progress),
            (l.step, l.i_g, l.off_in_interval, l.env_step),
            (t[0], t[1], t[2], t[3]),
        )
    }

    /// Lap metrics accumulated so far (``LapSim`` attribute names).
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = &self.lap.metrics;
        let d = PyDict::new(py);
        d.set_item("max_ey", m.max_ey)?;
        d.set_item("sum_ey2", m.sum_ey2)?;
        d.set_item("n_ey", m.n_ey)?;
        d.set_item("sum_dv", m.sum_dv)?;
        d.set_item("ell_max", m.ell_max)?;
        d.set_item("n_ell_over", m.n_ell_over)?;
        d.set_item("max_ay", m.max_ay)?;
        d.set_item("min_ax", m.min_ax)?;
        d.set_item("n_off", m.n_off)?;
        d.set_item("first_off_s", m.first_off_s)?;
        d.set_item("first_off_progress", m.first_off_progress)?;
        d.set_item("n_sat_rate", m.n_sat_rate)?;
        Ok(d)
    }
}

#[pymodule]
fn pilot_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyRefLine>()?;
    m.add_class::<PyLapKernel>()?;
    m.add_function(wrap_pyfunction!(st_dynamics, m)?)?;
    m.add_function(wrap_pyfunction!(rk4_step, m)?)?;
    Ok(())
}
