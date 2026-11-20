//! Vehicle parameters, read once from the Python params dict (``pilot.sim.vehicle_params``).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireLaw {
    /// Gym-Khana's linear ST equations (``vehicle_dynamics_st``).
    Linear,
    /// Simplified Magic Formula, ``pilot.tires.pacejka``.
    Pacejka,
    /// Fiala brush model, ``pilot.tires.fiala``.
    Fiala,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub mu: f64,
    pub c_sf: f64,
    pub c_sr: f64,
    pub lf: f64,
    pub lr: f64,
    pub h: f64,
    pub m: f64,
    pub i_z: f64,
    pub s_min: f64,
    pub s_max: f64,
    pub sv_min: f64,
    pub sv_max: f64,
    pub v_switch: f64,
    pub a_max: f64,
    pub v_min: f64,
    pub v_max: f64,
    // saturating-tire options (pilot.tires); defaults are the ``params.get`` defaults there
    pub tire: TireLaw,
    pub mu0: f64,
    pub shape_c: f64,
    pub stiffness_scales_with_mu: bool,
    pub drive_split_front: Option<f64>,
    pub brake_split_front: Option<f64>,
    pub derate_peak: bool,
}

fn req(d: &Bound<'_, PyDict>, k: &str) -> PyResult<f64> {
    match d.get_item(k)? {
        Some(v) => v.extract::<f64>(),
        None => Err(PyValueError::new_err(format!(
            "vehicle params: missing key {k:?}"
        ))),
    }
}

fn opt_f64(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Option<f64>> {
    match d.get_item(k)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract::<f64>()?)),
        _ => Ok(None),
    }
}

impl Params {
    pub fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mu = req(d, "mu")?;
        let tire = match d.get_item("tire_model")? {
            Some(v) if !v.is_none() => match v.extract::<String>()?.as_str() {
                "linear" => TireLaw::Linear,
                "pacejka" => TireLaw::Pacejka,
                "fiala" => TireLaw::Fiala,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown tire_model {other:?}"
                    )));
                }
            },
            _ => TireLaw::Linear,
        };
        let derate_peak = match d.get_item("tire_derate")? {
            Some(v) if !v.is_none() => match v.extract::<String>()?.as_str() {
                "peak" => true,
                "curve" => false,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown tire_derate {other:?}"
                    )));
                }
            },
            _ => true,
        };
        let stiffness_scales_with_mu = match d.get_item("tire_stiffness_scales_with_mu")? {
            Some(v) if !v.is_none() => v.is_truthy()?,
            _ => false,
        };
        Ok(Self {
            mu,
            c_sf: req(d, "C_Sf")?,
            c_sr: req(d, "C_Sr")?,
            lf: req(d, "lf")?,
            lr: req(d, "lr")?,
            h: req(d, "h")?,
            m: req(d, "m")?,
            i_z: req(d, "I")?,
            s_min: req(d, "s_min")?,
            s_max: req(d, "s_max")?,
            sv_min: req(d, "sv_min")?,
            sv_max: req(d, "sv_max")?,
            v_switch: req(d, "v_switch")?,
            a_max: req(d, "a_max")?,
            v_min: req(d, "v_min")?,
            v_max: req(d, "v_max")?,
            tire,
            mu0: opt_f64(d, "mu0")?.unwrap_or(mu),
            shape_c: opt_f64(d, "tire_shape_C")?.unwrap_or(1.3),
            stiffness_scales_with_mu,
            drive_split_front: opt_f64(d, "tire_drive_split_front")?,
            brake_split_front: opt_f64(d, "tire_brake_split_front")?,
            derate_peak,
        })
    }
}
