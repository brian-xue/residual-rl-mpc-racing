"""Saturating tires for the Gym-Khana ST plant (in-repo; the installed package is not modified).

The ST equations of ``gymkhana/envs/dynamic_models/single_track.py`` are rewritten in axle-force form,

    r'     = (lf Fyf - lr Fyr) / I
    beta'  = (Fyf + Fyr) / (m V) - r
    V'     = (Fxf + Fxr) / m
    alpha_f = delta - beta - lf r / V,   alpha_r = -beta + lr r / V      (ST's small-angle slip angles)
    Fzf = m (g lr - a h) / L,            Fzr = m (g lf + a h) / L       (ST's longitudinal load transfer)

With ``Fy = mu C_S Fz alpha`` and ``V' = a`` this is exactly gymkhana's ST (``law="linear"``, checked by
``pilot.sim_checks``). The nonlinear laws keep ST's identified small-slip stiffness and saturate at the
friction limit:

    C_alpha,i = mu_c C_S,i Fz,i          mu_c = mu0 (default: friction changes only the peak) or mu
    D_i       = xi_i mu Fz,i             peak force on the friction circle
    pacejka:  Fy = D sin(C atan(B alpha)),  B = C_alpha / (C D)   (simplified Magic Formula, Liniger et al.)
    fiala:    Fy = C_alpha alpha (1 - z + z^2/3) for z = C_alpha |alpha| / (3 D) < 1, else D sign(alpha)
              (brush model, Gerdes et al.; alpha is ST's small-angle slip, i.e. it plays the role of tan alpha)

Combined slip (force-based, no wheel-speed states, i.e. ideal traction control / ABS):

    Fx,tot = m a_cmd split front/rear by ``tire_drive_split_front`` / ``tire_brake_split_front``
             (None -> proportional to the axle normal loads); each axle |Fx,i| <= mu Fz,i
    a      = (Fxf + Fxr) / m;  Fz recomputed once with this realised acceleration, Fx re-saturated
    xi_i   = sqrt(1 - (Fx,i / (mu Fz,i))^2)
    derate "peak":  D_i = xi_i mu Fz,i with C_alpha kept (Gerdes' derated Fiala; default)
    derate "curve": Fy_i = xi_i Fy0_i(alpha) (whole curve scaled, stiffness reduced too)

Below V = 0.5 m/s gymkhana's kinematic branch is used unchanged.
"""

from __future__ import annotations

import math

import numpy as np
from gymkhana.envs.dynamic_models.single_track import vehicle_dynamics_st
from gymkhana.envs.dynamic_models.utils import accl_constraints, steering_constraint

G = 9.81
TIRE_MODELS = ("linear", "pacejka", "fiala")
_EPS = 1e-9


def pacejka(alpha: float, c_alpha: float, d: float, c: float) -> float:
    if d <= _EPS:
        return 0.0
    return d * math.sin(c * math.atan(c_alpha / (c * d) * alpha))


def fiala(alpha: float, c_alpha: float, d: float) -> float:
    if d <= _EPS:
        return 0.0
    z = c_alpha * abs(alpha) / (3.0 * d)
    if z >= 1.0:
        return math.copysign(d, alpha)
    return c_alpha * alpha * (1.0 - z + z * z / 3.0)


def _loads(a, m, lf, lr, h, L):
    return max(m * (G * lr - a * h) / L, 0.0), max(m * (G * lf + a * h) / L, 0.0)


def _long_forces(a_cmd, fzf, fzr, mu, m, split):
    lam = fzf / (fzf + fzr) if split is None else split
    cf, cr = mu * fzf, mu * fzr
    fxf = min(max(lam * m * a_cmd, -cf), cf)
    fxr = min(max((1.0 - lam) * m * a_cmd, -cr), cr)
    return fxf, fxr


def axle_forces(x: np.ndarray, a_cmd: float, params: dict, law: str) -> dict:
    """Axle loads, slip angles and forces for state ``x`` (V >= 0.5) and constrained acceleration ``a_cmd``."""
    V, delta, r, beta = x[3], x[2], x[5], x[6]
    m, lf, lr, h, mu = params["m"], params["lf"], params["lr"], params["h"], params["mu"]
    L = lf + lr
    af = delta - beta - lf * r / V
    ar = -beta + lr * r / V
    mu_c = mu if (law == "linear" or params.get("tire_stiffness_scales_with_mu", False)) else params.get("mu0", mu)
    if law == "linear":  # gymkhana ST: unsaturated, V' = a
        fzf, fzr = _loads(a_cmd, m, lf, lr, h, L)
        return dict(fzf=fzf, fzr=fzr, fxf=0.0, fxr=0.0, a=a_cmd, af=af, ar=ar,
                    fyf=mu_c * params["C_Sf"] * fzf * af, fyr=mu_c * params["C_Sr"] * fzr * ar, xif=1.0, xir=1.0)
    split = params.get("tire_drive_split_front") if a_cmd >= 0 else params.get("tire_brake_split_front")
    fzf, fzr = _loads(a_cmd, m, lf, lr, h, L)
    fxf, fxr = _long_forces(a_cmd, fzf, fzr, mu, m, split)
    a = (fxf + fxr) / m
    if a != a_cmd:  # one fixed-point pass: load transfer from the acceleration actually realised
        fzf, fzr = _loads(a, m, lf, lr, h, L)
        fxf, fxr = _long_forces(a_cmd, fzf, fzr, mu, m, split)
        a = (fxf + fxr) / m
    xif = math.sqrt(max(1.0 - (fxf / (mu * fzf)) ** 2, 0.0)) if fzf > 0 else 0.0
    xir = math.sqrt(max(1.0 - (fxr / (mu * fzr)) ** 2, 0.0)) if fzr > 0 else 0.0
    caf, car = mu_c * params["C_Sf"] * fzf, mu_c * params["C_Sr"] * fzr
    peak = params.get("tire_derate", "peak") == "peak"
    df, dr = (xif if peak else 1.0) * mu * fzf, (xir if peak else 1.0) * mu * fzr
    if law == "pacejka":
        c = params.get("tire_shape_C", 1.3)
        fyf, fyr = pacejka(af, caf, df, c), pacejka(ar, car, dr, c)
    elif law == "fiala":
        fyf, fyr = fiala(af, caf, df), fiala(ar, car, dr)
    else:
        raise ValueError(f"unknown tire law {law!r}; expected one of {TIRE_MODELS}")
    if not peak:
        fyf, fyr = xif * fyf, xir * fyr
    return dict(fzf=fzf, fzr=fzr, fxf=fxf, fxr=fxr, a=a, af=af, ar=ar, fyf=fyf, fyr=fyr, xif=xif, xir=xir)


def st_force_dynamics(x: np.ndarray, u_init: np.ndarray, params: dict, law: str) -> np.ndarray:
    """ST state derivative in axle-force form with tire ``law`` (same state, inputs and constraints as ST)."""
    V = x[3]
    if V < 0.5:
        return vehicle_dynamics_st(x, u_init, params)
    sv = steering_constraint(x[2], u_init[0], params["s_min"], params["s_max"], params["sv_min"], params["sv_max"])
    a_cmd = accl_constraints(V, u_init[1], params["v_switch"], params["a_max"], params["v_min"], params["v_max"])
    f = axle_forces(x, a_cmd, params, law)
    psi, r, beta = x[4], x[5], x[6]
    return np.array([
        V * math.cos(psi + beta),
        V * math.sin(psi + beta),
        sv,
        f["a"],
        r,
        (params["lf"] * f["fyf"] - params["lr"] * f["fyr"]) / params["I"],
        (f["fyf"] + f["fyr"]) / (params["m"] * V) - r,
    ])


def vehicle_dynamics_st_tire(x: np.ndarray, u_init: np.ndarray, params: dict) -> np.ndarray:
    """Drop-in for gymkhana's ``vehicle_dynamics_st``; the tire law is read from ``params["tire_model"]``.

    ``"linear"`` (or no key) calls gymkhana's own function, so existing ST results are reproduced exactly.
    """
    law = params.get("tire_model", "linear")
    if law == "linear":
        return vehicle_dynamics_st(x, u_init, params)
    return st_force_dynamics(x, u_init, params, law)
