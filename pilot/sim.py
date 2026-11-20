"""Thin wrapper around the pinned Gym-Khana 1.2.0 environment.

Verified against the installed source (gymkhana/envs/action.py, base_classes.py,
dynamic_models/single_track.py):

* action order is ``[steer, longitudinal]`` per agent, shape ``(num_agents, 2)``
* ``normalize_act=False`` -> physical units (rad, rad/s, m/s^2)
* ``steering_angle`` mode: target angle -> bang-bang steering velocity at ``sv_max``
* ``steering_speed`` mode: steering velocity passed through, clipped to ``sv_max``
* ``accl`` mode: acceleration passed through, clipped to ``a_max`` (scaled above ``v_switch``)
* every steering command passes a hidden 2-step FIFO (``steer_buffer_size=2``),
  so the command issued at step k acts at step k+2 (20 ms at dt = 0.01 s)
* ST state: ``[x, y, delta, v, yaw, yaw_rate, slip]``; (x, y) is the centre of
  gravity and ``v`` is the CG speed (x_dot = v cos(yaw + slip)).
* ``mu`` multiplies both axle forces of the linear ST tire model (no saturation).
* Optional saturating tires for ST (``tire:`` config key, see ``pilot.tires``): every ST agent's ``model`` is
  replaced by ``_STTirePlant``, whose ``f_dynamics`` dispatches on ``params["tire_model"]`` ("linear" calls
  gymkhana's own ``vehicle_dynamics_st``); state, reset and observations stay ST's. Nothing in site-packages
  is changed.
* STD state: ST state + ``[omega_front, omega_rear]``; PAC2002 tires saturate. ``params["mu"]`` is NOT
  read by the STD model: friction enters only through the Pacejka peak coefficients ``tire_p_dy1``
  (lateral) and ``tire_p_dx1`` (longitudinal). The pilot scales both by ``mu_scale`` (cornering stiffness
  ``tire_p_ky1`` unchanged). STD resets accept a full 7-state (wheel speeds set to free rolling).
"""

from __future__ import annotations

import copy
import warnings

import gymnasium as gym
import numpy as np

warnings.filterwarnings("ignore")

from gymkhana.envs.dynamic_models import DynamicModel  # noqa: E402
from gymkhana.envs.gymkhana_env import GKEnv  # noqa: E402

from .tires import TIRE_MODELS, vehicle_dynamics_st_tire  # noqa: E402

NOMINAL_PARAMS = GKEnv.f1tenth_vehicle_params()  # geometry (lf, lr, width, length) is identical in ST and STD
# ST tire options (config ``tire:``, a model name or a mapping of these keys; see pilot.tires)
TIRE_DEFAULTS = {
    "model": "linear",                  # "linear" (gymkhana ST), "pacejka" or "fiala"
    "C": 1.3,                           # Pacejka shape factor (Liniger et al. 1:43 cars: 1.2 front, 1.27 rear)
    "stiffness_scales_with_mu": False,  # False: mu changes only the peak force; True: also C_alpha (as linear ST)
    "drive_split_front": None,          # front share of the traction force; None -> proportional to axle load
    "brake_split_front": None,          # front share of the braking force; None -> proportional to axle load
    "derate": "peak",                   # friction circle: "peak" (D = xi mu Fz, C_alpha kept) or "curve" (xi Fy0)
}
_PLANT = {"model": "st", "tire": dict(TIRE_DEFAULTS)}


def tire_config(tire=None) -> dict:
    if tire is None:
        tire = {}
    elif isinstance(tire, str):
        tire = {"model": tire}
    unknown = set(tire) - set(TIRE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown tire option(s) {sorted(unknown)}; expected {sorted(TIRE_DEFAULTS)}")
    t = {**TIRE_DEFAULTS, **tire}
    if t["model"] not in TIRE_MODELS:
        raise ValueError(f"tire model must be one of {TIRE_MODELS}, got {t['model']}")
    if t["derate"] not in ("peak", "curve"):
        raise ValueError(f"tire derate must be 'peak' or 'curve', got {t['derate']}")
    for k in ("drive_split_front", "brake_split_front"):
        if t[k] is not None and not 0.0 <= t[k] <= 1.0:
            raise ValueError(f"tire {k} must be in [0, 1] or null, got {t[k]}")
    return t


def set_plant(model: str, tire=None):
    """``tire``: the config's ``tire:`` entry (None / model name / mapping); only the ST plant accepts non-linear."""
    if model not in ("st", "std"):
        raise ValueError(f"plant model must be 'st' or 'std', got {model}")
    t = tire_config(tire)
    if model == "std" and t["model"] != "linear":
        raise ValueError("the 'tire' option applies to the ST plant only (STD has its own PAC2002 tires)")
    _PLANT["model"], _PLANT["tire"] = model, t


def plant_model() -> str:
    return _PLANT["model"]


def plant_tire() -> dict:
    return dict(_PLANT["tire"])


def plant_label() -> str:
    t = _PLANT["tire"]["model"]
    return _PLANT["model"] if t == "linear" else f"{_PLANT['model']}+{t}"


class _STTirePlant:
    """Stands in for ``DynamicModel.ST`` on a RaceCar: ``update_pose`` reads ``self.model.f_dynamics`` every
    step, which here dispatches on ``params["tire_model"]``; every other attribute is ``DynamicModel.ST``'s."""

    f_dynamics = staticmethod(vehicle_dynamics_st_tire)

    def __getattr__(self, name):
        return getattr(DynamicModel.ST, name)


ST_TIRE_PLANT = _STTirePlant()


def vehicle_params(mu_scale: float = 1.0, model: str | None = None) -> dict:
    model = model or _PLANT["model"]
    if model == "st":
        p = copy.deepcopy(NOMINAL_PARAMS)
        p["mu"] = NOMINAL_PARAMS["mu"] * mu_scale
        t = _PLANT["tire"]
        p.update(mu0=NOMINAL_PARAMS["mu"], tire_model=t["model"], tire_shape_C=t["C"],
                 tire_stiffness_scales_with_mu=t["stiffness_scales_with_mu"],
                 tire_drive_split_front=t["drive_split_front"], tire_brake_split_front=t["brake_split_front"],
                 tire_derate=t["derate"])
        return p
    p = GKEnv.f1tenth_std_vehicle_params()
    p["tire_p_dy1"] *= mu_scale
    p["tire_p_dx1"] *= mu_scale
    p["mu"] *= mu_scale  # unused by STD, kept consistent
    return p


def make_env(track: str, mu_scale: float, steer_mode: str, dt: float = 0.01, max_steps: int = 100_000):
    cfg = {
        "map": track,
        "num_agents": 1,
        "timestep": dt,
        "integrator": "rk4",
        "model": _PLANT["model"],
        "control_input": ["accl", steer_mode],
        "normalize_act": False,
        "params": vehicle_params(mu_scale),
        "max_episode_steps": max_steps,
        "num_beams": 10,
        "predictive_collision": False,  # Frenet boundary check -> terminated when |ey| > half width
    }
    env = gym.make("gymkhana:gymkhana-v0", config=cfg).unwrapped
    if _PLANT["model"] == "st":  # always installed: the tire law follows params, also after update_params()
        for a in env.sim.agents:
            a.model = ST_TIRE_PLANT
    return env


def set_env_params(env, params: dict):
    """Give an existing env the params ``make_env`` would have built it with (agents and env-level copy)."""
    env.sim.update_params(params)
    env.params = env.config["params"] = params


def agent(env):
    return env.sim.agents[0]


def reset_to_state(env, state7):
    """ST: reset at a pose, then overwrite the full state (ST reset only accepts poses).
    STD: reset with the full 7-state; Gym-Khana sets free-rolling wheel speeds."""
    s = np.asarray(state7, dtype=float)
    if _PLANT["model"] == "std":
        env.reset(options={"states": s[None, :7]})
    else:
        env.reset(options={"poses": np.array([[s[0], s[1], s[4]]])})
        agent(env).state = s.copy()
    agent(env).curr_vel_cmd = s[3]
    return agent(env).state


class SteeringServo:
    """Turns a steering-angle target into a ``steering_speed`` command.

    Accounts for the simulator's 2-step command FIFO (predicts delta after the
    pending commands act) and enforces the shared rate limit. With
    ``steer_mode='steering_angle'`` the target angle is passed through instead and
    the simulator's own bang-bang servo is used.
    """

    def __init__(self, mode: str, dt: float, sv_max: float, delay_steps: int = 2):
        self.mode, self.dt, self.sv_max, self.d = mode, dt, sv_max, delay_steps
        self.pending = [0.0] * delay_steps

    def reset(self):
        self.pending = [0.0] * self.d

    def command(self, delta_target: float, delta_now: float) -> float:
        if self.mode == "steering_angle":
            return delta_target
        delta_pred = delta_now + self.dt * sum(self.pending)
        sv = float(np.clip((delta_target - delta_pred) / self.dt, -self.sv_max, self.sv_max))
        self.pending = self.pending[1:] + [sv]
        return sv
