"""Traced SubprocVecEnv for frozen-policy PPO rollouts (analysis only; the repository code is not modified).

``TracedSubprocVecEnv`` is SB3 2.9.2's ``SubprocVecEnv`` with a worker that is a copy of SB3's ``_worker`` plus
timestamps. All timestamps use ``time.perf_counter_ns()`` (``mach_absolute_time`` on macOS: one system-wide
monotonic clock, comparable across processes).

Per vectorised step and worker it records:
    dispatch   main, just before ``remote.send(("step", a))``
    recv       worker, right after ``remote.recv()`` returned the command
    step0/1    worker, around ``env.step`` (+ worker CPU time of the step)
    reset0/1   worker, around the auto-reset after a done (0 if none)
    send       worker, just before ``remote.send(result)``
    got        main, right after ``remote.recv()`` returned this worker's result (receipts are sequential, in
               worker order, as in SB3)
plus, inside env.step, the number and duration of ``LapSim.plan`` calls and the Rust kernel's OSQP solve time
(OSQP iteration counts are not exposed by the Rust kernel; see the report).
Main-process timestamps at the start of ``step_async`` and end of ``step_wait`` bound the main-process work
(policy forward, buffer, VecMonitor/VecNormalize) between vectorised steps.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _stack_obs

ns = time.perf_counter_ns


def _traced_worker(remote, parent_remote, env_fn_wrapper):
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.vec_env.patch_gym import _patch_env

    import pilot.closedloop as cl

    acc = {"n_plan": 0, "plan_ns": 0, "solve_ms": 0.0, "solve_ms_max": 0.0}
    f_plan = cl.LapSim.plan

    def plan(self):
        t0 = ns()
        r = f_plan(self)
        acc["plan_ns"] += ns() - t0
        acc["n_plan"] += 1
        if self.solve_times:
            s = self.solve_times[-1] * 1e3
            acc["solve_ms"] += s
            acc["solve_ms_max"] = max(acc["solve_ms_max"], s)
        return r

    cl.LapSim.plan = plan
    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info = {}
    while True:
        try:
            cmd, data = remote.recv()
            t_recv = ns()
            if cmd == "step":
                for k in acc:
                    acc[k] = 0 if k != "solve_ms_max" else 0.0
                acc["solve_ms"] = 0.0
                c0 = time.process_time_ns()
                t_s0 = ns()
                observation, reward, terminated, truncated, info = env.step(data)
                t_s1 = ns()
                c1 = time.process_time_ns()
                step_plan = dict(acc)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                t_r0 = t_r1 = 0
                reset_plan = {}
                if done:
                    info["terminal_observation"] = observation
                    for k in acc:
                        acc[k] = 0
                    acc["solve_ms"], acc["solve_ms_max"] = 0.0, 0.0
                    t_r0 = ns()
                    observation, reset_info = env.reset()
                    t_r1 = ns()
                    reset_plan = dict(acc)
                info["_trace"] = dict(recv=t_recv, step0=t_s0, step1=t_s1, cpu_step=c1 - c0, reset0=t_r0, reset1=t_r1,
                                      n_plan=step_plan["n_plan"], plan_ns=step_plan["plan_ns"],
                                      solve_ms=step_plan["solve_ms"], solve_ms_max=step_plan["solve_ms_max"],
                                      reset_plan_ns=reset_plan.get("plan_ns", 0), outcome=info.get("outcome"),
                                      send=ns())
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(cmd)
        except (EOFError, KeyboardInterrupt):
            break


class TracedSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, start_method="spawn"):
        self.waiting = False
        self.closed = False
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(len(env_fns))], strict=True)
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns, strict=True):
            p = ctx.Process(target=_traced_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.records = []  # one dict per vectorised step
        self._cur = None

    def step_async(self, actions):
        t0 = ns()
        c0 = time.process_time_ns()
        disp = []
        for remote, action in zip(self.remotes, actions, strict=True):
            disp.append(ns())
            remote.send(("step", action))
        self._cur = dict(async0=t0, async_cpu0=c0, dispatch=disp)
        self.waiting = True

    def step_wait(self):
        got, results = [], []
        for remote in self.remotes:
            results.append(remote.recv())
            got.append(ns())
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results, strict=True)
        tr = [i.pop("_trace") for i in infos]
        self._cur.update(got=got, wait1=ns(), wait_cpu1=time.process_time_ns(), workers=tr)
        self.records.append(self._cur)
        return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos


class EnvFactory:
    """Picklable ``ResidualRacingEnv`` factory (torch single-threaded in the worker, as in train_residual)."""

    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed

    def __call__(self):
        import torch

        from pilot.residual_env import ResidualRacingEnv

        torch.set_num_threads(1)
        return ResidualRacingEnv(self.cfg, seed=self.seed)
