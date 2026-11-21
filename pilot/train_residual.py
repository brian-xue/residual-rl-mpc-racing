"""Train the residual policy with PPO (Stable-Baselines3) on ``pilot.residual_env.ResidualRacingEnv``.

Usage:  uv run python -m pilot.train_residual [--config configs/pilot.yaml] [--steps N] [--holdout TRACK]
Writes checkpoints (ckpt_<steps>.zip), final.zip and train_log.csv to the config's residual.out.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from .protocol import TRACKS
from .residual_env import ResidualRacingEnv


def _make(cfg, seed):
    def f():
        torch.set_num_threads(1)
        return ResidualRacingEnv(cfg, seed=seed)
    return f


class LogAndSave(BaseCallback):
    def __init__(self, out: pathlib.Path, every: int, tracks=()):
        super().__init__()
        self.tracks = list(tracks) if len(tracks) > 1 else []  # per-track columns only for multi-track training
        self.out, self.every, self.next_save = out, every, every
        self.rows, self.ep_buf, self.t0 = [], [], time.time()

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.ep_buf.append(dict(ret=info["episode"]["r"], len=info["episode"]["l"], track=info.get("track"),
                                        crash=info.get("outcome") in ("wall_collision", "spin", "lateral_error")))
        if self.num_timesteps >= self.next_save:
            self.model.save(self.out / f"ckpt_{self.num_timesteps}.zip")
            self.next_save += self.every
        return True

    def _on_rollout_end(self):
        if not self.ep_buf:
            return
        df = pd.DataFrame(self.ep_buf)
        self.ep_buf = []
        row = dict(steps=self.num_timesteps, wall_s=round(time.time() - self.t0, 1), episodes=len(df),
                   ep_return=df.ret.mean(), ep_len=df.len.mean(), crash_rate=df.crash.mean(),
                   std=float(torch.exp(self.model.policy.log_std).mean()))
        for t in self.tracks:  # NaN when no episode of that track ended in this rollout
            d = df[df.track == t]
            row.update({f"{t}_episodes": len(d), f"{t}_return": d.ret.mean() if len(d) else float("nan"),
                        f"{t}_crash_rate": d.crash.mean() if len(d) else float("nan")})
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.out / "train_log.csv", index=False, float_format="%.4f")
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="override residual.ppo.seed (multi-seed runs)")
    ap.add_argument("--out", default=None, help="override residual.out")
    ap.add_argument("--holdout", default=None, choices=TRACKS,
                    help="multi-track fold: train on every benchmark track except this one (sets residual.tracks)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rc, pc = cfg["residual"], cfg["residual"]["ppo"]
    if args.seed is not None:
        pc["seed"] = args.seed
    if args.out is not None:
        rc["out"] = args.out
    if args.holdout is not None:
        rc["tracks"] = [t for t in TRACKS if t != args.holdout]
    out = pathlib.Path(rc["out"])
    out.mkdir(parents=True, exist_ok=True)
    steps = args.steps or pc["total_timesteps"]

    venv = SubprocVecEnv([_make(cfg, pc["seed"] * 1000 + i) for i in range(pc["n_envs"])], start_method="spawn")
    venv = VecNormalize(VecMonitor(venv, info_keywords=("outcome", "track")), norm_obs=False, norm_reward=True,
                        gamma=pc["gamma"])
    model = PPO("MlpPolicy", venv, n_steps=pc["n_steps"], batch_size=pc["batch_size"], n_epochs=pc["n_epochs"],
                learning_rate=pc["learning_rate"], gamma=pc["gamma"], gae_lambda=pc["gae_lambda"],
                clip_range=pc["clip_range"], ent_coef=pc["ent_coef"], seed=pc["seed"], device="cpu", verbose=0,
                policy_kwargs=dict(log_std_init=pc["log_std_init"], net_arch=list(pc["net_arch"]),
                                   activation_fn=torch.nn.Tanh))
    yaml.safe_dump(cfg, open(out / "train_config.yaml", "w"), sort_keys=False)
    model.learn(steps, callback=LogAndSave(out, pc["checkpoint_every"], rc.get("tracks") or [cfg["track"]]))
    model.save(out / "final.zip")
    venv.close()
    print(f"saved {out / 'final.zip'}")


if __name__ == "__main__":
    main()
