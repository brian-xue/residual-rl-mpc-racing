"""SlipMPC + RL, Stage 5 (docs/slip_rl_plan.md): diagnostics, lapping only (no training, no selection).

    uv run python results/analysis/slip_rl/stage5.py lap        # re-lap the 144-lap grid with traces (~40 min)
    uv run python results/analysis/slip_rl/stage5.py analyse    # tables from the per-lap file + the protocol reports

Controllers: S-E, S-E at 0.97 x speed, and the SELECTED checkpoint of every R and P' run (protocol_selected.txt).
``lap`` re-runs each one on the reporting grid with logging on and checks that every outcome matches the protocol
report (the new logging is passive). Per lap it keeps aggregates; the 20 Hz traces go to
results_st_pacejka/slip_rl/stage5/traces/ (not committed).

Questions (plan, Stage 5):
  1. the 8 + 9 feasibility cells (results/analysis/followup/feasibility): what R and P' do there;
  2. spin and wall locations;
  3. P': the speed scale it picks, against the true mu/mu0 and the curvature;
  4. R: residual size against slack activity, i.e. does R drive past SlipMPC's slip limits?
     Measured two ways: the MPC's own soft-limit slack (its predicted violation) and the plant's slip angles at the
     control instants, alpha_f = delta - beta - lf r / v and alpha_r = -beta + lr r / v, against the limits.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pilot.closedloop import Scenario, qualifies, run_many  # noqa: E402
from pilot.protocol import _hotspot, report_cells  # noqa: E402
from pilot.residual_env import ResidualPolicy  # noqa: E402
from pilot.sim import vehicle_params  # noqa: E402

D = ROOT / "results_st_pacejka/slip_rl"
OUT = pathlib.Path(__file__).resolve().parent / "stage5"
TRACES = D / "stage5/traces"
FEAS = ROOT / "results/analysis/followup/feasibility"
EPS = 1e-3  # slack "active" threshold [rad-equivalent]
P = vehicle_params(1.0)


def controllers():
    base = yaml.safe_load(open(ROOT / "configs/pilot_st_pacejka_slip_res.yaml"))
    se097 = copy.deepcopy(base)
    se097["longitudinal"]["slip_mpc"]["speed_scale"] = 0.97
    out = [("SE", "SE", base, None), ("SE097", "SE097", se097, None)]
    for arm in ("R", "P"):
        for s in range(3):
            rd = D / arm / f"s{s}"
            cfg = yaml.safe_load(open(rd / "train_config.yaml"))
            ck = rd / (rd / "protocol_selected.txt").read_text().strip()
            out.append((f"{arm}/s{s}", arm, cfg, ResidualPolicy.from_rc(str(ck), cfg["residual"])))
    return out


def lap_aggregates(tr: pd.DataFrame, lim_f, lim_r):
    v = np.maximum(tr.v.to_numpy(), 1.0)
    af = tr.delta.to_numpy() - tr.slip.to_numpy() - P["lf"] * tr.yaw_rate.to_numpy() / v
    ar = -tr.slip.to_numpy() + P["lr"] * tr.yaw_rate.to_numpy() / v
    act = (tr.slack0 > EPS).to_numpy()
    out = dict(n=len(tr), slack_active=act.mean(), slack_any_active=(tr.slack_max > EPS).mean(),
               slack0_mean=np.nanmean(tr.slack0), ar_over=(np.abs(ar) > lim_r).mean(),
               af_over=(np.abs(af) > lim_f).mean(), ar_max=np.abs(ar).max(), af_max=np.abs(af).max(),
               scale_mean=tr.speed_scale.mean())
    k = np.abs(tr.kappa.to_numpy())
    out.update(scale_corner=tr.speed_scale[k > 0.05].mean(), scale_straight=tr.speed_scale[k < 0.01].mean(),
               scale_min=tr.speed_scale.min(), v_over_plan=(tr.v / tr.v_plan.clip(lower=0.5)).mean())
    if "delta_res" in tr:
        dr, ar_ = tr.delta_res.abs().to_numpy(), tr.a_res.to_numpy()
        out.update(dres_mean=dr.mean(), ares_mean=ar_.mean(), ares_abs=np.abs(ar_).mean(),
                   dres_when_active=dr[act].mean() if act.any() else np.nan,
                   dres_when_inactive=dr[~act].mean() if (~act).any() else np.nan,
                   ares_corner=ar_[k > 0.05].mean(), ares_straight=ar_[k < 0.01].mean())
    last = tr.tail(20)  # last 1 s of the lap (the failure window for a failed lap)
    vl = np.maximum(last.v.to_numpy(), 1.0)
    out.update(last_slack_active=(last.slack0 > EPS).mean(),
               last_ar_max=np.abs(-last.slip.to_numpy() + P["lr"] * last.yaw_rate.to_numpy() / vl).max(),
               last_dres=last.delta_res.abs().mean() if "delta_res" in last else np.nan,
               last_ares=last.a_res.mean() if "a_res" in last else np.nan,
               last_scale=last.speed_scale.mean())
    return out


def lap(args):
    OUT.mkdir(parents=True, exist_ok=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    cells, rows = report_cells(), []
    for name, arm, cfg, pol in controllers():
        if args.only and name not in args.only:
            continue
        sw = cfg["longitudinal"]["slip_mpc"]
        q = cfg["qualifying"]
        ref = pd.read_csv((D / "baselines" / name / "s0" if pol is None else D / name) / "protocol_report.csv")
        traces = []
        for track in sorted({c[0] for c in cells}):
            tc = copy.deepcopy(cfg)
            tc["track"] = track
            cs = [c for c in cells if c[0] == track]
            K = cfg["residual"].get("base_K", 0.0)
            jobs = [(Scenario(c[1], c[2], c[3]), K, True, None) + ((pol,) if pol else ()) for c in cs]
            for c, (sm, tr) in zip(cs, run_many(tc, jobs)):
                ok, why = qualifies(sm, q)
                r = ref[(ref.track == track) & (ref.a_lat == c[1]) & (ref.mu_scale == c[2]) & (ref.start_s == c[3])]
                assert len(r) == 1 and r.status.iloc[0] == why, (name, c, why, r.status.tolist())
                fs = sm.get("fail_s")
                rows.append(dict(run=name, variant=arm, track=track, a_lat=c[1], mu_scale=c[2], start_s=c[3],
                                 grid=c[4], status=why, qualifying=bool(ok), fail_s=fs,
                                 hotspot=_hotspot(track, fs), lap_time_ratio=sm["lap_time_ratio"],
                                 **lap_aggregates(tr, sw["alpha_f_max"], sw["alpha_r_max"])))
                keep = [k for k in ("t", "s", "v", "v_plan", "kappa", "slip", "yaw_rate", "delta", "delta_res",
                                    "a_res", "a_mpc", "speed_scale", "slack0", "slack_max", "e_y") if k in tr]
                traces.append(tr[keep].assign(track=track, a_lat=c[1], mu_scale=c[2], start_s=c[3]))
        pd.concat(traces).to_csv(TRACES / f"{name.replace('/', '_')}.csv.gz", index=False, float_format="%.5g")
        print(name, "laps", sum(r["run"] == name for r in rows), "outcomes match the protocol report", flush=True)
    f = OUT / "laps.csv"
    new = pd.DataFrame(rows)
    if f.exists() and args.only:
        old = pd.read_csv(f)
        new = pd.concat([old[~old.run.isin(new.run.unique())], new])
    new.to_csv(f, index=False, float_format="%.5g")


def analyse(args):
    d = pd.read_csv(OUT / "laps.csv")
    pd.set_option("display.width", 200)
    out = []

    def show(title, tab):
        s = f"\n## {title}\n{tab.to_string(float_format=lambda x: f'{x:.3f}')}"
        print(s)
        out.append(s)

    # 1. feasibility cells
    key = ["track", "a_lat", "mu_scale", "start_s"]
    for lab, f in (("recovers (S-E fixed, wall policy failed)", "slipmpc_recovers.csv"),
                   ("loses (wall policy fixed, S-E failed)", "slipmpc_loses.csv")):
        c = pd.read_csv(FEAS / f)[key]
        m = d.merge(c, on=key)
        show(f"1. feasibility cells, S-E {lab}: qualifying laps of {len(c)}",
             m.groupby("run").qualifying.sum().to_frame().T)
    # 2. failure locations
    fl = d[d.status.isin(["spin", "wall_collision"])].copy()
    fl["where"] = np.where(fl.hotspot.fillna("").isin(["", "other"]), fl.track.str[:3] + "-" +
                           (fl.fail_s // 25 * 25).astype(int).astype(str), fl.hotspot)
    show("2. spins and walls by place (all runs; 25 m bins outside the named hotspots)",
         fl.pivot_table(index="where", columns="variant", values="status", aggfunc="size", fill_value=0)
         .assign(tot=lambda t: t.sum(1)).sort_values("tot", ascending=False).head(15))
    show("2b. spins and walls by mu/mu0 and a_lat (summed over runs of the variant)",
         fl.pivot_table(index=["mu_scale"], columns=["variant", "status"], values="a_lat", aggfunc="size",
                        fill_value=0))
    # 3. P' speed scale
    p = d[d.variant.isin(["P", "SE097"])]
    show("3. P': mean speed scale by mu/mu0 (and per run)",
         p.pivot_table(index="mu_scale", columns="run", values="scale_mean"))
    show("3b. P': scale in corners (|kappa| > 0.05) vs straights (|kappa| < 0.01), min scale",
         p.groupby("run")[["scale_mean", "scale_corner", "scale_straight", "scale_min"]].mean())
    show("3c. P': mean scale by a_lat", p.pivot_table(index="a_lat", columns="run", values="scale_mean"))
    # 4. slack and slip angles
    cols = ["slack_active", "slack_any_active", "slack0_mean", "ar_over", "af_over", "ar_max", "af_max",
            "v_over_plan"]
    show("4. soft-limit activity per run (mean over the 144 laps): stage-1 slack active, any stage active, "
         "plant |alpha_r| > 0.12, |alpha_f| > 0.25 (fraction of control instants), max |alpha|",
         d.groupby("run")[cols].mean())
    comp = d[d.status == "qualifying"].merge(d[(d.run == "SE") & (d.status == "qualifying")][key], on=key)
    show(f"4b. the same on the cells S-E qualifies ({comp[comp.run == 'SE'].shape[0]} laps), so the comparison is "
         "on equal laps", comp.groupby("run")[cols].mean())
    r = d[d.variant == "R"]
    show("4c. R: residual size, and |delta_res| when the stage-1 slack is active vs not",
         r.groupby("run")[["dres_mean", "dres_when_active", "dres_when_inactive", "ares_mean", "ares_abs",
                           "ares_corner", "ares_straight"]].mean())
    show("4d. last 1 s before a spin or wall: slack active fraction, max |alpha_r|, |delta_res|, a_res, scale",
         fl.groupby(["variant", "status"])[["last_slack_active", "last_ar_max", "last_dres", "last_ares",
                                            "last_scale"]].mean())
    (OUT / "analysis.txt").write_text("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lap", "analyse"])
    ap.add_argument("--only", nargs="*", help="lap: subset of controllers (SE, SE097, R/s0, ...)")
    args = ap.parse_args()
    {"lap": lap, "analyse": analyse}[args.what](args)
