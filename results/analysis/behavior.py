"""Part 2: learned behaviour, failure mechanisms, ablations and observation sensitivity (from replay.py data).

    uv run python results/analysis/behavior.py
Reads results/analysis/data/r1*, r3*; writes results/analysis/behavior/ (tables + figures).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rlana as R  # noqa: E402

D, B = R.OUT / "data", R.OUT / "behavior"
B.mkdir(parents=True, exist_ok=True)
POL = ["delta", "dual", "wall"]
COL = {"delta": "#2e7d32", "dual": "#1565c0", "wall": "#e65100", "SE": "#8e24aa", "ff_ell": "#757575", "ff_std": "#bdbdbd",
       "nominal_ell": "#9e9e9e", "nominal_std": "#e0e0e0"}
FAIL = ("wall_collision", "spin", "lateral_error", "timeout")
_TRACKS: dict = {}


def track(name):
    from pilot.closedloop import Track

    if name not in _TRACKS:
        _TRACKS[name] = Track(R.load_cfg("dual", name))
    return _TRACKS[name]


def load(name):
    sm = pd.read_csv(D / f"{name}_summary.csv")
    tr = pd.read_csv(D / f"{name}_traces.csv.gz")
    return sm, tr


def enrich(sm, tr):
    """Post-hoc wall gaps, slip angles, tire utilisation and plan acceleration for every lap."""
    out = []
    for lap_id, g in tr.groupby("lap_id"):
        r = sm.loc[sm.lap_id == lap_id].iloc[0]
        t = R.add_posthoc(g.reset_index(drop=True), track(r.track), r.mu_scale)
        tk = track(r.track)
        fe = r.ctrl not in ("delta", "nominal_std", "ff_std")
        v = tk.plan(r.a_lat, None, fe)
        aplan = v * np.gradient(v, tk.ref.ds)
        L = np.mod(t.s.values, tk.ref.length)
        t["a_plan"] = np.interp(L, np.r_[tk.ref.s, tk.ref.length], np.r_[aplan, aplan[0]])
        t["kappa_abs"] = t.kappa.abs()
        t["seg"] = np.where(t.kappa_abs > 0.05, "corner", np.where(t.a_plan < -0.5, "braking",
                                                                    np.where(t.kappa_abs < 0.01, "straight", "transition")))
        out.append(t)
    tr = pd.concat(out, ignore_index=True)
    return tr.merge(sm[["lap_id", "ctrl", "track", "a_lat", "mu_scale", "start_s", "status", "qualifying", "fail_s"]], on="lap_id")


# ---------------------------------------------------------------------------------------------- steering
def steering(tr):
    rows = []
    pol = tr[tr.ctrl.isin(POL)]
    for (ctrl, trk), g in pol.groupby(["ctrl", "track"]):
        st = g[g.seg == "straight"]
        tight = st[np.minimum(st.clear_left, st.clear_right) < 0.4]
        # side of the closer wall on tight straights: +1 left, -1 right; residual sign that would move toward it
        side = np.sign(tight.clear_right - tight.clear_left)  # +1: left wall closer
        rows.append(dict(ctrl=ctrl, track=trk, straight_res_mean=st.delta_res.mean(), straight_res_std=st.delta_res.std(),
                         straight_mpc_mean=st.delta_mpc.mean(), straight_ey_mean=st.e_y.mean(),
                         corner_res_mean_abs=g[g.seg == "corner"].delta_res.abs().mean(),
                         tight_n=len(tight), tight_res_mean=tight.delta_res.mean(), tight_ey_mean=tight.e_y.mean(),
                         tight_res_toward_closer_wall=float((np.sign(tight.delta_res) == side).mean()) if len(tight) else np.nan,
                         tight_gap_min_p05=tight.gap_min.quantile(0.05) if len(tight) else np.nan,
                         res_mpc_corr_straight=st[["delta_res", "delta_mpc"]].corr().iloc[0, 1]))
    sb = pd.DataFrame(rows)
    sb.to_csv(B / "steering_bias.csv", index=False, float_format="%.4f")

    # standardized linear regression of delta_res on state features, per policy (all tracks, all laps)
    feats = ["kappa", "e_y", "e_psi", "slip", "yaw_err", "v_ratio", "gap_left", "gap_right", "util_r"]
    reg = []
    for ctrl in POL:
        g = pol[pol.ctrl == ctrl].dropna(subset=["util_r"]).copy()
        g["yaw_err"] = g.yaw_rate - g.v * g.kappa
        g["v_ratio"] = g.v / g.v_plan
        X = g[feats].values
        mu_, sd = X.mean(0), X.std(0) + 1e-12
        Xs = np.c_[np.ones(len(g)), (X - mu_) / sd]
        for y in ("delta_res", "a_res"):
            if y == "a_res" and ctrl == "delta":
                continue
            beta_, *_ = np.linalg.lstsq(Xs, g[y].values, rcond=None)
            pred = Xs @ beta_
            r2 = 1 - np.var(g[y].values - pred) / np.var(g[y].values)
            reg.append(dict(ctrl=ctrl, target=y, r2=r2, intercept=beta_[0], **{f: b for f, b in zip(feats, beta_[1:])}))
    rg = pd.DataFrame(reg)
    rg.to_csv(B / "residual_regression.csv", index=False, float_format="%.4f")

    # figure: residual vs curvature (signed) and vs e_y on straights
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(-0.6, 0.6, 25)
    for ctrl in POL:
        g = pol[pol.ctrl == ctrl]
        c = pd.cut(g.kappa, bins)
        m = g.groupby(c, observed=True).agg(k=("kappa", "mean"), r=("delta_res", "mean"), m=("delta_mpc", "mean"))
        ax[0].plot(m.k, m.r, "-o", ms=3, color=COL[ctrl], label=f"{R.LABEL[ctrl]} δ_res")
        st = g[g.seg == "straight"]
        c2 = pd.cut(st.e_y, np.linspace(-0.2, 0.2, 17))
        m2 = st.groupby(c2, observed=True).agg(e=("e_y", "mean"), r=("delta_res", "mean"), m=("delta_mpc", "mean"))
        ax[1].plot(m2.e, m2.r, "-o", ms=3, color=COL[ctrl], label=f"{R.LABEL[ctrl]} δ_res")
        ax[1].plot(m2.e, m2.m, ":", color=COL[ctrl], lw=1)
    ax[0].set_xlabel("reference curvature κ [1/m]")
    ax[0].set_ylabel("mean steering residual [rad]")
    ax[0].set_title("Residual vs curvature (all laps, 4 tracks)", fontsize=9)
    ax[1].set_xlabel("lateral error e_y [m] (straights, |κ|<0.01)")
    ax[1].set_title("Straights: residual (solid) and MPC command (dotted) vs e_y", fontsize=9)
    for a in ax:
        a.axhline(0, color="k", lw=0.5)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(B / "fig_steering_residual.png", dpi=130)
    plt.close(fig)
    return sb, rg


# ------------------------------------------------------------------------------------------ longitudinal
def longitudinal(tr, sm):
    rows = []
    for (ctrl, trk), g in tr[tr.ctrl.isin(POL + ["SE", "ff_ell"])].groupby(["ctrl", "track"]):
        g = g[g.v_plan > 0.5]
        for seg, gg in g.groupby("seg"):
            rows.append(dict(ctrl=ctrl, track=trk, seg=seg, frac=len(gg) / len(g), v_ratio=(gg.v / gg.v_plan).mean(),
                             a_res=gg.a_res.mean() if "a_res" in gg and gg.a_res.notna().any() else np.nan,
                             abs_slip=gg.slip.abs().mean(), util_r_p95=gg.util_r.quantile(0.95),
                             util_f_p95=gg.util_f.quantile(0.95)))
    lg = pd.DataFrame(rows)
    lg.to_csv(B / "longitudinal_by_segment.csv", index=False, float_format="%.4f")
    # lap-time ratio of qualifying sweep laps by a_lat
    q = sm[sm.qualifying].groupby(["ctrl", "a_lat"]).lap_time_ratio.agg(["mean", "min", "max", "count"]).reset_index()
    q.to_csv(B / "lap_time_ratio_by_alat.csv", index=False, float_format="%.4f")

    # figure: v/v_plan and a_res along s for one Austin cell (a_lat 6, mu 0.8, start 0)
    cell = tr[(tr.track == "Austin") & (tr.a_lat == 6.0) & (tr.mu_scale == 0.8) & (tr.start_s == 0.0)]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ctrl in ("dual", "wall", "SE", "delta"):
        g = cell[cell.ctrl == ctrl].sort_values("progress")
        if g.empty:
            continue
        st = g.status.iloc[0]
        ax[0].plot(g.progress, g.v / g.v_plan, color=COL[ctrl], lw=0.9, label=f"{R.LABEL[ctrl]} ({st})")
        if ctrl in ("dual", "wall"):
            ax[1].plot(g.progress, g.a_res, color=COL[ctrl], lw=0.9, label=R.LABEL[ctrl])
        ax[2].plot(g.progress, g.slip, color=COL[ctrl], lw=0.9, label=R.LABEL[ctrl])
    g = cell[cell.ctrl == "dual"].sort_values("progress")
    ax2 = ax[1].twinx()
    ax2.plot(g.progress, g.a_plan, color="k", lw=0.6, alpha=0.5)
    ax2.set_ylabel("plan accel [m/s²] (grey)", fontsize=8)
    ax[0].axhline(1, color="k", lw=0.5)
    ax[0].set_ylabel("v / v_plan")
    ax[1].set_ylabel("a_res [m/s²]")
    ax[2].set_ylabel("sideslip β [rad]")
    ax[2].set_xlabel("progress [m] (Austin, a_lat 6, μ/μ₀ 0.8, start s = 0)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(B / "fig_speed_profile_austin.png", dpi=130)
    plt.close(fig)
    return lg, q


def low_alat():
    sm = pd.read_csv(D / "r1low_summary.csv")
    tr = pd.read_csv(D / "r1low_traces.csv.gz")
    rows = []
    for r in sm.itertuples():
        g = tr[tr.lap_id == r.lap_id]
        rows.append(dict(ctrl=r.ctrl, a_lat=r.a_lat, mu=r.mu_scale, status=r.status, lap_time_ratio=r.lap_time_ratio,
                         mean_a_res=g.a_res.mean(), mean_v_ratio=(g.v / g.v_plan).mean()))
    t = pd.DataFrame(rows)
    t.to_csv(B / "low_alat_slowdown.csv", index=False, float_format="%.4f")
    return t


# -------------------------------------------------------------------------------------------- failures
def failures(tr, sm):
    """Classify each failed lap from the last 1.5 s before the failure (explicit rules, see rl_analysis.md)."""
    rows = []
    fs = sm[sm.status.isin(FAIL)]
    for r in fs.itertuples():
        g = tr[tr.lap_id == r.lap_id].sort_values("t")
        w = g[g.t >= g.t.max() - 1.5]
        last = g.iloc[-1]
        straight = w.kappa_abs.max() < 0.02
        sat_r, sat_f = w.util_r.max() > 0.95, w.util_f.max() > 0.95
        fast = (w.v / w.v_plan).max() > 1.03
        if r.status == "spin":
            cls = "rear saturation (spin)" if sat_r else "spin without rear saturation"
        elif r.status == "wall_collision" and max(w.util_f.max(), w.util_r.max()) < 0.5:
            cls = "low grip use: clearance / steering offset"
        elif sat_f and not sat_r:
            cls = "corner: front saturation (understeer)"
        elif sat_r:
            cls = "corner: rear saturation"
        else:
            cls = "corner: insufficient correction (grip left)"
        rows.append(dict(lap_id=r.lap_id, ctrl=r.ctrl, track=r.track, a_lat=r.a_lat, mu=r.mu_scale, start_s=r.start_s,
                         status=r.status, fail_s=r.fail_s, cls=cls, excess_speed=fast,
                         straight=straight, max_util_f=w.util_f.max(), max_util_r=w.util_r.max(),
                         max_abs_slip=w.slip.abs().max(), ey_last=last.e_y, res_mean_last=w.delta_res.mean() if "delta_res" in w else np.nan,
                         mpc_mean_last=w.delta_mpc.mean(), clear_min=np.minimum(w.clear_left, w.clear_right).min(),
                         v_ratio_max=(w.v / w.v_plan).max()))
    fc = pd.DataFrame(rows)
    fc.to_csv(B / "failure_classification.csv", index=False, float_format="%.4f")
    summ = fc.groupby(["ctrl", "cls"]).size().unstack(fill_value=0)
    summ.to_csv(B / "failure_classes.csv")

    # location clusters: fail_s rounded to 10 m
    fc["loc"] = (fc.fail_s / 10).round() * 10
    loc = fc.groupby(["track", "ctrl", "loc"]).size().reset_index(name="n").sort_values(["track", "ctrl", "n"],
                                                                                        ascending=[True, True, False])
    loc.to_csv(B / "failure_locations.csv", index=False)

    # figure: failure locations per track with wall clearance along s
    fig, axes = plt.subplots(4, 1, figsize=(11, 10))
    order = ["SE", "delta", "dual", "wall"]
    for ax, trk in zip(axes, R.TRACKS):
        tk = track(trk)
        s_ext, left, right = tk.wall_side_clearance(1.5)
        ax.fill_between(s_ext, 0, np.minimum(left, right), color="#cfd8dc", step="mid",
                        label="min side clearance of the reference line [m]" if trk == "Austin" else None)
        ax.set_ylim(0, 1.6)
        ax.set_ylabel("clearance [m]", fontsize=8)
        ax3 = ax.twinx()
        for i, ctrl in enumerate(order):
            g = fc[(fc.track == trk) & (fc.ctrl == ctrl)]
            mk = {"wall_collision": "x", "spin": "o", "lateral_error": "^", "timeout": "s"}
            for st, gg in g.groupby("status"):
                ax3.scatter(gg.fail_s, np.full(len(gg), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(gg)),
                            marker=mk[st], color=COL[ctrl], s=22, label=f"{R.LABEL[ctrl]} {st}" if trk == "Austin" else None)
        ax3.set_yticks(range(len(order)), [R.LABEL[c] for c in order], fontsize=7)
        ax3.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(f"{trk}: failure location of every failed sweep lap", fontsize=9)
        ax.set_xlim(0, tk.ref.length)
    axes[-1].set_xlabel("arc length s [m]")
    h, l_ = [], []
    for a in fig.axes:
        hh, ll = a.get_legend_handles_labels()
        h += hh
        l_ += ll
    fig.legend(h, l_, fontsize=7, ncol=5, loc="lower center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(B / "fig_failure_locations.png", dpi=130)
    plt.close(fig)
    return fc, summ, loc


def hotspot_overlay(tr, fc):
    """Failed vs successful laps over the same segment: two most frequent RL failure spots on Spielberg/Silverstone."""
    spots = []
    for trk in ("Spielberg", "Silverstone"):
        g = fc[(fc.track == trk) & fc.ctrl.isin(["dual", "wall"])]
        for loc_, n in g.groupby((g.fail_s / 10).round() * 10).size().sort_values(ascending=False).head(2).items():
            spots.append((trk, loc_))
    fig, axes = plt.subplots(5, len(spots), figsize=(4.2 * len(spots), 10), sharex="col")
    for j, (trk, loc_) in enumerate(spots):
        lo, hi = loc_ - 40, loc_ + 15
        cell = tr[(tr.track == trk) & tr.ctrl.isin(["dual", "wall", "SE"])]
        for (lap_id, ctrl), g in cell.groupby(["lap_id", "ctrl"]):
            gg = g[(np.mod(g.s, track(trk).ref.length) >= lo) & (np.mod(g.s, track(trk).ref.length) <= hi)].sort_values("t")
            if gg.empty or gg.a_lat.iloc[0] != 6.0:
                continue
            fs = gg.fail_s.iloc[0]
            failed_here = gg.status.iloc[0] in FAIL and pd.notna(fs) and lo <= fs <= hi
            ls = "--" if failed_here else "-"
            x = np.mod(gg.s, track(trk).ref.length).values
            gg = gg.copy()
            brk = np.r_[False, np.diff(x) < 0]  # lap wrap: break the line
            gg.loc[brk, ["e_y", "gap_min", "delta_mpc", "v"]] = np.nan
            x = np.where(brk, np.nan, x)
            axes[0, j].plot(x, gg.e_y, ls, color=COL[ctrl], lw=0.8)
            axes[1, j].plot(x, gg.gap_min, ls, color=COL[ctrl], lw=0.8)
            axes[2, j].plot(x, gg.delta_mpc, ls, color=COL[ctrl], lw=0.8)
            if ctrl != "SE":
                gg.loc[brk, "delta_res"] = np.nan
                axes[3, j].plot(x, gg.delta_res, ls, color=COL[ctrl], lw=0.8)
            axes[4, j].plot(x, gg.v / gg.v_plan, ls, color=COL[ctrl], lw=0.8)
        axes[0, j].set_title(f"{trk}, s ≈ {loc_:.0f} m (a_lat 6, all μ/starts)", fontsize=9)
        axes[-1, j].set_xlabel("s [m]")
        for a in axes[:, j]:
            a.axvline(loc_, color="r", lw=0.5)
            a.grid(alpha=0.3)
    for a, lab in zip(axes[:, 0], ("e_y [m]", "min body-wall gap [m]", "δ_MPC [rad]", "δ_res [rad]", "v / v_plan")):
        a.set_ylabel(lab)
    from matplotlib.lines import Line2D

    fig.legend([Line2D([], [], color=COL[c]) for c in ("dual", "wall", "SE")] + [Line2D([], [], color="k"), Line2D([], [], color="k", ls="--")],
               ["RL δa", "RL δa+wall", "S-E", "passed this segment", "failed in this segment"], ncol=5, fontsize=8, loc="lower center")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(B / "fig_hotspots.png", dpi=120)
    plt.close(fig)
    return spots


# ------------------------------------------------------------------------------------- ablations + RL vs SE
def ablations():
    base = pd.read_csv(D / "r1_summary.csv")
    base = base[base.ctrl.isin(POL)].assign(ablation="none")
    ab = pd.read_csv(D / "r3_summary.csv")
    allr = pd.concat([base, ab], ignore_index=True)
    allr["status"] = allr.status.astype(str)
    t = allr.groupby(["ctrl", "ablation", "track"]).status.value_counts().unstack(fill_value=0)
    t = t.reindex(columns=[c for c in R.STATUSES if c in t.columns])
    t.to_csv(B / "ablations_by_track.csv")
    tot = allr.groupby(["ctrl", "ablation"]).agg(qual=("qualifying", "sum"), n=("qualifying", "size"),
                                                wall=("status", lambda s: (s == "wall_collision").sum()),
                                                spin=("status", lambda s: (s == "spin").sum()),
                                                lat=("status", lambda s: (s == "lateral_error").sum()),
                                                slow=("status", lambda s: (s == "speed_tolerance").sum()),
                                                ratio=("lap_time_ratio", "mean"))
    tot.to_csv(B / "ablations_total.csv", float_format="%.4f")
    return t, tot


def rl_vs_se(sm):
    piv = sm[sm.ctrl.isin(POL + ["SE"])].pivot_table(index=["track", "a_lat", "mu_scale", "start_s"], columns="ctrl",
                                                     values="status", aggfunc="first")
    piv.to_csv(B / "cell_outcomes.csv")
    rows = []
    for p in POL:
        rows.append(dict(policy=p, both=int(((piv[p] == "qualifying") & (piv.SE == "qualifying")).sum()),
                         rl_only=int(((piv[p] == "qualifying") & (piv.SE != "qualifying")).sum()),
                         se_only=int(((piv[p] != "qualifying") & (piv.SE == "qualifying")).sum()),
                         neither=int(((piv[p] != "qualifying") & (piv.SE != "qualifying")).sum())))
    t = pd.DataFrame(rows)
    by_mu = []
    for (trk, mu), g in piv.reset_index().groupby(["track", "mu_scale"]):
        by_mu.append(dict(track=trk, mu=mu, **{c: int((g[c] == "qualifying").sum()) for c in POL + ["SE"]}))
    pd.DataFrame(by_mu).to_csv(B / "qual_by_track_mu.csv", index=False)
    t.to_csv(B / "rl_vs_se_cells.csv", index=False)
    return t


def mismatch_signature(tr):
    """Does δ_res track the dynamics mismatch the slip-aware MPC models? Correlate δ_res with the difference
    between the measured yaw rate and the kinematic yaw rate the MPC assumes, binned by tire utilisation."""
    rows = []
    for ctrl in POL:
        g = tr[(tr.ctrl == ctrl)].dropna(subset=["util_r"]).copy()
        L = 0.15875 + 0.17145
        g["r_kin"] = g.v * np.tan(g.delta) / L * np.cos(np.arctan(0.17145 / L * np.tan(g.delta)))
        g["r_def"] = g.yaw_rate - g.r_kin  # <0 for understeer when turning left
        g["util"] = np.maximum(g.util_f, g.util_r)
        for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
            gg = g[(g.util >= lo) & (g.util < hi)]
            if len(gg) < 50:
                continue
            rows.append(dict(ctrl=ctrl, util_bin=f"{lo}-{hi}", n=len(gg), frac=len(gg) / len(g),
                             corr_res_rdef=gg[["delta_res", "r_def"]].corr().iloc[0, 1],
                             corr_res_slip=gg[["delta_res", "slip"]].corr().iloc[0, 1],
                             corr_res_kappa=gg[["delta_res", "kappa"]].corr().iloc[0, 1],
                             mean_abs_res=gg.delta_res.abs().mean()))
    t = pd.DataFrame(rows)
    t.to_csv(B / "mismatch_signature.csv", index=False, float_format="%.4f")
    return t


# ----------------------------------------------------------------------------------- observation sensitivity
OBS_NAMES = (["e_y", "e_psi", "v", "v-vp", "r", "beta", "r-vκ", "delta", "d_mpc", "prev_res"]
             + [f"κ@{t}" for t in (0, .15, .3, .5, .75, 1)] + [f"ay_plan@{t}" for t in (0, .15, .3, .5, .75, 1)])
DUAL_NAMES = ["prev_ares"] + [f"a_plan@{t}" for t in (0, .15, .3, .5, .75, 1)]
WALL_NAMES = [f"gapL@{t}" for t in (0, .25, .5, 1)] + [f"gapR@{t}" for t in (0, .25, .5, 1)]


def group(n):
    if n.startswith("gap"):
        return "wall gaps"
    if n.startswith(("κ@", "ay_plan@")):
        return "curvature/plan preview"
    if n.startswith("a_plan@"):
        return "accel-plan preview"
    if n in ("prev_res", "prev_ares"):
        return "previous residuals"
    if n in ("e_y", "e_psi", "r-vκ"):
        return "tracking errors"
    if n in ("d_mpc", "delta"):
        return "steering (MPC, actual)"
    return "vehicle state (v, r, β)"


def sensitivity():
    import torch

    rows = []
    sm = pd.read_csv(D / "r1_summary.csv")
    z = np.load(D / "r1_obs.npz")
    for ctrl in POL:
        names = OBS_NAMES + (DUAL_NAMES if ctrl != "delta" else []) + (WALL_NAMES if ctrl == "wall" else [])
        ids = sm[sm.ctrl == ctrl].lap_id.values
        O = np.concatenate([z[f"obs_{i}"] for i in ids if f"obs_{i}" in z])
        model = R._model(str(R.EXPS[ctrl][1] / R.SELECTED))
        x = torch.tensor(O, requires_grad=True)
        mean = model.policy.get_distribution(x).distribution.mean
        for k in range(mean.shape[1]):
            g, = torch.autograd.grad(mean[:, k].sum(), x, retain_graph=True)
            sens = (g.abs().mean(0) * x.detach().std(0)).numpy()  # |∂π/∂o| x std(o): output change per 1-σ input change
            for n, s_ in zip(names, sens):
                rows.append(dict(ctrl=ctrl, output="steer" if k == 0 else "accel", feature=n, group=group(n), sens=s_))
        # permutation importance (shuffle a feature group across time, mean |Δ action|)
        rng = np.random.default_rng(0)
        with torch.no_grad():
            base = model.policy.get_distribution(torch.tensor(O)).distribution.mean.numpy()
            for gname in sorted({group(n) for n in names}):
                idx = [i for i, n in enumerate(names) if group(n) == gname]
                Op = O.copy()
                Op[:, idx] = Op[rng.permutation(len(O))][:, idx]
                pm = model.policy.get_distribution(torch.tensor(Op)).distribution.mean.numpy()
                for k in range(base.shape[1]):
                    rows.append(dict(ctrl=ctrl, output=("steer" if k == 0 else "accel") + "_perm", feature=gname,
                                     group=gname, sens=float(np.abs(pm[:, k] - base[:, k]).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(B / "obs_sensitivity.csv", index=False, float_format="%.5f")
    g = t[~t.output.str.endswith("_perm")].groupby(["ctrl", "output", "group"]).sens.sum().unstack(0)
    g.to_csv(B / "obs_sensitivity_groups.csv", float_format="%.4f")
    gp = t[t.output.str.endswith("_perm")].pivot_table(index=["output", "group"], columns="ctrl", values="sens")
    gp.to_csv(B / "obs_permutation_groups.csv", float_format="%.4f")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, out in zip(ax, ("steer_perm", "accel_perm")):
        gg = gp.loc[out] if out in gp.index.get_level_values(0) else None
        if gg is None:
            continue
        gg = gg.sort_values("wall" if "wall" in gg else gg.columns[0])
        y = np.arange(len(gg))
        for i, c in enumerate([c for c in POL if c in gg]):
            a.barh(y + (i - 1) * 0.27, gg[c].values, 0.27, color=COL[c], label=R.LABEL[c])
        a.set_yticks(y, gg.index, fontsize=8)
        a.set_xlabel("mean |Δ action| when the group is shuffled (normalised action units)")
        a.set_title(f"{'steering' if out.startswith('steer') else 'acceleration'} output", fontsize=9)
        a.legend(fontsize=7)
        a.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(B / "fig_obs_importance.png", dpi=130)
    plt.close(fig)
    return g, gp


if __name__ == "__main__":
    sm, tr = load("r1")
    tr = enrich(sm, tr)
    tr.to_csv(D / "r1_traces_enriched.csv.gz", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    sb, rg = steering(tr)
    print(sb.to_string(index=False))
    print(rg.to_string(index=False))
    lg, q = longitudinal(tr, sm)
    print(lg.pivot_table(index=["ctrl", "seg"], columns="track", values="v_ratio").to_string())
    print(lg.pivot_table(index=["ctrl", "seg"], columns="track", values="a_res").to_string())
    print(q.to_string(index=False))
    print(low_alat().to_string(index=False))
    fc, summ, loc = failures(tr, sm)
    print(summ.to_string())
    print(loc.groupby(["track", "ctrl"]).head(3).to_string(index=False))
    print(hotspot_overlay(tr, fc))
    print(rl_vs_se(sm).to_string(index=False))
    print(mismatch_signature(tr).to_string(index=False))
    t, tot = ablations()
    print(tot.to_string())
    g, gp = sensitivity()
    print(g.to_string())
    print(gp.to_string())
