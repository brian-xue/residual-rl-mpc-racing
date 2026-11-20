"""Side-by-side GIF of both controllers in one scenario (follow camera, real geometry).

Body = controller colour, tire contact patches green when on the legal region (white lines
included) and red when off it, black squares = physical walls, yellow dashed = reference line.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon

from .drawing import draw_track

COLORS = {"nominal": "#e74c3c", "ff": "#3498db", "reference": "#9b59b6"}
NAMES = {"nominal": "Nominal MPC", "ff": "MPC + K v² κ", "reference": "Dynamic-model MPC (true μ)"}


def render(track, cfg, traces, summs, path, frame_dt=0.2, fps=20, window_m=9.0, trail_s=3.0, dpi=72):
    """``traces``/``summs`` are dicts keyed by controller ("nominal", "ff", "reference"); one panel per key,
    in dict order."""
    gcfg = cfg.get("gif", {})
    frame_dt, fps, window_m = gcfg.get("frame_dt_s", frame_dt), gcfg.get("fps", fps), gcfg.get("window_m", window_m)
    geo = track.geo
    t_end = max(float(tr.t.iloc[-1]) for tr in traces.values())
    times = np.arange(0.0, t_end + frame_dt, frame_dt)
    keys = list(traces)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 5.6))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.09, wspace=0.03)
    arts = {}
    for ax, ctrl in zip(axes, keys):
        draw_track(ax, track, lw_scale=2.5)
        trail, = ax.plot([], [], color=COLORS[ctrl], lw=2, alpha=0.8, zorder=6)
        body = MplPolygon(np.zeros((4, 2)), closed=True, fc=COLORS[ctrl], ec="k", lw=1, zorder=7)
        ax.add_patch(body)
        tires = [MplPolygon(np.zeros((4, 2)), closed=True, fc="lime", ec="k", lw=0.8, zorder=8) for _ in range(4)]
        for p in tires:
            ax.add_patch(p)
        title = ax.set_title("", fontsize=10, color=COLORS[ctrl], fontweight="bold")
        info = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, family="monospace",
                       color="w", bbox=dict(fc="black", alpha=0.55, lw=0), zorder=10)
        banner = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center", fontsize=18,
                         color="w", fontweight="bold", bbox=dict(fc="#c0392b", alpha=0.85, lw=0), zorder=11)
        banner.set_visible(False)
        timer = ax.text(0.98, 0.03, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=20,
                        family="monospace", fontweight="bold", color="w",
                        bbox=dict(fc="black", alpha=0.7, lw=0, boxstyle="round,pad=0.3"), zorder=12)
        arts[ctrl] = (ax, trail, body, tires, title, info, banner, timer)
    sm = summs[keys[0]]
    same_speed = len({summs[c]["a_lat"] for c in keys}) == 1
    speed_txt = f"speed plan a_lat = {sm['a_lat']} m/s² | " if same_speed else "different speed plans (see titles) | "
    fig.suptitle(f"{cfg['track']} | {speed_txt}μ/μ₀ = {sm['mu_scale']} | "
                 f"start s = {sm['start_s']:.0f} m | {fps * frame_dt:.0f}× real time\n"
                 "tires: green = on legal track (incl. white lines), red = off  |  black = walls (none on curves)",
                 fontsize=10)
    lt = {c: summs[c]["lap_time"] if summs[c]["outcome"] == "completed" else None for c in summs}
    gap_txt = fig.text(0.5, 0.015, "", ha="center", va="bottom", fontsize=14, fontweight="bold", color="w",
                       bbox=dict(fc="#2c3e50", alpha=0.9, lw=0, boxstyle="round,pad=0.4"), zorder=20)
    gap_txt.set_visible(False)

    def update(k):
        t = times[k]
        out = []
        for ctrl, (ax, trail, body, tires, title, info, banner, timer) in arts.items():
            tr = traces[ctrl]
            j = min(int(np.searchsorted(tr.t.values, t)), len(tr) - 1)
            r = tr.iloc[j]
            b, ps = geo.body(r.x, r.y, r.yaw), geo.tire_patches(r.x, r.y, r.yaw, r.delta)
            off, wall, tin = geo.check(r.x, r.y, r.yaw, r.delta)
            body.set_xy(np.asarray(b.exterior.coords))
            for p, poly, ok in zip(tires, ps, tin):
                p.set_xy(np.asarray(poly.exterior.coords))
                p.set_facecolor("lime" if ok else "red")
            m = (tr.t.values >= t - trail_s) & (tr.t.values <= t)
            trail.set_data(tr.x.values[m], tr.y.values[m])
            h = window_m / 2
            ax.set_xlim(r.x - h, r.x + h)
            ax.set_ylim(r.y - h * 0.8, r.y + h * 0.8)
            s_ = summs[ctrl]
            title.set_text(f"{NAMES[ctrl]} — {s_['status']}" + ("" if same_speed else f"  (a_lat {s_['a_lat']})"))
            if lt[ctrl] is not None and t >= lt[ctrl]:
                timer.set_text(f"LAP {lt[ctrl]:6.2f} s")
                timer.get_bbox_patch().set_facecolor("#27ae60" if s_["qualifying"] else "#c0392b")
            else:
                t_show = min(t, float(tr.t.iloc[-1]))
                timer.set_text(f"TIME {t_show:6.2f} s")
                timer.get_bbox_patch().set_facecolor("black")
            info.set_text(f"t {r.t:5.1f} s  v {r.v:4.1f} m/s\ne_y {r.e_y:+.2f} m  prog {100 * r.progress / track.ref.length:3.0f}%\n"
                          f"tires on track {int(np.sum(tin))}/4")
            ended = (lt[ctrl] is not None and t >= lt[ctrl]) or (j == len(tr) - 1 and t > float(tr.t.iloc[-1]))
            if ended and s_["outcome"] == "wall_collision":
                banner.set_text("WALL COLLISION")
                banner.set_visible(True)
            elif ended and s_["outcome"] == "completed":
                banner.set_text("QUALIFYING LAP" if s_["qualifying"] else f"LAP INVALID ({s_['status']})")
                banner.get_bbox_patch().set_facecolor("#27ae60" if s_["qualifying"] else "#c0392b")
                banner.set_visible(True)
            elif off:
                banner.set_text("OFF TRACK")
                banner.set_visible(True)
            else:
                banner.set_visible(False)
            out += [trail, body, *tires, title, info, banner, timer]
        if all(v is not None for v in lt.values()) and t >= max(lt.values()):
            if set(keys) == {"nominal", "ff"}:
                d = lt["nominal"] - lt["ff"]
                who = NAMES["ff"] if d > 0 else NAMES["nominal"]
                gap_txt.set_text(f"{who} faster by {abs(d):.2f} s per lap ({100 * abs(d) / max(lt.values()):.1f}%)")
            else:
                order = sorted(keys, key=lambda c: lt[c])
                best = lt[order[0]]
                gap_txt.set_text("  |  ".join(f"{NAMES[c]} {lt[c]:.2f} s" + ("" if c == order[0] else f" (+{lt[c] - best:.2f})")
                                              for c in order))
            gap_txt.set_visible(True)
        return out

    # hold the last frame for 2 s
    frames = list(range(len(times))) + [len(times) - 1] * int(2 * fps)
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    path = pathlib.Path(path)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return dict(path=str(path), frames=len(frames), size_mb=round(path.stat().st_size / 1e6, 2))
