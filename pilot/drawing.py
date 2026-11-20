"""Shared drawing of the track: grass, legal region, white lines, walls, reference line."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

GRASS, ASPHALT, WALL = "#7fa36b", "#5b5e63", "#111111"


def _poly_path(poly):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        xy = np.asarray(ring.coords)
        verts.extend(xy)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
    return Path(verts, codes)


def draw_track(ax, track, lw_scale=1.0, show_ref=True):
    geo = track.geo
    ax.set_facecolor(GRASS)
    polys = [geo.legal] if geo.legal.geom_type == "Polygon" else list(geo.legal.geoms)
    for p in polys:
        ax.add_patch(PathPatch(_poly_path(p), facecolor=ASPHALT, edgecolor="white", lw=1.2 * lw_scale, zorder=1))
    xy = geo.wall_xy_kept
    ax.scatter(xy[:, 0], xy[:, 1], s=(0.6 * lw_scale) ** 2, marker="s", color=WALL, lw=0, zorder=2)
    if show_ref:
        ax.plot(track.ref.x, track.ref.y, color="#f1c40f", lw=0.6 * lw_scale, ls=(0, (4, 3)), zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def car_polys(geo, x, y, yaw, delta):
    return geo.body(x, y, yaw), geo.tire_patches(x, y, yaw, delta)
