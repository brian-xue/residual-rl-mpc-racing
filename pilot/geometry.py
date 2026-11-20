"""Track-limit and wall geometry, checked every physics step (replaces Gym-Khana's
reference-point Frenet boundary termination).

* Legal track region: union of disks of radius w(s) swept along the closed track centerline
  from ``<track>_centerline.csv`` (columns x, y, w_tr_right, w_tr_left). The CSV width is taken
  as the outer edge of the painted white line, so the lines are part of the legal region.
* Physical walls: occupied cells of the simulator's own occupancy map (``env.track.occupancy_map``,
  pixel <= 128 after Gym-Khana's thresholding), each cell an axis-aligned square. With
  ``walls.remove_on_curves`` the cells whose nearest centerline point lies in a curve (|kappa| above a
  threshold, extended along the track before and after) are deleted, leaving open run-off in the
  corners and walls only along the straights. This only affects the validity checks; the simulator's
  laser scan still sees the original map (unused by the controllers).
* ``off_track``: none of the four tire contact patches intersects the legal region.
* ``wall_collision``: the vehicle body rectangle (Gym-Khana's ``get_vertices`` convention: centred on
  the CG, length x width) intersects any wall cell, regardless of tire positions.
"""

from __future__ import annotations

import pathlib

import numpy as np
import shapely
from gymkhana.envs.track.track_utils import find_track_dir
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def _rect(cx, cy, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([1, 1, -1, -1]) * length / 2
    dy = np.array([1, -1, -1, 1]) * width / 2
    return np.c_[cx + c * dx - s * dy, cy + s * dx + c * dy]


class TrackGeometry:
    def __init__(self, env_track, vehicle: dict, geo_cfg: dict):
        # raw CSV points (before Gym-Khana's spline resampling)
        d = find_track_dir(geo_cfg["track"])
        raw = np.loadtxt(pathlib.Path(d) / f"{geo_cfg['track']}_centerline.csv", delimiter=",", comments="#")
        xs, ys, wr, wl = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        if not np.allclose(wl, wr):
            raise NotImplementedError("asymmetric track widths not supported in the pilot")
        n = len(xs)
        # round-capped segment buffers: every point within w(s) of the centerline is legal
        pieces = [LineString([(xs[i], ys[i]), (xs[(i + 1) % n], ys[(i + 1) % n])]).buffer(min(wl[i], wl[(i + 1) % n]))
                  for i in range(n)]
        self.legal = unary_union(pieces)
        shapely.prepare(self.legal)
        self.half_width = float(wl.min())
        self.center_xy = np.c_[xs, ys]
        self.white_line_width = geo_cfg["white_line_width_m"]

        # walls from the simulator's occupancy grid (0 = occupied after Gym-Khana's thresholding)
        occ = np.asarray(env_track.occupancy_map) == 0
        res = float(env_track.spec.resolution)
        ox, oy = env_track.spec.origin[0], env_track.spec.origin[1]
        iy, ix = np.nonzero(occ)
        x0, y0 = ox + ix * res, oy + iy * res
        wc = geo_cfg.get("walls", {})
        self.n_wall_cells_map = len(x0)
        self.curve_mask = np.zeros(n, bool)
        if wc.get("remove_on_curves", False):
            from .reference import Reference  # local import: reference does not depend on geometry

            cref = Reference(xs, ys, ds=0.1, smooth_sigma_m=0.6, kappa_sigma_m=0.6)
            kap = np.abs(cref.kappa_at(cref.project_many(xs, ys)))
            seg = np.hypot(np.diff(np.r_[xs, xs[0]]), np.diff(np.r_[ys, ys[0]]))
            ext = int(np.ceil(wc["curve_extend_m"] / seg.mean()))
            curve = kap > wc["curve_kappa_threshold"]
            self.curve_mask = np.convolve(np.r_[curve[-ext:], curve, curve[:ext]].astype(float),
                                          np.ones(2 * ext + 1), "same")[ext:-ext] > 0 if ext > 0 else curve
            _, nearest = cKDTree(np.c_[xs, ys]).query(np.c_[x0 + res / 2, y0 + res / 2])
            keep = ~self.curve_mask[nearest]
            x0, y0 = x0[keep], y0[keep]
        self.wall_boxes = shapely.box(x0, y0, x0 + res, y0 + res)
        self.wall_tree = shapely.STRtree(self.wall_boxes)
        self.wall_xy_kept = np.c_[x0 + res / 2, y0 + res / 2]
        self.res = res

        # vehicle footprint
        self.length, self.width = vehicle["length"], vehicle["width"]
        lf, lr = vehicle["lf"], vehicle["lr"]
        ht = geo_cfg["tire_half_track_m"]
        self.tire_offsets = np.array([[lf, ht], [lf, -ht], [-lr, ht], [-lr, -ht]])  # FL FR RL RR
        self.patch_len, self.patch_w = geo_cfg["tire_patch_length_m"], geo_cfg["tire_patch_width_m"]
        # footprints() rows: FL FR RL RR tire patches, then the body; corners in _rect's order
        sizes = np.array([[self.patch_len, self.patch_w]] * 4 + [[self.length, self.width]])
        self._fp_dx = np.array([1, 1, -1, -1]) * sizes[:, :1] / 2
        self._fp_dy = np.array([1, -1, -1, 1]) * sizes[:, 1:] / 2
        self._fp_steer = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # front tires steer

    # ------------------------------------------------------------------ polygons
    def footprints(self, x, y, yaw, delta):
        """(5, 4, 2) corners of the four tire patches and the body, same vertices as tire_patches() / body()."""
        c, s = np.cos(yaw), np.sin(yaw)
        ox, oy = self.tire_offsets[:, 0], self.tire_offsets[:, 1]
        cx, cy = np.r_[x + c * ox - s * oy, x], np.r_[y + s * ox + c * oy, y]
        th = yaw + delta * self._fp_steer
        ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
        dx, dy = self._fp_dx, self._fp_dy
        return np.stack([cx[:, None] + ct * dx - st * dy, cy[:, None] + st * dx + ct * dy], axis=-1)

    def body(self, x, y, yaw):
        return Polygon(_rect(x, y, yaw, self.length, self.width))

    def tire_patches(self, x, y, yaw, delta):
        c, s = np.cos(yaw), np.sin(yaw)
        polys = []
        for k, (ox, oy) in enumerate(self.tire_offsets):
            px, py = x + c * ox - s * oy, y + s * ox + c * oy
            ty = yaw + (delta if k < 2 else 0.0)  # front tires steer
            polys.append(Polygon(_rect(px, py, ty, self.patch_len, self.patch_w)))
        return polys

    # ------------------------------------------------------------------ checks
    def check(self, x, y, yaw, delta):
        """Return (off_track, wall_collision, tires_in) for one pose. Runs at every physics step: the five
        polygons are built in one vectorised call (building them one by one cost ~90% of the check)."""
        polys = shapely.polygons(self.footprints(x, y, yaw, delta))
        tires_in = shapely.intersects(self.legal, polys[:4])
        off_track = not bool(np.any(tires_in))
        hit = self.wall_tree.query(polys[4], predicate="intersects")
        return off_track, bool(len(hit) > 0), tires_in

    def wall_clearance(self, x, y, yaw):
        """Distance from body to nearest wall cell (0 if touching)."""
        b = self.body(x, y, yaw)
        idx = self.wall_tree.query_nearest(b)
        return float(shapely.distance(b, self.wall_boxes[idx]).min())

    def wall_distance(self, xy):
        """Distance from each point to the nearest remaining wall cell edge."""
        pts = shapely.points(np.asarray(xy))
        idx = self.wall_tree.query_nearest(pts, return_distance=False)
        d = np.full(len(pts), np.inf)
        dist = shapely.distance(pts[idx[0]], self.wall_boxes[idx[1]])
        np.minimum.at(d, idx[0], dist)
        return d

    def centerline_wall_distance(self):
        """Smallest distance from the track centerline to any wall cell edge."""
        pts = shapely.points(self.center_xy)
        idx = self.wall_tree.query_nearest(pts, return_distance=False)
        return float(shapely.distance(pts[idx[0]], self.wall_boxes[idx[1]]).min())
