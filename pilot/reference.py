"""Fixed reference line (smoothed track centerline), curvature, speed plan, Frenet projection."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Reference:
    """Closed reference line resampled at ``ds`` and smoothed with a periodic Gaussian."""

    def __init__(self, xs, ys, ds=0.1, smooth_sigma_m=0.6, kappa_sigma_m=0.6):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if np.hypot(xs[0] - xs[-1], ys[0] - ys[-1]) < 1e-6:
            xs, ys = xs[:-1], ys[:-1]
        seg = np.hypot(np.diff(np.r_[xs, xs[0]]), np.diff(np.r_[ys, ys[0]]))
        s_raw = np.r_[0, np.cumsum(seg)]
        L = s_raw[-1]
        n = int(round(L / ds))
        s = np.linspace(0, L, n, endpoint=False)
        x = np.interp(s, s_raw, np.r_[xs, xs[0]])
        y = np.interp(s, s_raw, np.r_[ys, ys[0]])
        sig = smooth_sigma_m / ds
        if sig > 0:
            x = gaussian_filter1d(x, sig, mode="wrap")
            y = gaussian_filter1d(y, sig, mode="wrap")
        # re-parametrise by arc length after smoothing
        seg = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
        s_new = np.r_[0, np.cumsum(seg)]
        self.length = s_new[-1]
        n = int(round(self.length / ds))
        self.s = np.linspace(0, self.length, n, endpoint=False)
        self.ds = self.length / n
        self.x = np.interp(self.s, s_new, np.r_[x, x[0]])
        self.y = np.interp(self.s, s_new, np.r_[y, y[0]])
        dx = (np.roll(self.x, -1) - np.roll(self.x, 1)) / (2 * self.ds)
        dy = (np.roll(self.y, -1) - np.roll(self.y, 1)) / (2 * self.ds)
        self.psi = np.arctan2(dy, dx)
        dpsi = _wrap(np.roll(self.psi, -1) - np.roll(self.psi, 1)) / (2 * self.ds)
        self.kappa = gaussian_filter1d(dpsi, kappa_sigma_m / self.ds, mode="wrap")
        self.n = n

    # ------------------------------------------------------------------ lookup
    def idx(self, s):
        return np.round(np.asarray(s) / self.ds).astype(int) % self.n

    def kappa_at(self, s):
        return np.interp(np.mod(s, self.length), np.r_[self.s, self.length], np.r_[self.kappa, self.kappa[0]])

    def psi_at(self, s):
        i = self.idx(s)
        return self.psi[i]

    def project(self, x, y, i_guess=None, window=60):
        """Return (s, e_y, i). e_y > 0 means the point is left of the line."""
        if i_guess is None:
            ii = np.arange(self.n)
        else:
            ii = np.arange(i_guess - window, i_guess + window) % self.n
        d2 = (self.x[ii] - x) ** 2 + (self.y[ii] - y) ** 2
        i = ii[np.argmin(d2)]
        tx, ty = np.cos(self.psi[i]), np.sin(self.psi[i])
        rx, ry = x - self.x[i], y - self.y[i]
        along = rx * tx + ry * ty
        e_y = -rx * ty + ry * tx
        s = (self.s[i] + along) % self.length
        return s, e_y, i

    def project_many(self, xs, ys):
        out, i_g = [], None
        for x, y in zip(xs, ys):
            s, _, i_g = self.project(x, y, i_g, window=200) if i_g is not None else self.project(x, y)
            out.append(s)
        return np.array(out)

    def pose_at(self, s):
        i = self.idx(s)
        return self.x[i], self.y[i], self.psi[i]


def speed_plan(ref: Reference, a_lat: float, v_cap: float, a_acc: float, a_dec: float, v_min: float = 1.0,
               friction_ellipse: bool = False):
    """Curvature-limited speed profile v(s) = min(v_cap, sqrt(a_lat/|kappa|)) with accel/decel passes.

    ``friction_ellipse=False`` (pilot default): the passes use the full a_acc / a_dec everywhere, so braking
    and cornering demands add up (combined acceleration up to sqrt(a_dec^2 + a_lat^2)).
    ``friction_ellipse=True``: the passes only use what the ellipse (a_x/a_x,max)^2 + (v^2 kappa/a_lat)^2 <= 1
    leaves, a_x,avail = a_x,max sqrt(1 - (v^2 kappa / a_lat)^2), with a small floor so the passes can leave
    the curvature limit."""
    k = np.abs(ref.kappa)
    v = np.minimum(v_cap, np.sqrt(a_lat / np.maximum(k, 1e-6)))
    v = np.maximum(v, v_min)
    ds = ref.ds

    def avail(a_max, vi, ki):
        if not friction_ellipse:
            return a_max
        return a_max * np.sqrt(max(1.0 - (vi**2 * ki / a_lat) ** 2, 0.05**2))

    for _ in range(2):  # two sweeps handle the wrap-around
        for i in range(1, 2 * ref.n):  # forward (acceleration limit)
            a, b = (i - 1) % ref.n, i % ref.n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * avail(a_acc, v[a], k[a]) * ds))
        for i in range(2 * ref.n, 0, -1):  # backward (braking limit)
            a, b = i % ref.n, (i - 1) % ref.n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * avail(a_dec, v[a], k[a]) * ds))
    return v


def wall_safe_raceline(race_xy, center: Reference, n_max, smooth_sigma_m: float, ds_race: float,
                       lookaround_m: float = 0.0):
    """Keep the raceline's shape but clip its lateral offset from the centerline to |n| <= n_max.

    ``n_max`` is a scalar or a callable giving the allowed offset at each centerline point (x, y)
    (``inf`` where no wall is near). Returns (xs, ys, n_raw, n_safe). The clipped offset is smoothed
    (periodic Gaussian) only where clipping happened, so unclipped stretches keep the raceline exactly.
    """
    s_c, n = [], []
    i_g = None
    for x, y in race_xy:
        s, e, i_g = center.project(x, y, i_g, window=200) if i_g is not None else center.project(x, y)
        s_c.append(s), n.append(e)
    s_c, n = np.array(s_c), np.array(n)
    i = center.idx(s_c)
    lim = n_max(np.c_[center.x[i], center.y[i]]) if callable(n_max) else np.full(len(n), n_max)
    if lookaround_m > 0:  # the limit at s is the tightest limit within +/- lookaround_m (walls that start ahead)
        from scipy.ndimage import minimum_filter1d

        lim = minimum_filter1d(lim, size=2 * int(np.ceil(lookaround_m / ds_race)) + 1, mode="wrap")
    n_clip = np.clip(n, -lim, lim)
    n_safe = n_clip
    if smooth_sigma_m > 0 and np.any(n_clip != n):
        sm = gaussian_filter1d(n_clip, smooth_sigma_m / ds_race, mode="wrap")
        # blend: smoothed where clipping happened (+/- 3 sigma), raw elsewhere; never exceed the limit
        touched = gaussian_filter1d((n_clip != n).astype(float), smooth_sigma_m / ds_race, mode="wrap") > 1e-3
        n_safe = np.where(touched, sm, n)
        n_safe = np.clip(n_safe, -lim, lim)
    xs = center.x[i] - np.sin(center.psi[i]) * n_safe
    ys = center.y[i] + np.cos(center.psi[i]) * n_safe
    return xs, ys, n, n_safe
