//! Track-limit and wall checks of ``pilot.geometry.TrackGeometry.check``, without shapely.
//!
//! * legal region: the rings of the shapely polygon (same vertices), edges binned on a uniform grid;
//!   a tire patch intersects it iff a patch edge meets a ring edge, or the patch lies inside the region,
//!   or a whole ring lies inside the patch (GEOS ``intersects``: boundary contact counts);
//! * walls: the kept occupancy cells as boxes (same bounds as shapely's ``wall_boxes``), binned on a grid;
//!   the body intersects a box under the same rule.
//!
//! Orientation tests use Shewchuk's exact ``orient2d``, so only exactly-degenerate contacts can differ
//! from GEOS (whose orientation predicate is also exact).

use robust::{Coord, orient2d};

use crate::numerics::sin;

pub type P = [f64; 2];

#[inline]
fn orient(a: P, b: P, c: P) -> f64 {
    orient2d(
        Coord { x: a[0], y: a[1] },
        Coord { x: b[0], y: b[1] },
        Coord { x: c[0], y: c[1] },
    )
}

#[inline]
fn on_segment(a: P, b: P, p: P) -> bool {
    // p is collinear with a-b
    p[0] >= a[0].min(b[0])
        && p[0] <= a[0].max(b[0])
        && p[1] >= a[1].min(b[1])
        && p[1] <= a[1].max(b[1])
}

/// Closed segments p1-p2 and q1-q2 share at least one point.
#[inline]
pub fn segments_intersect(p1: P, p2: P, q1: P, q2: P) -> bool {
    let d1 = orient(q1, q2, p1);
    let d2 = orient(q1, q2, p2);
    let d3 = orient(p1, p2, q1);
    let d4 = orient(p1, p2, q2);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    (d1 == 0.0 && on_segment(q1, q2, p1))
        || (d2 == 0.0 && on_segment(q1, q2, p2))
        || (d3 == 0.0 && on_segment(p1, p2, q1))
        || (d4 == 0.0 && on_segment(p1, p2, q2))
}

/// Point inside or on a convex quadrilateral (any orientation).
#[inline]
fn in_convex_quad(q: &[P; 4], p: P) -> bool {
    let (mut pos, mut neg) = (false, false);
    for k in 0..4 {
        let o = orient(q[k], q[(k + 1) % 4], p);
        pos |= o > 0.0;
        neg |= o < 0.0;
    }
    !(pos && neg)
}

#[inline]
fn bbox(q: &[P; 4]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in q {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

/// Uniform grid over a bounding box; each cell lists item indices.
#[derive(Clone, Debug)]
struct Grid {
    x0: f64,
    y0: f64,
    cs: f64,
    nx: usize,
    ny: usize,
    cells: Vec<Vec<u32>>,
}

impl Grid {
    fn new(b: [f64; 4], cs: f64) -> Self {
        let nx = (((b[2] - b[0]) / cs).floor() as usize) + 1;
        let ny = (((b[3] - b[1]) / cs).floor() as usize) + 1;
        Self {
            x0: b[0],
            y0: b[1],
            cs,
            nx,
            ny,
            cells: vec![Vec::new(); nx * ny],
        }
    }

    #[inline]
    fn ix(&self, x: f64) -> usize {
        (((x - self.x0) / self.cs).floor().max(0.0) as usize).min(self.nx - 1)
    }

    #[inline]
    fn iy(&self, y: f64) -> usize {
        (((y - self.y0) / self.cs).floor().max(0.0) as usize).min(self.ny - 1)
    }

    fn insert(&mut self, b: [f64; 4], item: u32) {
        for j in self.iy(b[1])..=self.iy(b[3]) {
            for i in self.ix(b[0])..=self.ix(b[2]) {
                self.cells[j * self.nx + i].push(item);
            }
        }
    }

    /// Calls ``f`` for every item in the cells overlapping ``b`` (an item may be visited more than once);
    /// stops and returns true as soon as ``f`` does.
    #[inline]
    fn any(&self, b: [f64; 4], mut f: impl FnMut(u32) -> bool) -> bool {
        // boxes outside the grid cannot hit anything
        if b[2] < self.x0 || b[3] < self.y0 {
            return false;
        }
        let xmax = self.x0 + self.cs * self.nx as f64;
        let ymax = self.y0 + self.cs * self.ny as f64;
        if b[0] > xmax || b[1] > ymax {
            return false;
        }
        for j in self.iy(b[1])..=self.iy(b[3]) {
            for i in self.ix(b[0])..=self.ix(b[2]) {
                for &it in &self.cells[j * self.nx + i] {
                    if f(it) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[derive(Clone, Debug)]
pub struct Region {
    edges: Vec<[P; 2]>,
    grid: Grid,
    /// rows of height ``grid.cs``: edges whose y-range overlaps the row (for the crossing-number test)
    rows: Vec<Vec<u32>>,
    /// per ring: bounding box and first vertex (ring entirely inside a patch)
    rings: Vec<([f64; 4], P)>,
}

impl Region {
    /// ``rings``: closed coordinate rings (first point repeated last, as shapely returns them).
    pub fn new(rings: &[Vec<P>], cell: f64) -> Self {
        let mut edges = Vec::new();
        let mut ring_info = Vec::new();
        let mut tb = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for r in rings {
            let mut rb = [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            for w in r.windows(2) {
                if w[0] != w[1] {
                    edges.push([w[0], w[1]]);
                }
                for p in w {
                    rb[0] = rb[0].min(p[0]);
                    rb[1] = rb[1].min(p[1]);
                    rb[2] = rb[2].max(p[0]);
                    rb[3] = rb[3].max(p[1]);
                }
            }
            for k in 0..4 {
                tb[k] = if k < 2 {
                    tb[k].min(rb[k])
                } else {
                    tb[k].max(rb[k])
                };
            }
            ring_info.push((rb, r[0]));
        }
        let mut grid = Grid::new(tb, cell);
        let mut rows = vec![Vec::new(); grid.ny];
        for (k, e) in edges.iter().enumerate() {
            let b = [
                e[0][0].min(e[1][0]),
                e[0][1].min(e[1][1]),
                e[0][0].max(e[1][0]),
                e[0][1].max(e[1][1]),
            ];
            grid.insert(b, k as u32);
            for row in &mut rows[grid.iy(b[1])..=grid.iy(b[3])] {
                row.push(k as u32);
            }
        }
        Self {
            edges,
            grid,
            rows,
            rings: ring_info,
        }
    }

    /// Even-odd crossing number (points exactly on the boundary are handled by the edge test first).
    fn contains(&self, p: P) -> bool {
        let g = &self.grid;
        if p[1] < g.y0 || p[1] > g.y0 + g.cs * g.ny as f64 {
            return false;
        }
        let mut inside = false;
        for &k in &self.rows[g.iy(p[1])] {
            let [a, b] = self.edges[k as usize];
            if (a[1] > p[1]) != (b[1] > p[1]) {
                // the edge spans the horizontal line through p; ray to +x crosses it iff p is left of it
                let o = orient(a, b, p);
                if (b[1] > a[1] && o > 0.0) || (b[1] < a[1] && o < 0.0) {
                    inside = !inside;
                }
            }
        }
        inside
    }

    pub fn intersects_quad(&self, q: &[P; 4]) -> bool {
        let b = bbox(q);
        let hit = self.grid.any(b, |k| {
            let [a, c] = self.edges[k as usize];
            (0..4).any(|i| segments_intersect(q[i], q[(i + 1) % 4], a, c))
        });
        if hit || self.contains(q[0]) {
            return true;
        }
        self.rings.iter().any(|(rb, v)| {
            rb[0] >= b[0]
                && rb[2] <= b[2]
                && rb[1] >= b[1]
                && rb[3] <= b[3]
                && in_convex_quad(q, *v)
        })
    }
}

#[derive(Clone, Debug)]
pub struct Boxes {
    boxes: Vec<[f64; 4]>,
    grid: Option<Grid>,
}

impl Boxes {
    /// ``bounds``: (minx, miny, maxx, maxy) per box, as ``shapely.bounds(wall_boxes)``.
    pub fn new(bounds: Vec<[f64; 4]>, cell: f64) -> Self {
        if bounds.is_empty() {
            return Self {
                boxes: bounds,
                grid: None,
            };
        }
        let mut tb = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for b in &bounds {
            tb[0] = tb[0].min(b[0]);
            tb[1] = tb[1].min(b[1]);
            tb[2] = tb[2].max(b[2]);
            tb[3] = tb[3].max(b[3]);
        }
        let mut grid = Grid::new(tb, cell);
        for (k, b) in bounds.iter().enumerate() {
            grid.insert(*b, k as u32);
        }
        Self {
            boxes: bounds,
            grid: Some(grid),
        }
    }

    pub fn intersects_quad(&self, q: &[P; 4]) -> bool {
        let Some(grid) = &self.grid else { return false };
        let qb = bbox(q);
        grid.any(qb, |k| {
            let b = self.boxes[k as usize];
            if b[0] > qb[2] || b[2] < qb[0] || b[1] > qb[3] || b[3] < qb[1] {
                return false;
            }
            // shapely.box(x0, y0, x1, y1) vertex order: (x1, y0), (x1, y1), (x0, y1), (x0, y0)
            let r = [[b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]];
            for i in 0..4 {
                for j in 0..4 {
                    if segments_intersect(q[i], q[(i + 1) % 4], r[j], r[(j + 1) % 4]) {
                        return true;
                    }
                }
            }
            let p = q[0];
            (p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3])
                || in_convex_quad(q, r[0])
        })
    }
}

/// Vehicle footprint of ``TrackGeometry.footprints``: four tire patches (FL FR RL RR) and the body.
#[derive(Clone, Debug)]
pub struct Footprint {
    pub tire_offsets: [[f64; 2]; 4],
    /// per rectangle: corner offsets dx, dy (``np.array([1, 1, -1, -1]) * size / 2`` etc.)
    pub dx: [[f64; 4]; 5],
    pub dy: [[f64; 4]; 5],
    pub steer: [f64; 5],
}

impl Footprint {
    pub fn new(
        tire_offsets: [[f64; 2]; 4],
        patch_len: f64,
        patch_w: f64,
        length: f64,
        width: f64,
    ) -> Self {
        let sx = [1.0, 1.0, -1.0, -1.0];
        let sy = [1.0, -1.0, -1.0, 1.0];
        let mut dx = [[0.0; 4]; 5];
        let mut dy = [[0.0; 4]; 5];
        for r in 0..5 {
            let (l, w) = if r < 4 {
                (patch_len, patch_w)
            } else {
                (length, width)
            };
            for k in 0..4 {
                dx[r][k] = sx[k] * l / 2.0;
                dy[r][k] = sy[k] * w / 2.0;
            }
        }
        Self {
            tire_offsets,
            dx,
            dy,
            steer: [1.0, 1.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn corners(&self, x: f64, y: f64, yaw: f64, delta: f64) -> [[P; 4]; 5] {
        let (c, s) = (yaw.cos(), sin(yaw));
        let mut out = [[[0.0; 2]; 4]; 5];
        for r in 0..5 {
            let (cx, cy) = if r < 4 {
                let [ox, oy] = self.tire_offsets[r];
                (x + c * ox - s * oy, y + s * ox + c * oy)
            } else {
                (x, y)
            };
            let th = yaw + delta * self.steer[r];
            let (ct, st) = (th.cos(), sin(th));
            for k in 0..4 {
                let (dx, dy) = (self.dx[r][k], self.dy[r][k]);
                out[r][k] = [cx + ct * dx - st * dy, cy + st * dx + ct * dy];
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct Geometry {
    pub legal: Region,
    pub walls: Boxes,
    pub footprint: Footprint,
}

impl Geometry {
    /// (off_track, wall_collision, tires_in)
    pub fn check(&self, x: f64, y: f64, yaw: f64, delta: f64) -> (bool, bool, [bool; 4]) {
        let fp = self.footprint.corners(x, y, yaw, delta);
        let mut tires_in = [false; 4];
        for k in 0..4 {
            tires_in[k] = self.legal.intersects_quad(&fp[k]);
        }
        let off = !tires_in.iter().any(|&t| t);
        let wall = self.walls.intersects_quad(&fp[4]);
        (off, wall, tires_in)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touching_segments_intersect() {
        assert!(segments_intersect(
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.0]
        ));
        assert!(segments_intersect(
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ));
        assert!(!segments_intersect(
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ));
    }

    #[test]
    fn square_region() {
        let ring = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [0.0, 0.0],
        ];
        let hole = vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0], [4.0, 4.0]];
        let r = Region::new(&[ring, hole], 1.0);
        let q = |cx: f64, cy: f64| {
            [
                [cx - 0.1, cy - 0.1],
                [cx + 0.1, cy - 0.1],
                [cx + 0.1, cy + 0.1],
                [cx - 0.1, cy + 0.1],
            ]
        };
        assert!(r.intersects_quad(&q(1.0, 1.0)));
        assert!(!r.intersects_quad(&q(5.0, 5.0))); // inside the hole
        assert!(!r.intersects_quad(&q(20.0, 5.0)));
        assert!(r.intersects_quad(&q(10.05, 5.0))); // straddles the boundary
    }
}
