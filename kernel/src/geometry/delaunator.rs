//! Triangulation de Delaunay : traduction fidèle de Delaunator 5.1 (JavaScript), pour que les cartes
//! du noyau soient exactement celles de l'ancien générateur.

use super::predicates::orient2d;

const EPSILON: f64 = f64::EPSILON;
const EDGE_STACK_SIZE: usize = 512;
pub const NONE: usize = usize::MAX;

pub struct Delaunator {
    pub coords: Vec<f64>,
    pub triangles: Vec<usize>,
    /// Demi-arête jumelle, ou `NONE` sur l'enveloppe convexe.
    pub halfedges: Vec<usize>,
    pub hull: Vec<usize>,
}

fn dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
}

fn pseudo_angle(dx: f64, dy: f64) -> f64 {
    let p = dx / (dx.abs() + dy.abs());
    (if dy > 0.0 { 3.0 - p } else { 1.0 + p }) / 4.0
}

#[allow(clippy::too_many_arguments)]
fn in_circle(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, px: f64, py: f64) -> bool {
    let dx = ax - px;
    let dy = ay - py;
    let ex = bx - px;
    let ey = by - py;
    let fx = cx - px;
    let fy = cy - py;
    let ap = dx * dx + dy * dy;
    let bp = ex * ex + ey * ey;
    let cp = fx * fx + fy * fy;
    dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx) < 0.0
}

fn circumradius(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let ex = cx - ax;
    let ey = cy - ay;
    let bl = dx * dx + dy * dy;
    let cl = ex * ex + ey * ey;
    let d = 0.5 / (dx * ey - dy * ex);
    let x = (ey * bl - dy * cl) * d;
    let y = (dx * cl - ex * bl) * d;
    x * x + y * y
}

fn circumcenter(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> (f64, f64) {
    let dx = bx - ax;
    let dy = by - ay;
    let ex = cx - ax;
    let ey = cy - ay;
    let bl = dx * dx + dy * dy;
    let cl = ex * ex + ey * ey;
    let d = 0.5 / (dx * ey - dy * ex);
    (ax + (ey * bl - dy * cl) * d, ay + (dx * cl - ex * bl) * d)
}

/// Le tri rapide de Delaunator (mêmes échanges, donc même ordre pour les distances égales).
fn quicksort(ids: &mut [usize], dists: &[f64], left: usize, right: usize) {
    if right as isize - left as isize <= 20 {
        for i in left + 1..=right {
            let temp = ids[i];
            let temp_dist = dists[temp];
            let mut j = i as isize - 1;
            while j >= left as isize && dists[ids[j as usize]] > temp_dist {
                ids[(j + 1) as usize] = ids[j as usize];
                j -= 1;
            }
            ids[(j + 1) as usize] = temp;
        }
    } else {
        let median = (left + right) >> 1;
        let mut i = left + 1;
        let mut j = right;
        ids.swap(median, i);
        if dists[ids[left]] > dists[ids[right]] {
            ids.swap(left, right);
        }
        if dists[ids[i]] > dists[ids[right]] {
            ids.swap(i, right);
        }
        if dists[ids[left]] > dists[ids[i]] {
            ids.swap(left, i);
        }
        let temp = ids[i];
        let temp_dist = dists[temp];
        loop {
            loop {
                i += 1;
                if dists[ids[i]] >= temp_dist {
                    break;
                }
            }
            loop {
                j -= 1;
                if dists[ids[j]] <= temp_dist {
                    break;
                }
            }
            if j < i {
                break;
            }
            ids.swap(i, j);
        }
        ids[left + 1] = ids[j];
        ids[j] = temp;
        if right as isize - i as isize + 1 >= j as isize - left as isize {
            quicksort(ids, dists, i, right);
            if j > left {
                quicksort(ids, dists, left, j - 1);
            }
        } else {
            if j > left {
                quicksort(ids, dists, left, j - 1);
            }
            quicksort(ids, dists, i, right);
        }
    }
}

struct Builder<'a> {
    coords: &'a [f64],
    triangles: Vec<usize>,
    halfedges: Vec<usize>,
    hull_prev: Vec<usize>,
    hull_next: Vec<usize>,
    hull_tri: Vec<usize>,
    hull_hash: Vec<usize>,
    hash_size: usize,
    hull_start: usize,
    cx: f64,
    cy: f64,
}

impl Builder<'_> {
    fn hash_key(&self, x: f64, y: f64) -> usize {
        ((pseudo_angle(x - self.cx, y - self.cy) * self.hash_size as f64).floor() as usize) % self.hash_size
    }

    fn link(&mut self, a: usize, b: usize) {
        self.halfedges[a] = b;
        if b != NONE {
            self.halfedges[b] = a;
        }
    }

    fn add_triangle(&mut self, i0: usize, i1: usize, i2: usize, a: usize, b: usize, c: usize) -> usize {
        let t = self.triangles.len();
        self.triangles.extend([i0, i1, i2]);
        self.halfedges.extend([NONE, NONE, NONE]);
        self.link(t, a);
        self.link(t + 1, b);
        self.link(t + 2, c);
        t
    }

    fn legalize(&mut self, mut a: usize) -> usize {
        let mut stack = [0usize; EDGE_STACK_SIZE];
        let mut i = 0;
        let mut ar;
        loop {
            let b = self.halfedges[a];
            let a0 = a - a % 3;
            ar = a0 + (a + 2) % 3;
            if b == NONE {
                if i == 0 {
                    break;
                }
                i -= 1;
                a = stack[i];
                continue;
            }
            let b0 = b - b % 3;
            let al = a0 + (a + 1) % 3;
            let bl = b0 + (b + 2) % 3;
            let p0 = self.triangles[ar];
            let pr = self.triangles[a];
            let pl = self.triangles[al];
            let p1 = self.triangles[bl];
            let c = self.coords;
            let illegal = in_circle(
                c[2 * p0],
                c[2 * p0 + 1],
                c[2 * pr],
                c[2 * pr + 1],
                c[2 * pl],
                c[2 * pl + 1],
                c[2 * p1],
                c[2 * p1 + 1],
            );
            if illegal {
                self.triangles[a] = p1;
                self.triangles[b] = p0;
                let hbl = self.halfedges[bl];
                // arête retournée de l'autre côté de l'enveloppe (rare) : corriger la référence
                if hbl == NONE {
                    let mut e = self.hull_start;
                    loop {
                        if self.hull_tri[e] == bl {
                            self.hull_tri[e] = a;
                            break;
                        }
                        e = self.hull_prev[e];
                        if e == self.hull_start {
                            break;
                        }
                    }
                }
                self.link(a, hbl);
                let har = self.halfedges[ar];
                self.link(b, har);
                self.link(ar, bl);
                let br = b0 + (b + 1) % 3;
                if i < EDGE_STACK_SIZE {
                    stack[i] = br;
                    i += 1;
                }
            } else {
                if i == 0 {
                    break;
                }
                i -= 1;
                a = stack[i];
            }
        }
        ar
    }
}

impl Delaunator {
    /// Triangule les points `coords` = [x0, y0, x1, y1, …]. `None` si tous les points sont alignés.
    pub fn new(coords: Vec<f64>) -> Option<Delaunator> {
        let n = coords.len() >> 1;
        let (mut min_x, mut min_y, mut max_x, mut max_y) =
            (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut ids: Vec<usize> = (0..n).collect();
        for i in 0..n {
            let (x, y) = (coords[2 * i], coords[2 * i + 1]);
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
        }
        let cx = (min_x + max_x) / 2.0;
        let cy = (min_y + max_y) / 2.0;
        let (mut i0, mut i1, mut i2) = (0, 0, 0);
        let mut min_dist = f64::INFINITY;
        for i in 0..n {
            let d = dist(cx, cy, coords[2 * i], coords[2 * i + 1]);
            if d < min_dist {
                i0 = i;
                min_dist = d;
            }
        }
        let (i0x, i0y) = (coords[2 * i0], coords[2 * i0 + 1]);
        let mut min_dist = f64::INFINITY;
        for i in 0..n {
            if i == i0 {
                continue;
            }
            let d = dist(i0x, i0y, coords[2 * i], coords[2 * i + 1]);
            if d < min_dist && d > 0.0 {
                i1 = i;
                min_dist = d;
            }
        }
        let (mut i1x, mut i1y) = (coords[2 * i1], coords[2 * i1 + 1]);
        let mut min_radius = f64::INFINITY;
        for i in 0..n {
            if i == i0 || i == i1 {
                continue;
            }
            let r = circumradius(i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1]);
            if r < min_radius {
                i2 = i;
                min_radius = r;
            }
        }
        let (mut i2x, mut i2y) = (coords[2 * i2], coords[2 * i2 + 1]);
        if min_radius == f64::INFINITY {
            return None;
        }
        if orient2d(i0x, i0y, i1x, i1y, i2x, i2y) < 0.0 {
            core::mem::swap(&mut i1, &mut i2);
            core::mem::swap(&mut i1x, &mut i2x);
            core::mem::swap(&mut i1y, &mut i2y);
        }
        let (ccx, ccy) = circumcenter(i0x, i0y, i1x, i1y, i2x, i2y);
        let dists: Vec<f64> = (0..n).map(|i| dist(coords[2 * i], coords[2 * i + 1], ccx, ccy)).collect();
        if n > 0 {
            quicksort(&mut ids, &dists, 0, n - 1);
        }
        let hash_size = (n as f64).sqrt().ceil() as usize;
        let max_triangles = (2 * n).saturating_sub(5);
        let mut b = Builder {
            coords: &coords,
            triangles: Vec::with_capacity(max_triangles * 3),
            halfedges: Vec::with_capacity(max_triangles * 3),
            hull_prev: vec![0; n],
            hull_next: vec![0; n],
            hull_tri: vec![0; n],
            hull_hash: vec![NONE; hash_size],
            hash_size,
            hull_start: i0,
            cx: ccx,
            cy: ccy,
        };
        let mut hull_size = 3;
        b.hull_next[i0] = i1;
        b.hull_prev[i2] = i1;
        b.hull_next[i1] = i2;
        b.hull_prev[i0] = i2;
        b.hull_next[i2] = i0;
        b.hull_prev[i1] = i0;
        b.hull_tri[i0] = 0;
        b.hull_tri[i1] = 1;
        b.hull_tri[i2] = 2;
        let k = b.hash_key(i0x, i0y);
        b.hull_hash[k] = i0;
        let k = b.hash_key(i1x, i1y);
        b.hull_hash[k] = i1;
        let k = b.hash_key(i2x, i2y);
        b.hull_hash[k] = i2;
        b.add_triangle(i0, i1, i2, NONE, NONE, NONE);

        let (mut xp, mut yp) = (0.0, 0.0);
        for (k, &i) in ids.iter().enumerate() {
            let (x, y) = (coords[2 * i], coords[2 * i + 1]);
            // points presque confondus
            if k > 0 && (x - xp).abs() <= EPSILON && (y - yp).abs() <= EPSILON {
                continue;
            }
            xp = x;
            yp = y;
            if i == i0 || i == i1 || i == i2 {
                continue;
            }
            // une arête visible de l'enveloppe, par la table angulaire
            let mut start = 0;
            let key = b.hash_key(x, y);
            for j in 0..b.hash_size {
                start = b.hull_hash[(key + j) % b.hash_size];
                if start != NONE && start != b.hull_next[start] {
                    break;
                }
            }
            start = b.hull_prev[start];
            let mut e = start;
            let mut q;
            loop {
                q = b.hull_next[e];
                if orient2d(x, y, coords[2 * e], coords[2 * e + 1], coords[2 * q], coords[2 * q + 1]) >= 0.0 {
                    e = q;
                    if e == start {
                        e = NONE;
                        break;
                    }
                } else {
                    break;
                }
            }
            if e == NONE {
                continue;
            }
            let hn = b.hull_next[e];
            let ht = b.hull_tri[e];
            let mut t = b.add_triangle(e, i, hn, NONE, NONE, ht);
            b.hull_tri[i] = b.legalize(t + 2);
            b.hull_tri[e] = t;
            hull_size += 1;
            let mut nn = b.hull_next[e];
            loop {
                q = b.hull_next[nn];
                if orient2d(x, y, coords[2 * nn], coords[2 * nn + 1], coords[2 * q], coords[2 * q + 1]) >= 0.0 {
                    break;
                }
                let (hi, hnn) = (b.hull_tri[i], b.hull_tri[nn]);
                t = b.add_triangle(nn, i, q, hi, NONE, hnn);
                b.hull_tri[i] = b.legalize(t + 2);
                b.hull_next[nn] = nn;
                hull_size -= 1;
                nn = q;
            }
            if e == start {
                loop {
                    q = b.hull_prev[e];
                    if orient2d(x, y, coords[2 * q], coords[2 * q + 1], coords[2 * e], coords[2 * e + 1]) >= 0.0 {
                        break;
                    }
                    let (he, hq) = (b.hull_tri[e], b.hull_tri[q]);
                    t = b.add_triangle(q, i, e, NONE, he, hq);
                    b.legalize(t + 2);
                    b.hull_tri[q] = t;
                    b.hull_next[e] = e;
                    hull_size -= 1;
                    e = q;
                }
            }
            b.hull_start = e;
            b.hull_prev[i] = e;
            b.hull_next[e] = i;
            b.hull_prev[nn] = i;
            b.hull_next[i] = nn;
            let k = b.hash_key(x, y);
            b.hull_hash[k] = i;
            let k = b.hash_key(coords[2 * e], coords[2 * e + 1]);
            b.hull_hash[k] = e;
        }
        let mut hull = Vec::with_capacity(hull_size);
        let mut e = b.hull_start;
        for _ in 0..hull_size {
            hull.push(e);
            e = b.hull_next[e];
        }
        let (triangles, halfedges) = (b.triangles, b.halfedges);
        Some(Delaunator { coords, triangles, halfedges, hull })
    }
}
