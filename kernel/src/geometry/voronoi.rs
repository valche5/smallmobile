//! Diagramme de Voronoï découpé dans un rectangle : traduction fidèle de d3-delaunay 6.0 (JavaScript).

use super::delaunator::{Delaunator, NONE};

pub struct Voronoi {
    d: Delaunator,
    inedges: Vec<usize>,
    hull_index: Vec<usize>,
    circumcenters: Vec<f64>,
    vectors: Vec<f64>,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
}

const TOP_LEFT: u8 = 0b0101;
const TOP: u8 = 0b0100;
const TOP_RIGHT: u8 = 0b0110;
const RIGHT: u8 = 0b0010;
const BOTTOM_RIGHT: u8 = 0b1010;
const BOTTOM: u8 = 0b1000;
const BOTTOM_LEFT: u8 = 0b1001;
const LEFT: u8 = 0b0001;

impl Voronoi {
    /// `None` si les points sont tous alignés (cas que d3 traite en les décalant ; jamais sur nos cartes).
    pub fn new(points: &[(f64, f64)], [xmin, ymin, xmax, ymax]: [f64; 4]) -> Option<Voronoi> {
        let coords: Vec<f64> = points.iter().flat_map(|&(x, y)| [x, y]).collect();
        let d = Delaunator::new(coords)?;
        if d.hull.len() > 2 && Self::collinear(&d) {
            return None;
        }
        let n = points.len();
        let mut inedges = vec![NONE; n];
        let mut hull_index = vec![NONE; n];
        for e in 0..d.halfedges.len() {
            let p = d.triangles[if e % 3 == 2 { e - 2 } else { e + 1 }];
            if d.halfedges[e] == NONE || inedges[p] == NONE {
                inedges[p] = e;
            }
        }
        for (i, &h) in d.hull.iter().enumerate() {
            hull_index[h] = i;
        }
        if d.hull.len() <= 2 {
            return None;
        }
        let mut v = Voronoi {
            circumcenters: Vec::with_capacity(d.triangles.len() / 3 * 2),
            vectors: vec![0.0; n * 4],
            d,
            inedges,
            hull_index,
            xmin,
            ymin,
            xmax,
            ymax,
        };
        v.init();
        Some(v)
    }

    fn collinear(d: &Delaunator) -> bool {
        let c = &d.coords;
        d.triangles.as_chunks::<3>().0.iter().all(|t| {
            let (a, b, cc) = (2 * t[0], 2 * t[1], 2 * t[2]);
            let cross = (c[cc] - c[a]) * (c[b + 1] - c[a + 1]) - (c[b] - c[a]) * (c[cc + 1] - c[a + 1]);
            cross <= 1e-10
        })
    }

    fn init(&mut self) {
        let points = &self.d.coords;
        let (mut bx, mut by) = (None::<f64>, 0.0);
        for t in self.d.triangles.as_chunks::<3>().0 {
            let (t1, t2, t3) = (t[0] * 2, t[1] * 2, t[2] * 2);
            let (x1, y1) = (points[t1], points[t1 + 1]);
            let (x2, y2) = (points[t2], points[t2 + 1]);
            let (x3, y3) = (points[t3], points[t3 + 1]);
            let dx = x2 - x1;
            let dy = y2 - y1;
            let ex = x3 - x1;
            let ey = y3 - y1;
            let ab = (dx * ey - dy * ex) * 2.0;
            let (x, y);
            if ab.abs() < 1e-9 {
                // triangle dégénéré : centre à l'infini, à l'opposé du barycentre de l'enveloppe
                let cx = *bx.get_or_insert_with(|| {
                    let (mut sx, mut sy) = (0.0, 0.0);
                    for &i in &self.d.hull {
                        sx += points[i * 2];
                        sy += points[i * 2 + 1];
                    }
                    by = sy / self.d.hull.len() as f64;
                    sx / self.d.hull.len() as f64
                });
                let s = (cx - x1) * ey - (by - y1) * ex;
                let a = 1e9
                    * if s > 0.0 {
                        1.0
                    } else if s < 0.0 {
                        -1.0
                    } else {
                        s
                    };
                x = (x1 + x3) / 2.0 - a * ey;
                y = (y1 + y3) / 2.0 + a * ex;
            } else {
                let d = 1.0 / ab;
                let bl = dx * dx + dy * dy;
                let cl = ex * ex + ey * ey;
                x = x1 + (ey * bl - dy * cl) * d;
                y = y1 + (dx * cl - ex * bl) * d;
            }
            self.circumcenters.push(x);
            self.circumcenters.push(y);
        }
        // rayons des cellules extérieures
        let hull = &self.d.hull;
        let mut h = hull[hull.len() - 1];
        let mut p1 = h * 4;
        let (mut x1, mut y1) = (points[2 * h], points[2 * h + 1]);
        for &hh in hull {
            h = hh;
            let (p0, x0, y0) = (p1, x1, y1);
            p1 = h * 4;
            x1 = points[2 * h];
            y1 = points[2 * h + 1];
            self.vectors[p0 + 2] = y0 - y1;
            self.vectors[p1] = y0 - y1;
            self.vectors[p0 + 3] = x1 - x0;
            self.vectors[p1 + 1] = x1 - x0;
        }
    }

    /// Le polygone de la cellule `i`, sans répéter le premier point. `None` si la cellule est vide.
    pub fn cell_polygon(&self, i: usize) -> Option<Vec<(f64, f64)>> {
        let points = self.clip(i)?;
        if points.is_empty() {
            return None;
        }
        let mut out = vec![(points[0], points[1])];
        let mut n = points.len();
        while points[0] == points[n - 2] && points[1] == points[n - 1] && n > 1 {
            n -= 2;
        }
        let mut k = 2;
        while k < n {
            if points[k] != points[k - 2] || points[k + 1] != points[k - 1] {
                out.push((points[k], points[k + 1]));
            }
            k += 2;
        }
        Some(out)
    }

    fn cell(&self, i: usize) -> Option<Vec<f64>> {
        let e0 = self.inedges[i];
        if e0 == NONE {
            return None;
        }
        let mut points = Vec::new();
        let mut e = e0;
        loop {
            let t = e / 3;
            points.push(self.circumcenters[t * 2]);
            points.push(self.circumcenters[t * 2 + 1]);
            e = if e % 3 == 2 { e - 2 } else { e + 1 };
            if self.d.triangles[e] != i {
                break;
            }
            e = self.d.halfedges[e];
            if e == e0 || e == NONE {
                break;
            }
        }
        Some(points)
    }

    fn clip(&self, i: usize) -> Option<Vec<f64>> {
        let points = self.cell(i)?;
        let v = i * 4;
        let vv = &self.vectors;
        let clipped = if vv[v] != 0.0 || vv[v + 1] != 0.0 {
            self.clip_infinite(i, points, vv[v], vv[v + 1], vv[v + 2], vv[v + 3])
        } else {
            self.clip_finite(i, &points)
        };
        Self::simplify(clipped)
    }

    fn clip_finite(&self, i: usize, points: &[f64]) -> Option<Vec<f64>> {
        let n = points.len();
        let mut p: Option<Vec<f64>> = None;
        let (mut x1, mut y1) = (points[n - 2], points[n - 1]);
        let mut c1 = self.regioncode(x1, y1);
        let mut e1 = 0;
        let push = |p: &mut Option<Vec<f64>>, x: f64, y: f64| match p {
            Some(p) => p.extend([x, y]),
            None => *p = Some(vec![x, y]),
        };
        let mut j = 0;
        while j < n {
            let (x0, y0) = (x1, y1);
            x1 = points[j];
            y1 = points[j + 1];
            let c0 = c1;
            c1 = self.regioncode(x1, y1);
            if c0 == 0 && c1 == 0 {
                e1 = 0;
                push(&mut p, x1, y1);
            } else {
                let (sx1, sy1);
                if c0 == 0 {
                    let Some(s) = self.clip_segment(x0, y0, x1, y1, c0, c1) else {
                        j += 2;
                        continue;
                    };
                    (sx1, sy1) = (s[2], s[3]);
                } else {
                    let Some(s) = self.clip_segment(x1, y1, x0, y0, c1, c0) else {
                        j += 2;
                        continue;
                    };
                    let (sx0, sy0) = (s[2], s[3]);
                    (sx1, sy1) = (s[0], s[1]);
                    let e0 = e1;
                    e1 = self.edgecode(sx0, sy0);
                    if e0 != 0 && e1 != 0 {
                        let at = p.as_ref().map_or(0, Vec::len);
                        self.edge(i, e0, e1, p.get_or_insert_with(Vec::new), at);
                    }
                    push(&mut p, sx0, sy0);
                }
                let e0 = e1;
                e1 = self.edgecode(sx1, sy1);
                if e0 != 0 && e1 != 0 {
                    let at = p.as_ref().map_or(0, Vec::len);
                    self.edge(i, e0, e1, p.get_or_insert_with(Vec::new), at);
                }
                push(&mut p, sx1, sy1);
            }
            j += 2;
        }
        match p {
            Some(mut p) => {
                let e0 = e1;
                let e1 = self.edgecode(p[0], p[1]);
                if e0 != 0 && e1 != 0 {
                    let at = p.len();
                    self.edge(i, e0, e1, &mut p, at);
                }
                Some(p)
            }
            None if self.contains(i, (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0) => {
                Some(vec![self.xmax, self.ymin, self.xmax, self.ymax, self.xmin, self.ymax, self.xmin, self.ymin])
            }
            None => None,
        }
    }

    fn clip_segment(&self, x0: f64, y0: f64, x1: f64, y1: f64, c0: u8, c1: u8) -> Option<[f64; 4]> {
        let flip = c0 < c1;
        let (mut x0, mut y0, mut x1, mut y1, mut c0, mut c1) =
            if flip { (x1, y1, x0, y0, c1, c0) } else { (x0, y0, x1, y1, c0, c1) };
        loop {
            if c0 == 0 && c1 == 0 {
                return Some(if flip { [x1, y1, x0, y0] } else { [x0, y0, x1, y1] });
            }
            if c0 & c1 != 0 {
                return None;
            }
            let c = if c0 != 0 { c0 } else { c1 };
            let (x, y);
            if c & 0b1000 != 0 {
                x = x0 + (x1 - x0) * (self.ymax - y0) / (y1 - y0);
                y = self.ymax;
            } else if c & 0b0100 != 0 {
                x = x0 + (x1 - x0) * (self.ymin - y0) / (y1 - y0);
                y = self.ymin;
            } else if c & 0b0010 != 0 {
                y = y0 + (y1 - y0) * (self.xmax - x0) / (x1 - x0);
                x = self.xmax;
            } else {
                y = y0 + (y1 - y0) * (self.xmin - x0) / (x1 - x0);
                x = self.xmin;
            }
            if c0 != 0 {
                x0 = x;
                y0 = y;
                c0 = self.regioncode(x0, y0);
            } else {
                x1 = x;
                y1 = y;
                c1 = self.regioncode(x1, y1);
            }
        }
    }

    fn clip_infinite(&self, i: usize, points: Vec<f64>, vx0: f64, vy0: f64, vxn: f64, vyn: f64) -> Option<Vec<f64>> {
        let mut p = points;
        if let Some((x, y)) = self.project(p[0], p[1], vx0, vy0) {
            p.splice(0..0, [x, y]);
        }
        if let Some((x, y)) = self.project(p[p.len() - 2], p[p.len() - 1], vxn, vyn) {
            p.extend([x, y]);
        }
        match self.clip_finite(i, &p) {
            Some(mut p) => {
                let mut n = p.len();
                let mut c1 = self.edgecode(p[n - 2], p[n - 1]);
                let mut j = 0;
                while j < n {
                    let c0 = c1;
                    c1 = self.edgecode(p[j], p[j + 1]);
                    if c0 != 0 && c1 != 0 {
                        j = self.edge(i, c0, c1, &mut p, j);
                        n = p.len();
                    }
                    j += 2;
                }
                Some(p)
            }
            None if self.contains(i, (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0) => {
                Some(vec![self.xmin, self.ymin, self.xmax, self.ymin, self.xmax, self.ymax, self.xmin, self.ymax])
            }
            None => None,
        }
    }

    fn edge(&self, i: usize, mut e0: u8, e1: u8, p: &mut Vec<f64>, mut j: usize) -> usize {
        while e0 != e1 {
            let (x, y);
            match e0 {
                TOP_LEFT => {
                    e0 = TOP;
                    continue;
                }
                TOP => {
                    e0 = TOP_RIGHT;
                    (x, y) = (self.xmax, self.ymin);
                }
                TOP_RIGHT => {
                    e0 = RIGHT;
                    continue;
                }
                RIGHT => {
                    e0 = BOTTOM_RIGHT;
                    (x, y) = (self.xmax, self.ymax);
                }
                BOTTOM_RIGHT => {
                    e0 = BOTTOM;
                    continue;
                }
                BOTTOM => {
                    e0 = BOTTOM_LEFT;
                    (x, y) = (self.xmin, self.ymax);
                }
                BOTTOM_LEFT => {
                    e0 = LEFT;
                    continue;
                }
                LEFT => {
                    e0 = TOP_LEFT;
                    (x, y) = (self.xmin, self.ymin);
                }
                // un code de bord invalide : d3 boucle sans fin ; ici on s'arrête
                _ => return j,
            }
            let differs = p.get(j) != Some(&x) || p.get(j + 1) != Some(&y);
            if differs && self.contains(i, x, y) {
                p.splice(j..j, [x, y]);
                j += 2;
            }
        }
        j
    }

    fn project(&self, x0: f64, y0: f64, vx: f64, vy: f64) -> Option<(f64, f64)> {
        let mut t = f64::INFINITY;
        let (mut x, mut y) = (f64::NAN, f64::NAN);
        if vy < 0.0 {
            if y0 <= self.ymin {
                return None;
            }
            let c = (self.ymin - y0) / vy;
            if c < t {
                y = self.ymin;
                t = c;
                x = x0 + t * vx;
            }
        } else if vy > 0.0 {
            if y0 >= self.ymax {
                return None;
            }
            let c = (self.ymax - y0) / vy;
            if c < t {
                y = self.ymax;
                t = c;
                x = x0 + t * vx;
            }
        }
        if vx > 0.0 {
            if x0 >= self.xmax {
                return None;
            }
            let c = (self.xmax - x0) / vx;
            if c < t {
                x = self.xmax;
                t = c;
                y = y0 + t * vy;
            }
        } else if vx < 0.0 {
            if x0 <= self.xmin {
                return None;
            }
            let c = (self.xmin - x0) / vx;
            if c < t {
                x = self.xmin;
                t = c;
                y = y0 + t * vy;
            }
        }
        Some((x, y))
    }

    fn edgecode(&self, x: f64, y: f64) -> u8 {
        let h = if x == self.xmin {
            0b0001
        } else if x == self.xmax {
            0b0010
        } else {
            0
        };
        let v = if y == self.ymin {
            0b0100
        } else if y == self.ymax {
            0b1000
        } else {
            0
        };
        h | v
    }

    fn regioncode(&self, x: f64, y: f64) -> u8 {
        let h = if x < self.xmin {
            0b0001
        } else if x > self.xmax {
            0b0010
        } else {
            0
        };
        let v = if y < self.ymin {
            0b0100
        } else if y > self.ymax {
            0b1000
        } else {
            0
        };
        h | v
    }

    fn simplify(p: Option<Vec<f64>>) -> Option<Vec<f64>> {
        let mut p = p?;
        if p.len() > 4 {
            let mut i: isize = 0;
            while (i as usize) < p.len() {
                let len = p.len();
                let iu = i as usize;
                let j = (iu + 2) % len;
                let k = (iu + 4) % len;
                if (p[iu] == p[j] && p[j] == p[k]) || (p[iu + 1] == p[j + 1] && p[j + 1] == p[k + 1]) {
                    p.drain(j..j + 2);
                    i -= 2;
                }
                i += 2;
            }
            if p.is_empty() {
                return None;
            }
        }
        Some(p)
    }

    /// La cellule de `i` contient-elle (x, y) ? (le point de la triangulation le plus proche est `i`)
    fn contains(&self, i: usize, x: f64, y: f64) -> bool {
        if x.is_nan() || y.is_nan() {
            return false;
        }
        self.step(i, x, y) == i
    }

    fn step(&self, i: usize, x: f64, y: f64) -> usize {
        let points = &self.d.coords;
        let n = points.len() >> 1;
        if self.inedges[i] == NONE || points.is_empty() {
            return (i + 1) % n;
        }
        let sq = |t: usize| {
            let (dx, dy) = (x - points[t * 2], y - points[t * 2 + 1]);
            dx * dx + dy * dy
        };
        let mut c = i;
        let mut dc = sq(i);
        let e0 = self.inedges[i];
        let mut e = e0;
        loop {
            let t = self.d.triangles[e];
            let dt = sq(t);
            if dt < dc {
                dc = dt;
                c = t;
            }
            e = if e % 3 == 2 { e - 2 } else { e + 1 };
            if self.d.triangles[e] != i {
                break;
            }
            e = self.d.halfedges[e];
            if e == NONE {
                let hull = &self.d.hull;
                let h = hull[(self.hull_index[i] + 1) % hull.len()];
                if h != t && sq(h) < dc {
                    return h;
                }
                break;
            }
            if e == e0 {
                break;
            }
        }
        c
    }
}
