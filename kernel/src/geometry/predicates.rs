//! Prédicat d'orientation exact (Shewchuk, version JavaScript `robust-predicates`) : son signe est
//! toujours juste, même pour des points presque alignés.

const EPSILON: f64 = 1.110_223_024_625_156_5e-16;
const SPLITTER: f64 = 134_217_729.0;
const RESULT_ERR_BOUND: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;
const CCW_ERR_BOUND_A: f64 = (3.0 + 16.0 * EPSILON) * EPSILON;
const CCW_ERR_BOUND_B: f64 = (2.0 + 12.0 * EPSILON) * EPSILON;
const CCW_ERR_BOUND_C: f64 = (9.0 + 64.0 * EPSILON) * EPSILON * EPSILON;

/// Produit exact `a × b` sous forme de deux flottants (partie haute, erreur).
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let c = SPLITTER * a;
    let ahi = c - (c - a);
    let alo = a - ahi;
    let c = SPLITTER * b;
    let bhi = c - (c - b);
    let blo = b - bhi;
    (x, alo * blo - (x - ahi * bhi - alo * bhi - ahi * blo))
}

/// `(s1 + s0) − (t1 + t0)` exactement, en quatre composantes.
fn two_two_diff(s1: f64, s0: f64, t1: f64, t0: f64) -> [f64; 4] {
    let mut out = [0.0; 4];
    let i = s0 - t0;
    let bvirt = s0 - i;
    out[0] = s0 - (i + bvirt) + (bvirt - t0);
    let j = s1 + i;
    let bvirt = j - s1;
    let zero = s1 - (j - bvirt) + (i - bvirt);
    let i = zero - t1;
    let bvirt = zero - i;
    out[1] = zero - (i + bvirt) + (bvirt - t1);
    let u3 = j + i;
    let bvirt = u3 - j;
    out[2] = j - (u3 - bvirt) + (i - bvirt);
    out[3] = u3;
    out
}

/// `fast_expansion_sum_zeroelim` : somme exacte de deux développements ; renvoie la longueur de `h`.
fn sum(e: &[f64], f: &[f64], h: &mut [f64]) -> usize {
    let (elen, flen) = (e.len(), f.len());
    let (mut eindex, mut findex) = (0, 0);
    let at = |xs: &[f64], i: usize| xs.get(i).copied().unwrap_or(0.0);
    let mut enow = e[0];
    let mut fnow = f[0];
    let mut q;
    if (fnow > enow) == (fnow > -enow) {
        q = enow;
        eindex += 1;
        enow = at(e, eindex);
    } else {
        q = fnow;
        findex += 1;
        fnow = at(f, findex);
    }
    let mut hindex = 0;
    if eindex < elen && findex < flen {
        let (qnew, hh);
        if (fnow > enow) == (fnow > -enow) {
            qnew = enow + q;
            hh = q - (qnew - enow);
            eindex += 1;
            enow = at(e, eindex);
        } else {
            qnew = fnow + q;
            hh = q - (qnew - fnow);
            findex += 1;
            fnow = at(f, findex);
        }
        q = qnew;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
        while eindex < elen && findex < flen {
            let (qnew, hh);
            if (fnow > enow) == (fnow > -enow) {
                qnew = q + enow;
                let bvirt = qnew - q;
                hh = q - (qnew - bvirt) + (enow - bvirt);
                eindex += 1;
                enow = at(e, eindex);
            } else {
                qnew = q + fnow;
                let bvirt = qnew - q;
                hh = q - (qnew - bvirt) + (fnow - bvirt);
                findex += 1;
                fnow = at(f, findex);
            }
            q = qnew;
            if hh != 0.0 {
                h[hindex] = hh;
                hindex += 1;
            }
        }
    }
    while eindex < elen {
        let qnew = q + enow;
        let bvirt = qnew - q;
        let hh = q - (qnew - bvirt) + (enow - bvirt);
        eindex += 1;
        enow = at(e, eindex);
        q = qnew;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
    }
    while findex < flen {
        let qnew = q + fnow;
        let bvirt = qnew - q;
        let hh = q - (qnew - bvirt) + (fnow - bvirt);
        findex += 1;
        fnow = at(f, findex);
        q = qnew;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
    }
    if q != 0.0 || hindex == 0 {
        h[hindex] = q;
        hindex += 1;
    }
    hindex
}

fn tail(a: f64, b: f64, ab: f64) -> f64 {
    let bvirt = a - ab;
    a - (ab + bvirt) + (bvirt - b)
}

fn orient2d_adapt(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, detsum: f64) -> f64 {
    let acx = ax - cx;
    let bcx = bx - cx;
    let acy = ay - cy;
    let bcy = by - cy;
    let (s1, s0) = two_product(acx, bcy);
    let (t1, t0) = two_product(acy, bcx);
    let b = two_two_diff(s1, s0, t1, t0);
    let mut det = b.iter().sum::<f64>();
    let errbound = CCW_ERR_BOUND_B * detsum;
    if det >= errbound || -det >= errbound {
        return det;
    }
    let acxtail = tail(ax, cx, acx);
    let bcxtail = tail(bx, cx, bcx);
    let acytail = tail(ay, cy, acy);
    let bcytail = tail(by, cy, bcy);
    if acxtail == 0.0 && acytail == 0.0 && bcxtail == 0.0 && bcytail == 0.0 {
        return det;
    }
    let errbound = CCW_ERR_BOUND_C * detsum + RESULT_ERR_BOUND * det.abs();
    det += (acx * bcytail + bcy * acxtail) - (acy * bcxtail + bcx * acytail);
    if det >= errbound || -det >= errbound {
        return det;
    }
    let term = |a: f64, b: f64, c: f64, d: f64| {
        let (s1, s0) = two_product(a, b);
        let (t1, t0) = two_product(c, d);
        two_two_diff(s1, s0, t1, t0)
    };
    let mut c1 = [0.0; 8];
    let c1len = sum(&b, &term(acxtail, bcy, acytail, bcx), &mut c1);
    let mut c2 = [0.0; 12];
    let c2len = sum(&c1[..c1len], &term(acx, bcytail, acy, bcxtail), &mut c2);
    let mut d = [0.0; 16];
    let dlen = sum(&c2[..c2len], &term(acxtail, bcytail, acytail, bcxtail), &mut d);
    d[dlen - 1]
}

/// Positif si a, b, c tournent dans le sens des aiguilles d'une montre (axe y vers le bas), négatif
/// dans l'autre sens, nul s'ils sont alignés.
pub fn orient2d(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    let detleft = (ay - cy) * (bx - cx);
    let detright = (ax - cx) * (by - cy);
    let det = detleft - detright;
    let detsum = (detleft + detright).abs();
    if det.abs() >= CCW_ERR_BOUND_A * detsum {
        return det;
    }
    -orient2d_adapt(ax, ay, bx, by, cx, cy, detsum)
}

#[cfg(test)]
mod tests {
    use super::orient2d;

    #[test]
    fn exact_sign_for_nearly_collinear_points() {
        assert!(orient2d(0.0, 0.0, 1.0, 1.0, 0.0, 1.0) < 0.0);
        assert!(orient2d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0) > 0.0);
        assert_eq!(orient2d(0.0, 0.0, 1.0, 1.0, 2.0, 2.0), 0.0);
        // presque alignés : le calcul rapide se trompe, le calcul exact non
        let (x, y) = (0.5 + f64::EPSILON, 0.5);
        assert!(orient2d(0.0, 0.0, 1.0, 1.0, x, y) != 0.0);
    }
}
