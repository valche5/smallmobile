//! Le réseau de neurones, côté Rust : lecture des poids appris en Python (`lab/py/model.py`) et calcul
//! sur le processeur, une position à la fois. Même architecture que `model.py`, à l'identique :
//!
//! ```text
//! h_i = node_in(x_i · échelle)              g = glob_in(x_g · échelle)
//! L fois :
//!   a_i = moyenne des h_j voisins
//!   h_i += n2(relu(n1([h_i, a_i, g])))
//!   g   += g2(relu(g1([g, moyenne h, max h])))
//! politique : par région  node_pol([h_i, g]) → 5 sortes de coups ; globale glob_pol2(relu(glob_pol1(g)))
//! valeur    : val2(relu(val1([g, moyenne h]))) → 5 logits de victoire, 5 gains de pièces (÷ 50)
//! ```
//!
//! Format des poids : « SMNN1 », nombre de tenseurs, puis pour chacun nom, dimensions et `f32`
//! (petit-boutiste). Les poids des couches linéaires sont rangés comme dans PyTorch (sortie × entrée).
//! Les échelles des entrées (`node_scale`, `global_scale`) viennent avec les poids.

use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;

use smallmobile_kernel::Map;

use crate::encode::{Encoded, GLOBAL, NODE, REGIONS};
use crate::moves::KEYS;

/// Sortes de coups par région, dans l'ordre des clés (`moves::REGION`).
pub const REGION_KINDS: usize = 5;
/// La tête de valeur prédit les pièces gagnées d'ici la fin divisées par ce nombre (`model.py`).
const GAIN_SCALE: f32 = 50.0;
/// Clés qui ne désignent pas une région.
pub const GLOBAL_KEYS: usize = 128;

/// Une couche linéaire, poids rangés entrée × sortie pour le calcul.
struct Linear {
    inp: usize,
    out: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Linear {
    fn load(t: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> io::Result<Linear> {
        let (dims, w) = t.remove(&format!("{name}.weight")).ok_or_else(|| missing(name))?;
        let (_, b) = t.remove(&format!("{name}.bias")).ok_or_else(|| missing(name))?;
        let (out, inp) = (dims[0], dims[1]);
        let mut wt = vec![0.0; inp * out];
        for o in 0..out {
            for i in 0..inp {
                wt[i * out + o] = w[o * inp + i];
            }
        }
        Ok(Linear { inp, out, w: wt, b })
    }

    /// `y = b + x · W` (y de taille `out`).
    fn apply(&self, x: &[f32], y: &mut [f32]) {
        y.copy_from_slice(&self.b);
        self.accumulate(x, y);
    }

    /// `y += x · W`, en sautant les entrées nulles (fréquentes : présences, one-hot).
    fn accumulate(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.inp);
        for (i, &xi) in x.iter().enumerate() {
            if xi != 0.0 {
                let row = &self.w[i * self.out..(i + 1) * self.out];
                for (yo, &wo) in y.iter_mut().zip(row) {
                    *yo += xi * wo;
                }
            }
        }
    }
}

fn scale<const N: usize>(t: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> io::Result<[f32; N]> {
    let (_, v) = t.remove(name).ok_or_else(|| missing(name))?;
    v.try_into().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("{name} : mauvaise taille")))
}

fn missing(name: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("poids manquants : {name}"))
}

struct Layer {
    n1: Linear,
    n2: Linear,
    g1: Linear,
    g2: Linear,
}

pub struct Net {
    pub hidden: usize,
    node_in: Linear,
    glob_in: Linear,
    layers: Vec<Layer>,
    node_pol: Linear,
    glob_pol1: Linear,
    glob_pol2: Linear,
    val1: Linear,
    val2: Linear,
    node_scale: [f32; NODE],
    global_scale: [f32; GLOBAL],
}

/// Ce que le réseau dit d'une position, du point de vue du joueur qui agit (joueurs relatifs).
#[derive(Clone, Debug)]
pub struct Eval {
    /// Probabilité de victoire de chaque joueur relatif (somme 1 sur les joueurs présents).
    pub win: [f32; 5],
    /// Pièces que chaque joueur relatif gagnera encore d'ici la fin.
    pub gain: [f32; 5],
    /// Logit de chaque clé de coup (`moves::Key`).
    pub logits: Vec<f32>,
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

impl Net {
    pub fn load(path: &Path) -> io::Result<Net> {
        let mut f = io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0; 5];
        f.read_exact(&mut magic)?;
        if &magic != b"SMNN1" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "pas un fichier de poids SMNN1"));
        }
        let count = read_u32(&mut f)?;
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let len = read_u32(&mut f)? as usize;
            let mut name = vec![0; len];
            f.read_exact(&mut name)?;
            let ndim = read_u32(&mut f)? as usize;
            let dims: Vec<usize> =
                (0..ndim).map(|_| read_u32(&mut f).map(|d| d as usize)).collect::<io::Result<_>>()?;
            let size: usize = dims.iter().product();
            let mut bytes = vec![0u8; size * 4];
            f.read_exact(&mut bytes)?;
            let data = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            tensors.insert(String::from_utf8_lossy(&name).into_owned(), (dims, data));
        }
        let t = &mut tensors;
        let node_in = Linear::load(t, "node_in")?;
        let glob_in = Linear::load(t, "glob_in")?;
        let mut layers = Vec::new();
        while t.contains_key(&format!("layers.{}.n1.weight", layers.len())) {
            let l = layers.len();
            layers.push(Layer {
                n1: Linear::load(t, &format!("layers.{l}.n1"))?,
                n2: Linear::load(t, &format!("layers.{l}.n2"))?,
                g1: Linear::load(t, &format!("layers.{l}.g1"))?,
                g2: Linear::load(t, &format!("layers.{l}.g2"))?,
            });
        }
        let net = Net {
            hidden: node_in.out,
            node_in,
            glob_in,
            layers,
            node_pol: Linear::load(t, "node_pol")?,
            glob_pol1: Linear::load(t, "glob_pol1")?,
            glob_pol2: Linear::load(t, "glob_pol2")?,
            val1: Linear::load(t, "val1")?,
            val2: Linear::load(t, "val2")?,
            node_scale: scale(t, "node_scale")?,
            global_scale: scale(t, "global_scale")?,
        };
        if net.node_in.inp != NODE || net.glob_in.inp != GLOBAL || net.glob_pol2.out != GLOBAL_KEYS {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "réseau d'une autre version des entrées"));
        }
        Ok(net)
    }

    /// Évalue une position encodée de cette carte.
    pub fn eval(&self, e: &Encoded, map: &Map) -> Eval {
        self.eval_with(e, |i| &map.regions[i].adj)
    }

    /// Évalue une position encodée ; `adj(i)` : voisines de la région `i`.
    pub fn eval_with<'a>(&self, e: &Encoded, adj: impl Fn(usize) -> &'a [u8]) -> Eval {
        let hd = self.hidden;
        let r = e.regions;
        let mut h = vec![0.0f32; r * hd];
        let mut x = [0.0f32; GLOBAL];
        for i in 0..r {
            let mut xn = [0.0f32; NODE];
            for ((x, &v), &sc) in xn.iter_mut().zip(&e.node[i]).zip(&self.node_scale) {
                *x = v as f32 * sc;
            }
            self.node_in.apply(&xn, &mut h[i * hd..(i + 1) * hd]);
        }
        for ((x, &v), &sc) in x.iter_mut().zip(&e.global).zip(&self.global_scale) {
            *x = v as f32 * sc;
        }
        let mut g = vec![0.0f32; hd];
        self.glob_in.apply(&x, &mut g);

        let mut agg = vec![0.0f32; r * hd];
        let mut mid = vec![0.0f32; r * hd];
        let mut add = vec![0.0f32; r * hd];
        let mut cat = vec![0.0f32; 3 * hd];
        let mut gmid = vec![0.0f32; hd];
        let mut gadd = vec![0.0f32; hd];
        let mut gpart = vec![0.0f32; hd];
        for layer in &self.layers {
            // moyenne des voisines
            agg.iter_mut().for_each(|a| *a = 0.0);
            for i in 0..r {
                let adj = adj(i);
                let out = &mut agg[i * hd..(i + 1) * hd];
                for &j in adj {
                    let hj = &h[j as usize * hd..(j as usize + 1) * hd];
                    out.iter_mut().zip(hj).for_each(|(a, b)| *a += b);
                }
                let inv = 1.0 / adj.len().max(1) as f32;
                out.iter_mut().for_each(|a| *a *= inv);
            }
            // n1([h, a, g]) : la part de g est la même pour toutes les régions
            gpart.copy_from_slice(&layer.n1.b);
            accumulate_rows(&layer.n1, 2 * hd, &g, &mut gpart);
            for i in 0..r {
                mid[i * hd..(i + 1) * hd].copy_from_slice(&gpart);
            }
            let w = &layer.n1.w;
            gemm_acc(&h, r, hd, &w[..hd * hd], hd, &mut mid);
            gemm_acc(&agg, r, hd, &w[hd * hd..2 * hd * hd], hd, &mut mid);
            relu(&mut mid);
            for i in 0..r {
                add[i * hd..(i + 1) * hd].copy_from_slice(&layer.n2.b);
            }
            gemm_acc(&mid, r, hd, &layer.n2.w, hd, &mut add);
            h.iter_mut().zip(&add).for_each(|(a, b)| *a += b);
            let (mean, max) = pool(&h, r, hd);
            cat[..hd].copy_from_slice(&g);
            cat[hd..2 * hd].copy_from_slice(&mean);
            cat[2 * hd..].copy_from_slice(&max);
            layer.g1.apply(&cat, &mut gmid);
            relu(&mut gmid);
            layer.g2.apply(&gmid, &mut gadd);
            g.iter_mut().zip(&gadd).for_each(|(a, b)| *a += b);
        }
        let mut mid = gmid;

        let mut logits = vec![f32::NEG_INFINITY; KEYS];
        let mut gp = vec![0.0f32; hd];
        self.glob_pol1.apply(&g, &mut gp);
        relu(&mut gp);
        let mut gl = vec![0.0f32; GLOBAL_KEYS];
        self.glob_pol2.apply(&gp, &mut gl);
        logits[..GLOBAL_KEYS].copy_from_slice(&gl);
        let mut gpol = [0.0f32; REGION_KINDS];
        gpol.copy_from_slice(&self.node_pol.b);
        accumulate_rows(&self.node_pol, hd, &g, &mut gpol);
        for i in 0..r {
            let mut out = gpol;
            accumulate_rows(&self.node_pol, 0, &h[i * hd..(i + 1) * hd], &mut out);
            for (kind, &v) in out.iter().enumerate() {
                logits[GLOBAL_KEYS + kind * 64 + i] = v;
            }
        }

        let (mean, _) = pool(&h, r, hd);
        cat[..hd].copy_from_slice(&g);
        cat[hd..2 * hd].copy_from_slice(&mean);
        self.val1.apply(&cat[..2 * hd], &mut mid);
        relu(&mut mid);
        let mut v = [0.0f32; 10];
        self.val2.apply(&mid, &mut v);
        let mut win = [0.0f32; 5];
        let top = v[..e.players].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for q in 0..e.players {
            win[q] = (v[q] - top).exp();
            sum += win[q];
        }
        win.iter_mut().for_each(|w| *w /= sum);
        let mut gain = [0.0f32; 5];
        for q in 0..e.players {
            gain[q] = v[5 + q] * GAIN_SCALE;
        }
        Eval { win, gain, logits }
    }
}

/// `y += x · W[rows from..from+len]` : une partie des entrées d'une couche.
fn accumulate_rows(l: &Linear, from: usize, x: &[f32], y: &mut [f32]) {
    for (i, &xi) in x.iter().enumerate() {
        if xi != 0.0 {
            let row = &l.w[(from + i) * l.out..(from + i + 1) * l.out];
            for (yo, &wo) in y.iter_mut().zip(row) {
                *yo += xi * wo;
            }
        }
    }
}

/// `c += a · b` : `a` est `rows × k`, `b` est `k × n` (entrée × sortie), `c` est `rows × n`.
fn gemm_acc(a: &[f32], rows: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: le processeur a AVX2 et FMA (vérifié juste avant)
        unsafe { gemm_acc_avx2(a, rows, k, b, n, c) };
        return;
    }
    gemm_dispatch(a, rows, k, b, n, c);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_acc_avx2(a: &[f32], rows: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    gemm_dispatch(a, rows, k, b, n, c);
}

/// Une version par largeur courante : l'accumulateur d'une ligne tient alors dans les registres.
#[inline(always)]
fn gemm_dispatch(a: &[f32], rows: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    match n {
        32 => gemm_block::<32>(a, rows, k, b, n, 0, c),
        48 => gemm_block::<48>(a, rows, k, b, n, 0, c),
        // au-delà de 64 colonnes, l'accumulateur ne tient plus dans les registres : par blocs de 64
        _ if n.is_multiple_of(64) => {
            for col in (0..n).step_by(64) {
                gemm_block::<64>(a, rows, k, b, n, col, c);
            }
        }
        96 => {
            gemm_block::<48>(a, rows, k, b, n, 0, c);
            gemm_block::<48>(a, rows, k, b, n, 48, c);
        }
        _ => {
            for i in 0..rows {
                let ci = &mut c[i * n..(i + 1) * n];
                for (kk, &x) in a[i * k..(i + 1) * k].iter().enumerate() {
                    if x != 0.0 {
                        ci.iter_mut().zip(&b[kk * n..(kk + 1) * n]).for_each(|(cv, &bv)| *cv += x * bv);
                    }
                }
            }
        }
    }
}

/// Les colonnes `col..col + N` de `c += a · b` (`b` et `c` ont `n` colonnes).
#[inline(always)]
fn gemm_block<const N: usize>(a: &[f32], rows: usize, k: usize, b: &[f32], n: usize, col: usize, c: &mut [f32]) {
    for i in 0..rows {
        let at = i * n + col;
        let mut acc: [f32; N] = c[at..at + N].try_into().expect("bloc de N");
        // pas de test des zéros : un branchement imprévisible coûte plus que les calculs évités
        for (kk, &x) in a[i * k..(i + 1) * k].iter().enumerate() {
            let bk: &[f32; N] = b[kk * n + col..kk * n + col + N].try_into().expect("bloc de N");
            for j in 0..N {
                acc[j] = x.mul_add(bk[j], acc[j]);
            }
        }
        c[at..at + N].copy_from_slice(&acc);
    }
}

fn relu(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = v.max(0.0));
}

fn pool(h: &[f32], r: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0; hd];
    let mut max = vec![f32::NEG_INFINITY; hd];
    for i in 0..r {
        for k in 0..hd {
            let v = h[i * hd + k];
            mean[k] += v;
            max[k] = max[k].max(v);
        }
    }
    mean.iter_mut().for_each(|m| *m /= r as f32);
    (mean, max)
}

const _: () = assert!(GLOBAL_KEYS + REGION_KINDS * 64 == KEYS && REGIONS <= 64);
