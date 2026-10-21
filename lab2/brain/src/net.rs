//! Le réseau de neurones, côté Rust : lecture des poids appris en Python (`lab2/py/model.py`) et calcul
//! sur le processeur, une position à la fois. Même architecture que `model.py`, à l'identique :
//!
//! ```text
//! h_i = node_in(x_i · échelle)              g = glob_in(x_g · échelle)
//! L fois :
//!   a_i = moyenne des h_j voisins
//!   h_i += n2(relu(n1([h_i, a_i, g])))
//!   g   += g2(relu(g1([g, moyenne h, max h])))
//! politique : par région  node_pol([h_i, g]) → 9 sortes de coups ; globale glob_pol2(relu(glob_pol1(g)))
//! valeur    : val2(relu(val1([g, moyenne h]))) → 5 logits de victoire, 5 gains de pièces (÷ 50)
//! ```
//!
//! La politique dit le meilleur coup (appris de la politique améliorée de la recherche).
//!
//! Format des poids : « SMNN1 », nombre de tenseurs, puis pour chacun nom, dimensions et `f32`
//! (petit-boutiste). Les poids des couches linéaires sont rangés comme dans PyTorch (sortie × entrée).
//! Les échelles des entrées (`node_scale`, `global_scale`) viennent avec les poids.

use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;

use smallmobile_kernel::Map;

use crate::encode::{Encoded, GLOBAL, NODE, REGIONS};
use crate::moves::{GLOBAL_KEYS, KEYS, KINDS};

/// Sortes de coups par région, dans l'ordre des clés (`moves::Kind`).
pub const REGION_KINDS: usize = KINDS.len();
/// La tête de valeur prédit les pièces gagnées d'ici la fin divisées par ce nombre (`model.py`).
const GAIN_SCALE: f32 = 50.0;

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
    #[inline(always)]
    fn apply<const AVX2: bool>(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.inp);
        y.copy_from_slice(&self.b);
        accumulate_rows::<AVX2>(self, 0, x, y);
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

/// Une tête de politique : un logit par clé de coup, par région (`node`) et pour la partie
/// (`glob1`, `glob2`).
struct Policy {
    node: Linear,
    glob1: Linear,
    glob2: Linear,
}

impl Policy {
    fn load(t: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, names: [&str; 3]) -> io::Result<Policy> {
        Ok(Policy {
            node: Linear::load(t, names[0])?,
            glob1: Linear::load(t, names[1])?,
            glob2: Linear::load(t, names[2])?,
        })
    }

    /// Les logits des clés (`-inf` pour les régions absentes) : `g` le vecteur de la partie, `h` les
    /// `r` lignes des régions.
    #[inline(always)]
    fn logits<const AVX2: bool>(&self, g: &[f32], h: &[f32], r: usize, hd: usize) -> Vec<f32> {
        let mut logits = vec![f32::NEG_INFINITY; KEYS];
        let mut gp = vec![0.0f32; hd];
        self.glob1.apply::<AVX2>(g, &mut gp);
        relu(&mut gp);
        let mut gl = vec![0.0f32; GLOBAL_KEYS];
        self.glob2.apply::<AVX2>(&gp, &mut gl);
        logits[..GLOBAL_KEYS].copy_from_slice(&gl);
        let mut gpol = [0.0f32; REGION_KINDS];
        gpol.copy_from_slice(&self.node.b);
        accumulate_rows::<AVX2>(&self.node, hd, g, &mut gpol);
        region_policy::<AVX2>(&self.node, h, r, hd, &gpol, &mut logits);
        logits
    }
}

pub struct Net {
    pub hidden: usize,
    node_in: Linear,
    glob_in: Linear,
    layers: Vec<Layer>,
    policy: Policy,
    val1: Linear,
    val2: Linear,
    node_scale: [f32; NODE],
    global_scale: Vec<f32>,
}

/// Ce que le réseau dit d'une position, du point de vue du joueur qui agit (joueurs relatifs).
#[derive(Clone, Debug)]
pub struct Eval {
    /// Probabilité de victoire de chaque joueur relatif (somme 1 sur les joueurs présents).
    pub win: [f32; 5],
    /// Pièces que chaque joueur relatif gagnera encore d'ici la fin.
    pub gain: [f32; 5],
    /// Logit de chaque clé de coup (`moves::Key`) : le meilleur coup selon le réseau.
    pub logits: Vec<f32>,
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

impl Net {
    /// Lit un fichier de poids.
    pub fn load(path: &Path) -> io::Result<Net> {
        Net::from_bytes(&std::fs::read(path)?)
    }

    /// Lit des poids au format SMNN1.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Net> {
        let mut f = bytes;
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
            let data = bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
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
            policy: Policy::load(t, ["node_pol", "glob_pol1", "glob_pol2"])?,
            val1: Linear::load(t, "val1")?,
            val2: Linear::load(t, "val2")?,
            node_scale: scale(t, "node_scale")?,
            global_scale: t.remove("global_scale").ok_or_else(|| missing("global_scale"))?.1,
        };
        if net.node_in.inp != NODE
            || net.glob_in.inp != GLOBAL
            || net.global_scale.len() != GLOBAL
            || net.policy.glob2.out != GLOBAL_KEYS
        {
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
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: le processeur a AVX2 et FMA (vérifié juste avant)
            return unsafe { self.eval_avx2(e, adj) };
        }
        self.eval_generic::<false>(e, adj)
    }

    /// Le calcul simple, celui d'un processeur sans AVX2 : la référence du calcul rapide (ses
    /// multiplications matricielles ne fusionnent pas multiplication et addition, d'où de petits
    /// écarts d'arrondi).
    pub fn eval_portable<'a>(&self, e: &Encoded, adj: impl Fn(usize) -> &'a [u8]) -> Eval {
        self.eval_generic::<false>(e, adj)
    }

    /// Le même calcul avec AVX2, 8 nombres à la fois, sans changer un seul résultat : chaque nombre
    /// reçoit les mêmes opérations, dans le même ordre (`avx2`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn eval_avx2<'a>(&self, e: &Encoded, adj: impl Fn(usize) -> &'a [u8]) -> Eval {
        self.eval_generic::<true>(e, adj)
    }

    /// `AVX2` : le processeur a AVX2 et FMA. Les multiplications matricielles des couches utilisent
    /// alors FMA (`gemm_dispatch`), le reste des multiplications et additions séparées, comme sans.
    #[inline(always)]
    fn eval_generic<'a, const AVX2: bool>(&self, e: &Encoded, adj: impl Fn(usize) -> &'a [u8]) -> Eval {
        let hd = self.hidden;
        let r = e.regions;
        let mut h = vec![0.0f32; r * hd];
        let mut x = [0.0f32; GLOBAL];
        for i in 0..r {
            let mut xn = [0.0f32; NODE];
            for ((x, &v), &sc) in xn.iter_mut().zip(&e.node[i]).zip(&self.node_scale) {
                *x = v as f32 * sc;
            }
            self.node_in.apply::<AVX2>(&xn, &mut h[i * hd..(i + 1) * hd]);
        }
        for ((x, &v), &sc) in x.iter_mut().zip(&e.global).zip(&self.global_scale) {
            *x = v as f32 * sc;
        }
        let mut g = vec![0.0f32; hd];
        self.glob_in.apply::<AVX2>(&x, &mut g);

        let mut agg = vec![0.0f32; r * hd];
        let mut mid = vec![0.0f32; r * hd];
        let mut add = vec![0.0f32; r * hd];
        let mut cat = vec![0.0f32; 3 * hd];
        let mut gmid = vec![0.0f32; hd];
        let mut gadd = vec![0.0f32; hd];
        let mut gpart = vec![0.0f32; hd];
        for layer in &self.layers {
            // moyenne des voisines
            for i in 0..r {
                neighbours::<AVX2>(&h, hd, adj(i), &mut agg[i * hd..(i + 1) * hd]);
            }
            // n1([h, a, g]) : la part de g est la même pour toutes les régions
            gpart.copy_from_slice(&layer.n1.b);
            accumulate_rows::<AVX2>(&layer.n1, 2 * hd, &g, &mut gpart);
            for i in 0..r {
                mid[i * hd..(i + 1) * hd].copy_from_slice(&gpart);
            }
            let w = &layer.n1.w;
            gemm_dispatch::<AVX2>(&h, r, hd, &w[..hd * hd], hd, &mut mid);
            gemm_dispatch::<AVX2>(&agg, r, hd, &w[hd * hd..2 * hd * hd], hd, &mut mid);
            relu(&mut mid);
            for i in 0..r {
                add[i * hd..(i + 1) * hd].copy_from_slice(&layer.n2.b);
            }
            gemm_dispatch::<AVX2>(&mid, r, hd, &layer.n2.w, hd, &mut add);
            h.iter_mut().zip(&add).for_each(|(a, b)| *a += b);
            cat[..hd].copy_from_slice(&g);
            let (mean, max) = cat[hd..].split_at_mut(hd);
            pool(&h, r, hd, mean, max);
            layer.g1.apply::<AVX2>(&cat, &mut gmid);
            relu(&mut gmid);
            layer.g2.apply::<AVX2>(&gmid, &mut gadd);
            g.iter_mut().zip(&gadd).for_each(|(a, b)| *a += b);
        }
        let mut mid = gmid;

        let logits = self.policy.logits::<AVX2>(&g, &h, r, hd);

        cat[..hd].copy_from_slice(&g);
        let (mean, max) = cat[hd..].split_at_mut(hd);
        pool(&h, r, hd, mean, max);
        self.val1.apply::<AVX2>(&cat[..2 * hd], &mut mid);
        relu(&mut mid);
        let mut v = [0.0f32; 10];
        self.val2.apply::<AVX2>(&mid, &mut v);
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

/// `y += x · W[lignes from..from + len(x)]` : une partie des entrées d'une couche, en sautant les
/// entrées nulles (fréquentes : présences, one-hot).
#[inline(always)]
fn accumulate_rows<const AVX2: bool>(l: &Linear, from: usize, x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if AVX2 && l.out.is_multiple_of(8) {
        assert!(y.len() == l.out && l.w.len() >= (from + x.len()) * l.out, "dimensions");
        // SAFETY: AVX2 vérifié par `eval_with` ; dimensions vérifiées juste avant
        unsafe { avx2::accumulate(&l.w, l.out, from, x, y) };
        return;
    }
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
/// Avec AVX2 : multiplication-addition en une instruction (FMA). Sans FMA matériel (WebAssembly),
/// `mul_add` serait émulé, bien plus lent qu'une multiplication suivie d'une addition.
#[inline(always)]
fn gemm_dispatch<const AVX2: bool>(a: &[f32], rows: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    let tiled = matches!(n, 32 | 48 | 96) || n.is_multiple_of(64);
    #[cfg(target_arch = "x86_64")]
    if AVX2 && tiled {
        assert!(a.len() >= rows * k && b.len() >= k * n && c.len() >= rows * n, "dimensions");
        // SAFETY: AVX2 et FMA vérifiés par `eval_with` ; dimensions vérifiées juste avant
        unsafe { gemm_avx2(a, rows, k, b, n, c) };
        return;
    }
    match n {
        _ if tiled => {
            for i in 0..rows {
                let ci = &mut c[i * n..(i + 1) * n];
                // pas de test des zéros : un branchement imprévisible coûte plus que les calculs évités
                for (kk, &x) in a[i * k..(i + 1) * k].iter().enumerate() {
                    let bk = &b[kk * n..(kk + 1) * n];
                    for (cv, &bv) in ci.iter_mut().zip(bk) {
                        *cv += x * bv;
                    }
                }
            }
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

/// `gemm_dispatch` en AVX2 et FMA, par tuiles de 6 lignes × 16 colonnes : leurs 12 accumulateurs
/// restent dans les registres, et chaque morceau de ligne de `b` chargé sert à 6 lignes de `a` (ligne
/// par ligne, le calcul attendait la mémoire). Chaque case de `c` reçoit ses produits dans le même
/// ordre (`k` croissant), par la même instruction FMA, qu'en la calculant seule : le résultat ne
/// change pas, au bit près. `n` est un multiple de 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_avx2(a: &[f32], rows: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    let (a, b, c) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    for col in (0..n).step_by(16) {
        let mut row = 0;
        // SAFETY (pour chaque tuile) : lignes `row..row + R` < `rows` et colonnes `col..col + 16` <= `n`,
        // dans les tranches vérifiées par l'appelant
        while row + 6 <= rows {
            unsafe { tile::<6>(a.add(row * k), k, b.add(col), n, c.add(row * n + col)) };
            row += 6;
        }
        unsafe {
            match rows - row {
                1 => tile::<1>(a.add(row * k), k, b.add(col), n, c.add(row * n + col)),
                2 => tile::<2>(a.add(row * k), k, b.add(col), n, c.add(row * n + col)),
                3 => tile::<3>(a.add(row * k), k, b.add(col), n, c.add(row * n + col)),
                4 => tile::<4>(a.add(row * k), k, b.add(col), n, c.add(row * n + col)),
                5 => tile::<5>(a.add(row * k), k, b.add(col), n, c.add(row * n + col)),
                _ => {}
            }
        }
    }
}

/// Une tuile de `R` lignes × 16 colonnes : `a` pointe sur sa première ligne (`k` colonnes), `b` et `c`
/// sur sa première colonne (`n` colonnes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn tile<const R: usize>(a: *const f32, k: usize, b: *const f32, n: usize, c: *mut f32) {
    use std::arch::x86_64::*;
    // SAFETY : les pointeurs restent dans la tuile (voir `gemm_avx2`)
    unsafe {
        let mut lo: [__m256; R] = std::array::from_fn(|r| _mm256_loadu_ps(c.add(r * n)));
        let mut hi: [__m256; R] = std::array::from_fn(|r| _mm256_loadu_ps(c.add(r * n + 8)));
        for kk in 0..k {
            let b0 = _mm256_loadu_ps(b.add(kk * n));
            let b1 = _mm256_loadu_ps(b.add(kk * n + 8));
            for r in 0..R {
                let x = _mm256_set1_ps(*a.add(r * k + kk));
                lo[r] = _mm256_fmadd_ps(x, b0, lo[r]);
                hi[r] = _mm256_fmadd_ps(x, b1, hi[r]);
            }
        }
        for r in 0..R {
            _mm256_storeu_ps(c.add(r * n), lo[r]);
            _mm256_storeu_ps(c.add(r * n + 8), hi[r]);
        }
    }
}

#[inline(always)]
fn relu(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = v.max(0.0));
}

/// La politique de chaque région : `node_pol` sur sa ligne de `h`, plus `gpol` (la part commune).
#[inline(always)]
fn region_policy<const AVX2: bool>(
    node_pol: &Linear,
    h: &[f32],
    r: usize,
    hd: usize,
    gpol: &[f32; REGION_KINDS],
    logits: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if AVX2 {
        assert!(
            node_pol.out == REGION_KINDS && node_pol.w.len() >= hd * REGION_KINDS && h.len() >= r * hd,
            "dimensions"
        );
        // SAFETY: AVX2 vérifié par `eval_with` ; dimensions vérifiées juste avant
        unsafe { avx2::region_policy(&node_pol.w, h, r, hd, gpol, logits) };
        return;
    }
    for i in 0..r {
        let mut out = *gpol;
        accumulate_rows::<AVX2>(node_pol, 0, &h[i * hd..(i + 1) * hd], &mut out);
        for (kind, &v) in out.iter().enumerate() {
            logits[GLOBAL_KEYS + kind * 64 + i] = v;
        }
    }
}

/// Moyenne et maximum des `r` lignes de `h` (`hd` colonnes).
#[inline(always)]
fn pool(h: &[f32], r: usize, hd: usize, mean: &mut [f32], max: &mut [f32]) {
    mean.fill(0.0);
    max.fill(f32::NEG_INFINITY);
    for row in h[..r * hd].chunks_exact(hd) {
        for ((m, x), &v) in mean.iter_mut().zip(max.iter_mut()).zip(row) {
            *m += v;
            *x = x.max(v);
        }
    }
    mean.iter_mut().for_each(|m| *m /= r as f32);
}

/// `out` = moyenne des lignes `adj` de `h` (`hd` colonnes) : leur somme depuis 0, dans l'ordre, fois
/// l'inverse de leur nombre.
#[inline(always)]
fn neighbours<const AVX2: bool>(h: &[f32], hd: usize, adj: &[u8], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if AVX2 && hd.is_multiple_of(8) {
        assert!(out.len() == hd && adj.iter().all(|&j| (j as usize + 1) * hd <= h.len()), "dimensions");
        // SAFETY: AVX2 vérifié par `eval_with` ; dimensions vérifiées juste avant
        unsafe { avx2::mean_rows(h, hd, adj, out) };
        return;
    }
    out.fill(0.0);
    for &j in adj {
        out.iter_mut().zip(&h[j as usize * hd..(j as usize + 1) * hd]).for_each(|(a, b)| *a += b);
    }
    let inv = 1.0 / adj.len().max(1) as f32;
    out.iter_mut().for_each(|a| *a *= inv);
}

/// Les mêmes calculs que leurs versions simples, 8 sorties à la fois : chaque sortie reçoit les mêmes
/// multiplications et additions séparées, dans le même ordre, donc le même résultat au bit près.
/// Jusqu'à 64 sorties restent dans les registres pendant tout le calcul.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// `accumulate_rows` : `w` a `out` colonnes (multiple de 8) et au moins `from + len(x)` lignes.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn accumulate(w: &[f32], out: usize, from: usize, x: &[f32], y: &mut [f32]) {
        let (w, y) = (w.as_ptr(), y.as_mut_ptr());
        for col in (0..out).step_by(64) {
            // SAFETY (pour chaque bloc) : colonnes `col..col + 8 V` <= `out`, lignes vérifiées par l'appelant
            unsafe {
                let (w, y) = (w.add(from * out + col), y.add(col));
                match (out - col).min(64) / 8 {
                    8 => rows::<8>(w, out, x, y),
                    7 => rows::<7>(w, out, x, y),
                    6 => rows::<6>(w, out, x, y),
                    5 => rows::<5>(w, out, x, y),
                    4 => rows::<4>(w, out, x, y),
                    3 => rows::<3>(w, out, x, y),
                    2 => rows::<2>(w, out, x, y),
                    _ => rows::<1>(w, out, x, y),
                }
            }
        }
    }

    /// `8 V` colonnes : `y += x · w`, entrées nulles sautées.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn rows<const V: usize>(w: *const f32, out: usize, x: &[f32], y: *mut f32) {
        // SAFETY : `V` vecteurs de 8 dans `y` et dans chacune des `len(x)` lignes de `w` (voir `accumulate`)
        unsafe {
            let mut acc: [__m256; V] = std::array::from_fn(|v| _mm256_loadu_ps(y.add(8 * v)));
            for (i, &xi) in x.iter().enumerate() {
                if xi != 0.0 {
                    let xv = _mm256_set1_ps(xi);
                    let row = w.add(i * out);
                    for (v, acc) in acc.iter_mut().enumerate() {
                        *acc = _mm256_add_ps(*acc, _mm256_mul_ps(xv, _mm256_loadu_ps(row.add(8 * v))));
                    }
                }
            }
            for (v, acc) in acc.iter().enumerate() {
                _mm256_storeu_ps(y.add(8 * v), *acc);
            }
        }
    }

    /// `region_policy` pour 8 régions à la fois : chacune voit ses entrées dans l'ordre, et une entrée
    /// nulle ne change rien à sa somme, comme dans `accumulate_rows`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn region_policy(
        w: &[f32],
        h: &[f32],
        r: usize,
        hd: usize,
        gpol: &[f32; super::REGION_KINDS],
        logits: &mut [f32],
    ) {
        // les régions côte à côte : entrée par entrée, 8 régions dans un vecteur
        let rp = r.next_multiple_of(8);
        let mut ht = vec![0.0f32; hd * rp];
        for (i, row) in h[..r * hd].chunks_exact(hd).enumerate() {
            for (k, &v) in row.iter().enumerate() {
                ht[k * rp + i] = v;
            }
        }
        for i0 in (0..r).step_by(8) {
            let mut acc: [__m256; super::REGION_KINDS] = std::array::from_fn(|kind| _mm256_set1_ps(gpol[kind]));
            for k in 0..hd {
                // SAFETY : `k * rp + i0 + 8 <= hd * rp` ; `w` a au moins `hd` lignes de `REGION_KINDS` (vérifié par l'appelant)
                let x = unsafe { _mm256_loadu_ps(ht.as_ptr().add(k * rp + i0)) };
                let used = _mm256_cmp_ps::<_CMP_NEQ_UQ>(x, _mm256_setzero_ps());
                for (kind, acc) in acc.iter_mut().enumerate() {
                    let wk = w[k * super::REGION_KINDS + kind];
                    let sum = _mm256_add_ps(*acc, _mm256_mul_ps(x, _mm256_set1_ps(wk)));
                    *acc = _mm256_blendv_ps(*acc, sum, used);
                }
            }
            for (kind, acc) in acc.iter().enumerate() {
                let mut lanes = [0.0f32; 8];
                // SAFETY : 8 nombres dans `lanes`
                unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), *acc) };
                for (lane, &v) in lanes.iter().enumerate().take(r - i0) {
                    logits[super::GLOBAL_KEYS + kind * 64 + i0 + lane] = v;
                }
            }
        }
    }

    /// `neighbours` : `hd` multiple de 8, lignes `adj` dans `h`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn mean_rows(h: &[f32], hd: usize, adj: &[u8], out: &mut [f32]) {
        let inv = _mm256_set1_ps(1.0 / adj.len().max(1) as f32);
        let (h, out) = (h.as_ptr(), out.as_mut_ptr());
        for col in (0..hd).step_by(8) {
            // SAFETY : colonnes `col..col + 8` <= `hd`, lignes vérifiées par l'appelant
            unsafe {
                let mut acc = _mm256_setzero_ps();
                for &j in adj {
                    acc = _mm256_add_ps(acc, _mm256_loadu_ps(h.add(j as usize * hd + col)));
                }
                _mm256_storeu_ps(out.add(col), _mm256_mul_ps(acc, inv));
            }
        }
    }
}

const _: () = assert!(GLOBAL_KEYS + REGION_KINDS * 64 == KEYS && REGIONS <= 64);
