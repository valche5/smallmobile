//! Le réseau de neurones, côté Rust : lecture des poids appris en Python (`lab7/py/model.py`) et calcul
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
//! Calcul : en `f32` (AVX2 et FMA si le processeur les a) ; sur un processeur qui a AVX-512 BF16
//! (AMD Zen 4, machines de calcul AWS), les trois grands produits de chaque couche se font en bf16
//! (poids et entrées arrondis à 8 bits de mantisse, sommes en `f32`) : deux fois plus de produits
//! par instruction, pour un écart de l'ordre du centième (`eval_portable_bf16`, sa référence).
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
    /// Les poids en bf16 par paires d'entrées (`bf16::pack`), pour les produits en bf16 ; vide si
    /// le nombre d'entrées est impair.
    wp: Vec<u32>,
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
        let wp = if inp.is_multiple_of(2) { bf16::pack(&wt, inp, out) } else { Vec::new() };
        Ok(Linear { inp, out, w: wt, b, wp })
    }

    /// `y = b + x · W` pour chacune des `rows` lignes `x` de `xs` (`inp` nombres), dans les lignes `y`
    /// de `ys` (`out` nombres).
    #[inline(always)]
    fn apply_many<const AVX2: bool, const WIDE: bool>(&self, xs: &[f32], rows: usize, ys: &mut [f32]) {
        for y in ys[..rows * self.out].chunks_exact_mut(self.out) {
            y.copy_from_slice(&self.b);
        }
        accumulate_many::<AVX2, WIDE>(self, 0, xs, self.inp, rows, ys);
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

    /// Les logits des clés de chaque position (`-inf` pour les régions absentes) : `g` leurs vecteurs
    /// de partie (`hd` nombres chacun), `hs` leurs lignes de régions.
    #[inline(always)]
    fn logits_many<const AVX2: bool, const WIDE: bool, P: Position>(
        &self,
        g: &[f32],
        hs: &[Vec<f32>],
        items: &[P],
        hd: usize,
    ) -> Vec<Vec<f32>> {
        let n = items.len();
        let mut gp = vec![0.0f32; n * hd];
        self.glob1.apply_many::<AVX2, WIDE>(g, n, &mut gp);
        relu(&mut gp);
        let mut gl = vec![0.0f32; n * GLOBAL_KEYS];
        self.glob2.apply_many::<AVX2, WIDE>(&gp, n, &mut gl);
        let mut all = Vec::with_capacity(n);
        for (k, (p, h)) in items.iter().zip(hs).enumerate() {
            let mut logits = vec![f32::NEG_INFINITY; KEYS];
            logits[..GLOBAL_KEYS].copy_from_slice(&gl[k * GLOBAL_KEYS..(k + 1) * GLOBAL_KEYS]);
            let mut gpol = [0.0f32; REGION_KINDS];
            gpol.copy_from_slice(&self.node.b);
            accumulate_rows::<AVX2>(&self.node, hd, &g[k * hd..(k + 1) * hd], &mut gpol);
            region_policy::<AVX2>(&self.node, h, p.encoded().regions, hd, &gpol, &mut logits);
            all.push(logits);
        }
        all
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

impl Eval {
    /// Ce que disent les têtes du réseau d'une position à `players` joueurs : les logits des coups
    /// et la tête de valeur (5 logits de victoire, 5 gains de pièces ÷ `GAIN_SCALE`).
    pub fn from_heads(players: usize, logits: Vec<f32>, v: &[f32; 10]) -> Eval {
        let mut win = [0.0f32; 5];
        let top = v[..players].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for q in 0..players {
            win[q] = (v[q] - top).exp();
            sum += win[q];
        }
        win.iter_mut().for_each(|w| *w /= sum);
        let mut gain = [0.0f32; 5];
        for q in 0..players {
            gain[q] = v[5 + q] * GAIN_SCALE;
        }
        Eval { win, gain, logits }
    }
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
        // les têtes qui ne servent qu'à l'entraînement (possession des régions) restent dans `t`
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

    /// Évalue une position encodée ; `adj(i)` : voisines de la région `i`. Le calcul le plus rapide
    /// que permet le processeur : bf16 (`eval_bf16`), sinon `eval_f32`.
    pub fn eval_with<'a>(&self, e: &Encoded, adj: impl Fn(usize) -> &'a [u8]) -> Eval {
        self.eval_batch(&[One { e, adj }]).pop().expect("une évaluation")
    }

    /// Évalue plusieurs positions, couche après couche : chaque couche passe sur toutes les
    /// positions avant la suivante, si bien que ses poids restent dans le cache du cœur d'une
    /// position à l'autre (le réseau entier n'y tient pas). Chaque position reçoit les mêmes
    /// opérations, dans le même ordre, qu'évaluée seule : mêmes résultats au bit près. Le calcul le
    /// plus rapide que permet le processeur, comme `eval_with`.
    pub fn eval_batch<P: Position>(&self, items: &[P]) -> Vec<Eval> {
        #[cfg(target_arch = "x86_64")]
        if bf16::available() && self.bf16_ready() {
            // SAFETY: le processeur a AVX2, FMA et AVX-512 BF16 (vérifié juste avant)
            return unsafe { self.eval_avx512_bf16(items) };
        }
        self.eval_f32_batch(items)
    }

    /// Le calcul en `f32` : avec AVX2 et FMA si le processeur les a. Celui que `lab7/py/check.py`
    /// compare à PyTorch.
    pub fn eval_f32<'a>(&self, e: &Encoded, adj: impl Fn(usize) -> &'a [u8]) -> Eval {
        self.eval_f32_batch(&[One { e, adj }]).pop().expect("une évaluation")
    }

    fn eval_f32_batch<P: Position>(&self, items: &[P]) -> Vec<Eval> {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: le processeur a AVX2 et FMA (vérifié juste avant)
            return unsafe { self.eval_avx2(items) };
        }
        self.eval_generic::<false, false, P>(items)
    }

    /// Le calcul en bf16 si le processeur le permet (AVX-512 BF16), sinon `None`.
    pub fn eval_bf16<'a>(&self, e: &Encoded, adj: impl Fn(usize) -> &'a [u8]) -> Option<Eval> {
        #[cfg(target_arch = "x86_64")]
        if bf16::available() && self.bf16_ready() {
            // SAFETY: le processeur a AVX2, FMA et AVX-512 BF16 (vérifié juste avant)
            return unsafe { self.eval_avx512_bf16(&[One { e, adj }]) }.pop();
        }
        let _ = (e, adj);
        None
    }

    /// Le calcul simple, celui d'un processeur sans AVX2 : la référence du calcul rapide (ses
    /// multiplications matricielles ne fusionnent pas multiplication et addition, d'où de petits
    /// écarts d'arrondi).
    pub fn eval_portable<'a>(&self, e: &Encoded, adj: impl Fn(usize) -> &'a [u8]) -> Eval {
        self.eval_generic::<false, false, _>(&[One { e, adj }]).pop().expect("une évaluation")
    }

    /// La référence du calcul en bf16, sans instruction spéciale : mêmes arrondis et mêmes sommes,
    /// dans le même ordre (`bf16::gemm_reference`).
    pub fn eval_portable_bf16<'a>(&self, e: &Encoded, adj: impl Fn(usize) -> &'a [u8]) -> Eval {
        assert!(self.bf16_ready(), "réseau sans calcul bf16 (largeur)");
        self.eval_generic::<false, true, _>(&[One { e, adj }]).pop().expect("une évaluation")
    }

    /// Les produits en bf16 demandent une largeur multiple de 32 (conversion 32 nombres à la fois).
    fn bf16_ready(&self) -> bool {
        self.hidden.is_multiple_of(32)
    }

    /// Le même calcul avec AVX2, 8 nombres à la fois, sans changer un seul résultat : chaque nombre
    /// reçoit les mêmes opérations, dans le même ordre (`avx2`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn eval_avx2<P: Position>(&self, items: &[P]) -> Vec<Eval> {
        self.eval_generic::<true, false, P>(items)
    }

    /// Le calcul AVX2, avec les grands produits en bf16 (AVX-512 BF16).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,avx512f,avx512bf16")]
    unsafe fn eval_avx512_bf16<P: Position>(&self, items: &[P]) -> Vec<Eval> {
        self.eval_generic::<true, true, P>(items)
    }

    /// `AVX2` : le processeur a AVX2 et FMA. Les multiplications matricielles des couches utilisent
    /// alors FMA (`gemm_dispatch`), le reste des multiplications et additions séparées, comme sans.
    /// `BF16` : les trois grands produits de chaque couche en bf16 (`product`), avec AVX-512 BF16 si
    /// `AVX2`, sinon par leur référence. Les positions avancent ensemble, couche après couche
    /// (`eval_batch`) : chacune a ses lignes de régions (`hs`) ; les vecteurs de partie de toutes,
    /// côte à côte (`g`), passent ensemble dans les couches qui les lisent (`apply_many`). Chaque
    /// position reçoit les mêmes opérations, dans le même ordre, qu'évaluée seule.
    #[inline(always)]
    fn eval_generic<const AVX2: bool, const BF16: bool, P: Position>(&self, items: &[P]) -> Vec<Eval> {
        let (hd, n) = (self.hidden, items.len());
        let mut hs = Vec::with_capacity(n);
        let mut xn = vec![0.0f32; REGIONS * NODE];
        for p in items {
            let (e, r) = (p.encoded(), p.encoded().regions);
            for (xs, row) in xn.as_chunks_mut::<NODE>().0.iter_mut().zip(&e.node[..r]) {
                for ((x, &v), &sc) in xs.iter_mut().zip(row).zip(&self.node_scale) {
                    *x = v as f32 * sc;
                }
            }
            let mut h = vec![0.0f32; r * hd];
            self.node_in.apply_many::<AVX2, BF16>(&xn, r, &mut h);
            hs.push(h);
        }
        let mut xg = vec![0.0f32; n * GLOBAL];
        for (p, xs) in items.iter().zip(xg.as_chunks_mut::<GLOBAL>().0) {
            for ((x, &v), &sc) in xs.iter_mut().zip(&p.encoded().global).zip(&self.global_scale) {
                *x = v as f32 * sc;
            }
        }
        let mut g = vec![0.0f32; n * hd];
        self.glob_in.apply_many::<AVX2, BF16>(&xg, n, &mut g);

        let mut agg = vec![0.0f32; REGIONS * hd];
        let mut mid = vec![0.0f32; REGIONS * hd];
        let mut add = vec![0.0f32; REGIONS * hd];
        let mut gpart = vec![0.0f32; n * hd];
        let mut cat = vec![0.0f32; n * 3 * hd];
        let mut gmid = vec![0.0f32; n * hd];
        let mut gadd = vec![0.0f32; n * hd];
        let mut packed = Vec::new();
        for layer in &self.layers {
            // n1([h, a, g]) : la part de g est la même pour toutes les régions d'une position
            for gp in gpart.chunks_exact_mut(hd) {
                gp.copy_from_slice(&layer.n1.b);
            }
            accumulate_many::<AVX2, BF16>(&layer.n1, 2 * hd, &g, hd, n, &mut gpart);
            for (k, (p, h)) in items.iter().zip(&mut hs).enumerate() {
                let r = p.encoded().regions;
                let (agg, mid, add) = (&mut agg[..r * hd], &mut mid[..r * hd], &mut add[..r * hd]);
                // moyenne des voisines
                for i in 0..r {
                    neighbours::<AVX2>(h, hd, p.adj(i), &mut agg[i * hd..(i + 1) * hd]);
                }
                for i in 0..r {
                    mid[i * hd..(i + 1) * hd].copy_from_slice(&gpart[k * hd..(k + 1) * hd]);
                }
                product::<AVX2, BF16>(h, r, hd, &layer.n1, 0, mid, &mut packed);
                product::<AVX2, BF16>(agg, r, hd, &layer.n1, hd, mid, &mut packed);
                relu(mid);
                for i in 0..r {
                    add[i * hd..(i + 1) * hd].copy_from_slice(&layer.n2.b);
                }
                product::<AVX2, BF16>(mid, r, hd, &layer.n2, 0, add, &mut packed);
                h.iter_mut().zip(add.iter()).for_each(|(a, b)| *a += b);
                let cat = &mut cat[k * 3 * hd..(k + 1) * 3 * hd];
                cat[..hd].copy_from_slice(&g[k * hd..(k + 1) * hd]);
                let (mean, max) = cat[hd..].split_at_mut(hd);
                pool(h, r, hd, mean, max);
            }
            layer.g1.apply_many::<AVX2, BF16>(&cat, n, &mut gmid);
            relu(&mut gmid);
            layer.g2.apply_many::<AVX2, BF16>(&gmid, n, &mut gadd);
            g.iter_mut().zip(&gadd).for_each(|(a, b)| *a += b);
        }

        let logits = self.policy.logits_many::<AVX2, BF16, P>(&g, &hs, items, hd);

        // valeur : val1([g, moyenne h])
        let mut vin = vec![0.0f32; n * 2 * hd];
        let mut max = vec![0.0f32; hd];
        for (k, (p, h)) in items.iter().zip(&hs).enumerate() {
            let vin = &mut vin[k * 2 * hd..(k + 1) * 2 * hd];
            vin[..hd].copy_from_slice(&g[k * hd..(k + 1) * hd]);
            pool(h, p.encoded().regions, hd, &mut vin[hd..], &mut max);
        }
        self.val1.apply_many::<AVX2, BF16>(&vin, n, &mut gmid);
        relu(&mut gmid);
        let mut v = vec![0.0f32; n * 10];
        self.val2.apply_many::<AVX2, BF16>(&gmid, n, &mut v);
        items
            .iter()
            .zip(v.as_chunks::<10>().0)
            .zip(logits)
            .map(|((p, v), logits)| Eval::from_heads(p.encoded().players, logits, v))
            .collect()
    }
}

/// Une position à évaluer (`Net::eval_batch`) : son encodage et les voisines de chaque région.
pub trait Position {
    fn encoded(&self) -> &Encoded;
    fn adj(&self, region: usize) -> &[u8];
}

/// Une position et la carte où elle se joue.
impl Position for (&Encoded, &Map) {
    fn encoded(&self) -> &Encoded {
        self.0
    }

    fn adj(&self, region: usize) -> &[u8] {
        &self.1.regions[region].adj
    }
}

/// Une position seule, ses voisines données par une fonction (`eval_with`).
struct One<'a, 'e, F: Fn(usize) -> &'a [u8]> {
    e: &'e Encoded,
    adj: F,
}

impl<'a, F: Fn(usize) -> &'a [u8]> Position for One<'a, '_, F> {
    fn encoded(&self) -> &Encoded {
        self.e
    }

    fn adj(&self, region: usize) -> &[u8] {
        (self.adj)(region)
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

/// `accumulate_rows` pour `rows` lignes : `ys[r] += xs[r] · W[lignes from..from + len]` (`len`
/// entrées par ligne de `xs`, `l.out` sorties par ligne de `ys`), chaque ligne comme seule. Avec
/// AVX-512 (`WIDE`, le calcul bf16), trois lignes à la fois, toutes leurs sorties dans les
/// registres : chaque ligne de poids, lue une fois, sert aux trois (`avx512::accumulate_many`).
#[inline(always)]
fn accumulate_many<const AVX2: bool, const WIDE: bool>(
    l: &Linear,
    from: usize,
    xs: &[f32],
    len: usize,
    rows: usize,
    ys: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if AVX2 && WIDE && l.out.is_multiple_of(16) && l.out <= 128 {
        assert!(xs.len() >= rows * len && ys.len() >= rows * l.out && l.w.len() >= (from + len) * l.out, "dimensions");
        // SAFETY: AVX-512 vérifié par `eval_batch` (calcul bf16) ; dimensions vérifiées juste avant
        unsafe { avx512::accumulate_many(&l.w, l.out, from, xs, len, rows, ys) };
        return;
    }
    for r in 0..rows {
        accumulate_rows::<AVX2>(l, from, &xs[r * len..(r + 1) * len], &mut ys[r * l.out..(r + 1) * l.out]);
    }
}

/// `c += a · W[lignes from..from + k]` : `a` a `rows` lignes de `k` nombres, `c` a `rows` lignes de
/// `l.out` nombres. En bf16 si `BF16` (`bf16::gemm`, `packed` : place pour `a` arrondie), sinon
/// `gemm_dispatch`.
#[inline(always)]
fn product<const AVX2: bool, const BF16: bool>(
    a: &[f32],
    rows: usize,
    k: usize,
    l: &Linear,
    from: usize,
    c: &mut [f32],
    packed: &mut Vec<u32>,
) {
    let n = l.out;
    if BF16 {
        bf16::gemm::<AVX2>(a, rows, k, &l.wp[from / 2 * n..(from + k) / 2 * n], n, c, packed);
    } else {
        gemm_dispatch::<AVX2>(a, rows, k, &l.w[from * n..(from + k) * n], n, c);
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

/// Les produits en bf16. Un nombre de 32 bits porte une paire de bf16 (entrées `2p` et `2p + 1`,
/// la première dans les 16 bits de poids faible), comme l'instruction `vdpbf16ps` qui ajoute à
/// chaque somme `f32` les produits de deux paires.
mod bf16 {
    /// L'arrondi au plus proche (pair en cas d'égalité) d'un `f32` en bf16, comme
    /// `vcvtne2ps2bf16`.
    pub fn round(x: f32) -> u32 {
        let b = x.to_bits();
        (b + 0x7fff + ((b >> 16) & 1)) >> 16
    }

    fn lo(p: u32) -> f32 {
        f32::from_bits(p << 16)
    }

    fn hi(p: u32) -> f32 {
        f32::from_bits(p & 0xffff_0000)
    }

    /// Les poids `w` (`inp × out`, `inp` pair) par paires d'entrées : `inp / 2 × out` nombres.
    pub fn pack(w: &[f32], inp: usize, out: usize) -> Vec<u32> {
        let mut p = vec![0; inp / 2 * out];
        for k in 0..inp / 2 {
            for o in 0..out {
                p[k * out + o] = round(w[2 * k * out + o]) | round(w[(2 * k + 1) * out + o]) << 16;
            }
        }
        p
    }

    /// Le processeur a AVX2, FMA et AVX-512 BF16.
    #[cfg(target_arch = "x86_64")]
    pub fn available() -> bool {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bf16")
    }

    /// `c += a · b` : `a` (`rows × k`, `k` multiple de 32) arrondie en bf16 dans `packed`, `b` par
    /// paires (`pack`, `k / 2 × n`). Avec AVX-512 BF16 si `HW` (et `n` multiple de 16), sinon
    /// `gemm_reference`.
    #[inline(always)]
    pub fn gemm<const HW: bool>(
        a: &[f32],
        rows: usize,
        k: usize,
        b: &[u32],
        n: usize,
        c: &mut [f32],
        packed: &mut Vec<u32>,
    ) {
        assert!(a.len() >= rows * k && b.len() >= k / 2 * n && c.len() >= rows * n, "dimensions");
        packed.clear();
        packed.resize(rows * k / 2, 0);
        #[cfg(target_arch = "x86_64")]
        if HW && k.is_multiple_of(32) && n.is_multiple_of(16) {
            // SAFETY: AVX-512 BF16 vérifié par `eval_with` ; dimensions vérifiées juste avant
            unsafe {
                super::avx512::pack_rows(a, rows * k, packed);
                super::avx512::gemm(packed, rows, k / 2, b, n, c);
            }
            return;
        }
        for (p, pair) in packed.iter_mut().zip(a[..rows * k].as_chunks::<2>().0) {
            *p = round(pair[0]) | round(pair[1]) << 16;
        }
        gemm_reference(packed, rows, k / 2, b, n, c);
    }

    /// La référence de `vdpbf16ps` : à chaque somme, le produit des seconds éléments des paires puis
    /// celui des premiers, chacun exact (bf16 × bf16 tient dans un `f32`), chaque addition arrondie.
    pub fn gemm_reference(a: &[u32], rows: usize, kp: usize, b: &[u32], n: usize, c: &mut [f32]) {
        for r in 0..rows {
            let cr = &mut c[r * n..(r + 1) * n];
            for (p, &x) in a[r * kp..(r + 1) * kp].iter().enumerate() {
                let (x0, x1) = (lo(x), hi(x));
                for (cv, &w) in cr.iter_mut().zip(&b[p * n..(p + 1) * n]) {
                    *cv += x1 * hi(w);
                    *cv += x0 * lo(w);
                }
            }
        }
    }
}

/// Les produits en bf16 avec AVX-512 BF16 (`bf16`).
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use std::arch::x86_64::*;

    /// `accumulate_many` : trois lignes à la fois, leurs `out` sorties (`16 V`, au plus 128) dans
    /// les registres ; pour chaque entrée, la ligne de poids lue sert aux lignes où l'entrée n'est
    /// pas nulle (multiplication puis addition séparées, comme `avx2::accumulate`).
    #[target_feature(enable = "avx2,fma,avx512f")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn accumulate_many(
        w: &[f32],
        out: usize,
        from: usize,
        xs: &[f32],
        len: usize,
        rows: usize,
        ys: &mut [f32],
    ) {
        let (w, xs, ys) = (w[from * out..].as_ptr(), xs.as_ptr(), ys.as_mut_ptr());
        // SAFETY : `out / 16` vecteurs par ligne, lignes dans les tranches vérifiées par l'appelant
        unsafe {
            match out / 16 {
                8 => rows_wide::<8>(w, out, xs, len, rows, ys),
                7 => rows_wide::<7>(w, out, xs, len, rows, ys),
                6 => rows_wide::<6>(w, out, xs, len, rows, ys),
                5 => rows_wide::<5>(w, out, xs, len, rows, ys),
                4 => rows_wide::<4>(w, out, xs, len, rows, ys),
                3 => rows_wide::<3>(w, out, xs, len, rows, ys),
                2 => rows_wide::<2>(w, out, xs, len, rows, ys),
                _ => rows_wide::<1>(w, out, xs, len, rows, ys),
            }
        }
    }

    #[target_feature(enable = "avx2,fma,avx512f")]
    #[inline]
    unsafe fn rows_wide<const V: usize>(
        w: *const f32,
        out: usize,
        xs: *const f32,
        len: usize,
        rows: usize,
        ys: *mut f32,
    ) {
        let mut r = 0;
        // SAFETY (pour chaque tuile) : lignes `r..r + R` < `rows` (voir `accumulate_many`)
        unsafe {
            while r + 3 <= rows {
                tile_wide::<3, V>(w, out, xs.add(r * len), len, ys.add(r * out));
                r += 3;
            }
            match rows - r {
                1 => tile_wide::<1, V>(w, out, xs.add(r * len), len, ys.add(r * out)),
                2 => tile_wide::<2, V>(w, out, xs.add(r * len), len, ys.add(r * out)),
                _ => {}
            }
        }
    }

    #[target_feature(enable = "avx2,fma,avx512f")]
    #[inline]
    unsafe fn tile_wide<const R: usize, const V: usize>(
        w: *const f32,
        out: usize,
        xs: *const f32,
        len: usize,
        ys: *mut f32,
    ) {
        // SAFETY : `16 V` nombres de chacune des `R` lignes de `ys` et des `len` lignes de `w`, `len`
        // nombres de chacune des `R` lignes de `xs`
        unsafe {
            let mut acc: [[__m512; V]; R] =
                std::array::from_fn(|r| std::array::from_fn(|v| _mm512_loadu_ps(ys.add(r * out + 16 * v))));
            for i in 0..len {
                let row = w.add(i * out);
                for (r, acc) in acc.iter_mut().enumerate() {
                    let xi = *xs.add(r * len + i);
                    if xi != 0.0 {
                        let xv = _mm512_set1_ps(xi);
                        for (v, acc) in acc.iter_mut().enumerate() {
                            *acc = _mm512_add_ps(*acc, _mm512_mul_ps(xv, _mm512_loadu_ps(row.add(16 * v))));
                        }
                    }
                }
            }
            for (r, acc) in acc.iter().enumerate() {
                for (v, acc) in acc.iter().enumerate() {
                    _mm512_storeu_ps(ys.add(r * out + 16 * v), *acc);
                }
            }
        }
    }

    /// Les `len` premiers nombres de `a` (multiple de 32) arrondis en bf16, deux par nombre de
    /// `out`, dans l'ordre.
    #[target_feature(enable = "avx512f,avx512bf16")]
    pub unsafe fn pack_rows(a: &[f32], len: usize, out: &mut [u32]) {
        assert!(len.is_multiple_of(32) && a.len() >= len && out.len() >= len / 2, "dimensions");
        let (a, out) = (a.as_ptr(), out.as_mut_ptr());
        for i in (0..len).step_by(32) {
            // SAFETY : `i + 32 <= len` nombres de `a`, `i / 2 + 16 <= len / 2` de `out`
            unsafe {
                let low = _mm512_loadu_ps(a.add(i));
                let high = _mm512_loadu_ps(a.add(i + 16));
                // les 16 premiers bf16 viennent du second argument
                let both: __m512i = std::mem::transmute(_mm512_cvtne2ps_pbh(high, low));
                _mm512_storeu_si512(out.add(i / 2) as *mut __m512i, both);
            }
        }
    }

    /// `c += a · b` : `a` par paires (`rows × kp`), `b` par paires (`kp × n`, `n` multiple de 16),
    /// par tuiles de 6 lignes × 64 colonnes dont les 24 sommes restent dans les registres.
    #[target_feature(enable = "avx512f,avx512bf16")]
    pub unsafe fn gemm(a: &[u32], rows: usize, kp: usize, b: &[u32], n: usize, c: &mut [f32]) {
        let (a, b, c) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
        for col in (0..n).step_by(64) {
            // SAFETY (pour chaque tuile) : lignes `row..row + R` < `rows`, colonnes `col..col + 16 V`
            // <= `n`, dans les tranches vérifiées par l'appelant
            unsafe {
                match (n - col).min(64) / 16 {
                    4 => rows_of::<4>(a, rows, kp, b.add(col), n, c.add(col)),
                    3 => rows_of::<3>(a, rows, kp, b.add(col), n, c.add(col)),
                    2 => rows_of::<2>(a, rows, kp, b.add(col), n, c.add(col)),
                    _ => rows_of::<1>(a, rows, kp, b.add(col), n, c.add(col)),
                }
            }
        }
    }

    /// Toutes les lignes, `16 V` colonnes.
    #[target_feature(enable = "avx512f,avx512bf16")]
    #[inline]
    unsafe fn rows_of<const V: usize>(a: *const u32, rows: usize, kp: usize, b: *const u32, n: usize, c: *mut f32) {
        let mut row = 0;
        // SAFETY : voir `gemm`
        unsafe {
            while row + 6 <= rows {
                tile::<6, V>(a.add(row * kp), kp, b, n, c.add(row * n));
                row += 6;
            }
            match rows - row {
                1 => tile::<1, V>(a.add(row * kp), kp, b, n, c.add(row * n)),
                2 => tile::<2, V>(a.add(row * kp), kp, b, n, c.add(row * n)),
                3 => tile::<3, V>(a.add(row * kp), kp, b, n, c.add(row * n)),
                4 => tile::<4, V>(a.add(row * kp), kp, b, n, c.add(row * n)),
                5 => tile::<5, V>(a.add(row * kp), kp, b, n, c.add(row * n)),
                _ => {}
            }
        }
    }

    /// Une tuile de `R` lignes × `16 V` colonnes.
    #[target_feature(enable = "avx512f,avx512bf16")]
    #[inline]
    unsafe fn tile<const R: usize, const V: usize>(a: *const u32, kp: usize, b: *const u32, n: usize, c: *mut f32) {
        // SAFETY : les pointeurs restent dans la tuile (voir `gemm`)
        unsafe {
            let mut acc: [[__m512; V]; R] =
                std::array::from_fn(|r| std::array::from_fn(|v| _mm512_loadu_ps(c.add(r * n + 16 * v))));
            for p in 0..kp {
                let w: [__m512bh; V] = std::array::from_fn(|v| {
                    std::mem::transmute(_mm512_loadu_si512(b.add(p * n + 16 * v) as *const __m512i))
                });
                for (r, acc) in acc.iter_mut().enumerate() {
                    let x: __m512bh = std::mem::transmute(_mm512_set1_epi32(*a.add(r * kp + p) as i32));
                    for (acc, &w) in acc.iter_mut().zip(&w) {
                        *acc = _mm512_dpbf16_ps(*acc, x, w);
                    }
                }
            }
            for (r, acc) in acc.iter().enumerate() {
                for (v, &acc) in acc.iter().enumerate() {
                    _mm512_storeu_ps(c.add(r * n + 16 * v), acc);
                }
            }
        }
    }
}

const _: () = assert!(GLOBAL_KEYS + REGION_KINDS * 64 == KEYS && REGIONS <= 64);
