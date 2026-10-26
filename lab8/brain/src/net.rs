//! Le réseau de lab8, sur processeur : la référence de `lab8/py/model.py`, aux arrondis près.
//!
//! ```text
//! encodeur  jetons (tokens) → largeur d ; L couches : x += attn(ln1 x) avec un biais appris par
//!           relation et par tête, x += mlp(ln2 x) (gelu) ; enc_ln
//! têtes     [partie, moyenne des régions] → val1, relu, val2 : 5 logits de victoire, 5 gains de pièces
//!           (÷ GAIN_SCALE) d'ici la fin ; next1, next2 : de même d'ici le prochain tour du joueur ;
//!           own (apprentissage seulement, pas ici)
//! décodeur  un pas par coup : entrée = coup précédent (départ, coup global, ou sorte + région
//!           projetée), + dyn (entrées du pas), + nombre de coups déjà joués sur cette région ;
//!           couches : attention causale, attention aux jetons de l'encodeur, mlp ; dec_ln ;
//!           sorties : 32 coups globaux, et par sorte (9) le produit d'une requête avec chaque région
//! ```
//!
//! Les poids viennent d'un fichier SMNN1 (`model.export`) : tenseurs nommés comme dans PyTorch, avec
//! les échelles des entrées.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::encode::{Encoded, REGIONS};
use crate::moves::{KINDS, Key};
use crate::tokens::{self, DYN, GAME_IN, MARKET, MARKET_IN, PLAYER_IN, PLAYERS, REGION_IN, RELATIONS, TOKENS};

pub use crate::moves::{GLOBAL_KEYS, KEYS};

/// La tête de valeur prédit les pièces gagnées divisées par ce nombre.
pub const GAIN_SCALE: f32 = 50.0;
/// Plafond du nombre de coups déjà joués sur une région (`dec_count`).
pub const COUNTS: usize = 16;
/// Logit d'un coup impossible (région absente), et masque de l'attention.
pub const NEG: f32 = -1e9;
const LN_EPS: f32 = 1e-5;

struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    inp: usize,
    out: usize,
}

impl Linear {
    /// `y = W x + b`.
    fn apply(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.inp);
        for (o, y) in y.iter_mut().enumerate().take(self.out) {
            *y = self.b[o] + dot(&self.w[o * self.inp..][..self.inp], x);
        }
    }

    fn rows(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0.0; rows * self.out];
        for r in 0..rows {
            self.apply(&x[r * self.inp..][..self.inp], &mut y[r * self.out..][..self.out]);
        }
        y
    }
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn apply(&self, x: &[f32], y: &mut [f32]) {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for i in 0..x.len() {
            y[i] = (x[i] - mean) * inv * self.w[i] + self.b[i];
        }
    }

    fn rows(&self, x: &[f32], d: usize) -> Vec<f32> {
        let mut y = vec![0.0; x.len()];
        for (a, b) in x.chunks(d).zip(y.chunks_mut(d)) {
            self.apply(a, b);
        }
        y
    }
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

struct Mlp {
    a: Linear,
    b: Linear,
}

impl Mlp {
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        let mut h = vec![0.0; self.a.out];
        self.a.apply(x, &mut h);
        h.iter_mut().for_each(|v| *v = gelu(*v));
        self.b.apply(&h, out);
    }
}

struct Layer {
    ln1: Norm,
    attn: Attn,
    /// Biais de l'attention, `RELATIONS` × têtes.
    rel: Vec<f32>,
    ln2: Norm,
    mlp: Mlp,
}

struct DecLayer {
    ln1: Norm,
    me: Attn,
    ln2: Norm,
    cross: Attn,
    ln3: Norm,
    mlp: Mlp,
}

pub struct Net {
    pub d: usize,
    pub heads: usize,
    region_in: Linear,
    player_in: Linear,
    market_in: Linear,
    game_in: Linear,
    player_pos: Vec<f32>,
    market_pos: Vec<f32>,
    layers: Vec<Layer>,
    enc_ln: Norm,
    val: (Linear, Linear),
    next: (Linear, Linear),
    dec_start: Vec<f32>,
    dec_glob: Vec<f32>,
    dec_kind: Vec<f32>,
    dec_reg: Linear,
    dec_dyn: Linear,
    dec_count: Vec<f32>,
    dec_layers: Vec<DecLayer>,
    dec_ln: Norm,
    out_glob: Linear,
    out_kind: Linear,
    region_scale: Vec<f32>,
    player_scale: Vec<f32>,
    market_scale: Vec<f32>,
    game_scale: Vec<f32>,
    dyn_scale: Vec<f32>,
}

/// La valeur d'une position, en joueurs relatifs (0 : celui qui agit).
#[derive(Clone, Copy, Debug, Default)]
pub struct Value {
    /// Probabilités de victoire (softmax sur les joueurs présents).
    pub win: [f32; PLAYERS],
    /// Pièces gagnées attendues.
    pub gain: [f32; PLAYERS],
}

impl Value {
    fn from_heads(players: usize, v: &[f32]) -> Value {
        let mut out = Value::default();
        let top = v[..players].iter().copied().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        for (w, &x) in out.win.iter_mut().zip(v).take(players) {
            *w = (x - top).exp();
            sum += *w;
        }
        out.win[..players].iter_mut().for_each(|w| *w /= sum);
        for q in 0..players {
            out.gain[q] = v[PLAYERS + q] * GAIN_SCALE;
        }
        out
    }
}

/// Ce que l'encodeur a lu d'une position : ce dont le décodeur a besoin, et les deux valeurs.
pub struct Memory {
    pub players: usize,
    pub regions: usize,
    valid: [bool; TOKENS],
    /// Sorties de l'encodeur (`TOKENS` × d).
    x: Vec<f32>,
    /// Clés et valeurs de l'attention du décodeur aux jetons, par couche.
    cross: Vec<(Vec<f32>, Vec<f32>)>,
    /// `dec_reg` des régions (`regions` × d) : l'entrée d'un pas après un coup sur cette région.
    reg: Vec<f32>,
    /// Valeur finale, et au prochain tour du joueur qui agit.
    pub value: Value,
    pub next: Value,
    /// Les têtes avant softmax (10 et 10), pour les comparaisons.
    pub heads: [f32; 20],
}

/// Ce que le décodeur garde d'un pas au suivant, pendant un plan.
#[derive(Default)]
pub struct Cache {
    /// Clés et valeurs de l'attention causale, par couche (pas × d).
    me: Vec<(Vec<f32>, Vec<f32>)>,
    steps: usize,
    /// Coups déjà joués sur chaque région pendant le plan.
    counts: Vec<u8>,
}

impl Cache {
    pub fn steps(&self) -> usize {
        self.steps
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // huit sommes partielles : vectorisé par le compilateur
    let mut acc = [0.0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // √(2/π)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

fn softmax(v: &mut [f32]) {
    let top = v.iter().copied().fold(f32::MIN, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - top).exp();
        sum += *x;
    }
    v.iter_mut().for_each(|x| *x /= sum);
}

/// Les poids d'un fichier SMNN1 : nom → (forme, valeurs).
type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn read_tensors(bytes: &[u8]) -> io::Result<Tensors> {
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    let mut at = 0;
    let mut take = |n: usize| -> io::Result<&[u8]> {
        let s = bytes.get(at..at + n).ok_or_else(|| bad("fichier de poids tronqué"))?;
        at += n;
        Ok(s)
    };
    if take(5)? != b"SMNN1" {
        return Err(bad("pas un fichier de poids SMNN1"));
    }
    let u32_at = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap()) as usize;
    let count = u32_at(take(4)?);
    let mut out = HashMap::new();
    for _ in 0..count {
        let len = u32_at(take(4)?);
        let name = String::from_utf8(take(len)?.to_vec()).map_err(|_| bad("nom de tenseur"))?;
        let ndim = u32_at(take(4)?);
        let shape: Vec<usize> = (0..ndim).map(|_| take(4).map(u32_at)).collect::<io::Result<_>>()?;
        let n: usize = shape.iter().product();
        let data = take(4 * n)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        out.insert(name, (shape, data));
    }
    Ok(out)
}

struct Reader(Tensors);

impl Reader {
    fn get(&mut self, name: &str) -> io::Result<(Vec<usize>, Vec<f32>)> {
        self.0
            .remove(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("tenseur absent : {name}")))
    }

    fn vec(&mut self, name: &str, len: usize) -> io::Result<Vec<f32>> {
        let (shape, v) = self.get(name)?;
        if v.len() != len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{name} : forme {shape:?}")));
        }
        Ok(v)
    }

    fn linear(&mut self, name: &str, inp: usize, out: usize) -> io::Result<Linear> {
        Ok(Linear {
            w: self.vec(&format!("{name}.weight"), inp * out)?,
            b: self.vec(&format!("{name}.bias"), out)?,
            inp,
            out,
        })
    }

    fn norm(&mut self, name: &str, d: usize) -> io::Result<Norm> {
        Ok(Norm { w: self.vec(&format!("{name}.weight"), d)?, b: self.vec(&format!("{name}.bias"), d)? })
    }

    fn attn(&mut self, name: &str, d: usize) -> io::Result<Attn> {
        Ok(Attn {
            q: self.linear(&format!("{name}.q"), d, d)?,
            k: self.linear(&format!("{name}.k"), d, d)?,
            v: self.linear(&format!("{name}.v"), d, d)?,
            o: self.linear(&format!("{name}.o"), d, d)?,
        })
    }

    fn mlp(&mut self, name: &str, d: usize) -> io::Result<Mlp> {
        Ok(Mlp { a: self.linear(&format!("{name}.a"), d, 4 * d)?, b: self.linear(&format!("{name}.b"), 4 * d, d)? })
    }

    fn count(&self, prefix: &str) -> usize {
        (0..).take_while(|i| self.0.contains_key(&format!("{prefix}.{i}.ln1.weight"))).count()
    }
}

impl Net {
    pub fn load(path: &Path) -> io::Result<Net> {
        Net::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Net> {
        let mut r = Reader(read_tensors(bytes)?);
        let d = r.0.get("region_in.weight").map(|t| t.0[0]).ok_or_else(|| io::Error::other("region_in absent"))?;
        let heads =
            r.0.get("layers.0.rel.weight").map(|t| t.0[1]).ok_or_else(|| io::Error::other("layers.0 absent"))?;
        let (n_layers, n_dec) = (r.count("layers"), r.count("dec_layers"));
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let p = format!("layers.{i}");
            layers.push(Layer {
                ln1: r.norm(&format!("{p}.ln1"), d)?,
                attn: r.attn(&format!("{p}.attn"), d)?,
                rel: r.vec(&format!("{p}.rel.weight"), RELATIONS * heads)?,
                ln2: r.norm(&format!("{p}.ln2"), d)?,
                mlp: r.mlp(&format!("{p}.mlp"), d)?,
            });
        }
        let mut dec_layers = Vec::new();
        for i in 0..n_dec {
            let p = format!("dec_layers.{i}");
            dec_layers.push(DecLayer {
                ln1: r.norm(&format!("{p}.ln1"), d)?,
                me: r.attn(&format!("{p}.me"), d)?,
                ln2: r.norm(&format!("{p}.ln2"), d)?,
                cross: r.attn(&format!("{p}.cross"), d)?,
                ln3: r.norm(&format!("{p}.ln3"), d)?,
                mlp: r.mlp(&format!("{p}.mlp"), d)?,
            });
        }
        let net = Net {
            d,
            heads,
            region_in: r.linear("region_in", REGION_IN, d)?,
            player_in: r.linear("player_in", PLAYER_IN, d)?,
            market_in: r.linear("market_in", MARKET_IN, d)?,
            game_in: r.linear("game_in", GAME_IN, d)?,
            player_pos: r.vec("player_pos.weight", PLAYERS * d)?,
            market_pos: r.vec("market_pos.weight", MARKET * d)?,
            layers,
            enc_ln: r.norm("enc_ln", d)?,
            val: (r.linear("val1", 2 * d, d)?, r.linear("val2", d, 2 * PLAYERS)?),
            next: (r.linear("next1", 2 * d, d)?, r.linear("next2", d, 2 * PLAYERS)?),
            dec_start: r.vec("dec_start", d)?,
            dec_glob: r.vec("dec_glob.weight", GLOBAL_KEYS * d)?,
            dec_kind: r.vec("dec_kind.weight", KINDS.len() * d)?,
            dec_reg: r.linear("dec_reg", d, d)?,
            dec_dyn: r.linear("dec_dyn", DYN, d)?,
            dec_count: r.vec("dec_count.weight", COUNTS * d)?,
            dec_layers,
            dec_ln: r.norm("dec_ln", d)?,
            out_glob: r.linear("out_glob", d, GLOBAL_KEYS)?,
            out_kind: r.linear("out_kind", d, KINDS.len() * d)?,
            region_scale: r.vec("region_scale", REGION_IN)?,
            player_scale: r.vec("player_scale", PLAYER_IN)?,
            market_scale: r.vec("market_scale", MARKET_IN)?,
            game_scale: r.vec("game_scale", GAME_IN)?,
            dyn_scale: r.vec("dyn_scale", DYN)?,
        };
        // la tête des propriétaires ne sert qu'à l'apprentissage
        r.0.retain(|name, _| !name.starts_with("own."));
        if let Some(name) = r.0.keys().next() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("tenseur inconnu : {name}")));
        }
        Ok(net)
    }

    /// Paramètres (hors échelles et tête des propriétaires).
    pub fn parameters(&self) -> usize {
        let lin = |l: &Linear| l.w.len() + l.b.len();
        let norm = |n: &Norm| n.w.len() + n.b.len();
        let attn = |a: &Attn| lin(&a.q) + lin(&a.k) + lin(&a.v) + lin(&a.o);
        let mlp = |m: &Mlp| lin(&m.a) + lin(&m.b);
        let enc: usize =
            self.layers.iter().map(|l| norm(&l.ln1) + attn(&l.attn) + l.rel.len() + norm(&l.ln2) + mlp(&l.mlp)).sum();
        let dec: usize = self
            .dec_layers
            .iter()
            .map(|l| norm(&l.ln1) + attn(&l.me) + norm(&l.ln2) + attn(&l.cross) + norm(&l.ln3) + mlp(&l.mlp))
            .sum();
        enc + dec
            + lin(&self.region_in)
            + lin(&self.player_in)
            + lin(&self.market_in)
            + lin(&self.game_in)
            + lin(&self.val.0)
            + lin(&self.val.1)
            + lin(&self.next.0)
            + lin(&self.next.1)
            + lin(&self.dec_reg)
            + lin(&self.dec_dyn)
            + lin(&self.out_glob)
            + lin(&self.out_kind)
    }

    /// Les entrées des jetons, mises à l'échelle et projetées (`TOKENS` × d ; zéros pour les absents).
    fn embed(&self, e: &Encoded, valid: &[bool; TOKENS]) -> Vec<f32> {
        let d = self.d;
        let mut x = vec![0.0; TOKENS * d];
        let scaled = |v: &[u8], s: &[f32]| v.iter().zip(s).map(|(&a, &b)| a as f32 * b).collect::<Vec<f32>>();
        for i in 0..e.regions {
            self.region_in.apply(&scaled(&e.node[i], &self.region_scale), &mut x[i * d..][..d]);
        }
        for r in 0..PLAYERS {
            let t = tokens::T_PLAYER + r;
            if valid[t] {
                self.player_in.apply(&scaled(&tokens::player(&e.global, r), &self.player_scale), &mut x[t * d..][..d]);
                x[t * d..][..d].iter_mut().zip(&self.player_pos[r * d..][..d]).for_each(|(a, b)| *a += b);
            }
        }
        for m in 0..MARKET {
            let t = tokens::T_MARKET + m;
            if valid[t] {
                self.market_in.apply(&scaled(tokens::market(&e.global, m), &self.market_scale), &mut x[t * d..][..d]);
                x[t * d..][..d].iter_mut().zip(&self.market_pos[m * d..][..d]).for_each(|(a, b)| *a += b);
            }
        }
        let t = tokens::T_GAME;
        self.game_in.apply(&scaled(tokens::game(&e.global), &self.game_scale), &mut x[t * d..][..d]);
        x
    }

    /// L'encodeur et les têtes de valeur. `dist` : `tokens::distances` de la carte.
    pub fn encode(&self, e: &Encoded, dist: &[u8]) -> Memory {
        let (d, h) = (self.d, self.heads);
        let dh = d / h;
        let valid = tokens::valid(e);
        let rel = tokens::relations(e, dist);
        let live: Vec<usize> = (0..TOKENS).filter(|&t| valid[t]).collect();
        let mut x = self.embed(e, &valid);
        let mut y = vec![0.0; d];
        let mut scores = vec![0.0f32; TOKENS];
        for layer in &self.layers {
            let a = layer.ln1.rows(&x, d);
            let (q, k, v) =
                (layer.attn.q.rows(&a, TOKENS), layer.attn.k.rows(&a, TOKENS), layer.attn.v.rows(&a, TOKENS));
            let mut o = vec![0.0; TOKENS * d];
            let scale = 1.0 / (dh as f32).sqrt();
            for &i in &live {
                for head in 0..h {
                    let qi = &q[i * d + head * dh..][..dh];
                    for (j, s) in scores.iter_mut().enumerate() {
                        *s = if valid[j] {
                            dot(qi, &k[j * d + head * dh..][..dh]) * scale
                                + layer.rel[rel[i * TOKENS + j] as usize * h + head]
                        } else {
                            NEG
                        };
                    }
                    softmax(&mut scores);
                    let out = &mut o[i * d + head * dh..][..dh];
                    for &j in &live {
                        let p = scores[j];
                        for (a, b) in out.iter_mut().zip(&v[j * d + head * dh..][..dh]) {
                            *a += p * b;
                        }
                    }
                }
            }
            for &i in &live {
                layer.attn.o.apply(&o[i * d..][..d], &mut y);
                x[i * d..][..d].iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            }
            let mut n = vec![0.0; d];
            for &i in &live {
                layer.ln2.apply(&x[i * d..][..d], &mut n);
                layer.mlp.apply(&n, &mut y);
                x[i * d..][..d].iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            }
        }
        let mut out = vec![0.0; TOKENS * d];
        for &i in &live {
            self.enc_ln.apply(&x[i * d..][..d], &mut out[i * d..][..d]);
        }
        // les têtes de valeur : [partie, moyenne des régions]
        let mut pooled = vec![0.0; 2 * d];
        pooled[..d].copy_from_slice(&out[tokens::T_GAME * d..][..d]);
        for i in 0..e.regions {
            pooled[d..].iter_mut().zip(&out[i * d..][..d]).for_each(|(a, b)| *a += b / e.regions as f32);
        }
        let head = |(a, b): &(Linear, Linear)| {
            let mut hid = vec![0.0; d];
            a.apply(&pooled, &mut hid);
            hid.iter_mut().for_each(|v| *v = v.max(0.0));
            let mut v = vec![0.0; 2 * PLAYERS];
            b.apply(&hid, &mut v);
            v
        };
        let (val, next) = (head(&self.val), head(&self.next));
        let mut heads = [0.0; 20];
        heads[..10].copy_from_slice(&val);
        heads[10..].copy_from_slice(&next);
        let cross =
            self.dec_layers.iter().map(|l| (l.cross.k.rows(&out, TOKENS), l.cross.v.rows(&out, TOKENS))).collect();
        let reg = self.dec_reg.rows(&out[..e.regions * d], e.regions);
        Memory {
            players: e.players,
            regions: e.regions,
            valid,
            x: out,
            cross,
            reg,
            value: Value::from_heads(e.players, &val),
            next: Value::from_heads(e.players, &next),
            heads,
        }
    }

    /// Un pas du décodeur : les logits des `KEYS` coups (`NEG` pour une région absente), après le coup
    /// `prev` (`None` : premier pas du plan), dans une position aux entrées de pas `dy`.
    pub fn step(&self, mem: &Memory, cache: &mut Cache, prev: Option<Key>, dy: &[u8; DYN]) -> Vec<f32> {
        let (d, h) = (self.d, self.heads);
        let dh = d / h;
        if cache.steps == 0 {
            cache.me = vec![(Vec::new(), Vec::new()); self.dec_layers.len()];
            cache.counts = vec![0; REGIONS];
        }
        // l'entrée du pas
        let mut x = vec![0.0; d];
        let mut count = 0;
        match prev {
            None => x.copy_from_slice(&self.dec_start),
            Some(key) if (key as usize) < GLOBAL_KEYS => x.copy_from_slice(&self.dec_glob[key as usize * d..][..d]),
            Some(key) => {
                let (kind, region) = ((key as usize - GLOBAL_KEYS) / 64, (key as usize - GLOBAL_KEYS) % 64);
                cache.counts[region] = cache.counts[region].saturating_add(1);
                count = (cache.counts[region] as usize).min(COUNTS - 1);
                for ((x, a), b) in x.iter_mut().zip(&self.dec_kind[kind * d..][..d]).zip(&mem.reg[region * d..][..d]) {
                    *x = a + b;
                }
            }
        }
        let scaled: Vec<f32> = dy.iter().zip(&self.dyn_scale).map(|(&a, &b)| a as f32 * b).collect();
        let mut y = vec![0.0; d];
        self.dec_dyn.apply(&scaled, &mut y);
        for i in 0..d {
            x[i] += y[i] + self.dec_count[count * d + i];
        }
        let scale = 1.0 / (dh as f32).sqrt();
        let mut n = vec![0.0; d];
        let mut q = vec![0.0; d];
        let mut o = vec![0.0; d];
        for (l, layer) in self.dec_layers.iter().enumerate() {
            // attention causale : ce pas et les précédents
            layer.ln1.apply(&x, &mut n);
            layer.me.q.apply(&n, &mut q);
            let (keys, values) = &mut cache.me[l];
            let at = keys.len();
            keys.resize(at + d, 0.0);
            values.resize(at + d, 0.0);
            layer.me.k.apply(&n, &mut keys[at..]);
            layer.me.v.apply(&n, &mut values[at..]);
            let steps = keys.len() / d;
            attend(&q, keys, values, steps, None, h, scale, &mut o);
            layer.me.o.apply(&o, &mut y);
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            // attention aux jetons de l'encodeur
            layer.ln2.apply(&x, &mut n);
            layer.cross.q.apply(&n, &mut q);
            let (keys, values) = &mem.cross[l];
            attend(&q, keys, values, TOKENS, Some(&mem.valid), h, scale, &mut o);
            layer.cross.o.apply(&o, &mut y);
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            layer.ln3.apply(&x, &mut n);
            layer.mlp.apply(&n, &mut y);
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
        }
        cache.steps += 1;
        self.dec_ln.apply(&x, &mut n);
        let mut logits = vec![NEG; KEYS];
        self.out_glob.apply(&n, &mut logits[..GLOBAL_KEYS]);
        let mut kq = vec![0.0; KINDS.len() * d];
        self.out_kind.apply(&n, &mut kq);
        let norm = 1.0 / (d as f32).sqrt();
        for kind in 0..KINDS.len() {
            let qk = &kq[kind * d..][..d];
            for r in 0..mem.regions {
                logits[GLOBAL_KEYS + kind * 64 + r] = dot(qk, &mem.x[r * d..][..d]) * norm;
            }
        }
        logits
    }
}

/// Attention d'une requête `q` aux `n` clés et valeurs (lignes de d), têtes séparées ; `valid`
/// masque des clés.
#[allow(clippy::too_many_arguments)]
fn attend(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    n: usize,
    valid: Option<&[bool; TOKENS]>,
    h: usize,
    scale: f32,
    out: &mut [f32],
) {
    let d = q.len();
    let dh = d / h;
    let mut scores = vec![0.0f32; n];
    out.fill(0.0);
    for head in 0..h {
        let qh = &q[head * dh..][..dh];
        for (j, s) in scores.iter_mut().enumerate() {
            *s = if valid.is_none_or(|v| v[j]) { dot(qh, &keys[j * d + head * dh..][..dh]) * scale } else { NEG };
        }
        softmax(&mut scores);
        let oh = &mut out[head * dh..][..dh];
        for (j, &p) in scores.iter().enumerate() {
            if p > 0.0 {
                for (a, b) in oh.iter_mut().zip(&values[j * d + head * dh..][..dh]) {
                    *a += p * b;
                }
            }
        }
    }
}
