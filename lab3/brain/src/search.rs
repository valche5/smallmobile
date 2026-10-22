//! Recherche guidée par le réseau : Gumbel AlphaZero (Danihelka et al., *Policy improvement by
//! planning with Gumbel*, 2022), avec des nœuds de hasard.
//!
//! Chaque nœud est une position publique, telle que la voient les joueurs ; ses arêtes sont les
//! coups possibles. La suite d'un coup se range par résultat public de son hasard (dé, carte arrivée
//! au marché : `moves::chance`) : la recherche ne mêle pas deux résultats, et une décision ne dépend
//! jamais d'un hasard encore inconnu. Chaque itération tire un monde possible (ordre caché des
//! pioches, dés à venir : `determinize`), si bien que les résultats arrivent avec leur vraie
//! probabilité. Les coups forcés se jouent sans nœud.
//!
//! - À la racine, `CONSIDERED` coups sont tirés sans remise selon les logits du réseau plus un bruit
//!   de Gumbel (`explore` ; sans lui, les plus probables), puis départagés par halving séquentiel :
//!   les itérations se répartissent entre les coups retenus, et la moitié la moins bonne est écartée
//!   à chaque étape. Le coup joué est le dernier qui reste.
//! - Ailleurs, la sélection est déterministe : le coup qui rapproche le plus les visites de la
//!   politique améliorée (softmax des logits plus les valeurs des coups, `completed`).
//! - Une position nouvelle est jugée par le réseau (sa valeur), sans fin simulée.
//!
//! Chaque joueur maximise sa propre valeur. La politique améliorée de la racine est la cible de
//! politique de l'auto-apprentissage. Le réseau ne voit que ce que sait le joueur (`encode`).

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Game, Map, Phase};

use crate::encode::{Encoded, encode};
use crate::know::{Knowledge, determinize};
use crate::moves::{self, Choice, Key, Move};
use crate::net::{Eval, Net};
use crate::outcome::Outcome;
use crate::rand::Rng;

/// Coups retenus à la racine.
const CONSIDERED: usize = 16;
/// Les valeurs complétées des coups, ramenées entre 0 et 1 dans le nœud, sont multipliées par
/// `(VISIT_INIT + visites du coup le plus vu) × VALUE_SCALE` (réglages de mctx, la bibliothèque de
/// DeepMind) : plus la recherche avance, plus elles l'emportent sur les logits du réseau.
const VISIT_INIT: f32 = 50.0;
const VALUE_SCALE: f32 = 0.1;
/// Poids de la probabilité de victoire dans la valeur ; le reste vient de l'avance en pièces prévue
/// sur le meilleur rival (pièces actuelles + pièces que le réseau prévoit d'ici la fin), sigmoïde
/// d'échelle `SCALE`.
const WIN_WEIGHT: f32 = 0.5;
const SCALE: f32 = 8.0;

/// Les réglages de la recherche ; les autres sont fixes (constantes ci-dessus).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub iterations: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config { iterations: 200 }
    }
}

impl Config {
    /// Lit des réglages du type `n=200`.
    pub fn parse(text: &str) -> Option<Config> {
        let mut cfg = Config::default();
        for part in text.split(',').filter(|p| !p.is_empty()) {
            let (k, v) = part.split_once('=')?;
            match k {
                "n" => cfg.iterations = v.parse().ok()?,
                _ => return None,
            }
        }
        Some(cfg)
    }
}

/// Par joueur : la valeur (0 à 1), les pièces finales attendues (`COINS + joueur`), puis la part de
/// victoire attendue seule (`WINS + joueur`), sans l'avance en pièces que mêle la valeur.
type Value = [f32; VALUE];
const VALUE: usize = 3 * MAX_PLAYERS;
const COINS: usize = MAX_PLAYERS;
const WINS: usize = 2 * MAX_PLAYERS;
type PerPlayer = [f32; MAX_PLAYERS];

struct Edge {
    key: Key,
    mv: Move,
    logit: f32,
    visits: u32,
    total: Value,
    /// Les positions atteintes, par résultat du hasard du coup et des coups forcés qui suivent
    /// (`moves::chance`) ; une seule pour un coup sans hasard. Aucune si la partie s'y termine.
    children: Vec<(u64, u32)>,
}

struct Node {
    /// Le joueur qui agit.
    actor: usize,
    /// La valeur que le réseau donne à la position, pour chaque joueur.
    raw: PerPlayer,
    visits: u32,
    total: Value,
    edges: Vec<Edge>,
}

impl Node {
    /// La valeur moyenne d'un coup pour le joueur qui agit ; `None` s'il n'a jamais été essayé.
    fn q(&self, e: &Edge) -> Option<f32> {
        (e.visits > 0).then(|| e.total[self.actor] / e.visits as f32)
    }

    /// Les probabilités a priori des coups (softmax des logits).
    fn priors(&self) -> Vec<f32> {
        let mut out = Vec::new();
        self.priors_into(&mut out);
        out
    }

    /// `priors`, dans `out` (sans allocation une fois `out` assez grand).
    fn priors_into(&self, out: &mut Vec<f32>) {
        softmax_into(self.edges.iter().map(|e| e.logit), out);
    }

    /// Les valeurs complétées des coups, transformées pour s'ajouter aux logits : un coup jamais
    /// essayé reçoit la valeur mêlée du nœud (valeur du réseau et valeurs des coups essayés,
    /// pondérées par leur probabilité a priori) ; puis toutes sont ramenées entre 0 et 1 et
    /// multipliées par `(VISIT_INIT + visites du coup le plus vu) × VALUE_SCALE`.
    /// Dans `s.values` (`s.priors` : les probabilités a priori).
    fn completed_into(&self, s: &mut Scratch) {
        self.priors_into(&mut s.priors);
        let priors = &s.priors;
        let visits: u32 = self.edges.iter().map(|e| e.visits).sum();
        let raw = self.raw[self.actor];
        let mixed = if visits == 0 {
            raw
        } else {
            let (mut mass, mut weighted) = (0.0f32, 0.0f32);
            for (e, &p) in self.edges.iter().zip(priors) {
                if let Some(q) = self.q(e) {
                    mass += p;
                    weighted += p * q;
                }
            }
            (raw + visits as f32 * weighted / mass.max(f32::MIN_POSITIVE)) / (visits as f32 + 1.0)
        };
        let values = &mut s.values;
        values.clear();
        values.extend(self.edges.iter().map(|e| self.q(e).unwrap_or(mixed)));
        let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let most = self.edges.iter().map(|e| e.visits).max().unwrap_or(0);
        let scale = (VISIT_INIT + most as f32) * VALUE_SCALE;
        values.iter_mut().for_each(|v| *v = scale * (*v - lo) / (hi - lo).max(1e-8));
    }

    /// La politique améliorée : softmax des logits plus les valeurs complétées.
    fn improved(&self) -> Vec<f32> {
        let mut s = Scratch::default();
        self.improved_into(&mut s);
        s.values
    }

    /// `improved`, dans `s.values`.
    fn improved_into(&self, s: &mut Scratch) {
        self.completed_into(s);
        s.values.iter_mut().zip(&self.edges).for_each(|(c, e)| *c += e.logit);
        softmax_in_place(&mut s.values);
    }

    /// Le coup à suivre hors de la racine : celui dont la part des visites est la plus en retard
    /// sur la politique améliorée.
    fn select(&self, s: &mut Scratch) -> usize {
        let visits: u32 = self.edges.iter().map(|e| e.visits).sum();
        self.improved_into(s);
        let improved = &s.values;
        let score = |i: usize| improved[i] - self.edges[i].visits as f32 / (1.0 + visits as f32);
        (0..self.edges.len()).fold(0, |best, i| if score(i) > score(best) { i } else { best })
    }
}

/// Les tampons des calculs d'un nœud (`Node::select`), gardés d'une simulation à la suivante : ces
/// calculs ne demandent plus d'allocation (une dizaine par nœud traversé auparavant).
#[derive(Default)]
struct Scratch {
    priors: Vec<f32>,
    values: Vec<f32>,
}

/// La softmax des `xs`, dans `out`.
fn softmax_into(xs: impl Iterator<Item = f32>, out: &mut Vec<f32>) {
    out.clear();
    out.extend(xs);
    softmax_in_place(out);
}

/// La softmax de `xs`, à sa place : le maximum, les exponentielles, leur somme, puis chacune divisée
/// par la somme.
fn softmax_in_place(xs: &mut [f32]) {
    let top = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    xs.iter_mut().for_each(|x| *x = (*x - top).exp());
    let sum: f32 = xs.iter().sum();
    xs.iter_mut().for_each(|e| *e /= sum);
}

/// Le calendrier du halving séquentiel (`get_sequence_of_considered_visits` de mctx) : pour chaque
/// itération, le nombre de visites qu'a déjà le coup de la racine à suivre. `m` coups retenus,
/// `n` itérations ; à chaque étape, chaque coup encore retenu reçoit autant de visites, puis la
/// moitié est écartée (il en reste au moins 2).
fn considered_visits(m: usize, n: usize) -> Vec<u32> {
    if m <= 1 {
        return (0..n as u32).collect();
    }
    let log2 = (m as f64).log2().ceil() as usize;
    let mut out = Vec::with_capacity(n);
    let mut visits = vec![0u32; m];
    let mut considered = m;
    while out.len() < n {
        let extra = (n / (log2 * considered)).max(1);
        for _ in 0..extra {
            out.extend_from_slice(&visits[..considered]);
            visits[..considered].iter_mut().for_each(|v| *v += 1);
        }
        considered = (considered / 2).max(2);
    }
    out.truncate(n);
    out
}

/// Un tirage de la loi de Gumbel.
fn gumbel(rng: &mut Rng) -> f32 {
    -(-(rng.unit().max(1e-12)).ln()).ln() as f32
}

/// D'où viennent les évaluations du réseau : tout de suite (`Direct` : le jeu, l'arène), ou
/// groupées avec celles d'autres parties (la génération de parties du labo) ; la recherche attend
/// la sienne.
pub trait Oracle {
    fn eval(&self, net: &Net, e: &Encoded, map: &Map) -> impl Future<Output = Eval>;
}

/// Chaque évaluation tout de suite, seule.
pub struct Direct;

impl Oracle for Direct {
    fn eval(&self, net: &Net, e: &Encoded, map: &Map) -> impl Future<Output = Eval> {
        std::future::ready(net.eval(e, map))
    }
}

/// Le résultat d'un calcul qui n'attend jamais (évaluations `Direct`).
pub fn now<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(out) => out,
        Poll::Pending => unreachable!("une évaluation directe n'attend pas"),
    }
}

/// Les évaluations du réseau déjà faites, gardées d'une recherche à la suivante : un `Evals` par
/// partie (ou par décision). Une même position revient, dans une recherche (coups joués dans un
/// autre ordre) comme d'une recherche à la suivante (la décision d'après repasse par les positions
/// que la précédente a explorées) ; le réseau répond toujours pareil à la même entrée sur la même
/// carte : reprendre sa réponse ne change rien à la recherche, au bit près. Restent celles de la
/// recherche en cours et de la précédente ; sur une autre carte ou avec un autre réseau, tout est
/// oublié. Chacune tient en quelques centaines d'octets (`Judged`, `fingerprint`) : celles de
/// plusieurs parties tiennent ensemble dans le cache d'un cœur.
#[derive(Default)]
pub struct Evals {
    /// la carte et le réseau des évaluations gardées (adresses)
    owner: (usize, usize),
    known: HashMap<u128, Rc<Judged>>,
    /// celles de la recherche précédente, pas encore reprises par la recherche en cours
    older: HashMap<u128, Rc<Judged>>,
}

/// Ce que la recherche garde d'une évaluation : la valeur, et les logits des coups possibles dans
/// l'ordre de `moves::legal`.
struct Judged {
    win: [f32; 5],
    gain: [f32; 5],
    logits: Box<[f32]>,
}

impl Evals {
    /// Une recherche commence : les évaluations de la précédente restent, les plus vieilles s'en
    /// vont.
    fn next(&mut self, net: &Net, map: &Map) {
        let owner = (std::ptr::from_ref(map) as usize, std::ptr::from_ref(net) as usize);
        if owner != self.owner {
            self.owner = owner;
            self.known.clear();
            self.older.clear();
        } else {
            self.older = std::mem::take(&mut self.known);
        }
    }

    async fn get(&mut self, net: &Net, e: &Encoded, map: &Map, legal: &[Choice], oracle: &impl Oracle) -> Rc<Judged> {
        let key = fingerprint(e, legal);
        if let Some(judged) = self.known.get(&key) {
            return judged.clone();
        }
        // reprise de la recherche précédente : l'évaluation passe dans celles de la recherche en cours
        if let Some(judged) = self.older.remove(&key) {
            self.known.insert(key, judged.clone());
            return judged;
        }
        let eval = oracle.eval(net, e, map).await;
        let logits = legal.iter().map(|c| eval.logits[c.key as usize]).collect();
        let judged = Rc::new(Judged { win: eval.win, gain: eval.gain, logits });
        self.known.insert(key, judged.clone());
        judged
    }
}

/// L'empreinte de 128 bits de ce que lit le réseau et des coups possibles : deux hachages à la
/// façon de wyhash (produits de 128 bits repliés), de graines différentes. Deux positions
/// différentes ont la même avec une chance de l'ordre de 2⁻¹²⁸.
fn fingerprint(e: &Encoded, legal: &[Choice]) -> u128 {
    const SECRET: [u64; 4] =
        [0xa076_1d64_78bd_642f, 0xe703_7ed1_a0b4_28db, 0x8ebc_6af0_9c88_c6e3, 0x5899_65cc_7537_4cc3];
    fn mum(a: u64, b: u64) -> u64 {
        let r = u128::from(a) * u128::from(b);
        (r as u64) ^ ((r >> 64) as u64)
    }
    let mut lanes = [SECRET[0], SECRET[1]];
    let mut pair = |w0: u64, w1: u64| {
        lanes[0] = mum(w0 ^ SECRET[2], w1 ^ lanes[0]);
        lanes[1] = mum(w1 ^ SECRET[3], w0 ^ lanes[1]);
    };
    let mut bytes = |bytes: &[u8]| {
        let (pairs, rest) = bytes.as_chunks::<16>();
        for p in pairs {
            let (w0, w1) = p.split_at(8);
            pair(u64::from_le_bytes(w0.try_into().unwrap()), u64::from_le_bytes(w1.try_into().unwrap()));
        }
        if !rest.is_empty() {
            let mut t = [0u8; 16];
            t[..rest.len()].copy_from_slice(rest);
            let (w0, w1) = t.split_at(8);
            pair(
                u64::from_le_bytes(w0.try_into().unwrap()),
                u64::from_le_bytes(w1.try_into().unwrap()) ^ rest.len() as u64,
            );
        }
    };
    bytes(&[e.regions as u8, e.players as u8]);
    e.node[..e.regions].iter().for_each(|row| bytes(row));
    bytes(&e.global);
    let keys: Vec<u8> = legal.iter().flat_map(|c| c.key.to_le_bytes()).collect();
    bytes(&keys);
    (u128::from(mum(lanes[0], SECRET[1])) << 64) | u128::from(mum(lanes[1], SECRET[0]))
}

/// Le résultat d'une recherche.
pub struct Searched {
    pub mv: Move,
    /// La politique améliorée de la racine, par coup : la cible de politique de l'auto-apprentissage.
    pub policy: Vec<(Key, f32)>,
    /// Les visites de chaque coup de la racine.
    pub visits: Vec<(Key, u32)>,
    /// Pièces finales attendues de chaque joueur (réel) selon la recherche ; NaN si inconnues.
    pub finals: [f32; MAX_PLAYERS],
    /// Part de victoire attendue de chaque joueur (réel) selon la recherche ; NaN si inconnue.
    pub wins: [f32; MAX_PLAYERS],
    /// Pour chaque coup de la racine : a priori du réseau et valeur moyenne pour le joueur qui agit.
    pub details: Vec<(Key, f32, f32)>,
    /// Valeur de la racine pour le joueur qui agit ; NaN pour un coup forcé, joué sans chercher.
    pub value: f32,
}

pub struct Az {
    pub cfg: Config,
    pub net: Arc<Net>,
    /// Nom affiché (le fichier de poids).
    pub label: String,
}

/// La valeur du réseau (joueurs relatifs) remise en joueurs réels.
fn absolute(win: &[f32; 5], me: usize, n: usize) -> PerPlayer {
    let mut v = [0.0; MAX_PLAYERS];
    for (r, &w) in win.iter().enumerate().take(n) {
        v[(me + r) % n] = w;
    }
    v
}

/// Mêle au résultat d'un coup celui de son hasard (`moves::chance`).
fn fold(hash: u64, chance: u64) -> u64 {
    (hash ^ chance).wrapping_mul(0x0100_0000_01b3)
}

impl Az {
    pub fn new(cfg: Config, net: Arc<Net>, label: String) -> Self {
        Az { cfg, net, label }
    }

    /// Le nom affiché : le réseau et les réglages.
    pub fn describe(&self) -> String {
        format!("az({}, n={})", self.label, self.cfg.iterations)
    }

    /// Une position nouvelle : ses coups et leurs logits, et la valeur que lui donne le réseau.
    async fn expand(
        &self,
        evals: &mut Evals,
        w: &Game<'_>,
        k: &Knowledge,
        legal: &[Choice],
        oracle: &impl Oracle,
    ) -> (Node, Value) {
        let judged = evals.get(&self.net, &encode(w, k), w.map, legal, oracle).await;
        let value = self.judge(w, &judged);
        let edges = legal
            .iter()
            .zip(&judged.logits)
            .map(|(c, &logit)| Edge {
                key: c.key,
                mv: c.mv,
                logit,
                visits: 0,
                total: [0.0; VALUE],
                children: Vec::new(),
            })
            .collect();
        let actor = w.state.actor().expect("un joueur doit agir");
        let node = Node { actor, raw: std::array::from_fn(|q| value[q]), visits: 1, total: value, edges };
        (node, value)
    }

    /// La valeur que le réseau donne à une position (joueurs réels).
    fn judge(&self, w: &Game, eval: &Judged) -> Value {
        let s = &w.state;
        let me = s.actor().expect("un joueur doit agir");
        let n = s.player_count();
        let mut finals = [0.0f32; MAX_PLAYERS];
        for r in 0..n {
            let q = (me + r) % n;
            finals[q] = s.players[q].coins as f32 + eval.gain[r];
        }
        self.blend(&absolute(&eval.win, me, n), &finals, n)
    }

    /// La valeur d'une partie finie.
    fn terminal(&self, g: &Game) -> Value {
        let o = Outcome::of(g);
        let (mut win, mut finals) = ([0.0; MAX_PLAYERS], [0.0; MAX_PLAYERS]);
        for q in 0..o.players {
            win[q] = o.win[q] as f32;
            finals[q] = o.coins[q] as f32;
        }
        self.blend(&win, &finals, o.players)
    }

    /// La valeur de chaque joueur : victoire et avance en pièces (finales ou prévues) mêlées.
    fn blend(&self, win: &PerPlayer, finals: &PerPlayer, n: usize) -> Value {
        let mut v = [0.0; VALUE];
        for q in 0..n {
            v[COINS + q] = finals[q];
            v[WINS + q] = win[q];
            let rival = (0..n).filter(|&o| o != q).map(|o| finals[o]).fold(f32::NEG_INFINITY, f32::max);
            let lead = 1.0 / (1.0 + (-(finals[q] - rival) / SCALE).exp());
            v[q] = WIN_WEIGHT * win[q] + (1.0 - WIN_WEIGHT) * lead;
        }
        v
    }

    /// Une recherche de `cfg.iterations` itérations. `explore` : bruit de Gumbel à la racine
    /// (auto-apprentissage) ; sans lui, la recherche ne dépend du hasard que par les mondes possibles.
    /// `evals` : les évaluations gardées de la partie (ou de la décision). Chaque évaluation tout de
    /// suite (`Direct`).
    pub fn search(&self, root: &Game, k: &Knowledge, rng: &mut Rng, explore: bool, evals: &mut Evals) -> Searched {
        self.search_with(root, k, rng, self.cfg.iterations, explore, evals)
    }

    pub fn search_with(
        &self,
        root: &Game,
        k: &Knowledge,
        rng: &mut Rng,
        iterations: usize,
        explore: bool,
        evals: &mut Evals,
    ) -> Searched {
        now(self.run(root, k, rng, iterations, explore, evals, &Direct))
    }

    /// La recherche, qui attend chaque évaluation du réseau de `oracle`.
    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        &self,
        root: &Game<'_>,
        k: &Knowledge,
        rng: &mut Rng,
        iterations: usize,
        explore: bool,
        evals: &mut Evals,
        oracle: &impl Oracle,
    ) -> Searched {
        let legal = moves::legal(root, k);
        if legal.len() == 1 {
            return Searched {
                mv: legal[0].mv,
                policy: vec![(legal[0].key, 1.0)],
                visits: vec![(legal[0].key, 0)],
                finals: [f32::NAN; MAX_PLAYERS],
                wins: [f32::NAN; MAX_PLAYERS],
                details: vec![],
                value: f32::NAN,
            };
        }
        evals.next(&self.net, root.map);
        let (root_node, _) = self.expand(evals, root, k, &legal, oracle).await;
        let mut nodes = vec![root_node];
        let noise: Vec<f32> = legal.iter().map(|_| if explore { gumbel(rng) } else { 0.0 }).collect();
        let schedule = considered_visits(CONSIDERED.min(legal.len()), iterations);
        // le score d'un coup de la racine parmi ceux qui ont `visits` visites
        let mut scratch = Scratch::default();
        let root_pick = |root: &Node, visits: u32, s: &mut Scratch| -> usize {
            root.completed_into(s);
            let completed = &s.values;
            let score = |i: usize| noise[i] + root.edges[i].logit + completed[i];
            (0..root.edges.len())
                .filter(|&i| root.edges[i].visits == visits)
                .max_by(|&a, &b| score(a).total_cmp(&score(b)))
                .expect("un coup retenu")
        };
        let mut path: Vec<(u32, usize)> = Vec::new();
        for &considered in &schedule {
            let mut w = determinize(root, k, rng);
            let mut kw = *k;
            path.clear();
            let mut node = 0u32;
            let value: Value = loop {
                let n = &nodes[node as usize];
                let e = if node == 0 { root_pick(n, considered, &mut scratch) } else { n.select(&mut scratch) };
                let mv = n.edges[e].mv;
                path.push((node, e));
                // le coup, puis les coups forcés qui suivent, et le résultat de leur hasard
                kw.play(&mut w, &mv).expect("coup possible");
                let mut hash = fold(0, moves::chance(&w, &mv));
                let legal = loop {
                    if w.state.phase == Phase::GameOver {
                        break None;
                    }
                    let legal = moves::legal(&w, &kw);
                    if legal.len() > 1 {
                        break Some(legal);
                    }
                    kw.play(&mut w, &legal[0].mv).expect("coup forcé");
                    hash = fold(hash, moves::chance(&w, &legal[0].mv));
                };
                let Some(legal) = legal else { break self.terminal(&w) };
                let known = nodes[node as usize].edges[e].children.iter().find(|&&(h, _)| h == hash);
                if let Some(&(_, child)) = known {
                    node = child;
                    continue;
                }
                let (child, value) = self.expand(evals, &w, &kw, &legal, oracle).await;
                let id = nodes.len() as u32;
                nodes.push(child);
                nodes[node as usize].edges[e].children.push((hash, id));
                break value;
            };
            for &(nd, e) in &path {
                let n = &mut nodes[nd as usize];
                n.visits += 1;
                n.total.iter_mut().zip(value).for_each(|(t, v)| *t += v);
                let edge = &mut n.edges[e];
                edge.visits += 1;
                edge.total.iter_mut().zip(value).for_each(|(t, v)| *t += v);
            }
        }
        let root_node = &nodes[0];
        let most = root_node.edges.iter().map(|e| e.visits).max().unwrap_or(0);
        let chosen = root_pick(root_node, most, &mut scratch);
        let me = root_node.actor;
        let priors = root_node.priors();
        let policy = root_node.edges.iter().zip(root_node.improved()).map(|(e, p)| (e.key, p)).collect();
        let visits = root_node.edges.iter().map(|e| (e.key, e.visits)).collect();
        let details =
            root_node.edges.iter().zip(&priors).map(|(e, &p)| (e.key, p, root_node.q(e).unwrap_or(f32::NAN))).collect();
        let average = |at: usize| -> [f32; MAX_PLAYERS] {
            std::array::from_fn(|q| root_node.total[at + q] / root_node.visits as f32)
        };
        let (finals, wins) = (average(COINS), average(WINS));
        let value = root_node.total[me] / root_node.visits as f32;
        Searched { mv: root_node.edges[chosen].mv, policy, visits, finals, wins, details, value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Le calendrier de mctx : 4 coups, 10 itérations → chacun une fois, puis les 2 meilleurs.
    #[test]
    fn sequential_halving_follows_mctx() {
        assert_eq!(considered_visits(4, 10), vec![0, 0, 0, 0, 1, 1, 2, 2, 3, 3]);
        assert_eq!(considered_visits(1, 3), vec![0, 1, 2]);
        // 16 coups, 200 itérations : 3 visites chacun, puis 8 coups à 6, 4 à 12, 2 au reste
        let s = considered_visits(16, 200);
        assert_eq!(s.len(), 200);
        assert_eq!(&s[..16], &[0; 16]);
        assert_eq!(s.iter().filter(|&&v| v == 3).count(), 8);
    }
}
