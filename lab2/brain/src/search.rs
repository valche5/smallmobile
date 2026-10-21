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

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Game, Map, Phase};

use crate::encode::{Encoded, GLOBAL, NODE, encode};
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
        softmax(self.edges.iter().map(|e| e.logit))
    }

    /// Les valeurs complétées des coups, transformées pour s'ajouter aux logits : un coup jamais
    /// essayé reçoit la valeur mêlée du nœud (valeur du réseau et valeurs des coups essayés,
    /// pondérées par leur probabilité a priori) ; puis toutes sont ramenées entre 0 et 1 et
    /// multipliées par `(VISIT_INIT + visites du coup le plus vu) × VALUE_SCALE`.
    fn completed(&self) -> Vec<f32> {
        let priors = self.priors();
        let visits: u32 = self.edges.iter().map(|e| e.visits).sum();
        let raw = self.raw[self.actor];
        let mixed = if visits == 0 {
            raw
        } else {
            let (mut mass, mut weighted) = (0.0f32, 0.0f32);
            for (e, &p) in self.edges.iter().zip(&priors) {
                if let Some(q) = self.q(e) {
                    mass += p;
                    weighted += p * q;
                }
            }
            (raw + visits as f32 * weighted / mass.max(f32::MIN_POSITIVE)) / (visits as f32 + 1.0)
        };
        let values: Vec<f32> = self.edges.iter().map(|e| self.q(e).unwrap_or(mixed)).collect();
        let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let most = self.edges.iter().map(|e| e.visits).max().unwrap_or(0);
        let scale = (VISIT_INIT + most as f32) * VALUE_SCALE;
        values.iter().map(|&v| scale * (v - lo) / (hi - lo).max(1e-8)).collect()
    }

    /// La politique améliorée : softmax des logits plus les valeurs complétées.
    fn improved(&self) -> Vec<f32> {
        softmax(self.edges.iter().zip(self.completed()).map(|(e, c)| e.logit + c))
    }

    /// Le coup à suivre hors de la racine : celui dont la part des visites est la plus en retard
    /// sur la politique améliorée.
    fn select(&self) -> usize {
        let visits: u32 = self.edges.iter().map(|e| e.visits).sum();
        let improved = self.improved();
        let score = |i: usize| improved[i] - self.edges[i].visits as f32 / (1.0 + visits as f32);
        (0..self.edges.len()).fold(0, |best, i| if score(i) > score(best) { i } else { best })
    }
}

fn softmax(xs: impl Iterator<Item = f32>) -> Vec<f32> {
    let xs: Vec<f32> = xs.collect();
    let top = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|&x| (x - top).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
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

/// Les évaluations du réseau déjà faites pendant une recherche. Une même position peut revenir
/// (coups joués dans un autre ordre) ; le réseau répond toujours pareil à la même entrée sur la
/// même carte : reprendre sa réponse ne change rien à la recherche, au bit près. Sur une autre
/// carte, tout est oublié.
#[derive(Default)]
pub struct Evals {
    map: usize,
    known: HashMap<u64, Vec<Known>>,
}

/// Une position déjà évaluée : son entrée (`Evals::get`) et la réponse du réseau.
type Known = (Box<[u8]>, Rc<Eval>);

impl Evals {
    fn get(&mut self, net: &Net, e: &Encoded, map: &Map) -> Rc<Eval> {
        let at = std::ptr::from_ref(map) as usize;
        if at != self.map {
            self.known.clear();
            self.map = at;
        }
        // ce que lit le réseau
        let mut key = Vec::with_capacity(2 + e.regions * NODE + GLOBAL);
        key.extend_from_slice(&[e.regions as u8, e.players as u8]);
        e.node[..e.regions].iter().for_each(|row| key.extend_from_slice(row));
        key.extend_from_slice(&e.global);
        // une empreinte rapide pour trouver la place ; l'entrée entière pour reconnaître la position
        let hash = key.chunks(8).fold(0u64, |h, c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            (h.rotate_left(5) ^ u64::from_le_bytes(w)).wrapping_mul(0x517c_c1b7_2722_0a95)
        });
        let same = self.known.entry(hash).or_default();
        if let Some((_, eval)) = same.iter().find(|(k, _)| **k == *key) {
            return eval.clone();
        }
        let eval = Rc::new(net.eval(e, map));
        same.push((key.into_boxed_slice(), eval.clone()));
        eval
    }
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

/// Le coup préféré du réseau.
fn best(legal: &[Choice], logits: &[f32]) -> Move {
    legal.iter().max_by(|a, b| logits[a.key as usize].total_cmp(&logits[b.key as usize])).expect("des coups").mv
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
    fn expand(&self, evals: &mut Evals, w: &Game, k: &Knowledge, legal: &[Choice]) -> (Node, Value) {
        let eval = evals.get(&self.net, &encode(w, k), w.map);
        let value = self.judge(w, &eval);
        let edges = legal
            .iter()
            .map(|c| Edge {
                key: c.key,
                mv: c.mv,
                logit: eval.logits[c.key as usize],
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
    fn judge(&self, w: &Game, eval: &Eval) -> Value {
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

    /// Joue la partie jusqu'au bout, le réseau choisissant pour tous son coup préféré : les fins de
    /// partie qui rejugent les positions (`lab2 judge`). `w` est un monde possible (`determinize`).
    /// `evals` : les évaluations déjà faites, à garder d'une fin à l'autre d'une même position (les
    /// fins d'une position se ressemblent jusqu'au premier dé ou à la première carte tirée).
    pub fn play_out<'m>(&self, evals: &mut Evals, w: &Game<'m>, k: &Knowledge) -> Game<'m> {
        let mut end = *w;
        let mut kw = *k;
        while end.state.phase != Phase::GameOver {
            let legal = moves::legal(&end, &kw);
            let mv = if legal.len() == 1 {
                legal[0].mv
            } else {
                best(&legal, &evals.get(&self.net, &encode(&end, &kw), end.map).logits)
            };
            kw.play(&mut end, &mv).expect("coup de la fin de partie");
        }
        end
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
    pub fn search(&self, root: &Game, k: &Knowledge, rng: &mut Rng, explore: bool) -> Searched {
        self.search_with(root, k, rng, self.cfg.iterations, explore)
    }

    pub fn search_with(&self, root: &Game, k: &Knowledge, rng: &mut Rng, iterations: usize, explore: bool) -> Searched {
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
        let mut evals = Evals::default();
        let (root_node, _) = self.expand(&mut evals, root, k, &legal);
        let mut nodes = vec![root_node];
        let noise: Vec<f32> = legal.iter().map(|_| if explore { gumbel(rng) } else { 0.0 }).collect();
        let schedule = considered_visits(CONSIDERED.min(legal.len()), iterations);
        // le score d'un coup de la racine parmi ceux qui ont `visits` visites
        let root_pick = |root: &Node, visits: u32| -> usize {
            let completed = root.completed();
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
                let e = if node == 0 { root_pick(n, considered) } else { n.select() };
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
                let (child, value) = self.expand(&mut evals, &w, &kw, &legal);
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
        let chosen = root_pick(root_node, most);
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
