//! Recherche guidée par le réseau (style AlphaZero), sur mondes possibles en boucle ouverte.
//!
//! Chaque itération tire un monde possible (ordre caché des pioches, dés à venir) et descend l'arbre
//! des suites de coups, choisis par PUCT. Arrivé sur un nœud jamais vu, on demande au réseau quels
//! coups sont prometteurs (a priori des coups), puis on joue `DEPTH` tours de joueur où le réseau
//! joue pour tout le monde, et le réseau juge la position atteinte (valeur). Le réseau ne voit que ce
//! que sait le joueur (`encode`).

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Game, Map, Phase};

use crate::encode::{Encoded, GLOBAL, NODE, encode};
use crate::know::{Knowledge, determinize};
use crate::moves::{self, Key, Move};
use crate::net::{Eval, Net};
use crate::outcome::Outcome;
use crate::rand::Rng;

/// Exploration PUCT.
const C_PUCT: f32 = 2.5;
/// Un coup jamais essayé vaut la valeur du nœud (ramenée entre 0 et 1, comme celles des coups)
/// moins cette réduction.
const FPU: f32 = 0.2;
/// Concentration du bruit de Dirichlet à la racine.
const ALPHA: f32 = 0.3;
/// Poids de la probabilité de victoire dans la valeur ; le reste vient de l'avance en pièces prévue
/// sur le meilleur rival (pièces actuelles + pièces que le réseau prévoit d'ici la fin), sigmoïde
/// d'échelle `SCALE`.
const WIN_WEIGHT: f32 = 0.5;
const SCALE: f32 = 8.0;
/// Longueur des fins simulées, en tours de joueur : le réseau y joue pour tous, puis juge.
const DEPTH: usize = 3;

/// Les réglages de la recherche ; les autres sont fixes (constantes ci-dessus).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub iterations: usize,
    /// Part du bruit de Dirichlet à la racine (auto-apprentissage).
    pub noise: f32,
    /// Les adversaires jouent le coup que le réseau leur prévoit d'après ce qu'ils ont fait dans la
    /// partie (sa tête « coup réel », qui a appris des joueurs de tous styles), plutôt que le
    /// meilleur selon lui : 0, jamais ; 1, dans les fins simulées ; 2, aussi comme a priori de leurs
    /// coups dans l'arbre. Sans cette tête, le réseau leur donne le meilleur coup.
    pub behaviour: u8,
    /// Température du tirage des coups prévus des adversaires (0 : toujours le plus probable).
    pub behaviour_temp: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config { iterations: 200, noise: 0.0, behaviour: 0, behaviour_temp: 1.0 }
    }
}

impl Config {
    /// Lit des réglages du type `n=200,bh=1`.
    pub fn parse(text: &str) -> Option<Config> {
        let mut cfg = Config::default();
        for part in text.split(',').filter(|p| !p.is_empty()) {
            let (k, v) = part.split_once('=')?;
            match k {
                "n" => cfg.iterations = v.parse().ok()?,
                "noise" => cfg.noise = v.parse().ok()?,
                "bh" => cfg.behaviour = v.parse().ok()?,
                "bt" => cfg.behaviour_temp = v.parse().ok()?,
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
const NO_CHILD: u32 = u32::MAX;

struct Edge {
    key: Key,
    prior: f32,
    child: u32,
    visits: u32,
    total: Value,
}

#[derive(Default)]
struct Node {
    expanded: bool,
    visits: u32,
    total: Value,
    edges: Vec<Edge>,
}

/// Les évaluations du réseau déjà faites pendant une recherche. Une même position revient souvent
/// (environ un tiers des évaluations : coups joués dans un autre ordre, fins simulées qui se
/// recroisent) ; le réseau répond toujours pareil à la même entrée sur la même carte : reprendre sa
/// réponse ne change rien à la recherche, au bit près. Sur une autre carte, tout est oublié.
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

/// Le résultat d'une recherche : le coup choisi et les visites de chaque coup de la racine.
pub struct Searched {
    pub mv: Move,
    pub visits: Vec<(Key, u32)>,
    /// Pièces finales attendues de chaque joueur (réel) selon la recherche ; NaN si inconnues.
    pub finals: [f32; MAX_PLAYERS],
    /// Part de victoire attendue de chaque joueur (réel) selon la recherche ; NaN si inconnue.
    pub wins: [f32; MAX_PLAYERS],
    /// Pour chaque coup de la racine : a priori du réseau et valeur moyenne pour le joueur qui agit.
    pub details: Vec<(Key, f32, f32)>,
    /// Valeur de la racine pour le joueur qui agit.
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

fn dirichlet(rng: &mut Rng, alpha: f32, n: usize) -> Vec<f32> {
    // Gamma(alpha) par Marsaglia-Tsang sur alpha + 1, corrigé par U^(1/alpha)
    let gamma = |rng: &mut Rng| -> f64 {
        let a = alpha as f64 + 1.0;
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let (u1, u2) = (rng.unit().max(1e-12), rng.unit());
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = rng.unit().max(1e-12);
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                return d * v * rng.unit().max(1e-12).powf(1.0 / alpha as f64);
            }
        }
    };
    let xs: Vec<f64> = (0..n).map(|_| gamma(rng)).collect();
    let sum: f64 = xs.iter().sum::<f64>().max(1e-12);
    xs.iter().map(|x| (x / sum) as f32).collect()
}

/// Le coup préféré du réseau (`temp` = 0), ou tiré selon `softmax(logits / temp)`.
fn pick(legal: &[moves::Choice], logits: &[f32], temp: f32, rng: &mut Rng) -> Move {
    let best =
        legal.iter().max_by(|a, b| logits[a.key as usize].total_cmp(&logits[b.key as usize])).expect("des coups");
    if temp <= 0.0 {
        return best.mv;
    }
    let top = logits[best.key as usize];
    let weights: Vec<f64> = legal.iter().map(|c| (((logits[c.key as usize] - top) / temp) as f64).exp()).collect();
    let mut x = rng.unit() * weights.iter().sum::<f64>();
    for (c, w) in legal.iter().zip(&weights) {
        if x < *w {
            return c.mv;
        }
        x -= w;
    }
    best.mv
}

impl Az {
    pub fn new(cfg: Config, net: Arc<Net>, label: String) -> Self {
        Az { cfg, net, label }
    }

    /// Le nom affiché : le réseau et les réglages.
    pub fn describe(&self) -> String {
        let c = &self.cfg;
        let behaviour =
            if c.behaviour > 0 { format!(", bh={}, bt={}", c.behaviour, c.behaviour_temp) } else { String::new() };
        format!("az({}, n={}{behaviour})", self.label, c.iterations)
    }

    /// Valeur et a priori d'une position (le nœud n'a pas encore d'arêtes). `me` : le joueur qui
    /// cherche.
    #[allow(clippy::too_many_arguments)]
    fn expand(
        &self,
        evals: &mut Evals,
        node: &mut Node,
        w: &Game,
        k: &Knowledge,
        legal: &[moves::Choice],
        me: usize,
        rng: &mut Rng,
    ) -> Value {
        let eval = evals.get(&self.net, &encode(w, k), w.map);
        let actor = w.state.actor().expect("un joueur doit agir");
        let logits = if self.cfg.behaviour >= 2 { self.guess(&eval, actor != me).0 } else { &eval.logits[..] };
        let top = legal.iter().map(|c| logits[c.key as usize]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = legal.iter().map(|c| (logits[c.key as usize] - top).exp()).collect();
        let sum: f32 = exps.iter().sum();
        node.edges = legal
            .iter()
            .zip(&exps)
            .map(|(c, &e)| Edge { key: c.key, prior: e / sum, child: NO_CHILD, visits: 0, total: [0.0; VALUE] })
            .collect();
        node.expanded = true;
        self.simulate(evals, w, k, me, rng)
    }

    /// Les logits qui choisissent le coup du joueur qui agit, et leur température : la politique du
    /// réseau (son meilleur coup), ou pour un adversaire, avec `behaviour`, le coup qu'il lui prévoit.
    fn guess<'e>(&self, eval: &'e Eval, opponent: bool) -> (&'e [f32], f32) {
        match &eval.behaviour {
            Some(b) if opponent && self.cfg.behaviour > 0 => (b, self.cfg.behaviour_temp),
            _ => (&eval.logits, 0.0),
        }
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

    /// La fin simulée : `DEPTH` tours de joueur où le réseau joue pour tous (son coup préféré ; pour
    /// les adversaires, avec `behaviour`, le coup qu'il leur prévoit), puis le réseau juge.
    fn simulate(&self, evals: &mut Evals, w: &Game, k: &Knowledge, me: usize, rng: &mut Rng) -> Value {
        let mut end = *w;
        let mut kw = *k;
        let mut turns = 0;
        let mut current = end.state.current;
        while end.state.phase != Phase::GameOver {
            let reached = turns >= DEPTH;
            let actor = end.state.actor().expect("un joueur doit agir");
            let legal = moves::legal(&end);
            let mv = if legal.len() == 1 && !reached {
                legal[0].mv
            } else {
                let eval = evals.get(&self.net, &encode(&end, &kw), end.map);
                if reached {
                    return self.judge(&end, &eval);
                }
                let (logits, temp) = self.guess(&eval, actor != me);
                pick(&legal, logits, temp, rng)
            };
            kw.play(&mut end, &mv).expect("coup de la fin simulée");
            if end.state.current != current {
                current = end.state.current;
                turns += 1;
            }
        }
        self.terminal(&end)
    }

    /// Joue la partie jusqu'au bout, le réseau choisissant pour tous son coup préféré : les fins de
    /// partie qui rejugent les positions (`lab1 judge`). `w` est un monde possible (`determinize`).
    /// `evals` : les évaluations déjà faites, à garder d'une fin à l'autre d'une même position (les
    /// fins d'une position se ressemblent jusqu'au premier dé ou à la première carte tirée).
    pub fn play_out<'m>(&self, evals: &mut Evals, w: &Game<'m>, k: &Knowledge, rng: &mut Rng) -> Game<'m> {
        let mut end = *w;
        let mut kw = *k;
        while end.state.phase != Phase::GameOver {
            let legal = moves::legal(&end);
            let mv = if legal.len() == 1 {
                legal[0].mv
            } else {
                let eval = evals.get(&self.net, &encode(&end, &kw), end.map);
                pick(&legal, &eval.logits, 0.0, rng)
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
        let w = WIN_WEIGHT;
        let mut v = [0.0; VALUE];
        for q in 0..n {
            v[COINS + q] = finals[q];
            v[WINS + q] = win[q];
            let rival = (0..n).filter(|&o| o != q).map(|o| finals[o]).fold(f32::NEG_INFINITY, f32::max);
            let lead = 1.0 / (1.0 + (-(finals[q] - rival) / SCALE).exp());
            v[q] = w * win[q] + (1.0 - w) * lead;
        }
        v
    }

    pub fn search(&self, root: &Game, k: &Knowledge, rng: &mut Rng) -> Searched {
        self.search_with(root, k, rng, self.cfg.iterations, self.cfg.noise)
    }

    /// Une recherche de `iterations` itérations ; `noise` : part du bruit de Dirichlet à la racine.
    pub fn search_with(&self, root: &Game, k: &Knowledge, rng: &mut Rng, iterations: usize, noise: f32) -> Searched {
        let legal = moves::legal(root);
        let me = root.state.actor().expect("un joueur doit agir");
        if legal.len() == 1 {
            return Searched {
                mv: legal[0].mv,
                visits: vec![(legal[0].key, 1)],
                finals: [f32::NAN; MAX_PLAYERS],
                wins: [f32::NAN; MAX_PLAYERS],
                details: vec![],
                value: f32::NAN,
            };
        }
        let mut nodes: Vec<Node> = vec![Node::default()];
        let mut evals = Evals::default();
        // la racine aussi est jugée dans un monde possible : sa fin simulée ne voit pas les vrais dés
        let world = determinize(root, k, rng);
        let root_value = self.expand(&mut evals, &mut nodes[0], &world, k, &legal, me, rng);
        nodes[0].visits = 1;
        nodes[0].total = root_value;
        if noise > 0.0 {
            let dir = dirichlet(rng, ALPHA, legal.len());
            for (e, d) in nodes[0].edges.iter_mut().zip(dir) {
                e.prior = (1.0 - noise) * e.prior + noise * d;
            }
        }
        let mut path: Vec<(u32, usize)> = Vec::new();
        for _ in 0..iterations {
            let mut w = determinize(root, k, rng);
            let mut kw = *k;
            path.clear();
            let mut node = 0u32;
            let value: Value = loop {
                if w.state.phase == Phase::GameOver {
                    break self.terminal(&w);
                }
                let legal = moves::legal(&w);
                if legal.len() == 1 {
                    kw.play(&mut w, &legal[0].mv).expect("coup forcé");
                    continue;
                }
                let n = &mut nodes[node as usize];
                if !n.expanded {
                    break self.expand(&mut evals, n, &w, &kw, &legal, me, rng);
                }
                let pid = w.state.actor().expect("un joueur doit agir");
                // les coups de ce monde-ci ; un coup jamais vu reçoit un a priori moyen
                let mut seen: Vec<(usize, usize)> = Vec::with_capacity(legal.len());
                for (li, c) in legal.iter().enumerate() {
                    let ei = match n.edges.iter().position(|e| e.key == c.key) {
                        Some(ei) => ei,
                        None => {
                            let prior = 1.0 / legal.len() as f32;
                            n.edges.push(Edge { key: c.key, prior, child: NO_CHILD, visits: 0, total: [0.0; VALUE] });
                            n.edges.len() - 1
                        }
                    };
                    seen.push((li, ei));
                }
                let visits: u32 = seen.iter().map(|&(_, ei)| n.edges[ei].visits).sum();
                let prior_sum: f32 = seen.iter().map(|&(_, ei)| n.edges[ei].prior).sum::<f32>().max(1e-6);
                let parent_q = n.total[pid] / n.visits.max(1) as f32;
                let sqrt_n = (visits as f32 + 1.0).sqrt();
                // valeurs des coups ramenées entre 0 et 1 dans le nœud (comme MuZero) : les petits écarts
                // de valeur comptent autant que l'a priori
                let (mut lo, mut hi) = (parent_q, parent_q);
                for &(_, ei) in &seen {
                    let e = &n.edges[ei];
                    if e.visits > 0 {
                        let q = e.total[pid] / e.visits as f32;
                        lo = lo.min(q);
                        hi = hi.max(q);
                    }
                }
                let norm = |q: f32| if hi - lo > 1e-6 { (q - lo) / (hi - lo) } else { 0.5 };
                let mut best = 0;
                let mut best_score = f32::NEG_INFINITY;
                for (si, &(_, ei)) in seen.iter().enumerate() {
                    let e = &n.edges[ei];
                    let q = if e.visits > 0 {
                        norm(e.total[pid] / e.visits as f32)
                    } else {
                        (norm(parent_q) - FPU).max(0.0)
                    };
                    let u = C_PUCT * e.prior / prior_sum * sqrt_n / (1.0 + e.visits as f32);
                    if q + u > best_score {
                        best_score = q + u;
                        best = si;
                    }
                }
                let (li, ei) = seen[best];
                kw.play(&mut w, &legal[li].mv).expect("coup possible");
                path.push((node, ei));
                let child = n.edges[ei].child;
                node = if child == NO_CHILD {
                    let id = nodes.len() as u32;
                    nodes[node as usize].edges[ei].child = id;
                    nodes.push(Node::default());
                    id
                } else {
                    child
                };
            };
            let leaf = &mut nodes[node as usize];
            leaf.visits += 1;
            leaf.total.iter_mut().zip(value).for_each(|(t, v)| *t += v);
            for &(nd, ei) in &path {
                let n = &mut nodes[nd as usize];
                if nd != node {
                    n.visits += 1;
                    n.total.iter_mut().zip(value).for_each(|(t, v)| *t += v);
                }
                let e = &mut n.edges[ei];
                e.visits += 1;
                e.total.iter_mut().zip(value).for_each(|(t, v)| *t += v);
            }
        }
        let root_node = &nodes[0];
        let visits: Vec<(Key, u32)> = root_node.edges.iter().map(|e| (e.key, e.visits)).collect();
        let details = root_node.edges.iter().map(|e| (e.key, e.prior, e.total[me] / e.visits.max(1) as f32)).collect();
        let best = root_node.edges.iter().max_by_key(|e| (e.visits, (e.prior * 1e6) as u64)).expect("des coups").key;
        let mv = legal.iter().find(|c| c.key == best).expect("coup de la racine").mv;
        let average = |at: usize| -> [f32; MAX_PLAYERS] {
            std::array::from_fn(|q| root_node.total[at + q] / root_node.visits as f32)
        };
        let (finals, wins) = (average(COINS), average(WINS));
        Searched { mv, visits, finals, wins, details, value: root_node.total[me] / root_node.visits as f32 }
    }
}
