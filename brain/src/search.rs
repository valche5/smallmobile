//! Recherche guidée par le réseau (style AlphaZero), sur mondes possibles en boucle ouverte.
//!
//! Comme la recherche sans réseau du labo (`lab/src/mcts.rs`), chaque itération tire un monde possible
//! et descend l'arbre des suites de coups. La différence : arrivé sur un nœud jamais vu, on ne finit
//! pas la partie au hasard, on demande au réseau qui va gagner (valeur) et quels coups sont
//! prometteurs (a priori des coups), et l'on choisit ensuite les coups par PUCT. Le réseau ne voit que
//! ce que sait le joueur (`encode`).

use std::sync::Arc;

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Ai, Game, Level, Phase, Restrictions, Tuning};

use crate::encode::encode;
use crate::know::{Knowledge, Returns, determinize};
use crate::moves::{self, Key, Move};
use crate::net::Net;
use crate::opponents::{Beliefs, Model};
use crate::outcome::Outcome;
use crate::rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub iterations: usize,
    pub c_puct: f32,
    /// Un coup jamais essayé vaut la valeur du nœud moins cette réduction.
    pub fpu: f32,
    /// Bruit de Dirichlet à la racine (auto-apprentissage) : part et concentration.
    pub noise: f32,
    pub alpha: f32,
    /// Poids de la probabilité de victoire dans la valeur ; le reste vient de l'avance en pièces
    /// prévue (pièces actuelles + pièces que le réseau prévoit d'ici la fin), sigmoïde d'échelle `scale`.
    pub win_weight: f32,
    pub scale: f32,
    /// Part d'une fin de partie jouée par l'IA Normal dans la valeur d'une position nouvelle (le
    /// reste vient du réseau), comme le premier AlphaGo. 0 : réseau seul.
    pub rollout: f32,
    /// Longueur des fins simulées, en tours de joueur ; ensuite le réseau juge. 0 : jusqu'au bout.
    pub depth: usize,
    /// Nombre de fins simulées par position nouvelle (leur moyenne : moins de bruit, plus de calcul).
    pub rollouts: usize,
    /// Qui joue les fins simulées : 0, l'IA Normal du jeu ; 1, le réseau (d'après sa politique) ;
    /// 2, le réseau pour le joueur qui cherche, l'IA Normal pour les autres ; 3, le réseau pour le
    /// joueur qui cherche, et pour chaque adversaire le modèle qui prévoit le mieux ses coups
    /// (`opponents`, tiré à chaque fin simulée).
    pub net_rollouts: u8,
    /// Fins jouées par le réseau : température du tirage des coups (0 : toujours le préféré).
    pub rollout_temp: f32,
    /// Valeurs des coups ramenées entre 0 et 1 dans chaque nœud (comme MuZero) : les petits écarts
    /// de valeur comptent autant que l'a priori.
    pub normalize: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            iterations: 200,
            c_puct: 1.5,
            fpu: 0.2,
            noise: 0.0,
            alpha: 0.3,
            win_weight: 1.0,
            scale: 8.0,
            rollout: 0.0,
            depth: 0,
            rollouts: 1,
            net_rollouts: 0,
            rollout_temp: 0.0,
            normalize: true,
        }
    }
}

impl Config {
    /// Lit des réglages du type `n=400,c=1.5,fpu=0.2`.
    pub fn parse(text: &str) -> Option<Config> {
        let mut cfg = Config::default();
        for part in text.split(',').filter(|p| !p.is_empty()) {
            let (k, v) = part.split_once('=')?;
            match k {
                "n" => cfg.iterations = v.parse().ok()?,
                "c" => cfg.c_puct = v.parse().ok()?,
                "fpu" => cfg.fpu = v.parse().ok()?,
                "noise" => cfg.noise = v.parse().ok()?,
                "alpha" => cfg.alpha = v.parse().ok()?,
                "w" => cfg.win_weight = v.parse().ok()?,
                "s" => cfg.scale = v.parse().ok()?,
                "roll" => cfg.rollout = v.parse().ok()?,
                "norm" => cfg.normalize = v != "0",
                "depth" => cfg.depth = v.parse().ok()?,
                "rolls" => cfg.rollouts = v.parse::<usize>().ok()?.max(1),
                "rp" => cfg.net_rollouts = v.parse().ok()?,
                "rt" => cfg.rollout_temp = v.parse().ok()?,
                _ => return None,
            }
        }
        Some(cfg)
    }
}

/// Par joueur : la valeur (0 à 1), puis les pièces finales attendues (`COINS + joueur`).
type Value = [f32; 2 * MAX_PLAYERS];
const COINS: usize = MAX_PLAYERS;
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

/// Le résultat d'une recherche : le coup choisi et les visites de chaque coup de la racine.
pub struct Searched {
    pub mv: Move,
    pub visits: Vec<(Key, u32)>,
    /// Pièces finales attendues de chaque joueur (réel) selon la recherche ; NaN si inconnues.
    pub finals: [f32; MAX_PLAYERS],
    /// Pour chaque coup de la racine : a priori du réseau et valeur moyenne pour le joueur qui agit.
    pub details: Vec<(Key, f32, f32)>,
    /// Valeur de la racine pour le joueur qui agit.
    pub value: f32,
}

pub struct Az {
    pub cfg: Config,
    tuning: Tuning,
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
        Az { cfg, tuning: Tuning::default(), net, label }
    }

    /// Le nom affiché : le réseau et les réglages.
    pub fn describe(&self) -> String {
        let c = &self.cfg;
        let mut depth = if c.depth > 0 { format!(", depth={}", c.depth) } else { String::new() };
        if c.rollouts > 1 {
            depth += &format!(", rolls={}", c.rollouts);
        }
        if c.net_rollouts > 0 {
            depth += &format!(", rp={}, rt={}", c.net_rollouts, c.rollout_temp);
        }
        format!("az({}, n={}, c={}, w={}, roll={}{depth})", self.label, c.iterations, c.c_puct, c.win_weight, c.rollout)
    }

    /// Valeur et a priori d'une position (le nœud n'a pas encore d'arêtes).
    /// `me` : le joueur qui cherche ; `beliefs` : ce qu'il sait du jeu des autres.
    #[allow(clippy::too_many_arguments)]
    fn expand(
        &self,
        node: &mut Node,
        w: &Game,
        k: &Knowledge,
        legal: &[moves::Choice],
        me: usize,
        beliefs: &Beliefs,
        rng: &mut Rng,
    ) -> Value {
        let eval = self.net.eval(&encode(w, k.race_tail), w.map);
        let top = legal.iter().map(|c| eval.logits[c.key as usize]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = legal.iter().map(|c| (eval.logits[c.key as usize] - top).exp()).collect();
        let sum: f32 = exps.iter().sum();
        node.edges = legal
            .iter()
            .zip(&exps)
            .map(|(c, &e)| Edge {
                key: c.key,
                prior: e / sum,
                child: NO_CHILD,
                visits: 0,
                total: [0.0; 2 * MAX_PLAYERS],
            })
            .collect();
        node.expanded = true;
        let v = self.judge(w, &eval);
        let lambda = self.cfg.rollout;
        if lambda <= 0.0 {
            return v;
        }
        let mut r = [0.0; 2 * MAX_PLAYERS];
        for i in 0..self.cfg.rollouts {
            // la première dans ce monde-ci, les suivantes dans d'autres mondes possibles (pioches, dés)
            let world = if i == 0 { *w } else { determinize(w, k, rng) };
            let one = if self.cfg.net_rollouts > 0 {
                self.simulate_net(&world, k, me, beliefs, rng)
            } else {
                self.simulate(&world, k)
            };
            r.iter_mut().zip(one).for_each(|(a, b)| *a += b / self.cfg.rollouts as f32);
        }
        let mut out = [0.0; 2 * MAX_PLAYERS];
        for (o, (a, b)) in out.iter_mut().zip(v.iter().zip(r)) {
            *o = (1.0 - lambda) * a + lambda * b;
        }
        out
    }

    /// La valeur que le réseau donne à une position (joueurs réels).
    fn judge(&self, w: &Game, eval: &crate::net::Eval) -> Value {
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

    /// L'IA Normal joue la suite pour tout le monde : jusqu'au bout, ou `depth` tours de joueur puis
    /// le réseau juge.
    fn simulate(&self, w: &Game, k: &Knowledge) -> Value {
        let mut end = *w;
        for p in end.state.players.as_mut_slice() {
            p.level = Level::Normal;
        }
        let ai = Ai::new(&self.tuning);
        let mut turns = 0;
        let mut current = end.state.current;
        while end.state.phase != Phase::GameOver {
            if self.cfg.depth > 0 && turns >= self.cfg.depth {
                let eval = self.net.eval(&encode(&end, k.race_tail), end.map);
                return self.judge(&end, &eval);
            }
            let a = ai.decide(&end, Restrictions::default()).expect("l'IA du jeu trouve un coup");
            end.apply(&a, &mut smallmobile_kernel::NoJournal).expect("coup de l'IA du jeu");
            if end.state.current != current {
                current = end.state.current;
                turns += 1;
            }
        }
        self.terminal(&end)
    }

    /// Comme `simulate`, mais les coups sont ceux que préfère le réseau (tirés selon sa politique si
    /// `rollout_temp` > 0) : la suite d'un joueur fort plutôt que celle de l'IA Normal. Avec
    /// `net_rollouts` = 2, seul `me` joue ainsi ; l'IA Normal joue pour les autres ; avec 3, chaque
    /// adversaire joue selon un modèle tiré d'après `beliefs`.
    fn simulate_net(&self, w: &Game, k: &Knowledge, me: usize, beliefs: &Beliefs, rng: &mut Rng) -> Value {
        let who: [Model; MAX_PLAYERS] = std::array::from_fn(|q| match self.cfg.net_rollouts {
            _ if q == me => Model::Net,
            1 => Model::Net,
            2 => Model::Normal,
            _ => beliefs.sample(q, rng),
        });
        let mut end = *w;
        for p in end.state.players.as_mut_slice() {
            p.level = Level::Normal;
        }
        let ai = Ai::new(&self.tuning);
        let mut kw = *k;
        let mut turns = 0;
        let mut current = end.state.current;
        while end.state.phase != Phase::GameOver {
            let reached = self.cfg.depth > 0 && turns >= self.cfg.depth;
            let actor = end.state.actor().expect("un joueur doit agir");
            if !reached && who[actor] == Model::Normal {
                let a = ai.decide(&end, Restrictions::default()).expect("l'IA du jeu trouve un coup");
                let mut ret = Returns::default();
                moves::play(&mut end, &Move::Act(a), &mut ret).expect("coup de l'IA du jeu");
                kw.observe(ret.0, &end.state);
                if end.state.current != current {
                    current = end.state.current;
                    turns += 1;
                }
                continue;
            }
            let legal = moves::legal(&end);
            let mv = if legal.len() == 1 && !reached {
                legal[0].mv
            } else {
                let eval = self.net.eval(&encode(&end, kw.race_tail), end.map);
                if reached {
                    return self.judge(&end, &eval);
                }
                pick(&legal, &eval.logits, self.cfg.rollout_temp, rng)
            };
            let mut ret = Returns::default();
            moves::play(&mut end, &mv, &mut ret).expect("coup de la fin simulée");
            kw.observe(ret.0, &end.state);
            if end.state.current != current {
                current = end.state.current;
                turns += 1;
            }
        }
        self.terminal(&end)
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
        let w = self.cfg.win_weight;
        let mut v = [0.0; 2 * MAX_PLAYERS];
        for q in 0..n {
            v[COINS + q] = finals[q];
            let rival = (0..n).filter(|&o| o != q).map(|o| finals[o]).fold(f32::NEG_INFINITY, f32::max);
            let lead = 1.0 / (1.0 + (-(finals[q] - rival) / self.cfg.scale).exp());
            v[q] = w * win[q] + (1.0 - w) * lead;
        }
        v
    }

    pub fn search(&self, root: &Game, k: &Knowledge, rng: &mut Rng) -> Searched {
        self.search_with(root, k, rng, self.cfg.iterations, self.cfg.noise)
    }

    /// Une recherche qui tient compte de ce qu'on sait du jeu des adversaires (`net_rollouts` = 3).
    pub fn search_against(&self, root: &Game, k: &Knowledge, beliefs: &Beliefs, rng: &mut Rng) -> Searched {
        self.run(root, k, beliefs, rng, self.cfg.iterations, self.cfg.noise)
    }

    /// Une recherche de `iterations` itérations ; `noise` : part du bruit de Dirichlet à la racine.
    pub fn search_with(&self, root: &Game, k: &Knowledge, rng: &mut Rng, iterations: usize, noise: f32) -> Searched {
        self.run(root, k, &Beliefs::default(), rng, iterations, noise)
    }

    fn run(
        &self,
        root: &Game,
        k: &Knowledge,
        beliefs: &Beliefs,
        rng: &mut Rng,
        iterations: usize,
        noise: f32,
    ) -> Searched {
        let legal = moves::legal(root);
        let me = root.state.actor().expect("un joueur doit agir");
        if legal.len() == 1 {
            return Searched {
                mv: legal[0].mv,
                visits: vec![(legal[0].key, 1)],
                finals: [f32::NAN; MAX_PLAYERS],
                details: vec![],
                value: f32::NAN,
            };
        }
        let mut nodes: Vec<Node> = vec![Node::default()];
        let root_value = self.expand(&mut nodes[0], root, k, &legal, me, beliefs, rng);
        nodes[0].visits = 1;
        nodes[0].total = root_value;
        if noise > 0.0 {
            let dir = dirichlet(rng, self.cfg.alpha, legal.len());
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
                    let mut ret = Returns::default();
                    moves::play(&mut w, &legal[0].mv, &mut ret).expect("coup forcé");
                    kw.observe(ret.0, &w.state);
                    continue;
                }
                let n = &mut nodes[node as usize];
                if !n.expanded {
                    break self.expand(n, &w, &kw, &legal, me, beliefs, rng);
                }
                let pid = w.state.actor().expect("un joueur doit agir");
                // les coups de ce monde-ci ; un coup jamais vu reçoit un a priori moyen
                let mut seen: Vec<(usize, usize)> = Vec::with_capacity(legal.len());
                for (li, c) in legal.iter().enumerate() {
                    let ei = match n.edges.iter().position(|e| e.key == c.key) {
                        Some(ei) => ei,
                        None => {
                            let prior = 1.0 / legal.len() as f32;
                            n.edges.push(Edge {
                                key: c.key,
                                prior,
                                child: NO_CHILD,
                                visits: 0,
                                total: [0.0; 2 * MAX_PLAYERS],
                            });
                            n.edges.len() - 1
                        }
                    };
                    seen.push((li, ei));
                }
                let visits: u32 = seen.iter().map(|&(_, ei)| n.edges[ei].visits).sum();
                let prior_sum: f32 = seen.iter().map(|&(_, ei)| n.edges[ei].prior).sum::<f32>().max(1e-6);
                let parent_q = n.total[pid] / n.visits.max(1) as f32;
                let sqrt_n = (visits as f32 + 1.0).sqrt();
                let (mut lo, mut hi) = (parent_q, parent_q);
                if self.cfg.normalize {
                    for &(_, ei) in &seen {
                        let e = &n.edges[ei];
                        if e.visits > 0 {
                            let q = e.total[pid] / e.visits as f32;
                            lo = lo.min(q);
                            hi = hi.max(q);
                        }
                    }
                }
                let norm = |q: f32| if hi - lo > 1e-6 { (q - lo) / (hi - lo) } else { 0.5 };
                let mut best = 0;
                let mut best_score = f32::NEG_INFINITY;
                for (si, &(_, ei)) in seen.iter().enumerate() {
                    let e = &n.edges[ei];
                    let q = match (e.visits > 0, self.cfg.normalize) {
                        (true, true) => norm(e.total[pid] / e.visits as f32),
                        (true, false) => e.total[pid] / e.visits as f32,
                        (false, true) => (norm(parent_q) - self.cfg.fpu).max(0.0),
                        (false, false) => parent_q - self.cfg.fpu,
                    };
                    let u = self.cfg.c_puct * e.prior / prior_sum * sqrt_n / (1.0 + e.visits as f32);
                    if q + u > best_score {
                        best_score = q + u;
                        best = si;
                    }
                }
                let (li, ei) = seen[best];
                let mut ret = Returns::default();
                moves::play(&mut w, &legal[li].mv, &mut ret).expect("coup possible");
                kw.observe(ret.0, &w.state);
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
        let mut finals = [0.0; MAX_PLAYERS];
        for (q, f) in finals.iter_mut().enumerate() {
            *f = root_node.total[COINS + q] / root_node.visits as f32;
        }
        Searched { mv, visits, finals, details, value: root_node.total[me] / root_node.visits as f32 }
    }
}
