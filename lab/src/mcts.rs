//! Recherche arborescente Monte-Carlo sur mondes possibles (ISMCTS « en boucle ouverte »).
//!
//! Chaque itération tire un monde possible (ordre caché des pioches, dés à venir), descend l'arbre
//! en rejouant les coups, essaie un coup nouveau, puis finit la partie avec l'IA Normal du jeu pour
//! tout le monde. Les nœuds sont des suites de coups (désignés par leur clé), pas des états : le même
//! nœud recouvre des mondes différents.
//!
//! Plusieurs joueurs : chaque arête garde la récompense moyenne de chaque joueur, et chacun choisit
//! au mieux pour lui-même (max^n). Un coup qui n'est possible que dans certains mondes est jugé sur
//! les fois où il l'était (UCB avec « disponibilité »).

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Ai, Game, Level, NoJournal, Phase, Tuning};

use crate::agent::Agent;
use crate::know::{Knowledge, determinize};
use crate::moves::{self, Key, Move};
use crate::rand::Rng;
use crate::table::Outcome;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub iterations: usize,
    /// Constante d'exploration d'UCB (récompenses entre 0 et 1).
    pub c: f64,
    /// Poids de la victoire dans la récompense ; le reste vient de l'avance en pièces.
    pub win_weight: f64,
    /// Échelle de l'avance : une avance de `scale` pièces vaut ≈ 0,73.
    pub scale: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config { iterations: 1000, c: 0.7, win_weight: 0.5, scale: 8.0 }
    }
}

impl Config {
    /// Lit des réglages du type `n=1000,c=0.7,w=0.5,s=8`.
    pub fn parse(text: &str) -> Option<Config> {
        let mut cfg = Config::default();
        for part in text.split(',').filter(|p| !p.is_empty()) {
            let (k, v) = part.split_once('=')?;
            match k {
                "n" => cfg.iterations = v.parse().ok()?,
                "c" => cfg.c = v.parse().ok()?,
                "w" => cfg.win_weight = v.parse().ok()?,
                "s" => cfg.scale = v.parse().ok()?,
                _ => return None,
            }
        }
        Some(cfg)
    }
}

type Reward = [f64; MAX_PLAYERS];

const NO_CHILD: u32 = u32::MAX;

struct Edge {
    key: Key,
    child: u32,
    visits: u32,
    /// Nombre de fois où le coup était possible quand la recherche est passée par ce nœud.
    avail: u32,
    total: Reward,
}

#[derive(Default)]
struct Node {
    edges: Vec<Edge>,
}

pub struct Mcts {
    pub cfg: Config,
    tuning: Tuning,
}

impl Mcts {
    pub fn new(cfg: Config) -> Self {
        Mcts { cfg, tuning: Tuning::default() }
    }

    fn reward(&self, g: &Game) -> Reward {
        let o = Outcome::of(g);
        let mut r = [0.0; MAX_PLAYERS];
        let w = self.cfg.win_weight;
        for (q, x) in r.iter_mut().enumerate().take(o.players) {
            let lead = 1.0 / (1.0 + (-(o.margin(q) as f64) / self.cfg.scale).exp());
            *x = w * o.win[q] + (1.0 - w) * lead;
        }
        r
    }

    /// Finit la partie avec l'IA Normal du jeu pour tout le monde.
    fn rollout(&self, w: &mut Game) -> Reward {
        for p in w.state.players.as_mut_slice() {
            p.level = Level::Normal;
        }
        let ai = Ai::new(&self.tuning);
        ai.play(w, 50_000).expect("l'IA du jeu finit la partie");
        self.reward(w)
    }

    /// Le coup choisi et, pour chaque coup de la racine, (clé, visites, récompense moyenne du joueur).
    pub fn search(&self, root: &Game, k: &Knowledge, rng: &mut Rng) -> (Move, Vec<(Key, u32, f64)>) {
        let legal = moves::legal(root);
        if legal.len() == 1 {
            return (legal[0].mv, vec![]);
        }
        let me = root.state.actor().expect("un joueur doit agir");
        let mut nodes: Vec<Node> = vec![Node::default()];
        let mut path: Vec<(u32, usize)> = Vec::new();
        for _ in 0..self.cfg.iterations {
            let mut w = determinize(root, k, rng);
            path.clear();
            let mut node = 0u32;
            let reward = loop {
                if w.state.phase == Phase::GameOver {
                    break self.reward(&w);
                }
                let legal = moves::legal(&w);
                if legal.len() == 1 {
                    moves::play(&mut w, &legal[0].mv, &mut NoJournal).expect("coup forcé");
                    continue;
                }
                let pid = w.state.actor().expect("un joueur doit agir");
                let n = &mut nodes[node as usize];
                // les coups de ce monde-ci, avec leur arête (créée au besoin)
                let mut fresh: Vec<usize> = Vec::new();
                let mut seen: Vec<(usize, usize)> = Vec::with_capacity(legal.len());
                for (li, c) in legal.iter().enumerate() {
                    let ei = match n.edges.iter().position(|e| e.key == c.key) {
                        Some(ei) => ei,
                        None => {
                            n.edges.push(Edge { key: c.key, child: NO_CHILD, visits: 0, avail: 0, total: [0.0; 5] });
                            n.edges.len() - 1
                        }
                    };
                    let e = &mut n.edges[ei];
                    e.avail += 1;
                    if e.visits == 0 {
                        fresh.push(seen.len());
                    }
                    seen.push((li, ei));
                }
                if !fresh.is_empty() {
                    let (li, ei) = seen[fresh[rng.below(fresh.len())]];
                    moves::play(&mut w, &legal[li].mv, &mut NoJournal).expect("coup possible");
                    path.push((node, ei));
                    break self.rollout(&mut w);
                }
                let c = self.cfg.c;
                let mut best = 0;
                let mut best_score = f64::NEG_INFINITY;
                for (si, &(_, ei)) in seen.iter().enumerate() {
                    let e = &n.edges[ei];
                    let v = e.visits as f64;
                    let score = e.total[pid] / v + c * ((e.avail as f64).ln() / v).sqrt();
                    if score > best_score {
                        best_score = score;
                        best = si;
                    }
                }
                let (li, ei) = seen[best];
                moves::play(&mut w, &legal[li].mv, &mut NoJournal).expect("coup possible");
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
            for &(nd, ei) in &path {
                let e = &mut nodes[nd as usize].edges[ei];
                e.visits += 1;
                for (t, r) in e.total.iter_mut().zip(reward) {
                    *t += r;
                }
            }
        }
        let root_edges = &nodes[0].edges;
        let stats: Vec<(Key, u32, f64)> =
            root_edges.iter().map(|e| (e.key, e.visits, e.total[me] / e.visits.max(1) as f64)).collect();
        let best = root_edges.iter().max_by_key(|e| e.visits).expect("des coups").key;
        let mv = legal.iter().find(|c| c.key == best).expect("coup de la racine").mv;
        (mv, stats)
    }
}

impl Agent for Mcts {
    fn name(&self) -> String {
        let c = &self.cfg;
        format!("mcts(n={},c={},w={},s={})", c.iterations, c.c, c.win_weight, c.scale)
    }

    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Move {
        self.search(g, k, rng).0
    }
}
