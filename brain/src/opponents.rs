//! Deviner comment jouent les adversaires, d'après leurs coups dans la partie.
//!
//! Les fins simulées de la recherche font jouer les adversaires : le mieux est de les faire jouer
//! comme ils jouent vraiment (`lab/JOURNAL.md` : l'IA Normale prévoit bien l'IA Difficile, mal un
//! Expert). Plutôt que de le supposer, on l'observe : à chaque choix d'un adversaire (peuple,
//! conquêtes), chaque modèle de joueur dit ce qu'il aurait fait ; un modèle qui tombe juste gagne en
//! crédit (règle de Bayes). Les fins simulées tirent ensuite le modèle de chaque adversaire selon
//! ce crédit.
//!
//! Seuls les choix du peuple et des conquêtes comptent : ils se comparent action par action, quel que
//! soit le joueur. Les fins de tour et les défenses, découpées autrement par chacun, sont ignorées.

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Action, Ai, Game, Level, Phase, Restrictions, Tuning};

use crate::encode::encode;
use crate::know::{Knowledge, determinize};
use crate::moves::{self, Move};
use crate::net::Net;
use crate::rand::Rng;

/// Un modèle de joueur pour les fins simulées.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// L'IA Normale du jeu.
    Normal,
    /// Le coup préféré du réseau.
    Net,
}

pub const MODELS: [Model; 2] = [Model::Normal, Model::Net];

/// Part des coups qu'aucun modèle ne prévoit, même le bon (le hasard d'un joueur, un modèle
/// imparfait) : sans elle, un seul coup imprévu suffirait à écarter un modèle pour toujours.
const SURPRISE: f64 = 0.2;

/// Ce qu'on a appris de chaque joueur.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Beliefs {
    /// Log-vraisemblance de ses choix selon chaque modèle (`MODELS`).
    pub log: [[f32; MODELS.len()]; MAX_PLAYERS],
    /// Choix observés, et combien chaque modèle en a prévu.
    pub seen: [u32; MAX_PLAYERS],
    pub hits: [[u32; MODELS.len()]; MAX_PLAYERS],
}

impl Beliefs {
    /// Des croyances certaines : le joueur `q` joue comme `models[q]` (son type est connu, par
    /// exemple une IA du jeu ou un Expert).
    pub fn known(models: &[Model]) -> Beliefs {
        let mut b = Beliefs::default();
        for (q, model) in models.iter().enumerate() {
            for (m, candidate) in MODELS.iter().enumerate() {
                b.log[q][m] = if candidate == model { 0.0 } else { f32::NEG_INFINITY };
            }
        }
        b
    }

    /// Prend note de l'action `a` du joueur qui agit dans `g` (la partie juste avant l'action).
    /// Les modèles ne voient que ce que voit ce joueur (`determinize`) ; `net` : le réseau du modèle
    /// `Net`.
    pub fn observe(&mut self, net: &Net, g: &Game, a: &Action, rng: &mut Rng) {
        if !matches!(g.state.phase, Phase::Pick | Phase::Conquest) {
            return;
        }
        let legal = moves::legal(g);
        if legal.len() < 2 || !legal.iter().any(|c| c.mv == Move::Act(*a)) {
            return;
        }
        let pid = g.state.actor().expect("un joueur doit agir");
        let k = Knowledge::of(&g.state);
        let mut w = determinize(g, &k, rng);
        w.state.players[pid].level = Level::Normal;
        let tuning = Tuning::default();
        let normal = Ai::new(&tuning).decide(&w, Restrictions::default()).ok();
        let eval = net.eval(&encode(g, k.race_tail), g.map);
        let best = legal.iter().max_by(|x, y| eval.logits[x.key as usize].total_cmp(&eval.logits[y.key as usize]));
        let preferred = best.map(|c| c.mv);
        self.seen[pid] += 1;
        let miss = SURPRISE / legal.len() as f64;
        for (m, model) in MODELS.iter().enumerate() {
            let hit = match model {
                Model::Normal => normal == Some(*a),
                Model::Net => preferred == Some(Move::Act(*a)),
            };
            self.hits[pid][m] += hit as u32;
            self.log[pid][m] += (if hit { 1.0 - SURPRISE + miss } else { miss }).ln() as f32;
        }
    }

    /// Le crédit de chaque modèle pour le joueur `q` (somme 1 ; égal tant que rien n'est observé).
    pub fn weights(&self, q: usize) -> [f32; MODELS.len()] {
        let top = self.log[q].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut w = self.log[q].map(|l| (l - top).exp());
        let sum: f32 = w.iter().sum();
        w.iter_mut().for_each(|x| *x /= sum);
        w
    }

    /// Un modèle pour le joueur `q`, tiré selon son crédit.
    pub fn sample(&self, q: usize, rng: &mut Rng) -> Model {
        let mut x = rng.unit() as f32;
        for (m, w) in MODELS.iter().zip(self.weights(q)) {
            if x < w {
                return *m;
            }
            x -= w;
        }
        MODELS[MODELS.len() - 1]
    }
}
