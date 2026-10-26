//! Les plans : tout ce que décide un joueur entre deux hasards (`lab8/lab.md`).
//!
//! Une partie se découpe en plans. Un plan commence à la première décision d'un joueur (plus d'un
//! coup possible), et continue tant que ce joueur agit dans la même phase, jusqu'au premier coup à
//! hasard (`is_chance`) compris. Les coups forcés en font partie sans être des pas du décodeur :
//! seules les décisions sont écrites (`Writer`). L'encodeur lit la position au début du plan.

use smallmobile_kernel::{Game, Phase};

use crate::encode::encode;
use crate::know::Knowledge;
use crate::moves::{Choice, Key, Kind, Move};
use crate::net::{Cache, Memory, Net};
use crate::rand::Rng;
use crate::tokens::dynamics;

/// Un coup dont le résultat dépend du hasard et touche la suite du plan : le dé de renfort, le dé
/// des Berserks. (La carte qui arrive au marché après un choix de peuple ne touche que les joueurs
/// suivants.)
pub fn is_chance(mv: &Move) -> bool {
    matches!(mv, Move::Region(Kind::Reinforce, _) | Move::Berserk)
}

/// Où commence un plan : le joueur qui agit et la phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Start {
    pub actor: usize,
    pub phase: Phase,
}

impl Start {
    pub fn of(g: &Game) -> Start {
        Start { actor: g.state.actor().expect("un joueur doit agir"), phase: g.state.phase }
    }

    /// Le plan continue-t-il après `mv`, qui a mené à `after` ?
    pub fn goes_on(&self, mv: &Move, after: &Game) -> bool {
        !is_chance(mv) && after.state.phase == self.phase && after.state.actor() == Some(self.actor)
    }
}

/// L'écriture d'un plan par le décodeur, décision après décision.
pub struct Writer<'n> {
    net: &'n Net,
    pub mem: Memory,
    cache: Cache,
    prev: Option<Key>,
}

impl<'n> Writer<'n> {
    /// Lit la position `g` du début du plan. `dist` : `tokens::distances` de la carte.
    pub fn new(net: &'n Net, g: &Game, k: &Knowledge, dist: &[u8]) -> Writer<'n> {
        Writer { net, mem: net.encode(&encode(g, k), dist), cache: Cache::default(), prev: None }
    }

    /// Les logits des coups possibles `legal` de la décision suivante, dans la position `g`.
    pub fn logits(&mut self, g: &Game, k: &Knowledge, legal: &[Choice]) -> Vec<f32> {
        let all = self.net.step(&self.mem, &mut self.cache, self.prev, &dynamics(g, k));
        legal.iter().map(|c| all[c.key as usize]).collect()
    }

    /// Le coup choisi à la décision dont on vient de lire les logits.
    pub fn played(&mut self, key: Key) {
        self.prev = Some(key);
    }

    /// Décisions déjà écrites.
    pub fn steps(&self) -> usize {
        self.cache.steps()
    }
}

/// Un coup tiré des logits, à la température `t` (0 : le plus probable).
pub fn pick(logits: &[f32], t: f32, rng: &mut Rng) -> usize {
    let best = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).expect("un coup");
    if t <= 0.0 {
        return best;
    }
    let p: Vec<f64> = logits.iter().map(|&l| (((l - logits[best]) / t) as f64).exp()).collect();
    let mut u = rng.unit() * p.iter().sum::<f64>();
    for (i, &x) in p.iter().enumerate() {
        if u < x {
            return i;
        }
        u -= x;
    }
    best
}

/// Écrit et joue tout un plan depuis `g` (où le joueur a une décision), à la température `t` : les
/// coups joués, la position et le savoir à la fin du plan.
pub fn write<'m>(
    net: &Net,
    g: &Game<'m>,
    k: &Knowledge,
    dist: &[u8],
    t: f32,
    rng: &mut Rng,
) -> (Vec<Move>, Game<'m>, Knowledge) {
    let start = Start::of(g);
    let (mut w, mut kw) = (*g, *k);
    let mut writer: Option<Writer> = None;
    let mut played = Vec::new();
    loop {
        let legal = crate::moves::legal(&w, &kw);
        let mv = if legal.len() == 1 {
            legal[0].mv
        } else {
            let wr = writer.get_or_insert_with(|| Writer::new(net, &w, &kw, dist));
            let logits = wr.logits(&w, &kw, &legal);
            let c = legal[pick(&logits, t, rng)];
            wr.played(c.key);
            c.mv
        };
        kw.play(&mut w, &mv).expect("coup possible");
        played.push(mv);
        if w.state.phase == Phase::GameOver || !start.goes_on(&mv, &w) {
            return (played, w, kw);
        }
    }
}
