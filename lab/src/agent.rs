//! Les joueurs. Chacun reçoit la vraie partie et ce que son joueur sait (`Knowledge`) : un joueur
//! honnête ne lit la partie qu'à travers `know::determinize`.

use smallmobile_kernel::{Ai, Game, Level, Restrictions, Tuning};

use crate::know::{Knowledge, determinize};
use crate::moves::{self, Move};
use crate::rand::Rng;

pub trait Agent: Sync {
    fn name(&self) -> String;

    /// Le prochain coup du joueur qui doit agir. `rng` est propre à cette décision.
    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Move;
}

/// L'IA du jeu, à un niveau donné. Elle voit un monde possible, pas la vraie partie : ainsi elle ne
/// profite pas de l'ordre des pioches (ce que ses simulations feraient sinon).
pub struct KernelAi {
    pub level: Level,
    /// Voit la vraie partie (pioches et dés compris), comme dans le jeu : pour mesurer ce que ça change.
    pub sees_all: bool,
    tuning: Tuning,
}

impl KernelAi {
    pub fn new(level: Level) -> Self {
        KernelAi { level, sees_all: false, tuning: Tuning::default() }
    }
}

impl Agent for KernelAi {
    fn name(&self) -> String {
        let name = format!("{:?}", self.level).to_lowercase();
        if self.sees_all { name + "!" } else { name }
    }

    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Move {
        let mut w = if self.sees_all { *g } else { determinize(g, k, rng) };
        let pid = w.state.actor().expect("un joueur doit agir");
        w.state.players[pid].level = self.level;
        let ai = Ai::new(&self.tuning);
        Move::Act(ai.decide(&w, Restrictions::default()).expect("l'IA du jeu trouve toujours un coup"))
    }
}

/// Un coup au hasard parmi les coups possibles.
pub struct Random;

impl Agent for Random {
    fn name(&self) -> String {
        "random".into()
    }

    fn act(&self, g: &Game, _: &Knowledge, rng: &mut Rng) -> Move {
        let legal = moves::legal(g);
        legal[rng.below(legal.len())]
    }
}

/// Un joueur d'après son nom (`easy`, `normal`, `hard`, `random` ; `hard!` : voit tout).
pub fn by_name(name: &str) -> Option<Box<dyn Agent>> {
    if let Some(level) = name.strip_suffix('!') {
        return Some(Box::new(KernelAi { sees_all: true, ..KernelAi::new(level_named(level)?) }));
    }
    Some(match name {
        "easy" => Box::new(KernelAi::new(Level::Easy)),
        "normal" => Box::new(KernelAi::new(Level::Normal)),
        "hard" => Box::new(KernelAi::new(Level::Hard)),
        "random" => Box::new(Random),
        _ => return None,
    })
}

fn level_named(name: &str) -> Option<Level> {
    Some(match name {
        "easy" => Level::Easy,
        "normal" => Level::Normal,
        "hard" => Level::Hard,
        _ => return None,
    })
}
