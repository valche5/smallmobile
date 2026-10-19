//! Les joueurs. Chacun reçoit la vraie partie et ce que son joueur sait (`Knowledge`) : un joueur
//! honnête ne lit la partie qu'à travers `know::determinize`.

use smallmobile_brain::opponents::Model;
use smallmobile_kernel::{Ai, Game, Level, Restrictions, Tuning};

use crate::know::{Knowledge, determinize};
use crate::moves::{self, Move};
use crate::rand::Rng;

pub trait Agent: Sync {
    fn name(&self) -> String;

    /// Le prochain coup du joueur qui doit agir. `rng` est propre à cette décision.
    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Move;

    /// Le modèle qui simule le mieux ce joueur, pour qui connaît son type (dans le jeu, on voit
    /// quelles places sont des IA, et de quel niveau).
    fn model(&self) -> Model {
        Model::Normal
    }

    /// Comme `act`, en connaissant le modèle de chaque place (`seats`, joueurs réels).
    fn act_at(&self, g: &Game, k: &Knowledge, seats: &[Model], rng: &mut Rng) -> Move {
        let _ = seats;
        self.act(g, k, rng)
    }
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
        legal[rng.below(legal.len())].mv
    }
}

/// Un joueur d'après son nom (`easy`, `normal`, `hard`, `random` ; `hard!` : voit tout ;
/// `mcts:n=1000,c=0.7` : recherche, voir `mcts::Config::parse` ; `az:net=poids.bin,n=400` : recherche
/// guidée par le réseau, voir `az::Config::parse`).
pub fn by_name(name: &str) -> Option<Box<dyn Agent>> {
    if let Some(opts) = name.strip_prefix("mcts") {
        let cfg = crate::mcts::Config::parse(opts.strip_prefix(':').unwrap_or(opts))?;
        return Some(Box::new(crate::mcts::Mcts::new(cfg)));
    }
    if name.starts_with("az") {
        return Some(Box::new(az_named(name)?));
    }
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

/// `az:net=chemin,n=400,c=1.5,…`
fn az_named(name: &str) -> Option<crate::az::Az> {
    let opts = name.strip_prefix("az:")?;
    let mut net = None;
    let rest: Vec<&str> = opts
        .split(',')
        .filter(|p| match p.strip_prefix("net=") {
            Some(path) => {
                net = Some(path.to_string());
                false
            }
            None => true,
        })
        .collect();
    let net = net?;
    let cfg = crate::az::Config::parse(&rest.join(","))?;
    let loaded = crate::net::Net::load(std::path::Path::new(&net)).unwrap_or_else(|e| {
        eprintln!("réseau illisible ({net}) : {e}");
        std::process::exit(2)
    });
    let label = std::path::Path::new(&net).file_stem().map_or(net.clone(), |s| s.to_string_lossy().into_owned());
    Some(crate::az::Az::new(cfg, std::sync::Arc::new(loaded), label))
}

/// Une IA qui cherche, pour l'auto-apprentissage (`mcts:…`, `az:…`, ou `normal` : l'IA du jeu).
pub fn searcher_by_name(name: &str) -> Option<Box<dyn crate::selfplay::Searcher>> {
    if let Some(level) = level_named(name) {
        return Some(Box::new(KernelAi::new(level)));
    }
    if let Some(opts) = name.strip_prefix("mcts") {
        let cfg = crate::mcts::Config::parse(opts.strip_prefix(':').unwrap_or(opts))?;
        return Some(Box::new(crate::mcts::Mcts::new(cfg)));
    }
    Some(Box::new(az_named(name)?))
}
