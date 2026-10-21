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
    tuning: Tuning,
    /// Les réglages changés (`style`), pour le nom.
    style: String,
}

impl KernelAi {
    pub fn new(level: Level) -> Self {
        KernelAi { level, tuning: Tuning::default(), style: String::new() }
    }

    /// Un autre style de jeu au même niveau : des réglages de l'IA du jeu changés (`Tuning`), du type
    /// `foe=1,foe_declined=0.5`, ou un style nommé (`STYLES`). `None` si un réglage est inconnu.
    pub fn style(level: Level, style: &str) -> Option<Self> {
        let settings = STYLES.iter().find(|(name, _)| *name == style).map_or(style, |(_, s)| s);
        let mut ai = KernelAi::new(level);
        for part in settings.split(',').filter(|p| !p.is_empty()) {
            let (k, v) = part.split_once('=')?;
            if !ai.tuning.set(k, v.parse().ok()?) {
                return None;
            }
        }
        ai.style = style.to_string();
        Some(ai)
    }
}

/// Des styles de jeu de l'IA Normale, pour des tables variées (`lab2/journal.md`, P3) : nom, réglages.
/// Les conquêtes au rendement immédiat (`conquest_model=0`) laissent jouer les primes et malus des
/// régions visées ; le déclin à seuils (`decline_model=0`), ses seuils.
pub const STYLES: [(&str, &str); 5] = [
    // prend aux autres joueurs, même à ceux qui ne dominent pas
    ("aggressive", "conquest_model=0,foe=1,foe_declined=0.5"),
    // s'en prend au plus riche dès qu'il mène
    ("hunter", "conquest_model=0,dom_min=0,dom_share=0,dom_base=2,dom_max=4"),
    // garde ses jetons pour la défense, évite les régions exposées
    ("prudent", "conquest_model=0,min_ratio=0.6,exposure=0.3"),
    // décline dès que les conquêtes faiblissent
    ("early", "decline_model=0,decline_est=3,decline_left=1"),
    // garde son peuple jusqu'à l'épuisement
    ("late", "decline_model=0,decline_est=0,decline_left=4"),
];

impl Agent for KernelAi {
    fn name(&self) -> String {
        let mut name = format!("{:?}", self.level).to_lowercase();
        if !self.style.is_empty() {
            name += &format!(":{}", self.style);
        }
        name
    }

    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Move {
        let mut w = determinize(g, k, rng);
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

/// Un joueur d'après son nom (`easy`, `normal`, `hard`, `random` ; `normal:aggressive`,
/// `normal:foe=1` : un autre style, voir `KernelAi::style` ; `az:net=poids.bin,n=400` : l'Expert,
/// voir `az::Config::parse`).
pub fn by_name(name: &str) -> Option<Box<dyn Agent>> {
    if name.starts_with("az") {
        return Some(Box::new(az_named(name)?));
    }
    if let Some((level, style)) = name.split_once(':') {
        return Some(Box::new(KernelAi::style(level_named(level)?, style)?));
    }
    Some(match name {
        "random" => Box::new(Random),
        _ => Box::new(KernelAi::new(level_named(name)?)),
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

/// `az:net=chemin,n=400,bh=1`
pub fn az_named(name: &str) -> Option<crate::az::Az> {
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
