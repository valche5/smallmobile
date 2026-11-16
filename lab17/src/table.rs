//! Une table de jeu : une carte du jeu (`kernel/maps.bin`) pour un nombre de joueurs, ses tribus
//! oubliées et son nombre de tours, les extensions de la partie, et le résultat d'une partie.
//!
//! Les extensions viennent de `LAB17_EXPANSIONS` : `none` (le jeu de base, par défaut), `all` (les
//! trois), `mix` (tirées d'après la graine de la partie : un quart sans extension, trois huitièmes avec
//! les trois, le reste avec une ou deux, au hasard), ou un nombre (les bits de `Expansions`).

use std::sync::OnceLock;

use smallmobile_kernel::collections::RegionSet;
use smallmobile_kernel::ids::Expansions;
use smallmobile_kernel::maps;
use smallmobile_kernel::{Game, Level, Map, NoJournal};

pub use smallmobile_brain17::outcome::Outcome;

pub struct Table {
    pub players: usize,
    pub map: Map,
    pub lost: RegionSet,
    pub rounds: i32,
}

/// Les extensions jouées : toutes les mêmes, ou tirées partie par partie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Fixed(Expansions),
    Mix,
}

impl Mode {
    pub fn parse(text: &str) -> Result<Mode, String> {
        Ok(match text {
            "none" | "" => Mode::Fixed(Expansions::NONE),
            "all" => Mode::Fixed(Expansions::ALL),
            "mix" => Mode::Mix,
            bits => Mode::Fixed(
                bits.parse::<u8>().ok().and_then(Expansions::from_bits).ok_or(format!("extensions inconnues : {bits}"))?,
            ),
        })
    }

    /// Les extensions de la partie de graine `seed`.
    pub fn pick(self, seed: u32) -> Expansions {
        match self {
            Mode::Fixed(x) => x,
            Mode::Mix => {
                let h = (seed as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32;
                match h % 8 {
                    0 | 1 => Expansions::NONE,
                    2..=4 => Expansions::ALL,
                    // une ou deux extensions : les bits 1 à 6
                    _ => Expansions::from_bits(1 + ((h >> 8) % 6) as u8).expect("extensions connues"),
                }
            }
        }
    }

    /// Le mode du processus (`LAB17_EXPANSIONS`).
    pub fn current() -> Mode {
        static MODE: OnceLock<Mode> = OnceLock::new();
        *MODE.get_or_init(|| {
            let text = std::env::var("LAB17_EXPANSIONS").unwrap_or_default();
            Mode::parse(&text).unwrap_or_else(|e| panic!("LAB17_EXPANSIONS : {e}"))
        })
    }
}

impl Table {
    /// La table d'une carte du jeu (`maps` : graine modulo le nombre de cartes).
    pub fn new(players: usize, map_seed: u64) -> Table {
        let generated = maps::get(players, map_seed).expect("carte du jeu");
        let map = generated.to_map().expect("carte valide");
        let lost = generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
        Table { players, map, lost, rounds: generated.rounds }
    }

    /// Une partie neuve, avec les extensions du processus (`Mode::current`). Le niveau des places ne
    /// sert qu'aux IA du jeu.
    pub fn game(&self, seed: u32) -> Game<'_> {
        self.game_with(seed, Mode::current().pick(seed))
    }

    pub fn game_with(&self, seed: u32, expansions: Expansions) -> Game<'_> {
        let levels = vec![Level::Hard; self.players];
        Game::create(&self.map, self.lost, &levels, self.rounds, seed, expansions, &mut NoJournal)
    }
}
