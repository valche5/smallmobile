//! Une table de jeu : la carte générée pour un nombre de joueurs et une graine, comme dans le vrai jeu
//! (mêmes cartes, mêmes tribus oubliées, même nombre de tours), et le résultat d'une partie.

use smallmobile_kernel::collections::RegionSet;
use smallmobile_kernel::mapgen::{self, GeneratedMap};
use smallmobile_kernel::maps;
use smallmobile_kernel::{Game, Level, Map, NoJournal};

pub use smallmobile_brain14::outcome::Outcome;

pub struct Table {
    pub players: usize,
    pub map: Map,
    pub lost: RegionSet,
    pub rounds: i32,
}

impl Table {
    pub fn new(players: usize, map_seed: u64) -> Table {
        Table::of(mapgen::generate(players, map_seed).expect("carte générée"))
    }

    /// La table d'une carte du jeu (`maps` : graine modulo le nombre de cartes).
    pub fn from_pool(players: usize, seed: u64) -> Table {
        Table::of(maps::get(players, seed).expect("carte du jeu"))
    }

    fn of(generated: GeneratedMap) -> Table {
        let players = generated.players;
        let map = generated.to_map().expect("carte valide");
        let lost = generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
        Table { players, map, lost, rounds: generated.rounds }
    }

    /// Une partie neuve. Le niveau des places ne sert qu'aux IA du jeu.
    pub fn game(&self, seed: u32) -> Game<'_> {
        let levels = vec![Level::Hard; self.players];
        Game::create(&self.map, self.lost, &levels, self.rounds, seed, &mut NoJournal)
    }
}
