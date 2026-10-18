//! Une nouvelle partie : pioches mélangées, tribus oubliées en place, premier tour commencé.

use crate::action::Journal;
use crate::collections::{ArrayVec, RegionSet};
use crate::game::Game;
use crate::ids::{Level, MAX_PLAYERS, MAX_REGIONS, Phase, Power, Race};
use crate::map::Map;
use crate::rng;
use crate::state::{Occupant, Player, RegionState, State, Turn};

/// Mélange de Fisher-Yates, avec le générateur des dés initialisé par la graine de la partie.
fn shuffle<T: Copy>(items: &[T], rand: &mut u32) -> Vec<T> {
    let mut a = items.to_vec();
    for i in (1..a.len()).rev() {
        let (v, next) = rng::step(*rand);
        *rand = next;
        a.swap(i, (v * (i + 1) as f64).floor() as usize);
    }
    a
}

/// L'état de départ, avant le premier tour.
pub fn initial_state(map: &Map, lost_tribes: RegionSet, levels: &[Level], rounds: i32, seed: u32) -> State {
    assert!((2..=MAX_PLAYERS).contains(&levels.len()));
    let mut rand = seed;
    let race_deck = shuffle(Race::ALL, &mut rand).into_iter().collect();
    let power_deck = shuffle(Power::ALL, &mut rand).into_iter().collect();
    let mut regions = [RegionState::default(); MAX_REGIONS];
    for i in lost_tribes.iter().filter(|&i| i < map.len()) {
        regions[i] = RegionState { occupant: Occupant::Lost, tokens: 1, ..RegionState::default() };
    }
    State {
        players: levels.iter().map(|&level| Player { level, ..Player::default() }).collect(),
        races: [None; Race::COUNT],
        regions,
        market: ArrayVec::new(),
        race_deck,
        power_deck,
        race_tail: 0,
        power_tail: 0,
        round: 1,
        max_rounds: rounds,
        current: 0,
        phase: Phase::Pick,
        turn: Turn::default(),
        defense: ArrayVec::new(),
        peace: ArrayVec::new(),
        seq: 0,
        rng: seed ^ 0x9e37_79b9,
        last_die: None,
        last_region: None,
    }
}

impl<'m> Game<'m> {
    /// Une partie prête à jouer : marché rempli, premier tour commencé.
    pub fn create(
        map: &'m Map,
        lost_tribes: RegionSet,
        levels: &[Level],
        rounds: i32,
        seed: u32,
        journal: &mut impl Journal,
    ) -> Game<'m> {
        let mut g = Game::new(map, initial_state(map, lost_tribes, levels, rounds, seed));
        g.start(journal);
        g
    }
}
