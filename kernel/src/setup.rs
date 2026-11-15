//! Une nouvelle partie : pioches mélangées, tribus oubliées en place, premier tour commencé.

use crate::action::Journal;
use crate::collections::{ArrayVec, RegionSet};
use crate::game::Game;
use crate::ids::{Expansions, Level, MAX_PLAYERS, MAX_REGIONS, Phase, Race};
use crate::map::Map;
use crate::rng;
use crate::state::{Occupant, Player, RegionState, State, Turn};

/// Mélange de Fisher-Yates, avec le générateur des dés initialisé par la graine de la partie.
fn shuffle<T: Copy>(items: impl Iterator<Item = T>, rand: &mut u32) -> Vec<T> {
    let mut a: Vec<T> = items.collect();
    for i in (1..a.len()).rev() {
        let (v, next) = rng::step(*rand);
        *rand = next;
        a.swap(i, (v * (i + 1) as f64).floor() as usize);
    }
    a
}

/// L'état de départ, avant le premier tour. Les pioches mélangent le jeu de base et les extensions
/// choisies ; sans extension, la partie est celle d'avant les extensions (même tirage).
pub fn initial_state(
    map: &Map,
    lost_tribes: RegionSet,
    levels: &[Level],
    rounds: i32,
    seed: u32,
    expansions: Expansions,
) -> State {
    assert!((2..=MAX_PLAYERS).contains(&levels.len()));
    let mut rand = seed;
    let race_deck = shuffle(expansions.races(), &mut rand).into_iter().collect();
    let power_deck = shuffle(expansions.powers(), &mut rand).into_iter().collect();
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
        expansions,
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
        expansions: Expansions,
        journal: &mut impl Journal,
    ) -> Game<'m> {
        let mut g = Game::new(map, initial_state(map, lost_tribes, levels, rounds, seed, expansions));
        g.start(journal);
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::NoJournal;
    use crate::ids::{Expansion, Power};
    use crate::mapgen;

    fn sorted<T: Ord + Copy>(items: impl IntoIterator<Item = T>) -> Vec<T> {
        let mut v: Vec<T> = items.into_iter().collect();
        v.sort();
        v
    }

    /// Chaque ensemble d'extensions : les pioches de départ, puis pioches et marché d'une partie créée.
    #[test]
    fn decks_hold_the_base_game_and_the_chosen_expansions() {
        let generated = mapgen::generate(3, 11).unwrap();
        let map = generated.to_map().unwrap();
        for bits in 0..=Expansions::ALL.bits() {
            let x = Expansions::from_bits(bits).unwrap();
            let races: Vec<Race> =
                Race::ALL.iter().copied().filter(|r| r.expansion().is_none_or(|e| x.contains(e))).collect();
            let powers: Vec<Power> =
                Power::ALL.iter().copied().filter(|p| p.expansion().is_none_or(|e| x.contains(e))).collect();
            for seed in 0..8 {
                let s = initial_state(&map, RegionSet::EMPTY, &[Level::Normal; 3], 9, seed, x);
                assert_eq!(s.expansions, x);
                assert_eq!(sorted(s.race_deck.iter().copied()), races);
                assert_eq!(sorted(s.power_deck.iter().copied()), powers);
                let g = Game::create(&map, RegionSet::EMPTY, &[Level::Normal; 3], 9, seed, x, &mut NoJournal);
                let s = &g.state;
                assert_eq!(sorted(s.race_deck.iter().copied().chain(s.market.iter().map(|c| c.race))), races);
                assert_eq!(sorted(s.power_deck.iter().copied().chain(s.market.iter().map(|c| c.power))), powers);
            }
        }
        let base = Expansions::NONE.races().count() + Expansions::NONE.powers().count();
        assert_eq!(base, Race::BASE_COUNT + Power::BASE_COUNT);
        let all = Expansions::ALL;
        assert_eq!((all.races().count(), all.powers().count()), (Race::COUNT, Power::COUNT));
        let cursed = Expansions::NONE.with(Expansion::Cursed);
        assert_eq!(cursed.races().skip(Race::BASE_COUNT).collect::<Vec<_>>(), [Race::Goblins, Race::Kobolds]);
        assert_eq!(cursed.powers().count(), Power::BASE_COUNT + 5);
    }

    /// Sans extension, les pioches sont celles d'avant les extensions : le mélange du jeu de base seul.
    #[test]
    fn without_expansions_the_draw_is_unchanged() {
        let generated = mapgen::generate(2, 5).unwrap();
        let map = generated.to_map().unwrap();
        for seed in 0..32 {
            let s = initial_state(&map, RegionSet::EMPTY, &[Level::Normal; 2], 10, seed, Expansions::NONE);
            let mut rand = seed;
            let races = shuffle(Race::ALL[..Race::BASE_COUNT].iter().copied(), &mut rand);
            let powers = shuffle(Power::ALL[..Power::BASE_COUNT].iter().copied(), &mut rand);
            assert_eq!(s.race_deck.as_slice(), races.as_slice());
            assert_eq!(s.power_deck.as_slice(), powers.as_slice());
        }
    }
}
