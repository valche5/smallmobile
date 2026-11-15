//! Les extensions jouées en entier par l'IA du noyau : chaque partie se termine, aucune action de
//! l'IA n'est refusée, et les invariants tiennent à chaque coup.

use std::collections::BTreeMap;

use smallmobile_kernel::action::{Event, Logged};
use smallmobile_kernel::ai::{Ai, Restrictions, Tuning};
use smallmobile_kernel::collections::RegionSet;
use smallmobile_kernel::state::Occupant;
use smallmobile_kernel::wire::{self, Writer};
use smallmobile_kernel::{Expansions, Game, Level, Phase, Power, Race, mapgen};

/// Jetons, pièces et état relu : ce qui doit tenir après chaque action.
fn check(g: &Game, buf: &mut [u32]) {
    let s = &g.state;
    for &race in Race::ALL {
        let on_board = g.tokens_on_board(race);
        match s.races[race.index()] {
            None => assert_eq!(on_board, 0, "{race:?} absent mais sur le plateau"),
            Some(rs) => {
                // plateau + main + hors plateau + réserve = boîte (+2 pour la Horde active)
                let hordes = if rs.has(Power::Hordes) { 2 } else { 0 };
                assert!(rs.hand >= 0, "{race:?} : {rs:?}");
                assert!(g.supply(race) >= 0, "{race:?} : plus de jetons que la boîte");
                assert_eq!(on_board + rs.hand + i32::from(rs.aside) + g.supply(race), race.max() + hordes);
                assert!(rs.aside == 0 || matches!(race, Race::Barbarians | Race::Pixies));
                let floor = g.floor(race);
                for r in g.owned(race).iter() {
                    assert!(s.regions[r].tokens >= floor, "{race:?} : moins de {floor} jetons en {r}");
                    if race == Race::Priestesses && rs.declined {
                        assert_eq!(g.owned(race).len(), 1, "une seule Tour d'Ivoire");
                    }
                }
                if race == Race::Leprechauns {
                    assert!(g.cauldrons_on_board(race) + i32::from(rs.cauldrons) <= 16);
                }
            }
        }
    }
    for (i, r) in s.regions[..g.map.len()].iter().enumerate() {
        if r.ladies() {
            assert_eq!(r.occupant, Occupant::Race(Race::WhiteLadies), "région {i}");
            assert!(s.race(Race::WhiteLadies).declined);
        }
        if r.cauldron() {
            assert_eq!(r.occupant, Occupant::Race(Race::Leprechauns), "région {i}");
        }
        if r.catapult() {
            let race = r.occupant.race().expect("Catapulte sur une région vide");
            assert!(s.race(race).has(Power::Catapult));
        }
    }
    for p in &s.players {
        assert!(p.coins >= 0, "pièces négatives");
    }
    let mut w = Writer::new(buf);
    wire::encode_game(g.map, s, &mut w);
    let (_, back) = wire::decode_game(w.written()).expect("état relu");
    assert_eq!(&back, s);
}

/// Ce qui s'est vu pendant les parties : peuples et pouvoirs choisis, actions, événements.
type Seen = BTreeMap<String, usize>;

fn play(players: usize, map_seed: u64, seed: u32, levels: &[Level], seen: &mut Seen) -> usize {
    let generated = mapgen::generate(players, map_seed).unwrap();
    let map = generated.to_map().unwrap();
    let lost: RegionSet = generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
    let tuning = Tuning::default();
    let ai = Ai::new(&tuning);
    let mut events: Vec<Logged> = Vec::new();
    let mut g = Game::create(&map, lost, levels, generated.rounds, seed, Expansions::ALL, &mut events);
    let mut buf = vec![0; 16 * 1024];
    check(&g, &mut buf);
    let mut steps = 0;
    while g.state.phase != Phase::GameOver {
        let a = ai.decide(&g, Restrictions::default()).unwrap_or_else(|r| panic!("pas de décision : {r:?}"));
        g.apply(&a, &mut events).unwrap_or_else(|r| panic!("{a:?} refusée : {r:?} (graine {seed})"));
        check(&g, &mut buf);
        let name = format!("{a:?}");
        *seen.entry(format!("action:{}", name.split([' ', '(']).next().unwrap())).or_default() += 1;
        steps += 1;
        assert!(steps < 5_000, "partie sans fin");
    }
    for e in &events {
        let key = match e.event {
            Event::Picked { race, power, .. } => {
                *seen.entry(format!("race:{}", race.key())).or_default() += 1;
                format!("power:{}", power.key())
            }
            Event::Paid { kind, .. } => format!("paid:{kind:?}"),
            Event::PygmiesReturned { .. } => "pygmies".into(),
            Event::Scored { lines, .. } => {
                for l in lines.iter() {
                    *seen.entry(format!("income:{:?}", l.kind)).or_default() += 1;
                }
                continue;
            }
            Event::Declined { race, .. } => format!("declined:{}", race.key()),
            _ => continue,
        };
        *seen.entry(key).or_default() += 1;
    }
    steps
}

/// Tout ce que les extensions ajoutent a servi au moins une fois.
fn assert_covered(seen: &Seen, keys: &[&str]) {
    let missing: Vec<&&str> = keys.iter().filter(|k| !seen.contains_key(**k)).collect();
    assert!(missing.is_empty(), "jamais vus : {missing:?}\n{seen:#?}");
}

#[test]
fn the_ai_plays_whole_games_with_every_expansion() {
    let mut steps = 0;
    let mut seen = Seen::new();
    for players in 2..=5 {
        for k in 0..12u32 {
            let level = [Level::Easy, Level::Normal][(k % 2) as usize];
            let mut levels = vec![level; players];
            levels[(k as usize) % players] = [Level::Normal, Level::Easy][(k % 2) as usize];
            steps += play(players, 100 + k as u64, 1000 * players as u32 + k, &levels, &mut seen);
        }
    }
    assert!(steps > 1000);
    let mut keys: Vec<String> = Race::ALL[Race::BASE_COUNT..].iter().map(|r| format!("race:{}", r.key())).collect();
    keys.extend(Power::ALL[Power::BASE_COUNT..].iter().map(|p| format!("power:{}", p.key())));
    for k in [
        "action:Catapult",
        "action:Hire",
        "action:Regroup",
        "action:Cauldron",
        "paid:Bribe",
        "paid:Theft",
        "paid:Cauldron",
        "paid:Cauldrons",
        "paid:Historian",
        "paid:Hire",
        "pygmies",
        "income:Tower",
        "income:Imperial",
        "income:Barricade",
        "income:PeaceLoving",
        "declined:whiteLadies",
        "declined:priestesses",
    ] {
        keys.push(k.into());
    }
    assert_covered(&seen, &keys.iter().map(String::as_str).collect::<Vec<_>>());
    eprintln!("{seen:#?}");
}

/// Le niveau difficile simule des fins de partie : quelques parties seulement.
#[test]
fn the_hard_ai_plays_whole_games_with_every_expansion() {
    for players in 2..=5 {
        for k in 0..2u32 {
            let mut levels = vec![Level::Hard; players];
            levels[k as usize] = [Level::Easy, Level::Normal][k as usize];
            play(players, 7 + players as u64 + 10 * k as u64, 77 + players as u32 + k, &levels, &mut Seen::new());
        }
    }
}
