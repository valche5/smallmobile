//! Empreinte de non-régression : les mêmes graines doivent toujours donner exactement les mêmes
//! parties (cartes, règles, IA). L'empreinte est identique à celle de l'ancien moteur TypeScript ;
//! si une modification de l'IA la change volontairement, la mettre à jour ici.

use smallmobile_kernel::action::NoJournal;
use smallmobile_kernel::ai::{Ai, Restrictions, Tuning};
use smallmobile_kernel::collections::RegionSet;
use smallmobile_kernel::mapgen;
use smallmobile_kernel::rng::Key;
use smallmobile_kernel::state::Occupant;
use smallmobile_kernel::{Game, Level, Phase};

fn fingerprint(players: usize, games: u64) -> u32 {
    let tuning = Tuning::default();
    let ai = Ai::new(&tuning);
    let mut text = String::new();
    for seed in 1..=games {
        let generated = mapgen::generate(players, seed * 7).unwrap();
        let map = generated.to_map().unwrap();
        let lost: RegionSet =
            generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
        let mut g =
            Game::create(&map, lost, &vec![Level::Normal; players], generated.rounds, seed as u32, &mut NoJournal);
        while g.state.phase != Phase::GameOver {
            let a = ai.decide(&g, Restrictions::default()).unwrap();
            g.apply(&a, &mut NoJournal).unwrap();
        }
        let s = &g.state;
        let coins: Vec<String> = s.players.iter().map(|p| p.coins.to_string()).collect();
        let regions: String = s.regions[..map.len()]
            .iter()
            .map(|r| {
                let who = match r.occupant {
                    Occupant::Empty => "",
                    Occupant::Lost => "lost",
                    Occupant::Race(race) => race.key(),
                };
                format!("{who}{}", r.tokens)
            })
            .collect();
        text += &format!("{}:{}:{}:{};", s.seq, s.rng, coins.join(","), regions);
    }
    Key::new().text(&text).finish()
}

#[test]
fn same_seeds_same_games() {
    assert_eq!(fingerprint(2, 40), 0x506630de);
    assert_eq!(fingerprint(4, 40), 0xa096323f);
}
