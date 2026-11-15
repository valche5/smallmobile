//! Les raccourcis des cibles de conquête (`reachable_targets`, `attack_modifier`) disent la même
//! chose que le calcul complet, à chaque conquête de parties de l'IA sur les cartes du jeu.

use smallmobile_kernel::action::NoJournal;
use smallmobile_kernel::ai::{Ai, Restrictions, Tuning};
use smallmobile_kernel::collections::RegionSet;
use smallmobile_kernel::rules::Target;
use smallmobile_kernel::{Expansions, Game, Level, Phase, maps};

#[test]
fn shortcuts_match_the_full_computation() {
    let tuning = Tuning::default();
    let ai = Ai::new(&tuning);
    let mut checked = 0;
    for seed in 0..48u64 {
        // une partie sur deux avec toutes les extensions
        let expansions = if seed % 2 == 0 { Expansions::NONE } else { Expansions::ALL };
        let players = 2 + seed as usize % 4;
        let generated = maps::get(players, seed * 13).unwrap();
        let map = generated.to_map().unwrap();
        let lost: RegionSet =
            generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
        let levels = vec![Level::Normal; players];
        let mut g = Game::create(&map, lost, &levels, generated.rounds, seed as u32, expansions, &mut NoJournal);
        while g.state.phase != Phase::GameOver {
            let full: Vec<Target> = g.targets().iter().copied().filter(Target::reachable).collect();
            assert_eq!(g.reachable_targets().collect::<Vec<_>>(), full);
            if let Some(army) = g.state.turn.army.filter(|_| g.state.phase == Phase::Conquest) {
                for r in 0..map.len() {
                    let total: i32 = g.attack_modifiers(army, r).iter().map(|m| m.value()).sum();
                    assert_eq!(g.attack_modifier(army, r), -total);
                    let t = g.target(r, g.foothold(army));
                    assert_eq!(g.conquest_cost(army, r), t.cost);
                    assert!(t.cost >= g.floor(army));
                }
                checked += 1;
            }
            let a = ai.decide(&g, Restrictions::default()).unwrap();
            g.apply(&a, &mut NoJournal).unwrap();
        }
    }
    assert!(checked > 1000, "{checked} positions de conquête");
}
