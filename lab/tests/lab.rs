//! Le banc d'essai lui-même : les coups proposés sont toujours jouables, et ce que le joueur « sait »
//! des pioches est exact.

use smallmobile_kernel::Phase;
use smallmobile_kernel::action::{Event, Logged};
use smallmobile_kernel::ids::Race;
use smallmobile_lab::know::{Knowledge, Returns, determinize};
use smallmobile_lab::moves;
use smallmobile_lab::rand::Rng;
use smallmobile_lab::table::Table;

#[test]
fn every_legal_move_can_be_played() {
    let mut rng = Rng::new(7);
    for game in 0..60u64 {
        let table = Table::new(2 + game as usize % 4, 500 + game);
        let mut g = table.game(game as u32);
        while g.state.phase != Phase::GameOver {
            let legal = moves::legal(&g);
            assert!(!legal.is_empty());
            for m in &legal {
                let mut w = g;
                moves::play(&mut w, m, &mut Returns::default()).unwrap_or_else(|e| panic!("{m:?} : {e:?}"));
            }
            let m = legal[rng.below(legal.len())];
            moves::play(&mut g, &m, &mut Returns::default()).unwrap();
        }
    }
}

#[test]
fn known_deck_bottom_is_exact() {
    let mut rng = Rng::new(3);
    let mut checked = 0;
    for game in 0..200u64 {
        let table = Table::new(2 + game as usize % 4, 900 + game);
        let mut g = table.game(game as u32);
        let mut k = Knowledge::default();
        // les peuples rendus sous la pioche, dans l'ordre, tels qu'un joueur les a vus passer
        let mut seen: Vec<Race> = Vec::new();
        while g.state.phase != Phase::GameOver {
            let legal = moves::legal(&g);
            let m = legal[rng.below(legal.len())];
            let mut events: Vec<Logged> = Vec::new();
            moves::play(&mut g, &m, &mut events).unwrap();
            let returns: usize = events
                .iter()
                .map(|e| match &e.event {
                    Event::Declined { removed, .. } => {
                        seen.extend(removed.iter().copied());
                        removed.len()
                    }
                    _ => 0,
                })
                .sum();
            k.observe(returns, &g.state);
            let deck = g.state.race_deck.as_slice();
            let tail = &deck[deck.len() - k.race_tail..];
            assert_eq!(tail, &seen[seen.len() - k.race_tail..]);
            checked += k.race_tail;
            // un monde possible garde le marché, le fond connu et le contenu de la pioche
            let w = determinize(&g, &k, &mut rng);
            assert_eq!(w.state.market, g.state.market);
            assert_eq!(&w.state.race_deck.as_slice()[deck.len() - k.race_tail..], tail);
            let mut a = w.state.race_deck.as_slice().to_vec();
            let mut b = deck.to_vec();
            a.sort();
            b.sort();
            assert_eq!(a, b);
        }
    }
    assert!(checked > 0, "aucune carte rendue dans ces parties : le test ne vérifie rien");
}
