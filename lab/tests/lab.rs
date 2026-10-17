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
            let mut keys: Vec<_> = legal.iter().map(|c| c.key).collect();
            keys.sort();
            keys.dedup();
            assert_eq!(keys.len(), legal.len(), "deux coups de même clé");
            assert!(keys.iter().all(|&k| (k as usize) < moves::KEYS));
            for c in &legal {
                let mut w = g;
                let m = c.mv;
                moves::play(&mut w, &m, &mut Returns::default()).unwrap_or_else(|e| panic!("{m:?} : {e:?}"));
            }
            let m = legal[rng.below(legal.len())].mv;
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
            let m = legal[rng.below(legal.len())].mv;
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

#[test]
fn the_network_sees_only_what_the_player_knows() {
    use smallmobile_lab::encode::encode;
    let mut rng = Rng::new(11);
    for game in 0..40u64 {
        let table = Table::new(2 + game as usize % 4, 1300 + game);
        let mut g = table.game(game as u32);
        let mut k = Knowledge::default();
        while g.state.phase != Phase::GameOver {
            // deux mondes possibles donnent exactement la même entrée au réseau
            let a = encode(&determinize(&g, &k, &mut rng), k.race_tail);
            let b = encode(&determinize(&g, &k, &mut rng), k.race_tail);
            assert!(a.node == b.node && a.global == b.global, "l'encodage dépend de ce qui est caché");
            // « moi » est le joueur qui agit : ses pièces sont en tête des joueurs
            let me = g.state.actor().unwrap();
            assert_eq!(a.global[67 + 1] as i32, g.state.players[me].coins.clamp(0, 255));
            let legal = moves::legal(&g);
            let m = legal[rng.below(legal.len())].mv;
            let mut ret = Returns::default();
            moves::play(&mut g, &m, &mut ret).unwrap();
            k.observe(ret.0, &g.state);
        }
    }
}
