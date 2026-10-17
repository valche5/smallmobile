//! Une table de jeu : la carte générée pour un nombre de joueurs et une graine, comme dans le vrai jeu
//! (mêmes cartes, mêmes tribus oubliées, même nombre de tours), et le résultat d'une partie.

use smallmobile_kernel::collections::RegionSet;
use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::mapgen;
use smallmobile_kernel::{Game, Level, Map, NoJournal, Phase};

pub struct Table {
    pub players: usize,
    pub map: Map,
    pub lost: RegionSet,
    pub rounds: i32,
}

impl Table {
    pub fn new(players: usize, map_seed: u64) -> Table {
        let generated = mapgen::generate(players, map_seed).expect("carte générée");
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

/// Le résultat d'une partie finie, par joueur.
#[derive(Clone, Copy, Debug, Default)]
pub struct Outcome {
    pub players: usize,
    pub coins: [i32; MAX_PLAYERS],
    /// Part de la victoire : 1 pour le vainqueur, partagée en cas d'égalité parfaite.
    pub win: [f64; MAX_PLAYERS],
    /// Place, de 0 (premier) à `players - 1`.
    pub rank: [usize; MAX_PLAYERS],
}

impl Outcome {
    /// Classement du jeu : pièces, puis jetons sur la carte.
    pub fn of(g: &Game) -> Outcome {
        assert_eq!(g.state.phase, Phase::GameOver);
        let s = &g.state;
        let n = s.player_count();
        let key = |q: usize| {
            let p = &s.players[q];
            let tokens: i32 =
                p.active.into_iter().chain(p.declined.iter().copied()).map(|r| g.tokens_on_board(r)).sum();
            (p.coins, tokens)
        };
        let mut out = Outcome { players: n, ..Outcome::default() };
        let keys: Vec<_> = (0..n).map(key).collect();
        let best = *keys.iter().max().expect("des joueurs");
        let winners = keys.iter().filter(|&&k| k == best).count();
        for q in 0..n {
            out.coins[q] = keys[q].0;
            out.rank[q] = keys.iter().filter(|&&k| k > keys[q]).count();
            if keys[q] == best {
                out.win[q] = 1.0 / winners as f64;
            }
        }
        out
    }

    /// Avance de `q` sur son meilleur adversaire, en pièces.
    pub fn margin(&self, q: usize) -> i32 {
        let rival = (0..self.players).filter(|&o| o != q).map(|o| self.coins[o]).max().unwrap_or(0);
        self.coins[q] - rival
    }
}
