//! Le résultat d'une partie finie : pièces, victoire, places (classement du jeu).

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Game, Phase};

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
