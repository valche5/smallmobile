//! Compteurs de recherche, sans tirage aléatoire ni changement des décisions.

use smallmobile_kernel::{Game, Phase};

/// Totaux additifs, sauf `max_depth`. Une simulation est une descente, pas un tour de jeu.
/// Les recherches forcées (aucun choix) sont exclues. Les compteurs de tours se rapportent au
/// tour courant et au joueur qui décide à la racine, même lorsque celui-ci défend hors de son tour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Diagnostics {
    pub searches: u64,
    pub simulations: u64,
    pub legal: u64,
    pub visited: u64,
    pub wide_roots: u64,
    pub visited_outside16: u64,
    pub chosen_outside16: u64,
    /// Nombre de choix non forcés traversés, y compris celui de la racine.
    pub depth_sum: u64,
    pub max_depth: u64,
    pub turn_finished: u64,
    pub next_turn: u64,
    pub game_over: u64,
}

impl Diagnostics {
    pub fn add(&mut self, other: Self) {
        self.searches += other.searches;
        self.simulations += other.simulations;
        self.legal += other.legal;
        self.visited += other.visited;
        self.wide_roots += other.wide_roots;
        self.visited_outside16 += other.visited_outside16;
        self.chosen_outside16 += other.chosen_outside16;
        self.depth_sum += other.depth_sum;
        self.max_depth = self.max_depth.max(other.max_depth);
        self.turn_finished += other.turn_finished;
        self.next_turn += other.next_turn;
        self.game_over += other.game_over;
    }

    pub(crate) fn leaf(&mut self, horizon: &Horizon, depth: usize, over: bool) {
        self.simulations += 1;
        self.depth_sum += depth as u64;
        self.max_depth = self.max_depth.max(depth as u64);
        self.turn_finished += u64::from(horizon.turn_finished);
        self.next_turn += u64::from(horizon.next_turn);
        self.game_over += u64::from(over);
    }
}

/// Suit les franchissements pendant la descente (coups forcés compris), pas seulement à la
/// feuille : le prochain tour peut être passé avant que la simulation ne termine la partie.
pub(crate) struct Horizon {
    players: usize,
    start: i64,
    next: i64,
    turn_finished: bool,
    next_turn: bool,
}

impl Horizon {
    pub(crate) fn new(root: &Game) -> Self {
        let s = &root.state;
        let players = s.player_count();
        let actor = s.actor().expect("joueur à la racine");
        let start = s.round as i64 * players as i64 + s.current as i64;
        let distance = (actor + players - s.current) % players;
        let next = start + if distance == 0 { players } else { distance } as i64;
        Self { players, start, next, turn_finished: false, next_turn: false }
    }

    pub(crate) fn observe(&mut self, g: &Game) {
        self.observe_turn(g.state.round, g.state.current, g.state.phase == Phase::GameOver);
    }

    fn observe_turn(&mut self, round: i32, current: usize, over: bool) {
        let turn = round as i64 * self.players as i64 + current as i64;
        self.turn_finished |= over || turn > self.start;
        // GameOver conserve certains champs du dernier tour ; ce n'est pas un nouveau tour.
        self.next_turn |= !over && turn >= self.next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizon_counts_turns_not_defense_decisions_or_terminal_rounds() {
        let mut h = Horizon { players: 4, start: 9, next: 13, turn_finished: false, next_turn: false };
        h.observe_turn(2, 1, false); // défense ou redéploiement, même tour
        assert!(!h.turn_finished && !h.next_turn);
        h.observe_turn(2, 2, false);
        assert!(h.turn_finished && !h.next_turn);
        h.observe_turn(3, 0, false);
        assert!(!h.next_turn);
        h.observe_turn(3, 1, false);
        assert!(h.next_turn);
        h.observe_turn(3, 3, true);
        assert!(h.next_turn); // franchissement conservé jusqu'à une feuille terminale

        let mut last = Horizon { players: 4, start: 15, next: 19, turn_finished: false, next_turn: false };
        last.observe_turn(3, 3, true);
        assert!(last.turn_finished && !last.next_turn);
    }
}
