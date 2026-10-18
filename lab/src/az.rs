//! La recherche guidée par le réseau (`smallmobile_brain::search`), comme joueur du labo.

use smallmobile_kernel::Game;

pub use smallmobile_brain::search::*;

use crate::agent::Agent;
use crate::know::Knowledge;
use crate::moves::Move;
use crate::rand::Rng;

impl Agent for Az {
    fn name(&self) -> String {
        self.describe()
    }

    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Move {
        self.search(g, k, rng).mv
    }
}
