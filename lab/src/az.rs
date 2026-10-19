//! La recherche guidée par le réseau (`smallmobile_brain::search`), comme joueur du labo.

use smallmobile_brain::opponents::{Beliefs, Model};
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

    fn model(&self) -> Model {
        Model::Net
    }

    /// Avec `rp=4`, chaque adversaire des fins simulées joue selon le type de sa place ; sans place
    /// connue (`act`), c'est le mélange égal des modèles.
    fn act_at(&self, g: &Game, k: &Knowledge, seats: &[Model], rng: &mut Rng) -> Move {
        if self.cfg.net_rollouts == 4 {
            return self.search_against(g, k, &Beliefs::known(seats), rng).mv;
        }
        self.act(g, k, rng)
    }
}
