//! La recherche guidée par le réseau (`smallmobile_brain3::search`), comme joueur du labo : une
//! recherche par coup, une décision entière (fin de tour, défense) en actions groupées.

use smallmobile_kernel::{Action, Game};

pub use smallmobile_brain3::search::*;

use crate::agent::Agent;
use crate::know::Knowledge;
use crate::moves;
use crate::rand::Rng;

impl Agent for Az {
    fn name(&self) -> String {
        self.describe()
    }

    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Vec<Action> {
        moves::decision(g, k, |w, kw| self.search(w, kw, rng, false).mv).expect("coup de la recherche").0
    }
}
