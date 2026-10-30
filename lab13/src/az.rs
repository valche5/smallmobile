//! La recherche guidée par le réseau (`smallmobile_brain13::search`), comme joueur du labo : une
//! recherche par coup, une décision entière (fin de tour, défense) en actions groupées. Ses réglages
//! de joueur : `explore` (bruit de Gumbel pour jouer, comme le selfplay) et `history` (sans
//! l'histoire des joueurs, comme l'Expert du jeu).

use smallmobile_kernel::{Action, Game};

pub use smallmobile_brain13::search::*;

use crate::agent::Agent;
use crate::know::Knowledge;
use crate::moves;
use crate::rand::Rng;

impl Agent for Az {
    fn name(&self) -> String {
        self.describe()
    }

    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Vec<Action> {
        let mut evals = Evals::default();
        let k = known(self, g, k);
        let explore = self.cfg.explore;
        moves::decision(g, &k, |w, kw| self.search(w, kw, rng, explore, &mut evals).mv).expect("coup de la recherche").0
    }

    fn az(&self) -> Option<&Az> {
        Some(self)
    }
}

/// `Az::act`, chaque recherche attendant ses évaluations de `oracle` (l'arène sur GPU) : les mêmes
/// coups qu'avec des évaluations immédiates, aux nombres du réseau près.
pub async fn act_with(az: &Az, g: &Game<'_>, k: &Knowledge, rng: &mut Rng, oracle: &impl Oracle) -> Vec<Action> {
    let mut evals = Evals::default();
    let (mut w, mut kw) = (*g, known(az, g, k));
    let mut out = Vec::new();
    loop {
        let mv = az.run(&w, &kw, rng, az.cfg.iterations, az.cfg.explore, &mut evals, oracle).await.mv;
        out.extend(moves::actions(&w, &mv));
        kw.play(&mut w, &mv).expect("coup de la recherche");
        if !moves::goes_on(g, &w, &mv) {
            break;
        }
    }
    moves::compact(g, &out)
}

/// Ce que sait le joueur au début d'une décision : `k`, ou, sans histoire (`history=0`), ce que sait
/// l'Expert du jeu, qui ne reçoit que l'état de la partie (`Knowledge::of` : pioches suivies par le
/// noyau, histoire des joueurs vide).
fn known(az: &Az, g: &Game, k: &Knowledge) -> Knowledge {
    if az.cfg.history { *k } else { Knowledge::of(&g.state) }
}
