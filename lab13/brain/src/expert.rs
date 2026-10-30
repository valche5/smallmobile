//! L'IA Expert telle que le jeu l'emploie : la recherche guidée par le réseau (réglages de
//! `search`, ceux de lab6 par défaut), sur la partie telle que la voit le joueur qui doit agir.

use std::sync::Arc;

use smallmobile_kernel::{Action, Game, Reason};

use crate::know::Knowledge;
use crate::moves;
use crate::net::Net;
use crate::rand::{Rng, mix};
use crate::search::{Az, Config, Evals};

pub struct Expert {
    search: Az,
}

/// Une décision de l'Expert : les actions du coup choisi (toute une fin de tour ou une défense,
/// groupée), et ce qu'en pensait la recherche.
#[derive(Clone, Debug)]
pub struct Decision {
    pub actions: Vec<Action>,
    /// Chance de gagner estimée pour le joueur qui agit (0 à 1, valeur mêlée à l'avance en pièces),
    /// d'après la dernière recherche de la décision ; NaN si tous ses coups étaient forcés.
    pub value: f32,
    /// Itérations faites, toutes recherches de la décision comprises (0 : coups forcés).
    pub iterations: usize,
}

impl Expert {
    /// `cfg` : les réglages de la recherche (`Config::parse`), dont ses itérations.
    pub fn new(net: Arc<Net>, cfg: Config) -> Expert {
        Expert { search: Az::new(cfg, net, "expert".into()) }
    }

    /// La prochaine décision du joueur qui doit agir, en actions du noyau : un coup, ou toute une
    /// fin de tour ou une défense, cherchée coup par coup. L'Expert ne voit que ce que sait ce
    /// joueur : l'ordre caché des pioches et les dés à venir sont tirés au hasard. `seed` : le
    /// hasard de la recherche (même graine, même partie : même décision).
    pub fn decide(&self, g: &Game, seed: u64) -> Result<Decision, Reason> {
        let k = Knowledge::of(&g.state);
        let mut rng = Rng::new(mix(seed, g.state.seq as u64));
        let (mut value, mut iterations) = (f32::NAN, 0);
        let mut evals = Evals::default();
        let (actions, _) = moves::decision(g, &k, |w, kw| {
            let found = self.search.search(w, kw, &mut rng, false, &mut evals);
            if !found.value.is_nan() {
                value = found.value;
                iterations += self.search.cfg.iterations;
            }
            found.mv
        })?;
        Ok(Decision { actions, value, iterations })
    }
}
