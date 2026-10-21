//! L'IA Expert telle que le jeu l'emploie : la recherche guidée par le réseau (réglages fixes de
//! `search`), sur la partie telle que la voit le joueur qui doit agir.

use std::sync::Arc;

use smallmobile_kernel::{Action, Game, Reason};

use crate::know::{Knowledge, determinize};
use crate::moves;
use crate::net::Net;
use crate::rand::{Rng, mix};
use crate::search::{Az, Config};

pub struct Expert {
    search: Az,
}

/// Une décision de l'Expert : les actions du coup choisi, et ce qu'en pensait la recherche.
#[derive(Clone, Debug)]
pub struct Decision {
    pub actions: Vec<Action>,
    /// Chance de gagner estimée pour le joueur qui agit (0 à 1, valeur mêlée à l'avance en pièces) ;
    /// NaN pour un coup forcé, joué sans chercher.
    pub value: f32,
    /// Itérations faites (0 : coup forcé).
    pub iterations: usize,
}

impl Expert {
    pub fn new(net: Arc<Net>, iterations: usize) -> Expert {
        Expert { search: Az::new(Config { iterations, ..Config::default() }, net, "expert".into()) }
    }

    /// Le prochain coup du joueur qui doit agir, découpé en actions du noyau. L'Expert ne voit que ce
    /// que sait ce joueur : l'ordre caché des pioches et les dés à venir sont tirés au hasard.
    /// `seed` : le hasard de la recherche (même graine, même partie : même décision).
    pub fn decide(&self, g: &Game, seed: u64) -> Result<Decision, Reason> {
        let k = Knowledge::of(&g.state);
        let mut rng = Rng::new(mix(seed, g.state.seq as u64));
        let world = determinize(g, &k, &mut rng);
        let found = self.search.search(&world, &k, &mut rng);
        let iterations = if found.value.is_nan() { 0 } else { self.search.cfg.iterations };
        Ok(Decision { actions: moves::actions(g, &found.mv)?, value: found.value, iterations })
    }
}
