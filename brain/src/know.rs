//! Ce qu'un joueur sait vraiment, et les mondes possibles qui vont avec.
//!
//! L'état du noyau contient tout, y compris ce qu'aucun joueur ne voit :
//! - l'ordre des pioches (peuples et pouvoirs). Leur contenu est connu (tout ce qui n'est ni en jeu
//!   ni au marché) ; seul l'ordre est caché, sauf pour les cartes rendues sous la pioche quand un
//!   peuple disparaît : leur ordre est public, et un joueur attentif s'en souvient ;
//! - l'état du générateur des dés, qui donne les dés à venir.
//!
//! Les pièces des autres, cachées dans l'interface, se recalculent exactement à partir de ce qui est
//! annoncé (5 au départ, prix et pièces de chaque choix de peuple, gain de chaque tour) : un joueur
//! qui compte les connaît. L'IA a donc droit aux pièces de l'état, comme un joueur humain qui compte.
//!
//! Une IA honnête ne regarde l'état qu'à travers `determinize`, qui remplace ce qui est caché par un
//! tirage compatible avec ce que le joueur sait.

use smallmobile_kernel::action::{Event, Journal, Logged};
use smallmobile_kernel::{Game, State};

use crate::rand::Rng;

/// Ce qu'un joueur attentif sait des pioches : le nombre de cartes connues au fond de chacune.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Knowledge {
    pub race_tail: usize,
    pub power_tail: usize,
}

/// Un journal qui ne garde que ce qui sert à `Knowledge` (évite d'allouer pour chaque action).
#[derive(Default)]
pub struct Returns(pub usize);

impl Journal for Returns {
    fn record(&mut self, entry: Logged) {
        if let Event::Declined { removed, .. } = &entry.event {
            self.0 += removed.len();
        }
    }
}

impl Knowledge {
    /// Ce que sait un joueur de la partie en cours : le noyau suit le fond public des pioches.
    pub fn of(s: &State) -> Knowledge {
        Knowledge { race_tail: s.race_tail as usize, power_tail: s.power_tail as usize }
    }

    /// Met à jour ce qui est su après une action : `returns` cartes rendues sous les pioches (voir
    /// `Returns`), puis les pioches de l'état qui en résulte (les cartes tirées partent du dessus).
    pub fn observe(&mut self, returns: usize, after: &State) {
        self.race_tail = (self.race_tail + returns).min(after.race_deck.len());
        self.power_tail = (self.power_tail + returns).min(after.power_deck.len());
    }
}

/// Un monde possible : même partie vue par les joueurs, l'ordre caché des pioches et les dés à
/// venir tirés au hasard. Le monde ne dépend que de ce que sait le joueur et de `rng` : la partie
/// cachée des pioches est remise dans un ordre fixe avant d'être mélangée.
pub fn determinize<'m>(g: &Game<'m>, k: &Knowledge, rng: &mut Rng) -> Game<'m> {
    let mut w = *g;
    let s = &mut w.state;
    let hidden = s.race_deck.len() - k.race_tail.min(s.race_deck.len());
    let races = &mut s.race_deck.as_mut_slice()[..hidden];
    races.sort_unstable();
    rng.shuffle(races);
    let hidden = s.power_deck.len() - k.power_tail.min(s.power_deck.len());
    let powers = &mut s.power_deck.as_mut_slice()[..hidden];
    powers.sort_unstable();
    rng.shuffle(powers);
    reseed_dice(&mut w, rng);
    w
}

/// Les dés à venir redeviennent imprévisibles (sans toucher aux pioches).
pub fn reseed_dice(g: &mut Game, rng: &mut Rng) {
    g.state.rng = rng.next_u32();
}
