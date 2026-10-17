//! Les actions d'un joueur, et les événements qu'elles produisent pour le journal.

use crate::collections::ArrayVec;
use crate::ids::{MAX_REGIONS, Power, Race};
use crate::state::Occupant;

/// `Place` porte un tableau de 64 cases : les actions restent `Copy`, sans allocation.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Pick {
        index: usize,
    },
    /// Le marché est vide : passer son tour.
    SkipPick,
    Conquer {
        region: usize,
    },
    /// Dernière conquête, tentée avec le dé de renfort.
    Reinforce {
        region: usize,
    },
    BerserkRoll,
    Dragon {
        region: usize,
    },
    Sorcery {
        region: usize,
    },
    Abandon {
        region: usize,
    },
    Decline,
    EndConquest,
    Deploy {
        region: usize,
        delta: i32,
    },
    /// Pose les jetons en main ; `all` reprend d'abord tous les jetons (un seul reste par région).
    AutoDeploy {
        all: bool,
    },
    /// Fixe le nombre de jetons de chaque région du peuple ; le reste va en main.
    Place(Placement),
    Camp {
        region: usize,
        delta: i32,
    },
    Fortress {
        region: usize,
    },
    Hero {
        region: usize,
    },
    EndTurn {
        peace: Option<usize>,
        stout: bool,
    },
    EndDefense,
}

/// Jetons voulus par région (les régions qui ne sont pas au peuple sont ignorées).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement(pub [i32; MAX_REGIONS]);

impl Default for Placement {
    /// Un jeton par région : c'est ce que vaut une région absente d'un placement.
    fn default() -> Self {
        Placement([1; MAX_REGIONS])
    }
}

/// Comment une région a été prise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum How {
    Conquer,
    Die,
    Dragon,
    Sorcery,
}

/// Une ligne du décompte de fin de tour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IncomeLine {
    pub kind: IncomeKind,
    pub race: Race,
    /// Le peuple est en déclin.
    pub declined: bool,
    pub value: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IncomeKind {
    #[default]
    Regions,
    Mines,
    Farms,
    MagicSources,
    OrcLoot,
    Alchemist,
    Forests,
    Hills,
    Swamps,
    Merchant,
    Pillage,
    Fortresses,
    Wealthy,
}

pub type Income = ArrayVec<IncomeLine, 16>;

/// Ce qui s'est passé, dans l'ordre. Chaque événement augmente `State::seq` de 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    GameStarted,
    /// Début du tour d'un peuple (`ghouls` : les Goules en déclin jouent d'abord).
    TurnStarted {
        player: usize,
        race: Race,
        power: Power,
        hand: i32,
        ghouls: bool,
        amazon_bonus: i32,
    },
    Picked {
        player: usize,
        race: Race,
        power: Power,
        tokens: i32,
        paid: i32,
        gained: i32,
    },
    Conquered {
        player: usize,
        race: Race,
        region: usize,
        how: How,
        tokens: i32,
        victim: Option<usize>,
        victim_occupant: Occupant,
        /// La victime était un peuple actif (ses jetons rendus sont à replacer).
        victim_active: bool,
        /// Jetons perdus par la victime et jetons rendus, avant ce qu'en fait un peuple en déclin.
        lost: i32,
        returned: i32,
    },
    /// Dé de renfort (`region`) ou dé des Berserks (`region` vide, `success` vide).
    DieRolled {
        player: usize,
        region: Option<usize>,
        value: i32,
        need: i32,
        success: Option<bool>,
    },
    Abandoned {
        player: usize,
        region: usize,
    },
    Declined {
        player: usize,
        race: Race,
        power: Power,
        removed: ArrayVec<Race, 2>,
    },
    Scored {
        player: usize,
        gain: i32,
        lines: Income,
    },
    PeaceImposed {
        player: usize,
        target: usize,
        race: Option<Race>,
    },
    AmazonsReturned {
        player: usize,
        tokens: i32,
    },
    SkeletonsRaised {
        player: usize,
        tokens: i32,
    },
    GhoulsDone {
        player: usize,
    },
    Defended {
        player: usize,
        race: Race,
    },
    GameOver,
}

/// Un événement daté : son numéro (`State::seq` après lui) et la manche en cours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Logged {
    pub seq: u32,
    pub round: i32,
    pub event: Event,
}

/// Où vont les événements : nulle part dans les simulations, dans une liste pour le journal.
pub trait Journal {
    fn record(&mut self, entry: Logged);
}

/// Les simulations n'écrivent pas de journal.
pub struct NoJournal;

impl Journal for NoJournal {
    #[inline(always)]
    fn record(&mut self, _: Logged) {}
}

impl Journal for Vec<Logged> {
    fn record(&mut self, entry: Logged) {
        self.push(entry);
    }
}
