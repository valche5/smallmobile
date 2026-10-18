//! L'état d'une partie : tout ce qui change. `Copy` et sans allocation, pour que les simulations de
//! l'IA puissent en faire des copies à volonté.

use crate::collections::{ArrayVec, PlayerSet, RegionSet};
use crate::ids::{Level, MAX_PLAYERS, MAX_REGIONS, Phase, Power, Race};

/// Ce qui occupe une région.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Occupant {
    #[default]
    Empty,
    /// Une tribu oubliée.
    Lost,
    Race(Race),
}

impl Occupant {
    pub fn race(self) -> Option<Race> {
        match self {
            Occupant::Race(r) => Some(r),
            _ => None,
        }
    }

    pub fn is_empty(self) -> bool {
        self == Occupant::Empty
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionState {
    pub occupant: Occupant,
    pub tokens: i32,
    pub camps: i32,
    pub lair: bool,
    pub fortress: bool,
    pub hero: bool,
    pub dragon: bool,
    pub hole: bool,
}

impl RegionState {
    /// Un héros, un dragon ou un terrier rend la région imprenable.
    pub fn immunity(&self) -> Option<Immunity> {
        if self.hero {
            Some(Immunity::Hero)
        } else if self.hole {
            Some(Immunity::Hole)
        } else if self.dragon {
            Some(Immunity::Dragon)
        } else {
            None
        }
    }

    pub fn is_immune(&self) -> bool {
        self.hero || self.hole || self.dragon
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Immunity {
    Hero,
    Hole,
    Dragon,
}

/// Un peuple en jeu (choisi par un joueur).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaceState {
    pub power: Power,
    pub owner: usize,
    pub declined: bool,
    /// Jetons en main (pas encore posés sur la carte).
    pub hand: i32,
    pub holes_placed: i32,
    pub picked_round: i32,
}

impl RaceState {
    /// Le peuple est actif et possède ce pouvoir.
    pub fn has(&self, power: Power) -> bool {
        !self.declined && self.power == power
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Player {
    pub level: Level,
    pub coins: i32,
    pub active: Option<Race>,
    /// Peuples en déclin, du plus récent au plus ancien (au plus un Spirituel et un autre).
    pub declined: ArrayVec<Race, 2>,
    pub last_income: i32,
}

impl Default for Player {
    fn default() -> Self {
        Player { level: Level::Normal, coins: 5, active: None, declined: ArrayVec::new(), last_income: 0 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Combo {
    pub race: Race,
    pub power: Power,
    pub coins: i32,
}

/// Qui joue pendant le tour : les Goules en déclin passent avant le peuple actif.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stage {
    Ghouls,
    #[default]
    Active,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Turn {
    pub stage: Stage,
    /// Le peuple qui agit.
    pub army: Option<Race>,
    pub conquests: i32,
    /// Régions non vides conquises par le peuple actif (Orques, Pillards, Squelettes).
    pub non_empty: i32,
    /// Joueurs dont le peuple actif a été attaqué ce tour (Diplomates).
    pub attacked: PlayerSet,
    /// Joueurs déjà ensorcelés ce tour (Sorciers).
    pub sorcery: PlayerSet,
    pub dragon_used: bool,
    pub fortress_region: Option<usize>,
    pub berserk_die: Option<i32>,
    pub die_rolled: bool,
    pub declined: bool,
    pub picked: bool,
    pub amazon_bonus: i32,
    /// Jetons ramassés en début de tour, par région (pour rendre leurs jetons aux Goules qui déclinent).
    pub readied: [i32; MAX_REGIONS],
}

impl Default for Turn {
    fn default() -> Self {
        Turn {
            stage: Stage::Active,
            army: None,
            conquests: 0,
            non_empty: 0,
            attacked: PlayerSet::default(),
            sorcery: PlayerSet::default(),
            dragon_used: false,
            fortress_region: None,
            berserk_die: None,
            die_rolled: false,
            declined: false,
            picked: false,
            amazon_bonus: 0,
            readied: [0; MAX_REGIONS],
        }
    }
}

/// Un joueur attaqué qui doit replacer les jetons rendus avant la suite.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DefenseTask {
    pub player: usize,
    pub race: Race,
}

/// Paix imposée par des Diplomates : `target` ne peut pas attaquer le peuple actif de `diplomat`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Peace {
    pub diplomat: usize,
    pub target: usize,
}

/// Le dernier dé lancé (pour l'animation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DieRoll {
    pub seq: u32,
    pub player: usize,
    pub value: i32,
    pub need: i32,
    pub success: Option<bool>,
    pub region: Option<usize>,
    pub berserk: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub players: ArrayVec<Player, MAX_PLAYERS>,
    pub races: [Option<RaceState>; Race::COUNT],
    /// Seules les `map.len()` premières régions existent.
    pub regions: [RegionState; MAX_REGIONS],
    pub market: ArrayVec<Combo, 6>,
    pub race_deck: ArrayVec<Race, { Race::COUNT }>,
    pub power_deck: ArrayVec<Power, { Power::COUNT }>,
    /// Cartes du fond des pioches dont l'ordre est public : rendues sous la pioche quand un peuple
    /// disparaît, un joueur attentif s'en souvient. Le reste de l'ordre est caché.
    pub race_tail: u8,
    pub power_tail: u8,
    pub round: i32,
    pub max_rounds: i32,
    pub current: usize,
    pub phase: Phase,
    pub turn: Turn,
    pub defense: ArrayVec<DefenseTask, 16>,
    /// Au plus une par joueur, plus celle qu'il impose en finissant son tour.
    pub peace: ArrayVec<Peace, { MAX_PLAYERS + 1 }>,
    /// Numéro du dernier événement : il change à chaque événement du journal et sert de sel au hasard de l'IA.
    pub seq: u32,
    /// État du générateur des dés.
    pub rng: u32,
    pub last_die: Option<DieRoll>,
    /// Dernière région touchée par une action (pour l'animation).
    pub last_region: Option<usize>,
}

impl State {
    /// L'état du peuple, qui doit être en jeu (sinon c'est un bogue du noyau).
    pub fn race(&self, race: Race) -> &RaceState {
        self.races[race.index()].as_ref().expect("peuple absent")
    }

    pub fn race_mut(&mut self, race: Race) -> &mut RaceState {
        self.races[race.index()].as_mut().expect("peuple absent")
    }

    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    /// Les régions occupées par ce peuple (`n` : nombre de régions de la carte).
    pub fn owned(&self, race: Race, n: usize) -> RegionSet {
        let mut out = RegionSet::EMPTY;
        for (i, r) in self.regions[..n].iter().enumerate() {
            if r.occupant == Occupant::Race(race) {
                out.insert(i);
            }
        }
        out
    }

    /// Le joueur qui doit agir, ou `None` en fin de partie.
    pub fn actor(&self) -> Option<usize> {
        match self.phase {
            Phase::GameOver => None,
            Phase::Defend => self.defense.first().map(|d| d.player),
            _ => Some(self.current),
        }
    }

    /// Le peuple que contrôle le joueur qui agit.
    pub fn acting_race(&self) -> Option<Race> {
        match self.phase {
            Phase::Defend => self.defense.first().map(|d| d.race),
            Phase::Conquest | Phase::Redeploy => self.turn.army,
            _ => None,
        }
    }
}
