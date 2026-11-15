//! Les identifiants du jeu : peuples, pouvoirs, terrains, phases. Leur ordre est celui du TypeScript
//! (`RACE_IDS`, `POWER_IDS`, …) : c'est aussi leur numéro dans le protocole et dans les modèles appris.
//! Les peuples et pouvoirs des extensions viennent après ceux du jeu de base (`BASE_COUNT`), pour que
//! les numéros du jeu de base ne bougent pas.

macro_rules! id_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u8)]
        pub enum $name { $($variant),+ }

        /// Valeur des cases vides des listes de taille fixe ; sans signification de jeu.
        impl Default for $name {
            fn default() -> Self {
                Self::ALL[0]
            }
        }

        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$variant),+];
            pub const COUNT: usize = Self::ALL.len();

            pub fn index(self) -> usize {
                self as usize
            }

            pub fn from_index(i: usize) -> Option<Self> {
                Self::ALL.get(i).copied()
            }
        }
    };
}

id_enum! {
    /// Les peuples : les 14 du jeu de base, puis ceux des extensions.
    Race {
        Amazons, Dwarves, Elves, Ghouls, Giants, Halflings, Humans, Orcs, Ratmen, Skeletons, Sorcerers, Tritons,
        Trolls, Wizards,
        // Même pas peur
        Barbarians, Leprechauns, Pixies, Homunculi, Pygmies,
        // Maauuudits
        Goblins, Kobolds,
        // Honneur aux Dames
        WhiteLadies, Gypsies, Priestesses,
    }
}

id_enum! {
    /// Les pouvoirs spéciaux : les 20 du jeu de base, puis ceux des extensions.
    Power {
        Alchemist, Berserk, Bivouacking, Commando, Diplomat, DragonMaster, Flying, Forest, Fortified, Heroic, Hill,
        Merchant, Mounted, Pillaging, Seafaring, Spirit, Stout, Swamp, Underworld, Wealthy,
        // Même pas peur
        Catapult, Corrupt, Imperial, Mercenary, Barricade,
        // Maauuudits
        Hordes, Cursed, Marauding, Ransacking, Were,
        // Honneur aux Dames
        Historian, PeaceLoving,
    }
}

id_enum! {
    /// Les extensions, choisies à la création d'une partie.
    Expansion { BeNotAfraid, Cursed, GrandDames }
}

impl Expansion {
    /// Identifiant de l'extension côté TypeScript.
    pub fn key(self) -> &'static str {
        match self {
            Expansion::BeNotAfraid => "beNotAfraid",
            Expansion::Cursed => "cursed",
            Expansion::GrandDames => "grandDames",
        }
    }
}

/// Un ensemble d'extensions (un bit par extension, dans l'ordre de `Expansion`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Expansions(u8);

impl Expansions {
    /// Le jeu de base seul.
    pub const NONE: Expansions = Expansions(0);
    pub const ALL: Expansions = Expansions((1 << Expansion::COUNT) - 1);

    /// L'ensemble écrit en bits, s'il ne contient que des extensions connues.
    pub fn from_bits(bits: u8) -> Option<Self> {
        (bits & !Self::ALL.0 == 0).then_some(Expansions(bits))
    }

    pub fn bits(self) -> u8 {
        self.0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn contains(self, e: Expansion) -> bool {
        self.0 & 1 << e.index() != 0
    }

    pub fn with(self, e: Expansion) -> Self {
        Expansions(self.0 | 1 << e.index())
    }

    pub fn iter(self) -> impl Iterator<Item = Expansion> {
        Expansion::ALL.iter().copied().filter(move |&e| self.contains(e))
    }

    /// Le peuple est-il dans la boîte : jeu de base, ou une de ces extensions ?
    pub fn has_race(self, race: Race) -> bool {
        race.expansion().is_none_or(|e| self.contains(e))
    }

    pub fn has_power(self, power: Power) -> bool {
        power.expansion().is_none_or(|e| self.contains(e))
    }

    /// Les peuples de la partie, dans l'ordre des numéros.
    pub fn races(self) -> impl Iterator<Item = Race> {
        Race::ALL.iter().copied().filter(move |&r| self.has_race(r))
    }

    /// Les pouvoirs de la partie, dans l'ordre des numéros.
    pub fn powers(self) -> impl Iterator<Item = Power> {
        Power::ALL.iter().copied().filter(move |&p| self.has_power(p))
    }
}

impl FromIterator<Expansion> for Expansions {
    fn from_iter<I: IntoIterator<Item = Expansion>>(iter: I) -> Self {
        iter.into_iter().fold(Expansions::NONE, Expansions::with)
    }
}

id_enum! {
    Terrain { Farmland, Forest, Hill, Mountain, Swamp, Sea, Lake }
}

id_enum! {
    Phase { Pick, Conquest, Redeploy, Defend, GameOver }
}

id_enum! {
    /// Niveau de l'IA qui joue une place. Expert : l'IA du labo (recherche guidée par un réseau,
    /// hors du noyau) ; le noyau, lui, la joue comme Difficile.
    Level { Easy, Normal, Hard, Expert }
}

impl Level {
    /// Le niveau que joue l'IA du noyau.
    pub fn classic(self) -> Level {
        if self == Level::Expert { Level::Hard } else { self }
    }
}

impl Race {
    /// Les peuples du jeu de base : les `BASE_COUNT` premiers numéros.
    pub const BASE_COUNT: usize = 14;

    /// L'extension du peuple (`None` : jeu de base).
    pub const fn expansion(self) -> Option<Expansion> {
        use Race::*;
        match self {
            Barbarians | Leprechauns | Pixies | Homunculi | Pygmies => Some(Expansion::BeNotAfraid),
            Goblins | Kobolds => Some(Expansion::Cursed),
            WhiteLadies | Gypsies | Priestesses => Some(Expansion::GrandDames),
            _ => None,
        }
    }

    /// Identifiant du peuple côté TypeScript (et dans les noms d'entrées des modèles).
    pub fn key(self) -> &'static str {
        const KEYS: [&str; Race::COUNT] = [
            "amazons",
            "dwarves",
            "elves",
            "ghouls",
            "giants",
            "halflings",
            "humans",
            "orcs",
            "ratmen",
            "skeletons",
            "sorcerers",
            "tritons",
            "trolls",
            "wizards",
            "barbarians",
            "leprechauns",
            "pixies",
            "homunculi",
            "pygmies",
            "goblins",
            "kobolds",
            "whiteLadies",
            "gypsies",
            "priestesses",
        ];
        KEYS[self.index()]
    }

    /// Jetons du peuple seul (sans le pouvoir).
    pub fn tokens(self) -> i32 {
        const TOKENS: [i32; Race::COUNT] = [6, 3, 6, 5, 6, 6, 5, 5, 8, 6, 5, 6, 5, 5, 9, 6, 11, 5, 6, 6, 11, 2, 6, 4];
        TOKENS[self.index()]
    }

    /// Nombre de jetons de ce peuple dans la boîte.
    pub fn max(self) -> i32 {
        const MAX: [i32; Race::COUNT] =
            [15, 8, 11, 10, 11, 11, 10, 10, 13, 18, 18, 11, 10, 10, 14, 11, 16, 15, 11, 11, 16, 7, 11, 9];
        MAX[self.index()]
    }

    /// Jetons reçus en prenant ce peuple avec ce pouvoir.
    pub fn starting_tokens(self, power: Power) -> i32 {
        (self.tokens() + power.tokens()).min(self.max())
    }
}

impl Power {
    /// Les pouvoirs du jeu de base : les `BASE_COUNT` premiers numéros.
    pub const BASE_COUNT: usize = 20;

    /// L'extension du pouvoir (`None` : jeu de base).
    pub const fn expansion(self) -> Option<Expansion> {
        use Power::*;
        match self {
            Catapult | Corrupt | Imperial | Mercenary | Barricade => Some(Expansion::BeNotAfraid),
            Hordes | Cursed | Marauding | Ransacking | Were => Some(Expansion::Cursed),
            Historian | PeaceLoving => Some(Expansion::GrandDames),
            _ => None,
        }
    }

    /// Identifiant du pouvoir côté TypeScript.
    pub fn key(self) -> &'static str {
        const KEYS: [&str; Power::COUNT] = [
            "alchemist",
            "berserk",
            "bivouacking",
            "commando",
            "diplomat",
            "dragonMaster",
            "flying",
            "forest",
            "fortified",
            "heroic",
            "hill",
            "merchant",
            "mounted",
            "pillaging",
            "seafaring",
            "spirit",
            "stout",
            "swamp",
            "underworld",
            "wealthy",
            "catapult",
            "corrupt",
            "imperial",
            "mercenary",
            "barricade",
            "hordes",
            "cursed",
            "marauding",
            "ransacking",
            "were",
            "historian",
            "peaceLoving",
        ];
        KEYS[self.index()]
    }

    pub fn tokens(self) -> i32 {
        const TOKENS: [i32; Power::COUNT] =
            [4, 4, 5, 4, 5, 5, 5, 4, 3, 5, 4, 2, 5, 5, 5, 5, 4, 4, 5, 4, 4, 4, 4, 4, 4, 5, 0, 5, 4, 4, 5, 4];
        TOKENS[self.index()]
    }
}

impl Terrain {
    pub fn is_water(self) -> bool {
        matches!(self, Terrain::Sea | Terrain::Lake)
    }
}

// les numéros du jeu de base d'abord, ceux des extensions ensuite
const _: () = {
    let mut i = 0;
    while i < Race::COUNT {
        assert!(Race::ALL[i].expansion().is_none() == (i < Race::BASE_COUNT));
        i += 1;
    }
    let mut i = 0;
    while i < Power::COUNT {
        assert!(Power::ALL[i].expansion().is_none() == (i < Power::BASE_COUNT));
        i += 1;
    }
};

/// Faces du dé de renfort.
pub const DIE_FACES: [i32; 6] = [0, 0, 0, 1, 2, 3];

/// Probabilité qu'un jet du dé de renfort apporte au moins `need` jetons.
pub fn die_chance(need: i32) -> f64 {
    if need <= 0 {
        return 1.0;
    }
    DIE_FACES.iter().filter(|&&f| f >= need).count() as f64 / DIE_FACES.len() as f64
}

/// Plus grand nombre de joueurs et de régions qu'accepte le noyau.
pub const MAX_PLAYERS: usize = 5;
pub const MAX_REGIONS: usize = 64;
