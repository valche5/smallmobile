//! Les identifiants du jeu : peuples, pouvoirs, terrains, phases. Leur ordre est celui du TypeScript
//! (`RACE_IDS`, `POWER_IDS`, …) : c'est aussi leur numéro dans le protocole et dans les modèles appris.

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
    /// Les 14 peuples.
    Race { Amazons, Dwarves, Elves, Ghouls, Giants, Halflings, Humans, Orcs, Ratmen, Skeletons, Sorcerers, Tritons, Trolls, Wizards }
}

id_enum! {
    /// Les 20 pouvoirs spéciaux.
    Power {
        Alchemist, Berserk, Bivouacking, Commando, Diplomat, DragonMaster, Flying, Forest, Fortified, Heroic, Hill,
        Merchant, Mounted, Pillaging, Seafaring, Spirit, Stout, Swamp, Underworld, Wealthy,
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
        ];
        KEYS[self.index()]
    }

    /// Jetons du peuple seul (sans le pouvoir).
    pub fn tokens(self) -> i32 {
        const TOKENS: [i32; Race::COUNT] = [6, 3, 6, 5, 6, 6, 5, 5, 8, 6, 5, 6, 5, 5];
        TOKENS[self.index()]
    }

    /// Nombre de jetons de ce peuple dans la boîte.
    pub fn max(self) -> i32 {
        const MAX: [i32; Race::COUNT] = [15, 8, 11, 10, 11, 11, 10, 10, 13, 18, 18, 11, 10, 10];
        MAX[self.index()]
    }

    /// Jetons reçus en prenant ce peuple avec ce pouvoir.
    pub fn starting_tokens(self, power: Power) -> i32 {
        (self.tokens() + power.tokens()).min(self.max())
    }
}

impl Power {
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
        ];
        KEYS[self.index()]
    }

    pub fn tokens(self) -> i32 {
        const TOKENS: [i32; Power::COUNT] = [4, 4, 5, 4, 5, 5, 5, 4, 3, 5, 4, 2, 5, 5, 5, 5, 4, 4, 5, 4];
        TOKENS[self.index()]
    }
}

impl Terrain {
    pub fn is_water(self) -> bool {
        matches!(self, Terrain::Sea | Terrain::Lake)
    }
}

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
