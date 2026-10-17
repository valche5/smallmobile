//! Pourquoi une action est refusée, ou une région hors d'atteinte. Le noyau ne contient aucun texte :
//! l'interface traduit chaque raison en français (`src/engine/kernel.ts`).

use crate::state::Immunity;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    // --- phase -------------------------------------------------------------------------------
    NotPickPhase,
    NotConquestPhase,
    NotRedeployPhase,
    NotEndOfTurn,
    NoDefenseInProgress,
    /// Pouvoir de fin de tour hors du redéploiement du peuple actif.
    NotActiveRedeploy,
    GameOver,
    // --- choix du peuple ---------------------------------------------------------------------
    ComboUnavailable,
    NotEnoughCoins {
        needed: i32,
    },
    MarketNotEmpty,
    // --- conquêtes ---------------------------------------------------------------------------
    AlreadyYours,
    OwnActiveRace,
    GhoulsOwnRace,
    /// Paix imposée par les Diplomates de ce joueur.
    Peace {
        diplomat: usize,
    },
    Water {
        sea: bool,
    },
    Immune(Immunity),
    FirstConquest,
    NotAdjacent,
    DieRolled,
    NoTokenInHand,
    TokensNeeded {
        needed: i32,
    },
    EnoughTokensForNormal,
    DieCannotSucceed,
    NotBerserk,
    BerserkAlreadyRolled,
    NeedOneToken,
    DragonUsed,
    DragonImpossible,
    SorceryDeclinedTarget,
    SorceryNotIsolated,
    SorceryAlreadyCast {
        player: usize,
    },
    NoSorcererLeft,
    SorceryImpossible,
    // --- abandon et déclin -------------------------------------------------------------------
    AbandonNotNow,
    NotArmyRegion,
    AbandonAfterConquest,
    DeclineNotNow,
    GhoulsFirst,
    NoActiveRace,
    DeclinePickedThisTurn,
    DeclineAfterConquest,
    // --- redéploiement -----------------------------------------------------------------------
    NotYourRegion,
    KeepOneToken,
    NotEnoughTokens,
    MissingPower,
    AllCampsPlaced,
    NoCampHere,
    OneFortressPerTurn,
    FortressAlreadyHere,
    AllFortressesBuilt,
    AllHeroesPlaced,
    PlaceHandFirst {
        hand: i32,
    },
    OnlyDiplomats,
    InvalidOpponent,
    AttackedThisTurn,
    OnlyStout,
    // --- action mal formée -------------------------------------------------------------------
    InvalidRegion,
}

impl Reason {
    /// Numéro stable de la raison dans le protocole, et ses deux paramètres éventuels.
    pub fn code(self) -> (u32, i32) {
        use Reason::*;
        let (code, arg) = match self {
            NotPickPhase => (1, 0),
            NotConquestPhase => (2, 0),
            NotRedeployPhase => (3, 0),
            NotEndOfTurn => (4, 0),
            NoDefenseInProgress => (5, 0),
            NotActiveRedeploy => (6, 0),
            GameOver => (7, 0),
            ComboUnavailable => (8, 0),
            NotEnoughCoins { needed } => (9, needed),
            MarketNotEmpty => (10, 0),
            AlreadyYours => (11, 0),
            OwnActiveRace => (12, 0),
            GhoulsOwnRace => (13, 0),
            Peace { diplomat } => (14, diplomat as i32),
            Water { sea } => (15, sea as i32),
            Immune(Immunity::Hero) => (16, 0),
            Immune(Immunity::Hole) => (16, 1),
            Immune(Immunity::Dragon) => (16, 2),
            FirstConquest => (17, 0),
            NotAdjacent => (18, 0),
            DieRolled => (19, 0),
            NoTokenInHand => (20, 0),
            TokensNeeded { needed } => (21, needed),
            EnoughTokensForNormal => (22, 0),
            DieCannotSucceed => (23, 0),
            NotBerserk => (24, 0),
            BerserkAlreadyRolled => (25, 0),
            NeedOneToken => (26, 0),
            DragonUsed => (27, 0),
            DragonImpossible => (28, 0),
            SorceryDeclinedTarget => (29, 0),
            SorceryNotIsolated => (30, 0),
            SorceryAlreadyCast { player } => (31, player as i32),
            NoSorcererLeft => (32, 0),
            SorceryImpossible => (33, 0),
            AbandonNotNow => (34, 0),
            NotArmyRegion => (35, 0),
            AbandonAfterConquest => (36, 0),
            DeclineNotNow => (37, 0),
            GhoulsFirst => (38, 0),
            NoActiveRace => (39, 0),
            DeclinePickedThisTurn => (40, 0),
            DeclineAfterConquest => (41, 0),
            NotYourRegion => (42, 0),
            KeepOneToken => (43, 0),
            NotEnoughTokens => (44, 0),
            MissingPower => (45, 0),
            AllCampsPlaced => (46, 0),
            NoCampHere => (47, 0),
            OneFortressPerTurn => (48, 0),
            FortressAlreadyHere => (49, 0),
            AllFortressesBuilt => (50, 0),
            AllHeroesPlaced => (51, 0),
            PlaceHandFirst { hand } => (52, hand),
            OnlyDiplomats => (53, 0),
            InvalidOpponent => (54, 0),
            AttackedThisTurn => (55, 0),
            OnlyStout => (56, 0),
            InvalidRegion => (57, 0),
        };
        (code, arg)
    }
}

/// `Ok(())` si `ok`, sinon la raison.
pub fn ensure(ok: bool, reason: Reason) -> Result<(), Reason> {
    if ok { Ok(()) } else { Err(reason) }
}
