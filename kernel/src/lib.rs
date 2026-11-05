//! Le cœur du jeu SmallMobile : les règles et l'IA, en un seul endroit.
//!
//! - compilé en WebAssembly pour le jeu (serveur et téléphone, voir `abi`) ;
//! - l'état d'une partie (`State`) est une valeur `Copy` sans allocation : les simulations en font des
//!   copies à volonté ; la carte (`Map`) est partagée en lecture seule ;
//! - aucun texte : les événements (`Event`) et les raisons de refus (`Reason`) sont traduits par
//!   l'interface ;
//! - tout est déterministe : mêmes état et action, même résultat, sur toutes les plateformes.

pub mod abi;
pub mod action;
pub mod ai;
pub mod collections;
pub mod deploy;
pub mod game;
pub mod geometry;
pub mod ids;
pub mod map;
pub mod mapgen;
pub mod maps;
pub mod reason;
pub mod rng;
pub mod rules;
pub mod setup;
pub mod state;
pub mod wire;

pub use action::{Action, Event, Journal, NoJournal, Placement};
pub use ai::{Ai, Restrictions, Tuning};
pub use game::Game;
pub use ids::{Level, Phase, Power, Race, Terrain};
pub use map::{Map, RegionDef};
pub use reason::Reason;
pub use state::State;
