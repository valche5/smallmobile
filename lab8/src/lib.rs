//! Le laboratoire de l'IA hors jeu : tout ce qu'il faut pour chercher la meilleure façon de jouer,
//! sans limite de temps, et pour le mesurer. Le jeu lui-même (`smallmobile-kernel`) n'est pas touché :
//! on le joue tel quel.
//!
//! L'IA Expert (`smallmobile-brain`) est partagée avec le jeu : `know`, `moves`, `encode`, `net`,
//! `rand` en viennent, `az` en est la recherche vue comme un joueur du labo.
//!
//! - `table` : une partie à jouer (carte, joueurs), et son résultat ;
//! - `know` : ce qu'un joueur sait vraiment (pioches et dés cachés, histoire des joueurs) et des
//!   mondes possibles ;
//! - `moves` : les coups d'un joueur, une décision à la fois ;
//! - `agent` : les joueurs (IA du jeu et ses styles, hasard, Expert) ;
//! - `encode`, `net`, `az` : ce que voit le réseau, le réseau, la recherche guidée par le réseau ;
//! - `arena` : des milliers de parties pour comparer deux joueurs ;
//! - `selfplay` : parties d'entraînement du réseau, plusieurs à la fois par fil, leurs évaluations
//!   groupées (`batch`), par le processeur ou par le GPU (`gpu`, moteurs `vulkan` et `cuda`) ;
//!   `luck` : la part du hasard dans leurs résultats, pour une cible de la valeur sans elle.

pub mod agent;
pub mod arena;
pub mod az;
pub mod batch;
pub mod compact;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod gpu;
pub mod luck;
pub mod selfplay;
pub mod table;
#[cfg(feature = "vulkan")]
pub mod vulkan;

pub use smallmobile_brain8::{encode, know, moves, net, rand};
