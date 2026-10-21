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
//! - `moves` : les coups d'un joueur, regroupés pour que la recherche ait peu de choix à la fois ;
//! - `agent` : les joueurs (IA du jeu et ses styles, hasard, Expert) ;
//! - `encode`, `net`, `az` : ce que voit le réseau, le réseau, la recherche guidée par le réseau ;
//! - `arena` : des milliers de parties pour comparer deux joueurs ;
//! - `selfplay` : parties d'entraînement du réseau ;
//! - `judge` : positions rejugées par des fins de partie simulées (cibles de valeur presque sans
//!   bruit).

pub mod agent;
pub mod arena;
pub mod az;
pub mod judge;
pub mod selfplay;
pub mod table;

pub use smallmobile_brain2::{encode, know, moves, net, rand};
