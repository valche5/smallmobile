//! Le laboratoire de l'IA hors jeu : tout ce qu'il faut pour chercher la meilleure façon de jouer,
//! sans limite de temps, et pour le mesurer. Le jeu lui-même (`smallmobile-kernel`) n'est pas touché :
//! on le joue tel quel.
//!
//! - `table` : une partie à jouer (carte, joueurs), et son résultat ;
//! - `know` : ce qu'un joueur sait vraiment (pioches et dés cachés) et des mondes possibles ;
//! - `moves` : les coups d'un joueur, regroupés pour que la recherche ait peu de choix à la fois ;
//! - `agent` : les joueurs (IA du jeu, hasard, nos IA) ;
//! - `mcts` : recherche arborescente sur mondes possibles, parties finies par l'IA Normal ;
//! - `encode`, `net`, `az` : ce que voit le réseau, le réseau, la recherche guidée par le réseau ;
//! - `arena` : des milliers de parties pour comparer deux joueurs ;
//! - `selfplay` : parties d'entraînement du réseau ;
//! - `label` : positions jugées par des fins de partie simulées (cibles de valeur presque sans bruit).

pub mod agent;
pub mod arena;
pub mod az;
pub mod encode;
pub mod know;
pub mod label;
pub mod mcts;
pub mod moves;
pub mod net;
pub mod rand;
pub mod selfplay;
pub mod table;
