//! Le laboratoire de l'IA hors jeu : tout ce qu'il faut pour chercher la meilleure façon de jouer,
//! sans limite de temps, et pour le mesurer. Le jeu lui-même (`smallmobile-kernel`) n'est pas touché :
//! on le joue tel quel.
//!
//! - `table` : une partie à jouer (carte, joueurs), et son résultat ;
//! - `know` : ce qu'un joueur sait vraiment (pioches et dés cachés) et des mondes possibles ;
//! - `moves` : les coups d'un joueur, regroupés pour que la recherche ait peu de choix à la fois ;
//! - `agent` : les joueurs (IA du jeu, hasard, nos IA) ;
//! - `arena` : des milliers de parties pour comparer deux joueurs.

pub mod agent;
pub mod arena;
pub mod know;
pub mod moves;
pub mod rand;
pub mod table;
