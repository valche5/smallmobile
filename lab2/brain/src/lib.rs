//! L'IA Expert de SmallMobile : Gumbel AlphaZero avec nœuds de hasard, guidé par un réseau de
//! neurones appris hors du jeu (voir `lab2/journal.md`). Le même code joue dans le jeu (programme
//! `lab2-expert`, à côté du serveur) et dans le labo (`smallmobile-lab2`), qui l'entraîne et la mesure.
//!
//! - `know` : ce qu'un joueur sait vraiment (pioches et dés cachés, histoire des joueurs) et des
//!   mondes possibles ;
//! - `moves` : les coups d'un joueur, regroupés pour que la recherche ait peu de choix à la fois ;
//! - `encode`, `net` : ce que voit le réseau, le réseau ;
//! - `search` : la recherche guidée par le réseau (Gumbel AlphaZero, nœuds de hasard) ;
//! - `outcome` : le résultat d'une partie finie ;
//! - `expert` : l'IA Expert telle que le jeu l'emploie (réglages, coup découpé en actions du noyau).
//!
//! Rien ici ne dépend du système (fichiers, fils), hors du programme `lab2-expert`.

pub mod encode;
pub mod expert;
pub mod know;
pub mod moves;
pub mod net;
pub mod outcome;
pub mod rand;
pub mod search;
