//! L'IA Expert de SmallMobile : une recherche arborescente sur mondes possibles, guidée par un réseau
//! de neurones appris hors du jeu (voir `lab/JOURNAL.md`). Le même code joue dans le jeu (programme
//! `expert`, à côté du serveur) et dans le labo (`smallmobile-lab`), qui l'entraîne et la mesure.
//!
//! - `know` : ce qu'un joueur sait vraiment (pioches et dés cachés) et des mondes possibles ;
//! - `moves` : les coups d'un joueur, regroupés pour que la recherche ait peu de choix à la fois ;
//! - `encode`, `net` : ce que voit le réseau, le réseau ;
//! - `search` : la recherche guidée par le réseau ;
//! - `opponents` : comment jouent les adversaires, deviné d'après leurs coups ;
//! - `outcome` : le résultat d'une partie finie ;
//! - `expert` : l'IA Expert telle que le jeu l'emploie (réglages, coup découpé en actions du noyau).
//!
//! Rien ici ne dépend du système (fichiers, fils), hors du programme `expert`.

pub mod encode;
pub mod expert;
pub mod know;
pub mod moves;
pub mod net;
pub mod opponents;
pub mod outcome;
pub mod rand;
pub mod search;
