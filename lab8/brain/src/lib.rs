//! L'IA Expert du labo 8 (`lab8/lab.md`) : un transformer lit la position, un décodeur écrit le plan
//! du joueur qui agit, coup par coup ; la recherche choisit entre des plans entiers.
//!
//! Ce qui ne change pas vient de lab7 (figé) : les coups (`moves`), ce que sait un joueur et les
//! mondes possibles (`know`), le hasard (`rand`), le résultat (`outcome`) et les entrées du réseau
//! (`encode`), que `tokens` regroupe en jetons.
//!
//! - `tokens` : les jetons de l'encodeur, les relations entre jetons, les entrées d'un pas du décodeur ;
//! - `net` : le réseau (référence sur processeur, la même que `lab8/py/model.py`) ;
//! - `plan` : le découpage d'une partie en plans, et l'écriture d'un plan par le décodeur.
//!
//! Rien ici ne dépend du système (fichiers, fils).

pub mod net;
pub mod plan;
pub mod tokens;

pub use smallmobile_brain7::{encode, know, moves, outcome, rand};
