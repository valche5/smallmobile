//! La part du hasard dans le résultat d'une partie d'entraînement, pour une cible de la valeur qui ne
//! la garde pas (lab7, phase 2 : `selfplay --luck 1`, `train.py --luck`).
//!
//! À chaque coup dont la suite dépend du hasard (la carte qui arrive au marché après un choix de
//! peuple, le dé de renfort, le dé des Berserks) : δ = V(après) − E[V(après)], la part de victoire de
//! chaque joueur selon le réseau après le résultat tiré, moins sa moyenne sur les résultats possibles.
//! Chaque δ est d'espérance nulle, quel que soit le réseau : retrancher du résultat de la partie les δ
//! qui suivent une position laisse une cible de même espérance, sans biais, et bien moins dispersée
//! si le réseau est bon (variable de contrôle, comme AIVAT pour les parties de poker). Le TD de lab4,
//! lui, remplaçait le résultat par une estimation de la recherche : biaisé.
//!
//! - choix de peuple : la carte vient de la partie cachée des pioches ; la moyenne est estimée sur
//!   `DRAWS` autres tirages (`determinize`), indépendants du vrai : δ reste d'espérance nulle ;
//! - dé de renfort : réussite ou échec, de probabilité exacte (`ids::die_chance`) ; l'issue ne dépend
//!   que de la réussite (la conquête prend tous les jetons en main) ;
//! - dé des Berserks : ses quatre valeurs (`ids::DIE_FACES` : 0 la moitié du temps, 1, 2, 3).

use smallmobile_kernel::Game;
use smallmobile_kernel::ids::{DIE_FACES, MAX_PLAYERS, die_chance};
use smallmobile_kernel::rng::roll_die;

use crate::az::{Az, Oracle};
use crate::encode::encode;
use crate::know::{Knowledge, determinize};
use crate::moves::{Kind, Move};
use crate::rand::Rng;

/// Autres tirages du marché pour estimer la moyenne après un choix de peuple.
pub const DRAWS: usize = 4;

/// Part de victoire de chaque joueur (réel).
pub type Shares = [f32; MAX_PLAYERS];

/// Ce coup dépend-il du hasard ?
pub fn is_chance(mv: &Move) -> bool {
    matches!(mv, Move::Pick(_) | Move::Berserk | Move::Region(Kind::Reinforce, _))
}

/// δ du coup `mv`, joué depuis `before` (ce que savaient les joueurs : `kb`) jusqu'à `after` (`ka`) ;
/// `None` pour un coup sans hasard, ou dont l'issue était certaine. Les évaluations viennent de
/// `oracle` ; `rng` sert aux autres tirages du marché.
#[allow(clippy::too_many_arguments)]
pub async fn luck(
    az: &Az,
    before: &Game<'_>,
    kb: &Knowledge,
    mv: &Move,
    after: &Game<'_>,
    ka: &Knowledge,
    rng: &mut Rng,
    oracle: &impl Oracle,
) -> Option<Shares> {
    match mv {
        Move::Pick(_) => {
            let actual = shares(az, after, ka, oracle).await;
            let mut mean = [0.0f32; MAX_PLAYERS];
            for _ in 0..DRAWS {
                let mut w = determinize(before, kb, rng);
                let mut kw = *kb;
                kw.play(&mut w, mv).expect("le même choix, dans un autre monde");
                let v = shares(az, &w, &kw, oracle).await;
                mean.iter_mut().zip(v).for_each(|(m, x)| *m += x / DRAWS as f32);
            }
            Some(std::array::from_fn(|q| actual[q] - mean[q]))
        }
        Move::Region(Kind::Reinforce, _) => {
            let die = after.state.last_die.filter(|d| !d.berserk)?;
            let p = die_chance(die.need) as f32;
            if p <= 0.0 || p >= 1.0 {
                return None;
            }
            // la plus forte face réussit, la plus faible échoue (sinon l'issue serait certaine)
            let (w, kw) = faced(before, kb, mv, 3);
            let won = shares(az, &w, &kw, oracle).await;
            let (w, kw) = faced(before, kb, mv, 0);
            let lost = shares(az, &w, &kw, oracle).await;
            let actual = if die.success? { won } else { lost };
            Some(std::array::from_fn(|q| actual[q] - (p * won[q] + (1.0 - p) * lost[q])))
        }
        Move::Berserk => {
            let rolled = after.state.turn.berserk_die?;
            let (mut expected, mut actual) = ([0.0f32; MAX_PLAYERS], None);
            for face in 0..=3 {
                let p = DIE_FACES.iter().filter(|&&f| f == face).count() as f32 / DIE_FACES.len() as f32;
                let (w, kw) = faced(before, kb, mv, face);
                let v = shares(az, &w, &kw, oracle).await;
                expected.iter_mut().zip(v).for_each(|(e, x)| *e += p * x);
                if face == rolled {
                    actual = Some(v);
                }
            }
            let actual = actual?;
            Some(std::array::from_fn(|q| actual[q] - expected[q]))
        }
        _ => None,
    }
}

/// La partie après `mv` joué depuis `before`, le dé donnant `face` (son générateur réglé pour).
fn faced<'m>(before: &Game<'m>, kb: &Knowledge, mv: &Move, face: i32) -> (Game<'m>, Knowledge) {
    let seed = (0u32..)
        .find(|&s| {
            let mut r = s;
            roll_die(&mut r) == face
        })
        .expect("une graine pour chaque face");
    let (mut w, mut kw) = (*before, *kb);
    w.state.rng = seed;
    kw.play(&mut w, mv).expect("le même coup, avec une autre face du dé");
    (w, kw)
}

/// La part de victoire de chaque joueur (réel) selon le réseau, dans la position `w`.
async fn shares(az: &Az, w: &Game<'_>, k: &Knowledge, oracle: &impl Oracle) -> Shares {
    let eval = oracle.eval(&az.net, &encode(w, k), w.map).await;
    let (me, n) = (w.state.actor().expect("un joueur agit après un coup à hasard"), w.state.player_count());
    let mut out = [0.0; MAX_PLAYERS];
    for r in 0..n {
        out[(me + r) % n] = eval.win[r];
    }
    out
}
