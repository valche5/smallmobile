//! Ce que voit l'encodeur : les entrées de lab7 (`encode`), regroupées en jetons, et les relations
//! entre jetons qui portent la carte dans l'attention.
//!
//! Jetons, dans l'ordre :
//! - une région par jeton (`REGIONS` places, les absentes masquées) : sa ligne de `encode` (77) ;
//! - un joueur relatif par jeton (`PLAYERS`) : son bloc de la partie (81), ses habitudes (14), les
//!   régions qu'il a prises à chacun et que chacun lui a prises, depuis le début et pendant le
//!   dernier tour (4 × 5) ;
//! - une carte du marché par jeton (`MARKET`) : son bloc (38) ;
//! - la partie : les 67 premières entrées de la partie (manche, phase, tour en cours, pioches).
//!
//! Les 820 entrées de la partie sont toutes reprises ; les régions prises de chacun à chacun, deux
//! fois (chez celui qui prend, chez celui qui perd), les autres une seule. Les relations : entre deux
//! régions, leur distance en voisinages (0 à 4, 4 pour « 4 ou plus ») ; entre une région et un joueur,
//! le peuple qui l'occupe (actif ou en déclin), dans les deux sens ; ailleurs, aucune.
//!
//! Un pas du décodeur reçoit en plus `DYN` entrées de la position où il choisit (`dynamics`) : ce
//! qui change pendant un plan, hors des régions.

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::map::Map;
use smallmobile_kernel::state::Stage;
use smallmobile_kernel::{Game, Phase};

use crate::encode::{Encoded, GLOBAL, NODE, REGIONS};
use crate::know::Knowledge;

pub const PLAYERS: usize = MAX_PLAYERS;
pub const MARKET: usize = 6;
pub const TOKENS: usize = REGIONS + PLAYERS + MARKET + 1;
/// Le premier jeton des joueurs, du marché ; le jeton de la partie.
pub const T_PLAYER: usize = REGIONS;
pub const T_MARKET: usize = REGIONS + PLAYERS;
pub const T_GAME: usize = TOKENS - 1;

// les blocs de la partie dans `encode` (lab7/brain/src/encode.rs)
const G_PLAYER: usize = 67;
const P: usize = 81;
const G_MARKET: usize = G_PLAYER + PLAYERS * P;
const M: usize = 38;
const G_HABITS: usize = G_MARKET + MARKET * M;
const HABITS: usize = 14;
const G_ATTACKS: usize = G_HABITS + PLAYERS * HABITS;
const G_RECENT: usize = G_ATTACKS + PLAYERS * PLAYERS;
const _: () = assert!(G_RECENT + PLAYERS * PLAYERS == GLOBAL && NODE == 77 && G_HABITS == 700);

/// Entrées d'un jeton de région, de joueur, du marché, de la partie.
pub const REGION_IN: usize = NODE;
pub const PLAYER_IN: usize = P + HABITS + 4 * PLAYERS;
pub const MARKET_IN: usize = M;
pub const GAME_IN: usize = G_PLAYER;

/// Relations : aucune ; distance entre deux régions (1 + distance, de 0 à 4) ; région vers le joueur
/// dont le peuple l'occupe (actif, en déclin) ; joueur vers ces régions.
pub const RELATIONS: usize = 10;
const R_DISTANCE: u8 = 1;
const R_TO_ACTIVE: u8 = 6;
const R_TO_DECLINED: u8 = 7;
const R_FROM_ACTIVE: u8 = 8;
const R_FROM_DECLINED: u8 = 9;
/// Distance plafonnée : 4 voisinages ou plus, ou inaccessible.
pub const FAR: u8 = 4;

// entrées des régions utilisées pour les relations
const N_OWNER: usize = 17;
const N_ACTIVE: usize = 22;
const N_DECLINED: usize = 23;

/// Les entrées du jeton du joueur relatif `r`.
pub fn player(global: &[u8; GLOBAL], r: usize) -> [u8; PLAYER_IN] {
    let mut x = [0u8; PLAYER_IN];
    x[..P].copy_from_slice(&global[G_PLAYER + r * P..][..P]);
    x[P..P + HABITS].copy_from_slice(&global[G_HABITS + r * HABITS..][..HABITS]);
    let at = P + HABITS;
    for t in 0..PLAYERS {
        x[at + t] = global[G_ATTACKS + r * PLAYERS + t];
        x[at + PLAYERS + t] = global[G_ATTACKS + t * PLAYERS + r];
        x[at + 2 * PLAYERS + t] = global[G_RECENT + r * PLAYERS + t];
        x[at + 3 * PLAYERS + t] = global[G_RECENT + t * PLAYERS + r];
    }
    x
}

/// Les entrées du jeton de la carte `m` du marché.
pub fn market(global: &[u8; GLOBAL], m: usize) -> &[u8] {
    &global[G_MARKET + m * M..][..M]
}

/// Les entrées du jeton de la partie.
pub fn game(global: &[u8; GLOBAL]) -> &[u8] {
    &global[..GAME_IN]
}

/// Le joueur relatif `r` est-il à la table ? La carte `m` est-elle au marché ?
pub fn player_present(global: &[u8; GLOBAL], r: usize) -> bool {
    global[G_PLAYER + r * P] != 0
}

pub fn market_present(global: &[u8; GLOBAL], m: usize) -> bool {
    global[G_MARKET + m * M] != 0
}

/// Les jetons présents.
pub fn valid(e: &Encoded) -> [bool; TOKENS] {
    let mut v = [false; TOKENS];
    v[..e.regions].fill(true);
    for r in 0..PLAYERS {
        v[T_PLAYER + r] = player_present(&e.global, r);
    }
    for m in 0..MARKET {
        v[T_MARKET + m] = market_present(&e.global, m);
    }
    v[T_GAME] = true;
    v
}

/// Les distances entre les régions de la carte, en voisinages, plafonnées à `FAR` (`REGIONS` ×
/// `REGIONS`, ligne par ligne ; `FAR` hors de la carte).
pub fn distances(map: &Map) -> Vec<u8> {
    let n = map.len();
    let mut d = vec![FAR; REGIONS * REGIONS];
    for i in 0..n {
        d[i * REGIONS + i] = 0;
        let mut seen = 1u64 << i;
        let mut front = seen;
        for step in 1..FAR {
            let mut next = 0u64;
            let mut bits = front;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                next |= map.hoods[j].adj.0;
            }
            next &= !seen;
            seen |= next;
            front = next;
            let mut bits = next;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                d[i * REGIONS + j] = step;
            }
        }
    }
    d
}

/// Les relations entre jetons (`TOKENS` × `TOKENS`, ligne par ligne : la requête, puis la clé).
pub fn relations(e: &Encoded, dist: &[u8]) -> Vec<u8> {
    let mut rel = vec![0u8; TOKENS * TOKENS];
    for i in 0..e.regions {
        for j in 0..e.regions {
            rel[i * TOKENS + j] = R_DISTANCE + dist[i * REGIONS + j];
        }
        let row = &e.node[i];
        if let Some(r) = (0..PLAYERS).find(|&r| row[N_OWNER + r] != 0) {
            let (to, from) = if row[N_DECLINED] != 0 {
                (R_TO_DECLINED, R_FROM_DECLINED)
            } else {
                debug_assert!(row[N_ACTIVE] != 0);
                (R_TO_ACTIVE, R_FROM_ACTIVE)
            };
            rel[i * TOKENS + T_PLAYER + r] = to;
            rel[(T_PLAYER + r) * TOKENS + i] = from;
        }
    }
    rel
}

/// Entrées d'un pas du décodeur : ce qui change pendant un plan, hors des régions.
pub const DYN: usize = 20;
/// Leur place dans les entrées de la partie de `encode` (les deux dernières n'y sont pas : le
/// redéploiement commencé, les pouvoirs posés, `moves::Redeploy`).
pub const DYN_GLOBAL: [usize; DYN - 2] = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 63];

/// Les entrées d'un pas du décodeur dans la position `g` (les mêmes que `encode` aux places
/// `DYN_GLOBAL`, sans calculer le reste).
pub fn dynamics(g: &Game, k: &Knowledge) -> [u8; DYN] {
    let s = &g.state;
    let t = &s.turn;
    let put = |v: i32| v.clamp(0, 255) as u8;
    let mut x = [0u8; DYN];
    x[s.phase.index()] = 1;
    x[5] = (t.stage == Stage::Ghouls) as u8;
    x[6] = put(t.conquests);
    x[7] = put(t.non_empty);
    x[8] = t.die_rolled as u8;
    x[9] = t.berserk_die.is_some() as u8;
    x[10] = put(t.berserk_die.unwrap_or(0));
    x[11] = t.dragon_used as u8;
    x[12] = t.picked as u8;
    x[13] = t.declined as u8;
    x[14] = put(t.amazon_bonus);
    x[15] = t.fortress_region.is_some() as u8;
    if let Some(race) = s.acting_race() {
        x[16] = put(s.race(race).hand);
    }
    x[17] = put(s.defense.len() as i32);
    x[18] = (s.phase == Phase::Redeploy && k.redeploy.lifted) as u8;
    x[19] = (s.phase == Phase::Redeploy && k.redeploy.powers_done) as u8;
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode;
    use crate::moves;
    use crate::rand::Rng;
    use smallmobile_kernel::collections::RegionSet;
    use smallmobile_kernel::{Level, NoJournal, mapgen};

    /// Chaque entrée de la partie est reprise une seule fois dans les jetons ; les entrées d'un pas
    /// du décodeur sont celles de `encode` ; les distances sont symétriques et valent 1 entre voisines.
    #[test]
    fn tokens_take_every_input_once() {
        let mut rng = Rng::new(3);
        let mut checked = 0;
        for game in 0..12u64 {
            let n = 2 + game as usize % 4;
            let generated = mapgen::generate(n, 700 + game).unwrap();
            let map = generated.to_map().unwrap();
            let lost: RegionSet =
                generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
            let mut g =
                Game::create(&map, lost, &vec![Level::Normal; n], generated.rounds, game as u32, &mut NoJournal);
            let mut k = Knowledge::default();
            let dist = distances(&map);
            for i in 0..map.len() {
                for j in 0..map.len() {
                    assert_eq!(dist[i * REGIONS + j], dist[j * REGIONS + i]);
                }
                for &j in &map.regions[i].adj {
                    assert_eq!(dist[i * REGIONS + j as usize], 1);
                }
            }
            while g.state.phase != Phase::GameOver {
                let e = encode(&g, &k);
                let dy = dynamics(&g, &k);
                for (i, &at) in DYN_GLOBAL.iter().enumerate() {
                    assert_eq!(dy[i], e.global[at], "entrée {at} de la partie");
                }
                // chaque entrée de la partie, retrouvée par sa valeur marquée
                let mut marked = e.clone();
                let mut seen = vec![0u32; GLOBAL];
                for (at, x) in marked.global.iter_mut().enumerate() {
                    *x = (at % 251) as u8 + 1;
                }
                let mut count = |v: &[u8], from: &dyn Fn(usize) -> usize| {
                    for (i, &x) in v.iter().enumerate() {
                        let at = from(i);
                        assert_eq!(x as usize, at % 251 + 1);
                        seen[at] += 1;
                    }
                };
                count(game_inputs(&marked), &|i| i);
                for m in 0..MARKET {
                    count(market(&marked.global, m), &|i| G_MARKET + m * M + i);
                }
                for r in 0..PLAYERS {
                    let x = player(&marked.global, r);
                    count(&x[..P], &|i| G_PLAYER + r * P + i);
                    count(&x[P..P + HABITS], &|i| G_HABITS + r * HABITS + i);
                    let at = P + HABITS;
                    count(&x[at..at + PLAYERS], &|t| G_ATTACKS + r * PLAYERS + t);
                    count(&x[at + 2 * PLAYERS..at + 3 * PLAYERS], &|t| G_RECENT + r * PLAYERS + t);
                }
                assert!(seen.iter().all(|&c| c == 1), "chaque entrée une fois");
                let rel = relations(&e, &dist);
                assert!(rel.iter().all(|&r| (r as usize) < RELATIONS));
                let legal = moves::legal(&g, &k);
                let m = legal[rng.below(legal.len())].mv;
                k.play(&mut g, &m).unwrap();
                checked += 1;
            }
        }
        assert!(checked > 1000, "{checked} positions");
    }

    fn game_inputs(e: &Encoded) -> &[u8] {
        game(&e.global)
    }
}
