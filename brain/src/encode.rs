//! Ce que le réseau voit d'une position : uniquement ce qu'un joueur sait, du point de vue du joueur
//! qui doit agir (« moi » = joueur relatif 0, puis les suivants dans l'ordre du jeu).
//!
//! Toutes les entrées sont de petits entiers (0 à 255) : comptes, présences, numéros. Le réseau les
//! met à l'échelle lui-même (échelles choisies dans `lab/py/model.py`, livrées avec les poids). Ainsi
//! les positions s'enregistrent sur un octet par entrée.
//!
//! - une ligne par région (`NODE` entrées) : la carte, l'occupant, les jetons, ce que le peuple qui
//!   conquiert peut y faire ;
//! - une ligne pour la partie (`GLOBAL` entrées) : manche, phase, tour en cours, chaque joueur
//!   (pièces, peuples, revenus), le marché ; puis (version 2) l'âge des peuples, le contenu des
//!   pioches, les prochains peuples connus et toutes les paix.
//!
//! Les entrées de la version 2 suivent les `GLOBAL_V1` premières : un réseau de la version 1 lit
//! le début de la ligne et ignore le reste.

use smallmobile_kernel::ids::{MAX_PLAYERS, Power, Race};
use smallmobile_kernel::rules::TargetKind;
use smallmobile_kernel::state::{Occupant, Stage};
use smallmobile_kernel::{Game, Phase};

/// Plus grand nombre de régions des cartes générées (5 joueurs).
pub const REGIONS: usize = 48;

// --- entrées par région ----------------------------------------------------------------------------
const N_TERRAIN: usize = 0; // 7
const N_CAVERN: usize = 7;
const N_MAGIC: usize = 8;
const N_MINE: usize = 9;
const N_ENTRY: usize = 10;
const N_BORDER: usize = 11;
const N_COASTAL: usize = 12;
const N_CENTRE: usize = 13; // en dixièmes
const N_DEGREE: usize = 14;
const N_EMPTY: usize = 15;
const N_LOST: usize = 16;
const N_OWNER: usize = 17; // 5, joueur relatif
const N_ACTIVE: usize = 22;
const N_DECLINED: usize = 23;
const N_RACE: usize = 24; // 14
const N_POWER: usize = 38; // 20
const N_TOKENS: usize = 58;
const N_CAMPS: usize = 59;
const N_LAIR: usize = 60;
const N_FORTRESS: usize = 61;
const N_HERO: usize = 62;
const N_DRAGON: usize = 63;
const N_HOLE: usize = 64;
const N_DEFENSE: usize = 65;
const N_NORMAL: usize = 66;
const N_DIE: usize = 67;
const N_COST: usize = 68;
const N_NEED: usize = 69;
const N_SORCERY: usize = 70;
const N_DRAGON_OK: usize = 71;
const N_ABANDON: usize = 72;
const N_READIED: usize = 73;
const N_ARMY: usize = 74;
const N_MINE_OWN: usize = 75;
const N_ARMY_NEAR: usize = 76;
pub const NODE: usize = 77;

// --- entrées de la partie --------------------------------------------------------------------------
const G_PLAYERS: usize = 0; // 4 : 2 à 5 joueurs
const G_ROUND: usize = 4;
const G_MAX_ROUNDS: usize = 5;
const G_LEFT: usize = 6;
const G_CURRENT: usize = 7; // 5, joueur relatif dont c'est le tour
const G_PHASE: usize = 12; // 5
const G_GHOULS: usize = 17;
const G_CONQUESTS: usize = 18;
const G_NON_EMPTY: usize = 19;
const G_DIE_ROLLED: usize = 20;
const G_BERSERK_ROLLED: usize = 21;
const G_BERSERK_DIE: usize = 22;
const G_DRAGON_USED: usize = 23;
const G_PICKED: usize = 24;
const G_DECLINED: usize = 25;
const G_AMAZONS: usize = 26;
const G_FORTRESS_BUILT: usize = 27;
const G_HAND: usize = 28;
const G_ARMY_RACE: usize = 29; // 14
const G_ARMY_POWER: usize = 43; // 20
const G_DEFENSE_TASKS: usize = 63;
const G_RACE_DECK: usize = 64;
const G_POWER_DECK: usize = 65;
const G_KNOWN_TAIL: usize = 66;
const G_PLAYER: usize = 67; // 5 × P
const P_PRESENT: usize = 0;
const P_COINS: usize = 1;
const P_LAST_INCOME: usize = 2;
const P_INCOME: usize = 3;
const P_RACE: usize = 4; // 14
const P_POWER: usize = 18; // 20
const P_HAND: usize = 38;
const P_REGIONS: usize = 39;
const P_TOKENS: usize = 40;
const P_DECLINED_RACES: usize = 41; // 14
const P_DECLINED_POWERS: usize = 55; // 20
const P_DECLINED_REGIONS: usize = 75;
const P_ATTACKED: usize = 76;
const P_SORCERY: usize = 77;
/// Ce joueur m'a imposé la paix (je ne peux pas attaquer son peuple actif).
const P_PEACE_FROM: usize = 78;
/// Je lui ai imposé la paix.
const P_PEACE_TO: usize = 79;
/// A déjà joué cette manche.
const P_PLAYED: usize = 80;
const P: usize = 81;
const G_MARKET: usize = G_PLAYER + MAX_PLAYERS * P; // 6 × M
const M_PRESENT: usize = 0;
const M_RACE: usize = 1; // 14
const M_POWER: usize = 15; // 20
const M_COINS: usize = 35;
const M_TOKENS: usize = 36;
const M_AFFORDABLE: usize = 37;
const M: usize = 38;
/// Les entrées de la partie de la version 1 (réseaux jusqu'à gen10).
pub const GLOBAL_V1: usize = G_MARKET + 6 * M;
// --- version 2 : des informations publiques que la version 1 laissait de côté (`lab audit`) ----------
/// Âge du peuple actif de chaque joueur relatif : manches depuis son choix.
const G_AGE: usize = GLOBAL_V1; // 5
/// Peuples dans la pioche : ni en jeu ni au marché (contenu public, ordre caché).
const G_DECK_RACES: usize = G_AGE + MAX_PLAYERS; // 14
/// Les prochains peuples connus de la pioche (rendus dessous, ordre public), dans l'ordre où ils
/// sortiront après les cachés.
const G_NEXT_RACES: usize = G_DECK_RACES + Race::COUNT; // NEXT_KNOWN × 14
const NEXT_KNOWN: usize = 3;
/// Pouvoirs dans la pioche.
const G_DECK_POWERS: usize = G_NEXT_RACES + NEXT_KNOWN * Race::COUNT; // 20
/// Toutes les paix des Diplomates : `diplomate relatif × 5 + cible relative`.
const G_PEACE: usize = G_DECK_POWERS + Power::COUNT; // 25
pub const GLOBAL: usize = G_PEACE + MAX_PLAYERS * MAX_PLAYERS;

/// Une position encodée.
#[derive(Clone)]
pub struct Encoded {
    pub regions: usize,
    pub players: usize,
    /// La place (absolue) du joueur qui agit : pas une entrée du réseau, elle sert à traduire les
    /// clés des coups des réseaux de la version 1 (`net`).
    pub seat: usize,
    pub node: [[u8; NODE]; REGIONS],
    pub global: [u8; GLOBAL],
}

fn put(x: &mut u8, v: i32) {
    *x = v.clamp(0, 255) as u8;
}

/// Joueur relatif : 0 pour `me`, puis les suivants dans l'ordre du jeu.
pub fn relative(q: usize, me: usize, n: usize) -> usize {
    (q + n - me) % n
}

/// Encode la position du point de vue du joueur qui doit agir. `known_tail` : cartes connues au fond
/// de la pioche des peuples (voir `know::Knowledge`).
pub fn encode(g: &Game, known_tail: usize) -> Encoded {
    let s = &g.state;
    let n = s.player_count();
    let me = s.actor().unwrap_or(s.current);
    let rel = |q: usize| relative(q, me, n);
    let regions = g.map.len();
    assert!(regions <= REGIONS, "carte trop grande pour le réseau");
    let mut e = Encoded { regions, players: n, seat: me, node: [[0; NODE]; REGIONS], global: [0; GLOBAL] };

    let army = s.turn.army.filter(|_| matches!(s.phase, Phase::Conquest | Phase::Redeploy));
    let army_owned = army.map_or(Default::default(), |a| g.owned(a));
    let targets = if s.phase == Phase::Conquest { Some(g.targets()) } else { None };
    let abandon = s.phase == Phase::Conquest;

    for (i, def) in g.map.regions.iter().enumerate() {
        let x = &mut e.node[i];
        let r = &s.regions[i];
        x[N_TERRAIN + def.terrain.index()] = 1;
        x[N_CAVERN] = def.cavern as u8;
        x[N_MAGIC] = def.magic as u8;
        x[N_MINE] = def.mine as u8;
        x[N_ENTRY] = def.entry as u8;
        x[N_BORDER] = def.border as u8;
        x[N_COASTAL] = def.coastal as u8;
        put(&mut x[N_CENTRE], (def.centre * 10.0).round() as i32);
        put(&mut x[N_DEGREE], def.adj.len() as i32);
        match r.occupant {
            Occupant::Empty => x[N_EMPTY] = 1,
            Occupant::Lost => x[N_LOST] = 1,
            Occupant::Race(race) => {
                let rs = s.race(race);
                x[N_OWNER + rel(rs.owner)] = 1;
                x[if rs.declined { N_DECLINED } else { N_ACTIVE }] = 1;
                x[N_RACE + race.index()] = 1;
                x[N_POWER + rs.power.index()] = 1;
                x[N_MINE_OWN] = (rs.owner == me) as u8;
            }
        }
        put(&mut x[N_TOKENS], r.tokens);
        put(&mut x[N_CAMPS], r.camps);
        x[N_LAIR] = r.lair as u8;
        x[N_FORTRESS] = r.fortress as u8;
        x[N_HERO] = r.hero as u8;
        x[N_DRAGON] = r.dragon as u8;
        x[N_HOLE] = r.hole as u8;
        put(&mut x[N_DEFENSE], g.defense_value(i));
        if let Some(ts) = &targets {
            let t = &ts[i];
            if t.reachable() {
                x[N_NORMAL] = (t.kind == TargetKind::Normal) as u8;
                x[N_DIE] = (t.kind == TargetKind::Die) as u8;
                put(&mut x[N_COST], t.effective);
                put(&mut x[N_NEED], t.need);
                x[N_SORCERY] = t.sorcery.allowed() as u8;
                x[N_DRAGON_OK] = t.dragon.allowed() as u8;
            }
            if abandon {
                x[N_ABANDON] = g.can_abandon(i).is_ok() as u8;
            }
        }
        put(&mut x[N_READIED], s.turn.readied[i]);
        x[N_ARMY] = army_owned.contains(i) as u8;
        let near = def.adj.iter().filter(|&&a| army_owned.contains(a as usize)).count();
        put(&mut x[N_ARMY_NEAR], near as i32);
    }

    let x = &mut e.global;
    x[G_PLAYERS + n - 2] = 1;
    put(&mut x[G_ROUND], s.round);
    put(&mut x[G_MAX_ROUNDS], s.max_rounds);
    put(&mut x[G_LEFT], s.max_rounds - s.round);
    x[G_CURRENT + rel(s.current)] = 1;
    x[G_PHASE + s.phase.index()] = 1;
    let t = &s.turn;
    x[G_GHOULS] = (t.stage == Stage::Ghouls) as u8;
    put(&mut x[G_CONQUESTS], t.conquests);
    put(&mut x[G_NON_EMPTY], t.non_empty);
    x[G_DIE_ROLLED] = t.die_rolled as u8;
    x[G_BERSERK_ROLLED] = t.berserk_die.is_some() as u8;
    put(&mut x[G_BERSERK_DIE], t.berserk_die.unwrap_or(0));
    x[G_DRAGON_USED] = t.dragon_used as u8;
    x[G_PICKED] = t.picked as u8;
    x[G_DECLINED] = t.declined as u8;
    put(&mut x[G_AMAZONS], t.amazon_bonus);
    x[G_FORTRESS_BUILT] = t.fortress_region.is_some() as u8;
    let acting = s.acting_race();
    if let Some(race) = acting {
        let rs = s.race(race);
        put(&mut x[G_HAND], rs.hand);
        x[G_ARMY_RACE + race.index()] = 1;
        x[G_ARMY_POWER + rs.power.index()] = 1;
    }
    put(&mut x[G_DEFENSE_TASKS], s.defense.len() as i32);
    put(&mut x[G_RACE_DECK], s.race_deck.len() as i32);
    put(&mut x[G_POWER_DECK], s.power_deck.len() as i32);
    put(&mut x[G_KNOWN_TAIL], known_tail as i32);

    for q in 0..n {
        let b = G_PLAYER + rel(q) * P;
        let p = &s.players[q];
        x[b + P_PRESENT] = 1;
        put(&mut x[b + P_COINS], p.coins);
        put(&mut x[b + P_LAST_INCOME], p.last_income);
        put(&mut x[b + P_INCOME], g.income(q));
        if let Some(race) = p.active {
            let rs = s.race(race);
            x[b + P_RACE + race.index()] = 1;
            x[b + P_POWER + rs.power.index()] = 1;
            put(&mut x[b + P_HAND], rs.hand);
            put(&mut x[b + P_REGIONS], g.owned(race).len() as i32);
            put(&mut x[b + P_TOKENS], g.tokens_on_board(race));
        }
        let mut declined_regions = 0;
        for &race in p.declined.iter() {
            x[b + P_DECLINED_RACES + race.index()] = 1;
            x[b + P_DECLINED_POWERS + s.race(race).power.index()] = 1;
            declined_regions += g.owned(race).len() as i32;
        }
        put(&mut x[b + P_DECLINED_REGIONS], declined_regions);
        x[b + P_ATTACKED] = t.attacked.contains(q) as u8;
        x[b + P_SORCERY] = t.sorcery.contains(q) as u8;
        x[b + P_PEACE_FROM] = s.peace.iter().any(|z| z.diplomat == q && z.target == me) as u8;
        x[b + P_PEACE_TO] = s.peace.iter().any(|z| z.diplomat == me && z.target == q) as u8;
        x[b + P_PLAYED] = (q < s.current) as u8;
    }
    let coins = s.players[me].coins;
    for (m, c) in s.market.iter().enumerate() {
        let b = G_MARKET + m * M;
        x[b + M_PRESENT] = 1;
        x[b + M_RACE + c.race.index()] = 1;
        x[b + M_POWER + c.power.index()] = 1;
        put(&mut x[b + M_COINS], c.coins);
        put(&mut x[b + M_TOKENS], c.race.starting_tokens(c.power));
        x[b + M_AFFORDABLE] = (m as i32 <= coins) as u8;
    }

    for q in 0..n {
        if let Some(race) = s.players[q].active {
            put(&mut x[G_AGE + rel(q)], s.round - s.race(race).picked_round);
        }
    }
    for &race in s.race_deck.iter() {
        x[G_DECK_RACES + race.index()] = 1;
    }
    // la pioche se tire par le début ; les cartes connues sont à la fin
    let hidden = s.race_deck.len() - known_tail.min(s.race_deck.len());
    for (j, &race) in s.race_deck.iter().skip(hidden).take(NEXT_KNOWN).enumerate() {
        x[G_NEXT_RACES + j * Race::COUNT + race.index()] = 1;
    }
    for &power in s.power_deck.iter() {
        x[G_DECK_POWERS + power.index()] = 1;
    }
    for z in s.peace.iter() {
        x[G_PEACE + rel(z.diplomat) * MAX_PLAYERS + rel(z.target)] = 1;
    }
    e
}

// garde-fous : les tailles des blocs one-hot suivent le jeu
const _: () = assert!(Race::COUNT == 14 && Power::COUNT == 20 && MAX_PLAYERS == 5);
const _: () = assert!(N_POWER + 20 == N_TOKENS && P_POWER + 20 == P_HAND && M_POWER + 20 == M_COINS);
const _: () = assert!(GLOBAL_V1 == 700 && GLOBAL == 806);

#[cfg(test)]
mod tests {
    use smallmobile_kernel::collections::RegionSet;
    use smallmobile_kernel::{Level, NoJournal, mapgen};

    use super::*;
    use crate::know::{Knowledge, Returns};
    use crate::moves;
    use crate::rand::Rng;

    /// Les entrées de la version 2 disent ce que dit la partie : âge des peuples, pioches, prochains
    /// peuples connus, paix.
    #[test]
    fn version_2_inputs_follow_the_game() {
        let mut rng = Rng::new(3);
        let (mut known, mut peaces) = (0, 0);
        for game in 0..60u64 {
            let n = 2 + game as usize % 4;
            let generated = mapgen::generate(n, 500 + game).unwrap();
            let map = generated.to_map().unwrap();
            let lost: RegionSet =
                generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
            let levels = vec![Level::Hard; n];
            let mut g = Game::create(&map, lost, &levels, generated.rounds, game as u32, &mut NoJournal);
            let mut k = Knowledge::default();
            while g.state.phase != Phase::GameOver {
                let s = &g.state;
                let x = encode(&g, k.race_tail).global;
                let me = s.actor().unwrap();
                let rel = |q: usize| relative(q, me, n);
                for q in 0..n {
                    let age = s.players[q].active.map_or(0, |race| s.round - s.race(race).picked_round);
                    assert_eq!(x[G_AGE + rel(q)] as i32, age);
                }
                for race in Race::ALL {
                    assert_eq!(x[G_DECK_RACES + race.index()] == 1, s.race_deck.contains(race));
                }
                for power in Power::ALL {
                    assert_eq!(x[G_DECK_POWERS + power.index()] == 1, s.power_deck.contains(power));
                }
                let next: Vec<Race> =
                    s.race_deck.iter().skip(s.race_deck.len() - k.race_tail).take(3).copied().collect();
                for (j, row) in x[G_NEXT_RACES..G_DECK_POWERS].chunks(Race::COUNT).enumerate() {
                    let marked: Vec<usize> = (0..Race::COUNT).filter(|&i| row[i] == 1).collect();
                    assert_eq!(marked, next.get(j).map(|r| r.index()).into_iter().collect::<Vec<_>>());
                }
                known += !next.is_empty() as usize;
                for d in 0..n {
                    for t in 0..n {
                        let at = s.peace.iter().any(|z| z.diplomat == d && z.target == t);
                        assert_eq!(x[G_PEACE + rel(d) * MAX_PLAYERS + rel(t)] == 1, at);
                        peaces += at as usize;
                    }
                }
                let legal = moves::legal(&g);
                let m = legal[rng.below(legal.len())].mv;
                let mut ret = Returns::default();
                moves::play(&mut g, &m, &mut ret).unwrap();
                k.observe(ret.0, &g.state);
            }
        }
        assert!(known > 0 && peaces > 0, "{known} positions avec des peuples connus, {peaces} paix");
    }
}
