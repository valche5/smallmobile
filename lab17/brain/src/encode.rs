//! Ce que le réseau voit d'une position : uniquement ce qu'un joueur sait, du point de vue du joueur
//! qui doit agir (« moi » = joueur relatif 0, puis les suivants dans l'ordre du jeu).
//!
//! Toutes les entrées sont de petits entiers (0 à 255) : comptes, présences, numéros. Le réseau les
//! met à l'échelle lui-même (échelles choisies dans `lab17/py/model.py`, livrées avec les poids). Ainsi
//! les positions s'enregistrent sur un octet par entrée.
//!
//! - une ligne par région (`NODE` entrées) : la carte, l'occupant, les jetons, ce que le peuple qui
//!   conquiert peut y faire ;
//! - une ligne pour la partie (`GLOBAL` entrées) : manche, phase, tour en cours, chaque joueur
//!   (pièces, peuples, revenus), le marché ; puis l'histoire de chaque joueur depuis le début de la
//!   partie (`know::History`) : de quoi deviner son style.

use smallmobile_kernel::collections::RegionSet;
use smallmobile_kernel::ids::{Expansion, MAX_PLAYERS, Power, Race};
use smallmobile_kernel::rules::{Target, TargetKind};
use smallmobile_kernel::state::{Occupant, Stage};
use smallmobile_kernel::{Game, Phase};

use crate::know::{Knowledge, habit};

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
// --- extensions (lab17), après les entrées de lab13 pour que c130 s'élargisse sans rien changer ---
/// Peuples et pouvoirs des extensions (`Race::BASE_COUNT..`, `Power::BASE_COUNT..`).
const N_RACE_X: usize = 77; // 10
const N_POWER_X: usize = 87; // 12
const N_CATAPULT: usize = 99;
const N_CAULDRON: usize = 100;
const N_LADIES: usize = 101;
/// Mercenaires : conquête payée possible, et son coût en jetons.
const N_HIRE: usize = 102;
const N_HIRE_COST: usize = 103;
/// Catapulte : tir possible sur la région, et son coût.
const N_CATAPULT_OK: usize = 104;
const N_CATAPULT_COST: usize = 105;
/// Mercenaires : la région du paiement en attente (après un dé raté).
const N_HIRE_PENDING: usize = 106;
/// Gitanes : région abandonnée ce tour.
const N_ABANDONED: usize = 107;
pub const NODE: usize = 108;

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
// --- l'histoire de chaque joueur (`know::History`), d'après ses coups publics ----------------------
/// Les habitudes de chaque joueur relatif (`know::habit`).
const G_HABITS: usize = G_MARKET + 6 * M; // 5 × habit::COUNT
/// Régions prises par chaque joueur relatif à chaque joueur relatif (`attaquant × 5 + victime`) :
/// depuis le début de la partie, puis pendant le dernier tour de chacun.
const G_ATTACKS: usize = G_HABITS + MAX_PLAYERS * habit::COUNT; // 25
const G_RECENT: usize = G_ATTACKS + MAX_PLAYERS * MAX_PLAYERS; // 25
const G_BASE: usize = G_RECENT + MAX_PLAYERS * MAX_PLAYERS;
// --- extensions (lab17), après les entrées de lab13 ---
const G_ARMY_RACE_X: usize = G_BASE; // 10
const G_ARMY_POWER_X: usize = G_ARMY_RACE_X + 10; // 12
/// Les extensions de la partie (`Expansion`).
const G_EXPANSIONS: usize = G_ARMY_POWER_X + 12; // 3
const G_CATAPULT_USED: usize = G_EXPANSIONS + 3;
const G_REGROUPED: usize = G_CATAPULT_USED + 1;
const G_HIRE_PENDING: usize = G_REGROUPED + 1;
const G_TOWER_CHOSEN: usize = G_HIRE_PENDING + 1;
const G_PLAYER_X: usize = G_TOWER_CHOSEN + 1; // 5 × PX
const PX_RACE: usize = 0; // 10
const PX_POWER: usize = 10; // 12
const PX_DECLINED_RACES: usize = 22; // 10
const PX_DECLINED_POWERS: usize = 32; // 12
/// Jetons hors du plateau du peuple actif (Barbares, Fées).
const PX_ASIDE: usize = 44;
/// Chaudrons des Farfadets pas encore posés.
const PX_CAULDRONS: usize = 45;
const PX: usize = 46;
const G_MARKET_X: usize = G_PLAYER_X + MAX_PLAYERS * PX; // 6 × MX
const MX_RACE: usize = 0; // 10
const MX_POWER: usize = 10; // 12
/// Homoncules ajoutés sur la combinaison.
const MX_HOMUNCULI: usize = 22;
const MX: usize = 23;
/// Complété à un multiple de 4.
pub const GLOBAL: usize = (G_MARKET_X + 6 * MX).next_multiple_of(4);
/// Pour la tête du peuple (`net::Pick`) : la phase, le marché et la taille d'une carte du marché.
pub const PHASE_AT: usize = G_PHASE;
pub const MARKET_AT: usize = G_MARKET;
pub const CARD: usize = M;

/// Une position encodée.
#[derive(Clone)]
pub struct Encoded {
    pub regions: usize,
    pub players: usize,
    pub node: [[u8; NODE]; REGIONS],
    pub global: [u8; GLOBAL],
}

fn put(x: &mut u8, v: i32) {
    *x = v.clamp(0, 255) as u8;
}

/// La case d'un peuple : dans le bloc de lab13 (`base`) pour le jeu de base, dans celui des extensions
/// (`ext`) sinon.
fn race_at(race: Race, base: usize, ext: usize) -> usize {
    let i = race.index();
    if i < Race::BASE_COUNT { base + i } else { ext + i - Race::BASE_COUNT }
}

fn power_at(power: Power, base: usize, ext: usize) -> usize {
    let i = power.index();
    if i < Power::BASE_COUNT { base + i } else { ext + i - Power::BASE_COUNT }
}

/// Joueur relatif : 0 pour `me`, puis les suivants dans l'ordre du jeu.
pub fn relative(q: usize, me: usize, n: usize) -> usize {
    (q + n - me) % n
}

/// Encode la position du point de vue du joueur qui doit agir, avec ce qu'il sait (`Knowledge` :
/// cartes connues au fond de la pioche des peuples, histoire des joueurs).
pub fn encode(g: &Game, k: &Knowledge) -> Encoded {
    encode_with(g, k, None)
}

/// `encode`, avec les cibles des conquêtes (`Game::targets`) si elles sont déjà calculées.
pub fn encode_with(g: &Game, k: &Knowledge, targets: Option<&[Target]>) -> Encoded {
    let known_tail = k.race_tail;
    let s = &g.state;
    let n = s.player_count();
    let me = s.actor().unwrap_or(s.current);
    let rel = |q: usize| relative(q, me, n);
    let regions = g.map.len();
    assert!(regions <= REGIONS, "carte trop grande pour le réseau");
    let mut e = Encoded { regions, players: n, node: [[0; NODE]; REGIONS], global: [0; GLOBAL] };

    // les régions et les jetons de chaque peuple, en un passage
    let (mut owned, mut on_board) = ([RegionSet::EMPTY; Race::COUNT], [0i32; Race::COUNT]);
    for (i, r) in s.regions[..regions].iter().enumerate() {
        if let Occupant::Race(race) = r.occupant {
            owned[race.index()].insert(i);
            on_board[race.index()] += r.tokens;
        }
    }
    let army = s.turn.army.filter(|_| matches!(s.phase, Phase::Conquest | Phase::Redeploy));
    let army_owned = army.map_or(RegionSet::EMPTY, |a| owned[a.index()]);
    let computed;
    let targets = match (s.phase == Phase::Conquest, targets) {
        (false, _) => None,
        (true, Some(t)) => Some(t),
        (true, None) => {
            computed = g.targets();
            Some(&computed[..])
        }
    };
    let abandon = s.phase == Phase::Conquest;
    // les tirs de Catapulte possibles : le moins cher par région visée
    let mut catapult = [0i32; REGIONS];
    if s.phase == Phase::Conquest && s.expansions.contains(Expansion::BeNotAfraid) {
        for (_, region, cost) in g.catapult_options() {
            let c = &mut catapult[region];
            *c = if *c == 0 { cost + 1 } else { (*c).min(cost + 1) };
        }
    }

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
                x[race_at(race, N_RACE, N_RACE_X)] = 1;
                x[power_at(rs.power, N_POWER, N_POWER_X)] = 1;
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
            if t.reachable() && s.turn.hire.is_none() {
                let (hire, cost) = g.hire_target(t);
                if hire.allowed() {
                    x[N_HIRE] = 1;
                    put(&mut x[N_HIRE_COST], cost);
                }
            }
        }
        x[N_CATAPULT] = r.catapult() as u8;
        x[N_CAULDRON] = r.cauldron() as u8;
        x[N_LADIES] = r.ladies() as u8;
        if catapult[i] > 0 {
            x[N_CATAPULT_OK] = 1;
            put(&mut x[N_CATAPULT_COST], catapult[i] - 1);
        }
        x[N_HIRE_PENDING] = (s.turn.hire == Some(i as u8)) as u8;
        x[N_ABANDONED] = s.turn.abandoned.contains(i) as u8;
        put(&mut x[N_READIED], s.turn.readied[i]);
        x[N_ARMY] = army_owned.contains(i) as u8;
        let near = RegionSet(g.map.hoods[i].adj.0 & army_owned.0).len();
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
        x[race_at(race, G_ARMY_RACE, G_ARMY_RACE_X)] = 1;
        x[power_at(rs.power, G_ARMY_POWER, G_ARMY_POWER_X)] = 1;
    }
    for e in s.expansions.iter() {
        x[G_EXPANSIONS + e.index()] = 1;
    }
    x[G_CATAPULT_USED] = t.catapult_used as u8;
    x[G_REGROUPED] = t.regrouped as u8;
    x[G_HIRE_PENDING] = t.hire.is_some() as u8;
    x[G_TOWER_CHOSEN] = t.tower.is_some() as u8;
    put(&mut x[G_DEFENSE_TASKS], s.defense.len() as i32);
    put(&mut x[G_RACE_DECK], s.race_deck.len() as i32);
    put(&mut x[G_POWER_DECK], s.power_deck.len() as i32);
    put(&mut x[G_KNOWN_TAIL], known_tail as i32);

    for q in 0..n {
        let b = G_PLAYER + rel(q) * P;
        let bx = G_PLAYER_X + rel(q) * PX;
        let p = &s.players[q];
        x[b + P_PRESENT] = 1;
        put(&mut x[b + P_COINS], p.coins);
        put(&mut x[b + P_LAST_INCOME], p.last_income);
        put(&mut x[b + P_INCOME], g.income(q));
        if let Some(race) = p.active {
            let rs = s.race(race);
            x[race_at(race, b + P_RACE, bx + PX_RACE)] = 1;
            x[power_at(rs.power, b + P_POWER, bx + PX_POWER)] = 1;
            put(&mut x[bx + PX_ASIDE], i32::from(rs.aside));
            put(&mut x[bx + PX_CAULDRONS], i32::from(rs.cauldrons));
            put(&mut x[b + P_HAND], rs.hand);
            put(&mut x[b + P_REGIONS], owned[race.index()].len() as i32);
            put(&mut x[b + P_TOKENS], on_board[race.index()]);
        }
        let mut declined_regions = 0;
        for &race in p.declined.iter() {
            x[race_at(race, b + P_DECLINED_RACES, bx + PX_DECLINED_RACES)] = 1;
            x[power_at(s.race(race).power, b + P_DECLINED_POWERS, bx + PX_DECLINED_POWERS)] = 1;
            declined_regions += owned[race.index()].len() as i32;
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
        let bx = G_MARKET_X + m * MX;
        x[b + M_PRESENT] = 1;
        x[race_at(c.race, b + M_RACE, bx + MX_RACE)] = 1;
        x[power_at(c.power, b + M_POWER, bx + MX_POWER)] = 1;
        put(&mut x[b + M_COINS], c.coins);
        put(&mut x[b + M_TOKENS], c.starting_tokens());
        put(&mut x[bx + MX_HOMUNCULI], c.tokens);
        x[b + M_AFFORDABLE] = (g.pick_price(m) <= coins) as u8;
    }

    let h = &k.history;
    for q in 0..n {
        for (i, &v) in h.habits[q].iter().enumerate() {
            put(&mut x[G_HABITS + rel(q) * habit::COUNT + i], v as i32);
        }
        for t in 0..n {
            let at = rel(q) * MAX_PLAYERS + rel(t);
            put(&mut x[G_ATTACKS + at], h.attacks[q][t] as i32);
            x[G_RECENT + at] = h.recent[q][t];
        }
    }
    e
}

// garde-fous : les tailles des blocs one-hot suivent le jeu
const _: () = assert!(Race::BASE_COUNT == 14 && Power::BASE_COUNT == 20 && MAX_PLAYERS == 5);
const _: () = assert!(N_POWER + 20 == N_TOKENS && P_POWER + 20 == P_HAND && M_POWER + 20 == M_COINS);
const _: () = assert!(G_HABITS == 700 && G_BASE == 820 && GLOBAL == 1220);
const _: () = assert!(Race::COUNT - Race::BASE_COUNT == 10 && Power::COUNT - Power::BASE_COUNT == 12);
const _: () = assert!(N_ABANDONED + 1 == NODE && NODE <= 112);

#[cfg(test)]
mod tests {
    use smallmobile_kernel::collections::RegionSet;
    use smallmobile_kernel::{Level, NoJournal, mapgen};

    use super::*;
    use crate::know::Knowledge;
    use crate::moves;
    use crate::rand::Rng;

    /// Les entrées de l'histoire des joueurs, comptée d'après le journal des
    /// événements de la partie ; et la même histoire se refait action par action (comme en rejouant
    /// une partie enregistrée) que coup par coup (comme la recherche).
    #[test]
    fn history_inputs_tell_the_history() {
        use smallmobile_kernel::action::{Event, Journal, Logged};
        /// Le journal d'un coup, gardé aussi en entier.
        struct Both<'a>(&'a mut crate::know::Returns, &'a mut Vec<Logged>);
        impl Journal for Both<'_> {
            fn record(&mut self, entry: Logged) {
                self.0.record(entry);
                self.1.push(entry);
            }
        }
        let mut rng = Rng::new(5);
        let (mut attacks, mut picks) = (0, 0);
        for game in 0..40u64 {
            let n = 2 + game as usize % 4;
            let generated = mapgen::generate(n, 900 + game).unwrap();
            let map = generated.to_map().unwrap();
            let lost: RegionSet =
                generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
            let levels = vec![Level::Normal; n];
            let mut g = Game::create(
                &map,
                lost,
                &levels,
                generated.rounds,
                game as u32,
                smallmobile_kernel::Expansions::NONE,
                &mut NoJournal,
            );
            let (mut k, mut by_action, mut by_move) =
                (Knowledge::default(), Knowledge::default(), Knowledge::default());
            let mut events: Vec<Logged> = Vec::new();
            while g.state.phase != Phase::GameOver {
                let legal = moves::legal(&g, &by_move);
                let m = legal[rng.below(legal.len())].mv;
                // coup par coup, en suivant la fin de tour
                let mut played = g;
                by_move.play(&mut played, &m).unwrap();
                // action par action, avec un journal neuf pour chacune
                let mut w = g;
                for a in moves::actions(&g, &m) {
                    let mut ret = by_action.journal(&w.state);
                    w.apply(&a, &mut ret).unwrap();
                    by_action.observe(ret, &w.state);
                }
                let mut ret = k.journal(&g.state);
                moves::play(&mut g, &m, &mut Both(&mut ret, &mut events)).unwrap();
                k.observe(ret, &g.state);
                assert_eq!(g.state, w.state);
                assert_eq!(g.state, played.state);
                assert_eq!(k.history, by_action.history, "même histoire coup par coup et action par action");
                assert_eq!(k.history, by_move.history);
            }
            let h = &k.history;
            for q in 0..n {
                let taken = |p: Option<usize>| {
                    events
                        .iter()
                        .filter(|e| matches!(e.event, Event::Conquered { player, victim, .. } if player == q && victim == p))
                        .count()
                };
                for t in 0..n {
                    assert_eq!(h.attacks[q][t] as usize, taken(Some(t)));
                    attacks += taken(Some(t));
                }
                let x = &h.habits[q];
                assert_eq!((x[habit::EMPTY] + x[habit::LOST_TRIBES]) as usize, taken(None));
                let others: usize = (0..n).filter(|&t| t != q).map(|t| taken(Some(t))).sum();
                assert_eq!((x[habit::ACTIVE] + x[habit::DECLINED]) as usize, others);
                assert!(x[habit::LEADER] as usize <= others);
                let count = |f: &dyn Fn(&Event) -> bool| events.iter().filter(|e| f(&e.event)).count();
                assert_eq!(
                    x[habit::PICKS] as usize,
                    count(&|e| matches!(e, Event::Picked { player, .. } if *player == q))
                );
                assert_eq!(
                    x[habit::DECLINES] as usize,
                    count(&|e| matches!(e, Event::Declined { player, .. } if *player == q))
                );
                picks += x[habit::PICKS] as usize;
            }
            // vu par chaque joueur : ses comptes en tête, les autres dans l'ordre du jeu
            for me in 0..n {
                let mut seen = g;
                seen.state.current = me;
                let x = encode(&seen, &k).global;
                for q in 0..n {
                    let r = relative(q, me, n);
                    for i in 0..habit::COUNT {
                        assert_eq!(x[G_HABITS + r * habit::COUNT + i] as u16, h.habits[q][i].min(255));
                    }
                    for t in 0..n {
                        let at = r * MAX_PLAYERS + relative(t, me, n);
                        assert_eq!(x[G_ATTACKS + at] as u16, h.attacks[q][t].min(255));
                        assert_eq!(x[G_RECENT + at], h.recent[q][t]);
                    }
                }
            }
        }
        assert!(attacks > 0 && picks > 0, "{attacks} attaques, {picks} peuples choisis");
    }
}
