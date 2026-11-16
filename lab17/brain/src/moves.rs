//! Les coups d'un joueur, tels que la recherche les voit : une décision à la fois.
//!
//! - choix du peuple, déclin, chaque conquête, dé, abandon, fin des conquêtes : une action du noyau
//!   chacun ;
//! - fin de tour (redéploiement), découpée en décisions, dans l'ordre :
//!   1. `Lift` (forcé) : chaque région du peuple garde un jeton, les autres vont en main ; héros et
//!      campements sont relevés ;
//!   2. les pouvoirs, un à la fois et facultatifs : un héros, la forteresse, un campement sur une
//!      région ; jouer le coup suivant (un jeton, ou la fin du tour) met fin à leur pose ;
//!   3. les jetons, un à la fois, chacun sur une région ;
//!   4. la fin du tour : la paix des Diplomates (aucune, ou un adversaire) ou le déclin des
//!      Robustes, sinon forcée ;
//! - défense : les jetons rendus, un à la fois, puis la fin de la défense (forcée).
//!
//! Le noyau ne garde pas où en est une fin de tour : `Redeploy` le suit, dans ce que sait le joueur
//! (`Knowledge`). Chaque coup porte une clé (`Key`) qui le désigne d'un monde possible à l'autre, et
//! une sortie du réseau. Une décision entière du jeu (un coup, ou toute une fin de tour ou défense)
//! se joue en actions groupées (`decision`).

use smallmobile_kernel::action::{Journal, Placement};
use smallmobile_kernel::ids::{Power, Race};
use smallmobile_kernel::rules::{Target, TargetKind};
use smallmobile_kernel::state::Stage;
use smallmobile_kernel::{Action, Game, Phase, Reason};

use crate::know::Knowledge;

/// Les actions sur une région, dans l'ordre des clés (`REGION`) et des sorties du réseau.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Conquer,
    Reinforce,
    Dragon,
    Sorcery,
    Abandon,
    Hero,
    Fortress,
    Camp,
    /// Un jeton posé, au redéploiement ou à la défense.
    Token,
    // --- extensions (lab17), après celles de lab13 : les sorties de c130 ne bougent pas ---
    /// Mercenaires : la conquête payée 1 pièce, ou le paiement en attente après un dé raté.
    Hire,
    /// Catapultés : le tir sur la région, depuis la première région du peuple qui l'atteint.
    Catapult,
    /// Farfadets : un Chaudron posé.
    Cauldron,
}

pub const KINDS: [Kind; 12] = [
    Kind::Conquer,
    Kind::Reinforce,
    Kind::Dragon,
    Kind::Sorcery,
    Kind::Abandon,
    Kind::Hero,
    Kind::Fortress,
    Kind::Camp,
    Kind::Token,
    Kind::Hire,
    Kind::Catapult,
    Kind::Cauldron,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Move {
    /// Le peuple de cette place du marché.
    Pick(u8),
    /// Marché vide : passer.
    SkipPick,
    Decline,
    EndConquest,
    /// Le dé des Berserks.
    Berserk,
    /// Intrépides : reprendre ses jetons en main et continuer.
    Regroup,
    Region(Kind, u8),
    /// Début de la fin de tour : un jeton par région, le reste en main ; héros et campements relevés.
    Lift,
    /// Fin du tour : paix imposée à ce joueur (réel), déclin des Robustes.
    End {
        peace: Option<u8>,
        stout: bool,
    },
    EndDefense,
}

/// Où en est la fin de tour du joueur qui agit, que le noyau ne garde pas. Remis à zéro dès que
/// la partie quitte le redéploiement (`Knowledge::observe`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Redeploy {
    /// Jetons, héros et campements relevés (`Lift`).
    pub lifted: bool,
    /// La pose des pouvoirs est finie (un jeton a été posé).
    pub powers_done: bool,
}

/// Désigne un coup d'un monde possible à l'autre ; moins de `KEYS`.
pub type Key = u16;
/// Clés qui ne désignent pas une région.
pub const GLOBAL_KEYS: usize = 32;
pub const KEYS: usize = GLOBAL_KEYS + KINDS.len() * 64;

const PICK: Key = 0;
const SKIP_PICK: Key = 6;
const DECLINE: Key = 7;
const END_CONQUEST: Key = 8;
const BERSERK: Key = 9;
/// Plus la cible de la paix relativement au joueur qui finit son tour (0 : aucune, 1 : le suivant),
/// comme le réseau voit les joueurs.
pub const PEACE: Key = 10;
/// Plus 1 pour le déclin des Robustes.
pub const STOUT: Key = 15;
const REGROUP: Key = 17;
/// Les coups forcés (`Lift`, fin de tour ou de défense sans choix) : jamais appris.
const FORCED: Key = 31;
/// Plus sorte × 64 + région.
const REGION: Key = GLOBAL_KEYS as Key;

/// Un coup possible et sa clé.
#[derive(Clone, Copy, Debug)]
pub struct Choice {
    pub key: Key,
    pub mv: Move,
}

fn region(kind: Kind, r: usize) -> Choice {
    Choice { key: REGION + kind as Key * 64 + r as Key, mv: Move::Region(kind, r as u8) }
}

pub type Moves = Vec<Choice>;

/// Les coups possibles du joueur qui doit agir (vide en fin de partie).
pub fn legal(g: &Game, k: &Knowledge) -> Moves {
    legal_with(g, k, None)
}

/// `legal`, avec les cibles des conquêtes (`Game::targets`) si elles sont déjà calculées.
pub fn legal_with(g: &Game, k: &Knowledge, targets: Option<&[Target]>) -> Moves {
    let mut out = Vec::new();
    let s = &g.state;
    match s.phase {
        Phase::GameOver => {}
        Phase::Pick if s.market.is_empty() => out.push(Choice { key: SKIP_PICK, mv: Move::SkipPick }),
        Phase::Pick => {
            let coins = s.players[s.current].coins;
            // 1 pièce par combinaison au-dessus, 3 pour une Maudite (`Game::pick_price`)
            for index in (0..s.market.len()).take_while(|&i| g.pick_price(i) <= coins) {
                out.push(Choice { key: PICK + index as Key, mv: Move::Pick(index as u8) });
            }
        }
        Phase::Conquest => conquest(g, targets, &mut out),
        Phase::Redeploy => redeploy(g, k.redeploy, &mut out),
        Phase::Defend => {
            let race = s.defense[0].race;
            let own = g.owned(race);
            if s.races[race.index()].is_some_and(|rs| rs.hand > 0) && !own.is_empty() {
                out.extend(own.iter().map(|r| region(Kind::Token, r)));
            } else {
                out.push(Choice { key: FORCED, mv: Move::EndDefense });
            }
        }
    }
    out
}

/// Ce que la descente d'une recherche doit savoir d'une position (`forced`) : son seul coup
/// possible, s'il n'y en a qu'un ; sinon ses coups, s'ils ont été calculés.
pub enum Forced {
    One(Move),
    Several(Option<Moves>),
}

/// `legal`, réduit à savoir si un seul coup est possible. En conquête, la fin des conquêtes l'est
/// toujours : un autre coup suffit (déclin, abandon, berserk, la première région qui est une
/// cible), sans calculer les cibles de toutes les régions, le plus cher de la descente d'une
/// recherche. Dans les autres phases, `legal`, bon marché.
pub fn forced(g: &Game, k: &Knowledge) -> Forced {
    let s = &g.state;
    if s.phase != Phase::Conquest {
        let legal = legal(g, k);
        return if legal.len() == 1 { Forced::One(legal[0].mv) } else { Forced::Several(Some(legal)) };
    }
    let army = s.turn.army.expect("peuple en action");
    let rs = *s.race(army);
    let t = &s.turn;
    if t.hire.is_some() {
        return Forced::Several(None);
    }
    let abandon = t.conquests == 0 && !t.die_rolled && t.berserk_die.is_none();
    let mercenary = rs.has(Power::Mercenary);
    let other = g.can_decline().is_ok()
        || g.can_regroup().is_ok()
        || (rs.has(Power::Catapult) && !g.catapult_options().is_empty())
        || (abandon && g.owned(army).iter().any(|r| g.can_abandon(r).is_ok()))
        || (rs.has(Power::Berserk) && t.berserk_die.is_none() && rs.hand >= 1)
        || {
            // comme `conquest`, région par région
            let hold = g.foothold(army);
            (0..g.map.len()).any(|i| {
                let x = g.target(i, hold);
                x.reachable()
                    && (x.sorcery.allowed()
                        || x.dragon.allowed()
                        || (rs.hand >= 1 && x.kind != TargetKind::None)
                        || (mercenary && g.hire_target(&x).0.allowed()))
            })
        };
    if other { Forced::Several(None) } else { Forced::One(Move::EndConquest) }
}

fn conquest(g: &Game, targets: Option<&[Target]>, out: &mut Moves) {
    let s = &g.state;
    let army = s.turn.army.expect("peuple en action");
    let rs = *s.race(army);
    let t = &s.turn;
    // Mercenaires après un dé raté : payer, ou renoncer (la fin des conquêtes)
    if let Some(r) = t.hire {
        out.push(region(Kind::Hire, r as usize));
        out.push(Choice { key: END_CONQUEST, mv: Move::EndConquest });
        return;
    }
    if g.can_decline().is_ok() {
        out.push(Choice { key: DECLINE, mv: Move::Decline });
    }
    if t.conquests == 0 && !t.die_rolled && t.berserk_die.is_none() {
        for r in g.owned(army).iter() {
            if g.can_abandon(r).is_ok() {
                out.push(region(Kind::Abandon, r));
            }
        }
    }
    if rs.has(Power::Berserk) && t.berserk_die.is_none() && rs.hand >= 1 {
        out.push(Choice { key: BERSERK, mv: Move::Berserk });
    }
    if g.can_regroup().is_ok() {
        out.push(Choice { key: REGROUP, mv: Move::Regroup });
    }
    if rs.has(Power::Catapult) && !t.catapult_used {
        let mut seen = smallmobile_kernel::collections::RegionSet::EMPTY;
        for (_, r, _) in g.catapult_options() {
            if !seen.contains(r) {
                seen.insert(r);
                out.push(region(Kind::Catapult, r));
            }
        }
    }
    let mercenary = rs.has(Power::Mercenary);
    let computed;
    let targets = match targets {
        Some(t) => t,
        None => {
            computed = g.targets();
            &computed[..]
        }
    };
    for x in targets.iter().filter(|x| x.reachable()) {
        if x.sorcery.allowed() {
            out.push(region(Kind::Sorcery, x.region));
        }
        if x.dragon.allowed() {
            out.push(region(Kind::Dragon, x.region));
        }
        if rs.hand >= 1 {
            match x.kind {
                TargetKind::Normal => out.push(region(Kind::Conquer, x.region)),
                TargetKind::Die => out.push(region(Kind::Reinforce, x.region)),
                TargetKind::None => {}
            }
        }
        if mercenary && g.hire_target(x).0.allowed() {
            out.push(region(Kind::Hire, x.region));
        }
    }
    out.push(Choice { key: END_CONQUEST, mv: Move::EndConquest });
}

/// Le pouvoir à poser pendant la fin de tour du peuple actif, s'il en a un.
fn power_kind(rs: &smallmobile_kernel::state::RaceState) -> Option<Kind> {
    [(Power::Heroic, Kind::Hero), (Power::Fortified, Kind::Fortress), (Power::Bivouacking, Kind::Camp)]
        .into_iter()
        .find(|&(p, _)| rs.has(p))
        .map(|(_, kind)| kind)
}

fn redeploy(g: &Game, progress: Redeploy, out: &mut Moves) {
    let s = &g.state;
    let army = s.turn.army.expect("peuple en action");
    let rs = *s.race(army);
    let own = g.owned(army);
    if own.is_empty() {
        return finish(g, out);
    }
    // Barbares et Fées actifs ne redéploient pas : ni relève ni jetons, seulement leurs pouvoirs
    let frozen = s.turn.stage == Stage::Active && !rs.declined && matches!(army, Race::Barbarians | Race::Pixies);
    if !progress.lifted && !frozen {
        out.push(Choice { key: FORCED, mv: Move::Lift });
        return;
    }
    if !progress.powers_done && s.turn.stage == Stage::Active {
        match power_kind(&rs) {
            Some(Kind::Hero) if g.heroes_on_board(army) < 2 => {
                out.extend(own.iter().filter(|&r| !s.regions[r].hero).map(|r| region(Kind::Hero, r)));
            }
            Some(Kind::Fortress) if s.turn.fortress_region.is_none() && g.fortresses_on_board(army) < 6 => {
                out.extend(own.iter().filter(|&r| !s.regions[r].fortress).map(|r| region(Kind::Fortress, r)));
            }
            Some(Kind::Camp) if g.camps_on_board(army) < 5 => out.extend(own.iter().map(|r| region(Kind::Camp, r))),
            _ => {}
        }
        // Farfadets : un Chaudron sur une région, s'il reste un tour pour l'encaisser
        if army == Race::Leprechauns && !rs.declined && rs.cauldrons > 0 && s.round < s.max_rounds {
            out.extend(own.iter().filter(|&r| !s.regions[r].cauldron()).map(|r| region(Kind::Cauldron, r)));
        }
    }
    if rs.hand > 0 && !frozen {
        out.extend(own.iter().map(|r| region(Kind::Token, r)));
    } else {
        finish(g, out);
    }
}

/// Les fins de tour possibles : paix des Diplomates, déclin des Robustes, sinon la seule.
fn finish(g: &Game, out: &mut Moves) {
    let s = &g.state;
    let army = s.turn.army.expect("peuple en action");
    let rs = *s.race(army);
    let (pid, n) = (s.current, s.player_count());
    let active = s.turn.stage == Stage::Active;
    if active && rs.has(Power::Diplomat) {
        out.push(Choice { key: PEACE, mv: Move::End { peace: None, stout: false } });
        for q in (0..n).filter(|&q| q != pid && !s.turn.attacked.contains(q)) {
            let target = ((q + n - pid) % n) as Key;
            out.push(Choice { key: PEACE + target, mv: Move::End { peace: Some(q as u8), stout: false } });
        }
    } else if active && rs.has(Power::Stout) && s.turn.abandoned.is_empty() {
        for stout in [false, true] {
            out.push(Choice { key: STOUT + Key::from(stout), mv: Move::End { peace: None, stout } });
        }
    } else {
        out.push(Choice { key: FORCED, mv: Move::End { peace: None, stout: false } });
    }
}

/// Les actions du noyau qui jouent ce coup, dans l'ordre.
pub fn actions(g: &Game, m: &Move) -> Vec<Action> {
    match single(g, m) {
        Some(a) => vec![a],
        None => lift(g),
    }
}

/// `actions`, à la suite de `out` (sans allocation pour les coups d'une seule action).
pub fn actions_into(g: &Game, m: &Move, out: &mut Vec<Action>) {
    match single(g, m) {
        Some(a) => out.push(a),
        None => out.extend(lift(g)),
    }
}

/// L'action d'un coup qui n'en demande qu'une (tous sauf `Move::Lift`).
fn single(g: &Game, m: &Move) -> Option<Action> {
    Some(match *m {
        Move::Pick(index) => Action::Pick { index: index as usize },
        Move::SkipPick => Action::SkipPick,
        Move::Decline => Action::Decline,
        Move::EndConquest => Action::EndConquest,
        Move::Berserk => Action::BerserkRoll,
        Move::Regroup => Action::Regroup,
        Move::Region(kind, r) => {
            let region = r as usize;
            match kind {
                Kind::Conquer => Action::Conquer { region },
                Kind::Reinforce => Action::Reinforce { region },
                Kind::Dragon => Action::Dragon { region },
                Kind::Sorcery => Action::Sorcery { region },
                Kind::Abandon => Action::Abandon { region },
                Kind::Hero => Action::Hero { region },
                Kind::Fortress => Action::Fortress { region },
                Kind::Camp => Action::Camp { region, delta: 1 },
                Kind::Token => Action::Deploy { region, delta: 1 },
                Kind::Hire => Action::Hire { region },
                Kind::Catapult => {
                    let army = g.state.turn.army.expect("peuple en action");
                    // une région de tir valable, s'il y en a une (sinon le noyau refusera)
                    let from = g.catapult_from(army, region).unwrap_or(region);
                    Action::Catapult { from, region }
                }
                Kind::Cauldron => Action::Cauldron { region },
            }
        }
        Move::Lift => return None,
        Move::End { peace, stout } => Action::EndTurn { peace: peace.map(usize::from), stout },
        Move::EndDefense => Action::EndDefense,
    })
}

/// Les actions de `Move::Lift`.
fn lift(g: &Game) -> Vec<Action> {
    let s = &g.state;
    let army = s.turn.army.expect("peuple en action");
    let own = g.owned(army);
    // un jeton par région, deux pour des Kobolds (`Game::floor`)
    let mut out = vec![Action::Place(Placement([g.floor(army); smallmobile_kernel::ids::MAX_REGIONS]))];
    for r in own.iter() {
        if s.regions[r].hero {
            out.push(Action::Hero { region: r });
        }
        out.extend((0..s.regions[r].camps).map(|_| Action::Camp { region: r, delta: -1 }));
    }
    out
}

/// Joue un coup ; un coup refusé laisse la partie dans un état quelconque (c'est un bogue de
/// l'appelant). Le suivi de la fin de tour est l'affaire de `Knowledge::play`.
pub fn play(g: &mut Game, m: &Move, journal: &mut impl Journal) -> Result<(), Reason> {
    // sans `Vec` pour les coups d'une seule action (presque tous, à chaque simulation)
    match single(g, m) {
        Some(a) => g.apply(&a, journal)?,
        None => {
            for a in lift(g) {
                g.apply(&a, journal)?;
            }
        }
    }
    Ok(())
}

/// Le résultat public du hasard d'un coup qui vient d'être joué (`g` : la partie après le coup) :
/// le marché après un choix de peuple (les cartes qui y entrent : une d'habitude, plusieurs quand la
/// pioche des peuples, vidée, a reçu des cartes rendues), la réussite d'une conquête au dé, la valeur
/// du dé des Berserks ; 0 pour un coup sans hasard. Deux mondes possibles qui diffèrent sur ce
/// résultat donnent deux valeurs différentes ; le reste de la partie n'en dépend pas.
pub fn chance(g: &Game, m: &Move) -> u64 {
    let s = &g.state;
    let base = match m {
        Move::Pick(_) => return market(g),
        // Mercenaires : un dé raté peut laisser le paiement en attente
        Move::Region(Kind::Reinforce, _) => {
            1 + s.last_die.and_then(|d| d.success).unwrap_or(false) as u64 + 2 * s.turn.hire.is_some() as u64
        }
        Move::Berserk => 1 + s.turn.berserk_die.unwrap_or(0) as u64,
        _ => 0,
    };
    // Pygmées chassés d'une région : un dé dit combien reviennent en main
    let takes = matches!(
        m,
        Move::Region(Kind::Conquer | Kind::Reinforce | Kind::Dragon | Kind::Sorcery | Kind::Hire | Kind::Catapult, _)
    );
    match s.races[Race::Pygmies.index()] {
        Some(p) if takes => base | (1 + p.hand.max(0) as u64) << 8,
        _ => base,
    }
}

/// Le marché en empreinte : le peuple et le pouvoir de chaque carte, dans l'ordre (`chance` d'un
/// choix de peuple ; la recherche avec `market_late` le range aux choix de peuple suivants).
pub fn market(g: &Game) -> u64 {
    g.state.market.iter().fold(1, |h: u64, c| {
        let (race, power) = (c.race.index(), c.power.index());
        // le jeu de base garde l'empreinte de lab13 ; les extensions viennent après
        let id = if race < Race::BASE_COUNT && power < Power::BASE_COUNT {
            race * Power::BASE_COUNT + power
        } else {
            Race::BASE_COUNT * Power::BASE_COUNT + race * Power::COUNT + power
        };
        (h ^ (id as u64 | (c.tokens.max(0) as u64) << 16)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Une décision entière du joueur qui agit, en actions du noyau : un coup, ou toute une fin de tour
/// ou une défense, décidée coup par coup par `choose`, puis groupée (`compact`). `choose` reçoit la
/// partie et ce que sait le joueur, et rend un coup possible ; `k` est ce que sait le joueur au
/// début de la décision. Rend aussi les coups joués.
pub fn decision(
    g: &Game,
    k: &Knowledge,
    mut choose: impl FnMut(&Game, &Knowledge) -> Move,
) -> Result<(Vec<Action>, Vec<Move>), Reason> {
    let (mut w, mut kw) = (*g, *k);
    let (mut out, mut played) = (Vec::new(), Vec::new());
    loop {
        let mv = choose(&w, &kw);
        out.extend(actions(&w, &mv));
        kw.play(&mut w, &mv)?;
        played.push(mv);
        if !goes_on(g, &w, &mv) {
            break;
        }
    }
    Ok((compact(g, &out), played))
}

/// La décision commencée en `g` (`decision`) continue-t-elle, une fois `mv` joué (`w`) ? Oui
/// pendant une fin de tour ou une défense, tant que le même joueur agit dans la même phase.
pub fn goes_on(g: &Game, w: &Game, mv: &Move) -> bool {
    let phase = g.state.phase;
    let unit = matches!(phase, Phase::Redeploy | Phase::Defend);
    let ended = matches!(mv, Move::End { .. } | Move::EndDefense);
    unit && !ended && w.state.phase == phase && w.state.actor() == g.state.actor()
}

/// Les actions d'une fin de tour ou d'une défense, groupées : les héros et campements relevés puis
/// reposés au même endroit ne bougent pas, les jetons posés un à un rejoignent le placement (ou,
/// en défense, un ajout par région). Jouées depuis `g`, elles mènent exactement au même état.
pub fn compact(g: &Game, actions: &[Action]) -> Vec<Action> {
    let s = &g.state;
    let regions = g.map.len();
    let mut heroes = vec![false; regions];
    let mut camps = vec![0i32; regions];
    let mut deploy = vec![0i32; regions];
    let mut place: Option<Placement> = None;
    let (mut fortress, mut rest) = (Vec::new(), Vec::new());
    for a in actions {
        match *a {
            Action::Hero { region } => heroes[region] = !heroes[region],
            Action::Camp { region, delta } => camps[region] += delta,
            Action::Fortress { .. } => fortress.push(*a),
            Action::Place(p) => place = Some(p),
            Action::Deploy { region, delta } => match place.as_mut() {
                Some(p) => p.0[region] += delta,
                None => deploy[region] += delta,
            },
            _ => rest.push(*a),
        }
    }
    let mut out = Vec::new();
    // héros enlevés avant d'en poser (au plus 2), campements de même (au plus 5)
    for take in [true, false] {
        for r in 0..regions {
            if heroes[r] && s.regions[r].hero == take {
                out.push(Action::Hero { region: r });
            }
            let delta = camps[r];
            if (delta < 0) == take && delta != 0 {
                out.extend((0..delta.abs()).map(|_| Action::Camp { region: r, delta: delta.signum() }));
            }
        }
    }
    out.extend(fortress);
    out.extend(place.map(Action::Place));
    out.extend((0..regions).filter(|&r| deploy[r] != 0).map(|r| Action::Deploy { region: r, delta: deploy[r] }));
    out.extend(rest);
    out
}

/// Le coup en clair, pour l'analyse (régions désignées par leur numéro dans la carte).
pub fn describe(g: &Game, m: &Move) -> String {
    let s = &g.state;
    let region = |r: u8| format!("région {r} ({:?})", g.map.regions[r as usize].terrain);
    match *m {
        Move::Pick(index) => {
            let c = s.market[index as usize];
            format!("prendre {:?} {:?} (place {index}, {} pièces dessus)", c.race, c.power, c.coins)
        }
        Move::SkipPick => "passer (marché vide)".into(),
        Move::Decline => "passer en déclin".into(),
        Move::EndConquest => "finir les conquêtes".into(),
        Move::Berserk => "lancer le dé des Berserks".into(),
        Move::Regroup => "reprendre ses jetons (Intrépides)".into(),
        Move::Region(kind, r) => match kind {
            Kind::Conquer => format!("conquérir la {}", region(r)),
            Kind::Reinforce => format!("tenter la {} au dé", region(r)),
            Kind::Dragon => format!("lâcher le dragon sur la {}", region(r)),
            Kind::Sorcery => format!("ensorceler la {}", region(r)),
            Kind::Abandon => format!("abandonner la {}", region(r)),
            Kind::Hero => format!("un héros sur la {}", region(r)),
            Kind::Fortress => format!("la forteresse sur la {}", region(r)),
            Kind::Camp => format!("un campement sur la {}", region(r)),
            Kind::Token => format!("un jeton sur la {}", region(r)),
            Kind::Hire => format!("payer 1 pièce pour la {}", region(r)),
            Kind::Catapult => format!("la Catapulte sur la {}", region(r)),
            Kind::Cauldron => format!("un Chaudron sur la {}", region(r)),
        },
        Move::Lift => "relever jetons, héros et campements".into(),
        Move::End { peace, stout } => {
            let mut out = String::from("finir le tour");
            if let Some(q) = peace {
                out += &format!(", paix imposée au joueur {}", q + 1);
            }
            if stout {
                out += ", déclin (Robustes)";
            }
            out
        }
        Move::EndDefense => "finir la défense".into(),
    }
}

const _: () = assert!(KEYS == 800 && GLOBAL_KEYS > FORCED as usize && STOUT + 1 < REGROUP && REGROUP < FORCED);
const _: () = assert!(Race::BASE_COUNT == 14);

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use smallmobile_kernel::ids::Expansions;
    use smallmobile_kernel::{Level, NoJournal, maps};

    use super::*;
    use crate::rand::Rng;

    /// Des parties au hasard avec les trois extensions, 2 à 5 joueurs, sur les cartes du jeu : chaque
    /// coup de `legal` est accepté par le noyau, `forced` s'accorde avec `legal`, une décision groupée
    /// (`compact`) mène au même état que ses coups un à un, et chaque coup des extensions est joué.
    #[test]
    fn random_games_with_every_expansion() {
        let mut rng = Rng::new(17);
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for game in 0..160u64 {
            let n = 2 + game as usize % 4;
            let generated = maps::get(n, game).unwrap();
            let map = generated.to_map().unwrap();
            let lost = generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
            let x = if game % 5 == 0 { Expansions::NONE } else { Expansions::ALL };
            let levels = vec![Level::Normal; n];
            let mut g = Game::create(&map, lost, &levels, generated.rounds, game as u32, x, &mut NoJournal);
            let mut k = Knowledge::default();
            let mut moves = 0;
            while g.state.phase != Phase::GameOver {
                moves += 1;
                assert!(moves < 20_000, "partie sans fin");
                let legal = legal(&g, &k);
                assert!(!legal.is_empty());
                match forced(&g, &k) {
                    Forced::One(m) => assert_eq!((legal.len(), legal[0].mv), (1, m), "{legal:?}"),
                    Forced::Several(_) => assert!(legal.len() > 1, "{:?}", legal[0].mv),
                }
                let mut keys: Vec<Key> = legal.iter().map(|c| c.key).collect();
                keys.sort();
                keys.dedup();
                assert_eq!(keys.len(), legal.len(), "deux coups de même clé");
                assert!(keys.iter().all(|&key| (key as usize) < KEYS));
                // une décision entière au hasard, puis rejouée groupée
                let before = (g, k);
                let (actions, played) = decision(&g, &k, |w, kw| {
                    let l = super::legal(w, kw);
                    // les coups des extensions, plus souvent qu'à leur tour
                    let special = l.iter().filter(|c| is_expansion(&c.mv)).collect::<Vec<_>>();
                    if !special.is_empty() && rng.below(2) == 0 {
                        special[rng.below(special.len())].mv
                    } else {
                        l[rng.below(l.len())].mv
                    }
                })
                .unwrap_or_else(|r| panic!("coup refusé : {r:?}"));
                for m in &played {
                    if let Some(name) = expansion_name(m) {
                        *seen.entry(name).or_default() += 1;
                    }
                    k.play(&mut g, m).unwrap_or_else(|r| panic!("{} refusé : {r:?}", describe(&g, m)));
                }
                let mut w = before.0;
                for a in &actions {
                    w.apply(a, &mut NoJournal).unwrap_or_else(|r| panic!("action groupée refusée : {a:?} {r:?}"));
                }
                assert_eq!(w.state, g.state, "décision groupée : {played:?}");
            }
        }
        for name in ["hire", "catapult", "cauldron", "regroup"] {
            assert!(seen.get(name).copied().unwrap_or(0) > 0, "{name} jamais joué : {seen:?}");
        }
    }

    fn is_expansion(m: &Move) -> bool {
        expansion_name(m).is_some()
    }

    fn expansion_name(m: &Move) -> Option<&'static str> {
        match m {
            Move::Region(Kind::Hire, _) => Some("hire"),
            Move::Region(Kind::Catapult, _) => Some("catapult"),
            Move::Region(Kind::Cauldron, _) => Some("cauldron"),
            Move::Regroup => Some("regroup"),
            _ => None,
        }
    }
}
