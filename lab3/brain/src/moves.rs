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
use smallmobile_kernel::rules::TargetKind;
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
}

pub const KINDS: [Kind; 9] = [
    Kind::Conquer,
    Kind::Reinforce,
    Kind::Dragon,
    Kind::Sorcery,
    Kind::Abandon,
    Kind::Hero,
    Kind::Fortress,
    Kind::Camp,
    Kind::Token,
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
    let mut out = Vec::new();
    let s = &g.state;
    match s.phase {
        Phase::GameOver => {}
        Phase::Pick if s.market.is_empty() => out.push(Choice { key: SKIP_PICK, mv: Move::SkipPick }),
        Phase::Pick => {
            let coins = s.players[s.current].coins;
            for index in 0..s.market.len().min(coins.max(0) as usize + 1) {
                out.push(Choice { key: PICK + index as Key, mv: Move::Pick(index as u8) });
            }
        }
        Phase::Conquest => conquest(g, &mut out),
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

fn conquest(g: &Game, out: &mut Moves) {
    let s = &g.state;
    let army = s.turn.army.expect("peuple en action");
    let rs = *s.race(army);
    if g.can_decline().is_ok() {
        out.push(Choice { key: DECLINE, mv: Move::Decline });
    }
    let t = &s.turn;
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
    for x in g.targets().iter().filter(|x| x.reachable()) {
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
    if !progress.lifted {
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
    }
    if rs.hand > 0 {
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
    } else if active && rs.has(Power::Stout) {
        for stout in [false, true] {
            out.push(Choice { key: STOUT + Key::from(stout), mv: Move::End { peace: None, stout } });
        }
    } else {
        out.push(Choice { key: FORCED, mv: Move::End { peace: None, stout: false } });
    }
}

/// Les actions du noyau qui jouent ce coup, dans l'ordre.
pub fn actions(g: &Game, m: &Move) -> Vec<Action> {
    match single(m) {
        Some(a) => vec![a],
        None => lift(g),
    }
}

/// L'action d'un coup qui n'en demande qu'une (tous sauf `Move::Lift`).
fn single(m: &Move) -> Option<Action> {
    Some(match *m {
        Move::Pick(index) => Action::Pick { index: index as usize },
        Move::SkipPick => Action::SkipPick,
        Move::Decline => Action::Decline,
        Move::EndConquest => Action::EndConquest,
        Move::Berserk => Action::BerserkRoll,
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
    let mut out = vec![Action::Place(Placement::default())];
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
    match single(m) {
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
    match m {
        Move::Pick(_) => s.market.iter().fold(1, |h: u64, c| {
            (h ^ (c.race.index() * Power::COUNT + c.power.index()) as u64).wrapping_mul(0x0100_0000_01b3)
        }),
        Move::Region(Kind::Reinforce, _) => 1 + s.last_die.and_then(|d| d.success).unwrap_or(false) as u64,
        Move::Berserk => 1 + s.turn.berserk_die.unwrap_or(0) as u64,
        _ => 0,
    }
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
    let (actor, phase) = (g.state.actor(), g.state.phase);
    let (mut w, mut kw) = (*g, *k);
    let (mut out, mut played) = (Vec::new(), Vec::new());
    loop {
        let mv = choose(&w, &kw);
        out.extend(actions(&w, &mv));
        kw.play(&mut w, &mv)?;
        played.push(mv);
        let unit = matches!(phase, Phase::Redeploy | Phase::Defend);
        let ended = matches!(mv, Move::End { .. } | Move::EndDefense);
        if !unit || ended || w.state.phase != phase || w.state.actor() != actor {
            break;
        }
    }
    Ok((compact(g, &out), played))
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

const _: () = assert!(KEYS == 608 && GLOBAL_KEYS > FORCED as usize && STOUT + 1 < FORCED);
const _: () = assert!(Race::COUNT == 14);
