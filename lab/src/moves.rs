//! Les coups d'un joueur, tels que la recherche les voit.
//!
//! Le noyau accepte des actions très fines (un jeton de plus ici, un de moins là) : trop de façons
//! d'arriver au même résultat. Un coup regroupe donc les actions d'une même décision :
//! - choix du peuple, déclin, chaque conquête, dé, abandon, fin des conquêtes : une action chacun ;
//! - fin de tour : héros, forteresse et campements (placés selon la menace), placement complet des
//!   jetons parmi quelques plans, paix des Diplomates, déclin des Robustes, en un seul coup ;
//! - défense : un plan pour les jetons rendus, puis la fin de la défense.

use smallmobile_kernel::action::{Journal, Placement};
use smallmobile_kernel::collections::ArrayVec;
use smallmobile_kernel::deploy::Plan;
use smallmobile_kernel::ids::{MAX_REGIONS, Power, Race};
use smallmobile_kernel::rules::TargetKind;
use smallmobile_kernel::state::Stage;
use smallmobile_kernel::{Action, Game, Phase, Reason};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    /// Une seule action du noyau.
    Act(Action),
    /// Fin de tour complète.
    Finish(Finish),
    /// Jetons rendus replacés (ajouts par région), puis fin de la défense.
    Defend(Plan),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Finish {
    /// Héros, forteresse, campements, joués avant le placement.
    pub specials: Specials,
    pub placement: Option<Placement>,
    pub peace: Option<usize>,
    pub stout: bool,
}

/// Un héros, une forteresse ou un campement posé sur une région.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Special {
    #[default]
    Hero,
    Fortress,
    Camp,
}

pub type Specials = ArrayVec<(Special, u8), 8>;

fn special_action((what, region): (Special, u8)) -> Action {
    let region = region as usize;
    match what {
        Special::Hero => Action::Hero { region },
        Special::Fortress => Action::Fortress { region },
        Special::Camp => Action::Camp { region, delta: 1 },
    }
}

pub type Moves = Vec<Move>;

/// Les coups possibles du joueur qui doit agir (vide en fin de partie).
pub fn legal(g: &Game) -> Moves {
    let mut out = Vec::new();
    let s = &g.state;
    match s.phase {
        Phase::GameOver => {}
        Phase::Pick if s.market.is_empty() => out.push(Move::Act(Action::SkipPick)),
        Phase::Pick => {
            let coins = s.players[s.current].coins;
            for index in 0..s.market.len().min(coins.max(0) as usize + 1) {
                out.push(Move::Act(Action::Pick { index }));
            }
        }
        Phase::Conquest => conquest(g, &mut out),
        Phase::Redeploy => finishes(g, &mut out),
        Phase::Defend => defenses(g, &mut out),
    }
    out
}

fn conquest(g: &Game, out: &mut Moves) {
    let s = &g.state;
    let army = s.turn.army.expect("peuple en action");
    let rs = *s.race(army);
    if g.can_decline().is_ok() {
        out.push(Move::Act(Action::Decline));
    }
    let t = &s.turn;
    if t.conquests == 0 && !t.die_rolled && t.berserk_die.is_none() {
        for region in g.owned(army).iter() {
            if g.can_abandon(region).is_ok() {
                out.push(Move::Act(Action::Abandon { region }));
            }
        }
    }
    if rs.has(Power::Berserk) && t.berserk_die.is_none() && rs.hand >= 1 {
        out.push(Move::Act(Action::BerserkRoll));
    }
    for x in g.targets().iter().filter(|x| x.reachable()) {
        let region = x.region;
        if x.sorcery.allowed() {
            out.push(Move::Act(Action::Sorcery { region }));
        }
        if x.dragon.allowed() {
            out.push(Move::Act(Action::Dragon { region }));
        }
        if rs.hand >= 1 {
            match x.kind {
                TargetKind::Normal => out.push(Move::Act(Action::Conquer { region })),
                TargetKind::Die => out.push(Move::Act(Action::Reinforce { region })),
                TargetKind::None => {}
            }
        }
    }
    out.push(Move::Act(Action::EndConquest));
}

/// Héros, forteresse et campements, aux endroits les plus menacés (comme l'IA du jeu).
fn specials(g: &Game, army: Race) -> Specials {
    let mut out = ArrayVec::new();
    let mut w = *g;
    let s = &g.state;
    let rs = *s.race(army);
    if s.turn.stage != Stage::Active || rs.declined {
        return out;
    }
    let best = |w: &Game, ok: &dyn Fn(usize) -> bool| -> Option<usize> {
        let mut best = None;
        let mut best_score = f64::NEG_INFINITY;
        for r in w.owned(army).iter().filter(|&r| !w.state.regions[r].is_immune() && ok(r)) {
            let score = w.threat(army, r) / w.defense_value(r) as f64;
            if score > best_score {
                best_score = score;
                best = Some(r);
            }
        }
        best
    };
    let mut push = |w: &mut Game, what: Special, region: usize| {
        if w.apply(&special_action((what, region as u8)), &mut smallmobile_kernel::NoJournal).is_ok() {
            out.push((what, region as u8));
            true
        } else {
            false
        }
    };
    match rs.power {
        Power::Heroic => {
            while w.heroes_on_board(army) < 2 {
                let Some(region) = best(&w, &|_| true) else { break };
                if !push(&mut w, Special::Hero, region) {
                    break;
                }
            }
        }
        Power::Fortified if s.turn.fortress_region.is_none() && w.fortresses_on_board(army) < 6 => {
            if let Some(region) = best(&w, &|r| !w.state.regions[r].fortress) {
                push(&mut w, Special::Fortress, region);
            }
        }
        Power::Bivouacking => {
            while w.camps_on_board(army) < 5 {
                let Some(region) = best(&w, &|_| true) else { break };
                if !push(&mut w, Special::Camp, region) {
                    break;
                }
            }
        }
        _ => {}
    }
    out
}

fn finishes(g: &Game, out: &mut Moves) {
    let s = &g.state;
    let army = s.turn.army.expect("peuple en action");
    let specials = specials(g, army);
    let mut w = *g;
    for &x in specials.iter() {
        w.apply(&special_action(x), &mut smallmobile_kernel::NoJournal).expect("action spéciale déjà vérifiée");
    }
    let mut placements: ArrayVec<Option<Placement>, 6> = ArrayVec::new();
    if w.owned(army).is_empty() {
        placements.push(None);
    } else {
        for p in w.placement_candidates(army) {
            if !placements.contains(&Some(p)) {
                placements.push(Some(p));
            }
        }
    }
    let pid = s.current;
    let active = s.turn.stage == Stage::Active;
    let rs = *s.race(army);
    let mut peaces: ArrayVec<Option<usize>, 6> = ArrayVec::new();
    peaces.push(None);
    if active && rs.has(Power::Diplomat) {
        for q in (0..s.player_count()).filter(|&q| q != pid && !s.turn.attacked.contains(q)) {
            peaces.push(Some(q));
        }
    }
    let stouts: &[bool] = if active && rs.has(Power::Stout) { &[false, true] } else { &[false] };
    for &placement in placements.iter() {
        for &peace in peaces.iter() {
            for &stout in stouts {
                out.push(Move::Finish(Finish { specials, placement, peace, stout }));
            }
        }
    }
}

fn defenses(g: &Game, out: &mut Moves) {
    let race = g.state.defense[0].race;
    let pending = g.state.races[race.index()].is_some_and(|rs| rs.hand > 0) && !g.owned(race).is_empty();
    if !pending {
        out.push(Move::Defend([0; MAX_REGIONS]));
        return;
    }
    let a = g.plan_deployment(race, false);
    let b = g.plan_deployment(race, true);
    out.push(Move::Defend(a));
    if b != a {
        out.push(Move::Defend(b));
    }
}

/// Joue un coup ; un coup refusé laisse la partie dans un état quelconque (c'est un bogue de l'appelant).
pub fn play(g: &mut Game, m: &Move, journal: &mut impl Journal) -> Result<(), Reason> {
    match m {
        Move::Act(a) => g.apply(a, journal),
        Move::Finish(f) => {
            for &x in f.specials.iter() {
                g.apply(&special_action(x), journal)?;
            }
            if let Some(p) = f.placement {
                g.apply(&Action::Place(p), journal)?;
            }
            g.apply(&Action::EndTurn { peace: f.peace, stout: f.stout }, journal)
        }
        Move::Defend(plan) => {
            let race = g.state.defense[0].race;
            for r in g.owned(race).iter() {
                if plan[r] > 0 {
                    g.apply(&Action::Deploy { region: r, delta: plan[r] }, journal)?;
                }
            }
            g.apply(&Action::EndDefense, journal)
        }
    }
}
