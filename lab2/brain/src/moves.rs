//! Les coups d'un joueur, tels que la recherche les voit.
//!
//! Le noyau accepte des actions très fines (un jeton de plus ici, un de moins là) : trop de façons
//! d'arriver au même résultat. Un coup regroupe donc les actions d'une même décision :
//! - choix du peuple, déclin, chaque conquête, dé, abandon, fin des conquêtes : une action chacun ;
//! - fin de tour : héros, forteresse et campements (placés selon la menace), placement complet des
//!   jetons parmi quelques plans, paix des Diplomates, déclin des Robustes, en un seul coup ;
//!   La fin de tour de l'IA Normal du jeu (placement appris) est proposée en plus ;
//! - défense : un plan pour les jetons rendus, puis la fin de la défense.
//!
//! Chaque coup porte une clé (`Key`) qui le désigne indépendamment des détails : « le 3e plan de
//! placement » plutôt que le nombre exact de jetons par région. La recherche s'en sert pour
//! reconnaître le même choix d'un monde possible à l'autre.

use smallmobile_kernel::action::{Journal, Placement};
use smallmobile_kernel::collections::ArrayVec;
use smallmobile_kernel::deploy::Plan;
use smallmobile_kernel::ids::{MAX_REGIONS, Power, Race};
use smallmobile_kernel::rules::TargetKind;
use smallmobile_kernel::state::Stage;
use smallmobile_kernel::{Action, Ai, Game, Level, Phase, Reason, Restrictions, Tuning};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    /// Une seule action du noyau.
    Act(Action),
    /// Fin de tour complète.
    Finish(Finish),
    /// Jetons rendus replacés (ajouts par région), puis fin de la défense.
    Defend(Plan),
    /// Fin de tour jouée par l'IA Normal du jeu.
    Auto,
}

/// Désigne un coup d'un monde possible à l'autre ; moins de `KEYS`.
pub type Key = u16;
pub const KEYS: usize = 448;

const PICK: Key = 0;
const SKIP_PICK: Key = 6;
const DECLINE: Key = 7;
const END_CONQUEST: Key = 8;
const BERSERK: Key = 9;
const AUTO: Key = 10;
/// Plus plan × 12 + paix × 2 + Robustes ; paix : 0 (aucune), ou la cible relativement au joueur
/// qui finit son tour (1 : le suivant), comme le réseau voit les joueurs.
pub const FINISH: Key = 16;
const DEFEND: Key = 96;
/// + sorte × 64 + région.
const REGION: Key = 128;
const CONQUER: Key = 0;
const REINFORCE: Key = 1;
const DRAGON: Key = 2;
const SORCERY: Key = 3;
const ABANDON: Key = 4;

fn region_key(kind: Key, region: usize) -> Key {
    REGION + kind * 64 + region as Key
}

/// Un coup possible et sa clé.
#[derive(Clone, Copy, Debug)]
pub struct Choice {
    pub key: Key,
    pub mv: Move,
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

pub type Moves = Vec<Choice>;

fn push(out: &mut Moves, key: Key, mv: Move) {
    out.push(Choice { key, mv });
}

/// Les coups possibles du joueur qui doit agir (vide en fin de partie).
pub fn legal(g: &Game) -> Moves {
    let mut out = Vec::new();
    let s = &g.state;
    match s.phase {
        Phase::GameOver => {}
        Phase::Pick if s.market.is_empty() => push(&mut out, SKIP_PICK, Move::Act(Action::SkipPick)),
        Phase::Pick => {
            let coins = s.players[s.current].coins;
            for index in 0..s.market.len().min(coins.max(0) as usize + 1) {
                push(&mut out, PICK + index as Key, Move::Act(Action::Pick { index }));
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
        push(out, DECLINE, Move::Act(Action::Decline));
    }
    let t = &s.turn;
    if t.conquests == 0 && !t.die_rolled && t.berserk_die.is_none() {
        for region in g.owned(army).iter() {
            if g.can_abandon(region).is_ok() {
                push(out, region_key(ABANDON, region), Move::Act(Action::Abandon { region }));
            }
        }
    }
    if rs.has(Power::Berserk) && t.berserk_die.is_none() && rs.hand >= 1 {
        push(out, BERSERK, Move::Act(Action::BerserkRoll));
    }
    for x in g.targets().iter().filter(|x| x.reachable()) {
        let region = x.region;
        if x.sorcery.allowed() {
            push(out, region_key(SORCERY, region), Move::Act(Action::Sorcery { region }));
        }
        if x.dragon.allowed() {
            push(out, region_key(DRAGON, region), Move::Act(Action::Dragon { region }));
        }
        if rs.hand >= 1 {
            match x.kind {
                TargetKind::Normal => push(out, region_key(CONQUER, region), Move::Act(Action::Conquer { region })),
                TargetKind::Die => push(out, region_key(REINFORCE, region), Move::Act(Action::Reinforce { region })),
                TargetKind::None => {}
            }
        }
    }
    push(out, END_CONQUEST, Move::Act(Action::EndConquest));
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
    // (numéro du plan, plan) ; un plan identique à un précédent n'est pas repris
    let mut placements: ArrayVec<(Key, Option<Placement>), 6> = ArrayVec::new();
    if w.owned(army).is_empty() {
        placements.push((0, None));
    } else {
        for (i, p) in w.placement_candidates(army).into_iter().enumerate() {
            if !placements.iter().any(|&(_, q)| q == Some(p)) {
                placements.push((i as Key, Some(p)));
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
    for &(plan, placement) in placements.iter() {
        for &peace in peaces.iter() {
            for &stout in stouts {
                let target = peace.map_or(0, |q| (q + s.player_count() - pid) % s.player_count());
                let key = FINISH + plan * 12 + target as Key * 2 + Key::from(stout);
                push(out, key, Move::Finish(Finish { specials, placement, peace, stout }));
            }
        }
    }
    push(out, AUTO, Move::Auto);
}

fn defenses(g: &Game, out: &mut Moves) {
    let race = g.state.defense[0].race;
    let pending = g.state.races[race.index()].is_some_and(|rs| rs.hand > 0) && !g.owned(race).is_empty();
    if !pending {
        push(out, DEFEND, Move::Defend([0; MAX_REGIONS]));
        return;
    }
    let a = g.plan_deployment(race, false);
    let b = g.plan_deployment(race, true);
    push(out, DEFEND, Move::Defend(a));
    if b != a {
        push(out, DEFEND + 1, Move::Defend(b));
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
        Move::Auto => {
            let tuning = Tuning::default();
            let ai = Ai::new(&tuning);
            let pid = g.state.current;
            let level = g.state.players[pid].level;
            g.state.players[pid].level = Level::Normal;
            let mut result = Ok(());
            while result.is_ok() && g.state.phase == Phase::Redeploy {
                result = ai.decide(g, Restrictions::default()).and_then(|a| g.apply(&a, journal));
            }
            g.state.players[pid].level = level;
            result
        }
    }
}

/// Les actions du noyau qui jouent ce coup, dans l'ordre (le jeu les joue une à une). Mêmes actions
/// que `play`, calculées sur une copie de la partie.
pub fn actions(g: &Game, m: &Move) -> Result<Vec<Action>, Reason> {
    let mut out = Vec::new();
    match m {
        Move::Act(a) => out.push(*a),
        Move::Finish(f) => {
            out.extend(f.specials.iter().map(|&x| special_action(x)));
            if let Some(p) = f.placement {
                out.push(Action::Place(p));
            }
            out.push(Action::EndTurn { peace: f.peace, stout: f.stout });
        }
        Move::Defend(plan) => {
            let race = g.state.defense[0].race;
            for r in g.owned(race).iter() {
                if plan[r] > 0 {
                    out.push(Action::Deploy { region: r, delta: plan[r] });
                }
            }
            out.push(Action::EndDefense);
        }
        Move::Auto => {
            let tuning = Tuning::default();
            let ai = Ai::new(&tuning);
            let mut w = *g;
            w.state.players[w.state.current].level = Level::Normal;
            while w.state.phase == Phase::Redeploy {
                let a = ai.decide(&w, Restrictions::default())?;
                w.apply(&a, &mut smallmobile_kernel::NoJournal)?;
                out.push(a);
            }
        }
    }
    Ok(out)
}

/// Le coup en clair, pour l'analyse (régions désignées par leur numéro dans la carte).
pub fn describe(g: &Game, m: &Move) -> String {
    let s = &g.state;
    let region = |r: usize| format!("région {r} ({:?})", g.map.regions[r].terrain);
    match m {
        Move::Act(a) => match *a {
            Action::Pick { index } => {
                let c = s.market[index];
                format!("prendre {:?} {:?} (place {index}, {} pièces dessus)", c.race, c.power, c.coins)
            }
            Action::SkipPick => "passer (marché vide)".into(),
            Action::Conquer { region: r } => format!("conquérir la {}", region(r)),
            Action::Reinforce { region: r } => format!("tenter la {} au dé", region(r)),
            Action::BerserkRoll => "lancer le dé des Berserks".into(),
            Action::Dragon { region: r } => format!("lâcher le dragon sur la {}", region(r)),
            Action::Sorcery { region: r } => format!("ensorceler la {}", region(r)),
            Action::Abandon { region: r } => format!("abandonner la {}", region(r)),
            Action::Decline => "passer en déclin".into(),
            Action::EndConquest => "finir les conquêtes".into(),
            other => format!("{other:?}"),
        },
        Move::Finish(f) => {
            let mut out = String::from("fin de tour");
            if let Some(p) = f.placement {
                let army = s.turn.army.expect("peuple en action");
                let tokens: Vec<String> = g.owned(army).iter().map(|r| format!("{r}:{}", p.0[r])).collect();
                out += &format!(", jetons {}", tokens.join(" "));
            }
            for (what, r) in f.specials.iter() {
                out += &format!(", {what:?} en {r}");
            }
            if let Some(q) = f.peace {
                out += &format!(", paix imposée au joueur {}", q + 1);
            }
            if f.stout {
                out += ", déclin (Robustes)";
            }
            out
        }
        Move::Defend(plan) => {
            let added: Vec<String> =
                plan.iter().enumerate().filter(|(_, n)| **n > 0).map(|(r, n)| format!("+{n} en {r}")).collect();
            format!("défense : {}", if added.is_empty() { "rien".into() } else { added.join(", ") })
        }
        Move::Auto => "fin de tour à la manière de l'IA Normal".into(),
    }
}
