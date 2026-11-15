//! Les règles des peuples et pouvoirs des extensions, une à une, sur une petite carte en ligne.

use smallmobile_kernel::action::{Action, Event, IncomeKind, Logged, NoJournal, Payment, Placement};
use smallmobile_kernel::collections::RegionSet;
use smallmobile_kernel::map::{Map, RegionDef};
use smallmobile_kernel::rng;
use smallmobile_kernel::rules::{Special, Target, TargetKind};
use smallmobile_kernel::state::{Combo, Immunity, Occupant, RaceState, RegionState, Turn};
use smallmobile_kernel::{Expansions, Game, Level, Phase, Power, Race, Reason, Terrain};

use Power::*;
use Race::*;

/// Une ligne 0 – 1 – … – n−1 de champs (la région 0 au bord) ; `terrains` : les exceptions.
fn line(n: usize, terrains: &[(usize, Terrain)]) -> Map {
    let regions = (0..n)
        .map(|i| {
            let adj = [i.checked_sub(1), (i + 1 < n).then_some(i + 1)].into_iter().flatten().map(|a| a as u8).collect();
            let terrain = terrains.iter().find(|t| t.0 == i).map_or(Terrain::Farmland, |t| t.1);
            RegionDef {
                terrain,
                cavern: false,
                magic: false,
                mine: false,
                entry: i == 0,
                border: i == 0,
                coastal: false,
                adj,
                centre: 0.5,
            }
        })
        .collect();
    Map::new(1, regions).unwrap()
}

/// Une partie vide (marché et pioches vides), au début de la première manche.
fn game(map: &Map, players: usize) -> Game<'_> {
    let levels = vec![Level::Normal; players];
    let mut g = Game::create(map, RegionSet::EMPTY, &levels, 6, 1, Expansions::ALL, &mut NoJournal);
    let s = &mut g.state;
    s.market.clear();
    s.race_deck.clear();
    s.power_deck.clear();
    s.race_tail = 0;
    s.power_tail = 0;
    s.turn = Turn::default();
    g
}

/// Donne un peuple actif à `owner`, avec ses régions et sa main.
fn give(g: &mut Game, race: Race, power: Power, owner: usize, regions: &[(usize, i32)], hand: i32) {
    let s = &mut g.state;
    s.races[race.index()] = Some(RaceState {
        power,
        owner,
        declined: false,
        hand,
        holes_placed: 0,
        picked_round: 1,
        aside: 0,
        cauldrons: if race == Leprechauns { 16 } else { 0 },
    });
    s.players[owner].active = Some(race);
    for &(r, tokens) in regions {
        s.regions[r] = RegionState { occupant: Occupant::Race(race), tokens, ..RegionState::default() };
    }
}

/// Passe le peuple en déclin sans jouer (ses régions gardent leurs jetons).
fn set_declined(g: &mut Game, race: Race) {
    let owner = g.state.race(race).owner;
    g.state.race_mut(race).declined = true;
    let p = &mut g.state.players[owner];
    p.active = None;
    p.declined.push(race);
}

fn conquest(g: &mut Game, pid: usize, race: Race) {
    g.state.current = pid;
    g.state.phase = Phase::Conquest;
    g.state.turn = Turn { army: Some(race), ..Turn::default() };
}

fn redeploy(g: &mut Game, pid: usize, race: Race) {
    conquest(g, pid, race);
    g.state.phase = Phase::Redeploy;
}

/// Joue l'action ; renvoie ses événements.
fn play(g: &mut Game, a: Action) -> Vec<Event> {
    let mut log: Vec<Logged> = Vec::new();
    g.apply(&a, &mut log).unwrap_or_else(|r| panic!("{a:?} refusée : {r:?}"));
    log.into_iter().map(|l| l.event).collect()
}

fn target(g: &Game, region: usize) -> Target {
    g.target(region, g.foothold(g.state.turn.army.unwrap()))
}

fn paid(events: &[Event]) -> Vec<(Payment, Option<usize>, Option<usize>, i32)> {
    events
        .iter()
        .filter_map(|e| match *e {
            Event::Paid { kind, from, to, coins } => Some((kind, from, to, coins)),
            _ => None,
        })
        .collect()
}

fn income(g: &Game, pid: usize, kind: IncomeKind) -> Option<i32> {
    g.income_lines(pid).iter().find(|l| l.kind == kind).map(|l| l.value)
}

/// Un état du générateur dont le premier dé de renfort vaut `value`.
fn rng_for(value: i32) -> u32 {
    (0..).find(|&seed| rng::roll_die(&mut { seed }) == value).unwrap()
}

/// Placement : `tokens` par région listée, 1 ailleurs.
fn placement(tokens: &[(usize, i32)]) -> Placement {
    let mut p = Placement::default();
    for &(r, n) in tokens {
        p.0[r] = n;
    }
    p
}

// --- peuples ---------------------------------------------------------------------------------

#[test]
fn barbarians_never_redeploy_and_take_back_their_tokens_next_turn() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Barbarians, Alchemist, 0, &[(0, 3), (1, 2)], 4);
    conquest(&mut g, 0, Barbarians);
    play(&mut g, Action::EndConquest);
    let rs = *g.state.race(Barbarians);
    assert_eq!((rs.hand, rs.aside, g.supply(Barbarians)), (0, 4, 14 - 5 - 4));
    for a in [Action::Deploy { region: 0, delta: -1 }, Action::AutoDeploy { all: true }, Action::Place(placement(&[]))]
    {
        assert_eq!(g.apply(&a, &mut NoJournal), Err(Reason::NoRedeploy));
    }
    play(&mut g, Action::EndTurn { peace: None, stout: false });
    play(&mut g, Action::SkipPick);
    // leur tour : tout ce qui dépasse un par région, plus les jetons mis de côté
    assert_eq!((g.state.current, g.state.phase), (0, Phase::Conquest));
    let rs = *g.state.race(Barbarians);
    assert_eq!((rs.hand, rs.aside), (4 + 2 + 1, 0));
}

#[test]
fn barbarians_replace_the_tokens_returned_by_an_attack() {
    let map = line(5, &[]);
    let mut g = game(&map, 3);
    give(&mut g, Barbarians, Alchemist, 0, &[(0, 3), (1, 4)], 0);
    g.state.race_mut(Barbarians).aside = 2;
    give(&mut g, Humans, Alchemist, 1, &[(2, 1)], 8);
    conquest(&mut g, 1, Humans);
    play(&mut g, Action::Conquer { region: 1 });
    play(&mut g, Action::EndConquest);
    play(&mut g, Action::AutoDeploy { all: false });
    play(&mut g, Action::EndTurn { peace: None, stout: false });
    // les jetons rendus se replacent (phase de défense), ceux mis de côté restent de côté
    assert_eq!(g.state.phase, Phase::Defend);
    assert_eq!(g.state.race(Barbarians).hand, 3);
    play(&mut g, Action::Deploy { region: 0, delta: 3 });
    play(&mut g, Action::EndDefense);
    assert_eq!(g.state.regions[0].tokens, 6);
    assert_eq!(g.state.race(Barbarians).aside, 2);
}

#[test]
fn pixies_keep_one_per_region_between_turns() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Pixies, Stout, 0, &[(0, 3), (1, 1)], 2);
    redeploy(&mut g, 0, Pixies);
    conquest(&mut g, 0, Pixies);
    play(&mut g, Action::EndConquest);
    assert_eq!((g.state.regions[0].tokens, g.state.regions[1].tokens), (1, 1));
    let rs = *g.state.race(Pixies);
    assert_eq!((rs.hand, rs.aside), (0, 4));
    assert_eq!(g.apply(&Action::Deploy { region: 0, delta: 1 }, &mut NoJournal), Err(Reason::NoRedeploy));
    // en déclin (Vaillantes) : une par région, le reste disparaît
    play(&mut g, Action::EndTurn { peace: None, stout: true });
    let rs = *g.state.race(Pixies);
    assert!(rs.declined);
    assert_eq!((rs.hand, rs.aside, g.tokens_on_board(Pixies)), (0, 0, 2));
}

#[test]
fn kobolds_never_hold_a_region_with_fewer_than_two() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Kobolds, Commando, 0, &[(0, 2)], 3);
    conquest(&mut g, 0, Kobolds);
    // champ vide : 2 − 1 (Commandos), mais jamais moins de 2
    assert_eq!(target(&g, 1).cost, 2);
    play(&mut g, Action::Conquer { region: 1 });
    // un seul jeton en main : ni conquête, ni dé
    let t = target(&g, 2);
    assert_eq!((t.kind, t.cost), (TargetKind::None, 2));
    play(&mut g, Action::EndConquest);
    assert_eq!(g.apply(&Action::Deploy { region: 1, delta: -1 }, &mut NoJournal), Err(Reason::KeepTwoTokens));
    let one = placement(&[(0, 4), (1, 1)]);
    assert_eq!(g.apply(&Action::Place(one), &mut NoJournal), Err(Reason::KeepTwoTokens));
    // 0 : la région est abandonnée
    let events = play(&mut g, Action::Place(placement(&[(0, 5), (1, 0)])));
    assert!(events.contains(&Event::Abandoned { player: 0, region: 1 }));
    assert_eq!(g.owned(Kobolds), [0].into_iter().collect());
    assert_eq!((g.state.regions[0].tokens, g.state.race(Kobolds).hand), (5, 0));
    play(&mut g, Action::EndTurn { peace: None, stout: false });
    play(&mut g, Action::SkipPick);
    // leur tour : 2 restent dans la région
    assert_eq!((g.state.regions[0].tokens, g.state.race(Kobolds).hand), (2, 3));
    play(&mut g, Action::Decline);
    assert_eq!(g.state.regions[0].tokens, 1);
    assert_eq!(g.floor(Kobolds), 1);
}

#[test]
fn kobolds_need_two_tokens_for_the_dragon_and_the_die() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Kobolds, DragonMaster, 0, &[(0, 2)], 1);
    give(&mut g, Elves, Alchemist, 1, &[(1, 4)], 0);
    conquest(&mut g, 0, Kobolds);
    let t = target(&g, 1);
    assert_eq!(t.dragon, Special::Denied(Reason::TokensNeeded { needed: 2 }));
    assert_eq!(t.kind, TargetKind::None);
    g.state.race_mut(Kobolds).hand = 2;
    play(&mut g, Action::Dragon { region: 1 });
    assert_eq!((g.state.regions[1].tokens, g.state.race(Kobolds).hand), (2, 0));
}

#[test]
fn goblins_pay_one_less_against_declined_races() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Goblins, Alchemist, 0, &[(1, 2)], 6);
    give(&mut g, Elves, Alchemist, 1, &[(0, 1)], 0);
    set_declined(&mut g, Elves);
    give(&mut g, Humans, Alchemist, 1, &[(2, 1)], 0);
    conquest(&mut g, 0, Goblins);
    assert_eq!(target(&g, 0).cost, 2);
    // peuple actif ou région vide : prix normal
    assert_eq!(target(&g, 2).cost, 3);
    g.state.regions[3] = RegionState::default();
    g.state.regions[3].occupant = Occupant::Race(Goblins);
    assert_eq!(target(&g, 4).cost, 2);
}

#[test]
fn pygmies_roll_for_reinforcements_when_they_lose_tokens() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, Alchemist, 0, &[(0, 1)], 10);
    give(&mut g, Pygmies, Alchemist, 1, &[(1, 3)], 0);
    conquest(&mut g, 0, Humans);
    g.state.rng = rng_for(3);
    let events = play(&mut g, Action::Conquer { region: 1 });
    assert!(events.contains(&Event::PygmiesReturned { player: 1, value: 3, tokens: 3 }));
    assert_eq!(g.state.race(Pygmies).hand, 2 + 3);
    // presque tous sur le plateau : le dé ne rend que ce qui reste dans la boîte (le jeton perdu)
    let mut g = game(&map, 2);
    give(&mut g, Humans, Alchemist, 0, &[(0, 1)], 10);
    give(&mut g, Pygmies, Alchemist, 1, &[(1, 1), (2, 5), (3, 5)], 0);
    conquest(&mut g, 0, Humans);
    g.state.rng = rng_for(3);
    let events = play(&mut g, Action::Conquer { region: 1 });
    assert!(events.contains(&Event::PygmiesReturned { player: 1, value: 3, tokens: 1 }));
    assert_eq!(g.supply(Pygmies), 0);
    // en déclin : pas de renforts
    let mut g = game(&map, 2);
    give(&mut g, Humans, Alchemist, 0, &[(0, 1)], 10);
    give(&mut g, Pygmies, Alchemist, 1, &[(1, 1)], 0);
    set_declined(&mut g, Pygmies);
    conquest(&mut g, 0, Humans);
    let events = play(&mut g, Action::Conquer { region: 1 });
    assert!(!events.iter().any(|e| matches!(e, Event::PygmiesReturned { .. })));
}

#[test]
fn homunculi_grow_while_others_pick_below_them() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    let combo = |race, power| Combo { race, power, coins: 0, tokens: 0 };
    for c in [combo(Homunculi, Alchemist), combo(Humans, Merchant), combo(Elves, Flying)] {
        g.state.market.push(c);
    }
    g.state.phase = Phase::Pick;
    play(&mut g, Action::Pick { index: 2 });
    assert_eq!(g.state.market.as_slice()[0], Combo { race: Homunculi, power: Alchemist, coins: 1, tokens: 1 });
    assert_eq!(g.state.market.as_slice()[1].tokens, 0);
    g.state.current = 1;
    g.state.phase = Phase::Pick;
    let events = play(&mut g, Action::Pick { index: 0 });
    assert!(matches!(events[0], Event::Picked { tokens: 10, gained: 1, .. }));
    // 15 au plus au départ
    let mut g = game(&map, 2);
    g.state.market.push(Combo { race: Homunculi, power: Hordes, coins: 0, tokens: 14 });
    g.state.phase = Phase::Pick;
    play(&mut g, Action::Pick { index: 0 });
    assert_eq!(g.state.race(Homunculi).hand, 15 + 2);
}

#[test]
fn leprechauns_cash_their_cauldrons_next_turn() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Leprechauns, Alchemist, 0, &[(0, 2), (1, 2)], 0);
    redeploy(&mut g, 0, Leprechauns);
    play(&mut g, Action::Cauldron { region: 0 });
    assert_eq!((g.state.regions[0].cauldron(), g.state.race(Leprechauns).cauldrons), (true, 15));
    play(&mut g, Action::Cauldron { region: 0 });
    assert_eq!((g.state.regions[0].cauldron(), g.state.race(Leprechauns).cauldrons), (false, 16));
    assert_eq!(g.apply(&Action::Cauldron { region: 2 }, &mut NoJournal), Err(Reason::NotYourRegion));
    play(&mut g, Action::Cauldron { region: 0 });
    play(&mut g, Action::Cauldron { region: 1 });
    // un Chaudron n'ajoute pas de défense
    assert_eq!(g.defense_value(1), 4);
    play(&mut g, Action::EndTurn { peace: None, stout: false });
    let events = play(&mut g, Action::SkipPick);
    assert_eq!(paid(&events), [(Payment::Cauldrons, None, Some(0), 2)]);
    assert!(!g.state.regions[0].cauldron() && !g.state.regions[1].cauldron());
    assert_eq!(g.state.race(Leprechauns).cauldrons, 14);
}

#[test]
fn a_cauldron_taken_pays_the_attacker_and_a_decline_loses_them() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Leprechauns, Alchemist, 0, &[(1, 2), (2, 2)], 0);
    g.state.regions[1].mark(RegionState::CAULDRON, true);
    g.state.regions[2].mark(RegionState::CAULDRON, true);
    give(&mut g, Humans, Alchemist, 1, &[(0, 1)], 8);
    conquest(&mut g, 1, Humans);
    let events = play(&mut g, Action::Conquer { region: 1 });
    assert_eq!(paid(&events), [(Payment::Cauldron, None, Some(1), 1)]);
    assert!(!g.state.regions[1].cauldron());
    // déclin : Chaudrons posés et restants perdus
    conquest(&mut g, 0, Leprechauns);
    let events = play(&mut g, Action::Decline);
    assert!(paid(&events).is_empty());
    assert!(!g.state.regions[2].cauldron());
    assert_eq!(g.state.race(Leprechauns).cauldrons, 0);
}

#[test]
fn declined_white_ladies_cannot_be_taken() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, DragonMaster, 0, &[(0, 3)], 6);
    give(&mut g, WhiteLadies, Alchemist, 1, &[(1, 2)], 0);
    conquest(&mut g, 0, Humans);
    // actives : comme tout le monde
    assert_eq!(target(&g, 1).block, None);
    conquest(&mut g, 1, WhiteLadies);
    play(&mut g, Action::Decline);
    assert!(g.state.regions[1].ladies());
    conquest(&mut g, 0, Humans);
    let t = target(&g, 1);
    assert_eq!(t.block, Some(Reason::Immune(Immunity::Ladies)));
    assert_eq!(t.dragon, Special::Denied(Reason::Immune(Immunity::Ladies)));
    assert!(g.attackers(1, 1).is_empty());
}

#[test]
fn gypsies_earn_their_abandoned_regions_and_stay_out() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Gypsies, Stout, 0, &[(0, 2), (1, 2)], 0);
    conquest(&mut g, 0, Gypsies);
    play(&mut g, Action::Abandon { region: 1 });
    assert_eq!(income(&g, 0, IncomeKind::Abandoned), Some(1));
    assert_eq!(target(&g, 1).block, Some(Reason::AbandonedThisTurn));
    assert_eq!(g.can_decline(), Err(Reason::DeclineAfterAbandon));
    assert_eq!(g.apply(&Action::Decline, &mut NoJournal), Err(Reason::DeclineAfterAbandon));
    play(&mut g, Action::EndConquest);
    play(&mut g, Action::AutoDeploy { all: false });
    let stout = Action::EndTurn { peace: None, stout: true };
    assert_eq!(g.apply(&stout, &mut NoJournal), Err(Reason::DeclineAfterAbandon));
    let events = play(&mut g, Action::EndTurn { peace: None, stout: false });
    let Some(Event::Scored { gain, .. }) = events.first() else { panic!("décompte") };
    assert_eq!(*gain, 1 + 1);
}

#[test]
fn priestesses_stack_their_ivory_tower() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Priestesses, Alchemist, 0, &[(0, 2), (1, 3), (2, 2)], 0);
    conquest(&mut g, 0, Priestesses);
    assert_eq!(g.apply(&Action::Tower { region: 3 }, &mut NoJournal), Err(Reason::NotYourRegion));
    play(&mut g, Action::Tower { region: 2 });
    let events = play(&mut g, Action::Decline);
    assert_eq!(g.owned(Priestesses), [2].into_iter().collect());
    assert_eq!(g.state.regions[2].tokens, 3);
    let Some(Event::Scored { lines, .. }) = events.iter().find(|e| matches!(e, Event::Scored { .. })) else {
        panic!("décompte")
    };
    assert!(lines.iter().any(|l| l.kind == IncomeKind::Tower && l.value == 3));
    assert!(!lines.iter().any(|l| l.kind == IncomeKind::Regions));
    // sans choix : une seule région garde les jetons
    let mut g = game(&map, 2);
    give(&mut g, Priestesses, Alchemist, 0, &[(0, 2), (1, 3), (2, 2)], 4);
    conquest(&mut g, 0, Priestesses);
    let preview = g.declined_preview();
    assert_eq!(preview.owned(Priestesses).len(), 1);
    assert_eq!(preview.tokens_on_board(Priestesses), 3);
    // après une conquête, plus de déclin, donc plus de Tour
    play(&mut g, Action::Conquer { region: 3 });
    assert_eq!(g.apply(&Action::Tower { region: 0 }, &mut NoJournal), Err(Reason::DeclineAfterConquest));
    give(&mut g, Humans, Alchemist, 1, &[(4, 2)], 0);
    conquest(&mut g, 1, Humans);
    assert_eq!(g.apply(&Action::Tower { region: 4 }, &mut NoJournal), Err(Reason::MissingPower));
}

// --- pouvoirs --------------------------------------------------------------------------------

#[test]
fn the_catapult_shoots_over_a_region_and_guards_its_own() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, Catapult, 0, &[(0, 3)], 4);
    give(&mut g, Elves, Alchemist, 1, &[(4, 1)], 6);
    conquest(&mut g, 0, Humans);
    assert_eq!(target(&g, 2).block, Some(Reason::NotAdjacent));
    assert_eq!(g.catapult_target(0, 2), Ok(1));
    assert_eq!(g.catapult_target(0, 3), Err(Reason::CatapultOutOfRange));
    // exactement deux régions plus loin : pas sur une voisine
    assert_eq!(g.catapult_target(0, 1), Err(Reason::CatapultOutOfRange));
    assert_eq!(g.catapult_options(), [(0, 2, 1)]);
    let events = play(&mut g, Action::Catapult { from: 0, region: 2 });
    assert!(matches!(events[0], Event::Conquered { region: 2, tokens: 1, .. }));
    assert!(g.state.regions[0].catapult());
    assert_eq!(g.catapult_target(2, 4), Err(Reason::CatapultUsed));
    // la région de tir est imprenable
    conquest(&mut g, 1, Elves);
    g.state.regions[1].occupant = Occupant::Race(Elves);
    g.state.regions[1].tokens = 1;
    assert_eq!(target(&g, 0).block, Some(Reason::Immune(Immunity::Catapult)));
    // le déclin la retire
    conquest(&mut g, 0, Humans);
    play(&mut g, Action::Decline);
    assert!(!g.state.regions[0].catapult());
}

#[test]
fn the_catapult_cannot_shoot_over_the_sea_but_can_over_a_lake() {
    for (terrain, ok) in [(Terrain::Sea, false), (Terrain::Lake, true)] {
        let map = line(4, &[(1, terrain)]);
        let mut g = game(&map, 2);
        give(&mut g, Humans, Catapult, 0, &[(0, 2)], 4);
        conquest(&mut g, 0, Humans);
        let shot = g.catapult_target(0, 2);
        assert_eq!(shot.is_ok(), ok, "{terrain:?} : {shot:?}");
        // abandonner la région de tir retire la Catapulte
        if ok {
            play(&mut g, Action::Catapult { from: 0, region: 2 });
            conquest(&mut g, 0, Humans);
            play(&mut g, Action::Abandon { region: 0 });
            assert!(!g.state.regions[0].catapult());
        }
    }
}

#[test]
fn corrupt_races_are_paid_for_their_regions() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, Alchemist, 0, &[(0, 1)], 10);
    give(&mut g, Elves, Corrupt, 1, &[(1, 2), (2, 2)], 0);
    conquest(&mut g, 0, Humans);
    let events = play(&mut g, Action::Conquer { region: 1 });
    assert_eq!(paid(&events), [(Payment::Bribe, Some(0), Some(1), 1)]);
    assert_eq!((g.state.players[0].coins, g.state.players[1].coins), (4, 6));
    // sans pièce, rien à payer
    g.state.players[0].coins = 0;
    let events = play(&mut g, Action::Conquer { region: 2 });
    assert!(paid(&events).is_empty());
}

#[test]
fn imperial_and_barricade_count_regions() {
    let map = line(7, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, Imperial, 0, &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], 0);
    give(&mut g, Elves, Barricade, 1, &[(5, 1), (6, 1)], 0);
    assert_eq!(income(&g, 0, IncomeKind::Imperial), Some(2));
    assert_eq!(income(&g, 1, IncomeKind::Barricade), Some(3));
    let mut g = game(&map, 2);
    give(&mut g, Humans, Barricade, 0, &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], 0);
    give(&mut g, Elves, Imperial, 1, &[(5, 1), (6, 1)], 0);
    assert_eq!(income(&g, 0, IncomeKind::Barricade), None);
    assert_eq!(income(&g, 1, IncomeKind::Imperial), None);
    // Retranchés sans aucune région : +3 quand même
    g.state.regions[0..5].fill(RegionState::default());
    assert_eq!(income(&g, 0, IncomeKind::Barricade), Some(3));
}

#[test]
fn mercenaries_pay_a_coin_for_two_tokens() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, Mercenary, 0, &[(0, 1)], 3);
    give(&mut g, Elves, Alchemist, 1, &[(1, 3)], 0);
    g.state.players[0].coins = 1;
    conquest(&mut g, 0, Humans);
    let t = target(&g, 1);
    assert_eq!((t.cost, t.kind, g.hire_target(&t)), (5, TargetKind::Die, (Special::Allowed, 3)));
    let events = play(&mut g, Action::Hire { region: 1 });
    assert_eq!(paid(&events), [(Payment::Hire, Some(0), None, 1)]);
    assert_eq!((g.state.regions[1].tokens, g.state.race(Humans).hand, g.state.players[0].coins), (3, 0, 0));
    // plus de pièce
    give(&mut g, Elves, Alchemist, 1, &[(2, 1)], 0);
    g.state.race_mut(Humans).hand = 3;
    assert_eq!(g.hire_target(&target(&g, 2)).0, Special::Denied(Reason::NotEnoughCoins { needed: 1 }));
}

#[test]
fn mercenaries_choose_after_a_failed_die() {
    let map = line(5, &[]);
    for (roll, pay) in [(1, true), (1, false), (0, false)] {
        let mut g = game(&map, 2);
        give(&mut g, Humans, Mercenary, 0, &[(0, 1)], 2);
        give(&mut g, Elves, Alchemist, 1, &[(1, 3)], 0);
        conquest(&mut g, 0, Humans);
        // 3 jetons manquants : possible au dé grâce à la pièce
        assert_eq!(target(&g, 1).kind, TargetKind::Die);
        g.state.rng = rng_for(roll);
        play(&mut g, Action::Reinforce { region: 1 });
        if roll == 0 {
            // même en payant, c'est raté
            assert_eq!(g.state.phase, Phase::Redeploy);
            continue;
        }
        assert_eq!((g.state.phase, g.state.turn.hire), (Phase::Conquest, Some(1)));
        assert_eq!(g.apply(&Action::Conquer { region: 2 }, &mut NoJournal), Err(Reason::HirePending));
        assert_eq!(g.apply(&Action::Hire { region: 2 }, &mut NoJournal), Err(Reason::HirePending));
        if pay {
            play(&mut g, Action::Hire { region: 1 });
            assert_eq!(g.state.regions[1].occupant, Occupant::Race(Humans));
            assert_eq!((g.state.regions[1].tokens, g.state.players[0].coins), (2, 4));
        } else {
            play(&mut g, Action::EndConquest);
            assert_eq!(g.state.regions[1].occupant, Occupant::Race(Elves));
            assert_eq!(g.state.players[0].coins, 5);
        }
        assert_eq!((g.state.phase, g.state.turn.hire), (Phase::Redeploy, None));
    }
}

#[test]
fn hordes_bring_two_more_tokens_until_their_decline() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    g.state.market.push(Combo { race: Ratmen, power: Hordes, coins: 0, tokens: 0 });
    g.state.phase = Phase::Pick;
    play(&mut g, Action::Pick { index: 0 });
    assert_eq!((g.state.race(Ratmen).hand, g.supply(Ratmen)), (13 + 2, 0));
    // des Goules gardent leurs jetons au déclin, mais pas les 2 de la Horde
    let mut g = game(&map, 2);
    give(&mut g, Ghouls, Hordes, 0, &[(0, 6), (1, 6)], 0);
    assert_eq!(g.supply(Ghouls), 0);
    conquest(&mut g, 0, Ghouls);
    play(&mut g, Action::Decline);
    assert_eq!((g.tokens_on_board(Ghouls), g.supply(Ghouls)), (10, 0));
}

#[test]
fn passing_over_a_cursed_combo_costs_three_coins() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    let combo = |race, power| Combo { race, power, coins: 0, tokens: 0 };
    for c in [combo(Humans, Cursed), combo(Elves, Alchemist), combo(Orcs, Flying)] {
        g.state.market.push(c);
    }
    g.state.phase = Phase::Pick;
    assert_eq!(g.pick_price(2), 4);
    g.state.players[0].coins = 3;
    assert_eq!(g.apply(&Action::Pick { index: 2 }, &mut NoJournal), Err(Reason::NotEnoughCoins { needed: 4 }));
    g.state.players[0].coins = 4;
    play(&mut g, Action::Pick { index: 2 });
    assert_eq!(g.state.players[0].coins, 0);
    assert_eq!((g.state.market.as_slice()[0].coins, g.state.market.as_slice()[1].coins), (3, 1));
}

#[test]
fn marauding_races_take_their_tokens_back_once() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, Marauding, 0, &[(0, 3), (1, 1)], 0);
    conquest(&mut g, 0, Humans);
    play(&mut g, Action::Regroup);
    assert_eq!((g.state.regions[0].tokens, g.state.race(Humans).hand), (1, 2));
    assert_eq!(g.apply(&Action::Regroup, &mut NoJournal), Err(Reason::RegroupUsed));
    give(&mut g, Elves, Alchemist, 1, &[(3, 1)], 0);
    conquest(&mut g, 1, Elves);
    assert_eq!(g.can_regroup(), Err(Reason::MissingPower));
    // Kobolds : 2 restent
    let mut g = game(&map, 2);
    give(&mut g, Kobolds, Marauding, 0, &[(0, 5)], 0);
    conquest(&mut g, 0, Kobolds);
    play(&mut g, Action::Regroup);
    assert_eq!((g.state.regions[0].tokens, g.state.race(Kobolds).hand), (2, 3));
}

#[test]
fn ransacking_races_steal_from_active_races_only() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, Ransacking, 0, &[(0, 1)], 10);
    give(&mut g, Orcs, Alchemist, 1, &[(2, 1)], 0);
    set_declined(&mut g, Orcs);
    give(&mut g, Elves, Alchemist, 1, &[(1, 1)], 0);
    conquest(&mut g, 0, Humans);
    let events = play(&mut g, Action::Conquer { region: 1 });
    assert_eq!(paid(&events), [(Payment::Theft, Some(1), Some(0), 1)]);
    let events = play(&mut g, Action::Conquer { region: 2 });
    assert!(paid(&events).is_empty());
}

#[test]
fn were_races_pay_two_less_on_even_rounds() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, Were, 0, &[(0, 1)], 10);
    give(&mut g, Elves, Alchemist, 1, &[(1, 3)], 0);
    conquest(&mut g, 0, Humans);
    assert_eq!(target(&g, 1).cost, 5);
    g.state.round = 2;
    assert_eq!(target(&g, 1).cost, 3);
    // jamais moins de 1
    g.state.regions[1] = RegionState::default();
    assert_eq!(target(&g, 1).cost, 1);
    assert_eq!(g.attack_modifier(Humans, 1), -2);
}

#[test]
fn historians_earn_from_every_decline() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Elves, Alchemist, 1, &[(4, 1)], 0);
    set_declined(&mut g, Elves);
    give(&mut g, Orcs, Alchemist, 1, &[(3, 2)], 0);
    g.state.market.push(Combo { race: Humans, power: Historian, coins: 0, tokens: 0 });
    g.state.phase = Phase::Pick;
    let events = play(&mut g, Action::Pick { index: 0 });
    assert_eq!(paid(&events), [(Payment::Historian, None, Some(0), 1)]);
    conquest(&mut g, 1, Orcs);
    let events = play(&mut g, Action::Decline);
    assert_eq!(paid(&events), [(Payment::Historian, None, Some(0), 1)]);
    // leur propre déclin compte, puis plus rien
    g.state.players[0].active = Some(Humans);
    conquest(&mut g, 0, Humans);
    let events = play(&mut g, Action::Decline);
    assert_eq!(paid(&events), [(Payment::Historian, None, Some(0), 1)]);
    give(&mut g, Dwarves, Alchemist, 1, &[(2, 2)], 0);
    conquest(&mut g, 1, Dwarves);
    let events = play(&mut g, Action::Decline);
    assert!(paid(&events).is_empty());
}

#[test]
fn peace_loving_races_earn_when_they_attack_no_active_race() {
    let map = line(5, &[]);
    let mut g = game(&map, 2);
    give(&mut g, Humans, PeaceLoving, 0, &[(0, 1)], 10);
    give(&mut g, Elves, Alchemist, 1, &[(2, 1)], 0);
    conquest(&mut g, 0, Humans);
    play(&mut g, Action::Conquer { region: 1 });
    assert_eq!(income(&g, 0, IncomeKind::PeaceLoving), Some(3));
    play(&mut g, Action::Conquer { region: 2 });
    assert_eq!(income(&g, 0, IncomeKind::PeaceLoving), None);
}
