//! Le déroulement de la partie : chaque action est vérifiée, puis appliquée. Une action refusée ne
//! change rien (ni l'état, ni le hasard).

use crate::action::{Action, Event, How, Journal, Logged, Payment, Placement};
use crate::collections::ArrayVec;
use crate::ids::{MAX_REGIONS, Phase, Power, Race};
use crate::map::Map;
use crate::reason::{Reason, ensure};
use crate::rng;
use crate::rules::{Special, TargetKind};
use crate::state::{Combo, DefenseTask, DieRoll, Occupant, Peace, RaceState, RegionState, Stage, State, Turn};

/// Une partie : la carte (partagée) et l'état (copié par les simulations).
#[derive(Clone, Copy, Debug)]
pub struct Game<'m> {
    pub map: &'m Map,
    pub state: State,
}

/// Ce qu'une conquête fait à l'occupant précédent.
struct Eviction {
    player: Option<usize>,
    occupant: Occupant,
    active: bool,
    lost: i32,
    returned: i32,
}

impl<'m> Game<'m> {
    pub fn new(map: &'m Map, state: State) -> Self {
        Game { map, state }
    }

    /// Joue l'action du joueur qui doit agir, ou explique pourquoi elle est refusée.
    pub fn apply(&mut self, action: &Action, journal: &mut impl Journal) -> Result<(), Reason> {
        if self.state.phase == Phase::GameOver {
            return Err(Reason::GameOver);
        }
        let region_ok = |r: usize| ensure(r < self.map.len(), Reason::InvalidRegion);
        // Mercenaires : après un dé raté, payer ou renoncer, rien d'autre
        if self.state.turn.hire.is_some() && !matches!(action, Action::Hire { .. } | Action::EndConquest) {
            return Err(Reason::HirePending);
        }
        match *action {
            Action::Pick { index } => self.pick(index, journal),
            Action::SkipPick => self.skip_pick(journal),
            Action::Conquer { region } => {
                region_ok(region)?;
                self.conquer(region, journal)
            }
            Action::Reinforce { region } => {
                region_ok(region)?;
                self.reinforce(region, journal)
            }
            Action::BerserkRoll => self.berserk_roll(journal),
            Action::Dragon { region } => {
                region_ok(region)?;
                self.dragon(region, journal)
            }
            Action::Sorcery { region } => {
                region_ok(region)?;
                self.sorcery(region, journal)
            }
            Action::Abandon { region } => {
                region_ok(region)?;
                self.abandon(region, journal)
            }
            Action::Decline => self.decline(journal),
            Action::EndConquest => {
                self.require_conquest()?;
                self.state.turn.hire = None;
                self.enter_redeploy(journal);
                Ok(())
            }
            Action::Deploy { region, delta } => {
                region_ok(region)?;
                self.deploy(region, delta)
            }
            Action::AutoDeploy { all } => self.auto_deploy(all),
            Action::Place(placement) => self.place(&placement, journal),
            Action::Camp { region, delta } => {
                region_ok(region)?;
                self.camp(region, delta)
            }
            Action::Fortress { region } => {
                region_ok(region)?;
                self.fortress(region)
            }
            Action::Hero { region } => {
                region_ok(region)?;
                self.hero(region)
            }
            Action::EndTurn { peace, stout } => self.end_turn(peace, stout, journal),
            Action::EndDefense => self.end_defense(journal),
            Action::Catapult { from, region } => {
                region_ok(from)?;
                region_ok(region)?;
                self.catapult(from, region, journal)
            }
            Action::Hire { region } => {
                region_ok(region)?;
                self.hire(region, journal)
            }
            Action::Regroup => self.regroup(),
            Action::Cauldron { region } => {
                region_ok(region)?;
                self.cauldron(region)
            }
            Action::Tower { region } => {
                region_ok(region)?;
                self.can_tower(Some(region))?;
                self.state.turn.tower = Some(region as u8);
                Ok(())
            }
        }
    }

    /// Prépare une partie dont l'état initial (carte, joueurs, pioches) vient d'être construit : remplit
    /// le marché et commence le premier tour.
    pub fn start(&mut self, journal: &mut impl Journal) {
        self.refill_market();
        self.emit(journal, Event::GameStarted);
        self.start_turn(journal);
    }

    fn emit(&mut self, journal: &mut impl Journal, event: Event) {
        self.state.seq = self.state.seq.wrapping_add(1);
        journal.record(Logged { seq: self.state.seq, round: self.state.round, event });
    }

    // --- actions ---------------------------------------------------------------------------------

    fn pick(&mut self, index: usize, journal: &mut impl Journal) -> Result<(), Reason> {
        ensure(self.state.phase == Phase::Pick, Reason::NotPickPhase)?;
        let slot = *self.state.market.get(index).ok_or(Reason::ComboUnavailable)?;
        let pid = self.state.current;
        let paid = self.pick_price(index);
        ensure(self.state.players[pid].coins >= paid, Reason::NotEnoughCoins { needed: paid })?;
        let s = &mut self.state;
        for earlier in &mut s.market.as_mut_slice()[..index] {
            // une combinaison Maudite coûte 3 pièces à passer ; les Homoncules grossissent
            earlier.coins += if earlier.power == Power::Cursed { 3 } else { 1 };
            if earlier.race == Race::Homunculi {
                earlier.tokens = (earlier.tokens + 1).min(Race::Homunculi.max());
            }
        }
        s.players[pid].coins += slot.coins - paid;
        s.market.remove(index);
        self.refill_market();
        let s = &mut self.state;
        let (race, power) = (slot.race, slot.power);
        let tokens = slot.starting_tokens();
        s.races[race.index()] = Some(RaceState {
            power,
            owner: pid,
            declined: false,
            hand: tokens,
            holes_placed: 0,
            picked_round: s.round,
            aside: 0,
            cauldrons: if race == Race::Leprechauns { 16 } else { 0 },
        });
        s.players[pid].active = Some(race);
        s.turn.picked = true;
        s.turn.army = Some(race);
        s.turn.readied = [0; MAX_REGIONS];
        self.grant_amazons();
        self.state.phase = Phase::Conquest;
        let event = Event::Picked { player: pid, race, power, tokens, paid, gained: slot.coins };
        self.emit(journal, event);
        if power == Power::Historian {
            let declined = self.state.races.iter().flatten().filter(|r| r.declined).count() as i32;
            self.pay(Payment::Historian, None, Some(pid), declined, journal);
        }
        Ok(())
    }

    fn skip_pick(&mut self, journal: &mut impl Journal) -> Result<(), Reason> {
        ensure(self.state.phase == Phase::Pick, Reason::NotPickPhase)?;
        ensure(self.state.market.is_empty(), Reason::MarketNotEmpty)?;
        self.finish_turn(None, false, journal);
        Ok(())
    }

    fn require_conquest(&self) -> Result<Race, Reason> {
        self.state.turn.army.filter(|_| self.state.phase == Phase::Conquest).ok_or(Reason::NotConquestPhase)
    }

    fn conquer(&mut self, region: usize, journal: &mut impl Journal) -> Result<(), Reason> {
        let army = self.require_conquest()?;
        let t = self.target(region, self.foothold(army));
        if let Some(block) = t.block {
            return Err(block);
        }
        ensure(t.kind == TargetKind::Normal, Reason::TokensNeeded { needed: t.effective })?;
        self.state.race_mut(army).hand -= t.effective;
        self.occupy(region, army, t.effective, How::Conquer, journal);
        self.state.turn.berserk_die = None;
        Ok(())
    }

    fn reinforce(&mut self, region: usize, journal: &mut impl Journal) -> Result<(), Reason> {
        let army = self.require_conquest()?;
        let t = self.target(region, self.foothold(army));
        if let Some(block) = t.block {
            return Err(block);
        }
        ensure(t.kind != TargetKind::Normal, Reason::EnoughTokensForNormal)?;
        ensure(t.kind == TargetKind::Die, Reason::DieCannotSucceed)?;
        let roll = rng::roll_die(&mut self.state.rng);
        let rs = *self.state.race(army);
        let hand = rs.hand;
        let success = hand + roll >= t.cost;
        self.state.turn.die_rolled = true;
        let player = self.state.current;
        let event =
            Event::DieRolled { player, region: Some(region), value: roll, need: t.need, success: Some(success) };
        self.emit(journal, event);
        self.state.last_die = Some(DieRoll {
            seq: self.state.seq,
            player,
            value: roll,
            need: t.need,
            success: Some(success),
            region: Some(region),
            berserk: false,
        });
        if success {
            self.state.race_mut(army).hand = 0;
            self.occupy(region, army, hand, How::Die, journal);
        } else if rs.has(Power::Mercenary)
            && self.state.players[rs.owner].coins >= 1
            && hand + roll >= (t.cost - 2).max(self.floor(army))
        {
            // les Mercenaires choisiront : payer 1 pièce pour la prendre, ou renoncer
            self.state.turn.hire = Some(region as u8);
            return Ok(());
        }
        self.enter_redeploy(journal);
        Ok(())
    }

    fn hire(&mut self, region: usize, journal: &mut impl Journal) -> Result<(), Reason> {
        let army = self.require_conquest()?;
        let pid = self.state.race(army).owner;
        if let Some(pending) = self.state.turn.hire {
            ensure(usize::from(pending) == region, Reason::HirePending)?;
            self.pay(Payment::Hire, Some(pid), None, 1, journal);
            self.state.turn.hire = None;
            let hand = self.state.race(army).hand;
            self.state.race_mut(army).hand = 0;
            if let Some(d) = self.state.last_die.as_mut() {
                d.success = Some(true);
            }
            self.occupy(region, army, hand, How::Die, journal);
            self.enter_redeploy(journal);
            return Ok(());
        }
        let t = self.target(region, self.foothold(army));
        let (hire, hired) = self.hire_target(&t);
        match hire {
            Special::Allowed => {}
            Special::Denied(reason) => return Err(reason),
            Special::NotApplicable => return Err(t.block.unwrap_or(Reason::MissingPower)),
        }
        self.pay(Payment::Hire, Some(pid), None, 1, journal);
        self.state.race_mut(army).hand -= hired;
        self.occupy(region, army, hired, How::Conquer, journal);
        self.state.turn.berserk_die = None;
        Ok(())
    }

    fn catapult(&mut self, from: usize, region: usize, journal: &mut impl Journal) -> Result<(), Reason> {
        let army = self.require_conquest()?;
        let cost = self.catapult_target(from, region)?;
        self.state.race_mut(army).hand -= cost;
        for r in &mut self.state.regions {
            r.mark(RegionState::CATAPULT, false);
        }
        self.occupy(region, army, cost, How::Catapult, journal);
        self.state.regions[from].mark(RegionState::CATAPULT, true);
        self.state.turn.catapult_used = true;
        self.state.turn.berserk_die = None;
        Ok(())
    }

    /// Intrépides : les jetons repris en main, sauf ce qui doit rester par région.
    fn regroup(&mut self) -> Result<(), Reason> {
        self.can_regroup()?;
        let race = self.state.turn.army.expect("peuple en action");
        let floor = self.floor(race);
        let mut hand = 0;
        for i in self.owned(race).iter() {
            let r = &mut self.state.regions[i];
            if r.tokens > floor {
                self.state.turn.readied[i] += r.tokens - floor;
                hand += r.tokens - floor;
                r.tokens = floor;
            }
        }
        self.state.race_mut(race).hand += hand;
        self.state.turn.regrouped = true;
        Ok(())
    }

    /// Farfadets : pose un Chaudron, ou retire celui posé.
    fn cauldron(&mut self, region: usize) -> Result<(), Reason> {
        let s = &self.state;
        ensure(s.phase == Phase::Redeploy && s.turn.stage == Stage::Active, Reason::NotActiveRedeploy)?;
        let race = s.turn.army.expect("peuple en action");
        ensure(race == Race::Leprechauns && !s.race(race).declined, Reason::MissingPower)?;
        let r = s.regions[region];
        ensure(r.occupant == Occupant::Race(race), Reason::NotYourRegion)?;
        if r.cauldron() {
            self.state.race_mut(race).cauldrons += 1;
        } else {
            ensure(s.race(race).cauldrons > 0, Reason::AllCauldronsPlaced)?;
            self.state.race_mut(race).cauldrons -= 1;
        }
        self.state.regions[region].mark(RegionState::CAULDRON, !r.cauldron());
        Ok(())
    }

    /// Des pièces changent de main (au plus ce que `from` possède) ; rien si le montant est nul.
    fn pay(&mut self, kind: Payment, from: Option<usize>, to: Option<usize>, coins: i32, journal: &mut impl Journal) {
        let coins = match from {
            Some(p) => coins.min(self.state.players[p].coins),
            None => coins,
        };
        if coins <= 0 {
            return;
        }
        if let Some(p) = from {
            self.state.players[p].coins -= coins;
        }
        if let Some(p) = to {
            self.state.players[p].coins += coins;
        }
        self.emit(journal, Event::Paid { kind, from, to, coins });
    }

    fn berserk_roll(&mut self, journal: &mut impl Journal) -> Result<(), Reason> {
        let army = self.require_conquest()?;
        let rs = self.state.race(army);
        ensure(rs.has(Power::Berserk), Reason::NotBerserk)?;
        ensure(self.state.turn.berserk_die.is_none(), Reason::BerserkAlreadyRolled)?;
        ensure(rs.hand >= 1, Reason::NeedOneToken)?;
        let roll = rng::roll_die(&mut self.state.rng);
        self.state.turn.berserk_die = Some(roll);
        let player = self.state.current;
        self.emit(journal, Event::DieRolled { player, region: None, value: roll, need: 0, success: None });
        self.state.last_die = Some(DieRoll {
            seq: self.state.seq,
            player,
            value: roll,
            need: 0,
            success: None,
            region: None,
            berserk: true,
        });
        Ok(())
    }

    fn dragon(&mut self, region: usize, journal: &mut impl Journal) -> Result<(), Reason> {
        let army = self.require_conquest()?;
        match self.target(region, self.foothold(army)).dragon {
            Special::Allowed => {}
            Special::Denied(reason) => return Err(reason),
            Special::NotApplicable => return Err(Reason::DragonImpossible),
        }
        let cost = self.floor(army);
        self.state.race_mut(army).hand -= cost;
        self.occupy(region, army, cost, How::Dragon, journal);
        Ok(())
    }

    fn sorcery(&mut self, region: usize, journal: &mut impl Journal) -> Result<(), Reason> {
        let army = self.require_conquest()?;
        match self.target(region, self.foothold(army)).sorcery {
            Special::Allowed => {}
            Special::Denied(reason) => return Err(reason),
            Special::NotApplicable => return Err(Reason::SorceryImpossible),
        }
        self.occupy(region, army, 1, How::Sorcery, journal);
        Ok(())
    }

    fn abandon(&mut self, region: usize, journal: &mut impl Journal) -> Result<(), Reason> {
        self.can_abandon(region)?;
        let s = &mut self.state;
        let army = s.turn.army.expect("peuple en action");
        s.race_mut(army).hand += s.regions[region].tokens;
        s.regions[region] = RegionState::default();
        s.turn.readied[region] = 0;
        s.last_region = Some(region);
        if army == Race::Gypsies {
            s.turn.abandoned.insert(region);
        }
        let player = s.current;
        self.emit(journal, Event::Abandoned { player, region });
        Ok(())
    }

    fn decline(&mut self, journal: &mut impl Journal) -> Result<(), Reason> {
        self.can_decline()?;
        self.decline_active(true, journal);
        self.state.turn.declined = true;
        self.finish_turn(None, false, journal);
        Ok(())
    }

    /// Le peuple que l'on redéploie : celui du tour, ou celui d'un joueur attaqué.
    fn deploy_race(&self) -> Result<Race, Reason> {
        match self.state.phase {
            Phase::Redeploy => Ok(self.state.turn.army.expect("peuple en action")),
            Phase::Defend => Ok(self.state.defense[0].race),
            _ => Err(Reason::NotRedeployPhase),
        }
    }

    /// Barbares et Fées actifs : rien à redéployer à la fin de leur tour.
    fn no_redeploy(&self, race: Race) -> bool {
        self.state.phase == Phase::Redeploy
            && self.state.turn.stage == Stage::Active
            && matches!(race, Race::Barbarians | Race::Pixies)
            && !self.state.race(race).declined
    }

    /// La raison de refuser moins de `floor` jetons dans une région.
    fn keep_floor(floor: i32) -> Reason {
        if floor == 1 { Reason::KeepOneToken } else { Reason::KeepTwoTokens }
    }

    fn deploy(&mut self, region: usize, delta: i32) -> Result<(), Reason> {
        let race = self.deploy_race()?;
        ensure(!self.no_redeploy(race), Reason::NoRedeploy)?;
        let r = self.state.regions[region];
        ensure(r.occupant == Occupant::Race(race), Reason::NotYourRegion)?;
        if delta > 0 {
            ensure(self.state.race(race).hand >= delta, Reason::NoTokenInHand)?;
        } else if delta < 0 {
            let floor = self.floor(race);
            ensure(r.tokens as i64 + delta as i64 >= floor as i64, Self::keep_floor(floor))?;
        }
        self.state.race_mut(race).hand -= delta;
        self.state.regions[region].tokens += delta;
        Ok(())
    }

    fn auto_deploy(&mut self, all: bool) -> Result<(), Reason> {
        let race = self.deploy_race()?;
        ensure(!self.no_redeploy(race), Reason::NoRedeploy)?;
        if all {
            let target = self.full_deployment(race);
            self.set_tokens(race, &target);
        } else {
            self.auto_place(race);
        }
        Ok(())
    }

    fn place(&mut self, placement: &Placement, journal: &mut impl Journal) -> Result<(), Reason> {
        ensure(self.state.phase == Phase::Redeploy, Reason::NotRedeployPhase)?;
        let race = self.deploy_race()?;
        ensure(!self.no_redeploy(race), Reason::NoRedeploy)?;
        let own = self.owned(race);
        let total = self.state.race(race).hand as i64 + self.tokens_on_board(race) as i64;
        let floor = self.floor(race);
        let mut placed = 0i64;
        for r in own.iter() {
            let n = placement.0[r];
            // des Kobolds abandonnent la région qu'ils ne peuvent pas garder à 2 (0 jeton)
            ensure(n >= floor || (floor > 1 && n == 0), Self::keep_floor(floor))?;
            placed += n as i64;
        }
        ensure(placed <= total, Reason::NotEnoughTokens)?;
        self.set_tokens(race, placement);
        for r in own.iter().filter(|&r| placement.0[r] == 0) {
            self.state.regions[r] = RegionState::default();
            self.state.last_region = Some(r);
            let player = self.state.race(race).owner;
            self.emit(journal, Event::Abandoned { player, region: r });
        }
        Ok(())
    }

    /// Fixe les jetons de chaque région du peuple ; le reste va en main.
    fn set_tokens(&mut self, race: Race, placement: &Placement) {
        let mut total = self.state.race(race).hand + self.tokens_on_board(race);
        for r in self.owned(race).iter() {
            self.state.regions[r].tokens = placement.0[r];
            total -= placement.0[r];
        }
        self.state.race_mut(race).hand = total;
    }

    /// Le peuple actif en redéploiement, s'il a ce pouvoir.
    fn require_power(&self, power: Power) -> Result<Race, Reason> {
        let s = &self.state;
        ensure(s.phase == Phase::Redeploy && s.turn.stage == Stage::Active, Reason::NotActiveRedeploy)?;
        let race = s.turn.army.expect("peuple en action");
        ensure(s.race(race).has(power), Reason::MissingPower)?;
        Ok(race)
    }

    fn camp(&mut self, region: usize, delta: i32) -> Result<(), Reason> {
        let race = self.require_power(Power::Bivouacking)?;
        let r = self.state.regions[region];
        ensure(r.occupant == Occupant::Race(race), Reason::NotYourRegion)?;
        if delta > 0 {
            ensure(self.camps_on_board(race) < 5, Reason::AllCampsPlaced)?;
            self.state.regions[region].camps += 1;
        } else {
            ensure(r.camps > 0, Reason::NoCampHere)?;
            self.state.regions[region].camps -= 1;
        }
        Ok(())
    }

    fn fortress(&mut self, region: usize) -> Result<(), Reason> {
        let race = self.require_power(Power::Fortified)?;
        let r = self.state.regions[region];
        ensure(r.occupant == Occupant::Race(race), Reason::NotYourRegion)?;
        let built_this_turn = self.state.turn.fortress_region;
        if built_this_turn == Some(region) {
            self.state.regions[region].fortress = false;
            self.state.turn.fortress_region = None;
            return Ok(());
        }
        ensure(built_this_turn.is_none(), Reason::OneFortressPerTurn)?;
        ensure(!r.fortress, Reason::FortressAlreadyHere)?;
        ensure(self.fortresses_on_board(race) < 6, Reason::AllFortressesBuilt)?;
        self.state.regions[region].fortress = true;
        self.state.turn.fortress_region = Some(region);
        Ok(())
    }

    fn hero(&mut self, region: usize) -> Result<(), Reason> {
        let race = self.require_power(Power::Heroic)?;
        let r = self.state.regions[region];
        ensure(r.occupant == Occupant::Race(race), Reason::NotYourRegion)?;
        if !r.hero {
            ensure(self.heroes_on_board(race) < 2, Reason::AllHeroesPlaced)?;
        }
        self.state.regions[region].hero = !r.hero;
        Ok(())
    }

    fn end_turn(&mut self, peace: Option<usize>, stout: bool, journal: &mut impl Journal) -> Result<(), Reason> {
        ensure(self.state.phase == Phase::Redeploy, Reason::NotEndOfTurn)?;
        let race = self.state.turn.army.expect("peuple en action");
        let rs = *self.state.race(race);
        ensure(rs.hand <= 0 || self.owned(race).is_empty(), Reason::PlaceHandFirst { hand: rs.hand })?;
        let pid = self.state.current;
        if self.state.turn.stage == Stage::Ghouls {
            self.emit(journal, Event::GhoulsDone { player: pid });
            self.start_active_stage(journal);
            return Ok(());
        }
        if let Some(target) = peace {
            ensure(rs.has(Power::Diplomat), Reason::OnlyDiplomats)?;
            ensure(target != pid && target < self.state.player_count(), Reason::InvalidOpponent)?;
            ensure(!self.state.turn.attacked.contains(target), Reason::AttackedThisTurn)?;
        }
        ensure(!stout || rs.has(Power::Stout), Reason::OnlyStout)?;
        ensure(!stout || self.state.turn.abandoned.is_empty(), Reason::DeclineAfterAbandon)?;
        self.finish_turn(peace, stout, journal);
        Ok(())
    }

    fn end_defense(&mut self, journal: &mut impl Journal) -> Result<(), Reason> {
        ensure(self.state.phase == Phase::Defend, Reason::NoDefenseInProgress)?;
        let task = self.state.defense[0];
        if let Some(rs) = self.state.races[task.race.index()] {
            let pending = rs.hand > 0 && !self.owned(task.race).is_empty();
            ensure(!pending, Reason::PlaceHandFirst { hand: rs.hand })?;
        }
        self.emit(journal, Event::Defended { player: task.player, race: task.race });
        self.state.defense.remove(0);
        if self.state.defense.is_empty() {
            self.next_player(journal);
        }
        Ok(())
    }

    // --- déroulement du tour -------------------------------------------------------------------

    fn refill_market(&mut self) {
        let s = &mut self.state;
        while s.market.len() < 6 && !s.race_deck.is_empty() && !s.power_deck.is_empty() {
            let race = s.race_deck.remove(0);
            let power = s.power_deck.remove(0);
            s.market.push(Combo { race, power, coins: 0, tokens: 0 });
        }
        s.race_tail = s.race_tail.min(s.race_deck.len() as u8);
        s.power_tail = s.power_tail.min(s.power_deck.len() as u8);
    }

    /// Reprend en main les jetons du peuple, sauf un par région (deux pour des Kobolds), et ceux mis
    /// hors du plateau à son tour précédent (Barbares, Fées).
    fn ready_troops(&mut self, race: Race) {
        let mut readied = [0; MAX_REGIONS];
        let mut hand = 0;
        let floor = self.floor(race);
        for i in self.owned(race).iter() {
            let r = &mut self.state.regions[i];
            if r.tokens > floor {
                readied[i] = r.tokens - floor;
                hand += r.tokens - floor;
                r.tokens = floor;
            }
        }
        let rs = self.state.race_mut(race);
        rs.hand += hand + i32::from(rs.aside);
        rs.aside = 0;
        self.state.turn.readied = readied;
    }

    fn grant_amazons(&mut self) {
        if self.state.turn.army != Some(Race::Amazons) || self.state.turn.stage != Stage::Active {
            return;
        }
        let n = self.supply(Race::Amazons).min(4);
        self.state.race_mut(Race::Amazons).hand += n;
        self.state.turn.amazon_bonus = n;
    }

    fn start_turn(&mut self, journal: &mut impl Journal) {
        self.state.turn = Turn::default();
        let pid = self.state.current;
        if let Some(ghouls) = self.state.races[Race::Ghouls.index()] {
            let playing = !self.owned(Race::Ghouls).is_empty() || ghouls.hand > 0;
            if ghouls.owner == pid && ghouls.declined && playing {
                let turn = &mut self.state.turn;
                turn.stage = Stage::Ghouls;
                turn.army = Some(Race::Ghouls);
                self.ready_troops(Race::Ghouls);
                self.state.phase = Phase::Conquest;
                let event = Event::TurnStarted {
                    player: pid,
                    race: Race::Ghouls,
                    power: ghouls.power,
                    hand: self.state.race(Race::Ghouls).hand,
                    ghouls: true,
                    amazon_bonus: 0,
                };
                self.emit(journal, event);
                return;
            }
        }
        self.start_active_stage(journal);
    }

    fn start_active_stage(&mut self, journal: &mut impl Journal) {
        let pid = self.state.current;
        let turn = &mut self.state.turn;
        turn.stage = Stage::Active;
        turn.conquests = 0;
        turn.berserk_die = None;
        turn.die_rolled = false;
        turn.readied = [0; MAX_REGIONS];
        let Some(race) = self.state.players[pid].active else {
            self.state.turn.army = None;
            self.state.phase = Phase::Pick;
            return;
        };
        self.state.turn.army = Some(race);
        self.ready_troops(race);
        self.grant_amazons();
        self.state.phase = Phase::Conquest;
        let rs = self.state.race(race);
        let event = Event::TurnStarted {
            player: pid,
            race,
            power: rs.power,
            hand: rs.hand,
            ghouls: false,
            amazon_bonus: self.state.turn.amazon_bonus,
        };
        self.emit(journal, event);
        // Farfadets : chaque Chaudron resté en place rapporte 1 pièce et quitte le jeu
        if race == Race::Leprechauns {
            let mut cashed = 0;
            for i in self.owned(race).iter() {
                cashed += i32::from(self.state.regions[i].cauldron());
                self.state.regions[i].mark(RegionState::CAULDRON, false);
            }
            self.pay(Payment::Cauldrons, None, Some(pid), cashed, journal);
        }
    }

    /// Chasse l'occupant de la région ; ses jetons sont perdus ou rendus en main.
    fn evict(&mut self, region: usize, sorcery: bool) -> Eviction {
        let r = self.state.regions[region];
        let mut out = Eviction { player: None, occupant: r.occupant, active: false, lost: 0, returned: 0 };
        match r.occupant {
            Occupant::Empty => {}
            Occupant::Lost => out.lost = r.tokens,
            Occupant::Race(race) => {
                let ds = self.state.race_mut(race);
                out.player = Some(ds.owner);
                out.active = !ds.declined;
                out.lost = if sorcery {
                    r.tokens
                } else if race == Race::Elves && !ds.declined {
                    0
                } else {
                    r.tokens.min(1)
                };
                out.returned = r.tokens - out.lost;
                ds.hand += out.returned;
            }
        }
        self.state.regions[region] = RegionState::default();
        out
    }

    fn occupy(&mut self, region: usize, race: Race, tokens: i32, how: How, journal: &mut impl Journal) {
        let before = self.state.regions[region];
        let was_occupied = !before.occupant.is_empty();
        let victim = self.evict(region, how == How::Sorcery);
        let s = &mut self.state;
        let rs = *s.race(race);
        let r = &mut s.regions[region];
        r.occupant = Occupant::Race(race);
        r.tokens = tokens;
        if race == Race::Trolls {
            r.lair = true;
        }
        if race == Race::Halflings && !rs.declined && rs.holes_placed < 2 {
            r.hole = true;
            s.race_mut(race).holes_placed += 1;
        }
        if how == How::Dragon {
            for other in &mut s.regions {
                other.dragon = false;
            }
            s.regions[region].dragon = true;
            s.turn.dragon_used = true;
        }
        let t = &mut s.turn;
        t.conquests += 1;
        if was_occupied && t.stage == Stage::Active {
            t.non_empty += 1;
        }
        if let Some(v) = victim.player {
            if victim.active && v != rs.owner {
                t.attacked.insert(v);
            }
            if how == How::Sorcery {
                t.sorcery.insert(v);
            }
        }
        s.last_region = Some(region);
        let event = Event::Conquered {
            player: rs.owner,
            race,
            region,
            how,
            tokens,
            victim: victim.player,
            victim_occupant: victim.occupant,
            victim_active: victim.active,
            lost: victim.lost,
            returned: victim.returned,
        };
        self.emit(journal, event);
        if !self.state.expansions.is_empty() {
            self.conquest_effects(race, before, &victim, journal);
        }
    }

    /// Ce que les extensions ajoutent à une conquête : renforts des Pygmées, pots-de-vin des
    /// Corrompus, vol des Voleurs, Chaudron pris.
    fn conquest_effects(&mut self, race: Race, before: RegionState, victim: &Eviction, journal: &mut impl Journal) {
        let attacker = *self.state.race(race);
        if let Occupant::Race(v) = victim.occupant {
            let vs = *self.state.race(v);
            if v == Race::Pygmies && !vs.declined {
                for _ in 0..victim.lost {
                    let value = rng::roll_die(&mut self.state.rng);
                    let tokens = value.min(self.supply(v)).max(0);
                    self.state.race_mut(v).hand += tokens;
                    self.emit(journal, Event::PygmiesReturned { player: vs.owner, value, tokens });
                }
            }
            if vs.owner != attacker.owner {
                if vs.has(Power::Corrupt) {
                    self.pay(Payment::Bribe, Some(attacker.owner), Some(vs.owner), 1, journal);
                }
                if attacker.has(Power::Ransacking) && !vs.declined {
                    self.pay(Payment::Theft, Some(vs.owner), Some(attacker.owner), 1, journal);
                }
            }
        }
        if before.cauldron() {
            self.pay(Payment::Cauldron, None, Some(attacker.owner), 1, journal);
        }
    }

    fn enter_redeploy(&mut self, journal: &mut impl Journal) {
        let race = self.state.turn.army.expect("peuple en action");
        let player = self.state.race(race).owner;
        self.state.turn.berserk_die = None;
        let active_stage = self.state.turn.stage == Stage::Active;
        let bonus = self.state.turn.amazon_bonus;
        if active_stage && race == Race::Amazons && bonus > 0 {
            // les jetons prêtés pour conquérir sont rendus : d'abord ceux en main, puis sur les plus grosses piles
            let rs = self.state.race_mut(race);
            let from_hand = bonus.min(rs.hand);
            rs.hand -= from_hand;
            for _ in 0..bonus - from_hand {
                let own = self.owned(race);
                let mut best: Option<usize> = None;
                for i in own.iter() {
                    let tokens = self.state.regions[i].tokens;
                    if tokens > 1 && best.is_none_or(|b| tokens > self.state.regions[b].tokens) {
                        best = Some(i);
                    }
                }
                let Some(b) = best else { break };
                self.state.regions[b].tokens -= 1;
            }
            self.emit(journal, Event::AmazonsReturned { player, tokens: bonus });
            self.state.turn.amazon_bonus = 0;
        }
        if active_stage && matches!(race, Race::Barbarians | Race::Pixies) && !self.state.race(race).declined {
            // pas de redéploiement : les jetons en main (et pour les Fées, tout ce qui dépasse un par
            // région) quittent le plateau jusqu'à leur prochain tour
            let mut aside = 0;
            if race == Race::Pixies {
                for i in self.owned(race).iter() {
                    let r = &mut self.state.regions[i];
                    aside += r.tokens - 1;
                    r.tokens = 1;
                }
            }
            let rs = self.state.race_mut(race);
            rs.aside = u8::try_from(i32::from(rs.aside) + aside + rs.hand).expect("jetons hors du plateau");
            rs.hand = 0;
        }
        if active_stage && race == Race::Skeletons && !self.state.race(race).declined {
            let raised = (self.state.turn.non_empty / 2).min(self.supply(race));
            if raised > 0 {
                self.state.race_mut(race).hand += raised;
                self.emit(journal, Event::SkeletonsRaised { player, tokens: raised });
            }
        }
        self.state.phase = Phase::Redeploy;
    }

    /// Retire un peuple du jeu : ses régions se vident, ses cartes retournent sous les pioches.
    fn remove_race(&mut self, race: Race) {
        let rs = *self.state.race(race);
        for i in self.owned(race).iter() {
            self.state.regions[i] = RegionState::default();
        }
        let s = &mut self.state;
        s.races[race.index()] = None;
        // les cartes retournent sous les pioches (une carte en double, impossible en partie, est écartée)
        if s.race_deck.try_push(race).is_ok() {
            s.race_tail = (s.race_tail + 1).min(s.race_deck.len() as u8);
        }
        if s.power_deck.try_push(rs.power).is_ok() {
            s.power_tail = (s.power_tail + 1).min(s.power_deck.len() as u8);
        }
        let p = &mut s.players[rs.owner];
        p.declined.retain(|&r| r != race);
        if p.active == Some(race) {
            p.active = None;
        }
        s.defense.retain(|d| d.race != race);
    }

    /// Passe le peuple actif du joueur en déclin. `restore_ghouls` : des Goules qui déclinent en début
    /// de tour reprennent les jetons qu'elles venaient de ramasser.
    fn decline_active(&mut self, restore_ghouls: bool, journal: &mut impl Journal) {
        let pid = self.state.current;
        let race = self.state.players[pid].active.expect("peuple actif");
        let own = self.owned(race);
        let readied = self.state.turn.readied;
        // Historiens : +1 à chaque passage en déclin, tant qu'ils sont actifs (et pour le leur)
        let historian = match self.state.expansions.is_empty() {
            true => None,
            false => Race::ALL.iter().copied().find(|&h| {
                self.state.races[h.index()]
                    .is_some_and(|hs| hs.power == Power::Historian && (!hs.declined || h == race))
            }),
        };
        // Prêtresses : une seule région garde ses jetons, la Tour d'Ivoire
        let tower = if race == Race::Priestesses { self.tower_region() } else { None };
        for i in own.iter() {
            let r = &mut self.state.regions[i];
            if race != Race::Ghouls {
                r.tokens = 1;
            } else if restore_ghouls {
                r.tokens += readied[i];
            }
            r.camps = 0;
            r.hero = false;
            r.dragon = false;
            r.hole = false;
            r.marks = if race == Race::WhiteLadies { RegionState::LADIES } else { 0 };
        }
        if let Some(t) = tower {
            for i in own.iter().filter(|&i| i != t) {
                self.state.regions[i] = RegionState::default();
            }
            self.state.regions[t].tokens = own.len() as i32;
        }
        // de la Horde : les 2 jetons de plus disparaissent (sur les plus grosses piles des Goules, sinon
        // avec leur région)
        if self.state.race(race).power == Power::Hordes {
            for _ in 0..self.tokens_on_board(race) - race.max() {
                let own = self.owned(race);
                let stack = own.iter().filter(|&i| self.state.regions[i].tokens > 1);
                match stack.max_by_key(|&i| self.state.regions[i].tokens) {
                    Some(i) => self.state.regions[i].tokens -= 1,
                    None => {
                        let last = own.iter().last().expect("région du peuple");
                        self.state.regions[last] = RegionState::default();
                    }
                }
            }
        }
        let rs = self.state.race_mut(race);
        rs.hand = 0;
        rs.aside = 0;
        rs.cauldrons = 0;
        rs.declined = true;
        let power = rs.power;
        let p = &mut self.state.players[pid];
        p.active = None;
        // un seul peuple en déclin reste sur la carte, plus un Spirituel
        let all: ArrayVec<Race, 3> = core::iter::once(race).chain(p.declined.iter().copied()).collect();
        let spirit = |r: Race| self.state.race(r).power == Power::Spirit;
        let first_spirit = all.iter().copied().find(|&r| spirit(r));
        let first_other = all.iter().copied().find(|&r| !spirit(r));
        let keep = |r: Race| Some(r) == first_spirit || Some(r) == first_other;
        let removed: ArrayVec<Race, 2> = all.iter().copied().filter(|&r| !keep(r)).collect();
        self.state.players[pid].declined = all.iter().copied().filter(|&r| keep(r)).collect();
        for &r in removed.iter() {
            self.remove_race(r);
        }
        self.emit(journal, Event::Declined { player: pid, race, power, removed });
        if let Some(h) = historian.filter(|&h| self.state.races[h.index()].is_some()) {
            let owner = self.state.race(h).owner;
            self.pay(Payment::Historian, None, Some(owner), 1, journal);
        }
    }

    fn finish_turn(&mut self, peace: Option<usize>, stout: bool, journal: &mut impl Journal) {
        let pid = self.state.current;
        let lines = self.income_lines(pid);
        let gain = lines.iter().map(|l| l.value).sum();
        let p = &mut self.state.players[pid];
        p.coins += gain;
        p.last_income = gain;
        self.emit(journal, Event::Scored { player: pid, gain, lines });
        if stout {
            self.decline_active(false, journal);
        }
        self.state.peace.retain(|x| x.target != pid);
        if let Some(target) = peace {
            self.state.peace.push(Peace { diplomat: pid, target });
            let race = self.state.players[pid].active;
            self.emit(journal, Event::PeaceImposed { player: pid, target, race });
        }

        let n = self.state.player_count();
        let next = (pid + 1) % n;
        // tout dernier tour : personne ne rejouera, le moteur replace tout sans phase de défense
        // (les jetons en jeu départagent les égalités de pièces)
        let last = self.state.round == self.state.max_rounds && pid == n - 1;
        self.state.defense.clear();
        for k in 1..=n {
            let q = (pid + k) % n;
            let p = self.state.players[q];
            for race in p.active.into_iter().chain(p.declined.iter().copied()) {
                let Some(rs) = self.state.races[race.index()] else { continue };
                if rs.hand <= 0 {
                    continue;
                }
                if rs.declined && race != Race::Ghouls {
                    self.state.race_mut(race).hand = 0;
                    continue;
                }
                if self.owned(race).is_empty() {
                    continue;
                }
                // le joueur suivant reprend ses jetons au début de son tour : le moteur les replace pour lui
                if q == next || last {
                    self.auto_place(race);
                } else {
                    self.state.defense.push(DefenseTask { player: q, race });
                }
            }
        }
        if self.state.defense.is_empty() {
            self.next_player(journal);
        } else {
            self.state.phase = Phase::Defend;
        }
    }

    /// Pose les jetons en main là où ils protègent le mieux (même plan que l'action `AutoDeploy`).
    pub(crate) fn auto_place(&mut self, race: Race) {
        let plan = self.plan_deployment(race, false);
        for i in self.owned(race).iter() {
            self.state.regions[i].tokens += plan[i];
            self.state.race_mut(race).hand -= plan[i];
        }
    }

    fn next_player(&mut self, journal: &mut impl Journal) {
        let s = &mut self.state;
        s.current = (s.current + 1) % s.player_count();
        if s.current == 0 {
            s.round += 1;
        }
        if s.round > s.max_rounds {
            s.round = s.max_rounds;
            s.phase = Phase::GameOver;
            self.emit(journal, Event::GameOver);
            return;
        }
        self.start_turn(journal);
    }

    /// L'état juste après un passage en déclin (avant le décompte), pour les aperçus.
    pub fn declined_preview(&self) -> Game<'m> {
        let mut g = *self;
        g.decline_active(true, &mut crate::action::NoJournal);
        g
    }

    /// Nombre de régions que le peuple en action peut encore prendre ce tour sans dé (les moins chères d'abord).
    pub fn estimate_conquests(&self) -> i32 {
        let mut g = *self;
        let mut count = 0;
        for _ in 0..25 {
            let Some(region) = g.cheapest_target() else { break };
            g.conquer(region, &mut crate::action::NoJournal).expect("conquête la moins chère");
            count += 1;
        }
        count
    }
}
