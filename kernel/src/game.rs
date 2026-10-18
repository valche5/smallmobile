//! Le déroulement de la partie : chaque action est vérifiée, puis appliquée. Une action refusée ne
//! change rien (ni l'état, ni le hasard).

use crate::action::{Action, Event, How, Journal, Logged, Placement};
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
                self.enter_redeploy(journal);
                Ok(())
            }
            Action::Deploy { region, delta } => {
                region_ok(region)?;
                self.deploy(region, delta)
            }
            Action::AutoDeploy { all } => self.auto_deploy(all),
            Action::Place(placement) => self.place(&placement),
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
        let paid = index as i32;
        ensure(self.state.players[pid].coins >= paid, Reason::NotEnoughCoins { needed: paid })?;
        let s = &mut self.state;
        for earlier in &mut s.market.as_mut_slice()[..index] {
            earlier.coins += 1;
        }
        s.players[pid].coins += slot.coins - paid;
        s.market.remove(index);
        self.refill_market();
        let s = &mut self.state;
        let (race, power) = (slot.race, slot.power);
        let tokens = race.starting_tokens(power);
        s.races[race.index()] = Some(RaceState {
            power,
            owner: pid,
            declined: false,
            hand: tokens,
            holes_placed: 0,
            picked_round: s.round,
        });
        s.players[pid].active = Some(race);
        s.turn.picked = true;
        s.turn.army = Some(race);
        s.turn.readied = [0; MAX_REGIONS];
        self.grant_amazons();
        self.state.phase = Phase::Conquest;
        let event = Event::Picked { player: pid, race, power, tokens, paid, gained: slot.coins };
        self.emit(journal, event);
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
        let hand = self.state.race(army).hand;
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
        }
        self.enter_redeploy(journal);
        Ok(())
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
        self.state.race_mut(army).hand -= 1;
        self.occupy(region, army, 1, How::Dragon, journal);
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

    fn deploy(&mut self, region: usize, delta: i32) -> Result<(), Reason> {
        let race = self.deploy_race()?;
        let r = self.state.regions[region];
        ensure(r.occupant == Occupant::Race(race), Reason::NotYourRegion)?;
        if delta > 0 {
            ensure(self.state.race(race).hand >= delta, Reason::NoTokenInHand)?;
        } else if delta < 0 {
            ensure(r.tokens as i64 + delta as i64 >= 1, Reason::KeepOneToken)?;
        }
        self.state.race_mut(race).hand -= delta;
        self.state.regions[region].tokens += delta;
        Ok(())
    }

    fn auto_deploy(&mut self, all: bool) -> Result<(), Reason> {
        let race = self.deploy_race()?;
        if all {
            let target = self.full_deployment(race);
            self.set_tokens(race, &target);
        } else {
            self.auto_place(race);
        }
        Ok(())
    }

    fn place(&mut self, placement: &Placement) -> Result<(), Reason> {
        ensure(self.state.phase == Phase::Redeploy, Reason::NotRedeployPhase)?;
        let race = self.deploy_race()?;
        let own = self.owned(race);
        let total = self.state.race(race).hand as i64 + self.tokens_on_board(race) as i64;
        let mut placed = 0i64;
        for r in own.iter() {
            let n = placement.0[r];
            ensure(n >= 1, Reason::KeepOneToken)?;
            placed += n as i64;
        }
        ensure(placed <= total, Reason::NotEnoughTokens)?;
        self.set_tokens(race, placement);
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
            s.market.push(Combo { race, power, coins: 0 });
        }
        s.race_tail = s.race_tail.min(s.race_deck.len() as u8);
        s.power_tail = s.power_tail.min(s.power_deck.len() as u8);
    }

    /// Reprend en main les jetons du peuple, sauf un par région.
    fn ready_troops(&mut self, race: Race) {
        let mut readied = [0; MAX_REGIONS];
        let mut hand = 0;
        for i in self.owned(race).iter() {
            let r = &mut self.state.regions[i];
            if r.tokens > 1 {
                readied[i] = r.tokens - 1;
                hand += r.tokens - 1;
                r.tokens = 1;
            }
        }
        self.state.race_mut(race).hand += hand;
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
        let was_occupied = !self.state.regions[region].occupant.is_empty();
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
        }
        let rs = self.state.race_mut(race);
        rs.hand = 0;
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
