//! Lecture des règles : ce que coûte une région, qui peut l'atteindre, ce que rapporte un tour.
//! Rien ici ne modifie l'état.

use crate::action::{Income, IncomeKind, IncomeLine};
use crate::collections::{ArrayVec, RegionSet};
use crate::game::Game;
use crate::ids::{MAX_REGIONS, Phase, Power, Race, Terrain, die_chance};
use crate::reason::{Reason, ensure};
use crate::state::{Occupant, Stage};

/// Ce qu'il faut savoir des régions d'un peuple pour juger de ce qu'il peut atteindre.
#[derive(Clone, Copy, Debug)]
pub struct Foothold {
    /// Le peuple occupe au moins une région.
    pub any: bool,
    /// Il occupe au moins une caverne.
    pub cavern: bool,
    /// Les voisines de ses régions.
    pub near: RegionSet,
}

impl Foothold {
    const NONE: Foothold = Foothold { any: false, cavern: false, near: RegionSet::EMPTY };
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TargetKind {
    #[default]
    None,
    /// Assez de jetons en main.
    Normal,
    /// Possible seulement avec le dé de renfort.
    Die,
}

/// Un pouvoir spécial (dragon, sorcellerie) appliqué à une région.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Special {
    /// Le peuple n'a pas ce pouvoir, ou il ne s'applique pas à cette région.
    #[default]
    NotApplicable,
    Allowed,
    Denied(Reason),
}

impl Special {
    pub fn allowed(self) -> bool {
        self == Special::Allowed
    }
}

/// Une région vue par le peuple qui conquiert.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Target {
    pub region: usize,
    /// Pourquoi la région ne peut pas être visée, même avec assez de jetons.
    pub block: Option<Reason>,
    /// Coût en jetons (avant le dé des Berserks).
    pub cost: i32,
    /// Coût en tenant compte du dé des Berserks déjà lancé.
    pub effective: i32,
    pub kind: TargetKind,
    /// Jetons manquants pour une tentative au dé.
    pub need: i32,
    pub chance: f64,
    pub dragon: Special,
    pub sorcery: Special,
}

impl Target {
    pub fn reachable(&self) -> bool {
        self.block.is_none()
    }

    fn none(region: usize, block: Reason, cost: i32) -> Target {
        Target { region, block: Some(block), cost, effective: cost, ..Target::default() }
    }
}

/// Ce qui baisse le coût d'une conquête.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Modifier {
    #[default]
    Commando,
    Mounted,
    Underworld,
    Tritons,
    Giants,
    /// Gobelins contre un peuple en déclin.
    Goblins,
    /// Garous aux manches paires (−2).
    Were,
}

impl Modifier {
    /// Ce que le modificateur retire au coût.
    pub fn value(self) -> i32 {
        if self == Modifier::Were { 2 } else { 1 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttackHow {
    #[default]
    Conquer,
    Dragon,
    Sorcery,
}

/// Comment un attaquant lointain atteint la région.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Via {
    Flying,
    Cavern,
    /// Par-dessus une région, avec la Catapulte.
    Catapult,
}

/// Ce qui aide un attaquant (pour l'affichage).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttackHint {
    Modifier(Modifier),
    Berserk,
}

/// Un adversaire capable de prendre une région avant le prochain tour de son propriétaire.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Attacker {
    pub player: usize,
    /// `None` : il choisira un nouveau peuple, qui peut entrer par le bord de la carte.
    pub race: Option<Race>,
    /// Jetons à engager (0 : sorcellerie, 1 : dragon).
    pub cost: i32,
    pub how: AttackHow,
    pub hint: Option<AttackHint>,
    pub via: Option<Via>,
}

impl Attacker {
    /// Attaque possible seulement de loin (Volants, cavernes reliées).
    pub fn far(&self) -> bool {
        self.via.is_some()
    }
}

/// Au plus deux armées par joueur (ses Goules, puis son peuple actif).
pub type Attackers = ArrayVec<Attacker, 10>;

impl Game<'_> {
    pub fn region_count(&self) -> usize {
        self.map.len()
    }

    /// Régions du peuple.
    pub fn owned(&self, race: Race) -> RegionSet {
        self.state.owned(race, self.map.len())
    }

    pub fn tokens_on_board(&self, race: Race) -> i32 {
        self.owned(race).iter().map(|i| self.state.regions[i].tokens).sum()
    }

    /// Jetons encore dans la boîte pour ce peuple (de la Horde : 2 de plus tant qu'il est actif).
    pub fn supply(&self, race: Race) -> i32 {
        let (held, extra) = self.state.races[race.index()]
            .map_or((0, 0), |r| (r.hand + i32::from(r.aside), 2 * i32::from(r.has(Power::Hordes))));
        race.max() + extra - self.tokens_on_board(race) - held
    }

    /// Le moins de jetons que le peuple peut avoir dans une région, et le moins qu'une conquête lui
    /// coûte : 2 pour des Kobolds actifs, 1 sinon.
    #[inline]
    pub fn floor(&self, race: Race) -> i32 {
        if race == Race::Kobolds && self.state.races[race.index()].is_some_and(|r| !r.declined) { 2 } else { 1 }
    }

    /// Ce que coûte la combinaison `index` du marché : 1 pièce par combinaison au-dessus, 3 pour une
    /// combinaison Maudite.
    pub fn pick_price(&self, index: usize) -> i32 {
        let market = &self.state.market;
        market.as_slice()[..index.min(market.len())].iter().map(|c| if c.power == Power::Cursed { 3 } else { 1 }).sum()
    }

    /// Chaudrons des Farfadets posés sur le plateau.
    pub fn cauldrons_on_board(&self, race: Race) -> i32 {
        self.owned(race).iter().filter(|&i| self.state.regions[i].cauldron()).count() as i32
    }

    pub fn camps_on_board(&self, race: Race) -> i32 {
        self.owned(race).iter().map(|i| self.state.regions[i].camps).sum()
    }

    pub fn heroes_on_board(&self, race: Race) -> usize {
        self.owned(race).iter().filter(|&i| self.state.regions[i].hero).count()
    }

    pub fn fortresses_on_board(&self, race: Race) -> usize {
        self.owned(race).iter().filter(|&i| self.state.regions[i].fortress).count()
    }

    /// Le propriétaire de la région, s'il s'agit d'un peuple.
    pub fn owner_of(&self, region: usize) -> Option<usize> {
        self.state.regions[region].occupant.race().map(|r| self.state.race(r).owner)
    }

    /// Ce qu'il faut battre pour prendre la région.
    pub fn defense_value(&self, region: usize) -> i32 {
        let r = &self.state.regions[region];
        let occupied = if r.occupant.is_empty() { 0 } else { r.tokens };
        2 + occupied
            + i32::from(self.map.regions[region].terrain == Terrain::Mountain)
            + i32::from(r.lair)
            + i32::from(r.fortress)
            + r.camps
    }

    /// Ce qui baisse le coût d'une conquête pour ce peuple.
    pub fn attack_modifiers(&self, race: Race, region: usize) -> ArrayVec<Modifier, 5> {
        let mut out = ArrayVec::new();
        let rs = self.state.race(race);
        if rs.declined {
            return out;
        }
        let def = &self.map.regions[region];
        if rs.power == Power::Commando {
            out.push(Modifier::Commando);
        }
        if rs.power == Power::Mounted && matches!(def.terrain, Terrain::Hill | Terrain::Farmland) {
            out.push(Modifier::Mounted);
        }
        if rs.power == Power::Underworld && def.cavern {
            out.push(Modifier::Underworld);
        }
        if race == Race::Tritons && def.coastal {
            out.push(Modifier::Tritons);
        }
        if race == Race::Giants
            && def.adj.iter().any(|&a| {
                self.state.regions[a as usize].occupant == Occupant::Race(Race::Giants)
                    && self.map.regions[a as usize].terrain == Terrain::Mountain
            })
        {
            out.push(Modifier::Giants);
        }
        if race == Race::Goblins && self.declined_occupant(region) {
            out.push(Modifier::Goblins);
        }
        if rs.power == Power::Were && self.state.round % 2 == 0 {
            out.push(Modifier::Were);
        }
        out
    }

    /// La région est occupée par un peuple en déclin.
    fn declined_occupant(&self, region: usize) -> bool {
        self.state.regions[region].occupant.race().is_some_and(|r| self.state.race(r).declined)
    }

    /// Total de `attack_modifiers` (chacun vaut −1), sans les énumérer : une boucle chaude.
    pub fn attack_modifier(&self, race: Race, region: usize) -> i32 {
        let rs = self.state.race(race);
        if rs.declined {
            return 0;
        }
        let def = &self.map.regions[region];
        let giants = race == Race::Giants
            && self.map.hoods[region].adj.iter().any(|a| {
                self.state.regions[a].occupant == Occupant::Race(Race::Giants)
                    && self.map.regions[a].terrain == Terrain::Mountain
            });
        // les extensions : rien à regarder sans elles
        let extra = if self.state.expansions.is_empty() {
            0
        } else {
            i32::from(race == Race::Goblins && self.declined_occupant(region))
                + 2 * i32::from(rs.power == Power::Were && self.state.round % 2 == 0)
        };
        -(i32::from(rs.power == Power::Commando)
            + i32::from(rs.power == Power::Mounted && matches!(def.terrain, Terrain::Hill | Terrain::Farmland))
            + i32::from(rs.power == Power::Underworld && def.cavern)
            + i32::from(race == Race::Tritons && def.coastal)
            + i32::from(giants)
            + extra)
    }

    /// Coût d'une conquête, avant le dé des Berserks.
    #[inline]
    pub fn conquest_cost(&self, race: Race, region: usize) -> i32 {
        (self.defense_value(region) + self.attack_modifier(race, region)).max(self.floor(race))
    }

    pub fn foothold(&self, race: Race) -> Foothold {
        let own = self.owned(race);
        let near = RegionSet(own.iter().fold(0, |near, r| near | self.map.hoods[r].adj.0));
        Foothold { any: !own.is_empty(), cavern: own.intersects(self.map.caverns), near }
    }

    /// Le peuple peut-il atteindre la région (adjacence, première conquête) ?
    pub fn reach(&self, race: Race, region: usize, hold: Foothold) -> Result<(), Reason> {
        let def = &self.map.regions[region];
        let rs = self.state.race(race);
        let flying = rs.has(Power::Flying);
        if !hold.any {
            let free_entry = (race == Race::Halflings && !rs.declined) || flying || def.entry;
            return if free_entry { Ok(()) } else { Err(Reason::FirstConquest) };
        }
        if flying {
            return Ok(());
        }
        // les adjacences sont symétriques : voisine d'une région du peuple
        if hold.near.contains(region) {
            return Ok(());
        }
        if rs.has(Power::Underworld) && def.cavern && hold.cavern {
            return Ok(());
        }
        Err(Reason::NotAdjacent)
    }

    /// Ce qui empêche toute attaque de la région par ce peuple (hors coût).
    #[inline]
    pub fn block(&self, race: Race, region: usize, hold: Foothold) -> Option<Reason> {
        self.obstacle(race, region).or_else(|| self.reach(race, region, hold).err())
    }

    /// Ce qui empêche toute attaque de la région par ce peuple, hors coût et hors distance (la
    /// Catapulte tire par-dessus une région).
    #[inline(always)]
    pub fn obstacle(&self, race: Race, region: usize) -> Option<Reason> {
        let def = &self.map.regions[region];
        let r = &self.state.regions[region];
        let rs = self.state.race(race);
        let me = rs.owner;
        if r.occupant == Occupant::Race(race) {
            return Some(Reason::AlreadyYours);
        }
        if let Some(other) = r.occupant.race() {
            let target = self.state.race(other);
            if target.owner == me && !target.declined {
                return Some(Reason::OwnActiveRace);
            }
            if target.owner == me && rs.declined {
                return Some(Reason::GhoulsOwnRace);
            }
            if !target.declined && self.state.peace.iter().any(|p| p.diplomat == target.owner && p.target == me) {
                return Some(Reason::Peace { diplomat: target.owner });
            }
        }
        if def.is_water() && !rs.has(Power::Seafaring) {
            return Some(Reason::Water { sea: def.terrain == Terrain::Sea });
        }
        if let Some(immunity) = r.immunity() {
            return Some(Reason::Immune(immunity));
        }
        if race == Race::Gypsies && self.state.turn.abandoned.contains(region) {
            return Some(Reason::AbandonedThisTurn);
        }
        None
    }

    /// La région vue par le peuple qui conquiert.
    pub fn target(&self, region: usize, hold: Foothold) -> Target {
        let Some(race) = self.state.turn.army.filter(|_| self.state.phase == Phase::Conquest) else {
            return Target::none(region, Reason::NotConquestPhase, 0);
        };
        let rs = self.state.race(race);
        let turn = &self.state.turn;
        let cost = self.conquest_cost(race, region);
        let block = self.block(race, region, hold);
        if turn.die_rolled {
            return Target::none(region, Reason::DieRolled, cost);
        }
        let berserk = rs.has(Power::Berserk);
        let floor = self.floor(race);
        let effective = match turn.berserk_die {
            Some(die) if berserk => (cost - die).max(floor),
            _ => cost,
        };
        let mut t = Target { region, block, cost, effective, ..Target::default() };
        if block.is_none() && rs.hand >= floor {
            if rs.hand >= effective {
                t.kind = TargetKind::Normal;
            } else if !berserk {
                t.need = cost - rs.hand;
                // des Mercenaires qui ont une pièce paieront si le dé ne suffit pas seul
                let mercenary = rs.has(Power::Mercenary) && self.state.players[rs.owner].coins >= 1;
                let hired = if mercenary { (cost - 2).max(floor) } else { cost };
                t.chance = die_chance(hired - rs.hand);
                if t.chance > 0.0 {
                    t.kind = TargetKind::Die;
                }
            }
        }
        if rs.has(Power::DragonMaster) {
            t.dragon = if turn.dragon_used {
                Special::Denied(Reason::DragonUsed)
            } else if rs.hand < floor {
                Special::Denied(if floor == 1 { Reason::NeedOneToken } else { Reason::TokensNeeded { needed: floor } })
            } else if let Some(b) = block {
                Special::Denied(b)
            } else {
                Special::Allowed
            };
        }
        let foe = self.state.regions[region].occupant.race().map(|r| *self.state.race(r));
        if race == Race::Sorcerers
            && !rs.declined
            && let Some(foe) = foe
            && foe.owner != rs.owner
        {
            t.sorcery = if foe.declined {
                Special::Denied(Reason::SorceryDeclinedTarget)
            } else if self.state.regions[region].tokens != 1 {
                Special::Denied(Reason::SorceryNotIsolated)
            } else if turn.sorcery.contains(foe.owner) {
                Special::Denied(Reason::SorceryAlreadyCast { player: foe.owner })
            } else if self.supply(Race::Sorcerers) < 1 {
                Special::Denied(Reason::NoSorcererLeft)
            } else if let Some(b) = block {
                Special::Denied(b)
            } else {
                Special::Allowed
            };
        }
        t
    }

    /// Mercenaires : la conquête payée (1 pièce, 2 jetons de moins, au moins 1) de la région vue par
    /// `t`, et son coût en jetons. Calculée à la demande : `targets` reste aussi léger qu'avant.
    pub fn hire_target(&self, t: &Target) -> (Special, i32) {
        let s = &self.state;
        let Some(race) = s.turn.army.filter(|_| s.phase == Phase::Conquest && !s.turn.die_rolled) else {
            return (Special::NotApplicable, 0);
        };
        let rs = s.race(race);
        if !rs.has(Power::Mercenary) {
            return (Special::NotApplicable, 0);
        }
        let hired = (t.effective - 2).max(self.floor(race));
        let hire = if s.players[rs.owner].coins < 1 {
            Special::Denied(Reason::NotEnoughCoins { needed: 1 })
        } else if let Some(b) = t.block {
            Special::Denied(b)
        } else if rs.hand < hired {
            Special::Denied(Reason::TokensNeeded { needed: hired })
        } else {
            Special::Allowed
        };
        (hire, hired)
    }

    /// Pourquoi la région n'est pas conquérable maintenant (`None` : elle l'est, au moins au dé).
    pub fn target_reason(&self, t: &Target) -> Option<Reason> {
        if t.block.is_some() {
            return t.block;
        }
        let hand = self.state.turn.army.and_then(|r| self.state.races[r.index()]).map_or(0, |r| r.hand);
        (hand < 1).then_some(Reason::NoTokenInHand)
    }

    /// Toutes les régions, vues par le peuple qui conquiert.
    pub fn targets(&self) -> ArrayVec<Target, MAX_REGIONS> {
        let hold = match self.state.turn.army {
            Some(army) if self.state.phase == Phase::Conquest => self.foothold(army),
            _ => Foothold::NONE,
        };
        (0..self.map.len()).map(|i| self.target(i, hold)).collect()
    }

    /// Les régions que le peuple qui conquiert peut viser (`Target::reachable`), dans l'ordre : les
    /// mêmes que `targets`, sans rien calculer pour les autres. Une boucle chaude des labos.
    pub fn reachable_targets(&self) -> impl Iterator<Item = Target> + '_ {
        let army = self.state.turn.army.filter(|_| self.state.phase == Phase::Conquest);
        let (hold, candidates) = match army {
            Some(race) => {
                let hold = self.foothold(race);
                let rs = self.state.race(race);
                let all = RegionSet(u64::MAX >> (64 - self.map.len()));
                // hors de ces régions, `reach` refuse
                let candidates = if !hold.any || rs.has(Power::Flying) {
                    all
                } else if rs.has(Power::Underworld) && hold.cavern {
                    RegionSet(hold.near.0 | self.map.caverns.0)
                } else {
                    hold.near
                };
                (hold, candidates)
            }
            None => (Foothold::NONE, RegionSet::EMPTY),
        };
        candidates.iter().map(move |i| self.target(i, hold)).filter(Target::reachable)
    }

    /// La région la moins chère que le peuple qui conquiert peut prendre sans dé (la première en cas
    /// d'égalité). Même réponse que `targets`, en plus rapide : c'est une boucle chaude de l'IA.
    pub fn cheapest_target(&self) -> Option<usize> {
        let race = self.state.turn.army?;
        if self.state.phase != Phase::Conquest || self.state.turn.die_rolled {
            return None;
        }
        let rs = self.state.race(race);
        if rs.hand < 1 {
            return None;
        }
        let die = self.state.turn.berserk_die.filter(|_| rs.has(Power::Berserk));
        let hold = self.foothold(race);
        let floor = self.floor(race);
        let mut best = None;
        let mut best_cost = i32::MAX;
        for i in 0..self.map.len() {
            let mut cost = self.conquest_cost(race, i);
            if let Some(d) = die {
                cost = (cost - d).max(floor);
            }
            if cost >= best_cost || cost > rs.hand || self.block(race, i, hold).is_some() {
                continue;
            }
            best_cost = cost;
            best = Some(i);
        }
        best
    }

    pub fn can_decline(&self) -> Result<(), Reason> {
        let s = &self.state;
        let t = &s.turn;
        if s.phase != Phase::Conquest {
            Err(Reason::DeclineNotNow)
        } else if t.stage != Stage::Active {
            Err(Reason::GhoulsFirst)
        } else if s.players[s.current].active.is_none() {
            Err(Reason::NoActiveRace)
        } else if t.picked {
            Err(Reason::DeclinePickedThisTurn)
        } else if t.conquests > 0 || t.die_rolled || t.berserk_die.is_some() {
            Err(Reason::DeclineAfterConquest)
        } else if !t.abandoned.is_empty() {
            Err(Reason::DeclineAfterAbandon)
        } else {
            Ok(())
        }
    }

    pub fn can_abandon(&self, region: usize) -> Result<(), Reason> {
        let s = &self.state;
        let t = &s.turn;
        let Some(army) = t.army.filter(|_| s.phase == Phase::Conquest) else {
            return Err(Reason::AbandonNotNow);
        };
        if s.regions[region].occupant != Occupant::Race(army) {
            Err(Reason::NotArmyRegion)
        } else if t.conquests > 0 || t.die_rolled || t.berserk_die.is_some() {
            Err(Reason::AbandonAfterConquest)
        } else {
            Ok(())
        }
    }

    /// Les armées de `q` qui attaqueront encore avant le prochain tour de `owner` (`None` : un peuple
    /// à choisir). Au dernier tour, seuls comptent ceux qui n'ont pas encore joué ; le joueur en cours
    /// ne compte que tant qu'il n'a pas fini ses conquêtes.
    fn armies_before(&self, q: usize, owner: usize) -> ArrayVec<Option<Race>, 2> {
        let s = &self.state;
        let mut out = ArrayVec::new();
        if s.phase == Phase::GameOver || q == owner {
            return out;
        }
        let n = s.player_count();
        let c = s.current;
        let p = &s.players[q];
        let active = p.active.filter(|&r| !s.race(r).declined);
        let fresh = active.is_none() && !s.market.is_empty();
        if q == c {
            let t = &s.turn;
            if t.declined {
                return out;
            }
            if s.phase == Phase::Conquest && t.stage == Stage::Ghouls {
                out.push(Some(Race::Ghouls));
            }
            // après ses Goules, son peuple actif (ou celui qu'il va choisir) attaque encore
            let later = matches!(s.phase, Phase::Pick | Phase::Conquest)
                || (s.phase == Phase::Redeploy && t.stage == Stage::Ghouls);
            if later && (active.is_some() || fresh) {
                out.push(active);
            }
            return out;
        }
        let k = (q + n - c) % n;
        let m = match (owner + n - c) % n {
            0 => n,
            m => m,
        };
        if k >= m || (s.round >= s.max_rounds && q < c) {
            return out;
        }
        // comme au début d'un tour : des Goules sans région ni jeton en main ne rejouent plus
        if let Some(ghouls) = s.races[Race::Ghouls.index()] {
            let playing = !self.owned(Race::Ghouls).is_empty() || ghouls.hand > 0;
            if p.declined.contains(&Race::Ghouls) && ghouls.declined && playing {
                out.push(Some(Race::Ghouls));
            }
        }
        if active.is_some() || fresh {
            out.push(active);
        }
        out
    }

    /// Les adversaires qui pourraient prendre cette région à `owner` avant son prochain tour.
    pub fn attackers(&self, region: usize, owner: usize) -> Attackers {
        let s = &self.state;
        let def = &self.map.regions[region];
        let r = &s.regions[region];
        let mut out = Attackers::new();
        if r.is_immune() {
            return out;
        }
        let mine_active = match r.occupant.race() {
            Some(race) => Some(race) == s.players[owner].active && !s.race(race).declined,
            None => false,
        };
        for q in 0..s.player_count() {
            if mine_active && s.peace.iter().any(|p| p.diplomat == owner && p.target == q) {
                continue;
            }
            for army in self.armies_before(q, owner) {
                let Some(race) = army else {
                    // peuple encore inconnu : seulement l'entrée par le bord, au coût de base
                    if def.entry && !def.is_water() {
                        let cost = self.defense_value(region).max(1);
                        out.push(Attacker { player: q, race: None, cost, ..Attacker::default() });
                    }
                    continue;
                };
                let rs = s.race(race);
                let active = !rs.declined;
                if def.is_water() && !rs.has(Power::Seafaring) {
                    continue;
                }
                let hold = self.foothold(race);
                // la Catapulte tire par-dessus une région, à −1
                let catapult = if self.reach(race, region, hold).is_ok() {
                    false
                } else if rs.has(Power::Catapult) && self.catapult_from(race, region).is_some() {
                    true
                } else {
                    continue;
                };
                let own = self.owned(race);
                let near = own.intersects(self.map.hoods[region].adj)
                    || (own.is_empty() && (def.entry || (race == Race::Halflings && active)));
                let via = match near {
                    true => None,
                    false if catapult => Some(Via::Catapult),
                    false if rs.has(Power::Flying) => Some(Via::Flying),
                    false => Some(Via::Cavern),
                };
                let mods = self.attack_modifiers(race, region);
                let floor = self.floor(race);
                let mut cost = self.defense_value(region) - mods.iter().map(|m| m.value()).sum::<i32>();
                if catapult {
                    cost -= 1;
                }
                if rs.has(Power::Mercenary) && s.players[q].coins >= 1 {
                    cost -= 2;
                }
                let mut a = Attacker {
                    player: q,
                    race: Some(race),
                    cost: cost.max(floor),
                    how: AttackHow::Conquer,
                    hint: None,
                    via,
                };
                // la sorcellerie (gratuite) passe avant le dragon pour des Sorciers dompteurs
                if active && race == Race::Sorcerers && mine_active && r.tokens == 1 && self.supply(race) >= 1 {
                    a.how = AttackHow::Sorcery;
                    a.cost = 0;
                } else if rs.has(Power::DragonMaster) {
                    a.how = AttackHow::Dragon;
                    a.cost = floor;
                }
                if a.how == AttackHow::Conquer {
                    a.hint = match mods.first() {
                        Some(&m) => Some(AttackHint::Modifier(m)),
                        None if rs.has(Power::Berserk) => Some(AttackHint::Berserk),
                        None => None,
                    };
                }
                out.push(a);
            }
        }
        out
    }

    /// Le décompte que rapporterait la fin du tour pour ce joueur, maintenant.
    pub fn income_lines(&self, pid: usize) -> Income {
        let s = &self.state;
        let p = &s.players[pid];
        let mut lines = Income::new();
        let mut line = |kind, race, declined, value| lines.push(IncomeLine { kind, race, declined, value });
        for race in p.active.into_iter().chain(p.declined.iter().copied()) {
            let Some(rs) = s.races[race.index()] else { continue };
            let own = self.owned(race);
            let count = |f: &dyn Fn(usize) -> bool| own.iter().filter(|&i| f(i)).count() as i32;
            let terrain = |t: Terrain| count(&|i| self.map.regions[i].terrain == t);
            // Prêtresses en déclin : la Tour d'Ivoire rapporte ses jetons, pas sa région
            let tower = race == Race::Priestesses && rs.declined;
            if tower {
                let n: i32 = own.iter().map(|i| s.regions[i].tokens).sum();
                if n > 0 {
                    line(IncomeKind::Tower, race, true, n);
                }
            } else if !own.is_empty() {
                line(IncomeKind::Regions, race, rs.declined, own.len() as i32);
            }
            let t = &s.turn;
            let acting = pid == s.current && t.stage == Stage::Active && t.army == Some(race);
            let non_empty = if acting { t.non_empty } else { 0 };
            if race == Race::Dwarves {
                let n = count(&|i| self.map.regions[i].mine);
                if n > 0 {
                    line(IncomeKind::Mines, race, rs.declined, n);
                }
            }
            if rs.declined {
                continue;
            }
            let mut add = |kind, n: i32| {
                if n > 0 {
                    line(kind, race, false, n);
                }
            };
            match race {
                Race::Humans => add(IncomeKind::Farms, terrain(Terrain::Farmland)),
                Race::Wizards => add(IncomeKind::MagicSources, count(&|i| self.map.regions[i].magic)),
                Race::Orcs => add(IncomeKind::OrcLoot, non_empty),
                Race::Gypsies if acting => add(IncomeKind::Abandoned, t.abandoned.len() as i32),
                _ => {}
            }
            match rs.power {
                Power::Alchemist => add(IncomeKind::Alchemist, 2),
                Power::Forest => add(IncomeKind::Forests, terrain(Terrain::Forest)),
                Power::Hill => add(IncomeKind::Hills, terrain(Terrain::Hill)),
                Power::Swamp => add(IncomeKind::Swamps, terrain(Terrain::Swamp)),
                Power::Merchant => add(IncomeKind::Merchant, own.len() as i32),
                Power::Pillaging => add(IncomeKind::Pillage, non_empty),
                Power::Fortified => add(IncomeKind::Fortresses, count(&|i| s.regions[i].fortress)),
                Power::Wealthy if pid == s.current && rs.picked_round == s.round && t.picked => {
                    add(IncomeKind::Wealthy, 7)
                }
                Power::Imperial => add(IncomeKind::Imperial, own.len() as i32 - 3),
                Power::Barricade if own.len() <= 4 => add(IncomeKind::Barricade, 3),
                // les autres joueurs : comme s'ils n'attaquaient personne d'ici la fin de leur tour
                Power::PeaceLoving if pid != s.current || t.attacked.is_empty() => add(IncomeKind::PeaceLoving, 3),
                _ => {}
            }
        }
        lines
    }

    /// Pièces que rapporterait la fin du tour pour ce joueur, maintenant.
    pub fn income(&self, pid: usize) -> i32 {
        self.income_lines(pid).iter().map(|l| l.value).sum()
    }

    // --- extensions ------------------------------------------------------------------------------

    /// Une région intermédiaire (pas une mer) voisine de `a` et de `b`, par-dessus laquelle tire la
    /// Catapulte : `b` est exactement à deux régions de `a` (pas voisine).
    pub fn catapult_over(&self, a: usize, b: usize) -> Option<usize> {
        if a == b || self.map.hoods[a].adj.contains(b) {
            return None;
        }
        let both = RegionSet(self.map.hoods[a].adj.0 & self.map.hoods[b].adj.0);
        both.iter().find(|&m| self.map.regions[m].terrain != Terrain::Sea)
    }

    /// Une région du peuple d'où la Catapulte atteint `region` (la première).
    pub fn catapult_from(&self, race: Race, region: usize) -> Option<usize> {
        self.owned(race).iter().find(|&from| self.catapult_over(from, region).is_some())
    }

    /// Le coût d'un tir de Catapulte de `from` vers `region` par le peuple qui conquiert, ou pourquoi il
    /// est refusé : comme une conquête à −1 (2 au moins pour des Kobolds), payée avec la main, sans dé.
    pub fn catapult_target(&self, from: usize, region: usize) -> Result<i32, Reason> {
        let s = &self.state;
        let race = s.turn.army.filter(|_| s.phase == Phase::Conquest).ok_or(Reason::NotConquestPhase)?;
        let rs = s.race(race);
        ensure(rs.has(Power::Catapult), Reason::MissingPower)?;
        ensure(!s.turn.catapult_used, Reason::CatapultUsed)?;
        ensure(!s.turn.die_rolled, Reason::DieRolled)?;
        ensure(s.regions[from].occupant == Occupant::Race(race), Reason::NotYourRegion)?;
        if let Some(reason) = self.obstacle(race, region) {
            return Err(reason);
        }
        ensure(self.catapult_over(from, region).is_some(), Reason::CatapultOutOfRange)?;
        let floor = self.floor(race);
        let mut cost = (self.conquest_cost(race, region) - 1).max(floor);
        if let Some(die) = s.turn.berserk_die.filter(|_| rs.has(Power::Berserk)) {
            cost = (cost - die).max(floor);
        }
        ensure(rs.hand >= cost, Reason::TokensNeeded { needed: cost })?;
        Ok(cost)
    }

    /// Les tirs de Catapulte permis maintenant : région de tir, cible, coût.
    pub fn catapult_options(&self) -> Vec<(usize, usize, i32)> {
        let s = &self.state;
        let Some(race) = s.turn.army.filter(|_| s.phase == Phase::Conquest) else { return Vec::new() };
        if !s.race(race).has(Power::Catapult) || s.turn.catapult_used {
            return Vec::new();
        }
        let mut out = Vec::new();
        for from in self.owned(race).iter() {
            for region in 0..self.map.len() {
                if let Ok(cost) = self.catapult_target(from, region) {
                    out.push((from, region, cost));
                }
            }
        }
        out
    }

    /// Les Intrépides peuvent-ils reprendre leurs jetons maintenant ?
    pub fn can_regroup(&self) -> Result<(), Reason> {
        let s = &self.state;
        let race = s.turn.army.filter(|_| s.phase == Phase::Conquest).ok_or(Reason::NotConquestPhase)?;
        ensure(s.turn.hire.is_none(), Reason::HirePending)?;
        ensure(s.race(race).has(Power::Marauding), Reason::MissingPower)?;
        ensure(!s.turn.regrouped, Reason::RegroupUsed)?;
        ensure(!s.turn.die_rolled, Reason::DieRolled)
    }

    /// Les Prêtresses peuvent-elles choisir leur Tour d'Ivoire maintenant (avant un déclin possible) ?
    /// `region` : la région visée, s'il y en a une.
    pub fn can_tower(&self, region: Option<usize>) -> Result<(), Reason> {
        let s = &self.state;
        let t = &s.turn;
        let army = t.army.filter(|_| matches!(s.phase, Phase::Conquest | Phase::Redeploy));
        let Some(race) = army.filter(|_| t.stage == Stage::Active) else { return Err(Reason::TowerNotNow) };
        let rs = s.race(race);
        ensure(race == Race::Priestesses && !rs.declined, Reason::MissingPower)?;
        if s.phase == Phase::Conquest {
            self.can_decline()?;
        } else {
            ensure(rs.has(Power::Stout), Reason::OnlyStout)?;
            ensure(t.abandoned.is_empty(), Reason::DeclineAfterAbandon)?;
        }
        if let Some(r) = region {
            ensure(s.regions[r].occupant == Occupant::Race(race), Reason::NotYourRegion)?;
        }
        Ok(())
    }

    /// La Tour d'Ivoire des Prêtresses : celle choisie, sinon la région la moins menacée (la mieux
    /// défendue à menace égale, puis la première).
    pub fn tower_region(&self) -> Option<usize> {
        let own = self.owned(Race::Priestesses);
        if let Some(r) = self.state.turn.tower.map(usize::from).filter(|&r| own.contains(r)) {
            return Some(r);
        }
        let mut best = None;
        let mut best_key = (f64::INFINITY, i32::MIN);
        for r in own.iter() {
            let key = (self.threat(Race::Priestesses, r), self.defense_value(r));
            if key.0 < best_key.0 || (key.0 == best_key.0 && key.1 > best_key.1) {
                best_key = key;
                best = Some(r);
            }
        }
        best
    }
}
