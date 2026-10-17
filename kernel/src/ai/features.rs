//! Les entrées des modèles appris : ce que l'IA mesure de la partie avant chaque décision. L'ordre
//! des entrées est celui de l'apprentissage ; `Row` le vérifie, nom par nom, dans les compilations
//! de mise au point (tests, `kernel:check`).

use super::round_half_up;
use crate::collections::RegionSet;
use crate::game::Game;
use crate::ids::{Power, Race, Terrain};
use crate::rules::{Target, TargetKind};
use crate::state::{Combo, Occupant};

pub const MAX_INPUTS: usize = 128;

/// Les noms des entrées d'un modèle, dans l'ordre. Un modèle appris lit un préfixe de cette liste :
/// les indicateurs ajoutés après lui (en fin de liste) ne le gênent pas.
#[derive(Debug)]
pub struct Layout {
    pub names: Vec<String>,
}

impl Layout {
    fn new(head: &[&str], one_hots: bool, tail: &[&str]) -> Layout {
        let mut names: Vec<String> = head.iter().map(|s| s.to_string()).collect();
        if one_hots {
            names.extend(Race::ALL.iter().map(|r| format!("race:{}", r.key())));
            names.extend(Power::ALL.iter().map(|p| format!("power:{}", p.key())));
        }
        names.extend(tail.iter().map(|s| s.to_string()));
        assert!(names.len() <= MAX_INPUTS);
        Layout { names }
    }

    /// Les entrées du modèle sont-elles le début de cette liste ?
    pub fn accepts(&self, inputs: &[String]) -> bool {
        inputs.len() <= self.names.len() && self.names.iter().zip(inputs).all(|(a, b)| a == b)
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
}

macro_rules! layout {
    ($name:ident, $head:expr, $one_hots:expr, $tail:expr) => {
        pub fn $name() -> &'static Layout {
            static L: std::sync::OnceLock<Layout> = std::sync::OnceLock::new();
            L.get_or_init(|| Layout::new($head, $one_hots, $tail))
        }
    };
}

const PICK_SHARED: &[&str] = &[
    "players",
    "left",
    "myDeclined",
    "empty",
    "tribes",
    "entry",
    "weak",
    "oppActive",
    "oppDeclined",
    "oppStack",
    "farm",
    "forest",
    "hill",
    "swamp",
    "mountain",
    "mine",
    "magic",
    "cavern",
    "water",
    "freeFarm",
    "freeForest",
    "freeHill",
    "freeSwamp",
    "freeMountain",
    "freeMine",
    "freeMagic",
    "freeCavern",
    "freeCoast",
    "oppAttackMax",
    "oppAttackSum",
    "oppSpent",
    "oppNoRace",
    "oppAge",
    "oppDeclining",
    "nextAttack",
    "seat",
    "soft",
    "softEntry",
    "emptyCluster",
    "tokens",
];
const ENTRY: &[&str] = &[
    "players",
    "left",
    "hand",
    "choices",
    "myDeclined",
    "cost",
    "handAfter",
    "lost",
    "foeActive",
    "foeDeclined",
    "defenders",
    "bonus",
    "mountain",
    "border",
    "coastal",
    "cavern",
    "mine",
    "magic",
    "farmland",
    "forest",
    "hill",
    "swamp",
    "near",
    "nearEmpty",
    "nearLost",
    "nearFoe",
    "nearFoeTokens",
    "nearFoeMax",
    "nearFoeDeclined",
    "nearMine",
    "nearBonus",
    "nearCheap",
    "nearMountain",
    "nearWater",
    "far",
    "farEmpty",
    "farBonus",
    "farFoeTokens",
    "threat",
    "centre",
];
const CONQUEST: &[&str] = &[
    "players",
    "left",
    "hand",
    "choices",
    "myDeclined",
    "own",
    "conquests",
    "onBoard",
    "income",
    "cheapest",
    "declined",
    "stop",
    "cost",
    "handAfter",
    "lost",
    "foeActive",
    "foeDeclined",
    "defenders",
    "bonus",
    "mountain",
    "border",
    "coastal",
    "cavern",
    "mine",
    "magic",
    "farmland",
    "forest",
    "hill",
    "swamp",
    "near",
    "nearEmpty",
    "nearLost",
    "nearFoe",
    "nearFoeTokens",
    "nearFoeMax",
    "nearFoeDeclined",
    "nearArmy",
    "nearMine",
    "nearBonus",
    "nearCheap",
    "nearMountain",
    "nearWater",
    "far",
    "farEmpty",
    "farBonus",
    "farFoeTokens",
    "farArmy",
    "threat",
    "centre",
];
/// Indicateurs des conquêtes ajoutés ensuite, rangés en fin de liste.
const CONQUEST_LATER: &[&str] = &[
    "spare",
    "exposed",
    "foesAttack",
    "gain",
    "nextCheapest",
    "after",
    "ownerAttack",
    "ownerRegions",
    "chance",
    "need",
];
const DECLINE: &[&str] = &[
    "players",
    "left",
    "own",
    "est",
    "tokens",
    "hand",
    "free",
    "lost",
    "income",
    "fresh",
    "age",
    "market",
    "mine",
    "gap",
    "ghouls",
    "spirit",
    "incomeActive",
    "oppAttack",
    "danger",
];
const HOLD: &[&str] = &[
    "tokens",
    "defense",
    "players",
    "left",
    "own",
    "army",
    "mountain",
    "entry",
    "works",
    "attackers",
    "fresh",
    "far",
    "sorcery",
    "dragon",
    "cheapest",
    "freeMax",
    "freeSum",
    "nearFoeTokens",
    "nearFoeMax",
    "nearOwn",
    "nearEmpty",
    "nearFoeDeclined",
    "threat",
    "worth",
    "border",
    "coastal",
];

layout!(pick_layout, PICK_SHARED, true, &[]);
layout!(entry_layout, ENTRY, true, &[]);
layout!(conquest_layout, CONQUEST, true, CONQUEST_LATER);
layout!(decline_layout, DECLINE, true, &[]);
layout!(hold_layout, HOLD, false, &[]);

/// Les entrées d'un modèle, écrites dans l'ordre de leur liste (vérifié nom par nom dans les
/// compilations de mise au point).
pub struct Row {
    layout: &'static Layout,
    x: [f64; MAX_INPUTS],
    at: usize,
}

impl Row {
    pub fn new(layout: &'static Layout) -> Self {
        Row { layout, x: [0.0; MAX_INPUTS], at: 0 }
    }

    pub fn put(&mut self, name: &str, value: f64) -> &mut Self {
        debug_assert_eq!(self.layout.names[self.at], name, "entrée n° {}", self.at);
        self.x[self.at] = value;
        self.at += 1;
        self
    }

    /// Une case par peuple, à 1 pour `race`.
    pub fn races(&mut self, race: Race) -> &mut Self {
        for &r in Race::ALL {
            debug_assert_eq!(self.layout.names[self.at], format!("race:{}", r.key()));
            self.x[self.at] = f64::from(r == race);
            self.at += 1;
        }
        self
    }

    /// Une case par pouvoir, à 1 pour `power`.
    pub fn powers(&mut self, power: Power) -> &mut Self {
        for &p in Power::ALL {
            debug_assert_eq!(self.layout.names[self.at], format!("power:{}", p.key()));
            self.x[self.at] = f64::from(p == power);
            self.at += 1;
        }
        self
    }

    /// Laisse `n` entrées à 0.
    pub fn skip(&mut self, n: usize) -> &mut Self {
        self.at += n;
        self
    }

    pub fn finish(&self) -> Inputs {
        assert_eq!(self.at, self.layout.names.len(), "entrées manquantes");
        Inputs { x: self.x, len: self.at }
    }
}

/// Les entrées complètes d'un modèle.
#[derive(Clone, Copy)]
pub struct Inputs {
    x: [f64; MAX_INPUTS],
    len: usize,
}

impl Inputs {
    pub fn get(&self) -> &[f64] {
        &self.x[..self.len]
    }

    pub fn set(&mut self, i: usize, v: f64) {
        self.x[i] = v;
    }

    pub fn at(&self, i: usize) -> f64 {
        self.x[i]
    }
}

fn b(x: bool) -> f64 {
    f64::from(x)
}

/// Ce que vaut une région pour le peuple qui la tiendrait : 1 pièce de plus par Mine, Ferme, Source…
pub fn bonus(g: &Game, region: usize, race: Race, power: Power) -> i32 {
    let d = &g.map.regions[region];
    i32::from(race == Race::Dwarves && d.mine)
        + i32::from(race == Race::Humans && d.terrain == Terrain::Farmland)
        + i32::from(race == Race::Wizards && d.magic)
        + i32::from(power == Power::Forest && d.terrain == Terrain::Forest)
        + i32::from(power == Power::Hill && d.terrain == Terrain::Hill)
        + i32::from(power == Power::Swamp && d.terrain == Terrain::Swamp)
}

/// Une région vue par un joueur : 0 vide, 1 tribu oubliée, 2 à lui (autre peuple), 3 adversaire en
/// déclin, 4 adversaire actif, 5 le peuple qui conquiert.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Empty,
    Lost,
    Mine,
    FoeDeclined,
    FoeActive,
    Army,
}

impl Game<'_> {
    fn kind(&self, region: usize, pid: usize, army: Option<Race>) -> Kind {
        match self.state.regions[region].occupant {
            Occupant::Empty => Kind::Empty,
            Occupant::Lost => Kind::Lost,
            Occupant::Race(r) if Some(r) == army => Kind::Army,
            Occupant::Race(r) => {
                let rs = self.state.race(r);
                if rs.owner == pid {
                    Kind::Mine
                } else if rs.declined {
                    Kind::FoeDeclined
                } else {
                    Kind::FoeActive
                }
            }
        }
    }

    fn declined_regions(&self, pid: usize, except: Option<Race>) -> i32 {
        let p = &self.state.players[pid];
        p.declined.iter().filter(|&&r| Some(r) != except).map(|&r| self.owned(r).len() as i32).sum()
    }

    /// Jetons du peuple au-delà d'un par région : ce qu'il peut engager.
    fn spare_tokens(&self, race: Race) -> (i32, i32) {
        let held = self.owned(race).len() as i32;
        (self.tokens_on_board(race) + self.state.race(race).hand - held, held)
    }

    fn rounded_threat(&self, race: Race, region: usize) -> f64 {
        round_half_up(10.0 * self.threat(race, region)) / 10.0
    }
}

// --- choix du peuple ----------------------------------------------------------------------------

/// Ce qu'un joueur voit de la partie au moment de choisir un peuple (commun à toutes les combinaisons).
#[derive(Clone, Copy, Default)]
pub struct PickContext {
    players: f64,
    left: f64,
    my_declined: f64,
    empty: f64,
    tribes: f64,
    entry: f64,
    weak: f64,
    opp_active: f64,
    opp_declined: f64,
    opp_stack: f64,
    farm: f64,
    forest: f64,
    hill: f64,
    swamp: f64,
    mountain: f64,
    mine: f64,
    magic: f64,
    cavern: f64,
    water: f64,
    free_farm: f64,
    free_forest: f64,
    free_hill: f64,
    free_swamp: f64,
    free_mountain: f64,
    free_mine: f64,
    free_magic: f64,
    free_cavern: f64,
    free_coast: f64,
    opp_attack_max: f64,
    opp_attack_sum: f64,
    opp_spent: f64,
    opp_no_race: f64,
    opp_age: f64,
    opp_declining: f64,
    next_attack: f64,
    seat: f64,
    soft: f64,
    soft_entry: f64,
    empty_cluster: f64,
}

impl Game<'_> {
    pub fn pick_context(&self) -> PickContext {
        let s = &self.state;
        let pid = s.current;
        let n = s.player_count();
        let mut c = PickContext {
            players: n as f64,
            left: (s.max_rounds - s.round + 1) as f64,
            seat: (pid + 1) as f64,
            ..PickContext::default()
        };
        let mut land = 0;
        let mut empty = 0;
        let mut stack = 0;
        for (i, def) in self.map.regions.iter().enumerate() {
            let r = &s.regions[i];
            if def.is_water() {
                c.water += 1.0;
                continue;
            }
            land += 1;
            let free = r.occupant.is_empty();
            if free {
                empty += 1;
            }
            let terrain = match def.terrain {
                Terrain::Farmland => Some((&mut c.farm, &mut c.free_farm)),
                Terrain::Forest => Some((&mut c.forest, &mut c.free_forest)),
                Terrain::Hill => Some((&mut c.hill, &mut c.free_hill)),
                Terrain::Swamp => Some((&mut c.swamp, &mut c.free_swamp)),
                Terrain::Mountain => Some((&mut c.mountain, &mut c.free_mountain)),
                _ => None,
            };
            if let Some((all, free_count)) = terrain {
                *all += 1.0;
                if free {
                    *free_count += 1.0;
                }
            }
            for (flag, all, free_count) in [
                (def.mine, &mut c.mine, &mut c.free_mine),
                (def.magic, &mut c.magic, &mut c.free_magic),
                (def.cavern, &mut c.cavern, &mut c.free_cavern),
            ] {
                if flag {
                    *all += 1.0;
                    if free {
                        *free_count += 1.0;
                    }
                }
            }
            if free && def.coastal {
                c.free_coast += 1.0;
            }
            if free && def.entry {
                c.entry += 1.0;
            }
            let own_race = r.occupant.race().is_some_and(|race| s.race(race).owner == pid);
            if !free && r.tokens == 1 && !own_race {
                c.soft += 1.0;
                if def.entry {
                    c.soft_entry += 1.0;
                }
            }
            match r.occupant {
                Occupant::Empty => {}
                Occupant::Lost => c.tribes += 1.0,
                Occupant::Race(race) => {
                    let rs = s.race(race);
                    if rs.owner == pid {
                        c.my_declined += 1.0;
                    } else if rs.declined {
                        c.opp_declined += 1.0;
                    } else {
                        c.opp_active += 1.0;
                        stack += r.tokens;
                        if r.tokens == 1 {
                            c.weak += 1.0;
                        }
                    }
                }
            }
        }
        // ce que chaque adversaire pourra engager à son prochain tour : ses jetons au-delà d'un par
        // région ; sans peuple actif, il en choisira un neuf (une dizaine de jetons)
        let next = (pid + 1) % n;
        let mut ages = 0;
        let mut aged = 0;
        for q in (0..n).filter(|&q| q != pid) {
            let p = &s.players[q];
            let Some(race) = p.active.filter(|&r| !s.race(r).declined) else {
                c.opp_no_race += 1.0;
                if q == next {
                    c.next_attack = 10.0;
                }
                continue;
            };
            let (attack, regions) = self.spare_tokens(race);
            // la règle de déclin du jeu, vue de l'extérieur : presque plus de quoi conquérir, et peu à sacrifier
            let sacrificed = self.declined_regions(q, None);
            if attack < 4 && regions > 0 && sacrificed < regions && s.max_rounds - s.round >= 2 {
                c.opp_declining += 1.0;
            }
            if q == next {
                c.next_attack = attack as f64;
            }
            c.opp_attack_sum += attack as f64;
            c.opp_attack_max = c.opp_attack_max.max(attack as f64);
            if attack <= 3 {
                c.opp_spent += 1.0;
            }
            ages += s.round - s.race(race).picked_round;
            aged += 1;
        }
        if aged > 0 {
            c.opp_age = round_half_up(10.0 * (ages as f64 / aged as f64)) / 10.0;
        }
        c.empty_cluster = self.largest_empty_cluster() as f64;
        c.empty = round_half_up(100.0 * empty as f64 / land as f64);
        if c.opp_active > 0.0 {
            c.opp_stack = round_half_up(10.0 * stack as f64 / c.opp_active) / 10.0;
        }
        c
    }

    /// Plus grande zone de régions terrestres vides d'un seul tenant : la place pour s'étendre sans combattre.
    fn largest_empty_cluster(&self) -> usize {
        let open: RegionSet = (0..self.map.len())
            .filter(|&i| self.state.regions[i].occupant.is_empty() && !self.map.regions[i].is_water())
            .collect();
        let mut seen = RegionSet::EMPTY;
        let mut best = 0;
        for start in open.iter() {
            if seen.contains(start) {
                continue;
            }
            let mut zone = RegionSet::EMPTY;
            zone.insert(start);
            let mut todo = zone;
            while let Some(at) = todo.first() {
                todo.0 &= todo.0 - 1;
                let fresh = RegionSet(self.map.hoods[at].adj.0 & open.0 & !zone.0);
                zone.0 |= fresh.0;
                todo.0 |= fresh.0;
            }
            seen.0 |= zone.0;
            best = best.max(zone.len());
        }
        best
    }

    pub fn pick_inputs(&self, c: &PickContext, slot: Combo) -> Inputs {
        let mut x = Row::new(pick_layout());
        x.put("players", c.players)
            .put("left", c.left)
            .put("myDeclined", c.my_declined)
            .put("empty", c.empty)
            .put("tribes", c.tribes)
            .put("entry", c.entry)
            .put("weak", c.weak)
            .put("oppActive", c.opp_active)
            .put("oppDeclined", c.opp_declined)
            .put("oppStack", c.opp_stack)
            .put("farm", c.farm)
            .put("forest", c.forest)
            .put("hill", c.hill)
            .put("swamp", c.swamp)
            .put("mountain", c.mountain)
            .put("mine", c.mine)
            .put("magic", c.magic)
            .put("cavern", c.cavern)
            .put("water", c.water)
            .put("freeFarm", c.free_farm)
            .put("freeForest", c.free_forest)
            .put("freeHill", c.free_hill)
            .put("freeSwamp", c.free_swamp)
            .put("freeMountain", c.free_mountain)
            .put("freeMine", c.free_mine)
            .put("freeMagic", c.free_magic)
            .put("freeCavern", c.free_cavern)
            .put("freeCoast", c.free_coast)
            .put("oppAttackMax", c.opp_attack_max)
            .put("oppAttackSum", c.opp_attack_sum)
            .put("oppSpent", c.opp_spent)
            .put("oppNoRace", c.opp_no_race)
            .put("oppAge", c.opp_age)
            .put("oppDeclining", c.opp_declining)
            .put("nextAttack", c.next_attack)
            .put("seat", c.seat)
            .put("soft", c.soft)
            .put("softEntry", c.soft_entry)
            .put("emptyCluster", c.empty_cluster)
            .put("tokens", slot.race.starting_tokens(slot.power) as f64)
            .races(slot.race)
            .powers(slot.power);
        x.finish()
    }
}

// --- voisinage d'une région (entrée, conquêtes) -------------------------------------------------

/// Ce qu'il y a autour d'une région visée.
#[derive(Default)]
struct Surroundings {
    empty: i32,
    lost: i32,
    foe: i32,
    foe_tokens: i32,
    foe_max: i32,
    foe_declined: i32,
    mine: i32,
    army: i32,
    bonus: i32,
    cheap: i32,
    mountain: i32,
    far_empty: i32,
    far_bonus: i32,
    far_foe_tokens: i32,
    far_army: i32,
}

impl Game<'_> {
    /// `kinds` et `extra` : par région, ce qu'elle est pour le joueur et ce qu'elle rapporte en plus.
    fn surroundings(&self, region: usize, kinds: &[Kind], extra: &[i32]) -> Surroundings {
        let mut s = Surroundings::default();
        let hood = &self.map.hoods[region];
        for a in hood.near.iter() {
            let high = self.map.regions[a].terrain == Terrain::Mountain;
            if kinds[a] != Kind::Army {
                s.bonus += extra[a];
            }
            if high {
                s.mountain += 1;
            }
            match kinds[a] {
                Kind::Empty => {
                    s.empty += 1;
                    if !high {
                        s.cheap += 1;
                    }
                }
                Kind::Lost => s.lost += 1,
                Kind::Mine => s.mine += 1,
                Kind::FoeDeclined => s.foe_declined += 1,
                Kind::Army => s.army += 1,
                Kind::FoeActive => {
                    let tokens = self.state.regions[a].tokens;
                    s.foe += 1;
                    s.foe_tokens += tokens;
                    s.foe_max = s.foe_max.max(tokens);
                }
            }
        }
        for b in hood.far.iter() {
            if kinds[b] == Kind::Army {
                s.far_army += 1;
            } else {
                s.far_bonus += extra[b];
            }
            match kinds[b] {
                Kind::Empty => s.far_empty += 1,
                Kind::FoeActive => s.far_foe_tokens += self.state.regions[b].tokens,
                _ => {}
            }
        }
        s
    }

    /// Les entrées propres à la région visée, communes à l'entrée et aux conquêtes.
    fn put_region(&self, x: &mut Row, t: &Target, hand: i32, kinds: &[Kind], extra: &[i32]) {
        let def = &self.map.regions[t.region];
        let k = kinds[t.region];
        x.put("cost", t.cost as f64)
            .put("handAfter", (hand - t.cost) as f64)
            .put("lost", b(k == Kind::Lost))
            .put("foeActive", b(k == Kind::FoeActive))
            .put("foeDeclined", b(k == Kind::FoeDeclined))
            .put("defenders", self.state.regions[t.region].tokens as f64)
            .put("bonus", extra[t.region] as f64)
            .put("mountain", b(def.terrain == Terrain::Mountain))
            .put("border", b(def.border))
            .put("coastal", b(def.coastal))
            .put("cavern", b(def.cavern))
            .put("mine", b(def.mine))
            .put("magic", b(def.magic))
            .put("farmland", b(def.terrain == Terrain::Farmland))
            .put("forest", b(def.terrain == Terrain::Forest))
            .put("hill", b(def.terrain == Terrain::Hill))
            .put("swamp", b(def.terrain == Terrain::Swamp));
    }
}

// --- région d'entrée d'un peuple neuf -----------------------------------------------------------

impl Game<'_> {
    /// Une entrée se décide-t-elle ici ? (peuple actif sans région, aucune conquête ce tour)
    pub fn is_entry_choice(&self) -> bool {
        let s = &self.state;
        let Some(army) = s.turn.army else { return false };
        s.phase == crate::ids::Phase::Conquest
            && s.turn.stage == crate::state::Stage::Active
            && s.turn.conquests == 0
            && !s.race(army).declined
            && self.owned(army).is_empty()
    }

    /// Les entrées du modèle pour chaque région d'entrée possible (`options` : les conquêtes abordables).
    pub fn entry_inputs(&self, options: &[Target]) -> Vec<Inputs> {
        let s = &self.state;
        let pid = s.current;
        let army = s.turn.army.expect("peuple en action");
        let rs = s.race(army);
        let n = self.map.len();
        let kinds: Vec<Kind> = (0..n).map(|i| self.kind(i, pid, None)).collect();
        let extra: Vec<i32> = (0..n).map(|i| bonus(self, i, army, rs.power)).collect();
        let my_declined = self.declined_regions(pid, None);
        options
            .iter()
            .map(|t| {
                let hood = &self.map.hoods[t.region];
                let near = self.surroundings(t.region, &kinds, &extra);
                let mut x = Row::new(entry_layout());
                x.put("players", s.player_count() as f64)
                    .put("left", (s.max_rounds - s.round) as f64)
                    .put("hand", rs.hand as f64)
                    .put("choices", options.len() as f64)
                    .put("myDeclined", my_declined as f64);
                self.put_region(&mut x, t, rs.hand, &kinds, &extra);
                x.put("near", hood.near.len() as f64)
                    .put("nearEmpty", near.empty as f64)
                    .put("nearLost", near.lost as f64)
                    .put("nearFoe", near.foe as f64)
                    .put("nearFoeTokens", near.foe_tokens as f64)
                    .put("nearFoeMax", near.foe_max as f64)
                    .put("nearFoeDeclined", near.foe_declined as f64)
                    .put("nearMine", near.mine as f64)
                    .put("nearBonus", near.bonus as f64)
                    .put("nearCheap", near.cheap as f64)
                    .put("nearMountain", near.mountain as f64)
                    .put("nearWater", hood.water as f64)
                    .put("far", hood.far.len() as f64)
                    .put("farEmpty", near.far_empty as f64)
                    .put("farBonus", near.far_bonus as f64)
                    .put("farFoeTokens", near.far_foe_tokens as f64)
                    .put("threat", self.rounded_threat(army, t.region))
                    .put("centre", self.map.regions[t.region].centre)
                    .races(army)
                    .powers(rs.power);
                x.finish()
            })
            .collect()
    }
}

// --- conquêtes ----------------------------------------------------------------------------------

impl Game<'_> {
    /// Les entrées du modèle pour chaque conquête de `options`, puis une dernière ligne pour « arrêter là ».
    pub fn conquest_inputs(&self, options: &[Target]) -> Vec<Inputs> {
        let s = &self.state;
        let pid = s.current;
        let army = s.turn.army.expect("peuple en action");
        let rs = s.race(army);
        let n = self.map.len();
        let kinds: Vec<Kind> = (0..n).map(|i| self.kind(i, pid, Some(army))).collect();
        let extra: Vec<i32> = (0..n)
            .map(|i| match rs.declined {
                true => i32::from(army == Race::Dwarves && self.map.regions[i].mine),
                false => bonus(self, i, army, rs.power),
            })
            .collect();
        let mine = self.owned(army);
        let on_board = self.tokens_on_board(army);
        let cheapest = options.iter().map(|t| t.cost).fold(99, i32::min);
        // ce que chaque peuple actif adverse peut engager : ses jetons au-delà d'un par région
        let mut attack = [(0, 0); Race::COUNT];
        let mut foes_attack = 0;
        for q in (0..s.player_count()).filter(|&q| q != pid) {
            let Some(race) = s.players[q].active.filter(|&r| !s.race(r).declined) else { continue };
            attack[race.index()] = self.spare_tokens(race);
            foes_attack = foes_attack.max(attack[race.index()].0);
        }
        let exposed =
            mine.iter().filter(|&r| self.map.hoods[r].near.iter().any(|a| kinds[a] == Kind::FoeActive)).count();
        let loot = i32::from(!rs.declined && (army == Race::Orcs || rs.power == Power::Pillaging));
        let mut costs: Vec<i32> = options.iter().map(|t| t.cost).collect();
        costs.sort_unstable();
        let first_cost = costs.first().copied().unwrap_or(99);
        let my_declined = self.declined_regions(pid, Some(army));
        let income = self.income(pid);

        let shared = |x: &mut Row, stop: bool| {
            x.put("players", s.player_count() as f64)
                .put("left", (s.max_rounds - s.round) as f64)
                .put("hand", rs.hand as f64)
                .put("choices", options.len() as f64)
                .put("myDeclined", my_declined as f64)
                .put("own", mine.len() as f64)
                .put("conquests", s.turn.conquests as f64)
                .put("onBoard", on_board as f64)
                .put("income", income as f64)
                .put("cheapest", cheapest as f64)
                .put("declined", b(rs.declined))
                .put("stop", b(stop));
        };
        let mut rows: Vec<Inputs> = options
            .iter()
            .map(|t| {
                let def = &self.map.regions[t.region];
                let hood = &self.map.hoods[t.region];
                let k = kinds[t.region];
                let near = self.surroundings(t.region, &kinds, &extra);
                let mut x = Row::new(conquest_layout());
                shared(&mut x, false);
                self.put_region(&mut x, t, rs.hand, &kinds, &extra);
                x.put("near", hood.near.len() as f64)
                    .put("nearEmpty", near.empty as f64)
                    .put("nearLost", near.lost as f64)
                    .put("nearFoe", near.foe as f64)
                    .put("nearFoeTokens", near.foe_tokens as f64)
                    .put("nearFoeMax", near.foe_max as f64)
                    .put("nearFoeDeclined", near.foe_declined as f64)
                    .put("nearArmy", near.army as f64)
                    .put("nearMine", near.mine as f64)
                    .put("nearBonus", near.bonus as f64)
                    .put("nearCheap", near.cheap as f64)
                    .put("nearMountain", near.mountain as f64)
                    .put("nearWater", hood.water as f64)
                    .put("far", hood.far.len() as f64)
                    .put("farEmpty", near.far_empty as f64)
                    .put("farBonus", near.far_bonus as f64)
                    .put("farFoeTokens", near.far_foe_tokens as f64)
                    .put("farArmy", near.far_army as f64)
                    .put("threat", self.rounded_threat(army, t.region))
                    .put("centre", def.centre)
                    .races(army)
                    .powers(rs.power);
                // la suite : ce que la conquête rapporte tout de suite, ce qui reste abordable après, et qui l'on frappe
                let next = if first_cost == t.cost { costs.get(1).copied().unwrap_or(99) } else { first_cost };
                let owner = match (k, s.regions[t.region].occupant) {
                    (Kind::FoeActive, Occupant::Race(r)) => attack[r.index()],
                    _ => (0, 0),
                };
                let die = t.kind == TargetKind::Die;
                x.put("spare", (on_board - mine.len() as i32) as f64)
                    .put("exposed", exposed as f64)
                    .put("foesAttack", foes_attack as f64)
                    .put("gain", (1 + extra[t.region] + if k != Kind::Empty { loot } else { 0 }) as f64)
                    .put("nextCheapest", next as f64)
                    .put("after", (rs.hand - t.cost - next) as f64)
                    .put("ownerAttack", owner.0 as f64)
                    .put("ownerRegions", owner.1 as f64)
                    .put("chance", if die { t.chance } else { 1.0 })
                    .put("need", if die { t.need as f64 } else { 0.0 });
                x.finish()
            })
            .collect();

        let mut x = Row::new(conquest_layout());
        shared(&mut x, true);
        // rien à prendre : les entrées de la région visée restent à 0
        x.put("cost", 0.0).put("handAfter", rs.hand as f64).skip(REGION_TAIL).races(army).powers(rs.power);
        x.put("spare", (on_board - mine.len() as i32) as f64)
            .put("exposed", exposed as f64)
            .put("foesAttack", foes_attack as f64)
            .put("gain", 0.0)
            .put("nextCheapest", first_cost as f64)
            .put("after", (rs.hand - first_cost) as f64)
            .put("ownerAttack", 0.0)
            .put("ownerRegions", 0.0)
            .put("chance", 0.0)
            .put("need", 0.0);
        rows.push(x.finish());
        rows
    }
}

/// Entrées de la région visée après `cost` et `handAfter` (de `lost` à `centre`).
const REGION_TAIL: usize = 35;

// --- déclin ------------------------------------------------------------------------------------

impl Game<'_> {
    /// Le déclin se décide-t-il ici ? (début de tour, déclin permis, pas le dernier tour, pas les Robustes)
    pub fn is_decline_choice(&self) -> bool {
        let s = &self.state;
        if self.can_decline().is_err() || s.turn.conquests > 0 || s.max_rounds - s.round <= 0 {
            return false;
        }
        // les Robustes déclinent en fin de tour : autre décision
        s.race(s.turn.army.expect("peuple en action")).power != Power::Stout
    }

    /// `value` : la valeur d'une combinaison du marché selon la table de l'IA.
    pub fn decline_inputs(&self, value: impl Fn(&Game, Combo, usize) -> f64) -> Inputs {
        let s = &self.state;
        let pid = s.current;
        let p = &s.players[pid];
        let army = s.turn.army.expect("peuple en action");
        let rs = s.race(army);
        let own = self.owned(army);
        let tokens = self.tokens_on_board(army) + rs.hand;
        let affordable = || s.market.iter().enumerate().filter(|&(i, _)| i as i32 <= p.coins);
        let lost = self.declined_regions(pid, None);
        let market = affordable()
            .map(|(i, &slot)| value(self, slot, i))
            .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
        let market = market.unwrap_or(-20.0);
        let mine = value(self, Combo { race: army, power: rs.power, coins: 0 }, 0);
        let income = self.income(pid);
        let mut opp_attack = 0;
        for q in (0..s.player_count()).filter(|&q| q != pid) {
            if let Some(race) = s.players[q].active.filter(|&r| !s.race(r).declined) {
                opp_attack = opp_attack.max(self.spare_tokens(race).0);
            }
        }
        let fresh = affordable().map(|(_, c)| c.race.starting_tokens(c.power)).fold(0, i32::max);
        let danger = if own.is_empty() {
            0.0
        } else {
            let total = own.iter().fold(0.0, |sum, r| sum + self.threat(army, r));
            round_half_up(10.0 * total / own.len() as f64) / 10.0
        };
        let mut x = Row::new(decline_layout());
        x.put("players", s.player_count() as f64)
            .put("left", (s.max_rounds - s.round) as f64)
            .put("own", own.len() as f64)
            .put("est", self.estimate_conquests() as f64)
            .put("tokens", tokens as f64)
            .put("hand", rs.hand as f64)
            .put("free", (tokens - own.len() as i32) as f64)
            .put("lost", lost as f64)
            .put("income", income as f64)
            .put("fresh", fresh as f64)
            .put("age", (s.round - rs.picked_round) as f64)
            .put("market", market)
            .put("mine", mine)
            .put("gap", market - mine)
            .put("ghouls", b(army == Race::Ghouls))
            .put("spirit", b(rs.power == Power::Spirit))
            .put("incomeActive", (income - lost) as f64)
            .put("oppAttack", opp_attack as f64)
            .put("danger", danger)
            .races(army)
            .powers(rs.power);
        x.finish()
    }
}

// --- garder ses régions -------------------------------------------------------------------------

/// Position des entrées « jetons », « défense » et « valeur » dans le modèle de garde.
pub const HOLD_TOKENS: usize = 0;
pub const HOLD_DEFENSE: usize = 1;
pub const HOLD_WORTH: usize = 23;

impl Game<'_> {
    /// Les entrées du modèle pour une région de `race`, sur un état nivelé (`levelled`).
    pub fn hold_inputs(&self, race: Race, region: usize) -> Inputs {
        let s = &self.state;
        let rs = s.race(race);
        let pid = rs.owner;
        let def = &self.map.regions[region];
        let reg = &s.regions[region];
        let base = self.defense_value(region) - reg.tokens;
        let (mut attackers, mut fresh, mut far, mut sorcery, mut dragon) = (0, 0, 0, 0, 0);
        let (mut cheapest, mut free_max, mut free_sum) = (99, 0, 0);
        for a in self.attackers(region, pid) {
            let Some(foe) = a.race else {
                fresh += 1;
                continue;
            };
            attackers += 1;
            if a.far() {
                far = 1;
            }
            match a.how {
                crate::rules::AttackHow::Sorcery => sorcery = 1,
                crate::rules::AttackHow::Dragon => dragon = 1,
                // le coût de l'attaquant, hors jetons du défenseur : ce que ses pouvoirs et le terrain changent
                crate::rules::AttackHow::Conquer => cheapest = cheapest.min(a.cost - reg.tokens),
            }
            let free = self.spare_tokens(foe).0;
            free_max = free_max.max(free);
            free_sum += free;
        }
        let (mut foe_tokens, mut foe_max, mut own_near, mut empty_near, mut foe_declined) = (0, 0, 0, 0, 0);
        for &n in &def.adj {
            let n = n as usize;
            if self.map.regions[n].is_water() {
                continue;
            }
            let nr = &s.regions[n];
            match nr.occupant {
                Occupant::Empty => empty_near += 1,
                Occupant::Race(r) if r == race => own_near += 1,
                Occupant::Lost => {}
                Occupant::Race(r) => {
                    let other = s.race(r);
                    if other.owner == pid {
                        continue;
                    }
                    if other.declined {
                        foe_declined += 1;
                    } else {
                        foe_tokens += nr.tokens;
                        foe_max = foe_max.max(nr.tokens);
                    }
                }
            }
        }
        let mountain = i32::from(def.terrain == Terrain::Mountain);
        let mut x = Row::new(hold_layout());
        x.put("tokens", reg.tokens as f64)
            .put("defense", (base + reg.tokens) as f64)
            .put("players", s.player_count() as f64)
            .put("left", (s.max_rounds - s.round) as f64)
            .put("own", self.owned(race).len() as f64)
            .put("army", (self.tokens_on_board(race) + rs.hand) as f64)
            .put("mountain", mountain as f64)
            .put("entry", b(def.entry))
            .put("works", (base - 2 - mountain) as f64)
            .put("attackers", attackers as f64)
            .put("fresh", fresh as f64)
            .put("far", far as f64)
            .put("sorcery", sorcery as f64)
            .put("dragon", dragon as f64)
            .put("cheapest", if cheapest == 99 { base } else { cheapest } as f64)
            .put("freeMax", free_max as f64)
            .put("freeSum", free_sum as f64)
            .put("nearFoeTokens", foe_tokens as f64)
            .put("nearFoeMax", foe_max as f64)
            .put("nearOwn", own_near as f64)
            .put("nearEmpty", empty_near as f64)
            .put("nearFoeDeclined", foe_declined as f64)
            .put("threat", self.rounded_threat(race, region))
            .put("worth", (1 + bonus(self, region, race, rs.power)) as f64)
            .put("border", b(def.border))
            .put("coastal", b(def.coastal));
        x.finish()
    }
}
