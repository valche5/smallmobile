//! Placement automatique des jetons : la menace qui pèse sur chaque région, et les plans qui en
//! découlent. Il fait partie des règles (action `AutoDeploy`, défense automatique) et sert à l'IA.

use crate::action::Placement;
use crate::game::Game;
use crate::ids::{MAX_REGIONS, Race};
use crate::rules::AttackHow;
use crate::state::Occupant;

/// Jetons ajoutés par région.
pub type Plan = [i32; MAX_REGIONS];

impl<'m> Game<'m> {
    /// Danger qui pèse sur une région du peuple (plus c'est haut, plus il faut la défendre).
    pub fn threat(&self, race: Race, region: usize) -> f64 {
        let s = &self.state;
        let r = &s.regions[region];
        if r.is_immune() {
            return 0.0;
        }
        let owner = s.race(race).owner;
        let def = &self.map.regions[region];
        let mut t = if def.entry { 0.6 } else { 0.0 };
        for &n in &def.adj {
            let n = n as usize;
            match s.regions[n].occupant {
                Occupant::Race(other) if other == race => {}
                Occupant::Empty => {
                    if !self.map.regions[n].is_water() {
                        t += 0.3;
                    }
                }
                Occupant::Lost => t += 0.1,
                Occupant::Race(other) => {
                    if s.race(other).owner == owner {
                        t += 0.05;
                    }
                }
            }
        }
        // les adversaires qui peuvent vraiment l'atteindre, pouvoirs compris (Volants, cavernes, dragon, sorcellerie…)
        for a in self.attackers(region, owner) {
            // un peuple encore à choisir entre par le bord : le terme « bord de carte » le compte déjà
            let Some(foe) = a.race else { continue };
            if a.far() {
                t += 1.6;
            } else {
                let near: i32 = def
                    .adj
                    .iter()
                    .map(|&n| &s.regions[n as usize])
                    .filter(|nr| nr.occupant == Occupant::Race(foe))
                    .map(|nr| nr.tokens)
                    .sum();
                t += 1.0 + 0.3 * near as f64;
            }
            if self.attack_modifier(foe, region) != 0 {
                t += 0.5;
            }
            match a.how {
                AttackHow::Sorcery => t += 1.5,
                AttackHow::Dragon => t += 1.0,
                AttackHow::Conquer => {}
            }
        }
        t
    }

    /// Répartit les jetons en main sur les régions les plus menacées ; renvoie les ajouts par région.
    /// `flat` suit la racine de la menace et étale davantage : une défense fine partout dissuade mieux
    /// qu'une pile (mesuré par simulation).
    pub fn plan_deployment(&self, race: Race, flat: bool) -> Plan {
        let mut plan = [0; MAX_REGIONS];
        let own = self.owned(race);
        let hand = self.state.race(race).hand;
        let Some(first) = own.first() else { return plan };
        if hand <= 0 {
            return plan;
        }
        let mut threats = [0.0; MAX_REGIONS];
        let mut defense = [0; MAX_REGIONS];
        for r in own.iter() {
            let t = self.threat(race, r);
            threats[r] = if flat { t.sqrt() } else { t };
            defense[r] = self.defense_value(r);
        }
        for _ in 0..hand {
            let mut best = first;
            let mut best_score = f64::NEG_INFINITY;
            for r in own.iter() {
                let score = (threats[r] + 0.05) / defense[r] as f64;
                if score > best_score {
                    best_score = score;
                    best = r;
                }
            }
            plan[best] += 1;
            defense[best] += 1;
        }
        plan
    }

    /// La même partie, tous les jetons du peuple repris en main sauf un par région.
    pub fn levelled(&self, race: Race) -> Game<'m> {
        let mut g = *self;
        let mut hand = 0;
        for r in self.owned(race).iter() {
            hand += g.state.regions[r].tokens - 1;
            g.state.regions[r].tokens = 1;
        }
        g.state.race_mut(race).hand += hand;
        g
    }

    /// Placement complet de fin de tour : tous les jetons repris (un seul reste par région), puis
    /// répartis à plat. Renvoie le nombre de jetons visé par région.
    pub fn full_deployment(&self, race: Race) -> Placement {
        let g = self.levelled(race);
        Self::placement_from(&g, race, &g.plan_deployment(race, true))
    }

    fn placement_from(g: &Game, race: Race, plan: &Plan) -> Placement {
        let mut out = Placement::default();
        for r in g.owned(race).iter() {
            out.0[r] = 1 + plan[r];
        }
        out
    }

    /// Plusieurs placements complets plausibles, du plus fidèle à la menace au plus étalé ; le premier
    /// est celui de `full_deployment`. Le niveau difficile les départage par simulation.
    pub fn placement_candidates(&self, race: Race) -> [Placement; 6] {
        let g = self.levelled(race);
        let own = g.owned(race);
        let hand = g.state.race(race).hand;
        let owner = g.state.race(race).owner;
        let exposed: crate::collections::RegionSet = own
            .iter()
            .filter(|&r| !g.state.regions[r].is_immune() && g.attackers(r, owner).iter().any(|a| a.race.is_some()))
            .collect();
        // pose les jetons un par un, chaque fois sur la région au meilleur score
        let spread = |score: &dyn Fn(usize, i32) -> f64| {
            let mut added = [0; MAX_REGIONS];
            for _ in 0..hand {
                let mut best = own.first().unwrap_or(0);
                let mut best_score = f64::NEG_INFINITY;
                for r in own.iter() {
                    let sc = score(r, added[r]);
                    if sc > best_score {
                        best_score = sc;
                        best = r;
                    }
                }
                added[best] += 1;
            }
            Self::placement_from(&g, race, &added)
        };
        let immune = |r: usize| g.state.regions[r].is_immune();
        [
            Self::placement_from(&g, race, &g.plan_deployment(race, true)),
            spread(&|_, added| -added as f64),
            spread(&|r, added| {
                (if !exposed.is_empty() && !exposed.contains(r) { -1000.0 } else { 0.0 }) - added as f64
            }),
            spread(&|r, added| (if exposed.contains(r) { 2.0 } else { 1.0 }) / (1.0 + added as f64)),
            Self::placement_from(&g, race, &g.plan_deployment(race, false)),
            spread(&|r, added| (if immune(r) { -1000.0 } else { 0.0 }) - g.defense_value(r) as f64 - added as f64),
        ]
    }
}
