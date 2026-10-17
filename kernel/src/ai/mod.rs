//! L'IA du jeu, à trois niveaux :
//! - facile : table de valeurs des peuples, règle de déclin à seuils, rendement immédiat des conquêtes ;
//! - normal : modèles appris (choix du peuple, déclin, entrée, conquêtes, placement de fin de tour) ;
//! - difficile : choix du peuple, déclin, entrée et placement en jouant la suite de la partie.
//!
//! Le hasard de l'IA ne dépend que de l'état : une même position donne toujours la même décision.

pub mod boost;
pub mod features;
pub mod tuning;

use crate::action::{Action, NoJournal, Placement};
use crate::game::Game;
use crate::ids::{Level, MAX_PLAYERS, MAX_REGIONS, Phase, Power, Race, Terrain};
use crate::reason::Reason;
use crate::rng::{self, Key};
use crate::rules::{Target, TargetKind};
use crate::state::{Combo, Occupant, Stage};
use boost::Models;
use features::{HOLD_DEFENSE, HOLD_TOKENS, HOLD_WORTH};
pub use tuning::Tuning;

/// `Math.round` de JavaScript : au plus proche, les demis vers le haut.
pub fn round_half_up(x: f64) -> f64 {
    let floor = x.floor();
    if x - floor >= 0.5 { floor + 1.0 } else { floor }
}

/// Ce que vaut chaque peuple et chaque pouvoir, en pièces finales par rapport à la moyenne, selon le
/// moment : premier peuple, peuple suivant, fin de partie (4 tours ou moins). Mesuré par simulation.
const RACE_VALUE: [[f64; 3]; Race::COUNT] = [
    [5.3, 10.1, 4.0],
    [-2.3, -5.0, -3.0],
    [-1.5, -0.9, -1.4],
    [4.3, -0.8, -3.2],
    [0.1, -1.0, -0.7],
    [-1.0, -1.5, -1.3],
    [-2.3, -2.4, 1.4],
    [-3.9, -1.8, 2.1],
    [2.2, 2.8, 1.5],
    [-1.6, 2.9, -0.1],
    [-0.6, 0.0, 1.4],
    [3.5, 1.7, 1.2],
    [0.0, -1.8, -3.0],
    [-2.4, -2.4, 1.2],
];

const POWER_VALUE: [[f64; 3]; Power::COUNT] = [
    [-2.1, 0.0, 1.8],
    [2.0, -1.5, -1.1],
    [-0.5, -1.5, -2.2],
    [5.2, 1.2, 0.6],
    [-0.2, -0.9, -2.2],
    [4.5, 5.0, 0.5],
    [-1.1, -0.2, -0.4],
    [-2.0, -1.7, 0.4],
    [-3.5, -0.9, 0.3],
    [-0.2, -0.2, -1.9],
    [-2.0, -1.8, 0.5],
    [-5.6, -2.2, 4.0],
    [2.8, 1.3, -0.1],
    [-0.5, 2.9, 3.8],
    [0.2, 0.0, -1.5],
    [3.3, 3.2, -2.4],
    [-0.7, -2.0, -3.1],
    [-1.9, -1.6, 0.5],
    [1.6, 1.2, -0.2],
    [0.8, -0.5, 2.7],
];

/// Amplitude du hasard ajouté aux décisions.
fn noise_amplitude(level: Level) -> f64 {
    match level {
        Level::Easy => 2.5,
        Level::Normal => 0.4,
        Level::Hard => 0.0,
    }
}

/// Marge d'erreur sur les pièces des autres : elles sont cachées, l'IA ne compte que les gains annoncés.
fn coin_blur(level: Level) -> f64 {
    match level {
        Level::Easy => 0.35,
        Level::Normal => 0.2,
        Level::Hard => 0.1,
    }
}

type ArrayTargets = crate::collections::ArrayVec<Target, MAX_REGIONS>;

/// Ce que l'IA ne doit pas faire.
#[derive(Clone, Copy, Debug, Default)]
pub struct Restrictions {
    pub no_decline: bool,
    pub keep_placement: bool,
}

/// Les bornes des simulations : au-delà, la suite de partie est abandonnée.
const PLAY_TO_END_STEPS: usize = 20_000;
const ROLLOUT_STEPS: usize = 4_000;
const PLACEMENT_STEPS: usize = 2_000;

#[derive(Clone, Copy)]
pub struct Ai<'t> {
    pub tuning: &'t Tuning,
    models: &'t Models,
    /// Vrai pendant qu'une suite de partie est jouée pour le niveau difficile : certains modèles y sont
    /// coupés (ils n'y gagnent rien et ralentiraient tout).
    simulating: bool,
}

/// Une partie vue par l'IA qui décide pour le joueur `pid` au niveau `level`.
struct Seat<'a, 'm, 't> {
    ai: &'a Ai<'t>,
    g: &'a Game<'m>,
    pid: usize,
    level: Level,
}

impl<'t> Ai<'t> {
    /// L'IA du jeu, avec ces réglages.
    pub fn new(tuning: &'t Tuning) -> Self {
        Ai { tuning, models: Models::get(), simulating: false }
    }

    fn simulation(&self) -> Self {
        Ai { simulating: true, ..*self }
    }

    /// La prochaine action du joueur qui doit agir, à son niveau.
    pub fn decide(&self, g: &Game, restrictions: Restrictions) -> Result<Action, Reason> {
        let pid = g.state.actor().ok_or(Reason::GameOver)?;
        self.decide_as(g, g.state.players[pid].level, restrictions)
    }

    fn decide_as(&self, g: &Game, level: Level, restrictions: Restrictions) -> Result<Action, Reason> {
        let pid = g.state.actor().ok_or(Reason::GameOver)?;
        let seat = Seat { ai: self, g, pid, level };
        Ok(match g.state.phase {
            Phase::Pick if g.state.market.is_empty() => Action::SkipPick,
            Phase::Pick => Action::Pick { index: seat.choose_combo()? },
            Phase::Conquest => seat.conquest_step(restrictions.no_decline)?,
            Phase::Redeploy => seat.redeploy_step(restrictions.keep_placement)?,
            Phase::Defend => {
                let race = g.state.defense[0].race;
                let pending = g.state.races[race.index()].is_some_and(|rs| rs.hand > 0) && !g.owned(race).is_empty();
                if pending { Action::AutoDeploy { all: false } } else { Action::EndDefense }
            }
            Phase::GameOver => return Err(Reason::GameOver),
        })
    }

    /// Joue la partie jusqu'au bout, chaque joueur à son niveau ; renvoie le nombre d'actions jouées.
    pub fn play(&self, g: &mut Game, max_steps: usize) -> Result<usize, Reason> {
        for step in 0..max_steps {
            if g.state.phase == Phase::GameOver {
                return Ok(step);
            }
            let action = self.decide(g, Restrictions::default())?;
            g.apply(&action, &mut NoJournal)?;
        }
        Ok(max_steps)
    }

    /// Une action de la politique des simulations (niveau normal pour tout le monde).
    fn sim_step(&self, s: &mut Game) -> Result<(), Reason> {
        let action = self.decide_as(s, Level::Normal, Restrictions::default())?;
        s.apply(&action, &mut NoJournal)
    }

    /// Termine la partie avec la politique normale pour tout le monde ; renvoie l'avance finale de `pid`
    /// sur son meilleur adversaire. C'est la mesure la plus fiable de ce que vaut une décision.
    fn play_to_end(&self, g: &Game, pid: usize, seed: u32) -> Result<f64, Reason> {
        let sim = self.simulation();
        let mut s = *g;
        s.state.rng = seed;
        for _ in 0..PLAY_TO_END_STEPS {
            if s.state.phase == Phase::GameOver {
                break;
            }
            sim.sim_step(&mut s)?;
        }
        let players = &s.state.players;
        let rival = (0..players.len()).filter(|&q| q != pid).map(|q| players[q].coins).max().unwrap_or(i32::MIN);
        Ok((players[pid].coins - rival) as f64)
    }

    /// Joue `turns` tours de joueur avec la politique normale ; renvoie la valeur de la position de `pid`.
    fn rollout(&self, g: &Game, pid: usize, turns: usize, seed: u32) -> Result<f64, Reason> {
        let sim = self.simulation();
        let mut s = *g;
        s.state.rng = seed;
        let mut current = s.state.current;
        let mut played = 0;
        for _ in 0..ROLLOUT_STEPS {
            if s.state.phase == Phase::GameOver || played >= turns {
                break;
            }
            sim.sim_step(&mut s)?;
            if s.state.current != current {
                current = s.state.current;
                played += 1;
            }
        }
        Ok(position_value(&s, pid))
    }
}

/// Pièces plus valeur estimée de la position (revenu futur et potentiel de conquête).
fn position_value(s: &Game, pid: usize) -> f64 {
    let st = &s.state;
    let p = &st.players[pid];
    let left = if st.phase == Phase::GameOver { 0 } else { st.max_rounds - st.round + i32::from(st.current <= pid) };
    let mut regions: i32 = p.declined.iter().map(|&r| s.owned(r).len() as i32).sum();
    let mut spare = 0;
    if let Some(active) = p.active {
        let own = s.owned(active).len() as i32;
        regions += own;
        spare = (s.tokens_on_board(active) + st.race(active).hand - own).max(0);
    }
    p.coins as f64 + 0.75 * regions as f64 * left as f64 + 0.35 * spare as f64 * left.min(3) as f64
}

/// Graines des suites de partie jouées pour une décision.
fn seeds(g: &Game, count: usize) -> impl Iterator<Item = u32> {
    let (seq, state) = (g.state.seq as u64, g.state.rng as u64);
    (1..=count as u64).map(move |k| Key::new().num(seq).sep().num(state).sep().num(k).finish())
}

fn best_by<T: Copy>(items: impl IntoIterator<Item = T>, mut score: impl FnMut(T) -> f64) -> Option<T> {
    let mut best = None;
    let mut best_score = f64::NEG_INFINITY;
    for it in items {
        let s = score(it);
        if s > best_score {
            best_score = s;
            best = Some(it);
        }
    }
    best
}

/// Intérêt d'une combinaison du marché, en pièces : sa valeur propre, plus les pièces posées dessus,
/// moins son prix.
pub fn combo_value(g: &Game, slot: Combo, index: usize) -> f64 {
    let s = &g.state;
    let phase = if s.max_rounds - s.round + 1 > 4 {
        let p = &s.players[s.current];
        usize::from(p.declined.iter().any(|&r| !g.owned(r).is_empty()))
    } else {
        2
    };
    RACE_VALUE[slot.race.index()][phase]
        + POWER_VALUE[slot.power.index()][phase]
        + board_bonus(g, slot.race)
        + slot.coins as f64
        - index as f64
}

/// Ce que le plateau ajoute ou retire à un peuple dont la mécanique en dépend : des proies pour la
/// sorcellerie, des mines, des côtes ou des montagnes à prendre.
fn board_bonus(g: &Game, race: Race) -> f64 {
    let s = &g.state;
    if race == Race::Trolls {
        return if s.player_count() >= 3 { 0.8 } else { -1.0 };
    }
    if !matches!(race, Race::Sorcerers | Race::Dwarves | Race::Tritons | Race::Giants) {
        return 0.0;
    }
    let mut count = 0;
    for (i, def) in g.map.regions.iter().enumerate() {
        let r = &s.regions[i];
        if def.is_water() {
            continue;
        }
        let hit = match race {
            // une proie : un jeton isolé d'un peuple actif adverse
            Race::Sorcerers => {
                r.tokens == 1 && r.occupant.race().is_some_and(|o| !s.race(o).declined && s.race(o).owner != s.current)
            }
            Race::Dwarves => r.occupant.is_empty() && def.mine,
            Race::Tritons => r.occupant.is_empty() && def.coastal,
            _ => r.occupant.is_empty() && def.terrain == Terrain::Mountain,
        };
        if hit {
            count += 1;
        }
    }
    match race {
        Race::Sorcerers => {
            if count >= 1 {
                1.2
            } else {
                -2.0
            }
        }
        Race::Dwarves => {
            if count >= 1 {
                0.2
            } else {
                -0.5
            }
        }
        Race::Tritons => {
            if count >= 3 {
                0.2
            } else {
                -0.2
            }
        }
        _ => {
            if count >= 2 {
                0.1
            } else {
                -0.4
            }
        }
    }
}

impl Seat<'_, '_, '_> {
    fn tuning(&self) -> &Tuning {
        self.ai.tuning
    }

    fn noise(&self, salt: &str, index: Option<u64>) -> f64 {
        let amp = noise_amplitude(self.level);
        if amp == 0.0 {
            return 0.0;
        }
        let s = &self.g.state;
        let mut key = Key::new().num(s.seq as u64).sep().num(s.rng as u64).sep().text(salt);
        if let Some(i) = index {
            key = key.num(i);
        }
        (rng::draw(key.finish()) * 2.0 - 1.0) * amp
    }

    // --- choix du peuple ---------------------------------------------------------------------

    fn choose_combo(&self) -> Result<usize, Reason> {
        let s = &self.g.state;
        let t = self.tuning();
        let open: Vec<usize> = (0..s.market.len()).filter(|&i| i as i32 <= s.players[self.pid].coins).collect();
        // niveaux normal et difficile : le modèle appris donne la valeur de chaque combinaison dans ce contexte
        let learnt = self.level != Level::Easy && t.pick_model != 0.0;
        let base: Vec<f64> = if learnt {
            let model = &self.ai.models.pick;
            let context = self.g.pick_context();
            let rows: Vec<_> = open.iter().map(|&i| self.g.pick_inputs(&context, s.market[i])).collect();
            model.predict_all(rows.iter().map(|x| x.get()))
        } else {
            open.iter().map(|&i| combo_value(self.g, s.market[i], i)).collect()
        };
        let mut values: Vec<f64> = open
            .iter()
            .zip(base)
            .map(|(&i, value)| {
                let value = if learnt { value + s.market[i].coins as f64 - i as f64 } else { value };
                value + self.noise("pick", Some(i as u64))
            })
            .collect();
        if self.level == Level::Hard {
            // chaque combinaison est essayée, et la partie jouée jusqu'au bout plusieurs fois ; le modèle sert d'a priori
            let draws: Vec<u32> = seeds(self.g, t.pick_draws as usize).collect();
            let mut played = Vec::with_capacity(open.len());
            for &i in &open {
                let mut picked = *self.g;
                picked.apply(&Action::Pick { index: i }, &mut NoJournal)?;
                let mut sum = 0.0;
                for &seed in &draws {
                    sum += self.ai.play_to_end(&picked, self.pid, seed)?;
                }
                played.push(sum / draws.len() as f64);
            }
            let mean = |xs: &[f64]| xs.iter().fold(0.0, |a, b| a + b) / xs.len() as f64;
            let (m0, m1) = (mean(&played), mean(&values));
            let n = draws.len() as f64;
            for (k, v) in values.iter_mut().enumerate() {
                *v = (n * (played[k] - m0) + t.pick_prior * (*v - m1)) / (n + t.pick_prior);
            }
        }
        Ok(best_by(0..open.len(), |k| values[k]).map_or(0, |k| open[k]))
    }

    // --- adversaires -------------------------------------------------------------------------

    /// Pièces attribuées à `target` : exactes pour soi, à peu près pour les autres (même estimation tout le tour).
    fn guess_coins(&self, target: usize) -> f64 {
        let s = &self.g.state;
        let coins = s.players[target].coins as f64;
        if target == self.pid {
            return coins;
        }
        let key = Key::new().num(self.g.map.seed).sep().num(s.round as u64).sep().num(self.pid as u64).sep();
        let err = rng::draw(key.num(target as u64).finish()) * 2.0 - 1.0;
        coins * (1.0 + err * coin_blur(self.level))
    }

    /// Par joueur, l'intérêt de lui reprendre une région : 0 sauf s'il domine clairement la partie, et
    /// d'autant plus qu'il mène. Les pièces sont comparées à nombre de tours égal : ceux qui ont déjà
    /// joué cette manche ne comptent pas leur dernier gain. Au niveau difficile s'ajoute un bonus contre
    /// tout adversaire, d'autant plus fort qu'ils sont peu nombreux.
    fn dominance(&self) -> [f64; MAX_PLAYERS] {
        let s = &self.g.state;
        let t = self.tuning();
        let n = s.player_count();
        let mut standing = [0.0; MAX_PLAYERS];
        for (q, place) in standing.iter_mut().enumerate().take(n) {
            let behind = if q < s.current { s.players[q].last_income as f64 } else { 0.0 };
            *place = self.guess_coins(q) - behind;
        }
        let mut out = [0.0; MAX_PLAYERS];
        for q in (0..n).filter(|&q| q != self.pid) {
            let best_other = (0..n).filter(|&i| i != q).map(|i| standing[i]).fold(f64::NEG_INFINITY, f64::max);
            let lead = standing[q] - best_other;
            let clear = t.dom_min.max(t.dom_share * standing[q]);
            let rivalry = if self.level == Level::Hard { t.rivalry / (n - 1) as f64 } else { 0.0 };
            let domination =
                if lead < clear { 0.0 } else { t.dom_max.min(t.dom_base + (t.dom_base * (lead - clear)) / clear) };
            out[q] = rivalry + domination;
        }
        out
    }

    /// Intérêt d'une région pour le peuple qui attaque (≈ pièces par tour, plus le frein mis à un
    /// joueur qui domine).
    fn region_value(&self, army: Race, region: usize, dom: &[f64; MAX_PLAYERS]) -> f64 {
        let s = &self.g.state;
        let t = self.tuning();
        let def = &self.g.map.regions[region];
        let r = &s.regions[region];
        let rs = s.race(army);
        let active = !rs.declined;
        let mut v = 1.0;
        if army == Race::Dwarves && def.mine {
            v += 1.0;
        }
        if active {
            if army == Race::Humans && def.terrain == Terrain::Farmland {
                v += 1.0;
            }
            if army == Race::Wizards && def.magic {
                v += 1.0;
            }
            if (rs.power == Power::Forest && def.terrain == Terrain::Forest)
                || (rs.power == Power::Hill && def.terrain == Terrain::Hill)
                || (rs.power == Power::Swamp && def.terrain == Terrain::Swamp)
            {
                v += 1.0;
            }
            if rs.power == Power::Merchant {
                v += 1.0;
            }
            if army == Race::Giants && def.terrain == Terrain::Mountain {
                v += 0.3;
            }
            if rs.power == Power::Underworld && def.cavern {
                v += 0.3;
            }
        }
        if !r.occupant.is_empty() {
            if active && (army == Race::Orcs || rs.power == Power::Pillaging) {
                v += 1.0;
            }
            if active && army == Race::Skeletons {
                v += 0.5;
            }
            if let Some(foe_race) = r.occupant.race() {
                let foe = s.race(foe_race);
                v += if foe.owner == self.pid {
                    -1.0
                } else {
                    dom[foe.owner] + if foe.declined { t.foe_declined } else { t.foe }
                };
            }
        }
        let mut friends = 0;
        let mut enemies = 0;
        for &a in &def.adj {
            let n = &s.regions[a as usize];
            match n.occupant {
                Occupant::Race(o) if o == army => friends += 1,
                Occupant::Race(o) if t.exposure != 0.0 => {
                    let ns = s.race(o);
                    if ns.owner != self.pid && !ns.declined {
                        enemies += n.tokens;
                    }
                }
                _ => {}
            }
        }
        v + (t.adjacency * friends as f64 - t.exposure * enemies as f64)
    }

    /// Rendement d'une conquête : ce qu'elle rapporte par jeton engagé.
    fn target_score(&self, army: Race, target: &Target, dom: &[f64; MAX_PLAYERS]) -> f64 {
        let cost = (target.effective.max(1) as f64).powf(self.tuning().cost_power);
        self.region_value(army, target.region, dom) / cost + self.noise("t", Some(target.region as u64)) * 0.3
    }

    // --- déclin ------------------------------------------------------------------------------

    fn should_decline(&self) -> Result<bool, Reason> {
        let g = self.g;
        let s = &g.state;
        let t = self.tuning();
        if g.can_decline().is_err() {
            return Ok(false);
        }
        let rounds_left = s.max_rounds - s.round;
        if rounds_left <= 0 {
            return Ok(false);
        }
        // niveau normal : le modèle appris prédit ce que le déclin rapporte d'ici la fin
        if self.level == Level::Normal && t.decline_model != 0.0 && g.is_decline_choice() {
            let model = &self.ai.models.decline;
            return Ok(model.predict(g.decline_inputs(combo_value).get()) > 0.0);
        }
        let army = s.turn.army.expect("peuple en action");
        let own = g.owned(army).len() as i32;
        if own == 0 {
            return Ok(false);
        }
        let est = g.estimate_conquests();
        let p = &s.players[self.pid];
        let fresh = s
            .market
            .iter()
            .enumerate()
            .filter(|&(i, _)| i as i32 <= p.coins)
            .map(|(_, c)| c.race.starting_tokens(c.power))
            .fold(0, i32::max);
        match self.level {
            Level::Easy => Ok(est == 0 || (est <= 1 && self.noise("decl", None) > 1.5)),
            Level::Hard => {
                if fresh == 0 {
                    return Ok(false);
                }
                // « continuer » : jouer normalement ce tour sans décliner ; les deux suites sont jouées
                // jusqu'au bout de la partie, avec les mêmes dés
                let keep_action =
                    Seat { ai: &self.ai.simulation(), g, pid: self.pid, level: Level::Normal }.conquest_step(true)?;
                let mut keep = *g;
                keep.apply(&keep_action, &mut NoJournal)?;
                let mut decline = *g;
                decline.apply(&Action::Decline, &mut NoJournal)?;
                let mut gain = 0.0;
                for seed in seeds(g, t.decline_draws as usize) {
                    gain +=
                        self.ai.play_to_end(&decline, self.pid, seed)? - self.ai.play_to_end(&keep, self.pid, seed)?;
                }
                // le modèle appris sert d'a priori : il pèse autant que `decline_prior` fins de partie
                if t.decline_prior != 0.0 && g.is_decline_choice() {
                    let model = &self.ai.models.decline;
                    gain += t.decline_prior * model.predict(g.decline_inputs(combo_value).get());
                }
                Ok(gain > 0.0)
            }
            // Robustes en début de tour (hors modèle) : décliner à un tour de la fin est une perte, et pas
            // question de sacrifier plus de régions qu'on n'en garde
            Level::Normal => {
                let lost: i32 = p.declined.iter().map(|&r| g.owned(r).len() as i32).sum();
                Ok(est as f64 <= t.decline_est && rounds_left as f64 >= t.decline_left && lost < own)
            }
        }
    }

    // --- conquêtes ---------------------------------------------------------------------------

    /// La région d'entrée d'un peuple neuf. Normal : la meilleure selon le modèle appris. Difficile :
    /// les meilleures selon le modèle sont jouées jusqu'au bout plusieurs fois.
    fn choose_entry(&self, options: &[Target]) -> Result<usize, Reason> {
        if options.len() < 2 {
            return Ok(options[0].region);
        }
        let g = self.g;
        let model = &self.ai.models.entry;
        let xs = g.entry_inputs(options);
        let predicted = model.predict_all(xs.iter().map(|x| x.get()));
        let mut ranked: Vec<(usize, f64)> = options
            .iter()
            .zip(predicted)
            .map(|(t, p)| (t.region, p + self.noise("entry", Some(t.region as u64))))
            .collect();
        // tri stable, du meilleur au moins bon (comme `Array.sort` : les égalités gardent leur ordre)
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        let t = self.tuning();
        if self.level == Level::Hard && t.entry_draws > 0.0 {
            let draws: Vec<u32> = seeds(g, t.entry_draws as usize).collect();
            ranked.truncate(t.entry_tried as usize);
            for (region, value) in ranked.iter_mut() {
                let mut entered = *g;
                entered.apply(&Action::Conquer { region: *region }, &mut NoJournal)?;
                let mut sum = 0.0;
                for &seed in &draws {
                    sum += self.ai.play_to_end(&entered, self.pid, seed)?;
                }
                *value = sum / draws.len() as f64;
            }
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        }
        Ok(ranked[0].0)
    }

    fn conquest_step(&self, no_decline: bool) -> Result<Action, Reason> {
        let g = self.g;
        let s = &g.state;
        let t = self.tuning();
        let turn = &s.turn;
        let army = turn.army.expect("peuple en action");
        let rs = *s.race(army);
        if !no_decline && turn.stage == Stage::Active && self.should_decline()? {
            return Ok(Action::Decline);
        }
        // reprendre une région à son propre peuple en déclin ne rapporte rien
        let mut targets = g.targets();
        targets.retain(|x| g.owner_of(x.region) != Some(self.pid));
        let dom = self.dominance();
        let value = |x: &Target| self.region_value(army, x.region, &dom);

        if let Some(x) = best_by(targets.iter().filter(|x| x.sorcery.allowed()), value) {
            return Ok(Action::Sorcery { region: x.region });
        }
        let dragon_ready = rs.has(Power::DragonMaster) && !turn.dragon_used;
        if dragon_ready {
            let big = targets.iter().filter(|x| x.dragon.allowed() && x.cost as f64 >= t.dragon_min);
            if let Some(x) = best_by(big, |x| value(x) * x.cost as f64) {
                return Ok(Action::Dragon { region: x.region });
            }
        }
        if rs.has(Power::Berserk) && turn.berserk_die.is_none() && rs.hand >= 1 && targets.iter().any(|x| x.reachable())
        {
            return Ok(Action::BerserkRoll);
        }

        let affordable: ArrayTargets = targets.iter().filter(|x| x.kind == TargetKind::Normal).copied().collect();
        let normal = best_by(affordable.iter(), |x| self.target_score(army, x, &dom));
        if normal.is_some() && self.level != Level::Easy && t.entry_model != 0.0 && g.is_entry_choice() {
            // l'entrée d'un peuple neuf décide de tout son territoire : elle se choisit à part
            return Ok(Action::Conquer { region: self.choose_entry(&affordable)? });
        }
        // difficile : un modèle à part, appris sur l'avance finale sur le meilleur adversaire
        let lead = self.level == Level::Hard && t.conquest_lead > 0.0;
        if normal.is_some() && t.conquest_model != 0.0 && !self.ai.simulating && (self.level == Level::Normal || lead) {
            // le modèle choisit la conquête, ou d'arrêter là (pas dans les simulations : il n'y gagne rien
            // et elles seraient 40 % plus lentes)
            let model = if lead { &self.ai.models.conquest_lead } else { &self.ai.models.conquest };
            let rows = g.conquest_inputs(&affordable);
            let predicted = model.predict_all(rows.iter().map(|x| x.get()));
            let best = best_by(0..rows.len(), |k| predicted[k] + self.noise("c", Some(k as u64)) * 0.3);
            return Ok(match best.unwrap_or(0) {
                k if k < affordable.len() => Action::Conquer { region: affordable[k].region },
                _ => Action::EndConquest,
            });
        }
        if let Some(x) = normal {
            // garder ses jetons pour la défense plutôt que de les engager à trop faible rendement
            if t.min_ratio != 0.0 && turn.conquests > 0 && self.target_score(army, x, &dom) < t.min_ratio {
                return Ok(Action::EndConquest);
            }
            return Ok(Action::Conquer { region: x.region });
        }
        if dragon_ready {
            let any = targets.iter().filter(|x| x.dragon.allowed());
            if let Some(x) = best_by(any, |x| value(x) * x.cost as f64) {
                return Ok(Action::Dragon { region: x.region });
            }
        }
        let die = targets.iter().filter(|x| x.kind == TargetKind::Die);
        if let Some(x) = best_by(die, |x| x.chance * value(x)) {
            return Ok(Action::Reinforce { region: x.region });
        }
        Ok(Action::EndConquest)
    }

    // --- fin de tour -------------------------------------------------------------------------

    fn redeploy_step(&self, keep_placement: bool) -> Result<Action, Reason> {
        let g = self.g;
        let s = &g.state;
        let t = self.tuning();
        let turn = &s.turn;
        let army = turn.army.expect("peuple en action");
        let rs = *s.race(army);
        let own = g.owned(army);
        if turn.stage == Stage::Active && !rs.declined && !own.is_empty() {
            let open = || own.iter().filter(|&r| !s.regions[r].is_immune());
            let by_threat = |r: usize| g.threat(army, r) / g.defense_value(r) as f64;
            if rs.power == Power::Heroic && g.heroes_on_board(army) < 2 {
                let dom = self.dominance();
                let best = best_by(open(), |r| g.threat(army, r) + self.region_value(army, r, &dom) * 0.2);
                if let Some(region) = best {
                    return Ok(Action::Hero { region });
                }
            }
            if rs.power == Power::Fortified
                && turn.fortress_region.is_none()
                && g.fortresses_on_board(army) < 6
                && let Some(region) = best_by(open().filter(|&r| !s.regions[r].fortress), by_threat)
            {
                return Ok(Action::Fortress { region });
            }
            if rs.power == Power::Bivouacking
                && g.camps_on_board(army) < 5
                && let Some(region) = best_by(open(), by_threat)
            {
                return Ok(Action::Camp { region, delta: 1 });
            }
        }
        if !own.is_empty() && !keep_placement {
            // tout reprendre et replacer : les jetons qui ont servi à conquérir sont rarement là où il faut
            let differs = |target: &Placement| rs.hand > 0 || own.iter().any(|r| s.regions[r].tokens != target.0[r]);
            if self.level == Level::Hard {
                let target = self.best_placement(army)?;
                if differs(&target) {
                    return Ok(Action::Place(target));
                }
            } else if self.level == Level::Normal
                && t.hold_model != 0.0
                && !self.ai.simulating
                && turn.stage == Stage::Active
            {
                // chaque jeton là où il réduit le plus le risque de perdre une région (modèle appris)
                let target = self.hold_placement(army);
                if differs(&target) {
                    return Ok(Action::Place(target));
                }
            } else if differs(&g.full_deployment(army)) {
                return Ok(Action::AutoDeploy { all: true });
            }
        }
        if turn.stage == Stage::Ghouls {
            return Ok(Action::EndTurn { peace: None, stout: false });
        }

        let mut peace = None;
        if rs.has(Power::Diplomat) {
            let pressure = |q: usize| -> i32 {
                let Some(race) = s.players[q].active else { return 0 };
                own.iter()
                    .map(|r| {
                        let adj = &g.map.regions[r].adj;
                        adj.iter()
                            .map(|&n| &s.regions[n as usize])
                            .filter(|nr| nr.occupant == Occupant::Race(race))
                            .map(|nr| nr.tokens)
                            .sum::<i32>()
                    })
                    .sum()
            };
            let candidates = (0..s.player_count())
                .filter(|&q| q != self.pid && !turn.attacked.contains(q) && s.players[q].active.is_some());
            let best = best_by(candidates, |q| pressure(q) as f64 + self.guess_coins(q) * 0.01);
            if let Some(q) = best.filter(|&q| pressure(q) > 0) {
                peace = Some(q);
            }
        }
        let mut stout = false;
        if rs.has(Power::Stout) && s.max_rounds - s.round >= 1 {
            if self.level == Level::Hard {
                let mut yes = *g;
                yes.apply(&Action::EndTurn { peace, stout: true }, &mut NoJournal)?;
                let mut no = *g;
                no.apply(&Action::EndTurn { peace, stout: false }, &mut NoJournal)?;
                let turns = s.player_count() * (s.max_rounds - s.round).min(2) as usize;
                stout = self.compare_by_rollout(&yes, &no, turns)? > 0.5;
            } else {
                let ready = g.tokens_on_board(army) - own.len() as i32;
                stout = own.len() as f64 >= t.stout_own && ready as f64 <= t.stout_ready;
            }
        }
        Ok(Action::EndTurn { peace, stout })
    }

    /// Compare deux suites possibles par simulation ; renvoie l'avantage de `a` sur `b`.
    fn compare_by_rollout(&self, a: &Game, b: &Game, turns: usize) -> Result<f64, Reason> {
        let (mut va, mut vb) = (0.0, 0.0);
        for seed in seeds(self.g, 3) {
            va += self.ai.rollout(a, self.pid, turns, seed)?;
            vb += self.ai.rollout(b, self.pid, turns, seed)?;
        }
        Ok((va - vb) / 3.0)
    }

    /// Le placement de fin de tour qui résiste le mieux : chaque candidat est essayé, les adversaires
    /// jouent leur tour (politique normale, quelques tirages de dés), et l'on garde celui qui laisse le
    /// plus de revenu et de jetons au début du tour suivant.
    fn best_placement(&self, army: Race) -> Result<Placement, Reason> {
        let sim = self.ai.simulation();
        let candidates = self.g.placement_candidates(army);
        let mut best = candidates[0];
        let mut best_value = f64::NEG_INFINITY;
        for tokens in candidates {
            let mut value = 0.0;
            for seed in seeds(self.g, 3) {
                let mut s = *self.g;
                s.state.rng = seed;
                s.apply(&Action::Place(tokens), &mut NoJournal)?;
                s.apply(&Action::EndTurn { peace: None, stout: false }, &mut NoJournal)?;
                for _ in 0..PLACEMENT_STEPS {
                    let st = &s.state;
                    if st.phase == Phase::GameOver || (st.current == self.pid && st.phase != Phase::Defend) {
                        break;
                    }
                    sim.sim_step(&mut s)?;
                }
                let hand = s.state.players[self.pid].active.map_or(0.0, |a| 0.35 * s.state.race(a).hand as f64);
                value += s.income(self.pid) as f64 + hand;
            }
            if value > best_value + 1e-9 {
                best_value = value;
                best = tokens;
            }
        }
        Ok(best)
    }

    /// Placement complet de fin de tour selon le modèle de garde : un jeton par région, puis chaque jeton
    /// restant là où il réduit le plus la perte attendue (probabilité de perdre la région × ce qu'elle rapporte).
    fn hold_placement(&self, race: Race) -> Placement {
        let g = self.g.levelled(race);
        let own = g.owned(race);
        let mut spare = g.state.race(race).hand;
        let mut tokens = Placement::default();
        let open: Vec<usize> = own.iter().filter(|&r| !g.state.regions[r].is_immune()).collect();
        if open.is_empty() {
            if let Some(first) = own.first() {
                tokens.0[first] = 1 + spare;
            }
            return tokens;
        }
        let model = &self.ai.models.hold;
        let mut inputs = [None; MAX_REGIONS];
        for &r in &open {
            inputs[r] = Some(g.hold_inputs(race, r));
        }
        let risk = |r: usize, n: i32| {
            let mut x = inputs[r].expect("entrées de la région");
            x.set(HOLD_DEFENSE, x.at(HOLD_DEFENSE) - x.at(HOLD_TOKENS) + n as f64);
            x.set(HOLD_TOKENS, n as f64);
            model.predict(x.get()).clamp(0.0, 1.0)
        };
        let mut risks = [0.0; MAX_REGIONS];
        let mut next = [0.0; MAX_REGIONS];
        for &r in &open {
            risks[r] = risk(r, 1);
            next[r] = risk(r, 2);
        }
        while spare > 0 {
            let worth = |r: usize| inputs[r].expect("entrées de la région").at(HOLD_WORTH);
            // à gain égal (ou nul), la région la plus exposée d'abord
            let best = best_by(open.iter().copied(), |r| worth(r) * (risks[r] - next[r]).max(0.0) + 1e-3 * risks[r]);
            let best = best.unwrap_or(open[0]);
            tokens.0[best] += 1;
            risks[best] = next[best];
            next[best] = risk(best, tokens.0[best] + 1);
            spare -= 1;
        }
        tokens
    }
}
