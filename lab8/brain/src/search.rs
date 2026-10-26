//! La recherche sur des plans, version 1 (`lab8/lab.md`), comme SearchBot à un tour :
//!
//! 1. **candidats** : le plan le plus probable du décodeur, et des plans tirés à la température
//!    `temperature`, sans doublon, jusqu'à `candidates` (tirés depuis une seule lecture de la racine) ;
//! 2. **mondes** : `worlds` mondes possibles (pioches cachées, dés), les mêmes pour tous les
//!    candidats ; un candidat qui finit par un dé (renfort, Berserks) est joué à chacune de ses
//!    issues, pondérées par leur probabilité exacte ;
//! 3. dans chaque monde, chaque candidat est joué, puis la **fin simulée par plans** : chaque joueur
//!    joue le plan le plus probable du décodeur, jusqu'au début du prochain tour du joueur de la racine
//!    (ou la fin de la partie, au résultat exact) ;
//! 4. la valeur pour le joueur de la racine : celle de la fin simulée, mêlée pour une part `1 − mix`
//!    à celle de la première position lue après le candidat ; victoire et avance en pièces mêlées comme
//!    dans lab7 (`win`) ;
//! 5. le candidat de meilleure valeur moyenne sur les mondes.
//!
//! Toutes les fins simulées avancent ensemble (`run`) : à chaque tour, les positions à lire sont lues
//! d'un coup, puis les pas du décodeur sont faits d'un coup ; le moteur GPU les groupera en lots.

use std::rc::Rc;

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Game, Phase};

use crate::encode::{Encoded, encode};
use crate::know::{Knowledge, determinize};
use crate::moves::{self, Choice, Key, Move};
use crate::net::{Cache, Memory, Net, StepIn, Value};
use crate::outcome::Outcome;
use crate::plan::{Start, pick};
use crate::rand::{Rng, mix};
use crate::tokens::{distances, dynamics};

/// Le temps des recherches (ns) : lectures des positions, pas du décodeur, le reste (règles, coups) ;
/// et le nombre de lectures et de pas. Affiché par l'arène.
pub static TIMES: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
pub const T_READ: usize = 0;
pub const T_STEP: usize = 1;
pub const T_OTHER: usize = 2;
pub const N_READS: usize = 3;
pub const N_STEPS: usize = 4;

fn spent(slot: usize, since: std::time::Instant) {
    TIMES[slot].fetch_add(since.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// Échelle de l'avance en pièces dans la valeur (lab7).
const SCALE: f32 = 8.0;
/// Garde-fou : décisions d'une fin simulée.
const MAX_DECISIONS: usize = 2000;

/// La tête de valeur lue au bout des fins simulées.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    /// Le résultat final.
    Final,
    /// La valeur au prochain tour du joueur.
    Next,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub candidates: usize,
    pub worlds: usize,
    /// Température des candidats tirés.
    pub temperature: f32,
    /// Part de la fin simulée dans la valeur d'un candidat (1 : la fin simulée seule).
    pub mix: f32,
    /// Poids de la victoire dans la valeur, le reste à l'avance en pièces (lab7 : 0,5).
    pub win: f32,
    pub head: Head,
    /// Candidats ajoutés par modification locale du meilleur (DORA) : rejoué jusqu'à une de ses
    /// décisions tirée au hasard, puis réécrit par le décodeur ; évalués dans les mêmes mondes.
    pub mods: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config { candidates: 16, worlds: 4, temperature: 1.0, mix: 0.5, win: 0.5, head: Head::Final, mods: 0 }
    }
}

impl Config {
    /// `k=16,m=4,t=1,mix=0.5,win=0.5,head=next,mods=8` (tout facultatif).
    pub fn parse(text: &str) -> Option<Config> {
        let mut cfg = Config::default();
        for part in text.split(',').filter(|p| !p.is_empty()) {
            let (k, v) = part.split_once('=')?;
            match k {
                "k" => cfg.candidates = v.parse().ok().filter(|&k| k >= 1)?,
                "m" => cfg.worlds = v.parse().ok().filter(|&m| m >= 1)?,
                "t" => cfg.temperature = v.parse().ok()?,
                "mods" => cfg.mods = v.parse().ok()?,
                "mix" => cfg.mix = v.parse().ok()?,
                "win" => cfg.win = v.parse().ok()?,
                "head" => {
                    cfg.head = match v {
                        "final" => Head::Final,
                        "next" => Head::Next,
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
        Some(cfg)
    }
}

/// Le résultat d'une recherche.
pub struct Searched {
    /// Le plan choisi, coups forcés compris, depuis la racine.
    pub plan: Vec<Move>,
    /// Les candidats (leurs coups) et leur valeur moyenne, dans l'ordre où ils ont été tirés (le
    /// premier : le plus probable).
    pub candidates: Vec<(Vec<Move>, f32)>,
    /// Lectures de position et pas du décodeur demandés.
    pub encodes: usize,
    pub steps: usize,
}

/// Le tour en cours, compté depuis le début de la partie (lab7).
fn turn_index(g: &Game) -> i64 {
    let s = &g.state;
    s.round as i64 * s.player_count() as i64 + s.current as i64
}

/// Le début du prochain tour du joueur qui agit en `g` (en défense, hors de son tour : son tour).
fn horizon(g: &Game) -> i64 {
    let s = &g.state;
    let n = s.player_count() as i64;
    let (me, current) = (s.actor().expect("un joueur doit agir") as i64, s.current as i64);
    turn_index(g) + if me == current { n } else { (me - current).rem_euclid(n) }
}

/// La valeur de chaque joueur réel : victoire et avance en pièces (finales ou prévues) mêlées.
fn blend(win: &[f32; MAX_PLAYERS], finals: &[f32; MAX_PLAYERS], n: usize, w: f32) -> [f32; MAX_PLAYERS] {
    let mut v = [0.0; MAX_PLAYERS];
    for q in 0..n {
        let rival = (0..n).filter(|&o| o != q).map(|o| finals[o]).fold(f32::NEG_INFINITY, f32::max);
        let lead = 1.0 / (1.0 + (-(finals[q] - rival) / SCALE).exp());
        v[q] = w * win[q] + (1.0 - w) * lead;
    }
    v
}

/// La valeur lue par le réseau en `g` (joueurs réels).
fn judged(cfg: &Config, g: &Game, mem: &Memory) -> [f32; MAX_PLAYERS] {
    let s = &g.state;
    let n = s.player_count();
    let me = s.actor().expect("un joueur doit agir");
    let v: &Value = if cfg.head == Head::Next { &mem.next } else { &mem.value };
    let (mut win, mut finals) = ([0.0; MAX_PLAYERS], [0.0; MAX_PLAYERS]);
    for r in 0..n {
        let q = (me + r) % n;
        win[q] = v.win[r];
        finals[q] = s.players[q].coins as f32 + v.gain[r];
    }
    blend(&win, &finals, n, cfg.win)
}

fn terminal(cfg: &Config, g: &Game) -> [f32; MAX_PLAYERS] {
    let o = Outcome::of(g);
    let (mut win, mut finals) = ([0.0; MAX_PLAYERS], [0.0; MAX_PLAYERS]);
    for q in 0..o.players {
        win[q] = o.win[q] as f32;
        finals[q] = o.coins[q] as f32;
    }
    blend(&win, &finals, o.players, cfg.win)
}

/// Ce qu'attend une voie.
enum Need {
    /// Lire la position (début d'un plan, ou horizon atteint).
    Encode,
    /// Un pas du décodeur, parmi ces coups.
    Step(Vec<Choice>),
    Done,
}

/// Une partie qui avance plan après plan : une fin simulée, ou l'écriture d'un candidat.
struct Lane<'m> {
    w: Game<'m>,
    k: Knowledge,
    /// Le plan en cours : son début, la lecture de sa position, le décodeur.
    plan: Option<(Start, Rc<Memory>, Cache, Option<Key>)>,
    /// Coups imposés à jouer d'abord (le candidat), puis le décodeur.
    forced: std::vec::IntoIter<Move>,
    /// Décisions imposées au décodeur (leurs clés) : il les lit sans les choisir (modifications
    /// locales).
    prefix: std::vec::IntoIter<Key>,
    /// Arrêt : au début de ce tour (fin simulée), ou à la fin du premier plan (candidat).
    horizon: Option<i64>,
    /// Température du décodeur.
    t: f32,
    rng: Rng,
    /// Coups joués (candidat).
    played: Vec<Move>,
    decisions: usize,
    /// La valeur à la première lecture après le candidat, et celle du bout.
    first: Option<[f32; MAX_PLAYERS]>,
    end: Option<[f32; MAX_PLAYERS]>,
    /// L'issue imposée au dé qui finit le candidat (`Outcome`), son candidat et son poids.
    die: Option<DieOutcome>,
    cand: usize,
    weight: f32,
}

/// L'issue d'un dé : réussite ou échec du dé de renfort, valeur du dé des Berserks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DieOutcome {
    Reinforce(bool),
    Berserk(i32),
}

/// Les issues du dé qui finit le plan `plan` joué depuis `g` (`k`), avec leur probabilité exacte ;
/// aucune si le plan ne finit pas par un dé.
fn outcomes(g: &Game, k: &Knowledge, plan: &[Move]) -> Vec<(DieOutcome, f32)> {
    let (mut w, mut kw) = (*g, *k);
    let Some((last, before)) = plan.split_last() else { return Vec::new() };
    for mv in before {
        kw.play(&mut w, mv).expect("coup du candidat");
    }
    match *last {
        Move::Region(moves::Kind::Reinforce, r) => {
            let need = w.targets().iter().find(|t| t.region == r as usize).map_or(1, |t| t.need);
            let p = smallmobile_kernel::ids::die_chance(need) as f32;
            [(DieOutcome::Reinforce(true), p), (DieOutcome::Reinforce(false), 1.0 - p)]
                .into_iter()
                .filter(|o| o.1 > 0.0)
                .collect()
        }
        Move::Berserk => {
            let faces = smallmobile_kernel::ids::DIE_FACES;
            let mut out: Vec<(DieOutcome, f32)> = Vec::new();
            for f in faces {
                match out.iter_mut().find(|o| o.0 == DieOutcome::Berserk(f)) {
                    Some(o) => o.1 += 1.0 / faces.len() as f32,
                    None => out.push((DieOutcome::Berserk(f), 1.0 / faces.len() as f32)),
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

impl<'m> Lane<'m> {
    /// Joue les coups imposés et forcés jusqu'à ce qu'il faille lire ou décider.
    fn advance(&mut self, cfg: &Config) -> Need {
        loop {
            if self.end.is_some() {
                return Need::Done;
            }
            // un candidat s'arrête à la fin de son premier plan (il n'est pas jugé ici)
            if self.horizon.is_none() && self.plan.is_none() && !self.played.is_empty() {
                self.end = Some([0.0; MAX_PLAYERS]);
                return Need::Done;
            }
            if self.w.state.phase == Phase::GameOver {
                self.end = Some(terminal(cfg, &self.w));
                return Need::Done;
            }
            if let Some(mv) = self.forced.next() {
                if self.forced.len() == 0
                    && let Some(want) = self.die
                {
                    self.force_die(mv, want);
                }
                self.play(mv);
                continue;
            }
            if let Some(h) = self.horizon
                && (turn_index(&self.w) >= h || self.decisions >= MAX_DECISIONS)
            {
                return Need::Encode;
            }
            let legal = moves::legal(&self.w, &self.k);
            if legal.len() == 1 {
                self.play(legal[0].mv);
                continue;
            }
            return if self.plan.is_some() { Need::Step(legal) } else { Need::Encode };
        }
    }

    /// Change les dés à venir du monde pour que le dé de `mv` donne `want` (`DieOutcome`).
    fn force_die(&mut self, mv: Move, want: DieOutcome) {
        let base = self.w.state.rng;
        for i in 0..1000u32 {
            let (mut w, mut k) = (self.w, self.k);
            w.state.rng = base.wrapping_add(i.wrapping_mul(0x9e37_79b9));
            k.play(&mut w, &mv).expect("dé du candidat");
            let got = w.state.last_die.map(|d| match want {
                DieOutcome::Reinforce(_) => DieOutcome::Reinforce(d.success == Some(true)),
                DieOutcome::Berserk(_) => DieOutcome::Berserk(d.value),
            });
            if got == Some(want) {
                self.w.state.rng = base.wrapping_add(i.wrapping_mul(0x9e37_79b9));
                return;
            }
        }
        panic!("issue du dé introuvable : {want:?}");
    }

    fn play(&mut self, mv: Move) {
        let before = self.plan.as_ref().map(|p| p.0);
        self.k.play(&mut self.w, &mv).expect("coup possible");
        self.played.push(mv);
        if let Some(start) = before
            && !start.goes_on(&mv, &self.w)
        {
            self.plan = None;
        }
    }
}

/// Qui lit les positions (l'encodeur) : le processeur (`Net`), ou un moteur GPU qui lit tout un lot
/// d'un coup. Le décodeur reste sur le processeur (`Net::step`).
pub trait Reader: Sync {
    fn read(&self, items: Vec<(Encoded, Vec<u8>)>) -> Vec<Memory>;
}

impl Reader for Net {
    fn read(&self, items: Vec<(Encoded, Vec<u8>)>) -> Vec<Memory> {
        items.iter().map(|(e, d)| self.encode(e, d)).collect()
    }
}

/// Les lectures des positions `games`, d'un coup.
fn read_all(reader: &dyn Reader, games: &[(&Game, &Knowledge)], dist: &[u8]) -> Vec<Memory> {
    reader.read(games.iter().map(|(g, k)| (encode(g, k), dist.to_vec())).collect())
}

/// Fait avancer toutes les voies jusqu'au bout. Les candidats (`horizon` absent) s'arrêtent à la fin
/// de leur premier plan ; les fins simulées, à leur horizon. `count` : lectures et pas demandés.
fn run(net: &Net, reader: &dyn Reader, cfg: &Config, lanes: &mut [Lane], dist: &[u8], count: &mut (usize, usize)) {
    loop {
        let t0 = std::time::Instant::now();
        let needs: Vec<Need> = lanes.iter_mut().map(|l| l.advance(cfg)).collect();
        spent(T_OTHER, t0);
        if needs.iter().all(|n| matches!(n, Need::Done)) {
            return;
        }
        // les lectures
        let reading: Vec<usize> = (0..lanes.len()).filter(|&i| matches!(needs[i], Need::Encode)).collect();
        let t0 = std::time::Instant::now();
        let read = read_all(reader, &reading.iter().map(|&i| (&lanes[i].w, &lanes[i].k)).collect::<Vec<_>>(), dist);
        spent(T_READ, t0);
        TIMES[N_READS].fetch_add(read.len() as u64, std::sync::atomic::Ordering::Relaxed);
        count.0 += read.len();
        for (&i, mem) in reading.iter().zip(read) {
            let lane = &mut lanes[i];
            let value = judged(cfg, &lane.w, &mem);
            if lane.first.is_none() {
                lane.first = Some(value);
            }
            if lane.horizon.is_some_and(|h| turn_index(&lane.w) >= h || lane.decisions >= MAX_DECISIONS) {
                lane.end = Some(value);
            } else {
                lane.plan = Some((Start::of(&lane.w), Rc::new(mem), Cache::default(), None));
            }
        }
        // les pas du décodeur, tous ensemble
        let stepping: Vec<usize> = (0..lanes.len()).filter(|&i| matches!(needs[i], Need::Step(_))).collect();
        let t0 = std::time::Instant::now();
        let logits = {
            let mut items: Vec<StepIn> = Vec::with_capacity(stepping.len());
            for (i, lane) in lanes.iter_mut().enumerate() {
                if !matches!(needs[i], Need::Step(_)) {
                    continue;
                }
                let dy = dynamics(&lane.w, &lane.k);
                let (_, mem, cache, prev) = lane.plan.as_mut().expect("plan en cours");
                items.push(StepIn { mem, cache, prev: *prev, dy });
            }
            net.step_batch(&mut items)
        };
        spent(T_STEP, t0);
        TIMES[N_STEPS].fetch_add(logits.len() as u64, std::sync::atomic::Ordering::Relaxed);
        count.1 += logits.len();
        for (i, all) in stepping.into_iter().zip(logits) {
            let Need::Step(legal) = &needs[i] else { unreachable!() };
            let lane = &mut lanes[i];
            let c = match lane.prefix.next() {
                Some(key) => *legal.iter().find(|c| c.key == key).expect("décision du préfixe"),
                None => {
                    let choice: Vec<f32> = legal.iter().map(|c| all[c.key as usize]).collect();
                    legal[pick(&choice, lane.t, &mut lane.rng)]
                }
            };
            lane.plan.as_mut().expect("plan en cours").3 = Some(c.key);
            lane.decisions += 1;
            lane.play(c.mv);
        }
    }
}

/// Une voie neuve depuis `g` (ou un monde de `g`).
fn lane<'m>(w: &Game<'m>, k: &Knowledge) -> Lane<'m> {
    Lane {
        w: *w,
        k: *k,
        plan: None,
        forced: Vec::new().into_iter(),
        prefix: Vec::new().into_iter(),
        horizon: None,
        t: 0.0,
        rng: Rng::new(0),
        played: Vec::new(),
        decisions: 0,
        first: None,
        end: None,
        die: None,
        cand: 0,
        weight: 1.0,
    }
}

/// Écrit des plans depuis `g` avec le décodeur, d'une seule lecture de la racine (`root`) : un par
/// (préfixe de décisions imposées, température).
#[allow(clippy::too_many_arguments)]
fn write_plans(
    net: &Net,
    reader: &dyn Reader,
    cfg: &Config,
    g: &Game,
    k: &Knowledge,
    root: &Rc<Memory>,
    tries: Vec<(Vec<Key>, f32)>,
    rng: &mut Rng,
    dist: &[u8],
    count: &mut (usize, usize),
) -> Vec<Vec<Move>> {
    let mut lanes: Vec<Lane> = tries
        .into_iter()
        .enumerate()
        .map(|(i, (prefix, t))| {
            let mut l = lane(g, k);
            l.prefix = prefix.into_iter();
            l.t = t;
            l.rng = Rng::new(mix(rng.next_u64(), i as u64));
            l.plan = Some((Start::of(g), root.clone(), Cache::default(), None));
            l
        })
        .collect();
    run(net, reader, cfg, &mut lanes, dist, count);
    lanes.into_iter().map(|l| l.played).collect()
}

/// Les clés des décisions d'un plan joué depuis `g` (coups forcés exclus).
fn decisions(g: &Game, k: &Knowledge, plan: &[Move]) -> Vec<Key> {
    let (mut w, mut kw) = (*g, *k);
    let mut out = Vec::new();
    for mv in plan {
        let legal = moves::legal(&w, &kw);
        if legal.len() > 1 {
            out.push(legal.iter().find(|c| c.mv == *mv).expect("coup du plan").key);
        }
        kw.play(&mut w, mv).expect("coup du plan");
    }
    out
}

/// La valeur moyenne de chaque candidat pour le joueur `me`, dans les mondes `worlds`, chaque
/// candidat fini par un dé joué à chacune de ses issues.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    net: &Net,
    reader: &dyn Reader,
    cfg: &Config,
    g: &Game,
    k: &Knowledge,
    worlds: &[Game],
    candidates: &[Vec<Move>],
    dist: &[u8],
    count: &mut (usize, usize),
) -> Vec<f32> {
    let me = g.state.actor().expect("un joueur doit agir");
    let h = horizon(g);
    let mut lanes: Vec<Lane> = Vec::new();
    for w in worlds {
        for (c, plan) in candidates.iter().enumerate() {
            let dice = outcomes(g, k, plan);
            let branches: Vec<(Option<DieOutcome>, f32)> =
                if dice.is_empty() { vec![(None, 1.0)] } else { dice.iter().map(|&(o, p)| (Some(o), p)).collect() };
            for (die, weight) in branches {
                let mut l = lane(w, k);
                l.forced = plan.clone().into_iter();
                l.horizon = Some(h);
                l.die = die;
                l.cand = c;
                l.weight = weight;
                lanes.push(l);
            }
        }
    }
    run(net, reader, cfg, &mut lanes, dist, count);
    let mut scores = vec![0.0f32; candidates.len()];
    for l in &lanes {
        let end = l.end.expect("fin simulée finie");
        let first = l.first.unwrap_or(end);
        scores[l.cand] += l.weight * (cfg.mix * end[me] + (1.0 - cfg.mix) * first[me]) / worlds.len() as f32;
    }
    scores
}

/// Une recherche depuis `g`, où le joueur qui agit a une décision. `reader` lit les positions (`net`
/// lui-même, ou un moteur GPU) ; `net` décode.
pub fn search(net: &Net, reader: &dyn Reader, cfg: &Config, g: &Game, k: &Knowledge, rng: &mut Rng) -> Searched {
    let dist = distances(g.map);
    let mut count = (0, 0);
    // une seule lecture de la racine pour tous les candidats
    let root = Rc::new(read_all(reader, &[(g, k)], &dist).pop().expect("lecture"));
    count.0 += 1;
    // 1. les candidats : le plus probable, puis des tirages, sans doublon
    let tries = (0..3 * cfg.candidates).map(|i| (Vec::new(), if i == 0 { 0.0 } else { cfg.temperature })).collect();
    let mut candidates: Vec<Vec<Move>> = Vec::new();
    for plan in write_plans(net, reader, cfg, g, k, &root, tries, rng, &dist, &mut count) {
        if candidates.len() < cfg.candidates && !candidates.contains(&plan) {
            candidates.push(plan);
        }
    }
    // 2-4. chaque candidat dans chaque monde, avec sa fin simulée
    let worlds: Vec<Game> = (0..cfg.worlds).map(|_| determinize(g, k, rng)).collect();
    let mut scores = evaluate(net, reader, cfg, g, k, &worlds, &candidates, &dist, &mut count);
    // modifications locales du meilleur, évaluées dans les mêmes mondes
    if cfg.mods > 0 {
        let best = (0..candidates.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a)))
            .expect("un candidat");
        let keys = decisions(g, k, &candidates[best]);
        let tries: Vec<(Vec<Key>, f32)> =
            (0..3 * cfg.mods).map(|_| (keys[..rng.below(keys.len())].to_vec(), cfg.temperature)).collect();
        let mut fresh = Vec::new();
        for plan in write_plans(net, reader, cfg, g, k, &root, tries, rng, &dist, &mut count) {
            if fresh.len() < cfg.mods && !candidates.contains(&plan) && !fresh.contains(&plan) {
                fresh.push(plan);
            }
        }
        scores.extend(evaluate(net, reader, cfg, g, k, &worlds, &fresh, &dist, &mut count));
        candidates.extend(fresh);
    }
    let n = candidates.len();
    let best = (0..n).max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a))).expect("un candidat");
    Searched {
        plan: candidates[best].clone(),
        candidates: candidates.into_iter().zip(scores).collect(),
        encodes: count.0,
        steps: count.1,
    }
}
