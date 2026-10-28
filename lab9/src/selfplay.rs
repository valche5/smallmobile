//! Le selfplay de la phase 6 (`lab9/lab.md`) : chaque joueur choisit ses plans par la recherche sur
//! plans (`search`), et chaque plan cherché est enregistré dans la forme des parties d'imitation
//! (`record`, lue par `lab9/py/data.py`), avec ces cibles :
//!
//! - la politique : à chaque décision du plan joué, la politique améliorée par la recherche
//!   (`search::improved`), et l'a priori du décodeur à côté ;
//! - la valeur au prochain tour du joueur : celle que lit le réseau à la racine de sa recherche
//!   suivante (DORA : la valeur d'une copie du réseau, ici celle qui joue), et les pièces gagnées d'ici
//!   là ; le résultat final et les propriétaires, comme pour l'imitation ;
//! - `here` et `prior_win` : la valeur lue à la racine.
//!
//! Le plan joué est le meilleur candidat, sauf pendant les `explore` premières manches, où il est tiré
//! selon la politique améliorée sur les candidats (`search::sample`).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use smallmobile_kernel::Phase;
use smallmobile_lab7::selfplay::owners;
use smallmobile_lab7::table::Table;

use crate::encode::encode;
use crate::know::Knowledge;
use crate::moves;
use crate::plan::Start;
use crate::rand::{Rng, mix};
use crate::record::{Files, Game_, PROBABILITY, Segment, Step, coins, finish, relative};
use crate::search::{self, Engine};
use crate::tokens::dynamics;

pub struct Config {
    pub players: Vec<usize>,
    pub games: usize,
    pub threads: usize,
    pub seed: u64,
    pub search: search::Config,
    /// Échelle des valeurs, ramenées entre 0 et 1, dans la politique améliorée (`search::improved`).
    pub beta: f32,
    /// Manches où le plan joué est tiré plutôt que le meilleur.
    pub explore: i32,
    /// Arrêt au bout de ce temps (secondes ; 0 : aucun) : les parties en cours sont abandonnées (bancs
    /// de vitesse).
    pub seconds: f64,
}

/// Ce que comptent les fils : plans cherchés, décisions, et la somme de l'entropie croisée de la
/// politique améliorée face à l'a priori moins son entropie (KL, en millièmes de nat), aux décisions.
static PLANS: AtomicUsize = AtomicUsize::new(0);
static DECISIONS: AtomicUsize = AtomicUsize::new(0);
static KL: AtomicUsize = AtomicUsize::new(0);
/// Temps cumulés des fils (ns) : dans les parties, et dans les recherches.
static IN_GAMES: AtomicU64 = AtomicU64::new(0);
static IN_SEARCH: AtomicU64 = AtomicU64::new(0);

/// Les compteurs du selfplay à cet instant : recherches, lectures, pas, temps processeur (s), temps
/// des fils dans les recherches et dans l'attente des lectures et des pas (s).
pub fn snapshot() -> [f64; 7] {
    use search::{N_READS, N_STEPS, T_READ, T_STEP, TIMES};
    let t = |i: usize| TIMES[i].load(Ordering::Relaxed) as f64;
    [
        PLANS.load(Ordering::Relaxed) as f64,
        t(N_READS),
        t(N_STEPS),
        cpu_seconds(),
        IN_SEARCH.load(Ordering::Relaxed) as f64 / 1e9,
        t(T_READ) / 1e9,
        t(T_STEP) / 1e9,
    ]
}

/// Le temps processeur du programme (s), d'après /proc/self/stat (utime + stime).
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<&str> = stat.rsplit_once(')').map_or("", |r| r.1).split_whitespace().collect();
    let ticks = |i: usize| fields.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (ticks(11) + ticks(12)) / 100.0
}

/// Une partie où tous les joueurs cherchent leurs plans avec `engine`, découpée en plans ; `None` si
/// `deadline` passe avant la fin.
fn play<E: Engine>(
    engine: &E,
    cfg: &Config,
    table: &Table,
    seed: u32,
    rng: &mut Rng,
    deadline: Option<Instant>,
) -> Option<Vec<Segment>> {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let n = table.players;
    let adj: Vec<u64> = table.map.hoods.iter().map(|h| h.adj.0).collect();
    let mut segments: Vec<Segment> = Vec::new();
    // la suite du plan en cours
    let mut rest: VecDeque<moves::Move> = VecDeque::new();
    let mut turn = 0;
    while g.state.phase != Phase::GameOver {
        let mv = match rest.pop_front() {
            Some(mv) => mv,
            None => {
                let legal = moves::legal(&g, &k);
                if legal.len() == 1 {
                    legal[0].mv
                } else {
                    if deadline.is_some_and(|d| Instant::now() >= d) {
                        return None;
                    }
                    let plan = searched(engine, cfg, &g, &k, n, &adj, turn, &mut segments, rng);
                    rest.extend(plan);
                    rest.pop_front().expect("un plan non vide")
                }
            }
        };
        let current = g.state.current;
        k.play(&mut g, &mv).expect("coup possible");
        if g.state.current != current {
            turn += 1;
            let c = g.state.current;
            for p in segments.iter_mut().filter(|p| p.actor == c && p.owners.is_none()) {
                p.owners = Some(owners(&g, c));
            }
        }
    }
    finish(&mut segments, &g);
    Some(segments)
}

/// Une recherche au début d'un plan : le plan joué, et son enregistrement (`segments`).
#[allow(clippy::too_many_arguments)]
fn searched<E: Engine>(
    engine: &E,
    cfg: &Config,
    g: &smallmobile_kernel::Game,
    k: &Knowledge,
    n: usize,
    adj: &[u64],
    turn: usize,
    segments: &mut Vec<Segment>,
    rng: &mut Rng,
) -> Vec<moves::Move> {
    let t0 = Instant::now();
    let s = search::search(engine, &cfg.search, g, k, rng);
    IN_SEARCH.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let actor = g.state.actor().expect("un joueur doit agir");
    let value = s.root.0.win;
    // la première décision du prochain tour des joueurs qui l'attendaient
    if actor == g.state.current {
        let now = coins(g);
        for p in segments.iter_mut().filter(|p| p.actor == actor && p.turn < turn && p.next.is_none()) {
            let gain = std::array::from_fn(|q| (now[q] - p.coins[q]) as f32);
            p.next = Some((value, relative(&gain, actor, n)));
        }
    }
    let chosen = if g.state.round < cfg.explore {
        search::sample(&s, cfg.beta, rng)
    } else {
        s.candidates.iter().position(|c| c.0 == s.plan).expect("le plan choisi est un candidat")
    };
    let targets = search::improved(&s, chosen, cfg.beta);
    let plan = s.candidates[chosen].0.clone();
    // les décisions du plan, rejoué
    let (mut w, mut kw) = (*g, *k);
    let mut steps = Vec::new();
    let mut kl = 0.0f64;
    for mv in &plan {
        let legal = moves::legal(&w, &kw);
        if legal.len() > 1 {
            let at = &targets[steps.len()];
            debug_assert!(legal.iter().zip(at).all(|(c, t)| c.key == t.0), "coups possibles de la trace");
            let key = legal.iter().find(|c| c.mv == *mv).expect("coup du plan").key;
            for &(_, p, q) in at {
                if p > 0.0 {
                    kl += p as f64 * (p as f64 / (q as f64).max(1e-9)).ln();
                }
            }
            let probs = at
                .iter()
                .map(|&(key, p, q)| (key, (p * PROBABILITY).round() as u16, (q * PROBABILITY).round() as u16))
                .collect();
            steps.push(Step { key, dy: dynamics(&w, &kw), legal: probs });
        }
        kw.play(&mut w, mv).expect("coup du plan");
    }
    debug_assert!(Start::of(g).actor == actor);
    PLANS.fetch_add(1, Ordering::Relaxed);
    DECISIONS.fetch_add(steps.len(), Ordering::Relaxed);
    KL.fetch_add((kl * 1000.0) as usize, Ordering::Relaxed);
    let e = encode(g, k);
    segments.push(Segment {
        players: n,
        regions: e.regions,
        phase: g.state.phase,
        actor,
        turn,
        node: e.node[..e.regions].iter().flatten().copied().collect(),
        global: e.global.to_vec(),
        adj: adj.to_vec(),
        steps,
        coins: coins(g),
        here: value,
        prior: value,
        next: None,
        owners: None,
        win: [0.0; smallmobile_kernel::ids::MAX_PLAYERS],
        gain: [0.0; smallmobile_kernel::ids::MAX_PLAYERS],
    });
    plan
}

/// Les parties de selfplay, en parallèle, un jeu de fichiers par fil dans `dir` (`s{graine}-{fil}.*`).
/// Les fils se répartissent les moteurs `engines` (le fil t prend le moteur t % nombre) ; ceux d'un même
/// moteur partagent ses lots. Plusieurs moteurs GPU calculent en même temps : la carte, qu'un seul ne
/// remplit pas (lots trop petits), travaille alors à pleine puissance.
pub fn run<E: Engine>(cfg: &Config, engines: &[E], dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    // les fichiers de tous les fils d'abord : une erreur (trop de fichiers ouverts) arrête tout de suite
    let files: Vec<Files> = (0..cfg.threads)
        .map(|t| Files::create(&dir.join(format!("s{:x}-{t:02}", cfg.seed)), false))
        .collect::<std::io::Result<_>>()?;
    let (next, finished) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let start = Instant::now();
    let deadline = (cfg.seconds > 0.0).then(|| start + std::time::Duration::from_secs_f64(cfg.seconds));
    let cpu0 = cpu_seconds();
    std::thread::scope(|scope| -> std::io::Result<()> {
        let mut handles = Vec::new();
        for (t, mut files) in files.into_iter().enumerate() {
            let (next, finished) = (&next, &finished);
            let engine = &engines[t % engines.len()];
            handles.push(scope.spawn(move || -> std::io::Result<()> {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cfg.games {
                        break;
                    }
                    let players = cfg.players[i % cfg.players.len()];
                    let deal = mix(cfg.seed, i as u64);
                    let (map_seed, seed) = (deal % 1_000_000_007, mix(deal, 1) as u32);
                    let table = Table::new(players, map_seed);
                    let mut rng = Rng::new(mix(deal, 99));
                    let t0 = Instant::now();
                    let played = play(engine, cfg, &table, seed, &mut rng, deadline);
                    IN_GAMES.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let Some(segments) = played else { break };
                    files.write(&Game_ { deal, map_seed, seed, players }, &segments)?;
                    let count = finished.fetch_add(1, Ordering::Relaxed) + 1;
                    if count % 20 == 0 || count == cfg.games {
                        let secs = start.elapsed().as_secs_f64();
                        let (plans, decisions) = (PLANS.load(Ordering::Relaxed), DECISIONS.load(Ordering::Relaxed));
                        eprintln!(
                            "{count}/{} parties, {:.1} parties/min, {plans} plans, {:.0} plans/s, KL(cible ‖ a priori) {:.3} par décision",
                            cfg.games,
                            count as f64 * 60.0 / secs,
                            plans as f64 / secs,
                            KL.load(Ordering::Relaxed) as f64 / 1000.0 / decisions.max(1) as f64
                        );
                    }
                }
                files.flush()
            }));
        }
        for h in handles {
            h.join().expect("fil du selfplay")?;
        }
        report(cfg, start, cpu0, finished.load(Ordering::Relaxed));
        Ok(())
    })
}

/// Le bilan : débit, où passe le temps des fils, le processeur.
fn report(cfg: &Config, start: Instant, cpu0: f64, finished: usize) {
    use search::{N_READS, N_STEPS, T_OTHER, T_READ, T_STEP, TIMES};
    let secs = start.elapsed().as_secs_f64();
    let all = secs * cfg.threads as f64;
    let ns = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e9;
    let t = |i: usize| TIMES[i].load(Ordering::Relaxed) as f64 / 1e9;
    let pc = |x: f64| 100.0 * x / all;
    let (games, search) = (ns(&IN_GAMES), ns(&IN_SEARCH));
    let plans = PLANS.load(Ordering::Relaxed);
    eprintln!(
        "bilan : {secs:.1} s, {finished} parties finies, {plans} recherches ({:.1}/s), {} lectures ({:.0}/s), {} pas ({:.0}/s)",
        plans as f64 / secs,
        TIMES[N_READS].load(Ordering::Relaxed),
        TIMES[N_READS].load(Ordering::Relaxed) as f64 / secs,
        TIMES[N_STEPS].load(Ordering::Relaxed),
        TIMES[N_STEPS].load(Ordering::Relaxed) as f64 / secs
    );
    eprintln!(
        "fils ({} × {secs:.1} s) : recherches {:.1} % (attente des lectures {:.1}, des pas {:.1}, calcul {:.1}), \
         parties hors recherches {:.1} %, ailleurs {:.1} %",
        cfg.threads,
        pc(search),
        pc(t(T_READ)),
        pc(t(T_STEP)),
        pc(t(T_OTHER)),
        pc(games - search),
        pc(all - games)
    );
    eprintln!("processeur : {:.1} cœurs occupés en moyenne", (cpu_seconds() - cpu0) / secs);
}
