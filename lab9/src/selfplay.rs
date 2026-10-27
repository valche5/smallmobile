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
use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// Échelle des valeurs dans la politique améliorée (`search::improved`).
    pub beta: f32,
    /// Manches où le plan joué est tiré plutôt que le meilleur.
    pub explore: i32,
}

/// Ce que comptent les fils : plans cherchés, décisions, et la somme de l'entropie croisée de la
/// politique améliorée face à l'a priori moins son entropie (KL, en millièmes de nat), aux décisions.
static PLANS: AtomicUsize = AtomicUsize::new(0);
static DECISIONS: AtomicUsize = AtomicUsize::new(0);
static KL: AtomicUsize = AtomicUsize::new(0);

/// Une partie où tous les joueurs cherchent leurs plans avec `engine`, découpée en plans.
fn play<E: Engine>(engine: &E, cfg: &Config, table: &Table, seed: u32, rng: &mut Rng) -> Vec<Segment> {
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
    segments
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
    let s = search::search(engine, &cfg.search, g, k, rng);
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
/// Les fils partagent le moteur `engine` (le GPU rassemble leurs lectures).
pub fn run<E: Engine>(cfg: &Config, engine: &E, dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let (next, finished) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let start = Instant::now();
    std::thread::scope(|scope| -> std::io::Result<()> {
        let mut handles = Vec::new();
        for t in 0..cfg.threads {
            let (next, finished) = (&next, &finished);
            handles.push(scope.spawn(move || -> std::io::Result<()> {
                let mut files = Files::create(&dir.join(format!("s{:x}-{t:02}", cfg.seed)), false)?;
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
                    let segments = play(engine, cfg, &table, seed, &mut rng);
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
        Ok(())
    })
}
