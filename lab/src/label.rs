//! Positions jugées par des fins de partie simulées : beaucoup de positions variées, chacune suivie
//! de `rollouts` fins jouées par le juge (l'IA Normal, ou le réseau : `Judge`), chacune dans un
//! monde possible honnête. La moyenne donne, presque sans bruit, la chance de victoire et les
//! pièces finales de chaque joueur : ce que les simulations de la recherche mesurent une à une. Un
//! réseau qui l'apprend vaut une infinité de simulations, pour le prix d'une évaluation.
//!
//! Les parties qui fournissent les positions sont jouées par l'IA Normal (ou un autre joueur, pour
//! voir les positions d'un jeu plus fort), avec une part de coups au hasard pour couvrir aussi les
//! positions étranges que la recherche visite. `rejudge` juge plutôt les positions de parties déjà
//! jouées (`selfplay::GameLog`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Ai, Game, Level, Phase, Tuning};

use crate::agent::{Agent, KernelAi};
use crate::az::Az;
use crate::encode::encode;
use crate::know::{Knowledge, Returns, determinize};
use crate::moves::{self, KEYS};
use crate::rand::{Rng, mix};
use crate::selfplay::{GameLog, ILLEGAL, Record, games_file};
use crate::table::{Outcome, Table};

/// Qui joue les fins de partie qui jugent une position.
pub enum Judge {
    /// L'IA Normal pour tous.
    Normal,
    /// Le réseau pour tous (`Az::play_out` : son coup préféré, ou tiré selon `rt`).
    Net(Box<Az>),
}

pub struct Config {
    pub players: Vec<usize>,
    pub games: usize,
    pub threads: usize,
    pub seed: u64,
    /// Part des positions jugées (les autres sont seulement jouées).
    pub share: f64,
    pub rollouts: usize,
    /// Part des coups joués au hasard dans les parties qui fournissent les positions.
    pub random: f64,
    pub judge: Judge,
}

/// Moyenne de `count` fins de partie : part de victoire et pièces finales de chaque joueur réel.
pub fn judge(
    g: &Game,
    k: &Knowledge,
    count: usize,
    by: &Judge,
    rng: &mut Rng,
) -> ([f32; MAX_PLAYERS], [f32; MAX_PLAYERS]) {
    let tuning = Tuning::default();
    let ai = Ai::new(&tuning);
    let (mut win, mut coins) = ([0.0f32; MAX_PLAYERS], [0.0f32; MAX_PLAYERS]);
    let mut evals = crate::az::Evals::default();
    for _ in 0..count {
        let mut w = determinize(g, k, rng);
        let end = match by {
            Judge::Normal => {
                for p in w.state.players.as_mut_slice() {
                    p.level = Level::Normal;
                }
                ai.play(&mut w, 50_000).expect("l'IA du jeu finit la partie");
                w
            }
            Judge::Net(az) => az.play_out(&mut evals, &w, k, rng),
        };
        let o = Outcome::of(&end);
        for q in 0..o.players {
            win[q] += o.win[q] as f32 / count as f32;
            coins[q] += o.coins[q] as f32 / count as f32;
        }
    }
    (win, coins)
}

/// L'enregistrement d'une position jugée : cibles de valeur seulement (aucun coup visité).
fn judged(g: &Game, k: &Knowledge, count: usize, by: &Judge, rng: &mut Rng) -> Vec<u8> {
    let (win, finals) = judge(g, k, count, by, rng);
    let s = &g.state;
    let n = s.player_count();
    let me = s.actor().expect("un joueur doit agir");
    let mut visits = [ILLEGAL; KEYS];
    for c in &moves::legal(g) {
        visits[c.key as usize] = 0;
    }
    let (mut w, mut gain) = ([0.0; MAX_PLAYERS], [0.0; MAX_PLAYERS]);
    let mut qgain = [f32::NAN; MAX_PLAYERS];
    for r in 0..n {
        let q = (me + r) % n;
        w[r] = win[q];
        gain[r] = finals[q] - s.players[q].coins as f32;
        qgain[r] = gain[r];
    }
    let enc = encode(g, k.race_tail);
    Record { enc: &enc, map: g.map, win: w, gain, visits: &visits, qgain, qwin: w }.bytes()
}

/// Une partie qui fournit des positions : `random` de coups au hasard, le reste par `agent`.
fn label_game(
    table: &Table,
    seed: u32,
    cfg: &Config,
    agent: &dyn Agent,
    rng: &mut Rng,
    out: &mut impl Write,
) -> std::io::Result<usize> {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut count = 0;
    while g.state.phase != Phase::GameOver {
        let legal = moves::legal(&g);
        if legal.len() > 1 && rng.unit() < cfg.share {
            out.write_all(&judged(&g, &k, cfg.rollouts, &cfg.judge, rng))?;
            count += 1;
        }
        let mv = if rng.unit() < cfg.random { legal[rng.below(legal.len())].mv } else { agent.act(&g, &k, rng) };
        let mut ret = Returns::default();
        moves::play(&mut g, &mv, &mut ret).expect("coup possible");
        k.observe(ret.0, &g.state);
    }
    Ok(count)
}

/// Écrit des positions jugées en parallèle, un fichier par fil dans `dir` (et son `games_file`) :
/// `game(i, rng, out)` produit la partie `i` et rend son nombre de positions.
fn write_games(
    dir: &Path,
    prefix: &str,
    seed: u64,
    threads: usize,
    games: usize,
    game: &(dyn Fn(usize, &mut Vec<u8>) -> std::io::Result<usize> + Sync),
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let next = AtomicUsize::new(0);
    let positions = AtomicUsize::new(0);
    let start = Instant::now();
    std::thread::scope(|scope| -> std::io::Result<()> {
        let mut handles = Vec::new();
        for t in 0..threads {
            let (next, positions) = (&next, &positions);
            handles.push(scope.spawn(move || -> std::io::Result<()> {
                let path = dir.join(format!("{prefix}{seed:x}-{t:02}.v3.bin"));
                let mut out = BufWriter::new(File::create(&path)?);
                let mut counts = BufWriter::new(File::create(games_file(&path))?);
                let mut buffer = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= games {
                        break;
                    }
                    buffer.clear();
                    let count = game(i, &mut buffer)?;
                    out.write_all(&buffer)?;
                    counts.write_all(&(count as u32).to_le_bytes())?;
                    let total = positions.fetch_add(count, Ordering::Relaxed) + count;
                    if (i + 1) % 100 == 0 {
                        let secs = start.elapsed().as_secs_f64();
                        eprintln!(
                            "partie {}/{games} : {total} positions, {:.0} positions/s",
                            i + 1,
                            total as f64 / secs
                        );
                    }
                }
                out.flush()?;
                counts.flush()
            }));
        }
        for h in handles {
            h.join().expect("fil de jugement")?;
        }
        Ok(())
    })
}

/// `agent` : qui joue les parties d'où viennent les positions (l'IA Normal si `None`).
pub fn run(cfg: &Config, agent: Option<&dyn Agent>, dir: &Path) -> std::io::Result<()> {
    let normal = KernelAi::new(Level::Normal);
    let agent = agent.unwrap_or(&normal);
    write_games(dir, "l", cfg.seed, cfg.threads, cfg.games, &|i, out| {
        let n = cfg.players[i % cfg.players.len()];
        let deal = mix(cfg.seed, i as u64);
        let table = Table::new(n, deal % 1_000_000_007);
        let mut rng = Rng::new(mix(deal, 77));
        label_game(&table, mix(deal, 1) as u32, cfg, agent, &mut rng, out)
    })
}

/// Juge les positions de parties déjà jouées (`GameLog`, d'un jeu fort par exemple) : une part
/// `share` des positions où le joueur avait le choix, chacune par `rollouts` fins de `by`.
pub fn rejudge(
    logs: &[GameLog],
    share: f64,
    rollouts: usize,
    by: &Judge,
    seed: u64,
    threads: usize,
    dir: &Path,
) -> std::io::Result<()> {
    write_games(dir, "j", seed, threads, logs.len(), &|i, out| {
        let log = &logs[i];
        let table = log.table();
        let mut rng = Rng::new(mix(seed, log.deal));
        let mut count = 0;
        log.replay(&table, |g, k| {
            if moves::legal(g).len() > 1 && rng.unit() < share {
                out.extend(judged(g, k, rollouts, by, &mut rng));
                count += 1;
            }
        })
        .map_err(std::io::Error::other)?;
        Ok(count)
    })
}
