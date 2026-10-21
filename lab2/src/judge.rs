//! Positions rejugées par des fins de partie simulées : des positions de parties déjà jouées
//! (`selfplay::GameLog`), chacune suivie de `rollouts` fins où le réseau joue pour tous
//! (`Az::play_out`), chacune dans un monde possible honnête. La moyenne donne, presque sans bruit,
//! la chance de victoire et les pièces finales de chaque joueur : des cibles de valeur bien moins
//! bruitées que le résultat d'une seule partie.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use smallmobile_kernel::Game;
use smallmobile_kernel::ids::MAX_PLAYERS;

use crate::az::Az;
use crate::encode::encode;
use crate::know::{Knowledge, determinize};
use crate::moves::{self, KEYS};
use crate::rand::{Rng, mix};
use crate::selfplay::{GameLog, ILLEGAL, Record, games_file};
use crate::table::Outcome;

/// Moyenne de `count` fins de partie : part de victoire et pièces finales de chaque joueur réel.
pub fn judge(
    g: &Game,
    k: &Knowledge,
    count: usize,
    az: &Az,
    rng: &mut Rng,
) -> ([f32; MAX_PLAYERS], [f32; MAX_PLAYERS]) {
    let (mut win, mut coins) = ([0.0f32; MAX_PLAYERS], [0.0f32; MAX_PLAYERS]);
    let mut evals = crate::az::Evals::default();
    for _ in 0..count {
        let w = determinize(g, k, rng);
        let end = az.play_out(&mut evals, &w, k);
        let o = Outcome::of(&end);
        for q in 0..o.players {
            win[q] += o.win[q] as f32 / count as f32;
            coins[q] += o.coins[q] as f32 / count as f32;
        }
    }
    (win, coins)
}

/// L'enregistrement d'une position jugée : cibles de valeur seulement (politique toute nulle).
fn judged(g: &Game, k: &Knowledge, count: usize, az: &Az, rng: &mut Rng) -> Vec<u8> {
    let (win, finals) = judge(g, k, count, az, rng);
    let s = &g.state;
    let n = s.player_count();
    let me = s.actor().expect("un joueur doit agir");
    let mut policy = [ILLEGAL; KEYS];
    for c in &moves::legal(g) {
        policy[c.key as usize] = 0;
    }
    let (mut w, mut gain) = ([0.0; MAX_PLAYERS], [0.0; MAX_PLAYERS]);
    for r in 0..n {
        let q = (me + r) % n;
        w[r] = win[q];
        gain[r] = finals[q] - s.players[q].coins as f32;
    }
    let enc = encode(g, k);
    Record { enc: &enc, mixed: false, map: g.map, win: w, gain, policy: &policy }.bytes()
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
                let path = dir.join(format!("{prefix}{seed:x}-{t:02}{}", crate::selfplay::SUFFIX));
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

/// Juge les positions de parties déjà jouées (`GameLog`) : une part `share` des positions où le
/// joueur avait le choix, chacune par `rollouts` fins jouées par `az`.
pub fn rejudge(
    logs: &[GameLog],
    share: f64,
    rollouts: usize,
    az: &Az,
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
                out.extend(judged(g, k, rollouts, az, &mut rng));
                count += 1;
            }
        })
        .map_err(std::io::Error::other)?;
        Ok(count)
    })
}
