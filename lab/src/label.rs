//! Positions jugées par des fins de partie simulées : beaucoup de positions variées, chacune suivie
//! de `rollouts` fins jouées par l'IA Normal (chacune dans un monde possible honnête). La moyenne
//! donne, presque sans bruit, la chance de victoire et les pièces finales de chaque joueur : ce que
//! les simulations de la recherche mesurent une à une. Un réseau qui l'apprend vaut une infinité de
//! simulations, pour le prix d'une évaluation.
//!
//! Les parties qui fournissent les positions sont jouées par l'IA Normal (ou un autre joueur, pour
//! voir les positions d'un jeu plus fort), avec une part de coups au hasard pour couvrir aussi les
//! positions étranges que la recherche visite.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Ai, Game, Level, Phase, Tuning};

use crate::agent::{Agent, KernelAi};
use crate::encode::encode;
use crate::know::{Knowledge, Returns, determinize};
use crate::moves::{self, KEYS};
use crate::rand::{Rng, mix};
use crate::selfplay::{ILLEGAL, Record};
use crate::table::{Outcome, Table};

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
}

/// Moyenne de `count` fins de partie : part de victoire et pièces finales de chaque joueur réel.
pub fn judge(g: &Game, k: &Knowledge, count: usize, rng: &mut Rng) -> ([f32; MAX_PLAYERS], [f32; MAX_PLAYERS]) {
    let tuning = Tuning::default();
    let ai = Ai::new(&tuning);
    let (mut win, mut coins) = ([0.0f32; MAX_PLAYERS], [0.0f32; MAX_PLAYERS]);
    for _ in 0..count {
        let mut w = determinize(g, k, rng);
        for p in w.state.players.as_mut_slice() {
            p.level = Level::Normal;
        }
        ai.play(&mut w, 50_000).expect("l'IA du jeu finit la partie");
        let o = Outcome::of(&w);
        for q in 0..o.players {
            win[q] += o.win[q] as f32 / count as f32;
            coins[q] += o.coins[q] as f32 / count as f32;
        }
    }
    (win, coins)
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
            let (win, finals) = judge(&g, &k, cfg.rollouts, rng);
            let s = &g.state;
            let n = s.player_count();
            let me = s.actor().expect("un joueur doit agir");
            let mut visits = [ILLEGAL; KEYS];
            // pas de cible de politique : aucun coup visité
            for c in &legal {
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
            let enc = encode(&g, k.race_tail);
            out.write_all(&Record { enc: &enc, map: g.map, win: w, gain, visits: &visits, qgain }.bytes())?;
            count += 1;
        }
        let mv = if rng.unit() < cfg.random { legal[rng.below(legal.len())].mv } else { agent.act(&g, &k, rng) };
        let mut ret = Returns::default();
        moves::play(&mut g, &mv, &mut ret).expect("coup possible");
        k.observe(ret.0, &g.state);
    }
    Ok(count)
}

/// `agent` : qui joue les parties d'où viennent les positions (l'IA Normal si `None`).
pub fn run(cfg: &Config, agent: Option<&dyn Agent>, dir: &Path) -> std::io::Result<()> {
    let normal = KernelAi::new(Level::Normal);
    let agent = agent.unwrap_or(&normal);
    std::fs::create_dir_all(dir)?;
    let next = AtomicUsize::new(0);
    let positions = AtomicUsize::new(0);
    let start = Instant::now();
    std::thread::scope(|scope| -> std::io::Result<()> {
        let mut handles = Vec::new();
        for t in 0..cfg.threads {
            let (next, positions) = (&next, &positions);
            handles.push(scope.spawn(move || -> std::io::Result<()> {
                let mut out = BufWriter::new(File::create(dir.join(format!("l{:x}-{t:02}.v2.bin", cfg.seed)))?);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cfg.games {
                        break;
                    }
                    let n = cfg.players[i % cfg.players.len()];
                    let deal = mix(cfg.seed, i as u64);
                    let table = Table::new(n, deal % 1_000_000_007);
                    let mut rng = Rng::new(mix(deal, 77));
                    let count = label_game(&table, mix(deal, 1) as u32, cfg, agent, &mut rng, &mut out)?;
                    let total = positions.fetch_add(count, Ordering::Relaxed) + count;
                    if (i + 1) % 100 == 0 {
                        let secs = start.elapsed().as_secs_f64();
                        eprintln!(
                            "partie {}/{} : {total} positions, {:.0} positions/s",
                            i + 1,
                            cfg.games,
                            total as f64 / secs
                        );
                    }
                }
                out.flush()
            }));
        }
        for h in handles {
            h.join().expect("fil de jugement")?;
        }
        Ok(())
    })
}
