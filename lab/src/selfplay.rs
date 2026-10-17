//! Parties d'entraînement : une IA qui cherche joue contre elle-même, et chaque position cherchée
//! devient un exemple pour le réseau (ce qu'elle voyait, les visites de chaque coup, puis le résultat
//! final).
//!
//! Format des fichiers : des enregistrements de taille fixe (`RECORD` octets), lus en Python par
//! `numpy.fromfile` avec le type décrit dans `lab/py/data.py` :
//!
//! ```text
//! u8 joueurs, u8 régions, u8 pleine recherche, u8 libre
//! f32 × 5  victoire de chaque joueur relatif (part, 0 à 1)
//! f32 × 5  pièces gagnées d'ici la fin par chaque joueur relatif
//! u16 × KEYS  visites de chaque coup (65535 : coup impossible)
//! u8 × REGIONS × 8  voisines (un bit par région, 64 bits par région)
//! u8 × REGIONS × NODE  entrées par région
//! u8 × GLOBAL  entrées de la partie
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Game, Phase};

use crate::az::Searched;
use crate::encode::{Encoded, GLOBAL, NODE, REGIONS, encode};
use crate::know::{Knowledge, Returns};
use crate::moves::{self, KEYS};
use crate::rand::{Rng, mix};
use crate::table::{Outcome, Table};

pub const RECORD: usize = 4 + 20 + 20 + KEYS * 2 + REGIONS * 8 + REGIONS * NODE + GLOBAL;
pub const ILLEGAL: u16 = u16::MAX;

/// Une IA qui cherche, et rend les visites de la racine.
pub trait Searcher: Sync {
    fn name(&self) -> String;
    fn search(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Searched;
}

impl Searcher for crate::az::Az {
    fn name(&self) -> String {
        crate::agent::Agent::name(self)
    }

    fn search(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Searched {
        crate::az::Az::search(self, g, k, rng)
    }
}

impl Searcher for crate::mcts::Mcts {
    fn name(&self) -> String {
        crate::agent::Agent::name(self)
    }

    fn search(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Searched {
        let (mv, stats) = crate::mcts::Mcts::search(self, g, k, rng);
        let total: u32 = stats.iter().map(|s| s.1).sum();
        let value = stats.iter().map(|s| s.2 * s.1 as f64).sum::<f64>() / total.max(1) as f64;
        let visits = if stats.is_empty() {
            moves::legal(g).iter().map(|c| (c.key, 1)).collect()
        } else {
            stats.iter().map(|s| (s.0, s.1)).collect()
        };
        Searched { mv, visits, value: value as f32 }
    }
}

struct Pending {
    enc: Encoded,
    actor: usize,
    coins: [i32; MAX_PLAYERS],
    visits: [u16; KEYS],
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Les coups sont tirés au sort selon les visites jusqu'à cette manche comprise (exploration).
    pub sample_rounds: i32,
}

/// Joue une partie ; écrit un enregistrement par position cherchée (plus d'un coup possible).
pub fn play(
    table: &Table,
    seed: u32,
    searcher: &dyn Searcher,
    opts: Options,
    rng: &mut Rng,
    out: &mut impl Write,
) -> std::io::Result<(Outcome, usize)> {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut pending: Vec<Pending> = Vec::new();
    while g.state.phase != Phase::GameOver {
        let legal = moves::legal(&g);
        let mv = if legal.len() == 1 {
            legal[0].mv
        } else {
            let s = searcher.search(&g, &k, rng);
            let actor = g.state.actor().expect("un joueur doit agir");
            let mut visits = [ILLEGAL; KEYS];
            for c in &legal {
                visits[c.key as usize] = 0;
            }
            for &(key, n) in &s.visits {
                if visits[key as usize] != ILLEGAL {
                    visits[key as usize] = n.min(60_000) as u16;
                }
            }
            let mut coins = [0; MAX_PLAYERS];
            for (q, c) in coins.iter_mut().enumerate().take(g.state.player_count()) {
                *c = g.state.players[q].coins;
            }
            pending.push(Pending { enc: encode(&g, k.race_tail), actor, coins, visits });
            if g.state.round <= opts.sample_rounds {
                let total: u32 = s.visits.iter().map(|v| v.1).sum();
                let mut pick = rng.below(total.max(1) as usize) as u32;
                let mut chosen = s.mv;
                for &(key, n) in &s.visits {
                    if pick < n {
                        chosen = legal.iter().find(|c| c.key == key).map_or(s.mv, |c| c.mv);
                        break;
                    }
                    pick -= n;
                }
                chosen
            } else {
                s.mv
            }
        };
        let mut ret = Returns::default();
        moves::play(&mut g, &mv, &mut ret).expect("coup possible");
        k.observe(ret.0, &g.state);
    }
    let o = Outcome::of(&g);
    let n = o.players;
    let mut buf = Vec::with_capacity(RECORD);
    for p in &pending {
        buf.clear();
        buf.extend_from_slice(&[n as u8, p.enc.regions as u8, 1, 0]);
        for r in 0..MAX_PLAYERS {
            let w = if r < n { o.win[(p.actor + r) % n] as f32 } else { 0.0 };
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for r in 0..MAX_PLAYERS {
            let q = (p.actor + r) % n;
            let gain = if r < n { (o.coins[q] - p.coins[q]) as f32 } else { 0.0 };
            buf.extend_from_slice(&gain.to_le_bytes());
        }
        for v in p.visits {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for i in 0..REGIONS {
            let mut bits = 0u64;
            if i < table.map.len() {
                for &a in &table.map.regions[i].adj {
                    bits |= 1 << a;
                }
            }
            buf.extend_from_slice(&bits.to_le_bytes());
        }
        for row in &p.enc.node {
            buf.extend_from_slice(row);
        }
        buf.extend_from_slice(&p.enc.global);
        debug_assert_eq!(buf.len(), RECORD);
        out.write_all(&buf)?;
    }
    Ok((o, pending.len()))
}

pub struct Config {
    pub players: Vec<usize>,
    pub games: usize,
    pub threads: usize,
    pub seed: u64,
    pub opts: Options,
}

/// Des parties d'entraînement en parallèle, un fichier par fil dans `dir`.
pub fn run(cfg: &Config, searcher: &dyn Searcher, dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let next = AtomicUsize::new(0);
    let positions = AtomicUsize::new(0);
    let start = Instant::now();
    std::thread::scope(|scope| -> std::io::Result<()> {
        let mut handles = Vec::new();
        for t in 0..cfg.threads {
            let (next, positions) = (&next, &positions);
            handles.push(scope.spawn(move || -> std::io::Result<()> {
                let path = dir.join(format!("s{:x}-{t:02}.bin", cfg.seed));
                let mut out = BufWriter::new(File::create(path)?);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cfg.games {
                        break;
                    }
                    let n = cfg.players[i % cfg.players.len()];
                    let deal = mix(cfg.seed, i as u64);
                    let table = Table::new(n, deal % 1_000_000_007);
                    let mut rng = Rng::new(mix(deal, 99));
                    let (_, count) = play(&table, mix(deal, 1) as u32, searcher, cfg.opts, &mut rng, &mut out)?;
                    let total = positions.fetch_add(count, Ordering::Relaxed) + count;
                    let secs = start.elapsed().as_secs_f64();
                    eprintln!(
                        "partie {}/{} ({n} j.) : {count} positions, {total} en tout, {:.0} positions/s",
                        i + 1,
                        cfg.games,
                        total as f64 / secs
                    );
                }
                out.flush()
            }));
        }
        for h in handles {
            h.join().expect("fil d'auto-apprentissage")?;
        }
        Ok(())
    })
}

/// Relit un enregistrement : la position encodée et les voisines de chaque région.
pub fn read_record(rec: &[u8]) -> (Encoded, Vec<Vec<u8>>) {
    assert_eq!(rec.len(), RECORD);
    let players = rec[0] as usize;
    let regions = rec[1] as usize;
    let mut at = 4 + 20 + 20 + KEYS * 2;
    let mut adj = Vec::with_capacity(regions);
    for i in 0..REGIONS {
        let bits = u64::from_le_bytes(rec[at..at + 8].try_into().expect("8 octets"));
        at += 8;
        if i < regions {
            adj.push((0..64u8).filter(|&b| bits >> b & 1 == 1).collect());
        }
    }
    let mut e = Encoded { regions, players, node: [[0; NODE]; REGIONS], global: [0; GLOBAL] };
    for row in e.node.iter_mut() {
        row.copy_from_slice(&rec[at..at + NODE]);
        at += NODE;
    }
    e.global.copy_from_slice(&rec[at..at + GLOBAL]);
    (e, adj)
}
