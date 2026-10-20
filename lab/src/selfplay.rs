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
//! u8 × GLOBAL  entrées de la partie (GLOBAL_V1 avant la version 4)
//! f32 × 5  pièces que la recherche attendait encore pour chaque joueur relatif (NaN : inconnu)
//! f32 × 5  part de victoire que la recherche attendait pour chaque joueur relatif (NaN : inconnu)
//! ```
//!
//! Les fichiers `*.v4.bin` ont tout, avec l'encodage de la version 2 (`encode`) et les clés de paix
//! relatives (`moves::FINISH`) ; les `*.v3.bin` ont l'encodage et les clés de la version 1 ; les
//! `*.v2.bin` s'arrêtent de plus avant la dernière ligne, les `*.bin` avant les deux dernières.
//! `lab reencode` passe des fichiers de la version 3 à la version 4. À côté de chaque fichier :
//! `games_file` (positions de chaque partie) et, depuis la version 3, `moves_file` (de quoi rejouer
//! chaque partie, `GameLog`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::wire::{self, Writer};
use smallmobile_kernel::{Action, Game, Phase};

use crate::az::Searched;
use crate::encode::{Encoded, GLOBAL, GLOBAL_V1, NODE, REGIONS, encode};
use crate::know::{Knowledge, Returns};
use crate::moves::{self, KEYS};
use crate::rand::{Rng, mix};
use crate::table::{Outcome, Table};

pub const RECORD_V1: usize = 4 + 20 + 20 + KEYS * 2 + REGIONS * 8 + REGIONS * NODE + GLOBAL_V1;
pub const RECORD_V2: usize = RECORD_V1 + 20;
pub const RECORD_V3: usize = RECORD_V2 + 20;
pub const RECORD: usize = RECORD_V3 - GLOBAL_V1 + GLOBAL;
/// Le suffixe des fichiers de positions écrits aujourd'hui.
pub const SUFFIX: &str = ".v4.bin";
pub const ILLEGAL: u16 = u16::MAX;

/// Une IA qui cherche, et rend les visites de la racine.
pub trait Searcher: Sync {
    fn name(&self) -> String;
    /// Itérations d'une recherche complète.
    fn iterations(&self) -> usize;
    /// `explore` : recherche complète d'auto-apprentissage (bruit à la racine si l'IA en a).
    fn search(&self, g: &Game, k: &Knowledge, rng: &mut Rng, iterations: usize, explore: bool) -> Searched;
}

impl Searcher for crate::az::Az {
    fn name(&self) -> String {
        crate::agent::Agent::name(self)
    }

    fn iterations(&self) -> usize {
        self.cfg.iterations
    }

    fn search(&self, g: &Game, k: &Knowledge, rng: &mut Rng, iterations: usize, explore: bool) -> Searched {
        let noise = if explore { self.cfg.noise } else { 0.0 };
        self.search_with(g, k, rng, iterations, noise)
    }
}

impl Searcher for crate::mcts::Mcts {
    fn name(&self) -> String {
        crate::agent::Agent::name(self)
    }

    fn iterations(&self) -> usize {
        self.cfg.iterations
    }

    fn search(&self, g: &Game, k: &Knowledge, rng: &mut Rng, iterations: usize, _: bool) -> Searched {
        let (mv, stats) = self.search_with(g, k, rng, iterations);
        let total: u32 = stats.iter().map(|s| s.1).sum();
        let value = stats.iter().map(|s| s.2 * s.1 as f64).sum::<f64>() / total.max(1) as f64;
        let visits = if stats.is_empty() {
            moves::legal(g).iter().map(|c| (c.key, 1)).collect()
        } else {
            stats.iter().map(|s| (s.0, s.1)).collect()
        };
        let unknown = [f32::NAN; MAX_PLAYERS];
        Searched { mv, visits, finals: unknown, wins: unknown, details: vec![], value: value as f32 }
    }
}

/// L'IA du jeu comme « recherche » : une seule visite, sur son coup. Ses parties ne coûtent presque
/// rien : elles apprennent au réseau à juger une position, et donnent un premier a priori des coups.
/// Sa fin de tour est le coup `Auto`, sa défense le premier plan.
impl Searcher for crate::agent::KernelAi {
    fn name(&self) -> String {
        crate::agent::Agent::name(self)
    }

    fn iterations(&self) -> usize {
        1
    }

    fn search(&self, g: &Game, k: &Knowledge, rng: &mut Rng, _: usize, _: bool) -> Searched {
        let legal = moves::legal(g);
        let chosen = match g.state.phase {
            Phase::Redeploy => legal.iter().find(|c| c.mv == moves::Move::Auto),
            Phase::Defend => legal.first(),
            _ => {
                let mv = crate::agent::Agent::act(self, g, k, rng);
                legal.iter().find(|c| c.mv == mv)
            }
        };
        let c = chosen.expect("le coup de l'IA du jeu est parmi les coups possibles");
        Searched {
            mv: c.mv,
            visits: vec![(c.key, 1)],
            finals: [f32::NAN; MAX_PLAYERS],
            wins: [f32::NAN; MAX_PLAYERS],
            details: vec![],
            value: f32::NAN,
        }
    }
}

/// Un exemple pour le réseau, cibles en joueurs relatifs (0 : celui qui agit).
pub struct Record<'a> {
    pub enc: &'a Encoded,
    pub map: &'a smallmobile_kernel::Map,
    pub win: [f32; MAX_PLAYERS],
    pub gain: [f32; MAX_PLAYERS],
    pub visits: &'a [u16; KEYS],
    pub qgain: [f32; MAX_PLAYERS],
    pub qwin: [f32; MAX_PLAYERS],
}

impl Record<'_> {
    pub fn bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(RECORD);
        buf.extend_from_slice(&[self.enc.players as u8, self.enc.regions as u8, 1, 0]);
        for x in self.win.iter().chain(&self.gain) {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for v in self.visits {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for i in 0..REGIONS {
            let mut bits = 0u64;
            if i < self.map.len() {
                for &a in &self.map.regions[i].adj {
                    bits |= 1 << a;
                }
            }
            buf.extend_from_slice(&bits.to_le_bytes());
        }
        for row in &self.enc.node {
            buf.extend_from_slice(row);
        }
        buf.extend_from_slice(&self.enc.global);
        for x in self.qgain.iter().chain(&self.qwin) {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        debug_assert_eq!(buf.len(), RECORD);
        buf
    }
}

struct Pending {
    enc: Encoded,
    actor: usize,
    coins: [i32; MAX_PLAYERS],
    visits: [u16; KEYS],
    /// Pièces finales et part de victoire attendues par la recherche (joueurs réels).
    finals: [f32; MAX_PLAYERS],
    wins: [f32; MAX_PLAYERS],
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Les coups sont tirés au sort selon les visites jusqu'à cette manche comprise (exploration).
    pub sample_rounds: i32,
    /// Part des coups cherchés à fond (et enregistrés) ; les autres le sont en `fast` itérations
    /// (« playout cap randomization » de KataGo : plus de parties pour le même calcul).
    pub full_share: f64,
    pub fast: usize,
}

/// Une partie jouée : son résultat, le nombre de positions écrites, et les actions du noyau de
/// chaque coup.
pub struct Played {
    pub outcome: Outcome,
    pub positions: usize,
    pub moves: Vec<Vec<Action>>,
}

/// Joue une partie ; écrit un enregistrement par position cherchée (plus d'un coup possible).
pub fn play(
    table: &Table,
    seed: u32,
    searcher: &dyn Searcher,
    opts: Options,
    rng: &mut Rng,
    out: &mut impl Write,
) -> std::io::Result<Played> {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut pending: Vec<Pending> = Vec::new();
    let mut played = Vec::new();
    while g.state.phase != Phase::GameOver {
        let legal = moves::legal(&g);
        let full = rng.unit() < opts.full_share;
        let mv = if legal.len() == 1 {
            legal[0].mv
        } else if !full {
            searcher.search(&g, &k, rng, opts.fast, false).mv
        } else {
            let s = searcher.search(&g, &k, rng, searcher.iterations(), true);
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
            pending.push(Pending {
                enc: encode(&g, k.race_tail),
                actor,
                coins,
                visits,
                finals: s.finals,
                wins: s.wins,
            });
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
        played.push(moves::actions(&g, &mv).expect("coup découpé en actions"));
        let mut ret = Returns::default();
        moves::play(&mut g, &mv, &mut ret).expect("coup possible");
        k.observe(ret.0, &g.state);
    }
    let o = Outcome::of(&g);
    let n = o.players;
    for p in &pending {
        let mut win = [0.0; MAX_PLAYERS];
        let mut gain = [0.0; MAX_PLAYERS];
        let (mut qgain, mut qwin) = ([f32::NAN; MAX_PLAYERS], [f32::NAN; MAX_PLAYERS]);
        for r in 0..n {
            let q = (p.actor + r) % n;
            win[r] = o.win[q] as f32;
            gain[r] = (o.coins[q] - p.coins[q]) as f32;
            qgain[r] = p.finals[q] - p.coins[q] as f32;
            qwin[r] = p.wins[q];
        }
        let rec = Record { enc: &p.enc, map: &table.map, win, gain, visits: &p.visits, qgain, qwin };
        out.write_all(&rec.bytes())?;
    }
    Ok(Played { outcome: o, positions: pending.len(), moves: played })
}

/// De quoi rejouer une partie : la donne et les actions du noyau de chaque coup. Une ligne par
/// partie dans `moves_file`, séparée par des tabulations : donne, joueurs, graine de la carte,
/// graine de la partie, pièces finales (séparées par des virgules), puis les coups séparés par des
/// espaces ; un coup est une suite d'actions séparées par « / », une action ses mots `wire`
/// séparés par des virgules.
#[derive(Clone, Debug, PartialEq)]
pub struct GameLog {
    pub deal: u64,
    pub players: usize,
    pub map_seed: u64,
    pub seed: u32,
    pub coins: Vec<i32>,
    /// Les mots de chaque action de chaque coup.
    pub moves: Vec<Vec<Vec<u32>>>,
}

impl GameLog {
    pub fn new(deal: u64, map_seed: u64, seed: u32, table: &Table, played: &Played) -> GameLog {
        let regions = table.map.len();
        let mut buffer = vec![0u32; 8 + regions];
        let moves = played
            .moves
            .iter()
            .map(|actions| {
                actions
                    .iter()
                    .map(|a| {
                        let mut w = Writer::new(&mut buffer);
                        wire::encode_action(a, regions, &mut w);
                        w.written().to_vec()
                    })
                    .collect()
            })
            .collect();
        let coins = played.outcome.coins[..table.players].to_vec();
        GameLog { deal, players: table.players, map_seed, seed, coins, moves }
    }

    pub fn line(&self) -> String {
        let join = |xs: &[u32]| xs.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let coins: Vec<String> = self.coins.iter().map(i32::to_string).collect();
        let moves: Vec<String> =
            self.moves.iter().map(|m| m.iter().map(|a| join(a)).collect::<Vec<_>>().join("/")).collect();
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.deal,
            self.players,
            self.map_seed,
            self.seed,
            coins.join(","),
            moves.join(" ")
        )
    }

    pub fn parse(line: &str) -> Option<GameLog> {
        let mut f = line.split('\t');
        let (deal, players, map_seed, seed) =
            (f.next()?.parse().ok()?, f.next()?.parse().ok()?, f.next()?.parse().ok()?, f.next()?.parse().ok()?);
        let coins = f.next()?.split(',').map(|c| c.parse().ok()).collect::<Option<Vec<i32>>>()?;
        let words = |a: &str| a.split(',').map(|w| w.parse().ok()).collect::<Option<Vec<u32>>>();
        let moves = f.next()?.split(' ').map(|m| m.split('/').map(words).collect()).collect::<Option<Vec<_>>>()?;
        Some(GameLog { deal, players, map_seed, seed, coins, moves })
    }

    /// La table de la partie.
    pub fn table(&self) -> Table {
        Table::new(self.players, self.map_seed)
    }

    /// Rejoue la partie sur sa table : `visit` voit chaque position avant un coup, avec ce que sait
    /// le joueur qui agit. Renvoie la partie finie.
    pub fn replay<'m>(
        &self,
        table: &'m Table,
        mut visit: impl FnMut(&Game<'m>, &Knowledge),
    ) -> Result<Game<'m>, String> {
        let mut g = table.game(self.seed);
        let mut k = Knowledge::default();
        for mv in &self.moves {
            visit(&g, &k);
            let mut ret = Returns::default();
            for words in mv {
                let a =
                    wire::decode_action(words, table.map.len()).map_err(|e| format!("action illisible : {}", e.0))?;
                g.apply(&a, &mut ret).map_err(|e| format!("action refusée {a:?} : {e:?}"))?;
            }
            k.observe(ret.0, &g.state);
        }
        Ok(g)
    }
}

pub struct Config {
    pub players: Vec<usize>,
    pub games: usize,
    pub threads: usize,
    pub seed: u64,
    pub opts: Options,
}

/// Le fichier qui accompagne un fichier de positions : le nombre de positions de chaque partie,
/// dans l'ordre (u32 petit-boutiste ; `GAMES` dans `lab/py/data.py`).
pub fn games_file(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".games");
    name.into()
}

/// Le fichier qui accompagne un fichier de positions (version 3) : une ligne `GameLog` par partie.
pub fn moves_file(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".moves");
    name.into()
}

/// Des parties d'entraînement en parallèle, un fichier par fil dans `dir` (avec `games_file` et
/// `moves_file`).
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
                let path = dir.join(format!("s{:x}-{t:02}{SUFFIX}", cfg.seed));
                let mut out = BufWriter::new(File::create(&path)?);
                let mut games = BufWriter::new(File::create(games_file(&path))?);
                let mut logs = BufWriter::new(File::create(moves_file(&path))?);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cfg.games {
                        break;
                    }
                    let n = cfg.players[i % cfg.players.len()];
                    let deal = mix(cfg.seed, i as u64);
                    let (map_seed, seed) = (deal % 1_000_000_007, mix(deal, 1) as u32);
                    let table = Table::new(n, map_seed);
                    let mut rng = Rng::new(mix(deal, 99));
                    let played = play(&table, seed, searcher, cfg.opts, &mut rng, &mut out)?;
                    let count = played.positions;
                    games.write_all(&(count as u32).to_le_bytes())?;
                    writeln!(logs, "{}", GameLog::new(deal, map_seed, seed, &table, &played).line())?;
                    let total = positions.fetch_add(count, Ordering::Relaxed) + count;
                    let secs = start.elapsed().as_secs_f64();
                    eprintln!(
                        "partie {}/{} ({n} j.) : {count} positions, {total} en tout, {:.0} positions/s",
                        i + 1,
                        cfg.games,
                        total as f64 / secs
                    );
                }
                out.flush()?;
                logs.flush()?;
                games.flush()
            }));
        }
        for h in handles {
            h.join().expect("fil d'auto-apprentissage")?;
        }
        Ok(())
    })
}

/// Relit un enregistrement : la position encodée et les voisines de chaque région. Avant la version
/// 4, les entrées de la version 2 restent nulles, et la place du joueur qui agit est inconnue (0).
pub fn read_record(rec: &[u8]) -> (Encoded, Vec<Vec<u8>>) {
    assert!([RECORD_V1, RECORD_V2, RECORD_V3, RECORD].contains(&rec.len()));
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
    let mut e = Encoded { regions, players, seat: 0, node: [[0; NODE]; REGIONS], global: [0; GLOBAL] };
    for row in e.node.iter_mut() {
        row.copy_from_slice(&rec[at..at + NODE]);
        at += NODE;
    }
    let global = if rec.len() == RECORD { GLOBAL } else { GLOBAL_V1 };
    e.global[..global].copy_from_slice(&rec[at..at + global]);
    (e, adj)
}
