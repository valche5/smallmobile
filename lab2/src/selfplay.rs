//! Parties d'entraînement : une IA qui cherche joue contre elle-même, et chaque position cherchée à
//! fond devient un exemple pour le réseau (ce qu'elle voyait, la politique améliorée de la
//! recherche, puis le résultat final).
//!
//! Format des fichiers : des enregistrements de taille fixe (`RECORD` octets), lus en Python par
//! `numpy.fromfile` avec le type décrit dans `lab2/py/data.py` :
//!
//! ```text
//! u8 joueurs, u8 régions
//! f32 × 5  victoire de chaque joueur relatif (part, 0 à 1)
//! f32 × 5  pièces gagnées d'ici la fin par chaque joueur relatif
//! u16 × KEYS  politique améliorée de chaque coup, × `PROBABILITY` (65535 : coup impossible)
//! u8 × REGIONS × 8  voisines (un bit par région, 64 bits par région)
//! u8 × REGIONS × NODE  entrées par région
//! u8 × GLOBAL  entrées de la partie
//! ```
//!
//! Fichiers `*.pos` (`SUFFIX`). À côté de chacun : `games_file` (positions de chaque partie) et
//! `moves_file` (de quoi rejouer chaque partie, `GameLog`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::wire::{self, Writer};
use smallmobile_kernel::{Action, Game, Phase};

use crate::az::Az;
use crate::encode::{Encoded, GLOBAL, NODE, REGIONS, encode};
use crate::know::Knowledge;
use crate::moves::{self, KEYS};
use crate::rand::{Rng, mix};
use crate::table::{Outcome, Table};

pub const RECORD: usize = 2 + 20 + 20 + KEYS * 2 + REGIONS * 8 + REGIONS * NODE + GLOBAL;
/// Le suffixe des fichiers de positions.
pub const SUFFIX: &str = ".pos";
pub const ILLEGAL: u16 = u16::MAX;
/// Une probabilité de la politique améliorée est enregistrée multipliée par ce nombre.
pub const PROBABILITY: f32 = 60_000.0;

/// Un exemple pour le réseau, cibles en joueurs relatifs (0 : celui qui agit).
pub struct Record<'a> {
    pub enc: &'a Encoded,
    pub map: &'a smallmobile_kernel::Map,
    pub win: [f32; MAX_PLAYERS],
    pub gain: [f32; MAX_PLAYERS],
    pub policy: &'a [u16; KEYS],
}

impl Record<'_> {
    pub fn bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(RECORD);
        buf.extend_from_slice(&[self.enc.players as u8, self.enc.regions as u8]);
        for x in self.win.iter().chain(&self.gain) {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for v in self.policy {
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
        debug_assert_eq!(buf.len(), RECORD);
        buf
    }
}

struct Pending {
    enc: Encoded,
    actor: usize,
    coins: [i32; MAX_PLAYERS],
    policy: [u16; KEYS],
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Part des coups cherchés à fond (et enregistrés) ; les autres le sont en `fast` itérations
    /// (« playout cap randomization » de KataGo : plus de parties pour le même calcul). Toutes les
    /// recherches de l'auto-apprentissage tirent leurs coups de racine avec le bruit de Gumbel.
    pub full_share: f64,
    pub fast: usize,
}

/// La recette des données : une recherche complète (enregistrée) pour 8 coups. La valeur apprend
/// un seul résultat par partie, quel que soit le nombre de ses positions : mieux vaut beaucoup de
/// parties que beaucoup de positions par partie (AlphaGo n'en gardait qu'une par partie pour sa
/// valeur).
pub const RECIPE: Options = Options { full_share: 0.125, fast: 50 };

/// Une partie jouée : son résultat, le nombre de positions écrites, les actions du noyau de chaque
/// coup.
pub struct Played {
    pub outcome: Outcome,
    pub positions: usize,
    pub moves: Vec<Vec<Action>>,
}

/// Joue une partie ; écrit un enregistrement par position cherchée à fond (plus d'un coup
/// possible).
pub fn play(
    table: &Table,
    seed: u32,
    az: &Az,
    opts: Options,
    rng: &mut Rng,
    out: &mut impl Write,
) -> std::io::Result<Played> {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut pending: Vec<Pending> = Vec::new();
    let mut played = Vec::new();
    while g.state.phase != Phase::GameOver {
        let actor = g.state.actor().expect("un joueur doit agir");
        let legal = moves::legal(&g, &k);
        let mv = if legal.len() == 1 {
            legal[0].mv
        } else if rng.unit() >= opts.full_share {
            az.search_with(&g, &k, rng, opts.fast, true).mv
        } else {
            let s = az.search(&g, &k, rng, true);
            let mut policy = [ILLEGAL; KEYS];
            for &(key, p) in &s.policy {
                policy[key as usize] = (p * PROBABILITY).round() as u16;
            }
            pending.push(Pending { enc: encode(&g, &k), actor, coins: coins(&g), policy });
            s.mv
        };
        played.push(moves::actions(&g, &mv));
        k.play(&mut g, &mv).expect("coup possible");
    }
    let o = Outcome::of(&g);
    let n = o.players;
    for p in &pending {
        let mut win = [0.0; MAX_PLAYERS];
        let mut gain = [0.0; MAX_PLAYERS];
        for r in 0..n {
            let q = (p.actor + r) % n;
            win[r] = o.win[q] as f32;
            gain[r] = (o.coins[q] - p.coins[q]) as f32;
        }
        let rec = Record { enc: &p.enc, map: &table.map, win, gain, policy: &p.policy };
        out.write_all(&rec.bytes())?;
    }
    Ok(Played { outcome: o, positions: pending.len(), moves: played })
}

/// Les pièces de chaque joueur.
fn coins(g: &Game) -> [i32; MAX_PLAYERS] {
    let mut coins = [0; MAX_PLAYERS];
    for (c, p) in coins.iter_mut().zip(&g.state.players) {
        *c = p.coins;
    }
    coins
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
            let actions = self.actions(mv, table)?;
            // le coup de `moves` qui donne ces actions : la fin de tour reste suivie comme en jouant
            // (`Knowledge::redeploy`)
            let c = moves::legal(&g, &k)
                .into_iter()
                .find(|c| moves::actions(&g, &c.mv) == actions)
                .ok_or_else(|| format!("aucun coup ne donne {actions:?}"))?;
            k.play(&mut g, &c.mv).map_err(|e| format!("coup refusé {:?} : {e:?}", c.mv))?;
        }
        Ok(g)
    }

    /// Les actions d'un coup (`moves`) de cette partie.
    pub fn actions(&self, mv: &[Vec<u32>], table: &Table) -> Result<Vec<Action>, String> {
        mv.iter()
            .map(|words| wire::decode_action(words, table.map.len()).map_err(|e| format!("action illisible : {}", e.0)))
            .collect()
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
/// dans l'ordre (u32 petit-boutiste ; `GAMES` dans `lab2/py/data.py`).
pub fn games_file(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".games");
    name.into()
}

/// Le fichier qui accompagne un fichier de positions : une ligne `GameLog` par partie.
pub fn moves_file(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".moves");
    name.into()
}

/// Des parties d'entraînement en parallèle, un fichier par fil dans `dir` (avec `games_file` et
/// `moves_file`).
pub fn run(cfg: &Config, az: &Az, dir: &Path) -> std::io::Result<()> {
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
                    let played = play(&table, seed, az, cfg.opts, &mut rng, &mut out)?;
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

/// Relit un enregistrement : la position encodée et les voisines de chaque région.
pub fn read_record(rec: &[u8]) -> (Encoded, Vec<Vec<u8>>) {
    assert_eq!(rec.len(), RECORD);
    let players = rec[0] as usize;
    let regions = rec[1] as usize;
    let mut at = 2 + 20 + 20 + KEYS * 2;
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
