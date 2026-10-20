//! Passer des positions enregistrées (`*.v3.bin`, encodage de la version 1 ; `*.v4.bin`, version 2)
//! à l'encodage d'aujourd'hui (`*.v5.bin`, version 3), sans aucune recherche : les parties se
//! rejouent depuis leurs fichiers `.moves`, et chaque position enregistrée est retrouvée par son
//! entrée de la version 1. Les cibles ne changent pas ; les visites des fichiers de la version 3
//! passent aux clés de paix relatives (`moves::relative_peace`) ; chaque position reçoit la clé du
//! coup joué ensuite dans la partie (`GameLog::played`). Vaut pour les positions de `lab selfplay`
//! comme pour celles de `lab judge`, à condition d'avoir les parties d'où elles viennent.
//!
//! Avec `observe`, les parties à tables variées d'un fichier (son `.moves`) donnent en plus les
//! coups des autres joueurs (`selfplay::Record::observed`), à leur place dans chaque partie : comme
//! si `lab selfplay` les avait enregistrés (`selfplay::Options::observe`).

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use smallmobile_kernel::Game;
use smallmobile_kernel::ids::MAX_PLAYERS;

use crate::encode::{Encoded, GLOBAL_V1, GLOBAL_V2, NODE, REGIONS, encode};
use crate::moves::{self, KEYS};
use crate::selfplay::{GameLog, ILLEGAL, RECORD, RECORD_V3, RECORD_V4, Record, SUFFIX, games_file, moves_file};
use crate::table::Outcome;

/// Ce que le réseau de la version 1 voyait d'une position : joueurs, régions, entrées par région,
/// les `GLOBAL_V1` entrées de la partie (le début de celles des versions suivantes).
fn old_input(players: usize, regions: usize, node: &[[u8; NODE]; REGIONS], global: &[u8]) -> Vec<u8> {
    let mut x = Vec::with_capacity(2 + REGIONS * NODE + GLOBAL_V1);
    x.extend_from_slice(&[players as u8, regions as u8]);
    node.iter().for_each(|row| x.extend_from_slice(row));
    x.extend_from_slice(&global[..GLOBAL_V1]);
    x
}

fn hash(bytes: &[u8]) -> u128 {
    let (mut a, mut b) = (0x243f_6a88_85a3_08d3u64, 0x1319_8a2e_0370_7344u64);
    for c in bytes.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        let w = u64::from_le_bytes(w);
        a = (a.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
        b = (b.rotate_left(29) ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    (a as u128) << 64 | b as u128
}

/// Où se trouve chaque position où un joueur avait le choix : (partie, coup).
fn index(logs: &[GameLog]) -> io::Result<HashMap<u128, (u32, u32)>> {
    let mut at = HashMap::new();
    for (i, log) in logs.iter().enumerate() {
        let table = log.table();
        let mut step = 0u32;
        log.replay(&table, |g, k| {
            if moves::legal(g).len() > 1 {
                let e = encode(g, k);
                at.entry(hash(&old_input(e.players, e.regions, &e.node, &e.global))).or_insert((i as u32, step));
            }
            step += 1;
        })
        .map_err(io::Error::other)?;
    }
    Ok(at)
}

/// Le format d'un fichier de positions à réencoder : taille d'un enregistrement, entrées de la
/// partie, clés de paix de la version 1 (place absolue de la cible).
fn format(path: &Path) -> Option<(usize, usize, bool)> {
    let name = path.to_string_lossy();
    if name.ends_with(".v3.bin") {
        Some((RECORD_V3, GLOBAL_V1, true))
    } else if name.ends_with(".v4.bin") {
        Some((RECORD_V4, GLOBAL_V2, false))
    } else {
        None
    }
}

/// Les fichiers de positions des versions 3 et 4 sous `dir`.
fn files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            files(&path, out)?;
        } else if format(&path).is_some() {
            out.push(path);
        }
    }
    Ok(())
}

/// Réencode les positions de `data` (fichiers `*.v3.bin` et `*.v4.bin`, dans des sous-dossiers ou
/// non) venues des parties `logs`, vers `out` (même arborescence, `*.v5.bin`, avec `.games` et
/// `.moves`) ; `observe` : voir l'en-tête. Renvoie le nombre de fichiers et de positions écrites.
pub fn run(logs: &[GameLog], data: &Path, out: &Path, threads: usize, observe: bool) -> io::Result<(usize, usize)> {
    let at = index(logs)?;
    let mut paths = Vec::new();
    files(data, &mut paths)?;
    paths.sort();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let failure: Mutex<Option<io::Error>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = paths.get(i) else { break };
                    match file(logs, &at, data, path, out, observe) {
                        Ok(n) => {
                            done.fetch_add(n, Ordering::Relaxed);
                        }
                        Err(e) => {
                            *failure.lock().expect("verrou") =
                                Some(io::Error::other(format!("{} : {e}", path.display())));
                            break;
                        }
                    }
                }
            });
        }
    });
    match failure.into_inner().expect("verrou") {
        Some(e) => Err(e),
        None => Ok((paths.len(), done.into_inner())),
    }
}

/// Un fichier : ses positions retrouvées dans leurs parties, puis réécrites ; et avec `observe`,
/// les coups observés de ses parties à tables variées.
fn file(
    logs: &[GameLog],
    at: &HashMap<u128, (u32, u32)>,
    data: &Path,
    path: &Path,
    out: &Path,
    observe: bool,
) -> io::Result<usize> {
    let (size, global, absolute_peace) = format(path).expect("fichier des versions 3 ou 4");
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(size) {
        return Err(io::Error::other("taille qui n'est pas un multiple d'un enregistrement"));
    }
    let records: Vec<&[u8]> = bytes.chunks(size).collect();
    // chaque position : sa partie et son coup ; puis, par partie, les coups à revoir
    let node_at = 4 + 40 + KEYS * 2 + REGIONS * 8;
    let global_at = node_at + REGIONS * NODE;
    let mut wanted: HashMap<u32, Vec<(u32, usize)>> = HashMap::new();
    let mut steps_of = Vec::with_capacity(records.len());
    for (r, rec) in records.iter().enumerate() {
        let input = [&rec[..2], &rec[node_at..global_at + GLOBAL_V1]].concat();
        let &(game, step) =
            at.get(&hash(&input)).ok_or_else(|| io::Error::other(format!("position {r} introuvable")))?;
        wanted.entry(game).or_default().push((step, r));
        steps_of.push(step as usize);
    }
    let mut rewritten: Vec<Option<Vec<u8>>> = vec![None; records.len()];
    for (&game, steps) in &wanted {
        let log = &logs[game as usize];
        let table = log.table();
        let mut step = 0u32;
        log.replay(&table, |g, k| {
            for &(_, r) in steps.iter().filter(|&&(s, _)| s == step) {
                let rec = records[r];
                let e = encode(g, k);
                // l'ancienne entrée doit être exactement celle de la position retrouvée
                let old = [&rec[..2], &rec[node_at..global_at + GLOBAL_V1]].concat();
                assert_eq!(old, old_input(e.players, e.regions, &e.node, &e.global), "position mal retrouvée");
                let played = log.played(step as usize, g, &moves::legal(g), &table);
                let mut new = Vec::with_capacity(RECORD);
                new.extend_from_slice(&rec[..global_at]);
                new.extend_from_slice(&e.global);
                new.extend_from_slice(&rec[global_at + global..]);
                new.extend_from_slice(&played.to_le_bytes());
                if absolute_peace {
                    // visites : clés de paix relatives au joueur qui agit
                    let mut visits: Vec<u16> =
                        rec[44..44 + KEYS * 2].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    moves::relative_peace(&mut visits, e.seat, e.players, ILLEGAL);
                    for (k, v) in visits.iter().enumerate() {
                        new[44 + 2 * k..46 + 2 * k].copy_from_slice(&v.to_le_bytes());
                    }
                }
                debug_assert_eq!(new.len(), RECORD);
                rewritten[r] = Some(new);
            }
            step += 1;
        })
        .map_err(io::Error::other)?;
    }
    let mut rewritten = rewritten
        .into_iter()
        .enumerate()
        .map(|(r, rec)| Ok((steps_of[r], rec.ok_or_else(|| io::Error::other(format!("position {r} non réécrite")))?)));
    let target = {
        let relative = path.strip_prefix(data).map_err(io::Error::other)?.to_string_lossy().into_owned();
        out.join(relative.replace(if absolute_peace { ".v3.bin" } else { ".v4.bin" }, SUFFIX))
    };
    std::fs::create_dir_all(target.parent().expect("dossier du fichier"))?;
    let mut all = Vec::with_capacity(records.len() * RECORD);
    let own = if observe && moves_file(path).exists() { file_games(path)? } else { None };
    match own {
        // partie par partie : ses positions et les coups observés des autres joueurs, dans l'ordre
        Some((counts, games)) => {
            let mut counts_out = Vec::with_capacity(counts.len());
            for (&count, log) in counts.iter().zip(&games) {
                let mut game = (0..count)
                    .map(|_| rewritten.next().ok_or_else(|| io::Error::other("moins de positions que de parties"))?)
                    .collect::<io::Result<Vec<_>>>()?;
                game.extend(observed(log)?);
                game.sort_by_key(|&(step, _)| step);
                counts_out.push(game.len() as u32);
                game.into_iter().for_each(|(_, rec)| all.extend(rec));
            }
            if rewritten.next().is_some() {
                return Err(io::Error::other("plus de positions que n'en comptent les parties"));
            }
            let counts: Vec<u8> = counts_out.iter().flat_map(|c| c.to_le_bytes()).collect();
            std::fs::write(games_file(&target), counts)?;
        }
        None => {
            for rec in rewritten {
                all.extend(rec?.1);
            }
            std::fs::copy(games_file(path), games_file(&target))?;
        }
    }
    std::fs::write(&target, &all)?;
    if moves_file(path).exists() {
        std::fs::copy(moves_file(path), moves_file(&target))?;
    }
    Ok(all.len() / RECORD)
}

/// Le nombre de positions de chaque partie d'un fichier et ses parties (`games_file`, `moves_file`) ;
/// `None` s'ils ne se correspondent pas.
fn file_games(path: &Path) -> io::Result<Option<(Vec<u32>, Vec<GameLog>)>> {
    let counts: Vec<u32> =
        std::fs::read(games_file(path))?.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let games = std::fs::read_to_string(moves_file(path))?
        .lines()
        .map(|l| GameLog::parse(l).ok_or_else(|| io::Error::other("partie illisible")))
        .collect::<io::Result<Vec<_>>>()?;
    Ok((counts.len() == games.len()).then_some((counts, games)))
}

/// Un coup observé : son numéro dans la partie, la position, le joueur qui agit, les pièces de
/// chacun, les coups possibles (visites nulles) et la clé du coup joué.
struct Seen {
    step: usize,
    enc: Encoded,
    actor: usize,
    coins: [i32; MAX_PLAYERS],
    visits: [u16; KEYS],
    played: u16,
}

/// Les coups des autres joueurs que l'IA qui apprend, à une table variée : un enregistrement par
/// coup qui est l'un des coups possibles, sans recherche (`Record::observed`), avec son numéro.
fn observed(log: &GameLog) -> io::Result<Vec<(usize, Vec<u8>)>> {
    if log.seats.is_empty() {
        return Ok(Vec::new());
    }
    let table = log.table();
    let mut seen: Vec<Seen> = Vec::new();
    let mut step = 0;
    let end = log
        .replay(&table, |g, k| {
            let actor = g.state.actor().expect("un joueur doit agir");
            let legal = moves::legal(g);
            if legal.len() > 1 && log.seats[actor] >= 0 {
                let played = log.played(step, g, &legal, &table);
                if played != ILLEGAL {
                    let mut visits = [ILLEGAL; KEYS];
                    for c in &legal {
                        visits[c.key as usize] = 0;
                    }
                    seen.push(Seen { step, enc: encode(g, k), actor, coins: coins(g), visits, played });
                }
            }
            step += 1;
        })
        .map_err(io::Error::other)?;
    let o = Outcome::of(&end);
    let n = o.players;
    Ok(seen
        .iter()
        .map(|x| {
            let (mut win, mut gain) = ([0.0; MAX_PLAYERS], [0.0; MAX_PLAYERS]);
            for r in 0..n {
                let q = (x.actor + r) % n;
                win[r] = o.win[q] as f32;
                gain[r] = (o.coins[q] - x.coins[q]) as f32;
            }
            let unknown = [f32::NAN; MAX_PLAYERS];
            let rec = Record {
                enc: &x.enc,
                mixed: true,
                observed: true,
                played: x.played,
                map: &table.map,
                win,
                gain,
                visits: &x.visits,
                qgain: unknown,
                qwin: unknown,
            };
            (x.step, rec.bytes())
        })
        .collect())
}

fn coins(g: &Game) -> [i32; MAX_PLAYERS] {
    let mut coins = [0; MAX_PLAYERS];
    for (c, p) in coins.iter_mut().zip(&g.state.players) {
        *c = p.coins;
    }
    coins
}
