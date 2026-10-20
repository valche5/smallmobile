//! Passer des positions enregistrées (`*.v3.bin`, encodage de la version 1) à l'encodage
//! d'aujourd'hui (`*.v4.bin`, version 2), sans aucune recherche : les parties se rejouent depuis
//! leurs fichiers `.moves`, et chaque position enregistrée est retrouvée par son ancienne entrée.
//! Les cibles ne changent pas ; les visites passent aux clés de paix relatives
//! (`moves::relative_peace`). Vaut pour les positions de `lab selfplay` comme pour celles de
//! `lab judge`, à condition d'avoir les parties d'où elles viennent.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::encode::{GLOBAL_V1, NODE, REGIONS, encode};
use crate::moves::{self, KEYS};
use crate::selfplay::{GameLog, ILLEGAL, RECORD, RECORD_V3, SUFFIX, games_file, moves_file};

/// Ce que l'ancien réseau voyait d'une position : joueurs, régions, entrées par région, les
/// `GLOBAL_V1` entrées de la partie (comme dans un enregistrement de la version 3).
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
                let e = encode(g, k.race_tail);
                at.entry(hash(&old_input(e.players, e.regions, &e.node, &e.global))).or_insert((i as u32, step));
            }
            step += 1;
        })
        .map_err(io::Error::other)?;
    }
    Ok(at)
}

/// Les fichiers de positions de la version 3 sous `dir`.
fn files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            files(&path, out)?;
        } else if path.to_string_lossy().ends_with(".v3.bin") {
            out.push(path);
        }
    }
    Ok(())
}

/// Réencode les positions de `data` (fichiers `*.v3.bin`, dans des sous-dossiers ou non) venues des
/// parties `logs`, vers `out` (même arborescence, `*.v4.bin`, avec `.games` et `.moves`). Renvoie le
/// nombre de fichiers et de positions.
pub fn run(logs: &[GameLog], data: &Path, out: &Path, threads: usize) -> io::Result<(usize, usize)> {
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
                    match file(logs, &at, data, path, out) {
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

/// Un fichier : ses positions retrouvées dans leurs parties, puis réécrites.
fn file(logs: &[GameLog], at: &HashMap<u128, (u32, u32)>, data: &Path, path: &Path, out: &Path) -> io::Result<usize> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(RECORD_V3) {
        return Err(io::Error::other("taille qui n'est pas un multiple d'un enregistrement de la version 3"));
    }
    let records: Vec<&[u8]> = bytes.chunks(RECORD_V3).collect();
    // chaque position : sa partie et son coup ; puis, par partie, les coups à revoir
    let node_at = 4 + 40 + KEYS * 2 + REGIONS * 8;
    let global_at = node_at + REGIONS * NODE;
    let mut wanted: HashMap<u32, Vec<(u32, usize)>> = HashMap::new();
    for (r, rec) in records.iter().enumerate() {
        let input = [&rec[..2], &rec[node_at..global_at + GLOBAL_V1]].concat();
        let &(game, step) =
            at.get(&hash(&input)).ok_or_else(|| io::Error::other(format!("position {r} introuvable")))?;
        wanted.entry(game).or_default().push((step, r));
    }
    let mut rewritten: Vec<Option<Vec<u8>>> = vec![None; records.len()];
    for (&game, steps) in &wanted {
        let log = &logs[game as usize];
        let table = log.table();
        let mut step = 0u32;
        log.replay(&table, |g, k| {
            for &(_, r) in steps.iter().filter(|&&(s, _)| s == step) {
                let rec = records[r];
                let e = encode(g, k.race_tail);
                let mut new = Vec::with_capacity(RECORD);
                new.extend_from_slice(&rec[..node_at]);
                new.extend_from_slice(&rec[node_at..global_at]);
                new.extend_from_slice(&e.global);
                new.extend_from_slice(&rec[global_at + GLOBAL_V1..]);
                // l'ancienne entrée doit être exactement celle de la position retrouvée
                let old = [&rec[..2], &rec[node_at..global_at + GLOBAL_V1]].concat();
                assert_eq!(old, old_input(e.players, e.regions, &e.node, &e.global), "position mal retrouvée");
                // visites : clés de paix relatives au joueur qui agit
                let mut visits: Vec<u16> =
                    rec[44..44 + KEYS * 2].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                moves::relative_peace(&mut visits, e.seat, e.players, ILLEGAL);
                for (k, v) in visits.iter().enumerate() {
                    new[44 + 2 * k..46 + 2 * k].copy_from_slice(&v.to_le_bytes());
                }
                debug_assert_eq!(new.len(), RECORD);
                rewritten[r] = Some(new);
            }
            step += 1;
        })
        .map_err(io::Error::other)?;
    }
    let target = {
        let relative = path.strip_prefix(data).map_err(io::Error::other)?.to_string_lossy().into_owned();
        out.join(relative.replace(".v3.bin", SUFFIX))
    };
    std::fs::create_dir_all(target.parent().expect("dossier du fichier"))?;
    let mut all = Vec::with_capacity(records.len() * RECORD);
    for (r, rec) in rewritten.into_iter().enumerate() {
        all.extend(rec.ok_or_else(|| io::Error::other(format!("position {r} non réécrite")))?);
    }
    std::fs::write(&target, all)?;
    std::fs::copy(games_file(path), games_file(&target))?;
    if moves_file(path).exists() {
        std::fs::copy(moves_file(path), moves_file(&target))?;
    }
    Ok(records.len())
}
