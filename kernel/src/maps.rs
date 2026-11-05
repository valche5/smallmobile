//! Les cartes du jeu : `COUNT` cartes par nombre de joueurs, générées une fois pour toutes par `mapgen`
//! et rangées dans `kernel/maps.bin` (`build` ; `cargo run --release -p smallmobile-kernel --example maps`
//! le réécrit). Le jeu ne génère plus de carte : le serveur garde le fichier en mémoire, et le noyau
//! décode la carte demandée (`decode`).
//!
//! La carte d'une graine est celle de rang `graine % COUNT`, générée pour la graine
//! `1000 × joueurs + rang` (`seed_of`) : les cartes classiques (2024, 3033, 4044, 5055) en font partie.
//!
//! Le fichier, petit-boutiste : `MAGIC`, puis en u32 le début de chaque carte (2 à 5 joueurs, rangs 0
//! à `COUNT` − 1) et la fin du fichier, puis les cartes. Une carte : sa graine (u32), nombre de joueurs,
//! tours et nombre de régions (u8), puis par région : terrain, drapeaux (caverne, magie, mine, tribu oubliée, bord,
//! entrée, côte), centre (x, y en u16), nom (0 tiré : nom commun et complément ; 1 numéroté : numéro
//! et 0), voisines (nombre puis rangs), contour (nombre de points en u16, puis x et y en dixièmes,
//! u16).

use crate::ids::Terrain;
use crate::mapgen::{self, GeneratedMap, GeneratedRegion, Name};

/// Cartes par nombre de joueurs.
pub const COUNT: usize = 100;
pub const MAGIC: [u8; 4] = *b"SMM1";
pub const PLAYERS: std::ops::RangeInclusive<usize> = 2..=5;

/// Le fichier des cartes du jeu.
pub static FILE: &[u8] = include_bytes!("../maps.bin");

/// La graine pour laquelle a été générée la carte de cette graine.
pub fn seed_of(players: usize, seed: u64) -> u64 {
    1000 * players as u64 + seed % COUNT as u64
}

/// Les octets de la carte de cette graine dans `file`.
pub fn record(file: &[u8], players: usize, seed: u64) -> Option<&[u8]> {
    if !PLAYERS.contains(&players) || file.get(..4)? != MAGIC {
        return None;
    }
    let slot = (players - PLAYERS.start()) * COUNT + (seed % COUNT as u64) as usize;
    let offset = |i: usize| Some(u32::from_le_bytes(file.get(4 + 4 * i..8 + 4 * i)?.try_into().ok()?) as usize);
    file.get(offset(slot)?..offset(slot + 1)?)
}

/// La carte de cette graine (`FILE`).
pub fn get(players: usize, seed: u64) -> Option<GeneratedMap> {
    decode(record(FILE, players, seed)?)
}

/// Le fichier entier, généré : les cartes de chaque nombre de joueurs, toutes différentes.
pub fn build() -> Vec<u8> {
    let slots = PLAYERS.count() * COUNT;
    let mut records = Vec::new();
    let mut starts = Vec::with_capacity(slots + 1);
    let header = MAGIC.len() + 4 * (slots + 1);
    for players in PLAYERS {
        let mut seen = std::collections::HashSet::new();
        for rank in 0..COUNT as u64 {
            let seed = seed_of(players, rank);
            let map = mapgen::generate(players, seed).expect("carte générée");
            let shape: Vec<_> = map.regions.iter().map(|r| (r.terrain, r.cx, r.cy, r.adj.clone())).collect();
            assert!(seen.insert(shape), "deux cartes identiques à {players} joueurs (graine {seed})");
            starts.push((header + records.len()) as u32);
            encode(&map, &mut records);
        }
    }
    starts.push((header + records.len()) as u32);
    let mut out = MAGIC.to_vec();
    out.extend(starts.iter().flat_map(|s| s.to_le_bytes()));
    out.extend(records);
    out
}

/// Une carte, au format du fichier ; contour arrondi au dixième.
pub fn encode(m: &GeneratedMap, out: &mut Vec<u8>) {
    let u16_of = |v: i64| u16::try_from(v).expect("valeur hors du format des cartes");
    out.extend(u32::try_from(m.seed).expect("graine sur 32 bits").to_le_bytes());
    out.extend([m.players as u8, u8::try_from(m.rounds).unwrap(), u8::try_from(m.regions.len()).unwrap()]);
    for r in &m.regions {
        let flags = r.cavern as u8
            | (r.magic as u8) << 1
            | (r.mine as u8) << 2
            | (r.lost_tribe as u8) << 3
            | (r.border as u8) << 4
            | (r.entry as u8) << 5
            | (r.coastal as u8) << 6;
        out.extend([r.terrain.index() as u8, flags]);
        out.extend(u16_of(r.cx as i64).to_le_bytes());
        out.extend(u16_of(r.cy as i64).to_le_bytes());
        let (kind, a, b) = match r.name {
            Name::Drawn { noun, suffix } => (0, noun, suffix),
            Name::Numbered(n) => (1, n, 0),
        };
        out.extend([kind, u8::try_from(a).unwrap(), u8::try_from(b).unwrap(), r.adj.len() as u8]);
        out.extend(&r.adj);
        out.extend(u16_of(r.outline.len() as i64).to_le_bytes());
        for &(x, y) in &r.outline {
            for v in [x, y] {
                out.extend(u16_of((v * 10.0).round() as i64).to_le_bytes());
            }
        }
    }
}

/// Une carte du fichier ; `None` si les octets ne sont pas une carte valide.
pub fn decode(bytes: &[u8]) -> Option<GeneratedMap> {
    let mut r = Bytes(bytes);
    let seed = r.u32()? as u64;
    let players = r.u8()? as usize;
    let rounds = r.u8()? as i32;
    let n = r.u8()? as usize;
    let mut regions = Vec::with_capacity(n);
    for _ in 0..n {
        let terrain = Terrain::from_index(r.u8()? as usize)?;
        let flags = r.u8()?;
        if flags >= 1 << 7 {
            return None;
        }
        let (cx, cy) = (r.u16()? as i32, r.u16()? as i32);
        let (kind, a, b) = (r.u8()?, r.u8()? as u32, r.u8()? as u32);
        let name = match kind {
            0 => Name::Drawn { noun: a, suffix: b },
            1 => Name::Numbered(a),
            _ => return None,
        };
        let degree = r.u8()? as usize;
        let adj = r.take(degree)?.to_vec();
        if adj.iter().any(|&a| a as usize >= n) {
            return None;
        }
        let points = r.u16()? as usize;
        let outline = (0..points).map(|_| Some((r.u16()? as f64 / 10.0, r.u16()? as f64 / 10.0))).collect::<Option<_>>()?;
        let flag = |bit: u8| flags & 1 << bit != 0;
        regions.push(GeneratedRegion {
            terrain,
            cavern: flag(0),
            magic: flag(1),
            mine: flag(2),
            lost_tribe: flag(3),
            border: flag(4),
            entry: flag(5),
            coastal: flag(6),
            adj,
            outline,
            cx,
            cy,
            name,
        });
    }
    if !r.0.is_empty() {
        return None;
    }
    Some(GeneratedMap { seed, players, rounds, regions })
}

struct Bytes<'a>(&'a [u8]);

impl<'a> Bytes<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_holds_the_generated_maps() {
        assert!(FILE == build(), "kernel/maps.bin périmé : cargo run --release -p smallmobile-kernel --example maps");
    }

    #[test]
    fn maps_read_back_as_generated() {
        for players in PLAYERS {
            for seed in [0, 24, 33, 99, 2024, 123_456_789] {
                let map = get(players, seed).unwrap();
                let generated = mapgen::generate(players, seed_of(players, seed)).unwrap();
                assert_eq!((map.seed, map.players, map.rounds), (generated.seed, generated.players, generated.rounds));
                assert_eq!(map.to_map().unwrap().regions, generated.to_map().unwrap().regions);
                for (a, b) in map.regions.iter().zip(&generated.regions) {
                    assert_eq!((a.lost_tribe, a.cx, a.cy, a.name), (b.lost_tribe, b.cx, b.cy, b.name));
                    assert_eq!(a.outline.len(), b.outline.len());
                    for (p, q) in a.outline.iter().zip(&b.outline) {
                        assert!((p.0 - q.0).abs() <= 0.05 + 1e-9 && (p.1 - q.1).abs() <= 0.05 + 1e-9);
                    }
                }
            }
        }
        assert_eq!(seed_of(2, 2024), 2024);
        assert!(get(6, 0).is_none());
        assert!(decode(&record(FILE, 3, 7).unwrap()[1..]).is_none());
    }
}
