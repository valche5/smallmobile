//! Génération des cartes : des cellules de Voronoï relâchées, des mers en bordure, des lacs, des
//! terrains équilibrés, des particularités et des tribus oubliées, puis des frontières irrégulières.
//! Mêmes graines, mêmes cartes que l'ancien générateur TypeScript (d3-delaunay).
//!
//! Le noyau ne contient aucun texte : le nom d'une région est donné par des numéros (nom commun du
//! terrain, complément), que l'interface traduit avec ses listes.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::geometry::voronoi::Voronoi;
use crate::ids::Terrain;
use crate::map::{Map, MapError, RegionDef};
use crate::rng::{self, Key};

pub const MAP_W: f64 = 1000.0;
pub const MAP_H: f64 = 1300.0;
/// Tolérance pour dire qu'un sommet est sur le bord du cadre.
const EPS: f64 = 0.5;
/// Les arêtes plus courtes sont fondues (deux régions qui ne se touchent qu'en un point).
const MIN_EDGE: f64 = 26.0;
/// Nombre de noms communs par terrain, et de compléments (listes de l'interface).
pub const NOUNS: [u32; Terrain::COUNT] = [8, 7, 6, 6, 6, 5, 3];
pub const SUFFIXES: u32 = 54;

/// Ce qui dépend du nombre de joueurs.
#[derive(Clone, Copy, Debug)]
pub struct Preset {
    pub regions: usize,
    pub rounds: i32,
    pub seas: usize,
    pub lakes: usize,
}

pub fn preset(players: usize) -> Option<Preset> {
    let (regions, rounds, seas, lakes) = match players {
        2 => (23, 10, 2, 1),
        3 => (30, 10, 2, 1),
        4 => (39, 9, 3, 1),
        5 => (48, 8, 3, 2),
        _ => return None,
    };
    Some(Preset { regions, rounds, seas, lakes })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Name {
    Drawn {
        noun: u32,
        suffix: u32,
    },
    /// Plus de nom libre : le premier nom commun du terrain suivi d'un numéro.
    Numbered(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedRegion {
    pub terrain: Terrain,
    pub cavern: bool,
    pub magic: bool,
    pub mine: bool,
    pub lost_tribe: bool,
    pub border: bool,
    pub entry: bool,
    pub coastal: bool,
    pub adj: Vec<u8>,
    /// Contour, frontières irrégulières comprises.
    pub outline: Vec<(f64, f64)>,
    pub cx: i32,
    pub cy: i32,
    pub name: Name,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedMap {
    pub seed: u64,
    pub players: usize,
    pub rounds: i32,
    pub regions: Vec<GeneratedRegion>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapGenError {
    NoPreset,
    EmptyCell,
    NoValidMap,
}

type Pt = (f64, f64);

struct Rand(u32);

impl Rand {
    fn next(&mut self) -> f64 {
        let (v, next) = rng::step(self.0);
        self.0 = next;
        v
    }

    fn index(&mut self, len: usize) -> usize {
        (self.next() * len as f64).floor() as usize
    }
}

/// `Math.round` de JavaScript.
fn round(x: f64) -> f64 {
    crate::ai::round_half_up(x)
}

/// Les bords du cadre touchés : 1 gauche, 2 droite, 4 haut, 8 bas.
fn on_side(p: Pt) -> u8 {
    let mut m = 0;
    if p.0 <= EPS {
        m |= 1;
    }
    if p.0 >= MAP_W - EPS {
        m |= 2;
    }
    if p.1 <= EPS {
        m |= 4;
    }
    if p.1 >= MAP_H - EPS {
        m |= 8;
    }
    m
}

fn centroid(poly: &[Pt]) -> Pt {
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..poly.len() {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % poly.len()];
        let f = x0 * y1 - x1 * y0;
        a += f;
        cx += (x0 + x1) * f;
        cy += (y0 + y1) * f;
    }
    a /= 2.0;
    if a.abs() < 1e-6 {
        return poly[0];
    }
    (cx / (6.0 * a), cy / (6.0 * a))
}

fn area(poly: &[Pt]) -> f64 {
    let mut a = 0.0;
    for i in 0..poly.len() {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % poly.len()];
        a += x0 * y1 - x1 * y0;
    }
    (a / 2.0).abs()
}

fn distance(a: Pt, b: Pt) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// Des points bien répartis : pour chacun, le meilleur de 12 tirages (le plus loin des précédents).
fn sample_points(n: usize, rand: &mut Rand) -> Vec<Pt> {
    let margin = 40.0;
    let mut pts: Vec<Pt> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut best = (0.0, 0.0);
        let mut best_d = -1.0;
        for _ in 0..12 {
            let c = (margin + rand.next() * (MAP_W - 2.0 * margin), margin + rand.next() * (MAP_H - 2.0 * margin));
            let d = pts.iter().fold(f64::INFINITY, |d, p| d.min((p.0 - c.0) * (p.0 - c.0) + (p.1 - c.1) * (p.1 - c.1)));
            if d > best_d {
                best_d = d;
                best = c;
            }
        }
        pts.push(best);
    }
    pts
}

fn cell_polygons(pts: &[Pt]) -> Result<Vec<Vec<Pt>>, MapGenError> {
    let v = Voronoi::new(pts, [0.0, 0.0, MAP_W, MAP_H]).ok_or(MapGenError::EmptyCell)?;
    (0..pts.len()).map(|i| v.cell_polygon(i).ok_or(MapGenError::EmptyCell)).collect()
}

/// Sommets partagés entre cellules, et cellules décrites par leurs sommets ; les sommets trop proches
/// sont fondus (un sommet du cadre reste en place).
fn build_topology(polys: &[Vec<Pt>]) -> (Vec<Pt>, Vec<Vec<usize>>) {
    let mut verts: Vec<Pt> = Vec::new();
    let mut index: HashMap<(i64, i64), usize> = HashMap::new();
    let cells: Vec<Vec<usize>> = polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&p| {
                    let key = (round(p.0 * 10.0) as i64, round(p.1 * 10.0) as i64);
                    *index.entry(key).or_insert_with(|| {
                        verts.push(p);
                        verts.len() - 1
                    })
                })
                .collect()
        })
        .collect();
    let mut parent: Vec<usize> = (0..verts.len()).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] == i {
            return i;
        }
        let root = find(parent, parent[i]);
        parent[i] = root;
        root
    }
    for cell in &cells {
        for i in 0..cell.len() {
            let a = find(&mut parent, cell[i]);
            let b = find(&mut parent, cell[(i + 1) % cell.len()]);
            if a == b {
                continue;
            }
            let (pa, pb) = (verts[a], verts[b]);
            if distance(pa, pb) >= MIN_EDGE {
                continue;
            }
            let (sa, sb) = (on_side(pa), on_side(pb));
            if sa != 0 && sb != 0 {
                continue; // arête du cadre : on la garde
            }
            if sa != 0 {
                parent[b] = a;
            } else if sb != 0 {
                parent[a] = b;
            } else {
                verts[a] = ((pa.0 + pb.0) / 2.0, (pa.1 + pb.1) / 2.0);
                parent[b] = a;
            }
        }
    }
    let merged = cells
        .iter()
        .map(|cell| {
            let mut out: Vec<usize> = Vec::new();
            for &v in cell {
                let r = find(&mut parent, v);
                if out.last() != Some(&r) {
                    out.push(r);
                }
            }
            while out.len() > 1 && out[0] == out[out.len() - 1] {
                out.pop();
            }
            out
        })
        .collect();
    let resolved = (0..verts.len()).map(|i| verts[find(&mut parent, i)]).collect();
    (resolved, merged)
}

/// Frontière irrégulière entre a et b : des points intermédiaires déplacés de part et d'autre.
fn displace(a: Pt, b: Pt, rand: &mut Rand, depth: u32, amp: f64, out: &mut Vec<Pt>) {
    if depth == 0 {
        return;
    }
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = dx.hypot(dy);
    if len < 6.0 {
        return;
    }
    let off = (rand.next() * 2.0 - 1.0) * amp * len;
    let m = ((a.0 + b.0) / 2.0 + (-dy / len) * off, (a.1 + b.1) / 2.0 + (dx / len) * off);
    displace(a, m, rand, depth - 1, amp, out);
    out.push(m);
    displace(m, b, rand, depth - 1, amp, out);
}

fn pick_weighted(items: &[usize], weight: impl Fn(usize) -> f64, rand: &mut Rand) -> Option<usize> {
    let total = items.iter().fold(0.0, |s, &t| s + weight(t));
    if total <= 0.0 {
        return None;
    }
    let mut r = rand.next() * total;
    for &t in items {
        r -= weight(t);
        if r <= 0.0 {
            return Some(t);
        }
    }
    items.last().copied()
}

fn shuffle(items: &[usize], rand: &mut Rand) -> Vec<usize> {
    let mut a = items.to_vec();
    for i in (1..a.len()).rev() {
        let j = rand.index(i + 1);
        a.swap(i, j);
    }
    a
}

fn try_generate(preset: Preset, seed: u32) -> Result<Option<Vec<GeneratedRegion>>, MapGenError> {
    let mut rand = Rand(seed);
    let n = preset.regions;
    let mut pts = sample_points(n, &mut rand);
    for _ in 0..4 {
        pts = cell_polygons(&pts)?.iter().map(|p| centroid(p)).collect();
    }
    let (verts, cells) = build_topology(&cell_polygons(&pts)?);
    if cells.iter().any(|c| c.len() < 3) {
        return Ok(None);
    }
    let polys: Vec<Vec<Pt>> = cells.iter().map(|c| c.iter().map(|&v| verts[v]).collect()).collect();
    let average = MAP_W * MAP_H / n as f64;
    if polys.iter().any(|p| area(p) < average * 0.38) {
        return Ok(None);
    }

    // adjacence par arêtes partagées
    let mut edge_owners: Vec<((usize, usize), Vec<usize>)> = Vec::new();
    let mut edge_index: HashMap<(usize, usize), usize> = HashMap::new();
    for (ci, cell) in cells.iter().enumerate() {
        for i in 0..cell.len() {
            let (a, b) = (cell[i], cell[(i + 1) % cell.len()]);
            let key = (a.min(b), a.max(b));
            let at = *edge_index.entry(key).or_insert_with(|| {
                edge_owners.push((key, Vec::new()));
                edge_owners.len() - 1
            });
            edge_owners[at].1.push(ci);
        }
    }
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for (_, owners) in &edge_owners {
        if owners.len() == 2 {
            adj[owners[0]].insert(owners[1]);
            adj[owners[1]].insert(owners[0]);
        }
    }
    let border: Vec<bool> = cells
        .iter()
        .map(|cell| (0..cell.len()).any(|i| on_side(verts[cell[i]]) & on_side(verts[cell[(i + 1) % cell.len()]]) != 0))
        .collect();
    let centers: Vec<Pt> = polys.iter().map(|p| centroid(p)).collect();
    let dist = |a: usize, b: usize| distance(centers[a], centers[b]);

    // mers : cellules en bordure, bien espacées
    let mut terrain: Vec<Option<Terrain>> = vec![None; n];
    let border_cells: Vec<usize> = (0..n).filter(|&i| border[i]).collect();
    let mut seas: Vec<usize> = Vec::new();
    for _ in 0..preset.seas {
        let cands: Vec<usize> = border_cells
            .iter()
            .copied()
            .filter(|&c| !seas.contains(&c) && !seas.iter().any(|&x| adj[x].contains(&c)))
            .collect();
        if cands.is_empty() {
            return Ok(None);
        }
        let c = if seas.is_empty() {
            cands[rand.index(cands.len())]
        } else {
            let mut best = (cands[0], f64::NEG_INFINITY);
            for &c in &cands {
                let nearest = seas.iter().map(|&x| dist(x, c)).fold(f64::INFINITY, f64::min);
                let d = nearest * (0.75 + rand.next() * 0.5);
                if d > best.1 {
                    best = (c, d);
                }
            }
            best.0
        };
        seas.push(c);
        terrain[c] = Some(Terrain::Sea);
    }
    for _ in 0..preset.lakes {
        let cands: Vec<usize> = (0..n)
            .filter(|&i| !border[i] && terrain[i].is_none() && !adj[i].iter().any(|&j| terrain[j].is_some()))
            .collect();
        if cands.is_empty() {
            return Ok(None);
        }
        terrain[cands[rand.index(cands.len())]] = Some(Terrain::Lake);
    }

    // les terres doivent rester d'un seul tenant, chaque région touchant au moins deux terres
    let land: Vec<usize> = (0..n).filter(|&i| terrain[i].is_none()).collect();
    let mut seen = BTreeSet::from([land[0]]);
    let mut queue = std::collections::VecDeque::from([land[0]]);
    while let Some(c) = queue.pop_front() {
        for &nb in &adj[c] {
            if terrain[nb].is_none() && seen.insert(nb) {
                queue.push_back(nb);
            }
        }
    }
    if seen.len() != land.len() {
        return Ok(None);
    }
    if land.iter().any(|&i| adj[i].iter().filter(|&&j| terrain[j].is_none()).count() < 2) {
        return Ok(None);
    }

    // terrains équilibrés, en évitant les voisins identiques
    const KINDS: [Terrain; 5] = [Terrain::Farmland, Terrain::Forest, Terrain::Hill, Terrain::Mountain, Terrain::Swamp];
    let mut counts = [0usize; 5];
    for c in shuffle(&land, &mut rand) {
        let used: Vec<Option<Terrain>> = adj[c].iter().map(|&j| terrain[j]).collect();
        let free: Vec<usize> = (0..5).filter(|&k| !used.contains(&Some(KINDS[k]))).collect();
        let pool = if free.is_empty() { (0..5).collect() } else { free };
        let min = pool.iter().map(|&k| counts[k]).min().unwrap_or(0);
        let best: Vec<usize> = pool.into_iter().filter(|&k| counts[k] == min).collect();
        let k = best[rand.index(best.len())];
        terrain[c] = Some(KINDS[k]);
        counts[k] += 1;
    }
    let terrain: Vec<Terrain> = terrain.into_iter().map(|t| t.expect("terrain attribué")).collect();

    // particularités
    let mut flags = [vec![false; n], vec![false; n], vec![false; n]]; // mine, caverne, magie
    let features = 2.max(round(land.len() as f64 * 0.18) as usize);
    let prefer: [&[Terrain]; 3] = [
        &[Terrain::Mountain, Terrain::Hill],
        &[Terrain::Mountain, Terrain::Hill, Terrain::Forest],
        &[Terrain::Swamp, Terrain::Forest, Terrain::Farmland],
    ];
    for (f, prefer) in prefer.iter().enumerate() {
        for _ in 0..features {
            let cands: Vec<usize> =
                land.iter().copied().filter(|&i| !(flags[0][i] || flags[1][i] || flags[2][i])).collect();
            let arr = &flags[f];
            let weight = |i: usize| {
                let liked = if prefer.contains(&terrain[i]) { 3.0 } else { 1.0 };
                liked * if adj[i].iter().any(|&j| arr[j]) { 0.3 } else { 1.0 }
            };
            let Some(c) = pick_weighted(&cands, weight, &mut rand) else { break };
            flags[f][c] = true;
        }
    }
    let [mine, cavern, magic] = flags;
    let mut lost = vec![false; n];
    for _ in 0..round(land.len() as f64 * 0.3) as usize {
        let cands: Vec<usize> = land.iter().copied().filter(|&i| !lost[i] && terrain[i] != Terrain::Mountain).collect();
        let weight = |i: usize| if adj[i].iter().any(|&j| lost[j]) { 0.35 } else { 1.0 };
        let Some(c) = pick_weighted(&cands, weight, &mut rand) else { break };
        lost[c] = true;
    }

    // bords irréguliers, identiques des deux côtés d'une frontière
    let mut noisy: BTreeMap<(usize, usize), Vec<Pt>> = BTreeMap::new();
    let mut edge_points = |a: usize, b: usize| -> Vec<Pt> {
        let (lo, hi) = (a.min(b), a.max(b));
        let pts = noisy.entry((lo, hi)).or_insert_with(|| {
            let (pa, pb) = (verts[lo], verts[hi]);
            let mut out = Vec::new();
            if on_side(pa) & on_side(pb) == 0 {
                let len = distance(pa, pb);
                let key = Key::new().num(seed as u64).sep().num(lo as u64).text("-").num(hi as u64);
                let mut r = Rand(key.finish());
                displace(pa, pb, &mut r, if len > 60.0 { 3 } else { 2 }, 0.16, &mut out);
            }
            out
        });
        if a == lo { pts.clone() } else { pts.iter().rev().copied().collect() }
    };

    let water = |i: usize| terrain[i].is_water();
    let mut used_names: BTreeSet<(Terrain, u32, u32)> = BTreeSet::new();
    let mut regions = Vec::with_capacity(n);
    for (i, cell) in cells.iter().enumerate() {
        let mut outline = Vec::new();
        for k in 0..cell.len() {
            let (a, b) = (cell[k], cell[(k + 1) % cell.len()]);
            outline.push(verts[a]);
            outline.extend(edge_points(a, b));
        }
        let t = terrain[i];
        let mut name = Name::Numbered(used_names.len() as u32);
        for _ in 0..200 {
            let noun = rand.index(NOUNS[t.index()] as usize) as u32;
            let suffix = rand.index(SUFFIXES as usize) as u32;
            if used_names.insert((t, noun, suffix)) {
                name = Name::Drawn { noun, suffix };
                break;
            }
        }
        let coastal = !water(i) && adj[i].iter().any(|&j| water(j));
        // point d'entrée : bord de carte, côte d'une mer en bordure, ou la mer en bordure elle-même (Marins)
        let entry = if water(i) {
            t == Terrain::Sea && border[i]
        } else {
            border[i] || adj[i].iter().any(|&j| terrain[j] == Terrain::Sea && border[j])
        };
        regions.push(GeneratedRegion {
            terrain: t,
            cavern: cavern[i],
            magic: magic[i],
            mine: mine[i],
            lost_tribe: lost[i],
            border: border[i],
            entry,
            coastal,
            adj: adj[i].iter().map(|&j| j as u8).collect(),
            outline,
            cx: round(centers[i].0) as i32,
            cy: round(centers[i].1) as i32,
            name,
        });
    }
    Ok(Some(regions))
}

/// La carte d'une partie à `players` joueurs pour cette graine.
pub fn generate(players: usize, seed: u64) -> Result<GeneratedMap, MapGenError> {
    let preset = preset(players).ok_or(MapGenError::NoPreset)?;
    for attempt in 0..500u64 {
        let attempt_seed = seed.wrapping_add(attempt * 7919) as u32;
        if let Some(regions) = try_generate(preset, attempt_seed)? {
            return Ok(GeneratedMap { seed, players, rounds: preset.rounds, regions });
        }
    }
    Err(MapGenError::NoValidMap)
}

impl GeneratedMap {
    /// Éloignement de chaque région au centre de la carte, en dixièmes (0 à 10).
    pub fn centre_tenths(&self) -> Vec<u32> {
        centre_tenths(&self.regions.iter().map(|r| (r.cx as f64, r.cy as f64)).collect::<Vec<_>>())
    }

    /// La carte vue par les règles.
    pub fn to_map(&self) -> Result<Map, MapError> {
        let centre = self.centre_tenths();
        let regions = self
            .regions
            .iter()
            .zip(centre)
            .map(|(r, c)| RegionDef {
                terrain: r.terrain,
                cavern: r.cavern,
                magic: r.magic,
                mine: r.mine,
                entry: r.entry,
                border: r.border,
                coastal: r.coastal,
                adj: r.adj.clone(),
                centre: c as f64 / 10.0,
            })
            .collect();
        Map::new(self.seed, regions)
    }
}

/// Éloignement au centre (moyenne des centres des régions), rapporté au plus éloigné, en dixièmes.
pub fn centre_tenths(centres: &[(f64, f64)]) -> Vec<u32> {
    let n = centres.len() as f64;
    let (mut cx, mut cy) = (0.0, 0.0);
    for &(x, y) in centres {
        cx += x / n;
        cy += y / n;
    }
    let reach = centres.iter().fold(1.0f64, |r, &(x, y)| r.max((x - cx).hypot(y - cy)));
    centres.iter().map(|&(x, y)| round(10.0 * (x - cx).hypot(y - cy) / reach) as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_are_valid_and_deterministic() {
        for players in 2..=5 {
            let a = generate(players, 1234).unwrap();
            assert_eq!(a, generate(players, 1234).unwrap());
            assert_eq!(a.regions.len(), preset(players).unwrap().regions);
            assert!(a.to_map().is_ok());
        }
    }
}
