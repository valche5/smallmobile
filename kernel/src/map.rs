//! La carte : ce qui ne change jamais pendant une partie. Elle est partagée, en lecture seule, par
//! toutes les copies de l'état.

use crate::collections::RegionSet;
use crate::ids::{MAX_REGIONS, Terrain};

#[derive(Clone, Debug, PartialEq)]
pub struct RegionDef {
    pub terrain: Terrain,
    pub cavern: bool,
    pub magic: bool,
    pub mine: bool,
    /// Peut servir de première conquête (bord de carte ou côte d'une mer en bordure).
    pub entry: bool,
    /// Touche le bord de la carte.
    pub border: bool,
    /// Adjacente à une mer ou un lac.
    pub coastal: bool,
    /// Voisines, dans l'ordre de la carte.
    pub adj: Vec<u8>,
    /// Éloignement du centre de la carte, de 0 à 1 au dixième près (calculé avec la géométrie, côté TS).
    pub centre: f64,
}

impl RegionDef {
    pub fn is_water(&self) -> bool {
        self.terrain.is_water()
    }
}

/// Voisinage d'une région, déduit de la carte une fois pour toutes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Neighbourhood {
    pub adj: RegionSet,
    /// Voisines terrestres.
    pub near: RegionSet,
    /// Régions terrestres à deux pas (par une voisine terrestre), hors voisines.
    pub far: RegionSet,
    /// Nombre de voisines d'eau.
    pub water: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapError {
    TooManyRegions,
    NoRegion,
    BadNeighbour,
    AsymmetricNeighbours,
}

#[derive(Clone, Debug)]
pub struct Map {
    /// Graine de la carte (sert aussi de sel au hasard de l'IA).
    pub seed: u64,
    pub regions: Vec<RegionDef>,
    pub hoods: Vec<Neighbourhood>,
    pub caverns: RegionSet,
}

impl Map {
    pub fn new(seed: u64, regions: Vec<RegionDef>) -> Result<Map, MapError> {
        let n = regions.len();
        if n == 0 {
            return Err(MapError::NoRegion);
        }
        if n > MAX_REGIONS {
            return Err(MapError::TooManyRegions);
        }
        for (i, def) in regions.iter().enumerate() {
            let mut seen = RegionSet::EMPTY;
            for &a in &def.adj {
                let a = a as usize;
                if a >= n || a == i || seen.contains(a) {
                    return Err(MapError::BadNeighbour);
                }
                seen.insert(a);
                // le moteur suppose des adjacences symétriques (une région touche celles qui la touchent)
                if !regions[a].adj.contains(&(i as u8)) {
                    return Err(MapError::AsymmetricNeighbours);
                }
            }
        }
        let hoods = (0..n).map(|i| Self::hood(&regions, i)).collect();
        let caverns = (0..n).filter(|&i| regions[i].cavern).collect();
        Ok(Map { seed, regions, hoods, caverns })
    }

    fn hood(regions: &[RegionDef], i: usize) -> Neighbourhood {
        let mut h = Neighbourhood::default();
        let mut seen = RegionSet::EMPTY;
        seen.insert(i);
        for &a in &regions[i].adj {
            let a = a as usize;
            h.adj.insert(a);
            seen.insert(a);
            if regions[a].is_water() {
                h.water += 1;
            } else {
                h.near.insert(a);
            }
        }
        for a in h.near.iter() {
            for &b in &regions[a].adj {
                let b = b as usize;
                if !seen.contains(b) {
                    seen.insert(b);
                    if !regions[b].is_water() {
                        h.far.insert(b);
                    }
                }
            }
        }
        h
    }

    pub fn len(&self) -> usize {
        self.regions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Une ligne de régions : 0 — 1 — 2 — … ; `water` donne les lacs.
    pub fn line(n: usize, water: &[usize]) -> Map {
        let regions = (0..n)
            .map(|i| RegionDef {
                terrain: if water.contains(&i) { Terrain::Lake } else { Terrain::Farmland },
                cavern: false,
                magic: false,
                mine: false,
                entry: i == 0,
                border: i == 0,
                coastal: false,
                adj: [i.checked_sub(1), (i + 1 < n).then_some(i + 1)].into_iter().flatten().map(|a| a as u8).collect(),
                centre: 0.0,
            })
            .collect();
        Map::new(1, regions).unwrap()
    }

    #[test]
    fn neighbourhoods_skip_water() {
        let map = line(5, &[2]);
        let h = map.hoods[1];
        assert_eq!(h.near.iter().collect::<Vec<_>>(), vec![0]);
        assert_eq!(h.water, 1);
        // on ne traverse pas l'eau pour aller à deux pas
        assert!(h.far.is_empty());
        assert_eq!(map.hoods[3].far.iter().collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(map.hoods[4].far.iter().collect::<Vec<_>>(), Vec::<usize>::new());
        let map = line(5, &[]);
        assert_eq!(map.hoods[2].far.iter().collect::<Vec<_>>(), vec![0, 4]);
    }

    #[test]
    fn rejects_asymmetric_maps() {
        let mut map = line(3, &[]);
        map.regions[0].adj.clear();
        assert_eq!(Map::new(1, map.regions).unwrap_err(), MapError::AsymmetricNeighbours);
    }
}
