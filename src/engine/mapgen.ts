/** Les cartes : générées par le noyau, nommées et dessinées ici. */
import { engine } from './engine';
import type { MapDef } from './types';

export interface MapPreset {
  players: number;
  name: string;
  size: string;
  /** La carte proposée par défaut. */
  seed: number;
}

export const PRESETS: Record<number, MapPreset> = {
  2: { players: 2, name: 'Le Val Paisible', size: 'Petite', seed: 2024 },
  3: { players: 3, name: 'Les Trois Couronnes', size: 'Moyenne', seed: 3033 },
  4: { players: 4, name: 'Le Grand Continent', size: 'Grande', seed: 4044 },
  5: { players: 5, name: "L'Archipel Sans Fin", size: 'Immense', seed: 5055 },
};

/** Taille des cartes générées. */
export const MAP_W = 1000;
export const MAP_H = 1300;

const cache = new Map<string, MapDef>();

export function generateMap(players: number, seed: number): MapDef {
  const preset = PRESETS[players];
  if (!preset) throw new Error(`Pas de carte pour ${players} joueurs`);
  const key = `${players}:${seed}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const { rounds, regions } = engine().generateRegions(players, seed);
  const map: MapDef = {
    id: `p${players}`,
    name: seed === preset.seed ? preset.name : `${preset.size} carte n°${seed % 10000}`,
    seed,
    width: MAP_W,
    height: MAP_H,
    players,
    rounds,
    regions,
  };
  cache.set(key, map);
  return map;
}

/** La carte proposée par défaut pour ce nombre de joueurs. */
export function presetMap(players: number): MapDef {
  return generateMap(players, PRESETS[players]!.seed);
}
