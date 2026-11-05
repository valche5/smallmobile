/**
 * Les cartes : 100 par nombre de joueurs, générées une fois pour toutes dans `kernel/maps.bin` (voir
 * kernel/src/maps.rs), lues par le noyau, nommées et dessinées ici. La carte d'une graine est celle de
 * rang `graine % MAP_COUNT`. Le serveur garde le fichier en mémoire (`loadMaps`) ; le téléphone lui
 * demande la carte qu'il veut montrer (`fetchMap`, route `/maps/:joueurs/:rang`).
 */
import { engine } from './engine';
import type { MapDef } from './types';

export interface MapPreset {
  players: number;
  name: string;
  size: string;
  /** La carte proposée par défaut. */
  seed: number;
  /** Régions et tours de toutes les cartes de cette taille. */
  regions: number;
  rounds: number;
}

export const PRESETS: Record<number, MapPreset> = {
  2: { players: 2, name: 'Le Val Paisible', size: 'Petite', seed: 2024, regions: 23, rounds: 10 },
  3: { players: 3, name: 'Les Trois Couronnes', size: 'Moyenne', seed: 3033, regions: 30, rounds: 10 },
  4: { players: 4, name: 'Le Grand Continent', size: 'Grande', seed: 4044, regions: 39, rounds: 9 },
  5: { players: 5, name: "L'Archipel Sans Fin", size: 'Immense', seed: 5055, regions: 48, rounds: 8 },
};

/** Cartes par nombre de joueurs (`maps::COUNT`). */
export const MAP_COUNT = 100;
const MAGIC = 'SMM1';

/** Taille des cartes. */
export const MAP_W = 1000;
export const MAP_H = 1300;

let file: Uint8Array<ArrayBuffer> | null = null;
const cache = new Map<string, MapDef>();

/** Garde en mémoire le fichier des cartes (`kernel/maps.bin`). */
export function loadMaps(bytes: ArrayBuffer) {
  const f = new Uint8Array(bytes);
  if (new TextDecoder().decode(f.subarray(0, 4)) !== MAGIC) throw new Error('Fichier des cartes illisible.');
  file = f;
  cache.clear();
}

function rank(seed: number): number {
  if (!Number.isSafeInteger(seed) || seed < 0) throw new Error('Graine de carte invalide.');
  return seed % MAP_COUNT;
}

function key(players: number, seed: number): string {
  if (!PRESETS[players]) throw new Error(`Pas de carte pour ${players} joueurs`);
  return `${players}:${rank(seed)}`;
}

/** Les octets de la carte de cette graine, dans le fichier chargé (`loadMaps`). */
export function mapRecord(players: number, seed: number): Uint8Array<ArrayBuffer> {
  if (!file) throw new Error("Le fichier des cartes n'est pas chargé (loadMaps).");
  key(players, seed);
  const slot = (players - 2) * MAP_COUNT + rank(seed);
  const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
  return file.subarray(view.getUint32(4 + 4 * slot, true), view.getUint32(8 + 4 * slot, true));
}

function decode(players: number, bytes: Uint8Array): MapDef {
  const preset = PRESETS[players]!;
  const { seed, rounds, regions } = engine().readMap(bytes);
  return {
    id: `p${players}`,
    name: seed === preset.seed ? preset.name : `${preset.size} carte n°${seed % MAP_COUNT}`,
    seed,
    width: MAP_W,
    height: MAP_H,
    players,
    rounds,
    regions,
  };
}

/** La carte de cette graine, depuis le fichier chargé (serveur). */
export function getMap(players: number, seed: number): MapDef {
  const k = key(players, seed);
  let map = cache.get(k);
  if (!map) cache.set(k, (map = decode(players, mapRecord(players, seed))));
  return map;
}

/** La carte de cette graine si elle est déjà connue. */
export function cachedMap(players: number, seed: number): MapDef | null {
  return cache.get(key(players, seed)) ?? null;
}

/** La carte de cette graine, demandée au serveur (téléphone). */
export async function fetchMap(players: number, seed: number): Promise<MapDef> {
  const k = key(players, seed);
  const hit = cache.get(k);
  if (hit) return hit;
  const res = await fetch(`maps/${players}/${rank(seed)}`);
  if (!res.ok) throw new Error(`Carte introuvable (${res.status}).`);
  const map = decode(players, new Uint8Array(await res.arrayBuffer()));
  cache.set(k, map);
  return map;
}

/** Une autre carte au hasard pour ce nombre de joueurs (sa graine). */
export function otherMap(players: number, seed: number): number {
  return 1000 * players + ((rank(seed) + 1 + Math.floor(Math.random() * (MAP_COUNT - 1))) % MAP_COUNT);
}
