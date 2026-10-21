import type { Action, AiLevel, GameState, LogEntry, MapDef } from '../engine/types';

/** Messages échangés entre les téléphones et le serveur (JSON sur WebSocket). */

export type AiSpeed = 'lent' | 'normal' | 'rapide';

export interface SeatView {
  name: string;
  color: string;
  kind: 'human' | 'ai';
  ai: AiLevel;
  /** Un joueur humain occupe ce siège. */
  taken: boolean;
  /** Au moins un téléphone est connecté pour ce siège. */
  online: boolean;
  /** Siège humain confié temporairement à l'IA (joueur parti). */
  replaced: boolean;
}

export interface RoomView {
  code: string;
  size: number;
  mapSeed: number;
  aiLevel: AiLevel;
  /** Le labo dont l'Expert joue les sièges de niveau `expert` (`experts.ts`). */
  expertLab: string;
  speed: AiSpeed;
  /** Siège qui a réglé la vitesse en dernier (null : réglage d'origine). */
  speedBy: number | null;
  started: boolean;
  seats: SeatView[];
}

/** État de partie envoyé aux téléphones : sans la carte (envoyée une fois) ni la graine du hasard. */
export type GameWire = Omit<GameState, 'map' | 'record'>;

/** Un état passé de la partie, gardé par le serveur pour rejouer fidèlement ce qu'un téléphone a manqué. */
export interface Snapshot {
  seq: number;
  /** L'état, sans journal (le client le reconstruit à partir du journal final). */
  game: Omit<GameWire, 'log'>;
  /** Entrées de journal apparues avec cet état. */
  events: LogEntry[];
}

/** Ce qu'un téléphone n'a pas encore vu : à rejouer sur les vrais états intermédiaires. */
export interface Replay {
  /** Dernier état déjà vu par ce téléphone (null : plus dans l'historique). */
  base: Snapshot | null;
  frames: Snapshot[];
  /** Entrées d'autres joueurs plus anciennes que l'historique conservé. */
  missing: number;
}

export type ClientMsg =
  /** `opponent` : 'ai' pour jouer seul contre l'IA (la partie commence aussitôt). */
  | { t: 'create'; name: string; color: string; size: number; aiLevel: AiLevel; expertLab?: string; mapSeed: number; opponent?: 'human' | 'ai' }
  | { t: 'join'; code: string; name: string; color: string }
  /** `shown` : numéro de journal de la dernière trame affichée sur ce téléphone. */
  | { t: 'resume'; code: string; token: string; shown?: number }
  | { t: 'lobby'; size?: number; aiLevel?: AiLevel; expertLab?: string; mapSeed?: number; name?: string; color?: string }
  | { t: 'start' }
  | { t: 'action'; id: number; action: Action }
  | { t: 'undo' }
  | { t: 'speed'; speed: AiSpeed }
  | { t: 'replace'; seat: number }
  | { t: 'leave' };

export type ServerMsg =
  | { t: 'joined'; code: string; token: string; seat: number }
  | { t: 'room'; room: RoomView }
  | { t: 'map'; map: MapDef }
  /** `sync` : état de resynchronisation (reconnexion, action refusée), à afficher directement. */
  | { t: 'state'; game: GameWire; ack: { seat: number; id: number } | null; undo: number; sync?: boolean; replay?: Replay }
  | { t: 'error'; message: string; fatal?: boolean }
  | { t: 'left' };

export const CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
export const CODE_LENGTH = 4;

export function normalizeCode(raw: string): string {
  return raw
    .toUpperCase()
    .replace(/[^A-Z]/g, '')
    .slice(0, CODE_LENGTH);
}
