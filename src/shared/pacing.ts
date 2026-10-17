import type { LogEntry, LogKind } from '../engine/types';
import type { AiSpeed } from './protocol';

/** Temps laissé pour regarder chaque type d'événement (vitesse normale, ms). Partagé par le client et le serveur. */
export const EVENT_MS: Record<LogKind, number> = {
  turn: 1000,
  // le temps de lire ce que font le peuple et le pouvoir choisis
  pick: 2800,
  conquer: 1300,
  die: 2000,
  decline: 2000,
  abandon: 1000,
  score: 1200,
  power: 1000,
  defense: 0,
  info: 0,
  end: 1500,
};

/** Conquête qui fait une victime, quelle qu'elle soit. */
export const VICTIM_BONUS = 700;
/** Paix imposée. */
export const PEACE_MS = 1700;
/** Dé berserk (sans région). */
export const BERSERK_MS = 1000;
/** Trame « annule sa dernière action » (client seulement). */
export const UNDO_MS = 1000;
/** Groupe de redéploiement, une fois par groupe. */
export const REDEPLOY_MS = 1200;
/** Phase 1 du dé adverse (le dé roule), comprise dans la durée de la trame. */
export const DIE_ROLL_MS = 600;
/** Carte d'entête « pendant votre absence ». */
export const LATE_HEADER_MS = 1500;
/** Délai avant l'état « réfléchit… » quand la file est vide (contre le clignotement entre deux actions). */
export const GRACE_MS = 800;

export const FACTOR: Record<AiSpeed, number> = { lent: 1.5, normal: 1, rapide: 0.45 };

/** Durée propre d'une entrée de journal (vitesse normale). */
export function entryMs(e: LogEntry): number {
  if (e.quiet) return 0;
  if (e.kind === 'conquer' && e.victim !== undefined) return EVENT_MS.conquer + VICTIM_BONUS;
  if (e.kind === 'power' && e.victim !== undefined) return PEACE_MS;
  if (e.kind === 'die' && e.region === undefined) return BERSERK_MS;
  return EVENT_MS[e.kind];
}

/** Durée d'affichage d'une trame (1 entrée, ou 2 pour dé + conquête). */
export function frameDuration(frame: LogEntry[], speed: AiSpeed): number {
  if (!frame.length) return 0;
  const base = Math.max(...frame.map(entryMs)) + 400 * (frame.length - 1);
  return Math.min(3400, base) * FACTOR[speed];
}

/** Entrée qui mérite d'être regardée (les autres restent dans le Journal). */
export function isShown(e: LogEntry): boolean {
  return !e.quiet && e.kind !== 'defense' && e.kind !== 'info';
}

/**
 * Découpe des entrées de journal en trames : une entrée = une trame, sauf un dé avec région
 * suivi de la conquête du même joueur sur la même région (une seule trame).
 */
export function splitFrames(events: LogEntry[]): LogEntry[][] {
  const frames: LogEntry[][] = [];
  for (const e of events) {
    if (!isShown(e)) continue;
    const last = frames[frames.length - 1];
    const prev = last?.[last.length - 1];
    if (
      last &&
      last.length === 1 &&
      prev!.kind === 'die' &&
      prev!.region !== undefined &&
      e.kind === 'conquer' &&
      e.player === prev!.player &&
      e.region === prev!.region
    ) {
      last.push(e);
    } else frames.push([e]);
  }
  return frames;
}

/** Temps total pour montrer des événements (somme de leurs trames). Le serveur s'en sert pour rythmer les IA. */
export function playbackMs(events: LogEntry[], speed: AiSpeed): number {
  return splitFrames(events).reduce((sum, f) => sum + frameDuration(f, speed), 0);
}

/** Petite pause de « réflexion » avant chaque action d'une IA. */
export function thinkDelay(speed: AiSpeed): number {
  return 350 * FACTOR[speed];
}

export function newEvents(prev: { seq: number } | null, next: { log: LogEntry[] }): LogEntry[] {
  return prev ? next.log.filter((l) => l.seq > prev.seq) : [];
}
