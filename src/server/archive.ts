/**
 * Les parties jouées sur le serveur, gardées pour le labo de l'IA : un fichier JSON par partie dans
 * `data/games/`, avec de quoi la rejouer à l'identique (graines et actions) et le temps de chaque
 * décision d'IA. Contrairement aux salles (`store.ts`), rien n'est jamais effacé.
 *
 *   bun lab/tools/games.ts data/games    (temps de décision des IA)
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { AiLevel, GameRecord } from '../engine/types';

export interface ArchivedGame {
  /** Le nom du fichier : début de la partie et code de la salle. */
  id: string;
  code: string;
  updatedAt: number;
  players: { name: string; kind: 'human' | 'ai'; ai: AiLevel }[];
  /** Fin de partie : pièces de chaque joueur. */
  coins: number[] | null;
  record: GameRecord;
}

export interface GameArchive {
  save(game: ArchivedGame): void;
}

export class FileGameArchive implements GameArchive {
  private pending = new Map<string, { game: ArchivedGame; timer: ReturnType<typeof setTimeout> }>();

  constructor(private dir: string) {
    mkdirSync(dir, { recursive: true });
  }

  save(game: ArchivedGame) {
    const prev = this.pending.get(game.id);
    if (prev) clearTimeout(prev.timer);
    // pendant le tour d'une IA, une écriture toutes les quelques secondes suffit
    const timer = setTimeout(() => this.flushOne(game.id), 2000);
    this.pending.set(game.id, { game, timer });
  }

  private flushOne(id: string) {
    const entry = this.pending.get(id);
    if (!entry) return;
    this.pending.delete(id);
    try {
      writeFileSync(join(this.dir, `${id}.json`), JSON.stringify(entry.game));
    } catch (e) {
      console.error(`Enregistrement impossible de la partie ${id}`, e);
    }
  }

  /** Écrit immédiatement tout ce qui attend (à l'arrêt du serveur). */
  flush() {
    for (const [id, entry] of this.pending) {
      clearTimeout(entry.timer);
      this.flushOne(id);
    }
  }
}

/** Le nom de fichier d'une partie : son début (heure locale du serveur) et le code de la salle. */
export function gameId(code: string, startedAt: number): string {
  const d = new Date(startedAt);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}_${code}`;
}
