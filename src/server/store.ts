import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { RoomData, RoomStore } from './rooms';

/** Sauvegarde chaque partie dans un fichier JSON, pour survivre à un redémarrage du serveur. */
export class FileRoomStore implements RoomStore {
  private pending = new Map<string, { data: RoomData; timer: ReturnType<typeof setTimeout> }>();

  constructor(private dir: string) {
    mkdirSync(dir, { recursive: true });
  }

  private file(code: string): string {
    return join(this.dir, `${code}.json`);
  }

  save(data: RoomData) {
    const prev = this.pending.get(data.code);
    if (prev) clearTimeout(prev.timer);
    const timer = setTimeout(() => this.flushOne(data.code), 500);
    this.pending.set(data.code, { data, timer });
  }

  private flushOne(code: string) {
    const entry = this.pending.get(code);
    if (!entry) return;
    this.pending.delete(code);
    try {
      writeFileSync(this.file(code), JSON.stringify(entry.data));
    } catch (e) {
      console.error(`Sauvegarde impossible de la partie ${code}`, e);
    }
  }

  /** Écrit immédiatement tout ce qui attend (à l'arrêt du serveur). */
  flush() {
    for (const [code, entry] of this.pending) {
      clearTimeout(entry.timer);
      this.flushOne(code);
    }
  }

  remove(code: string) {
    const entry = this.pending.get(code);
    if (entry) clearTimeout(entry.timer);
    this.pending.delete(code);
    rmSync(this.file(code), { force: true });
  }

  loadAll(): RoomData[] {
    const out: RoomData[] = [];
    for (const name of readdirSync(this.dir)) {
      if (!name.endsWith('.json')) continue;
      try {
        out.push(JSON.parse(readFileSync(join(this.dir, name), 'utf8')) as RoomData);
      } catch {
        console.warn(`Fichier de partie illisible ignoré : ${name}`);
      }
    }
    return out;
  }
}
