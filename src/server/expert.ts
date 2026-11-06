/**
 * L'IA Expert, jouée par un processus natif à côté du serveur, un par labo (`labN/brain/src/bin/expert.rs`,
 * programme `labN-expert` construit par `bun run kernel:build` ; variantes : `src/shared/experts.ts`) :
 * le serveur lui envoie la partie, elle renvoie les actions de son coup. Un processus à part ne bloque
 * jamais le serveur ; s'il tombe, il est relancé à la demande suivante.
 */
import type { Subprocess } from 'bun';
import { existsSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { basename } from 'node:path';
import { gameWords, readAction } from '../engine/kernel';
import type { Action, GameState } from '../engine/types';
import { EXPERT_LABS, type ExpertLab } from '../shared/experts';

/** Un coup de l'Expert : ses actions, dans l'ordre, et ce qu'en pensait la recherche. */
export interface ExpertMove {
  actions: Action[];
  /** Chance de gagner estimée pour le joueur qui agit (NaN : coup forcé, joué sans chercher). */
  value: number;
  iterations: number;
  /** Temps de calcul dans le processus (ms), sans l'aller-retour. */
  computeMs: number;
  /** Le labo et le réseau qui ont joué (`lab1/gen11`). */
  net: string;
}

export interface Expert {
  decide(g: GameState): Promise<ExpertMove>;
}

/** Les Expert disponibles, par labo. */
export type Experts = ReadonlyMap<string, Expert>;

export interface ExpertOptions {
  /** Le labo (`lab1`). */
  lab: string;
  /** Le programme natif. */
  binary: string;
  /** Les poids du réseau (labN/nets/*.bin). */
  net: string;
  /** Itérations de recherche par décision. */
  iterations: number;
  /** Autres réglages de la recherche (`ExpertLab.search`). */
  search?: string;
  /** Fils de calcul du processus (`--threads`), si le programme les accepte (`ExpertLab.threads`). */
  threads?: number;
  /** Au-delà, la décision est abandonnée et le processus relancé (ms). */
  timeoutMs?: number;
}

/** Le programme et le réseau de l'Expert d'un labo ; les cœurs de la machine moins un s'il sait s'en servir. */
export function expertOptions(lab: ExpertLab): ExpertOptions {
  return {
    lab: lab.id,
    binary: `./target/release/${lab.id}-expert${process.platform === 'win32' ? '.exe' : ''}`,
    net: `./${lab.id}/nets/${lab.net}.bin`,
    iterations: lab.iterations,
    search: lab.search,
    threads: lab.threads ? Math.max(1, availableParallelism() - 1) : undefined,
  };
}

/** Un processus par labo dont le programme et le réseau sont là ; `warn` signale les autres. */
export function startExperts(warn: (why: string) => void): Map<string, ExpertProcess> {
  const out = new Map<string, ExpertProcess>();
  for (const lab of EXPERT_LABS) {
    const opts = expertOptions(lab);
    const missing = ExpertProcess.available(opts);
    if (missing) warn(`IA Expert ${lab.id} indisponible : ${missing}`);
    else out.set(lab.id, new ExpertProcess(opts));
  }
  return out;
}

interface Pending {
  resolve: (words: Uint32Array) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class ExpertProcess implements Expert {
  private proc: Subprocess<'pipe', 'pipe', 'inherit'> | null = null;
  /** Les demandes envoyées, dans l'ordre : le processus y répond une à une. */
  private waiting: Pending[] = [];
  private chunks: Uint8Array[] = [];
  private buffered = 0;

  constructor(private readonly opts: ExpertOptions) {}

  /** Le programme et les poids sont-ils là ? */
  static available(opts: ExpertOptions): string | null {
    if (!existsSync(opts.binary)) return `${opts.binary} manque (bun run kernel:build)`;
    if (!existsSync(opts.net)) return `${opts.net} manque`;
    return null;
  }

  decide(g: GameState): Promise<ExpertMove> {
    const seed = Math.floor(Math.random() * 2 ** 32);
    const game = gameWords(g);
    const request = new Uint32Array(4 + game.length);
    request.set([3 + game.length, this.opts.iterations, seed, 0]);
    request.set(game, 4);
    return this.ask(request).then((words) => ({ ...readMove(words, g), net: `${this.opts.lab}/${basename(this.opts.net, '.bin')}` }));
  }

  /** Arrête le processus (fin du serveur). */
  close() {
    this.proc?.kill();
    this.proc = null;
  }

  private ask(request: Uint32Array): Promise<Uint32Array> {
    const proc = this.spawn();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.fail(new Error("l'Expert ne répond plus")), this.opts.timeoutMs ?? 60_000);
      this.waiting.push({ resolve, reject, timer });
      proc.stdin.write(new Uint8Array(request.buffer, request.byteOffset, request.byteLength));
      proc.stdin.flush();
    });
  }

  private spawn() {
    if (this.proc) return this.proc;
    const args = [
      this.opts.binary,
      '--net',
      this.opts.net,
      ...(this.opts.search ? ['--search', this.opts.search] : []),
      ...(this.opts.threads && this.opts.threads > 1 ? ['--threads', String(this.opts.threads)] : []),
    ];
    const proc = Bun.spawn(args, { stdin: 'pipe', stdout: 'pipe', stderr: 'inherit' });
    this.proc = proc;
    this.chunks = [];
    this.buffered = 0;
    void this.read(proc);
    void proc.exited.then((code) => {
      if (this.proc !== proc) return;
      this.proc = null;
      this.fail(new Error(`l'Expert s'est arrêté (code ${code})`));
    });
    return proc;
  }

  private async read(proc: Subprocess<'pipe', 'pipe', 'inherit'>) {
    try {
      for await (const chunk of proc.stdout) {
        if (this.proc !== proc) return;
        this.chunks.push(chunk);
        this.buffered += chunk.length;
        this.deliver();
      }
    } catch {
      /* processus arrêté : `exited` s'en occupe */
    }
  }

  /** Les réponses complètes arrivées : un nombre de mots, puis les mots. */
  private deliver() {
    for (;;) {
      if (this.buffered < 4) return;
      const bytes = this.chunks.length === 1 ? this.chunks[0]! : Buffer.concat(this.chunks);
      this.chunks = [bytes];
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const len = view.getUint32(0, true);
      const size = 4 + 4 * len;
      if (bytes.length < size) return;
      const words = new Uint32Array(len);
      for (let i = 0; i < len; i++) words[i] = view.getUint32(4 + 4 * i, true);
      const rest = bytes.subarray(size);
      this.chunks = rest.length ? [rest] : [];
      this.buffered = rest.length;
      const p = this.waiting.shift();
      if (!p) continue;
      clearTimeout(p.timer);
      p.resolve(words);
    }
  }

  /** Toutes les demandes en cours échouent ; le processus sera relancé à la prochaine. */
  private fail(e: Error) {
    const waiting = this.waiting;
    this.waiting = [];
    for (const p of waiting) {
      clearTimeout(p.timer);
      p.reject(e);
    }
    const proc = this.proc;
    this.proc = null;
    proc?.kill();
  }
}

/** La réponse du processus (voir `labN/brain/src/bin/expert.rs`). */
function readMove(xs: Uint32Array, g: GameState): Omit<ExpertMove, 'net'> {
  if ((xs[0]! | 0) < 0) {
    const bytes = new Uint8Array(xs.buffer, xs.byteOffset + 8, Math.min(xs[1]!, 4 * (xs.length - 2)));
    throw new Error(`l'Expert refuse : ${new TextDecoder().decode(bytes)}`);
  }
  const value = new Float32Array(new Uint32Array([xs[1]!]).buffer)[0]!;
  const iterations = xs[2]!;
  const computeMs = xs[3]! / 1000;
  const count = xs[4]!;
  const actions: Action[] = [];
  let at = 5;
  for (let i = 0; i < count; i++) {
    const len = xs[at]!;
    actions.push(readAction(xs.subarray(at + 1, at + 1 + len), g));
    at += 1 + len;
  }
  if (!actions.length) throw new Error("l'Expert n'a renvoyé aucune action");
  return { actions, value, iterations, computeMs };
}
