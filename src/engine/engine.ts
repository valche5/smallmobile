/**
 * Le noyau du jeu (kernel/, compilé en WebAssembly) partagé par tout le moteur : il est chargé une fois
 * au démarrage (`initEngine`), puis interrogé de façon synchrone.
 */
import { Kernel, wasmBackend } from './kernel';
import type { GameState } from './types';

let kernel: Kernel | null = null;

/** Charge le noyau (le contenu de `game-kernel.wasm`). */
export async function initEngine(wasm: BufferSource | Promise<BufferSource>): Promise<void> {
  kernel ??= new Kernel(await wasmBackend(await wasm));
}

export function engineReady(): boolean {
  return kernel !== null;
}

/** Le noyau, sans partie particulière (génération de cartes, nouvelle partie). */
export function engine(): Kernel {
  if (!kernel) throw new Error("Le noyau du jeu n'est pas chargé (initEngine).");
  return kernel;
}

/** Le noyau, avec cette partie en mémoire. */
export function kernelFor(g: GameState): Kernel {
  const k = engine();
  k.use(g);
  return k;
}
