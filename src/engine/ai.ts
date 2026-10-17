/** L'IA du jeu (trois niveaux) : elle est entièrement dans le noyau (kernel/src/ai/). */
import { kernelFor } from './engine';
import type { Action, GameState } from './types';

/**
 * Prochaine action de l'IA pour le joueur qui doit agir, à son niveau. `noDecline` lui interdit de
 * passer en déclin, `keepPlacement` de replacer ses jetons en fin de tour.
 */
export function aiAction(g: GameState, noDecline = false, keepPlacement = false): Action {
  return kernelFor(g).decide(noDecline, keepPlacement);
}
