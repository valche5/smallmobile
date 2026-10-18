import { foeThreat, lostRegions } from '../../engine/preview';
import { actingRace, analyzeTarget } from '../../engine/rules';
import type { GameState } from '../../engine/types';

/**
 * Pastilles et marquages de la carte, calculés pour le joueur de ce téléphone (rendus par MapView).
 * - 'normal' | 'die' | 'special' : cible de ma conquête (pastille ⚔N, 🎲N%, 🐉1, 🔮) ;
 * - 'threat' : une de mes régions qu'un ennemi peut prendre (redéploiement, défense) ;
 * - 'own' : une de mes régions, sans pastille ; 'none' : hors de portée (assombrie) ;
 * - 'plain' : rien à signaler pour la phase (seuls les drapeaux comme `lost` comptent, pas d'assombrissement).
 */
export type MarkKind = 'normal' | 'die' | 'special' | 'threat' | 'own' | 'none' | 'plain';

export interface Mark {
  kind: MarkKind;
  label?: string;
  /** Couleur de l'anneau (redéploiement : l'ennemi le plus dangereux pour cette région). */
  ring?: string;
  /** Pastille à mettre en avant (prise bon marché, dragon, sorcellerie). */
  hot?: boolean;
  /** Région qu'un adversaire m'a prise depuis mon dernier décompte (contour rouge pointillé). */
  lost?: boolean;
}

/** Ce qu'il faut mettre en évidence sur la carte pour `pid`, le joueur de ce téléphone (`humanTurn` : c'est à lui). */
export function computeMarks(g: GameState, pid: number | null, humanTurn: boolean): Map<number, Mark> {
  const marks = new Map<number, Mark>();
  const race = actingRace(g);
  const deploying = humanTurn && (g.phase === 'redeploy' || g.phase === 'defend');
  if (humanTurn && g.phase === 'conquest') {
    for (const def of g.map.regions) {
      if (g.regions[def.id]!.race === race) {
        marks.set(def.id, { kind: 'own' });
        continue;
      }
      const t = analyzeTarget(g, def.id);
      if (t.kind === 'normal') marks.set(def.id, { kind: 'normal', label: `⚔${t.effective}` });
      else if (t.sorcery) marks.set(def.id, { kind: 'special', label: '🔮' });
      else if (t.dragon) marks.set(def.id, { kind: 'special', label: '🐉1' });
      else if (t.kind === 'die') marks.set(def.id, { kind: 'die', label: `🎲${Math.round(t.chance * 100)}%` });
      else marks.set(def.id, { kind: 'none' });
    }
  } else if (deploying && pid !== null) {
    // ce qu'un ennemi paiera vraiment pour me prendre chaque région, pouvoirs compris
    for (const def of g.map.regions) {
      if (g.regions[def.id]!.race !== race) {
        marks.set(def.id, { kind: 'none' });
        continue;
      }
      const t = foeThreat(g, def.id, pid);
      if (!t) {
        marks.set(def.id, { kind: 'own' });
        continue;
      }
      const label = t.how === 'sorcery' ? '🔮' : t.how === 'dragon' ? '🐉1' : `⚔${t.cost}`;
      marks.set(def.id, { kind: 'threat', label, ring: g.players[t.pid]!.color, hot: t.how !== 'conquer' || t.cost <= 3 });
    }
  }
  // régions perdues : visibles partout sauf pendant mon redéploiement et ma défense
  if (pid !== null && !deploying) {
    for (const id of lostRegions(g, pid)) marks.set(id, { ...(marks.get(id) ?? { kind: 'plain' }), lost: true });
  }
  return marks;
}
