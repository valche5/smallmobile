import { PLAYER_COLORS } from '../engine/data';

const KEY = 'smallmobile:me:v1';

export interface Me {
  name: string;
  color: string;
}

/** Nom et couleur préférés de la personne qui tient ce téléphone. */
export function loadMe(): Me {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? 'null') as Me | null;
    if (raw && typeof raw.name === 'string' && PLAYER_COLORS.some((c) => c.color === raw.color)) return raw;
  } catch {
    /* ignore */
  }
  return { name: '', color: PLAYER_COLORS[0]!.color };
}

export function saveMe(me: Me) {
  try {
    localStorage.setItem(KEY, JSON.stringify(me));
  } catch {
    /* ignore */
  }
}
