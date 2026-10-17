import { FEATURE_ICONS, POWERS, RACES } from '../engine/data';
import type { GameState, LogEntry, RaceId } from '../engine/types';

/** Une trame du tour adverse racontée pour la carte d'événement. */
export interface Narration {
  who: number | null;
  /** L2 : l'action. */
  main: string;
  /** L3 : la conséquence (absente si elle n'apporte rien). */
  detail?: string;
  /** L4 : seulement pour un choix de peuple (2e texte « foe »). */
  extra?: string;
  /** 'hit' : l'action me touche (rouge, 2e personne). */
  tone: 'hit' | 'neutral';
  region?: number;
  /** Dé de renfort : la carte le fait d'abord rouler (phase 1), puis donne le verdict. */
  dieRoll?: { need: number; value: number; success: boolean };
  /** L2 pendant que le dé roule. */
  rolling?: string;
  /** Jetons que je perds (bulle « −N » sur la carte). */
  loss?: number;
}

const ICONS: Record<string, string> = {
  pick: '👑',
  conquer: '⚔️',
  die: '🎲',
  decline: '💤',
  abandon: '🏳️',
  score: '💰',
  defense: '🛡️',
  power: '✨',
  info: 'ℹ️',
  turn: '▶',
  end: '🏁',
};

function plural(n: number, word: string): string {
  return `${n} ${word}${n > 1 ? 's' : ''}`;
}

/** Nom de région coupé pour tenir sur une ligne. */
export function cutName(name: string, max = 18): string {
  const chars = [...name];
  return chars.length > max ? `${chars.slice(0, max - 1).join('')}…` : name;
}

function where(g: GameState, region: number | undefined): string {
  return `« ${cutName(g.map.regions[region ?? -1]?.name ?? '?')} »`;
}

function nameOf(g: GameState, pid: number | null | undefined): string {
  return pid === null || pid === undefined ? '?' : (g.players[pid]?.name ?? '?');
}

/** Repli pour les entrées sans données structurées (anciennes sauvegardes). */
function fallback(e: LogEntry): string {
  return `${ICONS[e.kind] ?? '•'} ${e.short ?? e.detail ?? e.text}`;
}

/** Ce que je perds quand on me prend une région (L3, en rouge). */
function lossLine(g: GameState, e: LogEntry, viewer: number): string | undefined {
  const race = e.victimRace;
  if (!race || race === 'lost') return e.detail;
  const n = g.players.length;
  const actor = e.player ?? -1;
  // le joueur suivant reprend tout en main au début de son tour ; les autres replacent leurs jetons en défense
  const next = (actor + 1) % n === viewer;
  const back = e.returned ?? 0;
  const tail = back > 0 ? ` · ${back} ${next ? (back > 1 ? 'reviennent en main' : 'revient en main') : 'à replacer'}` : '';
  if (e.how === 'sorcery') return `Votre jeton ${RACES[race].name} devient un Sorcier`;
  const lost = e.lost ?? 1;
  if (lost === 0) return `${RACES[race].icon} Vous ne perdez rien${tail}`;
  const declined = g.races[race]?.declined ? ' (en déclin)' : '';
  return `Vous perdez ${plural(lost, 'jeton')} ${RACES[race].name}${declined}${tail}`;
}

/** L2 d'une conquête : qui perd quoi, vu du joueur de ce téléphone. */
function conquerMain(g: GameState, e: LogEntry, viewer: number | null, verb = 'prend'): string {
  const at = where(g, e.region);
  // L2 tient sur une ligne : le nombre de jetons de l'attaquant compte moins que le nom de la région
  if (e.how === 'dragon') return `🐉 lâche son dragon sur ${at}`;
  if (e.how === 'sorcery') {
    if (e.victim !== undefined && e.victim === viewer) return `🔮 ensorcelle votre ${at}`;
    return `🔮 ensorcelle ${at}${e.victim !== undefined ? ` de ${nameOf(g, e.victim)}` : ''}`;
  }
  if (e.victimRace === 'lost') return `${verb} ${at} · chasse la tribu`;
  if (e.victim === undefined) return `${verb} ${at}`;
  if (e.victim === viewer) return `${verb} votre ${at}`;
  if (e.victim === e.player && e.victimRace) return `${verb === 'prend' ? 'reprend' : verb} ${at} à ses ${RACES[e.victimRace].name}`;
  return `${verb} ${at} à ${nameOf(g, e.victim)}`;
}

/** Raconte une trame (une entrée de journal, ou dé + conquête) pour le joueur `viewer`. */
export function narrate(g: GameState, frame: LogEntry[], viewer: number | null): Narration {
  const e = frame[0]!;
  const who = e.player;
  const out: Narration = { who, main: fallback(e), tone: 'neutral' };
  switch (e.kind) {
    case 'turn': {
      const ghouls = e.race === 'ghouls' && g.races.ghouls?.declined;
      // juste après ses Goules (« a fini avec ses Goules ») : c'est la suite du même tour
      const afterGhouls = g.log.some((l) => l.seq === e.seq - 1 && l.kind === 'info' && l.quiet && l.player === e.player);
      if (ghouls || (!e.race && e.short?.includes('Goules'))) out.main = '🧟 joue d\'abord ses Goules en déclin';
      else if (e.tokens !== undefined && afterGhouls) out.main = `▶ puis son peuple actif · ${plural(e.tokens, 'jeton')} en main`;
      else if (e.tokens !== undefined) out.main = `▶ commence son tour · ${plural(e.tokens, 'jeton')} en main`;
      break;
    }
    case 'pick': {
      const race = e.race ?? (who !== null ? g.players[who]?.active : null) ?? null;
      const power = e.power ?? (race ? g.races[race]?.power : undefined);
      if (race && power) {
        out.main = `👑 choisit ${RACES[race].icon} ${RACES[race].name} ${POWERS[power].icon} ${POWERS[power].name}`;
        out.detail = `${RACES[race].icon} ${RACES[race].foe}`;
        out.extra = `${POWERS[power].icon} ${POWERS[power].foe}`;
      }
      break;
    }
    case 'conquer': {
      out.region = e.region;
      if (e.how === undefined) break;
      out.main = e.how === 'dragon' || e.how === 'sorcery' ? conquerMain(g, e, viewer) : `⚔️ ${conquerMain(g, e, viewer)}`;
      break;
    }
    case 'die': {
      out.region = e.region;
      const d = e.die ?? (g.lastDie && g.lastDie.seq === e.seq ? g.lastDie : null);
      if (e.region === undefined || d?.success === null) {
        // dé berserk : pas de région, il réduit le coût de la prochaine conquête
        const v = d?.value;
        if (v === undefined) break;
        out.main = v ? `🎲 dé berserk : ${v} → prochaine conquête −${v}` : '🎲 dé berserk : 0, aucun bonus';
        break;
      }
      if (!d) break;
      const success = !!d.success;
      out.dieRoll = { need: d.need, value: d.value, success };
      // le dé qui roule est dessiné en tête de ligne par la carte d'événement
      // le chiffre à faire d'abord : c'est lui qu'on doit lire avant le verdict
      out.rolling = `il faut ${d.need} sur ${where(g, e.region)}`;
      const conquer = frame.find((x) => x.kind === 'conquer');
      if (success && conquer) out.main = `🎲 ${d.value} : réussi, ${conquerMain(g, conquer, viewer)}`;
      else if (success) out.main = `🎲 ${d.value} : réussi`;
      else out.main = `🎲 ${d.value} : raté · fin de ses conquêtes`;
      break;
    }
    case 'abandon':
      out.region = e.region;
      if (e.region !== undefined) out.main = `🏳️ abandonne ${where(g, e.region)}`;
      break;
    case 'decline': {
      if (!e.race) break;
      out.main = `💤 passe ses ${RACES[e.race].icon} ${RACES[e.race].name} en déclin`;
      if (e.removed?.length) out.detail = `ses ${e.removed.map((r) => `${RACES[r].icon} ${RACES[r].name}`).join(', ')} disparaissent`;
      else if (e.race === 'ghouls') out.detail = '🧟 ses Goules gardent leurs jetons et attaqueront encore';
      break;
    }
    case 'power': {
      if (e.victim === undefined) break;
      if (e.victim === viewer) {
        out.main = '🕊️ vous impose la paix';
        const race: RaceId | null | undefined = e.race ?? (who !== null ? g.players[who]?.active : null);
        out.detail = race
          ? `Vous ne pourrez pas attaquer ses ${RACES[race].name} à votre prochain tour`
          : 'Vous ne pourrez pas l\'attaquer à votre prochain tour';
        out.tone = 'hit';
      } else out.main = `🕊️ impose la paix à ${nameOf(g, e.victim)}`;
      break;
    }
    case 'score':
      if (e.gain !== undefined) out.main = `💰 termine son tour · +${e.gain}`;
      break;
    case 'end':
      out.main = '🏁 Fin de la partie';
      break;
  }
  // une conquête (seule, ou après le dé) qui me prend une région
  const hit = frame.find((x) => x.kind === 'conquer' && viewer !== null && x.victim === viewer && x.player !== viewer);
  if (hit) {
    out.tone = 'hit';
    out.detail = lossLine(g, hit, viewer!);
    out.loss = hit.lost ?? 1;
  }
  return out;
}

/** Régions voisines des miennes (sans les miennes). */
function nearMe(g: GameState, me: number): Set<number> {
  const mine = new Set<number>();
  g.regions.forEach((r, i) => {
    if (r.race && r.race !== 'lost' && g.races[r.race]?.owner === me) mine.add(i);
  });
  const near = new Set<number>();
  for (const id of mine) for (const n of g.map.regions[id]!.adj) if (!mine.has(n)) near.add(n);
  return near;
}

/** Ce qu'un adversaire a renforcé près de mes régions pendant son redéploiement (null : rien qui me concerne). */
export function narrateGroup(from: GameState, to: GameState, actor: number | null, viewer: number | null): Narration | null {
  if (actor === null) return null;
  const near = viewer === null ? null : nearMe(to, viewer);
  const items: { size: number; text: string }[] = [];
  to.regions.forEach((r, i) => {
    if (near && !near.has(i)) return;
    if (!r.race || r.race === 'lost' || to.races[r.race]?.owner !== actor) return;
    const was = from.regions[i]!;
    const same = was.race === r.race;
    const name = cutName(to.map.regions[i]!.name, 14);
    const tokens = r.tokens - (same ? was.tokens : 0);
    if (tokens > 0) items.push({ size: tokens, text: `${name} +${tokens}` });
    const camps = r.camps - (same ? was.camps : 0);
    if (camps > 0) items.push({ size: camps, text: `${FEATURE_ICONS.camp} ${name}${camps > 1 ? ` ×${camps}` : ''}` });
    if (r.fortress && !(same && was.fortress)) items.push({ size: 1, text: `${FEATURE_ICONS.fortress} ${name}` });
    if (r.hero && !(same && was.hero)) items.push({ size: 1, text: `${FEATURE_ICONS.hero} ${name}` });
  });
  if (!items.length) return null;
  items.sort((a, b) => b.size - a.size);
  const shown = items.slice(0, 3).map((x) => x.text);
  if (items.length > 3) shown.push(`+${items.length - 3}`);
  return { who: actor, main: '🔁 renforce près de vous', detail: shown.join(' · '), tone: 'neutral' };
}
