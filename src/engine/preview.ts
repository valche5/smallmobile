import { DIE_FACES, FEATURE_ICONS, POWERS, RACES, TERRAINS } from './data';
import { applyAction, declinePreviewState } from './game';
import {
  allTargets,
  analyzeTarget,
  attackersOf,
  campsOnBoard,
  cloneState,
  defenseValue,
  fortressesOnBoard,
  heroesOnBoard,
  isWater,
  raceOf,
  regionsOf,
  scoreBreakdown,
  tokensOnBoard,
  type Attacker,
  type CostPart,
  type ScoreBreakdown,
} from './rules';
import type { GameState, RaceId, RegionState } from './types';

/** Une conséquence lisible d'une action. */
export interface Line {
  icon: string;
  text: string;
  tone?: 'good' | 'bad' | 'warn' | 'info';
}

function pl(n: number, word: string, plural?: string): string {
  return `${n} ${n > 1 ? (plural ?? `${word}s`) : word}`;
}

function pct(x: number): string {
  return `${Math.round(x * 100)} %`;
}

/** Pièces par tour que rapporte une région à son occupant (hors bonus de conquête). */
export function regionIncome(g: GameState, region: number): { total: number; parts: string[] } {
  const r = g.regions[region]!;
  const def = g.map.regions[region]!;
  if (!r.race || r.race === 'lost') return { total: 0, parts: [] };
  const rs = raceOf(g, r.race);
  const parts = ['Région +1'];
  if (r.race === 'dwarves' && def.mine) parts.push('Nains (Mine) +1');
  if (!rs.declined) {
    if (r.race === 'humans' && def.terrain === 'farmland') parts.push('Humains (Ferme) +1');
    if (r.race === 'wizards' && def.magic) parts.push('Mages (Source magique) +1');
    if (rs.power === 'forest' && def.terrain === 'forest') parts.push('Sylvestres +1');
    if (rs.power === 'hill' && def.terrain === 'hill') parts.push('des Collines +1');
    if (rs.power === 'swamp' && def.terrain === 'swamp') parts.push('des Marais +1');
    if (rs.power === 'merchant') parts.push('Marchands +1');
    if (rs.power === 'fortified' && r.fortress) parts.push('Forteresse +1');
  }
  return { total: parts.length, parts };
}

function victimIncomeLoss(g: GameState, after: GameState, pid: number): number {
  return scoreBreakdown(g, pid).total - scoreBreakdown(after, pid).total;
}

function futureLine(after: GameState): Line {
  const army = after.turn.army;
  if (!army || after.phase !== 'conquest') return { icon: '➡️', text: 'Vos conquêtes seront terminées.', tone: 'info' };
  const hand = raceOf(after, army).hand;
  const targets = allTargets(after);
  const normal = targets.filter((t) => t.kind === 'normal');
  if (normal.length) {
    const cheapest = Math.min(...normal.map((t) => t.effective));
    return {
      icon: '➡️',
      text: `Ensuite : ${pl(hand, 'jeton')} en main, ${pl(normal.length, 'région')} encore à portée (la moins chère : ${cheapest}).`,
      tone: 'info',
    };
  }
  const die = targets.filter((t) => t.kind === 'die');
  if (die.length) {
    const best = Math.max(...die.map((t) => t.chance));
    return {
      icon: '➡️',
      text: `Ensuite : plus de conquête sûre. Il restera le dé de renfort (jusqu'à ${pct(best)} de réussite).`,
      tone: 'warn',
    };
  }
  return { icon: '➡️', text: 'Ensuite : plus aucune conquête possible ce tour.', tone: 'warn' };
}

export type ConquestHow = 'conquer' | 'dragon' | 'sorcery';

export interface ConquestPreview {
  tokens: number;
  lines: Line[];
}

export function previewConquest(g: GameState, region: number, how: ConquestHow): ConquestPreview | null {
  const info = analyzeTarget(g, region);
  const allowed = how === 'conquer' ? info.kind === 'normal' : how === 'dragon' ? info.dragon : info.sorcery;
  if (!allowed) return null;
  const me = g.current;
  const army = g.turn.army!;
  const rs = raceOf(g, army);
  const r = g.regions[region]!;
  const after = applyAction(g, { type: how, region });
  const lines: Line[] = [];
  const spend = how === 'conquer' ? info.effective : how === 'dragon' ? 1 : 0;

  if (how === 'conquer') {
    lines.push({ icon: '🖐️', text: `Vous posez ${pl(spend, 'jeton')} ici : il vous en restera ${rs.hand - spend} en main.` });
  } else if (how === 'dragon') {
    lines.push({ icon: '🐉', text: `Le dragon prend la région avec 1 seul jeton au lieu de ${info.cost}. Il vous en restera ${rs.hand - 1} en main.`, tone: 'good' });
    lines.push({ icon: '🛡️', text: 'Région imprenable tant que le dragon y reste (jusqu\'à son prochain envol).', tone: 'good' });
    const old = g.regions.findIndex((x) => x.dragon);
    if (old >= 0 && old !== region) lines.push({ icon: '⚠️', text: `Le dragon quitte « ${g.map.regions[old]!.name} », qui redevient attaquable.`, tone: 'warn' });
  } else {
    lines.push({ icon: '🔮', text: `Un nouveau jeton Sorcier sort de la réserve : votre main reste à ${rs.hand}.`, tone: 'good' });
  }

  if (r.race === 'lost') {
    lines.push({ icon: '🛖', text: 'La tribu oubliée est éliminée.' });
  } else if (r.race) {
    const ds = raceOf(g, r.race);
    const owner = g.players[ds.owner]!;
    const rname = RACES[r.race].name;
    if (ds.owner === me) {
      lines.push({ icon: '⚠️', text: `Vous sacrifiez une région de vos propres ${rname} en déclin.`, tone: 'warn' });
    } else if (how === 'sorcery') {
      lines.push({ icon: '💥', text: `${owner.name} perd son jeton ${rname} : il est retiré du jeu.`, tone: 'good' });
    } else {
      const elves = r.race === 'elves' && !ds.declined;
      const lose = elves ? 0 : 1;
      const back = r.tokens - lose;
      if (elves) lines.push({ icon: '🧝', text: `Les Elfes ne perdent aucun jeton : ${owner.name} récupère ses ${r.tokens} jetons.`, tone: 'warn' });
      else
        lines.push({
          icon: '💥',
          text: `${owner.name} perd 1 jeton ${rname} pour de bon${back > 0 ? ` et en récupère ${back}, qu'il redéploiera à la fin de votre tour` : ''}.`,
          tone: 'good',
        });
    }
    if (r.lair) lines.push({ icon: '🪨', text: 'La tanière de troll est détruite.' });
    if (r.fortress) lines.push({ icon: '🏰', text: 'La forteresse est détruite.' });
    if (r.camps) lines.push({ icon: '⛺', text: `${pl(r.camps, 'campement')} retourne${r.camps > 1 ? 'nt' : ''} à ${owner.name}.` });
    if (ds.owner !== me) {
      const loss = victimIncomeLoss(g, after, ds.owner);
      if (loss > 0) lines.push({ icon: '📉', text: `${owner.name} gagnera ${pl(loss, 'pièce')} de moins à son prochain décompte.`, tone: 'good' });
      if (!ds.declined && rs.power === 'diplomat' && !rs.declined && !g.turn.attacked.includes(ds.owner))
        lines.push({ icon: '🕊️', text: `Vous ne pourrez plus imposer la paix à ${owner.name} ce tour.`, tone: 'warn' });
    }
  } else {
    lines.push({ icon: '🌿', text: 'Région vide : aucun défenseur ne perd de jeton.' });
  }

  const gain = scoreBreakdown(after, me).total - scoreBreakdown(g, me).total;
  const income = regionIncome(after, region);
  const extra = gain - income.total;
  const detail = [...income.parts, ...(extra > 0 ? [`Bonus de conquête +${extra}`] : [])].join(', ');
  lines.push({
    icon: '💰',
    text: `${gain >= 0 ? '+' : ''}${pl(gain, 'pièce')} à la fin de ce tour (${detail}).`,
    tone: gain > 0 ? 'good' : gain < 0 ? 'bad' : undefined,
  });

  if (army === 'halflings' && !rs.declined && rs.holesPlaced < 2)
    lines.push({ icon: '🏡', text: `Terrier (${rs.holesPlaced + 1}/2) : cette région deviendra imprenable.`, tone: 'good' });
  if (army === 'trolls') lines.push({ icon: '🪨', text: 'Une tanière de troll y sera bâtie : +1 défense, même en déclin.', tone: 'good' });
  if (army === 'skeletons' && !rs.declined && r.race)
    lines.push({ icon: '💀', text: 'Compte pour les renforts des Squelettes (1 jeton pour 2 régions non vides).', tone: 'good' });

  lines.push({
    icon: '🛡️',
    text: `Pour vous la reprendre avant votre redéploiement, il faudra ${defenseValue(after, region)} jetons.`,
    tone: 'info',
  });
  lines.push(futureLine(after));
  return { tokens: spend, lines };
}

/** Coupe un nom de région trop long pour tenir sur une ligne (« Sylve de Mirelun… »). */
export function cut(name: string, max = 18): string {
  const chars = [...name];
  return chars.length > max ? `${chars.slice(0, max).join('').trimEnd()}…` : name;
}

/** Une conquête possible sur cette région : conquête normale, dragon, sorcellerie ou dé de renfort. */
export type AttackHow = ConquestHow | 'die';

/** État juste après la conquête (le dé est supposé réussi). */
function conquestAfter(g: GameState, region: number, how: AttackHow): GameState {
  if (how !== 'die') return applyAction(g, { type: how, region });
  const c = cloneState(g);
  const rs = raceOf(c, c.turn.army!);
  rs.hand = Math.max(rs.hand, analyzeTarget(c, region).effective);
  return applyAction(c, { type: 'conquer', region });
}

/** Pourquoi la conquête rapporte plus (ou moins) que la pièce habituelle : « Ferme +1 », « victime +1 ». */
function gainReasons(g: GameState, region: number, race: RaceId): string[] {
  const def = g.map.regions[region]!;
  const r = g.regions[region]!;
  const rs = raceOf(g, race);
  const out: string[] = [];
  if (race === 'dwarves' && def.mine) out.push('Mine +1');
  if (rs.declined) return out;
  if (race === 'humans' && def.terrain === 'farmland') out.push('Ferme +1');
  if (race === 'wizards' && def.magic) out.push('Source +1');
  if (rs.power === 'forest' && def.terrain === 'forest') out.push('Forêt +1');
  if (rs.power === 'hill' && def.terrain === 'hill') out.push('Colline +1');
  if (rs.power === 'swamp' && def.terrain === 'swamp') out.push('Marais +1');
  if (rs.power === 'merchant') out.push('Marchands +1');
  const victims = (race === 'orcs' ? 1 : 0) + (rs.power === 'pillaging' ? 1 : 0);
  if (r.race !== null && g.turn.stage === 'active' && victims) out.push(`victime +${victims}`);
  return out;
}

/** L'essentiel d'une conquête : 3 conséquences au plus, d'une ligne chacune ; le reste part dans Détails. */
export interface AttackBrief {
  lines: Line[];
  more: Line[];
}

export function attackBrief(g: GameState, region: number, how: AttackHow): AttackBrief {
  const me = g.current;
  const army = g.turn.army!;
  const rs = raceOf(g, army);
  const r = g.regions[region]!;
  const after = conquestAfter(g, region, how);
  const all: Line[] = [];
  const ds = r.race && r.race !== 'lost' ? raceOf(g, r.race) : null;
  const owner = ds ? g.players[ds.owner]!.name : '';
  if (ds && r.race && r.race !== 'lost' && ds.owner === me)
    all.push({ icon: '🪦', text: `Vos ${RACES[r.race].name} en déclin perdent la région`, tone: 'warn' });
  if (how === 'dragon') {
    const old = g.regions.findIndex((x) => x.dragon);
    if (old >= 0 && old !== region) all.push({ icon: '⚠️', text: `Le dragon quittera « ${cut(g.map.regions[old]!.name)} »`, tone: 'warn' });
  }
  if (ds && ds.owner !== me && !ds.declined && rs.power === 'diplomat' && !rs.declined && !g.turn.attacked.includes(ds.owner))
    all.push({ icon: '🕊️', text: `Plus de paix possible avec ${owner} ce tour`, tone: 'warn' });
  if (army === 'halflings' && !rs.declined && rs.holesPlaced < 2)
    all.push({ icon: FEATURE_ICONS.hole, text: `Terrier ${rs.holesPlaced + 1}/2 : imprenable ensuite`, tone: 'good' });
  if (ds && ds.owner !== me && r.race === 'elves' && !ds.declined && how !== 'sorcery')
    all.push({ icon: RACES.elves.icon, text: `Elfes : ${owner} ne perd aucun jeton`, tone: 'warn' });
  // une région rapporte normalement 1 pièce : on ne parle du gain que s'il sort de l'ordinaire
  const gain = scoreBreakdown(after, me).total - scoreBreakdown(g, me).total;
  if (gain !== 1) {
    const why = gainReasons(g, region, army);
    all.push({ icon: '💰', text: `${gain >= 0 ? '+' : '−'}${Math.abs(gain)} ce tour${why.length ? ` (${why.join(', ')})` : ''}`, tone: gain > 1 ? 'good' : 'warn' });
  }
  return { lines: all.slice(0, 3), more: all.slice(3) };
}

/** Ce que « Détails ▾ » ajoute sous le tableau du coût : la victime, la défense ensuite, la suite du tour. */
export function attackDetails(g: GameState, region: number, how: AttackHow): Line[] {
  const me = g.current;
  const r = g.regions[region]!;
  const after = conquestAfter(g, region, how);
  const lines: Line[] = [];
  if (r.race === 'lost') lines.push({ icon: FEATURE_ICONS.lost, text: 'La tribu oubliée est chassée' });
  else if (r.race) {
    const ds = raceOf(g, r.race);
    const owner = g.players[ds.owner]!.name;
    const rname = RACES[r.race].name;
    if (ds.owner !== me) {
      const loss = victimIncomeLoss(g, after, ds.owner);
      const income = loss > 0 ? ` · −${loss} 💰/tour pour lui` : '';
      if (how === 'sorcery') lines.push({ icon: '💥', text: `${owner} perd son jeton ${rname}, remplacé par un Sorcier${income}` });
      else if (r.race === 'elves' && !ds.declined) lines.push({ icon: RACES.elves.icon, text: `${owner} ne perd aucun jeton · ${r.tokens} lui reviennent${income}` });
      else if (ds.declined && r.race !== 'ghouls') lines.push({ icon: '💥', text: `${owner} perd ${pl(r.tokens, 'jeton')} ${rname} (en déclin)${income}` });
      else {
        const back = r.tokens - 1;
        lines.push({ icon: '💥', text: `${owner} perd 1 jeton${back > 0 ? ` · ${back} lui ${back > 1 ? 'reviennent' : 'revient'}` : ''}${income}` });
      }
    }
    if (r.lair) lines.push({ icon: FEATURE_ICONS.lair, text: 'La tanière est détruite' });
    if (r.fortress) lines.push({ icon: FEATURE_ICONS.fortress, text: 'La forteresse est détruite' });
    if (r.camps) lines.push({ icon: FEATURE_ICONS.camp, text: `${pl(r.camps, 'campement')} rendu${r.camps > 1 ? 's' : ''} à ${owner}` });
  }
  if (how === 'dragon') lines.push({ icon: FEATURE_ICONS.dragon, text: "Imprenable tant que le dragon y reste" });
  if (how === 'die') {
    const info = analyzeTarget(g, region);
    lines.push({ icon: '🎲', text: `Faces : ${DIE_FACES.join(', ')} · il faut ${info.need} ou plus` });
  } else {
    // dragon ou terrier : imprenable ensuite (déjà dit par sa propre ligne), pas de coût de reprise
    if (!immuneKind(after.regions[region]!))
      lines.push({ icon: '⚔️', text: `Pour vous la reprendre : ${pl(defenseValue(after, region), 'jeton')}` });
    lines.push(nextLine(after));
  }
  return lines;
}

/** « Ensuite : 2 jetons, 4 régions à portée (la moins chère : 2) ». */
function nextLine(after: GameState): Line {
  const army = after.turn.army;
  if (!army || after.phase !== 'conquest') return { icon: '➡️', text: 'Ensuite : conquêtes terminées' };
  const hand = raceOf(after, army).hand;
  const targets = allTargets(after);
  const normal = targets.filter((t) => t.kind === 'normal');
  if (normal.length)
    return {
      icon: '➡️',
      text: `Ensuite : ${pl(hand, 'jeton')}, ${pl(normal.length, 'région')} à portée (la moins chère : ${Math.min(...normal.map((t) => t.effective))})`,
    };
  if (targets.some((t) => t.kind === 'die')) return { icon: '➡️', text: `Ensuite : ${pl(hand, 'jeton')}, plus que le dé de renfort` };
  return { icon: '➡️', text: `Ensuite : ${pl(hand, 'jeton')}, plus de conquête possible` };
}

const PART_ICONS: Record<string, string> = {
  montagne: TERRAINS.mountain.icon,
  tanière: FEATURE_ICONS.lair,
  forteresse: FEATURE_ICONS.fortress,
  campement: FEATURE_ICONS.camp,
  campements: FEATURE_ICONS.camp,
  Commandos: POWERS.commando.icon,
  Montés: POWERS.mounted.icon,
  Souterrains: POWERS.underworld.icon,
  Tritons: RACES.tritons.icon,
  Géants: RACES.giants.icon,
  dé: '🎲',
};

const RACE_NAMES = new Set(Object.values(RACES).map((r) => r.name));

/** Les jetons de l'occupant (peuple ou tribu oubliée). */
function isTokens(p: CostPart): boolean {
  return p.short === 'tribu' || (p.value > 0 && RACE_NAMES.has(p.short ?? ''));
}

/** « 2 + 2 jetons + 🏔️ 1 − 🗡️ 1 » : la formule avec des icônes. */
export function iconFormula(parts: CostPart[]): string {
  return parts
    .filter((p) => p.value !== 0 && p.short !== 'minimum')
    .map((p, i) => {
      const n = Math.abs(p.value);
      const term = p.short === 'base' ? `${n}` : isTokens(p) ? pl(n, 'jeton') : `${PART_ICONS[p.short ?? ''] ?? p.short ?? p.label} ${n}`;
      if (i === 0) return p.value < 0 ? `−${term}` : term;
      return `${p.value < 0 ? '−' : '+'} ${term}`;
    })
    .join(' ');
}

/** « Coût 2 + 2 jetons + 🏔️ 1 = 5 », seulement s'il y a autre chose que la base et les jetons. */
export function costLine(parts: CostPart[], total: number): string | null {
  if (!parts.some((p) => p.value !== 0 && p.short !== 'base' && !isTokens(p))) return null;
  const raw = parts.filter((p) => p.short !== 'minimum').reduce((s, p) => s + p.value, 0);
  return `Coût ${iconFormula(parts)} = ${raw}${raw !== total ? ` → ${total} (minimum)` : ''}`;
}

/** Pourquoi on ne peut pas conquérir cette région, en une ligne (texte exact de la fiche). */
export function conquestBlock(g: GameState, region: number): string | null {
  const army = g.turn.army;
  if (g.phase !== 'conquest' || !army) return null;
  const rs = raceOf(g, army);
  const r = g.regions[region]!;
  const def = g.map.regions[region]!;
  const info = analyzeTarget(g, region);
  if (info.kind !== 'none' || info.dragon || info.sorcery) return null;
  if (g.turn.dieRolled) return '⛔ Dé lancé : conquêtes terminées';
  if (r.race && r.race !== 'lost') {
    const target = raceOf(g, r.race);
    if (target.owner === rs.owner && rs.declined) return '⛔ Vos Goules ne peuvent pas attaquer vos peuples';
    if (!target.declined && g.peace.some((p) => p.diplomat === target.owner && p.target === rs.owner))
      return `🕊️ Paix : ${g.players[target.owner]!.name} vous interdit d'attaquer ses ${RACES[r.race].name}`;
  }
  if (isWater(def) && !(rs.power === 'seafaring' && !rs.declined))
    return def.terrain === 'sea' ? '⛔ Mer : seuls les Marins la prennent' : '⛔ Lac : seuls les Marins le prennent';
  const immune = immuneKind(r);
  if (immune) return `⛔ Imprenable : ${IMMUNE_TEXT[immune]}`;
  if (!info.reachable) return regionsOf(g, army).length ? '⛔ Pas voisine de vos régions' : '⛔ Première conquête : par le bord de la carte';
  if (rs.hand < 1) return '⛔ Plus de jeton en main';
  return `⛔ Il faut ${pl(info.effective, 'jeton')}, vous en avez ${rs.hand}`;
}

export type ImmuneKind = 'héros' | 'dragon' | 'terrier';
const IMMUNE_TEXT: Record<ImmuneKind, string> = { héros: 'un héros la garde', dragon: 'le dragon la garde', terrier: 'terrier des Semi-hommes' };

export function immuneKind(r: RegionState): ImmuneKind | null {
  if (r.hero) return 'héros';
  if (r.dragon) return 'dragon';
  if (r.hole) return 'terrier';
  return null;
}

// --- Menace ennemie : ce qu'un adversaire paiera pour MA région --------------------------

/** L'adversaire qui prendrait cette région le moins cher avant mon prochain tour (à égalité, celui qui joue le plus tôt). */
export function foeThreat(g: GameState, region: number, me: number): Attacker | null {
  const n = g.players.length;
  // compté depuis le joueur en cours : c'est moi pendant mon redéploiement, l'attaquant pendant ma défense
  const order = (a: Attacker) => (a.pid - g.current + n) % n;
  let best: Attacker | null = null;
  for (const a of attackersOf(g, region, me)) if (!best || a.cost < best.cost || (a.cost === best.cost && order(a) < order(best))) best = a;
  return best;
}

export interface ThreatText {
  text: string;
  /** Rien à craindre (imprenable, hors de portée) : en gris. */
  muted: boolean;
  /** Couleur de l'ennemi le plus dangereux. */
  color: string | null;
}

/** « ⚔ Bob la prend avec 4 (🗡️ −1) », identique dans le panneau de redéploiement et la consultation. */
export function foeThreatText(g: GameState, region: number, me: number): ThreatText {
  const immune = immuneKind(g.regions[region]!);
  if (immune) return { text: `✓ Imprenable (${immune})`, muted: true, color: null };
  const t = foeThreat(g, region, me);
  if (!t) return { text: '✓ Hors de portée des ennemis', muted: true, color: null };
  const p = g.players[t.pid]!;
  const who = p.name;
  let text: string;
  if (t.how === 'sorcery') text = `${RACES.sorcerers.icon} ${who} la prend gratis : 1 seul jeton`;
  else if (t.how === 'dragon') text = `${FEATURE_ICONS.dragon} ${who} la prend avec 1 jeton (dragon)`;
  else if (t.via === 'flying') text = `${POWERS.flying.icon} ${who} la prend avec ${t.cost}, de n'importe où`;
  else if (t.via === 'cavern') text = `${POWERS.underworld.icon} ${who} la prend avec ${t.cost} par les souterrains`;
  else if (!t.race) text = `⚔ ${who} la prend avec ${t.cost} (nouveau peuple)`;
  else text = `⚔ ${who} la prend avec ${t.cost}${t.mod ? ` (${t.mod})` : ''}`;
  return { text, muted: false, color: p.color };
}

/** La raison de la menace, au toucher : « Bob : 3 jetons à côté », « Bob a des Volants ». */
export function foeThreatReason(g: GameState, region: number, me: number): string | null {
  const t = foeThreat(g, region, me);
  if (!t) return null;
  const who = g.players[t.pid]!.name;
  if (!t.race) return `${who} peut entrer par le bord`;
  const rs = raceOf(g, t.race);
  if (t.how === 'sorcery') return `${who} a des Sorciers`;
  if (t.how === 'dragon') return `${who} a un dragon (${POWERS.dragonMaster.name})`;
  if (t.via === 'flying') return `${who} a des Volants`;
  if (t.via === 'cavern') return `${who} a des Souterrains`;
  const near = g.map.regions[region]!.adj.reduce((s, n) => s + (g.regions[n]!.race === t.race ? g.regions[n]!.tokens : 0), 0);
  const ghouls = rs.declined ? ' (Goules)' : '';
  if (!near) return `${who} peut entrer ici${ghouls}`;
  return `${who} : ${pl(near, 'jeton')} à côté${ghouls}`;
}

// --- État de mon pouvoir et rappels -------------------------------------------------------

/** Ce que mon pouvoir permet encore ce tour-ci (seulement ce qui change) : « 🐉 Dragon prêt », « ⛺ Placez ou déplacez des bivouacs · 2 en réserve ». */
export function powerStatus(g: GameState, pid: number, compact = false): string | null {
  if (g.current !== pid || g.turn.stage !== 'active') return null;
  const race = g.players[pid]!.active;
  if (!race || g.turn.army !== race) return null;
  const rs = g.races[race];
  if (!rs || rs.declined) return null;
  if (g.phase === 'conquest') {
    if (rs.power === 'dragonMaster' && !g.turn.dragonUsed && rs.hand >= 1) return `${FEATURE_ICONS.dragon} Dragon prêt`;
    if (race === 'sorcerers') {
      const n = allTargets(g).filter((t) => t.sorcery).length;
      if (n) return `${RACES.sorcerers.icon} ${pl(n, 'cible ensorcelable', 'cibles ensorcelables')}`;
    }
    if (rs.power === 'berserk' && g.turn.berserkDie !== null)
      return g.turn.berserkDie ? `🎲 −${g.turn.berserkDie} sur la prochaine` : '🎲 0 : aucun bonus';
    if (race === 'halflings' && rs.holesPlaced < 2) return `${FEATURE_ICONS.hole} Terriers : ${rs.holesPlaced}/2`;
  } else if (g.phase === 'redeploy') {
    const left = powerLeft(g, race);
    // Bivouaqueurs : les campements se posent ET se déplacent à chaque redéploiement, même la réserve vide
    // (`compact` : la ligne dit déjà les jetons à placer, seule la réserve tient encore)
    if (rs.power === 'bivouacking' && regionsOf(g, race).length) {
      if (compact) return left.camps > 0 ? `${FEATURE_ICONS.camp} ${left.camps} en réserve` : null;
      return `${FEATURE_ICONS.camp} Placez ou déplacez des bivouacs${left.camps > 0 ? ` · ${left.camps} en réserve` : ''}`;
    }
    if (left.fortress) return `${FEATURE_ICONS.fortress} Forteresse à bâtir`;
    if (left.heroes > 0) return `${POWERS.heroic.icon} ${pl(left.heroes, 'héros', 'héros')} à placer`;
  }
  return null;
}

/** Ce que le pouvoir du peuple actif peut encore poser ce redéploiement, là où le moteur l'accepte (régions possédées). */
function powerLeft(g: GameState, race: RaceId): { camps: number; fortress: boolean; heroes: number } {
  const rs = raceOf(g, race);
  const own = regionsOf(g, race);
  const none = { camps: 0, fortress: false, heroes: 0 };
  if (rs.declined || !own.length) return none;
  if (rs.power === 'bivouacking') return { ...none, camps: 5 - campsOnBoard(g, race) };
  if (rs.power === 'fortified')
    return { ...none, fortress: g.turn.fortressRegion === null && fortressesOnBoard(g, race) < 6 && own.some((i) => !g.regions[i]!.fortress) };
  // retirer un héros pour le reposer ne compte pas : seulement les régions sans héros
  if (rs.power === 'heroic') return { ...none, heroes: Math.min(2 - heroesOnBoard(g, race), own.filter((i) => !g.regions[i]!.hero).length) };
  return none;
}

/** Le premier oubli de pouvoir avant de finir le tour (« ⚠️ Forteresse pas encore bâtie »). */
export function endTurnReminder(g: GameState): string | null {
  if (g.phase !== 'redeploy' || g.turn.stage !== 'active') return null;
  const race = g.turn.army;
  if (!race) return null;
  const left = powerLeft(g, race);
  if (left.fortress) return '⚠️ Forteresse pas encore bâtie';
  if (left.heroes > 0) return `⚠️ ${pl(left.heroes, 'héros non placé', 'héros non placés')}`;
  if (left.camps > 0) return `⚠️ ${pl(left.camps, 'campement')} en réserve`;
  return null;
}

/** Ce que le peuple a gagné ou rendu en passant au redéploiement (Amazones, Squelettes), dit une seule fois. */
export function redeployNote(g: GameState): string | null {
  if (g.phase !== 'redeploy' || g.turn.stage !== 'active') return null;
  const race = g.turn.army;
  if (race !== 'amazons' && race !== 'skeletons') return null;
  let start = 0;
  for (const l of g.log) if (l.player === g.current && (l.kind === 'turn' || l.kind === 'pick')) start = l.seq;
  const e = [...g.log].reverse().find((l) => l.seq > start && l.player === g.current && l.kind === 'power');
  // le nombre vient des données de l'entrée, jamais du texte (qui commence par le nom du joueur)
  const n = e?.tokens ?? 0;
  if (!n) return null;
  return race === 'amazons' ? `${RACES.amazons.icon} ${n} rendus` : `${RACES.skeletons.icon} +${pl(n, 'jeton')}`;
}

// --- Déclin et abandon -------------------------------------------------------------------

export interface DeclineBrief {
  gain: number;
  /** Les 2 lignes toujours présentes, puis 2 au plus par priorité. */
  lines: Line[];
  more: Line[];
  score: ScoreBreakdown;
}

export function declineBrief(g: GameState): DeclineBrief {
  const me = g.current;
  const p = g.players[me]!;
  const race = p.active!;
  const rs = raceOf(g, race);
  const after = declinePreviewState(g);
  const score = scoreBreakdown(after, me);
  const own = regionsOf(g, race).length;
  const lines: Line[] = [{ icon: '💰', text: `+${score.total} maintenant, puis votre tour s'arrête.` }];
  if (race === 'ghouls') lines.push({ icon: RACES.ghouls.icon, text: 'Vos Goules gardent leurs jetons et attaqueront encore.' });
  else if (own) lines.push({ icon: '', text: own > 1 ? `Vos ${own} régions gardent 1 jeton chacune.` : 'Votre région garde 1 jeton.' });
  else lines.push({ icon: '', text: 'Aucune région ne reste sur la carte.' });
  const extra: Line[] = [];
  const removed = p.declined.filter((r) => !after.players[me]!.declined.includes(r));
  for (const r of removed) extra.push({ icon: '🪦', text: `Vos ${RACES[r].icon} ${RACES[r].name} en déclin disparaissent.`, tone: 'bad' });
  if (rs.power === 'spirit' && p.declined.length && !removed.length)
    extra.push({ icon: POWERS.spirit.icon, text: `Vos ${RACES[p.declined[0]!].name} en déclin restent (Spirituels).` });
  if (g.round >= g.maxRounds) extra.push({ icon: '⚠️', text: 'Dernier tour : pas de nouveau peuple ensuite.', tone: 'bad' });
  if (rs.power === 'stout') extra.push({ icon: POWERS.stout.icon, text: "Astuce : conquérez d'abord, déclinez en fin de tour." });
  const keeps = RACES[race].inDecline;
  if (keeps && race !== 'ghouls') extra.push({ icon: '', text: `Reste actif : ${keeps.charAt(0).toLowerCase()}${keeps.slice(1)}.` });
  const more: Line[] = extra.slice(2);
  const markers: string[] = [];
  for (const i of regionsOf(g, race)) {
    const r = g.regions[i]!;
    if (r.camps) markers.push(pl(r.camps, 'campement'));
    if (r.hero) markers.push('1 héros');
    if (r.dragon) markers.push('le dragon');
    if (r.hole) markers.push('1 terrier');
  }
  if (markers.length) more.push({ icon: '📦', text: `Retirés de la carte : ${markers.join(', ')}.` });
  if (g.regions.some((r) => r.race === race && (r.fortress || r.lair))) more.push({ icon: FEATURE_ICONS.fortress, text: 'Forteresses et tanières restent.' });
  return { gain: score.total, lines: [...lines, ...extra.slice(0, 2)], more, score };
}

/** Abandon d'une de mes régions : une alerte au plus, le reste dans Détails. */
export function abandonBrief(g: GameState, region: number): { alert: string | null; more: Line[] } {
  const r = g.regions[region]!;
  const army = g.turn.army!;
  const rs = raceOf(g, army);
  const alerts: string[] = [];
  const anywhere = !rs.declined && (rs.power === 'flying' || army === 'halflings');
  if (regionsOf(g, army).length === 1 && !anywhere) alerts.push('⚠️ Dernière région : il faudra repartir du bord');
  if (r.hole) alerts.push('⚠️ Le terrier sera perdu');
  if (r.fortress) alerts.push('⚠️ La forteresse sera détruite');
  if (r.lair) alerts.push('⚠️ La tanière sera détruite');
  const more: Line[] = alerts.slice(1).map((a) => ({ icon: '', text: a }));
  if (r.hero) more.push({ icon: FEATURE_ICONS.hero, text: 'Le héros revient dans votre réserve' });
  if (r.camps) more.push({ icon: FEATURE_ICONS.camp, text: `${pl(r.camps, 'campement')} revien${r.camps > 1 ? 'nent' : 't'} dans votre réserve` });
  if (r.dragon) more.push({ icon: FEATURE_ICONS.dragon, text: 'Le dragon quitte la région' });
  return { alert: alerts[0] ?? null, more };
}

export interface DeclinePreview {
  lines: Line[];
  score: ScoreBreakdown;
}

export function previewDecline(g: GameState): DeclinePreview {
  const me = g.current;
  const p = g.players[me]!;
  const race = p.active!;
  const rs = raceOf(g, race);
  const after = declinePreviewState(g);
  const lines: Line[] = [];
  const own = regionsOf(g, race);
  // les jetons temporaires des Amazones ne sont pas une perte
  const before = tokensOnBoard(g, race) + rs.hand - (race === 'amazons' ? g.turn.amazonBonus : 0);
  const kept = tokensOnBoard(after, race);
  lines.push({
    icon: '💤',
    text:
      race === 'ghouls'
        ? `Vos Goules gardent leurs ${pl(kept, 'jeton')} sur ${pl(own.length, 'région')} et pourront encore conquérir aux prochains tours.`
        : `Vos ${pl(own.length, 'région')} ${RACES[race].name} gardent chacune 1 seul jeton : ${pl(before - kept, 'jeton')} retiré${before - kept > 1 ? 's' : ''}.`,
    tone: race === 'ghouls' ? 'good' : 'warn',
  });
  lines.push({ icon: '🚫', text: `Le pouvoir ${POWERS[rs.power].name} est perdu${rs.power === 'spirit' ? ' (sauf son effet de déclin)' : ''}, ainsi que les bonus de peuple actif.`, tone: 'warn' });
  const markers: string[] = [];
  for (const i of own) {
    const r = g.regions[i]!;
    if (r.camps) markers.push(pl(r.camps, 'campement'));
    if (r.hero) markers.push('1 héros');
    if (r.dragon) markers.push('le dragon');
    if (r.hole) markers.push('1 terrier');
  }
  if (markers.length) lines.push({ icon: '📦', text: `Retirés de la carte : ${markers.join(', ')}.`, tone: 'warn' });
  if (g.regions.some((r) => r.race === race && (r.fortress || r.lair)))
    lines.push({ icon: '🏰', text: 'Forteresses et tanières restent en place (défense seulement).' });
  const removed = p.declined.filter((r) => !after.players[me]!.declined.includes(r));
  for (const r of removed) {
    const n = regionsOf(g, r).length;
    lines.push({ icon: '🪦', text: `Vos ${RACES[r].name} en déclin disparaissent : ${pl(n, 'région')} libérée${n > 1 ? 's' : ''} (−${n} pièce${n > 1 ? 's' : ''} par tour).`, tone: 'bad' });
  }
  if (rs.power === 'spirit' && p.declined.length)
    lines.push({ icon: '👻', text: `Spirituels : vos ${RACES[p.declined[0]!].name} en déclin sont conservés.`, tone: 'good' });
  const score = scoreBreakdown(after, me);
  const now = scoreBreakdown(g, me).total;
  lines.push({ icon: '💰', text: `Décompte immédiat : +${pl(score.total, 'pièce')} (sans déclin, vos régions actuelles valent ${now}).` });
  lines.push({ icon: '⏭️', text: 'Votre tour s\'arrête là : au prochain tour, vous choisirez un nouveau peuple.', tone: 'info' });
  if (rs.power === 'stout')
    lines.push({ icon: '🍺', text: 'Astuce : vos Robustes peuvent aussi conquérir ce tour, puis décliner en le terminant.', tone: 'good' });
  if (g.round >= g.maxRounds)
    lines.push({ icon: '⚠️', text: "C'est le dernier tour : décliner ne vous rapportera aucun nouveau peuple.", tone: 'bad' });
  return { lines, score };
}

export interface EndTurnPreview {
  score: ScoreBreakdown;
  lines: Line[];
}

export function previewEndTurn(g: GameState, opts: { peace?: number | null; stout?: boolean }): EndTurnPreview {
  const me = g.current;
  const p = g.players[me]!;
  const score = scoreBreakdown(g, me);
  const lines: Line[] = [];
  lines.push({ icon: '💰', text: `Vous gagnez ${pl(score.total, 'pièce')}.`, tone: 'good' });
  if (opts.stout && p.active) {
    lines.push({
      icon: '🍺',
      text: `Robustes : après le décompte, vos ${RACES[p.active].name} passent en déclin (1 jeton par région). Nouveau peuple au prochain tour.`,
      tone: 'warn',
    });
    const after = declinePreviewState(g);
    for (const r of p.declined.filter((r) => !after.players[me]!.declined.includes(r)))
      lines.push({ icon: '🪦', text: `Vos ${RACES[r].name} en déclin disparaîtront.`, tone: 'bad' });
  }
  if (opts.peace !== undefined && opts.peace !== null && p.active)
    lines.push({
      icon: '🕊️',
      text: `${g.players[opts.peace]!.name} ne pourra pas attaquer vos ${RACES[p.active].name} pendant son prochain tour.`,
      tone: 'good',
    });
  const pending = new Map<string, number>();
  for (const [race, rs] of Object.entries(g.races)) {
    if (!rs || rs.owner === me || rs.hand <= 0 || !regionsOf(g, race as RaceId).length) continue;
    if (rs.declined && race !== 'ghouls') continue;
    pending.set(`${g.players[rs.owner]!.name} (${RACES[race as RaceId].name})`, rs.hand);
  }
  for (const [who, n] of pending)
    lines.push({ icon: '🔁', text: `${who} redéploiera ${pl(n, 'jeton')} récupéré${n > 1 ? 's' : ''}.`, tone: 'info' });
  const last = g.round >= g.maxRounds && me === g.players.length - 1;
  if (last) lines.push({ icon: '🏁', text: 'C\'est le tout dernier tour : la partie se terminera.', tone: 'info' });
  else {
    const next = g.players[(me + 1) % g.players.length]!;
    lines.push({ icon: '⏭️', text: `Au tour de ${next.name}${next.kind === 'ai' ? ' (IA)' : ''}.`, tone: 'info' });
  }
  return { score, lines };
}

export function previewAbandon(g: GameState, region: number): Line[] {
  const r = g.regions[region]!;
  const def = g.map.regions[region]!;
  const army = g.turn.army!;
  const lines: Line[] = [
    { icon: '🖐️', text: `+${pl(r.tokens, 'jeton')} en main pour vos conquêtes.`, tone: 'good' },
    { icon: '📉', text: `Vous ne toucherez plus les ${pl(regionIncome(g, region).total, 'pièce')} de cette région à chaque tour.`, tone: 'bad' },
    {
      icon: '🌿',
      text: `La région redevient vide : n'importe qui pourra la prendre pour ${2 + (def.terrain === 'mountain' ? 1 : 0)} jetons.`,
      tone: 'warn',
    },
  ];
  if (r.hole) lines.push({ icon: '🏡', text: 'Le terrier est perdu définitivement.', tone: 'bad' });
  if (r.lair) lines.push({ icon: '🪨', text: 'La tanière est détruite.', tone: 'bad' });
  if (r.fortress) lines.push({ icon: '🏰', text: 'La forteresse est détruite.', tone: 'bad' });
  if (r.hero) lines.push({ icon: FEATURE_ICONS.hero, text: 'Le héros revient dans votre réserve.' });
  if (r.camps) lines.push({ icon: '⛺', text: `${pl(r.camps, 'campement')} revien${r.camps > 1 ? 'nent' : 't'} dans votre réserve.` });
  if (r.dragon) lines.push({ icon: '🐉', text: 'Le dragon quitte la région.' });
  const rs = raceOf(g, army);
  const anywhere = !rs.declined && (rs.power === 'flying' || army === 'halflings');
  if (regionsOf(g, army).length === 1 && !anywhere)
    lines.push({ icon: '⚠️', text: 'C\'était votre dernière région : votre prochaine conquête devra repartir du bord de la carte.', tone: 'warn' });
  return lines;
}

/** Régions qu'un adversaire m'a prises depuis mon dernier décompte (ou déclin), et qui ne sont plus à moi. */
export function lostRegions(g: GameState, pid: number): number[] {
  let since = 0;
  for (const l of g.log) if (l.player === pid && (l.kind === 'score' || l.kind === 'decline')) since = l.seq;
  const out = new Set<number>();
  for (const l of g.log) {
    if (l.seq <= since || l.kind !== 'conquer' || l.victim !== pid || l.player === pid || l.region === undefined) continue;
    const r = g.regions[l.region]!;
    const owner = r.race && r.race !== 'lost' ? g.races[r.race]?.owner : undefined;
    if (owner !== pid) out.add(l.region);
  }
  return [...out];
}
