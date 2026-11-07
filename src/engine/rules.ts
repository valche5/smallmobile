/**
 * Ce que disent les règles d'une partie : le noyau répond (coûts, portée, attaquants, décompte…), ce
 * module met ses réponses en mots pour l'interface. S'y ajoutent de simples lectures de l'état.
 */
import { POWERS, RACES, dieChance } from './data';
import { kernelFor } from './engine';
import type { IncomeLine, Modifier, Special } from './kernel';
import { reasonText } from './reasons';
import type { GameState, PowerId, RaceId, RaceState, RegionDef, RegionState } from './types';

// --- lectures de l'état ------------------------------------------------------------------------

/** Copie indépendante de l'état (la carte et les entrées du journal, qui ne changent pas, sont partagées). */
export function cloneState(g: GameState): GameState {
  const races: GameState['races'] = {};
  for (const id in g.races) races[id as RaceId] = { ...g.races[id as RaceId]! };
  const t = g.turn;
  return {
    ...g,
    players: g.players.map((p) => ({ ...p, declined: p.declined.slice() })),
    races,
    regions: g.regions.map((r) => ({ ...r })),
    market: g.market.map((s) => ({ ...s })),
    raceDeck: g.raceDeck.slice(),
    powerDeck: g.powerDeck.slice(),
    turn: { ...t, attacked: t.attacked.slice(), sorcery: t.sorcery.slice(), readied: { ...t.readied } },
    defense: g.defense.map((d) => ({ ...d })),
    peace: g.peace.map((x) => ({ ...x })),
    log: g.log.slice(),
    lastDie: g.lastDie ? { ...g.lastDie } : null,
  };
}

export function isWater(def: RegionDef): boolean {
  return def.terrain === 'sea' || def.terrain === 'lake';
}

export function regionsOf(g: GameState, race: RaceId): number[] {
  const out: number[] = [];
  g.regions.forEach((r, i) => {
    if (r.race === race) out.push(i);
  });
  return out;
}

export function ownerOf(g: GameState, region: number): number | null {
  const r = g.regions[region]!;
  if (!r.race || r.race === 'lost') return null;
  return g.races[r.race]?.owner ?? null;
}

export function raceOf(g: GameState, race: RaceId): RaceState {
  const rs = g.races[race];
  if (!rs) throw new Error(`Peuple absent : ${race}`);
  return rs;
}

export function hasPower(g: GameState, race: RaceId | null, power: PowerId): boolean {
  const rs = race ? g.races[race] : undefined;
  return !!rs && !rs.declined && rs.power === power;
}

export function tokensOnBoard(g: GameState, race: RaceId): number {
  return g.regions.reduce((s, r) => s + (r.race === race ? r.tokens : 0), 0);
}

export function campsOnBoard(g: GameState, race: RaceId): number {
  return g.regions.reduce((s, r) => s + (r.race === race ? r.camps : 0), 0);
}

export function heroesOnBoard(g: GameState, race: RaceId): number {
  return g.regions.filter((r) => r.race === race && r.hero).length;
}

export function fortressesOnBoard(g: GameState, race: RaceId): number {
  return g.regions.filter((r) => r.race === race && r.fortress).length;
}

/** Le joueur qui doit agir. */
export function currentActor(g: GameState): number | null {
  if (g.phase === 'gameOver') return null;
  if (g.phase === 'defend') return g.defense[0]?.player ?? null;
  return g.current;
}

/** Le peuple que contrôle le joueur en train d'agir. */
export function actingRace(g: GameState): RaceId | null {
  if (g.phase === 'defend') return g.defense[0]?.race ?? null;
  if (g.phase === 'conquest' || g.phase === 'redeploy') return g.turn.army;
  return null;
}

export function isImmune(r: RegionState): string | null {
  if (r.hero) return 'Un héros protège cette région : elle est imprenable.';
  if (r.hole) return 'Un terrier de Semi-hommes protège cette région : elle est imprenable.';
  if (r.dragon) return 'Un dragon garde cette région : elle est imprenable.';
  return null;
}

/** Classement final (pièces, puis jetons sur la carte). */
export function ranking(g: GameState): { pid: number; coins: number; tokens: number }[] {
  return g.players
    .map((p) => ({
      pid: p.id,
      coins: p.coins,
      tokens: [p.active, ...p.declined].filter((r): r is RaceId => !!r).reduce((s, r) => s + tokensOnBoard(g, r), 0),
    }))
    .sort((a, b) => b.coins - a.coins || b.tokens - a.tokens);
}

// --- coût d'une conquête -----------------------------------------------------------------------

export interface CostPart {
  label: string;
  value: number;
  /** Libellé court pour la formule (« 3 Orques », « montagne »). */
  short?: string;
}

/** Détail de la défense d'une région (ce qu'il faut battre). */
export function defenseParts(g: GameState, region: number): CostPart[] {
  const def = g.map.regions[region]!;
  const r = g.regions[region]!;
  const parts: CostPart[] = [{ label: 'Base de toute conquête', value: 2, short: 'base' }];
  if (r.race === 'lost') parts.push({ label: 'Tribu oubliée', value: r.tokens, short: 'tribu' });
  else if (r.race) {
    const rs = g.races[r.race];
    const owner = rs ? g.players[rs.owner]!.name : '?';
    parts.push({ label: `${RACES[r.race].name}${rs?.declined ? ' en déclin' : ''} (${owner})`, value: r.tokens, short: RACES[r.race].name });
  }
  if (def.terrain === 'mountain') parts.push({ label: 'Montagne', value: 1, short: 'montagne' });
  if (r.lair) parts.push({ label: 'Tanière de troll', value: 1, short: 'tanière' });
  if (r.fortress) parts.push({ label: 'Forteresse', value: 1, short: 'forteresse' });
  if (r.camps > 0) parts.push({ label: `Campement${r.camps > 1 ? ` ×${r.camps}` : ''}`, value: r.camps, short: r.camps > 1 ? 'campements' : 'campement' });
  return parts;
}

/** Ce qu'il faut battre pour prendre la région. */
export function defenseValue(g: GameState, region: number): number {
  return defenseParts(g, region).reduce((s, p) => s + p.value, 0);
}

/** Formule compacte : « 2 + 3 Orques + 1 montagne − 1 Commandos ». */
export function costFormula(parts: CostPart[]): string {
  return parts
    .filter((p) => p.value !== 0)
    .map((p, i) => {
      const term = p.short === 'base' ? `${Math.abs(p.value)}` : `${Math.abs(p.value)} ${p.short ?? p.label}`;
      if (i === 0) return p.value < 0 ? `−${term}` : term;
      return `${p.value < 0 ? '−' : '+'} ${term}`;
    })
    .join(' ');
}

const MODIFIER_PARTS: Record<Modifier, CostPart> = {
  commando: { label: 'Commandos', value: -1, short: 'Commandos' },
  mounted: { label: 'Montés (Colline/Ferme)', value: -1, short: 'Montés' },
  underworld: { label: 'Souterrains (Souterrain)', value: -1, short: 'Souterrains' },
  tritons: { label: 'Tritons (côte)', value: -1, short: 'Tritons' },
  giants: { label: 'Géants (près de leur Montagne)', value: -1, short: 'Géants' },
};

const MODIFIER_ICONS: Record<Modifier, string> = {
  commando: POWERS.commando.icon,
  mounted: POWERS.mounted.icon,
  underworld: POWERS.underworld.icon,
  tritons: RACES.tritons.icon,
  giants: RACES.giants.icon,
};

export type TargetKind = 'normal' | 'die' | 'none';

export interface TargetInfo {
  region: number;
  /** La région peut être visée (même si les jetons manquent). */
  reachable: boolean;
  reason: string | null;
  parts: CostPart[];
  /** Coût en jetons (avant dé berserk). */
  cost: number;
  /** Coût effectif tenant compte du dé berserk déjà lancé. */
  effective: number;
  kind: TargetKind;
  /** Jetons manquants pour une tentative au dé. */
  need: number;
  chance: number;
  dragon: boolean;
  dragonReason: string | null;
  sorcery: boolean;
  sorceryReason: string | null;
}

function specialReason(g: GameState, s: Special): string | null {
  return s.kind === 'denied' ? reasonText(g, s.reason) : null;
}

/** Toutes les régions vues par le peuple qui conquiert. `lean` : sans le détail du coût. */
export function allTargets(g: GameState, lean = false): TargetInfo[] {
  const conquest = g.phase === 'conquest' && !!g.turn.army;
  const hand = conquest ? (g.races[g.turn.army!]?.hand ?? 0) : 0;
  return kernelFor(g)
    .targets()
    .map((t) => {
      const parts: CostPart[] = [];
      if (conquest && !lean) {
        parts.push(...defenseParts(g, t.region), ...t.modifiers.map((m) => MODIFIER_PARTS[m]));
        const raw = parts.reduce((s, p) => s + p.value, 0);
        if (raw < 1) parts.push({ label: 'Minimum 1 jeton', value: 1 - raw, short: 'minimum' });
        if (hasPower(g, g.turn.army, 'berserk') && g.turn.berserkDie !== null)
          parts.push({ label: `Dé berserk (${g.turn.berserkDie})`, value: t.effective - t.cost, short: 'dé' });
      }
      return {
        region: t.region,
        reachable: !t.block,
        reason: t.block ? reasonText(g, t.block) : hand < 1 ? "Vous n'avez plus de jeton en main." : null,
        parts,
        cost: t.cost,
        effective: t.effective,
        kind: t.kind,
        need: t.need,
        chance: t.kind === 'die' ? dieChance(t.need) : 0,
        dragon: t.dragon.kind === 'allowed',
        dragonReason: specialReason(g, t.dragon),
        sorcery: t.sorcery.kind === 'allowed',
        sorceryReason: specialReason(g, t.sorcery),
      };
    });
}

/** Les cibles sans le détail des coûts. */
export function scanTargets(g: GameState): TargetInfo[] {
  return allTargets(g, true);
}

export function analyzeTarget(g: GameState, region: number): TargetInfo {
  return allTargets(g)[region]!;
}

// --- menaces -----------------------------------------------------------------------------------

/** Un adversaire capable de prendre une région avant le prochain tour de son propriétaire. */
export interface Attacker {
  pid: number;
  /** null : il choisira un nouveau peuple, qui peut entrer par le bord de la carte. */
  race: RaceId | null;
  /** Jetons à engager (0 : sorcellerie, 1 : dragon). */
  cost: number;
  how: 'conquer' | 'dragon' | 'sorcery';
  /** Premier modificateur qui baisse le coût (« 🗡️ −1 »), ou le dé des Berserks. */
  mod: string | null;
  /** Attaque possible seulement de loin (Volants, souterrains reliés). */
  far: boolean;
  via: 'flying' | 'cavern' | null;
}

/** Les adversaires qui pourraient prendre cette région à `ownerPid` avant son prochain tour. */
export function attackersOf(g: GameState, region: number, ownerPid: number): Attacker[] {
  return kernelFor(g)
    .attackers(region, ownerPid)
    .map((a) => ({
      pid: a.pid,
      race: a.race,
      cost: a.cost,
      how: a.how,
      mod: a.hint?.kind === 'modifier' ? `${MODIFIER_ICONS[a.hint.modifier]} −1` : a.hint?.kind === 'berserk' ? `${POWERS.berserk.icon} jusqu'à −3` : null,
      far: a.via !== null,
      via: a.via,
    }));
}

/** Danger qui pèse sur une région du peuple (plus c'est haut, plus il faut la défendre). */
export function threat(g: GameState, race: RaceId, region: number): number {
  return kernelFor(g).threat(race, region);
}

/** Placement complet de fin de tour (« tout replacer ») : jetons visés par région du peuple. */
export function fullDeployment(g: GameState, race: RaceId): Record<number, number> {
  const target = kernelFor(g).fullDeployment(race);
  return Object.fromEntries(regionsOf(g, race).map((r) => [r, target[r]!]));
}

// --- ce qui est permis -------------------------------------------------------------------------

export function canDecline(g: GameState): string | null {
  const r = kernelFor(g).permissions().decline;
  return r ? reasonText(g, r) : null;
}

export function canAbandon(g: GameState, region: number): string | null {
  const r = kernelFor(g).permissions().abandon[region];
  return r ? reasonText(g, r) : null;
}

// --- décompte ----------------------------------------------------------------------------------

export interface ScoreLine {
  label: string;
  value: number;
  icon: string;
}

export interface ScoreBreakdown {
  lines: ScoreLine[];
  total: number;
}

const s = (n: number) => (n > 1 ? 's' : '');

/** Une ligne du décompte, en mots. */
export function scoreLine(l: IncomeLine): ScoreLine {
  const n = l.value;
  const power = (p: PowerId) => POWERS[p].icon;
  switch (l.kind) {
    case 'regions': return { label: `${n} région${s(n)} ${RACES[l.race].name}${l.declined ? ' (déclin)' : ''}`, value: n, icon: RACES[l.race].icon };
    case 'mines': return { label: `Nains : ${n} Mine${s(n)}`, value: n, icon: '⛏️' };
    case 'farms': return { label: `Humains : ${n} Ferme${s(n)}`, value: n, icon: '🌾' };
    case 'magic': return { label: `Mages : ${n} Source${s(n)} magique${s(n)}`, value: n, icon: '✨' };
    case 'orcLoot': return { label: `Orques : ${n} région(s) non vide(s) conquise(s)`, value: n, icon: '👹' };
    case 'alchemist': return { label: 'Alchimistes', value: n, icon: power('alchemist') };
    case 'forests': return { label: `Sylvestres : ${n} Forêt${s(n)}`, value: n, icon: power('forest') };
    case 'hills': return { label: `des Collines : ${n} Colline${s(n)}`, value: n, icon: power('hill') };
    case 'swamps': return { label: `des Marais : ${n} Marais`, value: n, icon: power('swamp') };
    case 'merchant': return { label: `Marchands : ${n} région(s)`, value: n, icon: power('merchant') };
    case 'pillage': return { label: `Pillards : ${n} région(s) non vide(s) conquise(s)`, value: n, icon: power('pillaging') };
    case 'fortresses': return { label: `Fortifiés : ${n} forteresse${s(n)}`, value: n, icon: power('fortified') };
    case 'wealthy': return { label: 'Fortunés (premier tour)', value: n, icon: power('wealthy') };
  }
}

export function breakdown(lines: IncomeLine[]): ScoreBreakdown {
  const out = lines.map(scoreLine);
  return { lines: out, total: out.reduce((a, l) => a + l.value, 0) };
}

/** Pièces que rapporterait la fin du tour pour ce joueur, maintenant. */
export function scoreBreakdown(g: GameState, pid: number): ScoreBreakdown {
  return breakdown(kernelFor(g).income(pid));
}
