/**
 * Le pont vers le noyau Rust (kernel/), qui contient les règles, l'IA et la génération des cartes,
 * compilé en WebAssembly pour le téléphone et le serveur : des mots de 32 bits échangés dans un
 * tampon. Le format est décrit dans kernel/src/wire.rs ; les deux fichiers vont ensemble.
 */
import { NOUNS, POWER_IDS, RACE_IDS, SUFFIXES } from './data';
import type { Action, AiLevel, GameState, MapDef, Phase, PowerId, RaceId, RegionDef, Terrain } from './types';

/** Version du protocole attendue (`ABI_VERSION` côté Rust). */
export const KERNEL_ABI = 3;
const MAGIC = 0x534d4b33;
const NONE = 255;
const LOST = 254;

const TERRAINS: Terrain[] = ['farmland', 'forest', 'hill', 'mountain', 'swamp', 'sea', 'lake'];
const PHASES: Phase[] = ['pick', 'conquest', 'redeploy', 'defend', 'gameOver'];
const LEVELS: AiLevel[] = ['facile', 'normal', 'difficile', 'expert'];
const ACTIONS: Action['type'][] = [
  'pick', 'conquer', 'reinforce', 'berserkRoll', 'dragon', 'sorcery', 'abandon', 'decline', 'endConquest',
  'deploy', 'autoDeploy', 'place', 'camp', 'fortress', 'hero', 'endTurn', 'endDefense', 'skipPick',
];

const raceIndex = (r: RaceId) => RACE_IDS.indexOf(r);
const raceWord = (r: RaceId | null) => (r === null ? NONE : raceIndex(r));
const raceOrNull = (w: number): RaceId | null => (w === NONE ? null : RACE_IDS[w]!);
const occupantWord = (r: RaceId | 'lost' | null) => (r === 'lost' ? LOST : raceWord(r));
const occupantOf = (w: number): RaceId | 'lost' | null => (w === LOST ? 'lost' : raceOrNull(w));
const mask = (players: number[]) => players.reduce((m, p) => m | (1 << p), 0);
const regionFlags = (r: GameState['regions'][number]) => +r.lair | (+r.fortress << 1) | (+r.hero << 2) | (+r.dragon << 3) | (+r.hole << 4);

/** Lecture séquentielle d'une réponse du noyau. */
class Words {
  at = 0;
  constructor(readonly xs: Uint32Array) {}
  get done() {
    return this.at >= this.xs.length;
  }
  u() {
    return this.xs[this.at++]!;
  }
  i() {
    return this.u() | 0;
  }
  yes() {
    return this.u() === 1;
  }
  opt() {
    const x = this.i();
    return x < 0 ? null : x;
  }
  f64() {
    const v = new DataView(this.xs.buffer, this.xs.byteOffset + this.at * 4, 8).getFloat64(0, true);
    this.at += 2;
    return v;
  }
  end(what: string) {
    if (this.at !== this.xs.length) throw new Error(`${what} du noyau illisible (longueur inattendue).`);
  }
}

// --- cartes ------------------------------------------------------------------------------------

/** Éloignement de chaque région au centre de la carte, en dixièmes (0 à 10). */
export function centreTenths(map: MapDef): number[] {
  const defs = map.regions;
  let cx = 0;
  let cy = 0;
  for (const d of defs) {
    cx += d.cx / defs.length;
    cy += d.cy / defs.length;
  }
  let reach = 1;
  for (const d of defs) reach = Math.max(reach, Math.hypot(d.cx - cx, d.cy - cy));
  return defs.map((d) => Math.round((10 * Math.hypot(d.cx - cx, d.cy - cy)) / reach));
}

function mapWords(map: MapDef, out: number[]) {
  if (map.regions.length > 64) throw new Error('Le noyau accepte au plus 64 régions.');
  if (!Number.isSafeInteger(map.seed) || map.seed < 0) throw new Error('Graine de carte invalide.');
  out.push(map.seed % 2 ** 32, Math.floor(map.seed / 2 ** 32), map.regions.length);
  const centre = centreTenths(map);
  map.regions.forEach((d, i) => {
    const flags = +d.cavern | (+d.magic << 1) | (+d.mine << 2) | (+d.entry << 3) | (+d.border << 4) | (+d.coastal << 5);
    out.push(TERRAINS.indexOf(d.terrain), flags, centre[i]!, d.adj.length, ...d.adj);
  });
}

/** Les régions d'une carte générée par le noyau (`wire::encode_generated_map`). */
export function readGeneratedRegions(xs: Uint32Array): { rounds: number; regions: RegionDef[] } {
  const r = new Words(xs);
  const rounds = r.u();
  const regions = Array.from({ length: r.u() }, (_, id): RegionDef => {
    const terrain = TERRAINS[r.u()]!;
    const flags = r.u();
    const cx = r.i();
    const cy = r.i();
    const [named, a, b] = [r.u(), r.u(), r.u()];
    const nouns = NOUNS[terrain];
    const name = named === 0 ? `${nouns[a]!} ${SUFFIXES[b]!}` : `${nouns[0]!} ${a}`;
    const adj = Array.from({ length: r.u() }, () => r.u());
    const points = Array.from({ length: r.u() }, () => [r.f64(), r.f64()] as const);
    return {
      id,
      name,
      terrain,
      cavern: !!(flags & 1),
      magic: !!(flags & 2),
      mine: !!(flags & 4),
      lostTribe: !!(flags & 8),
      border: !!(flags & 16),
      entry: !!(flags & 32),
      coastal: !!(flags & 64),
      adj,
      path: `M${points.map((p) => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('L')}Z`,
      cx,
      cy,
    };
  });
  r.end('Carte');
  return { rounds, regions };
}

// --- état --------------------------------------------------------------------------------------

/** L'état de la partie, dans l'ordre de `wire::encode_state`. */
export function stateWords(g: GameState): Uint32Array {
  const out: number[] = [g.players.length, g.current, PHASES.indexOf(g.phase), g.round, g.maxRounds, g.seq, g.rng, g.lastRegion ?? -1];
  const d = g.lastDie;
  out.push(+!!d);
  if (d) out.push(d.seq, d.player, d.value, d.need, d.success === null ? -1 : +d.success, d.region ?? -1, +d.berserk);
  for (const p of g.players) out.push(LEVELS.indexOf(p.ai), p.coins, raceWord(p.active), p.lastIncome, p.declined.length, ...p.declined.map(raceIndex));
  for (const id of RACE_IDS) {
    const rs = g.races[id];
    if (rs) out.push(POWER_IDS.indexOf(rs.power), rs.owner, +rs.declined, rs.hand, rs.holesPlaced, rs.pickedRound);
    else out.push(NONE, 0, 0, 0, 0, 0);
  }
  for (const r of g.regions) out.push(occupantWord(r.race), r.tokens, regionFlags(r), r.camps);
  out.push(g.market.length);
  for (const s of g.market) out.push(raceIndex(s.race), POWER_IDS.indexOf(s.power), s.coins);
  out.push(g.raceDeck.length, ...g.raceDeck.map(raceIndex), g.powerDeck.length, ...g.powerDeck.map((p) => POWER_IDS.indexOf(p)));
  // anciennes sauvegardes : rien de connu au fond des pioches
  out.push(Math.min(g.raceTail ?? 0, g.raceDeck.length), Math.min(g.powerTail ?? 0, g.powerDeck.length));
  const t = g.turn;
  out.push(
    +(t.stage === 'ghouls'), raceWord(t.army), t.conquests, t.nonEmpty, mask(t.attacked), mask(t.sorcery), +t.dragonUsed,
    t.fortressRegion ?? -1, t.berserkDie ?? -1, +t.dieRolled, +t.declined, +t.picked, t.amazonBonus,
  );
  for (let i = 0; i < g.regions.length; i++) out.push(t.readied[i] ?? 0);
  out.push(g.defense.length);
  for (const x of g.defense) out.push(x.player, raceIndex(x.race));
  out.push(g.peace.length);
  for (const x of g.peace) out.push(x.diplomat, x.target);
  if (out.some((w) => !Number.isInteger(w))) throw new Error('État de partie invalide.');
  return Uint32Array.from(out);
}

/** Une partie complète (carte et état), pour `kernel_load`. */
export function gameWords(g: GameState): Uint32Array {
  const head: number[] = [MAGIC];
  mapWords(g.map, head);
  const state = stateWords(g);
  const out = new Uint32Array(head.length + state.length);
  out.set(head);
  out.set(state, head.length);
  return out;
}

/** Relit un état écrit par le noyau ; ce qu'il ne connaît pas (noms, journal, carte) vient de `template`. */
export function readState(xs: Uint32Array, template: GameState): GameState {
  const r = new Words(xs);
  const list = <T>(item: () => T): T[] => Array.from({ length: r.u() }, item);
  const count = r.u();
  const current = r.u();
  const phase = PHASES[r.u()]!;
  const round = r.i();
  const maxRounds = r.i();
  const seq = r.u();
  const rng = r.u();
  const lastRegion = r.opt();
  let lastDie: GameState['lastDie'] = null;
  if (r.yes()) {
    const [dseq, player, value, need, success, region, berserk] = [r.u(), r.u(), r.i(), r.i(), r.i(), r.opt(), r.yes()];
    lastDie = { seq: dseq, player, value, need, success: success < 0 ? null : success === 1, region, berserk };
  }
  const players = Array.from({ length: count }, (_, i) => ({
    ...template.players[i]!,
    ai: LEVELS[r.u()]!,
    coins: r.i(),
    active: raceOrNull(r.u()),
    lastIncome: r.i(),
    declined: list(() => RACE_IDS[r.u()]!),
  }));
  const races: GameState['races'] = {};
  for (const id of RACE_IDS) {
    const [power, owner, declined, hand, holesPlaced, pickedRound] = [r.u(), r.u(), r.yes(), r.i(), r.i(), r.i()];
    if (power !== NONE) races[id] = { race: id, power: POWER_IDS[power]!, owner, declined, hand, holesPlaced, pickedRound };
  }
  const regions = template.map.regions.map(() => {
    const [occupant, tokens, flags, camps] = [r.u(), r.i(), r.u(), r.i()];
    return {
      race: occupantOf(occupant),
      tokens,
      lair: !!(flags & 1),
      fortress: !!(flags & 2),
      hero: !!(flags & 4),
      dragon: !!(flags & 8),
      hole: !!(flags & 16),
      camps,
    };
  });
  const market = list(() => ({ race: RACE_IDS[r.u()]!, power: POWER_IDS[r.u()]!, coins: r.i() }));
  const raceDeck = list(() => RACE_IDS[r.u()]!);
  const powerDeck = list(() => POWER_IDS[r.u()]!);
  const raceTail = r.u();
  const powerTail = r.u();
  const members = (m: number) => Array.from({ length: count }, (_, i) => i).filter((i) => m & (1 << i));
  const stage = r.yes() ? ('ghouls' as const) : ('active' as const);
  const army = raceOrNull(r.u());
  const [conquests, nonEmpty, attacked, sorcery, dragonUsed, fortressRegion, berserkDie, dieRolled, declined, picked, amazonBonus] = [
    r.i(), r.i(), members(r.u()), members(r.u()), r.yes(), r.opt(), r.opt(), r.yes(), r.yes(), r.yes(), r.i(),
  ];
  const readied: Record<number, number> = {};
  for (let i = 0; i < regions.length; i++) {
    const x = r.i();
    if (x) readied[i] = x;
  }
  const defense = list(() => ({ player: r.u(), race: RACE_IDS[r.u()]! }));
  const peace = list(() => ({ diplomat: r.u(), target: r.u() }));
  r.end('État');
  return {
    ...template,
    players,
    races,
    regions,
    market,
    raceDeck,
    powerDeck,
    raceTail,
    powerTail,
    round,
    maxRounds,
    current,
    phase,
    seq,
    rng,
    lastRegion,
    lastDie,
    defense,
    peace,
    turn: { stage, army, conquests, nonEmpty, attacked, sorcery, dragonUsed, fortressRegion, berserkDie, dieRolled, declined, picked, amazonBonus, readied },
  };
}

// --- actions -----------------------------------------------------------------------------------

/** Une action, dans l'ordre de `wire::decode_action` : type, deux arguments, puis les jetons d'un placement. */
export function actionWords(a: Action, regions: number): Uint32Array {
  const kind = ACTIONS.indexOf(a.type);
  if (kind < 0) throw new Error('Action inconnue.');
  let x = 0;
  let y = 0;
  const tail: number[] = [];
  switch (a.type) {
    case 'pick':
      x = a.index;
      break;
    case 'conquer': case 'reinforce': case 'dragon': case 'sorcery': case 'abandon': case 'fortress': case 'hero':
      x = a.region;
      break;
    case 'deploy': case 'camp':
      x = a.region;
      y = a.delta;
      break;
    case 'autoDeploy':
      x = +!!a.all;
      break;
    case 'place':
      for (let i = 0; i < regions; i++) tail.push(a.tokens[i] ?? 1);
      break;
    case 'endTurn':
      x = a.peace ?? -1;
      y = +!!a.stout;
      break;
  }
  const words = [kind, x, y, ...tail];
  if (words.some((w) => !Number.isInteger(w) || w < -(2 ** 31) || w >= 2 ** 31)) throw new Error('Action mal formée.');
  return Uint32Array.from(words);
}

/** Une action écrite par le noyau (décision de l'IA). Un placement ne garde que les régions du peuple. */
export function readAction(xs: Uint32Array, g: GameState): Action {
  const type = ACTIONS[xs[0]!]!;
  const x = xs[1]! | 0;
  const y = xs[2]! | 0;
  switch (type) {
    case 'pick':
      return { type, index: x };
    case 'conquer': case 'reinforce': case 'dragon': case 'sorcery': case 'abandon': case 'fortress': case 'hero':
      return { type, region: x };
    case 'deploy': case 'camp':
      return { type, region: x, delta: y };
    case 'autoDeploy':
      return x ? { type, all: true } : { type };
    case 'place': {
      const race = g.phase === 'defend' ? g.defense[0]?.race : g.turn.army;
      const tokens: Record<number, number> = {};
      g.regions.forEach((r, i) => {
        if (r.race === race) tokens[i] = xs[3 + i]! | 0;
      });
      return { type, tokens };
    }
    case 'endTurn':
      return { type, peace: x < 0 ? null : x, stout: !!y };
    default:
      return { type } as Action;
  }
}

// --- réponses ----------------------------------------------------------------------------------

/** Une raison de refus (`Reason::code` côté Rust) : code et paramètre ; 0 : aucune. */
export interface Reason {
  code: number;
  arg: number;
}

function reason(r: Words): Reason | null {
  const code = r.u();
  const arg = r.i();
  return code ? { code, arg } : null;
}

/** Un pouvoir spécial sur une région : sans objet, permis, ou refusé pour une raison. */
export type Special = { kind: 'none' } | { kind: 'allowed' } | { kind: 'denied'; reason: Reason };

function special(r: Words): Special {
  const kind = r.u();
  const why = reason(r);
  return kind === 1 ? { kind: 'allowed' } : kind === 2 ? { kind: 'denied', reason: why! } : { kind: 'none' };
}

/** Ce qui baisse le coût d'une conquête, dans l'ordre du noyau. */
export const MODIFIERS = ['commando', 'mounted', 'underworld', 'tritons', 'giants'] as const;
export type Modifier = (typeof MODIFIERS)[number];
const modifiersOf = (bits: number) => MODIFIERS.filter((_, i) => bits & (1 << i));

/** Une région vue par le peuple qui conquiert (`wire::encode_targets`). */
export interface KernelTarget {
  region: number;
  block: Reason | null;
  cost: number;
  effective: number;
  kind: 'none' | 'normal' | 'die';
  need: number;
  dragon: Special;
  sorcery: Special;
  modifiers: Modifier[];
}

export function readTargets(xs: Uint32Array): KernelTarget[] {
  const r = new Words(xs);
  const out: KernelTarget[] = [];
  while (!r.done) {
    const block = reason(r);
    const [cost, effective, kind, need] = [r.i(), r.i(), r.u(), r.i()];
    const dragon = special(r);
    const sorcery = special(r);
    out.push({ region: out.length, block, cost, effective, kind: kind === 1 ? 'normal' : kind === 2 ? 'die' : 'none', need, dragon, sorcery, modifiers: modifiersOf(r.u()) });
  }
  return out;
}

/** Un adversaire capable de prendre une région (`wire::encode_attackers`). */
export interface KernelAttacker {
  pid: number;
  race: RaceId | null;
  cost: number;
  how: 'conquer' | 'dragon' | 'sorcery';
  hint: { kind: 'modifier'; modifier: Modifier } | { kind: 'berserk' } | null;
  via: 'flying' | 'cavern' | null;
}

export function readAttackers(xs: Uint32Array): KernelAttacker[] {
  const r = new Words(xs);
  const out: KernelAttacker[] = [];
  while (!r.done) {
    const [pid, race, cost, how, hint, modifier, via] = [r.u(), raceOrNull(r.u()), r.i(), r.u(), r.u(), r.u(), r.u()];
    out.push({
      pid,
      race,
      cost,
      how: (['conquer', 'dragon', 'sorcery'] as const)[how]!,
      hint: hint === 1 ? { kind: 'modifier', modifier: modifiersOf(modifier)[0]! } : hint === 2 ? { kind: 'berserk' } : null,
      via: via === 1 ? 'flying' : via === 2 ? 'cavern' : null,
    });
  }
  return out;
}

/** Les lignes du décompte, dans l'ordre du noyau (`IncomeKind`). */
export const INCOME_KINDS = [
  'regions', 'mines', 'farms', 'magic', 'orcLoot', 'alchemist', 'forests', 'hills', 'swamps', 'merchant', 'pillage', 'fortresses', 'wealthy',
] as const;
export interface IncomeLine {
  kind: (typeof INCOME_KINDS)[number];
  race: RaceId;
  declined: boolean;
  value: number;
}

function incomeLine(r: Words): IncomeLine {
  return { kind: INCOME_KINDS[r.u()]!, race: RACE_IDS[r.u()]!, declined: r.yes(), value: r.i() };
}

export function readIncome(xs: Uint32Array): IncomeLine[] {
  const r = new Words(xs);
  const out: IncomeLine[] = [];
  while (!r.done) out.push(incomeLine(r));
  return out;
}

/** Un événement du noyau (`wire::encode_events`), avec son numéro et la manche en cours. */
export type KernelEvent = { seq: number; round: number } & (
  | { type: 'started' }
  | { type: 'turn'; player: number; race: RaceId; power: PowerId; hand: number; ghouls: boolean; amazonBonus: number }
  | { type: 'pick'; player: number; race: RaceId; power: PowerId; tokens: number; paid: number; gained: number }
  | {
      type: 'conquer';
      player: number;
      race: RaceId;
      region: number;
      how: 'conquer' | 'die' | 'dragon' | 'sorcery';
      tokens: number;
      victim: number | null;
      victimRace: RaceId | 'lost' | null;
      victimActive: boolean;
      lost: number;
      returned: number;
    }
  | { type: 'die'; player: number; region: number | null; value: number; need: number; success: boolean | null }
  | { type: 'abandon'; player: number; region: number }
  | { type: 'decline'; player: number; race: RaceId; power: PowerId; removed: RaceId[] }
  | { type: 'score'; player: number; gain: number; lines: IncomeLine[] }
  | { type: 'peace'; player: number; target: number; race: RaceId | null }
  | { type: 'amazons'; player: number; tokens: number }
  | { type: 'skeletons'; player: number; tokens: number }
  | { type: 'ghoulsDone'; player: number }
  | { type: 'defended'; player: number; race: RaceId }
  | { type: 'over' }
);

export function readEvents(xs: Uint32Array): KernelEvent[] {
  const out: KernelEvent[] = [];
  let at = 0;
  while (at < xs.length) {
    const len = xs[at]!;
    const r = new Words(xs.subarray(at + 1, at + len));
    const head = { seq: r.u(), round: r.i() };
    const kind = r.u();
    const race = () => RACE_IDS[r.u()]!;
    const power = () => POWER_IDS[r.u()]!;
    let e: KernelEvent;
    switch (kind) {
      case 0:
        e = { ...head, type: 'turn', player: r.u(), race: race(), power: power(), hand: r.i(), ghouls: r.yes(), amazonBonus: r.i() };
        break;
      case 1:
        e = { ...head, type: 'pick', player: r.u(), race: race(), power: power(), tokens: r.i(), paid: r.i(), gained: r.i() };
        break;
      case 2: {
        const [player, by, region, how, tokens, victim, occupant, victimActive, lost, returned] = [
          r.u(), race(), r.u(), r.u(), r.i(), r.opt(), r.u(), r.yes(), r.i(), r.i(),
        ];
        const hows = ['conquer', 'die', 'dragon', 'sorcery'] as const;
        e = { ...head, type: 'conquer', player, race: by, region, how: hows[how]!, tokens, victim, victimRace: occupantOf(occupant), victimActive, lost, returned };
        break;
      }
      case 3: {
        const [player, region, value, need, success] = [r.u(), r.opt(), r.i(), r.i(), r.i()];
        e = { ...head, type: 'die', player, region, value, need, success: success < 0 ? null : success === 1 };
        break;
      }
      case 4:
        e = { ...head, type: 'abandon', player: r.u(), region: r.u() };
        break;
      case 5:
        e = { ...head, type: 'decline', player: r.u(), race: race(), power: power(), removed: Array.from({ length: r.u() }, race) };
        break;
      case 6:
        e = { ...head, type: 'score', player: r.u(), gain: r.i(), lines: Array.from({ length: r.u() }, () => incomeLine(r)) };
        break;
      case 7:
        e = { ...head, type: 'peace', player: r.u(), target: r.u(), race: raceOrNull(r.u()) };
        break;
      case 8:
        e = { ...head, type: 'amazons', player: r.u(), tokens: r.i() };
        break;
      case 9:
        e = { ...head, type: 'skeletons', player: r.u(), tokens: r.i() };
        break;
      case 10:
        e = { ...head, type: 'ghoulsDone', player: r.u() };
        break;
      case 11:
        e = { ...head, type: 'defended', player: r.u(), race: race() };
        break;
      case 12:
        e = { ...head, type: 'over' };
        break;
      case 13:
        e = { ...head, type: 'started' };
        break;
      default:
        throw new Error(`Événement inconnu du noyau : ${kind}`);
    }
    r.end('Événement');
    out.push(e);
    at += len;
  }
  return out;
}

// --- le noyau ----------------------------------------------------------------------------------

/** Une action refusée par les règles. */
export class KernelRefusal extends Error {
  constructor(readonly reason: Reason) {
    super(`Action refusée (raison ${reason.code})`);
  }
}

/** Les fonctions exportées par le noyau (kernel/src/abi.rs). */
export interface KernelExports {
  kernel_version(): number;
  kernel_capacity(): number;
  kernel_load(len: number): number;
  kernel_set_state(len: number): number;
  kernel_state(): number;
  kernel_apply(len: number): number;
  kernel_decide(flags: number): number;
  kernel_targets(): number;
  kernel_generate_map(players: number, seedLow: number, seedHigh: number): number;
  kernel_new_game(len: number): number;
  kernel_attackers(region: number, owner: number): number;
  kernel_income(player: number): number;
  kernel_permissions(): number;
  kernel_threat(race: number, region: number): number;
  kernel_full_deployment(race: number): number;
  kernel_decline_preview(): number;
}

/** Un noyau chargé : ses fonctions, et une vue sur son tampon d'échange (à redemander après chaque appel). */
export interface KernelBackend {
  exports: KernelExports;
  buffer(): Uint32Array;
}

/** Le noyau WebAssembly (`game-kernel.wasm`). */
export async function wasmBackend(wasm: BufferSource): Promise<KernelBackend> {
  const instance = await WebAssembly.instantiate(await WebAssembly.compile(wasm), {});
  const exports = instance.exports as unknown as KernelExports & { memory: WebAssembly.Memory; kernel_buffer(): number };
  const pointer = exports.kernel_buffer();
  const capacity = exports.kernel_capacity();
  // la mémoire peut grandir pendant un appel : la vue se refait à chaque fois
  return { exports, buffer: () => new Uint32Array(exports.memory.buffer, pointer, capacity) };
}

/**
 * Le noyau et la partie qu'il a en mémoire. Chaque question commence par lui donner la partie
 * (`use`) : la carte n'est rechargée que si elle change.
 */
export class Kernel {
  private map: MapDef | null = null;
  private template: GameState | null = null;

  constructor(private readonly backend: KernelBackend) {
    const version = backend.exports.kernel_version();
    if (version !== KERNEL_ABI) throw new Error(`Noyau incompatible (protocole ${version}, attendu ${KERNEL_ABI}).`);
  }

  private get x() {
    return this.backend.exports;
  }

  private send(words: Uint32Array): number {
    const buffer = this.backend.buffer();
    if (words.length > buffer.length) throw new Error('Demande trop grande pour le noyau.');
    buffer.set(words);
    return words.length;
  }

  private check(result: number): number {
    if (result >= 0) return result;
    if (result === -3) {
      const b = this.backend.buffer();
      throw new KernelRefusal({ code: b[0]!, arg: b[1]! | 0 });
    }
    if (result === -2) throw new Error('Aucune partie chargée dans le noyau.');
    const b = this.backend.buffer();
    const bytes = new Uint8Array(b.buffer, b.byteOffset + 4, Math.min(b[0]!, 256));
    throw new Error(`Demande mal formée pour le noyau : ${new TextDecoder().decode(bytes)}.`);
  }

  private reply(result: number): Uint32Array {
    return this.backend.buffer().slice(0, this.check(result));
  }

  /** Donne la partie au noyau. */
  use(g: GameState) {
    if (g.map === this.map) this.check(this.x.kernel_set_state(this.send(stateWords(g))));
    else {
      this.check(this.x.kernel_load(this.send(gameWords(g))));
      this.map = g.map;
    }
    this.template = g;
  }

  /** L'état de la partie en mémoire, relu (sans journal : il se reconstruit à partir des événements). */
  snapshot(): GameState {
    return readState(this.reply(this.x.kernel_state()), this.template!);
  }

  stateWords(): Uint32Array {
    return this.reply(this.x.kernel_state());
  }

  /** Joue une action ; renvoie les événements produits. */
  apply(a: Action): KernelEvent[] {
    return readEvents(this.reply(this.x.kernel_apply(this.send(actionWords(a, this.template!.regions.length)))));
  }

  /** Les régions de la carte générée pour `players` joueurs et cette graine. */
  generateRegions(players: number, seed: number): { rounds: number; regions: RegionDef[] } {
    if (!Number.isSafeInteger(seed) || seed < 0) throw new Error('Graine de carte invalide.');
    return readGeneratedRegions(this.reply(this.x.kernel_generate_map(players, seed % 2 ** 32, Math.floor(seed / 2 ** 32))));
  }

  /** Crée et charge une partie sur la carte de `template` (générée pour sa graine) ; renvoie les événements du début. */
  newGame(template: GameState, seed: number): KernelEvent[] {
    const { map, players } = template;
    const words = [players.length, ...players.map((p) => LEVELS.indexOf(p.ai)), map.seed % 2 ** 32, Math.floor(map.seed / 2 ** 32), seed >>> 0];
    const events = readEvents(this.reply(this.x.kernel_new_game(this.send(Uint32Array.from(words)))));
    this.map = map;
    this.template = template;
    return events;
  }

  /** La décision de l'IA pour le joueur qui doit agir. */
  decide(noDecline = false, keepPlacement = false): Action {
    return readAction(this.reply(this.x.kernel_decide(+noDecline | (+keepPlacement << 1))), this.template!);
  }

  targets(): KernelTarget[] {
    return readTargets(this.reply(this.x.kernel_targets()));
  }

  attackers(region: number, owner: number): KernelAttacker[] {
    return readAttackers(this.reply(this.x.kernel_attackers(region, owner)));
  }

  income(player: number): IncomeLine[] {
    return readIncome(this.reply(this.x.kernel_income(player)));
  }

  /** Raison de refuser le déclin, puis l'abandon de chaque région (`null` : permis). */
  permissions(): { decline: Reason | null; abandon: (Reason | null)[] } {
    const r = new Words(this.reply(this.x.kernel_permissions()));
    const decline = reason(r);
    const abandon: (Reason | null)[] = [];
    while (!r.done) abandon.push(reason(r));
    return { decline, abandon };
  }

  threat(race: RaceId, region: number): number {
    return new Words(this.reply(this.x.kernel_threat(raceIndex(race), region))).f64();
  }

  /** Jetons visés par région si l'on replace tout (0 hors des régions du peuple). */
  fullDeployment(race: RaceId): number[] {
    return Array.from(this.reply(this.x.kernel_full_deployment(raceIndex(race))), (w) => w | 0);
  }

  /** L'état juste après un passage en déclin du joueur en cours. */
  declinePreview(): GameState {
    return readState(this.reply(this.x.kernel_decline_preview()), this.template!);
  }
}
