import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { aiAction } from '../src/engine/ai';
import { PLAYER_COLORS } from '../src/engine/data';
import { applyAction, createGame } from '../src/engine/game';
import { PRESETS } from '../src/engine/mapgen';
import { analyzeTarget, cloneState, currentActor, regionsOf } from '../src/engine/rules';
import type { GameState, LogEntry, PowerId, RaceId } from '../src/engine/types';
import { RoomManager, type Client, type Scheduler } from '../src/server/rooms';
import { FACTOR, frameDuration, isShown, playbackMs, splitFrames } from '../src/shared/pacing';
import type { ClientMsg, GameWire, RoomView, ServerMsg } from '../src/shared/protocol';
import { narrate } from '../src/ui/narrate';

// --- Un navigateur minimal, juste ce qu'il faut au store --------------------------

class MemStorage {
  private m = new Map<string, string>();
  getItem(k: string) {
    return this.m.has(k) ? this.m.get(k)! : null;
  }
  setItem(k: string, v: string) {
    this.m.set(k, String(v));
  }
  removeItem(k: string) {
    this.m.delete(k);
  }
  clear() {
    this.m.clear();
  }
}

const fakeDocument = {
  visibilityState: 'visible',
  title: 'SmallMobile',
  addEventListener() {},
  removeEventListener() {},
};

/** WebSocket factice : branchée directement sur un RoomManager (messages livrés de façon asynchrone, comme un vrai réseau). */
class FakeSocket {
  static OPEN = 1;
  static CONNECTING = 0;
  static CLOSED = 3;
  static rooms: RoomManager | null = null;
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  client: Client;
  constructor(public url: string) {
    this.client = {
      code: null,
      seat: null,
      send: (m: ServerMsg) => queueMicrotask(() => this.readyState === 1 && this.onmessage?.({ data: JSON.stringify(m) })),
    };
    queueMicrotask(() => {
      this.readyState = 1;
      this.onopen?.();
    });
  }
  send(raw: string) {
    FakeSocket.rooms?.handle(this.client, JSON.parse(raw) as ClientMsg);
  }
  close() {
    if (this.readyState === 3) return;
    this.readyState = 3;
    FakeSocket.rooms?.disconnect(this.client);
    queueMicrotask(() => this.onclose?.());
  }
}

const G = globalThis as Record<string, unknown>;
const saved: Record<string, unknown> = {};
const FAKES: Record<string, unknown> = {
  window: { addEventListener() {}, history: { replaceState() {} } },
  document: fakeDocument,
  localStorage: new MemStorage(),
  sessionStorage: new MemStorage(),
  location: { protocol: 'http:', host: 'test', search: '', pathname: '/' },
  WebSocket: FakeSocket,
};

type Store = typeof import('../src/ui/store');
let store: Store;

beforeAll(async () => {
  for (const [k, v] of Object.entries(FAKES)) {
    saved[k] = G[k];
    G[k] = v;
  }
  store = await import('../src/ui/store');
});

afterAll(() => {
  store.onlineQuit(false);
  store.quitGame();
  for (const k of Object.keys(FAKES)) G[k] = saved[k];
});

const flush = async (n = 4) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

// --- Parties fabriquées -------------------------------------------------------------

const ALICE = PLAYER_COLORS[0]!.color;
const BOB = PLAYER_COLORS[1]!.color;

function newGame(seed = 42): GameState {
  return createGame({
    players: [
      { name: 'Alice', color: ALICE, kind: 'human', ai: 'normal' },
      { name: 'Bob', color: BOB, kind: 'human', ai: 'normal' },
    ],
    mapSeed: PRESETS[2]!.seed,
    seed,
  });
}

/** Met en tête du marché la combinaison voulue. */
function offer(g0: GameState, race: RaceId, power: PowerId): GameState {
  const g = cloneState(g0);
  g.market = [{ race, power, coins: 0 }, ...g.market.filter((m) => m.race !== race && m.power !== power).slice(0, 5)];
  g.raceDeck = g.raceDeck.filter((r) => r !== race);
  g.powerDeck = g.powerDeck.filter((p) => p !== power);
  return g;
}

/** Alice prend un peuple et passe son tour sans conquérir : c'est à Bob. */
function aliceSkips(g0 = newGame(), alice: [RaceId, PowerId] = ['tritons', 'alchemist']): GameState {
  let g = applyAction(offer(g0, ...alice), { type: 'pick', index: 0 });
  g = applyAction(g, { type: 'endConquest' });
  g = applyAction(g, { type: 'endTurn' });
  expect(g.current).toBe(1);
  return g;
}

function target(g: GameState, kind: 'normal' | 'die' = 'normal', not: number[] = []): number {
  return g.map.regions.map((r) => analyzeTarget(g, r.id)).find((t) => t.kind === kind && !not.includes(t.region))!.region;
}

function wire(g: GameState): GameWire {
  const { map: _m, ...rest } = g;
  return JSON.parse(JSON.stringify({ ...rest, rng: 0 })) as GameWire;
}

function roomView(g: GameState): RoomView {
  return {
    code: 'ABCD',
    size: g.players.length,
    mapSeed: g.map.seed,
    aiLevel: 'normal',
    speed: 'normal',
    speedBy: null,
    started: true,
    seats: g.players.map((p) => ({ name: p.name, color: p.color, kind: p.kind, ai: p.ai, taken: true, online: true, replaced: false })),
  };
}

/** Le téléphone d'Alice (siège 0) affiche `g0` ; le test joue ensuite le serveur. */
async function phoneShowing(g0: GameState) {
  store.onlineQuit(false);
  FakeSocket.rooms = null;
  store.onlineJoin('ABCD', 'Alice', ALICE);
  await flush();
  store.onMessage({ t: 'joined', code: 'ABCD', token: 'tok', seat: 0 });
  store.onMessage({ t: 'room', room: roomView(g0) });
  store.onMessage({ t: 'map', map: g0.map });
  store.onMessage({ t: 'state', game: wire(g0), ack: null, undo: 0 });
  expect(store.getState().game?.seq).toBe(g0.seq);
}

function serverSends(g: GameState) {
  store.onMessage({ t: 'state', game: wire(g), ack: null, undo: 0 });
}

/** Lit les trames une à une (comme si l'on touchait la carte d'événement à chaque fois). */
function readFrames(limit = 60) {
  const out: NonNullable<ReturnType<Store['getState']>['playback']>[] = [];
  for (let k = 0; k < limit && store.getState().playback; k++) {
    expect(store.canAct()).toBe(false);
    out.push(store.getState().playback!);
    store.nextFrame();
  }
  return out;
}

// --- Rythme et découpage -----------------------------------------------------------

function entry(kind: LogEntry['kind'], extra: Partial<LogEntry> = {}): LogEntry {
  return { seq: 1, round: 1, player: 1, kind, text: kind, ...extra };
}

describe('découpage en trames et durées', () => {
  test('une entrée = une trame, sauf dé + conquête', () => {
    expect(splitFrames([entry('score'), entry('decline'), entry('power', { victim: 0 })]).length).toBe(3);
    expect(splitFrames([entry('die', { region: 4 }), entry('conquer', { region: 4 })]).length).toBe(1);
    expect(splitFrames([entry('die', { region: 4 }), entry('conquer', { region: 5 })]).length).toBe(2);
    expect(splitFrames([entry('power', { quiet: true }), entry('defense'), entry('info')]).length).toBe(0);
  });

  test('durées de contrôle', () => {
    expect(frameDuration([entry('pick')], 'normal')).toBe(2800);
    expect(frameDuration([entry('conquer', { victim: 0 })], 'normal')).toBe(2000);
    expect(frameDuration([entry('conquer')], 'normal')).toBe(1300);
    expect(frameDuration([entry('die', { region: 3 }), entry('conquer', { region: 3, victim: 0 })], 'normal')).toBe(2400);
    expect(frameDuration([entry('die', { region: 3 })], 'normal')).toBe(2000);
    expect(frameDuration([entry('die')], 'normal')).toBe(1000);
    expect(frameDuration([entry('power', { victim: 0 })], 'normal')).toBe(1700);
    expect(frameDuration([entry('score')], 'normal')).toBe(1200);
    expect(frameDuration([entry('pick')], 'rapide')).toBeCloseTo(2800 * 0.45);
    expect(frameDuration([entry('power', { victim: 0 })], 'rapide')).toBeCloseTo(1700 * FACTOR.rapide);
    const events = [entry('pick'), entry('die', { region: 3 }), entry('conquer', { region: 3 }), entry('score'), entry('defense')];
    expect(playbackMs(events, 'lent')).toBeCloseTo((2800 + 2400 + 1200) * 1.5);
  });
});

describe('récit des trames', () => {
  test('une région prise : 2e personne, en rouge, avec ce que je perds', () => {
    const g = newGame();
    const name = g.map.regions[3]!.name;
    const hit = entry('conquer', { player: 1, region: 3, victim: 0, how: 'conquer', tokens: 3, lost: 1, returned: 2, victimRace: 'tritons' });
    const n = narrate(g, [hit], 0);
    expect(n.main).toContain('prend votre');
    expect(n.main).toContain(name.slice(0, 10));
    expect(n.tone).toBe('hit');
    expect(n.detail).toBe('Vous perdez 1 jeton Tritons · 2 reviennent en main');
    expect(n.loss).toBe(1);
    // vu par un tiers : neutre, sans conséquence
    const other = narrate(g, [hit], null);
    expect(other.tone).toBe('neutral');
    expect(other.main).toContain('à Alice');
  });

  test('Elfes, peuple en déclin, sorcellerie', () => {
    const g = newGame();
    const elves = narrate(g, [entry('conquer', { player: 1, region: 3, victim: 0, how: 'conquer', tokens: 3, lost: 0, returned: 3, victimRace: 'elves' })], 0);
    expect(elves.detail).toBe('🧝 Vous ne perdez rien · 3 reviennent en main');
    const sorc = narrate(g, [entry('conquer', { player: 1, region: 3, victim: 0, how: 'sorcery', tokens: 1, lost: 1, returned: 0, victimRace: 'tritons' })], 0);
    expect(sorc.main.startsWith('🔮 ensorcelle votre')).toBe(true);
    expect(sorc.detail).toBe('Votre jeton Tritons devient un Sorcier');
  });

  test('anciennes sauvegardes : repli sur le texte court', () => {
    const n = narrate(newGame(), [entry('conquer', { short: 'prend « Bois »', detail: 'Alice perd 1 jeton' })], 0);
    expect(n.main).toBe('⚔️ prend « Bois »');
  });
});

// --- File de lecture -----------------------------------------------------------------

describe('file de lecture du tour adverse', () => {
  test('5 états reçus d\'un coup : 5 trames distinctes, dans l\'ordre', async () => {
    const g0 = aliceSkips();
    await phoneShowing(g0);
    const s1 = applyAction(g0, { type: 'pick', index: 0 });
    const army = s1.turn.army!;
    const a = target(s1);
    const s2 = applyAction(s1, { type: 'conquer', region: a });
    const s3 = applyAction(s2, { type: 'conquer', region: target(s2) });
    const low = cloneState(s3);
    low.races[army]!.hand = 1;
    const s4 = applyAction(low, { type: 'reinforce', region: target(low, 'die') });
    const done = cloneState(s4);
    done.races[army]!.hand = 0;
    const s5 = applyAction(done, { type: 'endTurn' });
    for (const s of [s1, s2, s3, s4, s5]) serverSends(s);
    const shown = readFrames();
    expect(shown.length).toBe(5);
    expect(shown.map((p) => p.events[0]!.kind)).toEqual(['pick', 'conquer', 'conquer', 'die', 'score']);
    const seqs = shown.map((p) => p.events[0]!.seq);
    expect([...seqs].sort((x, y) => x - y)).toEqual(seqs);
    for (const p of shown) expect(p.events.length).toBe(p.events[0]!.kind === 'die' && p.events.length === 2 ? 2 : 1);
    // le dé adverse : pas de plein écran, « il faut N » d'abord
    const die = shown[3]!;
    expect(die.story!.rolling).toContain('il faut');
    expect(store.getState().overlay).toBeNull();
    // Alice reprend la main, avec le flash
    expect(store.canAct()).toBe(true);
    expect(store.getState().flash?.text).toContain('À vous');
  });

  test('Bob m\'impose la paix : une trame rouge distincte du décompte', async () => {
    const g0 = offer(aliceSkips(), 'humans', 'diplomat');
    await phoneShowing(g0);
    const s1 = applyAction(g0, { type: 'pick', index: 0 });
    const s2 = applyAction(s1, { type: 'endConquest' });
    const s3 = applyAction(s2, { type: 'endTurn', peace: 0 });
    for (const s of [s1, s2, s3]) serverSends(s);
    const shown = readFrames();
    const peace = shown.find((p) => p.story?.tone === 'hit');
    expect(peace?.story?.main).toBe('🕊️ vous impose la paix');
    expect(peace?.story?.detail).toBe('Vous ne pourrez pas attaquer ses Humains à votre prochain tour');
    const score = shown.find((p) => p.events[0]?.kind === 'score');
    expect(score).toBeDefined();
    expect(score).not.toBe(peace);
  });

  /** Bob a conquis 2 régions et redéploie ; `near` : Alice tient une région voisine de la 1re. */
  function redeploying(near: boolean) {
    let g = applyAction(offer(aliceSkips(), 'ratmen', 'commando'), { type: 'pick', index: 0 });
    const r1 = target(g);
    g = applyAction(g, { type: 'conquer', region: r1 });
    const r2 = target(g, 'normal', [r1]);
    g = applyAction(g, { type: 'conquer', region: r2 });
    g = applyAction(g, { type: 'endConquest' });
    if (near) {
      g = cloneState(g);
      const n = g.map.regions[r1]!.adj.find((x) => g.regions[x]!.race !== 'ratmen' && g.map.regions[x]!.terrain !== 'sea' && g.map.regions[x]!.terrain !== 'lake')!;
      Object.assign(g.regions[n]!, { race: 'tritons', tokens: 1, lair: false, fortress: false, camps: 0, hero: false, dragon: false, hole: false });
    }
    return { g, r1, r2 };
  }

  for (const near of [false, true]) {
    test(`12 ajustements de redéploiement puis le décompte : 2 trames (${near ? 'près de moi' : 'loin de moi'})`, async () => {
      const { g: g0, r1, r2 } = redeploying(near);
      await phoneShowing(g0);
      const army = g0.turn.army!;
      expect(g0.races[army]!.hand).toBeGreaterThan(2);
      const states: GameState[] = [];
      let g = g0;
      const moves = [r1, r1, r2, r1, r2, r1, r1, r2, r1, r2, r1];
      moves.forEach((r, i) => {
        const delta = i % 4 === 3 ? -1 : 1;
        g = applyAction(g, { type: 'deploy', region: delta < 0 ? r1 : r, delta });
        states.push(g);
      });
      g = applyAction(g, { type: 'autoDeploy' });
      states.push(g);
      expect(states.length).toBe(12);
      for (const s of states) serverSends(s);
      serverSends(applyAction(g, { type: 'endTurn' }));
      await wait(350); // le miroir du groupe se met à jour au plus toutes les 300 ms
      const group = store.getState().playback!;
      expect(group.ms).toBeGreaterThanOrEqual(1200 * FACTOR.normal);
      if (near) {
        expect(group.kind).toBe('group');
        expect(group.story!.main).toBe('🔁 renforce près de vous');
        expect(group.story!.detail).toContain('+');
      } else {
        expect(group.kind).toBe('wait');
        expect(group.story?.detail).toBeUndefined();
      }
      const shown = readFrames();
      expect(shown.length).toBe(2);
      expect(shown[1]!.events[0]!.kind).toBe('score');
    });
  }

  test("annulation d'une conquête déjà montrée : trame « annule »", async () => {
    const g0 = applyAction(aliceSkips(), { type: 'pick', index: 0 });
    await phoneShowing(g0);
    serverSends(applyAction(g0, { type: 'conquer', region: target(g0) }));
    expect(store.getState().playback!.events[0]!.kind).toBe('conquer');
    serverSends(g0);
    store.nextFrame();
    const pb = store.getState().playback!;
    expect(pb.kind).toBe('undo');
    expect(pb.story!.main).toBe('↶ annule sa dernière action');
    expect(store.getState().game!.seq).toBe(g0.seq);
    store.skipPlayback();
  });

  test('annulation avec 2 trames en file : comparée au dernier état reçu', async () => {
    const g0 = applyAction(offer(aliceSkips(), 'ratmen', 'commando'), { type: 'pick', index: 0 });
    await phoneShowing(g0);
    const s1 = applyAction(g0, { type: 'conquer', region: target(g0) });
    const s2 = applyAction(s1, { type: 'conquer', region: target(s1) });
    const s3 = applyAction(s2, { type: 'conquer', region: target(s2) });
    for (const s of [s1, s2, s3]) serverSends(s);
    expect(store.playbackQueue().length).toBe(2);
    serverSends(s2); // Bob annule la 3e conquête
    const q = store.playbackQueue();
    expect(q.some((f) => f.seq === s3.seq)).toBe(false);
    expect(q.some((f) => f.undo)).toBe(false);
    expect(q.filter((f) => f.events.length).map((f) => f.seq)).toEqual([s2.seq]);
    const shown = readFrames();
    expect(shown.map((p) => p.kind)).toEqual(['event', 'event']);
    expect(store.getState().game!.seq).toBe(s2.seq);
  });

  test('en direct, un retard se rattrape, mais jamais sur ce qui me touche', async () => {
    const g0 = applyAction(offer(aliceSkips(), 'ratmen', 'commando'), { type: 'pick', index: 0 });
    await phoneShowing(g0);
    let g = g0;
    for (let k = 0; k < 7; k++) {
      g = applyAction(g, { type: 'conquer', region: target(g) });
      if (k === 1) {
        // la 2e conquête me prend une région
        g = cloneState(g);
        Object.assign(g.log[g.log.length - 1]!, { victim: 0, victimRace: 'tritons', lost: 1, returned: 0 });
      }
      serverSends(g);
    }
    store.nextFrame(); // 2e trame, 5 en attente, mais elle me touche
    const hit = store.getState().playback!;
    expect(hit.story!.tone).toBe('hit');
    expect(hit.ms).toBe(frameDuration(hit.events, 'normal'));
    store.nextFrame(); // 3e trame : 4 en attente, on rattrape
    const pb = store.getState().playback!;
    expect(pb.ms).toBeCloseTo(Math.max(700, frameDuration(pb.events, 'normal') * 0.7));
    store.skipPlayback();
    expect(store.getState().playback).toBeNull();
    expect(store.getState().game!.seq).toBe(g.seq);
  });
});

// --- Avec le vrai serveur -------------------------------------------------------------

class ManualScheduler implements Scheduler {
  tasks: (() => void)[] = [];
  set(fn: () => void) {
    this.tasks.push(fn);
    return fn;
  }
  clear(h: unknown) {
    this.tasks = this.tasks.filter((t) => t !== h);
  }
  run(limit = 100000) {
    let n = 0;
    while (this.tasks.length && n++ < limit) this.tasks.shift()!();
  }
}

class Phone implements Client {
  code: string | null = null;
  seat: number | null = null;
  inbox: ServerMsg[] = [];
  map: GameState['map'] | null = null;
  send(msg: ServerMsg) {
    this.inbox.push(msg);
    if (msg.t === 'map') this.map = msg.map;
  }
  last<T extends ServerMsg['t']>(t: T) {
    return [...this.inbox].reverse().find((m) => m.t === t) as Extract<ServerMsg, { t: T }> | undefined;
  }
  game(): GameState {
    return { ...this.last('state')!.game, map: this.map! } as GameState;
  }
}

describe('avec le serveur', () => {
  test('reconnexion après une longue absence : une trame par entrée, sur les états de chaque moment', async () => {
    store.onlineQuit(false);
    const sched = new ManualScheduler();
    const rooms = new RoomManager(null, sched);
    FakeSocket.rooms = rooms;
    const host = new Phone();
    const guest = new Phone();
    rooms.handle(host, { t: 'create', name: 'Alice', color: ALICE, size: 4, aiLevel: 'normal', mapSeed: PRESETS[4]!.seed });
    const { code, token } = host.last('joined')!;
    rooms.handle(guest, { t: 'join', code, name: 'Bob', color: BOB });
    rooms.handle(host, { t: 'start' });
    let id = 0;
    while (currentActor(host.game()) === 0) rooms.handle(host, { t: 'action', id: ++id, action: aiAction(host.game()) });
    const s = host.game().seq;
    rooms.disconnect(host);
    rooms.handle(guest, { t: 'replace', seat: 0 });
    for (let k = 0; k < 30; k++) {
      const g = guest.game();
      if (currentActor(g) === 1) rooms.handle(guest, { t: 'action', id: ++id, action: aiAction(g) });
      else sched.run(1);
    }
    // Alice rouvre la page : sa session connaît la dernière trame affichée
    localStorage.setItem('smallmobile:online:v1', JSON.stringify({ code, token, seat: 0, shown: s }));
    store.onlineResume();
    await flush(10);
    const st = store.getState();
    expect(st.online?.seat).toBe(0);
    const first = st.playback!;
    expect(first.kind).toBe('header');
    expect(first.header).toContain('pendant votre absence');
    const replay = host.inbox.length; // l'ancien téléphone d'Alice ne reçoit plus rien
    expect(replay).toBeGreaterThan(0);
    // ce que les autres ont fait depuis la trame vue, une trame par entrée (dé + conquête : une seule)
    const final = guest.game();
    const others = final.log.filter((l) => l.seq > s && l.player !== 0 && isShown(l));
    const expected = splitFrames(others).length;
    store.nextFrame();
    let frames = 0;
    for (let k = 0; k < 200 && store.getState().playback; k++) {
      const pb = store.getState().playback!;
      if (pb.kind === 'event') {
        frames++;
        const e = pb.events[pb.events.length - 1]!;
        if (e.kind === 'conquer' && !pb.rolling) {
          // la carte montrée est celle de ce moment-là, pas l'état final
          const g = store.getState().game!;
          const race = g.regions[e.region!]!.race;
          expect(race && race !== 'lost' ? g.races[race]!.owner : null).toBe(e.player);
          expect(g.seq).toBeLessThanOrEqual(final.seq);
        }
      }
      store.nextFrame();
    }
    expect(frames).toBe(expected);
    expect(frames).toBeGreaterThan(12);
    expect(store.getState().game!.seq).toBe(final.seq);
    void regionsOf;
  });

  test('vitesse changée par un autre siège : un toast, pas pour son auteur', async () => {
    store.onlineQuit(false);
    const rooms = new RoomManager(null, new ManualScheduler());
    FakeSocket.rooms = rooms;
    store.onlineCreate({ name: 'Alice', color: ALICE, size: 2, aiLevel: 'normal', mapSeed: PRESETS[2]!.seed });
    await flush(10);
    const code = store.getState().online!.code!;
    const bob = new Phone();
    rooms.handle(bob, { t: 'join', code, name: 'Bob', color: BOB });
    await flush(10);
    rooms.handle(bob, { t: 'speed', speed: 'rapide' });
    await flush(10);
    expect(bob.last('room')!.room.speedBy).toBe(1);
    expect(store.getState().toast?.text).toBe('⚙️ Bob : vitesse ⚡ Rapide');
    const before = store.getState().toast!.id;
    store.onlineSend({ t: 'speed', speed: 'lent' });
    await flush(10);
    expect(store.getState().online!.room!.speed).toBe('lent');
    expect(store.getState().toast!.id).toBe(before);
  });
});

describe('actions en vol et reconnexions', () => {
  test('une action perdue avec la connexion n\'est pas rejouée sur la resynchronisation', async () => {
    const g0 = newGame();
    await phoneShowing(g0);
    expect(store.act({ type: 'pick', index: 0 })).toBe(true);
    expect(store.getState().online!.pending.length).toBe(1);
    // le serveur n'a jamais reçu le choix : la resynchronisation fait foi, Alice peut rejouer
    store.onMessage({ t: 'state', game: wire(g0), ack: null, undo: 0, sync: true });
    expect(store.getState().online!.pending.length).toBe(0);
    expect(store.getState().game!.seq).toBe(g0.seq);
    expect(store.canAct()).toBe(true);
  });

  test('un accusé de réception perdu ne fait pas appliquer deux fois la même action', async () => {
    const g0 = newGame();
    await phoneShowing(g0);
    store.act({ type: 'pick', index: 0 });
    const g1 = applyAction(g0, { type: 'pick', index: 0 });
    store.onMessage({ t: 'state', game: wire(g1), ack: null, undo: 1, sync: true });
    const g = store.getState().game!;
    expect(g.seq).toBe(g1.seq);
    expect(g.players[0]!.coins).toBe(g1.players[0]!.coins);
    expect(g.races[g1.players[0]!.active!]!.hand).toBe(g1.races[g1.players[0]!.active!]!.hand);
  });

  test('↶ en attente de réponse : pas de 2e envoi, la main revient avec l\'état suivant', async () => {
    const g1 = applyAction(newGame(), { type: 'pick', index: 0 });
    await phoneShowing(newGame());
    store.onMessage({ t: 'state', game: wire(g1), ack: null, undo: 1, sync: true });
    expect(store.canUndo()).toBe(true);
    store.undoLast();
    expect(store.getState().online!.inflight).toBe(true);
    expect(store.canAct()).toBe(false);
    expect(store.canUndo()).toBe(false);
    serverSends(newGame());
    expect(store.getState().online!.inflight).toBe(false);
  });
});

describe('passe-et-joue', () => {
  test('la relecture repart de la carte telle que le joueur l\'a laissée', () => {
    store.onlineQuit(false);
    store.startGame({
      players: [
        { name: 'Alice', color: ALICE, kind: 'human', ai: 'normal' },
        { name: 'Bob', color: BOB, kind: 'human', ai: 'normal' },
      ],
      mapSeed: PRESETS[2]!.seed,
      seed: 3,
    });
    store.dismissOverlay();
    store.act({ type: 'pick', index: 0 });
    store.act({ type: 'endConquest' });
    store.act({ type: 'endTurn' });
    const g = store.getState().game!;
    const lastAlice = [...g.log].reverse().find((l) => l.player === 0)!;
    expect(store.getState().seen[0]).toBe(lastAlice.seq);
    store.quitGame();
  });
});
