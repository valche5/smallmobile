import { describe, expect, test } from 'bun:test';
import { basename } from 'node:path';
import { aiAction } from '../src/engine/ai';
import { PLAYER_COLORS } from '../src/engine/data';
import { applyAction, createGame } from '../src/engine/game';
import { stateWords } from '../src/engine/kernel';
import { currentActor } from '../src/engine/rules';
import type { Action, GameState, MapDef } from '../src/engine/types';
import type { ArchivedGame, GameArchive } from '../src/server/archive';
import { DEFAULT_EXPERT, ExpertProcess, type Expert, type ExpertMove } from '../src/server/expert';
import { RoomManager, type Client, type Scheduler } from '../src/server/rooms';
import type { ServerMsg } from '../src/shared/protocol';

class Phone implements Client {
  code: string | null = null;
  seat: number | null = null;
  inbox: ServerMsg[] = [];
  map: MapDef | null = null;
  send(msg: ServerMsg) {
    this.inbox.push(msg);
    if (msg.t === 'map') this.map = msg.map;
  }
  last<T extends ServerMsg['t']>(t: T): Extract<ServerMsg, { t: T }> | undefined {
    return [...this.inbox].reverse().find((m) => m.t === t) as Extract<ServerMsg, { t: T }> | undefined;
  }
  game(): GameState {
    const st = this.last('state');
    if (!st || !this.map) throw new Error('pas de partie');
    return { ...st.game, map: this.map } as GameState;
  }
}

class ManualScheduler implements Scheduler {
  tasks: (() => void)[] = [];
  set(fn: () => void) {
    this.tasks.push(fn);
    return fn;
  }
  clear(h: unknown) {
    this.tasks = this.tasks.filter((t) => t !== h);
  }
}

/** Un Expert de test : l'IA du noyau, qui joue d'un coup toute sa fin de tour (comme les coups de l'Expert). */
class FakeExpert implements Expert {
  calls = 0;
  failing = false;
  async decide(g: GameState): Promise<ExpertMove> {
    this.calls++;
    if (this.failing) throw new Error('panne simulée');
    const actor = currentActor(g);
    const actions: Action[] = [];
    let cur = g;
    do {
      const a = aiAction(cur);
      actions.push(a);
      cur = applyAction(cur, a);
    } while (cur.phase === 'redeploy' && g.phase === 'redeploy' && currentActor(cur) === actor);
    return { actions, value: 0.5, iterations: 200, computeMs: 3, net: 'test' };
  }
}

class MemoryArchive implements GameArchive {
  games = new Map<string, ArchivedGame>();
  save(game: ArchivedGame) {
    this.games.set(game.id, game);
  }
}

/** Fait jouer la partie jusqu'au bout : l'humain joue comme l'IA du noyau, les IA à leur rythme. */
async function playOut(sched: ManualScheduler, host: Phone, rooms: RoomManager, limit = 20000) {
  for (let i = 0; i < limit; i++) {
    const g = host.game();
    if (g.phase === 'gameOver') return g;
    if (currentActor(g) === host.seat && !sched.tasks.length) {
      rooms.handle(host, { t: 'action', id: i, action: aiAction(g) });
      continue;
    }
    const task = sched.tasks.shift();
    if (task) task();
    // la réponse de l'Expert arrive par une promesse
    else await new Promise((r) => setTimeout(r, 0));
  }
  throw new Error('partie trop longue');
}

function soloRoom(size: number, aiLevel: 'difficile' | 'expert', expert: Expert | null) {
  const sched = new ManualScheduler();
  const archive = new MemoryArchive();
  const rooms = new RoomManager(null, sched, { expert, archive, log: () => {} });
  const host = new Phone();
  rooms.handle(host, { t: 'create', name: 'Alice', color: PLAYER_COLORS[0]!.color, size, aiLevel, mapSeed: 2024, opponent: 'ai' });
  return { sched, archive, rooms, host };
}

describe('seul contre l’IA, sur le serveur', () => {
  test('la partie commence aussitôt, toutes les autres places à l’IA', () => {
    const { host } = soloRoom(3, 'difficile', null);
    const room = host.last('room')!.room;
    expect(room.started).toBe(true);
    expect(room.seats.map((s) => s.kind)).toEqual(['human', 'ai', 'ai']);
    const g = host.game();
    expect(g.players.map((p) => p.ai)).toEqual(['difficile', 'difficile', 'difficile']);
    expect(new Set(g.players.map((p) => p.name)).size).toBe(3);
    // l'enregistrement reste sur le serveur
    expect(g.record).toBeUndefined();
  });

  test('l’Expert joue ses coups en plusieurs actions ; chaque décision est chronométrée et archivée', async () => {
    const expert = new FakeExpert();
    const { sched, archive, rooms, host } = soloRoom(3, 'expert', expert);
    const final = await playOut(sched, host, rooms);
    expect(final.phase).toBe('gameOver');
    const [game] = [...archive.games.values()];
    expect(game!.coins).toEqual(final.players.map((p) => p.coins));
    expect(game!.players.map((p) => p.ai)).toEqual(['expert', 'expert', 'expert']);
    const moves = game!.record.moves;
    const thoughts = moves.filter((m) => m.ai);
    // une décision par appel à l'Expert, et des fins de tour jouées sans redemander
    expect(thoughts.length).toBe(expert.calls);
    expect(moves.filter((m) => m.p !== 0).length).toBeGreaterThan(thoughts.length);
    expect(thoughts.every((m) => m.ai!.level === 'expert' && m.ai!.iterations === 200 && m.ai!.ms >= 0)).toBe(true);
    expect(moves.filter((m) => m.p === 0).every((m) => !m.ai)).toBe(true);
    // le réseau de chaque décision, et en tête de la partie
    expect(thoughts.every((m) => m.ai!.net === 'test')).toBe(true);
    expect(game!.expert).toEqual(['test']);
  });

  test('Expert en panne : l’IA Difficile joue à sa place, et c’est noté', async () => {
    const expert = new FakeExpert();
    expert.failing = true;
    const { sched, archive, rooms, host } = soloRoom(2, 'expert', expert);
    const final = await playOut(sched, host, rooms);
    expect(final.phase).toBe('gameOver');
    const [game] = [...archive.games.values()];
    const thoughts = game!.record.moves.filter((m) => m.ai);
    expect(thoughts.length).toBeGreaterThan(10);
    expect(thoughts.every((m) => m.ai!.fallback)).toBe(true);
    expect(game!.expert).toBeUndefined();
  });

  test('une partie archivée se rejoue à l’identique', async () => {
    const { sched, archive, rooms, host } = soloRoom(4, 'difficile', null);
    const final = await playOut(sched, host, rooms);
    const game = [...archive.games.values()][0]!;
    const r = game.record;
    let g = createGame({
      // une place de LLM se rejoue comme une place humaine : personne ne la joue à sa place
      players: game.players.map((p, i) => ({ name: p.name, color: final.players[i]!.color, kind: p.kind === 'llm' ? 'human' : p.kind, ai: p.ai })),
      mapSeed: r.mapSeed,
      seed: r.seed,
    });
    for (const m of r.moves) {
      expect(currentActor(g)).toBe(m.p);
      g = applyAction(g, m.a);
    }
    expect(g.phase).toBe('gameOver');
    expect(g.players.map((p) => p.coins)).toEqual(final.players.map((p) => p.coins));
    // l'état complet, dés compris (le téléphone ne reçoit pas leur graine)
    expect(stateWords({ ...g, rng: 0 })).toEqual(stateWords({ ...final, rng: 0 }));
  });
});

describe('le processus natif de l’Expert', () => {
  test('il répond avec des actions jouables, et se relance après un arrêt', async () => {
    expect(ExpertProcess.available(DEFAULT_EXPERT)).toBeNull();
    const expert = new ExpertProcess({ ...DEFAULT_EXPERT, iterations: 10 });
    try {
      let g = createGame({
        players: [
          { name: 'A', color: PLAYER_COLORS[0]!.color, kind: 'ai', ai: 'expert' },
          { name: 'B', color: PLAYER_COLORS[1]!.color, kind: 'ai', ai: 'expert' },
          { name: 'C', color: PLAYER_COLORS[2]!.color, kind: 'ai', ai: 'expert' },
        ],
        mapSeed: 7,
        seed: 11,
      });
      for (let decision = 0; decision < 12; decision++) {
        const move = await expert.decide(g);
        expect(move.actions.length).toBeGreaterThan(0);
        expect(move.net).toBe(basename(DEFAULT_EXPERT.net, '.bin'));
        for (const a of move.actions) g = applyAction(g, a);
        if (decision === 5) expert.close();
      }
      // les cartes rendues sous la pioche sont suivies par le noyau
      expect(g.raceTail).toBeGreaterThanOrEqual(0);
    } finally {
      expert.close();
    }
  });
});
