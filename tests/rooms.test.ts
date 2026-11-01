import { describe, expect, test } from 'bun:test';
import { aiAction } from '../src/engine/ai';
import { PLAYER_COLORS } from '../src/engine/data';
import { applyAction } from '../src/engine/game';
import { currentActor } from '../src/engine/rules';
import { isShown } from '../src/shared/pacing';
import type { GameState, MapDef } from '../src/engine/types';
import { RoomManager, type Client, type Scheduler } from '../src/server/rooms';
import type { ClientMsg, ServerMsg } from '../src/shared/protocol';

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
  run(limit = 100000) {
    let n = 0;
    while (this.tasks.length && n++ < limit) this.tasks.shift()!();
  }
}

function setup(size = 2, firstSeat = 0) {
  const sched = new ManualScheduler();
  const rooms = new RoomManager(null, sched, { firstSeat: () => firstSeat });
  const host = new Phone();
  const guest = new Phone();
  const send = (p: Phone, m: ClientMsg) => rooms.handle(p, m);
  send(host, { t: 'create', name: 'Alice', color: PLAYER_COLORS[0]!.color, size, aiLevel: 'normal', mapSeed: 2024 });
  const code = host.last('joined')!.code;
  send(guest, { t: 'join', code: code.toLowerCase(), name: 'Bob', color: PLAYER_COLORS[0]!.color });
  return { rooms, sched, host, guest, send, code };
}

describe('parties en ligne', () => {
  test('créer, rejoindre, couleurs distinctes, lancer', () => {
    const { host, guest, send } = setup(3);
    expect(guest.last('joined')!.seat).toBe(1);
    const room = guest.last('room')!.room;
    expect(room.seats.map((s) => s.kind)).toEqual(['human', 'human', 'ai']);
    // même couleur demandée : c'est le nouveau venu qui en change
    expect(room.seats[0]!.color).toBe(PLAYER_COLORS[0]!.color);
    expect(room.seats[1]!.color).not.toBe(PLAYER_COLORS[0]!.color);
    expect(new Set(room.seats.map((s) => s.color)).size).toBe(3);
    expect(room.seats[1]!.online).toBe(true);
    send(guest, { t: 'start' });
    expect(guest.last('error')!.message).toContain("l'hôte");
    send(host, { t: 'start' });
    expect(host.map).not.toBeNull();
    expect(guest.map).not.toBeNull();
    const g = guest.game();
    expect(g.players.map((p) => p.name)).toEqual(['Alice', 'Bob', 'Gorbak']);
    expect(g.rng).toBe(0);
  });

  test('le premier joueur tiré commence, les sièges tournent', () => {
    const { rooms, host, guest, send, code } = setup(3, 1);
    const token = host.last('joined')!.token;
    send(host, { t: 'start' });
    const g = host.game();
    expect(g.players.map((p) => p.name)).toEqual(['Bob', 'Gorbak', 'Alice']);
    expect(g.current).toBe(0);
    expect(host.last('joined')).toMatchObject({ seat: 2, token });
    expect(guest.last('joined')!.seat).toBe(0);
    expect(host.last('room')!.room.seats.map((s) => s.name)).toEqual(['Bob', 'Gorbak', 'Alice']);
    // le siège de chacun suit : l'hôte attend son tour, un second « lancer » ne fait rien
    send(host, { t: 'start' });
    expect(host.last('error')).toBeUndefined();
    send(host, { t: 'action', id: 1, action: { type: 'pick', index: 0 } });
    expect(host.last('error')!.message).toContain('pas à vous');
    send(guest, { t: 'action', id: 1, action: { type: 'pick', index: 0 } });
    expect(guest.last('state')!.ack).toEqual({ seat: 0, id: 1 });
    // à la reconnexion aussi
    rooms.disconnect(host);
    const back = new Phone();
    send(back, { t: 'resume', code, token });
    expect(back.last('joined')!.seat).toBe(2);
  });

  test('par défaut, le premier joueur est tiré au hasard', () => {
    const rooms = new RoomManager(null, new ManualScheduler());
    const firsts = new Set<string>();
    for (let i = 0; i < 64; i++) {
      const host = new Phone();
      rooms.handle(host, { t: 'create', opponent: 'ai', name: 'Alice', color: PLAYER_COLORS[0]!.color, size: 2, aiLevel: 'normal', mapSeed: 2024 });
      firsts.add(host.game().players[0]!.name);
    }
    expect(firsts.size).toBe(2);
  });

  test('un troisième téléphone ne peut pas rejoindre', () => {
    const { send, code } = setup(2);
    const third = new Phone();
    send(third, { t: 'join', code, name: 'Carl', color: '' });
    expect(third.last('error')).toMatchObject({ fatal: true });
  });

  test("seul le joueur dont c'est le tour peut agir, et il peut annuler", () => {
    const { host, guest, send } = setup(2);
    send(host, { t: 'start' });
    send(guest, { t: 'action', id: 1, action: { type: 'pick', index: 0 } });
    expect(guest.last('error')!.message).toContain("pas à vous");
    send(host, { t: 'action', id: 7, action: { type: 'pick', index: 0 } });
    const st = guest.last('state')!;
    expect(st.ack).toEqual({ seat: 0, id: 7 });
    expect(st.undo).toBe(1);
    expect(guest.game().phase).toBe('conquest');
    send(host, { t: 'undo' });
    expect(guest.game().phase).toBe('pick');
    send(host, { t: 'action', id: 8, action: { type: 'conquer', region: 999 } });
    expect(host.last('error')).toBeDefined();
  });

  test('partie complète à 3 (2 humains + 1 IA) jusqu\'au bout', () => {
    const { host, guest, send, sched } = setup(3);
    send(host, { t: 'start' });
    let id = 0;
    for (let k = 0; k < 5000; k++) {
      sched.run();
      const g = host.game();
      if (g.phase === 'gameOver') break;
      const actor = currentActor(g)!;
      const phone = actor === 0 ? host : actor === 1 ? guest : null;
      expect(phone).not.toBeNull();
      send(phone!, { t: 'action', id: ++id, action: aiAction(g) });
    }
    const end = host.game();
    expect(end.phase).toBe('gameOver');
    expect(JSON.stringify(guest.last('state')!.game)).toBe(JSON.stringify(host.last('state')!.game));
  });

  test('reconnexion et remplacement par une IA', () => {
    const { rooms, host, guest, send, sched, code } = setup(2);
    send(host, { t: 'start' });
    const token = host.last('joined')!.token;
    rooms.disconnect(host);
    expect(guest.last('room')!.room.seats[0]!.online).toBe(false);
    // le joueur 2 confie le siège absent à l'IA, qui joue son tour
    send(guest, { t: 'replace', seat: 0 });
    expect(guest.last('room')!.room.seats[0]!.replaced).toBe(true);
    sched.run(30);
    expect(guest.game().log.some((l) => l.kind === 'pick' && l.player === 0)).toBe(true);
    // Alice revient : elle reprend la main
    const back = new Phone();
    send(back, { t: 'resume', code, token });
    expect(back.last('joined')!.seat).toBe(0);
    expect(back.map).not.toBeNull();
    expect(back.game().players[0]!.kind).toBe('human');
    expect(guest.last('room')!.room.seats[0]).toMatchObject({ replaced: false, online: true });
  });

  /** Alice joue son tour, son téléphone se met en veille, puis Bob et les IA jouent `n` actions (ou jusqu'au tour `round`). */
  function sleepThenPlay(n: number, round = Infinity) {
    const env = setup(4);
    const { rooms, host, guest, send, sched } = env;
    send(host, { t: 'start' });
    const token = host.last('joined')!.token;
    let id = 0;
    while (currentActor(host.game()) === 0) send(host, { t: 'action', id: ++id, action: aiAction(host.game()) });
    const s = host.game().seq;
    rooms.disconnect(host);
    send(guest, { t: 'replace', seat: 0 });
    let actions = 0;
    for (let k = 0; k < 2000 && actions < n; k++) {
      const g = guest.game();
      if (g.phase === 'gameOver' || g.round >= round) break;
      if (currentActor(g) === 1) send(guest, { t: 'action', id: ++id, action: aiAction(g) });
      else sched.run(1);
      actions++;
    }
    const back = new Phone();
    send(back, { t: 'resume', code: env.code, token, shown: s });
    return { ...env, s, back, final: guest.game(), actions };
  }

  test('reconnexion après 20 actions : le serveur rejoue les états manqués, sur leurs vrais états', () => {
    const { s, back, final, guest, actions } = sleepThenPlay(20);
    expect(actions).toBe(20);
    const st = back.last('state')!;
    expect(st.sync).toBe(true);
    const replay = st.replay!;
    expect(replay).toBeDefined();
    expect(replay.base!.seq).toBeLessThanOrEqual(s);
    expect(replay.missing).toBe(0);
    expect(replay.frames.every((f) => f.seq > s)).toBe(true);
    // les ajustements sans événement consécutifs ne font qu'un état
    for (let i = 1; i < replay.frames.length; i++) {
      expect(replay.frames[i]!.events.length + replay.frames[i - 1]!.events.length).toBeGreaterThan(0);
    }
    // toutes les entrées des autres depuis la trame vue sont là, chacune avec l'état de son moment
    const others = final.log.filter((l) => l.seq > s && l.player !== 0 && isShown(l)).map((l) => l.seq);
    const replayed = replay.frames.flatMap((f) => f.events.map((e) => e.seq));
    expect(others.length).toBeGreaterThan(5);
    for (const q of others) expect(replayed).toContain(q);
    for (const f of replay.frames) {
      expect(f.game.seq).toBe(f.seq);
      const last = f.events[f.events.length - 1];
      if (last?.kind === 'conquer') {
        const race = f.game.regions[last.region!]!.race;
        expect(race && race !== 'lost' ? f.game.races[race]!.owner : null).toBe(last.player);
      }
    }
    // Bob ne reçoit pas de relecture : son téléphone suit en direct
    expect(guest.inbox.some((m) => m.t === 'state' && m.replay)).toBe(false);
  });

  test("absence très longue : les entrées plus anciennes que l'historique sont comptées", () => {
    const { s, back, final } = sleepThenPlay(10000, 4);
    const replay = back.last('state')!.replay!;
    expect(replay.base).toBeNull();
    const replayed = new Set(replay.frames.flatMap((f) => f.events.map((e) => e.seq)));
    const others = final.log.filter((l) => l.seq > s && l.player !== 0 && isShown(l));
    expect(replay.missing).toBe(others.filter((l) => !replayed.has(l.seq)).length);
    expect(replay.missing).toBeGreaterThan(0);
  });

  test("reprise d'un siège confié à l'IA : resynchronisation, pas une action adverse", () => {
    const { rooms, host, guest, send, sched, code } = setup(2);
    send(host, { t: 'start' });
    const token = host.last('joined')!.token;
    rooms.disconnect(host);
    send(guest, { t: 'replace', seat: 0 });
    sched.run(30);
    const back = new Phone();
    send(back, { t: 'resume', code, token });
    const states = back.inbox.filter((m) => m.t === 'state') as Extract<ServerMsg, { t: 'state' }>[];
    expect(states.length).toBe(1);
    expect(states[0]!.sync).toBe(true);
    expect(guest.last('state')!.sync).toBeUndefined();
  });

  test("une annulation retire l'action de l'historique", () => {
    const { rooms, host, guest, send, code } = setup(2);
    send(host, { t: 'start' });
    const token = host.last('joined')!.token;
    let id = 0;
    while (currentActor(host.game()) === 0) send(host, { t: 'action', id: ++id, action: aiAction(host.game()) });
    const s = host.game().seq;
    send(guest, { t: 'action', id: 1, action: { type: 'pick', index: 0 } });
    const g = guest.game();
    const target = g.map.regions.map((r) => r.id).find((r) => {
      try {
        return applyAction(g, { type: 'conquer', region: r }) !== null;
      } catch {
        return false;
      }
    })!;
    send(guest, { t: 'action', id: 2, action: { type: 'conquer', region: target } });
    send(guest, { t: 'undo' });
    rooms.disconnect(host);
    const back = new Phone();
    send(back, { t: 'resume', code, token, shown: s });
    const replay = back.last('state')!.replay!;
    const kinds = replay.frames.flatMap((f) => f.events.map((e) => e.kind));
    expect(kinds).toContain('pick');
    expect(kinds).not.toContain('conquer');
  });

  test('vitesse : réglage de salle, avec son auteur', () => {
    const { host, guest, send } = setup(2);
    send(host, { t: 'start' });
    send(guest, { t: 'speed', speed: 'rapide' });
    expect(host.last('room')!.room).toMatchObject({ speed: 'rapide', speedBy: 1 });
  });

  test("quitter le salon : le siège se libère, l'hôte ferme la partie", () => {
    const { host, guest, send, code } = setup(2);
    send(guest, { t: 'leave' });
    expect(host.last('room')!.room.seats[1]!.taken).toBe(false);
    const other = new Phone();
    send(other, { t: 'join', code, name: 'Chloé', color: '' });
    expect(other.last('joined')!.seat).toBe(1);
    send(host, { t: 'leave' });
    expect(other.last('error')).toMatchObject({ fatal: true });
  });

  test("l'hôte règle la carte avant de lancer", () => {
    const { host, guest, send } = setup(2);
    send(host, { t: 'lobby', size: 4, aiLevel: 'difficile' });
    let room = guest.last('room')!.room;
    expect(room.seats.length).toBe(4);
    expect(room.seats[3]!.ai).toBe('difficile');
    send(guest, { t: 'lobby', size: 2 });
    room = guest.last('room')!.room;
    expect(room.seats.length).toBe(4);
    send(guest, { t: 'lobby', name: 'Bobby', color: room.seats[0]!.color });
    expect(guest.last('error')!.message).toContain('Alice');
    send(guest, { t: 'lobby', color: room.seats[3]!.color });
    room = host.last('room')!.room;
    expect(room.seats[1]!.name).toBe('Bobby');
    expect(room.seats[1]!.color).toBe(guest.last('room')!.room.seats[1]!.color);
    expect(new Set(room.seats.map((s) => s.color)).size).toBe(4);
  });
});
