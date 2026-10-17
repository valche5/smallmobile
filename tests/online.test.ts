import { afterAll, describe, expect, test } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { ServerWebSocket } from 'bun';
import { RoomManager, type Client, type Scheduler } from '../src/server/rooms';
import { socketHandlers, upgradeSocket, type SocketData } from '../src/server/socket';
import { FileRoomStore } from '../src/server/store';
import type { ClientMsg, ServerMsg } from '../src/shared/protocol';

/** Les IA jouent sans attendre pendant les tests. */
const fast: Scheduler = { set: (fn) => setTimeout(fn, 0), clear: (h) => clearTimeout(h as ReturnType<typeof setTimeout>) };

const rooms = new RoomManager(null, fast);
const sockets = new Set<ServerWebSocket<SocketData>>();
const server = Bun.serve<SocketData>({
  port: 0,
  fetch: (req, srv) => upgradeSocket(req, srv),
  websocket: socketHandlers(rooms, sockets),
});
afterAll(() => server.stop(true));

class Phone {
  ws: WebSocket;
  inbox: ServerMsg[] = [];
  private waiters: { pred: (m: ServerMsg) => boolean; resolve: (m: ServerMsg) => void }[] = [];
  ready: Promise<void>;
  constructor() {
    this.ws = new WebSocket(`ws://localhost:${server.port}/ws`);
    this.ready = new Promise((r) => (this.ws.onopen = () => r()));
    this.ws.onmessage = (e) => {
      const msg = JSON.parse(String(e.data)) as ServerMsg;
      this.inbox.push(msg);
      this.waiters = this.waiters.filter((w) => {
        if (!w.pred(msg)) return true;
        w.resolve(msg);
        return false;
      });
    };
  }
  send(m: ClientMsg) {
    this.ws.send(JSON.stringify(m));
  }
  next<T extends ServerMsg['t']>(t: T, pred: (m: Extract<ServerMsg, { t: T }>) => boolean = () => true) {
    return new Promise<Extract<ServerMsg, { t: T }>>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`attente de « ${t} » trop longue`)), 3000);
      this.waiters.push({
        pred: (m) => m.t === t && pred(m as Extract<ServerMsg, { t: T }>),
        resolve: (m) => {
          clearTimeout(timer);
          resolve(m as Extract<ServerMsg, { t: T }>);
        },
      });
    });
  }
}

describe('serveur WebSocket', () => {
  test('deux téléphones créent, rejoignent et jouent', async () => {
    const alice = new Phone();
    const bob = new Phone();
    await Promise.all([alice.ready, bob.ready]);
    const joinedA = alice.next('joined');
    alice.send({ t: 'create', name: 'Alice', color: '#d62839', size: 3, aiLevel: 'facile', mapSeed: 3033 });
    const { code } = await joinedA;
    const joinedB = bob.next('joined');
    bob.send({ t: 'join', code, name: 'Bob', color: '#1f6fe0' });
    expect((await joinedB).seat).toBe(1);

    const mapB = bob.next('map');
    const stateB = bob.next('state');
    alice.send({ t: 'start' });
    expect((await mapB).map.regions.length).toBe(30);
    expect((await stateB).game.players.map((p) => p.name)).toEqual(['Alice', 'Bob', 'Gorbak']);

    const acked = bob.next('state', (m) => m.ack?.id === 1);
    alice.send({ t: 'action', id: 1, action: { type: 'pick', index: 0 } });
    expect((await acked).game.phase).toBe('conquest');

    // Alice termine son tour sans conquérir ; Bob prend la main
    alice.send({ t: 'action', id: 2, action: { type: 'endConquest' } });
    const bobTurn = bob.next('state', (m) => m.game.current === 1);
    alice.send({ t: 'action', id: 3, action: { type: 'endTurn' } });
    expect((await bobTurn).game.phase).toBe('pick');

    // Bob joue, puis l'IA (Gorbak) enchaîne toute seule jusqu'au tour 2 d'Alice
    bob.send({ t: 'action', id: 1, action: { type: 'pick', index: 0 } });
    bob.send({ t: 'action', id: 2, action: { type: 'endConquest' } });
    const back = alice.next('state', (m) => m.game.round === 2 && m.game.current === 0);
    bob.send({ t: 'action', id: 3, action: { type: 'endTurn' } });
    const round2 = await back;
    expect(round2.game.log.some((l) => l.player === 2 && l.kind === 'pick')).toBe(true);
    alice.ws.close();
    bob.ws.close();
  });

  test('un code inconnu est refusé', async () => {
    const p = new Phone();
    await p.ready;
    const err = p.next('error');
    p.send({ t: 'join', code: 'ZZZZ', name: 'X', color: '' });
    expect((await err).fatal).toBe(true);
    p.ws.close();
  });
});

describe('sauvegarde sur disque', () => {
  test('une partie survit au redémarrage du serveur', () => {
    const dir = mkdtempSync(join(tmpdir(), 'smallmobile-'));
    try {
      const store = new FileRoomStore(dir);
      const first = new RoomManager(store, fast);
      const inbox: ServerMsg[] = [];
      const host: Client = { code: null, seat: null, send: (m) => inbox.push(m) };
      first.handle(host, { t: 'create', name: 'Alice', color: '#d62839', size: 2, aiLevel: 'normal', mapSeed: 2024 });
      const joined = inbox.find((m) => m.t === 'joined') as Extract<ServerMsg, { t: 'joined' }>;
      const guest: Client = { code: null, seat: null, send: () => {} };
      first.handle(guest, { t: 'join', code: joined.code, name: 'Bob', color: '' });
      first.handle(host, { t: 'start' });
      store.flush();

      const second = new RoomManager(new FileRoomStore(dir), fast);
      const again: ServerMsg[] = [];
      const phone: Client = { code: null, seat: null, send: (m) => again.push(m) };
      second.handle(phone, { t: 'resume', code: joined.code, token: joined.token });
      expect(again.some((m) => m.t === 'joined')).toBe(true);
      const st = again.find((m) => m.t === 'state') as Extract<ServerMsg, { t: 'state' }>;
      expect(st.game.players[1]!.name).toBe('Bob');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
