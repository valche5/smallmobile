import { aiAction } from '../engine/ai';
import { AI_NAMES, PLAYER_COLORS } from '../engine/data';
import { applyAction, createGame, RuleError } from '../engine/game';
import { PRESETS } from '../engine/mapgen';
import { randomSeed } from '../engine/rng';
import { currentActor } from '../engine/rules';
import type { Action, AiLevel, AiThought, GameState } from '../engine/types';
import { FACTOR, isShown, newEvents, playbackMs, REDEPLOY_MS, splitFrames, thinkDelay } from '../shared/pacing';
import {
  CODE_ALPHABET,
  CODE_LENGTH,
  normalizeCode,
  type AiSpeed,
  type ClientMsg,
  type GameWire,
  type Replay,
  type RoomView,
  type ServerMsg,
  type Snapshot,
} from '../shared/protocol';
import { EXPERT_LABS, expertLab } from '../shared/experts';
import { expertNets, gameId, type GameArchive } from './archive';
import type { Expert, Experts } from './expert';

/** Un téléphone connecté. */
export interface Client {
  send(msg: ServerMsg): void;
  code: string | null;
  seat: number | null;
}

interface Seat {
  name: string;
  color: string;
  kind: 'human' | 'ai';
  ai: AiLevel;
  /** Jeton secret qui permet au joueur humain de se reconnecter. */
  token: string | null;
  /** Joueur humain absent, remplacé par l'IA. */
  replaced: boolean;
}

/** Ce qui est sauvegardé sur disque. */
export interface RoomData {
  code: string;
  createdAt: number;
  updatedAt: number;
  size: number;
  mapSeed: number;
  aiLevel: AiLevel;
  /** Le labo dont l'Expert joue les sièges de niveau `expert` (`src/shared/experts.ts`). */
  expertLab: string;
  speed: AiSpeed;
  /** Siège qui a réglé la vitesse en dernier. */
  speedBy?: number | null;
  seats: Seat[];
  game: GameState | null;
}

interface Room extends RoomData {
  clients: Set<Client>;
  undo: GameState[];
  timer: unknown;
  /** Les téléphones affichent encore la dernière action jusqu'à cet instant (ms). */
  busyUntil: number;
  /** Un groupe de redéploiement est en cours d'affichage (son temps minimal est déjà compté). */
  groupOpen: boolean;
  /** États récents, pour rejouer à un téléphone ce qu'il a manqué (en mémoire seulement). */
  history: Snapshot[];
  /** Le reste du coup choisi par l'Expert, joué action par action tant que la partie n'a pas bougé. */
  plan: { seq: number; actions: Action[] } | null;
}

/** Au plus autant d'états gardés par partie. */
const MAX_HISTORY = 400;

export interface Scheduler {
  set(fn: () => void, ms: number): unknown;
  clear(handle: unknown): void;
}

export interface RoomStore {
  save(room: RoomData): void;
  remove(code: string): void;
  loadAll(): RoomData[];
}

const IRREVERSIBLE = new Set<Action['type']>(['reinforce', 'berserkRoll', 'decline', 'endTurn', 'endDefense', 'skipPick']);
const LEVELS: AiLevel[] = ['facile', 'normal', 'difficile', 'expert'];
const SPEEDS: AiSpeed[] = ['lent', 'normal', 'rapide'];

const realScheduler: Scheduler = {
  set: (fn, ms) => setTimeout(fn, ms),
  clear: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

function cleanName(raw: unknown, fallback: string): string {
  const s = typeof raw === 'string' ? raw.trim().replace(/\s+/g, ' ').slice(0, 14) : '';
  return s || fallback;
}

function validColor(raw: unknown): string | null {
  return PLAYER_COLORS.some((c) => c.color === raw) ? (raw as string) : null;
}

function toWire(game: GameState): GameWire {
  const { map: _map, record: _record, ...rest } = game;
  // la graine du hasard reste sur le serveur : personne ne peut prédire les dés
  return { ...rest, rng: 0 };
}

/** Ce que le serveur peut confier à d'autres : l'IA Expert, l'archive des parties jouées, le journal. */
export interface RoomServices {
  experts?: Experts;
  archive?: GameArchive | null;
  /** Le journal du serveur (temps de décision de l'Expert). */
  log?: (line: string) => void;
  /** Tire le siège du premier joueur, parmi `players` (au hasard par défaut). */
  firstSeat?: (players: number) => number;
}

export class RoomManager {
  private rooms = new Map<string, Room>();
  private experts: Experts;
  private archive: GameArchive | null;
  private log: (line: string) => void;
  private firstSeat: (players: number) => number;

  constructor(
    private store: RoomStore | null = null,
    private scheduler: Scheduler = realScheduler,
    services: RoomServices = {},
  ) {
    this.experts = services.experts ?? new Map();
    this.archive = services.archive ?? null;
    this.log = services.log ?? ((line) => console.log(line));
    this.firstSeat = services.firstSeat ?? ((players) => Math.floor(Math.random() * players));
    for (const data of store?.loadAll() ?? []) {
      // salles sauvegardées avant les variantes de l'Expert : celle par défaut
      const room: Room = {
        ...data,
        expertLab: expertLab(data.expertLab),
        clients: new Set(),
        undo: [],
        timer: null,
        busyUntil: 0,
        groupOpen: false,
        history: [],
        plan: null,
      };
      this.rooms.set(room.code, room);
      this.scheduleAi(room);
    }
  }

  get size(): number {
    return this.rooms.size;
  }

  // --- Entrées --------------------------------------------------------------

  handle(client: Client, msg: ClientMsg) {
    try {
      this.dispatch(client, msg);
    } catch (e) {
      console.error('Message invalide', e);
      client.send({ t: 'error', message: 'Requête invalide.' });
    }
  }

  disconnect(client: Client) {
    const room = this.roomOf(client);
    if (!room) return;
    room.clients.delete(client);
    client.code = null;
    client.seat = null;
    this.broadcastRoom(room);
  }

  /** Supprime les parties inactives depuis plus de `maxAgeMs`. */
  prune(maxAgeMs: number, now = Date.now()) {
    for (const room of this.rooms.values()) {
      if (now - room.updatedAt > maxAgeMs && room.clients.size === 0) this.removeRoom(room);
    }
  }

  // --- Logique ----------------------------------------------------------------

  private dispatch(client: Client, msg: ClientMsg) {
    switch (msg.t) {
      case 'create':
        return this.create(client, msg);
      case 'join':
        return this.join(client, msg);
      case 'resume':
        return this.resume(client, msg);
      case 'lobby':
        return this.updateLobby(client, msg);
      case 'start':
        return this.start(client);
      case 'action':
        return this.action(client, msg.id, msg.action);
      case 'undo':
        return this.undo(client);
      case 'speed': {
        const room = this.roomOf(client);
        if (!room || !SPEEDS.includes(msg.speed)) return;
        room.speed = msg.speed;
        room.speedBy = client.seat;
        this.broadcastRoom(room);
        this.touch(room);
        return;
      }
      case 'replace':
        return this.replace(client, msg.seat);
      case 'leave':
        return this.leave(client);
    }
  }

  private create(client: Client, msg: Extract<ClientMsg, { t: 'create' }>) {
    const size = PRESETS[msg.size] ? msg.size : 2;
    const aiLevel = LEVELS.includes(msg.aiLevel) ? msg.aiLevel : 'normal';
    const solo = msg.opponent === 'ai';
    const code = this.newCode();
    const token = crypto.randomUUID();
    const now = Date.now();
    const room: Room = {
      code,
      createdAt: now,
      updatedAt: now,
      size,
      mapSeed: Number.isInteger(msg.mapSeed) ? msg.mapSeed : PRESETS[size]!.seed,
      aiLevel,
      expertLab: expertLab(msg.expertLab),
      speed: 'normal',
      seats: [
        { name: cleanName(msg.name, 'Joueur 1'), color: validColor(msg.color) ?? PLAYER_COLORS[0]!.color, kind: 'human', ai: aiLevel, token, replaced: false },
        solo
          ? { name: AI_NAMES[size - 2]!, color: '', kind: 'ai', ai: aiLevel, token: null, replaced: false }
          : { name: 'Joueur 2', color: '', kind: 'human', ai: aiLevel, token: null, replaced: false },
      ],
      game: null,
      clients: new Set(),
      undo: [],
      timer: null,
      busyUntil: 0,
      groupOpen: false,
      history: [],
      plan: null,
    };
    this.resizeSeats(room);
    this.rooms.set(code, room);
    this.attach(client, room, 0);
    client.send({ t: 'joined', code, token, seat: 0 });
    this.broadcastRoom(room);
    this.touch(room);
    // seul contre l'IA : personne à attendre
    if (solo) this.start(client);
  }

  private join(client: Client, msg: Extract<ClientMsg, { t: 'join' }>) {
    const room = this.rooms.get(normalizeCode(String(msg.code)));
    if (!room) return client.send({ t: 'error', message: 'Aucune partie ne porte ce code.', fatal: true });
    if (room.game) return client.send({ t: 'error', message: 'Cette partie a déjà commencé.', fatal: true });
    const seat = room.seats.findIndex((s) => s.kind === 'human' && !s.token);
    if (seat < 0) return client.send({ t: 'error', message: 'Cette partie est déjà complète.', fatal: true });
    const s = room.seats[seat]!;
    s.name = cleanName(msg.name, `Joueur ${seat + 1}`);
    s.color = validColor(msg.color) ?? '';
    s.token = crypto.randomUUID();
    this.fixColors(room, 0); // les joueurs déjà présents gardent leur couleur
    this.attach(client, room, seat);
    client.send({ t: 'joined', code: room.code, token: s.token, seat });
    this.broadcastRoom(room);
    this.touch(room);
  }

  private resume(client: Client, msg: Extract<ClientMsg, { t: 'resume' }>) {
    const room = this.rooms.get(normalizeCode(String(msg.code)));
    if (!room) return client.send({ t: 'error', message: "Cette partie n'existe plus.", fatal: true });
    const seat = room.seats.findIndex((s) => s.token !== null && s.token === msg.token);
    if (seat < 0) return client.send({ t: 'error', message: "Vous ne faites pas partie de cette partie.", fatal: true });
    const s = room.seats[seat]!;
    const wasReplaced = s.replaced;
    if (wasReplaced) {
      // le joueur revient : il reprend la main sur son siège
      s.replaced = false;
      if (room.game) room.game.players[seat]!.kind = 'human';
      // les états annulables datent de quand l'IA tenait ce siège : y revenir lui redonnerait l'étiquette « IA »
      room.undo = [];
    }
    this.attach(client, room, seat);
    client.send({ t: 'joined', code: room.code, token: s.token!, seat });
    this.broadcastRoom(room);
    if (room.game) {
      client.send({ t: 'map', map: room.game.map });
      // le siège revient de l'IA : les autres le voient ; ce téléphone, lui, repart toujours d'un état de resynchronisation
      if (wasReplaced) this.broadcastState(room, null, client);
      this.sendState(client, room, null, typeof msg.shown === 'number' ? msg.shown : null);
    }
    this.touch(room);
  }

  private updateLobby(client: Client, msg: Extract<ClientMsg, { t: 'lobby' }>) {
    const room = this.roomOf(client);
    if (!room || client.seat === null) return;
    if (room.game) return client.send({ t: 'error', message: 'La partie a déjà commencé.' });
    const me = room.seats[client.seat]!;
    if (msg.name !== undefined) me.name = cleanName(msg.name, me.name);
    if (msg.color !== undefined && validColor(msg.color)) {
      const other = room.seats.find((s, i) => i !== client.seat && s.color === msg.color);
      if (other?.kind === 'human' && other.token) {
        client.send({ t: 'error', message: `Cette couleur est déjà celle de ${other.name}.` });
      } else {
        if (other) other.color = me.color;
        me.color = msg.color;
      }
    }
    if (client.seat === 0) {
      if (msg.size !== undefined && PRESETS[msg.size] && msg.size !== room.size) {
        room.size = msg.size;
        room.mapSeed = PRESETS[msg.size]!.seed;
        this.resizeSeats(room);
      }
      if (msg.aiLevel !== undefined && LEVELS.includes(msg.aiLevel)) {
        room.aiLevel = msg.aiLevel;
        room.seats.forEach((s) => (s.ai = msg.aiLevel!));
      }
      if (EXPERT_LABS.some((l) => l.id === msg.expertLab)) room.expertLab = msg.expertLab!;
      if (msg.mapSeed !== undefined && Number.isInteger(msg.mapSeed)) room.mapSeed = msg.mapSeed;
    }
    this.fixColors(room, client.seat);
    this.broadcastRoom(room);
    this.touch(room);
  }

  private start(client: Client) {
    const room = this.roomOf(client);
    if (!room || room.game) return;
    if (client.seat !== 0) return client.send({ t: 'error', message: "Seul l'hôte peut lancer la partie." });
    if (room.seats.some((s) => s.kind === 'human' && !s.token))
      return client.send({ t: 'error', message: "Attendez que l'autre joueur ait rejoint la partie." });
    this.rotateSeats(room);
    room.game = createGame({
      players: room.seats.map((s) => ({ name: s.name, color: s.color, kind: s.kind, ai: s.ai })),
      mapSeed: room.mapSeed,
      seed: randomSeed(),
    });
    room.undo = [];
    room.history = [];
    room.groupOpen = false;
    room.plan = null;
    for (const c of room.clients) c.send({ t: 'map', map: room.game.map });
    this.broadcastRoom(room);
    this.broadcastState(room, null);
    this.touch(room);
    this.scheduleAi(room);
  }

  private action(client: Client, id: number, action: Action) {
    const room = this.roomOf(client);
    const g = room?.game;
    if (!room || !g) return client.send({ t: 'error', message: 'Aucune partie en cours.' });
    const actor = currentActor(g);
    if (actor === null || actor !== client.seat || this.isAi(room, actor)) {
      client.send({ t: 'error', message: "Ce n'est pas à vous de jouer." });
      return this.sendState(client, room, null);
    }
    let next: GameState;
    try {
      next = applyAction(g, action);
    } catch (e) {
      client.send({ t: 'error', message: e instanceof RuleError ? e.message : 'Action invalide.' });
      return this.sendState(client, room, null);
    }
    const irreversible = IRREVERSIBLE.has(action.type) || currentActor(next) !== actor;
    room.undo = irreversible ? [] : [...room.undo.slice(-30), g];
    room.game = next;
    this.markBusy(room, g, next);
    this.broadcastState(room, { seat: actor, id });
    this.touch(room);
    this.scheduleAi(room);
  }

  private undo(client: Client) {
    const room = this.roomOf(client);
    if (!room?.game) return;
    const prev = room.undo[room.undo.length - 1];
    if (!prev || currentActor(room.game) !== client.seat) return client.send({ t: 'error', message: 'Rien à annuler.' });
    room.undo = room.undo.slice(0, -1);
    room.game = prev;
    this.broadcastState(room, null);
    this.touch(room);
  }

  private replace(client: Client, seat: number) {
    const room = this.roomOf(client);
    if (!room?.game || client.seat === null) return;
    const s = room.seats[seat];
    if (!s || s.kind !== 'human' || seat === client.seat) return;
    if (this.online(room, seat)) return client.send({ t: 'error', message: 'Ce joueur est connecté.' });
    s.replaced = true;
    const p = room.game.players[seat]!;
    p.kind = 'ai';
    p.ai = room.aiLevel;
    room.undo = [];
    this.broadcastRoom(room);
    this.broadcastState(room, null);
    this.touch(room);
    this.scheduleAi(room);
  }

  private leave(client: Client) {
    const room = this.roomOf(client);
    if (!room || client.seat === null) return client.send({ t: 'left' });
    const seat = client.seat;
    this.disconnect(client);
    client.send({ t: 'left' });
    if (room.game) return; // en cours de partie, le siège reste réservé (reconnexion ou remplacement par l'IA)
    if (seat === 0) {
      for (const c of room.clients) {
        c.send({ t: 'error', message: "L'hôte a fermé la partie.", fatal: true });
        c.code = null;
        c.seat = null;
      }
      this.removeRoom(room);
      return;
    }
    const s = room.seats[seat]!;
    s.token = null;
    s.name = `Joueur ${seat + 1}`;
    this.broadcastRoom(room);
    this.touch(room);
  }

  // --- IA ---------------------------------------------------------------------

  private isAi(room: Room, seat: number): boolean {
    const s = room.seats[seat]!;
    return s.kind === 'ai' || s.replaced;
  }

  /** Les téléphones vont afficher l'action qui vient d'être jouée : on la laisse se dérouler avant la suivante. */
  private markBusy(room: Room, before: GameState, after: GameState) {
    const events = newEvents(before, after);
    let ms = playbackMs(events, room.speed);
    if (splitFrames(events).length) room.groupOpen = false;
    else if (after.phase === 'redeploy' && !room.groupOpen) {
      // les petits ajustements de redéploiement forment un seul groupe, montré une fois
      room.groupOpen = true;
      ms += REDEPLOY_MS * FACTOR[room.speed];
    }
    room.busyUntil = Math.max(room.busyUntil, this.now()) + ms;
  }

  private now(): number {
    return Date.now();
  }

  private scheduleAi(room: Room) {
    if (room.timer) return;
    const g = room.game;
    if (!g || g.phase === 'gameOver') return;
    const actor = currentActor(g);
    if (actor === null || !this.isAi(room, actor)) return;
    const delay = Math.max(thinkDelay(room.speed), room.busyUntil - this.now());
    room.timer = this.scheduler.set(() => {
      room.timer = null;
      const cur = room.game;
      if (!cur || cur.phase === 'gameOver') return;
      const who = currentActor(cur);
      if (who === null || !this.isAi(room, who)) return;
      const level = cur.players[who]!.ai;
      const planned = room.plan?.seq === cur.seq ? room.plan.actions[0] : undefined;
      room.plan = null;
      if (planned && this.play(room, cur, planned)) return;
      const expert = level === 'expert' ? this.experts.get(room.expertLab) : undefined;
      if (expert) return this.askExpert(room, cur, who, expert);
      const start = performance.now();
      let action: Action;
      try {
        action = aiAction(cur);
      } catch (e) {
        console.error(`IA bloquée dans la partie ${room.code}`, e);
        return;
      }
      this.play(room, cur, action, { level, ms: Math.round(performance.now() - start) });
    }, delay);
  }

  /** L'Expert réfléchit hors du serveur ; la partie ne doit pas avoir bougé quand il répond. */
  private askExpert(room: Room, cur: GameState, who: number, expert: Expert) {
    const start = performance.now();
    // la salle reste occupée : aucun autre tour d'IA ne part pendant la réflexion
    room.timer = 'expert';
    expert.decide(cur).then(
      (move) => {
        room.timer = null;
        if (room.game !== cur || !this.rooms.has(room.code)) return this.scheduleAi(room);
        const ms = Math.round(performance.now() - start);
        const value = Number.isNaN(move.value) ? undefined : Math.round(move.value * 1000) / 1000;
        const thought: AiThought = {
          level: 'expert',
          ms,
          actions: move.actions.length,
          iterations: move.iterations,
          computeMs: Math.round(move.computeMs),
          ...(value !== undefined ? { value } : {}),
          net: move.net,
        };
        if (move.iterations) {
          const chance = value !== undefined ? `, chance ${Math.round(value * 100)} %` : '';
          this.log(
            `[expert] ${room.code} siège ${who + 1} manche ${cur.round} : ${ms} ms (calcul ${thought.computeMs} ms, n=${move.iterations}, ${move.actions.length} action(s)${chance})`,
          );
        }
        const [first, ...rest] = move.actions;
        if (!this.play(room, cur, first!, thought)) return this.fallback(room, cur, start, 'coup refusé par les règles');
        if (rest.length) room.plan = { seq: room.game!.seq, actions: rest };
      },
      (e: Error) => {
        room.timer = null;
        if (room.game !== cur || !this.rooms.has(room.code)) return this.scheduleAi(room);
        this.fallback(room, cur, start, e.message);
      },
    );
  }

  /** L'Expert n'a pas pu jouer : l'IA Difficile joue à sa place cette fois. */
  private fallback(room: Room, cur: GameState, start: number, why: string) {
    this.log(`[expert] ${room.code} : ${why} ; l'IA Difficile joue à sa place`);
    let action: Action;
    try {
      action = aiAction(cur);
    } catch (e) {
      console.error(`IA bloquée dans la partie ${room.code}`, e);
      return;
    }
    this.play(room, cur, action, { level: 'expert', ms: Math.round(performance.now() - start), fallback: true });
  }

  /** Joue une action d'IA et prévoit la suite ; faux si les règles la refusent. */
  private play(room: Room, cur: GameState, action: Action, thought?: AiThought): boolean {
    try {
      room.game = applyAction(cur, action, thought);
    } catch (e) {
      console.error(`Action d'IA refusée dans la partie ${room.code}`, e);
      return false;
    }
    room.undo = [];
    this.markBusy(room, cur, room.game);
    this.broadcastState(room, null);
    this.touch(room);
    if (room.game.phase === 'gameOver') this.logThinking(room);
    this.scheduleAi(room);
    return true;
  }

  /** Fin de partie : le temps de réflexion de l'Expert, en bref (le détail est dans l'archive). */
  private logThinking(room: Room) {
    const times = (room.game?.record?.moves ?? []).flatMap((m) => (m.ai?.level === 'expert' && m.ai.iterations ? [m.ai.ms] : []));
    if (!times.length) return;
    times.sort((a, b) => a - b);
    const at = (q: number) => times[Math.min(times.length - 1, Math.floor(q * times.length))]!;
    const total = times.reduce((a, b) => a + b, 0);
    this.log(
      `[expert] ${room.code} fin de partie : ${times.length} décisions, médiane ${at(0.5)} ms, 90 % sous ${at(0.9)} ms, max ${times[times.length - 1]} ms, total ${Math.round(total / 1000)} s`,
    );
  }

  // --- Outils -----------------------------------------------------------------

  private roomOf(client: Client): Room | undefined {
    return client.code ? this.rooms.get(client.code) : undefined;
  }

  private attach(client: Client, room: Room, seat: number) {
    const prev = this.roomOf(client);
    if (prev && prev !== room) {
      prev.clients.delete(client);
      this.broadcastRoom(prev);
    }
    client.code = room.code;
    client.seat = seat;
    room.clients.add(client);
  }

  private online(room: Room, seat: number): boolean {
    for (const c of room.clients) if (c.seat === seat) return true;
    return false;
  }

  private view(room: Room): RoomView {
    return {
      code: room.code,
      size: room.size,
      mapSeed: room.mapSeed,
      aiLevel: room.aiLevel,
      expertLab: room.expertLab,
      speed: room.speed,
      speedBy: room.speedBy ?? null,
      started: !!room.game,
      seats: room.seats.map((s, i) => ({
        name: s.name,
        color: s.color,
        kind: s.kind,
        ai: s.ai,
        taken: s.kind === 'human' && !!s.token,
        online: this.online(room, i),
        replaced: s.replaced,
      })),
    };
  }

  private broadcastRoom(room: Room) {
    const msg: ServerMsg = { t: 'room', room: this.view(room) };
    for (const c of room.clients) c.send(msg);
  }

  /** État de resynchronisation ; `shown` : dernière trame affichée par ce téléphone, pour lui rejouer la suite. */
  private sendState(client: Client, room: Room, ack: { seat: number; id: number } | null, shown: number | null = null) {
    if (!room.game) return;
    const replay = shown === null || client.seat === null ? undefined : this.replay(room, shown, client.seat);
    client.send({ t: 'state', game: toWire(room.game), ack, undo: room.undo.length, sync: true, ...(replay ? { replay } : {}) });
  }

  private broadcastState(room: Room, ack: { seat: number; id: number } | null, except: Client | null = null) {
    if (!room.game) return;
    this.record(room);
    const msg: ServerMsg = { t: 'state', game: toWire(room.game), ack, undo: room.undo.length };
    for (const c of room.clients) if (c !== except) c.send(msg);
  }

  /** Ajoute l'état courant à l'historique (les états sans événement consécutifs se remplacent). */
  private record(room: Room) {
    const g = room.game!;
    const h = room.history;
    // annulation : rien de ce qui suit l'état rétabli ne doit être rejoué
    while (h.length && h[h.length - 1]!.seq > g.seq) h.pop();
    const prev = h[h.length - 1];
    const events = prev ? g.log.filter((l) => l.seq > prev.seq) : [];
    const { log: _log, ...game } = toWire(g);
    const snap: Snapshot = { seq: g.seq, game, events };
    if (prev && !events.length && !prev.events.length) h[h.length - 1] = snap;
    else h.push(snap);
    let drop = 0;
    while (drop < h.length - 1 && (h.length - drop > MAX_HISTORY || h[drop]!.game.round < g.round - 1)) drop++;
    if (drop) h.splice(0, drop);
  }

  /** Ce que le siège `seat` n'a pas vu depuis la trame `shown`, ou rien s'il n'a rien manqué des autres. */
  private replay(room: Room, shown: number, seat: number): Replay | undefined {
    const g = room.game!;
    const h = room.history;
    if (!h.length || !Number.isInteger(shown)) return undefined;
    const frames = h.filter((x) => x.seq > shown);
    let base: Snapshot | null = null;
    for (const x of h) if (x.seq <= shown) base = x;
    let missing = 0;
    if (!base) {
      const first = h[0]!;
      const from = first.events[0]?.seq ?? first.seq + 1;
      missing = g.log.filter((l) => l.seq > shown && l.seq < from && l.player !== seat && isShown(l)).length;
    }
    const others = frames.some((x) => x.events.some((e) => e.player !== seat && isShown(e)));
    if (!others && !missing) return undefined;
    return { base, frames, missing };
  }

  /** Le noyau fait commencer le joueur 0 : le premier joueur tiré y passe, l'ordre de jeu reste le même. */
  private rotateSeats(room: Room) {
    const n = room.seats.length;
    const first = this.firstSeat(n);
    room.seats = [...room.seats.slice(first), ...room.seats.slice(0, first)];
    for (const c of room.clients) {
      c.seat = (c.seat! - first + n) % n;
      c.send({ t: 'joined', code: room.code, token: room.seats[c.seat]!.token!, seat: c.seat });
    }
  }

  private resizeSeats(room: Room) {
    const humans = room.seats.slice(0, 2);
    const ais = Array.from({ length: room.size - 2 }, (_, k): Seat => ({
      name: AI_NAMES[k]!,
      color: '',
      kind: 'ai',
      ai: room.aiLevel,
      token: null,
      replaced: false,
    }));
    room.seats = [...humans, ...ais];
    this.fixColors(room, 0);
  }

  /** Garantit des couleurs distinctes ; le siège `keep` garde la sienne en priorité. */
  private fixColors(room: Room, keep: number) {
    const used = new Set<string>();
    const order = [keep, ...room.seats.map((_, i) => i).filter((i) => i !== keep)];
    for (const i of order) {
      const s = room.seats[i]!;
      if (!s.color || used.has(s.color)) s.color = PLAYER_COLORS.find((c) => !used.has(c.color))!.color;
      used.add(s.color);
    }
  }

  private newCode(): string {
    for (;;) {
      let code = '';
      for (let i = 0; i < CODE_LENGTH; i++) code += CODE_ALPHABET[Math.floor(Math.random() * CODE_ALPHABET.length)];
      if (!this.rooms.has(code)) return code;
    }
  }

  private touch(room: Room) {
    room.updatedAt = Date.now();
    this.archiveGame(room);
    if (!this.store) return;
    const { clients: _c, undo: _u, timer: _t, busyUntil: _b, groupOpen: _o, history: _h, plan: _p, ...data } = room;
    this.store.save(data);
  }

  /** La partie en cours, pour le labo : rejouable à l'identique, avec le temps de chaque décision d'IA. */
  private archiveGame(room: Room) {
    const g = room.game;
    if (!this.archive || !g?.record) return;
    this.archive.save({
      id: gameId(room.code, g.record.startedAt),
      code: room.code,
      updatedAt: room.updatedAt,
      players: g.players.map((p) => ({ name: p.name, kind: p.kind, ai: p.ai })),
      coins: g.phase === 'gameOver' ? g.players.map((p) => p.coins) : null,
      record: g.record,
      ...expertNets(g.record),
    });
  }

  private removeRoom(room: Room) {
    // une réflexion de l'Expert en cours s'achèvera sur une salle disparue, sans effet
    if (room.timer && room.timer !== 'expert') this.scheduler.clear(room.timer);
    this.rooms.delete(room.code);
    this.store?.remove(room.code);
  }
}
