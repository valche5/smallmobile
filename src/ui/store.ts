import { useEffect, useState } from 'preact/hooks';
import { applyAction, RuleError } from '../engine/game';
import { RACES } from '../engine/data';
import { currentActor } from '../engine/rules';
import type { Action, AiLevel, DieEvent, GameState, LogEntry, MapDef } from '../engine/types';
import {
  DIE_ROLL_MS,
  FACTOR,
  frameDuration,
  GRACE_MS,
  isShown,
  LATE_HEADER_MS,
  newEvents,
  REDEPLOY_MS,
  splitFrames,
  UNDO_MS,
} from '../shared/pacing';
import type { AiSpeed, ClientMsg, Replay, RoomView, ServerMsg, Snapshot } from '../shared/protocol';
import { narrate, narrateGroup, type Narration } from './narrate';
import { Connection, type NetStatus } from './net';

export type { AiSpeed } from '../shared/protocol';

/** 'rules' s'ouvre depuis le menu ☰, sans onglet. */
export type Tab = 'map' | 'market' | 'log' | 'rules';

export type Sheet =
  | { kind: 'region'; region: number }
  | { kind: 'decline' }
  | { kind: 'endTurn' }
  | { kind: 'legend' }
  | { kind: 'menu' }
  | { kind: 'people'; player: number }
  | null;

export type Overlay = { kind: 'die'; event: DieEvent; auto: boolean } | null;

export interface Toast {
  id: number;
  text: string;
  tone: 'good' | 'bad' | 'info';
}

/** Partie en ligne : chaque joueur sur son téléphone, le serveur fait foi. */
export interface OnlineState {
  code: string | null;
  token: string | null;
  /** Le siège de ce téléphone dans la partie. */
  seat: number | null;
  room: RoomView | null;
  status: NetStatus;
  map: MapDef | null;
  /** Dernier état confirmé par le serveur. */
  server: GameState | null;
  /** Actions envoyées, déjà appliquées localement, en attente de confirmation. */
  pending: { id: number; action: Action }[];
  /** Nombre d'actions annulables côté serveur. */
  undo: number;
  /** Un dé ou une annulation attend la réponse du serveur : pas d'autre action d'ici là. */
  inflight?: boolean;
}

/** La trame du tour adverse en cours d'affichage (carte d'événement). */
export interface Playback {
  id: number;
  /** 'wait' : groupe de redéploiement sans rien près de moi, montré comme une attente. */
  kind: 'event' | 'group' | 'undo' | 'header' | 'wait';
  events: LogEntry[];
  /** Durée d'affichage prévue (ms). */
  ms: number;
  /** Nombre de trames encore en attente d'affichage. */
  backlog: number;
  /** Le joueur qui agit dans cette trame. */
  actor: number | null;
  /** Ce que dit la carte (absent pour l'entête et l'attente). */
  story: Narration | null;
  /** Phase 1 du dé adverse : il roule, l'état n'est pas encore appliqué. */
  rolling: boolean;
  /** Texte de la carte d'entête (« ▶ 14 actions pendant votre absence… »). */
  header?: string;
  /** Tour auquel appartient la trame (`round:current`), pour ne pas la montrer au tour suivant. */
  turn: string;
}

/** Région de la trame en cours, mise en évidence sur la carte. */
export interface Highlight {
  region: number;
  color: string;
  /** L'action me touche : éclair rouge. */
  hit: boolean;
  /** Jetons que je perds (bulle « −N »). */
  loss: number;
  key: number;
}

/** Message bref et non bloquant affiché dans la zone d'action (« À vous ! »), à la place de l'indication. */
export interface Flash {
  id: number;
  icon: string;
  text: string;
  color: string;
  /** 'hit' : on vient d'être attaqué (fond rouge). */
  tone: 'turn' | 'hit';
}

export interface UIState {
  /** État affiché (peut être en retard sur le serveur pendant la lecture du tour adverse). */
  game: GameState | null;
  playback: Playback | null;
  /** Dernière trame montrée : la carte d'événement la garde pendant l'attente de la suivante. */
  card: Playback | null;
  /** La file est vide depuis plus de GRACE_MS : la carte d'événement passe en attente (« réfléchit… »). */
  idle: boolean;
  /** La caméra suit la région de la trame affichée. */
  follow: { region: number; n: number } | null;
  highlight: Highlight | null;
  /** Début du tour d'un adversaire : ses régions clignotent une fois. */
  blink: { regions: number[]; color: string; key: number } | null;
  /** Dernière région touchée par une action de ce téléphone (pulsation). */
  ownPulse: { region: number; key: number } | null;
  online: OnlineState | null;
  tab: Tab;
  sheet: Sheet;
  overlay: Overlay;
  toast: Toast | null;
  flash: Flash | null;
  /** L'écran de fin (podium) est affiché. */
  podium: boolean;
  aiSpeed: AiSpeed;
  showCosts: boolean;
  /** Dernier numéro de journal vu par chaque joueur humain. */
  seen: Record<number, number>;
  /** Le dernier début de tour de ce téléphone déjà annoncé (« À vous ! »). */
  handoffKey: string | null;
  /** Incrémenté pour demander à la carte de recentrer une région. */
  focus: { region: number; n: number } | null;
}

const PREFS_KEY = 'smallmobile:prefs:v1';
const ONLINE_KEY = 'smallmobile:online:v1';

function loadPrefs(): Pick<UIState, 'aiSpeed' | 'showCosts'> {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (raw) return { aiSpeed: 'normal', showCosts: true, ...JSON.parse(raw) };
  } catch {
    /* stockage indisponible */
  }
  return { aiSpeed: 'normal', showCosts: true };
}

/** Champs de lecture remis à zéro quand on quitte ou change de partie. */
const NO_PLAYBACK: Pick<UIState, 'playback' | 'card' | 'idle' | 'highlight' | 'blink' | 'follow' | 'ownPulse' | 'flash'> = {
  playback: null,
  card: null,
  idle: false,
  highlight: null,
  blink: null,
  follow: null,
  ownPulse: null,
  flash: null,
};

let state: UIState = {
  game: null,
  ...NO_PLAYBACK,
  online: null,
  tab: 'map',
  sheet: null,
  overlay: null,
  toast: null,
  podium: false,
  seen: {},
  handoffKey: null,
  focus: null,
  ...loadPrefs(),
};

const listeners = new Set<() => void>();

export function getState(): UIState {
  return state;
}

export function setState(patch: Partial<UIState>) {
  state = { ...state, ...patch };
  if ('aiSpeed' in patch || 'showCosts' in patch) {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify({ aiSpeed: state.aiSpeed, showCosts: state.showCosts }));
    } catch {
      /* ignore */
    }
  }
  listeners.forEach((l) => l());
}

export function useStore(): UIState {
  const [, setTick] = useState(0);
  useEffect(() => {
    const listener = () => setTick((t) => t + 1);
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);
  return state;
}

function patchOnline(patch: Partial<OnlineState>) {
  if (state.online) setState({ online: { ...state.online, ...patch } });
}

function pageHidden(): boolean {
  return typeof document !== 'undefined' && document.visibilityState === 'hidden';
}

// Écran verrouillé ou onglet fermé : on retient la partie en ligne, et la lecture se met en pause
if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', () => {
    if (state.online) saveSession();
  });
  document.addEventListener('visibilitychange', () => {
    if (pageHidden()) {
      if (state.online) saveSession();
      pausePlayback();
    } else {
      document.title = TITLE;
      resumePlayback();
    }
  });
}

/** Remet à zéro tout ce qui appartient à la partie affichée (file de lecture, historique, minuteurs). */
function leaveGame(): Partial<UIState> {
  resetPlayback();
  timeline = [];
  turnSignaled = null;
  flashPending = false;
  if (flashTimer) clearTimeout(flashTimer);
  flashTimer = null;
  return { ...NO_PLAYBACK };
}

/** Quitte la partie et y abandonne sa place. */
export function quitGame() {
  onlineQuit(true);
}

// --- Notifications ---------------------------------------------------------

const TITLE = 'SmallMobile';
let toastId = 0;
let toastTimer: ReturnType<typeof setTimeout> | null = null;
export function toast(text: string, tone: Toast['tone'] = 'info') {
  const id = ++toastId;
  setState({ toast: { id, text, tone } });
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    if (state.toast?.id === id) setState({ toast: null });
  }, 2600);
}

function vibrate(pattern: number | number[]) {
  try {
    // avant tout toucher sur la page, le navigateur refuse (et le signale en console)
    const nav = navigator as Navigator & { userActivation?: { hasBeenActive: boolean } };
    if (nav.userActivation && !nav.userActivation.hasBeenActive) return;
    nav.vibrate?.(pattern);
  } catch {
    /* ignore */
  }
}

/** C'est au tour de ce téléphone : on vibre (si le téléphone sait le faire), et on le signale dans l'onglet s'il est en arrière-plan. */
function notifyTurn() {
  vibrate([90, 60, 90]);
  if (pageHidden()) document.title = '🔔 À vous ! — SmallMobile';
}

function turnKey(g: GameState, actor: number): string {
  return `${actor}:${g.round}:${g.phase === 'defend' ? `d${g.seq}` : g.current}`;
}

/** Ce joueur se joue sur ce téléphone. */
function playsHere(pid: number): boolean {
  return pid === state.online?.seat;
}

/** Défense « contre soi-même » : mon peuple actif vient de prendre une région à mes Goules en déclin. */
function selfDefense(g: GameState): boolean {
  return g.phase === 'defend' && g.defense[0]?.player === g.current;
}

let turnSignaled: string | null = null;

/** Le tour revient à ce téléphone (côté serveur ou moteur) : vibration et titre d'onglet tout de suite, même si la lecture continue. */
function signalTurn(g: GameState) {
  const actor = currentActor(g);
  if (actor === null || !playsHere(actor)) return;
  const key = turnKey(g, actor);
  if (turnSignaled === key) return;
  turnSignaled = key;
  if (!selfDefense(g)) notifyTurn();
}

// --- Historique des états (relecture fidèle) ---------------------------------

/** États récents (sans journal), comme l'historique du serveur : de quoi rejouer un tour sur ses vrais états. */
let timeline: Snapshot[] = [];
const MAX_HISTORY = 400;

function remember(g: GameState) {
  const h = timeline;
  while (h.length && h[h.length - 1]!.seq > g.seq) h.pop();
  const prev = h[h.length - 1];
  const events = prev ? g.log.filter((l) => l.seq > prev.seq) : [];
  const { log: _log, map: _map, ...game } = g;
  const snap: Snapshot = { seq: g.seq, game, events };
  if (prev && !events.length && !prev.events.length) h[h.length - 1] = snap;
  else h.push(snap);
  let drop = 0;
  while (drop < h.length - 1 && (h.length - drop > MAX_HISTORY || h[drop]!.game.round < g.round - 1)) drop++;
  if (drop) h.splice(0, drop);
}

/** Reconstruit un état de l'historique avec la carte et le journal de l'état final. */
function rebuild(snap: Snapshot, final: GameState): GameState {
  return { ...snap.game, map: final.map, log: final.log.filter((l) => l.seq <= snap.seq) } as GameState;
}

// --- Lecture action par action ---------------------------------------------------

/** Une trame de la file : un état de la partie et ce qu'il faut en raconter. */
export interface Frame {
  game: GameState;
  /** Entrées montrées ensemble (une seule, ou dé + conquête). */
  events: LogEntry[];
  seq: number;
  actor: number | null;
  /** Groupe de redéploiement : `from` est l'état d'avant le premier ajustement. */
  group?: { from: GameState };
  /** Reçue page cachée, ou rejouée après une absence : ni accélération ni raccourci. */
  late?: boolean;
  /** Relecture demandée (Journal). */
  replay?: boolean;
  /** Appliquée tout de suite, sans carte (défense d'un tiers, annulation jamais vue, état final d'une relecture). */
  silent?: boolean;
  undo?: boolean;
  header?: string;
}

let frames: Frame[] = [];
/** Trame affichée (son minuteur tourne). */
let current: Frame | null = null;
/** Groupe de redéploiement qui reçoit encore les ajustements de l'adversaire. */
let openGroup: Frame | null = null;
/** Dernière trame montrée (le groupe y reste ouvert tant que rien d'autre ne passe). */
let lastShown: Frame | null = null;
/** Pas de démarrage de trame pendant qu'on remplit la file (relecture). */
let holding = false;
let playbackId = 0;

/** Minuteurs de la trame affichée : mis en pause quand la page est cachée. */
interface FrameTimer {
  fn: () => void;
  at: number;
  left: number;
  h: ReturnType<typeof setTimeout> | null;
}
let timers: FrameTimer[] = [];
let paused = pageHidden();
let graceTimer: ReturnType<typeof setTimeout> | null = null;
let mirrorTimer: ReturnType<typeof setTimeout> | null = null;
let mirrorAt = 0;

function frameAfter(ms: number, fn: () => void) {
  const t: FrameTimer = { fn, at: Date.now() + ms, left: ms, h: null };
  const run = () => {
    timers = timers.filter((x) => x !== t);
    fn();
  };
  t.fn = run;
  if (!paused) t.h = setTimeout(run, ms);
  timers.push(t);
}

function clearFrameTimers() {
  for (const t of timers) if (t.h) clearTimeout(t.h);
  timers = [];
}

function playbackSpeed(): AiSpeed {
  return state.online?.room?.speed ?? state.aiSpeed;
}

function speedFactor(): number {
  return FACTOR[playbackSpeed()];
}

/** Le joueur pour qui on raconte (« vous ») : le siège de ce téléphone. */
function meSeat(): number | null {
  return viewerSeat();
}

function turnOf(g: GameState): string {
  return `${g.round}:${g.current}`;
}

/** Ajoute une trame ; la lecture démarre si rien n'est affiché. */
function pushFrame(f: Frame) {
  frames.push(f);
  if (holding) return;
  if (!current) showNextFrame();
  else if (state.playback) setState({ playback: { ...state.playback, backlog: backlog() } });
}

function backlog(): number {
  return frames.filter((f) => !f.silent).length;
}

/**
 * Un nouvel état produit par un autre joueur (ou une IA), comparé au dernier état reçu (`before`) :
 * une trame par entrée de journal, montrée sur l'état réel de la carte à ce moment-là.
 */
function enqueueState(game: GameState, before: GameState | null, opts: { late?: boolean; replay?: boolean; events?: LogEntry[] } = {}) {
  const late = opts.late || pageHidden() || undefined;
  const replay = opts.replay || undefined;
  const viewer = meSeat();
  // annulation adverse : ce qu'on n'a pas encore vu disparaît sans bruit, ce qu'on a vu est « annulé »
  if (before && game.seq < before.seq) {
    frames = frames.filter((f) => f.header || f.seq <= game.seq);
    if (openGroup && openGroup.seq > game.seq) openGroup = null;
    const seen = (state.game?.seq ?? -1) > game.seq;
    pushFrame({ game, events: [], seq: game.seq, actor: currentActor(before), undo: seen || undefined, silent: !seen || undefined, late });
    if (seen) openGroup = null;
    return;
  }
  // mes propres entrées ne sont jamais racontées (mon début de tour est annoncé par le flash)
  const events = (opts.events ?? newEvents(before, game)).filter((e) => viewer === null || e.player !== viewer);
  const split = splitFrames(events);
  const actor = currentActor(game);
  if (!split.length) {
    if (game.phase === 'redeploy' && actor !== null && actor !== viewer) {
      const g = openGroup;
      const last = frames[frames.length - 1];
      // le groupe en cours absorbe les ajustements suivants, tant que rien d'autre ne s'est intercalé
      if (g && g.actor === actor && (last === g || (!last && (current === g || (!current && lastShown === g))))) {
        g.game = game;
        g.seq = game.seq;
        if (current === g || (!current && lastShown === g)) mirror(g);
        return;
      }
      const f: Frame = { game, events: [], seq: game.seq, actor, group: { from: before ?? game }, late, replay };
      openGroup = f;
      pushFrame(f);
      return;
    }
    pushFrame({ game, events: [], seq: game.seq, actor, silent: true, late });
    return;
  }
  openGroup = null;
  for (const ev of split) pushFrame({ game, events: ev, seq: game.seq, actor: ev[0]!.player, late, replay });
}

/** Le groupe affiché suit en direct les ajustements de l'adversaire (au plus une mise à jour toutes les 300 ms). */
function mirror(g: Frame) {
  if (mirrorTimer) return;
  const run = () => {
    mirrorTimer = null;
    mirrorAt = Date.now();
    // le groupe a pu être fermé entre-temps par une action : on montre quand même son dernier état s'il est encore à l'écran
    if (current !== g && !(current === null && lastShown === g)) return;
    const story = narrateGroup(g.group!.from, g.game, g.actor, meSeat());
    const pb = (state.playback?.kind === 'group' || state.playback?.kind === 'wait') && current === g ? state.playback : null;
    const patch: Partial<UIState> = { game: g.game, idle: false };
    const shown = pb ?? state.card;
    if (shown) {
      const next: Playback = { ...shown, kind: story ? 'group' : 'wait', story: story ?? (shown.kind === 'wait' ? shown.story : null) };
      if (pb) patch.playback = next;
      patch.card = next;
    }
    setState(patch);
    if (!current) startGrace();
  };
  const wait = 300 - (Date.now() - mirrorAt);
  if (wait <= 0) run();
  else mirrorTimer = setTimeout(run, wait);
}

function startGrace() {
  if (graceTimer) clearTimeout(graceTimer);
  graceTimer = setTimeout(() => {
    graceTimer = null;
    if (!current) setState({ idle: true });
  }, GRACE_MS);
}

/** Affiche la trame suivante de la file (ou termine la lecture). */
function showNextFrame() {
  clearFrameTimers();
  // la trame qu'on quitte laisse la carte dans son état (dé encore en train de rouler, entête…)
  if (current && !current.header && state.game !== current.game) setState({ game: current.game });
  current = null;
  if (paused) return;
  for (;;) {
    const f = frames.shift();
    if (!f) return endPlayback();
    if (f.silent) {
      if (state.game !== f.game) setState({ game: f.game });
      lastShown = f;
      continue;
    }
    display(f);
    return;
  }
}

function endPlayback() {
  current = null;
  const patch: Partial<UIState> = { playback: null };
  // la dernière trame reste lisible : elle ne passe en attente qu'après le délai de grâce
  setState(patch);
  startGrace();
  if (!state.overlay) checkHandoff();
}

function display(f: Frame) {
  current = f;
  lastShown = f;
  if (graceTimer) clearTimeout(graceTimer);
  graceTimer = null;
  const k = speedFactor();
  const viewer = meSeat();
  const id = ++playbackId;
  const patch: Partial<UIState> = { idle: false, ownPulse: null, blink: null };
  let kind: Playback['kind'] = 'event';
  let story: Narration | null = null;
  let ms: number;
  let rolling = false;
  if (f.header) {
    kind = 'header';
    ms = LATE_HEADER_MS;
  } else if (f.undo) {
    kind = 'undo';
    ms = UNDO_MS * k;
    story = { who: f.actor, main: '↶ annule sa dernière action', tone: 'neutral' };
    patch.game = f.game;
    patch.highlight = null;
  } else if (f.group) {
    story = narrateGroup(f.group.from, f.game, f.actor, viewer);
    kind = story ? 'group' : 'wait';
    // rien près de moi : la carte précédente du même joueur reste, en attente (« · redéploie… »)
    if (!story && state.card?.actor === f.actor && state.card.turn === turnOf(f.game)) story = state.card.story;
    ms = REDEPLOY_MS * k;
    patch.game = f.game;
    patch.highlight = null;
  } else {
    story = narrate(f.game, f.events, viewer);
    ms = frameDuration(f.events, playbackSpeed());
    const hit = story.tone === 'hit';
    // en retard sur la partie en direct : on rattrape un peu, jamais sur ce qui me touche
    if (!f.late && !f.replay && !hit && ms > 0) {
      const waiting = frames.filter((x) => !x.late && !x.replay && !x.silent && !x.header).length;
      const speedup = waiting > 8 ? 0.5 : waiting > 3 ? 0.7 : 1;
      if (speedup < 1) ms = Math.max(Math.min(ms, 700), ms * speedup);
    }
    const color = f.actor !== null ? (f.game.players[f.actor]?.color ?? '#ffe066') : '#ffe066';
    const region = story.region;
    const camera = region !== undefined && f.events.some((e) => e.kind === 'conquer' || e.kind === 'die' || e.kind === 'abandon');
    if (camera) patch.follow = { region: region!, n: (state.follow?.n ?? 0) + 1 };
    const conquest = f.events.some((e) => e.kind === 'conquer');
    const turn = f.events[0]!;
    if (turn.kind === 'turn' && turn.player !== null && !(turn.race === 'ghouls' && f.game.races.ghouls?.declined)) {
      // début de son tour : ses régions clignotent (elles retombent à 1 jeton)
      const regions = f.game.regions.flatMap((r, i) => (r.race && r.race !== 'lost' && f.game.races[r.race]?.owner === turn.player ? [i] : []));
      if (regions.length) patch.blink = { regions, color, key: id };
    }
    const after: Partial<UIState> = {
      game: f.game,
      highlight: region !== undefined ? { region, color, hit: hit && conquest, loss: hit && conquest ? (story.loss ?? 0) : 0, key: id } : null,
    };
    if (story.dieRoll) {
      // phase 1 : le dé roule sur l'état d'avant ; phase 2 : verdict et nouvel état
      rolling = true;
      patch.highlight = region !== undefined ? { region, color, hit: false, loss: 0, key: -id } : null;
      const s = story;
      frameAfter(Math.min(DIE_ROLL_MS * k, ms / 2), () => {
        const pb = state.playback?.id === id ? { ...state.playback, rolling: false } : state.playback;
        setState({ ...after, playback: pb, card: pb });
        if (s.tone === 'hit' && conquest) vibrate([40, 50, 40]);
      });
    } else {
      Object.assign(patch, after);
      if (hit && conquest) vibrate([40, 50, 40]);
    }
  }
  if (f.header) patch.highlight = null;
  const pb: Playback = { id, kind, events: f.events, ms, backlog: backlog(), actor: f.actor, story, rolling, header: f.header, turn: turnOf(f.game) };
  patch.playback = pb;
  if (!f.header) patch.card = pb;
  setState(patch);
  if (state.online) saveSession();
  frameAfter(ms, showNextFrame);
}

function pausePlayback() {
  if (paused) return;
  paused = true;
  const now = Date.now();
  for (const t of timers) {
    if (t.h) clearTimeout(t.h);
    t.h = null;
    t.left = Math.max(0, t.at - now);
  }
}

/** Retour sur la page : entête si l'on a manqué au moins 3 actions, puis lecture sans accélération. */
function resumePlayback() {
  if (!paused) return;
  paused = false;
  const late = frames.filter((f) => f.late && f.events.length).length;
  if (late >= 3 && !frames.some((f) => f.header)) {
    frames.unshift(headerFrame(`▶ ${late} actions pendant votre absence · ⏩ pour passer`));
    // la trame en cours a déjà été vue : on passe directement à l'entête
    if (current) {
      clearFrameTimers();
      if (!current.header && state.game !== current.game) setState({ game: current.game });
      current = null;
    }
  }
  if (current) {
    const now = Date.now();
    for (const t of timers) {
      t.at = now + t.left;
      t.h = setTimeout(t.fn, t.left);
    }
  } else if (frames.length) showNextFrame();
  else if (flashPending) checkHandoff();
}

function headerFrame(text: string): Frame {
  const g = state.game!;
  return { game: g, events: [], seq: -1, actor: null, header: text };
}

/** Passe à la trame suivante (toucher la carte d'événement). */
export function nextFrame() {
  if (current) showNextFrame();
}

/** Saute la lecture : affiche directement le dernier état connu et rend la main. */
export function skipPlayback() {
  const last = [...frames].reverse().find((f) => !f.header)?.game ?? current?.game ?? null;
  resetPlayback();
  // la dernière carte montrée ne correspond plus à la carte de jeu : on n'en garde rien
  const patch: Partial<UIState> = { playback: null, highlight: null, card: null };
  if (last && last !== state.game) patch.game = last;
  if (state.overlay?.kind === 'die' && state.overlay.auto) patch.overlay = null;
  setState(patch);
  startGrace();
  if (!state.overlay) checkHandoff();
}

function resetPlayback() {
  frames = [];
  current = null;
  openGroup = null;
  lastShown = null;
  holding = false;
  clearFrameTimers();
  if (graceTimer) clearTimeout(graceTimer);
  graceTimer = null;
  if (mirrorTimer) clearTimeout(mirrorTimer);
  mirrorTimer = null;
}

/** La lecture est-elle en cours (actions adverses pas encore toutes affichées) ? */
export function playbackBusy(): boolean {
  return current !== null || frames.length > 0 || mirrorTimer !== null;
}

/** File de lecture (pour les tests et le débogage). */
export function playbackQueue(): readonly Frame[] {
  return frames;
}

/** Rejoue des états passés sur leurs vrais états intermédiaires (`late` : retour d'absence, sinon relecture demandée). */
function replayStates(
  base: GameState | null,
  snaps: { game: GameState; events: LogEntry[] }[],
  final: GameState,
  opts: { late?: boolean; header: (n: number) => string | null },
) {
  resetPlayback();
  holding = true;
  let prev = base;
  if (base && base !== state.game) setState({ game: base });
  // rien à l'écran (page rechargée) et plus d'état de départ : on part du premier état rejoué
  else if (!base && !state.game && snaps[0]) setState({ game: snaps[0].game });
  for (const s of snaps) {
    enqueueState(s.game, prev, { events: s.events, late: opts.late, replay: !opts.late });
    prev = s.game;
  }
  frames.push({ game: final, events: [], seq: final.seq, actor: currentActor(final), silent: true });
  // même compte que le « ⏩ N » de la carte d'événement
  const n = frames.filter((f) => !f.silent && !f.header).length;
  const header = opts.header(n);
  if (header) frames.unshift(headerFrame(header));
  holding = false;
  showNextFrame();
}

/** Rejoue, action par action, ce qui s'est passé depuis ce numéro de journal (« ▶ Revoir »). */
export function replaySince(seq: number) {
  const g = state.game;
  if (!g || playbackBusy()) return;
  const viewer = meSeat();
  const after = timeline.filter((x) => x.seq > seq);
  if (!after.some((x) => x.events.some((e) => e.player !== viewer && isShown(e)))) return;
  let base: Snapshot | null = null;
  for (const x of timeline) if (x.seq <= seq) base = x;
  const first = timeline[0]!;
  const from = first.events[0]?.seq ?? first.seq + 1;
  const missing = base ? 0 : g.log.filter((l) => l.seq > seq && l.seq < from && l.player !== viewer && isShown(l)).length;
  replayStates(
    base ? rebuild(base, g) : null,
    after.map((x) => ({ game: rebuild(x, g), events: x.events })),
    g,
    {
      header: (n) =>
        n >= 3 || missing ? `▶ ${n} actions depuis votre dernier tour · ${missing ? `${missing} plus anciennes dans le Journal` : '⏩ pour passer'}` : null,
    },
  );
}

/** Alias du nom employé par la spécification. */
export const replayFrom = replaySince;

// --- Qui peut jouer ? ---------------------------------------------------------

/** Ce téléphone peut-il agir maintenant ? */
export function canAct(s: UIState = state): boolean {
  const g = s.game;
  if (!g || s.playback || playbackBusy()) return false;
  const actor = currentActor(g);
  if (actor === null || !s.online) return false;
  return actor === s.online.seat && s.online.status === 'open' && !s.online.inflight;
}

/** « Moi » : le siège de ce téléphone dans la partie. */
export function viewerSeat(s: UIState = state): number | null {
  return s.online?.seat ?? null;
}

export function canUndo(s: UIState = state): boolean {
  if (!canAct(s) || !s.online) return false;
  return s.online.undo > 0 && s.online.pending.length === 0;
}

// --- Actions ---------------------------------------------------------------

/** Actions qui tirent un dé : seul le serveur peut les résoudre. */
const RANDOM = new Set<Action['type']>(['reinforce', 'berserkRoll']);
const CLOSES_REGION_SHEET = new Set<Action['type']>(['conquer', 'dragon', 'sorcery', 'reinforce', 'abandon', 'pick', 'decline', 'endTurn', 'endDefense']);

function sheetAfter(a: Action): Partial<UIState> {
  // Après une action, on referme les fenêtres de confirmation
  if (state.sheet && state.sheet.kind !== 'region') return { sheet: null };
  if (state.sheet?.kind === 'region' && CLOSES_REGION_SHEET.has(a.type)) return { sheet: null };
  return {};
}

/** Ma propre action a touché une région : elle pulse (les actions adverses ont leur surbrillance). */
function pulseAfter(prev: GameState, next: GameState): Partial<UIState> {
  if (next.lastRegion === null || next.lastRegion === prev.lastRegion) return {};
  return { ownPulse: { region: next.lastRegion, key: next.seq }, highlight: null };
}

/** Joue une action de ce téléphone (affichée tout de suite, confirmée par le serveur). */
export function act(a: Action): boolean {
  return actOnline(a);
}

/** Ma propre action a aussi fait commencer le tour de l'adversaire : sa trame « ▶ commence son tour » passe par la file. */
function showOthers(before: GameState, after: GameState, me: number | null) {
  const viewer = meSeat();
  // la fin de partie se montre aussi quand c'est moi qui la déclenche
  const split = splitFrames(newEvents(before, after).filter((e) => e.kind === 'end' || (e.player !== me && e.player !== viewer && e.player !== null)));
  for (const ev of split) pushFrame({ game: after, events: ev, seq: after.seq, actor: ev[0]!.player });
}

let actionSeq = 0;

function actOnline(a: Action): boolean {
  const o = state.online;
  const g = state.game;
  if (!o || !g || !canAct()) return false;
  const id = ++actionSeq;
  if (RANDOM.has(a.type)) {
    // le dé est lancé par le serveur : on attend sa réponse
    if (!conn().send({ t: 'action', id, action: a })) return lostConnection();
    // rien ne change à l'écran avant la réponse : un 2e toucher ne doit pas relancer le dé
    setState({ ...sheetAfter(a), online: { ...o, inflight: true } });
    return true;
  }
  let next: GameState;
  try {
    next = applyAction(g, a);
  } catch (e) {
    toast(e instanceof RuleError ? e.message : 'Erreur inattendue.', 'bad');
    return false;
  }
  if (!conn().send({ t: 'action', id, action: a })) return lostConnection();
  // affichage immédiat, confirmé (ou corrigé) par le serveur ensuite
  setState({ game: next, online: { ...o, pending: [...o.pending, { id, action: a }] }, ...sheetAfter(a), ...pulseAfter(g, next) });
  feedback(a);
  if (!state.overlay) checkHandoff();
  return true;
}

function lostConnection(): false {
  toast('Connexion au serveur perdue : nouvelle tentative…', 'bad');
  return false;
}

/** Retour discret après une action : la carte et la barre du haut montrent déjà le résultat. */
function feedback(a: Action) {
  if (a.type === 'conquer' || a.type === 'dragon' || a.type === 'sorcery') vibrate(15);
  if (a.type === 'pick') setState({ tab: 'map' });
}

export function undoLast() {
  const o = state.online;
  if (!o) return;
  // un 2e ↶ avant la réponse annulerait une action de plus
  if (canUndo() && conn().send({ t: 'undo' })) setState({ sheet: null, online: { ...o, inflight: true } });
  else setState({ sheet: null });
}

export function dismissOverlay() {
  setState({ overlay: null });
  checkHandoff();
}

let flashSeq = 0;
let flashTimer: ReturnType<typeof setTimeout> | null = null;
/** Le tour m'est revenu page cachée : le flash attend le retour sur la page (et la fin de la lecture). */
let flashPending = false;

function showFlash(b: Omit<Flash, 'id'>) {
  const id = ++flashSeq;
  if (flashTimer) clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {
    if (state.flash?.id === id) setState({ flash: null });
  }, 1500);
  setState({ flash: { ...b, id } });
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n > 1 ? 's' : ''}`;
}

/** Les régions qu'on m'a prises depuis mon dernier passage : « Bob vous a pris 2 régions ». */
function takenSince(g: GameState, me: number, since: number): string | null {
  const hits = g.log.filter((l) => l.seq > since && l.kind === 'conquer' && l.victim === me && l.player !== me && l.player !== null);
  if (!hits.length) return null;
  const regions = new Set(hits.map((h) => h.region)).size;
  const attackers = new Set(hits.map((h) => h.player));
  const who = attackers.size === 1 ? `${g.players[hits[0]!.player!]!.name} vous a pris` : 'On vous a pris';
  return `${who} ${plural(regions, 'région')}`;
}

/** Quand la main revient à ce téléphone : le flash « À vous ! ». */
export function checkHandoff() {
  const g = state.game;
  if (!g || state.overlay || playbackBusy()) return;
  const actor = currentActor(g);
  if (actor === null) return;
  const defense = g.phase === 'defend';
  const key = turnKey(g, actor);
  if (state.handoffKey === key) return;
  if (!playsHere(actor)) return;
  signalTurn(g);
  // page cachée : le flash attendra qu'on revienne
  if (pageHidden()) {
    flashPending = true;
    return;
  }
  flashPending = false;
  const p = g.players[actor]!;
  const since = state.seen[actor] ?? 0;
  if (graceTimer) clearTimeout(graceTimer);
  graceTimer = null;
  setState({ handoffKey: key, seen: { ...state.seen, [actor]: g.seq }, card: null, highlight: null, idle: false });
  saveSession();
  // les fenêtres de confirmation ouvertes pendant le tour adverse n'ont plus de sens
  if (state.sheet && state.sheet.kind !== 'region' && state.sheet.kind !== 'people') setState({ sheet: null });
  const task = defense ? g.defense[0] : null;
  const taken = takenSince(g, actor, since);
  if (task && selfDefense(g)) {
    // mon peuple actif a pris une région à mes Goules : rien d'alarmant
    const hand = g.races[task.race]?.hand ?? 0;
    showFlash({ icon: '🛡️', text: `Replacez ${plural(hand, 'jeton')} ${RACES[task.race].name}`, color: p.color, tone: 'turn' });
  } else if (task) {
    const text = taken ?? `Replacez vos ${RACES[task.race].name}`;
    showFlash({ icon: '🛡️', text, color: p.color, tone: 'hit' });
  } else if (g.phase === 'pick') showFlash({ icon: '👑', text: 'À vous ! Choisissez un peuple', color: p.color, tone: 'turn' });
  else if (g.turn.stage === 'ghouls') showFlash({ icon: '🧟', text: "À vous ! Vos Goules d'abord", color: p.color, tone: 'turn' });
  else if (taken) showFlash({ icon: '🔔', text: `À vous ! ${taken}`, color: p.color, tone: 'turn' });
  else if (g.round === g.maxRounds) showFlash({ icon: '🔔', text: 'À vous ! Dernier tour', color: p.color, tone: 'turn' });
  else showFlash({ icon: '🔔', text: 'À vous !', color: p.color, tone: 'turn' });
}

/** Recentre la carte sur une région. */
export function showRegion(region: number) {
  setState({ follow: { region, n: (state.follow?.n ?? 0) + 1 } });
}

export function openRegion(region: number | null) {
  if (region === null) setState({ sheet: null });
  else setState({ sheet: { kind: 'region', region }, focus: { region, n: (state.focus?.n ?? 0) + 1 } });
}

// --- En ligne -------------------------------------------------------------------

interface Session {
  code: string;
  token: string;
  seat: number;
  seen?: Record<number, number>;
  /** Numéro de journal de la dernière trame affichée sur ce téléphone. */
  shown?: number;
}

export function savedSession(): Session | null {
  try {
    return JSON.parse(localStorage.getItem(ONLINE_KEY) ?? 'null') as Session | null;
  } catch {
    return null;
  }
}

function saveSession() {
  const o = state.online;
  if (!o?.code || !o.token || o.seat === null) return;
  try {
    const shown = state.game?.seq ?? savedSession()?.shown;
    localStorage.setItem(ONLINE_KEY, JSON.stringify({ code: o.code, token: o.token, seat: o.seat, seen: state.seen, shown }));
  } catch {
    /* ignore */
  }
}

export function forgetSession() {
  try {
    localStorage.removeItem(ONLINE_KEY);
  } catch {
    /* ignore */
  }
}

let connection: Connection | null = null;
/** À la reprise d'une partie en ligne : dernier numéro de journal affiché avant de quitter. */
let resumeShown: number | null = null;
/** Message d'entrée (créer / rejoindre) à envoyer dès que la connexion est ouverte. */
let intro: ClientMsg | null = null;

function conn(): Connection {
  connection ??= new Connection({
    status: (status) => patchOnline({ status }),
    open: () => {
      const o = state.online;
      if (!o) return;
      if (o.code && o.token) {
        // la dernière trame affichée : le serveur nous rejoue la suite sur ses vrais états
        const shown = state.game?.seq ?? resumeShown;
        conn().send({ t: 'resume', code: o.code, token: o.token, ...(shown !== null && shown !== undefined ? { shown } : {}) });
      } else if (intro) conn().send(intro);
    },
    message: onMessage,
  });
  return connection;
}

function enterOnline(patch: Partial<OnlineState> = {}) {
  setState({
    ...leaveGame(),
    online: {
      code: null,
      token: null,
      seat: null,
      room: null,
      status: 'connecting',
      map: null,
      server: null,
      pending: [],
      undo: 0,
      ...patch,
    },
    game: null,
    sheet: null,
    overlay: null,
    handoffKey: null,
    tab: 'map',
  });
  conn().connect();
}

export function onlineCreate(p: { name: string; color: string; size: number; aiLevel: AiLevel; mapSeed: number; opponent: 'human' | 'ai' }) {
  intro = { t: 'create', ...p };
  resumeShown = null;
  setState({ seen: {} });
  enterOnline();
}

export function onlineJoin(code: string, name: string, color: string) {
  intro = { t: 'join', code, name, color };
  resumeShown = null;
  setState({ seen: {} });
  enterOnline();
}

const HOME_KEY = 'smallmobile:accueil';

/** Le joueur est revenu volontairement à l'accueil : pas de reconnexion automatique dans cet onglet. */
export function wantsHome(): boolean {
  try {
    return sessionStorage.getItem(HOME_KEY) === '1';
  } catch {
    return false;
  }
}

function setWantsHome(v: boolean) {
  try {
    if (v) sessionStorage.setItem(HOME_KEY, '1');
    else sessionStorage.removeItem(HOME_KEY);
  } catch {
    /* ignore */
  }
}

export function onlineResume() {
  const s = savedSession();
  if (!s) return;
  setWantsHome(false);
  intro = null;
  setState({ seen: s.seen ?? {} });
  resumeShown = s.shown ?? null;
  enterOnline({ code: s.code, token: s.token, seat: s.seat });
}

/** Quitte l'écran de partie en ligne. `forget` : abandonne définitivement sa place. */
export function onlineQuit(forget: boolean) {
  if (forget) {
    conn().send({ t: 'leave' });
    forgetSession();
  } else {
    saveSession();
    setWantsHome(true);
  }
  intro = null;
  connection?.close();
  setState({ ...leaveGame(), online: null, game: null, sheet: null, overlay: null, handoffKey: null });
}

export function onlineSend(msg: ClientMsg) {
  if (!conn().send(msg)) lostConnection();
}

const SPEED_LABEL: Record<AiSpeed, string> = { lent: '🐢 Lente', normal: '🚶 Normale', rapide: '⚡ Rapide' };

/** Message du serveur (exporté pour les tests). */
export function onMessage(msg: ServerMsg) {
  const o = state.online;
  if (!o) return;
  switch (msg.t) {
    case 'joined':
      intro = null;
      patchOnline({ code: msg.code, token: msg.token, seat: msg.seat });
      saveSession();
      if (new URLSearchParams(location.search).has('partie')) history_replace();
      break;
    case 'room': {
      const prev = o.room;
      const by = msg.room.speedBy;
      // un autre joueur a changé la vitesse de la table : on le dit, c'est la seule annonce de ce genre
      if (prev && prev.speed !== msg.room.speed && by !== null && by !== undefined && by !== o.seat) {
        toast(`⚙️ ${msg.room.seats[by]?.name ?? '?'} : vitesse ${SPEED_LABEL[msg.room.speed]}`, 'info');
      }
      patchOnline({ room: msg.room });
      break;
    }
    case 'map':
      patchOnline({ map: msg.map });
      break;
    case 'state':
      receiveState(msg);
      break;
    case 'error':
      toast(msg.message, 'bad');
      if (msg.fatal) {
        forgetSession();
        intro = null;
        connection?.close();
        setState({ ...leaveGame(), online: null, game: null, sheet: null, overlay: null, handoffKey: null });
      } else if (o.server && o.pending.length) {
        setState({ game: o.server, online: { ...o, pending: [], inflight: false } });
      } else if (o.inflight) setState({ online: { ...o, inflight: false } });
      break;
    case 'left':
      break;
  }
}

function history_replace() {
  try {
    window.history.replaceState(null, '', location.pathname);
  } catch {
    /* ignore */
  }
}

function receiveState(msg: Extract<ServerMsg, { t: 'state' }>) {
  const o = state.online!;
  if (!o.map) return;
  const server = { ...msg.game, map: o.map } as GameState;
  const before = o.server;
  const actor = before ? currentActor(before) : null;
  const mine = (msg.ack !== null && msg.ack.seat === o.seat) || (actor === o.seat && !playbackBusy());
  signalTurn(server);

  if (!msg.sync && before && !mine) {
    // action d'un adversaire (ou d'une IA) : affichée à son tour, une trame à la fois
    remember(server);
    setState({ online: { ...o, server, pending: [], undo: msg.undo, inflight: false } });
    enqueueState(server, before);
    return;
  }

  // notre propre action, ou une resynchronisation : affichage immédiat (sauf ce qu'on a manqué, rejoué d'abord)
  // une resynchronisation fait foi : une action non confirmée a été soit appliquée (elle est dans cet état),
  // soit perdue avec la connexion ; la rejouer par-dessus la doublerait ou bloquerait la partie
  let pending = msg.sync ? [] : o.pending;
  if (msg.ack && msg.ack.seat === o.seat) {
    const acked = msg.ack.id;
    pending = pending.filter((p) => p.id > acked);
  }
  // on rejoue par-dessus l'état du serveur les actions pas encore confirmées
  let game = server;
  for (const p of pending) {
    try {
      game = applyAction(game, p.action);
    } catch {
      pending = [];
      game = server;
      break;
    }
  }
  const prev = state.game;
  resumeShown = null;
  if (msg.sync && msg.replay && msg.replay.frames.length) {
    setState({ online: { ...o, server, pending, undo: msg.undo, inflight: false }, ...(prev ? {} : { tab: 'map' as Tab }) });
    replayMissed(msg.replay, server, game);
    return;
  }
  resetPlayback();
  remember(server);
  const patch: Partial<UIState> = { game, playback: null, online: { ...o, server, pending, undo: msg.undo, inflight: false } };
  if (prev && !msg.sync && game.lastDie && game.lastDie.seq !== prev.lastDie?.seq && game.lastDie.player === o.seat) {
    // mon propre lancer : plein écran ; ceux des autres passent par la carte d'événement
    patch.overlay = { kind: 'die', event: game.lastDie, auto: false };
  } else if (state.overlay?.kind === 'die' && state.overlay.auto) patch.overlay = null;
  if (prev && !msg.sync) Object.assign(patch, pulseAfter(prev, game));
  if (!prev) patch.tab = 'map';
  setState(patch);
  if (before && !msg.sync && msg.ack?.seat === o.seat) showOthers(before, game, o.seat);
  if (!state.overlay) checkHandoff();
}

/** Reconnexion : on rejoue ce que ce téléphone n'a pas vu, sur les états gardés par le serveur. */
function replayMissed(replay: Replay, server: GameState, final: GameState) {
  timeline = [];
  const snaps = replay.frames.map((s) => ({ game: rebuild(s, server), events: s.events }));
  if (replay.base) remember(rebuild(replay.base, server));
  for (const s of snaps) remember(s.game);
  remember(server);
  const missing = replay.missing;
  replayStates(replay.base ? rebuild(replay.base, server) : null, snaps, final, {
    late: true,
    header: (n) =>
      n >= 3 || missing ? `▶ ${n} actions pendant votre absence · ${missing ? `${missing} plus anciennes dans le Journal` : '⏩ pour passer'}` : null,
  });
}
