/**
 * Le déroulement d'une partie, vu du jeu : le noyau applique les règles et renvoie des événements, que
 * ce module raconte dans le journal (en français).
 */
import { POWERS, RACES, comboName } from './data';
import { engine, kernelFor } from './engine';
import { KernelRefusal, type KernelEvent } from './kernel';
import { generateMap } from './mapgen';
import { reasonText } from './reasons';
import { scoreLine } from './rules';
import type { Action, AiLevel, GameState, LogEntry, LogKind, TurnState } from './types';

export class RuleError extends Error {}

export interface PlayerSetup {
  name: string;
  color: string;
  kind: 'human' | 'ai';
  ai: AiLevel;
}

export interface GameSetup {
  players: PlayerSetup[];
  mapSeed: number;
  seed: number;
}

/** Le journal ne garde que les dernières entrées. */
const LOG_SIZE = 500;

function freshTurn(): TurnState {
  return {
    stage: 'active',
    army: null,
    conquests: 0,
    nonEmpty: 0,
    attacked: [],
    sorcery: [],
    dragonUsed: false,
    fortressRegion: null,
    berserkDie: null,
    dieRolled: false,
    declined: false,
    picked: false,
    amazonBonus: 0,
    readied: {},
  };
}

export function createGame(setup: GameSetup): GameState {
  const n = setup.players.length;
  const map = generateMap(n, setup.mapSeed);
  const template: GameState = {
    version: 1,
    map,
    players: setup.players.map((p, i) => ({ id: i, name: p.name, color: p.color, kind: p.kind, ai: p.ai, coins: 5, active: null, declined: [], lastIncome: 0 })),
    races: {},
    regions: map.regions.map(() => ({ race: null, tokens: 0, lair: false, fortress: false, camps: 0, hero: false, dragon: false, hole: false })),
    market: [],
    raceDeck: [],
    powerDeck: [],
    round: 1,
    maxRounds: map.rounds,
    current: 0,
    phase: 'pick',
    turn: freshTurn(),
    defense: [],
    peace: [],
    log: [],
    seq: 0,
    rng: 0,
    lastDie: null,
    lastRegion: null,
  };
  const k = engine();
  const events = k.newGame(template, setup.seed);
  const g = k.snapshot();
  g.log = narrate(g, events);
  return g;
}

/** Joue l'action et renvoie le nouvel état ; `RuleError` (avec la raison) si les règles la refusent. */
export function applyAction(g0: GameState, a: Action): GameState {
  const k = kernelFor(g0);
  let events: KernelEvent[];
  try {
    events = k.apply(a);
  } catch (e) {
    if (e instanceof KernelRefusal) throw new RuleError(reasonText(g0, e.reason));
    throw e;
  }
  const g = k.snapshot();
  const log = [...g0.log, ...narrate(g, events)];
  g.log = log.length > LOG_SIZE ? log.slice(log.length - LOG_SIZE) : log;
  return g;
}

/** État juste après un passage en déclin (avant le décompte), pour les aperçus. */
export function declinePreviewState(g: GameState): GameState {
  return { ...kernelFor(g).declinePreview(), log: g.log };
}

// --- journal -----------------------------------------------------------------------------------

function plural(n: number, word: string): string {
  return `${n} ${word}${n > 1 ? 's' : ''}`;
}

/** Les entrées du journal pour ces événements ; `g` : l'état juste après (noms des joueurs et des régions). */
export function narrate(g: GameState, events: KernelEvent[]): LogEntry[] {
  const name = (pid: number) => g.players[pid]!.name;
  const regionName = (region: number) => `« ${g.map.regions[region]!.name} »`;
  const out: LogEntry[] = [];
  const log = (e: KernelEvent, player: number | null, kind: LogKind, text: string, extra: Partial<LogEntry> = {}) =>
    out.push({ seq: e.seq, round: e.round, player, kind, text, ...extra });
  for (const e of events) {
    switch (e.type) {
      case 'started':
        log(e, null, 'info', `La partie commence sur « ${g.map.name} » : ${g.map.rounds} tours, ${g.players.length} joueurs.`);
        break;
      case 'turn':
        if (e.ghouls)
          log(e, e.player, 'turn', `${name(e.player)} commence son tour avec ses Goules en déclin.`, {
            short: "joue d'abord ses Goules en déclin",
            race: 'ghouls',
            tokens: e.hand,
          });
        else
          log(e, e.player, 'turn', `${name(e.player)} commence son tour avec ${plural(e.hand, 'jeton')} ${RACES[e.race].name} en main.`, {
            short: `commence son tour · ${plural(e.hand, 'jeton')} en main`,
            race: e.race,
            power: e.power,
            tokens: e.hand,
            detail: e.amazonBonus ? `dont ${e.amazonBonus} jetons Amazones pour ce tour seulement` : 'ses régions gardent 1 jeton chacune',
          });
        break;
      case 'pick': {
        const money = [e.paid ? `paie ${plural(e.paid, 'pièce')}` : '', e.gained ? `récupère ${plural(e.gained, 'pièce')}` : ''].filter(Boolean).join(' et ');
        log(e, e.player, 'pick', `${name(e.player)} choisit les ${comboName(e.race, e.power)} (${e.tokens} jetons)${money ? ` — ${money}` : ''}.`, {
          short: `choisit les ${comboName(e.race, e.power)}`,
          detail: `${e.tokens} jetons · ${POWERS[e.power].short}`,
          race: e.race,
          power: e.power,
          tokens: e.tokens,
        });
        break;
      }
      case 'conquer': {
        const verb = e.how === 'dragon' ? 'lâche son dragon sur' : e.how === 'sorcery' ? 'ensorcelle' : e.how === 'die' ? 'conquiert grâce au dé' : 'conquiert';
        const shortVerb = e.how === 'dragon' ? 'prend avec son dragon' : e.how === 'sorcery' ? 'ensorcelle' : e.how === 'die' ? 'prend au dé' : 'prend';
        // un peuple en déclin (hors Goules) ne replace pas ses jetons : tout ce qui quitte la région est perdu
        const keeps = e.victimActive || e.victimRace === 'ghouls';
        log(e, e.player, 'conquer', `${name(e.player)} (${RACES[e.race].name}) ${verb} ${regionName(e.region)} avec ${plural(e.tokens, 'jeton')} — ${victimText(g, e)}.`, {
          region: e.region,
          victim: e.victim ?? undefined,
          short: `${shortVerb} ${regionName(e.region)}`,
          detail: victimShort(g, e),
          how: e.how,
          tokens: e.tokens,
          lost: e.victim === null || keeps ? e.lost : e.lost + e.returned,
          returned: e.victim === null || keeps ? e.returned : 0,
          victimRace: e.victimRace,
        });
        break;
      }
      case 'die':
        if (e.region === null)
          log(e, e.player, 'die', `${name(e.player)} (Berserks) lance le dé : ${e.value}. La prochaine conquête coûtera ${e.value} jeton(s) de moins.`, {
            short: `lance le dé berserk : ${e.value}`,
            detail: e.value ? `prochaine conquête : −${e.value} jeton${e.value > 1 ? 's' : ''}` : 'aucun bonus',
            die: { value: e.value, need: 0, success: null },
          });
        else
          log(e, e.player, 'die', `${name(e.player)} lance le dé de renfort pour ${regionName(e.region)} : ${e.value} — ${e.success ? 'réussite !' : 'échec.'}`, {
            region: e.region,
            short: `tente le dé sur ${regionName(e.region)} : ${e.value}`,
            detail: e.success ? 'Réussite' : `Échec (il fallait ${e.need})`,
            die: { value: e.value, need: e.need, success: e.success },
          });
        break;
      case 'abandon':
        log(e, e.player, 'abandon', `${name(e.player)} abandonne ${regionName(e.region)}.`, { region: e.region, short: `abandonne ${regionName(e.region)}` });
        break;
      case 'decline': {
        const removed = e.removed.map((r) => RACES[r].name).join(', ');
        const gone = e.removed.length ? ` Les ${removed} disparaissent de la carte.` : '';
        log(e, e.player, 'decline', `${name(e.player)} : les ${comboName(e.race, e.power)} passent en déclin.${gone}`, {
          short: `passe ses ${RACES[e.race].name} en déclin`,
          detail: e.removed.length ? `${removed} : disparaissent` : e.race === 'ghouls' ? 'les Goules gardent leurs jetons et attaqueront encore' : '1 jeton par région',
          race: e.race,
          power: e.power,
          removed: e.removed.length ? e.removed : undefined,
        });
        break;
      }
      case 'score': {
        const extras = e.lines.map(scoreLine).slice(1).map((l) => `${l.label} +${l.value}`);
        log(e, e.player, 'score', `${name(e.player)} marque ${e.gain} pièce${e.gain > 1 ? 's' : ''}${extras.length ? ` (${extras.join(', ')})` : ''}.`, {
          short: `termine son tour : +${e.gain} 💰`,
          detail: extras.length ? extras.join(', ') : undefined,
          gain: e.gain,
        });
        break;
      }
      case 'peace':
        log(e, e.player, 'power', `${name(e.player)} impose la paix à ${name(e.target)} pour son prochain tour.`, {
          victim: e.target,
          short: `impose la paix à ${name(e.target)} (Diplomates)`,
          detail: `${name(e.target)} ne pourra pas attaquer son peuple actif au prochain tour`,
          race: e.race ?? undefined,
        });
        break;
      case 'amazons':
        log(e, e.player, 'power', `${name(e.player)} : les Amazones rendent leurs ${e.tokens} jetons de conquête.`, {
          short: `rend ses ${e.tokens} jetons de conquête (Amazones)`,
          tokens: e.tokens,
          quiet: true,
        });
        break;
      case 'skeletons': {
        const raised = e.tokens > 1 ? `${e.tokens} nouveaux jetons` : '1 nouveau jeton';
        log(e, e.player, 'power', `${name(e.player)} : les Squelettes lèvent ${raised}.`, { short: `lève ${raised} (Squelettes)`, tokens: e.tokens, quiet: true });
        break;
      }
      case 'ghoulsDone':
        log(e, e.player, 'info', `${name(e.player)} : les Goules ont fini, place au peuple actif.`, { short: 'a fini avec ses Goules', quiet: true });
        break;
      case 'defended':
        log(e, e.player, 'defense', `${name(e.player)} redéploie ses ${RACES[e.race].name}.`, { short: `replace ses ${RACES[e.race].name} après l'attaque` });
        break;
      case 'over': {
        const best = [...g.players].sort((a, b) => b.coins - a.coins)[0]!;
        log(e, null, 'end', `Fin de la partie ! ${best.name} mène avec ${best.coins} pièces.`, { short: 'Fin de la partie' });
        break;
      }
    }
  }
  return out;
}

type Conquest = Extract<KernelEvent, { type: 'conquer' }>;

function victimText(g: GameState, e: Conquest): string {
  if (e.victimRace === 'lost') return 'la tribu oubliée est chassée';
  if (!e.victimRace || e.victim === null) return 'région vide';
  const loss = e.lost ? `perd ${plural(e.lost, 'jeton')}` : 'ne perd aucun jeton';
  const parts = [`${g.players[e.victim]!.name} (${RACES[e.victimRace].name}) ${loss}`];
  if (e.returned > 0) parts.push(`en récupère ${e.returned} à redéployer`);
  return parts.join(' et ');
}

/** Conséquence courte d'une conquête, pour l'affichage en direct. */
function victimShort(g: GameState, e: Conquest): string | undefined {
  if (e.victimRace === 'lost') return 'Tribu oubliée chassée';
  if (!e.victimRace || e.victim === null) return undefined;
  const who = g.players[e.victim]!.name;
  if (!e.lost) return `${who} ne perd aucun jeton (${RACES[e.victimRace].name})`;
  return `${who} perd ${plural(e.lost, 'jeton')} ${RACES[e.victimRace].name}`;
}
