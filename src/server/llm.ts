/**
 * Parties pour un LLM (ou tout programme) par une API HTTP simple, en JSON : un joueur, des IA du jeu
 * en face (l'Expert par défaut). Chaque requête du joueur fait jouer les IA jusqu'à son tour suivant ;
 * la réponse décrit la partie en texte, avec la liste des coups permis. Règles et mode d'emploi :
 * `GET /llm.md` (src/server/llm.md).
 *
 * Le joueur voit ce qu'un humain attentif saurait : la carte, le marché, les pièces de chacun (elles se
 * reconstituent à partir des gains annoncés), jamais l'ordre caché des pioches ni les dés à venir.
 * Les parties sont gardées dans `data/llm/` (une par fichier), à part des vraies parties du jeu.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { aiAction } from '../engine/ai';
import { AI_NAMES, PLAYER_COLORS, POWER_IDS, POWERS, RACE_IDS, RACES, TERRAINS } from '../engine/data';
import { applyAction, createGame, RuleError } from '../engine/game';
import { kernelFor } from '../engine/engine';
import { KernelRefusal } from '../engine/kernel';
import { randomSeed } from '../engine/rng';
import { allTargets, currentActor, ranking, regionsOf, tokensOnBoard } from '../engine/rules';
import type { Action, AiLevel, AiThought, GameState, RaceId } from '../engine/types';
import type { Expert } from './expert';

const LEVELS: AiLevel[] = ['facile', 'normal', 'difficile', 'expert'];
const PHASES: Record<GameState['phase'], string> = {
  pick: 'choix d’un peuple',
  conquest: 'conquêtes',
  redeploy: 'redéploiement',
  defend: 'défense (replacer les jetons rendus après une attaque)',
  gameOver: 'partie finie',
};
/** Au-delà, une partie qui n'avance plus est déclarée bloquée. */
const MAX_AI_ACTIONS = 5000;

export interface LlmGame {
  id: string;
  createdAt: number;
  updatedAt: number;
  /** La place du joueur (0 = premier à jouer). */
  seat: number;
  game: GameState;
  /** Dernière entrée du journal avant le dernier coup du joueur : la vue raconte la suite. */
  seen: number;
}

/** Ce que reçoit le joueur après chaque requête. */
export interface LlmView {
  id: string;
  seat: number;
  round: number;
  maxRounds: number;
  phase: GameState['phase'];
  yourTurn: boolean;
  gameOver: boolean;
  /** Ce qui s'est passé depuis le dernier coup du joueur, ce coup compris (dés, adversaires). */
  events: string[];
  /** Toute la situation en texte (Markdown), coups permis compris. */
  text: string;
  /** Les coups permis, à renvoyer tels quels dans `POST /api/llm/games/:id/actions`. */
  legal: { action: Action; note: string }[];
  /** Fin de partie : pièces de chacun, du premier au dernier. */
  ranking?: { seat: number; name: string; coins: number }[];
}

/** Une erreur à renvoyer au client, avec son code HTTP. */
export class LlmError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export interface LlmOptions {
  expert: Expert | null;
  /** Dossier des parties ; `null` : en mémoire seulement (tests). */
  dir: string | null;
}

export class LlmGames {
  private games = new Map<string, LlmGame>();
  /** Parties dont une requête est en cours (les IA réfléchissent). */
  private busy = new Set<string>();

  constructor(private readonly opts: LlmOptions) {
    if (opts.dir) mkdirSync(opts.dir, { recursive: true });
  }

  /**
   * Nouvelle partie. `players` : 2 à 5 ; `opponents` : un niveau pour tous, ou un par adversaire
   * (facile, normal, difficile, expert) ; `seat` : la place du joueur (au hasard sinon) ; `seed` :
   * pour rejouer la même carte et le même hasard.
   */
  async create(body: unknown): Promise<LlmView> {
    const b = (body ?? {}) as Record<string, unknown>;
    const players = b.players === undefined ? 2 : Number(b.players);
    if (!Number.isInteger(players) || players < 2 || players > 5) throw new LlmError(400, '`players` : de 2 à 5.');
    const raw = b.opponents ?? 'expert';
    const opponents = Array.isArray(raw) ? raw : Array.from({ length: players - 1 }, () => raw);
    if (opponents.length !== players - 1 || !opponents.every((o) => LEVELS.includes(o as AiLevel)))
      throw new LlmError(400, `\`opponents\` : un niveau, ou ${players - 1} niveaux parmi ${LEVELS.join(', ')}.`);
    const seat = b.seat === undefined ? Math.floor(Math.random() * players) : Number(b.seat);
    if (!Number.isInteger(seat) || seat < 0 || seat >= players) throw new LlmError(400, `\`seat\` : de 0 à ${players - 1}.`);
    const seed = b.seed === undefined ? randomSeed() : Number(b.seed);
    if (!Number.isSafeInteger(seed) || seed < 0) throw new LlmError(400, '`seed` : un entier positif.');
    const name = typeof b.name === 'string' && b.name.trim() ? b.name.trim().slice(0, 14) : 'LLM';
    const levels = [...opponents] as AiLevel[];
    const setup = Array.from({ length: players }, (_, i) =>
      i === seat
        ? { name, color: PLAYER_COLORS[i]!.color, kind: 'human' as const, ai: 'normal' as const }
        : { name: AI_NAMES[i]!, color: PLAYER_COLORS[i]!.color, kind: 'ai' as const, ai: levels.shift()! },
    );
    const now = Date.now();
    const entry: LlmGame = {
      id: newId(),
      createdAt: now,
      updatedAt: now,
      seat,
      game: createGame({ players: setup, mapSeed: seed, seed: seed + 1 }),
      seen: 0,
    };
    this.games.set(entry.id, entry);
    return this.run(entry, async () => {});
  }

  view(id: string): LlmView {
    return view(this.get(id));
  }

  /** L'état complet de la partie (tests et outils ; jamais envoyé au joueur). */
  game(id: string): GameState {
    return this.get(id).game;
  }

  /** Joue l'action du joueur, puis celles des IA jusqu'à son tour suivant. */
  async act(id: string, action: unknown): Promise<LlmView> {
    const entry = this.get(id);
    if (entry.game.phase === 'gameOver') throw new LlmError(409, 'La partie est finie.');
    if (currentActor(entry.game) !== entry.seat) throw new LlmError(409, "Ce n'est pas votre tour.");
    if (!action || typeof action !== 'object' || typeof (action as Action).type !== 'string')
      throw new LlmError(400, 'Le corps doit être une action en JSON, par exemple {"type":"endConquest"}.');
    return this.run(entry, async () => {
      try {
        const before = entry.game.log.at(-1)?.seq ?? 0;
        entry.game = applyAction(entry.game, action as Action);
        entry.seen = before;
      } catch (e) {
        if (e instanceof RuleError) throw new LlmError(422, `Action refusée : ${e.message}`);
        throw new LlmError(400, `Action illisible : ${(e as Error).message}`);
      }
    });
  }

  /** Une requête à la fois par partie : le coup, les IA, la sauvegarde, la réponse. */
  private async run(entry: LlmGame, move: () => Promise<void>): Promise<LlmView> {
    if (this.busy.has(entry.id)) throw new LlmError(409, 'Une requête est déjà en cours pour cette partie.');
    this.busy.add(entry.id);
    try {
      await move();
      await this.opponents(entry);
      entry.updatedAt = Date.now();
      this.save(entry);
      return view(entry);
    } finally {
      this.busy.delete(entry.id);
    }
  }

  /** Les IA jouent tant que ce n'est pas au joueur (ou jusqu'à la fin). */
  private async opponents(entry: LlmGame) {
    for (let n = 0; n < MAX_AI_ACTIONS; n++) {
      const g = entry.game;
      const who = currentActor(g);
      if (g.phase === 'gameOver' || who === null || who === entry.seat) return;
      const level = g.players[who]!.ai;
      const start = performance.now();
      if (level === 'expert' && this.opts.expert) {
        try {
          const move = await this.opts.expert.decide(g);
          const thought: AiThought = {
            level,
            ms: Math.round(performance.now() - start),
            actions: move.actions.length,
            iterations: move.iterations,
            computeMs: Math.round(move.computeMs),
          };
          // le coup entier de l'Expert ; s'il est refusé en route, il redemande au tour de boucle suivant
          let played = 0;
          for (const a of move.actions) {
            try {
              entry.game = applyAction(entry.game, a, played ? undefined : thought);
              played++;
            } catch {
              break;
            }
          }
          if (played) continue;
        } catch (e) {
          console.warn(`[llm] ${entry.id} : l'Expert n'a pas répondu (${(e as Error).message}) ; l'IA Difficile joue à sa place`);
        }
      }
      const fallback = level === 'expert';
      entry.game = applyAction(g, aiAction(g), { level, ms: Math.round(performance.now() - start), ...(fallback ? { fallback } : {}) });
    }
    throw new LlmError(500, 'Partie bloquée : les IA ne finissent pas leur tour.');
  }

  private get(id: string): LlmGame {
    const known = this.games.get(id);
    if (known) return known;
    if (this.opts.dir && /^[a-z0-9]{8}$/.test(id)) {
      try {
        const entry = JSON.parse(readFileSync(join(this.opts.dir, `${id}.json`), 'utf8')) as LlmGame;
        this.games.set(id, entry);
        return entry;
      } catch {
        // partie inconnue
      }
    }
    throw new LlmError(404, `Partie inconnue : ${id}.`);
  }

  private save(entry: LlmGame) {
    if (!this.opts.dir) return;
    try {
      writeFileSync(join(this.opts.dir, `${entry.id}.json`), JSON.stringify(entry));
    } catch (e) {
      console.error(`[llm] sauvegarde impossible de la partie ${entry.id}`, e);
    }
    // la mémoire ne garde que les parties récentes ; les autres se relisent sur disque
    const old = Date.now() - 24 * 3600 * 1000;
    for (const [id, g] of this.games) if (g.updatedAt < old && !this.busy.has(id)) this.games.delete(id);
  }
}

/** Le mode d'emploi (`src/server/llm.md`), complété par les peuples et les pouvoirs du jeu. */
export function llmRules(template: string): string {
  const races = RACE_IDS.map((id) => {
    const r = RACES[id];
    return `- **${r.name}** (\`${id}\`, ${r.tokens} jetons, ${r.max} au plus) : ${r.desc}${r.inDecline ? ` En déclin : ${r.inDecline}.` : ''}`;
  });
  const powers = POWER_IDS.map((id) => {
    const p = POWERS[id];
    return `- **${p.name}** (\`${id}\`, +${p.tokens} jetons) : ${p.desc}`;
  });
  const glossary = [`### Les ${RACE_IDS.length} peuples`, '', ...races, '', `### Les ${POWER_IDS.length} pouvoirs`, '', ...powers].join('\n');
  return template.replace('<!-- peuples et pouvoirs -->', glossary);
}

function newId(): string {
  const alphabet = 'abcdefghijkmnpqrstuvwxyz23456789';
  return Array.from(crypto.getRandomValues(new Uint8Array(8)), (b) => alphabet[b % alphabet.length]).join('');
}

/** Les réponses HTTP de l'API (montées par server.ts). */
export function llmRoutes(games: LlmGames, rules: () => string) {
  const answer = async (req: Request, work: () => Promise<LlmView> | LlmView): Promise<Response> => {
    try {
      const v = await work();
      return new URL(req.url).searchParams.get('format') === 'text' ? text(`${v.text}\n`) : Response.json(v);
    } catch (e) {
      if (e instanceof LlmError) return Response.json({ error: e.message }, { status: e.status });
      console.error('[llm]', e);
      return Response.json({ error: 'Erreur du serveur.' }, { status: 500 });
    }
  };
  const body = async (req: Request): Promise<unknown> => {
    const raw = await req.text();
    if (!raw.trim()) return {};
    try {
      return JSON.parse(raw);
    } catch {
      throw new LlmError(400, 'Le corps de la requête doit être du JSON.');
    }
  };
  return {
    '/llm.md': () => text(rules(), 'text/markdown; charset=utf-8'),
    '/api/llm/games': { POST: (req: Request) => answer(req, async () => games.create(await body(req))) },
    '/api/llm/games/:id': { GET: (req: Request & { params: { id: string } }) => answer(req, () => games.view(req.params.id)) },
    '/api/llm/games/:id/actions': {
      POST: (req: Request & { params: { id: string } }) => answer(req, async () => games.act(req.params.id, await body(req))),
    },
  };
}

function text(s: string, type = 'text/plain; charset=utf-8'): Response {
  return new Response(s, { headers: { 'Content-Type': type } });
}

// --- ce que voit le joueur ---------------------------------------------------------------------

function view(entry: LlmGame): LlmView {
  const g = entry.game;
  const fresh = g.log.filter((l) => l.seq > entry.seen && !l.quiet).map((l) => l.text);
  const yourTurn = g.phase !== 'gameOver' && currentActor(g) === entry.seat;
  const legal = yourTurn ? legalActions(g) : [];
  const final =
    g.phase === 'gameOver'
      ? ranking(g).map((r) => ({ seat: r.pid, name: g.players[r.pid]!.name, coins: r.coins }))
      : undefined;
  return {
    id: entry.id,
    seat: entry.seat,
    round: g.round,
    maxRounds: g.maxRounds,
    phase: g.phase,
    yourTurn,
    gameOver: g.phase === 'gameOver',
    events: fresh,
    text: describe(entry, fresh, legal, final),
    legal,
    ...(final ? { ranking: final } : {}),
  };
}

const raceName = (r: RaceId | 'lost') => (r === 'lost' ? 'tribu oubliée' : RACES[r].name);

function describe(entry: LlmGame, events: string[], legal: LlmView['legal'], final: LlmView['ranking']): string {
  const g = entry.game;
  const me = entry.seat;
  const out: string[] = [];
  const who = currentActor(g);
  const turn = g.phase === 'gameOver' ? '' : who === me ? ' — **à vous de jouer**' : ` — au tour de J${(who ?? g.current) + 1}`;
  out.push(`# Partie ${entry.id} : manche ${g.round}/${g.maxRounds}, phase ${PHASES[g.phase]}${turn}`);
  out.push(`Vous êtes **J${me + 1}** (${g.players[me]!.name}).`);
  if (events.length) out.push('', '## Depuis votre dernier coup (compris)', ...events.map((e) => `- ${e}`));
  if (final) {
    out.push('', '## Classement final');
    final.forEach((r, i) => out.push(`${i + 1}. J${r.seat + 1} ${r.name}${r.seat === me ? ' (vous)' : ''} : ${r.coins} pièces`));
    return out.join('\n');
  }

  out.push('', '## Joueurs');
  for (const p of g.players) {
    const parts = [`${p.coins} pièces`];
    if (p.active) {
      const rs = g.races[p.active]!;
      const n = regionsOf(g, p.active).length;
      parts.push(`actif : ${RACES[p.active].name} + ${POWERS[rs.power].name} (${n} régions, ${tokensOnBoard(g, p.active)} jetons posés, ${rs.hand} en main)`);
    } else parts.push('aucun peuple actif');
    for (const d of p.declined) {
      const rs = g.races[d];
      parts.push(`en déclin : ${RACES[d].name}${rs ? ` + ${POWERS[rs.power].name}` : ''} (${regionsOf(g, d).length} régions)`);
    }
    const level = p.kind === 'ai' ? `IA ${p.ai}` : 'vous';
    out.push(`- J${p.id + 1} ${p.name} (${p.id === me ? 'vous' : level}) : ${parts.join(' ; ')}`);
  }
  for (const peace of g.peace) out.push(`- Paix : J${peace.target + 1} ne peut pas attaquer J${peace.diplomat + 1} à son prochain tour.`);

  const army = g.turn.army;
  if (g.phase !== 'gameOver' && army && who === g.current) {
    const t = g.turn;
    const extra = [
      `${t.conquests} conquête(s) ce tour`,
      t.stage === 'ghouls' ? 'les Goules en déclin jouent avant le peuple actif' : '',
      t.berserkDie !== null ? `dé berserk lancé : ${t.berserkDie}` : '',
      t.amazonBonus ? `${t.amazonBonus} jetons prêtés aux Amazones pour conquérir` : '',
    ].filter(Boolean);
    out.push('', `Peuple en action : ${RACES[army].name}, ${g.races[army]!.hand} jetons en main ; ${extra.join(' ; ')}.`);
  }
  if (g.phase === 'defend' && g.defense[0]) {
    const d = g.defense[0];
    out.push('', `Défense : J${d.player + 1} replace les jetons rendus de ses ${RACES[d.race].name} (${g.races[d.race]?.hand ?? 0} en main).`);
  }

  out.push('', '## Marché (prendre la combinaison n coûte n pièces ; les pièces posées dessus sont gagnées)');
  g.market.forEach((m, i) =>
    out.push(`${i}. ${RACES[m.race].name} + ${POWERS[m.power].name} : ${RACES[m.race].tokens + POWERS[m.power].tokens} jetons, ${m.coins} pièce(s) dessus, coût ${i}`),
  );
  out.push(`Pioche : ${g.raceDeck.length} peuples et ${g.powerDeck.length} pouvoirs (ordre caché).`);

  out.push('', '## Carte', 'id · nom · terrain · particularités · voisines → occupant');
  g.map.regions.forEach((def, i) => {
    const r = g.regions[i]!;
    const flags = [
      def.entry ? 'entrée' : '',
      def.border ? 'bord' : '',
      def.coastal ? 'côte' : '',
      def.cavern ? 'caverne' : '',
      def.magic ? 'source magique' : '',
      def.mine ? 'mine' : '',
    ].filter(Boolean);
    let occupant = 'vide';
    if (r.race) {
      const owner = r.race === 'lost' ? null : g.races[r.race]?.owner;
      const state = r.race !== 'lost' && g.races[r.race]?.declined ? ' en déclin' : '';
      occupant = `${owner !== null && owner !== undefined ? `J${owner + 1} ` : ''}${raceName(r.race)}${state} ×${r.tokens}`;
    }
    const marks = [
      r.lair ? 'tanière' : '',
      r.fortress ? 'forteresse' : '',
      r.camps ? `${r.camps} campement(s)` : '',
      r.hero ? 'héros' : '',
      r.dragon ? 'dragon' : '',
      r.hole ? 'terrier' : '',
    ].filter(Boolean);
    out.push(
      `${i} · ${def.name} · ${TERRAINS[def.terrain].name} · ${flags.join(', ') || '-'} · ${def.adj.join(', ')}` +
        ` → ${occupant}${marks.length ? ` (${marks.join(', ')})` : ''}`,
    );
  });

  if (legal.length) {
    out.push('', '## Coups permis (corps de POST /api/llm/games/:id/actions)');
    for (const l of legal) out.push(`- \`${JSON.stringify(l.action)}\` : ${l.note}`);
  } else if (g.phase !== 'gameOver') out.push('', "Ce n'est pas votre tour.");
  return out.join('\n');
}

/** Les coups permis au joueur qui doit agir : chaque candidat est essayé par le noyau (l'état ne change pas). */
export function legalActions(g: GameState): LlmView['legal'] {
  const allowed = (a: Action) => {
    try {
      kernelFor(g).apply(a);
      return true;
    } catch (e) {
      if (e instanceof KernelRefusal) return false;
      throw e;
    }
  };
  const name = (r: number) => `${r} « ${g.map.regions[r]!.name} »`;
  const out: LlmView['legal'] = [];
  const add = (action: Action, note: string) => {
    if (allowed(action)) out.push({ action, note });
  };
  const regions = g.map.regions.map((_, i) => i);
  switch (g.phase) {
    case 'pick':
      g.market.forEach((m, i) =>
        add({ type: 'pick', index: i }, `prendre ${RACES[m.race].name} + ${POWERS[m.power].name} pour ${i} pièce(s), en gagner ${m.coins}`),
      );
      add({ type: 'skipPick' }, 'passer (marché vide)');
      break;
    case 'conquest': {
      const targets = allTargets(g, true);
      for (const t of targets) {
        if (t.kind === 'normal') add({ type: 'conquer', region: t.region }, `conquérir ${name(t.region)} pour ${t.effective} jeton(s)`);
        if (t.kind === 'die')
          add(
            { type: 'reinforce', region: t.region },
            `dernière conquête au dé sur ${name(t.region)} : il manque ${t.need} jeton(s), ${Math.round(t.chance * 100)} % de chances ; finit les conquêtes`,
          );
        if (t.dragon) add({ type: 'dragon', region: t.region }, `conquérir ${name(t.region)} avec le dragon (1 jeton)`);
        if (t.sorcery) add({ type: 'sorcery', region: t.region }, `ensorceler ${name(t.region)} (remplace un jeton adverse seul)`);
      }
      add({ type: 'berserkRoll' }, 'lancer le dé berserk avant la prochaine conquête');
      for (const r of regions) add({ type: 'abandon', region: r }, `abandonner ${name(r)} (ses jetons reviennent en main)`);
      add({ type: 'decline' }, 'passer le peuple actif en déclin (finit le tour)');
      add({ type: 'endConquest' }, 'finir les conquêtes et passer au redéploiement');
      break;
    }
    case 'redeploy':
    case 'defend':
      for (const r of regions) {
        add({ type: 'deploy', region: r, delta: 1 }, `poser 1 jeton de la main sur ${name(r)}`);
        add({ type: 'deploy', region: r, delta: -1 }, `reprendre 1 jeton de ${name(r)} en main`);
      }
      add({ type: 'autoDeploy' }, 'répartir automatiquement les jetons en main');
      add({ type: 'autoDeploy', all: true }, 'tout reprendre (1 jeton reste par région) et tout répartir automatiquement');
      for (const r of regions) {
        add({ type: 'camp', region: r, delta: 1 }, `poser un campement sur ${name(r)}`);
        add({ type: 'camp', region: r, delta: -1 }, `retirer un campement de ${name(r)}`);
        add({ type: 'fortress', region: r }, `forteresse sur ${name(r)} (ou la retirer si posée ce tour)`);
        add({ type: 'hero', region: r }, `héros sur ${name(r)} (ou le retirer)`);
      }
      add({ type: 'endTurn' }, 'finir le tour (décompte des pièces)');
      add({ type: 'endTurn', stout: true }, 'finir le tour et passer en déclin juste après le décompte (Robustes)');
      g.players.forEach((_, p) =>
        add({ type: 'endTurn', peace: p }, `finir le tour en offrant la paix à J${p + 1} (Diplomates : il ne pourra pas vous attaquer)`),
      );
      add({ type: 'endDefense' }, 'finir de replacer les jetons rendus');
      break;
  }
  return out;
}
