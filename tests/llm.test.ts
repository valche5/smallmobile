import { describe, expect, test } from 'bun:test';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { aiAction } from '../src/engine/ai';
import { PLAYER_COLORS, RACE_IDS, RACES } from '../src/engine/data';
import { applyAction, createGame } from '../src/engine/game';
import { currentActor } from '../src/engine/rules';
import type { Action, GameState } from '../src/engine/types';
import { FileGameArchive, type ArchivedGame, type GameArchive } from '../src/server/archive';
import type { Expert, ExpertMove } from '../src/server/expert';
import { legalActions, LlmError, LlmGames, llmRoutes, llmRules, type LlmView } from '../src/server/llm';

/** Un Expert de test : l'IA du noyau, qui joue d'un coup toute sa fin de tour (comme l'Expert). */
class FakeExpert implements Expert {
  calls = 0;
  async decide(g: GameState): Promise<ExpertMove> {
    this.calls++;
    const actor = currentActor(g);
    const actions: Action[] = [];
    let cur = g;
    do {
      const a = aiAction(cur);
      actions.push(a);
      cur = applyAction(cur, a);
    } while (cur.phase === 'redeploy' && g.phase === 'redeploy' && currentActor(cur) === actor);
    return { actions, value: 0.5, iterations: 200, computeMs: 3 };
  }
}

class MemoryArchive implements GameArchive {
  games = new Map<string, ArchivedGame>();
  save(g: ArchivedGame) {
    this.games.set(g.id, g);
  }
}

/** Un joueur simple qui ne joue que des coups de la liste : il avance toujours. */
const ORDER: Action['type'][] = ['pick', 'conquer', 'endConquest', 'endTurn', 'endDefense', 'autoDeploy', 'skipPick'];
function choose(v: LlmView): Action {
  for (const type of ORDER) {
    const l = v.legal.find((x) => x.action.type === type);
    if (l) return l.action;
  }
  return v.legal[0]!.action;
}

async function rejected(p: Promise<unknown>): Promise<LlmError> {
  try {
    await p;
  } catch (e) {
    if (e instanceof LlmError) return e;
    throw e;
  }
  throw new Error('requête acceptée');
}

describe('parties pour un LLM', () => {
  test('une partie se joue jusqu’au bout avec les coups proposés, tous permis', async () => {
    const expert = new FakeExpert();
    const archive = new MemoryArchive();
    const games = new LlmGames({ expert, archive, dir: null });
    let v = await games.create({ model: 'claude-test', client: 'tests', players: 3, opponents: ['expert', 'difficile'], seat: 1, seed: 42, name: 'Testeur' });
    expect(v.model).toBe('claude-test');
    expect(v.text).toContain('modèle claude-test');
    expect(v.seat).toBe(1);
    expect(v.yourTurn).toBe(true);
    expect(v.events.length).toBeGreaterThan(0);
    let moves = 0;
    while (!v.gameOver) {
      expect(v.yourTurn).toBe(true);
      expect(v.legal.length).toBeGreaterThan(0);
      // chaque coup proposé est accepté par les règles
      const g = games.game(v.id);
      for (const l of v.legal) expect(() => applyAction(g, l.action)).not.toThrow();
      expect(v.text).toContain('## Coups permis');
      v = await games.act(v.id, choose(v));
      moves++;
      expect(moves).toBeLessThan(2000);
    }
    expect(expert.calls).toBeGreaterThan(0);
    expect(v.ranking!.map((r) => r.seat).sort()).toEqual([0, 1, 2]);
    expect(v.text).toContain('Classement final');
    expect(v.legal).toEqual([]);
    expect((await rejected(games.act(v.id, { type: 'endTurn' }))).status).toBe(409);

    // le bilan est demandé, puis gardé avec la partie et dans l'archive
    expect(v.debrief).toBeNull();
    expect(v.text).toContain(`/api/llm/games/${v.id}/debrief`);
    expect((await rejected(Promise.resolve().then(() => games.debrief(v.id, { text: '  ' })))).status).toBe(400);
    const d = games.debrief(v.id, { text: 'J’ai trop attendu pour passer en déclin.' });
    expect(d.debrief).toBe('J’ai trop attendu pour passer en déclin.');
    expect(d.text).toContain('trop attendu');
    const [kept] = [...archive.games.values()];
    expect(archive.games.size).toBe(1);
    expect(kept!.id).toMatch(/_llm-[a-z0-9]{8}$/);
    expect(kept!.players.map((p) => p.kind)).toEqual(['ai', 'llm', 'ai']);
    expect(kept!.llm).toEqual({ seat: 1, model: 'claude-test', client: 'tests', debrief: 'J’ai trop attendu pour passer en déclin.' });
    expect(kept!.coins).toEqual(games.game(v.id).players.map((p) => p.coins));
  });

  test('le modèle est obligatoire ; le bilan attend la fin de la partie', async () => {
    const games = new LlmGames({ expert: null, archive: null, dir: null });
    expect((await rejected(games.create({ players: 2 }))).status).toBe(400);
    expect((await rejected(games.create({ model: '   ', players: 2 }))).status).toBe(400);
    const v = await games.create({ model: 'test', players: 2, seat: 0, seed: 1 });
    expect((await rejected(Promise.resolve().then(() => games.debrief(v.id, { text: 'trop tôt' })))).status).toBe(409);
  });

  test('les erreurs disent ce qui ne va pas, sans rien changer', async () => {
    const games = new LlmGames({ expert: null, archive: null, dir: null });
    const v = await games.create({ model: 'test', players: 2, opponents: 'normal', seat: 0, seed: 7 });
    expect(v.phase).toBe('pick');
    const before = games.game(v.id);
    const refused = await rejected(games.act(v.id, { type: 'conquer', region: 0 }));
    expect(refused.status).toBe(422);
    expect(refused.message).toMatch(/^Action refusée : /);
    expect((await rejected(games.act(v.id, { type: 'conquer', region: 'x' }))).status).toBe(400);
    expect((await rejected(games.act(v.id, 'pick'))).status).toBe(400);
    expect(games.game(v.id)).toBe(before);
    expect((await rejected(games.act('inconnue', { type: 'decline' }))).status).toBe(404);
    expect((await rejected(games.create({ model: 'test', players: 6 }))).status).toBe(400);
    expect((await rejected(games.create({ model: 'test', players: 3, opponents: ['expert'] }))).status).toBe(400);
    expect((await rejected(games.create({ model: 'test', opponents: 'champion' }))).status).toBe(400);
  });

  test('la vue ne montre ni l’ordre des pioches ni le hasard', async () => {
    const games = new LlmGames({ expert: null, archive: null, dir: null });
    const v = await games.create({ model: 'test', players: 4, opponents: 'difficile', seat: 2, seed: 9 });
    const g = games.game(v.id);
    expect(Object.keys(v).sort()).toEqual(['events', 'gameOver', 'id', 'legal', 'maxRounds', 'model', 'phase', 'round', 'seat', 'text', 'yourTurn']);
    // les peuples de la pioche n'apparaissent nulle part, sauf s'ils sont aussi en jeu
    const shown = new Set([...g.market.map((m) => m.race), ...(Object.keys(g.races) as (keyof typeof RACES)[])]);
    for (const r of g.raceDeck.filter((r) => !shown.has(r))) expect(v.text).not.toContain(RACES[r].name);
  });

  test('une partie se reprend après un redémarrage du serveur', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'llm-'));
    try {
      const first = new LlmGames({ expert: null, archive: null, dir });
      const v = await first.create({ model: 'test', players: 2, opponents: 'facile', seed: 3 });
      const again = new LlmGames({ expert: null, archive: null, dir });
      const w = again.view(v.id);
      expect(w.round).toBe(v.round);
      expect(w.legal).toEqual(v.legal);
      await again.act(v.id, choose(w));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test('une partie marquée après coup comme jouée par un LLM le reste quand le serveur la réécrit', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'archive-'));
    try {
      const games = new LlmGames({ expert: null, archive: null, dir: null });
      const v = await games.create({ model: 'test', players: 2, seat: 0, seed: 11 });
      const g = games.game(v.id);
      const room: ArchivedGame = {
        id: 'salle',
        code: 'ABCD',
        updatedAt: 1,
        players: g.players.map((p) => ({ name: p.name, kind: p.kind, ai: p.ai })),
        coins: [40, 90],
        record: g.record!,
      };
      const file = join(dir, 'salle.json');
      const llm = { seat: 0, model: 'deepseek', client: 'interface du jeu' };
      writeFileSync(file, JSON.stringify({ ...room, players: room.players.map((p, i) => (i ? p : { ...p, kind: 'llm' })), llm }));
      const archive = new FileGameArchive(dir);
      archive.save({ ...room, updatedAt: 2 });
      archive.flush();
      const kept = JSON.parse(readFileSync(file, 'utf8')) as ArchivedGame;
      expect(kept.updatedAt).toBe(2);
      expect(kept.llm).toEqual(llm);
      expect(kept.players.map((p) => p.kind)).toEqual(['llm', 'ai']);
      // une partie en cours n'est jamais relue
      archive.save({ ...room, coins: null, updatedAt: 3 });
      archive.flush();
      expect((JSON.parse(readFileSync(file, 'utf8')) as ArchivedGame).llm).toBeUndefined();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test('à la fin de l’étape des Goules en déclin, ni paix ni déclin des Robustes', () => {
    let stages = 0;
    for (let seed = 1; seed <= 60 && stages < 3; seed++) {
      const players = [0, 1, 2].map((i) => ({ name: `IA ${i}`, color: PLAYER_COLORS[i]!.color, kind: 'ai' as const, ai: 'difficile' as const }));
      let g = createGame({ players, mapSeed: seed, seed });
      while (g.phase !== 'gameOver') {
        if (g.phase === 'redeploy' && g.turn.stage === 'ghouls') {
          const ends = legalActions(g).filter((l) => l.action.type === 'endTurn');
          expect(ends.map((l) => l.action)).toEqual([{ type: 'endTurn' }]);
          stages++;
        }
        g = applyAction(g, aiAction(g));
      }
    }
    expect(stages).toBeGreaterThan(0);
  });

  test('les routes HTTP et le mode d’emploi', async () => {
    const games = new LlmGames({ expert: null, archive: null, dir: null });
    const routes = llmRoutes(games, () => 'règles');
    const post = (url: string, body: string, id?: string) =>
      Object.assign(new Request(`http://x${url}`, { method: 'POST', body }), { params: { id: id ?? '' } });
    const created = await routes['/api/llm/games'].POST(post('/api/llm/games', '{"model":"test","players":2,"seat":0,"seed":5}'));
    expect(created.status).toBe(200);
    const v = (await created.json()) as LlmView;
    const get = Object.assign(new Request(`http://x/api/llm/games/${v.id}?format=text`), { params: { id: v.id } });
    const text = await routes['/api/llm/games/:id'].GET(get);
    expect(text.headers.get('Content-Type')).toContain('text/plain');
    expect(await text.text()).toBe(`${v.text}\n`);
    const bad = await routes['/api/llm/games/:id/actions'].POST(post(`/api/llm/games/${v.id}/actions`, '{nope', v.id));
    expect(bad.status).toBe(400);
    expect(((await bad.json()) as { error: string }).error).toContain('JSON');
    const played = await routes['/api/llm/games/:id/actions'].POST(post(`/api/llm/games/${v.id}/actions`, JSON.stringify(v.legal[0]!.action), v.id));
    expect(played.status).toBe(200);
    expect(await routes['/llm.md']().text()).toBe('règles');

    const guide = llmRules(await Bun.file(new URL('../src/server/llm.md', import.meta.url).pathname).text());
    expect(guide).not.toContain('<!-- peuples et pouvoirs -->');
    for (const id of RACE_IDS) expect(guide).toContain(`**${RACES[id].name}**`);
  });
});
