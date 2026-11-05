import { describe, expect, test } from 'bun:test';
import { aiAction } from '../src/engine/ai';
import { RACES, RACE_IDS } from '../src/engine/data';
import { applyAction, createGame, type PlayerSetup } from '../src/engine/game';
import { getMap, PRESETS } from '../src/engine/maps';
import {
  allTargets,
  analyzeTarget,
  cloneState,
  defenseParts,
  defenseValue,
  regionsOf,
  scanTargets,
  scoreBreakdown,
  tokensOnBoard,
} from '../src/engine/rules';
import type { AiLevel, GameState, RaceId } from '../src/engine/types';

const COLORS = ['#d62839', '#1f6fe0', '#8e3ccb', '#ec7f00', '#2b2d42'];

function setup(n: number, levels: AiLevel[] = ['normal']): PlayerSetup[] {
  return Array.from({ length: n }, (_, i) => ({
    name: `IA ${i + 1}`,
    color: COLORS[i]!,
    kind: 'ai' as const,
    ai: levels[i % levels.length]!,
  }));
}

function checkInvariants(g: GameState) {
  const onBoard = new Set<RaceId>();
  g.regions.forEach((r, i) => {
    if (r.race === null) {
      expect(r.tokens).toBe(0);
      expect(r.lair || r.fortress || r.hero || r.dragon || r.hole || r.camps > 0).toBe(false);
    } else {
      expect(r.tokens).toBeGreaterThanOrEqual(1);
      if (r.race !== 'lost') {
        onBoard.add(r.race);
        expect(g.races[r.race]).toBeDefined();
      }
    }
    const def = g.map.regions[i]!;
    if ((def.terrain === 'sea' || def.terrain === 'lake') && r.race) {
      expect(r.race).not.toBe('lost');
      expect(g.races[r.race as RaceId]!.power).toBe('seafaring');
    }
  });
  for (const [race, rs] of Object.entries(g.races)) {
    expect(rs!.hand).toBeGreaterThanOrEqual(0);
    expect(tokensOnBoard(g, race as RaceId) + rs!.hand).toBeLessThanOrEqual(RACES[race as RaceId].max);
    const p = g.players[rs!.owner]!;
    if (rs!.declined) expect(p.declined).toContain(race as RaceId);
    else expect(p.active).toBe(race as RaceId);
  }
  for (const p of g.players) {
    expect(p.coins).toBeGreaterThanOrEqual(0);
    expect(p.declined.length).toBeLessThanOrEqual(2);
    if (p.active) expect(g.races[p.active]!.owner).toBe(p.id);
  }
  const all = [...g.raceDeck, ...g.market.map((m) => m.race), ...(Object.keys(g.races) as RaceId[])];
  expect(new Set(all).size).toBe(all.length);
  expect(all.length).toBe(RACE_IDS.length);
  expect(g.market.length).toBeLessThanOrEqual(6);
}

function playOut(g: GameState, check = true): { g: GameState; steps: number } {
  let steps = 0;
  while (g.phase !== 'gameOver') {
    const a = aiAction(g);
    g = applyAction(g, a);
    if (check) checkInvariants(g);
    steps++;
    if (steps > 20000) throw new Error('Partie interminable');
  }
  return { g, steps };
}

/** Régions et tours de chaque taille de carte. */
const SIZES: Record<number, [number, number]> = { 2: [23, 10], 3: [30, 10], 4: [39, 9], 5: [48, 8] };

describe('cartes', () => {
  for (const n of [2, 3, 4, 5]) {
    test(`carte ${n} joueurs`, () => {
      const m = getMap(n, PRESETS[n]!.seed);
      expect(m.regions.length).toBe(SIZES[n]![0]);
      expect(m.rounds).toBe(SIZES[n]![1]);
      for (const r of m.regions) for (const a of r.adj) expect(m.regions[a]!.adj).toContain(r.id);
      expect(m.regions.filter((r) => r.entry).length).toBeGreaterThan(3);
      // déterministe
      expect(getMap(n, PRESETS[n]!.seed)).toBe(m);
    });
  }
  test('cartes aléatoires', () => {
    for (let s = 1; s < 25; s++) for (const n of [2, 5]) expect(getMap(n, s * 977).regions.length).toBe(SIZES[n]![0]);
  });
});

describe('parties complètes IA contre IA', () => {
  for (const n of [2, 3, 4, 5]) {
    for (const seed of [1, 2, 3]) {
      test(`${n} joueurs, graine ${seed}`, () => {
        const g0 = createGame({ players: setup(n, ['facile', 'normal', 'difficile']), mapSeed: PRESETS[n]!.seed, seed });
        const { g } = playOut(g0);
        expect(g.phase).toBe('gameOver');
        expect(g.players.every((p) => p.coins > 5)).toBe(true);
      }, 30000);
    }
  }
  test('beaucoup de parties rapides (robustesse)', () => {
    for (let seed = 10; seed < 50; seed++) {
      const n = 2 + (seed % 4);
      const g0 = createGame({ players: setup(n, ['normal', 'difficile', 'facile']), mapSeed: seed * 31, seed });
      const { g } = playOut(g0, seed % 5 === 0);
      expect(g.phase).toBe('gameOver');
    }
  }, 120000);
});

function freshGame(): GameState {
  const players = setup(2);
  players[0]!.kind = 'human';
  players[1]!.kind = 'human';
  return createGame({ players, mapSeed: PRESETS[2]!.seed, seed: 42 });
}

describe('règles', () => {
  test('choix de peuple : paiement et pièces récupérées', () => {
    let g = freshGame();
    g.market[0]!.coins = 0;
    g.market[1]!.coins = 0;
    g.market[2]!.coins = 3;
    const slot = g.market[2]!;
    g = applyAction(g, { type: 'pick', index: 2 });
    expect(g.players[0]!.coins).toBe(5 - 2 + 3);
    expect(g.market[0]!.coins).toBe(1);
    expect(g.market[1]!.coins).toBe(1);
    expect(g.market.length).toBe(6);
    expect(g.players[0]!.active).toBe(slot.race);
    expect(g.phase).toBe('conquest');
  });

  test('coût de conquête : base 2 + jetons + montagne', () => {
    let g = freshGame();
    g = applyAction(g, { type: 'pick', index: 0 });
    const army = g.turn.army!;
    const entry = g.map.regions.find((r) => r.entry && r.terrain === 'mountain' && !r.lostTribe);
    if (entry) {
      const info = analyzeTarget(g, entry.id);
      const rs = g.races[army]!;
      const mods = info.parts.filter((p) => p.value < 0).reduce((s, p) => s + p.value, 0);
      expect(info.cost).toBe(Math.max(1, 3 + mods));
      expect(info.reachable).toBe(true);
      expect(rs.hand).toBeGreaterThan(0);
    }
    const interior = g.map.regions.find((r) => !r.entry && r.terrain !== 'sea' && r.terrain !== 'lake');
    if (interior && army !== 'halflings' && g.races[army]!.power !== 'flying') {
      expect(analyzeTarget(g, interior.id).reachable).toBe(false);
    }
  });

  test('conquête : le défenseur perd 1 jeton et récupère le reste', () => {
    let g = freshGame();
    // On fabrique une situation : joueur 0 = Ratmen, joueur 1 = Orques
    g = cloneState(g);
    g.market = [
      { race: 'ratmen', power: 'commando', coins: 0 },
      { race: 'orcs', power: 'alchemist', coins: 0 },
      ...g.market.filter((m) => m.race !== 'ratmen' && m.race !== 'orcs').slice(0, 4),
    ];
    g.raceDeck = g.raceDeck.filter((r) => r !== 'ratmen' && r !== 'orcs');
    g.powerDeck = g.powerDeck.filter((p) => p !== 'commando' && p !== 'alchemist');
    g = applyAction(g, { type: 'pick', index: 0 });
    const a = g.map.regions.find((r) => r.entry && !r.lostTribe && r.terrain !== 'mountain')!;
    g = applyAction(g, { type: 'conquer', region: a.id });
    expect(g.regions[a.id]!.race).toBe('ratmen');
    expect(g.regions[a.id]!.tokens).toBe(1); // 2 - 1 (commando)
    // Mettre 3 jetons sur la région puis finir
    g = applyAction(g, { type: 'endConquest' });
    while (g.races.ratmen!.hand > 0) g = applyAction(g, { type: 'deploy', region: a.id, delta: 1 });
    g = applyAction(g, { type: 'endTurn' });
    expect(g.current).toBe(1);
    g = applyAction(g, { type: 'pick', index: 0 });
    expect(g.players[1]!.active).toBe('orcs');
    // Les Orques attaquent la région des Ratmen (tokens = 1 + 12 de main)
    const tokens = g.regions[a.id]!.tokens;
    const info = analyzeTarget(g, a.id);
    expect(info.cost).toBe(2 + tokens);
    g.races.orcs!.hand = 30; // triche pour le test (hors limite)
    g = applyAction(g, { type: 'conquer', region: a.id });
    expect(g.regions[a.id]!.race).toBe('orcs');
    expect(g.races.ratmen!.hand).toBe(tokens - 1);
    expect(g.turn.nonEmpty).toBe(1);
    expect(g.turn.attacked).toContain(0);
  });

  test('déclin : 1 jeton par région, ancien déclin retiré', () => {
    let g = freshGame();
    g = applyAction(g, { type: 'pick', index: 0 });
    const race = g.turn.army!;
    // conquérir quelques régions
    for (let k = 0; k < 3; k++) {
      const t = g.map.regions.map((r) => analyzeTarget(g, r.id)).find((x) => x.kind === 'normal');
      if (!t) break;
      g = applyAction(g, { type: 'conquer', region: t.region });
    }
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    g = applyAction(g, { type: 'endTurn', stout: false });
    // Joueur 2 joue un tour minimal
    g = applyAction(g, { type: 'pick', index: 0 });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    g = applyAction(g, { type: 'endTurn' });
    while (g.phase === 'defend') {
      g = applyAction(g, { type: 'autoDeploy' });
      g = applyAction(g, { type: 'endDefense' });
    }
    expect(g.current).toBe(0);
    const own = regionsOf(g, race);
    const before = g.players[0]!.coins;
    const expected = scoreBreakdown(g, 0);
    g = applyAction(g, { type: 'decline' });
    expect(g.races[race]!.declined).toBe(true);
    for (const r of regionsOf(g, race)) {
      if (race !== 'ghouls') expect(g.regions[r]!.tokens).toBe(1);
    }
    expect(regionsOf(g, race).length).toBe(own.length);
    // Le score de déclin ne compte plus les bonus actifs
    expect(g.players[0]!.coins - before).toBeLessThanOrEqual(expected.total);
    expect(g.players[0]!.active).toBeNull();
  });

  test('le dé de renfort termine les conquêtes', () => {
    let g = freshGame();
    g = applyAction(g, { type: 'pick', index: 0 });
    const army = g.turn.army!;
    g.races[army]!.hand = 1;
    const t = g.map.regions.map((r) => analyzeTarget(g, r.id)).find((x) => x.kind === 'die');
    if (!t) return;
    g = applyAction(g, { type: 'reinforce', region: t.region });
    expect(g.phase).toBe('redeploy');
    expect(g.lastDie).not.toBeNull();
    expect(g.turn.dieRolled).toBe(true);
  });
});

describe('défense du joueur suivant (résolue par le moteur)', () => {
  /** Joueur 0 = `mine` (Commandos), joueur 1 = Orques qui lui prennent une région bien garnie. */
  function attacked(mine: RaceId, lastRound = false) {
    let g = freshGame();
    g = cloneState(g);
    g.market = [
      { race: mine, power: 'commando', coins: 0 },
      { race: 'orcs', power: 'alchemist', coins: 0 },
      ...g.market.filter((m) => m.race !== mine && m.race !== 'orcs').slice(0, 4),
    ];
    g.raceDeck = g.raceDeck.filter((r) => r !== mine && r !== 'orcs');
    g.powerDeck = g.powerDeck.filter((p) => p !== 'commando' && p !== 'alchemist');
    g = applyAction(g, { type: 'pick', index: 0 });
    const a = g.map.regions.find((r) => r.entry && !r.lostTribe && r.terrain !== 'mountain' && r.terrain !== 'sea' && r.terrain !== 'lake')!;
    g = applyAction(g, { type: 'conquer', region: a.id });
    // une 2e région, pour que le plan de redéploiement ait un vrai choix
    const b = g.map.regions.map((r) => analyzeTarget(g, r.id)).find((t) => t.kind === 'normal' && t.region !== a.id);
    if (b) g = applyAction(g, { type: 'conquer', region: b.region });
    g = applyAction(g, { type: 'endConquest' });
    while (g.races[mine]!.hand > 0) g = applyAction(g, { type: 'deploy', region: a.id, delta: 1 });
    g = applyAction(g, { type: 'endTurn' });
    expect(g.current).toBe(1);
    if (lastRound) {
      g = cloneState(g);
      g.round = g.maxRounds;
    }
    g = applyAction(g, { type: 'pick', index: 0 });
    g.races.orcs!.hand = 40; // triche pour le test (hors limite)
    g = applyAction(g, { type: 'conquer', region: a.id });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    expect(g.races[mine]!.hand).toBeGreaterThan(0);
    return g;
  }

  for (const mine of ['ratmen', 'ghouls'] as RaceId[]) {
    test(`à 2 joueurs, pas de phase de défense, même résultat qu'Auto + Valider (${RACES[mine].name})`, () => {
      const before = attacked(mine);
      const after = applyAction(before, { type: 'endTurn' });
      expect(after.phase).not.toBe('defend');
      expect(after.current).toBe(0);
      expect(after.log.some((l) => l.kind === 'defense')).toBe(false);
      // l'ancien chemin : phase de défense, 🪄 Auto, puis Valider
      let old = cloneState(before);
      old.phase = 'defend';
      old.defense = [{ player: 0, race: mine }];
      old = applyAction(old, { type: 'autoDeploy' });
      old = applyAction(old, { type: 'endDefense' });
      expect(old.current).toBe(0);
      expect(after.races[mine]!.hand).toBe(old.races[mine]!.hand);
      expect(after.turn.readied).toEqual(old.turn.readied);
      if (mine === 'ghouls') expect(Object.keys(after.turn.readied).length).toBeGreaterThan(0);
      expect(after.regions.map((r) => r.tokens)).toEqual(old.regions.map((r) => r.tokens));
    });
  }

  test('au tout dernier tour, aucune défense', () => {
    const before = attacked('ratmen', true);
    const after = applyAction(before, { type: 'endTurn' });
    expect(after.phase).toBe('gameOver');
    expect(after.defense.length).toBe(0);
    expect(after.log.some((l) => l.kind === 'defense')).toBe(false);
  });

  test('le journal porte les données de la conquête', () => {
    const g = attacked('ratmen');
    const c = [...g.log].reverse().find((l) => l.kind === 'conquer')!;
    expect(c).toMatchObject({ how: 'conquer', victim: 0, victimRace: 'ratmen', lost: 1 });
    expect(c.returned).toBeGreaterThan(0);
    expect(c.tokens).toBeGreaterThan(0);
    const pick = [...g.log].reverse().find((l) => l.kind === 'pick')!;
    expect(pick).toMatchObject({ race: 'orcs', power: 'alchemist' });
    const end = applyAction(g, { type: 'endTurn' });
    const score = [...end.log].reverse().find((l) => l.kind === 'score')!;
    expect(score.gain).toBe(end.players[1]!.lastIncome);
  });
});

describe('copie de l\'état', () => {
  test('cloneState copie tout, comme structuredClone, sans rien partager de modifiable', () => {
    let g = createGame({ players: setup(3), seed: 11, mapSeed: 11 });
    for (let k = 0; k < 400 && g.phase !== 'gameOver'; k++) {
      const c = cloneState(g);
      expect(c).toEqual(structuredClone(g));
      expect(c.map).toBe(g.map);
      for (const key of ['players', 'races', 'regions', 'market', 'raceDeck', 'powerDeck', 'turn', 'defense', 'peace', 'log'] as const)
        expect(c[key]).not.toBe(g[key]);
      expect(c.regions[0]).not.toBe(g.regions[0]);
      expect(c.players[0]!.declined).not.toBe(g.players[0]!.declined);
      expect(c.turn.readied).not.toBe(g.turn.readied);
      g = applyAction(g, aiAction(g));
    }
  });
});

describe('calculs allégés', () => {
  test('les cibles allégées et les totaux numériques disent la même chose que le détail', () => {
    for (const seed of [5, 6, 7]) {
      let g = createGame({ players: setup(4), seed, mapSeed: seed });
      for (let k = 0; k < 600 && g.phase !== 'gameOver'; k++) {
        if (g.phase === 'conquest') {
          const full = allTargets(g);
          expect(scanTargets(g)).toEqual(full.map((t) => ({ ...t, parts: [] })));
          full.forEach((t, i) => expect(analyzeTarget(g, i)).toEqual(t));
          // le détail du coût (pour l'affichage) donne bien le coût du noyau
          full.forEach((t) => {
            if (t.reason === "Ce n'est pas la phase de conquête.") return;
            expect(Math.max(1, t.parts.filter((p) => p.short !== 'dé').reduce((s, p) => s + p.value, 0))).toBe(t.cost);
            expect(t.parts.reduce((s, p) => s + p.value, 0)).toBe(t.effective);
          });
          g.map.regions.forEach((_, i) => expect(defenseValue(g, i)).toBe(defenseParts(g, i).reduce((s, p) => s + p.value, 0)));
        }
        g = applyAction(g, aiAction(g));
      }
    }
  });
});

describe('placement de fin de tour', () => {
  test('« tout replacer » et « place » conservent les jetons et laissent au moins un jeton par région', () => {
    let checked = 0;
    for (const seed of [21, 22]) {
      let g = createGame({ players: setup(3), seed, mapSeed: seed });
      for (let k = 0; k < 600 && g.phase !== 'gameOver'; k++) {
        const army = g.turn.army;
        if (g.phase === 'redeploy' && army && regionsOf(g, army).length > 1) {
          const total = tokensOnBoard(g, army) + g.races[army]!.hand;
          const all = applyAction(g, { type: 'autoDeploy', all: true });
          expect(tokensOnBoard(all, army) + all.races[army]!.hand).toBe(total);
          expect(all.races[army]!.hand).toBe(0);
          expect(regionsOf(all, army).every((r) => all.regions[r]!.tokens >= 1)).toBe(true);
          // replacer deux fois ne change plus rien
          expect(applyAction(all, { type: 'autoDeploy', all: true }).regions).toEqual(all.regions);
          const own = regionsOf(g, army);
          const placed = applyAction(g, { type: 'place', tokens: { [own[0]!]: total - own.length + 1 } });
          expect(placed.regions[own[0]!]!.tokens).toBe(total - own.length + 1);
          expect(placed.races[army]!.hand).toBe(0);
          expect(() => applyAction(g, { type: 'place', tokens: { [own[0]!]: total + 1 } })).toThrow();
          expect(() => applyAction(g, { type: 'place', tokens: { [own[0]!]: 0 } })).toThrow();
          checked++;
        }
        g = applyAction(g, aiAction(g));
      }
    }
    expect(checked).toBeGreaterThan(10);
  });
});
