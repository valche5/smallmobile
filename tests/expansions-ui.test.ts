/** Ce que l'interface montre des extensions : prix du marché, aperçus, carte, récits du tour adverse. */
import { describe, expect, test } from 'bun:test';
import { EXPANSION_IDS, expansionBlurb, pickPrice, startingTokens } from '../src/engine/data';
import { applyAction, createGame } from '../src/engine/game';
import { PRESETS } from '../src/engine/maps';
import { abandonBrief, declineBrief, immuneKind, noRedeployText, previewConquest, towerPreview } from '../src/engine/preview';
import { analyzeTarget, cloneState } from '../src/engine/rules';
import type { GameState, LogEntry, PowerId, RaceId } from '../src/engine/types';
import { narrate } from '../src/ui/narrate';
import { computeMarks } from '../src/ui/views/marks';

function game(n = 2): GameState {
  return createGame({
    players: Array.from({ length: n }, (_, i) => ({ name: `J${i + 1}`, color: '#000', kind: 'human' as const, ai: 'normal' as const })),
    mapSeed: PRESETS[n]!.seed,
    seed: 7,
    expansions: [...EXPANSION_IDS],
  });
}

/** Place la combinaison voulue en tête du marché puis la choisit. */
function pick(g: GameState, race: RaceId, power: PowerId): GameState {
  const c = cloneState(g);
  c.market = c.market.filter((m) => m.race !== race && m.power !== power);
  c.raceDeck = c.raceDeck.filter((r) => r !== race);
  c.powerDeck = c.powerDeck.filter((p) => p !== power);
  while (c.market.length >= 6) {
    const last = c.market.pop()!;
    c.raceDeck.unshift(last.race);
    c.powerDeck.unshift(last.power);
  }
  c.market.unshift({ race, power, coins: 0 });
  return applyAction(c, { type: 'pick', index: 0 });
}

/** Régions terrestres en bord de carte, vides, sans montagne. */
function entries(g: GameState): number[] {
  return g.map.regions.filter((r) => r.entry && !r.lostTribe && r.terrain !== 'mountain' && g.regions[r.id]!.race === null).map((r) => r.id);
}

/** Deux régions voisines prenables d'entrée : une en bord de carte, l'autre à côté. */
function pair(g: GameState): [number, number] {
  for (const a of entries(g)) {
    const b = g.map.regions[a]!.adj.find((x) => {
      const d = g.map.regions[x]!;
      return d.terrain !== 'sea' && d.terrain !== 'lake' && d.terrain !== 'mountain' && !d.lostTribe && g.regions[x]!.race === null;
    });
    if (b !== undefined) return [a, b];
  }
  throw new Error('pas de paire');
}

describe('extensions : ce que montre l’interface', () => {
  test('extensions décrites en bref', () => {
    expect(expansionBlurb('beNotAfraid')).toBe('5 peuples, 5 pouvoirs');
    expect(expansionBlurb('cursed')).toBe('2 peuples, 5 pouvoirs');
    expect(expansionBlurb('grandDames')).toBe('3 peuples, 2 pouvoirs');
  });

  test('marché : 3 pièces pour passer des Maudits, jetons de départ justes', () => {
    const market = [
      { race: 'humans', power: 'cursed', coins: 0 },
      { race: 'homunculi', power: 'alchemist', coins: 0, tokens: 3 },
      { race: 'orcs', power: 'hordes', coins: 0 },
    ] as const;
    expect([0, 1, 2].map((i) => pickPrice(market, i))).toEqual([0, 3, 4]);
    // Homoncules : 5 + 4 + 3 ajoutés ; de la Horde : 5 + 5 + 2
    expect(startingTokens(market[1])).toBe(12);
    expect(startingTokens(market[2])).toBe(12);
  });

  test('Mercenaires : la chance au dé compte le rattrapage payé', () => {
    let g = pick(game(), 'humans', 'mercenary');
    const r = entries(g)[0]!;
    g = cloneState(g);
    g.races.humans!.hand = 1;
    const t = analyzeTarget(g, r);
    expect(t.kind).toBe('die');
    expect(t.free).toBe(0.5);
    // il manque 1 : 1 pièce rattrape n'importe quel dé
    expect(t.chance).toBe(1);
  });

  test('Kobolds dompteurs de dragon : le dragon coûte 2 jetons', () => {
    const g = pick(game(), 'kobolds', 'dragonMaster');
    const r = entries(g)[0]!;
    expect(previewConquest(g, r, 'dragon')!.tokens).toBe(2);
    expect(computeMarks(g, 0, true).get(r)!.label).toBe('⚔2');
  });

  test('Catapultés : les cibles hors de portée normale ont leur pastille', () => {
    let g = pick(game(), 'humans', 'catapult');
    g = applyAction(g, { type: 'conquer', region: entries(g)[0]! });
    const marks = computeMarks(g, 0, true);
    expect([...marks.values()].some((m) => m.label?.startsWith('☄️'))).toBe(true);
  });

  test("Prêtresses : la Tour d'Ivoire proposée, puis celle choisie, dans l'aperçu du déclin", () => {
    let g = pick(game(), 'priestesses', 'merchant');
    const [a, b] = pair(g);
    g = applyAction(g, { type: 'conquer', region: a! });
    g = applyAction(g, { type: 'conquer', region: b! });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    g = applyAction(g, { type: 'endTurn' });
    // tour de J2, puis retour à J1 qui peut décliner
    g = pick(g, 'humans', 'alchemist');
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'endTurn' });
    expect(g.current).toBe(0);
    const proposed = towerPreview(g)!;
    expect(proposed.chosen).toBe(false);
    expect(proposed.tokens).toBe(2);
    const other = proposed.region === a ? b! : a!;
    g = applyAction(g, { type: 'tower', region: other });
    const chosen = towerPreview(g)!;
    expect(chosen).toEqual({ region: other, tokens: 2, chosen: true });
    expect(declineBrief(g).lines.some((l) => l.icon === '🗼' && l.text.includes(g.map.regions[other]!.name.slice(0, 10)))).toBe(true);
  });

  test('Dames Blanches en déclin : imprenables', () => {
    let g = pick(game(), 'whiteLadies', 'merchant');
    const r = entries(g)[0]!;
    g = applyAction(g, { type: 'conquer', region: r });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    g = applyAction(g, { type: 'endTurn' });
    g = pick(g, 'humans', 'alchemist');
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'endTurn' });
    g = applyAction(g, { type: 'decline' });
    expect(immuneKind(g.regions[r]!)).toBe('Dames Blanches');
  });

  test('Barbares : rien à replacer, et pourquoi', () => {
    let g = pick(game(), 'barbarians', 'alchemist');
    g = applyAction(g, { type: 'conquer', region: entries(g)[0]! });
    g = applyAction(g, { type: 'endConquest' });
    expect(noRedeployText(g)).toMatch(/^⚔️ Barbares : jamais redéployés · \d+ hors plateau$/);
  });

  test('Gitanes : l’abandon annonce sa pièce', () => {
    let g = pick(game(), 'gypsies', 'alchemist');
    const [a, b] = pair(g);
    g = applyAction(g, { type: 'conquer', region: a! });
    g = applyAction(g, { type: 'conquer', region: b! });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    g = applyAction(g, { type: 'endTurn' });
    g = pick(g, 'humans', 'alchemist');
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'endTurn' });
    expect(abandonBrief(g, a!).alert).toBe('🃏 +1 💰 au décompte, pas de retour');
    g = applyAction(g, { type: 'abandon', region: a! });
    const entry = g.log.at(-1)!;
    expect(entry.race).toBe('gypsies');
    expect(narrate(g, [entry], 1).main).toContain('+1 💰');
    expect(computeMarks(g, 0, true).get(a!)).toEqual({ kind: 'none', label: '🚫' });
  });

  test('récits du tour adverse : vol, pot-de-vin, dé des Pygmées', () => {
    const g = game(3);
    const pay = (kind: NonNullable<LogEntry['pay']>['kind'], from: number | null, to: number | null, player: number | null): LogEntry => ({
      seq: 1, round: 1, player, kind: 'power', text: '', pay: { kind, from, to, coins: 1 }, ...(kind === 'theft' ? { victim: from! } : {}),
    });
    const theft = narrate(g, [pay('theft', 1, 0, 0)], 1);
    expect(theft).toMatchObject({ main: '🦹 vous vole 1 💰', tone: 'hit' });
    expect(narrate(g, [pay('theft', 1, 0, 0)], 2).main).toBe('🦹 vole 1 💰 à J2');
    expect(narrate(g, [pay('bribe', 0, 1, 0)], 1).main).toBe('💸 vous verse 1 💰 (Corrompus)');
    // le pot-de-vin ne doit pas passer pour une paix des Diplomates
    expect(narrate(g, [pay('bribe', 0, 1, 0)], 2).main).not.toContain('paix');
    const pygmies: LogEntry = { seq: 2, round: 1, player: 1, kind: 'power', text: '', race: 'pygmies', tokens: 2, roll: 2 };
    expect(narrate(g, [pygmies], 1)).toMatchObject({ main: '🪃 dé des Pygmées : 2', detail: '2 Pygmées vous reviennent en main' });
  });
});
