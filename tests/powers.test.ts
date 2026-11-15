import { describe, expect, test } from 'bun:test';
import { EXPANSION_IDS } from '../src/engine/data';
import { applyAction, createGame, narrate } from '../src/engine/game';
import { actionWords, readAction, readEvents, readState, stateWords } from '../src/engine/kernel';
import { engine } from '../src/engine/engine';
import { PRESETS } from '../src/engine/maps';
import { previewConquest, previewDecline, previewEndTurn } from '../src/engine/preview';
import {
  analyzeTarget,
  canChooseTower,
  canRegroup,
  catapultShots,
  cloneState,
  defenseValue,
  regionsOf,
  scoreBreakdown,
} from '../src/engine/rules';
import type { Action, GameState, PowerId, RaceId } from '../src/engine/types';

function game(n = 2, expansions = false): GameState {
  return createGame({
    players: Array.from({ length: n }, (_, i) => ({ name: `J${i + 1}`, color: '#000', kind: 'human' as const, ai: 'normal' as const })),
    mapSeed: PRESETS[n]!.seed,
    seed: 7,
    ...(expansions ? { expansions: [...EXPANSION_IDS] } : {}),
  });
}

/** Place la combinaison voulue en tête du marché puis la choisit. */
function pick(g: GameState, race: RaceId, power: PowerId): GameState {
  const c = cloneState(g);
  c.market = c.market.filter((m) => m.race !== race && m.power !== power);
  c.raceDeck = c.raceDeck.filter((r) => r !== race);
  c.powerDeck = c.powerDeck.filter((p) => p !== power);
  // le marché garde 6 combinaisons : la dernière retourne sous les pioches
  while (c.market.length >= 6) {
    const last = c.market.pop()!;
    c.raceDeck.unshift(last.race);
    c.powerDeck.unshift(last.power);
  }
  c.market.unshift({ race, power, coins: 0 });
  return applyAction(c, { type: 'pick', index: 0 });
}

/** Région terrestre en bord de carte, vide, sans montagne. */
function entryRegion(g: GameState, exclude: number[] = []): number {
  return g.map.regions.find(
    (r) => r.entry && !r.lostTribe && r.terrain !== 'mountain' && !exclude.includes(r.id) && g.regions[r.id]!.race === null,
  )!.id;
}

/** Termine le tour courant en posant tous les jetons sur la première région. */
function finishTurn(g: GameState, opts: { stout?: boolean; peace?: number | null } = {}): GameState {
  if (g.phase === 'conquest') g = applyAction(g, { type: 'endConquest' });
  g = applyAction(g, { type: 'autoDeploy' });
  g = applyAction(g, { type: 'endTurn', ...opts });
  while (g.phase === 'defend') {
    g = applyAction(g, { type: 'autoDeploy' });
    g = applyAction(g, { type: 'endDefense' });
  }
  return g;
}

/** Donne au joueur courant une région adjacente à `region` pour la tester. */
function neighborOf(g: GameState, region: number): number {
  return g.map.regions[region]!.adj.find((a) => {
    const t = g.map.regions[a]!.terrain;
    return t !== 'sea' && t !== 'lake';
  })!;
}

describe('pouvoirs et peuples', () => {
  test('Amazones : +4 jetons en conquête, rendus au redéploiement', () => {
    let g = pick(game(), 'amazons', 'alchemist');
    expect(g.races.amazons!.hand).toBe(6 + 4 + 4);
    const r = entryRegion(g);
    g = applyAction(g, { type: 'conquer', region: r });
    const before = g.races.amazons!.hand;
    g = applyAction(g, { type: 'endConquest' });
    expect(g.races.amazons!.hand).toBe(before - 4);
  });

  test('Elfes : ne perdent aucun jeton quand ils sont attaqués', () => {
    let g = pick(game(), 'elves', 'alchemist');
    const r = entryRegion(g);
    g = applyAction(g, { type: 'conquer', region: r });
    g = finishTurn(g);
    const tokens = g.regions[r]!.tokens;
    g = pick(g, 'ratmen', 'flying');
    g.races.ratmen!.hand = 20;
    g = applyAction(g, { type: 'conquer', region: r });
    expect(g.races.elves!.hand).toBe(tokens);
  });

  test('Trolls : tanière +1 défense, conservée en déclin', () => {
    let g = pick(game(), 'trolls', 'alchemist');
    const r = entryRegion(g);
    g = applyAction(g, { type: 'conquer', region: r });
    expect(g.regions[r]!.lair).toBe(true);
    const tokens = g.regions[r]!.tokens;
    expect(defenseValue(g, r)).toBe(2 + tokens + 1);
    g = finishTurn(g);
    g = pick(g, 'ratmen', 'alchemist');
    g = finishTurn(g);
    g = applyAction(g, { type: 'decline' });
    expect(g.regions[r]!.lair).toBe(true);
    expect(g.regions[r]!.tokens).toBe(1);
  });

  test('Semi-hommes : entrée partout, 2 terriers imprenables, perdus en déclin', () => {
    let g = pick(game(), 'halflings', 'alchemist');
    const interior = g.map.regions.find((r) => !r.entry && r.terrain !== 'sea' && r.terrain !== 'lake' && !r.lostTribe)!.id;
    expect(analyzeTarget(g, interior).reachable).toBe(true);
    g = applyAction(g, { type: 'conquer', region: interior });
    const second = neighborOf(g, interior);
    g = applyAction(g, { type: 'conquer', region: second });
    expect(g.regions[interior]!.hole && g.regions[second]!.hole).toBe(true);
    expect(g.races.halflings!.holesPlaced).toBe(2);
    g = finishTurn(g);
    g = pick(g, 'ratmen', 'flying');
    g.races.ratmen!.hand = 30;
    const info = analyzeTarget(g, interior);
    expect(info.reachable).toBe(false);
    expect(info.reason).toContain('terrier');
    g = finishTurn(g);
    g = applyAction(g, { type: 'decline' });
    expect(g.regions[interior]!.hole).toBe(false);
  });

  test('Squelettes : +1 jeton pour 2 régions non vides conquises', () => {
    let g = pick(game(), 'skeletons', 'alchemist');
    const lost = g.map.regions.filter((r) => r.entry && r.lostTribe).map((r) => r.id);
    g.races.skeletons!.hand = 12;
    let conquered = 0;
    for (const r of lost) {
      if (analyzeTarget(g, r).kind === 'normal' && conquered < 2) {
        g = applyAction(g, { type: 'conquer', region: r });
        conquered++;
      }
    }
    if (conquered < 2) return;
    const hand = g.races.skeletons!.hand;
    g = applyAction(g, { type: 'endConquest' });
    expect(g.races.skeletons!.hand).toBe(hand + 1);
  });

  test('Fortunés : +7 au premier tour seulement', () => {
    let g = pick(game(), 'ratmen', 'wealthy');
    g = applyAction(g, { type: 'conquer', region: entryRegion(g) });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    expect(scoreBreakdown(g, 0).lines.some((l) => l.value === 7)).toBe(true);
    g = finishTurn(g);
    g = pick(g, 'orcs', 'alchemist');
    g = finishTurn(g);
    expect(scoreBreakdown(g, 0).lines.some((l) => l.value === 7)).toBe(false);
  });

  test('Commandos et Montés réduisent le coût', () => {
    let g = pick(game(), 'ratmen', 'commando');
    const r = entryRegion(g);
    expect(analyzeTarget(g, r).cost).toBe(1);
    g = pick(game(), 'ratmen', 'mounted');
    const farm = g.map.regions.find((x) => x.entry && !x.lostTribe && (x.terrain === 'farmland' || x.terrain === 'hill'))!;
    expect(analyzeTarget(g, farm.id).cost).toBe(1);
  });

  test('Marins : peuvent conquérir la mer', () => {
    let g = pick(game(), 'ratmen', 'seafaring');
    const sea = g.map.regions.find((r) => r.terrain === 'sea')!.id;
    const info = analyzeTarget(g, sea);
    expect(info.reachable).toBe(true);
    g = applyAction(g, { type: 'conquer', region: sea });
    expect(g.regions[sea]!.race).toBe('ratmen');
    const other = pick(game(), 'ratmen', 'alchemist');
    expect(analyzeTarget(other, sea).reachable).toBe(false);
  });

  test('Dompteurs de dragon : 1 jeton, région imprenable', () => {
    let g = pick(game(), 'ratmen', 'dragonMaster');
    const r = g.map.regions.find((x) => x.entry && x.lostTribe)!.id;
    const hand = g.races.ratmen!.hand;
    g = applyAction(g, { type: 'dragon', region: r });
    expect(g.regions[r]!.tokens).toBe(1);
    expect(g.regions[r]!.dragon).toBe(true);
    expect(g.races.ratmen!.hand).toBe(hand - 1);
    expect(analyzeTarget(g, entryRegion(g)).dragon).toBe(false);
    g = finishTurn(g);
    g = pick(g, 'orcs', 'flying');
    expect(analyzeTarget(g, r).reachable).toBe(false);
  });

  test('Sorciers : remplacent un jeton adverse isolé', () => {
    let g = pick(game(), 'ratmen', 'alchemist');
    const r = entryRegion(g);
    g = applyAction(g, { type: 'conquer', region: r });
    g = applyAction(g, { type: 'endConquest' });
    // laisser 1 seul jeton sur r : tout le reste en main ne peut pas être gardé, on crée une 2e région
    const r2 = entryRegion(g, [r]);
    g = cloneState(g);
    g.regions[r2] = { ...g.regions[r2]!, race: 'ratmen', tokens: 1 };
    g.races.ratmen!.hand = 0;
    g.regions[r]!.tokens = 1;
    g = applyAction(g, { type: 'endTurn' });
    g = pick(g, 'sorcerers', 'flying');
    const info = analyzeTarget(g, r);
    expect(info.sorcery).toBe(true);
    const preview = previewConquest(g, r, 'sorcery');
    expect(preview).not.toBeNull();
    g = applyAction(g, { type: 'sorcery', region: r });
    expect(g.regions[r]!.race).toBe('sorcerers');
    expect(g.races.ratmen!.hand).toBe(0);
    // une seule fois par adversaire et par tour
    expect(analyzeTarget(g, r2).sorcery).toBe(false);
  });

  test('Diplomates : la cible ne peut pas attaquer leur peuple actif', () => {
    let g = pick(game(), 'ratmen', 'diplomat');
    const r = entryRegion(g);
    g = applyAction(g, { type: 'conquer', region: r });
    g = finishTurn(g);
    g = pick(g, 'orcs', 'flying');
    g = finishTurn(g);
    // tour 2 du joueur 1 : paix imposée au joueur 2
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    const pv = previewEndTurn(g, { peace: 1 });
    expect(pv.lines.some((l) => l.text.includes('ne pourra pas attaquer'))).toBe(true);
    g = applyAction(g, { type: 'endTurn', peace: 1 });
    expect(g.current).toBe(1);
    g.races.orcs!.hand = 30;
    const info = analyzeTarget(g, r);
    expect(info.reachable).toBe(false);
    expect(info.reason).toContain('Paix');
    g = finishTurn(g);
    expect(g.peace.length).toBe(0);
  });

  test('Berserks : le dé réduit le coût de la conquête suivante', () => {
    let g = pick(game(), 'ratmen', 'berserk');
    g = applyAction(g, { type: 'berserkRoll' });
    const die = g.turn.berserkDie!;
    const r = g.map.regions.find((x) => x.entry && x.lostTribe)!.id;
    const info = analyzeTarget(g, r);
    expect(info.effective).toBe(Math.max(1, info.cost - die));
    expect(applyAction.bind(null, g, { type: 'decline' })).toThrow();
  });

  test('Spirituels : deux peuples en déclin à la fois', () => {
    let g = pick(game(), 'ratmen', 'alchemist');
    g = applyAction(g, { type: 'conquer', region: entryRegion(g) });
    g = finishTurn(g);
    g = pick(g, 'orcs', 'alchemist');
    g = finishTurn(g);
    g = applyAction(g, { type: 'decline' }); // J1 : Hommes-rats en déclin
    g = finishTurn(g); // J2
    g = pick(g, 'elves', 'spirit'); // J1
    g = applyAction(g, { type: 'conquer', region: entryRegion(g) });
    g = finishTurn(g);
    g = finishTurn(g); // J2
    const pv = previewDecline(g);
    expect(pv.lines.some((l) => l.text.includes('Spirituels'))).toBe(true);
    g = applyAction(g, { type: 'decline' });
    expect(g.players[0]!.declined.sort()).toEqual(['elves', 'ratmen']);
  });

  test('Robustes : déclin en fin de tour, après le décompte', () => {
    let g = pick(game(), 'ratmen', 'stout');
    g = applyAction(g, { type: 'conquer', region: entryRegion(g) });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    const coins = g.players[0]!.coins;
    const expected = scoreBreakdown(g, 0).total;
    g = applyAction(g, { type: 'endTurn', stout: true });
    expect(g.players[0]!.coins).toBe(coins + expected);
    expect(g.races.ratmen!.declined).toBe(true);
    expect(g.players[0]!.active).toBeNull();
  });

  test('Goules : gardent leurs jetons en déclin et jouent avant le peuple actif', () => {
    let g = pick(game(), 'ghouls', 'alchemist');
    const r = entryRegion(g);
    g = applyAction(g, { type: 'conquer', region: r });
    g = finishTurn(g);
    g = pick(g, 'orcs', 'alchemist');
    g = finishTurn(g);
    const total = regionsOf(g, 'ghouls').reduce((s, i) => s + g.regions[i]!.tokens, 0) + g.races.ghouls!.hand;
    g = applyAction(g, { type: 'decline' });
    const after = regionsOf(g, 'ghouls').reduce((s, i) => s + g.regions[i]!.tokens, 0);
    expect(after).toBe(total);
    g = finishTurn(g); // J2
    expect(g.turn.stage).toBe('ghouls');
    expect(g.turn.army).toBe('ghouls');
    expect(g.phase).toBe('conquest');
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'autoDeploy' });
    g = applyAction(g, { type: 'endTurn' });
    expect(g.turn.stage).toBe('active');
    expect(g.phase).toBe('pick');
  });

  test('Bivouaqueurs, Fortifiés, Héroïques : marqueurs de redéploiement', () => {
    let g = pick(game(), 'ratmen', 'bivouacking');
    const r = entryRegion(g);
    g = applyAction(g, { type: 'conquer', region: r });
    g = applyAction(g, { type: 'endConquest' });
    const d = defenseValue(g, r);
    g = applyAction(g, { type: 'camp', region: r, delta: 1 });
    expect(defenseValue(g, r)).toBe(d + 1);

    g = pick(game(), 'ratmen', 'fortified');
    g = applyAction(g, { type: 'conquer', region: r });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'fortress', region: r });
    expect(g.regions[r]!.fortress).toBe(true);
    expect(scoreBreakdown(g, 0).lines.some((l) => l.label.startsWith('Fortifiés'))).toBe(true);

    g = pick(game(), 'ratmen', 'heroic');
    g = applyAction(g, { type: 'conquer', region: r });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'hero', region: r });
    expect(g.regions[r]!.hero).toBe(true);
  });

  test('aperçu de conquête : cohérent avec le résultat réel', () => {
    let g = pick(game(), 'orcs', 'pillaging');
    const r = g.map.regions.find((x) => x.entry && x.lostTribe)!.id;
    const pv = previewConquest(g, r, 'conquer')!;
    const before = scoreBreakdown(g, 0).total;
    const after = applyAction(g, { type: 'conquer', region: r });
    const gain = scoreBreakdown(after, 0).total - before;
    expect(pv.lines.some((l) => l.text.startsWith(`+${gain} pièce`))).toBe(true);
    expect(gain).toBe(3); // région +1, Orques +1, Pillards +1
  });
});

describe('extensions', () => {
  test('les nouvelles actions passent par le format du noyau', () => {
    const g = game(2, true);
    const actions: Action[] = [
      { type: 'catapult', from: 3, region: 7 },
      { type: 'hire', region: 4 },
      { type: 'regroup' },
      { type: 'cauldron', region: 5 },
      { type: 'tower', region: 6 },
    ];
    for (const a of actions) expect(readAction(actionWords(a, g.regions.length), g)).toEqual(a);
  });

  test("l'état des extensions fait l'aller-retour par le noyau", () => {
    // 39 régions : les régions abandonnées s'écrivent sur deux mots
    let g = pick(game(4, true), 'leprechauns', 'mercenary');
    g = cloneState(g);
    const [a, b] = g.map.regions.filter((r) => r.entry && !r.lostTribe && g.regions[r.id]!.race === null).map((r) => r.id);
    g.regions[a!] = { ...g.regions[a!]!, race: 'leprechauns', tokens: 2, cauldron: true, catapult: true };
    g.regions[b!] = { ...g.regions[b!]!, race: 'leprechauns', tokens: 2, ladies: true };
    g.races.leprechauns = { ...g.races.leprechauns!, aside: 3, cauldrons: 12 };
    g.market[0] = { ...g.market[0]!, tokens: 2 };
    g.turn = { ...g.turn, catapultUsed: true, regrouped: true, abandoned: [1, 35], tower: a!, hire: b! };
    const k = engine();
    k.use(g);
    expect(Array.from(k.stateWords())).toEqual(Array.from(stateWords(g)));
    const back = readState(stateWords(g), g);
    expect(back.regions[a!]).toMatchObject({ cauldron: true, catapult: true, ladies: false });
    expect(back.races.leprechauns).toMatchObject({ aside: 3, cauldrons: 12 });
    expect(back.market[0]!.tokens).toBe(2);
    expect(back.turn).toMatchObject({ catapultUsed: true, regrouped: true, abandoned: [1, 35], tower: a, hire: b });
  });

  test('les nouveaux événements sont lus et racontés', () => {
    const g = game(2, true);
    const words = [7, 9, 1, 14, 1, 3, 2, 8, 10, 1, 15, 0, 0, 1, 1, 8, 10, 1, 15, 2, -1, 0, 1];
    const events = readEvents(Uint32Array.from(words.map((w) => w >>> 0)));
    expect(events).toEqual([
      { seq: 9, round: 1, type: 'pygmies', player: 1, value: 3, tokens: 2 },
      { seq: 10, round: 1, type: 'paid', kind: 'bribe', from: 0, to: 1, coins: 1 },
      { seq: 10, round: 1, type: 'paid', kind: 'cauldron', from: null, to: 0, coins: 1 },
    ]);
    const log = narrate(g, events);
    expect(log.length).toBe(3);
    expect(log[1]!.text).toContain('Corrompus');
    expect(log[2]!.short).toBe('+1 💰 (Chaudron des Farfadets pris)');
  });

  test('Kobolds : une conquête coûte au moins 2', () => {
    let g = pick(game(2, true), 'kobolds', 'commando');
    const r = entryRegion(g);
    const t = analyzeTarget(g, r);
    expect(t.cost).toBe(2);
    expect(t.parts.some((p) => p.label === 'Minimum 2 jetons (Kobolds)')).toBe(true);
    g = applyAction(g, { type: 'conquer', region: r });
    expect(g.regions[r]!.tokens).toBe(2);
  });

  test('Barbares : pas de redéploiement, les jetons restés en main attendent le prochain tour', () => {
    let g = pick(game(2, true), 'barbarians', 'alchemist');
    g = applyAction(g, { type: 'conquer', region: entryRegion(g) });
    const hand = g.races.barbarians!.hand;
    g = applyAction(g, { type: 'endConquest' });
    expect(g.races.barbarians!).toMatchObject({ hand: 0, aside: hand });
    expect(() => applyAction(g, { type: 'autoDeploy', all: true })).toThrow('ne redéploie pas');
    g = applyAction(g, { type: 'endTurn' });
    expect(g.current).toBe(1);
  });

  test('Farfadets : Chaudrons posés au redéploiement', () => {
    let g = pick(game(2, true), 'leprechauns', 'alchemist');
    const r = entryRegion(g);
    g = applyAction(g, { type: 'conquer', region: r });
    g = applyAction(g, { type: 'endConquest' });
    g = applyAction(g, { type: 'cauldron', region: r });
    expect(g.regions[r]!.cauldron).toBe(true);
    expect(g.races.leprechauns!.cauldrons).toBe(15);
  });

  test('Mercenaires : 1 pièce pour 2 jetons de moins', () => {
    let g = pick(game(2, true), 'humans', 'mercenary');
    const r = entryRegion(g);
    const t = analyzeTarget(g, r);
    expect(t).toMatchObject({ cost: 2, hire: true, hired: 1, hireReason: null });
    const coins = g.players[0]!.coins;
    g = applyAction(g, { type: 'hire', region: r });
    expect(g.players[0]!.coins).toBe(coins - 1);
    expect(g.regions[r]!.tokens).toBe(1);
    expect(g.log.some((l) => l.text.includes('Mercenaires'))).toBe(true);
  });

  test('Catapultés : un tir par-dessus une région, qui protège la région de tir', () => {
    let g = pick(game(2, true), 'humans', 'catapult');
    g = applyAction(g, { type: 'conquer', region: entryRegion(g) });
    const shots = catapultShots(g);
    expect(shots.length).toBeGreaterThan(0);
    const { from, region, cost } = shots[0]!;
    expect(g.map.regions[from]!.adj).not.toContain(region);
    g = applyAction(g, { type: 'catapult', from, region });
    expect(g.regions[region]!.tokens).toBe(cost);
    expect(g.regions[from]!.catapult).toBe(true);
    expect(g.log.at(-1)!.how).toBe('catapult');
    expect(catapultShots(g)).toEqual([]);
  });

  test('Intrépides : reprendre ses jetons une fois par tour', () => {
    let g = pick(game(2, true), 'humans', 'marauding');
    expect(canRegroup(g)).toBeNull();
    g = applyAction(g, { type: 'regroup' });
    expect(canRegroup(g)).toContain('déjà repris');
    expect(canChooseTower(g)).toBe('Votre peuple actif ne possède pas ce pouvoir.');
  });
});
