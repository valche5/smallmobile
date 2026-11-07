import { describe, expect, test } from 'bun:test';
import { applyAction, createGame } from '../src/engine/game';
import { PRESETS } from '../src/engine/maps';
import {
  abandonBrief,
  attackBrief,
  conquestBlock,
  costLine,
  declineBrief,
  endTurnReminder,
  foeThreat,
  foeThreatReason,
  foeThreatText,
  powerStatus,
  redeployNote,
} from '../src/engine/preview';
import { analyzeTarget, attackersOf, isWater, threat } from '../src/engine/rules';
import type { GameState, PowerId, RaceId } from '../src/engine/types';

function game(n = 2): GameState {
  const g = createGame({
    players: Array.from({ length: n }, (_, i) => ({ name: ['Moi', 'Bob', 'Carla', 'Dan'][i]!, color: `#00000${i}`, kind: 'human' as const, ai: 'normal' as const })),
    mapSeed: PRESETS[n]!.seed,
    seed: 7,
  });
  // on part d'une carte vide (sans tribus oubliées) pour maîtriser les voisinages
  g.regions.forEach((r) => Object.assign(r, { race: null, tokens: 0 }));
  return g;
}

/** Donne un peuple actif au joueur. */
function give(g: GameState, pid: number, race: RaceId, power: PowerId, declined = false) {
  g.races[race] = { race, power, owner: pid, declined, hand: 0, holesPlaced: 0, pickedRound: 1 };
  if (declined) g.players[pid]!.declined.push(race);
  else g.players[pid]!.active = race;
}

function put(g: GameState, region: number, race: RaceId, tokens: number) {
  Object.assign(g.regions[region]!, { race, tokens });
}

const land = (g: GameState, id: number) => !isWater(g.map.regions[id]!);

/** Région intérieure simple (ni montagne, ni souterrain, ni côte, ni bord) dont tous les voisins sont des terres. */
function interior(g: GameState): number {
  return g.map.regions.find(
    (r) =>
      !r.entry &&
      !r.border &&
      !r.coastal &&
      !r.cavern &&
      r.terrain !== 'mountain' &&
      r.terrain !== 'farmland' &&
      r.terrain !== 'hill' &&
      !isWater(r) &&
      r.adj.every((a) => land(g, a)),
  )!.id;
}

/** Une région terrestre à au moins 2 pas de `region`. */
function far(g: GameState, region: number): number {
  const near = new Set([region, ...g.map.regions[region]!.adj]);
  return g.map.regions.find((r) => !near.has(r.id) && !isWater(r) && !r.adj.includes(region) && !r.adj.some((a) => g.map.regions[region]!.adj.includes(a)) && !r.cavern)!.id;
}

describe('menace ennemie (foeThreat)', () => {
  test('Commandos voisins : coût 3 sur une plaine à 2 jetons, modificateur 🗡️ −1', () => {
    const g = game();
    give(g, 0, 'humans', 'alchemist');
    give(g, 1, 'ratmen', 'commando');
    const r = interior(g);
    put(g, r, 'humans', 2);
    put(g, g.map.regions[r]!.adj[0]!, 'ratmen', 3);
    const t = foeThreat(g, r, 0)!;
    expect(t.pid).toBe(1);
    expect(t.cost).toBe(3);
    expect(t.how).toBe('conquer');
    expect(t.mod).toBe('🗡️ −1');
    expect(t.far).toBe(false);
    expect(foeThreatText(g, r, 0).text).toBe('⚔ Bob la prend avec 3 (🗡️ −1)');
    expect(foeThreatReason(g, r, 0)).toBe('Bob : 3 jetons à côté');
  });

  test('Volants sans voisinage : menace de loin', () => {
    const g = game();
    give(g, 0, 'humans', 'alchemist');
    give(g, 1, 'ratmen', 'flying');
    const r = interior(g);
    put(g, r, 'humans', 2);
    put(g, far(g, r), 'ratmen', 2);
    const t = foeThreat(g, r, 0);
    expect(t).not.toBeNull();
    expect(t!.far).toBe(true);
    expect(t!.via).toBe('flying');
    expect(foeThreatText(g, r, 0).text).toBe("🦅 Bob la prend avec 4, de n'importe où");
    expect(foeThreatReason(g, r, 0)).toBe('Bob a des Volants');
  });

  test('Sorciers : région de mon peuple actif à 1 jeton, prise gratuite', () => {
    const g = game();
    give(g, 0, 'humans', 'alchemist');
    give(g, 1, 'sorcerers', 'alchemist');
    const r = interior(g);
    put(g, r, 'humans', 1);
    put(g, g.map.regions[r]!.adj[0]!, 'sorcerers', 2);
    const t = foeThreat(g, r, 0)!;
    expect(t.how).toBe('sorcery');
    expect(t.cost).toBe(0);
    expect(foeThreatText(g, r, 0).text).toBe('🔮 Bob la prend gratis : 1 seul jeton');
    // à 2 jetons, la sorcellerie ne passe plus
    put(g, r, 'humans', 2);
    expect(foeThreat(g, r, 0)!.how).toBe('conquer');
  });

  test('Dompteurs : prise avec 1 jeton', () => {
    const g = game();
    give(g, 0, 'humans', 'alchemist');
    give(g, 1, 'ratmen', 'dragonMaster');
    const r = interior(g);
    put(g, r, 'humans', 4);
    put(g, g.map.regions[r]!.adj[0]!, 'ratmen', 1);
    const t = foeThreat(g, r, 0)!;
    expect(t.how).toBe('dragon');
    expect(t.cost).toBe(1);
    expect(foeThreatText(g, r, 0).text).toBe('🐉 Bob la prend avec 1 jeton (dragon)');
  });

  test('aucun ennemi à portée : null et texte en gris', () => {
    const g = game();
    give(g, 0, 'humans', 'alchemist');
    give(g, 1, 'ratmen', 'alchemist');
    const r = interior(g);
    put(g, r, 'humans', 2);
    put(g, far(g, r), 'ratmen', 2);
    expect(foeThreat(g, r, 0)).toBeNull();
    expect(foeThreatText(g, r, 0)).toEqual({ text: '✓ Hors de portée des ennemis', muted: true, color: null });
  });

  test('héros : imprenable ; paix imposée : pas d’attaquant', () => {
    const g = game();
    give(g, 0, 'humans', 'heroic');
    give(g, 1, 'ratmen', 'alchemist');
    const r = interior(g);
    put(g, r, 'humans', 2);
    put(g, g.map.regions[r]!.adj[0]!, 'ratmen', 2);
    expect(foeThreat(g, r, 0)).not.toBeNull();
    g.peace.push({ diplomat: 0, target: 1 });
    expect(attackersOf(g, r, 0)).toEqual([]);
    g.peace = [];
    g.regions[r]!.hero = true;
    expect(foeThreat(g, r, 0)).toBeNull();
    expect(foeThreatText(g, r, 0).text).toBe('✓ Imprenable (héros)');
  });

  test('Goules en déclin : comptent comme attaquants', () => {
    const g = game();
    give(g, 0, 'humans', 'alchemist');
    give(g, 1, 'ghouls', 'alchemist', true);
    const r = interior(g);
    put(g, r, 'humans', 2);
    put(g, g.map.regions[r]!.adj[0]!, 'ghouls', 3);
    const t = foeThreat(g, r, 0)!;
    expect(t.race).toBe('ghouls');
    expect(t.cost).toBe(4);
  });

  test('égalité : celui qui joue le plus tôt après moi', () => {
    const g = game(3);
    give(g, 1, 'humans', 'alchemist');
    give(g, 0, 'ratmen', 'alchemist');
    give(g, 2, 'orcs', 'alchemist');
    const r = interior(g);
    put(g, r, 'humans', 2);
    const [a, b] = g.map.regions[r]!.adj;
    put(g, a!, 'ratmen', 2);
    put(g, b!, 'orcs', 2);
    // pendant mon redéploiement (joueur 1), le joueur 2 joue avant le joueur 0
    g.current = 1;
    g.phase = 'redeploy';
    g.turn.army = 'humans';
    expect(foeThreat(g, r, 1)!.pid).toBe(2);
  });

  test('threat() : une région intérieure face à des Volants est menacée', () => {
    const g = game();
    give(g, 0, 'humans', 'alchemist');
    const r = interior(g);
    put(g, r, 'humans', 2);
    for (const a of g.map.regions[r]!.adj) put(g, a, 'humans', 1);
    give(g, 1, 'ratmen', 'alchemist');
    const spot = g.map.regions.find((x) => !isWater(x) && x.id !== r && !g.map.regions[r]!.adj.includes(x.id) && !x.adj.includes(r) && g.regions[x.id]!.race === null && !x.adj.some((y) => g.regions[y]!.race === 'humans'))!.id;
    put(g, spot, 'ratmen', 2);
    expect(threat(g, 'humans', r)).toBe(0);
    g.races.ratmen!.power = 'flying';
    expect(threat(g, 'humans', r)).toBeGreaterThan(0);
  });
});

describe('fiche d’attaque et barre', () => {
  /** Pire cas : je joue des Humains Diplomates, Bob a des Elfes Fortifiés sur une Ferme avec forteresse. */
  function worst() {
    const g = game();
    give(g, 0, 'humans', 'diplomat');
    give(g, 1, 'elves', 'fortified');
    g.races.humans!.hand = 9;
    g.phase = 'conquest';
    g.turn.army = 'humans';
    const farm = g.map.regions.find((r) => r.terrain === 'farmland' && r.adj.some((a) => land(g, a) && g.map.regions[a]!.terrain !== 'mountain'))!;
    const mine = farm.adj.find((a) => land(g, a))!;
    put(g, mine, 'humans', 1);
    put(g, farm.id, 'elves', 2);
    g.regions[farm.id]!.fortress = true;
    return { g, farm: farm.id };
  }

  test('pire cas : formule, 3 conséquences d’une ligne', () => {
    const { g, farm } = worst();
    const info = analyzeTarget(g, farm);
    expect(info.kind).toBe('normal');
    expect(costLine(info.parts, info.effective)).toBe('Coût 2 + 2 jetons + 🏰 1 = 5');
    const brief = attackBrief(g, farm, 'conquer');
    expect(brief.lines.map((l) => `${l.icon} ${l.text}`)).toEqual([
      '🕊️ Plus de paix possible avec Bob ce tour',
      '🧝 Elfes : Bob ne perd aucun jeton',
      '💰 +2 ce tour (Ferme +1)',
    ]);
    for (const l of brief.lines) expect([...`${l.icon} ${l.text}`].length).toBeLessThanOrEqual(46);
  });

  test('cas simple : pas de ligne « Coût », aucune conséquence', () => {
    const g = game();
    give(g, 0, 'ratmen', 'alchemist');
    give(g, 1, 'orcs', 'alchemist');
    g.races.ratmen!.hand = 7;
    g.phase = 'conquest';
    g.turn.army = 'ratmen';
    const r = interior(g);
    put(g, g.map.regions[r]!.adj[0]!, 'ratmen', 1);
    put(g, r, 'orcs', 1);
    const info = analyzeTarget(g, r);
    expect(info.effective).toBe(3);
    expect(costLine(info.parts, info.effective)).toBeNull();
    expect(attackBrief(g, r, 'conquer').lines).toEqual([]);
  });

  test('raisons de blocage : textes exacts', () => {
    const g = game();
    give(g, 0, 'ratmen', 'alchemist');
    give(g, 1, 'orcs', 'alchemist');
    g.races.ratmen!.hand = 2;
    g.phase = 'conquest';
    g.turn.army = 'ratmen';
    const r = interior(g);
    expect(conquestBlock(g, r)).toBe('⛔ Première conquête : par le bord de la carte');
    put(g, g.map.regions[r]!.adj[0]!, 'ratmen', 1);
    expect(conquestBlock(g, far(g, r))).toBe('⛔ Pas voisine de vos régions');
    put(g, r, 'orcs', 4);
    expect(conquestBlock(g, r)).toBe('⛔ Il faut 6 jetons, vous en avez 2');
    g.peace.push({ diplomat: 1, target: 0 });
    expect(conquestBlock(g, r)).toBe("🕊️ Paix : Bob vous interdit d'attaquer ses Orques");
  });

  test('état du pouvoir : seulement ce qui change', () => {
    const g = game();
    give(g, 0, 'ratmen', 'dragonMaster');
    g.phase = 'conquest';
    g.turn.army = 'ratmen';
    expect(powerStatus(g, 0)).toBeNull(); // main vide : le dragon ne peut pas partir
    g.races.ratmen!.hand = 3;
    expect(powerStatus(g, 0)).toBe('🐉 Dragon prêt');
    g.turn.dragonUsed = true;
    expect(powerStatus(g, 0)).toBeNull();
    g.races.ratmen!.power = 'bivouacking';
    g.phase = 'redeploy';
    expect(powerStatus(g, 0)).toBeNull(); // aucune région où poser un campement
    put(g, interior(g), 'ratmen', 2);
    expect(powerStatus(g, 0)).toBe('⛺ Placez ou déplacez des bivouacs · 5 en réserve');
    expect(powerStatus(g, 0, true)).toBe('⛺ 5 en réserve'); // jetons encore à placer : version courte
    g.races.ratmen!.power = 'alchemist';
    expect(powerStatus(g, 0)).toBeNull();
  });

  test('fin du tour : premier oubli de pouvoir', () => {
    const g = game();
    give(g, 0, 'ratmen', 'fortified');
    g.phase = 'redeploy';
    g.turn.army = 'ratmen';
    put(g, interior(g), 'ratmen', 3);
    expect(endTurnReminder(g)).toBe('⚠️ Forteresse pas encore bâtie');
    g.races.ratmen!.power = 'commando';
    expect(endTurnReminder(g)).toBeNull();
  });

  test('abandon : la dernière région n’inquiète pas les Volants', () => {
    const g = game();
    give(g, 0, 'ratmen', 'alchemist');
    g.phase = 'conquest';
    g.turn.army = 'ratmen';
    const r = interior(g);
    put(g, r, 'ratmen', 3);
    expect(abandonBrief(g, r).alert).toBe('⚠️ Dernière région : il faudra repartir du bord');
    g.races.ratmen!.power = 'flying';
    expect(abandonBrief(g, r).alert).toBeNull();
  });

  test('déclin : 2 lignes fixes, puis 2 au plus', () => {
    const g = game();
    give(g, 0, 'trolls', 'stout');
    give(g, 0, 'dwarves', 'alchemist', true);
    g.phase = 'conquest';
    g.turn.army = 'trolls';
    g.round = g.maxRounds;
    const r = interior(g);
    const [a, b] = g.map.regions[r]!.adj;
    put(g, r, 'trolls', 3);
    put(g, a!, 'trolls', 2);
    put(g, b!, 'dwarves', 1);
    const brief = declineBrief(g);
    expect(brief.lines.map((l) => l.text)).toEqual([
      "+2 maintenant, puis votre tour s'arrête.",
      'Vos 2 régions gardent 1 jeton chacune.',
      'Vos 🪓 Nains en déclin disparaissent.',
      'Dernier tour : pas de nouveau peuple ensuite.',
    ]);
    expect(brief.more.some((l) => l.text.startsWith('Astuce'))).toBe(true);
  });
});

describe('pastilles de carte (computeMarks)', () => {
  test('redéploiement : prix de l’ennemi le plus dangereux, anneau à sa couleur', async () => {
    const { computeMarks } = await import('../src/ui/views/marks');
    const g = game();
    g.players[1]!.kind = 'ai';
    give(g, 0, 'humans', 'alchemist');
    give(g, 1, 'ratmen', 'commando');
    const r = interior(g);
    const quiet = far(g, r);
    put(g, r, 'humans', 2);
    put(g, quiet, 'humans', 2);
    put(g, g.map.regions[r]!.adj[0]!, 'ratmen', 3);
    g.phase = 'redeploy';
    g.turn.army = 'humans';
    const marks = computeMarks(g, 0, true);
    expect(marks.get(r)).toEqual({ kind: 'threat', label: '⚔3', ring: '#000001', hot: true });
    expect(marks.get(quiet)!.kind).toBe('own');
    expect(marks.get(quiet)!.label).toBeUndefined();
    // hors de mon tour : aucune pastille
    expect([...computeMarks(g, 0, false).values()].some((m) => m.label)).toBe(false);
  });

  test('régions perdues : signalées hors de mon redéploiement', async () => {
    const { computeMarks } = await import('../src/ui/views/marks');
    const g = game();
    g.players[1]!.kind = 'ai';
    give(g, 0, 'humans', 'alchemist');
    give(g, 1, 'ratmen', 'alchemist');
    const r = interior(g);
    put(g, r, 'ratmen', 3);
    put(g, g.map.regions[r]!.adj[0]!, 'humans', 2);
    g.log.push({ seq: ++g.seq, round: g.round, player: 1, kind: 'conquer', text: '', region: r, victim: 0 });
    g.current = 1;
    g.phase = 'conquest';
    g.turn.army = 'ratmen';
    expect(computeMarks(g, 0, false).get(r)).toEqual({ kind: 'plain', lost: true });
    g.current = 0;
    g.phase = 'redeploy';
    g.turn.army = 'humans';
    expect(computeMarks(g, 0, true).get(r)?.lost).toBeUndefined();
  });

  test('Amazones : les jetons rendus sont dits une fois, au redéploiement', () => {
    let g = game();
    give(g, 0, 'amazons', 'alchemist');
    g.races.amazons!.hand = 10;
    g.phase = 'conquest';
    g.turn.army = 'amazons';
    g.turn.amazonBonus = 4;
    put(g, interior(g), 'amazons', 1);
    g = applyAction(g, { type: 'endConquest' });
    expect(redeployNote(g)).toBe('🏹 4 rendus');
  });
});
