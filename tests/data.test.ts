import { describe, expect, test } from 'bun:test';
import { EXPANSION_IDS, FEATURE_ICONS, KIND_ORDER, POWER_IDS, POWERS, RACE_IDS, RACES, powersIn, racesIn, tagsOf } from '../src/engine/data';

const ALL = [...RACE_IDS.map((id) => RACES[id]), ...POWER_IDS.map((id) => POWERS[id])];

describe('étiquettes des peuples et pouvoirs', () => {
  test('14 peuples et 20 pouvoirs dans le jeu de base, 24 et 32 avec les extensions', () => {
    expect(racesIn().length).toBe(14);
    expect(powersIn().length).toBe(20);
    expect(RACE_IDS.length).toBe(24);
    expect(POWER_IDS.length).toBe(32);
    // les numéros du noyau : le jeu de base d'abord
    expect(RACE_IDS.slice(0, 14)).toEqual(racesIn());
    expect(POWER_IDS.slice(0, 20)).toEqual(powersIn());
    expect(racesIn(EXPANSION_IDS)).toEqual(RACE_IDS);
    expect(powersIn(EXPANSION_IDS)).toEqual(POWER_IDS);
    expect(racesIn(['beNotAfraid']).slice(14)).toEqual(['barbarians', 'leprechauns', 'pixies', 'homunculi', 'pygmies']);
  });

  test('tag, foe et kind présents et assez courts', () => {
    for (const d of ALL) {
      expect(d.tag.trim().length, d.name).toBeGreaterThan(0);
      expect(d.foe.trim().length, d.name).toBeGreaterThan(0);
      expect(KIND_ORDER, d.name).toContain(d.kind);
      expect([...d.tag].length, `${d.name} : ${d.tag}`).toBeLessThanOrEqual(18);
      expect([...d.foe].length, `${d.name} : ${d.foe}`).toBeLessThanOrEqual(52);
    }
  });

  test('56 icônes distinctes, ni 💰 ni 🛡️', () => {
    const icons = ALL.map((d) => d.icon);
    expect(new Set(icons).size).toBe(56);
    expect(icons).not.toContain('💰');
    expect(icons).not.toContain('🛡️');
    expect(POWERS.merchant.icon).toBe('⚖️');
    expect(POWERS.heroic.icon).toBe('🦸');
    expect(FEATURE_ICONS.hero).toBe('🦸');
  });

  test('foe raccourcis', () => {
    expect(POWERS.dragonMaster.foe).toBe('1 seul jeton suffit (dragon), région imprenable');
    expect(RACES.sorcerers.foe).toBe('Prennent gratis vos régions gardées par 1 jeton');
    expect(RACES.orcs.foe).toBe("+1 💰 pour chaque région occupée qu'ils prennent");
    expect(POWERS.pillaging.foe).toBe(RACES.orcs.foe);
    expect(POWERS.flying.foe).toBe('Attaquent partout : la distance ne protège pas');
    expect(POWERS.diplomat.foe).toBe('Peuvent vous interdire de les attaquer 1 tour');
  });
});

describe('tagsOf', () => {
  test('attaque, puis défense, spécial, revenu', () => {
    // Elfes (défense) + Volants (attaque) : le pouvoir passe devant
    expect(tagsOf('elves', 'flying')).toEqual([
      { power: 'flying', icon: '🦅', tag: 'attaquent partout', kind: 'attaque' },
      { race: 'elves', icon: '🧝', tag: 'aucune perte', kind: 'défense' },
    ]);
    // Humains (revenu) + Marins (spécial)
    expect(tagsOf('humans', 'seafaring').map((t) => t.kind)).toEqual(['spécial', 'revenu']);
    // Bivouaqueurs (défense) + Nains (revenu)
    expect(tagsOf('dwarves', 'bivouacking').map((t) => t.icon)).toEqual(['⛺', '🪓']);
  });

  test('même nature : le peuple d\'abord', () => {
    expect(tagsOf('sorcerers', 'flying').map((t) => t.icon)).toEqual(['🔮', '🦅']);
    expect(tagsOf('humans', 'merchant').map((t) => t.tag)).toEqual(['+1 💰 par Ferme', '+1 💰 par région']);
  });

  test('toujours 2 étiquettes ordonnées, pour toutes les combinaisons', () => {
    for (const r of RACE_IDS)
      for (const p of POWER_IDS) {
        const t = tagsOf(r, p);
        expect(t.length).toBe(2);
        expect(KIND_ORDER.indexOf(t[0]!.kind)).toBeLessThanOrEqual(KIND_ORDER.indexOf(t[1]!.kind));
      }
  });
});
