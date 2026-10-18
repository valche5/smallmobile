// Les parties archivées par le serveur (data/games/*.json) : qui a gagné, et le temps de décision
// des IA pendant les vraies parties, par niveau et par nombre de joueurs.
//
//   bun lab/tools/games.ts [data/games]
//   (sur mini : ssh mini.local 'cd ~/repos/smallmobile && ~/.bun/bin/bun lab/tools/games.ts')
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ArchivedGame } from '../../src/server/archive';

const dir = process.argv[2] ?? 'data/games';
const games: ArchivedGame[] = readdirSync(dir)
  .filter((f) => f.endsWith('.json'))
  .sort()
  .map((f) => JSON.parse(readFileSync(join(dir, f), 'utf8')) as ArchivedGame);

const quantile = (xs: number[], q: number) => xs[Math.min(xs.length - 1, Math.floor(q * xs.length))]!;
const secs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

console.log(`${games.length} partie(s) dans ${dir}\n`);
for (const g of games) {
  const coins = g.coins;
  const best = coins ? Math.max(...coins) : null;
  const who = g.players.map((p, i) => {
    const tag = p.kind === 'human' ? p.name : `${p.name} (${p.ai})`;
    return coins ? `${tag} ${coins[i]}${coins[i] === best ? ' 🏆' : ''}` : tag;
  });
  const state = coins ? '' : ` — en cours, ${g.record.moves.length} actions`;
  console.log(`${g.id} : ${who.join(' · ')}${state}`);
}

// temps de décision : une décision = la première action d'un coup d'IA
const groups = new Map<string, { ms: number[]; compute: number[]; fallback: number }>();
for (const g of games) {
  for (const m of g.record.moves) {
    // les coups forcés de l'Expert sont joués sans chercher : ils ne disent rien de son temps de calcul
    if (!m.ai || (m.ai.level === 'expert' && m.ai.iterations === 0 && !m.ai.fallback)) continue;
    const key = `${m.ai.level}, ${g.players.length} j.`;
    const e = groups.get(key) ?? { ms: [], compute: [], fallback: 0 };
    if (m.ai.fallback) e.fallback++;
    else {
      e.ms.push(m.ai.ms);
      if (m.ai.computeMs !== undefined) e.compute.push(m.ai.computeMs);
    }
    groups.set(key, e);
  }
}
if (groups.size) {
  console.log('\nTemps de décision des IA (attente comprise ; calcul seul pour l’Expert) :');
  console.log('niveau, table        décisions   médiane    90 %       max        calcul médian   repli');
  for (const [key, e] of [...groups].sort()) {
    const ms = [...e.ms].sort((a, b) => a - b);
    const compute = [...e.compute].sort((a, b) => a - b);
    const cols = ms.length ? [secs(quantile(ms, 0.5)), secs(quantile(ms, 0.9)), secs(ms[ms.length - 1]!)] : ['-', '-', '-'];
    const c = compute.length ? secs(quantile(compute, 0.5)) : '-';
    console.log(`${key.padEnd(20)} ${String(ms.length).padStart(9)}   ${cols.map((x) => x.padEnd(10)).join(' ')} ${c.padEnd(15)} ${e.fallback}`);
  }
}
