// Les parties archivées par le serveur (data/games/*.json) : qui a gagné (entre crochets, les réseaux de
// l'Expert qui y ont joué), le temps de décision des IA
// par niveau et par nombre de joueurs, et les résultats des LLM (parties de l'API llm, fichiers
// `*_llm-*`, place marquée `llm`) par modèle.
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
    const tag = p.kind === 'human' ? p.name : p.kind === 'llm' ? `${p.name} (LLM ${g.llm?.model ?? '?'})` : `${p.name} (${p.ai})`;
    return coins ? `${tag} ${coins[i]}${coins[i] === best ? ' 🏆' : ''}` : tag;
  });
  const state = coins ? '' : ` — en cours, ${g.record.moves.length} actions`;
  const nets = g.expert ? ` [${g.expert.join(', ')}]` : '';
  console.log(`${g.id}${nets} : ${who.join(' · ')}${state}`);
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

// les LLM : parties finies par modèle et réseaux de l'Expert en face, victoires, place moyenne (d'après
// les pièces), bilans donnés
const models = new Map<string, { games: number; wins: number; places: number; debriefs: number }>();
for (const g of games) {
  if (!g.llm || !g.coins) continue;
  const mine = g.coins[g.llm.seat]!;
  const place = 1 + g.coins.filter((c) => c > mine).length;
  const key = g.expert ? `${g.llm.model} / ${g.expert.join('+')}` : g.llm.model;
  const e = models.get(key) ?? { games: 0, wins: 0, places: 0, debriefs: 0 };
  e.games++;
  e.wins += +(place === 1);
  e.places += place;
  e.debriefs += +!!g.llm.debrief;
  models.set(key, e);
}
if (models.size) {
  console.log('\nLLM (parties finies) :');
  console.log('modèle / réseau de l’Expert          parties   victoires   place moy.   bilans');
  for (const [model, e] of [...models].sort()) {
    const cols = [String(e.games).padStart(7), String(e.wins).padStart(11), (e.places / e.games).toFixed(2).padStart(12), String(e.debriefs).padStart(8)];
    console.log(`${model.padEnd(35)} ${cols.join(' ')}`);
  }
}
