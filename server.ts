import type { ServerWebSocket } from 'bun';
import { networkInterfaces } from 'node:os';
import index from './src/index.html';
import { initEngine } from './src/engine/engine';
import { FileGameArchive } from './src/server/archive';
import { DEFAULT_EXPERT, ExpertProcess } from './src/server/expert';
import { LlmGames, llmRoutes, llmRules } from './src/server/llm';
import { RoomManager } from './src/server/rooms';
import { socketHandlers, upgradeSocket, type SocketData } from './src/server/socket';
import { FileRoomStore } from './src/server/store';

const KERNEL = './public/game-kernel.wasm';
if (!(await Bun.file(KERNEL).exists())) throw new Error(`${KERNEL} manque : bun run kernel:build`);
await initEngine(await Bun.file(KERNEL).arrayBuffer());

const port = Number(process.env.PORT ?? 3000);
const dev = process.env.NODE_ENV !== 'production';
const store = new FileRoomStore(process.env.DATA_DIR ?? './data/rooms');
const archive = new FileGameArchive(process.env.GAMES_DIR ?? './data/games');

// En développement (bun --hot), ce module est réexécuté à chaque modification : ce qui vit au-delà
// d'une exécution (connexions, minuteur, processus de l'Expert) est retrouvé ici et remplacé.
const shared = globalThis as {
  __sockets?: Set<ServerWebSocket<SocketData>>;
  __prune?: ReturnType<typeof setInterval>;
  __expert?: ExpertProcess;
  __llmExpert?: ExpertProcess;
};

// l'IA Expert : un processus natif ; sans lui, ses sièges jouent comme l'IA Difficile
const missing = ExpertProcess.available(DEFAULT_EXPERT);
if (missing) console.warn(`  IA Expert indisponible : ${missing}`);
shared.__expert?.close();
const expert = missing ? null : (shared.__expert = new ExpertProcess(DEFAULT_EXPERT));
const rooms = new RoomManager(store, undefined, { expert, archive });

// parties jouées par un LLM via l'API (src/server/llm.ts) : leur propre Expert, pour ne pas retarder
// les parties du jeu
shared.__llmExpert?.close();
const llmExpert = missing ? null : (shared.__llmExpert = new ExpertProcess(DEFAULT_EXPERT));
const llm = new LlmGames({ expert: llmExpert, dir: process.env.LLM_DIR ?? './data/llm' });
const llmGuide = llmRules(await Bun.file('./src/server/llm.md').text());

const lan = Object.values(networkInterfaces())
  .flat()
  .filter((i) => i && i.family === 'IPv4' && !i.internal)
  .map((i) => `http://${i!.address}:${port}`);

// les anciennes connexions se ferment : les téléphones se reconnectent au nouveau gestionnaire
for (const ws of shared.__sockets ?? []) ws.close(1012, 'Redémarrage du serveur');
const sockets = (shared.__sockets = new Set<ServerWebSocket<SocketData>>());
if (shared.__prune) clearInterval(shared.__prune);
shared.__prune = setInterval(() => rooms.prune(7 * 24 * 3600 * 1000), 3600 * 1000);

const server = Bun.serve<SocketData>({
  port,
  hostname: process.env.HOST ?? '0.0.0.0',
  routes: {
    '/': index,
    '/api/info': () => Response.json({ lan }),
    '/manifest.webmanifest': () =>
      new Response(Bun.file('./public/manifest.webmanifest'), { headers: { 'Content-Type': 'application/manifest+json' } }),
    '/icon.svg': () => new Response(Bun.file('./public/icon.svg'), { headers: { 'Content-Type': 'image/svg+xml' } }),
    '/game-kernel.wasm': () => new Response(Bun.file(KERNEL), { headers: { 'Content-Type': 'application/wasm' } }),
    ...llmRoutes(llm, () => llmGuide),
  },
  // une action d'un LLM attend que les adversaires aient joué : parfois plus que les 10 s par défaut
  idleTimeout: 120,
  development: dev ? { hmr: true, console: true } : false,
  fetch(req, srv) {
    if (new URL(req.url).pathname === '/ws') return upgradeSocket(req, srv);
    return new Response('Introuvable', { status: 404 });
  },
  websocket: socketHandlers(rooms, sockets),
});

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => {
    store.flush();
    archive.flush();
    expert?.close();
    llmExpert?.close();
    process.exit(0);
  });
}

console.log(`\n  🗺️  SmallMobile (${dev ? 'développement' : 'production'})`);
console.log(`  → http://localhost:${server.port}`);
for (const url of lan) console.log(`  → ${url}  (à ouvrir sur les téléphones, même Wi-Fi)`);
if (rooms.size) console.log(`  ${rooms.size} partie(s) en ligne restaurée(s).`);
console.log('');
