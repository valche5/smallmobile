import type { ServerWebSocket } from 'bun';
import { networkInterfaces } from 'node:os';
import index from './src/index.html';
import { initEngine } from './src/engine/engine';
import { RoomManager } from './src/server/rooms';
import { socketHandlers, upgradeSocket, type SocketData } from './src/server/socket';
import { FileRoomStore } from './src/server/store';

const KERNEL = './public/game-kernel.wasm';
if (!(await Bun.file(KERNEL).exists())) throw new Error(`${KERNEL} manque : bun run kernel:build`);
await initEngine(await Bun.file(KERNEL).arrayBuffer());

const port = Number(process.env.PORT ?? 3000);
const dev = process.env.NODE_ENV !== 'production';
const store = new FileRoomStore(process.env.DATA_DIR ?? './data/rooms');
const rooms = new RoomManager(store);

const lan = Object.values(networkInterfaces())
  .flat()
  .filter((i) => i && i.family === 'IPv4' && !i.internal)
  .map((i) => `http://${i!.address}:${port}`);

// En développement (bun --hot), ce module est réexécuté à chaque modification :
// on ferme les anciennes connexions pour que les téléphones se reconnectent au nouveau gestionnaire.
const shared = globalThis as { __sockets?: Set<ServerWebSocket<SocketData>>; __prune?: ReturnType<typeof setInterval> };
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
  },
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
    process.exit(0);
  });
}

console.log(`\n  🗺️  SmallMobile (${dev ? 'développement' : 'production'})`);
console.log(`  → http://localhost:${server.port}`);
for (const url of lan) console.log(`  → ${url}  (à ouvrir sur les téléphones, même Wi-Fi)`);
if (rooms.size) console.log(`  ${rooms.size} partie(s) en ligne restaurée(s).`);
console.log('');
