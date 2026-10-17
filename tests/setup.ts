/** Chargé avant les tests (bunfig.toml) : le noyau du jeu, compilé par `bun run kernel:build`. */
import { initEngine } from '../src/engine/engine';

const file = Bun.file(new URL('../public/game-kernel.wasm', import.meta.url).pathname);
if (!(await file.exists())) throw new Error('public/game-kernel.wasm manque : bun run kernel:build');
await initEngine(await file.arrayBuffer());
