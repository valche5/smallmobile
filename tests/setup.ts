/** Chargé avant les tests (bunfig.toml) : le noyau du jeu, compilé par `bun run kernel:build`, et les cartes. */
import { initEngine } from '../src/engine/engine';
import { loadMaps } from '../src/engine/maps';

const file = Bun.file(new URL('../public/game-kernel.wasm', import.meta.url).pathname);
if (!(await file.exists())) throw new Error('public/game-kernel.wasm manque : bun run kernel:build');
await initEngine(await file.arrayBuffer());
loadMaps(await Bun.file(new URL('../kernel/maps.bin', import.meta.url).pathname).arrayBuffer());
