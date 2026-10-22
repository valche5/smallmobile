/**
 * Construit le module WebAssembly du jeu (public/game-kernel.wasm), chargé par le téléphone et le serveur,
 * et l'IA Expert native de chaque labo (target/release/labN-expert, voir src/server/expert.ts).
 *   bun kernel/tools/build.ts
 */
import { existsSync, readdirSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { homedir } from 'node:os';
import { join } from 'node:path';

const root = new URL('../..', import.meta.url).pathname;
const cargo = Bun.which('cargo') ?? join(homedir(), '.cargo/bin/cargo');
export const WASM_FILE = join(root, 'public/game-kernel.wasm');
/** Taille maximale du module livré au téléphone, compressé (les modèles appris en font plus de la moitié). */
const WASM_BUDGET = 256 * 1024;

async function run(cmd: string[]) {
  const p = Bun.spawn(cmd, { cwd: root, stdout: 'inherit', stderr: 'inherit' });
  if ((await p.exited) !== 0) throw new Error(`Échec : ${cmd.join(' ')}`);
}

export async function build() {
  // la bibliothèque en cdylib pour WebAssembly seulement (kernel/Cargo.toml)
  await run([
    cargo, 'rustc', '--manifest-path', join(root, 'kernel/Cargo.toml'), '--lib', '--crate-type', 'cdylib',
    '--target', 'wasm32-unknown-unknown', '--profile', 'web',
  ]);
  // les labos (labN/) dont le cerveau (labN/brain, paquet smallmobile-brainN) donne un Expert
  const brains = readdirSync(root)
    .filter((d) => /^lab\d+$/.test(d) && existsSync(join(root, d, 'brain/Cargo.toml')))
    .flatMap((d) => ['-p', `smallmobile-brain${d.slice(3)}`]);
  await run([cargo, 'build', '--release', '-q', ...brains, '--bins']);
  const wasm = await Bun.file(join(root, 'target/wasm32-unknown-unknown/web/smallmobile_kernel.wasm')).arrayBuffer();
  const module = await WebAssembly.compile(wasm);
  if (WebAssembly.Module.imports(module).length) throw new Error('Le module WebAssembly ne doit rien importer.');
  const gzip = gzipSync(new Uint8Array(wasm), { level: 9 }).length;
  if (gzip > WASM_BUDGET) throw new Error(`Module WebAssembly trop gros : ${gzip} octets compressés.`);
  const previous = (await Bun.file(WASM_FILE).exists()) ? Buffer.from(await Bun.file(WASM_FILE).arrayBuffer()) : null;
  if (!previous?.equals(Buffer.from(wasm))) await Bun.write(WASM_FILE, wasm);
}

if (import.meta.main) await build();
