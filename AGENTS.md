# Notes pour les agents

## Règle : commandes longues

**Toujours lancer une commande potentiellement longue avec un délai maximal (timeout), estimé
avant de la lancer.** Estimer soi-même la durée (au besoin en mesurant d'abord sur un petit
échantillon), puis fixer le délai avec une marge raisonnable, par exemple
`timeout 900 cargo test --release` ou le paramètre de délai de l'outil. Une commande
sans délai peut bloquer la session indéfiniment.

## Le projet en bref

- `kernel/` : le jeu en Rust (règles, IA, cartes), compilé en WebAssembly pour le serveur et les
  téléphones (`public/game-kernel.wasm`, hors git, généré par `bun run kernel:build`).
- `src/` : interface Preact, serveur Bun, et le pont TypeScript vers le noyau (`src/engine/`).
- Vérifications : `bun run test`, `bun run typecheck`, `cargo test`, `cargo clippy --workspace --all-targets`.

## Déployer sur mini

```
ssh mini.local 'export PATH=$HOME/.cargo/bin:$HOME/.bun/bin:$PATH; cd ~/repos/smallmobile && git pull && bun install && bun run kernel:build && systemctl --user restart smallmobile.service'
```

Le service lance `bun server.ts` directement : il ne recompile pas le noyau, d'où le
`kernel:build` avant le redémarrage.
