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
- `brain/` : l'IA Expert (recherche guidée par le réseau `lab/nets/gen8.bin`), partagée par le labo et
  le jeu. Dans le jeu, elle tourne seulement sur le serveur : programme natif `target/release/expert`
  (construit aussi par `bun run kernel:build`), piloté par `src/server/expert.ts`.
- `src/` : interface Preact, serveur Bun, et le pont TypeScript vers le noyau (`src/engine/`). Toutes
  les parties se jouent sur le serveur (plus de mode hors ligne) ; il les archive dans `data/games/`
  (graines, actions, temps de chaque décision d'IA) : `bun lab/tools/games.ts` les résume.
- Vérifications : `bun run test`, `bun run typecheck`, `cargo test`, `cargo clippy --workspace --all-targets`.

## Déployer sur mini

```
ssh mini.local 'export PATH=$HOME/.cargo/bin:$HOME/.bun/bin:$PATH; cd ~/repos/smallmobile && git pull && bun install && bun run kernel:build && systemctl --user restart smallmobile.service'
```

Le service lance `bun server.ts` directement : il ne recompile pas le noyau, d'où le
`kernel:build` avant le redémarrage.

Le dépôt de travail sur mini est aussi celui que sert la production (`smallmobile.service`, port
3120, même utilisateur) : ne jamais arrêter un serveur de test par motif (`pkill -f "bun server.ts"`
arrête aussi la production) ; tester sur un autre port et arrêter le processus par son numéro.

## Calcul du labo sur AWS

mini sert le jeu : **aucun calcul lourd sur mini** (`lab arena`, `label`, `selfplay`, `train.py`).
Tout part sur des machines AWS via `lab/tools/cloud.sh` (en-tête du script : mode d'emploi ;
`lab/JOURNAL.md`, « Les machines AWS » : le compte, les prix, les limites).

- `lab/tools/cloud.sh job --type c7a.16xlarge [--data D1,D2] [--get chemins] -- 'commande'` : machine,
  installation, données du bucket, calcul, rapatriement, destruction. Une seule machine à la fois
  (quota 64 vCPU, à la demande ; le Spot est interdit dans ce projet AWS).
- Après chaque calcul : `lab/tools/cloud.sh status` (rien ne doit rester) et donner la dépense
  (estimée par le script à la destruction ; AWS l'affiche avec quelques heures de retard).
- Budget : les crédits AWS (160 $ en octobre 2026 ; `cloud.sh status` donne ce qu'il en reste). Le
  plafond du compte est bien au-dessus (l'ancien plafond de 50 $/mois ne vaut plus).
- Les données produites sur la machine vont dans le bucket (`aws s3 sync` depuis la machine), puis
  sur mini (`AWS_PROFILE=smallai aws s3 sync s3://smallai-lab-631245464763/data/X lab/data/X`).
- Ne pas modifier `cloud.sh` sur place pendant qu'un calcul l'utilise (bash relit le fichier) :
  écrire une copie puis la renommer par-dessus.
- Un calcul long en arrière-plan ne réveille l'agent qu'à sa fin : une commande par étape, et
  tenir l'utilisateur au courant à chaque étape.
