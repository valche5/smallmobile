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
- `labN/` : les labos de l'IA Expert, un par approche (règles communes : `labtools/README.md`).
  Chacun a son journal, ses données et son cerveau `labN/brain/` (recherche guidée par un réseau
  `labN/nets/*.bin`), partagé par le labo et le jeu. Dans le jeu, l'Expert de chaque labo listé
  dans `src/shared/experts.ts` tourne seulement sur le serveur : programme natif
  `target/release/labN-expert` (construit aussi par `bun run kernel:build`), piloté par
  `src/server/expert.ts`. Un labo figé ne change plus ; les nouveaux travaux vont dans le dernier.
- `labtools/` : outils communs aux labos (AWS, score des arènes, parties archivées).
- `src/` : interface Preact, serveur Bun, et le pont TypeScript vers le noyau (`src/engine/`). Toutes
  les parties se jouent sur le serveur (plus de mode hors ligne) ; il les archive dans `data/games/`
  (graines, actions, temps de chaque décision d'IA) : `bun labtools/games.ts` les résume.
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
Dans ce dépôt, `bun run test` et `bun run kernel:build` reconstruisent `target/release/labN-expert`,
que le serveur relance à la demande, et `public/game-kernel.wasm` : c'est déjà déployer le code en
cours (`bun test` seul ne reconstruit rien). Pendant le développement, s'en tenir à `cargo test` et `cargo clippy` (dossier
`target/debug`), ou construire ailleurs (`CARGO_TARGET_DIR`).

## Calcul du labo sur AWS

mini sert le jeu : **aucun calcul lourd sur mini** (`lab arena`, `label`, `selfplay`, `train.py`).
Tout part sur des machines AWS via `labtools/cloud.sh` (en-tête du script : mode d'emploi ;
`labtools/README.md`, « Les machines AWS » : le compte, les prix, les limites).

- `LAB=labN labtools/cloud.sh job --type c7a.16xlarge [--data D1,D2] [--save chemins] [--get chemins] -- 'commande'` :
  machine, installation, données du bucket, calcul, rapatriement, destruction. Une seule machine à
  la fois (quota Spot : voir le journal).
- Spot : AWS peut reprendre la machine en cours de calcul. Pour un calcul long, `--save` garde des
  dossiers dans le bucket (repris au lancement suivant), et le script saute les morceaux déjà
  faits (en-tête de `cloud.sh`).
- Après chaque calcul : `labtools/cloud.sh status` (rien ne doit rester) et donner la dépense
  (estimée par le script à la destruction ; AWS l'affiche avec quelques heures de retard).
- Budget : rester dans les crédits AWS. Budget en cours, alertes et solde des crédits, datés :
  `labtools/README.md`, « Les machines AWS » ; les revérifier avant chaque campagne. `cloud.sh`
  impose Spot (maximum 1 $/h), au plus 12 h ; aucun repli à la demande. La session SmallAI
  locale est `--profile smallai`.
- Les données produites sur la machine vont dans le bucket, qui reprend les chemins du dépôt
  (`aws s3 sync` depuis la machine), puis sur mini
  (`AWS_PROFILE=smallai aws s3 sync s3://smallai-lab-631245464763/labN/data/X labN/data/X`).
- Ne pas modifier `cloud.sh` sur place pendant qu'un calcul l'utilise (bash relit le fichier) :
  écrire une copie puis la renommer par-dessus.
- Un calcul long en arrière-plan ne réveille l'agent qu'à sa fin : une commande par étape, et
  tenir l'utilisateur au courant à chaque étape.
