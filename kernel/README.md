# Noyau du jeu

Les règles, l'IA et la génération des cartes de SmallMobile, en Rust. Le jeu (serveur et
téléphones) l'utilise compilé en WebAssembly (`public/game-kernel.wasm`), à travers
`src/engine/kernel.ts`. Aucun texte ici :
le noyau renvoie des événements et des raisons de refus, que l'interface raconte en français.

| Commande | Rôle |
| --- | --- |
| `bun run kernel:build` | Module WebAssembly du jeu (aussi fait par `bun start`, `bun run dev`, `bun test`) |
| `cargo test` | Tests du noyau, dont l'empreinte de non-régression (`tests/regression.rs`) |

Organisation (`src/`) :

- `ids`, `map`, `state`, `action` : identifiants, carte, état d'une partie, actions et événements ;
- `rules` : lecture des règles (coût, portée, attaquants, décompte) ; `game` : déroulement d'un
  tour ; `deploy` : placement automatique des jetons ; `setup` : nouvelle partie ;
- `mapgen` et `geometry/` : génération des cartes (Delaunator et Voronoï de d3, traduits à
  l'identique) ;
- `ai/` : décisions des trois niveaux, entrées des modèles (`features`), arbres appris
  (`boost`, fichiers `models/*.bin`), réglages (`tuning`) ;
- `wire`, `abi` : protocole et fonctions exportées pour TypeScript.

Tout est déterministe : mêmes graines et mêmes actions, mêmes parties, en natif comme en
WebAssembly. Le noyau a été vérifié identique, état par état et décision par décision, à
l'ancien moteur TypeScript avant que celui-ci ne soit retiré.
