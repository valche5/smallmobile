# SmallMobile 🗺️⚔️

Un clone de **Small World** pensé pour le **navigateur mobile** : chaque joueur
joue sur **son propre téléphone**, connecté à un petit serveur Bun. Des **IA**
complètent la table sur les cartes plus grandes. Chaque action affiche ses
conséquences **avant** d'être confirmée. Interface entièrement en français.

## Démarrer

Il faut [Bun](https://bun.sh) ≥ 1.2 et [Rust](https://rustup.rs) avec la cible WebAssembly
(`rustup target add wasm32-unknown-unknown`) : les règles, l'IA et les cartes sont écrites en Rust
(`kernel/`) et compilées en WebAssembly, chargé par le serveur et par les téléphones.

```bash
bun install
bun start        # ou « bun run dev » pour le rechargement à chaud
```

Le serveur affiche son adresse sur le réseau local, par exemple
`http://192.168.1.20:3000`. Ouvrez-la sur les deux téléphones, connectés au
même Wi-Fi que l'ordinateur qui fait tourner le serveur.

1. Sur le premier téléphone : **En ligne → Créer la partie**. Un code de
   4 lettres et un QR code s'affichent.
2. Sur le second : scannez le QR code avec l'appareil photo (ou ouvrez le lien
   partagé, ou tapez le code dans **Rejoindre une partie**).
3. L'hôte choisit la carte puis **Lance la partie**.

Pour jouer à distance (hors du même Wi-Fi), hébergez le serveur sur une
machine accessible depuis Internet (n'importe quel hébergeur qui exécute Bun),
idéalement derrière HTTPS ; les WebSockets passent alors en `wss://`
automatiquement.

| Commande            | Rôle                                                              |
| ------------------- | ----------------------------------------------------------------- |
| `bun start`         | Serveur de jeu (`PORT=8080 bun start` pour changer de port)       |
| `bun run dev`       | Développement avec rechargement à chaud                           |
| `bun run kernel:build` | Noyau du jeu en WebAssembly (`public/game-kernel.wasm`), fait par les autres commandes |
| `bun run build`     | Version statique dans `dist/` (mode « Sur ce téléphone » seulement) |
| `bun test`          | Tests : règles, pouvoirs, parties IA, serveur et WebSockets       |
| `cargo test`        | Tests du noyau Rust (dont l'empreinte de non-régression de l'IA)  |
| `bun run typecheck` | Vérification TypeScript                                           |

Les parties en ligne sont enregistrées dans `data/rooms/` (modifiable avec
`DATA_DIR`) : elles survivent à un redémarrage du serveur et sont effacées
après 7 jours d'inactivité.

## Jouer

- **En ligne** (chacun sur son téléphone) : le serveur applique les règles et
  fait jouer les IA ; chaque téléphone affiche la partie en direct.
  - « À vous ! » s'affiche (et le téléphone vibre) quand c'est votre tour,
    avec le résumé de ce que les autres ont fait. Quand ce n'est pas votre
    tour, la barre du bas indique qui joue et sa dernière action ; vous pouvez
    toujours explorer la carte et les autres vues.
  - Un téléphone en veille ou un rechargement de page se reconnecte tout seul.
    Si un joueur reste hors ligne pendant son tour, l'autre peut confier ses
    tours à l'IA ; il reprend la main dès qu'il revient.
- **Sur ce téléphone** : contre l'IA, ou en passe-et-joue (un écran invite à
  passer le téléphone). Sauvegarde automatique sur l'appareil.
- **Cartes** : Petite (2 joueurs, 23 régions, 10 tours), Moyenne (+1 IA),
  Grande (+2 IA), Immense (+3 IA). Les IA ont trois niveaux ; « Difficile »
  simule les tours suivants (déclin, placement des jetons) et s'en prend
  davantage à ses adversaires, surtout en duel.
  « 🎲 Autre carte » génère un nouveau monde.
- **Vues** (barre du bas) : 🗺️ Carte, 👑 Peuples (le marché), 👥 Joueurs
  (scores, revenus, connexion), 📜 Journal, 📖 Règles.

### Comprendre avant d'agir

- Sur la carte, chaque région attaquable porte une pastille : **⚔3** = coût en
  jetons, **🎲 33 %** = possible seulement au dé de renfort, **🐉 / 🔮** = pouvoir
  spécial. Les régions hors de portée sont assombries.
- Toucher une région ouvre une feuille avec le **détail du coût** et la liste
  des **conséquences** : jetons restants, pertes du défenseur, pièces gagnées,
  défense de la région ensuite, conquêtes encore possibles…
- Choisir un peuple, passer en déclin, finir les conquêtes, finir le tour,
  abandonner une région, lancer le dé berserk : tout passe par une feuille de
  confirmation qui chiffre l'effet (pièces, jetons perdus, peuple qui disparaît…).
- **↶** annule la dernière action tant qu'aucun dé n'a été lancé.
- L'écran « À vous ! » résume ce qui s'est passé depuis votre dernier tour
  (attaques subies en rouge). Quand vous êtes attaqué, il vous invite à
  redéployer vos jetons.

## Règles implémentées

Les règles de base de Small World : 14 peuples, 20 pouvoirs, choix d'un
peuple avec pièces posées sur les peuples sautés, conquête (2 + défenseurs +
montagne + tanières/forteresses/campements), première conquête par le bord de
la carte, dé de renfort (faces 0, 0, 0, 1, 2, 3), redéploiement, déclin (un seul
peuple en déclin sauf Spirituels), redéploiement défensif des joueurs attaqués,
décompte et égalité départagée au nombre de jetons en jeu.

Choix d'adaptation :

- Les cartes sont **générées** (diagramme de Voronoï) et dimensionnées pour un
  écran vertical ; les quatre cartes « classiques » utilisent une graine fixe.
- Les pièces restent cachées jusqu'au podium, même les siennes : seuls les gains de chaque tour sont annoncés.
- Les Goules en déclin jouent leurs conquêtes avant le peuple actif.
- Les Robustes déclinent après le décompte du tour.

## Architecture

```
server.ts            serveur Bun : pages, WebSocket /ws, manifeste PWA
src/server/          parties en ligne (salons, IA côté serveur, sauvegarde disque)
src/shared/          protocole des messages et rythme des IA
kernel/              le jeu en Rust : règles, IA, cartes (compilé en WebAssembly)
  src/rules.rs       calculs (coût, portée, attaquants, décompte)
  src/game.rs        actions → nouvel état et événements
  src/mapgen.rs      génération des cartes (Voronoï, comme l'ancien générateur d3)
  src/ai/            IA : décisions, entrées des modèles, arbres appris, réglages
  models/            modèles appris de l'IA (sommes de petits arbres)
  src/wire.rs, abi.rs   protocole avec TypeScript
src/engine/          le jeu vu de TypeScript, partagé par le serveur et les téléphones
  kernel.ts, engine.ts  pont vers le noyau WebAssembly
  rules.ts           réponses du noyau mises en mots (coûts, raisons, décompte)
  game.ts            actions → nouvel état, journal raconté à partir des événements
  ai.ts, mapgen.ts   IA et cartes (dans le noyau)
  data.ts            peuples, pouvoirs, terrains, noms des régions
  preview.ts         textes de conséquences affichés avant chaque action
src/ui/              interface Preact (accueil, salon, vues, carte SVG zoomable)
tests/               bun test
```

---

Projet de fan non officiel. *Small World* est un jeu de Philippe Keyaerts édité
par Days of Wonder ; aucun élément graphique original n'est utilisé.
