# Lab17 : l'Expert des extensions

Base : **lab13** (l'Expert en production : c130, fins simulées, `light=1,place=1`, évaluations sur
plusieurs fils) au commit `bc63a59`, plus `loop.sh` de lab11. Lab13 ne change plus. Règles communes :
[labtools/README.md](../labtools/README.md). Résultats : [journal.md](journal.md).

## Pourquoi

Le jeu a reçu trois extensions (`5195eaa` : Même pas peur, Maauuudits, Honneur aux Dames ; 10 peuples,
12 pouvoirs, choisies à la création d'une partie). Les réseaux et les encodages des labos ne connaissent
que le jeu de base : dès qu'une extension est cochée, le serveur remplace l'Expert par la Difficile
(`src/server/rooms.ts`, « l'Expert ne connaît que le jeu de base ») et chaque `labN-expert` refuse la
partie (`brain/src/bin/expert.rs`).

**But** : un Expert qui joue les parties avec extensions, sans rien perdre au jeu de base.
**Échéance** : l'utilisateur veut une IA qui marche bien **le matin du 16 novembre 2026** (labo ouvert
le 15 à 22 h 30). Le temps de chaque étape compte : voir « Budget de temps ».

## L'idée

Partir de c130 plutôt que de zéro : élargir ses entrées et ses sorties avec des **poids nuls** (comme
`lab14/py/widen.py` pour c130x). Le réseau élargi joue le jeu de base comme c130 et il est aveugle aux
nouveautés ; la recherche joue les vraies règles (le noyau), ce qui corrige déjà une partie de son
ignorance. Puis une boucle d'auto-jeu avec extensions lui apprend le reste.

Les boucles depuis c130 sont toutes restées plates (`labtools/bilan.md`) parce que le professeur
n'était pas plus fort que l'élève. Ici l'élève ignore tout du nouveau contenu et les cibles de valeur
viennent des résultats réels des parties : un gain rapide est attendu sur les parties avec extensions.

## Budget de temps (PC : i7-12700KF 20 fils, RX 9070 XT, Vulkan)

Débits mesurés dans lab11 sur ce PC (`lab11/journal.md`), à revérifier car les règles des extensions
ont pu ralentir le noyau :

- auto-jeu ordinaire de c130 à n = 400 : **65 parties/s** (25 000 parties en 387 s, 35 positions par
  partie) ;
- auto-jeu avec fins simulées (`n=400,roll=1,mix=0.5`) : **2,6 parties/s**, 25 fois plus lent ;
- un pas d'apprentissage : 50 à 90 s ; un duel de 460 donnes arrêté par le test séquentiel : ~40 s.

Donc, pour cette nuit, **la boucle se fait en auto-jeu ordinaire, sans fins simulées** (la recette de
lab2 et lab3, qui a mené de zéro à c060 en 60 pas) : ~8 min par pas. Les fins simulées restent à la
décision, dans le jeu.

| Étape | Durée visée | Fin visée |
| --- | --- | --- |
| 1. Encodage, coups, réseau élargi, tests | 2 h 30 | 01 h 00 |
| 2. Mesure sans entraînement (étape déjà livrable) | 15 min | 01 h 15 |
| 3. Boucle d'auto-jeu avec extensions, sans surveillance | 5 h | 06 h 15 |
| 4. Mesures finales, branchement dans le jeu | 45 min | 07 h 00 |

Règles : mesurer un pas à blanc avant de lancer la boucle et fixer le nombre de pas d'après la durée
mesurée ; toute commande longue avec un délai (`AGENTS.md`) ; la boucle tourne en arrière-plan avec un
délai global, elle ne réveille l'agent qu'à sa fin. **Chaque étape laisse quelque chose de livrable** :
si l'étape 1 déborde, réduire le périmètre (liste « v1 » ci-dessous) plutôt que la boucle.

## Ce que le noyau a ajouté (lu dans `kernel/src`, à relire avant de coder)

- `ids.rs` : `Race::COUNT` 24 et `Power::COUNT` 32 ; les numéros du jeu de base ne bougent pas
  (`BASE_COUNT` 14 et 20, extensions à la suite). `Expansions` (3 bits) est dans `State::expansions` ;
  `Game::create` et `initial_state` le prennent en argument ; `Expansions::NONE` rend le tirage
  d'avant au bit près.
- Nouvelles actions (`action.rs`, `game.rs`) :
  - `Catapult { from, region }` (pouvoir Catapultés) : une fois par tour, cible à deux régions d'une
    région du peuple, coût −1, sans dé ; la région de tir porte la Catapulte et devient imprenable.
    `Game::catapult_options()` liste (tir, cible, coût).
  - `Hire { region }` (Mercenaires) : conquête payée 1 pièce pour 2 jetons de moins
    (`Game::hire_target(&Target)`). Après un dé de renfort raté que la pièce rattraperait,
    `turn.hire = Some(région)` : seuls `Hire` sur cette région ou `EndConquest` (renoncer) sont
    acceptés (`Reason::HirePending`).
  - `Regroup` (Intrépides, `Power::Marauding`) : une fois par tour, reprend en main les jetons
    au-dessus du plancher et continue (`Game::can_regroup`).
  - `Cauldron { region }` (Farfadets) : au redéploiement du peuple actif, pose ou retire un Chaudron
    sur une de ses régions (`RaceState::cauldrons` restants) ; encaissés au début de son tour suivant.
  - `Tower { region }` (Prêtresses) : choisit la Tour d'Ivoire avant un déclin ; sans choix, le noyau
    prend la région la moins menacée (`Game::tower_region`).
- Nouvel état (`state.rs`) : `RegionState::marks` (`catapult()`, `cauldron()`, `ladies()` ; Catapulte
  et Dames Blanches en déclin rendent la région imprenable, `is_immune`) ; `RaceState::aside` (jetons
  hors du plateau jusqu'au prochain tour : Barbares, Fées) et `cauldrons` ; `Combo::tokens` (Homoncules
  ajoutés sur une combinaison, `Combo::starting_tokens()`) ; `Turn::catapult_used`, `regrouped`,
  `abandoned` (Gitanes), `tower`, `hire`.
- Règles qui touchent les coups du labo :
  - **Barbares et Fées actifs ne redéploient pas** (`no_redeploy`, `enter_redeploy`) : `Place`,
    `Deploy`, `AutoDeploy` sont refusés (`Reason::NoRedeploy`), la main part en `aside`. `Move::Lift`
    (qui joue `Place`) ne doit pas leur être proposé : aller droit à la fin du tour.
  - **Kobolds : plancher de 2 jetons par région** (`Game::floor(race)`). `Placement::default()` (un
    jeton partout) est refusé ; `place` accepte 0 (région abandonnée) ou au moins le plancher. `Lift`
    et la pose jeton par jeton doivent suivre le plancher.
  - Gitanes : une région abandonnée ne se reprend pas ce tour, et pas de déclin après un abandon
    (`can_decline`, fin de tour `stout`).
  - Hasard nouveau : le dé des **Pygmées** quand ils perdent une région (`Event::PygmiesReturned`), et
    l'état « paiement en attente » après un dé raté des Mercenaires. `moves::chance` doit distinguer
    ces issues, sinon la recherche confond des mondes différents.
- L'IA du jeu joue déjà les extensions (`kernel/src/ai/mod.rs`, `expansion_conquest`, Chaudrons dans
  `redeploy_step`) : c'est l'adversaire témoin, et un modèle de la manière d'énumérer ces coups.

## Étape 1 : le cerveau élargi (`lab17/brain`, puis `lab17/src`, `shaders`, `py`)

Le différentiel `git diff --no-index lab13 lab14` montre **tous les endroits à toucher** quand les
entrées changent de taille (lab14 a ajouté 16 entrées par région) : `encode.rs`, `net.rs`,
`shaders/net.comp` (`NODE`, `NODE16` multiple de 16, `GLOBAL`, `KEYS`), `vulkan.rs`, `cuda.rs`,
`gpu.rs`, `batch.rs`, `compact.rs`, `py/data.py`, `py/model.py` (échelles), `py/widen.py`.

1. **Entrées** (`encode.rs`). Blocs peuples 14 → 24 et pouvoirs 20 → 32 partout (région, armée, chaque
   joueur actif et en déclin, marché). Entrées neuves : par région, Catapulte, Chaudron, Dames ; par
   joueur, `aside` et Chaudrons restants ; par carte du marché, les jetons avec Homoncules
   (`Combo::starting_tokens()`) ; tour : `catapult_used`, `regrouped`, paiement en attente (et sa
   région, par région), régions abandonnées par les Gitanes (par région) ; partie : les 3 extensions.
   Par région aussi : cible de Catapulte possible et son coût, conquête payée possible et son coût
   (comme `N_NORMAL`, `N_COST`).
   Le plus simple pour garder c130 : **laisser les anciens numéros en place et ajouter les entrées
   neuves à la fin** de chaque ligne (région, partie) ; `widen.py` n'ajoute alors que des colonnes
   nulles. Attention à la tête du peuple (`net::Pick`, `MARKET_AT`, `CARD`) : elle lit les cartes du
   marché, il lui faut aussi les entrées neuves de chaque carte.
2. **Coups** (`moves.rs`). Nouvelles sortes à la suite de `KINDS` (les clés et sorties existantes ne
   bougent pas) : `Hire`, `Catapult` (clé par région visée ; le tir part de la première région
   valable de `catapult_options`, v1), `Cauldron` (avec les pouvoirs de fin de tour, comme `Camp`).
   Clé globale neuve pour `Regroup` (16 à 30 sont libres). Paiement en attente : `Region(Hire, r)` ou
   `EndConquest`. `Tower` : pas en v1 (choix du noyau). Mettre à jour `forced`, `single`, `compact`
   (les Chaudrons passent après `Place`), `chance` (Pygmées, paiement en attente), `market` (le hachage
   multiplie par `Power::BASE_COUNT` : passer à `Power::COUNT`), `describe`, `power_kind`, `Lift`
   (Barbares, Fées, Kobolds, ci-dessus). Vérifier aussi `know.rs` (suivi du redéploiement, histoire) et
   les fins simulées de `search.rs` (`light`, `place`) avec ces peuples.
3. **Réseau** (`net.rs`, `py/model.py`, GPU). Sorties par région 9 → 12 sortes. `py/widen.py` : c130 →
   `nets/c130w.bin`, colonnes d'entrée nulles, lignes de sortie neuves à biais choisi pour que les coups
   neufs ne soient ni ignorés ni préférés d'office (regarder l'ordre de grandeur des logits de c130
   avant de fixer le biais ; la recherche de Gumbel ne visite que les coups les mieux notés a priori).
4. **Programme Expert** (`brain/src/bin/expert.rs`) : retirer le refus des extensions.
5. **Table et outils** (`src/table.rs`, `bin/lab.rs`) : les cartes du jeu (`Table::from_pool`, à
   reprendre de `lab14/src/table.rs` : le jeu ne joue plus que `kernel/maps.bin`) ; une option
   `--expansions` pour `arena` et `selfplay` (`none`, `all`, `mix` : tirage par partie, par exemple la
   moitié avec les trois, le reste réparti entre les autres ensembles, dont aucune).

Tests à écrire (courts, ils remplacent des heures de calcul faux) :

- parties au hasard avec toutes les extensions, 2 à 5 joueurs : tout coup de `moves::legal` est
  accepté par le noyau, `forced` s'accorde avec `legal`, la partie se termine, `compact` mène au même
  état (le test `history_inputs_tell_the_history` en est le modèle) ; vérifier que chaque coup neuf a
  été joué au moins une fois ;
- jeu de base : c130w dans lab17 décide comme c130 dans lab13 sur des positions du jeu de base
  (dépendance de test `smallmobile-brain13` ; égalité des sorties à une tolérance près si l'ordre des
  sommes change, mêmes coups choisis) ;
- `lab17 check-gpu` : le GPU s'accorde avec le processeur pour c130w.

Vérifications : `cargo test -p smallmobile-lab17 -p smallmobile-brain17`, `cargo clippy -p
smallmobile-lab17 -p smallmobile-brain17 --all-targets`. `cargo` est dans `~/.cargo/bin` (pas dans le
`PATH` par défaut). Python : `lab3/py/.venv/bin/python` (PyTorch ROCm).

## Étape 2 : mesure sans entraînement

Arènes courtes avec toutes les extensions, sur les cartes du jeu, 2 à 5 joueurs :

- l'Expert du jeu avec c130w (`n=100,roll=1,mix=0.5,light=1,place=1`) contre la Difficile, ~100
  donnes : bat-il la Difficile, qui, elle, connaît les extensions ? Noter le temps par décision.
- le même sans extension contre la Difficile : le témoin.

S'il bat nettement la Difficile avec extensions, **c'est déjà livrable** (étape 4 sans attendre).

## Étape 3 : la boucle (`lab17/loop.sh`, reprise de lab11, à adapter)

- Départ `c000` = `nets/c130w.bin` ; parties `--expansions mix`, cartes du jeu, auto-jeu ordinaire
  (`AGENT` vide, `SIMS=400` ou 200 selon le débit mesuré), `LR=1.5e-4`.
- Fenêtre : `LAB17_WINDOW_MIN` (2 millions de positions par défaut, soit ~2 cycles de 25 000 parties) ;
  le remplissage par c000 est utile ici (ses parties avec extensions portent déjà les résultats réels).
- Duel de chaque pas : contre c000 **avec toutes les extensions** (le progrès attendu). Toutes les 5
  à 10 étapes et à la fin, le même duel **sans extension** : la régression au jeu de base (tolérance
  du protocole : 2 points).
- Un pas à blanc d'abord (`RUN=blanc`, petits nombres), puis le vrai, en arrière-plan avec un délai
  global ; nombre de pas fixé pour finir vers 06 h 15. `touch lab17/runs/RUN/stop` arrête proprement.

Si le jeu de base recule de plus de 2 points : augmenter la part des parties sans extension, baisser
le taux (5e-5). Si rien ne monte avec extensions après ~10 pas : vérifier les entrées neuves (sont-elles
remplies ?) et le biais des sorties neuves avant de conclure.

## Étape 4 : mesures finales et branchement

- Confirmation sur des graines neuves : le dernier réseau contre c000, avec la recherche du jeu
  (`n=100,roll=1,mix=0.5,light=1,place=1`), toutes extensions (230 donnes par taille), puis sans
  extension contre c130 de lab13 ; et contre la Difficile avec extensions.
- Branchement : entrée `lab17` dans `src/shared/experts.ts` (avec un champ du genre
  `expansions: true`), la garde de `src/server/rooms.ts` levée pour les labos qui l'ont,
  `kernel/tools/build.ts` construit `lab17-expert` ; tests du serveur (`tests/rooms.test.ts`,
  `tests/online.test.ts`), `bun run test`, `bun run typecheck`. Les anciennes variantes restent
  proposées pour le jeu de base.
- Le déploiement sur mini se fait sur demande de l'utilisateur (`AGENTS.md`).

## Si la boucle plafonne

Repartir de zéro avec la recette de lab3 et ces entrées (~9 min par pas dans lab3, de l'ordre d'une
journée de PC pour 130 pas) ; ou des cycles au professeur à fins simulées, 25 fois plus chers.
