# Labo 2 : Gumbel AlphaZero, nœuds de hasard, coups jeton après jeton

Ouvert le 21 octobre 2026, à partir de lab1, figé ce jour-là (`lab1/JOURNAL.md` : tout ce qui a
été mesuré jusqu'à gen11). Règles communes des labos, protocole de mesure, machines AWS :
`labtools/README.md`. **Figé le 22 octobre 2026 après sa troisième boucle** ; la suite, un réseau
par nombre de joueurs : `lab3/journal.md`.

## Objectif

Le même que lab1 : l'IA la plus forte possible sur les cartes générées, à 2, 3, 4 et 5 joueurs,
mesurée aussi au temps de réflexion du jeu sur mini. Mêmes règles d'honnêteté : l'Expert ne sait
que ce qu'un joueur attentif peut savoir (ordre caché des pioches et dés à venir tirés au sort), et
ne sait pas contre qui il joue ; il peut se faire un profil de chaque joueur d'après ses coups
publics (histoire des joueurs en entrée du réseau).

## Pourquoi ce labo

Revue du 21 octobre : lab1 s'écartait d'AlphaZero et de ses successeurs sur plusieurs points,
certains justifiés par le jeu, d'autres hérités de l'époque où le réseau était faible. Décisions
de l'utilisateur, prises sans mesure préalable (régressions passagères acceptées) :

1. **Gumbel AlphaZero** (Danihelka et al., *Policy improvement by planning with Gumbel*, 2022)
   au lieu de PUCT : racine par tirage de Gumbel et halving séquentiel, sélection déterministe
   ailleurs, cible de politique = politique améliorée (valeurs complétées), plus de bruit de
   Dirichlet. Fait pour les petits budgets et les nombreux coups possibles : notre cas.
2. **Nœuds de hasard** (la recherche de Stochastic MuZero, sans modèle appris : le noyau est un
   simulateur exact) au lieu de la boucle ouverte : chaque nœud est une position publique ; la
   suite d'un coup se range par résultat public (dé, carte arrivée au marché).
3. **Valeur du réseau aux feuilles**, sans les fins simulées de 3 tours de lab1 (valeur biaisée
   vers le jeu du réseau sans recherche, 8 évaluations par itération) : jugées suspectes, retirées.
4. **Coups repensés** : pouvoirs (héros, forteresse, campements) posés un par un et facultatifs,
   héros et campements relevés à chaque tour ; jetons posés un par un ; paix des Diplomates ou
   déclin des Robustes en décision à part. Plus de coup `Auto` (IA Normale) ni de plans de
   placement du noyau dans la recherche.
5. **Retirés** : la tête « coup réel », `bh`/`bt` et les coups observés (P3 de lab1, jamais mesuré).

Gardés de lab1 : l'encodage (820 entrées, histoire des joueurs comprise), la valeur (0,5 ×
victoire + 0,5 × sigmoïde de l'avance en pièces, échelle 8), le tronc du réseau (graphe des
régions, 4 couches de 64), l'auto-apprentissage avec recherches courtes, la recette
d'entraînement. Rien de ses poids ni de ses parties. La taille du réseau pourra augmenter si les
mesures le demandent (accord de l'utilisateur, 21 octobre).

**Positions rejugées supprimées** (décision de l'utilisateur, 21 octobre) : lab1 les gardait pour
des cibles de valeur moins bruitées (moyenne de 32 fins jouées par le réseau), mais leur gain y
était faible et fragile (+2,4 ± 2,2 points sur les résultats seuls, une mesure, P2), elles sont
circulaires (le réseau juge ses propres fins) et coûtent environ 4 fois le prix des parties. Le
calcul va plutôt à plus de parties réelles.

**Départ de zéro** (décision de l'utilisateur, 21 octobre) : aucun poids, aucune partie venus de
lab1 ni des IA du jeu. gen0 est un réseau neuf qui joue au hasard (sorties à zéro : politique
uniforme, mêmes chances pour tous) ; les données ne viennent que de l'auto-jeu, sans tables
variées. lab1 et les IA du jeu ne servent qu'à mesurer, en arène. Plus lent à démarrer, mais ce
que le labo apprend ne doit rien à lab1 ; on compare ensuite ses progrès à lab1 et aux IA du jeu.

## Système actuel

- **Recherche** (`lab2/brain/src/search.rs`) : Gumbel AlphaZero. Racine : 16 coups au plus, tirés
  selon logits + bruit de Gumbel (auto-apprentissage ; sans bruit ailleurs), halving séquentiel
  (calendrier de mctx). Hors racine : sélection déterministe vers la politique améliorée
  `softmax(logits + σ(q complétées))`, `σ = (50 + visites max) × 0,1 × q ramenée entre 0 et 1`,
  valeur mêlée pour les coups jamais essayés (constantes de mctx). Nœuds = positions publiques,
  suite d'un coup rangée par résultat de son hasard (`moves::chance` : cartes arrivées au marché,
  réussite du dé de renfort, dé des Berserks) ; un monde possible tiré par itération ; coups
  forcés sans nœud. Feuille : valeur du réseau. Seul réglage : `n` (itérations).
- **Coups** (`lab2/brain/src/moves.rs`) : une décision à la fois. Fin de tour : `Lift` (forcé :
  un jeton par région, le reste en main, héros et campements relevés), puis les pouvoirs un par un
  (héros, forteresse, campement sur une région ; facultatifs : jouer un jeton ou finir le tour met
  fin à leur pose), puis les jetons un par un, puis la fin du tour (paix des Diplomates : aucune ou
  un adversaire relatif ; déclin des Robustes ; sinon forcée). Défense : les jetons rendus un par
  un. Le suivi de la fin de tour (`Redeploy`) est dans `Knowledge`. 608 clés : 32 globales, 9
  sortes × 64 régions (conquérir, dé, dragon, sorcellerie, abandon, héros, forteresse, campement,
  jeton). Une décision entière du jeu (`moves::decision`) se joue en actions groupées : héros et
  campements reposés au même endroit ne bougent pas, les jetons forment un seul placement.
- **Réseau** (`lab2/brain/src/net.rs`, `lab2/py/model.py`) : graphe des régions, 4 couches de 128
  depuis la deuxième boucle (64 dans la première) ;
  têtes de politique, de valeur (victoire, pièces gagnées) et, à l'entraînement seulement, de
  possession (à qui appartient chaque région au prochain tour du joueur). gen0
  (`lab2/nets/gen0.bin`, écrit par `python lab2/py/model.py`) est un réseau neuf : sorties à zéro,
  il joue au hasard. Calcul Rust en `f32` (AVX2) ; sur un processeur qui a AVX-512 BF16 (AWS),
  les grands produits en bf16, choisis automatiquement : mêmes résultats que leur référence
  scalaire, écarts au `f32` de 10⁻⁴ sur la victoire et 0,004 sur les logits (`lab2 bench-net`).
- **Jeu** : `lab2-expert` avec c060 (`lab2/nets/c060.bin`), n = 800 (`src/shared/experts.ts`).
  Arène d'essai sur mini avec c040 (même taille), un cœur, 3 joueurs face à Normale, 20 donnes :
  179 ms en moyenne par décision à n = 800, p95 624 ms, fins de tour comprises (c020, 64x4 :
  65 ms, p95 224 ms ; lab1 dans le jeu sur mini : 80 à 250 ms, médiane 170 ms).
- **Arène** : un joueur rend une décision entière en actions du noyau ; témoin `lab1:net=…,n=…`
  (l'Expert de lab1 tel qu'il joue dans le jeu).
- **Auto-jeu** (`lab2 selfplay`) : le réseau joue toutes les places ; bruit de Gumbel dans toutes
  les recherches, le coup joué est celui de la recherche. Recette `selfplay::RECIPE` : une
  recherche complète (`n` de l'agent) pour 8 coups, enregistrée ; les autres à 50 itérations.
  Positions `*.pos` : politique améliorée de la racine, victoire, pièces gagnées, possession des
  régions au prochain tour du joueur, et un poids (surprise de la recherche, voir Entraînement).
- **Entraînement** (`lab2/py/train.py`) : politique, victoire, pièces et possession (poids 1,5,
  comme la propriété des cases de KataGo : des dizaines d'étiquettes réelles par position, là où
  la victoire n'en donne qu'une par partie). Positions tirées en proportion de leur poids : la
  moitié répartie également, l'autre selon la surprise de la recherche (écart de sa politique à
  l'a priori du réseau, « policy surprise weighting » de KataGo). Calcul en bf16 (poids en 32
  bits). Erreurs par genre de position : `lab2/py/value.py`.
- **Boucle** (`lab2/loop.sh`, idées de KataGo) : à chaque pas k, le réseau k s'entraîne depuis le
  réseau k−1 sur une fenêtre des derniers cycles (toutes les positions jusqu'à 1 000 000, puis
  1 000 000 × (1 + 0,4 × ((total / 1 000 000)^0,75 − 1) / 0,75) ; 250 000 dans la première
  boucle), chaque nouvelle position servant en moyenne 4 fois ; pendant ce temps, le réseau k−1
  joue les 25 000 parties du cycle k+1 (un cycle de retard, comme les joueurs d'AlphaZero ou de
  KataGo pendant l'apprentissage) ; puis arènes du réseau k face à Difficile et Normale (460
  donnes chacune, mêmes donnes à chaque pas). Pas de sélection du réseau, comme AlphaZero. En
  fin de boucle (`lab2/runs/final.sh`) : les réseaux sur les mêmes positions de validation
  (`lab2/py/curve.py` ; celles de chaque pas changent avec la fenêtre), et le dernier réseau face
  à ceux d'avant et à lab1.
- **Données** : 5,3 Go de positions par cycle de 25 000 parties (128x4). `lab2/prune.sh` efface
  du bucket les positions des cycles sortis de la fenêtre (un cycle de marge), sans machine
  allumée ; les parties (`*.pos.games`, `*.pos.moves`) restent, de quoi les rejouer.
- **Temps** (`c7a.16xlarge`, mesures du 21 octobre avec un réseau de 3 cycles d'essai) : 124
  parties/s sur 64 cœurs en bf16 (82 en `f32`), 104 sur 48 cœurs, 102 pendant l'entraînement ;
  entraînement 12 800 positions/s sur 16 cœurs en bf16 (9 900 en `f32` ; 10 000 pendant les
  parties ; plus rien à gagner au-delà de 16 à 32 cœurs, la mémoire limite) ; arènes 57 s
  (Difficile) et 22 s (Normale) à 920 donnes. En même temps, avec c020 : 48 cœurs de parties et
  16 d'entraînement, 62 parties/s et 59,6 ms par lot ; 56 et 8 (entraînement compilé, lots
  préparés à côté) : 67 parties/s mais 85,9 ms par lot, moins bien. `torch.compile` : 3 % gagnés à
  16 cœurs en bf16 pour 9 à 18 s de compilation par cycle, pas gardé. Couches sans concaténation
  (la part de g calculée une fois par position, comme en Rust) : même réseau, 11 % de moins sur
  mini. Un pas de la première boucle : 5,5 à 7,3 min (les parties s'allongent avec le niveau),
  contre environ 10 min en `f32` sans recouvrement. Avec 128x4 (deuxième boucle) : 56 cœurs de
  parties et 8 d'entraînement, 1 060 à 1 075 s de parties et 790 à 1 180 s d'entraînement par
  pas, environ 20 min par pas. Sur mini (6 cœurs, AVX2) : l'entraînement
  plafonne à 4 cœurs, et le bf16 y est émulé (15 fois plus lent) : aucun entraînement réel sur
  mini.

## Plan

1. ~~Recherche : Gumbel AlphaZero et nœuds de hasard.~~ Fait le 21 octobre.
2. ~~Coups : pouvoirs et jetons un par un, paix et déclin à part ; l'Expert du jeu renvoie sa fin
   de tour en actions groupées.~~ Fait le 21 octobre.
3. ~~Réseau, format des positions, entraînement ; départ de zéro.~~ Fait le 21 octobre.
4. ~~Boucle par petits cycles à la manière de KataGo (possession, surprise, fenêtre), essai sur
   mini, mesures sur AWS, bf16 et recouvrement.~~ Fait le 21 octobre.
5. ~~Première boucle sur AWS depuis gen0 : 20 pas, arènes à chaque pas ; lab2 revient dans le jeu
   avec son meilleur réseau.~~ Faite le 21 octobre (voir Résultats).
6. ~~Comparaison avec lab1 dans les conditions du jeu.~~ Faite le 21 octobre : −12,6.
7. ~~Taille du réseau.~~ Mesurée le 21 octobre : avec assez de parties, plus large aide la valeur,
   la possession et la politique (voir Résultats).
8. ~~Sortir du plateau (+19 face à Difficile depuis c015, −12,6 face à lab1) : réseau plus large
   et fenêtre plus large.~~ Deuxième boucle, 128x4, finie le 22 octobre : −2,2 face à lab1.
9. ~~Troisième boucle, même recette depuis c040 : pas 41 à 60.~~ Faite le 22 octobre.

## Résultats

Scores du protocole (`labtools/score.py`, tailles à poids égaux), n = 200.

### Première boucle (21 octobre)

`lab2/loop.sh 20` depuis gen0 (au hasard), sur une `c7a.16xlarge` Spot à 0,846 $/h : 2 h 12 de
boucle, 500 000 parties ; la machine a aussi servi aux mesures du bf16 et de l'entraînement
(179 min en tout, 2,58 $). Parties `lab2/data/loop` dans le bucket, réseaux et arènes
`lab2/runs/loop` (hors Git) ; c020 gardé dans `lab2/nets/`.

Arènes de chaque réseau, 460 donnes (115 par taille), mêmes donnes à chaque pas :

| Réseau | Difficile (1002) | Normale (1003) | Réseau | Difficile | Normale |
| --- | --- | --- | --- | --- | --- |
| c001 | −31,4 ± 0,4 | −29,2 ± 1,1 | c011 | +6,7 ± 2,2 | +10,6 ± 2,1 |
| c002 | −29,1 ± 1,0 | −28,2 ± 1,2 | c012 | +10,4 ± 2,3 | +10,4 ± 2,0 |
| c003 | −21,7 ± 1,8 | −16,2 ± 2,1 | c013 | +14,7 ± 2,3 | +14,0 ± 2,1 |
| c004 | −18,8 ± 1,9 | −9,6 ± 2,3 | c014 | +15,9 ± 2,4 | +17,3 ± 2,2 |
| c005 | −14,0 ± 2,1 | −7,2 ± 2,3 | c015 | +18,5 ± 2,4 | +16,2 ± 2,1 |
| c006 | −9,8 ± 2,1 | −3,8 ± 2,1 | c016 | +19,0 ± 2,4 | +17,7 ± 2,2 |
| c007 | −6,0 ± 2,0 | −1,4 ± 1,8 | c017 | +18,8 ± 2,2 | +17,9 ± 2,1 |
| c008 | −5,6 ± 1,9 | +0,4 ± 1,9 | c018 | +19,6 ± 2,2 | +15,7 ± 2,1 |
| c009 | +0,4 ± 2,2 | +5,0 ± 1,9 | c019 | +19,9 ± 2,3 | +17,5 ± 2,1 |
| c010 | +3,1 ± 2,3 | +6,8 ± 1,8 | c020 | +19,8 ± 2,4 | +17,2 ± 2,1 |

c020 par taille (2, 3, 4, 5 joueurs) : Difficile +36,7, +13,8, +13,0, +15,7 ; Normale +45,2,
+8,7, +6,7, +8,3.

- **La boucle apprend depuis le hasard, sans rien de lab1 ni des IA du jeu** : à égalité avec
  Normale au pas 8, avec Difficile au pas 9, puis 3,5 à 4 points par pas jusqu'à c015 ; c003
  (75 000 parties) dépasse déjà l'ancien gen1 par génération (150 000 parties au hasard :
  −29,7 et −27,6).
- **Plateau depuis c015** : +18,5 à +19,9 face à Difficile, +15,7 à +17,9 face à Normale.
- **Fort à 2 joueurs, plus faible à 3, 4 et 5**, comme les essais précédents.
- **Face à lab1, dans les conditions du jeu : −12,6 ± 1,3** (c020 à n = 800 face à l'Expert de
  lab1, gen11 à n = 200 ; 920 donnes, graine 1004 ; par taille −11,2, −12,8, −13,4, −13,2 ;
  `lab2/runs/lab1/arena-c020.tsv`, `lab2/runs/lab1.sh`). Le même écart à toutes les tailles,
  alors que c020 domine Difficile bien plus à 2 joueurs qu'à 3, 4 et 5. 56 ms par décision en
  moyenne pour c020 (bf16), p95 198 ms ; 23 min de machine, 0,29 $.
- Fenêtre : le dernier cycle seul jusqu'à c009, deux cycles de c010 à c019, trois pour c020.
  Parties : 10 positions par partie au hasard, 22 puis 33 en fin de boucle ; 25 000 parties en
  109 s (hasard, 64 cœurs) puis 245 à 406 s (48 cœurs) ; entraînement 128 à 368 s.

### Taille du réseau (21 octobre)

`lab2/py/size.py`, scripts `lab2/runs/size.sh` et `size2.sh` (résultats `lab2/runs/size*`) : des
réseaux entraînés depuis zéro, recette de train.py, sur les mêmes parties (cycles 19 et 20 de la
première boucle, 50 000 parties, 1,57 million de positions d'entraînement, 81 000 de validation), validation sur les
mêmes parties réservées. Coût d'une évaluation Rust en bf16 sur un cœur (c7a) :

| Taille | Paramètres | Évaluation | 2 passages : valeur, possession, politique | 8 passages |
| --- | --- | --- | --- | --- |
| 64x4 | 206 713 | 36 µs | 1,161, 0,622, 1,507 | 1,074, 0,574, 1,441 |
| 64x6 | 272 761 | 52 µs | 1,154, 0,615, 1,495 | |
| 64x8 | 338 809 | 70 µs | 1,147, 0,610, 1,488 | |
| 96x4 | 417 561 | 75 µs | 1,138, 0,602, 1,479 | 1,071, 0,559, 1,421 |
| 96x6 | 565 785 | 105 µs | 1,137, 0,597, 1,472 | |
| 128x4 | 700 089 | 115 µs | 1,125, 0,590, 1,463 | 1,078, 0,550, 1,408 |

c020 (la boucle, 64x4) sur la même validation : 1,039, 0,547, 1,379 ; continué 4 passages sur ces
parties : 1,046, 0,546, 1,381.

- À 50 000 parties, entraînées jusqu'au bout, les trois largeurs finissent à la même valeur
  (1,07-1,08) : la valeur bute sur le nombre de parties, quelle que soit la taille (c020, qui a vu
  les 500 000 parties de la boucle, fait 1,039 et ne gagne rien à repasser sur ces 50 000). Plus
  large aide un peu la politique et la possession (−0,02 à −0,03) ; la largeur rapporte plus que
  la profondeur à coût égal. 14 + 33 min de machine, 0,18 + 0,49 $.
- **Objection de l'utilisateur, juste** : 50 000 parties ne suffisent pas à montrer ce qu'apporte
  la taille. Même mesure sur les cycles 11 à 20 (250 000 parties, une position sur trois, la même
  pour tous : 2,5 millions de positions d'entraînement, 132 000 de validation ; 5 passages, autant
  de positions vues qu'avant ; `lab2/runs/size3.sh`) :

  | Taille | Valeur | Possession | Politique | Écart de valeur à c020 (50 000 → 250 000 parties) |
  | --- | --- | --- | --- | --- |
  | 64x4 | 1,040 | 0,545 | 1,468 | +0,035 → +0,029 |
  | 96x4 | 1,026 | 0,531 | 1,448 | +0,032 → +0,015 |
  | 128x4 | 1,018 | 0,520 | 1,433 | +0,039 → +0,007 |

  c020 sur cette validation : 1,011, 0,530, 1,441 ; continué 2 passages sur ces 250 000 parties :
  1,003, 0,520, 1,418 (une fenêtre plus large aide un peu). **Avec plus de parties, les réseaux
  plus larges prennent l'avantage sur la valeur aussi** : 128x4, entraîné une fois depuis zéro sur
  la moitié des parties, rejoint presque c020. 35 min de machine, 0,51 $.


### Deuxième boucle (21 et 22 octobre)

Réseau 128x4, fenêtre d'au moins 1 000 000 de positions, et les parties de la première boucle
(choix de l'utilisateur : repartir de ces parties plutôt que du hasard). c020 y est le 128x4 de
la mesure de taille (`lab2/runs/size3/128x4.bin`, entraîné depuis zéro sur les cycles 11 à 20),
qui remplace le c020 64x4 de la première boucle (gardé : `lab2/runs/loop/c020-64x4.bin`) ; puis
`lab2/loop.sh 40` (script `lab2/runs/loop2.sh`), pas 21 à 40, 56 cœurs de parties et 8
d'entraînement. Une `c7a.16xlarge` Spot, 413 min, 5,29 $. Fenêtre : 5 cycles au pas 21, 8 au pas 40
(6,7 millions de positions) ; 33,6 positions par partie.

Arènes de chaque réseau (comme la première boucle), et mesures de validation de chaque pas : sur
les parties réservées de sa fenêtre, qui change à chaque pas (pas comparables d'un pas à l'autre,
voir la courbe plus bas).

| Réseau | Difficile | Normale | Valeur | Possession | Politique | Précision |
| --- | --- | --- | --- | --- | --- | --- |
| c020 (128x4) | +18,2 ± 2,3 | +17,3 ± 2,1 | | | | |
| c021 | +17,6 ± 2,3 | +17,9 ± 2,1 | 1,030 | 0,542 | 1,398 | 44,7 % |
| c022 | +18,6 ± 2,3 | +19,5 ± 2,1 | 1,028 | 0,538 | 1,393 | 45,1 % |
| c023 | +20,0 ± 2,2 | +19,8 ± 2,2 | 1,024 | 0,536 | 1,390 | 45,3 % |
| c024 | +20,4 ± 2,2 | +18,0 ± 2,2 | 1,022 | 0,534 | 1,386 | 45,5 % |
| c025 | +22,2 ± 2,2 | +22,2 ± 2,2 | 1,023 | 0,533 | 1,380 | 45,9 % |
| c026 | +19,9 ± 2,3 | +21,1 ± 2,1 | 1,023 | 0,530 | 1,374 | 46,3 % |
| c027 | +23,7 ± 2,2 | +20,2 ± 2,0 | 1,025 | 0,527 | 1,365 | 46,8 % |
| c028 | +20,6 ± 2,2 | +18,5 ± 2,1 | 1,025 | 0,526 | 1,358 | 47,1 % |
| c029 | +24,2 ± 2,3 | +20,9 ± 2,1 | 1,028 | 0,526 | 1,355 | 47,3 % |
| c030 | +23,6 ± 2,3 | +21,3 ± 2,1 | 1,027 | 0,526 | 1,355 | 47,3 % |
| c031 | +24,1 ± 2,3 | +19,6 ± 2,1 | 1,035 | 0,527 | 1,352 | 47,4 % |
| c032 | +25,3 ± 2,2 | +22,5 ± 2,1 | 1,037 | 0,528 | 1,349 | 47,4 % |
| c033 | +26,8 ± 2,2 | +20,3 ± 2,1 | 1,037 | 0,529 | 1,347 | 47,5 % |
| c034 | +24,3 ± 2,2 | +20,6 ± 2,1 | 1,037 | 0,529 | 1,344 | 47,6 % |
| c035 | +25,3 ± 2,2 | +19,1 ± 2,0 | 1,040 | 0,530 | 1,341 | 47,8 % |
| c036 | +25,8 ± 2,2 | +23,2 ± 2,2 | 1,040 | 0,530 | 1,342 | 47,7 % |
| c037 | +24,1 ± 2,2 | +19,0 ± 2,1 | 1,040 | 0,530 | 1,339 | 47,8 % |
| c038 | +23,7 ± 2,1 | +21,8 ± 2,1 | 1,038 | 0,530 | 1,336 | 47,9 % |
| c039 | +24,4 ± 2,3 | +20,3 ± 2,1 | 1,033 | 0,530 | 1,334 | 48,0 % |
| c040 | +25,9 ± 2,2 | +21,3 ± 2,1 | 1,030 | 0,528 | 1,332 | 47,9 % |

c040 par taille (2, 3, 4, 5 joueurs) : Difficile +41,3, +19,1, +18,0, +25,1 ; Normale +44,3,
+6,1, +17,4, +17,5.

Mesures de fin (`lab2/runs/final.sh`, résultats `lab2/runs/final`, 38 min de machine, 0,47 $).
Courbe : chaque réseau sur les mêmes positions, celles des parties réservées des cycles 21 et 22
(jouées par c019 et le c020 64x4 ; 83 869 positions) et des cycles 39 et 40 (jouées par c037 et
c038 ; 83 276 positions) :

| Réseau | Valeur 21-22 | Possession | Politique | Valeur 39-40 | Possession | Politique | Précision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c010 (64x4) | 1,076 | 0,661 | 1,588 | 1,140 | 0,722 | 1,686 | 33,7 % |
| c015 (64x4) | 1,037 | 0,578 | 1,448 | 1,088 | 0,632 | 1,539 | 38,4 % |
| c020 (64x4) | 1,020 | 0,551 | 1,404 | 1,065 | 0,591 | 1,481 | 40,6 % |
| c020 (128x4) | 1,030 | 0,555 | 1,417 | 1,077 | 0,601 | 1,487 | 40,4 % |
| c025 | 1,007 | 0,533 | 1,390 | 1,060 | 0,568 | 1,432 | 42,9 % |
| c030 | 1,003 | 0,535 | 1,412 | 1,048 | 0,552 | 1,388 | 45,0 % |
| c035 | 1,003 | 0,541 | 1,436 | 1,042 | 0,540 | 1,348 | 47,1 % |
| c040 | 1,003 | 0,545 | 1,454 | 1,037 | 0,533 | 1,330 | 47,8 % |

Arènes de c040, 920 donnes, graine 1004 :

| Face à | Score | 2 joueurs | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| c030 (n = 200 des deux côtés) | +5,1 ± 1,6 | +4,7 | +5,1 | +5,3 | +5,4 |
| c020 64x4 (n = 200) | +14,4 ± 1,7 | +13,7 | +13,6 | +13,7 | +16,7 |
| lab1 (c040 à n = 800, gen11 à n = 200) | −2,2 ± 1,6 | −0,5 | −2,6 | −1,4 | −4,2 |

- **Presque à égalité avec lab1** (−12,6 avec le c020 64x4), toujours sans rien de lab1. 134 ms
  par décision pour c040 sur AWS (bf16), p95 459 ms ; 179 ms sur un cœur de mini (`f32`).
- **Les arènes face aux IA du jeu ne mesurent plus les progrès** : plateau vers +25 face à
  Difficile et +21 face à Normale depuis c029, alors que c040 bat c030 de 5 points. Les mesures
  de fin (courbe, réseaux précédents, lab1) prennent le relais.
- **La valeur de validation de chaque pas montait** (1,025 vers 1,040) **parce que les parties
  deviennent plus difficiles à prédire**, pas parce que le réseau recule : sur des positions
  fixes, chaque réseau fait mieux que le précédent sur les parties récentes (valeur 1,077 → 1,037,
  possession 0,601 → 0,533, politique 1,487 → 1,330) ; sur les parties anciennes, la valeur
  stagne depuis c030 et la politique s'éloigne des recherches des anciens réseaux (cible
  périmée, pas une erreur).
- Le pont 128x4 part un peu en dessous du c020 64x4 (sur les deux jeux de positions comme en
  arène), le dépasse dès c025.
- Positions des cycles 1 à 31 effacées du bucket ensuite (`lab2/prune.sh`, 193 → 52 Go) : la
  courbe sur les cycles 21 et 22 ne peut plus être refaite.

### Troisième boucle (22 octobre)

Même recette depuis c040 (choix de l'utilisateur) : `lab2/loop.sh 60`, pas 41 à 60 ; les parties
du cycle 41 jouées d'abord par c039 sur 64 cœurs (la deuxième boucle s'arrêtait avant). Une
`c7a.16xlarge` Spot, 424 min, 5,35 $. Fenêtre : 8 cycles au pas 41, 11 au pas 60 (9,3 millions de
positions) ; parties 1 076 à 1 085 s par cycle, entraînement 816 à 1 188 s (il s'allonge avec la
fenêtre) : 20 à 22 min par pas.

| Réseau | Difficile | Normale | Valeur | Possession | Politique | Précision |
| --- | --- | --- | --- | --- | --- | --- |
| c041 | +26,4 ± 2,1 | +20,9 ± 2,0 | 1,028 | 0,528 | 1,329 | 48,1 % |
| c042 | +24,9 ± 2,2 | +22,1 ± 2,1 | 1,028 | 0,528 | 1,329 | 48,0 % |
| c043 | +25,5 ± 2,2 | +21,8 ± 2,2 | 1,027 | 0,527 | 1,327 | 48,1 % |
| c044 | +25,9 ± 2,3 | +21,1 ± 2,0 | 1,026 | 0,526 | 1,325 | 48,1 % |
| c045 | +25,6 ± 2,3 | +21,2 ± 2,0 | 1,023 | 0,526 | 1,323 | 48,3 % |
| c046 | +26,9 ± 2,2 | +23,4 ± 2,0 | 1,021 | 0,527 | 1,322 | 48,3 % |
| c047 | +25,5 ± 2,2 | +23,4 ± 2,1 | 1,021 | 0,527 | 1,320 | 48,4 % |
| c048 | +25,9 ± 2,3 | +23,2 ± 2,1 | 1,021 | 0,527 | 1,317 | 48,4 % |
| c049 | +28,5 ± 2,1 | +23,4 ± 2,1 | 1,019 | 0,526 | 1,315 | 48,5 % |
| c050 | +26,3 ± 2,3 | +24,5 ± 1,9 | 1,016 | 0,525 | 1,313 | 48,7 % |
| c051 | +28,8 ± 2,2 | +25,2 ± 2,1 | 1,017 | 0,525 | 1,311 | 48,7 % |
| c052 | +27,1 ± 2,2 | +22,8 ± 2,0 | 1,016 | 0,524 | 1,308 | 48,9 % |
| c053 | +27,9 ± 2,2 | +24,3 ± 2,1 | 1,014 | 0,525 | 1,307 | 48,9 % |
| c054 | +30,2 ± 2,1 | +26,8 ± 2,1 | 1,013 | 0,524 | 1,305 | 48,9 % |
| c055 | +30,7 ± 2,1 | +25,6 ± 2,0 | 1,014 | 0,524 | 1,305 | 49,0 % |
| c056 | +30,0 ± 2,1 | +23,7 ± 2,2 | 1,016 | 0,524 | 1,302 | 49,0 % |
| c057 | +29,4 ± 2,3 | +26,4 ± 2,1 | 1,019 | 0,523 | 1,301 | 49,0 % |
| c058 | +29,2 ± 2,2 | +26,2 ± 2,1 | 1,017 | 0,523 | 1,298 | 49,1 % |
| c059 | +28,7 ± 2,2 | +24,9 ± 2,1 | 1,016 | 0,523 | 1,299 | 49,1 % |
| c060 | +30,6 ± 2,2 | +27,8 ± 2,1 | 1,014 | 0,522 | 1,298 | 49,1 % |

c060 par taille (2, 3, 4, 5 joueurs) : Difficile +40,9, +24,6, +29,3, +27,7 ; Normale +46,5,
+22,0, +20,2, +22,6.

- **Le plateau de la deuxième boucle cède** : vers +25 et +21 jusqu'à c045, puis la montée
  reprend, vers +30 et +26 à partir de c054. Les gains vont aux tables de 3, 4 et 5 joueurs
  (Normale à 3 joueurs : +6,1 pour c040, +22,0 pour c060).
- La valeur de chaque pas baisse de nouveau (1,028 → 1,014), la politique continue (1,329 →
  1,298).

Mesures de fin (`lab2/runs/final3.sh`, résultats `lab2/runs/final3` ; machine reprise par AWS
pendant les arènes, faute de place, puis relancée : 11 + 39 min, 0,63 $). Courbe sur les parties
réservées des cycles 41 et 42 (jouées par c039 et c040 ; 84 443 positions) et 59 et 60 (jouées
par c057 et c058 ; 83 244 positions) :

| Réseau | Valeur 41-42 | Possession | Politique | Valeur 59-60 | Possession | Politique | Précision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c020 (64x4) | 1,054 | 0,590 | 1,484 | 1,050 | 0,603 | 1,507 | 39,7 % |
| c030 | 1,036 | 0,552 | 1,390 | 1,025 | 0,565 | 1,423 | 43,1 % |
| c040 | 1,023 | 0,531 | 1,322 | 1,015 | 0,543 | 1,361 | 46,0 % |
| c045 | 1,018 | 0,526 | 1,320 | 1,010 | 0,536 | 1,339 | 47,4 % |
| c050 | 1,014 | 0,525 | 1,325 | 1,009 | 0,530 | 1,320 | 48,2 % |
| c055 | 1,014 | 0,525 | 1,336 | 1,008 | 0,526 | 1,305 | 49,4 % |
| c060 | 1,011 | 0,527 | 1,345 | 1,006 | 0,523 | 1,296 | 49,7 % |

Arènes de c060, 920 donnes, graine 1004 :

| Face à | Score | 2 joueurs | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| c050 (n = 200 des deux côtés) | +0,2 ± 1,5 | −0,1 | +1,9 | +2,3 | −3,2 |
| c040 (n = 200) | +5,3 ± 1,6 | +5,4 | +4,7 | +6,5 | +4,5 |
| lab1 (c060 à n = 800, gen11 à n = 200) | +0,7 ± 1,6 | +3,6 | +0,5 | +1,0 | −2,3 |

- **À égalité avec lab1** (−2,2 pour c040, −12,6 pour le c020 64x4), toujours sans rien de lab1.
  132 ms par décision pour c060 sur AWS, p95 452 ms.
- **Tout le gain de la boucle vient de c041 à c050** : c060 bat c040 de 5,3 points mais pas c050
  (+0,2, rejoué à l'identique après la reprise de la machine) ; la valeur ne gagne plus que
  0,003 de c050 à c060 sur les parties récentes. La montée face aux IA du jeu après c050 (vers
  +30 et +26) ne se retrouve pas face à c050 : de quoi mieux exploiter ces IA, pas de progrès
  général mesurable.
- Positions des cycles 1 à 48 effacées du bucket ensuite (`lab2/prune.sh`, 70,5 Go gardés).
