# Labo 2 : Gumbel AlphaZero, nœuds de hasard, coups jeton après jeton

Ouvert le 21 octobre 2026, à partir de lab1, figé ce jour-là (`lab1/JOURNAL.md` : tout ce qui a
été mesuré jusqu'à gen11). Règles communes des labos, protocole de mesure, machines AWS :
`labtools/README.md`.

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
- **Réseau** (`lab2/brain/src/net.rs`, `lab2/py/model.py`) : graphe des régions, 4 couches de 64 ;
  têtes de politique, de valeur (victoire, pièces gagnées) et, à l'entraînement seulement, de
  possession (à qui appartient chaque région au prochain tour du joueur). gen0
  (`lab2/nets/gen0.bin`, écrit par `python lab2/py/model.py`) est un réseau neuf : sorties à zéro,
  il joue au hasard. Calcul Rust en `f32` (AVX2) ; sur un processeur qui a AVX-512 BF16 (AWS),
  les grands produits en bf16, choisis automatiquement : mêmes résultats que leur référence
  scalaire, écarts au `f32` de 10⁻⁴ sur la victoire et 0,004 sur les logits (`lab2 bench-net`).
- **Jeu** : lab2 n'y est pas tant que la boucle n'a pas de réseau qui vaille ; il reviendra
  (`src/shared/experts.ts`) à n = 800. Arène d'essai sur mini, un cœur, 3 joueurs face à Normale :
  65 ms en moyenne par décision à n = 800, p95 224 ms, fins de tour comprises (lab1 dans le jeu
  sur mini : 80 à 250 ms, médiane 170 ms).
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
  réseau k−1 sur une fenêtre des derniers cycles (toutes les positions jusqu'à 250 000, puis
  250 000 × (1 + 0,4 × ((total / 250 000)^0,75 − 1) / 0,75)), chaque nouvelle position servant en
  moyenne 4 fois ; pendant ce temps, le réseau k−1 joue les 25 000 parties du cycle k+1 (un cycle
  de retard, comme les joueurs d'AlphaZero ou de KataGo pendant l'apprentissage) ; puis arènes
  du réseau k face à Difficile et Normale (460 donnes chacune, mêmes donnes à chaque pas). Pas de
  sélection du réseau, comme AlphaZero.
- **Temps** (`c7a.16xlarge`, mesures du 21 octobre avec un réseau de 3 cycles d'essai) : 124
  parties/s sur 64 cœurs en bf16 (82 en `f32`), 104 sur 48 cœurs, 102 pendant l'entraînement ;
  entraînement 12 800 positions/s sur 16 cœurs en bf16 (9 900 en `f32` ; 10 000 pendant les
  parties ; plus rien à gagner au-delà de 16 à 32 cœurs, la mémoire limite) ; arènes 57 s
  (Difficile) et 22 s (Normale) à 920 donnes. Un pas : environ 5 min (25 000 parties en 245 s
  pendant 215 s d'entraînement, puis 40 s d'arènes), contre environ 10 min en `f32` sans
  recouvrement. Sur mini (6 cœurs, AVX2) : l'entraînement plafonne à 4 cœurs, et le bf16 y est
  émulé (15 fois plus lent) : aucun entraînement réel sur mini.

## Plan

1. ~~Recherche : Gumbel AlphaZero et nœuds de hasard.~~ Fait le 21 octobre.
2. ~~Coups : pouvoirs et jetons un par un, paix et déclin à part ; l'Expert du jeu renvoie sa fin
   de tour en actions groupées.~~ Fait le 21 octobre.
3. ~~Réseau, format des positions, entraînement ; départ de zéro.~~ Fait le 21 octobre.
4. ~~Boucle par petits cycles à la manière de KataGo (possession, surprise, fenêtre), essai sur
   mini, mesures sur AWS, bf16 et recouvrement.~~ Fait le 21 octobre.
5. Première boucle sur AWS depuis gen0 (accord de l'utilisateur) : 20 pas, environ 1 h 45, suivie
   par les arènes de chaque pas ; lab2 revient dans le jeu avec son meilleur réseau.

## Résultats

Aucun pour l'instant.
