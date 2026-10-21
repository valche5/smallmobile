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
régions, 4 couches de 64), l'auto-apprentissage avec recherches courtes, les tables variées. La
taille du réseau pourra augmenter si les mesures le demandent (accord de l'utilisateur, 21 octobre).

**Positions rejugées supprimées** (décision de l'utilisateur, 21 octobre, pendant les données de
gen1) : lab1 les gardait pour des cibles de valeur moins bruitées (moyenne de 32 fins jouées par le
réseau), mais leur gain y était faible et fragile (+2,4 ± 2,2 points sur les résultats seuls, une
mesure, P2), elles sont circulaires (le réseau juge ses propres fins) et coûtaient ici 72 s par
morceau de 500 parties contre 19 s pour les jouer. Avec gen0, elles étaient même faussées : ses
sorties de jetons et de pouvoirs valent toutes zéro, et à égalité son coup préféré met tous les
jetons sur une seule région. Le calcul va plutôt à plus de parties réelles.

## Système actuel

- **Recherche** (`lab2/brain/src/search.rs`) : Gumbel AlphaZero. Racine : 16 coups au plus, tirés
  selon logits + bruit de Gumbel (auto-apprentissage ; sans bruit ailleurs), halving séquentiel
  (calendrier de mctx). Hors racine : sélection déterministe vers la politique améliorée
  `softmax(logits + σ(q complétées))`, `σ = (50 + visites max) × 0,1 × q ramenée entre 0 et 1`,
  valeur mêlée pour les coups jamais essayés (constantes de mctx). Nœuds = positions publiques,
  suite d'un coup rangée par résultat de son hasard (`moves::chance` : carte arrivée au marché,
  réussite du dé de renfort, dé des Berserks) ; un monde possible tiré par itération ; coups
  forcés sans nœud. Feuille : valeur du réseau. Seul réglage : `n` (itérations).
- **Vitesse** : 200 itérations en 18 ms en moyenne sur mini avec gen11 et les coups de lab1 (lab1 :
  231 ms), 12,6 ms avec gen0 et les nouveaux coups, sur 24 positions de
  `lab2 speed --positions 6 --seed 3` (21 octobre).
- **Coups** (`lab2/brain/src/moves.rs`) : une décision à la fois. Fin de tour : `Lift` (forcé :
  un jeton par région, le reste en main, héros et campements relevés), puis les pouvoirs un par un
  (héros, forteresse, campement sur une région ; facultatifs : jouer un jeton ou finir le tour met
  fin à leur pose), puis les jetons un par un, puis la fin du tour (paix des Diplomates : aucune ou
  un adversaire relatif ; déclin des Robustes ; sinon forcée). Défense : les jetons rendus un par
  un. Le suivi de la fin de tour (`Redeploy`) est dans `Knowledge`. 608 clés : 32 globales, 9
  sortes × 64 régions (conquérir, dé, dragon, sorcellerie, abandon, héros, forteresse, campement,
  jeton). Une décision entière du jeu (`moves::decision`) se joue en actions groupées : héros et
  campements reposés au même endroit ne bougent pas, les jetons forment un seul placement.
- **Réseau de départ gen0** (`lab2/nets/gen0.bin`, 21 octobre) : gen11 de lab1 converti par un
  script ponctuel (non gardé) ; tronc, valeur et sorties des coups qui existaient déjà repris
  (peuples, passer, déclin, fin des conquêtes, Berserks, paix et Robustes depuis la fin de tour
  « plan 0 » de lab1, 5 sortes de coups sur une région) ; héros, forteresse, campement et jeton
  partent de zéro (logits égaux : placement uniforme avant le premier entraînement).
- **Jeu** : `lab2-expert`, n = 800 (`src/shared/experts.ts`), variante par défaut. Arène
  d'essai sur mini, un cœur, 3 joueurs face à Normale (2 donnes) : 65 ms en moyenne par décision,
  p95 224 ms, fins de tour comprises (n = 200 : 21 ms, p95 61 ms ; lab1 dans le jeu sur mini : 80
  à 250 ms, médiane 170 ms).
- **Arène** : un joueur rend une décision entière en actions du noyau ; témoin `lab1:net=…,n=…`
  (l'Expert de lab1 tel qu'il joue dans le jeu).
- **Auto-apprentissage** : bruit de Gumbel dans toutes les recherches ; le coup joué est celui de
  la recherche ; recherches courtes (`--full-share`, `--fast`) gardées. Positions `*.pos` : cible
  de politique = politique améliorée de la racine.
- **Entraînement** (`lab2/py/train.py`) : politique, victoire, pièces ; poids lus et écrits en
  SMNN1 (`.bin`), plus de `.pt`.

## Plan

1. ~~Recherche : Gumbel AlphaZero et nœuds de hasard.~~ Fait le 21 octobre.
2. ~~Coups : pouvoirs et jetons un par un, paix et déclin à part ; l'Expert du jeu renvoie sa fin
   de tour en actions groupées.~~ Fait le 21 octobre.
3. ~~Réseau (clés des nouveaux coups), format des positions, entraînement ; réseau de départ gen0
   tiré de lab1/gen11.~~ Fait le 21 octobre ; reste à vérifier sur AWS que PyTorch et Rust
   calculent le même réseau (`lab2/py/check.py`, sans PyTorch sur mini).
4. Première génération sur AWS (accord de l'utilisateur pour la dépense) : parties de gen0,
   entraînement de gen1, arène face à l'Expert de lab1.

## Résultats

Aucun pour l'instant.
