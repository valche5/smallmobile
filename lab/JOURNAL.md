# Journal de l'IA hors jeu

Objectif : l'IA de SmallMobile la plus forte possible, sans limite de temps ni de calcul, hors du
jeu (elle peut réfléchir aussi longtemps qu'elle veut). Couvre toutes les tables : 2 à 5 joueurs,
cartes générées.

Règle d'honnêteté : l'IA sait tout ce qu'un joueur humain attentif peut savoir, rien de plus.
- Elle ne lit ni l'ordre caché des pioches ni les dés à venir (`know::determinize` les retire).
- Elle connaît l'ordre des cartes rendues sous la pioche quand un peuple disparaît (c'est public).
- Elle connaît les pièces de tous : cachées dans l'interface, elles se recalculent exactement à
  partir de ce qui est annoncé (5 au départ, prix et pièces de chaque choix de peuple, gain de
  chaque tour). Un humain qui compte les connaît aussi.

Méthode visée (voir le plan) : recherche arborescente (MCTS) sur des mondes possibles, guidée par
un réseau de neurones sur le graphe des régions, entraîné par auto-apprentissage (style AlphaZero).
Simulation en Rust (`lab/`), entraînement en Python (PyTorch).

Commandes (depuis la racine) :

```
cargo build --release -p smallmobile-lab
./target/release/lab bench --agent normal
./target/release/lab arena --hero hard --field normal --deals 1000
cargo test -p smallmobile-lab
```

Lecture de l'arène : format « duplicate ». Chaque donne (carte, pioches, dés) est rejouée en
mettant le joueur testé à chaque place, face au reste de la table. « Attendu » est le taux de
victoire à force égale (1/n). L'écart à l'attendu, toutes tables confondues, résume la force.

---

## 2026-10-17 — Étape 0 : banc d'essai

Fait :
- crate `lab/` dans l'espace de travail, sans toucher au noyau ;
- `table` : les mêmes cartes que le jeu (`mapgen::generate`), résultat d'une partie (pièces, puis
  jetons sur la carte pour départager, comme `ranking` dans l'interface) ;
- `know` : suivi des cartes connues au fond des pioches, et `determinize` (monde possible : ordre
  caché des pioches mélangé, dés réinitialisés) ;
- `moves` : les coups regroupés pour la recherche. Une décision = un coup : choix du peuple,
  déclin, chaque conquête / dé / abandon / sort / dragon, fin des conquêtes ; la fin de tour est un
  seul coup (héros, forteresse, campements posés selon la menace, puis un placement parmi les 6
  plans du noyau, paix des Diplomates, déclin des Robustes) ; la défense : 2 plans ;
- `agent` : IA du jeu (Facile, Normal, Difficile) vues à travers un monde possible, et un joueur au
  hasard ; `arena` : parties en parallèle, format duplicate ;
- tests : chaque coup proposé est jouable ; le fond connu des pioches est exact.

Mesures (un cœur, carte générée comprise) :

| table | Normal, ms/partie | Difficile, ms/partie | coups/partie |
| --- | --- | --- | --- |
| 2 joueurs | 0,85 | 114 | ~110 |
| 3 joueurs | 1,3 | 317 | ~160 |
| 4 joueurs | 1,8 | 565 | ~200 |
| 5 joueurs | 2,3 | 839 | ~225 |

Sur 20 cœurs : de l'ordre de 10 000 parties Normal par seconde. Le moteur n'est pas le goulot.

Étalons (taux de victoire du premier contre une table du second, attendu à force égale entre
parenthèses) :

| | 2 j. | 3 j. | 4 j. | 5 j. | écart total |
| --- | --- | --- | --- | --- | --- |
| Normal contre hasard | 100 % | 100 % | 98 % | 100 % | +71 pts |
| Normal contre Facile | 87 % (50) | 68 % (33) | 64 % (25) | 60 % (20) | +38 pts |
| Difficile contre Normal | 85 % (50) | 54 % (33) | 42 % (25) | 32 % (20) | +19 pts |
| Normal contre Difficile | 15 % (50) | 21 % (33) | 19 % (25) | 16 % (20) | −11 pts |
| Normal contre Normal | 49 % | 35 % | 25 % | 20 % | +0,4 pt (contrôle) |
| Difficile qui voit tout contre Difficile | 55 % | 29 % | 25 % | 22 % | +0,4 pt |

(1000 donnes, sauf hasard 400 et Normal contre Difficile / voit tout 400 ; ± 2 à 4 points.)

Constats :
- le contrôle Normal contre Normal tombe pile sur l'attendu : l'arène est juste ;
- l'IA Difficile est nettement plus forte que Normal à 2 joueurs, beaucoup moins à 5 ;
- voir les pioches et les dés n'apporte rien de mesurable à l'IA Difficile : la version honnête
  sert d'étalon sans perte.

Prochaine étape : recherche (MCTS) sur mondes possibles, sans réseau, avec l'IA Normal pour finir
les parties simulées. Premier but : battre nettement l'IA Difficile.

## 2026-10-17 — Étape 1 : recherche sans réseau (MCTS)

`mcts` : recherche Monte-Carlo sur mondes possibles « en boucle ouverte ». Chaque itération tire
un monde (ordre caché des pioches, dés), descend l'arbre des suites de coups, essaie un coup
nouveau, puis finit la partie avec l'IA Normal pour tout le monde. Plusieurs joueurs : chacun
maximise sa propre récompense (max^n), et un coup possible dans certains mondes seulement est jugé
sur ces mondes-là (UCB avec disponibilité). Un coup en plus à la fin de tour : « finir comme l'IA
Normal » (son placement appris), en plus des 6 plans.

Premier essai, 500 itérations par décision contre l'IA Difficile (100 donnes) :

| | 2 j. | 3 j. | 4 j. | 5 j. | écart total |
| --- | --- | --- | --- | --- | --- |
| MCTS n=500 contre Difficile | 60 % (50) | 35 % (33) | 40 % (25) | 36 % (20) | +12 pts |

Réglages, 300 itérations contre l'IA Normal (100 donnes, ± 4 à 5 points) — c : exploration,
w : poids de la victoire dans la récompense (le reste : avance en pièces, sigmoïde d'échelle 8) :

| c \ w | 0 | 0,5 | 1 |
| --- | --- | --- | --- |
| 0,3 | **79 %** (+51 pts) | 72 % (+44) | 65 % (+36) |
| 0,7 | 74 % (+46) | | |

Constats :
- la recherche bat déjà l'IA Difficile, surtout à 4-5 joueurs, où l'IA du jeu est la plus faible ;
- récompenser l'avance en pièces plutôt que la seule victoire rend les simulations bien plus
  informatives (la victoire seule est trop bruitée sur des fins de partie jouées par l'IA Normal) ;
- peu d'exploration (c=0,3) vaut mieux.

Coût : une décision à 200 itérations prend ~50 ms à 2 joueurs, plus à 5.

## 2026-10-17 — Étape 2 : réseau de neurones et auto-apprentissage (mise en place)

- `encode` : la position vue du joueur qui agit, 77 entrées par région et 700 pour la partie,
  toutes de petits entiers (un octet). Test : deux mondes possibles donnent exactement le même
  encodage, le réseau ne peut donc rien savoir de caché ;
- réseau sur le graphe des régions (`lab/py/model.py`, recalculé à l'identique en Rust par
  `net.rs`, vérifié par `lab/py/check.py` : écart 5·10⁻⁵) : chaque région échange avec ses
  voisines et avec un résumé de la partie, 4 couches de 64 ; sorties : a priori de chaque coup
  (448 clés), probabilité de victoire de chaque joueur, pièces qu'il gagnera encore ;
- calcul en Rust sur le processeur (AVX2/FMA) : ~85 µs par position pour 64 × 4 couches, soit
  moins qu'une fin de partie jouée par l'IA Normal (~250 µs) ;
- `az` : recherche guidée par le réseau (PUCT, comme AlphaZero), même arbre en boucle ouverte ;
- `selfplay` : parties d'entraînement, 5,7 Ko par position ; tirage des coups selon les visites
  pendant les 2 premières manches ; « playout cap randomization » de KataGo disponible ;
- `lab analyze` : le meilleur coup d'une vraie partie (exportée du serveur par
  `lab/tools/export-room.ts`), avec la chance de victoire estimée de chaque coup ;
- PyTorch 2.14 pour ROCm 7.14 sur la RX 9070 XT (16 TFLOPS fp32) : `lab/py/setup.sh`.

Génération 0 : 6 000 parties de MCTS (n=200, c=0,3, w=0) contre lui-même, toutes tables,
~250 000 positions par heure.

## 2026-10-17 — Premiers réseaux : ce qui ne marchait pas, et pourquoi

Réseau d'essai sur les 66 000 premières positions de la génération 0 (~500 parties) :
- la valeur apprend (Brier 0,65 → 0,36), la politique presque pas : les cibles de MCTS sont
  presque uniformes (9,5 coups en moyenne, le plus visité n'a que 33 % des visites) ;
- MCTS lui-même ne choisit le même coup que dans 43 % des cas sur deux recherches indépendantes,
  même à 1 000 itérations : beaucoup de coups se valent vraiment (l'ordre des conquêtes, par
  exemple). La politique de ce jeu est floue par nature ; seule la force de jeu fait foi ;
- la recherche guidée par ce réseau perd lourdement contre l'IA Normal (4 % de victoires, −24
  points) : elle passe en déclin un tour sur deux. En cours de tour, les pièces du tour ne sont pas
  encore comptées ; le déclin finit le tour et les encaisse : un réseau mal entraîné le préfère.

Corrections, mesurées contre l'IA Normal (40 donnes, même graine, ± 8 points) :

| recherche guidée (200 itérations) | écart à l'attendu |
| --- | --- |
| réseau d'essai, victoire seule | −24 |
| réseau d'essai, avance en pièces prévue (w=0) | −16 |
| réseau pré-entraîné sur l'IA Normal, w=1 | +11 |
| idem + fins simulées (roll=1) | +26 |
| idem + valeurs normalisées dans chaque nœud (MuZero) | +40 |
| idem, c=2,5 | **+42** |
| réseau pré-entraîné seul, valeurs normalisées | +12 |
| MCTS (n=200, c=0,3, w=0), pour comparer | +36 |

- pré-entraînement : 100 000 parties de l'IA Normal (2 ms chacune), 10 % des positions gardées,
  soit 1,3 million de positions en 2 minutes. Le réseau y apprend à juger une position et imite
  l'IA Normal (78 % de coups identiques) ;
- normalisation : les écarts de valeur entre coups (quelques centièmes) étaient écrasés par le
  terme d'exploration ; la recherche jouait presque toujours le coup de l'a priori, donc celui de
  l'IA Normal. Ramenées entre 0 et 1 dans chaque nœud, les valeurs reprennent la main ;
- fins simulées (le premier AlphaGo) : la valeur d'une position nouvelle mêle le réseau et une fin
  de partie jouée par l'IA Normal. Avec un réseau encore jeune, ça aide beaucoup.

La recherche hybride égale déjà MCTS, avec un réseau qui ne fera que progresser.

## 2026-10-18 — Le jugement du réseau est le maillon faible

Génération 1 : l'hybride (a priori du réseau + fins simulées) joue contre lui-même. Premier
réseau gen1 : gen0 (2 000 parties MCTS) appris en partant du réseau pré-entraîné.

Mesures contre l'IA Normal (60 donnes, graine 21, ± 7 points) ; « profondeur » : la fin simulée
s'arrête après ce nombre de tours de joueur et le réseau juge la suite :

| réseau | fin simulée complète | profondeur 5 | profondeur 2 | réseau seul |
| --- | --- | --- | --- | --- |
| pré-entraîné (IA Normal) | | +33 | +26 | +17 (100 donnes) |
| gen1 (MCTS seul) | **+51** | +20 | −3 | −11 (100 donnes) |
| gen1b (MCTS + 30 % Normal, 2 passes) | +45 (100 donnes) | | +24 | +13 |

- gen1 a appris par cœur ses 2 000 parties : ~130 positions par partie partagent le même
  résultat, le réseau retient les parties au lieu de juger. La validation tirait des positions au
  hasard, donc dans les mêmes parties : elle ne le voyait pas. Elle se fait maintenant par blocs de
  parties entières ;
- mêler des parties de l'IA Normal (100 000 parties différentes) évite l'apprentissage par cœur
  (gen1b), mais le jugement reste « de niveau Normal » ;
- pour guider l'hybride, les a priori appris sur des recherches (gen1, plus ouverts) valent mieux
  que l'imitation de l'IA Normal (gen1b) : plus d'exploration, et les fins simulées font le reste.

Correctif de fond : des cibles moins bruitées. La recherche suit maintenant aussi les pièces
finales attendues de chaque joueur (moyenne sur ses ~200 simulations), enregistrées avec chaque
position (format `*.v2.bin`). Le réseau apprend sur un mélange du résultat réel et de cette
attente (`--q-weight 0.5`), bien plus stable qu'une partie unique (exemple : attendu 116 et 98,
réel 119 et 93).

Génération 2 : 5 000 parties de l'hybride (gen1, fins complètes) au nouveau format, 166 000
positions ; gen2 apprend gen0 + gen1 + 10 % de l'IA Normal, en partant de gen1 (350 parties contre
l'IA Normal) :

| réseau | fin simulée complète | profondeur 5 | réseau seul |
| --- | --- | --- | --- |
| gen1 | **+52** | +20 | −11 |
| gen2 | +45 | +31 | −1 |

Pas de progrès : le réseau se trompe d'environ 9 pièces par position, et toujours dans le même
sens, alors que les coups ne se départagent souvent que de 1 à 3 pièces. Une fin simulée est plus
bruitée, mais sans biais : sur 200 simulations, ses erreurs se compensent.

## 2026-10-18 — Des positions jugées par 32 fins simulées

`lab label` : l'IA Normal joue des parties (10 % de coups au hasard, pour couvrir les positions
étranges que visite la recherche) ; une position sur dix est jugée par la moyenne de 32 fins de
partie, chacune dans un monde possible honnête. Cible de valeur presque exacte, sans cible de
politique : `train.py` masque l'une ou l'autre selon la source (`dossier:policy` : politique
seule).

`label1` : 125 000 parties, 1,96 million de positions jugées en 63 minutes (518 par seconde).
