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

gen3 : la valeur apprise sur `label1` seul, la politique sur les recherches de gen0 et gen1
(`:policy`), en partant de gen1 ; 2,4 millions de positions, 6 passes, 9 minutes. Mesures contre
l'IA Normal (100 donnes, graine 31, 350 parties, ± 7 points), recherche guidée c=2,5 :

| recherche | gen1 | gen2 | gen3 | durée (gen3) |
| --- | --- | --- | --- | --- |
| n=200, fin simulée complète | +52 | +45 | +52 | 194 s |
| n=200, profondeur 5 | +20 | +31 | **+54** | 88 s |
| n=200, profondeur 2 | −3 (60 donnes) | | +44 | 59 s |
| n=400, profondeur 2 | | | +44 | 121 s |
| n=200, réseau seul | −11 | −1 | +28 | 21 s |
| n=800, réseau seul | | | +32 | 93 s |

- le jugement appris sur des fins simulées moyennées vaut enfin quelque chose : la fin simulée
  s'arrête après 5 tours de joueur sans rien perdre, deux fois plus vite ;
- le réseau seul passe de −1 à +28, mais profite peu de plus d'itérations (+32 à n=800) : son
  jugement reste en deçà de quelques tours joués pour de vrai ;
- correctif : la validation faisait la moyenne par lot, faussée quand un lot n'a presque pas de
  cible de politique ; elle pondère maintenant par le nombre de positions de chaque cible.

Meilleure recherche à ce stade : gen3, n=200, profondeur 5 (+54 en 88 s).

## 2026-10-18 — Réseau plus gros, entraînement plus long, réglages : plateau

Mêmes données que gen3, 20 passes au lieu de 6 :

- gen3l (64×4, depuis gen1, 33 min) : valeur en validation 0,674 (elle baisse jusqu'au bout),
  politique figée dès la 5e passe (ses 450 000 positions sont trop peu) ;
- gen3b (128×6, depuis zéro, 51 min) : 0,678, et sa politique se dégrade (apprentissage par
  cœur) ; la recherche est 3,3 fois plus lente.

Contre l'IA Normal (200 donnes, graine 41, 700 parties, ± 3 % de victoires) :

| réseau | profondeur 5 | réseau seul |
| --- | --- | --- |
| gen3 | +50 | +28 |
| gen3l | +50 | +31 |
| gen3b | +49 | +28 |

Ce ne sont ni la taille du réseau ni la durée d'entraînement qui limitent, mais les données.

Réglages de la recherche (gen3l, n=200, mêmes donnes ; référence profondeur 5, c=2,5, w=0 : +50) :

| variante | écart |
| --- | --- |
| profondeur 3 | +49 |
| profondeur 8 | +55 |
| roll=0,5 (moitié réseau, moitié fin simulée) | +50 |
| c=1,5 / c=4 | +52 / +52 |
| w=0,5 / w=1 (probabilité de victoire dans la valeur) | +55 / +53 |
| w=0,5, s=4 / s=16 | +53 / +54 |
| w=0,5, profondeur 8 | +56 |

Tout se tient entre +49 et +56 : contre l'IA Normal (84 % de victoires), la mesure sature et ne
départage plus les réglages fins. Il faut un adversaire plus fort.

## 2026-10-18 — gen4 : la politique des recherches de gen3

Parties de gen3l contre lui-même (n=200, profondeur 5, bruit 0,25, la moitié des coups cherchés à
fond et enregistrés, les autres en 50 itérations) : 12 000 parties, 810 000 positions, 2 h 15.
gen4 : la valeur sur `label1`, la politique sur ces recherches et celles de gen1, depuis gen3l,
10 passes (précision de la politique en validation : 42 %).

Contre l'IA Difficile (200 donnes, graine 51 ; n=200, profondeur 5, w=0,5) :

| réseau | écart |
| --- | --- |
| gen3l | +27,5 |
| gen4 | +33,1 |

Face à face, réseau seul (n=200, w=0,5 ; 200 donnes) : gen4 contre des gen3l +2,1, gen3l contre
des gen4 −1,4. Léger mieux.

`lab label --agent` : les positions jugées peuvent venir de parties d'un autre joueur. `label2` :
60 000 parties jouées par gen3l (n=50, réseau seul, 10 % de coups au hasard).

## 2026-10-18 — Le jugement du réseau n'est plus le facteur limitant

`label3` (60 000 parties de gen4) et `label4` (80 000 parties de gen5) : ~0,8 et ~1,1 million de
positions jugées. `train.py --lazy` lit les données à la demande (`np.memmap`), plus grosses que la
mémoire. gen5 : valeur sur label1 + label2, politique sur gen3, depuis gen4 ; gen6 : + label3,
depuis gen5.

| réseau | contre l'IA Difficile (n=200, profondeur 5, w=0,5) | face-à-face, réseau seul |
| --- | --- | --- |
| gen5 | +31,0 (n=100 : +29,9 ; n=400 : +34,1) | +1,2 / −3,0 contre gen4 |
| gen6 | +30,8 | +0,1 / −2,1 contre gen5 |

`lab noise` : chaque position (parties de l'IA Normal) est jugée deux fois par 32 fins simulées ;
l'écart donne le bruit des cibles, le réseau (gen6) est comparé à leur moyenne. Avance du joueur qui
agit sur son meilleur rival, en pièces :

| positions | dispersion des cibles | bruit d'un jugement | erreur du réseau |
| --- | --- | --- | --- |
| toutes | 17,6 | 1,7 | 6,9 |
| 2 joueurs | 22,0 | 1,8 | 8,6 |
| 5 joueurs | 15,1 | 1,5 | 5,8 |
| dernière manche | 20,9 | 0,1 | 2,3 |
| 6 manches ou plus restantes | 14,2 | 2,4 | 8,6 |

Le réseau se trompe quatre fois plus que le bruit de ses cibles, et ni plus de données (gen3l →
gen6 : 6,96 → 6,89) ni un poids quatre fois plus fort sur les pièces à l'entraînement n'y changent
rien.

Mais est-ce ce qui limite la force ? `rolls=K` : K fins simulées par position nouvelle (la première
dans le monde de l'itération, les autres dans d'autres mondes possibles). gen6 contre l'IA
Difficile (200 donnes) :

| jugement d'une position nouvelle | écart | durée |
| --- | --- | --- |
| 1 fin de 5 tours, puis le réseau | +30,8 | 6 min |
| 4 fins de 5 tours, puis le réseau | +33,5 | 18 min |
| 4 fins complètes | +30,7 | 26 min |

Un jugement bien moins bruité n'apporte presque rien : la précision de la valeur ne limite plus.
Détail contre l'IA Difficile : 2 joueurs 75,5 % (attendu 50 %), 3 joueurs 64 %, 4 joueurs 58,5 %,
5 joueurs 50,8 % (attendu 20 %).

## 2026-10-18 — Des fins simulées jouées par le réseau

Contre l'IA Difficile (gen6, 200 donnes), plus de calcul aide un peu : n=800 +37,5, profondeur 8
+34,5, les deux +35,6 (n=200, profondeur 5 : +30,8).

Le réseau seul, sans recherche (n=1, son coup préféré), fait +27,6 contre l'IA Normal et +16,6
contre l'IA Difficile : il joue bien mieux que l'IA Normal qui joue les fins simulées.
`rp=1` : le réseau joue les fins simulées pour tout le monde ; `rp=2` : pour le joueur qui
cherche seulement, l'IA Normal pour les autres ; `rt` : température du tirage des coups.
Une évaluation du réseau : 72 µs (une décision de l'IA Normal : ~8 µs).

| fins simulées (n=200, profondeur 5) | 2 j. | 3 j. | 4 j. | 5 j. | écart | durée |
| --- | --- | --- | --- | --- | --- | --- |
| IA Normal pour tous | 75,5 % | 64,0 % | 58,5 % | 50,8 % | +30,8 | 6 min |
| réseau pour tous | 93,0 % | 55,7 % | 48,2 % | 44,8 % | +26,4 | 14 min |
| réseau pour tous, rt=0,5 | 91,0 % | 66,7 % | 62,7 % | 49,6 % | +34,4 | 13 min |
| réseau pour tous, profondeur 3 | 90,0 % | 60,7 % | 45,2 % | 54,4 % | +29,6 | 9 min |
| réseau pour moi, Normal pour les autres | 89,0 % | 71,3 % | 63,2 % | 56,2 % | +37,6 | 6 min |
| idem, rt=0,5 | 86,0 % | 70,7 % | 72,5 % | 58,2 % | **+40,4** | 6 min |

À deux, jouer soi-même fort dans les fins simulées change tout. À plusieurs, ce qui compte est de
prévoir les adversaires : l'IA Normal ressemble à l'IA Difficile (mêmes heuristiques), un réseau
qui simule des adversaires à son image se trompe sur eux. `rp=2` prend le meilleur des deux, pour
le même temps de calcul.

## 2026-10-18 — Point d'étape (le calcul passe sur AWS)

Meilleure IA à ce jour : `az:net=lab/nets/gen6.bin,n=200,w=0.5,c=2.5,roll=1,depth=5,rp=2,rt=0.5`,
+40 contre l'IA Difficile (200 donnes, graine 51), pour le même calcul que la référence (+31).

Pistes ouvertes, par ordre d'intérêt :
1. `rp=2` : confirmer sur plus de donnes, puis essayer rt (0,25 ; 1), la profondeur (3, 8) et n
   (400, 800) ; à n=800 la référence passait de +31 à +37.
2. Parties d'auto-apprentissage avec cette recherche (politique plus forte), puis gen7 : valeur sur
   label1 à label4 (label4 pas encore appris), politique sur ces parties.
3. Le réseau juge mal (erreur ~7 pièces sur l'avance, `lab noise`) mais ce n'est pas ce qui limite
   la force (`rolls=4` n'apporte que +3) : inutile d'y investir pour l'instant.

Organisation : le calcul part sur des machines AWS (EC2 Spot, processeur seul) ; mini (i5-8500T,
6 cœurs, sert aussi le jeu) ne fait qu'orchestrer : il garde les données (`lab/data`, 39 Go) et
les poids, lance les machines, y envoie le travail, rapatrie les résultats et les éteint. Pas de
calcul sur mini. `lab/py/setup.sh cpu` : entraînement sur processeur ; `train.py --lazy` si les
données ne tiennent pas en mémoire.

## 2026-10-18 — Les machines AWS

Compte AWS « nouvelle expérience » : projet SmallAI (compte 631245464763), plan payant, plafond de
dépense de 50 $/mois (compté avant les crédits). AWS y interdit le Spot (règle
`DenySpotMarketType`, plan gratuit comme payant ; seule l'activation des « fonctionnalités
avancées », irréversible, la lève, mais supprime le plafond) : machines à la demande, région
eu-north-1 (c7a.4xlarge 0,88 $/h, c7a.16xlarge 3,52 $/h). Quota à la demande : 64 vCPU (une
c7a.16xlarge à la fois).

- `lab/tools/cloud.sh` (profil `smallai`, utilisateur `smallai-orchestrator`) : `job [--type T]
  [--data D1,D2] [--get chemins] -- 'commande'` lance une machine (Debian 13, Rust, labo compilé,
  `setup.sh cpu`), copie les données du bucket, calcule, rapatrie, détruit, vérifie qu'il ne reste
  ni machine, ni disque, ni adresse ; `status` : machines et dépense du mois. Sur la machine,
  extinction (donc destruction) après 20 min sans calcul et au bout de 12 h quoi qu'il arrive.
- `lab/data` est copié dans `s3://smallai-lab-631245464763/data` (mini → bucket : ~33 Mo/s), lu
  gratuitement par les machines (rôle `smallai-worker`, ce seul bucket).
- Politiques : `lab/tools/aws/`.

Vitesse (`lab label`, 5 min) : c7a.4xlarge (16 cœurs AMD EPYC 9R14) 570 positions/s, i7-12700KF
518. Installation et compilation : 1 min 20 ; test complet : 6 min, 0,10 $.

## 2026-10-18 — `rp=2` confirmé sur 1 000 donnes

gen6 contre l'IA Difficile, 1 000 donnes (graine 61), n=200, profondeur 5, w=0,5 ; c7a.4xlarge :

| fins simulées | 2 j. | 3 j. | 4 j. | 5 j. | écart | durée |
| --- | --- | --- | --- | --- | --- | --- |
| IA Normal pour tous | 80,2 % | 68,1 % | 59,0 % | 51,8 % | +32,8 ± 1,6 | 14 min |
| réseau pour moi (`rp=2`) | 86,3 % | 76,6 % | 64,5 % | 57,6 % | **+39,2** ± 1,5 | 26 min |
| idem, rt=0,5 | 87,3 % | 74,8 % | 62,8 % | 53,9 % | +37,1 ± 1,6 | 25 min |

`rp=2` vaut +6 à +7 à tous les nombres de joueurs ; rt=0,5 ne fait pas mieux que rt=0 (l'avance
sur 200 donnes était du bruit). Sur cette machine, `rp=2` prend deux fois le temps de la référence
(sur l'i7, autant) : les fins de l'IA Normal y vont deux fois plus vite, le réseau à peine plus.
Coût : 67 min, 1,01 $.

Autres réglages avec `rp=2` (mêmes 1 000 donnes, c7a.16xlarge, 64 cœurs) :

| réglage | écart | durée |
| --- | --- | --- |
| n=200, profondeur 3 | +38,0 | 6 min |
| n=200, profondeur 5 | +39,2 | (26 min sur 16 cœurs) |
| n=200, profondeur 5, rt=1 | +35,4 | 8,5 min |
| n=200, profondeur 8 | +38,8 | 12 min |
| n=400, profondeur 5 | +40,0 | 17,5 min |
| n=800, profondeur 5 | **+42,0** | 35 min |

La profondeur ne compte pas (3 vaut 5 et 8, pour moitié moins de calcul) ; tirer les coups du
réseau au hasard (rt=1) coûte 4 points ; doubler n rapporte ~1,5 point. Coût : 81 min, 4,77 $.

## 2026-10-18 — gen7 : la politique des recherches `rp=2`

Parties de gen6 contre lui-même (n=200, profondeur 3, `rp=2`, bruit 0,25, la moitié des coups
cherchés à fond et enregistrés, les autres en 50 itérations) : 20 000 parties, 1,42 million de
positions (`gen6`), 61 min sur 64 cœurs. Compilée pour le processeur (`target-cpu=native`), la
recherche ne va pas plus vite.

Entraînement sur processeur (64 cœurs) : 63 s par passe sur 811 000 positions, aussi vite que la
carte graphique. gen7 : la valeur sur label1 à label4, la politique sur `gen6`, depuis gen6,
8 passes de 8 min (6,1 millions de positions) ; précision de la politique en validation 51 %
(gen6 : 44 %, sur d'autres parties), brier 0,108.

Contre l'IA Difficile (1 000 donnes, graine 61 ; n=200, profondeur 3, `rp=2`) :

| réseau | 2 j. | 3 j. | 4 j. | 5 j. | écart |
| --- | --- | --- | --- | --- | --- |
| gen6 | 85,9 % | 75,1 % | 63,1 % | 56,6 % | +38,0 |
| gen7 | 89,0 % | 79,4 % | 67,0 % | 57,7 % | **+40,9** |

Presque gen6 à n=800, profondeur 5 (+42,0), pour six fois moins de calcul. Face à face (même
recherche, 400 donnes) : gen7 contre des gen6 +2,0, gen6 contre des gen7 −3,0. Coût de gen7
(parties, entraînement, évaluation) : ~11 $.

## 2026-10-18 — Point d'étape (gen7)

Meilleure IA à ce jour : `az:net=lab/nets/gen7.bin,n=200,w=0.5,c=2.5,roll=1,depth=3,rp=2`, +40,9
contre l'IA Difficile (1 000 donnes, graine 61). Dépense de la journée sur AWS : ~16 $ (crédits
restants : 160 $) ; `lab/data/gen6` est sur mini et dans le bucket.

Pistes ouvertes, par ordre d'intérêt :
1. Un tour de plus : parties de gen7 contre lui-même (mêmes réglages que pour `gen6`, 20 000
   parties, ~1 h), gen8 (valeur sur label1 à label4, politique sur `gen7` et peut-être `gen6`,
   depuis gen7, 8 passes, ~1 h), puis les mêmes mesures (IA Difficile 1 000 donnes graine 61,
   face-à-face 400 donnes). ~10 $. Le gain tient-il (+2,9 de gen6 à gen7) ?
2. La valeur n'apprend plus de label1 à label4 (brier ~0,11 depuis gen5) : de nouvelles positions
   jugées (`lab label --agent` avec gen7) si elle redevient le facteur limitant.
3. Mettre cette IA dans le jeu : mesurer le temps d'une décision (n=200, `rp=2`, profondeur 3) sur
   un téléphone, chercher une version allégée.
