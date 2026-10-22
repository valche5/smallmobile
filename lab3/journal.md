# Labo 3 : un réseau par nombre de joueurs

Ouvert le 22 octobre 2026, à partir de lab2, figé après sa troisième boucle (`lab2/journal.md` :
la recherche, les coups, le réseau et la boucle viennent de là). Règles communes des labos,
protocole de mesure, machines AWS : `labtools/README.md`.

## Objectif

Le même que lab1 et lab2 : l'IA la plus forte possible sur les cartes générées, à 2, 3, 4 et 5
joueurs, mesurée aussi au temps de réflexion du jeu sur mini, avec les mêmes règles d'honnêteté
(l'Expert ne sait que ce qu'un joueur attentif peut savoir, et ne sait pas contre qui il joue).

## Pourquoi ce labo

Question de l'utilisateur (22 octobre) : **un réseau spécialisé dans un nombre de joueurs
joue-t-il mieux, à ce nombre, qu'un réseau qui apprend tous les nombres à la fois ?** Le jeu
change beaucoup d'une taille à l'autre (à 2 joueurs, lab2 domine les IA du jeu bien plus qu'à 3,
4 ou 5) ; un réseau par taille y consacre toute sa capacité et toutes ses parties, au prix de ce
qu'une taille apprend aux autres.

Décisions de l'utilisateur (22 octobre) : un réseau par nombre de joueurs, en commençant par 4
joueurs ; au départ, le petit réseau (64x4) et la petite fenêtre (250 000 positions) de la
première boucle de lab2 ; départ de zéro, comme lab2.

## Système actuel

Celui de lab2 (voir son journal), sauf :

- **Un nombre de joueurs par réseau** : `lab3/loop.sh P N` joue toutes ses parties
  d'auto-apprentissage et ses arènes à P joueurs ; parties `lab3/data/pP`, réseaux et arènes
  `lab3/runs/pP`. Le réseau garde les entrées et sorties de lab2 (jusqu'à 5 joueurs).
- **Réseau de départ** : le généraliste de lab2, son c060 (128x4, `lab3/nets/start.bin`), continué
  à un seul nombre de joueurs (décision de l'utilisateur, 22 octobre, après l'essai depuis zéro).
  `lab3/nets/gen0.bin`, le réseau neuf de lab2, ne sert plus qu'aux tests.
- **Fenêtre** : toutes les positions jusqu'à 4 000 000, puis
  4 000 000 × (1 + 0,4 × ((total / 4 000 000)^0,75 − 1) / 0,75) (250 000 pour l'essai depuis
  zéro, 1 000 000 pour le premier essai de spécialisation). Le réseau de départ joue d'abord les
  cycles qui la remplissent (4 à 4 joueurs, `data.filled`), et le premier entraînement part de
  là : un pas tire 4 fois les positions de son dernier cycle, soit une passe sur une fenêtre de
  quatre cycles, puis moins quand elle grandit (le c060 de lab2 : 0,36 passe par pas sur 11
  cycles, 275 000 parties).
- **Cœurs** : 48 de parties et 16 d'entraînement en même temps : avec le moteur de lab3, les
  parties d'un 128x4 n'ont plus besoin des 56 cœurs de lab2.
- **Témoin de l'arène** : l'Expert de lab2 (`lab2:net=lab2/nets/….bin,n=…`, `lab2/brain`), tel
  qu'il joue dans le jeu ; plus de témoin lab1.
- **Moteur de génération des parties** (`selfplay::run`, `batch`, demande de l'utilisateur :
  massivement parallèle, attentif aux caches du processeur). Chaque fil mène `CONCURRENT`
  parties à la fois ; la recherche attend ses évaluations (`search::Oracle`) : chaque partie
  avance jusqu'à sa prochaine évaluation, puis le fil les évalue toutes ensemble, couche après
  couche (`Net::eval_batch` : les poids d'une couche restent dans le cache du cœur d'une
  position à l'autre ; les vecteurs de partie de toutes les positions passent ensemble dans les
  couches qui les lisent, trois à la fois avec AVX-512). Les évaluations sont gardées d'une recherche à la suivante
  (`search::Evals`, une par partie : la décision d'après repasse par les positions que la
  précédente a explorées), chacune en quelques centaines d'octets (empreinte de 128 bits,
  valeur, logits des coups possibles). Tout ce que touche un fil lui appartient ; seuls les
  poids, en lecture, sont partagés. Chaque partie est la même, à l'octet près, que jouée seule
  (tests ; 1 024 parties comparées sur AWS). Le jeu et l'arène gardent une évaluation à la
  fois (`search::Direct`).
- **Vitesse** (22 octobre, c060 de lab2, n = 200, 4 joueurs, recette de l'auto-jeu ; ancien
  moteur : celui de lab2) : sur un cœur de mini, 4 parties en 21,0 s → 14,8 s (évaluations
  gardées seules) → 14,6 s ; sur `c7a.16xlarge`, 16 parties sur un cœur 33,6 s → 24,1 s, 1 024
  parties sur 64 cœurs 49,9 s → 34,0 s → 30,9 s avec les vecteurs de partie groupés (1,61 fois
  plus vite).
- **Profil** sur 64 cœurs (`perf`) : le réseau prend environ 82 % du temps, dont 53 % pour les
  grands produits bf16, 13 % pour les vecteurs de partie, 13 % pour le reste de son calcul ; le
  moteur du jeu environ 6 %. Seul ou à 64, un cœur fait 2,05 instructions par cycle et autant
  d'instructions par partie : les cœurs ne se gênent pas (caches, mémoire) ; à 64, le processeur
  passe de 3,68 à 2,78 GHz (AVX-512 sur tous les cœurs), d'où 1,32 fois moins vite par cœur.
- **Le réseau sur GPU** (préparé le 22 octobre, pas encore mesuré sur un vrai GPU) : `lab3
  selfplay --device gpu`, programme compilé avec `--features gpu` (`DEVICE=gpu lab3/loop.sh`). Un
  fil évaluateur appelle PyTorch (`lab3/py/serve.py`, par pyo3 : CUDA ou ROCm) ; les fils de jeu
  lui envoient leurs groupes, qu'il évalue tous ensemble ; chaque fil mène deux groupes de 32
  parties et fait avancer l'un pendant que le GPU évalue l'autre. PyTorch et Rust calculent le
  même réseau (écart de 5·10⁻⁵ en f32 ; test `the_gpu_evaluates_like_the_processor`), mais les
  parties ne sont plus identiques au bit près à celles du processeur (autre ordre des sommes,
  bf16 sur GPU). Machines : `labtools/cloud.sh` lance les types à GPU NVIDIA (g5, g6…) sur l'image
  d'AWS avec les pilotes, et compile avec l'option `gpu` ; leur quota Spot est à 0 sur le compte
  (demande à AWS nécessaire). g5.16xlarge (64 vCPU, A10G) à 0,52 $/h à Stockholm le 22 octobre.
- Retiré : `size.py` (mesure des tailles de réseau de lab2).

## Mesure

Le protocole commun garde le format duplicate à 2, 3, 4 et 5 joueurs ; un réseau spécialisé ne
joue qu'à sa taille, et se compare à lab2 à cette taille seulement :

- à chaque pas, arènes à P joueurs face à Difficile (graine 1002) et Normale (1003), 460 donnes ;
  les réseaux de lab2 rejoués sur ces mêmes donnes pour comparer ;
- en fin de boucle (`lab3/runs/final.sh`), 920 donnes à P joueurs (graine 1004), n = 200 des deux
  côtés, face au réseau de départ (le c060 de lab2) ; courbe des réseaux sur les mêmes positions
  de validation (`lab3/py/curve.py`). L'essai depuis zéro (`lab3/runs/final-zero.sh`) se mesurait
  au c020 64x4 de lab2 (même taille de réseau, même fenêtre, autant de parties, mais à toutes les
  tailles) et à son meilleur réseau.

## Plan

1. ~~Boucle à 4 joueurs depuis gen0, 20 pas (comme la première boucle de lab2), puis mesures de
   fin.~~ Faite le 22 octobre : à moyens égaux, le spécialiste bat le généraliste de 8,4 points à
   4 joueurs (voir Résultats).
2. Spécialiser le généraliste : c060 de lab2 continué sur des parties à 4 joueurs, jusqu'au
   cycle 20, avec le nouveau moteur. Premier essai arrêté au pas 3 le 22 octobre (fenêtre d'un
   cycle, le réseau baissait : voir Résultats) ; relancé le 22 octobre avec la fenêtre remplie
   d'abord par c000 (cycles 1 et 2 de l'essai repris : mêmes parties).
3. Selon le résultat : les autres nombres de joueurs, un Expert du jeu qui choisit son réseau
   d'après le nombre de joueurs.
4. Le réseau sur GPU (demande de l'utilisateur, 22 octobre) : le moteur groupe déjà les positions
   de toutes ses parties.

## Résultats

Scores du protocole (`labtools/score.py`), à 4 joueurs seulement : l'écart à une victoire sur
quatre, de −25 (jamais) à +75 (toujours) ; n = 200.

### Spécialisation à 4 joueurs, premier essai arrêté (22 octobre)

`lab3/loop.sh 4 20` depuis le c060 de lab2, fenêtre minimale de 1 000 000 positions : une
`c7a.16xlarge` Spot, 64 min, 0,80 $, arrêtée au pas 3. Parties d'un cycle 708 à 884 s (48 cœurs
sauf le premier), entraînement 697 à 738 s (16 cœurs). Validation sur les parties réservées de
la fenêtre de chaque pas.

| Réseau | Difficile | Normale | Valeur | Possession | Politique | Précision |
| --- | --- | --- | --- | --- | --- | --- |
| c000 (c060 de lab2) | +24,5 ± 2,3 | +17,6 ± 2,4 | | | | |
| c001 | +18,9 ± 2,2 | +11,1 ± 2,2 | 1,173 | 0,541 | 1,309 | 48,1 % |
| c002 | +18,2 ± 2,3 | +12,5 ± 2,2 | 1,211 | 0,546 | 1,312 | 48,0 % |
| c003 | +15,1 ± 2,3 | +9,5 ± 2,3 | 1,201 | 0,561 | 1,377 | 44,9 % |

- **Le réseau perd 8 à 9 points en trois pas.** Cause probable : à 990 000 positions par cycle, la
  fenêtre (minimum 1 000 000, croissance lente) ne gardait qu'un cycle, et un pas tire 4 fois
  les positions de son dernier cycle : quatre passes, au taux plein, sur les mêmes 25 000
  parties. La valeur retient leurs issues : 1,096 à l'entraînement contre 1,173 à 1,211 en
  validation. Le c060 apprenait sur 11 cycles (275 000 parties), 0,36 passe par pas.
- La fenêtre n'aurait pris un deuxième cycle qu'au pas 5, et 5 au pas 20. D'où la relance avec
  la fenêtre remplie avant le premier entraînement (Système actuel).
- Réseaux et arènes : `lab3/runs/p4-try1` sur mini. Les parties des cycles 1 et 2 (jouées par
  c000) resservent à la relance ; celles des cycles 3 et 4 (c001 et c002), périmées, effacées du
  bucket.

### Boucle à 4 joueurs depuis zéro (22 octobre)

`lab3/loop.sh 4 20` depuis gen0 (fenêtre de 250 000 positions, 64x4), puis
`lab3/runs/final-zero.sh` (résultats `lab3/runs/p4-zero`, `lab3/runs/p4-zero-final`, parties
`lab3/data/p4-zero` : dossiers renommés ensuite) : une `c7a.16xlarge` Spot, 172 min, 2,17 $. 25 000 parties par cycle, 11
positions par partie au hasard, 37,5 en fin de boucle ; parties 127 à 471 s, entraînement 145 à
339 s ; fenêtre de 3 cycles au pas 20 (2,8 millions de positions), comme la première boucle de
lab2.

Arènes de chaque réseau à 4 joueurs (460 donnes) ; à côté, pour comparer, le même pas de la
première boucle de lab2 (même réseau, même fenêtre, autant de parties, mais à toutes les
tailles) sur ses 115 donnes à 4 joueurs, environ ±4,5 ; mesures de validation de chaque pas sur
les parties réservées de sa fenêtre (à 4 joueurs : pas comparables à celles de lab2).

| Réseau | Difficile | Normale | lab2 : Difficile | lab2 : Normale | Valeur | Possession | Politique | Précision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c001 | −22,9 ± 0,6 | −23,6 ± 0,6 | −24,3 | −24,3 | 0,828 | 0,284 | 1,237 | 54,6 % |
| c002 | −22,2 ± 0,8 | −23,3 ± 0,6 | −23,3 | −22,8 | 0,714 | 0,272 | 1,202 | 54,9 % |
| c003 | −15,8 ± 1,4 | −13,7 ± 1,6 | −19,6 | −15,8 | 0,825 | 0,468 | 1,635 | 32,7 % |
| c004 | −15,9 ± 1,3 | −11,8 ± 1,6 | −17,7 | −14,1 | 0,881 | 0,468 | 1,606 | 33,3 % |
| c005 | −16,3 ± 1,4 | −14,0 ± 1,5 | −12,3 | −12,4 | 1,076 | 0,475 | 1,536 | 37,8 % |
| c006 | −14,3 ± 1,5 | −12,9 ± 1,5 | −13,8 | −14,6 | 1,097 | 0,469 | 1,510 | 38,4 % |
| c007 | −12,8 ± 1,5 | −12,5 ± 1,5 | −11,4 | −12,2 | 1,121 | 0,465 | 1,542 | 37,9 % |
| c008 | −11,6 ± 1,6 | −10,5 ± 1,7 | −14,2 | −11,5 | 1,108 | 0,448 | 1,476 | 40,1 % |
| c009 | +0,1 ± 1,9 | −0,6 ± 1,9 | −8,2 | −5,4 | 1,119 | 0,451 | 1,477 | 41,0 % |
| c010 | −0,8 ± 1,9 | −1,0 ± 2,0 | −6,5 | −5,4 | 1,120 | 0,448 | 1,473 | 41,0 % |
| c011 | +5,2 ± 2,1 | +3,8 ± 2,1 | −1,5 | +0,7 | 1,132 | 0,460 | 1,450 | 42,4 % |
| c012 | +6,1 ± 2,1 | +7,9 ± 2,2 | +5,7 | +3,3 | 1,138 | 0,471 | 1,421 | 43,6 % |
| c013 | +9,0 ± 2,3 | +8,3 ± 2,4 | +11,6 | +3,2 | 1,146 | 0,484 | 1,411 | 44,2 % |
| c014 | +10,9 ± 2,3 | +10,4 ± 2,3 | +9,5 | +9,3 | 1,147 | 0,494 | 1,409 | 44,2 % |
| c015 | +13,0 ± 2,3 | +11,2 ± 2,2 | +7,7 | +8,5 | 1,162 | 0,503 | 1,400 | 44,4 % |
| c016 | +16,1 ± 2,4 | +10,8 ± 2,3 | +11,4 | +8,6 | 1,164 | 0,518 | 1,386 | 44,9 % |
| c017 | +15,8 ± 2,3 | +10,3 ± 2,3 | +14,9 | +8,6 | 1,157 | 0,526 | 1,383 | 45,3 % |
| c018 | +13,3 ± 2,2 | +10,9 ± 2,2 | +16,1 | +8,2 | 1,152 | 0,527 | 1,384 | 45,3 % |
| c019 | +15,4 ± 2,4 | +10,8 ± 2,2 | +12,3 | +10,0 | 1,155 | 0,528 | 1,384 | 45,4 % |
| c020 | +13,6 ± 2,3 | +9,3 ± 2,2 | +13,0 | +6,7 | 1,155 | 0,530 | 1,380 | 45,4 % |

Les réseaux de lab2 rejoués sur les mêmes donnes à 4 joueurs :

| Réseau | Difficile | Normale |
| --- | --- | --- |
| lab3 c020 (64x4, 4 joueurs, 20 cycles) | +13,6 ± 2,3 | +9,3 ± 2,2 |
| lab2 c020 (64x4, toutes tailles, 20 cycles) | +8,8 ± 2,3 | +8,5 ± 2,3 |
| lab2 c060 (128x4, toutes tailles, 60 cycles) | +24,5 ± 2,3 | +17,6 ± 2,4 |

Courbe sur les mêmes positions, parties réservées des cycles 19 et 20 de lab3 (jouées par c017 et
c018 ; 91 968 positions) :

| Réseau | Valeur | Possession | Politique | Précision |
| --- | --- | --- | --- | --- |
| lab3 c005 | 1,381 | 0,730 | 1,824 | 29,9 % |
| lab3 c010 | 1,215 | 0,663 | 1,612 | 36,1 % |
| lab3 c015 | 1,182 | 0,563 | 1,439 | 42,7 % |
| lab3 c020 | 1,161 | 0,531 | 1,379 | 45,6 % |
| lab2 c020 (64x4) | 1,182 | 0,575 | 1,491 | 39,9 % |
| lab2 c060 (128x4) | 1,153 | 0,555 | 1,518 | 39,9 % |

Arènes directes, 920 donnes à 4 joueurs, graine 1004, lab3 c020 face à trois réseaux de lab2 :

| Face à | Score | Victoires | Rang moyen |
| --- | --- | --- | --- |
| lab2 c020 (64x4) | +8,4 ± 1,5 | 33,4 % | 2,27 |
| lab2 c060 (128x4) | −9,0 ± 1,2 | 16,0 % | 2,92 |

- **À moyens égaux, le spécialiste gagne à sa taille** : +8,4 ± 1,5 face au généraliste de même
  réseau, de même fenêtre et d'autant de parties (toutes à 4 joueurs pour lui, un quart pour
  lab2) ; il prédit mieux le vainqueur (valeur 1,161 contre 1,182) ; face aux IA du jeu, +4,8
  face à Difficile et +0,8 face à Normale sur les mêmes donnes. Son avance mêle la
  spécialisation et 4 fois plus de parties à 4 joueurs.
- **Même plafond que la première boucle de lab2** à partir de c016 (petit réseau, petite
  fenêtre) ; le généraliste plus grand et plus entraîné reste loin devant (c060 : −9,0 en arène
  directe, meilleure valeur même sur ces parties qu'il n'a jamais vues).
- La politique de la courbe avantage lab3 par construction (ses cibles sont les recherches de
  lab3, qui a joué ces parties) ; la valeur, elle, compare des prédictions du vainqueur.
- Positions des cycles 1 à 16 effacées du bucket ensuite (`lab3/prune.sh 4`, 25,8 Go gardés, aujourd'hui sous `p4-zero`).
