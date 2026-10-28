# Labo 9 : le réseau hybride de ResTNet, dans le cadre de lab8

Ouvert le 27 octobre 2026 à la demande de l'utilisateur, après lab8 (figé), pour remplacer
l'encodeur tout-transformer par des blocs de graphe et des blocs transformer alternés (ResTNet). Plan
et raisons : [lab.md](lab.md).

## Phase 0 : la base (27 octobre)

Code : copie de lab8 (figé) renommée (`smallmobile-lab9`, `smallmobile-brain9`, programme `lab9`,
`LAB9_`), avec le bloc R :
- `py/model.py` : `Conv` (le bloc R), `adjacency` et `neighbours` (moyenne des voisines, en f32) ; la
  disposition (`Net(d, "RRTRRT", têtes, décodeur)`, `--size 256,RRTRRT,8,2`) ; le nombre de têtes
  écrit dans les poids (`heads`), lu dans les biais des relations pour un réseau de lab8 ;
- `brain/src/net.rs` : les blocs `Block::T` et `Block::R`, la disposition lue dans les poids ;
- `src/vulkan.rs` : refuse encore les blocs R (phase 3).

Vérifications :
- Rust et PyTorch donnent les mêmes sorties sur un hybride au hasard (`tests/fixtures-hybrid`, 32 de
  large, RTR) : écart maximal des têtes 1,9e-6, des logits 3,0e-6 ;
- un réseau de lab8 (`tests/fixtures`, celui de lab8) se lit dans lab9 à l'identique : sorties
  PyTorch égales au bit près, Rust à 3,5e-6 ; i2 et s1 se chargent (`lab9 speed`) ;
- `cargo test -p smallmobile-lab9 -p smallmobile-brain9 --release` : 9 tests passés ; clippy passé ;
  compilation sans Vulkan passée.

Débit de l'apprentissage (`train.py --bench 300`, `lab8/data/c130`) :

| Réseau | Paramètres | Lots | Plans/s |
| --- | --- | --- | --- |
| h1, RRTRRT 256 × 6, 8 têtes | 6 040 026 | 256 | **9 450** |
| forme d'i2, TTTTTT 256 × 6 (même série) | 7 882 522 | 256 | 7 870 |
| h2, RRTRRT 128 × 6, 4 têtes | 1 545 482 | 512 | **16 500** |
| s1, TTTT 128 × 4 (lab8) | 1 611 354 | 512 | ~17 000 |

## Phase 2 : protocole (fixé le 27 octobre, avant le calcul)

Données : `lab8/data/c130` et `lab8/data/c130b`, lues en place ; validation : les 50 000 premiers
plans des parties mises de côté, ceux d'i2 et s1. Graine d'apprentissage 0, comme eux (même ordre des
plans).
- **h1** : `train.py --data ../../lab8/data/c130,../../lab8/data/c130b --size 256,RRTRRT,8,2 --steps
  150000 --val-every 5000 --value-every 4` (`lab9/runs/h1`), la recette d'i2 entière (i2 avait été
  arrêté au lot 103 000 sur 150 000 : comparaison aux mêmes lots, jusqu'à 100 000).
- **h2** : de même, `--size 128,RRTRRT,4,2 --batch 512 --steps 75000` (`lab9/runs/h2`), la recette de
  s1.
- Puis `best.py`, et la parité des politiques seules des deux meilleurs (920 donnes, graine 82001),
  appariée avec i2 (`lab8/runs/i2/parity-100000.tsv`) et s1 (`lab8/runs/s1/parity-075000.tsv`).

## Phase 3 : le bloc R en Vulkan (27 octobre, pendant l'apprentissage de h1)

`shaders/neighbours.comp` : pour chaque jeton, l'entrée d'un produit du bloc R, `[a, moyenne des
voisines, a de la partie]` ou `[a, moyenne des voisines]` en bf16 (voisines lues dans les distances,
en bits dans la mémoire partagée) ; le bloc : `layernorm`, `neighbours`, `gemm` (gelu), `layernorm`,
`neighbours`, `gemm` (résiduel). `check-gpu`, réseaux hybrides au hasard (sorties au hasard, σ 0,05),
200 positions, GPU partagé avec l'apprentissage :

| Réseau | Têtes | Régions | `dec_reg` | Attention croisée | Logits | Victoires (décodeur sur GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| RRTRRT 256 × 6 | 0,011 | 0,037 | 0,031 | 0,033 | 0,024 | 0,0027 |
| RRTRRT 128 × 6 | 0,010 | 0,039 | 0,028 | 0,030 | 0,018 | 0,0017 |
| lab8, TTTTTT 256 × 6 (même mesure) | 0,008 | 0,031 | — | — | 0,023 | — |

Mêmes écarts que le moteur de lab8 (l'arrondi bf16). Vitesse : à mesurer GPU libre.

## Phase 2 : h1, RRTRRT 256 × 6 (27 octobre)

150 000 lots en ~1 h 15 (9 200 plans/s ; i2 : 4 h 15 à l'époque, 7 870 plans/s aujourd'hui).
Validation (mêmes plans qu'i2) :

| Lots | h1 politique | h1 valeur | i2 politique | i2 valeur |
| --- | --- | --- | --- | --- |
| 10 000 | 1,284 | 1,090 | 1,341 | 1,088 |
| 25 000 | 1,218 | 1,047 | 1,238 | 1,051 |
| 50 000 | 1,186 | 1,031 | 1,195 | 1,034 |
| 75 000 | 1,171 | 1,024 | 1,177 | 1,025 |
| 100 000 | **1,162** | **1,019** | 1,166 | 1,019 |
| 150 000 | **1,154** | **1,014** | — | — |

- **h1 est meilleur qu'i2 à chaque validation**, des deux têtes, surtout au début (le local appris
  d'emblée par les blocs R) ; au lot 100 000, l'écart n'est plus que de 0,004 en politique. h1 au lot
  ~85 000 vaut i2 au lot 100 000, avec 0,77 fois les paramètres et ~0,55 fois le calcul.
- Meilleur réseau (`best.py`) : le dernier, lot 150 000 ; critère de la phase 2 atteint en tout et à
  chaque taille (politique 1,154 contre 1,179 pour c130 ; valeur 1,014 contre 1,032 ; précision
  58,3 % contre 60,6 %).
- **Parité des politiques seules** (`plan:net=lab9/runs/h1/h1-150000.bin` contre c130 à n = 1, 920
  donnes, graine 82001, `runs/h1/parity-150000.tsv`) : **+1,9 ± 1,5 points** (2 j. +3,4, 3 j. +1,0,
  4 j. +0,2, 5 j. +3,1) ; apparié avec i2 : +1,5 ± 2,0 (indécis), avec s1 : +6,0 ± 2,0. 9,6 ms par
  décision (i2 : 20,8), processeur partagé avec l'apprentissage de h2.

## Phase 2 : h2, RRTRRT 128 × 6 (28 octobre)

75 000 lots de 512 en ~50 min (12 000 à 16 000 plans/s, processeur partagé avec la parité de h1).

| Lots | h2 politique | h2 valeur | s1 politique | s1 valeur |
| --- | --- | --- | --- | --- |
| 10 000 | 1,294 | 1,086 | 1,321 | 1,087 |
| 25 000 | 1,224 | 1,046 | 1,229 | 1,051 |
| 50 000 | 1,194 | 1,031 | 1,198 | 1,035 |
| 75 000 | **1,186** | **1,027** | 1,189 | 1,031 |

- h2 est meilleur que s1 à chaque lot, de peu ; **il n'atteint pas i2** (1,166 / 1,019) : réponse
  négative à la question principale de la phase 2. Critère de la phase 2 non atteint (politique sous
  celle de c130, valeur au-dessus).
- Parité des politiques seules (`runs/h2/parity-075000.tsv`) : **−2,2 ± 1,5 points** (2 j. −1,6,
  3 j. −5,7, 4 j. −0,8, 5 j. −0,9) ; apparié avec s1 : +1,8 ± 2,0 (indécis), avec i2 : −2,7 ± 2,0.
  3,1 ms par décision (s1 : 3,9).

## Temps de calcul : les cinq réseaux (28 octobre)

Demandé par l'utilisateur. **La carte ne tourne pas à pleine puissance sur ces calculs** (100 à 127 W,
1,7 à 2,3 GHz, pour ~300 W et ~3 GHz au plus) et sa fréquence varie d'un lancement à l'autre : s1
en lots de 256 a donné 109 200 lectures/s le 27 octobre, 59 000 à 83 000 aujourd'hui (le binaire de
lab8 de même). Les débits du GPU ci-dessous sont des médianes de trois tours alternés, dans la même
séance ; ne pas les comparer aux mesures d'autres séances.

| Réseau | Paramètres | Processeur, un cœur (ms par lecture) | GPU, lots de 256 (lectures/s) | GPU, lots de 1 024 | En partie (ms par décision) | Validation (pol / val) | Politique seule contre c130 à n = 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c130 (graphe 128 × 7) | 1,1 M | ~0,15 | 1 053 000 | 1 172 000 | — | 1,179 / 1,032 | 0 |
| i2 (TTTTTT 256) | 7,9 M | 15,6 | 28 700 | 22 600 | 297 | 1,166 / 1,019 | +0,4 ± 1,6 |
| **h1 (RRTRRT 256)** | 6,0 M | **8,6** | **41 200** | **32 600** | 287 | **1,154 / 1,014** | **+1,9 ± 1,5** |
| s1 (TTTT 128) | 1,6 M | 2,8 | 69 800 | 84 200 | 258 | 1,189 / 1,031 | −4,1 ± 1,5 |
| h2 (RRTRRT 128) | 1,5 M | 2,4 | 67 700 | 87 000 | 279 | 1,186 / 1,027 | −2,2 ± 1,5 |

Processeur : `lab9 speed` (f32, une position à la fois ; h1 et h2 mesurés sur des réseaux au hasard
de même forme). En partie : la recherche sur plans (`gpu=1`, k = 16, m = 4) contre c130 à n = 1, 20
donnes, graine 9, les quatre l'un après l'autre (s1 avait fait 136 ms le 27 octobre).

- **h1 est le meilleur réseau, et le moins cher de sa largeur** : 1,4 fois i2 sur le GPU, 1,8 fois
  sur le processeur.
- **En partie, le gain fond** : h1 287 ms par décision contre 297 pour i2 ; h2 n'est pas plus rapide
  que s1. Profil (`LAB9_GPU_PROFILE`) : lectures ~70 % du GPU, pas du décodeur ~30 % (inchangés par
  l'hybride) ; dans les lectures, `neighbours` prend 10 à 11 % (un groupe par position, peu de travail
  par groupe dans les petits lots de la recherche) ; lectures en partie : h1 27 600/s, i2 24 900/s
  (1,1 fois seulement).
- Tous restent loin de c130 : h2 lit encore ~13 fois moins vite.

## Le GPU manque de travail (28 octobre)

Remarque de l'utilisateur : la carte n'est pas bridée, elle n'a pas assez de travail. Mesuré avec h1
(`LAB9_GPU_PROFILE` affiche maintenant la taille moyenne des lots et la part du temps passée à attendre
le GPU) :

| Travail | Lectures par lot | Voies par pas | GPU occupé | Puissance | Débit |
| --- | --- | --- | --- | --- | --- |
| Arène (20 fils, k = 16, m = 4) | 73 | 234 | 88 % | 100 à 155 W, 2,3 à 3,3 GHz | 290 ms par décision |
| Selfplay, 60 fils, k = 8, m = 2 | 52 | 175 | 88 % | 70 à 78 W | 50 parties/min |
| Selfplay, 200 fils | 156 | 525 | 85 % | 87 à 108 W | **82 parties/min** |
| Selfplay, 400 fils | — | — | — | 92 à 95 W | 21 parties en 10 min (arrêté) |
| `bench-gpu`, lots de 256 | 256 | — | — | 100 à 127 W | 43 000 lectures/s |

- Le GPU calcule ~85 % du temps, mais sur des lots de 50 à 150 positions : une recherche ne demande
  que quelques lectures à la fois, et les fils partagent le GPU un lot à la fois (`vulkan::Shared` :
  le fil qui le trouve libre emporte toutes les demandes, lectures puis pas, et attend chacun).
  Chaque kernel attend le précédent (une barrière après chacun) : sur de petits lots, la carte reste
  à moitié vide, d'où la puissance faible.
- Plus de parties en vol : ×1,6 à 200 fils ; à 400, tout se bloque (le partage sature : 85 % du
  processeur inoccupé). Deux bancs en même temps (deux processus) ne font que +15 % au total.
- Pour le selfplay : 200 fils. Pistes pour aller plus loin, non faites : plusieurs lots en vol sur le
  GPU dans un même processus (activations et commandes par flux), un `gemm` à tuiles plus petites pour
  les petits lots, `neighbours` fondu dans le produit qui le suit.

### La cible du selfplay

La cible de lab8 multipliait les valeurs brutes des candidats par `beta` = 10 : des écarts de quelques
centièmes ne pesaient rien (KL 0,005 nat par décision, la cible était l'a priori). Comme les racines de
lab7 et mctx : valeurs ramenées entre 0 et 1 sur les candidats, puis multipliées par
`(50 + m) × 0,1` (5,2 pour m = 2, 5,4 pour m = 4).

Avec la cible normalisée (h1, 200 fils) : k = 8, m = 2 : **84 parties/min**, KL 0,143 nat par
décision (0,005 avant) ; k = 16, m = 4 : ~13 parties/min, KL 0,140. La recherche plus large coûte 6
fois plus pour une cible aussi informative : le selfplay se fait en k = 8, m = 2. (Un premier essai
en k = 16, m = 4 a été arrêté à tort, cru bloqué : les fichiers `.games` ne s'écrivent sur le disque
qu'à la fin, et mon filtre cachait les lignes de progrès ; l'essai à 400 fils a pu être jugé de même.)

## Phase 6 : la boucle (protocole, fixé le 28 octobre avant le calcul)

Demandé par l'utilisateur : entraîner h1 par auto-apprentissage, comme dans les anciens labos.
`lab9/loop.sh` (sur le modèle de `lab7/loop.sh`), essai `p1`, depuis c000 = `h1-150000.bin` ; au pas k :
1. 5 000 parties jouées par le réseau k−1 (`lab9 selfplay`, k = 8, m = 2, `beta` de mctx 5,2, plan
   joué tiré aux 2 premières manches, 200 parties ensemble, graine 90 000 000 + k) ;
2. le réseau k appris depuis k−1 : 4 000 lots de 256, AdamW neuf, taux 1e-4 (montée, cosinus),
   moyenne glissante 0,998, victoire finale sur un plan sur quatre ; sur les 4 derniers cycles, un
   quart de chaque lot tiré des parties d'imitation (`lab8/data/c130`) ;
3. duel du réseau k contre c000, la recherche sur plans des deux côtés (k = 8, m = 2), 460 donnes au
   plus (graine 1004, les mêmes à chaque pas), arrêt séquentiel à 3 points.

Essai à blanc (200 parties, 200 lots, 24 donnes, deux pas) : la chaîne tourne ; 154 s pour 200
parties, un duel ~9 s par donne. Un pas réel : ~1 h de parties, ~5 min d'apprentissage, 15 à 70 min de
duel. Après la boucle : le dernier réseau en phase 5 (contre c130 à n = 400, graine 82002), appariée
avec h1 en phase 5 (à mesurer aussi) et i2.

## Où passe le temps du selfplay (28 octobre, mesuré)

La boucle p1, lancée à 01 h 48, s'est arrêtée à la fin de son premier cycle (« Too many open files » :
1 024 fichiers ouverts au plus sous systemd, 200 fils × 11 fichiers ; 108 fils sont morts au départ,
les 92 autres ont joué les 5 000 parties à 72,5 par minute). Parties gardées (`data/p1/c001`), `ulimit`
relevé, fichiers du selfplay ouverts avant de jouer ; relancée à 09 h 59 puis arrêtée à la demande de
l'utilisateur (machine sous-employée), c001 appris, son duel en cours.

Instrumentation (`selfplay --seconds S`, un bilan toutes les 30 s sur l'intervalle seul ; temps des
fils, des tours du partage du GPU, préparation et relecture des lots, cœurs occupés). Les mesures
d'hier (lots de 50 à 160, 70 à 100 W) étaient faussées : essais trop courts, dont la fin traîne (les
derniers fils jouent seuls), et profil des kernels activé. En régime (h1, 200 fils, k = 8, m = 2,
après la première minute) :

| Mesure | Valeur |
| --- | --- |
| Débit | **215 recherches/s** (~150 parties/min), 31 000 lectures/s, 110 000 pas/s |
| GPU | calcule 90 % du temps, **270 à 278 W** (~300 au plus) ; lots de 250 lectures et 870 voies |
| Fils | 98 % du temps à attendre le GPU (lectures 35 %, pas 63 %) ; calcul de la recherche 0,1 % |
| Processeur | **0,9 cœur** sur 20 |
| Hors calcul du GPU | préparation des lots 4 %, relecture 3,3 % (le GPU attend) |

Le GPU est saturé ; le processeur est inemployé. Répartition du temps du GPU (profil, 90 s) : lectures
75 % (`gemm` 48, `neighbours` 8, `layernorm` 7, `store` 6, `attention` 5), pas du décodeur 25 % (`gemm`
10, `dec_out` 8, `dec_cross` 4, `dec_self` 2). Une lecture coûte ~386 MFLOP de produits, à ~28 TFLOPS
(hipBLASLt : 28 à 47 sur ces formes, lab8). 141 lectures et 500 pas par recherche.

Essais (banc de 3 min, moyenne des intervalles après 60 s ; `lab9 bench-gpu` pour `neighbours`) :

| Changement | Effet |
| --- | --- |
| `neighbours` sur 4 groupes par position | aucun (885 → 925 ms) : limité par la mémoire (~1 To/s ; il recopie `a` et la partie) |
| Lectures et pas d'un tour en une soumission (`Encoder::turn`) | 215 → 215 recherches/s (gardé : un peu moins de temps de GPU par tour) |
| Deux moteurs (deux files de lots en parallèle) | 205 recherches/s : moins bien, la carte est déjà pleine |

Pistes, non faites :
- moteur (mêmes résultats) : ne pas calculer les jetons absents (~25 % des lignes des produits : 37,5
  régions en moyenne sur 48, joueurs et cartes absents), `dec_out` et la relecture limités aux coups
  possibles, `store` depuis du bf16, `neighbours` fondu dans le produit, `gemm` réglé : ×1,3 à ×1,6 au
  mieux, pour une journée de travail ;
- recherche : moins de lectures par recherche (141) : fins simulées jouées par un réseau plus petit
  (h2 : lectures deux fois moins chères), ou sans fins simulées (`mix=0`, la seconde étape prévue de la
  phase 6) ;
- processeur : h1 y lit une position en 8,6 ms, 19 cœurs n'ajouteraient que ~7 % de lectures.
