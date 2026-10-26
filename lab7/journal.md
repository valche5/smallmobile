# Labo 7 : les pistes de sortie du plateau de c130

Ouvert le 26 octobre 2026 à la demande de l'utilisateur, depuis **lab6** au commit `65d3daf`, pour
tester les pistes de la revue des six journaux. Lab6 est figé. Plan et raisons : [lab.md](lab.md).
**Figé le 26 octobre 2026** : la suite, la recherche sur des plans de tour écrits par un décodeur, avec un
transformer neuf, se fait dans `lab8/journal.md`.

**Bilan (26 octobre)** : les fins simulées à la décision battent c130 de **+5,4 ± 1,5 points** à
n = 100 (confirmé sur 920 donnes neuves) et +8,9 ± 2,3 à n = 400 : le premier progrès net depuis
lab3, au prix du calcul. Le marché branché aux choix de peuple et l'histoire vide de l'Expert du jeu
ne changent rien ; le bruit du selfplay coûte ~6 points aux coups des parties d'entraînement, mais
aucune des quatre recettes d'apprentissage (sans bruit, sans hasard, professeur à fins simulées,
témoin) n'a fait progresser c130 en dix pas. Moteur CUDA deux fois plus rapide. Rapport :
[rapport.html](rapport.html).

## Base

- Code de lab6 renommé (`smallmobile-lab7`, `smallmobile-brain7`, programme `lab7`, variables
  `LAB7_`), sans `py/bench_cuda.py`. `nets/c130.bin` identique à lab3 et lab6.
- Options d'agent dans `brain/src/search.rs` (`Config`) : `roll`, `mix`, `market=late`, `crn`,
  `win`, `explore`, `history` (tableau de `lab.md`). Sans option, la recherche est celle de lab6 au
  bit près. `lab7 arena` et `lab7 speed` affichent le coût des fins simulées.
- `moves::market` : l'empreinte du marché, sortie de `moves::chance` (même calcul).

## Vérifications

- `cargo test -p smallmobile-lab7 -p smallmobile-brain7 --no-default-features` : 11 tests passés.
  - `without_options_the_search_is_lab6s` : sans option, ou options à leur défaut écrites, mêmes
    coups, politique, visites, valeurs au bit près et même hasard consommé que la recherche de
    lab6 (c130, n = 64, 2 à 5 joueurs, avec et sans bruit).
  - `rollouts_stop_at_the_next_turn_of_the_root_player` : une fin simulée s'arrête au début du
    prochain tour du joueur de la racine (aussi depuis une défense hors de son tour), ou à la fin de
    la partie ; sa valeur reste entre 0 et 1.
  - `market_late_searches_the_turn_after_a_pick` : premier choix de peuple à 3 joueurs, n = 200 ;
    sans l'option, plus de 10 nœuds par choix (un par carte tirée) et 32 visites au-delà ; avec,
    un seul nœud par choix et 194 visites au-delà.
  - `market_late_trees_stay_legal_in_every_world` : n = 150, 160 positions de 2 à 5 joueurs ; les
    nœuds partagés par plusieurs mondes ont des coups possibles dans chacun.
  - `grouped_searches_with_rollouts_match_lone_ones` : fins simulées, marché tardif et mondes
    communs ; recherches menées ensemble et évaluées par groupes (`Batch`, le chemin du GPU au
    calcul près) : mêmes résultats, au bit près, que chacune seule.
  - `whole_games_with_every_option` et `without_history_the_agent_sees_what_the_game_expert_sees`.
- `cargo clippy -p smallmobile-lab7 -p smallmobile-brain7 --all-targets --no-default-features
  -- -D warnings` : passé. `cargo fmt` appliqué aux deux paquets.
- Compilation release avec Vulkan et CUDA (glslc 2026.4, dossier de `vast.sh`) : passée.

## Coût sur mini

`lab7 speed --positions 3 --seed 5` (12 recherches, 2 à 5 joueurs, un cœur) :

| Joueur | ms par recherche | Fins simulées |
| --- | --- | --- |
| `n=400` | 121,9 | |
| `n=400,market=late` | 120,7 | |
| `n=100,roll=1` | 1 496 | 1 198, 29,2 évaluations chacune |
| `n=100,roll=1,crn=1` | 1 510 | 1 198, 29,8 évaluations chacune |

## Machine

vast.ai, offre 53189390 (machine 119726, Pays-Bas : RTX 5080 bridée à 250 W, pilote 595.84, Ryzen 9
9950X, 31 fils), 0,421 $/h, `--device cuda` pour toutes les arènes ; binaire `7dd1d98dfa98`.

- Premier essai de `vast.sh up` (contrat 57736205, 3 min, 0,03 $) : `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
  `up` installait les derniers paquets pip `nvidia-cuda-nvrtc` (13.4) et `nvidia-cublas` (13.8),
  dont le PTX est trop récent pour le pilote 595 (lab6 utilisait ceux de PyTorch cu130). `vast.sh`
  fixe désormais ceux de CUDA 13.0 (`CUDA_PIP`). Vulkan bf16 reste faux sur NVIDIA (victoire 0,94).
- Deuxième essai (contrat 57736666) : CUDA juste (victoire 0,0004, pièces 0,012, logits 0,11).

## Arènes de la phase 1

Contre un champ de c130 à n = 400, 2 à 5 joueurs, `labtools/score.py` (intervalles à 95 %).

### A1 : fins simulées à n = 100 (graine 71002, 400 donnes)

`az:net=lab7/nets/c130.bin,n=100,roll=1` : **+5,2 ± 2,4 points** ; par taille +2,5 ± 5,3 (2 j.),
+6,8 ± 5,6 (3 j.), +3,5 ± 4,0 (4 j.), +8,0 ± 4,1 (5 j.). 31 min ; 9,1 s par décision du joueur
testé sur l'arène GPU, attentes comprises (p95 34 s ; le champ 0,88 s) ; 10,6 millions de fins
simulées, 25,1 évaluations chacune.

À calcul comparable (~12 fois n = 400), plus de recherche sans fin simulée ne donnait presque rien :
16× +1,8 ± 2,2 (lab6), 8× +2,0 ± 2,3 (lab4). Le gain vient surtout de 3 à 5 joueurs, là où la
recherche n'atteignait jamais le tour suivant du joueur. Seuil de `lab.md` atteint (≥ +3) : suites
prévues (n = 400, `mix=0.5`, `crn=1` sur les mêmes donnes, puis confirmation du meilleur sur 920
donnes neuves, graine 72002).

### Variantes de A1, mêmes donnes (graine 71002, 400 donnes)

| Joueur testé | Score | 2 j. | 3 j. | 4 j. | 5 j. | Face à A1 (apparié) |
| --- | --- | --- | --- | --- | --- | --- |
| `n=100,roll=1` (A1) | +5,2 ± 2,4 | +2,5 | +6,8 | +3,5 | +8,0 | |
| `n=100,roll=1,mix=0.5` | +5,7 ± 2,4 | +2,0 | +4,8 | +11,1 | +4,8 | +0,5 ± 3,2 |
| `n=100,roll=1,crn=1` | +4,5 ± 2,3 | +1,5 | +1,3 | +6,0 | +9,1 | −0,7 ± 3,3 |

Trois mesures de +4,5 à +5,7 : le gain des fins simulées tient. Ni le mélange avec la valeur du réseau
ni les mondes communs n'y changent rien de mesurable. 6,7 s par décision avec le moteur corrigé
(9,1 avant), ~23 min par arène : les fins simulées demandent aussi du processeur (coups possibles,
encodage à chaque coup joué). Confirmation de `mix=0.5` (le meilleur, de peu) sur 920 donnes neuves.

### Confirmation des fins simulées (graine 72002, 920 donnes neuves)

`az:net=lab7/nets/c130.bin,n=100,roll=1,mix=0.5` contre c130 à n = 400 : **+5,4 ± 1,5 points**,
positif à toutes les tailles et croissant avec elles : +3,6 ± 3,2 (2 j.), +4,4 ± 3,4 (3 j.),
+5,9 ± 2,7 (4 j.), +7,7 ± 2,4 (5 j.). 6,9 s par décision sur l'arène GPU (attentes comprises), ~50
min. **Premier progrès net sur c130 depuis lab3**, pour ~12 fois le calcul de n = 400, quand 16 fois
plus de recherche ordinaire ne donnait que +1,8 ± 2,2 (lab6). La réponse des adversaires, simulée
par le réseau jusqu'au prochain tour du joueur, était ce qui manquait à la recherche ; lab1 l'avait
mesuré (+11,7 parmi les Expert), lab2 l'avait retirée sans mesure.

### Le prix des fins simulées (graine 71002, 400 donnes, `mix=0.5`)

| Joueur testé | Score | 2 j. | 3 j. | 4 j. | 5 j. | s par décision (arène) |
| --- | --- | --- | --- | --- | --- | --- |
| `n=50,roll=1` | +2,9 ± 2,2 | −0,8 | +4,0 | +5,6 | +2,9 | 3,4 |
| `n=100,roll=1` | +5,7 ± 2,4 | +2,0 | +4,8 | +11,1 | +4,8 | 6,7 |
| `n=100,roll=2` (deux tours suivants) | +7,0 ± 2,3 | +7,0 | +4,8 | +7,5 | +8,8 | 14,3 |
| `n=400,roll=1` | **+8,9 ± 2,3** | +7,0 | +12,5 | +8,2 | +8,0 | 23,8 |

Face à `n=100,roll=1` (apparié) : `n=50` −2,7 ± 3,0, `roll=2` +1,3 ± 3,2, `n=400` +3,2 ± 3,3. Avec les
fins simulées, le calcul paie de nouveau : de +2,9 à +8,9 en multipliant les simulations par 8, quand
la recherche ordinaire ne gagnait que +1,8 pour 16 fois plus (lab6).

### A2 : le marché branché aux seuls choix de peuple (graine 71003, 920 donnes)

`az:net=lab7/nets/c130.bin,n=400,market=late` : **+0,3 ± 1,5 points** (2 j. −0,5 ± 3,2 ; 3 j.
+1,7 ± 3,4 ; 4 j. −0,7 ± 2,8 ; 5 j. +0,7 ± 2,2), 0,85 s par décision, ~20 min. Chercher la suite
d'un choix de peuple (194 visites au-delà au lieu de 32 dans le test) ne change rien de mesurable :
la valeur du réseau juste après le choix suffit à le départager, ou ce qui manque est plus loin
(le tour des adversaires, que A1 simule). Option gardée, sans emploi.

### A0 : l'Expert sans histoire, comme dans le jeu (graine 71001, 920 donnes)

`az:net=lab7/nets/c130.bin,n=400,history=0` : **−0,6 ± 1,5 points** (2 j. +0,4 ± 3,5 ; 3 j.
+0,3 ± 3,5 ; 4 j. −1,0 ± 2,5 ; 5 j. −2,2 ± 2,1), 0,83 s par décision, 17 min. Indécis : l'Expert du jeu,
qui part de `Knowledge::of` (histoire vide), ne perd rien de mesurable face à celui des arènes, au
plus un à deux points (à 5 joueurs surtout, à la limite de l'intervalle). Le seuil de `lab.md`
(intervalle sous 0) n'est pas atteint : transmettre l'histoire à l'Expert du jeu reste à faire pour
la propreté, sans urgence.

### A3 : ce que coûte le bruit du selfplay (graine 71004, 400 donnes chacune)

Les recherches rapides du selfplay (7 coups sur 8, n = 50, bruit de Gumbel à la racine), face à c130
à n = 400 :

| Joueur testé | Score | 2 j. | 3 j. | 4 j. | 5 j. |
| --- | --- | --- | --- | --- | --- |
| `n=50,explore=1` (le selfplay) | −10,7 ± 2,0 | −12,5 | −12,8 | −11,1 | −6,4 |
| `n=50` (sans bruit) | −4,9 ± 2,0 | −4,5 | −6,3 | −4,8 | −4,2 |

Différence appariée (mêmes donnes) : **−5,8 ± 2,6 points** pour le bruit seul. Les parties
d'entraînement sont jouées une dizaine de points sous le jeu que la valeur doit juger (arènes, jeu),
dont six par le bruit : de quoi biaiser les cibles de la valeur, ce que testent T1 et T2.

## Le moteur CUDA deux fois plus rapide (26 octobre)

Mesures sur RTX 5080 non bridée (vast.ai, machine t0, 360 W) : selfplay c130 à 770 positions/s avec
16 parties par fil, 890 avec 32 (lots de 233 et 470 positions) ; 64 dépassent la place d'un fil dans
l'anneau du GPU. Doubler les lots ne rapporte que 16 % : le coût est par position, pas dans les
lancements. `LAB7_CUDA_PHASES=1` chronomètre les étapes d'un calcul (événements CUDA, un calcul sur
8 ; affichées à la fin du programme) ; `lab7 bench-gpu` mesure le moteur seul par taille de lot. Sur
RTX 4090 (machine t3) : 4,72 µs par position, dont **53 % pour la moyenne des voisines**
(`neighbours`), qui lisait à chaque région le décalage de sa position dans le tampon épinglé de
l'hôte, donc à travers le PCIe (~90 000 lectures par calcul).

Correctif : places et décalages du calcul copiés dans la mémoire du GPU au début de chaque calcul
(16 Ko, `Work::stage`). Mêmes sorties (`check-gpu` : victoire 0,0004, pièces 0,008, logits 0,11).
`bench-gpu` sur RTX 4090, positions par seconde, avant → après : lots de 64 97 700 → 147 200, 256
214 400 → 419 200, 512 262 500 → 574 800, 1 024 287 700 → 655 200, 2 048 265 300 → 533 900 ;
2,35 µs par position. Les boucles de la phase 2 en profitent dès leur deuxième cycle (binaire
`74311ae103b7` envoyé à toutes les machines). Le reste du calcul est réparti sur une dizaine
d'étapes de 4 à 11 % chacune (entrées lues dans l'anneau épinglé, produits, `mid`, résiduel).

## Phase 2 lancée (26 octobre, 01 h 35 – 01 h 45)

Une machine vast.ai par bras, disque de 60 Go, PyTorch 2.14.1+cu130 installé par pip, boucle
`lab7/loop.sh` (32 parties par fil). Essai à blanc d'abord sur t0 (fenêtres réduites,
`LAB7_WINDOW_MIN`) : remplissage, apprentissage, duel, mesure finale, et le chemin `LUCK=1 NOISY=1`.

| Bras | Machine | Commande |
| --- | --- | --- |
| T0 | t0 : RTX 5080, Ryzen 9 9900X3D, France, 0,31 $/h | `RUN=t0 lab7/loop.sh 13` |
| T1 | t1 : RTX 4080 SUPER, i9-14900KF, Allemagne, 0,32 $/h | `RUN=t1 NOISY=1 lab7/loop.sh 13` |
| T2 | t2 : RTX 3090, Threadripper 7960X, Pays-Bas, 0,45 $/h | `RUN=t2 NOISY=1 LUCK=1 lab7/loop.sh 13` |
| T3 | t3 : RTX 4090, Ryzen 9 9950X, Nebraska, 0,51 $/h | `RUN=t3 SIMS=100 AGENT=roll=1 VALUE_MIX=0.5 GAMES=12500 FILL_SIMS=400 FILL_AGENT= FILL_GAMES=25000 lab7/loop.sh 14` |

T3 : fenêtre remplie par la recette ordinaire, puis des cycles de 12 500 parties dont les recherches
enregistrées sont celles de A1 (n = 100, fins simulées) ; le professeur donne la cible de politique
et, pour moitié, celle de la valeur (`--value-mix 0.5`). Une 5080 du Canada refusée par `up` (GPU
occupé au repos), détruite aussitôt.

Incident (26 octobre, 01 h 20 – 02 h 15) : `loop.sh` exportait `LAB7_CONCURRENT=32` pour toute la
boucle ; l'arène des duels le lit aussi, et partage l'anneau du GPU entre ses deux réseaux (256
places par fil et par réseau pour 32 × 16 positions en vol) : panique au premier duel (« plus de place
pour ce fil dans l'anneau du GPU »), les quatre boucles arrêtées après c004, machines inoccupées 40 à
55 min (~1,3 $). Corrigé : 32 parties par fil pour le selfplay seulement (`CONCURRENT`), et l'arène
vérifie d'entrée que ses parties tiennent dans l'anneau (binaire `d359df711107`). Boucles reprises à
leur duel de c004 (étapes déjà faites sautées).

Premiers duels (c004, un pas sur la fenêtre remplie par c130, n = 200, test séquentiel) : T0
−0,6 ± 2,5 ; T1 −1,2 ± 3,2 ; T2 −3,9 ± 3,1 ; T3 −2,9 ± 3,4 (son c004 vient encore de la recette
ordinaire).

## Résultats de la phase 2 (26 octobre)

Dix pas par bras depuis c130 (c004 à c013 ; T3 jusqu'à c014), duels de chaque pas contre c130
(n = 200, test séquentiel), mesure finale de c013 contre c130 à n = 400, 920 donnes (graine 72101,
mêmes donnes pour tous) :

| Bras | Duels des pas (de c004 à c013) | Mesure finale | Face à T0 (apparié) |
| --- | --- | --- | --- |
| T0, témoin | −0,6, −0,9, −1,3, −1,2, −1,2, −1,6, −2,2, −4,3, −0,8, −1,5 | −0,5 ± 1,4 | |
| T1, sans bruit après la première manche | −1,2, −0,2, −3,2, −5,0, −1,7, −0,2, −4,2, −1,4, −1,2, −1,2 | −2,1 ± 1,4 | −1,6 ± 2,0 |
| T2, sans bruit ni hasard | −3,9, −1,5, +0,0, −1,1, −2,2, −3,6, +1,9, −4,2, +0,1, −2,5 | −0,7 ± 1,4 | −0,2 ± 2,0 |
| T3, professeur à fins simulées (c004 à c014) | −2,9, …, +0,5, −1,1, +0,1, +0,3, −1,1, −0,7, −2,4, +0,0, +0,1 | −1,7 ± 1,4 (c014) | −1,2 ± 2,0 |

**Aucune recette n'a fait progresser c130 en dix pas.** Ni des parties d'entraînement jouées sans
bruit (qui coûtait pourtant 6 points à leurs coups, A3), ni la part du hasard retranchée de la cible
de la valeur, ni un professeur nettement plus fort à la décision (n = 100 avec fins simulées : +5,4
en arène) dont viennent la cible de politique et la moitié de celle de la valeur. Le réseau ne
retient pas ce que les fins simulées calculent, du moins à cette échelle (125 000 parties du
professeur, dans une fenêtre de 4 à 5 millions de positions). Le gain des fins simulées reste à la
décision : il se paie en calcul à chaque coup.

Coût et incident : les quatre machines de T0, T1, T2 et de la phase 1 sont restées allumées, inoccupées,
de la fin de leur calcul (04 h 42 à 06 h 31 UTC) à 08 h, faute d'attente programmée sur la fin des
boucles : ~3,8 $ perdus, en plus de ~1,3 $ pour l'incident des duels. Désormais, chaque machine qui
calcule a une commande en attente qui rapatrie ses résultats et la détruit à la fin.

T3 fini à 08 h 32 UTC (mesure finale de c014), rapatrié et détruit automatiquement à la fin de son
calcul (machine de 478 min, ~4,17 $). Plus aucune machine louée ; crédit restant 6,70 $.
