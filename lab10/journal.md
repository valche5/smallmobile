# Labo 10 : l'ancrage de la recherche de c130 sur sa politique (piKL)

Ouvert le 29 octobre 2026 à la demande de l'utilisateur, depuis **lab7** au commit `1ae5637`, après le
bilan des labos (`labtools/bilan.md`) : l'ancrage piKL était la piste inexplorée la moins chère. Plan et
protocole : [lab.md](lab.md).

## Base

- Code de lab7 renommé (`smallmobile-lab10`, `smallmobile-brain10`, programme `lab10`, variables
  `LAB10_`), sans `py/`, `loop.sh`, `rapport.html` ni les données. `nets/c130.bin` identique à lab3 et
  lab7 (SHA-256 `e534cc84…`).
- `search::Config::anchor` (`Anchor::Mctx`, `Fixed { scale }`, `Raw { lambda }`), lu par
  `Node::completed_into` et passé à `select`, `improved` et aux choix de la racine ; options
  `anchor=`, `scale=`, `lambda=` (l'une ou l'autre selon l'ancrage, sinon refus).
- `lab10 speed` affiche la KL de la politique de la racine à l'a priori du réseau et la part des
  décisions où le coup joué n'est pas le coup préféré du réseau.
- Pas d'ajout au sélecteur des Experts du jeu.

## Vérifications (29 octobre, mini, processeur seul)

- `cargo test -p smallmobile-lab10 -p smallmobile-brain10 --no-default-features` : 14 tests passés.
  - `without_anchor_the_search_is_lab7s` : sans ancrage, ou avec `anchor=mctx` écrit, avec ou sans les
    options de lab7 (`roll`, `mix`, `market`, `crn`) : mêmes coups, politique, visites, valeurs au bit
    près et même hasard consommé que la recherche de lab7 (c130, 2 à 5 joueurs, avec et sans bruit).
  - `anchors_scale_the_values_as_announced` : sur un même nœud, le facteur de mctx croît avec les
    visites, celui de `fixed` non, `raw` garde les écarts bruts divisés par λ.
  - `grouped_searches_with_rollouts_match_lone_ones` et `whole_games_with_every_option` rejoués avec
    les deux ancrages.
- `cargo clippy … -- -D warnings` : passé.
- Étalonnage de la KL sur mini : tableau de `lab.md`. Avec mctx, l'écart au réseau croît de 0,77 nat
  (n = 400) à 1,10 (n = 6 400) ; avec `fixed,scale=10`, 0,61 puis 0,39.

Aucun résultat de force à ce stade.

## Machine (29 octobre)

vast.ai, offre 49691112 (machine 58280633, Pays-Bas : RTX 5080, pilote 595.84, Ryzen 9 9950X, 31 fils,
30 Go), 0,354 $/h, image `stock-ubuntu26.04-2026-10-23`, binaire `1e5d19f04d4c` (commit `92b9221`).
L'offre française à 0,29 $/h (machine de T0 dans lab7) a été refusée à la location. `check-gpu` :
Vulkan bf16 faux comme toujours sur NVIDIA (victoire 0,90), **CUDA juste** (0,0004, 0,012, 0,111) ;
`--device cuda` pour toutes les arènes. Crédit au départ : 6,68 $.

## Phase 1 : présélection à n = 400 (graine 101001, 400 donnes, 29 octobre)

Contre un champ de c130 à n = 400, mêmes donnes pour les quatre ; 22 min 35 pour les quatre arènes
(~5 min 40 chacune, 1 400 parties ; ~0,58 s par décision du joueur testé sur l'arène GPU, attentes
comprises). `lab10/runs/{f10,f20,r2,r5}.tsv`.

| Candidat | Score | 2 j. | 3 j. | 4 j. | 5 j. |
| --- | --- | --- | --- | --- | --- |
| F10 `anchor=fixed,scale=10` | +0,1 ± 2,2 | −2,5 | +1,2 | −0,1 | +1,9 |
| F20 `anchor=fixed,scale=20` | −2,3 ± 2,2 | −2,0 | −3,2 | −0,1 | −4,1 |
| R2 `anchor=raw,lambda=0.002` | −1,5 ± 2,2 | −2,0 | −0,8 | −0,9 | −2,2 |
| R5 `anchor=raw,lambda=0.005` | +0,8 ± 2,2 | −2,5 | +5,2 | −0,1 | +0,8 |

À n = 400, aucun ancrage ne se distingue de mctx : rien n'atteint le seuil de +3, tous dans le bruit.
Attendu (`lab.md`) : F10 diffère peu de mctx à ce budget. Phase 2 avec F10 et R5, les meilleurs de
chaque famille.

## Phase 2 : à 16× (n = 6 400, graine 62001, 400 donnes, 29 octobre)

Les donnes de l'arène 16× de lab6 (`lab6/runs/arena16.tsv` : c130 à n = 6 400 avec mctx, +1,8 ± 2,2),
contre un champ de c130 à n = 400. F10 21 min (6,4 s par décision), R5 18 min (5,3 s).
`lab10/runs/{f10,r5}-16x.tsv`.

| Joueur testé | Score | 2 j. | 3 j. | 4 j. | 5 j. | Face au 16× de mctx (apparié) |
| --- | --- | --- | --- | --- | --- | --- |
| lab6 : `n=6400` (mctx) | +1,8 ± 2,2 | −0,8 | +4,9 | +0,5 | +2,4 | |
| F10 : `n=6400,anchor=fixed,scale=10` | **+6,2 ± 2,3** | +6,2 | +11,3 | +1,3 | +5,9 | **+4,4 ± 3,2** |
| R5 : `n=6400,anchor=raw,lambda=0.005` | **+7,9 ± 2,2** | +6,5 | +13,2 | +8,5 | +3,5 | **+6,1 ± 3,1** |

R5 face à F10 (apparié) : +1,7 ± 3,3, indécis.

- **La prédiction de la biblio se vérifie** : avec un ancrage qui ne faiblit pas avec les visites, seize
  fois plus de recherche rapporte +6 à +8 points là où la recette de mctx n'en donnait que +1,8 (lab6),
  et +2,0 pour 8× (lab4). À n = 400, les mêmes ancrages ne changeaient rien (phase 1) : c'est bien le
  comportement à grand budget qui est corrigé, pas la recherche elle-même.
- Les deux formes marchent ; `raw` (le λ fixe de piKL, sans ramener les valeurs entre 0 et 1) est
  un peu devant, sans que la différence soit établie.
- Ordre de grandeur : les fins simulées de lab7 à n = 400 (`roll=1`) donnaient +8,9 ± 2,3 pour 24 s par
  décision ; R5 à 16× donne +7,9 ± 2,2 pour 5,3 s.

## Phase 3 : confirmation de R5 à 16× (graine 102001, 920 donnes neuves, 29 octobre)

`az:net=lab10/nets/c130.bin,n=6400,anchor=raw,lambda=0.005` contre c130 à n = 400 : **+6,1 ± 1,5 points**
(2 j. +3,7 ± 3,1, 3 j. +9,6 ± 3,5, 4 j. +6,6 ± 3,1, 5 j. +4,7 ± 2,5), positif à toutes les tailles ; 5,5 s par
décision sur l'arène GPU (attentes comprises), 40 min. `lab10/runs/r5-16x-confirm.tsv`.

**Confirmé** : à poids fixes, l'ancrage à λ fixe fait rapporter au calcul ce que mctx laissait perdre
(16× : +1,8 ± 2,2 dans lab6). C'est le deuxième levier à la décision après les fins simulées, et il ne
coûte rien de plus que la recherche longue elle-même.

### Avec les fins simulées de lab7 (graine 72002, 920 donnes)

`n=100,roll=1,mix=0.5,anchor=raw,lambda=0.005` contre c130 à n = 400 : **+7,0 ± 1,6** (2 j. +4,2, 3 j. +6,9,
4 j. +7,8, 5 j. +9,0), 7,3 s par décision, 52 min. Apparié avec la confirmation de lab7 sans ancrage
(`lab7/runs/a1-confirm.tsv`, +5,4 ± 1,5) : **+1,6 ± 2,0, indécis**. À n = 100 l'ancrage ne change pas
grand-chose, comme à n = 400 en phase 1 : son effet est celui des grands budgets. Les deux leviers ne
s'excluent pas (le gain ne baisse pas), sans qu'on puisse dire qu'ils s'additionnent.
`lab10/runs/r5-roll-confirm.tsv`.

## Phase 4b : balayage de λ à 16× (PC, graine 104001, 400 donnes, 29 octobre)

PC (RX 9070 XT, Vulkan bf16 ; `check-gpu` : victoire 0,0021, pièces 0,069, logits 0,575, comme dans
lab8). Une arène 16× de 400 donnes en 7 à 8 min (1,5 à 1,7 s par décision), contre 18 à 21 min pour la
RTX 5080 en CUDA. `lab10/runs/sweep-{l25,l50,l100}.tsv`.

| λ | Score | 2 j. | 3 j. | 4 j. | 5 j. |
| --- | --- | --- | --- | --- | --- |
| 0,0025 | +4,8 ± 2,3 | +3,0 | +6,7 | +6,2 | +3,2 |
| 0,005 (R5) | +5,6 ± 2,4 | +7,2 | +4,2 | +7,8 | +3,0 |
| 0,01 | +5,4 ± 2,3 | +4,5 | +5,3 | +8,4 | +3,4 |

Face à 0,005 : 0,0025 −0,8 ± 3,0, 0,01 −0,2 ± 3,1. Le score est plat sur un facteur 4 de λ : la règle
garde **λ = 0,005**. L'effet ne demande pas de réglage fin. R5 sur le PC (+5,6) rejoint R5 en CUDA
(+7,9 et +6,1).

## Phase 4c : témoin mctx à 16× sur le PC (graine 105001, 920 donnes, 29 octobre)

`az:net=lab10/nets/c130.bin,n=6400` (mctx) contre c130 à n = 400 : **+2,9 ± 1,5** (2 j. +3,5, 3 j. +4,1,
4 j. +3,1, 5 j. +1,0), 2,2 s par décision (1,5 s avec l'ancrage), 22 min. `lab10/runs/mctx-16x-pc.tsv`.

Sur le même moteur, chaque λ de 4b fait mieux que ce témoin de +1,9 à +2,6 (± 2,8 chacun). Les trois
λ réunis (1 200 donnes, +5,3) le battent de **+2,4 ± 2,0**. mctx à 16× gagne un peu plus que lab6 ne le
laissait croire (+1,8 ± 2,2 sur 400 donnes) : le gain dû à l'ancrage est plus petit qu'estimé en
phase 2 (+6,1 ± 3,1 face à lab6).

## p4 : λ à 16× en CUDA (graine 62001, 400 donnes, 29 octobre ; hors protocole)

Lancé sur vast par la session précédente avant que la phase 4 soit écrite. Mêmes donnes que la phase 2.
`lab10/runs/{r2,r10}-16x.tsv`, 36 min pour les deux.

| λ | Score | 2 j. | 3 j. | 4 j. | 5 j. | Face à R5 (phase 2) |
| --- | --- | --- | --- | --- | --- | --- |
| 0,002 | +5,1 ± 2,2 | +6,2 | +8,7 | +5,8 | −0,3 | −2,8 ± 3,1 |
| 0,01 | +6,3 ± 2,4 | +8,0 | +7,2 | +7,2 | +2,8 | −1,6 ± 3,2 |

Même conclusion qu'en 4b, sur l'autre moteur : de λ = 0,002 à 0,01, le score à 16× est plat.

## Phase 4a : témoin mctx à 16× sur les donnes de la confirmation (graine 102001, 920 donnes, 29 octobre)

Lancé sur les deux moteurs. **PC (Vulkan)** : `az:net=lab10/nets/c130.bin,n=6400` contre c130 à n = 400 :
**+3,6 ± 1,5** (2 j. +4,3, 3 j. +5,6, 4 j. +2,3, 5 j. +2,0), 2,2 s par décision, 22 min.
`lab10/runs/mctx-16x-confirm.tsv`. Face à R5 sur les mêmes donnes (phase 3, CUDA) : R5 **+2,6 ± 2,1**.

**vast (CUDA, même machine que la phase 3)** : **+4,9 ± 1,5** (2 j. +6,5, 3 j. +6,5, 4 j. +3,2, 5 j. +3,5),
6,3 s par décision, 45 min. `lab10/runs/mctx-16x-confirm-cuda.tsv` (rapatrié sous ce nom). Face à R5
sur les mêmes donnes et le même moteur : R5 **+1,2 ± 2,0, indécis**. Les deux moteurs entre eux :
−1,4 ± 2,1.

Bilan à 16× (toutes les arènes ; écart-type de chaque arène, pondération par les donnes) :

| Recherche à 16× | Donnes | Score |
| --- | --- | --- |
| mctx (lab6, 4c, 4a PC, 4a CUDA) | 3 160 | +3,5 ± 0,8 |
| ancrage `raw`, λ = 0,005 (phases 2, 3, 4b) | 1 720 | +6,4 ± 1,1 |
| ancrage `raw`, λ de 0,002 à 0,01 (avec 4b et p4) | 3 320 | +5,9 ± 0,8 |

Par moteur : CUDA, R5 +6,7 contre mctx +4,0 (1 320 donnes chacun), **+2,7 ± 1,8** ; PC, les trois λ
+5,3 contre mctx +3,3 (1 200 et 1 840 donnes), **+2,0 ± 1,7**.

Chercher seize fois plus rapporte déjà environ +3,5 points avec mctx (lab6, sur 400 donnes, le
sous-estimait). **Gain dû à l'ancrage à 16× : +2,9 ± 1,4 (λ = 0,005), +2,4 ± 1,1 (tous les λ)** : positif
sur chaque moteur, mais deux fois plus petit que l'estimation de la phase 2 (+6,1 ± 3,1) et à peine
au-dessus du bruit sur un seul moteur. La phase 2 avait de la chance (R5 +7,9) et lab6 n'en avait pas
(mctx +1,8).

## Phase 5 : l'ancrage avec les fins simulées à n = 400 (PC, graine 71002, 400 donnes, 29 octobre)

Contre c130 à n = 400. `lab10/runs/p5{a,b}.tsv`, 21 et 20 min (5,1 et 4,7 s par décision).

| Joueur testé | Score | 2 j. | 3 j. | 4 j. | 5 j. |
| --- | --- | --- | --- | --- | --- |
| 5a `n=400,roll=1,mix=0.5,anchor=raw,lambda=0.005` | **+12,7 ± 2,4** | +6,0 | +13,7 | +16,5 | +14,5 |
| 5b `n=400,roll=1,mix=0.5` (témoin, PC) | +9,3 ± 2,5 | +4,0 | +14,2 | +10,9 | +8,2 |
| lab7 `a1-n400` (même joueur que 5b, CUDA) | +8,9 ± 2,3 | +7,0 | +12,5 | +8,2 | +8,0 |

5b rejoint lab7 (+0,4 ± 3,3) : le PC reproduit les fins simulées de lab7. 5a − 5b : **+3,4 ± 3,2** (gain à 4
et 5 joueurs : +5,6, +6,3). Face au témoin réuni (5b et lab7, 800 donnes, +9,1) : **+3,6 ± 2,9**, au-dessus du
seuil de +3. Le score de 5a est le plus haut jamais mesuré contre c130. Selon la règle, confirmation sur
920 donnes neuves (5c, 5d).
