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
