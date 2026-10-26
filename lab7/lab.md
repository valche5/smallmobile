# Lab7 : les pistes de sortie du plateau de c130

Base : copie de **lab6** au commit `65d3daf` (donc lab3/c130, moteur CUDA compris). Lab6 est figé.
Règles communes : [labtools/README.md](../labtools/README.md). Résultats : [journal.md](journal.md).

## Pourquoi

Relu dans les six journaux et le code (revue du 26 octobre) :

- À poids fixes, plus de recherche ne sert plus : 8× (+2,0 ± 2,3, lab4), 16× (+1,8 ± 2,2, lab6),
  toute la largeur, jetons ordonnés ou fusionnés : rien de net. La recherche optimise la valeur du
  réseau en fin de (morceau de) tour ; à 4 et 5 joueurs, moins de 4 % des simulations atteignent le
  prochain tour du joueur : la réponse des adversaires n'est jamais simulée.
- Lab1 avait mesuré l'inverse avec ses fins simulées, le réseau jouant pour tous (`rp=1`) : +11,7
  parmi les Expert, contre +0,7 pour la valeur seule à n = 800, mêmes donnes (graine 201). Lab2 les
  a retirées en ouvrant, avec quatre autres changements, sans mesure séparée.
- Le choix du peuple est cherché à un seul coup : la carte qui arrive au marché vient de la pioche
  cachée, différente dans chaque monde, et entre dans l'empreinte du coup (`moves::chance`) ; chaque
  simulation qui passe par un choix crée un nouveau nœud et s'y arrête.
- L'Expert du jeu part de `Knowledge::of` (`lab3/brain/src/expert.rs`) : l'histoire des joueurs,
  120 entrées du réseau, y est vide à chaque décision ; les arènes la tiennent à jour.
- En selfplay, chaque coup est tiré avec le bruit de Gumbel, 7 sur 8 à 50 simulations ; l'arène, le
  plafond de lab4 et le jeu jouent sans bruit à 400.

## Options d'agent (`search::Config`, `az:net=…,n=…,option=…`)

Toutes désactivées par défaut : la recherche est alors celle de lab6 au bit près (test
`without_options_the_search_is_lab6s`).

| Option | Sens |
| --- | --- |
| `roll=R` | Fins simulées : une position nouvelle est jugée au début du R-ième tour suivant du joueur de la racine (sa défense hors de son tour comprise), le coup préféré du réseau joué pour tous jusque-là, les coups forcés sans évaluation ; le résultat exact si la partie finit avant |
| `mix=λ` | Part de la fin simulée dans la valeur d'une position nouvelle (1 par défaut ; 0,5 comme AlphaGo) |
| `market=late` | La carte arrivée au marché ne range les suites qu'aux choix de peuple suivants, seuls coups qui en dépendent ; ailleurs, les nœuds servent à tous les mondes |
| `crn=1` | Un seul monde pour les simulations d'un même tour du halving (nombres aléatoires communs) |
| `win=w` | Poids de la victoire dans la valeur (0,5 ; lab4 : 0,75 donnait +1,3 ± 1,0) |
| `explore=1` | Bruit de Gumbel à la racine pour jouer, comme le selfplay (diagnostic) |
| `history=0` | L'histoire des joueurs vide à chaque décision, comme l'Expert du jeu (diagnostic) |

Coût sur un cœur de mini (`lab7 speed`, 12 positions, graine 5) : n = 400 122 ms par recherche,
`market=late` 121 ms, `n=100,roll=1` 1,5 s (29 évaluations par fin simulée en moyenne, ~12 fois
n = 400, de l'ordre des 16× de lab6).

## Phase 1 : arènes à poids fixes, fixées avant tout calcul

Toutes contre un champ de c130 à n = 400 (`az:net=lab7/nets/c130.bin,n=400`), 2 à 5 joueurs,
`--device cuda`, score de `labtools/score.py` (intervalle à 95 %, donnes rééchantillonnées).
Une machine vast.ai, la même pour toute la phase.

| Essai | Joueur testé | Donnes | Graine | Question |
| --- | --- | --- | --- | --- |
| A0 | `n=400,history=0` | 920 | 71001 | L'Expert du jeu, sans histoire, est-il plus faible que celui qu'on mesure ? |
| A1 | `n=100,roll=1` | 400 | 71002 | Les fins simulées sortent-elles du plateau, à calcul comparable aux 16× de lab6 ? |
| A2 | `n=400,market=late` | 920 | 71003 | Chercher la suite d'un choix de peuple aide-t-il, à calcul égal ? |
| A3 | `n=50,explore=1` et `n=50` | 400 chacun | 71004 | Combien le bruit du selfplay coûte-t-il aux parties d'entraînement ? |

Suites prévues selon les résultats, chacune fixée avant d'être lancée :

- A1 ≥ +3 points : `n=400,roll=1` sur les donnes de A1 (graine 71002), puis `mix=0.5` et `crn=1` ;
  confirmation du meilleur sur 920 donnes neuves (graine 72002).
- A2 positif (intervalle au-dessus de 0) : confirmation sur 920 donnes neuves (72003), puis
  combinaison avec le meilleur de A1.
- A0 négatif (intervalle sous 0) : transmettre l'histoire à l'Expert du jeu (le serveur archive déjà
  les actions de chaque partie : les rejouer, comme `GameLog::replay`).
- A3 : si le bruit coûte plusieurs points, phase 2 (selfplay sans bruit après la première manche,
  puis cible de valeur débarrassée de la chance), avec une boucle témoin.

Pas d'arrêt anticipé : les arènes vont au bout de leur taille.

## Phase 2 : apprentissage (boucle `lab7/loop.sh`)

Chaque bras part de c130 avec une fenêtre neuve, remplie par c130 avec la recette du bras, puis fait
ses pas (25 000 parties, n = 400 pour les recherches enregistrées, AdamW neuf à chaque pas, taux
maximal 1,5e-4 en un cycle) ; duel de chaque pas contre c130 (n = 200, test séquentiel). Une
machine vast.ai par bras, en parallèle (`VAST_NAME`), disque d'au moins 50 Go (PyTorch, fenêtre).

| Bras | Réglages | Hypothèse |
| --- | --- | --- |
| T0, témoin | recette de lab3 (bruit de Gumbel à chaque coup) | la boucle depuis c130 ne progresse plus (w2 de lab4 : +0,9 ± 1,4 en 20 pas) |
| T1, sans bruit | `NOISY=1` : bruit à la première manche seulement ; ensuite, recherches rapides sans bruit et coup sans bruit des recherches complètes (`Searched::greedy`), dont la cible de politique garde son bruit | la valeur apprend le jeu qu'elle juge |
| T2, sans bruit ni hasard | `NOISY=1 LUCK=1` : en plus, la part du hasard retranchée de la cible de la valeur (`luck`) | une cible moins dispersée, sans biais |
| T3, professeur | `AGENT=roll=1` (recherches enregistrées avec fins simulées), si la phase 1 les montre meilleures | un professeur plus fort que le réseau |

Coût d'un bras sur RTX 5080 (CUDA, ~47 s par 1 000 parties) : ~20 min de parties, ~3 min
d'apprentissage et ~3 min de duel par pas ; fenêtre de 5 cycles (~1 h 40), puis 10 pas (~4 h 20) :
~6 h, ~2,5 $. Mesure finale de chaque bras : son dernier réseau contre c130, 920 donnes neuves
(graine 72101), n = 400 des deux côtés ; les bras se comparent sur les mêmes donnes.

Ensuite, si un bras paie : architecture (combos et joueurs comme entités, d'abord hors ligne) ;
spécialistes par nombre de joueurs depuis le meilleur réseau, au taux bas.

## Commandes

Compilation sur mini, CPU seul, pour les tests (Debian n'a pas le glslc du shader bf16) :

```sh
timeout 1200 cargo test -p smallmobile-lab7 -p smallmobile-brain7 --no-default-features
timeout 900 cargo clippy -p smallmobile-lab7 -p smallmobile-brain7 --all-targets --no-default-features -- -D warnings
```

Sur vast.ai (`labtools/vast.sh`, binaire compilé sur mini avec Vulkan et CUDA), par exemple A1 :

```sh
LAB=lab7 labtools/vast.sh up OFFRE
labtools/vast.sh run a1 7200 'target/release/lab7 arena --hero az:net=lab7/nets/c130.bin,n=100,roll=1 \
  --field az:net=lab7/nets/c130.bin,n=400 --players 2,3,4,5 --deals 400 --seed 71002 --device cuda \
  --out lab7/runs/a1.tsv'
labtools/vast.sh wait a1 600
labtools/vast.sh get lab7/runs/a1.tsv
python3 labtools/score.py lab7/runs/a1.tsv
labtools/vast.sh down
```
