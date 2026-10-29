# Lab11 : une boucle d'apprentissage depuis c130 avec un professeur nettement plus fort

Base : copie de **lab10** au commit `443e9bc` (c130, moteurs Vulkan et CUDA, options de lab7, ancrage
piKL), plus la boucle de lab7 (`loop.sh`, `py/`), que lab10 n'avait pas reprise. Règles communes :
[labtools/README.md](../labtools/README.md). Résultats : [journal.md](journal.md).

## Pourquoi

- Bilan des labos (`labtools/bilan.md`) : toutes les boucles depuis c130 sont restées plates (w1, w2,
  T0 à T3, p1). Dénominateur commun : **le professeur n'était pas plus fort que l'élève** au budget du
  selfplay. T3 (lab7), le plus fort, avait des fins simulées à n = 100 (+5,4 en arène) : 10 pas, rien
  (−1,7 ± 1,4).
- Lab10, phase 5 : `n=400,roll=1,mix=0.5,anchor=raw,lambda=0.005` fait **+12,7 ± 2,4** contre c130 à
  n = 400 (400 donnes ; confirmation 5c/5d en cours), deux fois l'écart du professeur de T3.
- Le bilan demande aussi des boucles longues : lab2 → c130 a pris 60 à 130 pas ; toutes les boucles
  depuis c130 ont eu 4 à 20 pas.

## La recette, fixée avant tout calcul

Sur le PC (RX 9070 XT, Vulkan ; PyTorch ROCm de `lab3/py/.venv`), une seule boucle :

```sh
RUN=p1 SIMS=400 AGENT=roll=1,mix=0.5,anchor=raw,lambda=0.005 VALUE_MIX=0.5 GAMES=12500 \
  FILL_SIMS=400 FILL_AGENT= FILL_GAMES=25000 lab11/loop.sh 40
```

- **Professeur** : les recherches enregistrées (une décision sur 8) sont celles de 5a. Elles donnent la
  cible de politique (politique améliorée, ancrée) et la moitié de la cible de valeur (`VALUE_MIX=0.5` :
  la valeur de la recherche, fins simulées comprises, comme T3). Les autres décisions : recherches
  rapides à n = 50 avec bruit de Gumbel, comme dans lab7 (elles héritent de l'ancrage, qui ne change
  rien à ce budget : lab10, phase 1). Si 5c ne bat pas 5d, le professeur devient
  `roll=1,mix=0.5` sans ancrage (5b, +9,3).
- **Fenêtre** : 2 millions de positions au minimum (`py/data.py`), deux fois moins que lab7. Elle est
  remplie par c130 avec la recette ordinaire (n = 400, ~65 parties/s), puis ne contient plus que des
  parties du professeur après quatre pas. Dans T3, la fenêtre de 4 millions diluait le professeur.
- **Cycles** de 12 500 parties (~500 000 positions), AdamW neuf à chaque pas, taux maximal 1,5e-4.
- **Jusqu'au cycle 40** (`loop.sh 40`), sans arrêt sur les duels. Les cycles du remplissage comptent
  (deux ou trois de 25 000 parties), et le premier pas apprend sur la fenêtre de c130 seule, comme dans
  lab7 : environ 37 pas avec le professeur.

Avant le premier pas :

1. **Essai à blanc** (`LAB11_WINDOW_MIN=50000 GAMES=200 FILL_GAMES=1000 DEALS=40 FINAL=0`, 2 pas) :
   selfplay Vulkan, apprentissage ROCm, duel. Rien n'en est gardé.
2. **Débit du professeur** : 1 000 parties de selfplay avec la recette ci-dessus, sur le PC libre (500
   prévues d'abord ; mais la boucle mène 320 parties à la fois, 20 fils × 16). Si le débit est sous
   2 parties/s (cycle de plus de 1 h 45), les cycles passent à 6 250 parties (fenêtre inchangée).
   Estimation : ~2,5 parties/s (une recherche de 5a pour ~20 décisions enregistrées par partie,
   ~50 recherches de 5a par seconde dans l'arène de lab10), soit ~85 min de parties et ~95 min par pas,
   **~65 h pour 40 pas**.
3. **Le professeur** : avec ancrage si `score.py lab10/runs/p5c.tsv lab10/runs/p5d.tsv` conclut « A
   meilleur » (intervalle au-dessus de 0), sans sinon.

Ces trois étapes et le lancement de la boucle s'enchaînent seuls dès la fin de 5d
(`lab11/runs/start-p1.sh`, hors Git, recopié dans le journal).

## Mesures

- **Duel de chaque pas** : le réseau seul (n = 200 des deux côtés, sans option) contre c130, 460 donnes,
  graine 1004, test séquentiel (`loop.sh`). Indicatif.
- **Points d'étape** aux pas 10, 20 et 30 : le réseau du pas contre c130, n = 400 des deux côtés,
  400 donnes, graine 111001 (mêmes donnes à chaque étape). Seul arrêt prévu : au pas 20, si le point
  d'étape est sous −3 (le réseau se dégrade).
- **Mesure finale** (`loop.sh`, `FINAL=920`) : c040 contre c130 à n = 400 des deux côtés, 920 donnes,
  graine 112101. **Succès : intervalle au-dessus de 0.** Repères : T0 −0,5 ± 1,4, T3 −1,7 ± 1,4 (lab7,
  10 pas).
- Si le réseau progresse : le même professeur avec le nouveau réseau, contre c130 avec 5a (les gains de
  la recherche et du réseau s'additionnent-ils ?).

## Commandes

Tests sur le PC, processeur seul :

```sh
timeout 1500 cargo test -p smallmobile-lab11 -p smallmobile-brain11 --no-default-features
timeout 900 cargo clippy -p smallmobile-lab11 -p smallmobile-brain11 --all-targets --no-default-features -- -D warnings
```

Point d'étape, par exemple au pas 10 :

```sh
target/release/lab11 arena --hero az:net=lab11/runs/p1/c010.bin,n=400 --field az:net=lab11/runs/p1/c000.bin,n=400 \
  --players 2,3,4,5 --deals 400 --seed 111001 --device vulkan --out lab11/runs/p1/step-c010.tsv
python3 labtools/score.py lab11/runs/p1/step-c010.tsv --role 0
```
