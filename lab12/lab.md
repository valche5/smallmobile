# Lab12 : la riposte du rival dans les fins simulées

Base : copie de **lab10** au commit `e61f71d` (c130, moteurs Vulkan et CUDA, options de lab7, ancrage
piKL), sans boucle d'apprentissage. Règles communes : [labtools/README.md](../labtools/README.md).
Résultats : [journal.md](journal.md).

## Pourquoi

- Les fins simulées (lab7) sont le plus gros gain à la décision : +5,4 ± 1,5 à n = 100, +8,9 à
  n = 400. Elles prolongent chaque simulation jusqu'au prochain tour du joueur de la racine, chaque
  adversaire jouant le coup que le réseau préfère pour lui.
- Best Reply Search (Schadd & Winands, 2011 ; `labtools/biblio.md`) : entre deux tours du joueur,
  seul l'adversaire qui a la riposte la plus dangereuse joue. Son intérêt premier, voir plus vite le
  tour suivant du joueur, nos fins simulées l'ont déjà ; reste son modèle de l'adversaire. Dans nos
  fins simulées, aucun adversaire ne cherche à nuire au joueur de la racine.
- Paranoid (tous contre le joueur) écarté : à Small World, les adversaires ne se coalisent pas.

## Option (`search::Config`, `az:…,reply=K`)

`reply=K` (K ≥ 2) : dans une fin simulée, le **rival** du joueur de la racine (l'adversaire aux pièces
finales prévues les plus hautes, selon le réseau, à la position d'où part la fin simulée) joue chacun de
ses tours K fois : son tour préféré, puis K − 1 tours dont chaque coup est tiré de sa politique
(Gumbel-max). Les décisions des autres joueurs pendant ce tour (leurs défenses) restent à leur coup
préféré. On garde le tour qui laisse au joueur de la racine la plus basse valeur du réseau au bout du
tour ; la fin simulée continue de là, comme avant. Sans l'option, la recherche est celle de lab10 au
bit près (tests de lab10 repris).

Coût sur le PC (`lab12 speed --positions 3 --seed 5`, un cœur, PC partagé avec une arène) :
`n=100,roll=1,mix=0.5` 1,19 s par recherche (29,9 évaluations par fin simulée) ; avec `reply=4`,
2,28 s (58,7) : **1,9 fois**.

## Protocole, fixé avant tout calcul

Contre un champ de c130 à n = 400, 2 à 5 joueurs, `--device cuda`, une machine vast.ai (RTX 5080 ou
4090), `labtools/score.py`, sans arrêt anticipé.

1. **Présélection** : `n=100,roll=1,mix=0.5,reply=4`, graine 71002, 400 donnes. Témoin déjà mesuré en
   CUDA, même joueur sans `reply` : +5,7 ± 2,4 sur ces donnes (lab7), +5,4 ± 1,5 sur 920 autres
   (graine 72002) ; réunis, +5,5 sur 1 320 donnes. Seuil : **+3 points** sur ce témoin réuni.
2. **Confirmation**, si le seuil est atteint : le candidat et son témoin (`n=100,roll=1,mix=0.5`), 920
   donnes neuves chacun, graine 122001, même machine. Décisif si l'intervalle de la différence exclut 0.

Coût estimé : présélection ~45 min (lab7 : 23 min pour le témoin à 400 donnes sur RTX 5080), confirmation
~2 h 30 ; au plus ~3 h 30, ~1,2 $. Crédit au départ : 5,24 $. La machine est détruite à la fin de son
dernier calcul, automatiquement.

## Commandes

Tests sur le PC, processeur seul :

```sh
timeout 1500 cargo test -p smallmobile-lab12 -p smallmobile-brain12 --no-default-features
timeout 900 cargo clippy -p smallmobile-lab12 -p smallmobile-brain12 --all-targets --no-default-features -- -D warnings
```

Sur vast.ai, depuis mini :

```sh
LAB=lab12 labtools/vast.sh up OFFRE
labtools/vast.sh run r4 7200 'mkdir -p lab12/runs && target/release/lab12 arena --hero az:net=lab12/nets/c130.bin,n=100,roll=1,mix=0.5,reply=4 \
  --field az:net=lab12/nets/c130.bin,n=400 --players 2,3,4,5 --deals 400 --seed 71002 --device cuda \
  --out lab12/runs/r4.tsv'
labtools/vast.sh wait r4 600
labtools/vast.sh get lab12/runs/r4.tsv
python3 labtools/score.py lab12/runs/r4.tsv --role 0
labtools/vast.sh down
```
