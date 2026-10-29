# Lab10 : l'ancrage de la recherche de c130 sur sa politique (piKL)

Base : copie de **lab7** au commit `1ae5637` (donc lab3/c130, moteur CUDA et options de lab7 compris),
sans sa boucle d'apprentissage ni son Python. Lab7 est figé. Règles communes :
[labtools/README.md](../labtools/README.md). Résultats : [journal.md](journal.md).

## Pourquoi

- Bilan des labos (`labtools/bilan.md`) : à poids fixes, chercher plus ne rapporte rien (8× +2,0 ± 2,3,
  16× +1,8 ± 2,2) ; les fins simulées, si (+5,4 à +8,9). Hypothèse « exploitation » : la recherche
  exploite les erreurs de la valeur du réseau, et le fait d'autant plus qu'elle cherche longtemps.
- piKL (Jacob et al., 2022 ; `labtools/biblio.md`, note de lecture) : la recherche est pénalisée selon
  sa distance KL à une politique d'ancrage. Notre recherche Gumbel a déjà cette forme — politique
  améliorée = softmax(logits + σ(q)) — mais avec la recette de mctx :
  - les valeurs des coups sont **ramenées entre 0 et 1 dans chaque nœud** : un écart de bruit entre
    coups presque égaux pèse autant qu'un vrai écart ;
  - le facteur vaut **(50 + visites du coup le plus vu) × 0,1** : l'ancrage **faiblit quand on cherche
    plus**. Mesuré ici (`lab10 speed`, 24 recherches, c130) : KL(politique de la racine ‖ a priori)
    0,77 nat à n = 400, **1,10 à n = 6 400**.
- Prédiction de la biblio : si l'hypothèse est juste, un ancrage qui ne faiblit pas gagne plus à 16×
  qu'à n = 400, et réduit le gain des fins simulées (même défaut corrigé).

## Options (`search::Config`, `az:net=…,n=…,anchor=…`)

Sans option, la recherche est celle de lab7 au bit près, options de lab7 comprises (test
`without_anchor_the_search_is_lab7s`).

| Option | Sens |
| --- | --- |
| `anchor=mctx` | Le défaut (lab2 à lab7) : valeurs ramenées entre 0 et 1, fois (50 + visites max) × 0,1 |
| `anchor=fixed,scale=S` | Valeurs ramenées entre 0 et 1, fois S quelles que soient les visites : un ancrage qui ne faiblit pas |
| `anchor=raw,lambda=L` | Valeurs brutes (entre 0 et 1) divisées par L, sans normalisation dans le nœud : le λ fixe de piKL |

`lab10 speed` affiche, en plus des temps et des empreintes, la KL de la politique de la racine à
l'a priori et la part des décisions où le coup joué n'est pas le coup préféré du réseau.

Étalonnage sur mini (`speed --positions 6 --seed 5`, 24 recherches, un cœur) :

| Recherche | KL (nat) | Coup ≠ préféré | ms |
| --- | --- | --- | --- |
| `n=400` (mctx) | 0,769 | 46 % | 117 |
| `n=6400` (mctx) | 1,096 | 46 % | 1 392 |
| `n=400,anchor=fixed,scale=5` | 0,346 | 25 % | 112 |
| `n=400,anchor=fixed,scale=10` | 0,605 | 42 % | 116 |
| `n=400,anchor=fixed,scale=20` | 0,848 | 46 % | 117 |
| `n=6400,anchor=fixed,scale=10` | 0,392 | 46 % | 1 454 |
| `n=400,anchor=raw,lambda=0.05` | 0,028 | 8 % | 99 |
| `n=400,anchor=raw,lambda=0.02` | 0,069 | 12 % | 97 |
| `n=400,anchor=raw,lambda=0.01` | 0,169 | 21 % | 105 |
| `n=400,anchor=raw,lambda=0.005` | 0,273 | 29 % | 104 |
| `n=400,anchor=raw,lambda=0.002` | 0,527 | 42 % | 108 |
| `n=6400,anchor=raw,lambda=0.005` | 0,331 | 29 % | 1 248 |

## Protocole, fixé avant tout calcul

Toutes les arènes contre un champ de **c130 à n = 400** (`az:net=lab10/nets/c130.bin,n=400`), 2 à 5
joueurs, `--device cuda`, score de `labtools/score.py` (intervalle à 95 %, donnes rééchantillonnées),
sans arrêt anticipé. Une machine vast.ai (RTX 5080 ou 4090, `labtools/vast.sh`), la même pour toute la
campagne ; crédit au départ 6,68 $.

### Phase 1 : présélection à n = 400 (400 donnes, 100 par taille, graine 101001)

| Candidat | Joueur testé | KL étalonnée |
| --- | --- | --- |
| F10 | `n=400,anchor=fixed,scale=10` | 0,61 (≈ mctx à n = 400) |
| F20 | `n=400,anchor=fixed,scale=20` | 0,85 |
| R2 | `n=400,anchor=raw,lambda=0.002` | 0,53 |
| R5 | `n=400,anchor=raw,lambda=0.005` | 0,27 |

Mêmes donnes pour les quatre : différences appariées entre candidats. Seuil de pertinence pratique :
+3 points (±2,3 attendus sur 400 donnes). À n = 400 on n'attend pas grand-chose : F10 et F20 y
diffèrent peu de mctx ; c'est la mesure témoin de la phase 2.

### Phase 2 : à 16× (n = 6 400, 400 donnes, graine 62001)

La graine de l'arène 16× de lab6 (`lab6/runs/arena16.tsv`, c130 à n = 6 400 : +1,8 ± 2,2), pour une
différence appariée avec elle. Le meilleur `fixed` et le meilleur `raw` de la phase 1 (à égalité, F10 et
R5), à n = 6 400. Question : l'ancrage constant fait-il payer le calcul là où mctx ne gagne rien ?
Critère : différence appariée avec lab6 au-dessus de 0, et gain sur la phase 1 du même ancrage.

### Phase 3 : confirmation (920 donnes neuves, graine 102001)

Le meilleur candidat des phases 1 et 2 s'il atteint +3 points ; sinon, rien. Puis, s'il y a un gain, la
combinaison avec les fins simulées de lab7 (`roll=1,mix=0.5`) sur les donnes de la confirmation de lab7
(graine 72002, 920 donnes : `n=100,roll=1,mix=0.5` +5,4 ± 1,5), pour voir si les deux gains
s'additionnent ou corrigent le même défaut.

Coût estimé : une arène de 400 donnes à n = 400 ≈ 10 min sur RTX 5080 (lab7 : 0,85 s par décision) ;
à n = 6 400 ≈ 27 min (lab6) ; 920 donnes à n = 400 ≈ 20 min. Campagne entière ≈ 2 h 30, ~1 $.

## Commandes

Tests sur mini, processeur seul :

```sh
timeout 900 cargo test -p smallmobile-lab10 -p smallmobile-brain10 --no-default-features
timeout 600 cargo clippy -p smallmobile-lab10 -p smallmobile-brain10 --all-targets --no-default-features -- -D warnings
```

Sur vast.ai (`labtools/vast.sh`, binaire compilé sur mini avec Vulkan et CUDA), par exemple F10 :

```sh
LAB=lab10 labtools/vast.sh up OFFRE
labtools/vast.sh run f10 2400 'target/release/lab10 arena --hero az:net=lab10/nets/c130.bin,n=400,anchor=fixed,scale=10 \
  --field az:net=lab10/nets/c130.bin,n=400 --players 2,3,4,5 --deals 400 --seed 101001 --device cuda \
  --out lab10/runs/f10.tsv'
labtools/vast.sh wait f10 600
labtools/vast.sh get lab10/runs/f10.tsv
python3 labtools/score.py lab10/runs/f10.tsv --role 0
labtools/vast.sh down
```
