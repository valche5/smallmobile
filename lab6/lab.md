# Lab6 : c130 est-il loin de son plafond ?

Base : copie de **lab5** (donc de lab3/c130) au commit `bc6b820`, sans le diagnostic `probe`.
Lab5 est figé. Règles communes : [labtools/README.md](../labtools/README.md).

## Question

Lab5 a mesuré la marge d'un coup changé (+0,66 point, une borne basse : six candidats, c130
continue la partie). Ici, on donne à c130 **beaucoup plus de recherche** et on le fait jouer contre
des c130 ordinaires : la recherche longue peut trouver d'autres coups et des plans sur plusieurs
décisions. Si elle gagne nettement, c130 est loin du meilleur jeu à sa portée ; sinon, il est proche
du plafond de ce réseau et de cette recherche.

Lab4 l'a fait à 8 fois le budget (3 200 contre 400 simulations, 400 donnes) : **+2,0 ± 2,3**,
indécis. Lab6 va plus loin : **16 fois** (6 400), puis **64 fois** (25 600).

## Arène

`lab6 arena --hero az:net=lab6/nets/c130.bin,n=6400 --field az:net=lab6/nets/c130.bin,n=400
--players 2,3,4,5 --deals N --device vulkan --out r.tsv`, score par `labtools/score.py`
(points, par donne, intervalle à 95 %). Graines neuves, notées au journal avant chaque lancement.

1. Mini-arène de coût : 8 donnes par taille, durée et temps par décision.
2. 16× : 400 donnes (100 par taille), précision attendue ±2,3 comme lab4.
3. 64× : même nombre de donnes, après le coût mesuré en 2.

Taille fixée avant de voir les scores ; pas d'arrêt dès qu'un résultat devient favorable.

## Calcul

Sur vast.ai (`labtools/vast.sh`), une machine à la fois, même modèle de GPU pour une même arène.
À côté, `py/bench_cuda.py` mesure le réseau sur CUDA (PyTorch) contre `lab6 bench-vulkan` sur
la même carte : un portage CUDA ne se décide qu'avec ce chiffre.
