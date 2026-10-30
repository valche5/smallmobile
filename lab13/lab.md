# Lab13 : pourquoi c130 n'apprend pas de son professeur (diagnostic)

Base : copie de **lab11** au commit `a9de062` (c130, moteurs, options de lab7 et lab10, `py/` de lab7),
sans la boucle. Réseaux étudiés : ceux de la boucle p1 de lab11 (`lab11/runs/p1/c000.bin` = c130 à
`c011.bin`). Règles communes : [labtools/README.md](../labtools/README.md). Résultats :
[journal.md](journal.md).

## Pourquoi

La boucle p1 de lab11 (professeur `n=400,roll=1,mix=0.5`, ~+9 contre c130) n'a rien rapporté en neuf
pas : duels de −0,8 à −2,7, c010 −0,7 ± 2,2 contre c130 à n = 400. Sur ses propres cibles, la précision
de la politique stagne (46,6 % à c006, 45,7 % à c011) alors que la fenêtre ne contient plus que le
professeur. Deux explications, qui mènent à des pistes différentes :

- **la politique apprend, la valeur ou la recherche masque le gain** : travailler la cible de valeur
  (`VALUE_MIX=1`, valeur des fins simulées ; MuZero Reanalyse) ;
- **la politique n'apprend pas ce que le professeur calcule** (réponse des adversaires, invisible d'un
  seul regard) : entrées calculées à la main décrivant la menace (AlphaGo, KataGo), ou plus de capacité.

## Mesures, fixées avant tout calcul

Sur le PC (Vulkan ; PyTorch ROCm de `lab3/py/.venv`).

**D1, la politique seule.** c010 (huit pas de professeur) et c002 (un pas sur les parties de c130 seules,
témoin de l'apprentissage sans professeur), à n = 1 (le coup préféré du réseau, sans recherche), contre
un champ de c130 à n = 1 ; 920 donnes, graine 131001. La politique a appris si c010 bat c130
(intervalle au-dessus de 0) et fait mieux que c002.

**D2, l'accord avec le professeur** (`py/agree.py`), sur des parties neuves que personne n'a apprises :
1 000 parties du professeur avec c130 (`n=400,roll=1,mix=0.5`, graine 139 000 000, `data/prof`) et,
pour comparaison, 1 000 parties de la recette ordinaire (`n=400`, graine 139 000 001, `data/ord`).
Réseaux : c000 (c130), c002, c005, c008, c010, c011. Pour chacun : les mesures de la validation de
`train.py`, et le taux de coups préférés égaux à ceux de la cible, séparément là où c130 et le
professeur diffèrent (« écarts ») et là où ils s'accordent. La politique a appris le professeur si,
aux écarts, c010 et c011 dépassent nettement c002 (de plus de 5 points de pourcentage).

**D3, la recette** (ajouté le 30 octobre après D2, avant tout calcul). c130 réentraîné hors ligne
(`py/fit.py`) sur les parties du professeur de lab11 encore gardées (cycles c004 à c011, ~3,4 millions
de positions), 4 passages moyens, taux maximal en un cycle : **1,5e-4** (celui de la boucle), **5e-5**,
**1,5e-5**. Mesures toutes les 2 000 lots sur `data/prof` (le juge) et `data/ord`. **La recette
suffit** si la meilleure perte de politique sur `data/prof` passe sous **1,287** (c130 : 1,307, moins
0,02) avec un coup de la cible à **47,5 %** au moins (c130 : 46,5 %). Alors : arènes du meilleur réseau
contre c130 (n = 1 et n = 400). Sinon : D4, la capacité (un réseau plus large, même protocole).

**D5, le plafond des cibles** (ajouté le 30 octobre après D3, avant tout calcul). `lab13 consistency` :
1 500 positions de `data/prof` (et de `data/ord`), rejouées depuis leurs `.moves` ; sur chacune, 4
recherches du professeur (de la recherche ordinaire `n=400`), chacune sa graine, bruit de Gumbel à la
racine comme dans le selfplay ; graine 151001. Mesures : même coup préféré entre deux recherches (le
plafond de la précision), coup préféré de l'a priori de c130, entropies croisées de la cible à l'a priori
(la perte de politique) et à la moyenne des trois autres recherches (ce qu'atteindrait un réseau qui
saurait la cible moyenne), entropie de la cible (le plancher). **Plafond bas** si l'accord entre
recherches du professeur dépasse celui de l'a priori de moins de 5 points : ses cibles sont alors
surtout du bruit autour de c130.

**D6, des cibles ancrées** (ajouté le 30 octobre après D5, avant tout calcul). D5 refait sur les mêmes
positions (graine 151001) avec l'ancrage piKL de lab10 (`anchor=raw,lambda=0.005`) : le professeur
(`n=400,roll=1,mix=0.5,anchor=raw,lambda=0.005`, sur `data/prof`) et la recherche ordinaire
(`n=400,anchor=raw,lambda=0.005`, sur `data/ord`). Mesure clé : **la marge apprenable**, CE à l'a priori
moins CE à la moyenne des autres recherches (D5 : professeur sans ancrage 0,004, recherche ordinaire
0,219). **Les cibles ancrées sont apprenables** si la marge du professeur ancré atteint **0,10**. Alors :
une boucle dont le professeur est ancré.

## Commandes

```sh
target/release/lab13 arena --hero az:net=lab11/runs/p1/c010.bin,n=1 --field az:net=lab11/runs/p1/c000.bin,n=1 \
  --players 2,3,4,5 --deals 920 --seed 131001 --device vulkan --out lab13/runs/d1-c010.tsv
LAB13_CONCURRENT=16 target/release/lab13 selfplay --agent az:net=lab13/nets/c130.bin,n=400,roll=1,mix=0.5 \
  --players 2,3,4,5 --games 1000 --seed 139000000 --threads 20 --device vulkan --out lab13/data/prof
(cd lab13/py && ../../lab3/py/.venv/bin/python agree.py --data ../data/prof --nets ../../lab11/runs/p1/c000.bin …)
```
