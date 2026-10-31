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

**D7, des cibles apprenables font-elles un réseau plus fort ?** (ajouté le 30 octobre après D6, avant
tout calcul). 20 000 parties du professeur ancré mené par c130 (`n=400,roll=1,mix=0.5,anchor=raw,
lambda=0.005`, recette du selfplay, graine 171 000 000, `data/anc-train`, ~700 000 positions) et 1 000
neuves pour juger (graine 171 000 001, `data/anc-test`). c130 réentraîné dessus (`py/fit.py`, 4 passages,
`--value-mix 0.5` comme la boucle de lab11) à trois taux : 1,5e-4, 5e-5, 1,5e-5. Le réseau de la
meilleure perte de politique sur `anc-test` passe en arène contre c130, n = 400 des deux côtés (le
réseau seul change), 920 donnes, graine 172001 ; et n = 1 (la politique seule), mêmes donnes.
**Succès** : en arène à n = 400, intervalle au-dessus de 0. Alors : une boucle dont le professeur est
ancré, dans un nouveau labo.

**D9, par type de décision** (ajouté le 30 octobre après D8 de lab14, avant tout calcul). Sans
nouvelle partie. Chaque position de `data/anc-test` (neuves) et 40 000 de `data/anc-train` tirées au
hasard (apprises), graine 9, rangées par type d'après la phase et les coups possibles : **peuple**
(un choix au marché), **déclin** (décliner est possible), **conquête** (le reste de la phase de
conquête), **pouvoirs** (héros, forteresse, campement au redéploiement), **jetons** (le reste du
redéploiement), **défense**. Les apprentissages de D7 refaits à l'identique (1,5e-4 et 5e-5), le
réseau de la fin gardé (`fit.py --last`). Par type, pour c130 et ces deux réseaux : part des positions
et de la perte de politique de c130, perte de politique, entropie de la cible (le plancher), coup de
la cible. Lecture, fixée d'avance (± ~0,01 sur les gros types, plus sur les petits) :

- **un type fait l'essentiel de l'excès de c130** (perte moins entropie) : c'est là que le professeur
  diffère, la suite s'y concentre ;
- **l'apprentissage baisse la perte d'un type sur les apprises, pas sur les neuves** : trop peu
  d'exemples, ou une règle qui ne se généralise pas ; **sur aucune des deux** : ce type ne s'apprend
  pas avec ces entrées et cette sortie ;
- **un type monte sur les neuves** pendant que d'autres baissent : l'apprentissage déplace l'erreur.

**D10, la recherche courte hors des décisions stratégiques** (ajouté le 30 octobre après D9, avant
tout calcul). Option `light=N` de la recherche : au redéploiement et à la défense (jetons, pouvoirs,
fin de tour ; 53 % des positions, où D9 trouve la cible la plus incertaine), N itérations sans fins
simulées ; ailleurs (peuple, déclin, conquête), la recherche entière. Joueurs :
`n=100,roll=1,mix=0.5,light=16` et `…,light=1` (le coup préféré du réseau). Mesures :

1. **le temps** de l'Expert du jeu (`lab13-expert`, un cœur du PC, sans GPU comme sur mini), parties
   entières de 4 et 5 joueurs où tous les sièges sont l'Expert, comme pour sa mise en service ;
2. **la force** : arène contre c130 à n = 400, graine 72002, 920 donnes (celles de la confirmation de
   lab7, `lab7/runs/a1-confirm.tsv` : +5,4 ± 1,5 sans `light`), Vulkan ; différence appariée au témoin.

**Retenu** pour l'Expert du jeu : la variante la plus rapide dont la différence au témoin est
au-dessus de **−1,5** point, borne basse de l'intervalle au-dessus de **−3,5**. Alors elle remplace
l'Expert actuel, et le selfplay ciblé (piste B) s'en sert.

**D11, un selfplay du professeur plus rapide** (ajouté le 30 octobre après D10, avant tout calcul).
Le GPU est saturé (98 %) et les fins simulées font l'essentiel des évaluations, dont 55 % au
redéploiement (un jeton, une évaluation). Deux changements :

- `place=1` : dans les fins simulées, les jetons du redéploiement posés selon la répartition du noyau
  (`Game::full_deployment`, d'après la menace), sans évaluation ; les pouvoirs restent au réseau. Il
  change le professeur : arènes contre c130 à n = 400, graine 107001, 920 donnes, appariées à lab10
  `p5c` (même professeur sans `place`, +9,7 ± 1,6), à n = 400 et à n = 200. **Retenu** si la différence
  à `p5c` est au-dessus de **−1,5**, borne basse au-dessus de **−3,5** ; n = 200 seulement s'il passe
  aussi (deux fois plus de parties).
- `--full P,D,C,A` (selfplay) : la part des coups cherchés à fond et enregistrés par type (peuple,
  déclin possible, conquête, autres), au lieu de 12,5 % de tous les coups. Ne change pas le professeur.

**D12, des parties ciblées font-elles généraliser les décisions stratégiques ?** (ajouté le 31 octobre
après D11, avant tout calcul). 20 000 parties du professeur ancré avec `place=1`
(`n=400,roll=1,mix=0.5,anchor=raw,lambda=0.005,place=1`), `--full 1,1,0.125,0`, graine 181 000 000,
`data/tgt-train` (~1,1 million de positions, dont ~170 000 choix de peuple et ~750 000 déclins
possibles, contre ~22 000 et ~93 000 en D7). c130 réentraîné dessus comme en D7 (`fit.py`, 4 passages,
`--value-mix 0.5`, taux 1,5e-4, 5e-5, 1,5e-5, réseau de la fin gardé), jugé par type sur
`data/anc-test` (`d9.py`, parties neuves de D7). Le réseau retenu : celui de la fin dont la perte
moyenne sur les positions de peuple, de déclin et de conquête de `anc-test` est la plus basse.

- **Les décisions stratégiques se généralisent** si, pour ce réseau, la perte sur `anc-test` baisse au
  peuple et au déclin (écart à c130 sous zéro de plus de deux erreurs types), là où D9 la voyait monter
  (+0,05, +0,04).
- **Succès** : arène du réseau retenu contre c130, n = 400 des deux côtés, graine 183001, 920 donnes,
  intervalle au-dessus de 0 ; aussi n = 1 (politique seule), mêmes donnes.

**D13, des têtes construites pour le choix du peuple et le déclin** (ajouté le 31 octobre après D12,
avant tout calcul ; Python seul, hors ligne). Dans c130, ces deux décisions sortent du seul vecteur
global (`glob_pol2`) : les six cartes du marché y entrent à plat, chaque place a sa sortie sans
partage, et une carte ne rencontre les régions qu'à travers leur moyenne et leur maximum. `py/heads.py`
garde le tronc de c130 et ajoute aux logits de ces coups (corrections parties de zéro : au départ,
c130 exactement) :

- **peuple** : chaque carte (ses entrées, sa place, le vecteur global) résumée par un même petit
  réseau pour les six places ; chaque paire carte × région passée par une même couche, regroupée sur
  les régions (moyenne, maximum) ; un même score par carte (Deep Sets, sans attention) ;
- **déclin** : une même couche sur chaque région (son état, le vecteur global), regroupée sur mes
  régions et sur toutes ; un score.

Apprentissage sur `data/tgt-train` (D12), 4 passages, comme `fit.py` (`py/d13.py`) : **t0** sans têtes,
tout à 1,5e-5 (le témoin, D12) ; **h1** têtes à 1e-3, tronc figé ; **h2** têtes à 1e-3, tronc à
1,5e-5 ; **h3** têtes à 3e-4, tronc figé. Mesure par type sur `data/anc-test` (`d9.py`).
**L'architecture était la limite** si un essai à têtes baisse la perte de politique sur `anc-test` au
peuple ou au déclin d'au moins **0,03** sous c130, sans monter de plus de 0,01 ailleurs. Alors : les
têtes dans les moteurs (processeur, Vulkan) et arènes. Sinon : la structure de la sortie n'est pas en
cause.

*Arènes de D13* (ajouté le 31 octobre après le critère atteint au peuple, avant tout calcul). La tête
du peuple seule (h2 sans la tête du déclin, `--no-decline`, `runs/d13-h2p.bin`), dans le moteur du
processeur (`net::Pick`, mêmes logits que PyTorch à 1e-4 près). Contre c130, n = 1 puis n = 400 des
deux côtés, graine 185001, 920 donnes, processeur. **Succès** : à n = 400, intervalle au-dessus de 0.

**D14, la valeur est-elle le levier ?** (ajouté le 31 octobre après les arènes de D13, avant tout
calcul). Relecture de D1 à D13 : le gain des fins simulées est, par construction, une correction des
valeurs des feuilles (`Az::mixed`), avec le même a priori ; à n = 400 l'a priori pèse peu (lab10 :
0,77 nat d'écart à la racine). Un élève qui apprend la politique du professeur ne retrouve donc pas sa
recherche (gains à n = 1 : +1,2 à +3,0 ; à n = 400 : rien). Hypothèse : la valeur de c130 est apprise
sur des parties jouées aux sept huitièmes par des recherches rapides bruitées (A3 de lab7 : −10,7 au
jeu d'arène), des adversaires qui ne punissent pas ; la fin simulée gloutonne révèle la riposte
(Willemsen, Baier & Kaisers 2021, A0GB : la cible d'AlphaZero n'est juste que pour un agent qui
explore). Mesure, sans apprentissage :

- **le juge** : 1 000 parties de c130 à n = 400, **sans bruit** (`--noisy-rounds 0`), toutes les
  décisions cherchées à fond et enregistrées (`--full 1,1,1,1`), graine 191 000 001, `data/judge` : le
  résultat de ces parties est celui du jeu d'arène, pas du selfplay bruité (sinon il porte le même
  biais que la cible actuelle) ;
- à chaque position enregistrée, le selfplay écrit aussi (`--revalue 4`, `roll_win.npy`,
  `roll_gain.npy`) la **valeur par fin simulée** : depuis la position même, le coup préféré du réseau
  pour tous jusqu'au début du prochain tour du joueur, jetons posés par le noyau (`place=1`), têtes du
  réseau à l'horizon (ou le résultat si la partie finit), moyenne de 4 mondes possibles ; part de
  victoire et pièces à gagner d'ici la fin, vues du joueur qui agit (comme les têtes) ;
- `py/d14.py` : sur le juge, l'entropie croisée au résultat (et le Brier) de la **valeur du réseau**,
  de la **valeur de la recherche ordinaire à la racine** (`value`, déjà enregistrée) et de la **valeur
  par fin simulée**, plus leur moyenne à parts égales ; l'erreur quadratique des pièces (tête du
  réseau, fin simulée) au gain réel ; par type de décision (`d9.py`) et par nombre de joueurs ;
  différences appariées à c130 avec leur intervalle à 95 % (rééchantillonnage par partie).

**La valeur est le levier** si l'entropie croisée au résultat de la valeur par fin simulée est sous
celle du réseau de **0,010** au moins, intervalle sous zéro, sur toutes les positions du juge ; lecture
secondaire : où (types) l'écart est le plus grand. Sinon, le gain des fins simulées n'est pas une
meilleure estimation de la valeur position par position (c'est leur combinaison dans l'arbre, ou de la
variance) et D15 n'a pas lieu.

**D15, la valeur réapprise sur ce que calculent les fins simulées** (ajouté le 31 octobre avec D14,
avant tout calcul ; conditionné à D14). La cible de valeur de D7 et D12 (`--value-mix 0.5`) mêlait la
valeur de racine du professeur au résultat sans jamais être mesurée seule ; celle de lab4 (`--td`)
prenait la valeur à la position suivante de la trajectoire bruitée. Ici :

- **D15a** (données existantes) : c130 réentraîné sur `data/tgt-train` (D12) comme en D12 (`fit.py`,
  4 passages, taux 5e-5 et 1,5e-5, réseau de la fin) avec **`--value-mix 1`** : la valeur de racine du
  professeur (fins simulées, ancrage) pour seule cible de la victoire ;
- **D15b** : les 20 000 parties de D12 rejouées avec `--revalue 4` (mêmes réglages et graine,
  `data/tgt-roll` ; des parties équivalentes, non identiques au bit près : lab14 D8), puis c130
  réentraîné de même avec **`--roll-mix 1`** (victoire et pièces : la valeur par fin simulée) et
  `--roll-mix 0.5` (mêlée au résultat), aux deux taux.

Mesures, pour chaque réseau : sur le **juge** (parties neuves sans bruit), l'entropie croisée de la
valeur au résultat et au Brier, et l'erreur des pièces ; sur `anc-test`, la perte de politique par type
(`d9.py`, qui ne doit pas monter de plus de 0,01 en moyenne). **La valeur s'apprend** si l'entropie
croisée au résultat sur le juge baisse d'au moins **0,005** sous c130 (intervalle sous zéro). Le réseau
retenu : la meilleure entropie croisée sur le juge parmi ceux qui passent. Arènes contre c130, n = 400
des deux côtés puis n = 1, 920 donnes, graine 193001, Vulkan. **Succès** à n = 400 : intervalle
au-dessus de 0. Alors : la cible de valeur par fin simulée entre dans la boucle (le selfplay ciblé
`place=1` l'écrit déjà), avec le garde-fou de Lc0 (mélange au résultat, vérification contre le
résultat à chaque pas).

**D16, le coût de la recherche au jeu** (ajouté le 31 octobre après D14, avant tout calcul ; décision
de l'utilisateur : l'effort va au coût de la recherche, pas à l'apprentissage). L'Expert du jeu
(`lab13-expert`, c130, `n=100,roll=1,mix=0.5,light=1`) tourne sur un cœur de mini (i5-8500T, 6 cœurs,
~1,8 fois plus lent que le PC par cœur : lab7 mesurait 1 496 ms par recherche `n=100,roll=1` sur mini,
le PC 848). Sur le PC, un cœur, `lab13 speed --positions 3 --seed 5` : 627 ms par recherche, 32
évaluations par fin simulée dont la moitié au redéploiement ; avec `place=1`, 361 ms et 17,5
évaluations. Trois leviers, mesurés sur le PC :

1. **le temps par décision** : `lab13 expert-time`, parties entières de 4 et 5 joueurs où tous les
   sièges sont l'Expert du jeu (`Expert::decide`, comme le serveur), moyenne, médiane, p95 et maximum
   par décision (le format de D10, `runs/d10-time.txt` : light=1, 4 j. 819 ms / p95 1 410 / max 1 589 ;
   5 j. 1 206 / 2 061 / 2 365) ;
2. **la force** : arène contre c130 à n = 400, graine 72002, 920 donnes, Vulkan, appariée au témoin
   light=1 de D10 (+4,7 ± 1,5, `runs/d10-light1.tsv`).

- **E1, `place=1` au jeu** (sans code) : `n=100,roll=1,mix=0.5,light=1,place=1`. **Retenu** si la
  différence au témoin est au-dessus de −1,5, borne basse au-dessus de −3,5 (règle de D10 et D11).
- **E2, les évaluations en parallèle** : un oracle du processeur à plusieurs fils (`search::Parallel`),
  qui évalue ensemble les positions qu'une recherche demande en même temps (les feuilles d'un tour du
  halving, puis chaque pas de leurs fins simulées) ; `lab13-expert --threads T`. **Mêmes décisions au
  bit près** qu'avec un fil (test), donc pas d'arène : seul le temps compte, à 1, 2, 4 et 6 fils.
- **E3, racheter de la force avec le temps gagné** : `n=200` et `n=400` avec `place=1,light=1`, arènes
  appariées au témoin ; temps à 6 fils. **Pour le jeu** : la variante la plus forte (intervalle de la
  différence au témoin au-dessus de 0) dont le p95 à 5 joueurs, à 6 fils sur le PC, reste sous le
  p95 actuel à un fil (2 061 ms) ; sinon E1 seul.

Ensuite, selon le résultat : des fins simulées jouées par un réseau léger (politique seule, 64×4,
distillée de c130), ou la tête des propriétaires comme évaluation bon marché.

## Commandes

```sh
target/release/lab13 arena --hero az:net=lab11/runs/p1/c010.bin,n=1 --field az:net=lab11/runs/p1/c000.bin,n=1 \
  --players 2,3,4,5 --deals 920 --seed 131001 --device vulkan --out lab13/runs/d1-c010.tsv
LAB13_CONCURRENT=16 target/release/lab13 selfplay --agent az:net=lab13/nets/c130.bin,n=400,roll=1,mix=0.5 \
  --players 2,3,4,5 --games 1000 --seed 139000000 --threads 20 --device vulkan --out lab13/data/prof
(cd lab13/py && ../../lab3/py/.venv/bin/python agree.py --data ../data/prof --nets ../../lab11/runs/p1/c000.bin …)
```
