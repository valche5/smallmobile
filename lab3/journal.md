# Labo 3 : un réseau par nombre de joueurs

Ouvert le 22 octobre 2026, à partir de lab2, figé après sa troisième boucle (`lab2/journal.md` :
la recherche, les coups, le réseau et la boucle viennent de là). Règles communes des labos,
protocole de mesure, machines AWS : `labtools/README.md`.

## Objectif

Le même que lab1 et lab2 : l'IA la plus forte possible sur les cartes générées, à 2, 3, 4 et 5
joueurs, mesurée aussi au temps de réflexion du jeu sur mini, avec les mêmes règles d'honnêteté
(l'Expert ne sait que ce qu'un joueur attentif peut savoir, et ne sait pas contre qui il joue).

## Pourquoi ce labo

Question de l'utilisateur (22 octobre) : **un réseau spécialisé dans un nombre de joueurs
joue-t-il mieux, à ce nombre, qu'un réseau qui apprend tous les nombres à la fois ?** Le jeu
change beaucoup d'une taille à l'autre (à 2 joueurs, lab2 domine les IA du jeu bien plus qu'à 3,
4 ou 5) ; un réseau par taille y consacre toute sa capacité et toutes ses parties, au prix de ce
qu'une taille apprend aux autres.

Décisions de l'utilisateur (22 octobre) : un réseau par nombre de joueurs, en commençant par 4
joueurs ; au départ, le petit réseau (64x4) et la petite fenêtre (250 000 positions) de la
première boucle de lab2 ; départ de zéro, comme lab2.

## Système actuel

Celui de lab2 (voir son journal), sauf :

- **Un nombre de joueurs par réseau** : `lab3/loop.sh P N` joue toutes ses parties
  d'auto-apprentissage et ses arènes à P joueurs ; parties `lab3/data/pP`, réseaux et arènes
  `lab3/runs/pP`. Le réseau garde les entrées et sorties de lab2 (jusqu'à 5 joueurs).
- **Réseau** : 4 couches de 64 ; gen0 (`lab3/nets/gen0.bin`) est le réseau neuf de lab2 (sorties à
  zéro : il joue au hasard).
- **Fenêtre** : toutes les positions jusqu'à 250 000, puis
  250 000 × (1 + 0,4 × ((total / 250 000)^0,75 − 1) / 0,75), comme la première boucle de lab2.
- **Cœurs** : 48 de parties et 16 d'entraînement en même temps (mesures de lab2 pour 64x4).
- **Témoin de l'arène** : l'Expert de lab2 (`lab2:net=lab2/nets/….bin,n=…`, `lab2/brain`), tel
  qu'il joue dans le jeu ; plus de témoin lab1.
- Retiré : `size.py` (mesure des tailles de réseau de lab2).

## Mesure

Le protocole commun garde le format duplicate à 2, 3, 4 et 5 joueurs ; un réseau spécialisé ne
joue qu'à sa taille, et se compare à lab2 à cette taille seulement :

- à chaque pas, arènes à P joueurs face à Difficile (graine 1002) et Normale (1003), 460 donnes ;
  les réseaux de lab2 rejoués sur ces mêmes donnes pour comparer ;
- en fin de boucle (`lab3/runs/final.sh`), 920 donnes à P joueurs (graine 1004), n = 200 des deux
  côtés : face au c020 64x4 de lab2 (même taille de réseau, même fenêtre, autant de parties, mais
  à toutes les tailles : la comparaison directe) et face au meilleur réseau de lab2 ; courbe des
  réseaux, ceux de lab2 compris, sur les mêmes positions de validation (`lab3/py/curve.py`).

## Plan

1. Boucle à 4 joueurs depuis gen0, 20 pas (comme la première boucle de lab2), puis mesures de
   fin. Lancée le 22 octobre, après la troisième boucle de lab2.
2. Selon le résultat : les autres nombres de joueurs, un réseau et une fenêtre plus grands, un
   Expert du jeu qui choisit son réseau d'après le nombre de joueurs.

## Résultats
