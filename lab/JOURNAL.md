# Journal de l'IA hors jeu

Objectif : l'IA de SmallMobile la plus forte possible, sans limite de temps ni de calcul, hors du
jeu (elle peut réfléchir aussi longtemps qu'elle veut). Couvre toutes les tables : 2 à 5 joueurs,
cartes générées.

Règle d'honnêteté : l'IA sait tout ce qu'un joueur humain attentif peut savoir, rien de plus.
- Elle ne lit ni l'ordre caché des pioches ni les dés à venir (`know::determinize` les retire).
- Elle connaît l'ordre des cartes rendues sous la pioche quand un peuple disparaît (c'est public).
- Elle connaît les pièces de tous : cachées dans l'interface, elles se recalculent exactement à
  partir de ce qui est annoncé (5 au départ, prix et pièces de chaque choix de peuple, gain de
  chaque tour). Un humain qui compte les connaît aussi.

Méthode visée (voir le plan) : recherche arborescente (MCTS) sur des mondes possibles, guidée par
un réseau de neurones sur le graphe des régions, entraîné par auto-apprentissage (style AlphaZero).
Simulation en Rust (`lab/`), entraînement en Python (PyTorch).

Commandes (depuis la racine) :

```
cargo build --release -p smallmobile-lab
./target/release/lab bench --agent normal
./target/release/lab arena --hero hard --field normal --deals 1000
cargo test -p smallmobile-lab
```

Lecture de l'arène : format « duplicate ». Chaque donne (carte, pioches, dés) est rejouée en
mettant le joueur testé à chaque place, face au reste de la table. « Attendu » est le taux de
victoire à force égale (1/n). L'écart à l'attendu, toutes tables confondues, résume la force.

---

## 2026-10-17 — Étape 0 : banc d'essai

Fait :
- crate `lab/` dans l'espace de travail, sans toucher au noyau ;
- `table` : les mêmes cartes que le jeu (`mapgen::generate`), résultat d'une partie (pièces, puis
  jetons sur la carte pour départager, comme `ranking` dans l'interface) ;
- `know` : suivi des cartes connues au fond des pioches, et `determinize` (monde possible : ordre
  caché des pioches mélangé, dés réinitialisés) ;
- `moves` : les coups regroupés pour la recherche. Une décision = un coup : choix du peuple,
  déclin, chaque conquête / dé / abandon / sort / dragon, fin des conquêtes ; la fin de tour est un
  seul coup (héros, forteresse, campements posés selon la menace, puis un placement parmi les 6
  plans du noyau, paix des Diplomates, déclin des Robustes) ; la défense : 2 plans ;
- `agent` : IA du jeu (Facile, Normal, Difficile) vues à travers un monde possible, et un joueur au
  hasard ; `arena` : parties en parallèle, format duplicate ;
- tests : chaque coup proposé est jouable ; le fond connu des pioches est exact.

Mesures (un cœur, carte générée comprise) :

| table | Normal, ms/partie | Difficile, ms/partie | coups/partie |
| --- | --- | --- | --- |
| 2 joueurs | 0,85 | 114 | ~110 |
| 3 joueurs | 1,3 | 317 | ~160 |
| 4 joueurs | 1,8 | 565 | ~200 |
| 5 joueurs | 2,3 | 839 | ~225 |

Sur 20 cœurs : de l'ordre de 10 000 parties Normal par seconde. Le moteur n'est pas le goulot.

Étalons (taux de victoire du premier contre une table du second, attendu à force égale entre
parenthèses) :

| | 2 j. | 3 j. | 4 j. | 5 j. | écart total |
| --- | --- | --- | --- | --- | --- |
| Normal contre hasard | 100 % | 100 % | 98 % | 100 % | +71 pts |
| Normal contre Facile | 87 % (50) | 68 % (33) | 64 % (25) | 60 % (20) | +38 pts |
| Difficile contre Normal | 85 % (50) | 54 % (33) | 42 % (25) | 32 % (20) | +19 pts |
| Normal contre Difficile | 15 % (50) | 21 % (33) | 19 % (25) | 16 % (20) | −11 pts |
| Normal contre Normal | 49 % | 35 % | 25 % | 20 % | +0,4 pt (contrôle) |
| Difficile qui voit tout contre Difficile | 55 % | 29 % | 25 % | 22 % | +0,4 pt |

(1000 donnes, sauf hasard 400 et Normal contre Difficile / voit tout 400 ; ± 2 à 4 points.)

Constats :
- le contrôle Normal contre Normal tombe pile sur l'attendu : l'arène est juste ;
- l'IA Difficile est nettement plus forte que Normal à 2 joueurs, beaucoup moins à 5 ;
- voir les pioches et les dés n'apporte rien de mesurable à l'IA Difficile : la version honnête
  sert d'étalon sans perte.

Prochaine étape : recherche (MCTS) sur mondes possibles, sans réseau, avec l'IA Normal pour finir
les parties simulées. Premier but : battre nettement l'IA Difficile.
