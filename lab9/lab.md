# Lab9 : le réseau hybride de ResTNet, dans le cadre de lab8

Le cadre est celui de lab8 (figé) : plans de tour, décodeur, recherche sur plans, moteur Vulkan, selfplay.
Seul l'encodeur change. Règles communes : [labtools/README.md](../labtools/README.md). Résultats :
[journal.md](journal.md).

## Pourquoi

- lab8 a montré qu'un transformer neuf imite c130 mieux que son propre réseau (i2, 256 × 6), et que la
  recherche sur plans en tire parti : +3,6 ± 1,5 points contre c130 à n = 400 en 134 ms par décision.
  Mais une lecture coûte 13 fois le calcul de c130. Le petit transformer (s1, 128 × 4) est 3,4 fois
  plus rapide mais perd 4 points sans recherche (`lab8/journal.md`).
- ResTNet (Ju, Wu, Shih, Wu, 2024, arXiv 2410.05347) alterne des blocs résiduels convolutifs
  (**R**, le bloc d'AlphaZero : le local) et des blocs transformer (**T** : le global). Au Go 9 × 9
  (6 blocs, contre KataGo) : RRTRRT 60,8 %, tout R 54,6 %, tout T 39,9 %, T en premier (TRRRRT)
  43,9 %. Même ordre de gain au Go 19 × 19 (+7 points) et à Hex (+8). Leur conclusion : extraire le
  local avant de mêler le global ; le tout-transformer est la forme la moins bonne.
- Nos deux réseaux sont les deux extrêmes : c130 est un « tout R » sur graphe (7 blocs, moyenne des
  voisines à la place de la convolution 3 × 3) ; i2 est un « tout T » (la carte n'y entre que par
  le biais d'attention appris selon la distance).

## Décisions de l'utilisateur (27 octobre)

- Un nouveau labo avec l'architecture de ResTNet ; lab8 figé.
- Les parties d'imitation de lab8 sont **relues telles quelles** (`lab8/data/c130`, `lab8/data/c130b` :
  même format, aucune conversion), par exception à la règle des données propres à chaque labo :
  mêmes plans de validation qu'i1, i2 et s1, comparaison directe.
- Deux tailles en phase 2 : **256 × 6** (contre i2, à taille égale) et **128 × 6** (contre s1 : la qualité
  d'i2 à moindre coût ?).

## Le réseau

Tout ce qui n'est pas l'encodeur reste celui de lab8 : jetons (régions, joueurs, marché, partie),
têtes de valeur, décodeur de plans.

**Bloc T** : celui de lab8 (attention avec biais par relation, puis MLP 4d, normalisations avant).

**Bloc R** (neuf) : le bloc d'AlphaZero porté sur la carte, deux convolutions de graphe et un résiduel,
comme le bloc de c130 :

```text
a = LN(x)
z_i = gelu(W1 · [a_i, moyenne des a_j voisins, a_partie])      (3d → d)
z  = LN(z)
x_i += W2 · [z_i, moyenne des z_j voisins]                      (2d → d)
```

Les voisines sont celles de la carte (`adj`, déjà dans les parties). Les jetons hors carte (joueurs,
marché, partie) n'ont pas de voisines (moyenne nulle) : ils passent par les mêmes poids, avec le jeton
de la partie comme contexte. Coût par jeton : 5d² multiplications, contre ~12d² et l'attention pour un
bloc T. RRTRRT coûte donc ~0,6 fois le calcul de TTTTTT, avec deux attentions au lieu de six.

**Disposition** : RRTRRT, la meilleure de l'article à 6 blocs. Elle est lue dans les poids (une couche
avec `attn` est un bloc T, une couche avec `conv1` un bloc R) : les réseaux de lab8 se lisent comme
TTTTTT, et servent de témoins dans les arènes de lab9.

## Phases et critères, fixés avant tout calcul

Chaque commande longue a un délai maximal (AGENTS.md).

| Phase | Travail | Critère pour continuer |
| --- | --- | --- |
| 0, base | `smallmobile-lab9` et `smallmobile-brain9`, copie élaguée de lab8 ; le bloc R dans `py/model.py` et la référence Rust ; disposition lue dans les poids | Mêmes sorties en Rust et en PyTorch (réseau hybride au hasard) ; i2 lu par lab9 donne les mêmes sorties que dans lab8 ; débit de l'apprentissage mesuré |
| 1, données | Celles de lab8 (200 000 parties, 16 millions de plans) | — |
| 2, imitation | **h1** RRTRRT 256 × 6, 8 têtes, recette d'i2 (lots de 256, 150 000 lots, victoire finale sur un plan sur quatre) ; **h2** RRTRRT 128 × 6, 4 têtes, recette de s1 (lots de 512, 75 000 lots). Validation tous les 5 000 lots | h1 face à i2 au même lot (100 000 : politique 1,166, valeur 1,019) ; h2 face à s1 au lot 75 000 (1,189, 1,031). Question principale : **h2 atteint-il i2** (politique et valeur au moins aussi bonnes) ? Puis parité des politiques seules des meilleurs (920 donnes, graine 82001, contre c130 à n = 1), appariée avec i2 et s1 |
| 3, moteur | Le bloc R en Vulkan (moyenne des voisines, puis `gemm`) ; `check-gpu`, `bench-gpu --slots true` | Écarts du niveau de ceux d'i2 ; lectures par seconde comparées à i2 (31 700) et s1 (109 200) en lots de 256 |
| 5, plans | La recherche sur plans contre c130 à n = 400, 920 donnes, graine 82002 : (a) k = 16, m = 4 comme i2 ; (b) au temps par décision d'i2 (134 ms), k et m choisis d'après la vitesse mesurée avant l'arène. Appariée avec i2 (`lab8/runs/i2/phase5-100000.tsv`) et lab7 (`lab8/runs/i1/phase5-lab7.tsv`) | Mieux qu'i2 en apparié ; la cible de fond reste la différence appariée avec lab7 au-dessus de 0 |
| 6, selfplay | Celui de lab8 (`selfplay`, `train.py --imitation`), si la phase 5 montre que le calcul de la recherche paie ; la cible d'abord mesurée (plus de candidats, `beta`), lab8 l'ayant trouvée presque égale à l'a priori | Gain sur la phase 5 |

Si aucun hybride n'atteint i2 à moindre coût, on s'arrête pour décider ensemble.
