# Lab14 : la menace des adversaires en entrée du réseau

Base : copie de **lab13** au commit `0dcf777` (c130, moteurs, options de lab7 et lab10, `py/fit.py`),
sans ses journaux. Règles communes : [labtools/README.md](../labtools/README.md). Résultats :
[journal.md](journal.md).

## Pourquoi

Lab13 a écarté la recette (D3) et le bruit des cibles (D5 à D7) : même avec des cibles ancrées,
reproductibles d'une recherche à l'autre (marge apprenable 0,156), c130 réentraîné hors ligne sur
700 000 positions du professeur ne prédit pas mieux ses cibles sur des parties neuves (1,270 contre
1,271) ; il baisse la perte sur ses propres positions (1,45 → 1,35), sans règle qui se généralise.

Ce que le professeur sait de plus vient de ses fins simulées : la réponse des adversaires au tour
suivant. c130 ne voit que ce que **le peuple qui conquiert** peut faire maintenant (`N_COST`,
`N_NEED`, …, rien hors de sa phase de conquête) ; ce que chaque adversaire pourra prendre à son tour,
il doit le recalculer d'un seul regard. Hypothèse de lab13 : des entrées calculées à la main décrivant
la menace (comme les entrées d'AlphaGo et de KataGo) rendent apprenable ce que le professeur calcule.

## Les nouvelles entrées

16 entrées par région (`NODE` 77 → 93, `lab14/brain/src/encode.rs`), pour chaque joueur relatif
(moi compris), d'après son peuple actif `R` tel qu'il est maintenant (aucune s'il n'en a pas) :

- `X_REACH` : `R` pourrait viser la région à son tour (`Game::block` vide : ni à lui, ni sous la paix,
  ni immunisée, ni hors de portée) ;
- `X_COST` : son coût de conquête (`Game::conquest_cost`), si elle est visable ;
- `X_TAKE` : visable et coût au plus égal aux jetons qu'il aura en début de tour (en main, plus ceux
  sur la carte moins un par région gardée), dé de renfort non compté ;

puis `X_TAKERS` : le nombre d'adversaires qui peuvent la prendre (`X_TAKE`).

**c130x** : c130 dont `node_in` reçoit 16 colonnes de poids nuls. Il calcule exactement ce que
calcule c130 : il remplace c130 partout dans lab14 (lab14 ne lit que des réseaux à 93 entrées).

## Mesures, fixées avant tout calcul

**D8, la menace rend-elle les cibles apprenables ?** Sur le PC (Vulkan ; PyTorch ROCm de
`lab3/py/.venv`).

1. **Contrôle de c130x** : même politique et même valeur que c130 (`lab14 check-net` contre
   `lab13`, aux arrondis près) ; les parties de test de D7 rejouées à l'identique (même graine
   171 000 001 : mêmes parties, même nombre de positions, 1 000 parties).
2. **Les parties** : les 20 000 parties d'apprentissage de D7 refaites avec c130x et les mêmes
   réglages et graines (`n=400,roll=1,mix=0.5,anchor=raw,lambda=0.005`, graine 171 000 000,
   `data/anc-train`) : les mêmes parties que D7, avec les nouvelles entrées.
3. **L'apprentissage** : c130x réentraîné dessus exactement comme en D7 (`py/fit.py`, 4 passages,
   `--value-mix 0.5`, taux 1,5e-4, 5e-5, 1,5e-5), jugé sur `anc-test`. Le témoin est D7 (mêmes
   parties, même recette, sans les entrées) : meilleure perte de politique 1,270, coup de la cible
   56,4 % au départ.
4. **Critère** : **la menace aide** si la meilleure perte de politique sur `anc-test` passe sous
   **1,250** (D7 : 1,270, moins 0,02) avec un coup de la cible à **56,4 %** au moins. Alors : arènes
   du meilleur réseau contre c130x, n = 400 des deux côtés puis n = 1, 920 donnes, graine 182001.
   **Succès** en arène à n = 400 : intervalle au-dessus de 0 ; alors, une boucle au professeur ancré
   avec ces entrées. Sinon : la menace n'est pas ce qui manque, D8 clôt la piste des entrées.

Si les parties de D7 ne se rejouent pas à l'identique (étape 1), D8 continue sur des parties neuves
aux mêmes réglages et graines, et le témoin devient c130x réentraîné sur ces parties les nouvelles
entrées mises à zéro (même recette).
