# Labo 13 : pourquoi c130 n'apprend pas de son professeur

Ouvert le 30 octobre 2026 à la demande de l'utilisateur, après l'arrêt de la boucle p1 de lab11,
depuis **lab11** au commit `a9de062`. Plan et protocole : [lab.md](lab.md).

## Base

- Code de lab11 renommé (`smallmobile-lab13`, `smallmobile-brain13`, programme `lab13`, variables
  `LAB13_`), sans la boucle (`loop.sh`, `p1.sh`), les données ni les résultats.
- `py/agree.py` : les mesures de validation de `train.py` et l'accord avec le coup préféré de la
  cible, pour plusieurs réseaux, aux positions où la cible s'écarte de la référence et ailleurs.

## D1 : la politique seule (graine 131001, 920 donnes, n = 1 des deux côtés, 30 octobre)

| Réseau testé, contre c130 | Score | 2 j. | 3 j. | 4 j. | 5 j. |
| --- | --- | --- | --- | --- | --- |
| c010 (huit pas de professeur) | +1,4 ± 1,5 | +1,1 | +2,9 | +1,4 | +0,1 |
| c002 (un pas sur les parties de c130) | +0,1 ± 1,4 | +0,8 | −0,2 | +1,4 | −1,4 |

c010 − c002 : +1,3 ± 2,0, indécis. La politique seule de c010 ne bat pas c130 de façon établie.
`lab13/runs/d1-{c010,c002}.tsv` (2 s chacune).

## D2 : l'accord avec le professeur sur des parties neuves (30 octobre)

1 000 parties du professeur avec c130 (`data/prof`, 404 s, 34 708 positions) et 1 000 de la recette
ordinaire (`data/ord`, 16 s, 34 862 positions). Le coup préféré de la cible diffère de celui de c130 à
**53,5 %** des positions du professeur et à **49,4 %** de celles de la recherche ordinaire à n = 400.

| Réseau | Politique (prof) | Valeur (prof) | Cible (prof) | Écarts (prof) | Accords (prof) | Écarts (ord) | Accords (ord) | = c130 (prof) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c000 = c130 | 1,307 | 1,011 | 46,5 % | 0 % | 100 % | 0 % | 100 % | 100 % |
| c002 | 1,316 | 1,012 | 46,0 % | 8,8 % | 88,8 % | 9,9 % | 88,8 % | 82,7 % |
| c005 | 1,314 | 1,013 | 45,8 % | 12,8 % | 83,8 % | 13,6 % | 83,5 % | 75,3 % |
| c008 | 1,320 | 1,015 | 45,8 % | 16,0 % | 80,1 % | 15,8 % | 79,2 % | 69,9 % |
| c010 | 1,321 | 1,016 | 45,8 % | 16,3 % | 79,9 % | 16,3 % | 79,0 % | 69,2 % |
| c011 | 1,321 | 1,015 | 45,9 % | 16,4 % | 79,7 % | 16,3 % | 78,6 % | 69,0 % |

« Écarts » : positions où le coup préféré de la cible diffère de celui de c130 ; part des coups du
réseau égaux à celui de la cible. « Accords » : les autres positions.

- Le critère de `lab.md` est rempli à la lettre (aux écarts, c010 16,3 % contre 8,8 % pour c002), mais
  **le témoin le dément** : sur les parties de la recherche ordinaire, que le professeur n'a jamais
  jouées, les réseaux gagnent exactement autant aux écarts (9,9 % → 16,3 %) et perdent autant aux
  accords. Ce n'est pas le professeur qu'ils apprennent : ils s'éloignent de c130 (69 % de coups
  communs à c011), et ce qu'ils gagnent d'un côté, ils le perdent de l'autre. Le critère aurait dû
  comparer les deux jeux de parties ; c'était le rôle de `data/ord`.
- **Aucune mesure ne s'améliore sur les cibles du professeur** : coup de la cible 46,5 % → 45,9 %,
  politique 1,307 → 1,321, valeur 1,011 → 1,015. Neuf cycles de ses parties (~3,9 millions de
  positions) n'ont pas appris au réseau à mieux le prédire sur des parties neuves.

**Conclusion** : ce n'est pas la valeur ou la recherche qui masquent un progrès de la politique. Avec
la recette de la boucle (taux 1,5e-4 en un cycle, ~4 passages par position), c130 ne retient rien de
prédictible des cibles du professeur. Deux causes restent : la recette (des cibles très bruitées,
qu'un apprentissage court ne moyenne pas) ou la capacité de c130 (ce que calcule le professeur n'est
pas lisible d'un seul regard). Elles se séparent hors ligne, sur les parties de lab11 déjà jouées.

## D3 : la recette (30 octobre)

c130 réentraîné hors ligne (`py/fit.py`) sur les parties du professeur de lab11 encore gardées (c004 à
c011 : 3 446 887 positions, 99 998 parties), 4 passages moyens (26 928 lots de 512), ~6,5 min par taux.
`lab13/runs/d3-lr*.txt`.

| Taux maximal | Apprentissage (politique, fin) | `prof` : politique (meilleure / fin) | `prof` : coup de la cible (fin) | `ord` : politique (fin) |
| --- | --- | --- | --- | --- |
| c130, départ | | 1,307 | 46,5 % | 1,248 |
| 1,5e-4 | 1,458 | 1,319 / 1,319 | 46,0 % | 1,295 |
| 5e-5 | 1,469 | 1,313 / 1,314 | 46,5 % | 1,288 |
| 1,5e-5 | 1,479 | 1,313 / 1,313 | 46,8 % | 1,284 |

- **Le critère n'est pas atteint, à aucun taux** : la perte de politique sur les parties neuves du
  professeur ne descend jamais sous celle de c130 (1,307). Elle monte dès les 2 000 premiers lots,
  même au taux le plus bas, puis ne bouge presque plus ; la valeur fait de même (1,011 → 1,011 à 1,016).
- Le réseau apprend ses lots (1,50 → 1,46 au taux de la boucle), mais ce qu'il en tire ne se transpose
  pas aux cibles du professeur sur d'autres parties, et l'éloigne de la recherche ordinaire (`ord`
  1,248 → 1,284 à 1,295).
- **Réserve** : les cibles apprises viennent du professeur mené par les réseaux de la boucle (c003 à
  c010, qui jouent déjà 70 à 80 % des coups de c130) ; celles du juge, du professeur mené par c130. Une
  part du décalage peut venir de là. Le test sans ce biais demande des parties du professeur mené par
  c130 pour apprendre (~10 h pour autant de positions).

## D5 : le plafond des cibles (graine 151001, 1 500 positions, 4 recherches chacune, 30 octobre)

`lab13 consistency` sur 1 500 positions rejouées de `data/ord` et de `data/prof` (c130, bruit de Gumbel
à la racine comme dans le selfplay). Recherche ordinaire 40 s ; professeur 1 622 s (20 fils).
`lab13/runs/d5-*.{txt,tsv}`.

| Recherche | Même coup entre deux recherches | Coup de l'a priori de c130 | CE à l'a priori | CE à la moyenne des 3 autres | Entropie de la cible |
| --- | --- | --- | --- | --- | --- |
| ordinaire, `n=400` | 66,2 % | 49,4 % | 1,264 | 1,045 | 0,410 |
| professeur, `n=400,roll=1,mix=0.5` | 55,4 % | 46,9 % | 1,294 | 1,290 | 0,369 |

Par nombre de joueurs, le professeur s'accorde avec lui-même à 63 % à 2 joueurs, 52 à 55 % à 3 à 5.

- **Le seuil de `lab.md` n'est pas franchi** (8,5 points au-dessus de l'a priori, plus que 5) : les coups
  du professeur ne sont pas du bruit. Mais, comme l'a fait remarquer l'utilisateur, ce seuil ne suffisait
  pas à conclure : un professeur qui gagne ~9 points ne peut pas n'être que du bruit ; la question est
  ce qu'une cible porte, position par position.
- **En perte, ses cibles ne portent presque rien d'apprenable** : la moyenne de trois autres recherches
  du professeur ne prédit pas mieux une de ses cibles que l'a priori de c130 (1,290 contre 1,294).
  Ses cibles sont **trop sûres d'elles** : plus pointues que celles de la recherche ordinaire (entropie
  0,369 contre 0,410) et pourtant moins reproductibles (55 % contre 66 %). La recherche ordinaire, elle,
  laisse une vraie marge (1,264 → 1,045). C'est ce que voyaient D2 et D3 : l'entropie croisée sur une
  cible aussi bruitée ne peut presque rien apprendre, quelle que soit la recette.

**Leçons nettes** : positions où les 4 recherches s'accordent et contredisent l'a priori de c130.

| Leçons de… | Positions | c002 s'écarte de c130 / joue le coup | c010 s'écarte de c130 / joue le coup |
| --- | --- | --- | --- |
| professeur (`prof`) | 148 (9,9 %) | 23,0 % / 12,8 % | 36,5 % / 20,3 % |
| recherche ordinaire (`ord`) | 231 (15,4 %) | 27,3 % / 13,0 % | 44,2 % / 19,9 % |

c010 gagne autant de leçons de la recherche ordinaire, qu'il n'a jamais apprises, que de leçons du
professeur (+6,9 et +7,5 points sur c002, ± ~6,5 sur si peu de positions) : là encore de la dérive, aux
positions où l'a priori de c130 hésite, pas un apprentissage du professeur.

**Conclusion de D5** : le professeur choisit mieux que c130, mais ses cibles de politique, une recherche
à la fois, sont trop bruitées et trop pointues pour qu'un réseau en extraie quelque chose par entropie
croisée. Ce n'est ni la recette (D3) ni, à ce stade, la capacité : c'est la cible.

## D6 : des cibles ancrées (mêmes positions que D5, 30 octobre)

`anchor=raw,lambda=0.005` (lab10) ; recherche ordinaire 38 s, professeur 1 709 s.
`lab13/runs/d6-*.{txt,tsv}`.

| Recherche | Même coup entre recherches | Coup de l'a priori | CE à l'a priori | CE à la moyenne des autres | **Marge** | Entropie |
| --- | --- | --- | --- | --- | --- | --- |
| ordinaire (D5) | 66,2 % | 49,4 % | 1,264 | 1,045 | 0,219 | 0,410 |
| ordinaire, ancrée | 68,9 % | 59,7 % | 1,250 | 1,126 | 0,124 | 0,938 |
| professeur (D5) | 55,4 % | 46,9 % | 1,294 | 1,290 | 0,004 | 0,369 |
| **professeur, ancré** | **63,6 %** | 54,6 % | 1,267 | 1,111 | **0,156** | 0,842 |

Par nombre de joueurs, la marge du professeur ancré va de 0,123 (2 joueurs) à 0,175 (3 joueurs).

**Le critère est atteint** (0,156 ≥ 0,10) : ancrées, les cibles du professeur deviennent apprenables.
Elles sont deux fois moins pointues (entropie 0,84 contre 0,37) et plus reproductibles (63,6 % contre
55,4 %) : l'ancrage garde la politique améliorée près de l'a priori là où le professeur hésite, et ne
l'en écarte que là où ses valeurs le justifient. La marge est même sous-estimée : la moyenne de trois
recherches est elle-même bruitée.

Le professeur de lab11 était sans ancrage parce que l'ancrage n'ajoutait rien d'établi au jeu (lab10,
phase 5) ; c'est à l'apprentissage qu'il sert, comme le rapporte Diplodocus (réseaux réentraînés avec
piKL, presque rien à l'inférence seule).

## D7 : des cibles apprenables font-elles un réseau plus fort ? (30 octobre)

20 000 parties du professeur ancré mené par c130 (692 973 positions, 7 644 s) et 1 000 neuves pour
juger (`anc-test`). c130 réentraîné dessus (`py/fit.py`, 4 passages, `--value-mix 0.5`, 5 413 lots de
512, ~76 s par taux). `lab13/runs/d7-*.{txt,tsv}`.

| Taux maximal | Apprentissage (politique, fin) | `anc-test` : politique (meilleure / fin) | `anc-test` : coup de la cible (fin) | `prof` : politique (fin) | `ord` : politique (fin) |
| --- | --- | --- | --- | --- | --- |
| c130, départ | | 1,271 | 56,4 % | 1,307 | 1,248 |
| 1,5e-4 | 1,352 | 1,271 (départ) / 1,286 | 53,7 % | 1,332 | 1,283 |
| 5e-5 | | 1,271 / 1,272 | 55,1 % | | |
| 1,5e-5 | | **1,270** / 1,270 | 55,8 % | | |

Arènes du réseau de 1,5e-5 contre c130, 920 donnes, graine 172001 :

| Arène | Score (tailles à poids égaux) |
| --- | --- |
| n = 400 des deux côtés | **+0,1 ± 1,5** |
| n = 1 (la politique seule) | +1,2 ± 1,4 |

**Échec** : l'intervalle à n = 400 contient 0. Le résultat était acquis avant les arènes : le réseau
retenu ne s'écarte de c130 que d'un millième de perte, il le joue presque à l'identique.

Comme en D3 avec les cibles sans ancrage, l'apprentissage baisse la perte sur ses propres positions
(1,45 → 1,35 au taux fort) mais jamais sur des parties neuves, où la précision recule (56,4 % → 53,7 %
à 55,8 %). La marge de D6 (0,156) se mesurait position par position, contre la moyenne des autres
recherches **sur la même position** : elle dit qu'une cible reproductible existe, pas qu'un réseau sait
la généraliser à des positions qu'il n'a pas vues. Avec 700 000 positions, ce que le professeur ancré
sait de plus que c130 ne passe pas dans le réseau.

**Conclusion de D7** : rendre les cibles reproductibles ne suffit pas. Pas de boucle à professeur ancré
sur cette seule base.

## L'Expert du jeu passe aux fins simulées (30 octobre)

Hors du diagnostic, à la demande de l'utilisateur : le seul progrès établi depuis lab3, les fins
simulées à la décision (lab7, +5,4 ± 1,5 à n = 100, confirmé), entre dans le jeu. `lab13-expert` prend
les réglages de la recherche (`--search roll=1,mix=0.5`, `search::Config::parse`) ; variante `lab13` de
`src/shared/experts.ts` (c130, n = 100), désormais proposée par défaut.

Temps de calcul par décision dans le processus de l'Expert, sur un cœur du PC, parties entières où
tous les sièges sont l'Expert (une partie par cas) :

| Joueurs | lab3 (n = 400) : moyenne / p95 / max | lab13 (n = 100, fins simulées) : moyenne / p95 / max |
| --- | --- | --- |
| 2 | 63 ms / 178 / 372 | 359 ms / 915 / 2 183 |
| 4 | 91 ms / 201 / 462 | 1 768 ms / 7 207 / 9 681 |
| 5 | 112 ms / 292 / 484 | 2 239 ms / 7 810 / 13 314 |

n = 400 avec fins simulées (+8,9) coûterait environ quatre fois plus, jusqu'à ~50 s au pire à 5
joueurs, trop près de la limite de 60 s du serveur : écarté.

## D9 : par type de décision (30 octobre)

Les apprentissages de D7 refaits à l'identique (fin : 1,286 à 1,5e-4, 1,272 à 5e-5, comme D7), réseau
de la fin gardé (`runs/d9-lr*-last.bin`). `py/d9.py`, `runs/d9.txt`. Excès : perte de politique moins
entropie de la cible (ce qui sépare c130 de la cible, bruit de la cible compris). Écarts à c130 ± une
erreur type.

| Type | Positions | Perte de c130 | Excès de c130 (part) | Entropie | Apprises : 1,5e-4 / 5e-5 | Neuves : 1,5e-4 / 5e-5 |
| --- | --- | --- | --- | --- | --- | --- |
| peuple | 3,2 % | 0,721 | 0,511 (4 %) | 0,210 | −0,131 / −0,068 | **+0,052** ± 0,013 / +0,022 ± 0,007 |
| déclin possible | 13,4 % | 1,365 | **0,657** (22 %) | 0,708 | −0,105 / −0,046 | **+0,041** ± 0,006 / +0,005 ± 0,005 |
| conquête | 30,6 % | 1,169 | 0,424 (32 %) | 0,746 | −0,055 / −0,026 | **+0,029** ± 0,003 / +0,008 ± 0,002 |
| pouvoirs | 2,7 % | 1,690 | 0,404 (3 %) | 1,286 | −0,022 / −0,017 | −0,002 ± 0,004 / −0,011 ± 0,003 |
| jetons | 48,2 % | 1,327 | 0,321 (38 %) | 1,006 | −0,013 / −0,009 | −0,004 ± 0,001 / −0,006 ± 0,001 |
| défense | 1,9 % | 1,181 | 0,259 (1 %) | 0,921 | −0,017 / −0,013 | −0,002 ± 0,004 / −0,004 ± 0,003 |
| tous | | 1,271 | 0,404 | 0,867 | −0,042 / −0,022 | +0,014 / +0,000 |

(Neuves : toutes les positions de `anc-test` ; apprises : 40 000 de `anc-train`, graine 9.)

- **La moyenne cachait deux mouvements opposés.** Sur les décisions qui engagent la partie (peuple,
  déclin, conquête : 47 % des positions, 58 % de l'excès), l'apprentissage apprend beaucoup les parties
  vues (jusqu'à −0,13) et **empire** sur les neuves (jusqu'à +0,05) : il apprend ces cas par cœur. Sur
  les jetons, les pouvoirs et la défense, il gagne un peu, aussi sur les neuves (−0,004 à −0,011).
- **L'excès par position est le plus grand au déclin (0,66) et au choix du peuple (0,51)**, où le
  professeur est très sûr de lui (entropie 0,21 au peuple) : c'est là qu'il diffère le plus de c130,
  et là que l'apprentissage généralise le moins. Avec 20 000 parties, il n'y a que ~22 000 choix de
  peuple et ~93 000 positions de déclin possible, pour des situations qui se répètent peu.
- Les jetons (48 % des positions) ont la cible la plus incertaine (entropie 1,01) et l'excès par
  position le plus faible : la recherche y départage des coups presque équivalents.

**Conclusion de D9** : ce n'est pas une incapacité générale du réseau à apprendre, c'est un défaut de
**généralisation sur les décisions stratégiques**, qui manquent d'exemples. Le remède est dans les
données (beaucoup plus de ces décisions) ou dans ce qui les rend généralisables, pas dans une autre
cible de politique. Le budget de recherche peut aussi s'y concentrer.

Correction d'une piste proposée avant D9 : la tête « prochain tour » existe déjà. La tête des
propriétaires apprend à qui appartient chaque région **au début du prochain tour** du joueur qui agit
(`selfplay::owners`, poids 1,5 dans `train.py`), et non à la fin de la partie.

## La tête des propriétaires sait la riposte (30 octobre)

Question de l'utilisateur : la tête « prochain tour » (propriétaires au début du prochain tour du joueur
qui agit) ne sert qu'à l'apprentissage ; les moteurs de jeu ne la calculent pas. Ce qu'elle vaut, c130
sur `anc-test`, régions à moi au moment de la décision (`runs/owners-check.txt`) :

| Décisions | Régions | Perdues avant mon prochain tour | AUC | Prédit perdu | Perdues dans le décile le plus menacé |
| --- | --- | --- | --- | --- | --- |
| toutes | 199 788 | 19,1 % | **0,848** | 19,0 % | **65,9 %** |
| déclin possible | 23 922 | 29,3 % | 0,849 | 28,7 % | 85,8 % |
| conquête | 52 653 | 20,2 % | 0,830 | 20,1 % | 65,6 % |
| jetons | 110 221 | 16,8 % | 0,849 | 16,8 % | 60,1 % |

Bien classée et calibrée : le tronc de c130 contient ce que les fins simulées vont chercher (qui me
reprend quoi d'ici mon tour), que sa politique et sa valeur exploitent mal. Piste notée : s'en servir
en jeu (moteurs à compléter), par exemple comme évaluation bon marché à la place d'une part des fins
simulées.
