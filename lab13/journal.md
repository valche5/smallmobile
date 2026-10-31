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

## D10 : la recherche courte hors des décisions stratégiques (30 octobre)

`light=N` : au redéploiement et à la défense, N itérations sans fins simulées. Temps de l'Expert du
jeu, un cœur du PC, parties entières (`runs/d10-time.txt`) ; arènes contre c130 à n = 400, graine
72002, 920 donnes, Vulkan, ~8 min chacune ; témoin `lab7/runs/a1-confirm.tsv` (mêmes donnes, sans
`light`).

| Joueur | 4 j. : moyenne / max | 5 j. : moyenne / max | Score contre c130 | Différence au témoin |
| --- | --- | --- | --- | --- |
| `n=100,roll=1,mix=0.5` (Expert actuel, témoin) | 1,8 s / 9,7 s | 2,2 s / 13,3 s | +5,4 ± 1,5 | |
| `…,light=16` | 0,8 s / 1,6 s | 1,25 s / 2,4 s | +4,3 ± 1,5 | −1,1 ± 2,1 |
| `…,light=1` | 0,8 s / 1,6 s | 1,2 s / 2,4 s | **+4,7 ± 1,5** | **−0,6 ± 2,1** |

- Les deux passent le critère (au-dessus de −1,5, borne basse au-dessus de −3,5) et coûtent pareil :
  tout le temps restant est au peuple, au déclin et à la conquête. Le pire cas est divisé par 5,5, la
  moyenne par ~2 : les longues décisions étaient les redéploiements, une recherche entière avec fins
  simulées par jeton posé.
- **Retenu : `light=1`**, meilleur score des deux. L'Expert du jeu (`src/shared/experts.ts`) passe à
  `n=100,roll=1,mix=0.5,light=1`.
- Comme le laissait penser D9, presque toute la force des fins simulées est dans les décisions
  stratégiques ; au redéploiement et à la défense, le coup préféré du réseau suffit.

## D11 : un selfplay du professeur plus rapide (30-31 octobre)

**Où va le calcul** (300 parties du professeur de D7, `--full` absent, PC) : GPU à 98 %, processeur à
34 % ; les fins simulées font l'essentiel des évaluations (25 par fin simulée), dont **55 % au
redéploiement**, 40 % à la conquête.

**`place=1`** (jetons des fins simulées posés par `Game::full_deployment`) : 12,5 évaluations par fin
simulée (conquête 81 %, redéploiement 9 %), **1,9 fois plus vite** (82 s contre 155 pour 300 parties).
Arènes contre c130 à n = 400, graine 107001, 920 donnes, appariées à lab10 `p5c` (même professeur sans
`place` : +9,7 ± 1,6, 54 min) :

| Professeur | Score | Différence à `p5c` | Temps par décision (moyenne / p95) | Durée |
| --- | --- | --- | --- | --- |
| `n=400,roll=1,mix=0.5,anchor=raw,lambda=0.005` (`p5c`) | +9,7 ± 1,6 | | 6,1 s / 23,2 s | 54 min |
| … `place=1` | **+8,4 ± 1,5** | **−1,3 ± 2,1** | 2,7 s / 10,2 s | 25 min |
| … `place=1`, n = 200 | +5,7 ± 1,6 | −4,0 ± 2,2 | 1,3 s / 5,2 s | 14 min |

**`place=1` à n = 400 est retenu** (au-dessus de −1,5, borne basse −3,4) ; à n = 200, le professeur perd
4 points : écarté.

**Le ciblage** (`--full 1,1,0.125,0` : tout choix de peuple et tout déclin possible cherchés à fond, une
conquête sur huit, rien au redéploiement ni à la défense), avec `place=1`, 300 parties :

| Recette | Temps | Positions | Peuple | Déclin possible | Conquête |
| --- | --- | --- | --- | --- | --- |
| D7 (12,5 % de tous les coups) | 155 s | 10 353 | 344 | 1 406 | 3 111 |
| `place=1`, `--full 1,1,0.125,0` | 134 s | 16 879 | **2 587** | **11 190** | 3 102 |

Au même débit de positions (le GPU reste la limite), **8 à 9 fois plus de choix de peuple et de déclins
par heure**, autant de conquêtes.

## D12 : des parties ciblées font-elles généraliser les décisions stratégiques ? (31 octobre)

20 000 parties ciblées du professeur `place=1` en 6 799 s (**164 positions/s**, contre 90 en D7) :
1 114 401 positions, dont ~174 000 choix de peuple et ~729 000 déclins possibles (15,6 % et 65,4 %),
contre ~22 000 et ~93 000 en D7. Apprentissages (~110 s par taux), `runs/d12-types.txt` :

| Type | Neuves (`anc-test`) : 1,5e-4 / 5e-5 / 1,5e-5 | D9 (parties de D7) : 1,5e-4 / 5e-5 | Apprises : 1,5e-4 / 1,5e-5 |
| --- | --- | --- | --- |
| peuple | +0,015 / +0,006 / +0,010 ± 0,006 | +0,052 / +0,022 | −0,063 / −0,014 |
| déclin possible | +0,019 / +0,003 / **−0,001** ± 0,005 | +0,041 / +0,005 | −0,063 / −0,015 |
| conquête | +0,020 / +0,007 / +0,004 ± 0,002 | +0,029 / +0,008 | −0,017 / +0,000 |
| jetons (absents des parties) | +0,011 / +0,005 / +0,002 | −0,004 / −0,006 | |

(Écarts de perte de politique à c130.) Réseau retenu (perte stratégique la plus basse sur `anc-test`) :
1,5e-5. Arènes contre c130, graine 183001, 920 donnes : **n = 400 +1,3 ± 1,5** ; n = 1 +3,0 ± 1,5.
Hors protocole, mêmes réseaux sur 920 donnes neuves (graine 184001, `runs/d12-confirm-*`) : n = 1
+0,2 ± 1,4 ; n = 400 +0,2 ± 1,4. Le +3,0 était le hasard des donnes.

**Échec** : avec huit fois plus de décisions stratégiques, l'apprentissage les apprend par cœur deux
fois moins (−0,06 au lieu de −0,13 sur les apprises) et se dégrade moins sur les neuves, mais n'y gagne
toujours rien. Le manque d'exemples n'était pas la cause : ce que le professeur choisit de plus au
choix du peuple et au déclin ne se prédit pas depuis ce que voit le réseau, ou demande bien plus
qu'un ordre de grandeur de données en plus.

Acquis pour la suite : le selfplay ciblé `place=1` produit les décisions stratégiques du professeur
neuf fois plus vite, à force quasi égale (D11).

## D13 : des têtes construites pour le choix du peuple et le déclin (31 octobre)

`py/heads.py` sur le tronc de c130 (corrections parties de zéro : au départ, c130 au bit près),
236 034 paramètres ; appris sur `data/tgt-train` (D12), 8 706 lots, 78 à 140 s ; écarts de perte de
politique à c130 sur `anc-test` (parties neuves), `runs/d13-*.txt` :

| Essai | Peuple | Déclin possible | Conquête | Jetons | Tous | Coup de la cible au peuple |
| --- | --- | --- | --- | --- | --- | --- |
| t0 : sans têtes, tout à 1,5e-5 (témoin, D12) | +0,010 ± 0,006 | −0,001 ± 0,005 | +0,004 | +0,002 | +0,002 | 73,4 → 72,3 % |
| h1 : têtes à 1e-3, tronc figé | −0,027 ± 0,016 | −0,002 ± 0,003 | 0 | 0 | −0,001 | 73,4 → 74,3 % |
| **h2 : têtes à 1e-3, tronc à 1,5e-5** | **−0,048 ± 0,015** | +0,000 ± 0,005 | +0,004 | +0,003 | +0,001 | 73,4 → 74,3 % |
| h2, autre graine (hors protocole) | −0,047 ± 0,016 | +0,000 ± 0,005 | +0,004 | +0,002 | +0,001 | 73,4 → 74,5 % |
| h3 : têtes à 3e-4, tronc figé | −0,013 ± 0,017 | −0,004 ± 0,002 | 0 | 0 | −0,001 | 73,4 → 73,9 % |

- **Le critère est atteint au choix du peuple** (h2 : −0,048, reproduit à −0,047 ; ailleurs, au plus
  +0,004) : **pour la première fois, une décision stratégique se généralise** sur des parties neuves,
  là où l'ancienne tête montait (t0 : +0,010 ; D9 : +0,05). La tête partagée entre les places du marché
  et confrontée aux régions apprend ce que la sortie globale ne pouvait que retenir par cœur.
  L'architecture de la sortie était bien une limite pour cette décision.
- **Rien au déclin** : la tête par région regroupée ne gagne rien (au mieux −0,004). Ce qui décide le
  déclin du professeur ne se lit pas mieux ainsi.
- L'effet porte sur 3 % des positions (la perte moyenne ne bouge pas) ; ce que vaut en jeu un meilleur
  choix du peuple reste à mesurer : il faut la tête dans un moteur pour les arènes.

**Arènes de D13** (31 octobre). La tête du peuple seule (`runs/d13-h2p.bin` : h2 sans la tête du
déclin ; peuple −0,035 ± 0,016 sur `anc-test`), dans le moteur du processeur (`net::Pick`, mêmes
logits que PyTorch à 7e-5 près sur 40 choix de peuple). Contre c130, graine 185001, 920 donnes,
processeur :

| Arène | Score | Durée |
| --- | --- | --- |
| n = 1 (politique seule) | +0,9 ± 1,4 | 43 s |
| n = 400 | **−0,0 ± 1,4** | 88 min (processeur seul ; ~4 min en Vulkan) |

**Échec en jeu.** La tête prédit mieux le choix du peuple du professeur sur des parties neuves (de
73,4 à 74,3 % de coups communs), mais ni la politique seule ni la recherche n'en jouent mieux : un
écart de cette taille, sur 3 % des décisions, ne se voit pas en arène, et la recherche à n = 400 fait
déjà ses choix de peuple sans l'aide de la tête.

## D14 : la valeur est-elle le levier ? (31 octobre)

Relecture de D1 à D13 (`lab.md`, D14) : le gain des fins simulées corrige les valeurs des feuilles, pas
l'a priori ; les élèves de la politique gagnent à n = 1 et rien à n = 400. Hypothèse testée : la valeur
de c130, apprise sur des parties bruitées (A3 de lab7 : −10,7), sous-estime la riposte des adversaires,
que la fin simulée gloutonne révèle.

**Le juge** : 1 000 parties de c130 à n = 400 sans bruit, toutes les décisions cherchées à fond et
enregistrées (278 334 positions, 89 s, graine 191 000 001, `data/judge`), chacune avec sa valeur par fin
simulée (`selfplay --revalue 4` : coup préféré du réseau pour tous jusqu'au prochain tour du joueur,
`place=1`, têtes du réseau à l'horizon, 4 mondes). Hors protocole, le même juge avec **64 mondes** par
position (`data/judge64`, 508 s ; mêmes parties), pour séparer le bruit des mondes d'un biais.
`py/d14.py`, `runs/d14.txt`, `runs/d14-k64.txt` ; ± : intervalle à 95 % par rééchantillonnage des parties.

| Estimateur de la valeur, sur le juge | CE au résultat | Écart à c130 | Brier | Pièces (erreur²) |
| --- | --- | --- | --- | --- |
| c130, la tête de valeur | 1,029 | | 0,556 | 66,9 |
| la recherche ordinaire à la racine (`value`) | 1,016 | **−0,013 ± 0,001** | | |
| fin simulée, 4 mondes | 1,122 | **+0,093 ± 0,016** | 0,568 | 65,5 |
| fin simulée, 64 mondes | 1,078 | **+0,049 ± 0,013** | 0,561 | 64,7 |
| moyenne c130 / fin simulée (4 mondes) | 1,018 | −0,011 ± 0,004 | | |
| moyenne c130 / fin simulée (64 mondes) | 1,015 | −0,014 ± 0,004 | | |

Par type (64 mondes), écart de la fin simulée à c130 : déclin +0,079, conquête +0,047, jetons +0,046,
pouvoirs +0,042, peuple +0,033, défense −0,020 ± 0,018. Par joueurs : 2 j. +0,009 ± 0,011, 3 j.
+0,015 ± 0,016, 4 j. +0,060, 5 j. +0,084. Écart signé de la part de victoire du joueur qui agit, fin
simulée moins c130 : **+0,002 ± 0,001** (par type, de −0,004 au peuple à +0,011 aux pouvoirs) ; écart
absolu 0,068.

**Le critère n'est pas atteint, et de loin, à 4 comme à 64 mondes** : depuis la position même, la fin
simulée prédit le résultat *moins* bien que la tête de valeur. Seize fois plus de mondes ne comblent
que la moitié de l'écart et ne changent presque rien à l'écart absolu entre les deux estimations (0,072
→ 0,068) : ce qui sépare la fin simulée du réseau n'est pas le hasard des mondes, c'est un désaccord
systématique, position par position, **et il a tort plus souvent que raison**. L'écart signé est nul :
la fin simulée ne corrige aucun biais d'ensemble de la valeur (les adversaires de la valeur de c130 ne
sont pas « trop gentils »). Elle est d'autant moins bonne qu'il y a de joueurs (à 2 et 3 joueurs, à
parité), là où en jeu elle rapporte le plus.

Ce qui fait mieux que le réseau, c'est ce qui moyenne : la valeur de racine de la recherche ordinaire
(−0,013) et la moyenne à parts égales réseau / fin simulée (−0,011 et −0,014), la cible « q + z » de
Lc0. Un gain d'un peu plus de 1 % d'entropie croisée, de la variance en moins, pas une information que le
réseau n'aurait pas. Sur les pièces à gagner, la fin simulée fait un peu mieux que la tête (64,7 contre
66,9 ; au peuple 96 contre 105, à la défense 49 contre 57) : elle connaît les pièces du tour en cours.

**Conclusion de D14** : l'hypothèse est réfutée. Le gain des fins simulées en jeu n'est pas une meilleure
estimation de la valeur de la position courante qu'on pourrait apprendre à la tête de valeur ; il tient
à ce qu'elles évaluent *sous* les coups cherchés (la suite concrète d'un choix, comparée à celle des
autres choix) et à leur moyenne dans l'arbre. Selon le protocole, D15b (la valeur réapprise sur les fins
simulées) n'a pas lieu. Acquis : le juge sans bruit (`data/judge64`, 278 000 positions avec leur valeur
par fin simulée et celle de la recherche) mesure la qualité de la valeur de n'importe quel réseau sur du
jeu d'arène, ce qu'aucune mesure ne faisait ; `selfplay --revalue`.

## D15a, hors protocole : la valeur de racine du professeur pour seule cible (31 octobre)

D14 fermait D15 ; D15a ne coûtait que quatre minutes sur des données existantes et D14 venait de
montrer que la valeur de racine d'une recherche prédit le résultat un peu mieux que la tête
(−0,013). c130 réentraîné sur `data/tgt-train` (D12) avec `--value-mix 1` (la valeur de racine du
professeur, fins simulées et ancrage, pour seule cible de la victoire ; `fit.py`, 4 passages, réseau de
la fin), taux 5e-5 et 1,5e-5 ; jugé sur `data/judge64` (`d14.py`, `runs/d15-a-value.txt`) et sur
`anc-test` par type (`d9.py`, `runs/d15-a-types.txt`).

| Taux | CE de la valeur au résultat, juge (écart à c130) | au peuple | à 5 joueurs | Politique, `anc-test` (écart) |
| --- | --- | --- | --- | --- |
| c130 | 1,029 | 1,162 | 1,343 | 1,271 |
| 5e-5 | 1,037 (**+0,007 ± 0,002**) | +0,021 ± 0,003 | +0,010 ± 0,005 | 1,276 (+0,005) |
| 1,5e-5 | 1,035 (**+0,006 ± 0,002**) | +0,019 ± 0,003 | +0,007 ± 0,004 | 1,273 (+0,002) |

**Échec** (le critère demandait −0,005 ; pas d'arène) : apprise sur la valeur de racine du professeur,
la tête de valeur juge *moins* bien le jeu d'arène, surtout aux choix de peuple. Les parties du
professeur sont bruitées (recherches rapides, bruit de Gumbel) et sa valeur de racine, calculée sur des
fins simulées, porte le désaccord systématique que D14 mesure : la cible « q » de Lc0 ne vaut ici que
pour la recherche qui a joué les parties jugées, pas pour celle du professeur. La piste de la cible de
valeur est close avec D14.

## D16 : le coût de la recherche au jeu (31 octobre)

Décision de l'utilisateur après D14 : l'effort va au coût de la recherche au jeu, pas à
l'apprentissage. Protocole : `lab.md`, D16. Mini (i5-8500T) a 6 cœurs et l'Expert n'en utilisait qu'un.

**E2, les évaluations en parallèle.** `parallel::Parallel` : les positions qu'une recherche demande en
même temps (les feuilles d'un tour du halving, puis chaque pas de leurs fins simulées, en cadence)
attendent, puis sont évaluées ensemble, réparties entre les fils (`Net::eval_batch` par tranche) ;
la recherche reprend. `Expert::with_threads`, `lab13-expert --threads N` ; le serveur donne à l'Expert
les cœurs de la machine moins un (`src/server/expert.ts`, labos marqués `threads` dans
`src/shared/experts.ts`). Mêmes décisions au bit près qu'à un fil (test
`parallel_evaluations_give_the_same_decisions`) : pas d'arène.

**Le temps par décision** (`lab13 expert-time`, PC, parties entières de 4 et 5 joueurs où tous les
sièges sont l'Expert, ~200 décisions chacune ; `runs/d16-time.txt`) :

| Réglages | Fils | 4 j. : moyenne / p95 / max (ms) | 5 j. : moyenne / p95 / max (ms) | Gain sur la production à 1 fil |
| --- | --- | --- | --- | --- |
| production `n=100,roll=1,mix=0.5,light=1` | 1 | 609 / 1 264 / 1 524 | 973 / 1 949 / 2 233 | |
| production | 2 | 331 / 716 / 785 | 514 / 1 033 / 1 238 | 1,9× |
| production | 4 | 256 / 551 / 637 | 395 / 845 / 939 | 2,4× |
| production | 6 | 242 / 506 / 639 | 369 / 777 / 841 | 2,6× |
| `place=1` (E1) | 1 | 328 / 737 / 849 | 531 / 1 122 / 1 318 | 1,8× |
| `place=1` | 6 | **136 / 299 / 334** | **212 / 439 / 526** | **4,5×** |
| `place=1`, n = 200 | 1 | 594 / 1 404 / 1 717 | 891 / 1 877 / 2 435 | 1,1× |
| `place=1`, n = 200 | 6 | 255 / 619 / 821 | 368 / 804 / 1 033 | 2,6× |
| `place=1`, n = 400 (E3) | 6 | 473 / 1 069 / 1 234 | 731 / 1 618 / 1 997 | 1,3× |

- Les fils rapportent 1,9× à 2, 2,6× à 6 : le halving séquentiel n'offre que 16, 8, 4 puis 2
  simulations par tour, et ses derniers tours, les plus nombreux, n'occupent que deux fils. Au-delà
  de 4 fils, presque rien.
- `place=1` divise le temps par 1,8 à lui seul (les jetons du redéploiement des fins simulées étaient
  la moitié des évaluations) ; avec 6 fils, **4,5 fois moins de temps** que la production pour les
  mêmes réglages de recherche, s'il garde la force (arène E1).
- **n = 400 avec `place=1` à 6 fils tient dans le temps de la production à 1 fil** : p95 à 5 joueurs
  1 618 ms contre 1 949 (le critère de E3). Sur mini, ~1,8 fois plus lent par cœur et 5 fils, environ
  1,5 s en moyenne et 3 s au p95 à 5 joueurs, ce que la production fait aujourd'hui.

**E1 et E3, la force** (arènes contre c130 à n = 400, graine 72002, 920 donnes, Vulkan ; témoin apparié
light=1 de D10, +4,7 ± 1,5 ; `runs/d16-place-n*.tsv`, 12 + 25 + 48 min) :

| Joueur testé | Score | 2 j. | 3 j. | 4 j. | 5 j. | Différence au témoin |
| --- | --- | --- | --- | --- | --- | --- |
| `n=100,roll=1,mix=0.5,light=1` (production, D10) | +4,7 ± 1,5 | +2,9 | +5,4 | +4,8 | +5,9 | |
| **E1** `…,light=1,place=1` | +3,4 ± 1,6 | +2,4 | +4,3 | +3,2 | +3,8 | **−1,3 ± 2,1** |
| `…,place=1`, n = 200 | +4,5 ± 1,4 | +4,0 | +3,3 | +6,5 | +4,2 | −0,2 ± 2,0 |
| **E3** `…,place=1`, n = 400 | **+6,3 ± 1,5** | +3,8 | +7,1 | +7,1 | +7,2 | **+1,5 ± 2,0**, indécis |

- **E1 est retenu** (différence au-dessus de −1,5, borne basse −3,4 au-dessus de −3,5), comme `place=1`
  l'était pour le professeur en D11 (−1,3 ± 2,1, le même écart). Avec les six fils de mini : la même
  recherche qu'aujourd'hui pour **4,5 fois moins de temps**, 0,21 s en moyenne par décision à 5 joueurs
  sur le PC (0,44 au p95), soit environ 0,4 s (0,8 au p95) sur mini.
- **E3 n'est pas établi** : n = 400 avec `place=1` gagne +1,5 ± 2,0 sur la production, l'intervalle
  contient zéro ; le critère demandait un intervalle au-dessus de zéro. Il est pourtant positif à toutes
  les tailles, et de +7 à 3, 4 et 5 joueurs, cohérent avec lab7 (n = 400 avec fins simulées : +8,9) et
  D11 (+8,4 avec l'ancrage, sans `light`) ; `light=1` et l'absence d'ancrage en coûtent sans doute un
  ou deux. Au p95, il tient dans le temps de la production actuelle : c'est l'option « plus fort au même
  temps » si on la préfère à « aussi fort, 4,5 fois plus vite ».

**L'Expert du jeu passe à E1** (`src/shared/experts.ts` : `n=100,roll=1,mix=0.5,light=1,place=1`,
fils = cœurs moins un). À déployer sur mini (`bun run kernel:build` reconstruit `lab13-expert`).

**Ce qui reste du coût.** À 6 fils, les derniers tours du halving (2 simulations) laissent quatre cœurs
inoccupés ; une recherche à n = 400 fait 93 tours dont 75 à 4 simulations ou moins. Pour aller plus
loin sans changer les décisions : lancer plusieurs mondes par simulation ou évaluer d'avance les enfants
probables ; pour changer la recherche : des fins simulées jouées par un réseau léger (distillé de c130,
politique seule), à mesurer en arène.
