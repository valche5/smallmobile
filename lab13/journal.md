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
