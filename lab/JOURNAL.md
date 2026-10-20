# IA SmallMobile : résultats utiles et plan d'expériences

État au 20 octobre 2026. Ce document remplace le journal chronologique ; les anciens détails
restent dans Git. Les résultats ci-dessous ont été mesurés. Les étapes P0 à P7 sont un plan
à exécuter : la réécriture du document ne vaut pas réalisation de ces étapes.

## Objectif et priorité

Construire l'IA la plus forte possible sur les cartes générées, à **2, 3, 4 et 5 joueurs**.
Le labo peut chercher longtemps ; la version du jeu doit aussi être évaluée au temps de
réflexion disponible sur mini. Conserver ces deux mesures de force.

**Où on en est (20 octobre) :** apprendre valeur et politique sur du jeu fort a relancé le
progrès que les générations identiques à gen8 n'apportaient plus : C (+5 à +6 sur gen8), gen9
(+7,9 sur C, plus de +10 sur gen8, livré le 20 octobre), gen10 (fenêtre W : +6,3 sur gen9, mais
moins bon face à Normale). **Leviers suivants, dans l'ordre :** auditer l'encodage ; entraîner
face à une population d'adversaires (P6) ; cibles territoriales (P7) ; placements et recherche
(P5). Détail : « Suite du plan après gen10 ».

**Décision du 20 octobre : l'Expert ne sait pas contre qui il joue.** Il doit bien jouer dans tous
les cas : ni `rp=4` (simulation selon le type de chaque place), ni type de place (humain, IA,
niveau, réseau) en entrée du réseau. La robustesse face aux styles différents vient de
l'entraînement (P6), pas de l'identification des adversaires. **Permis, en revanche :** se faire un
profil de chaque joueur d'après ses coups publics dans la partie, comme un humain attentif
(historique en entrée du réseau ; P3 revue).

Règle d'honnêteté : connaître seulement ce qu'un humain attentif peut savoir. Les scores se
reconstituent à partir des gains annoncés. Le contenu des pioches et l'ordre des cartes rendues
sont publics ; leur ordre caché et les dés à venir doivent être échantillonnés, jamais consultés
pour décider. Voir `brain/src/know.rs` et le correctif P0.

## Système actuel et témoins

- `kernel/` : règles, simulation, cartes, IA Facile/Normale/Difficile.
- `brain/` : recherche guidée par réseau, commune au labo et à l'Expert. Arbre de suites
  d'actions en boucle ouverte, mondes possibles, sélection PUCT ; chaque joueur maximise sa
  propre valeur. Les placements sont regroupés en quelques plans candidats.
- `lab/` : arène, auto-apprentissage, cibles de valeur ; `lab/py/` : entraînement PyTorch.
- Réseau : graphe des régions, 4 couches de largeur 64 ; 77 entrées par région, 700 globales,
  448 clés d'actions. Sorties : politique, victoire par joueur, pièces encore à gagner.
  Inférence Rust AVX2/FMA ; concordance Python/Rust déjà vérifiée à environ 5·10⁻⁵.
- Expert livré : `lab/nets/gen10w.bin` depuis le 20 octobre au soir (gen9 le matin, gen8 avant), via `target/release/expert`
  et `src/server/expert.ts`.
  `bun run kernel:build` construit le noyau WebAssembly et cet exécutable.
- Mesure : `lab arena --hero A --field B`, ou `--table 'A|B|C'` pour une table fixe ; chaque
  joueur passe à chaque place. `--out r.tsv` écrit chaque place de chaque partie ;
  `python3 lab/py/score.py` en tire le score du protocole, ses intervalles par donne et les
  différences appariées (deux arènes de même graine, ou `--versus` pour deux rôles d'une table).
- Vitesse : `lab speed --agent az:…` chronomètre des recherches sur des positions fixes et donne
  deux empreintes de leurs résultats ; une optimisation qui ne change rien les garde.
- Archives réelles : `data/games/`, graines, actions et temps de décision, réseau de l'Expert de
  chaque décision (`ai.net`, et `expert` en tête du fichier ; ajouté après coup, gen8, aux parties
  d'avant le 20 octobre) ; résumé par `bun lab/tools/games.ts`. Les quelques parties humaines disponibles ne suffisent pas encore
  à entraîner un modèle des humains. Les 4 parties archivées au 19 octobre réunissent 1 ou
  2 humains et 1 à 3 Expert : l'Expert affronte surtout d'autres Expert, qu'il simule pourtant
  comme Normal (`rp=2`).

Réglage livré (gen10w depuis le 20 octobre au soir, gen9 avant ; gen8 reste le témoin figé des mesures
ci-dessous) :

```text
az:net=lab/nets/gen10w.bin,n=200,w=0.5,c=2.5,roll=1,depth=3,rp=1
```

`rp=1` dans le jeu depuis le 20 octobre, 20 h 47 (décision de l'utilisateur : plus d'IA Normale
dans les fins simulées ; environ 190 ms par décision sur mini avec le code accéléré). Les mesures
« aux réglages livrés » de ce journal ont été faites en `rp=2` jusqu'à gen10 ; la comparaison
`rp=1`/`rp=2` avec gen10w est prévue dans la campagne gen11.

| Paramètre | Sens |
| --- | --- |
| `n` | Itérations de recherche par décision |
| `w` | Poids de la victoire ; le reste est une sigmoïde de l'avance sur le meilleur rival |
| `c` | Exploration PUCT ; normalisation des valeurs par nœud activée par défaut |
| `roll` | Part des fins simulées dans l'évaluation ; 0 = réseau seul |
| `depth` | Nombre de **tours de joueur**, puis évaluation par le réseau ; 0 = fin complète |
| `rp=0` | Normal pour tous dans les fins simulées |
| `rp=1` | Politique du réseau pour tous, sans recherche imbriquée |
| `rp=2` | Réseau pour soi, Normal pour les adversaires |
| `rp=3` | Réseau pour soi, modèle adverse tiré selon les croyances |
| `rp=4` | Réseau pour soi, chaque adversaire selon le type de sa place (réseau pour un Expert, Normal pour une IA du jeu) ; place inconnue : mélange égal |
| `rt` | Température des coups du réseau dans les fins ; défaut 0, coup préféré |
| `rolls` | Fins simulées par position nouvelle ; défaut 1 |

Sur mini (i5-8500T, un cœur), l'Expert livré prend environ 80 à 250 ms par décision,
médiane 170 ms. Ce sont des repères mesurés, pas une garantie de latence.

## Résultats à retenir

### Comment lire les mesures historiques

L'arène rejoue chaque donne à toutes les places du joueur testé, face au reste de la table.
À force égale, sa probabilité de victoire est 1/n. « Écart » désigne les points de victoire
au-dessus de cet attendu, **pondérés par le nombre de parties**. À nombre de donnes égal par
taille, les tables de 5 joueurs pèsent donc plus que celles de 2. Le nombre de donnes indiqué
est le total sur les tailles, sauf mention contraire.

Les anciens intervalles ne regroupent pas les parties par donne. Les graines 61/62 ont beaucoup
servi à sélectionner les réglages. P0/P1 doivent confirmer les résultats avec une mesure
adaptée et des graines inédites.

Ordres de grandeur, tirés des ± des journaux (parties supposées indépendantes) : 400 donnes
donnent environ ±2,5 points sur le total d'un réglage, donc environ ±3,5 sur la différence de
deux réglages ; à 2 joueurs, 100 donnes donnent ±6,9 points. Les réglages `w`, `s` et `c` ont
été choisis sur 200 donnes contre Normal avec d'anciens réseaux, dans le bruit (`w=0.5`/`1` :
+55,4/+53,1, ±2,7 chacun) : ils ne sont pas établis.

Face à Difficile, le score est saturé autour de +40 : en P1, n=80 et n=200 y font jeu égal
(−1,1 ± 1,8) alors que n=80 perd de 6,4 ± 2,1 contre n=200. Difficile ne sert plus qu'à vérifier
l'absence de régression et à tenir la place des IA du jeu dans les tables mixtes ; la force se
compare entre joueurs forts.

### Recherche et premiers apprentissages

| Essai | Résultat utile | Conséquence |
| --- | --- | --- |
| Difficile contre Normal, 1 000 donnes | 85 % à 2 joueurs, 32 % à 5 ; écart global +19 | Normal est trop faible pour départager les modèles récents |
| MCTS sans réseau, n=500 contre Difficile, 100 donnes | Écart +12 | La recherche apporte déjà un gain |
| Premiers réseaux sur quelques milliers de parties MCTS | Valeur fragile, politique diffuse, déclin trop fréquent | Beaucoup de positions corrélées ne remplacent pas des parties diverses |
| Pré-entraînement sur 100 000 parties Normal, puis recherche hybride | La normalisation des petites différences de valeur améliore fortement la recherche | Garder une référence hybride et surveiller les échelles |
| Valeur apprise sur 32 fins Normal moyennées (`label1`) | gen3 : +54 contre Normal à profondeur 5, contre +52 en fins complètes, environ deux fois plus vite | Les cibles moyennées rendent les recherches tronquées efficaces |
| Réseau 128×6 ou 20 passes sur les mêmes données | Pas de gain convaincant en jeu ; grand réseau 3,3 fois plus lent | Faible priorité à agrandir le réseau avec ces mêmes cibles |
| Plus de données `label`, poids des pièces multiplié par quatre | Erreur d'avance autour de 7 pièces, presque inchangée | Répéter cette recette a peu de rendement |

La validation des premiers essais mélangeait les positions des mêmes parties. Le découpage
par blocs l'a améliorée, mais la séparation stricte par partie reste à faire (P0).

### Fins simulées et modèle des adversaires

Gen6 contre Difficile, 1 000 donnes, `n=200,depth=5,w=0.5` :

| Réglage | Écart |
| --- | --- |
| Normal pour tous (`rp=0`) | +32,8 |
| Réseau pour soi (`rp=2,rt=0`) | +39,2 |
| Réseau pour soi, coups échantillonnés (`rt=0.5`) | +37,1 |

Le gain de `rp=2` est confirmé ; celui de `rt=0.5` sur un petit échantillon ne l'était pas.
Avec `rp=2`, les profondeurs 3/5/8 donnent +38,0/+39,2/+38,8 : approfondir était peu rentable
dans ce contexte. À profondeur 5, n=400/800 donne +40,0/+42,0, avec davantage de calcul.

**Résultat déterminant : gen8 parmi les Expert**, 400 donnes, graine 71. Adversaires : gen8
avec le réglage livré (`rp=2`, 200 itérations, profondeur 3).

| Joueur testé | 2 j. | 3 j. | 4 j. | 5 j. | Écart |
| --- | --- | --- | --- | --- | --- |
| `rp=2`, témoin | 49,0 % | 33,3 % | 27,6 % | 16,9 % | −0,5 |
| `rp=1`, réseau pour tous | 65,0 % | 51,0 % | 37,4 % | 32,2 % | +13,8 |
| `rp=1,rt=0.5` | 64,0 % | 51,5 % | 41,9 % | 27,7 % | +13,5 |

Dans l'autre sens, `rp=2` parmi des `rp=1` fait −7,0. Mais contre Difficile, `rp=1` fait +34,6,
contre +42,4 pour `rp=2` sur la graine 61. Le classement dépend fortement des adversaires.
Battre davantage Difficile ne suffit plus à démontrer un progrès général.

Coût mesuré le 19 octobre (ci-dessous) : une décision `rp=1` prend 2,4 fois une décision `rp=2`.

### Mesures du 19 octobre : graines inédites, code corrigé (P0)

Graines 201 à 203, jamais utilisées ; recherche honnête à la racine ; scores du protocole
(`lab/py/score.py`, tailles à poids égaux, intervalles à 95 % par donne). Résultats par place :
`lab/runs/p0/*.tsv` sur mini ; script `lab/runs/p0-variance.sh`. 57 min de c7a.16xlarge, 0,87 $.

Parmi des Expert livrés (`rp=2`), 400 donnes, 100 par taille, graine 201 :

| Joueur testé | Score | Différence avec le témoin | Temps moyen / p95 |
| --- | --- | --- | --- |
| `rp=2`, témoin | −0,5 ± 2,1 | — | 132 / 219 ms |
| `rp=1` | +11,7 ± 2,4 | **+12,1 ± 3,0** | 320 / 561 ms |
| `roll=0,n=800` | +0,7 ± 2,3 | +1,2 ± 3,1, indécis | 109 / 158 ms |
| `w=1` | −0,8 ± 2,2 | −0,4 ± 3,0, indécis | 133 / 218 ms |

`rp=1` confirme son avance sur des graines neuves, à toutes les tailles (+15,0, +15,2, +11,6,
+6,8). `w` ne change rien de mesurable : garder 0,5. Temps sur une machine pleine (64 parties
à la fois) : à comparer entre eux, pas à la latence de mini.

Tables mixtes proches de la production, 400 donnes, chaque joueur à chaque place :

| Table | `rp=4` | `rp=2` | `rp=1` | Difficile |
| --- | --- | --- | --- | --- |
| 3 joueurs, graine 202 | +30,8 ± 2,6 | −9,0 ± 2,4 | — | −21,8 ± 1,8 |
| 4 joueurs, graine 203 | +17,4 ± 2,4 | −4,0 ± 1,9 | +1,7 ± 2,1 | −15,0 ± 1,4 |

Différences à la même table : `rp=4` − `rp=2` +39,9 ± 4,6 (3 j.) et +21,4 ± 3,7 (4 j.) ;
`rp=4` − `rp=1` +15,7 ± 3,8 ; `rp=1` − `rp=2` +5,7 ± 3,2. Simuler chaque adversaire selon son
type gagne nettement : `rp=2` se trompe sur les Expert, `rp=1` sur Difficile. Parmi des Expert
seulement, `rp=4` joue comme `rp=1`. Coût de `rp=4` : 202 ms (3 j.) et 296 ms (4 j.) par
décision, selon le nombre d'Expert à simuler par le réseau.

Variance : l'écart-type par donne du score d'un joueur va de 16 à 28 points selon la taille,
le plus fort à 2 joueurs. Entre deux arènes de même graine, la corrélation par donne est
presque nulle (−0,13 à 0,26) : l'appariement ne gagne presque rien, et une même arène témoin
peut servir à plusieurs candidats. Pour la différence de deux réglages : environ 100 donnes par
taille pour ±3 points, 230 pour ±2, 900 pour ±1. À la même table, deux candidats se disputent
la victoire (corrélation −0,3 à −0,7) : la comparaison y est moins précise par donne.

Valeur de gen8 (`lab noise`, 3 968 positions de l'IA Normal, avance en pièces) : erreur 6,2
(gen6 : 6,9) ; dernière manche 2,4 (bruit 0,13) ; 1 à 2 manches restantes 3,6 ; 3 à 5 : 6,6 ;
6 ou plus : 7,3 (gen6 : 8,6). gen8 a progressé en début de partie ; le plancher de fin de partie
n'a pas bougé.

### `rp=4` à temps égal (P1, 19 octobre)

Graines 301 à 304, inédites ; résultats `lab/runs/p1/*.tsv` sur mini, script
`lab/runs/p1-rp4.sh`. 85 min de c7a.16xlarge, 1,11 $. À n=80, `rp=4` coûte autant que l'Expert
livré (`rp=2`, n=200) parmi des Expert, et moins dès qu'il y a des IA du jeu : il les simule par
la politique Normal, bien moins chère que le réseau.

Parmi des Expert, 920 donnes, 230 par taille (graine 301 ; 302 pour la dernière ligne,
400 donnes). À force égale, le score attendu est 0 (témoin de P0 : −0,5 ± 2,1) :

| Joueur testé | Champ | Score | Temps moyen / p95 |
| --- | --- | --- | --- |
| `rp=4`, n=80 | livrés | **+5,8 ± 1,5** | 127 / 223 ms (champ : 133) |
| `rp=4`, n=200 | livrés | +12,2 ± 1,6 | 315 / 553 ms |
| livré | `rp=4`, n=200 | −8,9 ± 2,2 | 131 / 219 ms |

Par taille, `rp=4` n=80 : +6,6, +8,8, +3,9, +3,8. Sur les mêmes donnes, n=80 − n=200 :
−6,4 ± 2,1 : parmi des Expert, les itérations comptent encore. `rp=4` n=200 retrouve le score de
`rp=1` en P0 (+11,7), comme attendu : parmi des Expert, les deux simulent pareil.

Face à Difficile seul (graine 301, 920 donnes) : `rp=4` n=80 +39,2 ± 1,5 en 52 ms (p95 91),
livré +40,4 ± 1,5 en 130 ms (p95 221) ; différence −1,1 ± 1,8, indécise. Ici les deux simulent
Difficile par Normal ; au-delà de 80 itérations, le gain est faible.

Tables mixtes, chaque joueur à chaque place, 400 donnes (temps moyen par décision) :

| Table | `rp=4` n=80 | livré | `rp=4` n=200 | Difficile |
| --- | --- | --- | --- | --- |
| 3 joueurs, graine 303 | +23,9 ± 2,7 (81 ms) | −3,5 ± 2,5 (131 ms) | — | −20,4 ± 1,9 |
| 4 joueurs, graine 304 | +9,4 ± 2,4 (118 ms) | −0,1 ± 2,1 (127 ms) | +5,2 ± 2,2 (258 ms) | −14,5 ± 1,4 |

Différences à la même table : n=80 − livré +27,3 ± 4,8 (3 j.) et +9,5 ± 3,9 (4 j.) ;
n=200 − livré +5,4 ± 3,5 ; n=80 − n=200 +4,1 ± 4,0, indécis.

**Conclusion :** à temps égal ou moindre, `rp=4` n=80 bat l'Expert livré parmi des Expert et sur
les tables mixtes, et l'égale face à Difficile seul en 40 % du temps. Limites : temps mesurés sur
une machine pleine (rapport avec mini à étalonner) ; aucune table n'avait d'humain, pour qui
`rp=4` prendrait le mélange égal, non mesuré. **Non retenu pour le jeu (décision du
20 octobre) :** `rp=4` suppose de savoir qui occupe chaque place ; l'Expert ne doit pas le savoir.
Son gain vient de là : simuler chaque place selon son type.

### Progression des réseaux et rendement du calcul

Contre Difficile, `n=200,depth=3,rp=2,w=0.5,c=2.5` :

| Réseau | 1 000 donnes, graine 61 | 4 000 donnes, graines 61 et 62 |
| --- | --- | --- |
| gen6 | +38,0 | — |
| gen7 | +40,9 | +39,9 |
| gen8 | +42,4 | +40,9 |

Sur 1 200 donnes dans chaque sens, gen8 parmi des gen7 fait +1,6 ; gen7 parmi des gen8 fait
−1,4. Le gain de gen8 est modeste, autour d'un point contre Difficile sur le grand ensemble.
La recette rapporte moins ; cela ne prouve pas un plafond de l'architecture.

Gen8 contre Difficile, 1 000 donnes, graine 61 :

| Réglage | Écart | Observation |
| --- | --- | --- |
| `rp=2`, n=200 / 100 / 50 / 25, profondeur 3 | +42,4 / +41,6 / +37,6 / +35,1 | n=100 intéressant pour la latence, à confirmer face aux Expert |
| `rp=0`, n=200, profondeur 3 | +37,1 | Environ le coût de `rp=2,n=100`, moins fort contre Difficile |
| `roll=0`, n=200 / 800 / 3 200 | +33,2 / +35,5 / +36,6 | Comparaison parmi les Expert et à temps égal encore manquante |

Sur gen6, quatre fins de 5 tours par feuille donnaient +33,5 contre Difficile, contre +30,8
pour une seule, au prix d'environ trois fois plus de temps. Quatre fins complètes donnaient
+30,7. L'erreur de valeur était d'environ 6,9 pièces, contre 1,7 de bruit pour ses cibles.
**Cela limite l'intérêt de moyenner davantage ces fins ; cela n'établit pas que la valeur
serait suffisante face aux Expert.** Moyenner ne corrige pas un modèle adverse inadéquat.

La valeur seule passe mal à l'échelle : `roll=0` gagne 3,4 points de n=200 à n=3 200, seize
fois plus de calcul, quand `rp=2` fait déjà +35,1 à n=25. L'erreur de pièces de gen8 est la même
à l'entraînement et en validation (0,031 et 0,030) : le réseau n'apprend pas mieux ses propres
données. Plus de données n'a pas réduit l'erreur ; le réseau 128×6 et les 20 passes n'ont rien
apporté en jeu. Une information manquante dans l'encodage est une explication possible
(diagnostic en P0).

### Données et modèles à réutiliser

Les dossiers désignent le générateur : `data/gen7` contient les parties de gen7 qui ont servi
à entraîner gen8. Données volumineuses et checkpoints `.pt` : à retrouver sur mini et dans
le bucket. Réseaux exportés `.bin` : `lab/nets/`.

| Source | Usage et provenance |
| --- | --- |
| `normal`, `gen0`, `gen1` | Amorçage : Normal, MCTS, premières recherches hybrides |
| `label1` | Environ 1,96 million de positions issues de Normal avec exploration |
| `label2`, `label3`, `label4` | Positions issues de gen3l, gen4, gen5 ; 60 000, 60 000, 80 000 parties sources |
| Tous les `label1` à `label4` | Valeur calculée par **32 fins jouées par Normal**, même si la position vient d'un Expert |
| `gen6` | 20 000 parties, 1,42 million de positions ; `rp=2`, n=200, profondeur 3 |
| `gen7` | 20 000 parties, 1,48 million de positions, environ 8 Go ; même recette |
| gen8 | Depuis gen7 ; valeur sur `label1` à `label4`, politique sur `gen7:policy` et `gen6@0.5:policy`, 8 passes |
| `p2`, `p2-judged` | 5 000 parties gen8 `rp=1` (377 000 positions) ; 221 000 positions rejugées par 32 fins de gen8 |
| `gen9`, `gen9-judged` | 20 000 parties p2c `rp=1` (1,51 million de positions) ; 861 000 positions rejugées par 32 fins de p2c |
| gen9 | Depuis gen8 ; anciennes sources au quart, plus `p2`, `p2-judged`, `gen9`, `gen9-judged` (valeur et politique), 24 000 lots (P2 étendu) |
| `gen10`, `gen10-judged` | 20 000 parties gen9 `rp=1` (1,54 million de positions) ; 888 000 positions rejugées par 32 fins de gen9 |
| `gen10f`, `gen10w` | Depuis gen9, 24 000 lots : F comme gen9 plus `gen10` ; W seulement `p2`, `gen9`, `gen10` (fenêtre) |

Les générations récentes gardent 50 % des décisions cherchées à fond, les autres à 50 itérations,
avec bruit 0,25 pour les recherches complètes. Générer 20 000 parties prenait environ une heure
sur 64 cœurs ; entraîner gen8 environ 78 minutes. Ce sont des repères pour l'ancienne recette,
pas une estimation des fins entièrement jouées par le réseau.

Dans `train.py`, `:policy` exclut la valeur d'une source ; `@0.5` sous-échantillonne les blocs,
ce n'est pas un poids de perte. `--q-weight` mélange résultat et pièces attendues par la
recherche, sans modifier la cible de victoire. `--lazy` lit les données à la demande.
`train.py` n'utilise pas la validation pour choisir un état : nombre de passes fixe, dernier
état exporté. Le découpage par blocs n'a donc faussé que les chiffres affichés, pas les réseaux.
Les enregistrements gardent les pièces attendues par la recherche (`qgain`), pas ses chances
de victoire, ni la graine et les actions de la partie.

Les frontières des parties se retrouvent dans les fichiers existants : les positions d'une
partie se suivent et partagent la même carte. Sur `gen7/s6acfc7-00.v2.bin`, le triplet
(joueurs, régions, voisinages) découpe 23 122 positions en 322 segments, tous de cartes
distinctes, de longueur médiane 49 à 2 joueurs et 91 à 5 joueurs, au plus 117 : des parties
entières (64 fichiers, environ 20 000 parties).

## Hypothèses à départager

1. **Ancrage sur Normal.** La valeur apprend ses fins de partie ; la politique apprend des
   recherches dont les adversaires des fins simulées jouent Normal. C'est une explication
   plausible du plateau, pas une causalité isolée. `label --agent` ne change que les positions
   sources : le joueur qui les juge reste Normal.
2. **Adaptation adverse.** `brain/src/opponents.rs` contient deux modèles, Normal et le coup
   préféré du réseau, avec mise à jour bayésienne. `search_against` reçoit les croyances,
   mais leur observation n'est pas branchée dans l'arène ou le jeu. `rp=3` via `search` est
   donc un mélange 50/50 sans adaptation. Aucun gain adaptatif n'est mesuré. Les types de
   places (humain, Expert, niveau d'IA) sont publics : simuler chaque place selon son type est
   honnête et ne demande aucune croyance ; seul le jeu des humains reste à deviner.
   **Type de place exclu (décision du 20 octobre) :** l'Expert ne sait pas contre qui il joue ;
   seul le profil observé dans la partie est permis (P3).
3. **Cibles de politique.** Une recherche à 200 itérations peut bien jouer sans suffisamment
   améliorer la politique qu'elle doit enseigner. Comparer les recettes à coût total comparable.
4. **Coups absents.** `brain/src/moves.rs` fixe certains placements de pouvoirs et limite les
   plans de redéploiement/défense. La recherche ne peut choisir un plan absent. L'importance
   de cette restriction n'a pas été mesurée.
5. **Entrées de la valeur.** Avec les anciennes cibles, l'erreur de valeur ne baissait ni avec
   les données ni avec la taille du réseau. Les nouvelles la font baisser (9,3 → 8,4 → 6,6
   pièces face aux fins du réseau, gen8 → C → gen9), sauf à la dernière manche, presque
   déterministe (2,5 → 3,3). Deux informations publiques manquent dans `brain/src/encode.rs`
   (vérifié le 20 octobre) : une paix n'est encodée que si elle concerne le joueur qui agit,
   celle entre deux adversaires est invisible ; des cartes rendues sous la pioche, le réseau
   reçoit le nombre (`G_KNOWN_TAIL`), pas l'ordre, public. Le type de chaque place n'est pas
   non plus une entrée, et ne doit pas l'être (décision du 20 octobre). Effet non mesuré :
   l'erreur résiduelle seule ne prouve pas que ces manques en sont la cause.
6. **Angle mort de l'auto-apprentissage.** Des parties entre réseaux apprennent peu à exploiter
   des adversaires d'un autre style. Mesuré en gen10 : sans les anciennes parties de Normal, W
   bat F parmi des gen9 (+3,0) mais exploite moins bien Normale (−4,3). Essai direct : une
   population d'adversaires dans les parties d'entraînement (P6).
7. **Cibles trop globales.** Victoire et pièces finales disent mal quelle région ou quel choix a
   été mal jugé. Des cibles locales (propriétaire de chaque région après une rotation, revenu
   au prochain décompte, pertes d'une armée) pourraient enrichir le tronc commun (P7).

## Protocole commun

- Figer code, poids, paramètres et adversaires. Enregistrer commit, empreinte des poids,
  graines, machine, commandes, données et durée.
- Séparer graines d'entraînement, sélection et confirmation. Ne pas utiliser les graines
  historiques 61/62 comme preuve finale de progrès.
- Garder le format duplicate à 2/3/4/5 joueurs. Pour A/B face à un adversaire fixé, utiliser
  les mêmes donnes et places. Exporter les résultats individuels.
- Présenter victoires par taille, avance/rang en diagnostics, temps moyen et p95 par décision.
  Nouveau score global : moyenne **à poids égaux des quatre tailles** de l'écart à 1/n.
  L'ancien score peut être publié séparément ; ne pas mélanger les deux.
- Rééchantillonner les donnes entières pour les intervalles, séparément par taille, en conservant
  les paires A/B. Les places d'une donne ne sont pas indépendantes.
- Présélection : 400 donnes au total par confrontation, soit ±3 points sur une différence
  (mesuré le 19 octobre) : assez pour les réglages de P1, pas pour les branches de P2, qui
  passent directement à la taille de confirmation.
- Confirmation du candidat retenu : nouvelles donnes, 230 par taille pour ±2 points sur une
  différence, 900 pour ±1 (variance mesurée). Fixer ce nombre avant le calcul ; ne pas répéter
  jusqu'à obtenir un résultat favorable. Les campagnes coûtent peu : 400 donnes parmi des Expert
  `rp=2` prennent 8 à 11 min sur c7a.16xlarge, donc 3 600 donnes de 1 h 15 à 1 h 45, au plus
  2 $ de calcul au plafond Spot. Une arène témoin sert à plusieurs candidats de même graine.
  Dimensionner pour la précision.
- Arrêt anticipé (`lab arena --stop 3`, depuis le 19 octobre) : pour une question « le candidat
  gagne-t-il 3 points ou rien ? », l'arène s'arrête dès que le test séquentiel conclut (5 %
  d'erreur de chaque côté, un regard toutes les 23 donnes par taille). Rejoué sur les 9 arènes
  de P1 et P2 : 3,5 fois moins de donnes, mêmes conclusions. Le score d'une arène arrêtée est
  moins précis (C : +7,5 ± 3,8 après 184 donnes, +6,1 ± 1,6 sur 920) et un peu flatté par
  l'arrêt : jouer l'arène entière quand l'ampleur compte (comparer deux candidats entre eux,
  tableaux du journal).
- Promotion : gain sur le score principal choisi à l'avance, intervalle à 95 % de la différence
  au-dessus de zéro ; examiner chaque taille et le coût. Tolérance de régression : 2 points sur
  le score global, par étalon. Par taille, l'intervalle d'une différence reste trop large pour
  exclure 2 points (environ ±3 à ±4,5 à 2 joueurs pour 500 à 1 000 donnes) : une taille ne
  bloque la promotion que si son recul est significatif. Si l'incertitude ne permet pas
  d'exclure une régression supérieure à la tolérance, le résultat est **indécis**, pas une
  preuve d'équivalence.
- Panel de chaque génération (depuis le 20 octobre) : la génération précédente ne suffit pas.
  Ajouter Difficile et Normale à chaque génération (le recul de gen10 face à Normale ne se voyait
  pas entre réseaux) ; pour une recette nouvelle (P6), un panel tenu à l'écart de
  l'entraînement : générations figées et tables mixtes absentes des parties d'entraînement.
  Publier les résultats par adversaire et par taille : à plusieurs joueurs, un classement
  unique peut cacher une dépendance à la composition de la table.
- Tests ciblés pour chaque modification, puis vérifications Rust du projet avec timeout.
  Entraînement et campagnes de parties se font sur AWS.

## Plan priorisé

### P0 — Fiabiliser la recherche et les mesures

**Préalable aux nouveaux résultats.**

1. **Fait le 19 octobre.** La racine était développée avant `determinize` : dans le labo
   (`Az::act`, auto-apprentissage), sa première fin simulée utilisait les vrais dés/pioche. Une
   valeur sur n+1 (0,5 % à n=200, 2 % à n=50), qui agissait sur la valeur des coups jamais
   essayés, la normalisation et `qgain`. `Expert::decide` protégeait déjà le jeu. La racine est
   maintenant jugée dans un monde possible, et `determinize` remet la partie cachée des pioches
   dans un ordre fixe avant de la mélanger : le monde ne dépend plus que de ce que sait le
   joueur. Test `the_search_sees_only_what_the_player_knows` (Expert livré, réseau seul,
   Difficile) : il échouait avant le correctif. Pas de campagne de remesure dédiée : les mesures
   de P1, sur graines inédites, en tiennent lieu.
2. **Fait.** `lab selfplay` et `lab label` écrivent à côté de
   chaque fichier `x.bin.games` : le nombre de positions de chaque partie. `data.py` le lit, ou le
   reconstruit d'après les cartes et l'écrit (méthode vérifiée sur un fichier gen7, label4 et gen0
   v1 : 322, 5 031 et 99 parties, toutes de cartes distinctes). La validation se choisit par une
   règle fixe sur une clé de la carte (`in_validation`, 5 %) : la même pour toutes les
   générations, quel que soit l'ordre des fichiers ; `@0.2` garde des parties entières. Les
   `.games` de `label1` à `label4`, `gen6` et `gen7` sont dans le bucket (`lab/py/index.py`, P2) ;
   les autres données se liront au premier calcul qui les utilise. gen8 a vu 95 % des anciennes parties : pour les réseaux issus de gen8, ne
   juger que les parties de validation des données neuves (P2).
3. **Fait.** `lab arena --out`, `--table`, `rp=4` (modèle par type de place, transmis par
   l'arène) et `lab/py/score.py` ; temps moyen et p95 par décision pour chaque rôle.
4. **Fait le 19 octobre** (`lab/runs/p0-variance.sh`, graines 201 à 203) : variance par donne,
   corrélation des paires et présélection P1. Résultats : « Mesures du 19 octobre ».
5. **Fait pour gen8 le 19 octobre** (« Mesures du 19 octobre »). Déjà mesuré pour gen6 (`lab noise`, positions de
   l'IA Normal jugées deux fois par 32 fins, avance du joueur qui agit, en pièces) :

   | Positions | Dispersion des cibles | Bruit d'un jugement | Erreur du réseau |
   | --- | --- | --- | --- |
   | Toutes | 17,6 | 1,7 | 6,9 |
   | Dernière manche | 20,9 | 0,1 | 2,3 |
   | 6 manches ou plus restantes | 14,2 | 2,4 | 8,6 |

   L'erreur se concentre en début de partie, où la cible dépend d'un long avenir. En fin de
   partie, presque déterministe, il reste 2,3 pièces d'erreur (2,4 pour gen8) : piste pour
   l'encodage (P5).

**Livrable :** tests ciblés passants, partitions vérifiables, résultats individuels reproductibles,
variance par donne mesurée. Toutes les étapes suivantes utilisent ce code corrigé.

### P1 — Choisir une référence de recherche face aux joueurs forts

**Hypothèse :** le meilleur réglage gen8 dépend du champ adverse et du budget de réflexion.
P1 choisit l'Expert livré et les réglages des générations suivantes ; il ne bloque pas P2.

**Présélection faite le 19 octobre** (« Mesures du 19 octobre ») : `rp=4` l'emporte nettement
sur les tables mixtes et vaut `rp=1` parmi des Expert ; `roll=0,n=800` et `w=1` sont indécis
et sortent de la liste.

**Confirmation faite le 19 octobre** (« `rp=4` à temps égal ») : `rp=4` n=80 bat `rp=2` n=200 à
temps égal (+5,8 ± 1,5 parmi des Expert, +9,5 et +27,3 aux tables mixtes) et l'égale face à
Difficile. **Non intégré au jeu** (décision du 20 octobre : l'Expert ne sait pas contre qui il
joue). Restent possibles, sans identifier les adversaires : `rp=1` ou `rp=2`, le même modèle
pour toutes les places (livré : `rp=2`), et la profondeur 5.

- Poids figés gen8 ; comparer `roll=1,rp=2,n=200,depth=3`, `roll=1,rp=1,n=200,depth=3`,
  `roll=0,n=800` et le modèle par type de place (P0). Fixer `w=0.5,c=2.5,rt=0`, sans bruit à la
  racine ; ajouter `w=1` à la présélection, le critère étant la victoire. Ces n sont des points
  de départ, pas une affirmation d'égalité de coût.
- Adversaires figés : Difficile, gen8 `rp=2`, gen8 `rp=1`. Après présélection, ajouter des
  tables mixtes avec composition et rotation des places consignées. Pour l'Expert livré,
  privilégier des tables proches de la production : plusieurs Expert, des IA du jeu à la place
  des humains. Confronter aussi les candidats retenus entre eux, dans les deux sens à plusieurs
  joueurs.
- Mesurer les temps sur un court échantillon AWS et régler n pour une seconde comparaison
  à temps moyen comparable, sur la même machine ; contrôler aussi le p95. `rp=1` coûte 2,4 fois
  `rp=2`, et `rp=4` 1,5 à 2,2 fois selon la table : à n=200, de l'ordre de 0,3 à 0,4 s de
  médiane sur mini, contre 170 ms aujourd'hui. Étalonner le rapport de vitesse AWS/mini sur
  quelques décisions.
- Ensuite seulement, essayer profondeur 5 pour le meilleur hybride. À 4/5 joueurs, profondeur 3
  ne couvre pas une rotation complète : tester cette rotation est un essai distinct, avec surcoût.

**Décision :** figer une référence forte pour les générations suivantes et une référence au
temps du jeu. Si aucun réglage ne domine, garder plusieurs témoins et publier le compromis. Un
gain face à Difficile seul ne suffit pas pour changer l'Expert livré.

### P2 — Faire apprendre politique et valeur sur du jeu fort

**Priorité d'entraînement. Hypothèse :** renouveler les deux cibles permet de dépasser le plateau.
Dépend de P0 seulement, ni de P1 ni de P3.

**Pilote produit le 19 octobre** (`lab/runs/p2-pilot.sh`, 65 min de c7a.16xlarge, 0,86 $) :
5 000 parties gen8 `rp=1` (graines 8 000 000 à 8 000 009), 377 000 positions enregistrées en
43 min (`lab/data/p2`, 2,1 Go) ; 30 % des positions de ces parties rejugées chacune par 32 fins du
réseau, 221 000 positions en 20 min (`lab/data/p2-judged`, 1,3 Go). Les deux sont dans le bucket.
Entraînement et arènes faits le 19 octobre (`lab/runs/p2-train.sh`, réseaux et résultats dans
`lab/runs/p2` sur mini) : quatre branches depuis gen8, 12 000 lots de 512, lr 5e-4,
`q-weight 0`, anciennes sources au quart (1,73 million de positions) ; les nouvelles font 18 %
des positions de B et 26 % de C et D. 16 min d'entraînement, puis 18 min par arène. Arènes aux
réglages livrés (`rp=2`, n=200), graine 501, 920 donnes :

| Joueur testé | Champ | Score | Par taille (2, 3, 4, 5 j.) |
| --- | --- | --- | --- |
| A, témoin réentraîné | gen8 | −0,3 ± 1,5 | +0,2, +0,4, −1,0, −0,8 |
| B, politique du pilote | A | +2,9 ± 1,6 | +6,0, +3,8, +1,8, 0,0 |
| **C, valeur du pilote** | A | **+6,1 ± 1,6** | +6,6, +5,9, +5,9, +6,0 |
| D, les deux | A | +4,0 ± 1,6 | +5,4, +2,9, +4,1, +3,7 |

Mêmes donnes : C − B +3,2 ± 2,2 ; D − B +1,1 ± 2,1 et D − C −2,1 ± 2,2, indécis. Le simple
réentraînement ne change rien (A vaut gen8). **Les nouvelles cibles de valeur donnent le gain le
plus net, +6 points, égal à toutes les tailles** : premier progrès mesuré au-delà de gen8, et
appui à l'hypothèse 1 (la valeur apprise des fins de Normal plafonne). La politique du pilote
aide aussi, surtout à 2 joueurs ; les deux ensemble ne s'additionnent pas ici. Les pertes de
validation ne se comparent pas entre branches (ensembles de validation différents ; la valeur
des résultats de parties, presque binaire, est plus bruitée).

**Confirmation faite le 19 octobre** (`lab/runs/p2-confirm.sh`, résultats `lab/runs/p2-confirm`
sur mini ; 92 min de c7a.16xlarge, 1,19 $). Mêmes réglages ; C1 : résultats des parties seuls
(`p2:value`, 377 000 positions) ; C2 : fins du réseau seules (`p2-judged`, 221 000) ; C′ : C
refait avec la graine 1 (`train.py --seed 1` : autres parties anciennes gardées, autre ordre des
lots). Arènes parmi des A, graine 501, sauf la dernière (graine 502, donnes neuves) :

| Joueur testé | Champ | Score | Par taille (2, 3, 4, 5 j.) |
| --- | --- | --- | --- |
| C1, résultats seuls | A | +3,7 ± 1,5 | +4,3, +3,8, +3,9, +2,7 |
| C2, fins du réseau seules | A | +1,4 ± 1,5 | +0,2, +2,5, +0,2, +2,8 |
| C′, C graine 1 | A | +4,7 ± 1,5 | +3,6, +3,2, +8,2, +4,0 |
| C | gen8 | **+5,3 ± 1,6** | +3,5, +6,3, +6,7, +4,8 |

Mêmes donnes : C − C1 +2,4 ± 2,2 ; C − C2 +4,7 ± 2,1 ; C2 − C1 −2,3 ± 2,1 ; C′ − C −1,4 ± 2,1,
indécis. **Le gain de C se confirme** : deux graines (+6,1 et +4,7), et +5,3 face à gen8 sur des
donnes neuves. Les résultats des parties en font l'essentiel ; les fins du réseau, faibles seules,
ajoutent encore quelque chose aux résultats (C bat C1). Recette retenue pour la suite : valeur
apprise des résultats des parties et des fins du réseau, avec les anciennes sources.

Prochaine étape, à décider : étendre à 20 000 parties avec cette recette (génération et jugement
environ 4 h sur une machine, 3 $). C est aussi un candidat pour le jeu (+5,3 face à gen8 aux
réglages livrés) ; il cumulerait avec `rp=4`, non mesuré ensemble (`rp=4` écarté depuis).

**Extension lancée le 20 octobre** : 20 000 parties de C (`lab/nets/p2c.bin`) et leurs positions
rejugées par C (`lab/runs/gen9-data.sh`, `lab/data/gen9` et `gen9-judged`), puis entraînement de
gen9 (`lab/runs/gen9-train.sh`). L'Expert du jeu reste gen8 en attendant.

**Données produites le 20 octobre** (256 min de c7a.16xlarge, 3,31 $ ; graines 9 000 000 à
9 000 039, jugement 9 100 000 à 9 100 039) : 40 lots de 500 parties, chacun joué en 4 min 20
puis rejugé en 2 min ; 1,51 million de positions (`lab/data/gen9`, 8,2 Go) et 861 000 positions
rejugées par 32 fins de C (`lab/data/gen9-judged`, 4,9 Go), dans le bucket et sur mini. Valeur de
C sur les 3 968 positions de P0 (`lab/runs/gen9/noise-*.txt`), erreur en pièces, gen8 → C ;
chaque réseau est jugé par ses propres fins (dispersion et bruit presque égaux : 12,4 et 0,90
pour gen8, 12,6 et 0,92 pour C) :

| Positions | Face aux fins de Normal | Face aux fins du réseau |
| --- | --- | --- |
| Toutes | 6,2 → 6,5 | 9,3 → 8,4 |
| 2, 3, 4, 5 joueurs | 7,2, 7,2, 5,7, 5,2 → 7,1, 7,6, 6,0, 5,7 | 11,8, 9,1, 9,3, 8,0 → 12,5, 8,0, 7,9, 6,6 |
| Dernière manche | 2,4 → 2,7 | 2,5 → 2,7 |
| 1 à 2 manches restantes | 3,6 → 4,0 | 5,5 → 5,2 |
| 3 à 5 manches restantes | 6,6 → 6,9 | 10,3 → 9,0 |
| 6 manches ou plus | 7,3 → 7,6 | 10,6 → 9,9 |
| Pièces finales par joueur | 4,1 → 5,0 | 11,1 → 8,9 |

C s'est rapproché des fins du réseau (−0,9 pièce sur l'avance, −2,2 sur les pièces finales,
sauf à 2 joueurs) en s'éloignant un peu de celles de Normal : il a commencé à apprendre le niveau
de jeu du réseau. L'erreur reste loin du bruit (8,4 contre 0,9) : de la marge pour gen9.

**gen9 entraîné le 20 octobre** (`lab/runs/gen9-train.sh`, résultats `lab/runs/gen9` sur mini ;
74 min de c7a.16xlarge, 0,96 $). Depuis gen8, 24 000 lots de 512, lr 5e-4, `q-weight 0`,
anciennes sources au quart, 4,74 millions de positions de 113 790 parties ; deux entraînements
en parallèle, 22 min. V : nouvelle valeur seule (recette de C) ; VP : V et la politique des
nouvelles parties. Réglages livrés (`rp=2`, n=200), 130 à 135 ms par décision comme gen8 :

| Joueur testé | Champ | Donnes | Score | Par taille (2, 3, 4, 5 j.) |
| --- | --- | --- | --- | --- |
| V | C | 920 (graine 601) | +4,9 ± 1,5 | +2,9, +5,7, +8,0, +2,9 |
| **VP** | C | 920 (graine 601) | **+7,9 ± 1,7** | +5,8, +12,0, +9,3, +4,6 |
| V | gen8 | 184, arrêt anticipé (graine 602) | +13,4 ± 3,9 | +16,3, +17,4, +9,2, +10,7 |
| **VP** | gen8 | 184, arrêt anticipé (graine 602) | **+11,1 ± 3,8** | +12,0, +13,8, +12,0, +6,7 |

Mêmes donnes : VP − V +3,0 ± 2,2 parmi des C (920 donnes) ; −2,3 ± 5,2 parmi des gen8 (184
donnes), indécis. Les deux arènes face à gen8 se sont arrêtées au premier regard possible (gain
de 3 points plutôt que 0) ; un score lu à l'arrêt est un peu optimiste. **Deuxième progrès
d'affilée : environ +5 à +8 points sur C, plus de +10 sur gen8, à toutes les tailles.** Avec
quatre fois plus de parties, la politique des nouvelles parties aide nettement face à C ; face à
gen8, trop peu de donnes pour départager. Retenu : **VP = `lab/nets/gen9.bin`** (V :
`gen9v.bin`), candidat pour le jeu et enseignant de la génération suivante. Pertes de validation
finales, V puis VP : politique 1,574 et 1,578, valeur 0,984 et 0,983, Brier 0,325 et 0,324.
**gen9 livré dans le jeu le 20 octobre à 11 h 58.**

**gen10 lancé le 20 octobre** : 20 000 parties de gen9 et positions rejugées par gen9
(`lab/runs/gen10-data.sh`, `lab/data/gen10` et `gen10-judged`), puis deux branches depuis gen9
(`lab/runs/gen10-train.sh`, recette de VP, 24 000 lots) : **F, complet** (anciennes sources au
quart, comme gen9, plus `p2`, `gen9`, `gen10`) et **W, fenêtre** (`p2`, `gen9`, `gen10` seulement).
Question : les anciennes parties (fins de Normal, réseaux faibles) servent-elles encore ? Sans
elles, disque et mémoire cessent de croître à chaque génération (71 Go à télécharger pour F,
d'où `--disk 200`). Arènes aux réglages livrés : F et W parmi des gen9 (graine 701, 920 donnes,
différence W − F) ; face à Difficile et à Normale (graines 702 et 703), pour voir si W a oublié
les positions de jeu plus faible, que des arènes entre réseaux ne montreraient pas.

**Données de gen10 produites le 20 octobre** (272 min, 3,46 $ ; graines 10 000 000 à 10 000 039,
jugement 10 100 000 à 10 100 039) : 1,54 million de positions (8,4 Go) et 888 000 rejugées
(5,0 Go), dans le bucket et sur mini ; 4 min 35 par lot de 500 parties (C : 4 min 20), 2 min 05
de jugement. Erreur de valeur sur les 3 968 positions de P0, chaque réseau jugé par ses propres
fins, gen8 → C → gen9 :

| Positions | Face aux fins de Normal | Face aux fins du réseau |
| --- | --- | --- |
| Toutes | 6,2 → 6,5 → 7,5 | 9,3 → 8,4 → **6,6** |
| 2, 3, 4, 5 joueurs (réseau) | | 11,8, 9,1, 9,3, 8,0 → 12,5, 8,0, 7,9, 6,6 → 10,5, 6,5, 5,9, 4,7 |
| Dernière manche | 2,4 → 2,7 → 3,4 | 2,5 → 2,7 → 3,3 |
| 1 à 2 manches restantes | 3,6 → 4,0 → 4,7 | 5,5 → 5,2 → 4,9 |
| 3 à 5 manches restantes | 6,6 → 6,9 → 8,1 | 10,3 → 9,0 → 7,3 |
| 6 manches ou plus | 7,3 → 7,6 → 8,6 | 10,6 → 9,9 → 7,3 |
| Pièces finales par joueur | 4,1 → 5,0 → 7,4 | 11,1 → 8,9 → 7,2 |

gen9 apprend nettement mieux les fins de son propre jeu (−2,7 pièces depuis gen8, à toutes les
tailles et surtout tôt dans la partie) et s'éloigne de celles de Normal, comme attendu. Seule
régression : la dernière manche, presque déterministe, passe de 2,5 à 3,3 face aux deux juges.
Les résultats de parties, bruités, ont pu dégrader la précision de fin de partie : à surveiller
(piste P5, entrées de la valeur).

**gen10 entraîné le 20 octobre** (`lab/runs/gen10-train.sh`, résultats `lab/runs/gen10` sur mini ;
98 min de c7a.16xlarge avec un disque de 200 Go, 1,26 $ ; 74 Go de données sur le disque, 41 Go
de mémoire pour F et 32 Go pour W). F : 7,21 millions de positions de 133 788 parties ; W :
5,49 millions de 45 000 parties ; 24 000 lots chacun, 24 min à deux. Réglages livrés, 920 donnes
par arène, 140 à 147 ms par décision :

| Champ (graine) | F, complet | W, fenêtre | W − F, mêmes donnes |
| --- | --- | --- | --- |
| gen9 (701) | +3,3 ± 1,5 | **+6,3 ± 1,7** | **+3,0 ± 2,2** |
| Difficile (702) | +46,7 ± 1,3 | +46,7 ± 1,4 | +0,1 ± 1,7 |
| Normale (703) | +56,4 ± 1,0 | +52,1 ± 1,1 | **−4,3 ± 1,3** |

Par taille (2, 3, 4, 5 j.) : parmi des gen9, F +5,1, +3,8, +3,0, +1,3 et W +6,3, +6,8, +6,1,
+6,2 ; W − F face à Normale −0,2, −4,6, −7,7, −4,6 (victoires 80,4 % contre 85,2 %). **Sans
l'historique, W joue mieux entre réseaux, aussi bien face à Difficile, mais exploite moins bien
Normale à 3 joueurs et plus** : l'oubli redouté, dans sa forme la plus faible. Les positions de
`label` (parties de Normal, valeur des fins de Normal) apprennent à jouer contre des adversaires de
ce genre ; les parties entre réseaux ne le font pas. gen9 n'a pas été mesuré face à Normale : on
ne sait pas si W recule par rapport à gen9 ou si F progresse. Les pertes de validation ne se
comparent pas (ensembles différents). Réseaux : `lab/nets/gen10f.bin` et `gen10w.bin`.

**Partie perdue contre la paix des Diplomates** (API LLM, 20 octobre, `gpt-6-astra`, 110 à 65) :
à 2 joueurs, des Tritons Diplomates ont imposé la paix à l'Expert à chacun des 10 tours, ce que la
règle officielle permet (elle reste telle quelle). L'Expert le voit : au tour 3, il s'estime à
3,6 % de chances avec la paix, 67 % sur la même position sans elle. Mais, perdu, il cesse de
jouer : ses Squelettes, 11 jetons en main, finissent trois tours sans conquérir deux entrées
libres à 1 jeton. À 200 itérations, tous les coups valent environ 3,6 % ; gen8 donne 41 % d'a
priori à « finir les conquêtes » (les attaques sur les Tritons, qu'il préfère, sont interdites).
Sur cette position, à 200 itérations : gen8 et A finissent ; B conquiert de justesse ; **C et D
conquièrent nettement**, leur valeur jugeant « finir » plus bas d'un point environ. À 2 000
itérations, gen8 conquiert aussi. Mesurer l'avance par rapport à celle prévue à la racine (comme
KataGo) ne suffit pas à 200 itérations : essai abandonné. La nouvelle valeur corrige ce défaut.

Juge réseau sur les mêmes 3 968 positions de l'IA Normal qu'en P0 (`lab noise --judge`, même
graine ; `lab/runs/p2/noise-net.txt` contre `lab/runs/p0/noise-gen8.txt`), avance en pièces,
juge Normal → juge réseau :

| Positions | Dispersion des cibles | Bruit d'un jugement | Erreur de gen8 |
| --- | --- | --- | --- |
| Toutes | 15,2 → 12,4 | 1,76 → 0,90 | 6,2 → 9,3 |
| Dernière manche | 19,0 → 19,0 | 0,13 → 0,14 | 2,4 → 2,5 |
| 1 à 2 manches restantes | 17,2 → 15,2 | 0,47 → 0,37 | 3,6 → 5,5 |
| 3 à 5 manches restantes | 15,9 → 10,4 | 1,31 → 0,70 | 6,6 → 10,3 |
| 6 manches ou plus | 12,2 → 9,6 | 2,50 → 1,24 | 7,3 → 10,6 |

Les fins du réseau sont deux fois moins bruitées (seuls les dés et les pioches sont tirés au
sort) et resserrent les écarts. gen8, entraîné sur les fins de Normal, s'écarte de 9,3 pièces de
celles du réseau, 10 à 11 dès qu'il reste 3 manches : les deux juges ne donnent pas la même
valeur, et gen8 a appris la première. Sur les pièces finales de chaque joueur, son erreur (11,1)
égale la dispersion (11,2) : en partie un écart de niveau, les fins du réseau ne finissant pas
avec le même nombre de pièces. Lequel des deux juges enseigne la meilleure valeur pour jouer : les
arènes des branches C et D le diront.

1. Pilote de **5 000 parties**, équilibrées entre tailles, enseignant gen8 figé `rp=1` : en
   auto-apprentissage, les adversaires sont le réseau, que `rp=2` simulerait en Normal. P1 peut
   changer ce réglage pour les générations suivantes. Garder d'abord 200/50 itérations et 50 %
   de décisions complètes pour ne pas changer simultanément la qualité des recherches.
   Enregistrer visites, résultats finaux, chances de victoire estimées par la recherche pour
   chaque joueur, identifiant, graine et actions de chaque partie (pour réencoder ou rejuger
   plus tard). Réserver des parties entières à la validation. Durée estimée : environ 40 min sur
   64 cœurs (20 000 parties `rp=2` : 66 min ; `rp=1` 2,4 fois plus cher), à mesurer.
2. Nouvelles cibles de valeur, deux sources :
   - **positions jugées par le réseau** : `lab label` avec un juge dont les fins sont jouées par
     la politique de gen8 (`rt=0`) au lieu de Normal, sur des positions de jeu fort (pilote ou
     `label --agent`). Mêmes positions et même moyenne que l'ancienne recette : seul le juge
     change. C'est le test direct de l'hypothèse 1. Les fins vont jusqu'au bout : aucune erreur
     de valeur ne s'auto-entretient. Coût à mesurer ; si 32 fins sont trop chères, en garder 8 ;
   - **résultats finaux du pilote**, victoire et pièces, avec `q-weight=0` : seulement 5 000
     résultats indépendants, bruités ; en complément, pas comme seule source.

   Les anciennes cibles `label` restent leurs moyennes Normal. Si les deux sources échouent,
   essayer des fins tronquées jugées par un réseau figé, ou un mélange avec les estimations de
   recherche, en gardant une composante de résultats observés.
3. Depuis le même checkpoint gen8, entraîner les quatre branches suivantes. Les nouvelles
   sources **s'ajoutent** aux anciennes : les remplacer comparerait surtout des volumes de
   données (le pilote donne environ 370 000 positions, contre des millions d'anciennes).

| Branche | Cible de politique | Cible de valeur |
| --- | --- | --- |
| A, témoin réentraîné | Ancienne | Ancienne |
| B | Ancienne + nouvelle | Ancienne |
| C | Ancienne | Ancienne + nouvelle |
| D | Ancienne + nouvelle | Ancienne + nouvelle |

« Ancienne politique » : sources de gen8 ; « ancienne valeur » : `label1` à `label4`.
Ajouter si nécessaire `:value` : seul `:policy` existe actuellement. Changer une source ne
signifie pas geler une tête : le tronc reste partagé. Fixer proportions des pertes, nombre de
mises à jour, exemples par cible et sélection des checkpoints pour rendre les branches
comparables. Mesurer aussi gen8 intact pour détecter l'effet du simple réentraînement.

**Mesures :** arène P1, à la taille de confirmation dès le départ : les écarts attendus, de 1 à
3 points, sont sous le seuil de la présélection. Calibration de victoire et erreur en pièces sur
des parties fortes indépendantes, par phase et taille. La victoire reste le critère principal.

**Décision :** confirmer la branche gagnante, idéalement avec une seconde graine d'entraînement,
puis étendre à 20 000 parties (environ 2 h 40 en `rp=1`). Si le pilote est indécis, examiner variance et apprentissage
avant d'étendre. Si B gagne et C/D reculent, travailler les cibles de valeur ; si C/D gagnent,
renouveler la valeur dans la boucle suivante. Si aucune ne progresse, diagnostiquer via P4/P5
plutôt que répéter mécaniquement des générations.

Après un gain confirmé, comparer l'auto-jeu courant à un mélange comprenant quelques anciennes
générations figées, avec proportions fixées pour l'essai. Cette petite population vise à
préserver plusieurs styles, inspirée de la ligue
[AlphaStar](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/).
Son efficacité dans SmallMobile reste à mesurer.

### P3 — Mesurer l'intérêt de l'adaptation aux adversaires

**Revue le 20 octobre :** pas de type de place (décision du 20 octobre) ; le profil d'un joueur
vient seulement de ses coups publics dans la partie. Forme retenue : un historique résumé par
joueur en entrée du réseau (attaques subies et portées, déclins, prudence des placements…),
plutôt que les croyances bayésiennes de `opponents.rs` ci-dessous. Inutile en auto-apprentissage
pur (tous les joueurs sont le même réseau) : à entraîner sur des parties à tables variées (P6),
réencodées depuis leurs fichiers `.moves`. Côté jeu, le serveur devra transmettre l'historique à
`expert`. Le plan ci-dessous reste comme mémoire de l'ancienne forme.

**Hypothèse :** reconnaître un style pendant la partie améliore les simulations sur les tables mixtes.

- Brancher des croyances propres à chaque observateur et chaque partie dans l'arène, alimentées
  par les actions publiques. Réinitialiser entre parties ; utiliser un hasard distinct pour
  l'observation afin de ne pas perturber celui des décisions.
- Comparer `rp=1/2`, mélange fixe 50/50, croyances actuelles, et croyances utilisant une
  distribution de coups du réseau avec confiance limitée.
- Le modèle actuel récompense seulement le coup préféré. Plusieurs ordres de conquête peuvent
  se valoir et les coups sont corrélés : quelques accords peuvent produire trop de confiance.
  Calibrer température/plancher de probabilité sur des parties réservées.
- Témoin : le modèle par type de place (P0, mesuré en P1). Pour les bots, les croyances doivent
  faire au moins aussi bien ; leur vrai intérêt concerne les humains. Ne pas supposer qu'un
  humain joue comme le réseau.
- Mesurer victoires, coût d'observation, évolution des croyances et prédiction par adversaire.
  Une meilleure prédiction sans gain en jeu ne suffit pas.

**Décision :** retenir l'adaptation si elle améliore les tables mixtes sans régression au-delà
de la tolérance convenue sur les champs homogènes. Sinon garder mélange ou réglages spécialisés.
Intégration serveur après validation : croyances par partie, transmission des coups observés
à `expert`, diagnostics dans les archives.

### P4 — Améliorer les cibles enseignées par la recherche

**Hypothèse :** moins de cibles, issues d'une recherche plus forte, valent mieux à coût égal.
Même enseignant et même modèle adverse dans les deux recettes :

| Recette | Décisions complètes enregistrées | Autres décisions | Itérations moyennes théoriques |
| --- | --- | --- | --- |
| Référence | 50 % à n=200 | n=50 | 125 |
| Candidate | 10 % à n=800 | n=50 | 125 |

Mesurer le temps réel, le nombre de décisions et le coût d'entraînement : l'égalité de calcul
n'est qu'approximative, et la candidate garde moins de positions. Comparer les réseaux à budget
total comparable ; garder la source de valeur identique pour isoler les cibles de politique.

La distinction entre diversité des parties et qualité des cibles vient de
[KataGo, section 3.1](https://arxiv.org/html/1902.10565v5#S3.SS1). Si l'essai réussit, étudier
séparément l'exploration forcée et la correction des cibles de visites de sa section 3.2,
pour limiter l'apprentissage des coups essayés seulement par exploration.

**Décision :** retenir le meilleur progrès en jeu par heure de calcul. Un meilleur accord avec
l'enseignant sans gain de victoire ne suffit pas.

### P5 — Plafonds structurels et efficacité du calcul

**Espace des coups :** constituer un ensemble de positions où placement, défense ou pouvoirs
comptent. Ajouter quelques variantes locales des plans de `brain/src/moves.rs` : déplacement
d'un jeton, autre région pour un héros/forteresse, autre répartition des campements. Comparer
leurs suites, puis la force en arène. Vérifier légalité, clés d'actions et compatibilité des
formats ; de nouvelles clés nécessitent un apprentissage.

**Entrées du réseau :** pour les manques vérifiés (hypothèse 5), construire d'abord des
positions qui ne diffèrent que par eux (paix entre deux adversaires, ordre des cartes rendues)
et vérifier qu'elles ont le même encodage ; mesurer leur fréquence et leur effet, puis
corriger, séparément des cibles auxiliaires (P7). N'encoder que de l'information publique :
l'ordre caché des pioches et les dés restent tirés au sort. Un nouvel encodage rend les anciens
tenseurs inutilisables ; les parties de `p2`, `gen9` et `gen10` se rejouent à l'identique
(fichiers `.moves` : donne, graines, actions) et se réencodent sans relancer l'Expert,
contrairement à `label`, `gen6` et `gen7`. Les nouvelles entrées démarrent à poids nuls, pour affiner depuis le dernier réseau.

**Audit fait le 20 octobre** (`lab audit --games lab/data/gen10/c00` : 500 parties, 76 074
décisions, 1 s sur mini). Comparaison de tout l'état du noyau (`kernel/src/state.rs`) avec
l'encodage, puis fréquence de chaque information publique absente, aux décisions :

| Le réseau ne voit pas… | 2 j. | 3 j. | 4 j. | 5 j. | Toutes |
| --- | --- | --- | --- | --- | --- |
| une paix entre deux adversaires | 0 % | 1,6 % | 4,0 % | 6,1 % | 3,4 % |
| quels peuples sont connus au fond de la pioche, bientôt au marché (≤ 2 cachés) | 8,8 % | 35,7 % | 17,2 % | 9,2 % | 17,5 % |
| quels pouvoirs sont connus au fond, bientôt au marché | 0 % | 0 % | 0,2 % | 0,1 % | 0,1 % |
| la file des défenses au-delà de la première | 0 % | 0 % | 0 % | 0 % | 0 % |
| l'âge du peuple actif d'un adversaire (choisi il y a 3 manches ou plus) | 24 % | 41 % | 47 % | 45 % | 41 % |
| (politique) à qui imposer la paix : la clé désigne la place absolue de la cible | 0,4 % | 1,0 % | 1,0 % | 0,9 % | 0,9 % |

Le réseau reçoit le nombre de peuples connus au fond, pas lesquels ni leur ordre ; rien des
pouvoirs connus. L'âge des peuples (`picked_round`) sert à l'IA Normale du jeu, pas au réseau.
Aucune collision : dans une même partie, deux positions publiquement différentes n'ont jamais
donné la même entrée (l'information manque, mais d'autres entrées changent avec elle). Tout le
reste de l'état public est encodé ; le type des places est exclu volontairement. Candidats à
ajouter : l'âge des peuples actifs, les prochains peuples connus de la pioche, la paix entre
adversaires ; et le contenu des pioches (peuples et pouvoirs encore à venir), public mais
seulement déductible par élimination sur tout le plateau. Côté politique, la clé de fin de tour
des Diplomates (`moves::finishes`) désigne la cible par sa place absolue, alors que le réseau voit
les joueurs relativement à lui : « la paix au joueur suivant » change de clé selon sa place, même
à 2 joueurs. À rendre relative (et à convertir dans les visites des données réencodées).

**Encodage v2 écrit le 20 octobre** (`brain/src/encode.rs`, test `version_2_inputs_follow_the_game`) :
106 entrées de la partie ajoutées après les 700 premières (`GLOBAL_V1`), soit 806 : âge du peuple
actif de chaque joueur (5), peuples de la pioche (14), les 3 prochains peuples connus de la pioche
(3 × 14), pouvoirs de la pioche (20), toutes les paix (diplomate × cible relatifs, 25). Clé de paix
des Diplomates relative au joueur qui agit (`moves::FINISH`). Un réseau de la version 1 (jusqu'à
gen10) lit les 700 premières entrées et voit ses sorties de paix traduites (`moves::relative_peace`) :
gen9 et gen10w cherchent exactement comme avant (mêmes empreintes `lab speed`). Données : format
`*.v4.bin` (5 866 octets par position) ; `lab reencode` y passe les `*.v3.bin` d'après leurs parties
(500 parties et leurs positions rejugées en 3 s ; test aller-retour octet pour octet). Python :
`model.widen` met un réseau de la version 1 aux 806 entrées (poids nuls : mêmes sorties, sauf les
clés de paix qui changent de sens) ; `train.py --old-inputs` met les nouvelles entrées à zéro, pour
un témoin entraîné sur les mêmes données. Rust et PyTorch concordent toujours (5·10⁻⁵). Contrôle de bout en bout : des
parties d'auto-apprentissage de gen10w jouées par le nouveau code sont identiques, octet pour
octet, aux mêmes parties jouées par l'ancien code puis réencodées.

**Tables variées écrites le 20 octobre** (P6 ; `lab selfplay --field 'A|B|…' --mixed 0.5`, test
`varied_tables_record_only_the_learner`) : dans une part des parties, l'IA qui apprend n'occupe
que certaines places (chacune avec une chance sur deux, au moins une), les autres étant tirées
parmi les adversaires donnés. Seules ses positions sont enregistrées, marquées « table variée »
(4e octet de l'enregistrement) ; le fichier `.moves` note qui jouait à chaque place (numéro dans
la liste, -1 pour l'IA qui apprend), pour l'analyse et pour l'historique des coups (P3). Les
autres parties restent identiques à l'auto-apprentissage pur. Pouvoirs connus et file des défenses : trop rares pour compter. Leur effet ne se
mesure qu'en entraînant avec et sans, sur les mêmes parties réencodées.

**Recherche :** si davantage d'itérations aide toujours peu après P2/P4, examiner le regroupement
en boucle ouverte. Une même suite d'actions peut mener à des marchés ou dés différents ; tester
sur un petit cas la séparation des observations publiques avant une refonte de l'arbre.

**Vitesse :** profiler avant de choisir entre cache des évaluations, réutilisation de l'arbre
entre conquêtes, inférence quantifiée/distillée et plusieurs cœurs. La réutilisation doit tenir
compte des observations et croyances mises à jour. Pour un cache exact, vérifier des décisions
inchangées à graine fixée ; pour une approximation, remesurer la force à temps égal.

**Fait le 20 octobre, sans rien changer aux résultats** (mesures sur mini, un cœur). Profil :
le réseau prenait 92 % d'une recherche `rp=1` (auto-apprentissage) et 65 % d'une recherche
`rp=2` (jeu, arènes ; l'IA Normale des fins simulées 26 %), à 145 µs par évaluation ; un tiers des
évaluations d'une recherche refaisaient une position déjà évaluée. Changements :
- cache des évaluations par recherche, et entre les 32 fins d'une position rejugée (le réseau
  répond toujours pareil à la même entrée) ; entre recherches successives, 5 % seulement de
  positions communes : sans intérêt ;
- réseau en AVX2 : multiplications matricielles par tuiles de 6 × 16 dans les registres, couches
  du vecteur global et têtes 8 sorties à la fois, politique par région 8 régions à la fois.
  Chaque nombre reçoit les mêmes opérations dans le même ordre : résultats identiques au bit
  près. 145 → 93 µs par évaluation.

| Mesure | Avant | Après |
| --- | --- | --- |
| Recherche `rp=1`, n=200 (`lab speed`, 48 positions) | 386 ms | 191 ms (×2,0) |
| Recherche `rp=2`, n=200, réglage livré | 170 ms | 110 ms (×1,55) |
| Parties d'auto-apprentissage (2 parties à 3 joueurs) | 3 positions/s | 6 positions/s |
| Rejugement par 32 fins du réseau (43 positions) | 24 s | 12 s (×2,0) |

Vérifié : `lab speed` donne les mêmes empreintes (décisions et valeurs au bit près) ; fichiers de
`lab selfplay` et `lab judge` identiques octet pour octet entre l'ancien et le nouveau code ;
test du calcul rapide contre le calcul simple (`Net::eval_portable`). Reste, pour le réglage du
jeu : l'IA Normale du noyau (environ 40 % d'une recherche `rp=2`), code partagé avec le jeu.
Regrouper les évaluations de plusieurs parties gagnerait peu désormais : les multiplications
matricielles (60 % d'une évaluation) traitent déjà toutes les régions d'une position à la fois,
près des limites du processeur.

**Faible priorité aujourd'hui :** réseau plus gros sur les mêmes cibles, nouvelles positions
jugées uniquement par Normal, nombreuses générations identiques, modèle humain appris sur les
quelques parties disponibles. Réexaminer si les nouvelles mesures le justifient.

### P6 — Entraîner face à une population d'adversaires

**Hypothèse (6) :** des parties d'entraînement face à une population d'adversaires donnent une
stratégie plus robuste que l'auto-apprentissage seul, sans les anciennes cibles de Normal.

- Recette candidate : la fenêtre de W (parties récentes de jeu fort), dont une part des tables
  mêle le réseau courant à des générations figées (gen8, C, gen9), à Normale et à Difficile.
  Valeur apprise des **vrais résultats** de ces parties ; volume borné comme pour W. Témoin : la
  fenêtre pure, à calcul total égal, mêmes cibles, même réseau, même nombre de mises à jour.
- Évaluer sur un panel tenu à l'écart (protocole) : gen10 parmi les réseaux, Difficile, Normale,
  tables mixtes, au moins un adversaire absent de l'entraînement.
- **Succès :** progrès sur le panel sans effondrement face à un style. Battre la génération
  précédente seule ne suffit pas.
- Ensuite seulement : des adversaires spécialisés contre le champion, faibles ailleurs mais
  révélateurs de ses failles (« exploiteurs » d'AlphaStar). Le réseau ne reçoit jamais
  l'identité ni le profil de ses adversaires (décision du 20 octobre) : il doit apprendre une
  stratégie qui tient face à tous.
- Piste distincte : garder une sélection d'anciennes positions en renouvelant leurs cibles,
  plutôt que leurs valeurs de Normal. Le recul de W face à Normale ne prouve pas que ces
  positions sont inutiles, seulement que leurs anciennes cibles freinent ailleurs.

### P7 — Cibles territoriales auxiliaires

**Hypothèse (7) :** des cibles locales rendent chaque partie plus instructive que victoire,
pièces et politique seules.

Le réseau prédit déjà les pièces à venir : un score de plus ne serait pas nouveau. Décomposer :
propriétaire de chaque région après une rotation complète, revenu au prochain décompte,
territoires gardés et jetons perdus par une armée. La victoire reste l'objectif ; des têtes
utilisées seulement à l'entraînement ne coûtent rien en jeu. Inspiration : la propriété du
plateau de KataGo (section 4.1), à adapter, car les pièces s'accumulent à chaque tour : la carte
finale seule ne suffit pas.

**Premier essai :** cibles tirées des parties rejouables (`p2`, `gen9`, `gen10`), d'abord la
propriété après une rotation et le revenu au prochain décompte ; avec et sans, mêmes parties,
mêmes mises à jour, mêmes pertes principales. Critère : la force en arène à temps égal ;
calibration et erreurs par région en diagnostic. Après l'audit de l'encodage, et séparément.

### Suite du plan après gen10 (20 octobre)

Ordre retenu après la revue du 20 octobre (pistes proposées par un autre modèle, vérifiées dans
le code et confrontées aux mesures ci-dessus) :

La revue proposait d'abord de brancher `rp=4` dans le jeu : écarté (décision du 20 octobre,
l'Expert ne sait pas contre qui il joue).

1. **Audit de l'encodage** (hypothèse 5, P5) : du code et des tests, sans AWS ; peut avancer
   pendant les campagnes.
2. **gen11, essai de population (P6)** contre la fenêtre pure, à calcul égal. Choisir d'ici là
   le réseau livré et l'enseignant entre gen9, `gen10w` et `gen10f` ; repère manquant : gen9
   face à Normale et Difficile.
3. **Cibles territoriales (P7)**, après l'audit.
4. **Placements** (P5, espace des coups) : variantes locales avant tout générateur appris de
   plans. Le placement des héros, forteresses et campements est fixe, et la défense n'a que deux
   répartitions candidates : la restriction est certaine, son coût en jeu non mesuré.
5. **Recherche par observations publiques** (P5, recherche) : sur des cas construits d'abord ;
   plus de branches peuvent diluer 200 itérations, à valider à temps égal.

Garder les résultats de parties fortes au cœur des cibles de valeur. Faible priorité inchangée :
réseau plus gros (latence sur mini ; l'échec passé portait sur les anciennes cibles), davantage
de fins moyennées du même réseau, générations identiques sans mesurer leur gain marginal.

## Les machines AWS

Règles d'usage (aucun calcul lourd sur mini, `lab/tools/cloud.sh`, une commande par étape avec
délai estimé, utilisateur informé à chaque étape, `cloud.sh status` après chaque calcul) :
voir `AGENTS.md`. Aucun calcul AWS sans son accord pour l'instant ; ce plan n'autorise pas une
dépense.

Configuration constatée le 19 octobre 2026, à revérifier avant de dépenser :

- SmallAI `631245464763`, profil `smallai`, région `eu-north-1` (Stockholm). Fonctionnalités
  avancées activées, Spot autorisé, quota Standard Spot 64 vCPU ; une seule `c7a.16xlarge`
  à la fois. Le DryRun a vérifié les permissions, pas la disponibilité réelle.
- `cloud.sh` impose Spot sans repli à la demande (`SPOT=0` refusé). Par défaut :
  `c7a.16xlarge`, 128 Gio de mémoire, disque gp3 de 100 Gio. Plafond de **1 $/h pour le calcul**,
  hors disque, IPv4, S3 et autres frais. Lire le prix Spot effectif avant de chiffrer une étape ;
  les anciennes générations à environ 10 $ utilisaient le tarif à la demande.
- Maximum 12 h par machine, extinction après 20 min sans calcul, destruction à l'extinction
  avec suppression du disque. Spot ponctuel, interruption avec destruction, aucun remplacement
  automatique.
- Interruptions, relevé du 19 octobre : AWS classe `c7a.16xlarge` à Stockholm dans la tranche
  « 5 à 10 % » (part des machines reprises sur le mois écoulé ; `c7a.8xlarge` : plus de 20 %).
  Les prix Spot ont doublé en un mois (0,35 → 0,77 à 0,89 $/h selon la zone) ; eu-north-1c a
  dépassé le plafond de 1 $/h du 13 au 16 octobre (1,064 $ au plus haut). `cloud.sh up` prend la
  zone la moins chère sous le plafond. Pour un calcul long, `job --save lab/X,...` garde ces
  dossiers dans le bucket (`s3://…/X`) : repris au lancement, copiés toutes les 10 min, à
  l'annonce d'une reprise (2 min avant) et à la fin. Découper le calcul en morceaux déplacés
  dans le dossier gardé une fois complets, et sauter ceux déjà présents : un calcul relancé
  reprend alors où il en était.
- Bucket `s3://smallai-lab-631245464763/data/`, rôle machine `smallai-worker`, politiques dans
  `lab/tools/aws/`. Synchroniser ensuite les résultats sur mini.

Budget global avant crédits `Credits-Semaine-19-25-Oct-2026-110USD`, compte de gestion
`051299608012` : **110 $ du 19 octobre 00:00 UTC au 26 octobre 00:00 UTC**, alertes à
55/80/100/110 $. Ce sont des alertes, pas une coupure automatique. Les anciens plafonds mensuels
ne s'appliquent plus. Dernier relevé du 19 octobre : 8,31 $ consommés sur la période,
123,23 $ de crédits restants estimés, soit 101,69 $ de dépenses supplémentaires dans ce budget
et 21,54 $ de réserve. Depuis ce relevé, estimations de `cloud.sh` : P0 0,87 $, essais Spot
0,02 $, P1 1,11 $, pilote P2 0,86 $, entraînement et arènes P2 1,54 $ (dont 0,10 $ perdus sur un
script non exécutable, et une reprise par AWS à 20 h 30, sans perte grâce à `--save`),
confirmation P2 1,19 $, données de gen9 3,31 $, entraînement et arènes de gen9 0,96 $,
données de gen10 3,46 $, entraînement et arènes de gen10 1,26 $, soit 14,58 $. **Montants datés : consulter coûts et crédits du compte de gestion
avant chaque campagne. Les crédits ne se renouvellent pas chaque semaine.**

Pour chaque étape autorisée : mesurer un petit échantillon sur AWS, estimer durée et coût total
(installation, données, calcul, évaluation), fixer le timeout, puis lancer l'étape complète.
Ne pas lancer toute la feuille de route d'un seul bloc. L'en-tête de `tools/cloud.sh` documente
`job --type ... --data ... --get ... -- 'commande'` et le rapatriement des résultats.

## Références et tenue du document

- **David J. Wu — Accelerating Self-Play Learning in Go (KataGo, 2019/2020).**
  [Article](https://arxiv.org/abs/1902.10565),
  [texte intégral](https://arxiv.org/html/1902.10565v5).
  Sections 3.1/3.2 : allocation variable de recherche, exploration et cibles de politique.
  Application : P4. La randomisation du budget existe déjà dans le labo ; ses réglages et la
  qualité des cibles restent à comparer dans ce jeu. Section 4.1 : propriété du plateau et score
  comme cibles auxiliaires. Application : P7.
- **Équipe AlphaStar — Mastering the real-time strategy game StarCraft II (2019).**
  [Présentation des auteurs](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/).
  « How AlphaStar is trained » : population de concurrents, versions figées, diversité des
  stratégies. Application : P6, population d'adversaires puis exploiteurs. Cela inspire un essai,
  sans garantie de progrès pour un jeu à 2–5 joueurs.

Après chaque expérience, mettre à jour la synthèse et le statut de l'étape concernée. Garder
dans `lab/runs/` scripts, résultats détaillés et manifeste : hypothèse, code/poids, sources,
graines, paramètres, budget, résultats par taille, incertitude et décision. `lab/runs/` est
ignoré par Git : sauvegarder ces fichiers et les autres artefacts sur mini et/ou S3, avec leur
emplacement consigné. Ne pas recréer une
chronologie exhaustive ici : conserver ce qui change une décision, les échecs qui évitent
de refaire un essai et les prochaines actions concrètes.
