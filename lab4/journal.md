# Labo 4 : budget, largeur et horizon de recherche

Ouvert le 24 octobre 2026 à la demande de l'utilisateur, depuis lab3 au commit
`217e7437d814bdc3d7510dca07ba99dfc03621eb`. Lab3 est figé ; son c130 reste l'Expert du jeu.

Objectif : construire un expert de diagnostic plus coûteux et démontrer une marge de progression
avant de reprendre l'apprentissage. Plan des prochaines expériences : **[lab.md](lab.md)**.

## Création

- Copie autonome du cerveau, des outils CPU/Vulkan et Python de lab3. Les boucles et campagnes
  historiques ne sont pas reprises ; aucune donnée d'entraînement copiée.
- `nets/c130.bin` : poids inchangés, SHA-256
  `e534cc84c801fc3c5b7d377cd8b67a50a4e1a11b65b1f6644c32b20eeb52cba3`.
  `nets/gen0.bin` ne sert qu'aux tests.
- Recherche configurable avec `n` et `candidates=16|32|64|all` (tout entier positif est accepté).
  Défaut : 400 simulations, 16 candidats. Budget strict même si les candidats sont plus nombreux.
- Trois variantes préparées : A = 400/16, B = 3 200/16, C = 3 200/tous. Même réseau, mêmes
  règles de recherche, même objectif victoire/pièces.
- Diagnostics de couverture et d'horizon par taille, affichés en analyse/vitesse/arène et écrits
  à côté des résultats d'arène dans `.search.tsv`. Aucun compteur ne tire de nombre aléatoire.
- Recherche élargie : envois plafonnés à 16 feuilles pour respecter l'anneau GPU hérité de lab3,
  sans limiter le nombre de candidats. Test avec un évaluateur à capacité limitée, sans GPU.
- Pas de prolongation des feuilles ni de changement d'objectif à ce stade : essais conditionnels
  décrits dans le plan, à partir des diagnostics.

## Vérifications

- `timeout 600s cargo test --workspace --no-default-features` : passé.
- Après le découpage des envois élargis, `timeout 600s cargo test -p smallmobile-lab4
  -p smallmobile-brain4 --no-default-features` : 7 tests passés. Ils couvrent l'équivalence à
  lab3 sur c130 (2 à 5 joueurs, avec et sans bruit), les budgets et paramètres invalides,
  la couverture des coups, le classement hors top 16, l'export des compteurs, les franchissements
  de tours et la limite de 16 évaluations simultanées sans changement de résultat.
- `timeout 600s cargo clippy --workspace --all-targets --no-default-features` : passé ;
  avertissements préexistants de code inutilisé dans lab3, aucun dans lab4.
- Formatage vérifié pour les deux nouveaux paquets. Le contrôle global de formatage signale
  des écarts préexistants dans lab3, laissé figé. `git diff --check` : passé.
- Poids c130 vérifiés identiques par SHA-256. Aucun fichier de production reconstruit.
- Vérifications sur mini, en CPU seulement : `glslc` absent, compilation et exécution Vulkan
  non vérifiées ici. L'évaluateur à capacité limitée teste le découpage des recherches larges
  sans matériel GPU ; ce n'est pas un test des shaders ni du pilote.

## Pilote sur mini (24 octobre)

Lancé sur mini à la demande de l'utilisateur, pour un premier regard : CPU seul, `nice 19`,
4 fils sur 6, binaire à part (`CARGO_TARGET_DIR=target/lab4`, sans toucher aux programmes de la
production). Commit `835ad79`.

- `speed`, 2 positions par taille, un cœur : A 102 ms par recherche, B 730 ms (×7,2), C 740 ms.
  Aucune de ces 8 positions n'a plus de 16 coups : B et C y jouent au bit près pareil.
- Mini-arènes, graine 41001, 8 donnes (28 parties), 4 fils : 559 s (B contre A) et 551 s
  (C contre A). Scores **+8,8 ± 24,3** et **+7,7 ± 16,7** : bruit pur à cette taille, aucune
  conclusion de force. Temps par décision entière du candidat ~950 ms, p95 ~2,6 s (4 fils
  partagés). À ce débit, 400 donnes prendraient ~8 h sur mini.
- Diagnostics du candidat (B ; C quasi identique) :

| Joueurs | Racines > 16 coups | Profondeur moy. (A → B) | Tour fini (A → B) | Prochain tour atteint (A → B) |
| --- | --- | --- | --- | --- |
| 2 | 9 % | 6,2 → 9,8 | 57 → 77 % | 21 → 35 % |
| 3 | 11 % | 6,2 → 11,0 | 51 → 74 % | 2,5 → 11 % |
| 4 | 12 % | 6,6 → 10,7 | 49 → 66 % | 0,5 → 1,6 % |
| 5 | 10 % | 6,8 → 11,5 | 54 → 68 % | 0,6 → 2,1 % |

- Largeur : C explore 1 828 coups hors du top 16 sur ses racines larges, et n'en choisit
  **aucun** (0 sur 2 369 recherches). B laisse 6 à 12 % des coups légaux sans visite.
- Horizon : même à 3 200 simulations, à 4 et 5 joueurs, moins de 2 % des simulations atteignent
  le prochain tour du joueur racine ; un tiers ne finissent même pas le tour courant. Le budget
  s'use dans les décisions jeton par jeton du tour en cours, ce qui appuie l'étape 3 du plan
  (prolonger les feuilles) plus que l'élargissement.

## Présélection sur le PC (24 octobre)

Reprise sur GrosseBoite (i7-12700KF, RX 9070 XT), à la demande de l'utilisateur, de la session
qui tournait sur mini. Commit `294ab4b`, poids c130 SHA-256 `e534cc84…` (vérifié), binaire
`cargo build --release -p smallmobile-lab4` (Vulkan).

- `speed`, 4 positions par taille, un cœur : A 61 ms par recherche, B 394 ms (×6,4), C 399 ms.
- Pilote GPU, B contre A, 40 donnes, graine 40002 (hors des graines réservées) : 37 s ; temps
  par décision entière du candidat 413 ms, p95 1,4 s (20 fils sur le GPU).
- Présélection, 400 donnes (100 par taille), graine 41001, `--device vulkan`, 20 fils :

| Duel | Durée | Score (95 %) | 2 joueurs | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| B contre A | 300 s | +2,0 ± 2,3 | +3,0 | −1,5 | +5,0 | +1,4 |
| C contre A | 292 s | +2,8 ± 2,3 | +3,0 | +1,7 | +3,4 | +3,0 |
| C contre B | 897 s | +0,6 ± 2,1 | +1,5 | +1,5 | −1,0 | +0,4 |

  Différence appariée C − B face à A, mêmes donnes : +0,8 ± 2,3 (indécis). C choisit hors du
  top 16 dans 13 recherches sur ~110 000 (4 565 à 35 471 coups visités hors du top 16 selon la
  taille) : C et B jouent presque pareil, l'écart de leurs scores face à A est du bruit.
- Budget seul (B) : indécis, au-dessous du seuil pratique de +3. Largeur (C) : pas d'apport
  mesurable face à B. Diagnostics de B sur ces parties : prochain tour du joueur racine atteint
  par 29 % des simulations à 2 joueurs, 9,6 % à 3, 3,4 % à 4.
- Ce qui use la profondeur (parties de selfplay de c130) : ~11,7 décisions par tour, dont 4,6
  redéploiements d'un jeton et 2,5 conquêtes ; 7,9 coups légaux par décision en moyenne
  (médiane 6). À 4 joueurs, le prochain tour est à ~40 décisions. L'arbre déplie chaque ordre de
  pose des jetons redéployés séparément (4 jetons sur 5 régions : 625 suites pour 70
  répartitions) : piste suivante, un ordre imposé des jetons (une seule suite par répartition).

## Jetons redéployés dans l'ordre (24 octobre)

Idée de l'utilisateur : l'ordre de pose des jetons ne compte pas, l'arbre ne devrait pas le
déplier. Option `tokens=ordered` (défaut `free`, la recherche de lab3) : au redéploiement, un
jeton ne va que sur une région de numéro au moins celui du précédent (`Redeploy::last_token`,
`Knowledge::ordered_tokens` posé par la recherche sur sa copie). Une seule suite de coups par
répartition ; un jeton qui n'a plus qu'une région possible devient un coup forcé. Test
`ordered_tokens_reach_the_same_redeployments_once_each` : sur 60 parties au hasard, à chaque pose
assez petite pour être énumérée, les mêmes positions de fin de pose sont atteintes, une fois
chacune dans l'ordre. L'équivalence de A à lab3 tient (option désactivée).

- `speed`, 24 positions par taille, un cœur : D (`n=400,tokens=ordered`) 81 ms par recherche
  contre 60 pour A (+35 % : les simulations vont plus loin, plus d'actions jouées chacune).
- Diagnostics, A → D : tour courant terminé 45 → 68 % (2 joueurs), 50 → 66 % (3), 41 → 58 % (4),
  40 → 53 % (5) ; prochain tour atteint 11 → 20 % (2), 1,7 → 3,8 % (3), 4,1 → 4,1 % (4), 0 → 0 (5).
- **D contre A, 920 donnes (graine 42004) : −1,0 ± 1,4** (2 joueurs −0,1, 3 −0,2, 4 −2,2, 5 −1,6),
  226 s. Aucun gain malgré l'arbre plus profond. Hypothèse : l'ordre imposé change le sens des
  priors de c130, appris sur des poses libres (il met d'abord le jeton là où il compte le plus ;
  dans l'ordre, ce premier choix interdit ensuite les régions de numéro plus petit). Le tester
  vraiment demanderait des cibles apprises dans l'ordre, ou une fusion des positions identiques
  qui garde les poses libres.

## Jetons fusionnés (24 octobre)

Pour garder la pose libre que c130 a apprise : option `tokens=merged` (défaut désactivé). Les
jetons sont posés librement, mais les suites qui posent les mêmes jetons depuis la même position
mènent au même nœud (`Tree::merged`, empreinte de la suite sans son ordre : nœud de départ, somme
d'une constante par région, hasard du dernier coup). À la racine, le coup qui commence la suite
entre dans l'empreinte : les sous-arbres des coups de la racine restent à part (une simulation
chacun par tour). Les statistiques restent sur les coups : un nœud partagé cumule les visites de
ses parents. Tests : l'encodage d'une même position ne dépend pas de l'ordre de ses jetons
(`ordered_tokens_reach_the_same_redeployments_once_each`) ; à une racine de redéploiement, autant
de simulations vont plus loin (`merged_tokens_go_deeper`) ; A inchangé.

- `speed`, 24 positions par taille : E (`n=400,tokens=merged`) 77 ms par recherche (A 60, D 81).
  Tour courant terminé, A → E : 45 → 55 % (2 joueurs), 50 → 55 % (3), 41 → 53 % (4), 40 → 46 % (5) ;
  prochain tour atteint 11 → 15 %, 1,7 → 3,7 %, 4,1 → 4,5 %, 0 → 0.
- **E contre A, 920 donnes (graine 42005) : +0,4 ± 1,5** (2 joueurs +0,1, 3 −1,7, 4 +1,1, 5 +2,1),
  227 s.

Bilan des essais sur la recherche à poids fixes : 8 fois plus de simulations (B, +2,0 ± 2,3 sur
400 donnes), toute la largeur (C, rien de plus que B), un arbre plus profond dans le tour par
l'ordre imposé (D, −1,0 ± 1,4) ou la fusion des jetons (E, +0,4 ± 1,5) : rien de net. Finir plus
souvent son tour ne paie pas ; le prochain tour reste hors de portée à 4 et 5 joueurs.

## Critère de la valeur et diagnostic de la valeur (24 octobre)

Remarque de l'utilisateur : le mélange 50 % victoire, 50 % avance en pièces n'est peut-être pas le
bon, on veut la victoire. Option `win=` (poids de la victoire dans la valeur que la recherche
maximise, défaut 0,5 = lab3). Duels contre A (`win=0.5`), n = 400, 920 donnes chacun :

| Poids de la victoire | Graine | Score (95 %) | 2 joueurs | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0,75 | 42006 | +1,3 ± 1,4 | −0,2 | +2,3 | +1,8 | +1,3 |
| 0,75 | 42008 | +1,3 ± 1,5 | +2,6 | +0,6 | +0,9 | +1,2 |
| 1 | 42007 | +1,2 ± 1,5 | +0,2 | +2,1 | +3,1 | −0,4 |

Les deux duels à 0,75 réunis (1 840 donnes, moyenne pondérée par taille) : **+1,3 ± 1,0**,
positif à toutes les tailles (+1,2 à +1,5). Petit gain, mais le premier significatif de lab4. À 1,
l'avance en pièces s'effondre (−3,6 pièces à 2 joueurs) sans perte de victoires.

Diagnostic de la valeur (`py/stages.py`), parties de validation de lab3 c126–c130 (217 578
positions, 6 289 parties), entropie croisée de la victoire :

| Joueurs | Manches restantes | Uniforme | Pièces actuelles | Réseau c130 | Recherche (n = 400) |
| --- | --- | --- | --- | --- | --- |
| 2 | tout | 0,693 | 0,631 | 0,268 | 0,262 |
| 3 | tout | 1,099 | 1,107 | 0,950 | 0,940 |
| 4 | 6 et + | 1,386 | 1,413 | 1,354 | 1,351 |
| 4 | 4–5 | 1,386 | 1,436 | 1,291 | 1,285 |
| 4 | 2–3 | 1,386 | 1,410 | 1,128 | 1,112 |
| 4 | 0–1 | 1,386 | 1,351 | 0,630 | 0,588 |
| 5 | 6 et + | 1,609 | 1,628 | 1,585 | 1,582 |
| 5 | tout | 1,609 | 1,615 | 1,333 | 1,317 |

- À 3 à 5 joueurs, avec 4 manches ou plus à jouer, la valeur ne sait presque rien (à 4 joueurs :
  1,354 contre 1,386 sans rien savoir). Les pièces actuelles ne disent rien avant la fin.
- La recherche à 400 simulations ne fait guère mieux que le réseau (à 4 joueurs, 1,144 contre
  1,157 en tout) ; son avance n'apparaît qu'à la fin (0,588 contre 0,630). Apprendre sa valeur au
  réseau ne lui apprendrait presque rien de neuf.

## Apprentissage avec la victoire seule (24 octobre)

Demande de l'utilisateur : reprendre l'apprentissage depuis c130 avec un selfplay qui vise la
victoire seule (`win=1`) ; les pièces guidaient un réseau jeune, plus forcément un réseau mûr.

État des lieux d'un pas de la boucle de lab3 (moyennes de c121 à c130, deep7 : 128x7, n = 400,
25 000 parties, en continu) : **~568 s par pas**.

| Étape | Durée | Part | Détail |
| --- | --- | --- | --- |
| Parties (selfplay) | 387 s | 68 % | 863 690 positions enregistrées, 2 234 par s ; GPU à 100 %, ~300 W (au plafond), processeur ~80 % |
| Apprentissage | 146 s | 26 % | chargement 8 s, 6 748 lots de 512 en 130 s (~19 ms chacun, calcul du GPU), validation 4 s |
| Duel contre c000 | 35 s | 6 % | n = 200, arrêt séquentiel (184 à 460 donnes) |

Dans un cycle, une décision sur 8 est cherchée à fond (400 simulations, enregistrée), les autres
à 50 : ~6,9 millions de décisions non forcées, ~345 millions de simulations complètes et ~302
millions rapides (environ moitié-moitié du calcul).

Boucle `lab4/loop.sh` : reprise de celle de lab3, GPU et étapes successives seulement, depuis la
fenêtre d'un autre essai ; `WIN` règle l'objectif du selfplay (recherches complètes et rapides) et
des duels, joués contre `REF` avec le même objectif. Essai `w1` : fenêtre de deep7 liée dans
`data/w1`, c130 et son état d'optimiseur dans `runs/w1`, `WIN=1`, n = 400, taux 1e-4, en continu,
duels contre c130 à `win=1`, pas 131 à 150 (service `lab4-w1`, lancé à 23:02). Essai à blanc
(64 parties, `runs/w1test`) avant le lancement : les trois étapes s'enchaînent.

Essai `w1` arrêté après c131 (à la demande de l'utilisateur, pour l'essai suivant) ; les parties
de c132 sont jouées et gardées. Duels de c131 (win = 1) : contre c130 à win = 1, −0,1 ± 2,1 ; contre
c130 à win = 0,5, +1,2 ± 2,1 (368 et 460 donnes, intervalles à 95 %). Duels de la boucle désormais
contre la référence à son objectif d'avant (`REF_WIN`, 0,5). Suivi de la valeur sur des positions
fixes (`py/track.py`, parties de validation de c126–c130) : c130 et c131 identiques, valeur bien
calibrée (perte et entropie des prédictions presque égales dans chaque groupe).

## La tête des propriétaires (24 octobre)

Question de l'utilisateur : la tête des propriétaires (à qui appartiendra chaque région au prochain
tour du joueur, comme la propriété de KataGo ; apprise seulement, Rust ne la calcule pas) pèse
près d'un quart de la perte (1,5 × 0,50 contre 1,42 pour la politique et 1,02 pour la valeur).
`train.py --own-weight`, `lab4/own.sh` : depuis c130 et son optimiseur, 3 pas d'apprentissage sur
la fenêtre de deep7 (c099…c130, mêmes graines), poids 1,5 (témoin), 0,5 et 0.

- Valeur et politique sur les positions fixes de `track.py` : les trois identiques à c130 au
  millième près ; seule la tête elle-même se dégrade sans entraînement (validation 0,512 → 0,526).
- Duel, poids 0 contre 1,5, n = 400, 920 donnes (graine 42011) : **−0,3 ± 1,5**.
- Sur un réseau mûr et en 3 pas, cette tête n'aide ni ne gêne : son effet, s'il y en a, se joue
  pendant l'apprentissage depuis le début (le tronc l'apprend tôt). La retirer n'apporterait rien
  ici ; le tester vraiment demanderait un apprentissage depuis zéro.

## Retirés de lab4 : la tête des propriétaires et l'apprentissage en continu (25 octobre)

À la demande de l'utilisateur, après l'essai ci-dessus et ceux de lab3 :

- **Tête des propriétaires** retirée partout dans lab4 : le selfplay ne calcule plus à qui
  appartient chaque région au tour suivant (`selfplay::owners`), les `.pos` perdent ce champ
  (6 162 octets par position), la forme compacte n'écrit plus `owner.npy`, le réseau n'a plus que
  la politique et la valeur (victoire et pièces), la perte n'a plus ce terme. Les réseaux de lab3
  (c130) se lisent sans elle (`model.load` écarte `own.*`) ; les anciens cycles gardent leur
  `owner.npy`, que plus rien ne lit.
- **Apprentissage en continu** retiré (état d'AdamW et moyenne des poids gardés d'un pas à
  l'autre, taux constant) : dans lab3, pas mieux qu'un cycle par pas. Chaque pas repart d'un AdamW
  neuf, taux maximal 1,5e-4 en un cycle (`loop.sh`, `LR`), le réglage de c041–c050 de lab3.
- Un pas d'apprentissage réel depuis c130 (fenêtre de deep7) : **103 s** au lieu de ~130 (la tête
  par région et la moyenne des poids tous les 8 lots coûtaient ~20 % du pas) ; validation politique
  1,261, valeur 1,029 (1,258 et 1,027 en continu : l'optimiseur neuf remonte au taux maximal).
  Le réseau écrit se lit dans Rust (`lab4 speed`).
- Tests : `test_train.py` (la forme compacte de Rust redonne les `.pos` sans le champ, un pas
  d'apprentissage écrit un réseau qui a appris, c130 se lit sans la tête), tests Rust et clippy de
  lab4 passent.

## Plafond de la valeur (25 octobre)

`lab4 ceiling` (`src/ceiling.rs`) : 300 positions de parties neuves (100 à 3, 4 et 5 joueurs,
graine 11), jouées par c130 (n = 400) jusqu'au début d'une manche tirée au hasard ; depuis chacune,
32 suites jusqu'au bout par c130, chacune dans un monde tiré à nouveau (pioches cachées, dés à
venir). 338 s sur le GPU. Analyse `py/ceiling.py` : entropie croisée de la victoire, uniforme,
valeur de c130 à la position, et plafond = la répartition des vainqueurs des 31 autres suites
(a priori de Dirichlet 0,5 ; un peu pessimiste, le vrai plafond est un peu plus bas).

| Joueurs | Manches restantes | Suites | Uniforme | Réseau | Plafond | Écart comblé |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 6+ | 1 568 | 1,099 | 1,072 | 1,063 | 74 % |
| 3 | 4–5 | 640 | 1,099 | 1,050 | 0,976 | 40 % |
| 3 | 2–3 | 736 | 1,099 | 1,004 | 0,848 | 38 % |
| 3 | 1 | 256 | 1,099 | 0,808 | 0,683 | 70 % |
| 4 | 6+ | 1 056 | 1,386 | 1,360 | 1,341 | 57 % |
| 4 | 4–5 | 896 | 1,386 | 1,326 | 1,218 | 36 % |
| 4 | 2–3 | 864 | 1,386 | 1,119 | 0,923 | 58 % |
| 4 | 1 | 384 | 1,386 | 0,811 | 0,673 | 81 % |
| 5 | 6+ | 864 | 1,609 | 1,561 | 1,577 | (≥ 100 %) |
| 5 | 4–5 | 864 | 1,609 | 1,476 | 1,370 | 56 % |
| 5 | 2–3 | 768 | 1,609 | 1,383 | 1,254 | 64 % |
| 5 | 1 | 704 | 1,609 | 0,900 | 0,639 | 73 % |
| 3 / 4 / 5 | tout | 3 200 chacun | | 1,031 / 1,220 / 1,350 | 0,966 / 1,114 / 1,237 | 51 / 61 / 70 % |

- **Le début de partie est surtout de la chance** : à 6 manches ou plus, même une valeur parfaite
  pour ces joueurs ferait à peine mieux que l'uniforme (4 joueurs : 1,341 contre 1,386), et c130 y
  est déjà presque. Rien à gagner là.
- **La marge est au milieu et à la fin** : 0,07 à 0,20 d'entropie croisée de 4 à 2 manches de la
  fin, jusqu'à 0,26 à 5 joueurs dans la dernière manche (0,900 contre 0,639), là où l'issue dépend
  du jeu concret plus que du hasard. C'est là qu'une meilleure valeur, ou une recherche plus
  profonde, peut payer.
- Limites : 100 positions par taille, moins dans chaque tranche (bruit) ; le plafond vaut pour
  ces joueurs-là (c130 à n = 400) ; la valeur du réseau est lue au début d'une manche, pour tous
  les joueurs.

## Résultats de force

**Aucun tournoi complet ni entraînement lancé. Aucun gain revendiqué.** Les mini-arènes du pilote
ne mesurent pas la force.
