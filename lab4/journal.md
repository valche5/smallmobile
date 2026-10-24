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

## Résultats de force

**Aucun tournoi complet ni entraînement lancé. Aucun gain revendiqué.** Les mini-arènes du pilote
ne mesurent pas la force.
