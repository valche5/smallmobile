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

## Résultats de force

**Aucun tournoi complet ni entraînement lancé. Aucun gain revendiqué.** Les mini-arènes du pilote
ne mesurent pas la force.
