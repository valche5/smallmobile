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

## Résultats de force

**Aucun tournoi ni entraînement lancé. Aucun gain revendiqué.** Création, tests, commit et push
uniquement, conformément à la demande. Les futures campagnes se feront sur une machine de calcul
autorisée, jamais sur mini.
