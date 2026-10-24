# Lab4 : construire un meilleur professeur à partir de c130

Plan ouvert le 24 octobre 2026. État de l'implémentation et vérifications : [journal.md](journal.md).
Protocole commun : [labtools/README.md](../labtools/README.md).

## Question et hypothèses

Lab3 plafonne : c130 contre c110 = +0,7 ± 1,4 point après vingt cycles. Cela ne mesure pas la
distance au meilleur jeu possible. Avant de relancer l'apprentissage, chercher un expert plus
fort avec les mêmes poids, quitte à dépenser beaucoup plus de calcul par décision.

Trois freins possibles : budget insuffisant, bons coups exclus des 16 candidats initiaux,
horizon absorbé par les décisions jeton par jeton. Les expériences ci-dessous les séparent.
Une recherche plus coûteuse n'est pas présumée meilleure : le duel doit le démontrer.

## Base figée et variantes prêtes

- Poids : `lab4/nets/c130.bin`, copie exacte de lab3 c130 (128 × 7). Aucun entraînement au départ.
- Témoin A : `az:net=lab4/nets/c130.bin,n=400,candidates=16`.
- Candidat B, approfondi : `az:net=lab4/nets/c130.bin,n=3200,candidates=16`.
- Candidat C, élargi : `az:net=lab4/nets/c130.bin,n=3200,candidates=all`.
- La représentation des coups, l'information visible, les tirages de hasard, la sélection
  interne et le mélange 50 % victoire / 50 % avance transformée restent ceux de lab3.
- `candidates=N` permet aussi d'essayer 32 ou 64. `all` signifie tous les coups légaux à la
  racine si le budget le permet ; le nombre effectivement considéré est au plus `n`. À 3 200,
  le budget dépasse le nombre de clés de coups (608). Le budget de simulations n'est jamais
  augmenté implicitement. Cette réserve compte notamment pour le selfplay rapide à 50.
- Les évaluations sont envoyées par groupes d'au plus 16 feuilles, même avec `candidates=all`,
  pour respecter la réservation de mémoire GPU par partie. Tous les candidats du tour sont
  sélectionnés avant ces groupes ; leur découpage ne change pas la sélection.
- Les défauts de lab4 sont `n=400,candidates=16`, y compris si les options sont omises.

Lab3 reste figé et proposé dans le jeu. Lab4 n'est pas ajouté au sélecteur des Experts avant
validation. Les outils d'arène CPU/GPU, d'analyse de positions et d'entraînement sont repris ;
aucune ancienne donnée, campagne ou boucle automatique n'est reprise. Les variables GPU portent
le préfixe `LAB4_`.

## Ordre des expériences

### 0. Préparer la mesure

- [x] Vérifier sur des positions que le témoin A reproduit la recherche originale de lab3 :
  coups, visites, politique, valeurs et consommation du générateur aléatoire.
- [x] Instrumenter la couverture des coups et l'horizon, sans changer les tirages ou les choix.
- [ ] Sur la machine de calcul autorisée, mesurer un petit échantillon : coût par recherche,
  temps moyen et p95 par décision entière, mémoire, débit d'arène. Estimer chaque étape et fixer
  son timeout avant de la lancer. Les huit fois plus de simulations ne sont pas une mesure de temps.
- [ ] Figer commit, SHA-256 des poids, machine, commandes, graines et nombre de donnes dans le journal.

**Aucun tournoi ni entraînement sur mini.** Toute campagne AWS demande un accord de dépense,
la vérification des crédits et le passage par `LAB=lab4 labtools/cloud.sh` ; après chaque étape,
contrôler `cloud.sh status` et noter le coût. Aucun calcul lancé lors de la création de lab4.

### 1. Budget seul : B contre A

- [ ] Présélection sur 400 donnes au total, 100 par taille, graine réservée `41001`.
- [ ] Si le résultat est prometteur, confirmer sur 920 nouvelles donnes au total (230 par taille),
  graine réservée `42001`, sans arrêt anticipé. Fixer le nombre avant le lancement.
- [ ] Si 3 200 apporte un gain confirmé, essayer 12 800 simulations, toujours 16 candidats,
  avec une nouvelle présélection puis une nouvelle confirmation (`41002`, `42002`).

Critère principal : score de victoire, moyenne à poids égaux des quatre tailles. Une promotion
demande un intervalle à 95 % au-dessus de zéro, et l'examen des régressions par taille selon le
protocole commun. Un intervalle recouvrant zéro est indécis ; ne pas multiplier les essais sur les
mêmes graines jusqu'à obtenir un gain. Le seuil de pertinence pratique visé est +3 points.

### 2. Largeur : C contre A et B

- [ ] Mesurer C contre A sur les mêmes donnes de présélection que B contre A : différence appariée
  des deux candidats face au même témoin, analysée avec `labtools/score.py`.
- [ ] Jouer aussi C contre B, à budget égal, pour vérifier directement le bénéfice de la largeur.
- [ ] Confirmer le candidat retenu contre A sur une graine encore inutilisée (`42003`, 920 donnes).
- [ ] Si `all` dilue trop le budget, tester une largeur intermédiaire (32 puis éventuellement 64),
  une modification à la fois, avec de nouvelles graines réservées et un budget d'essais fixé.

Ne pas additionner les gains entre adversaires. À 3–5 joueurs, le duel standard est un candidat
face à une table d'adversaires identiques, avec rotation complète des places.

### 3. Comprendre l'horizon avant de modifier la recherche

- [ ] Comparer les diagnostics par nombre de joueurs ; examiner des positions de conquête,
  déclin, redéploiement et défense avec `analyze` pour isoler les phases problématiques.
- [ ] Rejouer les positions où le choix élargi sort du top 16 : bon coup oublié ou bruit ?
- [ ] Si le prochain tour reste rarement atteint, implémenter une variante séparée qui prolonge
  certaines feuilles avec c130 jusqu'au prochain tour du joueur racine, puis les évalue.
  Définir avant l'essai le budget des décisions intermédiaires, le nombre de suites et une limite
  de longueur ; conserver les nœuds de hasard et l'information publique. Ne pas révéler les
  futurs dés ou cartes au joueur simulé.
- [ ] Comparer cette variante au meilleur témoin précédent, puis à temps de calcul comparable.
  Les suites peuvent propager les erreurs de c130 : davantage d'horizon ne garantit pas un gain.

La prolongation des feuilles n'est **pas encore implémentée**. Si le coût vient surtout des
permutations de placements, étudier ensuite leur regroupement ou la réutilisation des états,
dans une expérience distincte avec des tests d'équivalence des coups.

### 4. Objectif et robustesse

- [ ] Après les essais budget/largeur, rendre configurable le poids victoire/pièces et comparer
  0,5 (actuel), 0,75, puis 1. Garder le réseau et le budget fixes. La tête de victoire peut être
  moins précise que celle des pièces : la victoire seule n'est pas présumée meilleure.
- [ ] Confirmer le meilleur candidat sur un panel tenu à l'écart de la sélection : lab2,
  Difficile, Normale et tables mixtes, avec résultats par taille et adversaire. Ne pas confondre
  une adaptation à c130 avec une amélioration générale.
- [ ] Mesurer séparément les temps CPU par décision entière sur un cœur. Les temps d'arène GPU
  incluent l'attente et l'avancement d'autres parties ; ils ne prédisent pas la latence du jeu.

### 5. Apprentissage, seulement après un professeur confirmé

- [ ] Produire des cibles avec le meilleur expert confirmé et comparer à une continuation témoin
  de c130, avec budget de calcul et protocole de sélection explicités.
- [ ] Tester séparément des parties mieux jouées dans leur ensemble : augmenter le budget des
  sept décisions sur huit actuellement jouées à 50 simulations. Ne pas confondre ce changement
  avec l'augmentation du budget des seules positions enregistrées.
- [ ] Comparer le réseau appris au même budget de jeu (400), sur des graines de confirmation neuves.

Si aucune recherche plus coûteuse n'améliore le jeu, examiner les erreurs de valeur et les données
avant d'augmenter encore la capacité. Ce résultat ne démontrerait pas que le jeu est résolu.

## Outils et diagnostics

Exemples à exécuter **sur la machine de calcul**, après l'estimation et l'autorisation de l'étape ;
ces commandes ne sont pas un lanceur automatique. Les délais ci-dessous sont des plafonds de
sécurité à ajuster au pilote. `--deals` compte les donnes totales, pas les donnes par taille.

```sh
timeout 900s cargo build --release -p smallmobile-lab4 --no-default-features
mkdir -p lab4/runs/search
timeout 120s target/release/lab4 speed --agent az:net=lab4/nets/c130.bin,n=3200,candidates=16 --positions 2 --seed 40001
timeout 3600s target/release/lab4 arena --hero az:net=lab4/nets/c130.bin,n=3200,candidates=16 --field az:net=lab4/nets/c130.bin,n=400,candidates=16 --players 2,3,4,5 --deals 400 --seed 41001 --out lab4/runs/search/b-v-a.tsv
timeout 30s python3 labtools/score.py lab4/runs/search/b-v-a.tsv --role 0
```

Pour GPU : compiler avec les fonctionnalités par défaut (`glslc` requis), puis ajouter
`--device vulkan` à l'arène. Le réseau est identique pour A, B et C.

Chaque arène écrit les résultats habituels et `F.search.tsv` à côté de `--out F`, avec une ligne
par rôle et taille. Les mêmes résumés sont affichés par `arena`, `speed` et `analyze` :

- `searches`, `simulations` : recherches avec un choix et descentes effectuées ; coups forcés exclus.
- `legal`, `visited` : sommes des nombres de coups légaux et de coups visités aux racines.
- `wide_roots` : recherches avec plus de 16 coups légaux.
- `visited_outside16`, `chosen_outside16` : coups explorés et décisions choisies hors des 16
  premiers logits du réseau, sans bruit, avec le départage des égalités du moteur.
- `depth_sum`, `max_depth` : profondeur en **choix non forcés**, pas en manches ; moyenne =
  `depth_sum / simulations`.
- `turn_finished` : simulations qui terminent le tour courant à la racine.
- `next_turn` : simulations qui atteignent le prochain tour du **joueur qui décide à la racine**.
  En défense hors de son tour, ce prochain tour peut arriver dans la même manche.
- `game_over` : simulations qui atteignent une fin de partie réelle.

Les trois derniers taux ont pour dénominateur `simulations`. Les compteurs incluent tous les
franchissements de la descente, même si celle-ci continue au-delà. Les recherches des fins de
tour sont comptées jeton par jeton ; les temps de décision de l'arène regroupent ces recherches.
Avec `--stop`, les diagnostics peuvent aussi inclure des recherches en vol de donnes finalement
écartées ; utiliser les arènes sans arrêt pour comparer précisément leurs distributions.

## Vérifications sans campagne

```sh
timeout 600s cargo test -p smallmobile-lab4 -p smallmobile-brain4 --no-default-features
timeout 600s cargo clippy --workspace --all-targets --no-default-features
```

Ces tests vérifient des positions, le budget et les compteurs ; ils ne mesurent pas la force.
