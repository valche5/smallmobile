# IA SmallMobile : résultats utiles et plan d'expériences

État au 19 octobre 2026. Ce document remplace le journal chronologique ; les anciens détails
restent dans Git. Les résultats ci-dessous ont été mesurés. Les étapes P0 à P5 sont un plan
à exécuter : la réécriture du document ne vaut pas réalisation de ces étapes.

## Objectif et priorité

Construire l'IA la plus forte possible sur les cartes générées, à **2, 3, 4 et 5 joueurs**.
Le labo peut chercher longtemps ; la version du jeu doit aussi être évaluée au temps de
réflexion disponible sur mini. Conserver ces deux mesures de force.

**Prochain levier : apprendre politique et valeur sur du jeu fort, après avoir fiabilisé les
mesures et choisi une référence parmi les Expert.** Une génération identique à gen8 a peu de
rendement. L'adaptation aux adversaires mérite un essai séparé ; sa réussite ne doit pas
conditionner le lancement de ce nouvel apprentissage.

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
- Expert livré : `lab/nets/gen8.bin`, via `target/release/expert` et `src/server/expert.ts`.
  `bun run kernel:build` construit le noyau WebAssembly et cet exécutable.
- Archives réelles : `data/games/`, graines, actions et temps de décision ; résumé par
  `bun lab/tools/games.ts`. Les quelques parties humaines disponibles ne suffisent pas encore
  à entraîner un modèle des humains.

Réglage livré, à garder comme témoin figé :

```text
az:net=lab/nets/gen8.bin,n=200,w=0.5,c=2.5,roll=1,depth=3,rp=2
```

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

Les générations récentes gardent 50 % des décisions cherchées à fond, les autres à 50 itérations,
avec bruit 0,25 pour les recherches complètes. Générer 20 000 parties prenait environ une heure
sur 64 cœurs ; entraîner gen8 environ 78 minutes. Ce sont des repères pour l'ancienne recette,
pas une estimation des fins entièrement jouées par le réseau.

Dans `train.py`, `:policy` exclut la valeur d'une source ; `@0.5` sous-échantillonne les blocs,
ce n'est pas un poids de perte. `--q-weight` mélange résultat et pièces attendues par la
recherche, sans modifier la cible de victoire. `--lazy` lit les données à la demande.

## Hypothèses à départager

1. **Ancrage sur Normal.** La valeur apprend ses fins de partie ; la politique apprend des
   recherches dont les adversaires des fins simulées jouent Normal. C'est une explication
   plausible du plateau, pas une causalité isolée. `label --agent` ne change que les positions
   sources : le joueur qui les juge reste Normal.
2. **Adaptation adverse.** `brain/src/opponents.rs` contient deux modèles, Normal et le coup
   préféré du réseau, avec mise à jour bayésienne. `search_against` reçoit les croyances,
   mais leur observation n'est pas branchée dans l'arène ou le jeu. `rp=3` via `search` est
   donc un mélange 50/50 sans adaptation. Aucun gain adaptatif n'est mesuré.
3. **Cibles de politique.** Une recherche à 200 itérations peut bien jouer sans suffisamment
   améliorer la politique qu'elle doit enseigner. Comparer les recettes à coût total comparable.
4. **Coups absents.** `brain/src/moves.rs` fixe certains placements de pouvoirs et limite les
   plans de redéploiement/défense. La recherche ne peut choisir un plan absent. L'importance
   de cette restriction n'a pas été mesurée.

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
- Présélection : 300 à 500 donnes au total par confrontation. Confirmation du candidat retenu :
  2 000 à 4 000 nouvelles donnes au total, soit 500 à 1 000 par taille. Fixer ce nombre avant
  le calcul ; ne pas répéter jusqu'à obtenir un résultat favorable.
- Promotion : gain sur le score principal choisi à l'avance, intervalle à 95 % de la différence
  au-dessus de zéro ; examiner chaque taille et le coût. Tolérance proposée de régression :
  2 points par taille/étalon. Si l'incertitude ne permet pas d'exclure une régression supérieure,
  le résultat est **indécis**, pas une preuve d'équivalence.
- Tests ciblés pour chaque modification, puis vérifications Rust du projet avec timeout.
  Entraînement et campagnes de parties se font sur AWS.

## Plan priorisé

### P0 — Fiabiliser la recherche et les mesures

**Préalable aux nouveaux résultats.**

1. Échantillonner un monde avant l'expansion initiale dans `brain/src/search.rs`. Actuellement,
   elle précède `determinize` et sa première fin simulée peut utiliser les vrais dés/pioche
   dans le labo. `Expert::decide` protège déjà son entrée. Ajouter un test d'indépendance aux
   dés cachés et vérifier arène/auto-apprentissage. Incidence historique inconnue : remesurer
   les témoins après correction.
2. Ajouter un identifiant stable de partie aux données ou à un index associé. Remplacer les
   blocs arbitraires de 256 positions de `lab/py/data.py`. Réserver une validation stable entre
   générations ; le checkpoint initial ne doit pas l'avoir déjà apprise. Pour les anciens
   fichiers, régénérer une validation indépendante plutôt que prétendre retrouver exactement
   des frontières non enregistrées.
3. Exporter les résultats par donne/place et calculer les différences selon le protocole.
   Préparer des tables mixtes : l'arène actuelle ne propose qu'un héros contre un champ homogène.

**Livrable :** tests ciblés passants, partitions vérifiables, résultats individuels reproductibles,
témoins remesurés. Toutes les étapes suivantes utilisent ce code corrigé.

### P1 — Choisir une référence de recherche face aux joueurs forts

**Hypothèse :** le meilleur réglage gen8 dépend du champ adverse et du budget de réflexion.

- Poids figés gen8 ; comparer `roll=1,rp=2,n=200,depth=3`,
  `roll=1,rp=1,n=200,depth=3` et `roll=0,n=800`.
  Fixer `w=0.5,c=2.5,rt=0`, sans bruit à la racine. Ces n sont des points de départ,
  pas une affirmation d'égalité de coût.
- Adversaires figés : Difficile, gen8 `rp=2`, gen8 `rp=1`. Après présélection, ajouter des
  tables mixtes avec composition et rotation des places consignées. Confronter aussi les
  candidats retenus entre eux, dans les deux sens à plusieurs joueurs.
- Mesurer les temps sur un court échantillon AWS et régler n pour une seconde comparaison
  à temps moyen comparable, sur la même machine ; contrôler aussi le p95.
- Ensuite seulement, essayer profondeur 5 pour le meilleur hybride. À 4/5 joueurs, profondeur 3
  ne couvre pas une rotation complète : tester cette rotation est un essai distinct, avec surcoût.

**Décision :** figer une référence forte pour l'apprentissage et une référence au temps du jeu.
Si aucun réglage ne domine, garder plusieurs témoins et publier le compromis. Un gain face à
Difficile seul ne suffit pas pour changer l'Expert livré.

### P2 — Faire apprendre politique et valeur sur du jeu fort

**Priorité d'entraînement. Hypothèse :** renouveler les deux cibles permet de dépasser le plateau.
Dépend de P0/P1, pas de P3.

1. Pilote de **5 000 parties**, équilibrées entre tailles, enseignant gen8 figé, `rp=1` ou
   réglage fort retenu par P1. Garder d'abord 200/50 itérations et 50 % de décisions complètes
   pour ne pas changer simultanément la qualité des recherches. Enregistrer visites, résultats
   finaux et identifiants de partie.
2. Nouvelles cibles de valeur : résultats finaux de ces parties, victoire et pièces.
   Fixer `q-weight=0` dans toutes les branches pour ce premier essai ; les anciennes cibles
   `label` restent leurs moyennes Normal. Si ces résultats sont trop bruités, essayer ensuite
   des fins jugées par un réseau figé ou un mélange avec les estimations de recherche.
   Garder une composante de résultats observés pour ne pas entretenir les erreurs du réseau.
3. Depuis le même checkpoint gen8, entraîner les quatre branches suivantes.

| Branche | Cible de politique | Cible de valeur |
| --- | --- | --- |
| A, témoin réentraîné | Ancienne | Ancienne |
| B | Nouvelle | Ancienne |
| C | Ancienne | Nouvelle |
| D | Nouvelle | Nouvelle |

« Ancienne politique » : sources de gen8 ; « ancienne valeur » : `label1` à `label4`.
Ajouter si nécessaire `:value` : seul `:policy` existe actuellement. Changer une source ne
signifie pas geler une tête : le tronc reste partagé. Fixer proportions des pertes, nombre de
mises à jour, exemples par cible et sélection des checkpoints pour rendre les branches
comparables. Mesurer aussi gen8 intact pour détecter l'effet du simple réentraînement.

**Mesures :** arène P1 ; calibration de victoire et erreur en pièces sur des parties fortes
indépendantes, par phase et taille. La victoire reste le critère principal.

**Décision :** confirmer la branche gagnante, idéalement avec une seconde graine d'entraînement,
puis étendre à 20 000 parties. Si le pilote est indécis, examiner variance et apprentissage
avant d'étendre. Si B gagne et C/D reculent, travailler les cibles de valeur ; si C/D gagnent,
renouveler la valeur dans la boucle suivante. Si aucune ne progresse, diagnostiquer via P4/P5
plutôt que répéter mécaniquement des générations.

Après un gain confirmé, comparer l'auto-jeu courant à un mélange comprenant quelques anciennes
générations figées, avec proportions fixées pour l'essai. Cette petite population vise à
préserver plusieurs styles, inspirée de la ligue
[AlphaStar](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/).
Son efficacité dans SmallMobile reste à mesurer.

### P3 — Mesurer l'intérêt de l'adaptation aux adversaires

**Hypothèse :** reconnaître un style pendant la partie améliore les simulations sur les tables mixtes.

- Brancher des croyances propres à chaque observateur et chaque partie dans l'arène, alimentées
  par les actions publiques. Réinitialiser entre parties ; utiliser un hasard distinct pour
  l'observation afin de ne pas perturber celui des décisions.
- Comparer `rp=1/2`, mélange fixe 50/50, croyances actuelles, et croyances utilisant une
  distribution de coups du réseau avec confiance limitée.
- Le modèle actuel récompense seulement le coup préféré. Plusieurs ordres de conquête peuvent
  se valoir et les coups sont corrélés : quelques accords peuvent produire trop de confiance.
  Calibrer température/plancher de probabilité sur des parties réservées.
- Pour les bots connus, ajouter un témoin utilisant leur type pour choisir le modèle de
  simulation. Ne pas supposer qu'un humain joue comme le réseau.
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

**Recherche :** si davantage d'itérations aide toujours peu après P2/P4, examiner le regroupement
en boucle ouverte. Une même suite d'actions peut mener à des marchés ou dés différents ; tester
sur un petit cas la séparation des observations publiques avant une refonte de l'arbre.

**Vitesse :** profiler avant de choisir entre cache des évaluations, réutilisation de l'arbre
entre conquêtes, inférence quantifiée/distillée et plusieurs cœurs. La réutilisation doit tenir
compte des observations et croyances mises à jour. Pour un cache exact, vérifier des décisions
inchangées à graine fixée ; pour une approximation, remesurer la force à temps égal.

**Faible priorité aujourd'hui :** réseau plus gros sur les mêmes cibles, nouvelles positions
jugées uniquement par Normal, nombreuses générations identiques, modèle humain appris sur les
quelques parties disponibles. Réexaminer si les nouvelles mesures le justifient.

## Les machines AWS

**Aucun calcul lourd sur mini**, qui sert la production. Arène, génération, étiquetage et
entraînement passent par `lab/tools/cloud.sh`. Une commande par étape, avec timeout estimé
avant lancement et marge raisonnable ; informer l'utilisateur à chaque étape. Aucun calcul
AWS sans son accord pour l'instant ; ce plan n'autorise pas une dépense.

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
  automatique. Sauvegarder les résultats utiles dans S3 au fil des étapes.
- Bucket `s3://smallai-lab-631245464763/data/`, rôle machine `smallai-worker`, politiques dans
  `lab/tools/aws/`. Synchroniser ensuite les résultats sur mini.
- Ne pas modifier `cloud.sh` sur place pendant qu'il tourne : écrire une copie puis la renommer.
  Après chaque calcul : `cloud.sh status`, vérifier l'absence de ressources restantes, donner
  la dépense estimée par le script. La facturation AWS arrive avec retard.

Budget global avant crédits `Credits-Semaine-19-25-Oct-2026-110USD`, compte de gestion
`051299608012` : **110 $ du 19 octobre 00:00 UTC au 26 octobre 00:00 UTC**, alertes à
55/80/100/110 $. Ce sont des alertes, pas une coupure automatique. Les anciens plafonds mensuels
ne s'appliquent plus. Dernier relevé du 19 octobre : 8,31 $ consommés sur la période,
123,23 $ de crédits restants estimés, soit 101,69 $ de dépenses supplémentaires dans ce budget
et 21,54 $ de réserve. **Montants datés : consulter coûts et crédits du compte de gestion
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
  qualité des cibles restent à comparer dans ce jeu.
- **Équipe AlphaStar — Mastering the real-time strategy game StarCraft II (2019).**
  [Présentation des auteurs](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/).
  « How AlphaStar is trained » : population de concurrents, versions figées, diversité des
  stratégies. Application : mélange d'adversaires après P2. Cela inspire un essai, sans garantie
  de progrès pour un jeu à 2–5 joueurs.

Après chaque expérience, mettre à jour la synthèse et le statut de l'étape concernée. Garder
dans `lab/runs/` scripts, résultats détaillés et manifeste : hypothèse, code/poids, sources,
graines, paramètres, budget, résultats par taille, incertitude et décision. Sauvegarder les
artefacts hors Git sur mini et/ou S3, avec leur emplacement consigné. Ne pas recréer une
chronologie exhaustive ici : conserver ce qui change une décision, les échecs qui évitent
de refaire un essai et les prochaines actions concrètes.
