# Lab5 : quelles occasions c130 manque-t-il ?

Base : **lab3/c130**, depuis `167ba76`, sans les variantes de lab4. Le cerveau, les coups,
l'encodage et les poids sont ceux de lab3. Lab3 et lab4 sont figés ; lab3 reste l'Expert du jeu.
Les règles communes sont dans [labtools/README.md](../labtools/README.md).

## Question mesurée

Depuis une position, remplacer **un coup** de c130 par un autre, puis laisser tous les joueurs
continuer avec c130 : est-ce que le joueur qui décide gagne davantage ? Toutes les suites vont
jusqu'au résultat réel du jeu, égalités partagées selon le noyau. Le critère de sélection est la
victoire seule ; le moteur qui propose le coup et continue les parties garde le mélange
victoire/pièces de lab3 et 400 simulations, sans exploration d'entraînement.

Ce diagnostic mesure une amélioration locale sous une continuation fixée. Il ne mesure ni un
« pourcentage du jeu résolu », ni le gain d'un nouvel Expert en arène. Un résultat positif donne
une marge exploitable sur ces positions. Un résultat négatif ne borne pas le meilleur jeu : les
alternatives sont limitées, c130 continue après le coup, et les plans qui demandent plusieurs
décisions différentes peuvent rester invisibles.

## Trois étapes indépendantes

1. `positions` : une partie source neuve par position, jouée par c130 à toutes les places.
   Répartition à poids égaux par nombre de joueurs et tiers de partie (`early`, `middle`, `late`).
   `--positions N` signifie **N par cellule**, donc `6 × N` positions avec `--players 2,4`.
   Dans chaque partie, une manche du tiers et un joueur sont tirés au hasard. Les choix de ce
   joueur dans cette manche (défenses hors tour comprises) sont échantillonnés par réservoir,
   séparément par phase ; une phase présente est tirée uniformément, puis son choix retenu.
   Les jetons de redéploiement ne dominent donc pas le lot par leur nombre. Si le joueur n'a
   aucun choix, l'étape signale cette position et s'arrête : aucune substitution cachée.
2. `select` : toutes les alternatives sont évaluées sur les mêmes mondes et graines de recherche.
   Le coup de c130 est toujours inclus ; déclin et fin des conquêtes aussi lorsqu'ils sont légaux.
   Puis viennent les coups aux meilleurs a priori, diversifiés par sorte de coup et propriétaire
   de la région ciblée. S'il faut tronquer, une place est réservée à un autre coup tiré au hasard.
   Au plus `--candidates K` coups au total. Le meilleur taux de victoire de sélection est retenu ;
   c130 gagne les égalités, puis les autres sont départagés par clé croissante.
3. `confirm` : **nouvelles graines**, uniquement le coup retenu et celui de c130. Le choix reste
   figé, même s'il perd lors de cette étape. Quand c130 a été retenu, son gain vaut exactement
   zéro et une seule suite par monde suffit. Les pertes et ces zéros entrent dans le score.

Chaque suite redétermine les pioches cachées et les dés. La recherche des joueurs n'accède jamais
au futur du monde réalisé : elle refait ses propres tirages comme dans lab3. Les alternatives
partagent les mêmes tirages initiaux et graines de recherche ; après divergence de leurs coups,
elles peuvent consommer ces tirages différemment. L'appariement réduit du bruit sans l'annuler.

## Lecture du résultat

Pour chaque position : moyenne de `victoire(coup retenu) − victoire(coup c130)` sur les paires de
confirmation. Le score principal est la moyenne à poids égaux des cellules joueurs × tiers.
On conserve aussi les scores par cellule, les fréquences des phases et les décisions concrètes.

`py/probe.py` rééchantillonne les **positions entières** à l'intérieur de chaque cellule pour un
intervalle bootstrap à 95 %. Une position vient d'une seule partie source indépendante ; ses
centaines de suites ne comptent pas comme autant de positions indépendantes. Le pilote à une
position par cellule ne reçoit aucun intervalle. Avec peu de positions, les intervalles restent
approximatifs. Le classement des « meilleurs écarts » est exploratoire, sans déclaration de
significativité individuelle : on ne sélectionne pas une deuxième fois sur la confirmation.

La taille de la campagne sera fixée après le pilote, selon son coût et la précision souhaitée,
avant de regarder ses confirmations. Ne pas arrêter dès qu'un résultat devient favorable.
Si des gains nets apparaissent, examiner les positions, puis construire et confirmer un agent
qui les exploite sur de nouvelles parties et plusieurs adversaires. Ce sera une étape distincte.

## Pilote prêt, aucune campagne autorisée par ce document

Premier pilote proposé : 2 et 4 joueurs, **1 position par cellule** (6 au total), jusqu'à 6 coups,
32 suites de sélection par coup, 128 suites de confirmation par coup, c130 à 400 simulations,
graine `51001`. Maximum : 1 152 suites de sélection et 1 536 de confirmation. Les parties sources
ne sont jouées que jusqu'à la manche choisie. Ce pilote sert au débit, au coût par phase et à la
vérification des sorties ; il ne mesure pas la force. Ces durées sont inconnues à la création.

À exécuter sur une machine de calcul autorisée. **Aucune campagne sur mini.** AWS : accord de
dépense et crédits à revérifier selon `labtools/README.md`. Un PC doit également être choisi
explicitement. Une commande par étape, timeout estimé avant chaque lancement ; ne pas lancer
les trois étapes d'un seul bloc. `probe.sh` refuse de lancer une campagne sur l'hôte `mini`.

Sur la machine de calcul, après compilation (budget initial de compilation : 900 s) :

```sh
timeout 900 cargo build --release -p smallmobile-lab5 --no-default-features
# Première étape seulement ; 900 s est un plafond initial à adapter à la machine autorisée.
RUN=pilot STAGE=positions LIMIT=900 lab5/probe.sh
```

Avant `select`, estimer son délai à partir d'un petit essai borné ou des premières positions du
pilote ; avant `confirm`, utiliser le débit mesuré de `select`, ventilé par tiers. Les délais ne
sont pas des prédictions. Le lanceur exige `LIMIT` et ne passe pas à l'étape suivante.

```sh
# Exemples : remplacer <secondes_estimees> avant exécution.
RUN=pilot STAGE=select LIMIT=<secondes_estimees> lab5/probe.sh
RUN=pilot STAGE=confirm LIMIT=<secondes_estimees> lab5/probe.sh
timeout 60 python3 lab5/py/probe.py lab5/runs/pilot
timeout 30 target/release/lab5 probe-show --out lab5/runs/pilot --position 0
```

Pour Vulkan : compiler avec les fonctionnalités par défaut (`glslc` récent, prenant en charge
`GL_EXT_bfloat16` ; 2026.4 validé, 2025.2 insuffisant), puis `DEVICE=vulkan`
pour **toutes** les étapes du lot. CPU et GPU ont des arrondis différents : dossiers séparés.
`THREADS`, `CONCURRENT`, `PLAYERS`, `POSITIONS`, `CANDIDATES`, `SELECT`, `CONFIRM`, `SIMS`, `SEED`
et `NET` règlent le lanceur ; les valeurs par défaut ci-dessus forment le pilote. Les réglages de
calcul du cerveau sont ceux de lab3, les variables d'environnement portent le préfixe `LAB5_`.
`LAB5_INSTANT` est refusé pour ce diagnostic.

Le parallélisme se répartit entre positions, puis les suites d'une position sont groupées sur son
fil. Le pilote à six positions utilise donc au plus six fils CPU, même sur une machine plus grande.
Les journaux donnent la durée de chaque position et son tiers : en tenir compte pour extrapoler
le coût d'un lot qui occupera davantage de cœurs. Une reprise qui saute des positions n'est pas
une mesure de débit.

AWS, exemple de **la première étape seulement**, après accord et estimation de l'installation :

```sh
LAB=lab5 timeout --kill-after=60s 2400 labtools/cloud.sh job --type c7a.16xlarge \
  --save lab5/runs/pilot --get lab5/runs/pilot -- \
  'RUN=pilot STAGE=positions LIMIT=900 lab5/probe.sh'
timeout 120 labtools/cloud.sh status
```

Les deux autres étapes reprennent le même dossier via `--save`, avec leurs propres délais
estimés. Toujours vérifier `status` et noter le coût après chaque calcul, même interrompu.
Le code générique de `cloud.sh` connaît déjà `LAB=lab5` ; il n'a pas été modifié.

## Reprise et traçabilité

- `manifest.tsv` : empreintes SHA-256 du code Rust/shaders/noyau et des poids, graines, nombres
  de positions/suites/candidats, simulations et backend. Une reprise incompatible est refusée.
- `pNNNNNN.position.tsv` : carte, graine, joueur, manche, phase, historique de clés permettant de
  reconstruire exactement la position et les connaissances publiques ; coup de c130, candidats,
  descriptions de tous les coups et estimations initiales de recherche.
- `pNNNNNN.select.tsv`, `.confirm.tsv` : chaque suite, son coup, son monde, sa victoire et son
  avance en pièces. Le choix issu de la sélection est enregistré puis vérifié à la confirmation.
- Fichiers écrits atomiquement **par position et par étape** ; une interruption ne perd que les
  positions en cours. Relancer la même commande valide et saute celles qui sont terminées.
  Les fichiers `.tmp` ne comptent pas comme résultats. Un verrou interdit deux écrivains.
- Le lanceur conserve l'empreinte du binaire, la machine, le commit si disponible et les journaux
  de chaque étape. Le manifeste reste la référence sur le code réellement compilé, y compris
  avant un commit. `invocations.tsv` note les démarrages, durées et paramètres de parallélisme.

Les outils hérités `arena`, `speed`, `analyze` et `selfplay` restent disponibles. Aucune boucle
d'entraînement historique, donnée de lab3/lab4 ni variante de lab4 n'est importée. Les fichiers
Python de lecture/contrôle du réseau sont conservés ; ils demandent PyTorch et NumPy si utilisés.
Le diagnostic et son analyse n'en ont pas besoin : `py/setup.sh` ne les installe pas.

## Vérifications locales, sans campagne

```sh
timeout 600 cargo test --workspace --no-default-features
timeout 600 cargo clippy --workspace --all-targets --no-default-features
timeout 30 python3 -m unittest discover -s lab5/py -p 'test_*.py'
```

Les tests nouveaux comparent le cerveau à lab3/c130, vérifient les graines indépendantes, les
vrais résultats terminaux, l'absence de dépendance au monde caché d'origine, le parcours complet
des trois étapes, la relecture Python, la reprise, le refus des fichiers incohérents et la stabilité
avec un autre parallélisme CPU. Les petites parties de fixture utilisent gen0 et une simulation.
