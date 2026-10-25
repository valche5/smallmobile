# Labo 5 : mesurer les coups manqués par c130

Ouvert le 25 octobre 2026 à la demande de l'utilisateur, depuis **lab3** au commit
`167ba76`. Lab4 est figé. Protocole et commandes : [lab.md](lab.md).

## Base

- Copie autonome et élaguée de lab3 : cerveau, moteur CPU/Vulkan, arène, lecture des parties et
  outils de contrôle du réseau. Aucune variante de recherche, cible TD, suppression de tête ou
  boucle d'entraînement de lab4 n'est reprise.
- `nets/c130.bin` identique à lab3, SHA-256
  `e534cc84c801fc3c5b7d377cd8b67a50a4e1a11b65b1f6644c32b20eeb52cba3`.
  `gen0.bin` ne sert qu'aux tests. Pas de données d'apprentissage copiées.
- Pas d'ajout au sélecteur des Experts : lab3/c130 reste le moteur du jeu.

## Diagnostic préparé

`lab5 probe` : positions indépendantes et stratifiées, comparaison des alternatives par suites
complètes, puis confirmation sur des mondes indépendants. Le moteur de continuation est c130
à 400 simulations. Le coup de référence est conservé parmi les candidats ; la confirmation ne
rechoisit pas le gagnant. Les zéros et régressions restent dans la moyenne.

Reprise par fichiers atomiques, manifeste des paramètres et empreintes, sorties individuelles,
commande `probe-show` pour relire une position sans calcul, analyse Python des confirmations avec
bootstrap par partie source. Le pilote proposé est de six positions, uniquement pour le coût.

## Vérifications

- `timeout 600 cargo test --workspace --no-default-features` : passé (93 tests).
- `timeout 600 cargo clippy --workspace --all-targets --no-default-features` : passé ; avertissements
  déjà présents dans lab3 figé. Lab5 passe aussi son Clippy ciblé avec `-D warnings`.
- Tests ciblés lab5/brain5 repassés après les corrections de reprise et de traçabilité CPU.
- `timeout 30 python3 -m unittest discover -s lab5/py -p 'test_*.py'` : 4 tests passés.
- Scripts vérifiés par `bash -n`. Poids c130 identiques à lab3 (empreinte ci-dessus).
- Vérification Vulkan : compilateur Debian 2025.2 insuffisant pour le shader bf16 hérité de lab3 ;
  compilateur 2026.4 extrait temporairement, sans installation système.
  `timeout 240 cargo check -p smallmobile-lab5 --all-targets` avec ce compilateur dans `PATH` :
  passé, shaders compris. Aucune partie GPU jouée sur mini ; le pilote sur GPU reste à faire.

Aucun résultat de force à ce stade.

## Calculs

Aucune campagne, dépense AWS ou formation de réseau lancée. Les tests locaux utilisent de
petites fixtures et ne constituent pas une mesure de force.

### Pilote sur vast.ai (25 octobre)

À la demande de l'utilisateur : une machine vast.ai (offre 57512964, Nevada, 0,35 $/h ; RTX 5080
16 Go, pilote 615.71, Vulkan bf16 et matrices coopératives ; Ryzen 9 7945HX, 31 fils ; 29 Go),
binaire compilé sur mini (Vulkan, shaderc 2026.4), image `vastai/base-image:stock-ubuntu26.04`.
Outil : `labtools/vast.sh`. 17 min de location, ~0,10 $ (plus ~0,13 $ d'essais sur deux hôtes
défectueux) ; crédit restant 24,78 $.

- `bench-vulkan` c130, réseau seul : 555 000 positions/s par lots de 64, 717 000 par lots de
  2 048 (au-delà, refus du moteur : la liste par défaut du banc va jusqu'à 4 096, à corriger).
- `selfplay` c130, 1 000 parties, graine 7 : GPU 19 s, ~1 885 positions/s en régime ; processeur
  seul 91 s, ~400 positions/s (charge ~17 sur 31 fils). Pas de référence c130 mesurée sur le PC.
- Pilote `probe` (paramètres de `lab.md`, graine 51001, `DEVICE=vulkan`), lot `lab5/runs/pilot` :
  `positions` 7,6 s ; `select` 82 s (six positions en parallèle, une par fil) ;
  `confirm` 102 s. Deux coups changés sur six : +11,3 points (2 j., début, conquête), −0,8
  (4 j., milieu, redéploiement) ; global équilibré +1,8, sans intervalle (pilote : coût seulement).

### Campagne c1, fixée avant tout lancement (25 octobre)

Question : c130 laisse-t-il passer de meilleurs coups, à quelle fréquence et pour quel gain
moyen ? Taille fixée sur le pilote seulement (son coût, et la dispersion de ses six écarts :
écart type ~4,6 points), avant de voir la moindre confirmation de c1.

- `RUN=c1`, **50 positions par case** (2 et 4 joueurs × début, milieu, fin : 300 positions),
  graine **52001** (hors de celle du pilote), autres paramètres ceux du pilote : 6 coups au plus,
  32 suites de sélection et 128 de confirmation par coup, c130 à 400 simulations, `DEVICE=vulkan`.
  Précision attendue (à revoir sur les données) : global ±0,5 point, par case ±1,3.
- Le score global compte toutes les positions, y compris celles dont l'issue est déjà jouée :
  on y mesure le gain moyen possible sur l'ensemble des décisions.
- Une seule machine vast.ai pour les trois étapes, même modèle de GPU de bout en bout ; à défaut
  (machine perdue), reprise sur le même modèle, notée ici.
- Lecture : `py/probe.py`, intervalles bootstrap par position. Pas d'arrêt anticipé selon le
  résultat ; positions à examiner ensuite seulement, dans une étape distincte.
