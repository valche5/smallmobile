# Labos de l'IA Expert : règles communes

Chaque approche de l'IA Expert a son labo, un dossier complet et autonome :

- `labN/journal.md` (`JOURNAL.md` pour lab1) : objectif, architecture, résultats et décisions de
  cette approche, et rien d'autre ;
- `labN/brain/` (paquet `smallmobile-brainN`) : son Expert (recherche, coups, encodage, réseau),
  joué par le serveur du jeu (programme `labN-expert`) et par les labos ;
- `labN/` (paquet `smallmobile-labN`, programme `labN`) : arène, auto-apprentissage, outils ;
- `labN/py/` : entraînement PyTorch ; `labN/nets/` : ses réseaux ;
- `labN/data/`, `labN/runs/` (hors Git) : ses données et résultats, aussi dans le bucket sous
  `labN/`.

**Un nouveau labo à chaque rupture** (format des données, coups, recherche, changement
d'approche) : on part d'une copie élaguée du dernier labo, avec un journal propre et ses propres
données ; rien n'est converti d'un labo à l'autre, hormis les poids d'un réseau de départ.
L'ancien labo est alors **figé** : il ne change plus, mais reste compilé tant qu'il sert, comme
variante de l'Expert dans le jeu (`src/shared/experts.ts`) ou comme adversaire témoin dans
l'arène d'un labo plus récent (qui dépend alors de son `brain`). Quand il ne sert plus, on le
supprime (Git le garde) et on le retire de `src/shared/experts.ts`.

Labos :

| Labo | Approche | État |
| --- | --- | --- |
| `lab1` | Recherche AlphaZero sur mondes possibles (boucle ouverte), fins simulées de 3 tours, coups regroupés ; réseaux gen0 à gen11 | Figé le 21 octobre 2026 ; Expert `lab1/gen11` du jeu |
| `lab2` | Gumbel AlphaZero, nœuds de hasard, coups jeton après jeton, départ de zéro (`lab2/journal.md`) | Figé le 22 octobre 2026 après sa troisième boucle ; Expert `lab2/c060` du jeu, témoin de l'arène de lab3 |
| `lab3` | Spécialisation puis reprise généraliste, c130 en 128×7 (`lab3/journal.md`) | Figé le 24 octobre 2026 ; Expert `lab3/c130` du jeu |
| `lab4` | Budget, largeur, horizon et cibles de valeur depuis c130 ; essais sans progrès net de l'apprentissage | Figé le 25 octobre 2026 ; résultats dans `lab4/journal.md` |
| `lab5` | Coups manqués par c130 : alternatives évaluées jusqu'à la fin, sélection et confirmation indépendantes ; reprise directe de lab3 | Figé le 25 octobre 2026 ; **résultats invalides** : calculés en Vulkan bf16 sur NVIDIA, qui calcule faux (erratum de `lab5/journal.md`) |
| `lab6` | La marge de c130 face à une recherche beaucoup plus longue (arène 16× et 64× les simulations) ; essai CUDA | Ouvert le 25 octobre 2026 ; plan `lab6/lab.md` |

Outils communs (ce dossier) :

- `cloud.sh` : machines de calcul AWS (`LAB=labN labtools/cloud.sh job …`, mode d'emploi en
  tête du script) ; `aws/` : politiques du compte ; `state/` (hors Git) : la machine en cours ;
- `vast.sh` : machines à GPU louées sur vast.ai (`LAB=labN labtools/vast.sh up OFFRE`, mode
  d'emploi en tête du script ; section « Les machines vast.ai ») ; journal dans `state/vast/` ;
- `score.py` : le score d'une arène (`labN arena --out r.tsv`) et ses intervalles par donne ;
- `games.ts` : résumé des parties archivées par le serveur (`data/games/`) ;
- `export-room.ts` : une salle du serveur au format du noyau, pour l'analyser dans un labo.

Le jeu propose l'Expert de chaque labo listé dans `src/shared/experts.ts` (le dernier par
défaut) ; `bun run kernel:build` construit le programme `labN-expert` de chaque labo.

## Protocole de mesure

Commun à tous les labos ; un labo peut le compléter dans son journal.

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
- Arrêt anticipé (`labN arena --stop 3`, depuis le 19 octobre) : pour une question « le candidat
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

## Les machines vast.ai

Constaté le 25 octobre 2026 : compte de l'utilisateur, 25 $ de crédit offert, clé SSH
`~/.ssh/id_ed25519` de mini déclarée ; client `vastai` de `lab4/py/.venv`. Mêmes règles que pour
AWS : accord de l'utilisateur avant toute dépense, étapes courtes, `vast.sh status` après chaque
calcul (aucune machine ne doit rester).

- Image `vastai/base-image:stock-ubuntu26.04-…` : SSH et Vulkan NVIDIA fournis, glibc assez récente
  pour le binaire compilé sur mini. Une image nue (`debian:trixie`) n'a pas de sshd : inutilisable.
- Hôtes inégaux : un GPU occupé à 100 % au repos et Vulkan bloqué à l'ouverture (`drm_open`) ; un
  `authorized_keys` aux droits refusés par sshd. `vast.sh up` contrôle les deux et corrige les droits.
- **Le shader Vulkan bf16 calcule faux sur NVIDIA** (RTX 5080, 25 octobre) : sur ces cartes, `--device
  cuda` (lab6) ou `LAB6_VULKAN_F32=1`. `vast.sh up` compare le GPU au processeur (`labN check-gpu`)
  et installe cuBLAS et NVRTC (paquets pip `nvidia-*`), chargés par les calculs lancés par `run`.
- Sur la machine, `ps` et `pgrep` peuvent rester bloqués pendant qu'un calcul tient le GPU : le
  script suit ses calculs par numéro de processus et journal.
- Hôtes, 25 octobre (numéro de machine vast, stable d'une location à l'autre) : **bons** 119726
  (Pays-Bas, RTX 5080 bridée à 250 W, Ryzen 9 9950X ; campagne c1 de lab5) et 153202 (Nevada,
  RTX 5080, Ryzen 9 7945HX ; pilote de lab5, puis GPU bloqué après une ouverture de Vulkan
  pendant le démarrage de l'image). **GPU occupé avant tout calcul** (100 %, 96 à 118 W au
  repos) : les RTX 5090 des offres 34461819 (Belgique) et 19559317 (Norvège). Préférer les bons.
- Le binaire Vulkan se compile sur mini avec shaderc 2026.4 extrait dans
  `~/.local/opt/shaderc-2026.4` (Debian 13 : 2025.2, insuffisant pour le shader bf16), dans
  `~/.cache/vast-target` pour ne pas toucher `target/release`.

## Les machines AWS

Règles d'usage (aucun calcul lourd sur mini, `labtools/cloud.sh`, une commande par étape avec
délai estimé, utilisateur informé à chaque étape, `cloud.sh status` après chaque calcul) :
voir `AGENTS.md`. Aucun calcul AWS sans l'accord de l'utilisateur : un plan de labo n'autorise
pas une dépense.

Configuration constatée le 19 octobre 2026, à revérifier avant de dépenser :

- SmallAI `631245464763`, profil `smallai`, région `eu-north-1` (Stockholm). Fonctionnalités
  avancées activées, Spot autorisé, quota Standard Spot 64 vCPU ; une seule `c7a.16xlarge`
  à la fois. Le DryRun a vérifié les permissions, pas la disponibilité réelle. Quota Spot des
  machines à GPU (« All G and VT Spot Instance Requests ») : 0 au 22 octobre, augmentation à 64
  vCPU demandée le 22 octobre à 17 h 23 (Service Quotas, en attente) ; `cloud.sh` sait
  lancer ces machines (`gpu_type` : image d'AWS avec les pilotes NVIDIA, Ubuntu, utilisateur
  `ubuntu`).
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
  zone la moins chère sous le plafond. Pour un calcul long, `job --save labN/X,...` garde ces
  dossiers dans le bucket (`s3://…/labN/X`) : repris au lancement, copiés toutes les 10 min, à
  l'annonce d'une reprise (2 min avant) et à la fin. Découper le calcul en morceaux déplacés
  dans le dossier gardé une fois complets, et sauter ceux déjà présents : un calcul relancé
  reprend alors où il en était.
- Bucket `s3://smallai-lab-631245464763/`, qui reprend les chemins du dépôt (`labN/data/X` y est
  `labN/data/X`) ; rôle machine `smallai-worker`, politiques dans `labtools/aws/`. Synchroniser
  ensuite les résultats sur mini.

Budget global avant crédits `Credits-Semaine-19-25-Oct-2026-110USD`, compte de gestion
`051299608012` : **110 $ du 19 octobre 00:00 UTC au 26 octobre 00:00 UTC**, alertes à
55/80/100/110 $. Ce sont des alertes, pas une coupure automatique. Les anciens plafonds mensuels
ne s'appliquent plus. Dernier relevé du 19 octobre : 8,31 $ consommés sur la période,
123,23 $ de crédits restants estimés, soit 101,69 $ de dépenses supplémentaires dans ce budget
et 21,54 $ de réserve. Depuis ce relevé, estimations de `cloud.sh` : P0 0,87 $, essais Spot
0,02 $, P1 1,11 $, pilote P2 0,86 $, entraînement et arènes P2 1,54 $ (dont 0,10 $ perdus sur un
script non exécutable, et une reprise par AWS à 20 h 30, sans perte grâce à `--save`),
confirmation P2 1,19 $, données de gen9 3,31 $, entraînement et arènes de gen9 0,96 $,
données de gen10 3,46 $, entraînement et arènes de gen10 1,26 $, données de gen11 2,38 $,
entraînement de gen11 1,43 $ (dont 0,18 $ perdus sur le premier lancement), gen12 abandonné
0,51 $, premiers essais de lab2 depuis les poids de lab1 (abandonnés, données effacées) 2,49 $,
lab2 gen1 0,44 $, première boucle de lab2 et mesures 2,58 $, lab2 face à lab1
0,29 $, tailles de réseau de lab2 1,18 $, deuxième boucle de lab2 5,29 $, ses mesures de fin
0,47 $, troisième boucle de lab2 5,35 $, boucle à 4 joueurs de lab3 et ses
mesures 2,17 $, mesures de fin de la troisième boucle de lab2 0,63 $, machine d'essai du moteur de lab3
0,39 $, spécialisation à 4 joueurs de lab3 arrêtée au pas 3 0,80 $, sa relance arrêtée au pas 18 3,53 $, soit 44,51 $. Relevé du 22 octobre (budget et Cost Explorer, environ un jour de retard :
le 21 octobre en partie, le 22 pas encore) : 28,91 $ consommés sur la période, entièrement payés
par les crédits. **Montants datés : consulter coûts et crédits du compte de gestion
avant chaque campagne. Les crédits ne se renouvellent pas chaque semaine.**

Pour chaque étape autorisée : mesurer un petit échantillon sur AWS, estimer durée et coût total
(installation, données, calcul, évaluation), fixer le timeout, puis lancer l'étape complète.
Ne pas lancer toute la feuille de route d'un seul bloc. L'en-tête de `cloud.sh` documente
`job --type ... --data ... --get ... -- 'commande'` et le rapatriement des résultats.
