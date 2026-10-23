# Labo 3 : un réseau par nombre de joueurs

Ouvert le 22 octobre 2026, à partir de lab2, figé après sa troisième boucle (`lab2/journal.md` :
la recherche, les coups, le réseau et la boucle viennent de là). Règles communes des labos,
protocole de mesure, machines AWS : `labtools/README.md`.

## Objectif

Le même que lab1 et lab2 : l'IA la plus forte possible sur les cartes générées, à 2, 3, 4 et 5
joueurs, mesurée aussi au temps de réflexion du jeu sur mini, avec les mêmes règles d'honnêteté
(l'Expert ne sait que ce qu'un joueur attentif peut savoir, et ne sait pas contre qui il joue).

## Pourquoi ce labo

Question de l'utilisateur (22 octobre) : **un réseau spécialisé dans un nombre de joueurs
joue-t-il mieux, à ce nombre, qu'un réseau qui apprend tous les nombres à la fois ?** Le jeu
change beaucoup d'une taille à l'autre (à 2 joueurs, lab2 domine les IA du jeu bien plus qu'à 3,
4 ou 5) ; un réseau par taille y consacre toute sa capacité et toutes ses parties, au prix de ce
qu'une taille apprend aux autres.

Décisions de l'utilisateur (22 octobre) : un réseau par nombre de joueurs, en commençant par 4
joueurs ; au départ, le petit réseau (64x4) et la petite fenêtre (250 000 positions) de la
première boucle de lab2 ; départ de zéro, comme lab2.

## Système actuel

Celui de lab2 (voir son journal), sauf :

- **Tailles des tables** : `lab3/loop.sh P N` joue ses parties d'auto-apprentissage et ses arènes
  aux tailles indiquées par P : un nombre pour une spécialisation, ou `2,3,4,5` pour un généraliste
  (reprise demandée le 23 octobre). Parties `lab3/data/pP`, réseaux et arènes `lab3/runs/pP`.
  Les listes utilisent des tirets dans les chemins (`p2-3-4-5`) : les virgules séparent les options
  de la notation des agents. Dans chaque cycle généraliste, les parties alternent 2, 3, 4, 5 joueurs.
  Le réseau garde les entrées et sorties de lab2 (jusqu'à 5 joueurs).
- **Réseau de départ** : le généraliste de lab2, son c060 (128x4, `lab3/nets/start.bin`), continué
  à un seul nombre de joueurs pour la spécialisation (décision du 22 octobre), ou à toutes les
  tailles pour la reprise généraliste du 23 octobre.
  `lab3/nets/gen0.bin`, le réseau neuf de lab2, ne sert plus qu'aux tests.
- **Fenêtre** : toutes les positions jusqu'à 4 000 000, puis
  4 000 000 × (1 + 0,4 × ((total / 4 000 000)^0,75 − 1) / 0,75) (250 000 pour l'essai depuis
  zéro, 1 000 000 pour le premier essai de spécialisation). Le réseau de départ joue d'abord les
  cycles qui la remplissent (4 à 4 joueurs, `data.filled`), et le premier entraînement part de
  là : un pas tire 4 fois les positions de son dernier cycle, soit une passe sur une fenêtre de
  quatre cycles, puis moins quand elle grandit (le c060 de lab2 : 0,36 passe par pas sur 11
  cycles, 275 000 parties).
- **Ordre des étapes** : sur GPU (`DEVICE=vulkan`), génération, entraînement et arènes sont
  successifs par défaut (`PARALLEL=0`) : pas de partage statique des cœurs ni de concurrence entre
  Vulkan et ROCm. Les parties du cycle k+1 y sont jouées par le réseau k, qui vient d'apprendre
  (jusqu'au 23 octobre midi, par k−1 : un retard qui ne servait qu'au recouvrement). La
  génération et les arènes utilisent `THREADS` fils (tous par défaut) ; `TRAIN` règle les
  opérations CPU de PyTorch (4 par défaut), sans réservation de cœurs. Sur CPU, `PARALLEL=1` garde
  la recette AWS historique : 48 fils pour les parties et 16 pour l'apprentissage simultanés, tous
  les 64 pour le remplissage et les arènes.
- **Mesure de chaque pas** (depuis le 23 octobre midi) : tous les `ARENA_EVERY` pas (4) et au
  dernier, un duel contre `REF` (c000), n = 200 des deux côtés, 460 donnes au plus (graine 1004),
  arrêté par le test séquentiel « 3 points ou rien » (`STOP`). Les arènes face à Difficile et
  Normale à chaque pas prenaient la moitié d'un pas (~540 s sur 1 070) et ne mesurent plus les
  progrès à ce niveau (lab2/journal.md, troisième boucle). Un duel sur le CPU (20 fils) :
  ~860 s quand le test ne conclut qu'à la fin (c015 contre c000) ; sur GPU (`--device vulkan`,
  ci-dessous) 47 s : un duel à chaque pas sur GPU, tous les 4 sur CPU.
- **L'arène sur GPU** (`lab3 arena --device vulkan`, `arena::run` avec un GPU) : comme le
  selfplay, chaque fil mène 16 parties ; un joueur `az` cherche en attendant ses évaluations du
  GPU (`az::act_with`), pendant que le fil fait avancer les autres parties ; les IA du jeu jouent
  directement. Un seul moteur pour tous les réseaux de la table (`Gpu::start_nets`, deux au
  plus, de même forme) : leurs poids à la suite dans les mêmes tampons, chaque position garde le
  rang du sien (un octet) et le shader décale sa table de poids d'autant (`T`). Deux moteurs à la
  fois sur ce GPU se bloquent, même sur deux files (graphique et calcul seul) : une étape soumise
  d'avance qui attend son feu vert retient l'autre moteur (constaté, mécanisme du pilote non
  élucidé). Test `one_gpu_evaluates_two_networks`. Duel c015 contre c000, 460 donnes : GPU
  **−0,5 ± 2,1** en 47 s, CPU −0,0 ± 2,2 en 859 s (18 fois plus long) ; avec l'arrêt séquentiel,
  34 s pour 184 donnes.
- **Fenêtre en RAM, sous forme compacte** (`data.py`, `pack`, `Packed`) : chaque fichier `.pos`
  a, à côté, un `.pos.packed.npz` fait une fois (~25 s par cycle de 25 000 parties) : les octets
  non nuls des régions et de la partie (indice de 16 bits, valeur), la politique des seuls coups
  possibles, voisines et propriétaires des régions présentes : 1 938 octets par position au lieu de
  6 210. La fenêtre de 6,8 millions de positions passe de ~42 Go (plus que les 31 Go du PC : lue
  au hasard dans les fichiers mappés, 81 ms par lot de 512, contre 8,6 ms de calcul sur le GPU)
  à 13,2 Go en RAM ; un lot se redéploie en 2 ms, dans un autre fil pendant que le GPU calcule.
  Champs identiques au bit près aux fichiers (test `test_train.py`, et sur des lots réels).
  Apprentissage de c016 sur cette fenêtre : 284 → **56 s**. La validation se mesure sur un
  sous-ensemble fixe des parties réservées (`VAL_MEASURE`, ~40 000 positions) : 45 → 2 s.
- **Témoin de l'arène** : l'Expert de lab2 (`lab2:net=lab2/nets/….bin,n=…`, `lab2/brain`), tel
  qu'il joue dans le jeu ; plus de témoin lab1.
- **Moteur de génération des parties** (`selfplay::run`, `batch`, demande de l'utilisateur :
  massivement parallèle, attentif aux caches du processeur). Chaque fil mène `CONCURRENT`
  parties à la fois ; la recherche attend ses évaluations (`search::Oracle`) : chaque partie
  avance jusqu'à sa prochaine évaluation, puis le fil les évalue toutes ensemble, couche après
  couche (`Net::eval_batch` : les poids d'une couche restent dans le cache du cœur d'une
  position à l'autre ; les vecteurs de partie de toutes les positions passent ensemble dans les
  couches qui les lisent, trois à la fois avec AVX-512). Les évaluations sont gardées d'une recherche à la suivante
  (`search::Evals`, une par partie : la décision d'après repasse par les positions que la
  précédente a explorées), chacune en quelques centaines d'octets (empreinte de 128 bits,
  valeur, logits des coups possibles). Tout ce que touche un fil lui appartient ; seuls les
  poids, en lecture, sont partagés. Chaque partie est la même, à l'octet près, que jouée seule
  (tests ; 1 024 parties comparées sur AWS). Le jeu et l'arène gardent une évaluation à la
  fois (`search::Direct`).
- **Vitesse** (22 octobre, c060 de lab2, n = 200, 4 joueurs, recette de l'auto-jeu ; ancien
  moteur : celui de lab2) : sur un cœur de mini, 4 parties en 21,0 s → 14,8 s (évaluations
  gardées seules) → 14,6 s ; sur `c7a.16xlarge`, 16 parties sur un cœur 33,6 s → 24,1 s, 1 024
  parties sur 64 cœurs 49,9 s → 34,0 s → 30,9 s avec les vecteurs de partie groupés (1,61 fois
  plus vite).
- **Profil** sur 64 cœurs (`perf`) : le réseau prend environ 82 % du temps, dont 53 % pour les
  grands produits bf16, 13 % pour les vecteurs de partie, 13 % pour le reste de son calcul ; le
  moteur du jeu environ 6 %. Seul ou à 64, un cœur fait 2,05 instructions par cycle et autant
  d'instructions par partie : les cœurs ne se gênent pas (caches, mémoire) ; à 64, le processeur
  passe de 3,68 à 2,78 GHz (AVX-512 sur tous les cœurs), d'où 1,32 fois moins vite par cœur.
- **Compilation et choix CPU/GPU** : Vulkan est inclus par défaut (`cargo build`,
  `cargo build --release`) et demande `glslc` à la compilation. `--no-default-features` produit un
  programme CPU seul, sans outil ni chargeur Vulkan ; les machines AWS CPU utilisent cette option.
  À l'exécution, `selfplay --device cpu` (défaut) et `--device vulkan` choisissent l'évaluateur ;
  `--device gpu` est un alias de Vulkan. `DEVICE=vulkan lab3/loop.sh …` utilise le GPU pour les
  parties. Le pont pyo3 et l'ancien évaluateur PyTorch ont été retirés le 22 octobre : Python
  et PyTorch ne servent plus qu'aux outils d'entraînement et d'analyse du réseau.
- **Premières mesures avec l'ancien évaluateur PyTorch sur le PC** (22 octobre ; i7-12700KF, 20 fils, sans AVX-512 ; RX 9070
  XT, ROCm 7.14, PyTorch 2.14) avec le c060 de lab2, n = 200, 2 à 5 joueurs : parties, processeur
  seul 207 positions/s ; GPU 141 (4 fils de jeu), 205 (8 fils), 184 (16 fils). Le GPU plafonne
  dès 8 fils, occupé à 76-83 % pour environ 1 TFLOP/s utile : le coût est fixe, à chaque appel
  (fil évaluateur seul, copies, lancements de PyTorch), pas dans le calcul. Entraînement, lots de
  512 : 0,3 lot/s sur le processeur (bf16 sans AVX-512 BF16), 39 sur GPU en f32, 103 en bf16 :
  `train.py` ne prenait le bf16 que sur processeur (`torch.autocast("cpu")`), corrigé. Rust 1.97
  plantait en compilant lab3 (LLVM, code AVX-512) : 1.99 compile.
- **Plafond sans réseau** (même PC, même recette) : un réseau factice qui répond aussitôt, 20 fils,
  2 356 positions/s, environ 3 400 sans le coût du faux réseau lui-même (31 % du profil) ; une
  simulation coûte environ 17 µs (moteur du jeu ~25 % du profil, encodage 7 %, empreinte 4 %).
- **Le réseau par Vulkan** (`lab3 selfplay --device vulkan`, option `vulkan` activée par défaut, `src/vulkan.rs`,
  `shaders/`), sans Python : un fil d'envoi recopie les groupes des fils de jeu dans un emplacement
  que le GPU lit directement (12 emplacements en vol), un fil de réception attend chaque lot
  (sémaphore à compteur) et lit les sorties là où le GPU les a écrites. Un seul calcul par lot, un
  groupe de travail par position, tout le réseau dans sa mémoire partagée (`net.comp`) : les grands
  produits des régions sur les matrices coopératives (16 × 16 × 16, bf16, sommes en f32), le flux
  résiduel en f32 dans les registres, moins de 32 Ko de mémoire partagée (deux positions à la fois
  par unité de calcul). Mêmes arrondis bf16 que le processeur pour h, la moyenne des voisines et
  relu(n1), au plus proche ; node_in et node_pol en deux parties bf16, presque f32. Écarts au
  calcul f32 sur le c060 (test `vulkan_evaluates_like_the_processor`) : victoire 0,002, pièces 0,06,
  logits 0,05 (le bf16 du processeur : 0,0015, 0,06, 0,04). `net32.comp`, tout en f32 (écart 2·10⁻⁴), sert de secours
  (GPU sans matrices coopératives bf16, llvmpipe pour les tests sans GPU).
- **Vitesse par Vulkan** (RX 9070 XT, RADV, c060, 20 fils, 2 560 parties) : 1 222 positions/s, 5,9
  fois le processeur seul et 6 fois PyTorch ; lots de 234 positions, GPU occupé 77 %. Étapes : 40
  petits calculs par lot (102 positions par lot, 0,84 ms de GPU chacun) 669 ; tout le réseau en un
  calcul, f32, 551 ; bf16 et matrices coopératives 794 ; moins de 32 Ko par position 1 029 ; quatre
  groupes par fil 1 130, quatre emplacements 1 222. Deux files de calcul : 836. Seul, le calcul
  (`lab3 bench-vulkan`) : 0,22 ms pour une position, 509 000 positions/s par lots de 4 096 ; le
  GPU attend surtout la mémoire (lectures des poids), pas ses unités de calcul.
- **Le shader, suite** (22-23 octobre ; profils : `LAB3_VULKAN_PHASES=1` donne la durée de chaque
  étape d'une position, lue sur l'horloge du GPU ; `bench-vulkan --stream` des lots à la suite).
  Piège : peu chargé, le GPU descend à 60-100 MHz (3,3 GHz en selfplay) ; mesurer sous charge.
  Lignes de h de 256 octets dans la mémoire partagée : les 16 lignes d'une tuile sur le même banc,
  lectures 16 fois plus lentes ; écartées de 8 nombres, le débit double (en flux, lots de 256 :
  433 000 → 900 000 positions/s ; de 2 048 : ~950 000). Deux sommes indépendantes par tuile au lieu
  d'une : +20 à 40 % (quatre : trop de registres, moins de positions en même temps). Le selfplay ne
  suit pas (1 250 à 1 290 positions/s) : le GPU n'y calcule plus que 45 % du temps, le reste est
  du côté du CPU et des allers-retours des lots.
- **Le trajet d'une position** (23 octobre, `LAB3_VULKAN_PROFILE=1` : temps des fils de jeu ;
  `LAB3_INSTANT=1` : évaluations instantanées, sans réseau ; `LAB3_GROUPS`, `LAB3_CONCURRENT` :
  parties menées par fil). En selfplay GPU, les fils de jeu calculent 83 % du temps, attendent 12 %,
  copient et décodent 4,5 % : ils sont le frein, pas le GPU. Le même jeu coûte deux fois plus de
  cycles à 128 parties par fil (4 × 32, le réglage GPU) qu'à 8 (mêmes parties, évaluations
  instantanées ; `perf stat`) : 68 contre 35 millions de cycles par position, 1,02 contre 1,61
  instruction par cycle, 12 fois plus de défauts du cache L3. Au pic (heaptrack), ~250 Ko par
  partie, dont ~65 Ko relus à chaque simulation (arbre, évaluations gardées) : 128 parties par fil
  débordent les caches.
- **Données réduites, recherche identique au bit près** (empreintes de `lab3 speed`, selfplay
  d'un fil) : l'arbre gardé par partie et réutilisé (nœuds de 16 octets ; coups en 16 octets lus à
  chaque passage, logit, a priori calculé une fois, valeur et visites du seul joueur qui agit,
  plus 24 octets lus pour le coup joué ; premier enfant dans le coup, les autres chaînés) au lieu
  de 15 valeurs et d'un `Vec` d'enfants par coup ; les évaluations gardées dans une seule table
  (dernière recherche qui s'en est servie) ; les positions à enregistrer sans les régions absentes
  ni les 608 clés de politique (~3,5 Ko au lieu de 5,8) ; le journal des coups d'un seul tenant. À
  128 parties par fil : défauts L3 65 000 → 28 000 par position, ~59 → ~48 millions de cycles.
  Selfplay GPU (c060, 20 fils) : 1 220 → 1 510 positions/s ; les fils attendent maintenant 23 %
  du temps, le GPU calcule ~66 % : le prochain frein est l'aller-retour des lots.
- **Mesurer le régime établi** (`selfplay` affiche toutes les 10 s le rythme des 10 dernières
  secondes, positions comptées dès leur recherche ; le compteur des parties finies retarde d'une
  durée de partie). Toutes les parties partent ensemble et leurs premiers coups coûtent moins : le
  rythme est trop haut tant que les parties n'ont pas l'âge d'une partie, ~60 s à 128 par fil,
  ~110 s à 256. Selfplay GPU (c060, 20 fils), régime établi : 4 × 32 ~1 650 positions/s (attente
  10 %) ; 8 × 32 ~1 655 (attente 4 %) : doubler les parties par fil ôte l'attente mais le jeu coûte
  d'autant plus (caches), aucun gain. Les 1 510 et 1 539 des mesures sur un nombre fixe de parties
  comptent la fin, où les fils n'ont plus assez de parties.
- **Simulations groupées par tour de la racine** (`search::Az::run`). Une
  partie ne cherchait qu'une feuille à la fois : pour remplir les lots du GPU, il fallait 128
  parties par fil, dont les arbres débordaient les caches (sans réseau, 41 millions de cycles
  par position contre 31 à 8 parties par fil, 23 000 défauts L3 contre 500, répartis sur tout
  l'arbre). Le halving séquentiel avance par tours où chaque coup encore retenu à la racine
  reçoit une visite : ces simulations suivent des coups différents de la racine, donc des
  sous-arbres disjoints. Elles descendent toutes, leurs positions nouvelles sont évaluées
  ensemble, puis elles remontent dans l'ordre ; seuls les choix de la racine se font au début du
  tour, sans les valeurs des simulations du même tour. n = 200, 16 coups : 49 tours de 16, 8, 4
  puis 2 simulations, ~4 positions par tour. Limitée à une simulation par tour, elle redonnait
  l'ancienne recherche au bit près (empreintes de `lab3 speed` et du selfplay de contrôle) ; ce
  réglage est retiré, et la force face à l'ancienne n'est pas mesurée. Selfplay GPU (c060, 20 fils), régime
  établi : 3 × 8 parties par fil ~1 750 positions/s (2 × 8, 2 × 12, 2 × 16, 4 × 8 : 1 700 à
  1 750), contre ~1 650 avec 4 × 32 avant. Les fils de jeu attendent alors 30 à 40 %, mais le GPU
  n'est pas plein : ses horodatages donnent ~770 000 positions par seconde de calcul (1,1 million
  en lots de 2 048, `bench-vulkan --stream`) pour ~500 000 demandées ; tous les réglages (groupes,
  emplacements, lots de 60 à 430 positions) butent sur ces 500 000 : l'aller-retour des lots
  limitait (le compteur `gpu_busy_percent`, 92 %, compte toute activité).
- **Flot continu** (`gpu.rs`) : chaque fil de jeu a ses places dans un anneau de la mémoire que lit
  le GPU (16 384 places en tout) et y écrit lui-même les positions d'une partie, puis leurs places
  à la suite dans une file d'ordre (65 536 numéros) ; un fil « gpu » soumet d'un calcul les numéros
  écrits à la suite (2 calculs en vol au plus : pendant que l'un tourne, le suivant grossit) et note
  leur fin ; le fil de jeu lit les sorties sur place et reprend la partie dès qu'elles sont là, une
  partie à la fois (24 par fil). Plus de fil d'envoi ni de réception, plus de copie, plus de
  groupes. Premier essai, des places à la suite dans l'ordre des envois : un fil lent à relire
  bloquait tout (l'anneau se renouvelle en ~27 ms ; un fil voulait écrire sur sa propre place pas
  encore lue) ; d'où la file d'ordre, dont un numéro revient dès que le GPU l'a calculé. Les
  écritures dans la mémoire du GPU (write-combining) passent une barrière avant leurs marques.
  Selfplay GPU (c060, 20 fils) : ~2 100 positions/s (jusqu'à 2 400), ~600 000 évaluations/s,
  lots de 240 à 420 positions ; fils de jeu 70 % de calcul, fil « gpu » 13 % d'un cœur. Le GPU
  tourne alors à 1,6-1,9 GHz au lieu de 3,3 (`pp_dpm_sclk`, réglage `auto`), mais ce n'est pas le
  frein : horloge figée à 2 420 MHz (niveau `profile_peak`, demandé sans root par l'appel des
  outils de profilage, `amdgpu_cs_ctx_stable_pstate` de libdrm, rendu à `auto` à la fin du
  programme), même débit (2 030 positions/s, attente des fils 25 %).
- **Le trajet d'une position dans le flot** (`LAB3_LATENCY=1`, avec `LAB3_VULKAN_PROFILE=1` :
  horodatages du GPU ramenés à `CLOCK_MONOTONIC` par `VK_KHR_calibrated_timestamps`). Un calcul
  soumis seul : appel de soumission ~30 µs, soumis → début sur le GPU 440 µs, calcul 280 µs (250
  positions), fin → vue 90 µs, et **80 à 150 µs de trou sur le GPU avant chaque calcul**, même
  avec 3 ou 4 calculs en vol (le trou baisse sans disparaître : un coût fixe par tâche du noyau,
  changement de tâche et caches). Écrite → reprise : 1,27 ms, dont 0,23 de calcul ; les fils
  attendent 25 à 30 % du temps.
- **Étapes soumises d'avance** (`vulkan::STEPS`, `Engine::launch`) : trois commandes enregistrées
  une fois, de 32 étapes chacune ; une étape attend un événement posé par l'hôte (`vkSetEvent`,
  permis après la soumission hors passe de rendu, `vkCmdWaitEvents`), calcule les positions que
  dit un petit tampon (dispatch indirect), puis pose un événement de fin que le fil « gpu » lit
  (`vkGetEventStatus`) ; un tour fini est soumis à nouveau, derrière les autres. Une soumission
  pour 32 calculs, sans tâche du noyau par calcul : trou avant un calcul ~30 µs, fin → vue 30 µs,
  feu vert sans coût ; écrite → reprise ~0,8 ms ; lots de ~75 positions. Selfplay GPU (c060,
  20 fils) : **~2 760 positions/s** (16 parties par fil ; 2 660 avec 24), contre ~2 100 ; les
  fils de jeu calculent ~90 % du temps, attendent 3 % : le jeu et la recherche sont de nouveau
  le frein.
- **Profil du jeu et de la recherche** (`perf record --call-graph=fp`, binaire avec pointeurs de
  cadre, selfplay GPU à 2 700 positions/s) : coups possibles (`moves::legal`, surtout
  `Game::targets` : portée, coût, modificateurs de chaque région) 33 % des cycles, encodage 19 %,
  arbre ~13 %, échange avec le GPU ~8 %, coups joués 5 %, allocations ~5 %, empreintes 4 % ; le
  fil « gpu » un cœur entier (il surveille les fins en boucle). La descente calculait tous les
  coups de chaque position traversée, pour savoir seulement si elle était forcée, puis les jetait
  quand le nœud existait déjà. Chercher d'abord l'enfant sans coup forcé est faux : le même
  résultat public du hasard peut être suivi d'un coup forcé dans un monde et pas dans un autre
  (le marché vient de la pioche cachée). `moves::forced` répond à la seule question : en
  conquête, un coup autre que la fin suffit (déclin, abandon, berserk, la première région qui est
  une cible, `Game::target` région par région) ; les coups ne sont calculés en entier qu'aux
  feuilles. Recherche identique au bit près (empreintes `lab3 speed` et selfplay de contrôle,
  test `forced_moves_agree_with_legal`) ; selfplay de contrôle 8,65 → 7,08 s ; selfplay GPU
  2 760 → **~3 300 positions/s**, GPU 940 000 évaluations/s, fils de jeu en attente 10 %.
- **Encodage allégé, même entrées au bit près** (`encode_with`, `legal_with`) : les régions et les
  jetons de chaque peuple relevés en un passage (au lieu d'un parcours de la carte par peuple et
  par usage), les voisines tenues par l'armée par un ET de bits (`Map::hoods`), et à une feuille
  en conquête les cibles calculées une fois pour ses coups et son encodage. Empreintes
  inchangées ; selfplay de contrôle 7,08 → 6,38 s. Selfplay GPU ~3 360 positions/s (+2 %) : le
  GPU est à son plafond (~950 000 évaluations/s, lots de ~150 positions), les fils de jeu
  l'attendent 17 % du temps. Le frein est de nouveau le GPU.
- **Le shader, étape par étape** (`LAB3_VULKAN_PHASES=1`, lots de 512) : une position prend
  ~134 µs du début à la fin de son groupe de travail (110 seule sur son unité de calcul, 147 à
  deux), quel que soit le nombre de régions (24 ou 40) ; 128 positions à la fois (2 groupes par
  unité, bornés ensemble par les registres, 192 par fil, et la mémoire partagée, 32 Ko) :
  128 / 134 µs ≈ 950 000 positions/s, le plafond du selfplay. Le shader est limité par la latence,
  pas par le calcul (~10 % de ce que peuvent les unités matricielles en bf16). Les étapes
  globales (un vecteur par position : glob_in, part de g, g1, g2, politique de la partie,
  valeur) en font ~40 %. Une position enregistrée coûte ~283 évaluations (recherches rapides de
  la recette comprises, la moitié des simulations retombant sur une évaluation gardée).
- **Produits matrice-vecteur raccourcis** (`gemv4`) : seize lectures de poids à la fois au lieu
  de douze (un seul aller-retour vers la mémoire pour 128 entrées : part de g 12,6 → 8,7 µs ;
  vingt-quatre n'aident pas g1 et ralentissent le reste) ; pour les entrées creuses (octets de la
  partie, sorties de relu), un vote du sous-groupe ne garde que les entrées non nulles, dans le
  même ordre (g2 12 → 8 µs ; sur l'entrée dense de g1, le vote coûte plus qu'il ne gagne, 14 →
  25 µs : pas là). Écarts au processeur inchangés au bit près ; banc 821 000 → 907 000
  positions/s de GPU ; selfplay GPU 3 360 → **~3 520 positions/s**, ~1 000 000 évaluations/s.
- **Essai de `g1` sur les matrices coopératives** (23 octobre, non retenu) : `ub`, les moyennes
  des voisines, n'est plus lu au moment de `g1` ; son espace peut recevoir les entrées globales
  décomposées en bf16 haute/basse, sans ajouter de mémoire partagée. Deux prototypes, trois
  produits pour approcher le f32 : quatre sous-groupes en deux passages avec sorties f32 dans
  `scratch`, puis huit sous-groupes en un passage avec sorties haute/basse dans `ub`. Les deux
  passent le contrôle numérique sur les réseaux 64 et 128, sans élargir les tolérances. Le
  compilateur confirme les mêmes ressources pour la référence et les prototypes : 192 VGPR,
  32 Kio de mémoire partagée allouée, sans débordement des registres. Mais le
  banc en flux (lots de 128) ralentit : ~1,20–1,21 million d'évaluations/s pour le premier,
  ~1,03 pour le second, contre ~1,25–1,29 pour la référence. En vrai selfplay, le meilleur
  prototype perd **4,7 %** : référence 3 837 positions/s, prototype 3 657 (ordre ABBA, deux
  essais de 42 s chacun par version, première tranche de 10 s écartée ; c060, n = 200, 20 fils,
  16 parties par fil, graines identiques, 2 à 5 joueurs). Chaque matrice répète un seul vecteur
  sur ses 16 lignes ; le coût des produits, conversions et barrières annule le bénéfice.
  Prototypes retirés ; sources de référence restaurées. Détails et variantes conservés hors
  Git dans `lab3/runs/g1-matrix-20261023-031627/RESULTATS.md`. Cet essai ne tranche pas la piste
  de plusieurs positions par groupe ni celle des trois groupes résidents par unité de calcul.
- **Essai de trois groupes résidents** (23 octobre, non retenu) : `ub` réduit à une tranche de
  16 colonnes pour toutes les régions, réutilisée pour les entrées basses de `node_in` puis les
  moyennes des voisines pendant `n1`. Poids chargés par tranche, une chaîne de produits par
  tuile pour `n1` et `n2`, prélecture de `gemv4` ramenée à huit poids. Le shader de 256 fils
  descend de 192 à **120 VGPR**, de 32 à **21 Kio de LDS allouée**, sans débordement des
  registres ; 12 sous-groupes/SIMD au lieu de 8, ressources compatibles avec trois groupes
  résidents. Le contrôle numérique existant passe pour les réseaux 64 et 128, sans modifier les
  tolérances. Une variante à 128 fils (240 VGPR, 6 sous-groupes/SIMD) est moins rapide au banc.
  En flux, 20 000 lots de 512 : temps GPU −7,1 %, débit mural +5,4 %. Mais par lots de 128, le
  temps GPU augmente de 5,5 % (le débit mural augmente malgré cela : temps hors calcul GPU
  différent). En vrai selfplay, c060, n = 200, 20 fils, 2 à 5 joueurs, graine 777 :
  à 16 parties par fil, référence **3 837 positions/s**, prototype **3 718 (−3,1 %)**, ordre
  ABBA, deux essais de 42 s chacun par version, première tranche de 10 s écartée. Lots de
  ~133 et ~149 positions : la capacité supplémentaire profite peu aux petits lots, dont les
  synchronisations et les lectures supplémentaires augmentent la latence. Dernier contrôle
  avec 24 parties par fil (un essai de 42 s chacun) : référence 3 912, prototype 3 828 (−2,1 %).
  Aucun changement du shader retenu. Variantes, statistiques du pilote et journaux conservés
  hors Git dans `lab3/runs/three-groups-20261023-032833/RESULTATS.md`.
- **Deux défauts du compilateur de RADV** (Mesa 26.2.4, ACO, gfx1201), trouvés en comparant son
  assembleur (`RADV_DEBUG=shaders`) pour des variantes justes et fausses du shader. (1) En formant
  des paires d'instructions VOPD, ACO met une addition avec l'instruction qui produit l'une de ses
  entrées (conversion bf16 → f32 de valeurs lues ensemble) : la paire lit l'ancienne valeur du
  registre, une colonne de la moyenne des voisines reçoit la valeur d'une autre. Il apparaît ou non
  selon l'ordre global des instructions (tableau ou `vec4`, écart des lignes, pooling changé
  ailleurs) ; `ACO_DEBUG=nosched-vopd` le fait disparaître, pour 1 % de vitesse : `src/vulkan.rs`
  le met avant de charger le pilote. (2) La conversion f32 → bf16 tronque au lieu d'arrondir au
  plus proche : arrondi fait dans le shader, les écarts au f32 reviennent au niveau du bf16 du
  processeur (logits 0,19 → 0,05).
- Retiré : `size.py` (mesure des tailles de réseau de lab2).

## Reprise généraliste sur le PC (23 octobre)

À la demande de l'utilisateur : continuer c060 de lab2 en mélangeant les tailles de tables dans
chaque cycle (2, 3, 4, 5 joueurs, même nombre de parties par taille), avec le moteur optimisé de lab3.
Le départ `lab3/runs/p2-3-4-5/c000.bin` est identique à `lab2/nets/c060.bin` ; les nouveaux réseaux
restent dans lab3. Les graines commencent à 30 000 061 pour ne pas rejouer les graines des 60 cycles
de lab2. La fenêtre est remplie avec de nouvelles parties avant le premier apprentissage.

Lancement à 03:55 CEST, sur GrosseBoite (RX 9070 XT), jusqu'à c020, 25 000 parties par cycle,
200 simulations par recherche, arènes de 460 donnes face à Difficile puis Normale. Génération
Vulkan, apprentissage ROCm bf16, arènes CPU. Les trois étapes sont successives : les fils CPU ne
sont plus partagés entre deux tâches simultanées. La fenêtre est mappée depuis les fichiers,
sans copie complète en RAM (32 Gio sur le PC).

Commande de reprise avec la même configuration (ne pas lancer en double) :

```sh
timeout --kill-after=30s 8h env DEVICE=vulkan MMAP=1 PARALLEL=0 THREADS=20 \
  GAMES=25000 DEALS=460 SEED_BASE=30000060 LAB3_CONCURRENT=16 \
  PYTHONUNBUFFERED=1 lab3/loop.sh 2,3,4,5 20
```

Lancée en service utilisateur `lab3-generalist-20261023.service`, indépendant du chat, avec une
limite de 8 h et arrêt de tout son groupe de processus. Journal des étapes :
`lab3/runs/p2-3-4-5/log.txt` ; détails du remplissage en cours :
`lab3/data/tmp/p2-3-4-5-c001.err` (rythme et parties terminées) et `.txt` (bilan final). Pendant l'apprentissage :
`lab3/runs/p2-3-4-5/cNNN-train.log`. Arrêt :
`systemctl --user stop lab3-generalist-20261023.service`.

Vérifications avant lancement : lecture RAM/mmap identique sur deux fichiers (clés, poids, indices
désordonnés, répétés et aux frontières) ; essai réel de 128 parties mêlant les quatre tailles,
35 lots GPU, validation, export et relecture des poids finis ; orchestration successive/parallèle
et reprise vérifiées avec des exécutables factices ; compilation release et `git diff --check`.

Premier cycle : 25 000 parties en 295 s, régime établi autour de 2 870 positions/s. Le deuxième
cycle atteint ensuite 3 715–3 752 positions/s avec le même exécutable, le même modèle et les mêmes
réglages. Le mmap n'intervient pas encore (uniquement à l'apprentissage) ; les anciens essais
écrivaient eux aussi les fichiers de parties. Un relevé `pidstat` pendant le premier cycle n'a
montré aucune attente disque.

Incident du diagnostic à 04:03–04:05 : l'agent a suspendu les processus de la campagne par SIGSTOP
pour lancer des essais comparatifs sans perdre les positions déjà produites. Mauvaise méthode
avec la file Vulkan soumise à l'avance : perte du contexte GPU, arrêt du cycle c002. Les essais
réalisés pendant cette suspension (`runs/rate-check-20261023`) sont non concluants et ne doivent
pas servir à comparer les graines ou les exécutables. Reprise à 04:05:54, c001 conservé, c002
rejoué ; échéance de sécurité maintenue à 11:55:58 CEST. Ne pas suspendre les processus GPU de
cette façon pour de futurs diagnostics.

## Mesure

Le protocole commun garde le format duplicate à 2, 3, 4 et 5 joueurs ; un réseau spécialisé ne
joue qu'à sa taille, et se compare à lab2 à cette taille seulement :

- à chaque pas, arènes à P joueurs face à Difficile (graine 1002) et Normale (1003), 460 donnes ;
  les réseaux de lab2 rejoués sur ces mêmes donnes pour comparer ;
- en fin de boucle (`lab3/runs/final.sh`), 920 donnes à P joueurs (graine 1004), n = 200 des deux
  côtés, face au réseau de départ (le c060 de lab2) ; courbe des réseaux sur les mêmes positions
  de validation (`lab3/py/curve.py`). L'essai depuis zéro (`lab3/runs/final-zero.sh`) se mesurait
  au c020 64x4 de lab2 (même taille de réseau, même fenêtre, autant de parties, mais à toutes les
  tailles) et à son meilleur réseau.

## Plan

1. ~~Boucle à 4 joueurs depuis gen0, 20 pas (comme la première boucle de lab2), puis mesures de
   fin.~~ Faite le 22 octobre : à moyens égaux, le spécialiste bat le généraliste de 8,4 points à
   4 joueurs (voir Résultats).
2. Spécialiser le généraliste : c060 de lab2 continué sur des parties à 4 joueurs, jusqu'au
   cycle 20, avec le nouveau moteur. Premier essai arrêté au pas 3 le 22 octobre (fenêtre d'un
   cycle, le réseau baissait : voir Résultats) ; relancé le 22 octobre avec la fenêtre remplie
   d'abord par c000 (cycles 1 et 2 de l'essai repris : mêmes parties) Relance arrêtée au pas 18 :
   gain bref, puis retour au niveau de c060 (voir Résultats).
3. Selon le résultat : les autres nombres de joueurs, un Expert du jeu qui choisit son réseau
   d'après le nombre de joueurs.
4. Le réseau sur GPU (demande de l'utilisateur, 22 octobre) : le moteur groupe déjà les positions
   de toutes ses parties.

## Résultats

Scores du protocole (`labtools/score.py`), à 4 joueurs seulement : l'écart à une victoire sur
quatre, de −25 (jamais) à +75 (toujours) ; n = 200.

### Sortir du plateau : réseau approfondi 128x6, puis n = 400 (23 octobre, en cours)

Le plateau ne date pas de la reprise : dans lab2, c060 ne battait déjà plus c050 (+0,2 ± 1,5),
tout le gain de la troisième boucle venait de c041 à c050. La reprise de cette nuit repart de ce
plateau avec le même réseau 128x4 et la même recette : c015 contre c000 = −0,0 ± 2,2 (CPU),
−0,5 ± 2,1 (GPU). Pistes, de la plus prometteuse à la moins :

1. **Plus de capacité** : lab2 a mesuré qu'un réseau plus large aide la valeur, la possession et
   la politique quand les parties suffisent. Le shader accepte jusqu'à 7 couches en largeur 128
   (256 demanderait de le refaire : sa mémoire partagée ne tiendrait plus). Plutôt que de repartir
   de zéro, **approfondir** : `py/deepen.py` ajoute des couches dont les branches résiduelles partent
   de zéro, le réseau joue d'abord exactement comme l'ancien puis apprend avec plus de capacité.
2. **De meilleures cibles** : n = 400 pour les recherches complètes (avec n = 200 et 16 coups
   retenus, la politique améliorée reste proche du réseau).
3. **Un taux d'apprentissage plus bas** (ou une moyenne des poids) : chaque pas refait un cycle
   complet à 5e-4 sur un réseau mûr.
4. **Vérifier la recherche groupée** (jamais comparée en arène à l'ancienne ; toutes les parties
   de lab3 en viennent). Écartée : même distribution que l'ancienne (voir plus bas).

Essai de la piste 1, `deep6` : c015 approfondi à 6 couches (`runs/deep6/c016.bin`, mêmes sorties
que c015), fenêtre de départ c010…c016 de la reprise (liens dans `data/deep6`), puis pas 17 à 30
avec la boucle accélérée (duel contre c000 sur GPU à chaque pas). Selfplay à 6 couches ~3 280
positions/s (−12 %). Lancé à 13:52 en service utilisateur, 3 h au plus :

```sh
systemd-run --user --unit=lab3-deep6 --working-directory=$PWD --setenv=DEVICE=vulkan \
  --setenv=PARALLEL=0 --setenv=THREADS=20 --setenv=GAMES=25000 --setenv=SEED_BASE=30000060 \
  --setenv=RUN=deep6 --setenv=START=17 --setenv=LAB3_CONCURRENT=16 --setenv=PYTHONUNBUFFERED=1 \
  --setenv=PATH="$PATH" timeout --kill-after=30s 3h lab3/loop.sh 2,3,4,5 30
```

Suivi : `lab3/runs/deep6/log.txt` ; arrêt : `systemctl --user stop lab3-deep6`.

Premier pas (c017) : la boucle accélérée tourne de bout en bout en **407 s par pas** (parties
260 s, apprentissage 111 s dont 35 de chargement et compactage, duel 36 s), contre ~1 070 la
nuit dernière. c017 contre c000 : −3,4 ± 3,3 (arrêt après 184 donnes, « 0 plutôt que 3 »).

Pas 17 à 30, fini à 15:31 (14 pas, 1 h 39). Duels contre c000 (460 donnes au plus, arrêt
séquentiel) : c017 −3,4, c018 +1,3, c019 +1,7, c020 +0,6, c021 +2,4, c022 +0,7, c023 +4,4 et
c024 +3,2 (« gain de 3 points plutôt que 0 »), c025 +1,2, c026 −2,5, c027 −2,3 (arrêtés à 184
donnes), c028 +2,1, c029 +1,8, c030 +1,2. Confirmation sur 920 donnes neuves (graine 1005, sans
arrêt, 99 s sur GPU) : **c030 contre c000 = +0,9 ± 1,6**, pas significatif. Les mesures
d'apprentissage progressent pourtant à chaque pas, ce que le 128x4 ne faisait plus (sa politique
montait de 1,430 à 1,445) : politique 1,447 → 1,428, propriétaires 0,523 → 0,514 ; en
validation, politique 1,307 → 1,290, précision 49,0 → 49,6 %, valeur stable (~1,01 / 1,02 à
1,04). La capacité aide à apprendre, mais pas encore à jouer mieux : elle n'est pas le seul frein.

Suite, piste 2 : le même réseau de c031 à c040 avec n = 400 pour les recherches du selfplay
(`SIMS=400` ; duels toujours à n = 200), lancé à 15:40 de la même façon (`START=31`, jusqu'à
c040).

Pas 31 à 40, fini à 17:01 (1 h 21, ~530 s par pas : parties 355 s). Tous les duels contre c000
sont positifs, et 8 sur 10 concluent « gain de 3 points plutôt que 0 » : c031 +3,8, c032 +3,8,
c033 +3,8, c034 +3,6, c035 +3,0, c036 +4,0, c037 +2,2, c038 +4,4, c039 +3,1, c040 +3,6.
Confirmation sur 920 donnes neuves (graine 1005) : **c040 contre c000 = +4,9 ± 1,5** (2 joueurs
+5,8, 3 +6,5, 4 +5,0, 5 +2,2), contre +0,9 ± 1,6 pour c030. **De meilleures cibles sortent du
plateau**, d'un coup : le saut est là dès c031 (le premier réseau appris sur des parties à
n = 400), puis les duels restent vers +3,5 : un nouveau palier. La politique de validation monte
(1,289 → 1,307, précision 49,4 → 47,9 %) : les cibles à n = 400 sont plus tranchées, donc plus dures
à prédire, ce n'est pas un recul.

Suite, piste 3 : c041 à c050, n = 400, `LR=1.5e-4` (service `lab3-deep6-lr`, lancé à 17:04),
mesurée par un duel direct c050 contre c040 sur 920 donnes neuves.

Arrêté à 17:44 à la demande de l'utilisateur, pendant les parties de c045 (effacées à la reprise).
Duels contre c000 : c041 +4,6, c042 +3,1, c043 +7,1, c044 +5,7, tous « gain de 3 points plutôt
que 0 » (c043 et c044 arrêtés dès 184 donnes, ± 3,5 à 3,7). Pertes d'entraînement un peu plus
basses qu'à 5e-4 (valeur 1,012 → 1,003, propriétaires 0,514 → 0,510), validation stable. Prometteur,
pas encore mesuré : confirmation c044 contre c040 (920 donnes, graine 1005) à faire. Reprise :
même commande avec `START=45`.

Confirmation sur 920 donnes neuves (graine 1005) : **c044 contre c040 = +1,9 ± 1,5** (2 joueurs
+2,0, 3 +2,5, 4 +2,0, 5 +1,1) : même sens à toutes les tailles, pas encore significatif (1,3 écart
type). Boucle reprise à 18:46 (`START=45`, jusqu'à c050, 90 min au plus) ; mesure prévue : c050
contre c040.

Pas 45 à 50, fini à 19:40. Duels contre c000, tous « gain de 3 points plutôt que 0 » : c045 +3,8,
c046 +5,8, c047 +4,5, c048 +4,8, c049 +5,8, c050 +4,1. Confirmations sur 920 donnes neuves
(graine 1005) :

| Duel | Score | 2 joueurs | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| c050 contre c040 | **+2,9 ± 1,5** | +2,4 | +2,4 | +4,5 | +2,5 |
| c050 contre c000 | **+5,8 ± 1,6** | +5,8 | +4,3 | +7,1 | +6,0 |
| (c040 contre c000) | +4,9 ± 1,5 | +5,8 | +6,5 | +5,0 | +2,2 |

**Le taux plus bas aide** : là où les dix pas à 5e-4 restaient sur le palier de n = 400 (duels
contre c000 à plat vers +3,5), les dix pas à 1,5e-4 battent c040 de 2,9 points (environ 2 écarts
types), dans le même sens à toutes les tailles. Le gain face à c000 (+0,9 de plus) est plus petit
que le duel direct : les scores ne s'additionnent pas d'un adversaire à l'autre. Suite prévue :
l'apprentissage en continu (`CONTINUOUS=1`) depuis c050.

Pendant c032 : la fenêtre compacte passe de la RAM à un mappage des fichiers `.pos.packed.npz`
(`MAP_POPULATE`, lus d'un trait au chargement ; seuls les petits champs sont copiés en RAM). En
RAM, la fenêtre aurait pris ~31 Go vers c060 sur les 32 du PC ; mappée, le noyau peut rendre les
pages. Lot de 512 : 11,3 ms (2 ms en RAM, 33,5 ms sans `MAP_POPULATE`, 81 ms pour l'ancien
mmap des fichiers denses). Sur c032 : chargement 23 s au lieu de 45, entraînement 80 s au lieu
de 72, 109 s en tout au lieu de 122 ; 7 Go utilisés.

Piste 4 écartée sans arène : dans un tour de la recherche groupée, chaque coup de la racine qui a
`considered` visites en reçoit une (`root_picks` prend tous ceux du tour), et le score d'un coup
encore à visiter ne change pas pendant le tour (seuls les coups déjà visités, qui n'y sont plus
éligibles, reçoivent des valeurs). L'ancienne recherche visitait donc les mêmes coups, dans un autre
ordre ; les sous-arbres sont disjoints et les évaluations les mêmes (empreintes communes : un seul
envoi). Seul l'ordre des tirages (déterminisation) change : même distribution de recherches.

Piste 3 préparée : `LR` dans la boucle (`train.py --lr`, 5e-4 par défaut). Chaque pas repart d'un
AdamW neuf et remonte au taux maximal ; une moyenne des poids pendant le cycle ferait doublon avec
la descente de OneCycle, qui finit presque à zéro.

Plus loin que la piste 3 : `CONTINUOUS=1` dans la boucle (`train.py --state-in/--state-out`).
L'optimiseur AdamW, le réseau qui apprend et une moyenne exponentielle de ses poids (0,9998 par
lot, ~5 000 lots) passent d'un pas à l'autre dans `runs/RUN/cNNN.state.pt` ; taux constant (`LR`,
montée de 500 lots au premier pas seulement) ; le réseau exporté, qui joue et se mesure, est la
moyenne, comme dans KataGo. Ordre prévu : piste 3 (c041–c050, `LR=1.5e-4`, OneCycle), puis le
continu depuis c050.

### Reprise généraliste depuis c060, arrêt à la validation de c016 (23 octobre)

Campagne locale du 23 octobre, de 03:55:58 à 07:56:47 CEST, environ 4 h : 16 cycles de 25 000
parties à 2–5 joueurs mélangés, 12 réseaux exportés (c004 à c015), c016 non exporté à cause d'une
erreur de conversion NumPy/PyTorch en validation. Les données de c016 sont conservées.

Scores face aux IA du jeu, mêmes graines 1002/1003, 460 donnes et 200 simulations ; tailles
pondérées également, intervalles à 95 %. Les résultats de c060 sont ceux archivés de lab2.

| Réseau | Difficile | Normale |
| --- | --- | --- |
| Départ : lab2 c060 | +30,6 ± 2,2 | +27,8 ± 2,1 |
| c004 | +28,8 ± 2,0 | +27,1 ± 2,1 |
| c005 | +26,8 ± 2,2 | +25,2 ± 2,1 |
| c006 | +28,7 ± 2,2 | +25,3 ± 2,0 |
| c007 | +28,0 ± 2,1 | +24,7 ± 2,2 |
| c008 | +28,4 ± 2,2 | +25,8 ± 2,0 |
| c009 | +28,3 ± 2,2 | +26,3 ± 2,0 |
| c010 | +28,8 ± 2,3 | +24,5 ± 2,1 |
| c011 | +29,1 ± 2,1 | +25,9 ± 2,0 |
| c012 | +28,4 ± 2,2 | +26,6 ± 2,2 |
| c013 | +29,0 ± 2,0 | +24,3 ± 2,1 |
| c014 | +29,5 ± 2,2 | +25,2 ± 2,0 |
| c015 (dernier exporté) | +29,8 ± 2,2 | +25,6 ± 2,0 |

**Aucun gain démontré face à c060.** Comparaison appariée sur les donnes communes avec
`labtools/score.py` : c015 − c060 = **−0,9 ± 2,8 points face à Difficile**, **−2,2 ± 2,5 face à
Normale**, les deux indécis. Duel direct ensuite (23 octobre, 460 donnes, graine 1004, n = 200
des deux côtés, 859 s sur le CPU) : **c015 contre c000 = −0,0 ± 2,2 points** (2 joueurs +0,2,
3 joueurs −1,8, 4 joueurs −2,4, 5 joueurs −3,0 d'avance moyenne) ; test séquentiel : « 0 plutôt
que 3 points ». C'est une comparaison de performances contre les IA du jeu, pas
un duel direct entre ces réseaux. Les conclusions de lab2 sur les limites de ces adversaires
restent valables. Rapports : `runs/p2-3-4-5/comparison-c015-c060-{hard,normal}.txt`.

Précision de validation 49,4 % à c004, 48,5 % à c015 : fenêtres différentes, donc pas une mesure
isolée de progrès ou de régression. Le départ c060 conserve sa place de référence.

**Temps observés sur le PC** : 25 000 parties en 227–232 s après le premier cycle ; calcul
d'apprentissage GPU en 130–284 s, étape entière en 173–414 s (fenêtre de 3,4 à 6,8 millions de
positions) ; les deux arènes CPU en environ 9 min par génération, devenues l'étape la plus longue.

**Correction du plantage, sans reprise de l'apprentissage** : 339 969 positions de validation
à c016, soit 166 lots de 2048 et un dernier lot d'une seule position. Dans ce cas, la vue d'un
champ f32 est jugée contiguë par NumPy, mais conserve le pas de 6210 octets du format `.pos`,
non divisible par 4. `np.ascontiguousarray` ne copie pas cette vue ; `torch.from_numpy` la
refuse. Erreur reproduite aussi sur un tableau en RAM : indépendante du mmap. `batches` force
désormais une copie C de chaque champ, y compris pour un lot singleton.

Tests : trois régressions `py/test_train.py` (singleton, dernier lot singleton en RAM et sur
plusieurs fichiers mappés, sélection vide) ; lecture du dernier singleton réel de c016 sur CPU
et GPU ; inférence GPU de validation sur les 2049 dernières positions réelles, moyenne pondérée
vérifiée. Aucun optimiseur exécuté, aucun modèle modifié ou créé. Campagne laissée arrêtée et
suivi automatique déjà supprimé, conformément à la demande de l'utilisateur.

### Spécialisation à 4 joueurs, relance arrêtée au pas 18 (22 octobre)

Fenêtre remplie par c000 (4 cycles) avant le premier entraînement : une `c7a.16xlarge` Spot,
286 min, 3,53 $. Arrêtée au pas 18 par l'utilisateur, sans mesures de fin. Réseaux et arènes :
`lab3/runs/p4`.

| Réseau | Difficile | Normale | Valeur | Possession | Politique | Précision |
| --- | --- | --- | --- | --- | --- | --- |
| c000 (c060 de lab2) | +24,5 ± 2,3 | +17,6 ± 2,4 | | | | |
| c004 | +29,9 ± 2,4 | +21,4 ± 2,2 | 1,151 | 0,538 | 1,291 | 49,4 % |
| c005 | +30,8 ± 2,2 | +23,6 ± 2,2 | 1,159 | 0,538 | 1,290 | 49,4 % |
| c006 | +26,7 ± 2,3 | +21,4 ± 2,3 | 1,161 | 0,538 | 1,292 | 49,2 % |
| c007 | +26,9 ± 2,2 | +20,1 ± 2,3 | 1,159 | 0,538 | 1,293 | 49,2 % |
| c008 | +26,7 ± 2,3 | +18,3 ± 2,4 | 1,158 | 0,539 | 1,295 | 49,0 % |
| c009 | +26,0 ± 2,2 | +18,3 ± 2,3 | 1,161 | 0,539 | 1,296 | 49,1 % |
| c010 | +24,2 ± 2,3 | +16,2 ± 2,2 | 1,188 | 0,539 | 1,298 | 48,9 % |
| c011 | +25,1 ± 2,3 | +19,2 ± 2,2 | 1,200 | 0,541 | 1,302 | 48,7 % |
| c012 | +26,3 ± 2,3 | +16,5 ± 2,2 | 1,205 | 0,542 | 1,307 | 48,7 % |
| c013 | +25,7 ± 2,3 | +19,5 ± 2,3 | 1,394 | 0,548 | 1,338 | 47,5 % |
| c014 | +25,2 ± 2,3 | +20,8 ± 2,3 | 1,330 | 0,549 | 1,329 | 47,8 % |
| c015 | +23,0 ± 2,4 | +18,7 ± 2,3 | 1,334 | 0,553 | 1,339 | 47,5 % |
| c016 | +16,8 ± 2,3 | +13,3 ± 2,3 | 1,378 | 0,554 | 1,345 | 47,1 % |
| c017 | +22,3 ± 2,3 | +18,9 ± 2,1 | 1,320 | 0,553 | 1,346 | 46,9 % |
| c018 | +24,5 ± 2,3 | +18,4 ± 2,3 | 1,336 | 0,555 | 1,357 | 46,5 % |

- **Gain bref, puis rien** : +6 points à c004–c005 (appris sur les parties de c060 lui-même), puis
  retour au niveau de c060 dès c010, et un creux à c016.
- **La valeur se dégrade par sauts** (pièces au pas 10, valeur au pas 13 : 1,19 → 1,29 à
  l'entraînement), alors que les parties ne changent pas (longueur, parts partagées, pièces, poids
  de tirage vérifiés sur la machine, cycles 1 à 14) : c'est l'entraînement, pas les données.
- L'auto-jeu à 4 joueurs seul, depuis le généraliste, n'améliore pas le réseau.

### Spécialisation à 4 joueurs, premier essai arrêté (22 octobre)

`lab3/loop.sh 4 20` depuis le c060 de lab2, fenêtre minimale de 1 000 000 positions : une
`c7a.16xlarge` Spot, 64 min, 0,80 $, arrêtée au pas 3. Parties d'un cycle 708 à 884 s (48 cœurs
sauf le premier), entraînement 697 à 738 s (16 cœurs). Validation sur les parties réservées de
la fenêtre de chaque pas.

| Réseau | Difficile | Normale | Valeur | Possession | Politique | Précision |
| --- | --- | --- | --- | --- | --- | --- |
| c000 (c060 de lab2) | +24,5 ± 2,3 | +17,6 ± 2,4 | | | | |
| c001 | +18,9 ± 2,2 | +11,1 ± 2,2 | 1,173 | 0,541 | 1,309 | 48,1 % |
| c002 | +18,2 ± 2,3 | +12,5 ± 2,2 | 1,211 | 0,546 | 1,312 | 48,0 % |
| c003 | +15,1 ± 2,3 | +9,5 ± 2,3 | 1,201 | 0,561 | 1,377 | 44,9 % |

- **Le réseau perd 8 à 9 points en trois pas.** Cause probable : à 990 000 positions par cycle, la
  fenêtre (minimum 1 000 000, croissance lente) ne gardait qu'un cycle, et un pas tire 4 fois
  les positions de son dernier cycle : quatre passes, au taux plein, sur les mêmes 25 000
  parties. La valeur retient leurs issues : 1,096 à l'entraînement contre 1,173 à 1,211 en
  validation. Le c060 apprenait sur 11 cycles (275 000 parties), 0,36 passe par pas.
- La fenêtre n'aurait pris un deuxième cycle qu'au pas 5, et 5 au pas 20. D'où la relance avec
  la fenêtre remplie avant le premier entraînement (Système actuel).
- Réseaux et arènes : `lab3/runs/p4-try1` sur mini. Les parties des cycles 1 et 2 (jouées par
  c000) resservent à la relance ; celles des cycles 3 et 4 (c001 et c002), périmées, effacées du
  bucket.

### Boucle à 4 joueurs depuis zéro (22 octobre)

`lab3/loop.sh 4 20` depuis gen0 (fenêtre de 250 000 positions, 64x4), puis
`lab3/runs/final-zero.sh` (résultats `lab3/runs/p4-zero`, `lab3/runs/p4-zero-final`, parties
`lab3/data/p4-zero` : dossiers renommés ensuite) : une `c7a.16xlarge` Spot, 172 min, 2,17 $. 25 000 parties par cycle, 11
positions par partie au hasard, 37,5 en fin de boucle ; parties 127 à 471 s, entraînement 145 à
339 s ; fenêtre de 3 cycles au pas 20 (2,8 millions de positions), comme la première boucle de
lab2.

Arènes de chaque réseau à 4 joueurs (460 donnes) ; à côté, pour comparer, le même pas de la
première boucle de lab2 (même réseau, même fenêtre, autant de parties, mais à toutes les
tailles) sur ses 115 donnes à 4 joueurs, environ ±4,5 ; mesures de validation de chaque pas sur
les parties réservées de sa fenêtre (à 4 joueurs : pas comparables à celles de lab2).

| Réseau | Difficile | Normale | lab2 : Difficile | lab2 : Normale | Valeur | Possession | Politique | Précision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c001 | −22,9 ± 0,6 | −23,6 ± 0,6 | −24,3 | −24,3 | 0,828 | 0,284 | 1,237 | 54,6 % |
| c002 | −22,2 ± 0,8 | −23,3 ± 0,6 | −23,3 | −22,8 | 0,714 | 0,272 | 1,202 | 54,9 % |
| c003 | −15,8 ± 1,4 | −13,7 ± 1,6 | −19,6 | −15,8 | 0,825 | 0,468 | 1,635 | 32,7 % |
| c004 | −15,9 ± 1,3 | −11,8 ± 1,6 | −17,7 | −14,1 | 0,881 | 0,468 | 1,606 | 33,3 % |
| c005 | −16,3 ± 1,4 | −14,0 ± 1,5 | −12,3 | −12,4 | 1,076 | 0,475 | 1,536 | 37,8 % |
| c006 | −14,3 ± 1,5 | −12,9 ± 1,5 | −13,8 | −14,6 | 1,097 | 0,469 | 1,510 | 38,4 % |
| c007 | −12,8 ± 1,5 | −12,5 ± 1,5 | −11,4 | −12,2 | 1,121 | 0,465 | 1,542 | 37,9 % |
| c008 | −11,6 ± 1,6 | −10,5 ± 1,7 | −14,2 | −11,5 | 1,108 | 0,448 | 1,476 | 40,1 % |
| c009 | +0,1 ± 1,9 | −0,6 ± 1,9 | −8,2 | −5,4 | 1,119 | 0,451 | 1,477 | 41,0 % |
| c010 | −0,8 ± 1,9 | −1,0 ± 2,0 | −6,5 | −5,4 | 1,120 | 0,448 | 1,473 | 41,0 % |
| c011 | +5,2 ± 2,1 | +3,8 ± 2,1 | −1,5 | +0,7 | 1,132 | 0,460 | 1,450 | 42,4 % |
| c012 | +6,1 ± 2,1 | +7,9 ± 2,2 | +5,7 | +3,3 | 1,138 | 0,471 | 1,421 | 43,6 % |
| c013 | +9,0 ± 2,3 | +8,3 ± 2,4 | +11,6 | +3,2 | 1,146 | 0,484 | 1,411 | 44,2 % |
| c014 | +10,9 ± 2,3 | +10,4 ± 2,3 | +9,5 | +9,3 | 1,147 | 0,494 | 1,409 | 44,2 % |
| c015 | +13,0 ± 2,3 | +11,2 ± 2,2 | +7,7 | +8,5 | 1,162 | 0,503 | 1,400 | 44,4 % |
| c016 | +16,1 ± 2,4 | +10,8 ± 2,3 | +11,4 | +8,6 | 1,164 | 0,518 | 1,386 | 44,9 % |
| c017 | +15,8 ± 2,3 | +10,3 ± 2,3 | +14,9 | +8,6 | 1,157 | 0,526 | 1,383 | 45,3 % |
| c018 | +13,3 ± 2,2 | +10,9 ± 2,2 | +16,1 | +8,2 | 1,152 | 0,527 | 1,384 | 45,3 % |
| c019 | +15,4 ± 2,4 | +10,8 ± 2,2 | +12,3 | +10,0 | 1,155 | 0,528 | 1,384 | 45,4 % |
| c020 | +13,6 ± 2,3 | +9,3 ± 2,2 | +13,0 | +6,7 | 1,155 | 0,530 | 1,380 | 45,4 % |

Les réseaux de lab2 rejoués sur les mêmes donnes à 4 joueurs :

| Réseau | Difficile | Normale |
| --- | --- | --- |
| lab3 c020 (64x4, 4 joueurs, 20 cycles) | +13,6 ± 2,3 | +9,3 ± 2,2 |
| lab2 c020 (64x4, toutes tailles, 20 cycles) | +8,8 ± 2,3 | +8,5 ± 2,3 |
| lab2 c060 (128x4, toutes tailles, 60 cycles) | +24,5 ± 2,3 | +17,6 ± 2,4 |

Courbe sur les mêmes positions, parties réservées des cycles 19 et 20 de lab3 (jouées par c017 et
c018 ; 91 968 positions) :

| Réseau | Valeur | Possession | Politique | Précision |
| --- | --- | --- | --- | --- |
| lab3 c005 | 1,381 | 0,730 | 1,824 | 29,9 % |
| lab3 c010 | 1,215 | 0,663 | 1,612 | 36,1 % |
| lab3 c015 | 1,182 | 0,563 | 1,439 | 42,7 % |
| lab3 c020 | 1,161 | 0,531 | 1,379 | 45,6 % |
| lab2 c020 (64x4) | 1,182 | 0,575 | 1,491 | 39,9 % |
| lab2 c060 (128x4) | 1,153 | 0,555 | 1,518 | 39,9 % |

Arènes directes, 920 donnes à 4 joueurs, graine 1004, lab3 c020 face à trois réseaux de lab2 :

| Face à | Score | Victoires | Rang moyen |
| --- | --- | --- | --- |
| lab2 c020 (64x4) | +8,4 ± 1,5 | 33,4 % | 2,27 |
| lab2 c060 (128x4) | −9,0 ± 1,2 | 16,0 % | 2,92 |

- **À moyens égaux, le spécialiste gagne à sa taille** : +8,4 ± 1,5 face au généraliste de même
  réseau, de même fenêtre et d'autant de parties (toutes à 4 joueurs pour lui, un quart pour
  lab2) ; il prédit mieux le vainqueur (valeur 1,161 contre 1,182) ; face aux IA du jeu, +4,8
  face à Difficile et +0,8 face à Normale sur les mêmes donnes. Son avance mêle la
  spécialisation et 4 fois plus de parties à 4 joueurs.
- **Même plafond que la première boucle de lab2** à partir de c016 (petit réseau, petite
  fenêtre) ; le généraliste plus grand et plus entraîné reste loin devant (c060 : −9,0 en arène
  directe, meilleure valeur même sur ces parties qu'il n'a jamais vues).
- La politique de la courbe avantage lab3 par construction (ses cibles sont les recherches de
  lab3, qui a joué ces parties) ; la valeur, elle, compare des prédictions du vainqueur.
- Positions des cycles 1 à 16 effacées du bucket ensuite (`lab3/prune.sh 4`, 25,8 Go gardés, aujourd'hui sous `p4-zero`).
