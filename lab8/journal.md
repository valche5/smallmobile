# Labo 8 : la recherche sur des plans de tour, avec un transformer neuf

Ouvert le 26 octobre 2026 à la demande de l'utilisateur, après lab7 (figé), pour chercher sur des
tours entiers écrits par un décodeur, avec un réseau neuf qui commence par imiter c130. Plan et
raisons : [lab.md](lab.md).

Un premier lab8 (commit `fb6456a`, sorties « plan de tour » ajoutées au réseau de c130) a été retiré
par l'utilisateur (`c1b3131`) ; celui-ci repart de la discussion du 26 octobre. Le retrait avait
laissé `lab8` et `lab8/brain` dans les membres de `Cargo.toml`, ce qui cassait l'espace de travail :
retirés, ils reviendront avec le code de la phase 0.

## Machine

Mesures du 26 octobre sur ce PC (i7-12700KF, RX 9070 XT, RADV) avec `target/release/lab7`
(`cargo build --release -p smallmobile-lab7 --no-default-features --features vulkan`) :

- `lab7 check-gpu --net lab7/nets/c130.bin --device vulkan` : 148 positions, écart maximal victoire
  0,0021, pièces 0,069, logits 0,575.
- `lab7 bench-gpu` : positions par seconde selon la taille des lots : 64 → 689 000, 256 →
  1 007 000, 512 → 1 081 000, 1 024 → 1 148 000, 2 048 → 1 184 000.
- `lab7 selfplay --agent az:net=lab7/nets/c130.bin,n=400 --games 640 --device vulkan` (recette de
  lab7 : bruit, une recherche complète sur 8) : 640 parties en 11 s.
- `lab7 arena`, c130 contre c130 à n = 400, 80 donnes (graine 81002) : 21,3 s, 13 223 décisions,
  100 ms en moyenne, p95 336 ms.
- PyTorch 2.14.1+rocm7.14 (`lab3/py/.venv`) voit la carte.

## Phase 0 : la base (26 octobre)

Code :
- `smallmobile-brain8` : `tokens` (jetons, relations, entrées d'un pas du décodeur), `net` (le réseau
  sur processeur, référence de `py/model.py`), `plan` (découpage en plans, écriture d'un plan par le
  décodeur). Coups, savoir, hasard, encodage : ceux de `smallmobile-brain7`, par dépendance.
- `smallmobile-lab8` : `record` (parties d'imitation découpées en plans, sur GPU comme le selfplay de
  lab7), `agent` (le décodeur seul comme joueur des arènes de lab7 : `plan:net=…`), `fixture`
  (comparaison à PyTorch) ; programme `lab8` : `record`, `arena`, `fixture`, `check-net`, `speed`.
- `py/` : `model.py` (encodeur, têtes, décodeur), `data.py` (lecture des parties en lots),
  `train.py` (apprentissage, validation face au réseau de c130), `check.py`, `setup.sh`.

Choix précisés en écrivant le code :
- Un plan s'arrête aussi quand la phase change : le choix du peuple est un plan à lui seul (les
  coûts des conquêtes ne sont encodés qu'en phase de conquête), le redéploiement commence un plan
  neuf (une lecture de position de plus, mais à jour). Les coups forcés font partie du plan sans être
  des pas du décodeur.
- Un pas du décodeur reçoit, en plus du coup précédent, 20 entrées de la position (phase, tour en
  cours, jetons en main, défenses en attente, redéploiement commencé) et le nombre de coups déjà
  joués sur la région du coup précédent.
- L'enregistreur garde aussi ce que dit le réseau de c130 seul (a priori à chaque décision, valeur au
  début du plan) : la référence du critère de la phase 2.

Vérifications :
- `cargo test -p smallmobile-lab8 -p smallmobile-brain8 --release` : 4 tests passés. Les jetons
  reprennent chaque entrée de lab7 ; Rust et PyTorch donnent les mêmes têtes et logits (écart
  maximal 3e-6 sur 49 plans de parties au hasard, réseau au hasard 32 × 2 : `tests/fixtures`) ; le
  décodeur joue des parties entières de 2 à 5 joueurs sans coup impossible ; les parties
  d'imitation (c130 à n = 8) se découpent en plans dont la politique, l'a priori et les parts de
  victoire font 1.
- `cargo clippy -p smallmobile-lab8 -p smallmobile-brain8 --all-targets -- -D warnings` : passé ;
  compilation sans Vulkan (`--no-default-features`) : passée.
- Un réseau entraîné (1 500 lots sur 400 parties) donne aussi les mêmes sorties en Rust et en
  PyTorch : écart maximal 7e-6.

Débits (machine ci-dessus) :
- `lab8 record --games 400 --device vulkan` (c130, n = 400 à chaque décision, sans bruit) : 26 s,
  32 101 plans, 134 Mo.
- `train.py --bench 60`, réseau 256 × 6 (7 882 522 paramètres) : 2 354 plans par seconde.
- `lab8 speed` (un cœur, f32) : encodeur 19,5 ms par position, pas du décodeur 0,30 ms.

Incident : faute mémoire du GPU à la validation (`Memory access fault`), reproductible. Cause : les
distances entre régions (`data.distances`) calculées sous `autocast` bf16 sur un tenseur étiré (pas
nul entre les lots) ; le produit par lots bf16 de ROCm 7.14 lit alors hors de la mémoire. Corrigé :
en f32, sur des tenseurs contigus.

Essai de bout en bout, 1 500 lots sur ces 400 parties seulement : la valeur apprend les résultats par
cœur (perte 0,13 à l'apprentissage, 3,6 à la validation), la politique atteint 37 % de précision sur
les parties mises de côté (c130 seul : 60 %). Rien à en conclure, sinon que la chaîne marche : la
phase 2 se fera sur les 100 000 parties.

### Forme compacte des parties d'imitation (26 octobre)

Demande de l'utilisateur, d'après les labos précédents : les données doivent tenir en mémoire à
l'apprentissage (31 Go). Le premier format faisait 4,4 ko par plan, 33 Go pour 100 000 parties. Sur
400 parties : les entrées des régions faisaient 66 % du total (87 % d'octets nuls, et les colonnes de
la carte répétées à chaque plan), celles de la partie 19 % (85 % de nuls), les voisines 7 %
(identiques pour toute la partie).

Forme compacte (`record.rs`, en tête) : la carte une fois par partie (15 colonnes des régions,
voisines) ; des autres entrées, un masque de bits et les seuls octets non nuls ; les 30 valeurs en
f16. Sur les mêmes 400 parties (graine 81000000) : 35,2 Mo au lieu de 141,8 (× 4,0), 1 097 octets
par plan ; 100 000 parties ≈ 8,8 Go. Le plus gros : les masques des régions (27 %), les coups
possibles (15 %).

Vérifications : le test `imitation_games_split_into_plans` relit la forme compacte en Rust et la
compare aux entrées pleines (`lab8 record --raw true`), octet par octet ; `python data.py DIR` fait de
même pour la lecture Python (32 101 plans identiques) ; `half_rounds_to_nearest` vérifie la
conversion en f16. Décodage d'un lot de 256 plans : 5,2 ms sur un fil (le GPU en apprend un en
~100 ms) ; apprentissage `--bench 80` : 2 546 plans par seconde, comme avant.

### Débit des parties d'imitation (26 octobre)

Question de l'utilisateur : peut-on accélérer la génération ? `lab8 record` affiche maintenant la
répartition du temps des fils de jeu et ce qu'ils envoient au GPU ; `--concurrent` règle le nombre
de parties par fil. Sur 2 000 parties (graine 81000000) :

| Parties par fil | Durée | Parties/s | Attente du GPU | Évaluations/s | Lot moyen |
| --- | --- | --- | --- | --- | --- |
| 16 | 119 s | 16,8 | 7,4 % | 708 000 | 101 |
| 32 | 130 s | 15,4 | 2,7 % | 648 000 | 90 |
| 64 | — | — | — | — | l'anneau du GPU ne les tient pas |

Le GPU n'est pas le frein : les fils l'attendent 7 % du temps, et une partie envoie seule ses
positions (le lot ne grossit pas avec le nombre de parties ; plus de parties par fil chargent les
caches et ralentissent). Les fils calculent ~86 % du temps (processeur occupé à ~78 %). `perf`
(1 000 parties) : copies mémoire 15 % (parties copiées par `determinize`, positions encodées copiées
dans les requêtes du GPU, cibles des conquêtes), allocations ~6 %, `encode` 8 %, l'arbre de la
recherche (`fingerprint`, `select`, `completed_into`) 16 %, les règles (`reach`, `target`, `block`,
`income_lines`…) ~15 %. Tout est dans le code de lab7 (figé) et du noyau : gagner 15 à 20 %
demanderait de copier et réécrire la recherche de c130 dans lab8, pour une génération faite une
fois (≈ 2 h). Non fait : l'effort d'optimisation va au moteur du transformer et à la recherche sur
plans, qui servent à chaque coup. 16 parties par fil reste le réglage par défaut.

## Nuit du 26 au 27 octobre (en autonomie)

Phase 1 lancée à 22 h 50 : `lab8 record --games 100000 --out lab8/data/c130 --seed 81000000
--device vulkan` (journal `lab8/data/c130.log`), 16,5 parties par seconde.

### L'apprentissage long (`train.py`)

Moyenne glissante des poids (EMA 0,9996 tous les 4 lots, mesurée et écrite à chaque validation),
reprise après interruption (`--state`), chaque validation écrite à part (`x-012000.bin`) pour choisir
la meilleure après coup, validation détaillée par phase et par nombre de joueurs, avec à côté c130
seul (`c130_pol`, `c130_acc`, `c130_val`), sur 50 000 plans mis de côté.

### La recherche sur des plans, version 1 (`brain/src/search.rs`)

Comme prévu par `lab.md` : candidats tirés du décodeur depuis une seule lecture de la racine (le plus
probable, puis des tirages à la température `t`, sans doublon), `m` mondes communs à tous les
candidats, fin simulée par plans (le plan le plus probable de chaque joueur jusqu'au prochain tour du
joueur de la racine), valeur mêlée à la première lecture après le candidat (`mix`), meilleure moyenne.
Toutes les voies avancent ensemble : à chaque tour, leurs lectures partent d'un coup au lecteur
(`search::Reader` : le processeur, ou le GPU). Le joueur des arènes (`plans:net=…,k=…,m=…[,gpu=1]`)
garde la suite du plan choisi entre deux décisions du jeu. Test : des parties entières de 2 à 5 joueurs
sans plan trop court, suites reprises.

Ajouts de la nuit :
- un candidat qui finit par un dé est joué à chacune de ses issues (renfort : réussite et échec ;
  Berserks : 0 à 3), pondérées par leur probabilité exacte (`die_chance`) : le dé du monde est
  changé jusqu'à donner l'issue voulue ;
- les pas du décodeur de toutes les voies d'un tour sont faits ensemble (`Net::step_batch`) : un
  pas seul lit ~9 Mo de poids, plus que les caches du cœur ; ensemble, chaque ligne de poids sert à
  toutes les voies. Mêmes logits au bit près (test `batched_steps_match_lone_ones`) ;
- le produit scalaire en AVX2 et FMA, choisi à l'exécution (les binaires sont compilés pour le x86-64
  de base) : encodeur sur processeur −24 %, pas seul −12 % (`LAB8_NO_AVX=1` pour comparer, même
  charge) ;
- l'arène affiche le temps des recherches : `plans:…,gpu=1,k=16,m=4` (réseau au hasard, pendant la
  phase 1) : lectures 227 s pour 133 159 positions, pas du décodeur 103 s pour 369 245 pas, reste
  0,7 s (temps cumulés des fils) ; 592 ms par décision. Le GPU, partagé, est le frein.

- modifications locales du meilleur candidat (DORA, option `mods=N`, éteinte par défaut) : rejoué
  jusqu'à une de ses décisions tirée au hasard (le décodeur lit les décisions imposées), puis réécrit
  à la température `t` ; les nouveaux candidats sont évalués dans les mêmes mondes.

Pas encore : le halving.

### Le décodeur sur le GPU

Lectures gardées dans la mémoire du GPU (6 144 places : sortie de l'encodeur des régions, `dec_reg`,
clés et valeurs de l'attention croisée, en bf16 ; `store.comp`), caches des voies (3 072, clés et
valeurs de l'attention causale, 128 pas au plus par plan), pas du décodeur en lots (`dec_in`,
`dec_self`, `dec_cross`, `dec_out`, et `gemm`, `layernorm`) : seuls les 608 logits reviennent à
l'hôte. La recherche est générique sur son moteur (`search::Engine`) : `Cpu`, `vulkan::Gpu` (tout sur le
GPU, `gpu=1`), `vulkan::GpuEncoder` (l'encodeur sur le GPU, le décodeur sur le processeur,
`gpu=enc`). `check-gpu` : logits du décodeur sur le GPU à 0,03 près de ceux du processeur (deux pas).

Arène de contrôle (k = 16, m = 4, pendant la phase 1) : `gpu=1` 2 023 ms par décision, `gpu=enc`
588 ms. Une recherche fait ~40 allers-retours successifs vers le GPU (lectures et pas de toutes ses
voies), et chacun coûte ~16 ms tant que la phase 1 occupe le GPU : à remesurer GPU libre.

### L'encodeur sur GPU (`src/vulkan.rs`, `shaders/`), phase 3 commencée

Le transformer ne tient pas dans la mémoire partagée d'un groupe (61 Ko d'activations par position) :
des kernels successifs sur des lots de positions, `gemm` (bf16 sur les matrices coopératives,
128 × 128 par groupe, A et W par la mémoire partagée, biais, gelu ou relu et résiduel au passage),
`embed`, `fixup`, `layernorm`, `attention` (relations calculées sur place), `pool`, `pack`. Le décodeur
reste sur le processeur : le GPU rend, par position, la sortie de l'encodeur des régions, `dec_reg` et
les clés et valeurs de l'attention croisée (172 ko en bf16) et les 20 têtes. Les fils de l'arène
partagent un seul lot (`vulkan::Shared`).

- `lab8 check-gpu`, réseau 256 × 6 initialisé comme PyTorch (sorties au hasard, σ 0,05) : écarts
  maximaux sur 200 positions : têtes 0,008, sorties des régions 0,031, logits du premier pas 0,023
  (logits jusqu'à 2,7) : l'arrondi bf16. Avec un réseau aux poids démesurés (σ 0,2 partout) les écarts
  montent (logits 0,74 pour 12) : activations énormes, même arrondi relatif.
- Débit mesuré pendant la phase 1, qui occupe le GPU : ~4 200 positions/s quelle que soit la taille
  des lots ; `LAB8_GPU_PROFILE=1` : `gemm` 73 %, `pack` 15 %, `attention` 8 %. Le premier `gemm`
  (64 × 64, W lue en mémoire globale à chaque pas de 16) prenait 1,7 fois plus. À remesurer GPU
  libre.
- Arène de contrôle (réseau au hasard) : `plans:…,gpu=1,k=8,m=2` contre `plan:` (le décodeur seul),
  3 joueurs, 6 donnes : 18 victoires sur 18 (la fin simulée joue les vraies règles), 284 ms par
  décision.

## Phase 1 : les parties d'imitation (27 octobre, 0 h 25)

`lab8 record --games 100000 --out lab8/data/c130 --seed 81000000 --device vulkan` : 100 000 parties
en 5 804 s (17,2 parties/s), 8 068 556 plans, 8,3 Go (forme compacte) ; GPU : 723 000 évaluations/s,
lots de 116 en moyenne ; fils de jeu en attente du GPU 7,6 % du temps.

Mesures GPU libre, aussitôt après :
- `bench-gpu` (réseau 256 × 6 au hasard) : 3 125 positions/s en lots de 16, 5 990 en lots de 64,
  5 540 en lots de 1 024 ; `gemm` 77 %, `pack` 10 %, `attention` 8,5 %. Les produits tournent à ~5
  TFLOPS : le kernel `gemm` est à reprendre (c130 en Vulkan atteint ~40 TFLOPS sur la même carte).
- Arène de contrôle, k = 16, m = 4 (mêmes donnes que plus haut) : `gpu=1` 9,4 s en tout (lectures
  49 s, pas 74 s, temps cumulés des fils), `gpu=enc` 17,4 s (lectures 157 s, pas 77 s). GPU libre,
  le tout-GPU l'emporte : la lenteur d'avant venait du GPU partagé avec la phase 1.

## Phase 2 : protocole (fixé le 26 octobre, avant le calcul)

- Données : `lab8/data/c130` (100 000 parties, graine 81000000) ; validation : les parties dont la
  donne est un multiple de 20 (5 %), 50 000 de leurs plans mesurés à chaque validation.
- Réseau : 256 × 6, 8 têtes, décodeur de 2 couches (7,9 millions de paramètres), neuf.
- Apprentissage : lots de 256, AdamW (taux 3e-4, montée sur 500 lots, cosinus jusqu'à 10 %,
  décroissance 0,01), 125 000 lots (~4 passes, ~3 h 30 à 2 500 plans/s), EMA des poids ; validation
  tous les 5 000 lots, chaque réseau écrit (`lab8/runs/i1/i1-NNNNNN.bin`), état de reprise.
- Critère (`lab.md`) : sur les plans mis de côté, le meilleur réseau (par la somme pol + val de la
  validation) a une politique (entropie croisée face à la recherche de c130) et une valeur finale
  (entropie croisée face au vainqueur) au moins aussi bonnes que le réseau de c130 seul, en tout et
  par nombre de joueurs.
- Ensuite (remplace la phase 4, la recherche coup par coup du nouveau réseau n'étant pas écrite) :
  parité des politiques seules, `plan:net=…` (le décodeur, le coup le plus probable) contre
  `az:net=lab7/nets/c130.bin,n=1` (le coup préféré du réseau de c130), 920 donnes, graine 82001 ;
  attendu : dans ±2 points de 0 si l'imitation est réussie.
- Puis la phase 5 : `plans:net=…` contre c130 à n = 400 (sur le processeur, pour ne pas partager le
  GPU), 920 donnes, graine 82002.

## Phase 2 : l'apprentissage (nuit du 27 octobre)

De 0 h 30 à 4 h 05 (`lab8/runs/i1`, 2 500 plans/s). Validations (plans mis de côté ; c130 seul :
politique 1,179, valeur 1,032, précision 60,6 %) :

| Lots | Politique | Valeur | Précision |
| --- | --- | --- | --- |
| 5 000 | 1,444 | 1,159 | 45,3 % |
| 10 000 | 1,328 | 1,071 | 50,8 % |
| 15 000 | 1,270 | 1,054 | 53,8 % |
| 20 000 | 1,240 | 1,045 | 55,1 % |
| 30 000 | 1,211 | 1,035 | 56,3 % |
| 40 000 | 1,195 | 1,030 | 56,9 % |
| 50 000 | 1,186 | **1,027** | 57,2 % |
| 60 000 | 1,180 | 1,029 | 57,5 % |
| 70 000 | 1,178 | 1,039 | 57,5 % |
| 85 000 | **1,176** | 1,077 | 57,5 % |
| 100 000 | 1,176 | 1,132 | 57,5 % |
| 125 000 | 1,177 | 1,234 | 57,3 % |

- **La valeur apprend par cœur** à partir du lot 50 000 (1,027, puis 1,234 au dernier lot) : 100 000
  parties ne donnent que 100 000 vainqueurs, répétés sur ~80 plans chacune et vus 4 fois. La
  politique, elle, progresse jusqu'au lot 85 000 et passe sous celle de c130 dès le lot 70 000 (son
  entropie croisée est meilleure, sa précision moins bonne : 57,5 % contre 60,6 %).
- Meilleur réseau selon la règle fixée (`best.py`, somme pol + val) : **lot 60 000**. Critère de la
  phase 2 manqué de peu : politique 1,1803 contre 1,1788 (en tout, 2, 4 et 5 j.) ; valeur meilleure
  qu'à c130 en tout, à 2, 3 et 5 j., moins bonne à 4 j. (1,215 contre 1,203). Aucun lot ne réunit
  les deux : la valeur est au mieux quand la politique n'a pas fini.
- Remèdes possibles (non faits) : plus de parties, moins de poids pour la valeur, ou la valeur
  apprise sur une partie seulement des plans de chaque partie (AlphaGo : une position par partie).

Contrôles en cours de route (le décodeur seul contre le coup préféré du réseau de c130, `plan:` contre
`az:…,n=1`, graine 9, hors protocole) :
- lot 5 000, 40 donnes : −26,4 points. Trace d'une partie (`LAB8_TRACE=1`) : déclin un tour sur deux,
  tous les jetons sur une région au redéploiement. Sur les plans mis de côté, ce réseau choisit le
  déclin deux fois plus souvent que la recherche de c130 quand il est possible (26 % contre 13 %) :
  un réseau trop jeune, pas une erreur de la chaîne (distances Python et Rust vérifiées identiques).
- Corrigé au passage : le joueur `plan:` recommençait un plan à chaque décision du jeu (chaque
  conquête), le décodeur toujours à son premier pas, ce qu'il ne voit jamais à l'apprentissage ; il
  écrit maintenant le plan d'un trait et garde la suite, comme `plans:`. Peu d'effet à ce lot.
- lot 20 000, 80 donnes : −8,6 points (2 j. 35,0 %, 3 j. 25,8 %, 4 j. 16,9 %, 5 j. 13,0 %).
- La recherche sur plans (GPU) contre c130 à n = 400 sur le processeur : la chaîne complète tourne
  (2 donnes de contrôle, 167 ms par décision pendant l'apprentissage).
- lot 50 000, 200 donnes : **+0,6 point** (2 j. 50,0 %, 3 j. 32,7 %, 4 j. 25,5 %, 5 j. 21,8 %).
- La recherche sur plans au lot 50 000 (`gpu=1,k=16,m=4`) contre c130 à n = 400, 80 donnes :
  **−2,2 ± 4,3 points** (2 j. +0,0, 3 j. −3,3, 4 j. −9,4, 5 j. +4,0), 449 ms par décision (GPU
  partagé avec l'apprentissage). Note : `labtools/score.py` donne tous les rôles par défaut ; le
  joueur étudié est `--role 0`.

## Parité des politiques seules (27 octobre, protocole)

`plan:net=lab8/runs/i1/i1-060000.bin` contre `az:net=lab7/nets/c130.bin,n=1`, 920 donnes, graine
82001 (`lab8/runs/i1/parity-060000.tsv`) : **−0,1 ± 1,5 point** (2 j. −0,7, 3 j. −0,9, 4 j. +1,0,
5 j. +0,2) ; 22,7 ms par décision (p95 62,5 ms, décodeur sur le processeur), 206 s. Dans ±2 points :
le transformer joue aussi bien que le réseau de c130, sans recherche ni l'un ni l'autre.

## Phase 5 : la recherche sur plans contre c130 (27 octobre, protocole)

920 donnes, graine 82002, c130 à n = 400 (processeur pour lab8, GPU pour lab7 ; mêmes donnes) :

| Joueur | Score | 2 j. | 3 j. | 4 j. | 5 j. | Temps par décision |
| --- | --- | --- | --- | --- | --- | --- |
| `plans:net=i1-060000,gpu=1` (k = 16, m = 4) | **+0,4 ± 1,5** | +2,4 | −3,4 | +1,2 | +1,3 | 131 ms (p95 516 ms) |
| lab7, `az:c130,n=400,roll=1,mix=0.5` | **+9,7 ± 1,5** | +6,8 | +8,8 | +13,2 | +9,9 | 5 104 ms (p95 20 106 ms) |

- Différence appariée lab8 − lab7 : **−9,3 ± 2,2 points** (2 j. −4,5, 3 j. −12,2, 4 j. −12,0, 5 j.
  −8,6). Critère de la phase 5 non atteint.
- À temps très inégal : lab7 pense 39 fois plus longtemps. La recherche sur plans à 131 ms rejoint
  c130 à n = 400 (lui-même nettement au-dessus de sa politique seule, à parité avec la nôtre) : ce
  que la recherche ajoute vaut à peu près 400 simulations de c130.
- Temps de la recherche (tous fils) : lectures 8 543 s (39,6 millions de positions), pas du décodeur
  11 210 s (129 millions), règles 164 s. Durée 4 231 s (c130 sur le processeur en compte la plus
  grande part).
- Fichiers : `lab8/runs/i1/phase5-060000.tsv`, `lab8/runs/i1/phase5-lab7.tsv`.

### Plus de calcul pour la recherche sur plans (hors protocole)

Les 200 premières donnes de la graine 82002 (50 par taille), appariées avec les arènes ci-dessus. Les
places et caches du GPU ont dû être agrandis (16 384 et 12 288 : 6 Go) ; moins de fils pour qu'ils
suffisent.

| Recherche | Contre c130 | Contre k = 16, m = 4 | Contre lab7 | Temps par décision |
| --- | --- | --- | --- | --- |
| k = 32, m = 16, 12 fils | +4,2 ± 3,6 | +2,8 ± 4,5 | −2,7 ± 4,7 | 743 ms (p95 2,8 s) |
| k = 32, m = 32, 8 fils | +4,6 ± 3,2 | (contre m = 16 : +0,5 ± 4,3) | −2,2 ± 4,2 | 1 159 ms (p95 4,4 s) |

La recherche profite du calcul : à 0,7 s par décision, l'écart avec lab7 (5,1 s) n'est plus décidé.
Doubler les mondes au-delà de 16 n'apporte plus rien de visible : la recherche plafonne à un tour
(candidats, puis le plan le plus probable de chacun) ; ce qui reste à gagner est plutôt dans la
valeur (apprise par cœur) et dans les candidats.

## i2 : la valeur sans apprentissage par cœur (protocole, fixé le 27 octobre avant le calcul)

Demandé après la phase 5 : corriger la valeur apprise par cœur, puis refaire la phase 5.

- Données : `lab8/data/c130` et 100 000 parties de plus, `lab8/data/c130b` (même commande que la
  phase 1, graine 81100000 : autres donnes) ; 200 000 parties, ~16 millions de plans. Validation :
  mêmes règles (donne multiple de 20), les 50 000 premiers plans mesurés, ceux de i1 : comparables.
- `train.py --value-every 4` : la victoire et les pièces finales apprises sur un plan sur quatre de
  chaque partie (indice multiple de 4) ; les cibles du prochain tour (valeur de la recherche de c130,
  propre à chaque position) et les propriétaires sur tous les plans. Avec deux fois plus de parties
  et ~2,4 passes, chaque vainqueur est vu ~50 fois au lieu de ~320.
- Réseau neuf 256 × 6, mêmes réglages que i1, 150 000 lots (~4 h 10), validation tous les 5 000 lots
  (`lab8/runs/i2`).
- Meilleur réseau par la même règle (`best.py`) ; puis la parité des politiques seules (graine 82001)
  et la phase 5 (graine 82002, k = 16, m = 4, appariée avec `phase5-lab7.tsv` et avec i1).

## i2 : l'apprentissage (27 octobre, 10 h 40 – 13 h 30, arrêté)

Chargement 14 s, 16 139 039 plans, 23,4 Go en mémoire (2,4 Go libres : rien d'autre de lourd en même
temps), 2 556 plans/s. Arrêté au lot 103 000 sur 150 000 à la demande de l'utilisateur, pour un
bilan ; l'état du lot 100 000 est gardé (`lab8/runs/i2/state.pt` : la même commande reprend là).

| Lots | i1 politique | i1 valeur | i2 politique | i2 valeur |
| --- | --- | --- | --- | --- |
| 50 000 | 1,186 | **1,027** | 1,195 | 1,034 |
| 60 000 | 1,180 | 1,029 | 1,187 | 1,029 |
| 70 000 | 1,178 | 1,039 | 1,180 | 1,027 |
| 80 000 | 1,176 | 1,063 | 1,174 | 1,024 |
| 90 000 | 1,176 | 1,097 | 1,170 | 1,020 |
| 100 000 | 1,176 | 1,132 | **1,166** | **1,019** |

- **La valeur n'apprend plus par cœur** : elle baisse encore au lot 100 000 ; sur les plans
  d'apprentissage 1,001 (lot 103 000), à peine sous la validation.
- **Critère de la phase 2 atteint au lot 100 000** (`best.py`), en tout et à chaque taille :
  politique 1,1664 contre 1,1788 (2 j. 1,098 / 1,107 ; 3 j. 1,214 / 1,230 ; 4 j. 1,162 / 1,174 ;
  5 j. 1,164 / 1,176), valeur 1,0193 contre 1,0318 (2 j. 0,185 / 0,191 ; 3 j. 0,932 / 0,939 ; 4 j.
  1,191 / 1,203 ; 5 j. 1,317 / 1,337). Précision 58,0 % (c130 60,6 %).
- Pas encore mesuré en jeu : ni parité des politiques, ni phase 5 avec `i2-100000.bin`.

## Bilan au 27 octobre, 13 h 30

Fait :
- **La base** (phase 0) : transformer neuf (jetons régions, joueurs, marché, partie ; attention
  biaisée par les distances), décodeur qui écrit un plan de tour coup par coup, référence Rust égale
  à PyTorch, forme compacte des parties (33 Go → 8,3 Go pour 100 000 parties).
- **Les données** (phase 1) : 200 000 parties de c130 à n = 400, 16 millions de plans, ~1 h 35 par
  100 000 sur le PC.
- **Le moteur GPU** (phase 3, en partie) : encodeur et décodeur en Vulkan, bf16, mémoire des lectures
  et caches des voies gardés sur le GPU. Le `gemm` plafonne à ~5 TFLOPS (c130 en Vulkan : ~40) : à
  reprendre.
- **L'imitation** : i1 (100 000 parties) joue à parité de c130 sans recherche (−0,1 ± 1,5 point, 920
  donnes), mais sa valeur apprend par cœur après le lot 50 000. i2 (200 000 parties, victoire finale
  sur un plan sur quatre) corrige cela et passe le critère de la phase 2 partout.
- **La recherche sur plans** (phase 5, avec i1) : un tour de profondeur, candidats tirés du
  décodeur, mondes possibles communs, issues exactes du dé, fins simulées par plans.

Mesures en jeu (contre c130 à n = 400) :

| Joueur | Donnes | Score | Temps par décision |
| --- | --- | --- | --- |
| i1, politique seule (contre c130 à n = 1) | 920 | −0,1 ± 1,5 | 23 ms |
| i1, plans k = 16, m = 4 | 920 | +0,4 ± 1,5 | 131 ms |
| i1, plans k = 32, m = 16 | 200 | +4,2 ± 3,6 | 743 ms |
| i1, plans k = 32, m = 32 | 200 | +4,6 ± 3,2 | 1 159 ms |
| lab7, `n=400,roll=1,mix=0.5` | 920 | **+9,7 ± 1,5** | 5 104 ms |

Ce qu'on en retient :
- Un transformer neuf, appris en quelques heures sur ce PC, imite c130 aussi bien que son propre
  réseau ; la cible du vainqueur final, répétée sur tous les plans d'une partie, s'apprend par cœur
  (même piège qu'AlphaGo) : une position sur quatre et deux fois plus de parties suffisent.
- La recherche sur plans vaut 400 simulations de c130 en 30 fois moins de temps ; avec plus de
  calcul elle dépasse c130 de ~4,5 points, mais plafonne vite sur les mondes. Le meilleur joueur
  reste celui de lab7 (−9,3 ± 2,2 points d'écart apparié à k = 16 ; −2,2 ± 4,2 à k = 32, m = 32,
  indécis). Phase 5 non atteinte avec i1.

Pistes, dans l'ordre :
1. Mesurer i2 (`i2-100000.bin`) : parité des politiques (graine 82001), phase 5 (graine 82002),
   appariées avec i1 et lab7 ; la valeur meilleure doit surtout servir la recherche.
2. Finir i2 jusqu'au lot 150 000 si la politique y gagne encore (reprise par `--state`).
3. La recherche : plus de candidats, modifications locales du meilleur plan (`mods`), et un second
   tour de profondeur, plutôt que plus de mondes.
4. Le `gemm` du GPU (×5 à gagner sur les lectures), puis la phase 6 (selfplay).

## Optimisation du transformer (27 octobre, après-midi)

Demandé : accélérer l'apprentissage et le jeu du nouveau réseau, en regardant les bibliothèques HIP
d'AMD. Mesures d'abord, GPU libre, réseau i2-100000.

### Ce que coûte une lecture, et où en est le moteur Vulkan

- Calcul par position lue : c130 ~46 MFLOP (48 régions × 7 couches de 128, deux produits par couche),
  le transformer ~620 MFLOP (60 jetons × 6 couches de 256, plus les clés et valeurs du décodeur) :
  **13 fois plus**, par la taille choisie.
- Moteur Vulkan, chemin de la recherche (`bench-gpu --slots true`, sorties gardées sur le GPU) : 22 800
  positions/s en lots de 64, **31 700 en lots de 256**, 22 800 en lots de 1 024 (`gemm` 72-76 %,
  `attention` 12-15 %, `layernorm` 6-7 %). Les 6 000 positions/s notées plus haut comptaient la copie
  des 172 ko de sorties de chaque position vers le processeur (`encode`), que la recherche ne fait pas.
  Soit ~20 TFLOPS utiles, contre ~34 pour c130 (745 000 positions/s × 46 MFLOP) : 1,7 fois moins
  efficace, pas 9.
- hipBLASLt (par `torch.matmul`, bf16) sur nos formes, lots de 256 positions (15 360 lignes) : 28 à 47
  TFLOPS selon le produit, ~37 en moyenne sur une couche ; notre `gemm` en fait ~27. L'encodeur
  entier en PyTorch : 12 900 positions/s (lots de 256), 29 500 compilé (`torch.compile`) ; 16 300 et
  37 900 en lots de 1 024. **Un moteur HIP gagnerait au plus ~1,3 fois** : non fait.
- Dans une arène (`plans:…,gpu=1`, k = 16, m = 4, 20 donnes contre c130 à n = 1, `LAB8_GPU_PROFILE=1`,
  le profil sépare maintenant lectures et pas du décodeur) : GPU occupé 92 % du temps ; lectures 74 %
  (25 600 positions/s ; `gemm` 24 s sur 32,5), pas du décodeur 26 % (248 000 pas/s) ; 253 lectures et
  853 pas par décision. La recherche est bornée par le calcul de l'encodeur.

### L'apprentissage : 2 550 → 7 400 plans/s

`torch.profiler` sur un lot (256 plans) : le calcul seul prenait 97,9 ms, la préparation des lots
(5,8 ms par fil, 4 fils) et l'envoi (16 ms) se recouvrent avec lui.

| Changement | Calcul d'un lot | Boucle réelle (`--bench 300`) |
| --- | --- | --- |
| Départ | 97,9 ms | 2 550 plans/s |
| Biais des relations par un produit avec le one-hot (`bmm` par position) au lieu de `nn.Embedding` : le gradient d'une table de 10 lignes lue 921 600 fois par lot prenait **62 %** du GPU | 42,5 ms | 5 600 plans/s |
| Attention efficace au lieu du chemin `math` : biais au type des activations (bf16) et dernière dimension contiguë (biais calculés transposés) | 33,8 ms | **7 400 plans/s** |

Sorties identiques au bit près en f32 (`check.py` sur `tests/fixtures`, test Rust `net`). Un
apprentissage comme i2 (150 000 lots) passe de 4 h 15 à ~1 h 30.

Essayé, abandonné : `torch.compile`. Le réseau entier, ou l'encodeur seul dans la vraie boucle, fait
une faute mémoire du GPU dans un produit bf16 avec biais de hipBLASLt (ROCm 7.14), au bout de quelques
dizaines à centaines de lots ; le décodeur seul compilé est plus lent (formes variables). Reste dans
un lot : produits ~30 %, conversions bf16 15 %, attention arrière 11 %, normalisations 9 % ; aller
plus loin demande de fusionner des kernels.

### Ce qu'il reste

- Le levier du jeu est le calcul de l'encodeur : un réseau plus petit (128 × 4 : ~6 fois moins de
  calcul par position) appris sur les mêmes 200 000 parties, s'il garde la parité ; maintenant ~1 h
  d'apprentissage.
- Le `gemm` Vulkan au niveau de hipBLASLt : ~1,2 fois sur la recherche entière.

## s1 : un transformer 128 × 4 (protocole, fixé le 27 octobre avant le calcul)

Demandé : un réseau plus petit, appris avec l'apprentissage accéléré, puis la parité des politiques
seules et sa vitesse dans la recherche.

- Réseau 128 × 4, 4 têtes (32 par tête, comme i2), décodeur de 2 couches : 1,6 million de paramètres
  (i2 : 7,9) ; ~6 fois moins de calcul par position lue. Moteur GPU vérifié à cette taille
  (`check-gpu`, réseau au hasard : écarts du même ordre que pour i2 appris, l'arrondi bf16).
- Mêmes 200 000 parties, `--value-every 4`, lots de 512 (12 400 plans/s en lots de 256, 17 000 en lots
  de 512), taux 3e-4 inchangé, 75 000 lots (38 millions de plans, 2,4 passes, ~40 min), validation
  tous les 5 000 lots (`lab8/runs/s1`) ; meilleur réseau par `best.py`.
- Parité des politiques seules : `plan:net=…` contre `az:net=lab7/nets/c130.bin,n=1`, 920 donnes,
  graine 82001, appariée avec i1 ; attendu dans ±2 points.
- Vitesse, mesurée de deux façons et comparée à i2 mesuré de même :
  - vitesse brute du GPU (`bench-gpu --slots true`) : des lots de positions lues (passées dans
    l'encodeur) aussi vite que possible, sorties gardées sur le GPU comme dans la recherche ; i2 :
    31 700 positions lues par seconde en lots de 256 ;
  - vitesse en partie : notre joueur (recherche sur plans, 16 candidats, 4 mondes) joue 20 donnes
    (graine 9 : les mêmes que pour i2) contre c130 sans recherche (`n=1`, presque instantané : le temps
    mesuré est le nôtre), avec le profil du GPU ; i2 : 25 600 positions lues par seconde, 261,5 ms de
    réflexion par décision.

## s1 : résultats (27 octobre)

Apprentissage : 75 000 lots de 512 en ~38 min (16 200 à 17 400 plans/s ; 20 Go en mémoire, une
machine virtuelle de l'utilisateur à côté : 9 à 10 Go d'échange, sans effet visible sur le débit).

| Lots | Politique | Valeur | Précision |
| --- | --- | --- | --- |
| 25 000 | 1,229 | 1,051 | 55,3 % |
| 50 000 | 1,198 | 1,035 | 56,6 % |
| 75 000 | **1,189** | **1,031** | 56,9 % |
| i2, lot 100 000 | 1,166 | 1,019 | 58,0 % |
| c130 seul | 1,179 | 1,032 | 60,6 % |

Meilleur réseau : le dernier (lot 75 000), qui progressait encore un peu. Valeur au niveau de c130,
politique un peu en dessous (critère de la phase 2 non atteint).

- **Parité des politiques seules** (920 donnes, graine 82001, `lab8/runs/s1/parity-075000.tsv`) :
  **−4,1 ± 1,5 points** (2 j. −4,6, 3 j. −9,7, 4 j. −2,6, 5 j. +0,6) ; apparié avec i1 : −4,0 ± 1,9.
  Hors des ±2 attendus : le petit réseau perd en qualité. 3,9 ms par décision sur le processeur
  (i1 : 22,7).
- **Vitesse brute du GPU** (`bench-gpu --slots true`) : 52 900 positions lues par seconde en lots de
  64, **109 200 en lots de 256**, 101 300 en lots de 1 024 ; 3,4 fois i2 (pour ~6 fois moins de
  calcul : les petits produits occupent moins bien la carte).
- **Vitesse en partie** (mêmes 20 donnes que i2) : **135,6 ms par décision** (i2 : 261,5), soit 1,9
  fois plus vite ; lectures 62 800 positions/s (i2 : 25 600), pas du décodeur 381 000/s (i2 :
  248 000). Le GPU est occupé 85 % du temps : lectures 65 %, pas du décodeur 35 % ; le décodeur
  (2 couches, inchangé) pèse maintenant autant que la moitié de l'encodeur.

À retenir : 1,6 million de paramètres coûtent ~4 points de jeu sans recherche, pour une recherche
deux fois plus rapide. Doubler le calcul de la recherche rapportait moins (k = 32, m = 16 contre
k = 16, m = 4 : 8 fois le calcul pour +3,8 points) : s1 ne vaut pas i2 à temps égal, sans doute. Une
taille intermédiaire (192 × 5, 256 × 4) ou s1 appris plus longtemps restent à essayer.

## À faire ensuite (décidé le 27 octobre)

**La phase 5 avec i2**, à lancer plus tard (pas tout de suite, à la demande de l'utilisateur) :
1. parité des politiques seules : `plan:net=lab8/runs/i2/i2-100000.bin` contre
   `az:net=lab7/nets/c130.bin,n=1`, 920 donnes, graine 82001, appariée avec
   `lab8/runs/i1/parity-060000.tsv` (~4 min) ;
2. phase 5 : `plans:net=lab8/runs/i2/i2-100000.bin,gpu=1` (k = 16, m = 4) contre
   `az:net=lab7/nets/c130.bin,n=400` sur le processeur, 920 donnes, graine 82002, appariée avec
   `lab8/runs/i1/phase5-060000.tsv` (i1) et `lab8/runs/i1/phase5-lab7.tsv` (le meilleur de lab7)
   (~1 h 10, GPU et processeur occupés : rien d'autre de lourd en même temps).
