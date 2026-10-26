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

Pas encore : le dé de renfort à ses deux issues pondérées (pour l'instant tiré dans chaque monde),
les modifications locales de candidats, le halving.

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

Reste, pour les phases suivantes : la recherche sur plans et la recherche coup par coup du nouveau
réseau (phases 4 et 5), le moteur GPU du transformer (phase 3).
