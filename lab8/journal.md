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

Reste, pour les phases suivantes : la recherche sur plans et la recherche coup par coup du nouveau
réseau (phases 4 et 5), le moteur GPU du transformer (phase 3).
