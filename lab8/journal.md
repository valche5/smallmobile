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

## Phase 0

À faire : voir [lab.md](lab.md).
