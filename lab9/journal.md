# Labo 9 : le réseau hybride de ResTNet, dans le cadre de lab8

Ouvert le 27 octobre 2026 à la demande de l'utilisateur, après lab8 (figé), pour remplacer
l'encodeur tout-transformer par des blocs de graphe et des blocs transformer alternés (ResTNet). Plan
et raisons : [lab.md](lab.md).

## Phase 0 : la base (27 octobre)

Code : copie de lab8 (figé) renommée (`smallmobile-lab9`, `smallmobile-brain9`, programme `lab9`,
`LAB9_`), avec le bloc R :
- `py/model.py` : `Conv` (le bloc R), `adjacency` et `neighbours` (moyenne des voisines, en f32) ; la
  disposition (`Net(d, "RRTRRT", têtes, décodeur)`, `--size 256,RRTRRT,8,2`) ; le nombre de têtes
  écrit dans les poids (`heads`), lu dans les biais des relations pour un réseau de lab8 ;
- `brain/src/net.rs` : les blocs `Block::T` et `Block::R`, la disposition lue dans les poids ;
- `src/vulkan.rs` : refuse encore les blocs R (phase 3).

Vérifications :
- Rust et PyTorch donnent les mêmes sorties sur un hybride au hasard (`tests/fixtures-hybrid`, 32 de
  large, RTR) : écart maximal des têtes 1,9e-6, des logits 3,0e-6 ;
- un réseau de lab8 (`tests/fixtures`, celui de lab8) se lit dans lab9 à l'identique : sorties
  PyTorch égales au bit près, Rust à 3,5e-6 ; i2 et s1 se chargent (`lab9 speed`) ;
- `cargo test -p smallmobile-lab9 -p smallmobile-brain9 --release` : 9 tests passés ; clippy passé ;
  compilation sans Vulkan passée.

Débit de l'apprentissage (`train.py --bench 300`, `lab8/data/c130`) :

| Réseau | Paramètres | Lots | Plans/s |
| --- | --- | --- | --- |
| h1, RRTRRT 256 × 6, 8 têtes | 6 040 026 | 256 | **9 450** |
| forme d'i2, TTTTTT 256 × 6 (même série) | 7 882 522 | 256 | 7 870 |
| h2, RRTRRT 128 × 6, 4 têtes | 1 545 482 | 512 | **16 500** |
| s1, TTTT 128 × 4 (lab8) | 1 611 354 | 512 | ~17 000 |

## Phase 2 : protocole (fixé le 27 octobre, avant le calcul)

Données : `lab8/data/c130` et `lab8/data/c130b`, lues en place ; validation : les 50 000 premiers
plans des parties mises de côté, ceux d'i2 et s1. Graine d'apprentissage 0, comme eux (même ordre des
plans).
- **h1** : `train.py --data ../../lab8/data/c130,../../lab8/data/c130b --size 256,RRTRRT,8,2 --steps
  150000 --val-every 5000 --value-every 4` (`lab9/runs/h1`), la recette d'i2 entière (i2 avait été
  arrêté au lot 103 000 sur 150 000 : comparaison aux mêmes lots, jusqu'à 100 000).
- **h2** : de même, `--size 128,RRTRRT,4,2 --batch 512 --steps 75000` (`lab9/runs/h2`), la recette de
  s1.
- Puis `best.py`, et la parité des politiques seules des deux meilleurs (920 donnes, graine 82001),
  appariée avec i2 (`lab8/runs/i2/parity-100000.tsv`) et s1 (`lab8/runs/s1/parity-075000.tsv`).

## Phase 3 : le bloc R en Vulkan (27 octobre, pendant l'apprentissage de h1)

`shaders/neighbours.comp` : pour chaque jeton, l'entrée d'un produit du bloc R, `[a, moyenne des
voisines, a de la partie]` ou `[a, moyenne des voisines]` en bf16 (voisines lues dans les distances,
en bits dans la mémoire partagée) ; le bloc : `layernorm`, `neighbours`, `gemm` (gelu), `layernorm`,
`neighbours`, `gemm` (résiduel). `check-gpu`, réseaux hybrides au hasard (sorties au hasard, σ 0,05),
200 positions, GPU partagé avec l'apprentissage :

| Réseau | Têtes | Régions | `dec_reg` | Attention croisée | Logits | Victoires (décodeur sur GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| RRTRRT 256 × 6 | 0,011 | 0,037 | 0,031 | 0,033 | 0,024 | 0,0027 |
| RRTRRT 128 × 6 | 0,010 | 0,039 | 0,028 | 0,030 | 0,018 | 0,0017 |
| lab8, TTTTTT 256 × 6 (même mesure) | 0,008 | 0,031 | — | — | 0,023 | — |

Mêmes écarts que le moteur de lab8 (l'arrondi bf16). Vitesse : à mesurer GPU libre.
