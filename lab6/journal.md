# Labo 6 : la marge de c130 face à une recherche beaucoup plus longue

Ouvert le 25 octobre 2026 à la demande de l'utilisateur, depuis **lab5** au commit `bc6b820`
(copie sans `probe`). Plan : [lab.md](lab.md).

## Base

- Code de lab5 renommé (`smallmobile-lab6`, `smallmobile-brain6`, variables `LAB6_`), sans le
  diagnostic `probe` ; `bench-vulkan` s'arrête à 2 048 positions par lot (la limite du moteur).
- `nets/c130.bin` identique à lab3 et lab5.
- `cargo test` et `cargo clippy -D warnings` sur lab6 : passés.

## Machine et CUDA (25 octobre)

vast.ai, machine 119726 (Pays-Bas, RTX 5080 **bridée à 250 W**, pilote 595.84, Ryzen 9 9950X),
image `stock-ubuntu26.04-py312`, PyTorch 2.14.1 (CUDA 13.0). Réseau c130 seul, 40 régions,
positions par seconde, même carte :

| Lot | Vulkan (`bench-vulkan`) | CUDA f32 | CUDA bf16 | bf16 compilé | + graphes CUDA |
| --- | --- | --- | --- | --- | --- |
| 64 | 288 143 | 83 770 | 51 034 | 152 065 | 215 842 |
| 256 | 330 203 | 313 034 | 201 750 | 613 399 | 456 622 |
| 1 024 | 379 997 | 310 150 | 294 758 | **760 952** | 487 449 |
| 2 048 | 386 894 | 202 131 | 263 220 | 584 311 | 469 734 |

PyTorch compilé (`max-autotune-no-cudagraphs`) : ~2 fois le shader Vulkan dès 256 positions par
lot. Un portage CUDA du réseau pourrait diviser par deux le temps GPU sur NVIDIA. Le même shader
faisait 717 000 positions/s sur la 5080 non bridée du Nevada (machine 153202) : ici, le bridage à
250 W coûte environ la moitié ; à vérifier dans le choix des machines (`power.limit`).

## Arènes

Graines réservées : 61001 (mini-arène de coût), 62001 (16×), 63001 (64×).
- Mini-arène 61001 (8 donnes, 28 parties, 6 400 contre 400) : 116 s, 1,24 s par décision de la
  recherche longue (p95 4,3 s), GPU à 99 %. Score sans valeur à cette taille (−10,7 ± 14,4).
  Le fichier de sortie manquait (dossier absent) : `lab6 arena` crée désormais `--out` avant de
  jouer ; après l'échec, `exit` laissait le pilote NVIDIA attendre les commandes soumises d'avance.
- Arène 16× lancée : graine 62001, 400 donnes, même machine.

## Le shader Vulkan bf16 faux sur NVIDIA ; moteur CUDA

`lab6 check-gpu` (positions de l'IA Normale, 2 à 5 joueurs, GPU contre processeur), RTX 5080 :
Vulkan bf16 écart maximal victoire 0,37, pièces 10,0, logits 112 (faux) ; Vulkan f32 0,002, 0,06,
0,63 ; **CUDA** (`--device cuda`, `src/cuda.rs` : cuBLAS bf16 pour les grands produits, noyaux NVRTC,
régions à la suite) 0,0003, 0,006, 0,11, et pareil avec deux réseaux mêlés dans les calculs.
Selfplay c130, 1 000 parties : CUDA 47 s (~790 positions/s), Vulkan f32 73 s, Vulkan bf16 (faux)
33 s. Le banc PyTorch laisse de la marge au moteur CUDA (premier jet). La première arène 16×,
lancée en Vulkan bf16, a été arrêtée sans score (même graine reprise ensuite).

### Arène 16× (graine 62001, 400 donnes, CUDA, 27 min)

c130 à 6 400 simulations face à des c130 à 400 (`runs/arena16.tsv`, `labtools/score.py`) :
2 joueurs −0,8 ± 4,3 ; 3 joueurs +4,9 ± 5,7 ; 4 joueurs +0,5 ± 4,2 ; 5 joueurs +2,4 ± 3,4 ;
**score +1,8 ± 2,2 points**, indécis, comme 8× dans lab4 (+2,0 ± 2,3). 8,3 s par décision de la
recherche longue (p95 28 s). Seize fois plus de recherche n'apporte presque rien : c130 est proche du
plafond de ce que sa recherche tire de ce réseau.
