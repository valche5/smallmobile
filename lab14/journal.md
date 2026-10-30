# Labo 14 : la menace des adversaires en entrée du réseau

Ouvert le 30 octobre 2026 à la demande de l'utilisateur, depuis **lab13** au commit `0dcf777`, pour
la piste des entrées calculées à la main laissée par lab13 (D7). Plan : [lab.md](lab.md).

## Base

- Code de lab13 renommé (`smallmobile-lab14`, `smallmobile-brain14`, programme `lab14`, variables
  `LAB14_`). 16 entrées de la menace par région (`encode::threat`, `NODE` 93) ; shaders, moteur CUDA
  et `py/` suivent (`NODE16` 96). `py/widen.py` : c130 → `nets/c130x.bin` (poids nuls pour la menace),
  seul réseau de lab14.
- `threat_inputs_match_the_conquest_targets` : au début d'une conquête, la menace du joueur qui
  conquiert est ce que propose le noyau (`Game::targets`), régions et coûts.
- `without_anchor_the_search_is_lab7s` : lab14 avec c130x rend la recherche de lab7 avec c130 au bit
  près (processeur). `lab14 check-gpu` (Vulkan) : mêmes écarts au processeur que c130 dans lab13
  (victoire 0,0021, pièces 0,069, logits 0,575).
- `cargo test` (lab14, brain14) : passé.
