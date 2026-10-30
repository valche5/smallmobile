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

## D8 : la menace rend-elle les cibles apprenables ? (30 octobre)

**Contrôle de c130x.** Les 1 000 parties de test de D7 refaites dans lab14 avec c130x (graine
171 000 001, 432 s) : **999 identiques** coup pour coup à celles de lab13, la dernière diverge au coup
433 (arrondi du GPU) ; même nombre de coups. Les fichiers diffèrent seulement par la répartition des
parties entre fils. Les 20 000 parties d'apprentissage refaites de même (7 877 s) : le témoin est D7.

Les entrées sont remplies (5 000 positions de `anc-test`, 186 647 régions) : un adversaire peut prendre
la région (`X_TAKERS` > 0) dans 33 % des régions ; l'adversaire suivant peut viser 17 % des régions, à
un coût moyen de 3,3.

| Taux maximal | Apprentissage (politique, fin) D7 → D8 | `anc-test` : politique (meilleure / fin) D7 → D8 | `anc-test` : coup de la cible (fin) D7 → D8 |
| --- | --- | --- | --- |
| départ | | 1,271 → 1,271 | 56,4 % → 56,4 % |
| 1,5e-4 | 1,352 → 1,349 | 1,271 / 1,286 → 1,271 / 1,286 | 53,7 % → 53,8 % |
| 5e-5 | → 1,406 | 1,271 / 1,272 → 1,271 / 1,272 | 55,1 % → 55,2 % |
| 1,5e-5 | → 1,439 | 1,270 / 1,270 → 1,270 / 1,270 | 55,8 % → 55,8 % |

**Le critère n'est pas atteint** (1,270, seuil 1,250) : pas d'arène. Les entrées de la menace ne
changent rien, ni sur des parties neuves ni même sur les parties apprises (1,349 contre 1,352). Ce que
le professeur sait de plus que c130 n'est pas la carte des prises possibles au tour suivant : c130 la
lisait déjà, ou ce n'est pas elle qui fait ses choix. **D8 clôt la piste des entrées** (`lab14/lab.md`).
