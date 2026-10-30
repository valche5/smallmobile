# Labo 11 : une boucle depuis c130 avec un professeur nettement plus fort

Ouvert le 29 octobre 2026 à la demande de l'utilisateur, depuis **lab10** au commit `443e9bc`, plus la
boucle de lab7 (`loop.sh`, `py/`). Plan et protocole : [lab.md](lab.md).

## Base

- Code de lab10 renommé (`smallmobile-lab11`, `smallmobile-brain11`, programme `lab11`, variables
  `LAB11_`), sans les données ni les résultats. `nets/c130.bin` identique à lab3, lab7 et lab10.
- `py/` et `loop.sh` de lab7, renommés : boucle sur le PC (`DEVICE=vulkan`, PyTorch ROCm de
  `lab3/py/.venv`, 16 parties par fil), graines 110 000 000 + cycle, mesure finale sur la graine
  112101 ; fenêtre minimale de 2 millions de positions (4 dans lab7).

## Vérifications (29 octobre, PC)

- `cargo test -p smallmobile-lab11 -p smallmobile-brain11 --no-default-features` : 14 tests passés ;
  `cargo clippy … -- -D warnings` : passé. Modules Python importés par PyTorch 2.14.1+rocm7.14.
- Essai à blanc (`RUN=blanc LAB11_WINDOW_MIN=50000 GAMES=200 FILL_GAMES=1000 DEALS=40 FINAL=0`, recette du
  professeur, 4 cycles), pendant que 5c de lab10 occupait le PC : remplissage par c130, 1 000 parties en
  44 s (35 positions par partie) ; pas d'apprentissage en 10 à 19 s (ROCm) ; duels de 40 donnes en
  29 s ; la fenêtre glisse (c002…c004 au dernier pas). **Professeur : 200 parties en 395 à 444 s,
  ~0,5 partie/s**, cinq fois sous l'estimation de `lab.md` (PC partagé, et 200 parties ne remplissent
  pas les 320 places de la boucle). Mesure propre avant le lancement.

## Lancement de p1 (29–30 octobre, PC ; `lab11/runs/start-p1.sh`)

- **Professeur** : 5c − 5d de lab10 +1,0 ± 2,1, indécis : selon la règle, `n=400,roll=1,mix=0.5`, sans
  ancrage (+8,9 ± 1,2 contre c130 à n = 400, toutes arènes de lab7 et lab10 réunies).
- **Débit**, PC libre : 1 000 parties en 386 s, **2,59 parties/s** (l'essai à blanc, PC partagé et 200
  parties, sous-estimait) : cycles de **12 500 parties** (~80 min).
- Remplissage par c130 : c001 et c002, 25 000 parties chacun en 387 s (65 parties/s), 1,73 million de
  positions. Premier pas (c002, sur la fenêtre de c130 seule) : 91 s d'apprentissage ; duel −1,3 ± 2,8
  (arrêt après 276 donnes : 0 plutôt que 3 points). Le professeur joue à partir de c003.

## Pas 3 à 10 (30 octobre, 00 h 20 – 10 h 30)

~75 min par pas : 12 500 parties du professeur en 4 380 à 4 460 s (~430 000 positions), apprentissage
~50 s, duel ~40 s. Fenêtre entièrement du professeur depuis c006 (c004…c010 : 3,0 millions de
positions au pas 10).

| Pas | c002 | c003 | c004 | c005 | c006 | c007 | c008 | c009 | c010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Duel contre c130 (n = 200) | −1,3 | −2,5 | −1,2 | −1,8 | −2,0 | −0,8 | −1,9 | −2,7 | −2,3 |
| Validation : politique | 1,251 | 1,283 | 1,293 | 1,308 | 1,325 | 1,333 | 1,331 | 1,331 | 1,324 |
| Validation : précision | 50,0 % | 49,0 % | 48,1 % | 46,9 % | 46,6 % | 46,1 % | 46,0 % | 45,9 % | 45,8 % |

La perte de politique monte avec la part du professeur dans la fenêtre : ses cibles s'écartent plus de
l'a priori que celles de c130 à n = 400. La valeur reste stable (1,04 à 1,07).

**Point d'étape au pas 10** (c010 contre c130, n = 400 des deux côtés, graine 111001, 400 donnes,
4,5 min) : **−0,7 ± 2,2** (2 j. −0,2, 3 j. −2,5, 4 j. +0,8, 5 j. −0,7). Rien de gagné ni de perdu ; au-dessus
du seuil d'arrêt (−3, au pas 20). `lab11/runs/p1/step-c010.tsv`.
