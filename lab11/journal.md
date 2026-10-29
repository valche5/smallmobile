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
