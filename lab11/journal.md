# Labo 11 : une boucle depuis c130 avec un professeur nettement plus fort

Ouvert le 29 octobre 2026 à la demande de l'utilisateur, depuis **lab10** au commit `443e9bc`, plus la
boucle de lab7 (`loop.sh`, `py/`). Plan et protocole : [lab.md](lab.md).

## Base

- Code de lab10 renommé (`smallmobile-lab11`, `smallmobile-brain11`, programme `lab11`, variables
  `LAB11_`), sans les données ni les résultats. `nets/c130.bin` identique à lab3, lab7 et lab10.
- `py/` et `loop.sh` de lab7, renommés : boucle sur le PC (`DEVICE=vulkan`, PyTorch ROCm de
  `lab3/py/.venv`, 16 parties par fil), graines 110 000 000 + cycle, mesure finale sur la graine
  112101 ; fenêtre minimale de 2 millions de positions (4 dans lab7).
