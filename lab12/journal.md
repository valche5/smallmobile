# Labo 12 : la riposte du rival dans les fins simulées

Ouvert le 29 octobre 2026 à la demande de l'utilisateur, depuis **lab10** au commit `e61f71d`. Plan et
protocole : [lab.md](lab.md).

## Base

- Code de lab10 renommé (`smallmobile-lab12`, `smallmobile-brain12`, programme `lab12`, variables
  `LAB12_`), sans les données ni les résultats. `nets/c130.bin` identique à lab3 et lab10.
- `search::Config::reply`, option `reply=K` : `Az::rival_turn` joue K fois le tour du rival dans une
  fin simulée (`Az::rollout`), le rival choisi à la position d'où elle part (`Az::rival`).

## Vérifications (29 octobre, PC)

- `cargo test -p smallmobile-lab12 -p smallmobile-brain12 --no-default-features` : 15 tests passés, dont
  ceux de lab10 (sans option, la recherche de lab7 au bit près) et
  `the_rival_turn_is_the_worst_for_the_root_player` : le tour gardé finit au tour suivant, se rejoue à
  l'identique, ne laisse jamais au joueur de la racine plus que son tour préféré, et le fait baisser
  dans une partie des positions. `cargo clippy … -- -D warnings` : passé.
- Coût (`speed`, un cœur, PC partagé) : `reply=4` 1,9 fois `n=100,roll=1,mix=0.5` (2,28 s contre 1,19 s
  par recherche ; 58,7 évaluations par fin simulée contre 29,9).
