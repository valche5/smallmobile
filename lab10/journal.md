# Labo 10 : l'ancrage de la recherche de c130 sur sa politique (piKL)

Ouvert le 29 octobre 2026 à la demande de l'utilisateur, depuis **lab7** au commit `1ae5637`, après le
bilan des labos (`labtools/bilan.md`) : l'ancrage piKL était la piste inexplorée la moins chère. Plan et
protocole : [lab.md](lab.md).

## Base

- Code de lab7 renommé (`smallmobile-lab10`, `smallmobile-brain10`, programme `lab10`, variables
  `LAB10_`), sans `py/`, `loop.sh`, `rapport.html` ni les données. `nets/c130.bin` identique à lab3 et
  lab7 (SHA-256 `e534cc84…`).
- `search::Config::anchor` (`Anchor::Mctx`, `Fixed { scale }`, `Raw { lambda }`), lu par
  `Node::completed_into` et passé à `select`, `improved` et aux choix de la racine ; options
  `anchor=`, `scale=`, `lambda=` (l'une ou l'autre selon l'ancrage, sinon refus).
- `lab10 speed` affiche la KL de la politique de la racine à l'a priori du réseau et la part des
  décisions où le coup joué n'est pas le coup préféré du réseau.
- Pas d'ajout au sélecteur des Experts du jeu.

## Vérifications (29 octobre, mini, processeur seul)

- `cargo test -p smallmobile-lab10 -p smallmobile-brain10 --no-default-features` : 14 tests passés.
  - `without_anchor_the_search_is_lab7s` : sans ancrage, ou avec `anchor=mctx` écrit, avec ou sans les
    options de lab7 (`roll`, `mix`, `market`, `crn`) : mêmes coups, politique, visites, valeurs au bit
    près et même hasard consommé que la recherche de lab7 (c130, 2 à 5 joueurs, avec et sans bruit).
  - `anchors_scale_the_values_as_announced` : sur un même nœud, le facteur de mctx croît avec les
    visites, celui de `fixed` non, `raw` garde les écarts bruts divisés par λ.
  - `grouped_searches_with_rollouts_match_lone_ones` et `whole_games_with_every_option` rejoués avec
    les deux ancrages.
- `cargo clippy … -- -D warnings` : passé.
- Étalonnage de la KL sur mini : tableau de `lab.md`. Avec mctx, l'écart au réseau croît de 0,77 nat
  (n = 400) à 1,10 (n = 6 400) ; avec `fixed,scale=10`, 0,61 puis 0,39.

Aucun résultat de force à ce stade.
