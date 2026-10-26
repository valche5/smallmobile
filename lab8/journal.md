# Labo 8 : le plan de tour en un passage du réseau

Ouvert le 26 octobre 2026 à la demande de l'utilisateur, depuis **lab7** au commit `adb2c72`, pour
rendre les fins simulées moins chères : un réseau qui sort tout un plan de tour (régions à prendre,
répartition des jetons) en un passage. Lab7 est figé. Plan et raisons : [lab.md](lab.md).

## Base

- Code de lab7 renommé (`smallmobile-lab8`, `smallmobile-brain8`, programmes `lab8` et
  `lab8-expert`, variables `LAB8_`), sans `runs/`, `journal.md`, `lab.md` ni `rapport.html`.
  `nets/c130.bin` et `nets/gen0.bin` identiques à lab7.
- Témoin des tests inchangé : sans option, la recherche est toujours celle de lab6 au bit près
  (`without_options_the_search_is_lab6s`), donc aussi celle de lab7.
- Lab8 n'est pas un Expert du jeu (absent de `src/shared/experts.ts`).

## Vérifications

- `cargo test -p smallmobile-lab8 -p smallmobile-brain8 --no-default-features` : 13 tests passés
  (6 dans `brain`, 7 dans `tests/options.rs`), comme lab7.
- `cargo clippy -p smallmobile-lab8 -p smallmobile-brain8 --all-targets --no-default-features
  -- -D warnings` : passé.
