# Labo 6 : la marge de c130 face à une recherche beaucoup plus longue

Ouvert le 25 octobre 2026 à la demande de l'utilisateur, depuis **lab5** au commit `bc6b820`
(copie sans `probe`). Plan : [lab.md](lab.md).

## Base

- Code de lab5 renommé (`smallmobile-lab6`, `smallmobile-brain6`, variables `LAB6_`), sans le
  diagnostic `probe` ; `bench-vulkan` s'arrête à 2 048 positions par lot (la limite du moteur).
- `nets/c130.bin` identique à lab3 et lab5.
- `cargo test` et `cargo clippy -D warnings` sur lab6 : passés.
