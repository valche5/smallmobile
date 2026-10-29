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

## Machine (29 octobre)

vast.ai, offre 49691112 (Pays-Bas : RTX 5080, Ryzen 9 9950X), 0,34 $/h, comme lab10. Deux essais de
`up` perdus : `cargo` absent du PATH en ssh non interactif (rien loué) ; puis CUDA juste (victoire
0,0004) refusé par le test de `vast.sh`, qui comparait des textes selon la locale (machine louée
3 min, 0,02 $ ; corrigé, `91d60d2`). Premier lancement de l'arène sans `lab12/runs` sur la machine :
arrêt en 0 s, relancée.

## Présélection (graine 71002, 400 donnes, CUDA, 29–30 octobre)

`az:net=lab12/nets/c130.bin,n=100,roll=1,mix=0.5,reply=4` contre c130 à n = 400 : **+5,1 ± 2,5** (2 j.
+1,2, 3 j. +7,3, 4 j. +3,5, 5 j. +8,5), 14,2 s par décision sur l'arène (6,7 s pour le témoin dans lab7),
43 min. `lab12/runs/r4.tsv`.

Face au témoin sur les mêmes donnes (`lab7/runs/a1-n-100_roll-1_mix-0p5.tsv`, +5,7) : −0,5 ± 3,1 ; face
au témoin réuni (+5,5 sur 1 320 donnes) : **−0,4 ± 2,8**. Seuil de +3 loin : pas de confirmation.

**La riposte du rival n'apporte rien**, pour deux fois le calcul. Modéliser un adversaire qui cherche
à nuire au joueur, dans les fins simulées, ne change pas ce qu'elles mesurent : le jeu de c130 à n = 400
qu'affronte le joueur ne contient pas de telles ripostes, et le coup préféré du réseau en est déjà un
bon modèle. Machine détruite après 47 min, 0,28 $ ; crédit restant 4,89 $.
