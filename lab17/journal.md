# Labo 17 : l'Expert des extensions

Ouvert le 15 novembre 2026 à la demande de l'utilisateur, depuis **lab13** au commit `bc63a59`, pour
que l'Expert joue les parties avec extensions (`5195eaa`). Plan et budget de temps : [lab.md](lab.md).

## Base (15 novembre, 22 h)

- Code de lab13 renommé (`smallmobile-lab17`, `smallmobile-brain17`, programmes `lab17` et
  `lab17-expert`, variables `LAB17_`), sans les données, les résultats ni les scripts de diagnostic
  (`py/d9.py`, `d13.py`, `d14.py`). `nets/c130.bin` identique à lab3. `loop.sh` repris de lab11.

## Le cerveau élargi (15 novembre, 22 h – 22 h 10)

- **Entrées** (`brain/src/encode.rs`) : celles de lab13 ne bougent pas ; les neuves viennent après.
  Par région (77 → 108) : peuples et pouvoirs des extensions (10 et 12), Catapulte, Chaudron, Dames
  Blanches, conquête payée possible et son coût (Mercenaires), tir de Catapulte possible et son coût,
  paiement en attente, région abandonnée par les Gitanes. Partie (820 → 1 220, complété à un
  multiple de 4) : peuple et pouvoir de l'armée, les 3 extensions, Catapulte tirée, reprise des
  Intrépides faite, paiement en attente, Tour d'Ivoire choisie ; par joueur, peuples et pouvoirs des
  extensions (actif et en déclin), jetons hors du plateau, Chaudrons restants ; par carte du marché,
  peuple, pouvoir et Homoncules. Le prix d'une carte (`M_AFFORDABLE`) suit `Game::pick_price`
  (3 pièces pour une Maudite) ; ses jetons, `Combo::starting_tokens` (Homoncules, Hordes).
- **Coups** (`brain/src/moves.rs`) : trois sortes de coups par région après celles de lab13 (`Hire`,
  `Catapult`, `Cauldron` : 12 sortes, 800 clés) et la reprise des Intrépides (clé globale 17).
  Paiement en attente : payer ou finir les conquêtes. Tir de Catapulte depuis la première région du
  peuple qui atteint la cible (`Game::catapult_from`). Chaudrons posés avec les pouvoirs de fin de
  tour, s'il reste une manche. Barbares et Fées actifs ne relèvent ni ne posent de jetons ; la relève
  des Kobolds en garde deux par région (`Game::floor`) ; pas de déclin des Robustes après un abandon
  des Gitanes. Tour d'Ivoire : laissée au noyau. Hasard : le dé raté des Mercenaires qui laisse le
  paiement en attente, et les Pygmées rendus en main (`moves::chance`) ; l'empreinte du marché garde
  celle de lab13 au jeu de base.
- **Réseau** : `py/widen.py` écrit `nets/c130w.bin` depuis c130 : colonnes d'entrée neuves nulles,
  sorties des coups neufs copiées d'un coup proche (conquête payée et Catapulte ← conquête, Chaudron
  ← campement, reprise ← fin des conquêtes). Shader Vulkan, moteur CUDA, `py/` suivent (`NODE16` 112).
- `src/table.rs` : les cartes du jeu (`kernel/maps.bin`) ; `LAB17_EXPANSIONS` = `none` (défaut),
  `all`, `mix` (par partie : un quart sans extension, trois huitièmes avec les trois, le reste avec
  une ou deux) ou les bits.
- `lab17-expert` accepte les parties avec extensions.

Vérifications :

- `random_games_with_every_expansion` (neuf) : 160 parties au hasard sur les cartes du jeu, 2 à 5
  joueurs, les trois extensions (une sur cinq sans) : chaque coup de `legal` accepté par le noyau,
  `forced` d'accord avec `legal`, clés distinctes, décisions groupées (`compact`) qui mènent au même
  état ; les quatre coups neufs joués.
- `without_anchor_the_search_is_lab7s` : **lab17 avec c130w rend la recherche de lab7 avec c130 au bit
  près** au jeu de base (coups, politique, visites, valeurs).
- `cargo test -p smallmobile-lab17 -p smallmobile-brain17` : 18 tests passés ; `cargo clippy …
  --all-targets -- -D warnings` : passé.
- `lab17 check-gpu` (Vulkan, RX 9070 XT) sur un c130w aux poids neufs bruités, positions avec les trois
  extensions : écarts au processeur victoire 0,0024, pièces 0,071, logits 0,595 (bf16, comme lab13).

## E1 : c130w sans apprentissage (15 novembre, 22 h 10)

L'Expert du jeu (`n=100,roll=1,mix=0.5,light=1,place=1`, c130w) contre la Difficile, 200 donnes
duplicate, graine 1702, cartes du jeu. Écart à l'attendu en points de victoires (pondéré par les
parties, sortie de l'arène).

| Extensions | 2 j. | 3 j. | 4 j. | 5 j. | Écart | Temps par décision (arène) |
| --- | --- | --- | --- | --- | --- | --- |
| aucune | 93,0 % | 66,3 % | 63,0 % | 63,8 % | **+39,7** | 4,7 ms |
| les trois | 61,0 % | 18,0 % | 26,5 % | 26,0 % | **+0,9 ± 3,4** | 4,6 ms |

Le réseau aveugle aux extensions n'est qu'à égalité avec la Difficile, qui, elle, les joue : la
recherche à fins simulées ne suffit pas, il faut apprendre. `runs/e1-c130w-hard-{all,none}.tsv`.

## Débit (15 novembre, PC)

- Auto-jeu c130w à n = 400, extensions `mix`, 20 fils, 16 parties par fil : 2 000 parties en 39 s
  (51 parties/s, 1 520 positions/s, ~30 positions par partie).
- Essai à blanc de la boucle (`RUN=blanc`, 2 000 puis 1 000 parties par cycle, fenêtre de 50 000
  positions, duels de 40 donnes) : de bout en bout en 2 min 20. Apprentissage 6 à 9 s. Dès le premier
  pas, duel avec les trois extensions +24,3 contre c000, mais **−18,2 au jeu de base** : la fenêtre
  minuscule fait dériver le réseau ; à surveiller dans la vraie boucle.

## x1 : la boucle (lancée le 15 novembre à 22 h 17)

`RUN=x1 lab17/loop.sh 5` puis par tranches : 25 000 parties par cycle, n = 400, extensions `mix`, taux
1,5e-4, fenêtre minimale 2 millions de positions (remplie par c000 = c130w), duel à chaque pas avec
les trois extensions (graine 1704) et sans extension tous les 5 pas (graine 1705), n = 200 des deux
côtés.

Remplissage : c001 par c000 (25 000 parties en 479 s, ~740 000 positions). Chaque pas : parties
~480 s, apprentissage 80 s, duel 47 s (arrêté après 184 donnes), soit ~10,5 min.

| Pas | c002 | c003 | c004 | c005 |
| --- | --- | --- | --- | --- |
| Duel contre c000, trois extensions (n = 200) | +42,6 | +48,3 | +49,3 | +50,8 |
| Duel contre c000, jeu de base (n = 200) | −6,4 | | | −7,1 |
| Recherche du jeu contre la Difficile, trois extensions (E2, graine 1702, 200 donnes) | | +13,9 ± 3,7 | | +29,1 ± 3,7 |

Le duel avec extensions contre c000 sature dès le deuxième pas : la mesure qui compte est E2, la
recherche du jeu (`n=100,roll=1,mix=0.5,light=1,place=1`) contre la Difficile, mêmes donnes qu'E1
(c130w : +0,9 ; au jeu de base, +39,7). Le jeu de base recule de ~7 points face à c130 : **décision**,
l'Expert du Labo 13 (c130) garde les parties sans extension, celui du Labo 17 joue celles avec
extensions (`expertFor`, ci-dessous). Boucle relancée jusqu'au pas 36 (23 h 09).

E2 par taille, c005 : 95,0 %, 55,3 %, 55,0 %, 46,2 % de victoires à 2, 3, 4, 5 joueurs (attendu 50,
33, 25, 20 %).

Pas 6 à 36 (23 h 09 – 4 h 51), en trois tranches (une tâche de fond dure au plus 2 h : la première,
lancée pour 7 h, a été coupée à 1 h 09 juste après l'apprentissage de c016 ; la reprise a refait son
duel). ~10,7 min par pas ; fenêtre de 4,8 millions de positions au pas 16. Aucune erreur.

| Pas | c010 | c015 | c020 | c025 | c030 | c035 | c036 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Duel contre c000, trois extensions | +49,8 | +48,4 | +47,7 | +48,3 | +49,5 | +52,8 | +47,9 |
| Duel contre c000, jeu de base | −7,3 | −1,2 | −4,2 | −1,6 | −1,9 | −3,7 | |
| E2 : contre la Difficile, trois extensions | +34,9 ± 3,6 | +35,1 ± 3,6 | +34,1 ± 3,6 | | | | |

E2 plafonne vers +35 dès c010 (à 2 joueurs, 95 à 99 % de victoires) ; les duels directs, eux,
montrent que l'apprentissage continue, lentement (trois extensions, n = 200, 460 donnes) :

- c020 contre c010 (graine 1706) : **+3,8 ± 2,3** ;
- c036 contre c020 (graine 1709) : **+2,5 ± 2,2**.

## Mesures finales de c036 (16 novembre, 4 h 53 – 5 h 10)

Recherche du jeu (`n=100,roll=1,mix=0.5,light=1,place=1`), graines neuves.

| Arène | Donnes | 2 j. | 3 j. | 4 j. | 5 j. | Écart à l'attendu |
| --- | --- | --- | --- | --- | --- | --- |
| c036 contre la Difficile, trois extensions (1707) | 460 | 96,5 % | 65,5 % | 57,4 % | 54,7 % | **+35,2 ± 2,3** |
| c130w contre la Difficile, mêmes donnes | 460 | | | | | +4,2 ± 2,3 |
| c036 contre c130w, trois extensions (1708) | 200 | 80,5 % | 84,3 % | 66,0 % | 56,0 % | **+39,9 ± 3,4** |
| c036 contre c130w, jeu de base (1708) | 200 | 49,0 % | 33,0 % | 22,5 % | 15,6 % | −2,5 ± 3,2 |

Pour situer : au jeu de base, c130 fait +39,7 contre la Difficile (E1). **Avec les extensions, c036
atteint le niveau qu'a c130 au jeu de base.** Au jeu de base il reste un peu sous c130 (−2,5 ± 3,2,
indécis, mais le recul se voit à chaque mesure).

Temps par décision sur le processeur (`lab17 expert-time`, PC, 6 fils, parties entières où tous les
sièges sont l'Expert) : trois extensions, 4 j. 117 ms en moyenne (p95 248, max 313), 5 j. 170 ms (p95
398, max 589) ; jeu de base 136 et 193 ms. Comme l'Expert du jeu de lab13 (136 et 212 ms, D16).

## Dans le jeu

- `nets/x1-c036.bin` (empreinte `909bf6bbc89f244a`, copie de `runs/x1/c036.bin`) ; `nets/c130w.bin`
  (`aa42fca594ebf594`) reste pour les tests.
- `src/shared/experts.ts` : entrée `lab17` (`x1-c036`, réglages de lab13, `expansions: true`).
  `expertFor` : sans extension, le labo de la salle (par défaut `DEFAULT_EXPERT_LAB`, désormais le
  dernier labo du jeu de base seul, lab13/c130) ; avec des extensions, le labo de la salle s'il les
  connaît, sinon le dernier qui les connaît (lab17). `levelFor` ne retombe plus sur Difficile tant
  qu'un labo connaît les extensions. Serveur (`rooms.ts`) et parties pour LLM (`llm.ts`) passent par
  `expertFor` ; le choix du niveau dit quel Expert joue avec des extensions (`Expansions.tsx`).
- Tests : `tests/rooms.test.ts`, `online.test.ts`, `llm.test.ts` suivent la nouvelle règle ;
  `tests/expert.test.ts` joue une partie entière à 4 joueurs avec les trois extensions par le vrai
  programme `lab17-expert`. `bun test` : 156 passés ; `bun run typecheck` : passé.
- Pas encore déployé sur mini.

## Pistes

- Le jeu de base : quelques pas de plus avec une plus grande part de parties sans extension, ou un
  taux plus bas, pour qu'un seul réseau remplace c130 partout.
- E2 sature : mesurer les pas suivants en duel direct, ou contre c036.
- La Tour d'Ivoire des Prêtresses est laissée au noyau ; la Catapulte tire toujours de la première
  région possible.

## Pas 37 à 46 (16 novembre, 5 h 12 – 7 h 02)

PC libre : la boucle a continué. Duels contre c000 avec les trois extensions +49,8 à +53,0 ; au jeu de
base −1,8 (c040) et −2,5 (c045). **c046 contre c036** (trois extensions, n = 200, 460 donnes, graine
1710) : +1,0 ± 2,2, indécis. c036 reste l'Expert du jeu.
