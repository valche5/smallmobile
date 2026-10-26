# Lab8 : la recherche sur des plans de tour, avec un transformer neuf

Pas de code au départ : la phase 0 construit le labo, sur la base de lab7 (figé). Règles communes :
[labtools/README.md](../labtools/README.md). Lectures : [labtools/biblio.md](../labtools/biblio.md).
Résultats : [journal.md](journal.md).

## Pourquoi

- Les fins simulées de lab7 battent c130 de +5,4 ± 1,5 points (n = 100) et +8,9 ± 2,3 (n = 400),
  quand 16 fois plus de recherche ordinaire ne donnait que +1,8 ± 2,2 (lab6). Un tour fait 25 à 30
  décisions : la recherche coup par coup n'atteint presque jamais la réponse des adversaires (moins
  de 4 % des simulations à 4 et 5 joueurs).
- Le réseau n'apprend pas ce que calculent les fins simulées (lab7, T3 : rien en dix pas).
- Diplomacy, le jeu le plus proche (carte irrégulière, jusqu'à 7 joueurs, beaucoup d'ordres par
  tour) : les meilleurs agents cherchent sur des **tours entiers** tirés de la politique, gardent
  des fins simulées courtes (2 à 3 tours), écrivent les ordres avec un **décodeur**, et le
  transformer y a battu le réseau de graphe (DORA : 898 → 1 123 Elo).
- Le réseau de c130 (graphe à moyenne des voisines, 128 × 7, 1,1 million de paramètres) n'a jamais
  été remis en cause depuis lab2.

## Décisions de l'utilisateur (26 octobre)

- La recherche porte sur des **plans de tour**, écrits par un **décodeur**.
- Un réseau **neuf**, un transformer, si son entraînement tient sur cette machine (pas quinze
  jours). Il commence par **imiter c130**, puis passe au selfplay.
- Les **fins simulées** sont gardées au départ, retirées ensuite quand la valeur suffit.
- Le temps de réflexion peut être long : on vise un joueur fort, pas rapide.
- Le progrès se mesure face à **c130** à n = 400.
- La **cible de valeur au tour suivant** (DORA), mesurée (voir sa réserve plus bas).
- Tout le calcul se fait sur ce PC : plus d'AWS ni de vast.ai.

## Le plan de tour

Un plan, c'est **tout ce que décide un joueur entre deux hasards**. Un tour ordinaire en a deux :

1. choix du peuple ou déclin, abandons, conquêtes dans l'ordre (dragon, sorcellerie, héros…),
   puis la tentative au dé de renfort ou non ;
2. **hasard** : le dé de renfort, réussi ou raté, aux probabilités exactes (il n'a lieu qu'à la
   dernière conquête : le noyau passe au redéploiement dès qu'il est lancé) ;
3. redéploiement (jetons, héros, forteresse, campements), paix des Diplomates, déclin des Robustes.

Cas particuliers :
- les **Berserks** lancent un dé avant chaque conquête : leur tour se coupe à chaque dé ;
- la **défense** se joue après le tour de l'attaquant (`Game::finish_turn`, `kernel/src/game.rs`) :
  c'est un plan à part, celui du défenseur ;
- la **carte qui arrive au marché** après un choix de peuple ne touche pas le plan du joueur, seulement
  les suivants : elle est tirée comme les autres hasards.

Le moteur joue chaque coup : il donne les coups légaux, l'état qui suit et les probabilités exactes du
dé. Le réseau choisit seulement.

## Le réseau

**Encodeur** (un transformer), une passe par plan :
- des **jetons** : un par région (48 au plus), un par joueur (5), un par carte du marché (6), un pour
  la partie. Les entrées sont celles de `lab7/brain/src/encode.rs`, regroupées par jeton (région :
  77 ; joueur : 81, plus son histoire ; marché : 38 ; partie : le reste) ;
- la **carte** passe par l'attention : un biais appris selon la distance entre deux régions (0, 1,
  2, 3, 4 voisinages ou plus, calculée une fois par carte), et un biais région ↔ joueur selon la
  relation (son peuple actif, en déclin, aucun) ;
- taille de départ : largeur 256, 6 couches, 8 têtes (≈ 5 millions de paramètres), à ajuster en
  phase 2 sur le budget d'entraînement.

**Décodeur** : il écrit le plan coup par coup.
- Le vocabulaire est celui des coups de lab7 : 32 coups globaux, et 9 sortes de coups par région
  (conquérir, renforcer au dé, dragon, sorcellerie, abandonner, héros, forteresse, campement, jeton).
  Un coup sur une région se choisit en **pointant** le jeton de la région : produit de la requête du
  décodeur avec la description de la région, une tête par sorte.
- À chaque pas, il reçoit le coup précédent et ce que le moteur a mis à jour (jetons en main,
  conquêtes faites, dé lancé…). Il regarde les coups déjà écrits (attention causale) et les jetons de
  l'encodeur (attention croisée). Les coups illégaux sont masqués par le moteur.
- 2 couches, avec cache des clés et valeurs : un pas coûte une petite fraction de l'encodeur.

**Têtes sur l'encodeur** :
- la valeur finale : 5 probabilités de victoire et 5 gains de pièces, comme lab7 ;
- la **valeur au prochain tour** du joueur qui agit (même forme) ;
- à l'entraînement seulement, les propriétaires de chaque région au prochain tour du joueur (lab7).

## La cible de valeur au tour suivant

La valeur au prochain tour apprend la valeur estimée au début du prochain tour du même joueur : celle
de la recherche de c130 pendant l'imitation, puis celle d'une copie retardée du réseau en selfplay
(DORA), plutôt que le vainqueur final, qui accumule le hasard de toute la fin de partie.

**Réserve** : lab4 a déjà essayé une cible proche (`--td 0.5` : moitié résultat, moitié valeur de la
recherche à la position suivante du joueur), depuis c130 : rien de mesurable en 3 pas, ni en 20 pas
dans la boucle w2 (`lab4/journal.md`, « Cibles de valeur à court terme », « Essai w2 »). Le cadre
diffère ici : un réseau neuf, et une recherche qui s'arrête justement au tour suivant. Les deux têtes
sont donc apprises ensemble, et une option de la recherche choisit celle qu'elle utilise : mesure hors
ligne (laquelle prédit le mieux le résultat, et la valeur après fin simulée), puis arène.

## La recherche sur des plans

**Version 1**, à un tour, comme SearchBot :
1. **Candidats** : K plans pour le joueur qui agit (K = 32 au départ), tirés du décodeur sans
   doublon ; plus des modifications locales des meilleurs (DORA : reprendre un plan à partir d'un
   pas et réécrire la suite). Les doublons se repèrent à l'état qu'ils produisent.
2. Le moteur joue chaque candidat. Au dé : les deux issues, pondérées exactement, avec un
   redéploiement écrit par le décodeur pour chacune.
3. **Fin simulée par plans** : les adversaires jouent jusqu'au prochain tour du joueur, avec des plans
   tirés du décodeur. On fait M tirages (M = 4 à 16), **les mêmes pour tous les candidats** (mêmes
   plans adverses, mêmes dés, même pioche), pour que la comparaison soit juste.
4. Valeur au prochain tour du joueur (tête au choix), mêlée ou non à la valeur directe (`mix`).
5. Choix : halving séquentiel de Gumbel sur les candidats (comme la racine de lab7), budget compté
   en passes de l'encodeur.

**Version 2**, quand la valeur suffit : un arbre de plans (les adversaires aussi choisissent parmi
des candidats), sans fins simulées.

Coût d'une fin simulée par plans à 5 joueurs : une passe de l'encodeur par plan, environ 10 jusqu'au
prochain tour (2 par joueur, plus les défenses), plus les pas du décodeur. Une fin simulée de lab7 en
demande environ 29.

## Données et apprentissage

- **Imitation** : parties de c130 sans bruit, n = 400 à chaque décision, comme le champ des arènes.
  Un enregistrement par plan : l'entrée de l'encodeur au début du plan ; ses coups, avec la
  politique de la recherche de c130 à chaque pas ; le résultat final ; la valeur de c130 au prochain
  tour du joueur ; les propriétaires. Pertes : entropie croisée du décodeur sur la politique de c130
  à chaque pas (en suivant les coups joués), les deux valeurs, les propriétaires.
- **Selfplay** ensuite, avec la recherche sur plans. La cible de politique est la répartition de la
  recherche sur ses candidats, corrigée pour des candidats tirés (Sampled MuZero, biblio). Les cibles
  de valeur sont les deux têtes.

## La machine (mesuré le 26 octobre)

i7-12700KF (20 fils), 31 Go, Radeon RX 9070 XT (RADV, Mesa 26.2.4), Ubuntu 26.10, rustc 1.99.1,
glslc. PyTorch 2.14.1+rocm7.14 voit la carte (`lab3/py/.venv`).

- c130 sur Vulkan bf16 (`lab7 check-gpu`) : écart maximal victoire 0,0021, pièces 0,069, logits
  0,575 (en CUDA, lab7 : 0,0004, 0,012, 0,11 ; l'écart des logits est à surveiller).
- `lab7 bench-gpu` : 1,18 million de positions par seconde en lots de 2 048 (RTX 4090 en CUDA, lab7 :
  0,66 million).
- Arène c130 contre c130, n = 400, 80 donnes de 2 à 5 joueurs (280 parties : chaque donne est
  rejouée à toutes les places) : 21,3 s, 100 ms par décision en moyenne (p95 336 ms).
- Parties d'imitation (`lab8 record`, phase 0) : 15 parties par seconde, 80 plans par partie,
  335 ko par partie. 100 000 parties : ≈ 2 h et 33 Go.
- Transformer de départ (256 × 6, 8 têtes, décodeur de 2 couches : 7,9 millions de paramètres) :
  apprentissage à 2 350 plans par seconde sur la carte (bf16, lots de 256, sans optimisation), soit
  ≈ 1 h par passe sur 8 millions de plans ; sur un cœur, en f32, 19,5 ms par lecture de position et
  0,3 ms par pas du décodeur (le moteur GPU de la phase 3 est indispensable à la recherche).

## Phases et critères, fixés avant tout calcul

Chaque commande longue a un délai maximal (AGENTS.md).

| Phase | Travail | Critère pour continuer |
| --- | --- | --- |
| 0, base | Paquets `smallmobile-lab8` et `smallmobile-brain8`, qui reprennent de lab7 (figé), par dépendance, ce qui ne change pas : coups, savoir, hasard, encodage, arène, c130 et son moteur GPU. Neufs : les jetons, le réseau (référence Rust sur processeur et PyTorch), le décodeur, le découpage en plans, l'enregistreur des parties d'imitation, l'apprentissage. | Tests : mêmes sorties en Rust et en PyTorch ; tout plan joué est légal ; débits mesurés. **Fait le 26 octobre** (journal). |
| 1, données | 100 000 parties de c130 (graines 81 000 000 et suivantes), 5 % de parties mises de côté | — |
| 2, imitation | Tailles choisies sur les parties mises de côté, entraînement de 24 h au plus | Sur les parties mises de côté, politique et valeur au moins aussi bonnes que le réseau de c130 face à sa propre recherche (par phase et par nombre de joueurs) |
| 3, moteur | Encodeur et décodeur en Vulkan (`check-gpu` contre le processeur, `bench-gpu`) | Écarts du niveau de c130 |
| 4, parité | Le nouveau réseau avec une recherche coup par coup comme celle de lab7 (à adapter au décodeur), n = 400, contre c130 à n = 400 ; 920 donnes, graine 82001 | Score dans ±2 points de 0 |
| 5, plans | La recherche sur plans (écrite après la phase 2 : probabilités exactes du dé, candidats, fins simulées par plans) contre c130 à n = 400, 920 donnes, graine 82002. Sur les mêmes donnes, le meilleur de lab7 (`n=400,roll=1,mix=0.5`), pour une comparaison appariée | Différence appariée avec lab7 au-dessus de 0. Temps par décision (moyenne, p95) noté, sans plafond |
| 6, selfplay | Boucle avec la recherche sur plans ; puis la même recherche sans fins simulées | Gain sur la phase 5 ; les fins simulées retirées quand l'arène ne les distingue plus |

Si l'imitation (phase 2) ne rejoint pas c130, ou si l'entraînement dépasse le budget, on s'arrête pour
décider ensemble : réseau plus petit, ou recherche sur plans avec le réseau de c130.
