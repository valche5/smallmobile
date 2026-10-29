# Bilan des labos 1 à 9 (19–28 octobre 2026)

Relu le 29 octobre 2026 dans les neuf journaux, les plans (`labN/lab.md`), `labtools/README.md` et
`labtools/biblio.md`. Ce document résume ce qui a marché, ce qui n'a pas marché, les pistes
restées ouvertes, et la recommandation pour une reprise de l'apprentissage. Les chiffres sont des
scores du protocole (`labtools/score.py`, intervalles à 95 %) ; les détails sont dans chaque journal.

## Chronologie, un labo par ligne

| Labo | Approche | Résultat |
| --- | --- | --- |
| lab1 | PUCT, boucle ouverte, fins simulées de 3 tours, gen0 → gen11 | Progrès relancé par des **cibles de valeur issues du jeu fort** (+6,1, puis +7,9, +3,7). Fins simulées jouées par le réseau pour tous (`rp=1`) : **+11,7** parmi les Expert |
| lab2 | Gumbel AlphaZero, nœuds de hasard, coups jeton par jeton, départ de zéro, **sans fins simulées** | 60 pas : rattrape lab1 (+0,7 ± 1,6). Plateau dès c050 |
| lab3 | Spécialiste par nombre de joueurs, puis généraliste 128×7 | Spécialiste depuis zéro : +8,4 à sa taille à moyens égaux ; spécialiser c060 le dégrade. Sortie du plateau par **n = 400** (+4,9) et **taux bas** (+2,9) → **c130 = +10,1 sur c060**, Expert du jeu |
| lab4 | Poids fixes : budget, largeur, horizon, critère de valeur | 8× simulations +2,0 ± 2,3 ; largeur, jetons ordonnés ou fusionnés : rien ; `win=0.75` : +1,3 ± 1,0. Boucles w1/w2 (victoire seule, cible TD) : rien en 20 pas |
| lab5 | Sonde des coups manqués par c130 | **Résultats invalides** (shader bf16 Vulkan faux sur NVIDIA) ; méthode jugée biaisée de toute façon |
| lab6 | 16× la recherche | +1,8 ± 2,2 : indécis. Moteur CUDA |
| lab7 | Fins simulées, marché tardif, histoire vide, bruit du selfplay ; quatre boucles | **Fins simulées : +5,4 ± 1,5 (n = 100), +8,9 ± 2,3 (n = 400)**, premier gain net depuis lab3. Marché, histoire : rien. Quatre recettes d'apprentissage (T0 à T3) : rien en 10 pas |
| lab8 | Transformer neuf, décodeur de plans de tour, recherche sur plans | i2 imite c130 mieux que son propre réseau ; recherche sur plans +3,6 ± 1,5 en 134 ms par décision, mais **−6,1 ± 2,1 face aux fins simulées de lab7** ; cible du selfplay ≈ a priori |
| lab9 | Encodeur hybride ResTNet (RRTRRT) dans le cadre de lab8 | h1 : meilleur réseau d'imitation (+1,9 ± 1,5 sans recherche contre c130 à n = 1), 1,4× moins cher qu'i2 ; en partie, le gain fond. Boucle p1 : 4 pas sans progrès |

## Ce qui a marché

1. **La qualité des cibles, pas leur quantité.** Valeur apprise sur du jeu fort (lab1 P2), recherches
   enregistrées à n = 400 au lieu de 200 (lab3), taux d'apprentissage bas sur un réseau mûr (lab3).
   Chaque fois un saut net ; le reste (plus de données, plus de passes) ne donnait rien.
2. **Les fins simulées à la décision.** +11,7 (lab1), +5,4 et +8,9 (lab7). Le gain le plus gros et le
   plus reproductible, croissant avec le nombre de joueurs. Avec elles, le calcul paie de nouveau
   (de +2,9 à n = 50 à +8,9 à n = 400) alors que la recherche ordinaire ne gagnait que +1,8 pour
   16 fois plus. **Non déployé** : `src/shared/experts.ts` sert toujours c130 à n = 400 sans fins
   simulées.
3. **La largeur du réseau** aide dès qu'il y a assez de parties (lab2 : 128×4 rejoint c020 avec la
   moitié des parties).
4. **L'imitation fonctionne.** Un transformer ou un hybride apprend c130 en 1 à 4 h sur le PC ;
   l'hybride R/T bat le tout-transformer à chaque lot (lab9), comme dans ResTNet. Piège rencontré :
   la cible du vainqueur final, répétée sur tous les plans d'une partie, s'apprend par cœur ; un plan
   sur quatre et deux fois plus de parties le corrigent (i2).
5. **L'ingénierie.** Moteur Vulkan de c130 à ~1 million de lectures/s, selfplay à ~3 500 positions/s
   sur le PC, données compactes en RAM, arènes sur GPU 18 fois plus vites. C'est le socle d'un
   selfplay rapide.

## Ce qui n'a pas marché

1. **Plus de recherche sans fins simulées** : 8× (lab4), 16× (lab6), toute la largeur, jetons
   ordonnés ou fusionnés : rien de net. À 4 et 5 joueurs, moins de 4 % des simulations atteignent
   le tour suivant du joueur ; la recherche optimise la valeur du réseau en fin de morceau de tour
   et n'évalue jamais la réponse adverse (même constat que Justesen 2017 et le mémoire sur Small
   World).
2. **Toute boucle d'apprentissage depuis c130** : w1, w2 (lab4), T0 à T3 (lab7), p1 (lab9), toutes
   plates. Dénominateur commun : **le professeur n'est pas plus fort que l'élève** au budget du
   selfplay. Même T3, dont le professeur avait des fins simulées (+5,4 en arène), n'a rien donné en
   10 pas et 125 000 parties.
3. Sur un réseau mûr, en 3 à 20 pas : 7e couche, apprentissage continu, n = 800, valeur de la
   recherche comme cible, cible TD, victoire seule, tête des propriétaires, selfplay sans bruit ni
   hasard : rien de mesurable. La capacité aide à apprendre (validation), pas à jouer.
4. **Spécialiser un généraliste mûr** (lab3, depuis c060 au taux plein) : gain bref, puis
   dégradation.
5. **La recherche sur plans** (lab8, lab9) : efficace par milliseconde (vaut 400 simulations de c130
   en 30 fois moins de temps), mais plafonne à un tour de profondeur et reste nettement sous les
   fins simulées ; sa cible de selfplay s'écarte peu de l'a priori (KL 0,14 nat avec la
   normalisation de mctx, 0,005 avant).
6. **Les arènes face à Difficile et Normale** ne mesurent plus rien depuis lab2 ; seules les arènes
   entre réseaux comptent, et la mesure à 920 donnes ne voit que des tranches de 10 à 20 pas.
7. Le shader bf16 Vulkan calcule faux sur NVIDIA (lab5 perdu) ; sur ces cartes, CUDA.

## Ce que dit la bibliographie (`labtools/biblio.md`)

- **Diplomacy** (SearchBot, DORA, Diplodocus) : recherche sur des tours entiers tirés de la
  politique, fins simulées courtes (2 à 3 tours) gardées à la décision, **ancrage piKL** de la
  recherche sur la politique. Nos fins simulées sont la méthode la mieux soutenue.
- **Jeux à plusieurs actions par tour** (Justesen 2017, Baier & Cowling 2018) : le MCTS s'épuise
  dans le tour courant ; la meilleure méthode publiée est une suite entière prolongée par un
  adversaire glouton bon marché, sans branchement.
- **Diplodocus** : à plusieurs joueurs, un agent jugé contre ses propres copies peut tromper ;
  évaluer sur une population.
- **Blomqvist (Risk)** : mêmes symptômes que nos paliers (la politique aide, la valeur n'apprend
  plus).
- **Sampled MuZero** : les règles pour rester juste quand on cherche sur des plans tirés.

## Pistes inexplorées

| Piste | Coût | Pourquoi |
| --- | --- | --- |
| **Ancrage piKL** : λ fixe, sans ramener les valeurs entre 0 et 1 dans chaque nœud | Une constante, aucun entraînement | Notre ancrage Gumbel faiblit avec les visites ; peut expliquer que chercher plus ne rapporte rien |
| **Déployer les fins simulées dans le jeu** (c130, `n=100,roll=1,mix=0.5`, code de lab7) | Nul ; ~1,5 s par recherche sur un cœur de mini | +5,4 confirmé sur 920 donnes neuves |
| **Selfplay depuis zéro avec fins simulées** | Long | lab2 les a retirées sans mesure ; toutes les boucles depuis ont appris sans. T3 partait de c130, 10 pas |
| Fins simulées par un **réseau léger** (64×4, h2) | Moyen | Une fin simulée coûte ~29 lectures ; dans lab9 les lectures font 75 % du GPU |
| **Boucles longues** | Long | lab2 → c130 a demandé 60 à 130 pas ; toutes les boucles post-c130 ont 4 à 20 pas |
| Spécialistes **depuis c130 au taux bas** | Moyen | Prévu dans lab7 (« Ensuite »), jamais fait ; lab3 spécialisait au taux plein |
| Réseau de graphe **plus large** (256) | Réécrire le shader de c130 | lab2 : la largeur paie avec assez de parties ; c130 n'a que 1,1 million de paramètres |
| Têtes **politique et valeur séparées** (DORA) | Moyen | Non essayé |
| Candidats par **mutation** du meilleur plan (`mods` de lab8, écrit, jamais mesuré), second tour de profondeur | Moyen | La recherche sur plans plafonne sur les mondes, pas sur les candidats |
| Évaluation sur **population** et sur les parties humaines archivées (`data/games/`) | Faible | Nos arènes sont c130 contre c130 |
| Best Reply Search, adversaires typés (Pluribus) dans les fins simulées, quantités grossières au redéploiement (1 / moitié / tout) | Moyen | Non essayés |
| Adaptation aux adversaires (histoire des joueurs, lab1 P3) | Moyen | gen12 abandonné ; l'histoire vide ne coûte rien de mesurable (lab7 A0) : faible priorité |

## Recommandation : reprendre avec un selfplay rapide

Le selfplay le plus rapide dont on dispose est, de loin, la filière c130 (réseau de graphe 128×7,
moteur Vulkan de lab3/lab4/lab7), mesuré sur le PC (RX 9070 XT) :

| Pipeline | Débit |
| --- | --- |
| lab3/lab4 `loop.sh`, c130, n = 400 (une décision sur 8 enregistrée), Vulkan | **~65 parties/s** (25 000 en ~390 s), pas complet ~570 s |
| lab8 `record`, c130 à n = 400 à chaque décision, sans bruit | 17 parties/s |
| lab9 `selfplay`, h1, plans k = 8, m = 2, 200 fils | ~3 parties/s (5 000 en 28 min), 20 fois plus lent |

Ordre proposé :

1. **Sans entraînement, deux mesures** : l'ancrage piKL sur c130 (à n = 400 et à 16×), et les fins
   simulées de lab7 en production, qui sont un gain acquis.
2. **Une boucle depuis c130 avec la filière rapide** (lab4/lab7 `loop.sh`, Vulkan), mais un
   **professeur réellement plus fort** : fins simulées aux seules décisions enregistrées (une sur 8,
   voire une sur 16 pour garder 15 à 20 parties/s), cible de valeur après fin simulée, et **au
   moins 40 à 60 pas** avant de conclure : T3 a eu 10 pas, et lab3 a eu besoin de 60 et plus pour
   gagner 10 points.
3. Garder h1 et h2 pour l'imitation et comme témoins ; le selfplay sur plans reste trop lent et sa
   cible trop proche de l'a priori tant que la recherche ne dépasse pas un tour.
