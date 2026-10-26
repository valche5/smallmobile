# Bibliographie : IA pour les jeux proches de Small World

Articles sur les jeux qui ressemblent au nôtre : carte irrégulière, 2 à 5 joueurs, tours de
nombreuses décisions, dés et pioche cachée. Chaque entrée dit ce que l'article apporte et ce
qu'il veut dire pour les labos.

Degré de lecture, indiqué pour chaque entrée :
- **lu** : article lu en entier ;
- **parcouru** : passages clés lus ;
- **résumé** : seulement le résumé et des sources secondaires.

Recherche faite le 26 octobre 2026. Six articles sont lus en entier ; leurs notes de lecture sont à
la fin, avec ce qu'elles changent pour nos pistes. Les PDF payants (IEEE) sont dans
`labtools/pdf/` (hors Git).

## Ce qu'il faut retenir

1. **Small World lui-même** n'a qu'un mémoire de licence (Nieuwenhuizen 2021), avec une recherche
   MCTS sans réseau. Sa conclusion rejoint lab6 : l'agent « ne regarde presque que son propre tour ».
2. **Le plus proche de notre problème, c'est Diplomacy** : carte irrégulière, 7 joueurs, beaucoup
   d'ordres par tour. Les meilleurs agents (Meta, DeepMind) cherchent sur des **tours entiers**
   tirés de la politique. Ils gardent des **fins simulées courtes** (2 à 3 tours) au moment de
   jouer. Et ils **tiennent la recherche en laisse** près de la politique (piKL).
3. **Les jeux à plusieurs actions par tour** (Hero Academy, Blood Bowl, jeux de stratégie en temps
   réel) ont tous buté sur le même mur. La recherche s'épuise à combiner les actions du tour et ne
   voit pas la réponse de l'adversaire. Les solutions publiées cherchent sur des séquences
   entières (évolution en ligne, MCTS évolutionnaire) ou échantillonnent les combinaisons.
4. **Les fins simulées tronquées** suivies d'une évaluation sont une méthode classique et étudiée :
   Tesauro 1996, Bertsekas, AlphaGo 2016. Ce n'est pas un bricolage.

## Small World

- **Nieuwenhuizen, *The Effect of Monte Carlo Tree Search on Modern Board Game Elements*,
  mémoire de licence, LIACS, Université de Leiden, 2021.**
  [PDF](https://theses.liacs.nl/pdf/2020-2021-NieuwenhuizenD.pdf) — *parcouru.*
  - **Le cadre.** MCTS (UCT) avec fins simulées au hasard jusqu'à la fin de la partie, à 2
    joueurs seulement.
  - **La décomposition du tour.** Le tour est découpé en actions successives, comme chez nous : un
    nœud de l'arbre correspond à une décision, pas à un tour entier. L'auteur estime à plus de
    600 000 les fins de tour possibles dès le premier tour.
  - **Le constat.** Le découpage règle le facteur de branchement, mais l'agent « regarde presque
    exclusivement son propre tour ». Il ne planifie rien sur le long terme et choisit mal les
    combos.
  - **Pour nous.** C'est le seul travail trouvé sur Small World. Le même diagnostic que lab6, sans
    réseau.

## Diplomacy

Le domaine le plus proche. Pas de dés, tours simultanés, mais une carte de provinces irrégulière, 7
joueurs, et des ordres donnés à toutes les unités à chaque tour.

- **Paquette et al., « No Press Diplomacy: Modeling Multi-Agent Gameplay » (DipNet), NeurIPS 2019.**
  [arXiv 1909.02128](https://arxiv.org/abs/1909.02128) — *lu.*
  - **Le réseau.** Réseau de graphe de 16 couches sur les 81 lieux de la carte, appris sur des
    parties humaines.
  - **La sortie.** Un décodeur émet les ordres unité par unité, d'un trait.
  - **Pour nous.** C'est la base de tout ce qui suit, et notre architecture est de la même famille.

- **Anthony et al. (DeepMind), « Learning to Play No-Press Diplomacy with Best Response Policy
  Iteration » (BRPI), NeurIPS 2020.** [arXiv 2006.04635](https://arxiv.org/abs/2006.04635) — *lu.*
  - **La recherche.** On tire des tours complets candidats et des tours adverses. Chaque candidat
    est jugé par la valeur un tour plus tard, en moyenne sur ces mêmes tours adverses.
  - **Pour nous.** C'est l'équivalent de notre option `crn`.

- **Gray et al. (Meta), « Human-Level Performance in No-Press Diplomacy via Equilibrium Search »
  (SearchBot), ICLR 2021.** [arXiv 2010.02923](https://arxiv.org/abs/2010.02923) — *lu.*
  - **Les candidats.** Les tours complets les plus probables selon la politique, pour chaque
    joueur.
  - **Le choix.** Le meilleur mélange de candidats face aux autres, trouvé par *regret matching*.
  - **L'évaluation.** Une fin simulée de 2 à 3 tours. Ils n'observent aucun gain au-delà de 3 ou 4.
  - **Le coût.** 2 à 20 minutes par tour.
  - **Pour nous.** La même profondeur courte que notre `roll=1`, à une autre échelle de calcul.

- **Bakhtin et al. (Meta), « No-Press Diplomacy from Scratch » (DORA), NeurIPS 2021.**
  [arXiv 2110.02924](https://arxiv.org/abs/2110.02924) — *lu.*
  - **Le réseau.** Un transformer remplace le réseau de graphe (898 → 1 123 Elo), avec politique et
    valeur séparées.
  - **La cible de la valeur.** Elle est amorcée : la valeur au tour suivant, pas le vainqueur.
  - **L'exploration.** Les nouveaux coups viennent de modifications locales des bons coups.
  - **Les fins simulées.** Gardées à 7 joueurs au moment de jouer (2 tours), mais pas pendant
    l'apprentissage.
  - **Pour nous.** Lab4 a essayé une cible de ce genre (`--td 0.5`), sans effet mesurable en 20 pas
    (`lab4/journal.md`). La séparation politique/valeur et la génération des candidats n'ont pas
    été essayées.

- **Jacob et al. (Meta), « Modeling Strong and Human-Like Gameplay with KL-Regularized Search »
  (piKL), ICML 2022.** [arXiv 2112.07544](https://arxiv.org/abs/2112.07544) — *lu (note de
  lecture).*
  - **L'idée.** La recherche est pénalisée selon sa distance à une politique d'ancrage. Elle
    prédit mieux les coups des joueurs forts et joue mieux que l'imitation seule (échecs, go,
    Diplomacy).
  - **Pour nous.** Notre recherche Gumbel a déjà cette forme, mais son ancrage faiblit quand les
    visites augmentent (voir la note).

- **Bakhtin et al. (Meta), « Mastering the Game of No-Press Diplomacy via Human-Regularized
  Reinforcement Learning and Planning » (Diplodocus), ICLR 2023.**
  [arXiv 2210.05492](https://arxiv.org/abs/2210.05492) — *lu (note de lecture).*
  - **La méthode.** piKL appliqué pendant le selfplay (RL-DiL-piKL).
  - **Le résultat.** Premier et troisième d'un tournoi de 200 parties contre 62 humains.
  - **La thèse.** Le selfplay seul ne suffit pas quand l'agent joue avec des humains.
  - **Pour nous.** L'Expert est appris en selfplay pur et joue contre des humains dans l'app.
    Nos arènes ne mesurent que l'Expert contre lui-même.

- **Meta FAIR Diplomacy Team, « Human-level play in the game of Diplomacy by combining language
  models with strategic reasoning » (Cicero), *Science* 378, 2022.**
  [DOI 10.1126/science.ade9097](https://doi.org/10.1126/science.ade9097) — *résumé.*
  - **Ce que c'est.** Diplomacy avec négociation en langage naturel. Planification piKL et modèle
    de langue.
  - **Pour nous.** Hors sujet, sauf pour situer la lignée.

- **Kramár et al. (DeepMind), « Negotiation and honesty in artificial intelligence methods for the
  board game of Diplomacy », *Nature Communications* 13, 7214, 2022.**
  [Article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9726752/) — *résumé.*
  - **Ce que c'est.** Des agents qui passent des accords et sanctionnent ceux qui les rompent.
  - **Pour nous.** Hors sujet : Small World n'a pas de négociation.

## Jeux à plusieurs actions par tour

Le problème de profondeur de lab6 : un tour de 25 à 30 décisions, et la recherche n'atteint jamais
l'adversaire.

- **Justesen, Mahlmann, Risi & Togelius, « Playing Multi-Action Adversarial Games: Online
  Evolutionary Planning versus Tree Search », *IEEE Transactions on Games* 10(3), 2018.**
  [Notice](https://pure.itu.dk/en/publications/playing-multi-action-adversarial-games-online-evolutionary-planni/)
  — *lu (note de lecture ; PDF dans `labtools/pdf/justesen2017.pdf`).*
  - **Le jeu.** Hero Academy, 5 actions par tour.
  - **La méthode.** Un algorithme évolutionnaire cherche directement la séquence entière du tour,
    jugée par une fonction d'évaluation. Il bat MCTS, d'autant plus nettement que le tour compte
    d'actions.
  - **La version de 2016.** EvoApplications 2016, sous le nom « Online Evolution ».

- **Baier & Cowling, « Evolutionary MCTS for Multi-Action Adversarial Games », IEEE CIG 2018**, et
  **« Evolutionary MCTS with Flexible Search Horizon », AIIDE 2018.**
  [AIIDE](https://ojs.aaai.org/index.php/AIIDE/article/view/13023) — *les deux lus (notes de
  lecture ; PDF de CIG dans `labtools/pdf/baier2018.pdf`).*
  - **La méthode.** Un arbre dont les nœuds sont des séquences d'actions entières et les branches
    des mutations de ces séquences.
  - **Sa limite.** La première version ne voyait pas au-delà du tour du joueur.
  - **La correction.** La seconde la prolonge avec un modèle simple du jeu des deux joueurs.
  - **Pour nous.** C'est exactement notre « fin simulée », greffée sur une recherche par tours
    entiers.

- **Ontañón, « Combinatorial Multi-armed Bandits for Real-Time Strategy Games », *JAIR* 58, 2017.**
  [arXiv 1710.04805](https://arxiv.org/abs/1710.04805) — *résumé.*
  - **La méthode.** NaïveMCTS considère une action comme la combinaison de sous-actions (une par
    unité) et les échantillonne séparément. Son avantage croît avec le facteur de branchement.
  - **Pour nous.** Une piste pour choisir les conquêtes et le redéploiement ensemble.

- **Moraes, Mariño, Lelis & Nascimento, « Action Abstractions for Combinatorial Multi-Armed Bandit
  Tree Search », AIIDE 2018.** [AIIDE](https://ojs.aaai.org/index.php/AIIDE/article/view/13018)
  — *résumé.* Restreindre les combinaisons à celles qu'engendrent quelques scripts simples.

- **Justesen et al., « Blood Bowl: A New Board Game Challenge and Competition for AI », IEEE CoG
  2019.** [Notice](https://pure.itu.dk/en/publications/blood-bowl-a-new-board-game-challenge-and-competition-for-ai/)
  — *résumé.*
  - **Le jeu.** Chaque pièce bouge plusieurs fois par tour, il y a des dés à chaque action, et les
    points sont rares.
  - **Le bilan.** En 2022, le tournoi Bot Bowl a été gagné par un programme à règles écrites.
  - **Sur le même jeu.** « MimicBot » ([arXiv 2108.09478](https://arxiv.org/abs/2108.09478)) part
    de l'imitation puis passe à l'apprentissage par renforcement. « Planning in the midst of
    chaos » ([HAL 03345360](https://hal.archives-ouvertes.fr/hal-03345360), CoG 2021) propose un
    modèle stochastique du tour.

- **Hubert et al. (DeepMind), « Learning and Planning in Complex Action Spaces » (Sampled MuZero),
  ICML 2021.** [arXiv 2104.06303](https://arxiv.org/abs/2104.06303) — *lu (note de lecture).*
  - **La méthode.** La recherche porte sur un échantillon de coups tirés de la politique, avec une
    correction pour que l'amélioration de politique reste juste.
  - **Pour nous.** Le cadre théorique pour chercher sur des plans de tour échantillonnés plutôt que
    sur tous les coups légaux.

## Autres jeux de société modernes

- **Szita, Chaslot & Spronck, « Monte-Carlo Tree Search in Settlers of Catan », Advances in
  Computer Games 12, 2009 (LNCS 6048, 2010).**
  [Proceedings](https://chessprogramming.org/Advances_in_Computer_Games_12) — *résumé.* Un MCTS
  avec des fins simulées guidées par des connaissances du jeu.

- **Gendre & Kaneko, « Playing Catan with Cross-dimensional Neural Network », 2020.**
  [arXiv 2008.07079](https://arxiv.org/abs/2008.07079) — *résumé.*
  - **L'idée.** Une architecture pour des entrées et des sorties de formes mêlées : plateau,
    cartes, joueurs.
  - **Le résultat.** L'agent bat jSettler.
  - **Pour nous.** Le même problème que notre marché et nos joueurs aplatis dans 820 nombres.

- **Xenou, Chalkiadakis & Afantenos, « Deep Reinforcement Learning in Strategic Board Game
  Environments », EUMAS 2018.** [HAL](https://hal.archives-ouvertes.fr/hal-02124411) — *résumé.*
  Catan, valeurs Q par LSTM.

- **Blomqvist, *Playing the Game of Risk with an AlphaZero Agent*, mémoire de master, KTH, 2020,
  avec l'agence suédoise de recherche pour la défense (FOI).**
  [PDF](https://forums.triplea-game.org/assets/uploads/files/1637032761173-fulltext01.pdf)
  — *lu (note de lecture).*
  - **Le jeu.** Risk : conquêtes au dé, plusieurs joueurs. Le plus proche de Small World par le
    thème.
  - **La méthode.** MCTS guidé par un réseau appris par Expert Iteration. Les entrées sont plates,
    avec **une tête de politique par type de décision** (cinq).
  - **Le résultat.** La politique apprise améliore la recherche, mais plus rien ne progresse après
    quelques itérations. Le réseau **n'apprend pas de bonne valeur** de position. C'est le même
    symptôme que nos paliers.

- **Gaina, Balla, Dockhorn, Montoliu & Perez-Liebana, « Design and Implementation of TAG: A
  Tabletop Games Framework », 2020.** [arXiv 2009.12065](https://arxiv.org/abs/2009.12065) —
  *résumé.*
  - **Ce que c'est.** Un cadre Java pour la recherche sur les jeux de société modernes, avec des
    dizaines de jeux.
  - **La suite.** Goodman, Perez-Liebana & Lucas (2022,
    [arXiv 2202.05773](https://arxiv.org/abs/2202.05773)) y étudient l'effet du nombre de
    joueurs sur huit jeux.

## Jeux à plusieurs joueurs

- **Petosa & Balch, « Multiplayer AlphaZero », NeurIPS 2019 Deep RL Workshop.**
  [arXiv 1910.13012](https://arxiv.org/abs/1910.13012) — *résumé.* Une valeur par joueur et une
  remontée façon max^n. C'est ce que font nos labos depuis lab1.

- **Schadd & Winands, « Best Reply Search for Multiplayer Games », *IEEE TCIAIG* 3(1), 2011.**
  [PDF](https://dke.maastrichtuniversity.nl/m.winands/documents/BestReplySearch.pdf) — *résumé.*
  - **L'idée.** À chaque tour adverse, seul l'adversaire qui a la meilleure riposte joue. La
    recherche voit ainsi plus de tours du joueur.
  - **Le résultat.** Elle bat max^n et paranoid sur plusieurs jeux.
  - **Pour nous.** Une autre façon d'atteindre la réponse adverse sans dérouler les 4 adversaires.

- **Nijssen & Winands, aperçu des techniques de recherche à plusieurs joueurs.**
  [PDF](https://dke.maastrichtuniversity.nl/m.winands/documents/Multi_Overview.pdf) — *résumé.*
  Voir aussi Sturtevant, « An Analysis of UCT in Multi-Player Games », 2008 (non vérifié).

- **Brown & Sandholm, « Superhuman AI for multiplayer poker » (Pluribus), *Science* 365, 2019.**
  [Article](https://par.nsf.gov/servlets/purl/10119653) — *résumé.*
  - **La méthode.** Recherche à profondeur limitée. Au-delà de l'horizon, chaque adversaire est
    réduit à 4 stratégies : le plan de base et trois versions biaisées.
  - **Pour nous.** Une idée transposable à nos fins simulées : plusieurs réponses adverses typées
    au lieu d'une seule réponse préférée.

## Hasard et information cachée

- **Cowling, Powley & Whitehouse, « Information Set Monte Carlo Tree Search », *IEEE TCIAIG*
  4(2), 2012.** [PDF](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf)
  — *résumé.* Les « mondes » de nos labos : tirer des versions possibles de l'information cachée.

- **Antonoglou et al. (DeepMind), « Planning in Stochastic Environments with a Learned Model »
  (Stochastic MuZero), ICLR 2022.** [Notice](https://mlanthology.org/iclr/2022/antonoglou2022iclr-planning)
  — *résumé.*
  - **La méthode.** Des « états d'après-coup » et des nœuds de hasard dans un modèle appris.
  - **Le résultat.** État de l'art au backgammon et au 2048.
  - **Pour nous.** Nos nœuds de hasard (lab2) en sont l'équivalent avec le vrai moteur.

- **Perolat et al. (DeepMind), « Mastering the game of Stratego with model-free multiagent
  reinforcement learning » (DeepNash), *Science* 378, 2022.**
  [arXiv 2206.15378](https://arxiv.org/abs/2206.15378) — *résumé.* Aucune recherche, une
  dynamique d'apprentissage régularisée (R-NaD). Trois premiers de tous les temps sur Gravon.

- **Schmid et al. (DeepMind), « Student of Games », *Science Advances* 9, 2023.**
  [arXiv 2112.03178](https://arxiv.org/abs/2112.03178) — *résumé.* Une recherche unique pour
  l'information parfaite et cachée (GT-CFR) : échecs, go, poker, Scotland Yard.

## Fins simulées et recherche guidée par réseau

- **Tesauro & Galperin, « On-line Policy Improvement using Monte-Carlo Search », NIPS 1996.**
  [Article](https://proceedings.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html)
  — *résumé.*
  - **L'idée.** Juger chaque coup en jouant la suite avec la politique de base. C'est l'ancêtre de
    `roll=1`.
  - **Le résultat.** Au backgammon, les erreurs de TD-Gammon baissent jusqu'à 5 fois.

- **Bertsekas, *Rollout, Policy Iteration, and Distributed Reinforcement Learning*, 2020.**
  *Non vérifié.* Le cadre théorique de la fin simulée tronquée suivie d'une évaluation.

- **Silver et al., « Mastering the game of Go with deep neural networks and tree search »
  (AlphaGo), *Nature* 529, 2016.** [DOI 10.1038/nature16961](https://doi.org/10.1038/nature16961)
  — *de mémoire.*
  - **Le mélange.** La valeur d'une feuille mélange fin simulée et réseau de valeur ; λ = 0,5 bat
    chacune des deux seule. C'est notre `mix=0.5`.
  - **Les entrées.** 48 plans d'entrée, dont des plans tactiques calculés à la main (échelles,
    libertés).

- **Lanctot, Winands, Pepels & Sturtevant, « Monte Carlo Tree Search with Heuristic Evaluations
  using Implicit Minimax Backups », CIG 2014.** [arXiv 1406.0486](https://arxiv.org/abs/1406.0486)
  — *résumé.* Faire remonter séparément les résultats simulés et les évaluations.

- **Silver et al., AlphaZero, *Science* 362, 2018.** [arXiv 1712.01815](https://arxiv.org/abs/1712.01815)
  — *de mémoire.* Sans fins simulées : à deux joueurs alternés, l'arbre voit l'adversaire dès le
  premier niveau.

- **Danihelka, Guez, Schrittwieser & Silver, « Policy Improvement by Planning with Gumbel », ICLR
  2022.** [OpenReview](https://openreview.net/forum?id=bERaNdoegnO) — *résumé.* La recherche Gumbel
  des labos depuis lab2.

- **Anthony, Tian & Barber, « Thinking Fast and Slow with Deep Learning and Tree Search » (Expert
  Iteration), NeurIPS 2017.** [arXiv 1705.08439](https://arxiv.org/abs/1705.08439) — *de mémoire.*
  Apprendre au réseau à imiter la recherche (piste « professeur » de lab7).

- **Wu, « Accelerating Self-Play Learning in Go » (KataGo), 2019.**
  [arXiv 1902.10565](https://arxiv.org/abs/1902.10565) — *résumé.* Des cibles auxiliaires (dont la
  propriété des cases, reprise par notre tête des propriétaires) et des entrées calculées à la
  main.

## Réseaux de graphe pour les jeux

- **Ben-Assayag & El-Yaniv, « Train on Small, Play the Large: Scaling Up Board Games with AlphaZero
  and GNN », 2021.** [arXiv 2107.08387](https://arxiv.org/abs/2107.08387) — *résumé.*
  - **L'idée.** AlphaZero avec un réseau de graphe, pour jouer sur des plateaux de taille
    quelconque.
  - **Pour nous.** Le même argument que pour nos cartes générées.

## Ce qui manque

- **Aucun travail avec apprentissage sur Small World**, ni sur un jeu de conquête au dé à 5
  joueurs. Le mémoire sur Risk est le plus proche.
- **Rien sur la façon de découper un tour** (une décision à la fois ou un tour entier) dans un jeu
  séquentiel avec hasard au milieu du tour. Diplomacy n'a pas de dés, Hero Academy n'a que 2
  joueurs.

## Notes de lecture (26 octobre 2026)

Six articles lus en entier. Chaque note sépare ce que dit l'article (section ou tableau entre
parenthèses) de notre interprétation (« pour nous »). Les chiffres clés ont été revérifiés dans
les textes.

### Jeux à plusieurs actions par tour : Justesen 2017, Baier & Cowling 2018 (CIG et AIIDE)

**Le cadre commun.** Hero AIcademy, un clone simplifié de Hero Academy :
- 2 joueurs, 5 points d'action par tour (jusqu'à 15 dans les essais) ;
- environ 7,8 × 10⁸ tours possibles ;
- rendu déterministe et à information complète pour les expériences ;
- une évaluation écrite à la main (la valeur des unités), très bon marché, rien n'est appris.

**Justesen et al.**
- **OEP.** Un algorithme évolutionnaire dont le génome est la séquence des actions du tour :
  population de 100, croisement et mutation avec réparation des séquences illégales, 6 s par tour.
- **Le résultat.** OEP bat le MCTS de base à 98 %. Il fait jeu égal avec deux MCTS « contraints » :
  sans exploration, ou en ne gardant que le meilleur fils de la racine à chaque phase (tableau II).
- **L'arbre ne voit pas l'adversaire.** Le MCTS de base fait 258 488 itérations par tour, mais
  n'atteint que 201 fins de tour distinctes. Ses feuilles sont en moyenne à 4,86 actions de
  profondeur : à peine le tour adverse (§VII-A). C'est notre constat de lab6.
- **L'adversaire est ignoré.** Le simuler au hasard n'apportait rien (§IV).

**Baier & Cowling, CIG 2018 (EMCTS).**
- **La méthode.** Un nœud est un tour entier, une arête est une mutation (changer une action). La
  racine est le tour glouton, et une feuille est jugée par l'évaluation statique.
- **Le résultat.** Elle bat OEP de 52 à 65 % selon la longueur du tour et le temps (tableau I).
- **Sa limite.** Elle ne regarde toujours pas l'adversaire.

**Baier & Cowling, AIIDE 2018 (« Flexible Search Horizon »).**
- **La méthode.** Le génome continue au-delà du tour. Les actions adverses y sont insérées et
  **toujours régénérées par un modèle glouton** après chaque mutation, sans branchement.
- **Le résultat.** Elle bat EMCTS de 62 à 85 % (tableau 1). Un modèle adverse faible suffit, et
  l'horizon aide surtout quand les tours sont courts.

**Pour nous.**
- **Ce que ces articles valident.** Nos fins simulées : un adversaire déterministe et bon marché
  inséré dans l'évaluation, sans branchement, c'est leur meilleure méthode.
- **Ce qu'ils ne valident pas.** Un arbre de plans de tour avec branchement sur les réponses
  adverses : aucun des trois ne le fait, et ils le présentent comme un problème ouvert.
- **Ce qu'on peut reprendre.**
  - Engendrer les candidats par **mutation et réparation autour du plan glouton du réseau**,
    plutôt qu'avec une liste fixe.
  - Le **redéploiement** s'y prête bien : il est déterministe, puisqu'il vient après le dé.
  - Une fin simulée jouée par une **politique plus légère** pourrait garder une bonne part du gain
    pour moins que 12 fois le calcul.
- **La prudence.** Ces études portent sur 2 joueurs, un jeu déterministe, une évaluation qui
  permet 10⁴ à 10⁵ évaluations par tour (nous en avons 100 à 400, coûteuses), et des MCTS de
  référence sans réseau. L'ampleur de leurs gains ne se transpose pas.

### piKL (Jacob et al., ICML 2022)

**Le problème.** L'imitation prédit bien les coups humains mais joue faiblement. Le selfplay joue
fort mais d'une façon étrangère aux humains. La critique porte sur la ressemblance aux humains,
pas sur les erreurs de la valeur.

**La méthode.**
- **L'idée.** Maximiser l'utilité moins λ fois la distance (KL) à une politique d'ancrage τ, ici
  un réseau d'imitation humaine.
- **Aux échecs et au go.** Rien de nouveau : le MCTS PUCT fait déjà cela, avec un λ qui dépend de
  c_puct et diminue avec les visites (annexe C).
- **En Diplomacy.** Un algorithme pour coups simultanés (piKL-Hedge) avec λ fixe.

**Les résultats.**
- **Aux échecs et au go**, moins d'ancrage donne un agent plus fort : de 72 % à 94 % contre le
  réseau brut aux échecs (tableau 4). Mais la prédiction des coups humains est la meilleure pour
  un λ intermédiaire.
- **En Diplomacy**, un ancrage léger (λ = 10⁻³) améliore à la fois le score et la prédiction
  (§4.3.2).
- **Au Hanabi**, il passe de 20,23 à 23,11 points (tableau 13). Les auteurs l'expliquent par un
  modèle appris utilisé hors de sa distribution, une situation proche de notre hypothèse
  « exploitation ».

**Pour nous.**
- **Notre recherche Gumbel a déjà cette forme.** La politique améliorée, softmax(logits + σ(q)),
  est une régularisation KL vers notre réseau. Le λ effectif vaut l'écart des valeurs divisé par
  `(50 + visites du coup le plus vu) × 0,1` (`lab7/brain/src/search.rs`, `completed_into`).
- **Deux différences avec piKL, deux leviers possibles pour l'hypothèse « exploitation » :**
  - Notre ancrage **faiblit quand on cherche plus**. Le facteur vaut environ 10 à n = 200 et
    dépasse 80 à 16 fois plus de simulations (estimation). C'est la recette Gumbel standard
    (mctx), pas une erreur, mais elle pourrait expliquer que chercher plus ne rapporte rien.
  - Les valeurs sont **ramenées entre 0 et 1 dans chaque nœud** : des écarts de bruit entre coups
    presque égaux deviennent aussi décisifs qu'un vrai écart.
- **Limite.** Le gain de force de l'ancrage n'est démontré qu'en Diplomacy, avec un ancrage
  humain qui apporte une information extérieure. Le nôtre n'en apporte pas.
- **Coût.** Une constante ou une formule à changer, sans réapprentissage.

### Diplodocus (Bakhtin et al., ICLR 2023)

**Le problème.** À 2 joueurs en somme nulle, le selfplay converge vers un jeu sûr. Dès qu'il y a
de la coopération ou plus de 2 joueurs, il ne garantit plus rien : beaucoup de conventions sont
possibles, et les humains n'en utilisent que quelques-unes (§1).

**La méthode.**
- **DiL-piKL.** C'est piKL avec un λ tiré à chaque itération : chaque valeur représente un type
  de joueur.
- **Deux λ distincts.** Les adversaires sont modélisés près de l'ancrage, tandis que l'agent joue
  avec un λ bas, presque en meilleure réponse (§3.1).
- **RL-DiL-piKL.** Le selfplay de DORA avec cette recherche, l'ancrage restant fixe pendant tout
  l'apprentissage.

**Les résultats.**
- **Contre les humains.** 1er et 3e d'un tournoi de 200 parties contre 62 humains (tableau 2).
  Les auteurs reconnaissent des marges larges.
- **À l'inférence seulement**, la méthode n'apporte presque rien. Le gain vient des réseaux
  réentraînés avec elle (fig. 6).

**Ce qu'ils disent du selfplay pur.**
- DORA, superhumain à 2, joue mal à 7 avec des agents de type humain (§2).
- Deux agents appris séparément, aussi forts l'un que l'autre, paraissent faibles l'un au milieu
  de l'autre : leurs conventions sont incompatibles (annexe D). Les auteurs évaluent donc sur une
  **population** d'agents, jamais un agent contre des copies de lui-même.

**Pour nous.**
- **Nos arènes sont ce genre de mesure.** L'Expert y joue contre des copies de c130. Une partie de
  nos gains pourrait n'être qu'une adaptation à c130.
- **Ce qui se transpose sans humains : les deux λ.** On modéliserait les adversaires plus près du
  réseau que soi-même. Nos fins simulées, où tous jouent le coup préféré du réseau, en sont une
  forme extrême.

### Blomqvist, Risk (KTH, 2020)

**Le cadre.**
- 1 contre 1 plus un joueur neutre, sur la carte standard de 42 territoires, sans information
  cachée.
- Le tour est découpé en décisions, avec des **quantités grossières** : placer 1 troupe, la
  moitié ou tout, et des nœuds « attaquer ou arrêter », « fortifier ou non ».

**La méthode.**
- **Le réseau.** Entrées plates (troupes par territoire, sans voisinages), 3 couches denses.
- **Les sorties.** Cinq têtes de politique (une par type de décision) et une valeur scalaire.
- **La recherche.** Expert Iteration, MCTS à 10 000 simulations, rollouts au hasard coupés après
  8 tours.

**Les résultats.**
- **La politique aide.** La recherche guidée par le réseau bat le MCTS pur à 72,5 % (tableau 5.4).
- **Puis plus rien ne progresse** : le réseau 1 vaut les réseaux 3 à 6 (tableau 5.2).
- **La valeur dégrade le jeu.** Les agents qui l'utilisent perdent à 76 % et 89 % (tableau 5.3).
  Selon l'auteur, elle agit en classifieur du vainqueur plutôt qu'en estimation (ch. 6).

**Pour nous.**
- **Les symptômes ressemblent aux nôtres, mais pas la cause.** Leur échelle est minuscule
  (environ 100 000 positions, une décision par partie), donc on n'en tire aucun diagnostic.
- **À retenir.** Les quantités grossières (1 / moitié / tout), qui raccourciraient notre
  redéploiement jeton par jeton, et la remarque sur des constantes d'exploration identiques pour
  des décisions très différentes.

### Sampled MuZero (Hubert et al., ICML 2021)

**Le problème.** Chercher quand on ne peut pas énumérer les coups.

**La méthode.**
- On tire K coups d'une loi β (en pratique la politique elle-même).
- Dans la recherche, l'a priori est remplacé par (β̂/β)·π, où β̂ est la fréquence des tirages.
  Avec β = π, l'a priori devient simplement la fréquence des tirages.
- Sans cette correction, un coup probable compte deux fois : plus souvent tiré, et favorisé par
  son a priori (§4–5, ablation de la fig. 13).
- La cible de la politique reste la part des visites à la racine.

**Les résultats.**
- **Au go**, K = 50 coups sur 362 approchent la recherche complète (fig. 2).
- **En contrôle continu**, K = 3 à 10 suffisent.

**Pour nous, si un coup devient un plan de tour tiré en déroulant la politique.**
- **Calculer β exactement** : le produit des probabilités de chaque décision, température, bruit
  et masques compris.
- **Corriger l'a priori comme ci-dessus**, sans compter π deux fois.
- **Se méfier du dédoublonnage.** Retirer les doublons revient à tirer sans remise : la correction
  n'est plus exacte. C'est le cas de Gumbel-Top-k, à creuser.
- **L'apprentissage peut rester décision par décision.** La cible se décompose en entropies
  croisées sur chaque décision des plans, pondérées par les visites.
- **Le dé de renfort coupe le plan** : un plan de conquêtes, un nœud de hasard, puis un plan de
  redéploiement.
- **Ce que la méthode ne fait pas.** Elle corrige la recherche, pas la valeur. Avec des plans,
  chaque feuille devient une fin de tour, et la faiblesse de la valeur pèse encore plus.

### Ce que ces lectures changent pour nos pistes

1. **Un essai nouveau, presque gratuit : l'ancrage de la recherche.** Il découle de piKL.
   - **Les variantes.** Comparer la recette actuelle à un ancrage qui ne faiblit pas avec les
     visites, puis à un λ fixe sans ramener les valeurs entre 0 et 1, à n = 200 et à 16 fois plus.
   - **La prédiction.** Si l'hypothèse « exploitation » est juste, ces variantes gagnent plus à
     16 fois qu'à n = 200, et elles réduisent le gain des fins simulées, puisqu'elles corrigent le
     même défaut.
   - **Des diagnostics sans arène.** Mesurer l'écart entre la politique de la racine et le réseau
     en fonction de n, et l'écart entre la valeur du coup choisi et sa valeur après fin simulée.
2. **Les fins simulées sont la méthode la mieux soutenue.** Elles le sont par la littérature
   multi-actions et par Diplomacy. Piste de coût : une fin simulée jouée par une politique plus
   légère.
3. **Les plans de tour restent sans preuve directe.** Si on y va, Sampled MuZero donne les règles
   pour rester juste, et EMCTS une façon d'engendrer les candidats. Le redéploiement, déterministe,
   est le bon premier terrain. Ce chantier ne règle pas la valeur, il la sollicite davantage.
4. **Nos mesures.** Diplodocus montre qu'à plusieurs joueurs, un agent jugé contre ses propres
   copies peut tromper. Deux remèdes :
   - évaluer contre une **population** (réseau brut, Expert témoin, Expert avec fins simulées) ;
   - regarder les **parties contre les humains** archivées dans `data/games/`.
