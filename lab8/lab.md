# Lab8 : le plan de tour en un passage du réseau

Base : copie de **lab7** au commit `adb2c72` (c130, fins simulées et moteurs Vulkan et CUDA
compris). Lab7 est figé. Règles communes : [labtools/README.md](../labtools/README.md). Lectures :
[labtools/biblio.md](../labtools/biblio.md). Résultats : [journal.md](journal.md).

## Pourquoi

- Les fins simulées de lab7 (`roll=1`, le réseau joue pour tous jusqu'au prochain tour du joueur)
  battent c130 de +5,4 ± 1,5 points à n = 100 et de +8,9 ± 2,3 à n = 400. C'est le seul progrès net
  depuis lab3. Elles coûtent cher : environ 29 évaluations du réseau par fin simulée, une par
  conquête ou par jeton, soit ~12 fois le calcul de n = 400 à n = 100.
- Dans la littérature (Diplomacy, jeux à plusieurs actions par tour), les tours se jouent et se
  cherchent **entiers**, et les fins simulées courtes sont gardées (biblio : SearchBot, DORA, Baier &
  Cowling 2018).
- Un réseau qui sort **tout un plan de tour en un passage** rendrait une fin simulée environ cinq
  fois moins chère (de ~29 à ~5 évaluations). Les mêmes plans seraient plus tard les candidats d'une
  recherche sur des tours entiers (lab9 éventuel).

## Ce que le réseau sortirait (proposition, à valider avant d'écrire le code)

Le corps du réseau ne change pas (c130 : graphe 128 × 7, mêmes entrées). On ajoute trois sorties,
lues sur les descriptions de région et de partie que la politique actuelle utilise déjà :

| Sortie | Lue sur la position | Sens | Cible d'apprentissage |
| --- | --- | --- | --- |
| **conquête normale** : un logit par région | début des conquêtes du joueur | « je prends cette région avec mes jetons pendant ce tour » | 1 si la région a été prise sans dé pendant le tour, sinon 0 (entropie croisée binaire) |
| **dé de renfort** : un logit par région, plus un logit « aucun » lu sur la partie | début des conquêtes du joueur | la région où tenter le dé à la fin, ou aucune | la région du `Reinforce` joué, **tenté**, réussi ou non ; « aucun » sinon (entropie croisée sur une softmax) |
| **jetons** : un logit par région | début du redéploiement, après `Lift` ; défense | part des jetons libres posés sur cette région (softmax sur les régions du peuple) | la répartition réellement jouée |

Le dé de renfort est forcément la dernière conquête : il n'est permis que sur une région que les
jetons en main ne suffisent pas à prendre (`TargetKind::Die`, il manque 1 à 3 jetons), et le noyau
passe au redéploiement dès qu'il est lancé, réussi ou non (`Game::reinforce`). Une sortie à part
dit lequel des plans du tour finit au dé ; sans elle, l'exécuteur ne saurait pas s'il doit payer
une région avec ses jetons ou garder les jetons pour la tenter au dé. Les cibles sont des
décisions, pas des issues : un dé raté reste la cible « dé » de sa région, sans le hasard du dé.

Deux lectures par tour plutôt qu'un seul plan : le dé de renfort, les Berserks et l'issue des
conquêtes décident des régions à redéployer. La seconde lecture coûte une évaluation.

**L'exécuteur** (moteur, sans réseau) transforme les sorties en coups légaux :
- conquêtes normales : tant qu'une région de probabilité ≥ 0,5, autre que la région du dé, est une
  cible normale, prendre la plus probable ; dragon et sorcellerie si la politique actuelle les
  préfère sur cette région ;
- puis le dé : si la région du dé (le plus fort logit, « aucun » compris) est alors une cible au dé,
  `Reinforce` ; si elle est devenue une cible normale (plus de jetons que prévu), la prendre
  normalement ; sinon fin des conquêtes ;
- redéploiement et défense : un jeton par région, le reste au prorata des parts (plus forts restes) ;
- Berserks (pas de dé de renfort ; un dé avant chaque conquête, qui baisse son coût, au minimum 1,
  et ne coûte rien : `Game::berserk_roll`) : la sortie « conquête normale » sert de liste de
  priorités, lue une fois au début des conquêtes. L'exécuteur lance le dé, prend la région de
  probabilité ≥ 0,5 la plus probable que le coût réduit permet, puis recommence ; il s'arrête quand
  aucune région de la liste n'est payable. Leur cible reste les régions prises, donc dépendante des
  dés (tours peu nombreux) ; mesure à part (ci-dessous). Si l'accord est mauvais, ces tours
  reviennent à la politique coup par coup ;
- décisions rares (choix du peuple, déclin, héros, forteresse, campements, paix, Robustes) : la
  politique actuelle, coup par coup, comme aujourd'hui.

Une fin simulée « par plans » (`roll=1,plan=1`) coûterait par joueur : 1 évaluation (plan
d'attaque) + 1 (jetons) + celles des décisions rares, contre ~29 aujourd'hui.

## Données

- Pas de fenêtre de positions conservée (bucket AWS supprimé, machines vast.ai détruites).
- Parties neuves jouées par c130 **sans bruit** à n = 400, comme le champ des arènes : les plans
  appris doivent être ceux du joueur que la fin simulée imite, pas ceux du selfplay bruité (lab7,
  A3 : le bruit coûte ~6 points). Un enregistrement par début de conquêtes, par début de
  redéploiement et par défense, avec sa cible. Les positions ordinaires et la valeur au prochain
  tour du joueur sont écrites en même temps (piste B).
- Calcul sur le PC fixe de l'utilisateur (i7-12700KF, RX 9070 XT, RADV, `--device vulkan` ; le
  shader bf16 a été écrit et vérifié sur cette carte, il ne calcule faux que sur NVIDIA : lab5).

## Mesures, fixées avant tout calcul

1. **Hors ligne** (positions mises de côté, par phase et par nombre de joueurs) :
   - conquêtes normales : précision, rappel et F1 de l'ensemble des régions prises ;
   - dé de renfort : part des tours où la région du dé (ou « aucun ») est celle jouée ;
   - tours de Berserks à part : mêmes mesures que les conquêtes normales, et accord de
     l'exécuteur avec la politique coup par coup sur les mêmes dés ;
   - jetons : écart moyen, en jetons, à la répartition jouée ;
   - accord de l'exécuteur avec la politique jouée coup par coup sans bruit, sur les mêmes
     positions : part des tours aux mêmes régions prises, écart en jetons.
2. **Coût** (`lab8 speed`) : évaluations et millisecondes par fin simulée, `roll=1` contre
   `roll=1,plan=1`.
3. **Arènes** contre un champ de c130 à n = 400, 2 à 5 joueurs, `labtools/score.py` :
   - `n=100,roll=1,mix=0.5,plan=1`, 920 donnes, graine 72002 (les donnes de la confirmation de
     lab7 : +5,4 ± 1,5 sans plans) ;
   - si le coût le permet, plus de simulations ou `roll=2` à calcul égal à `n=100,roll=1`.
   - Succès : un score au moins égal à celui de `roll=1` moins 1,5 point, pour au moins trois fois
     moins de calcul par décision.

## Piste B : la valeur au prochain tour (en attente)

Déjà essayée dans lab4 (`train.py --td 0.5`, boucle w2 de 20 pas depuis c130, avec `win=1`) : rien
de mesurable (`lab4/journal.md`, « Cibles de valeur à court terme » et « Essai w2 »). Ne la relancer
que sous une forme nouvelle, à décider : par exemple λ = 1 comme DORA, ou une mesure hors ligne
(la valeur apprise sur cette cible se rapproche-t-elle de la valeur après fin simulée ?), avec les
données ci-dessus.

## Commandes

Sur mini, processeur seul, pour les tests (Debian n'a pas le glslc du shader bf16) :

```sh
timeout 1500 cargo test -p smallmobile-lab8 -p smallmobile-brain8 --no-default-features
timeout 900 cargo clippy -p smallmobile-lab8 -p smallmobile-brain8 --all-targets --no-default-features -- -D warnings
```
