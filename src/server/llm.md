# SmallMobile pour un LLM : règles et mode d'emploi

Vous jouez à **SmallMobile**, un clone de *Small World*, contre les IA du jeu (l'**Expert** par
défaut, la plus forte). Vous jouez par une API HTTP en JSON : chaque réponse décrit la partie en
texte et donne la liste des coups permis. Ce document contient toutes les règles, puis l'API.

## 1. Le but

Avoir le plus de **pièces** à la fin de la dernière manche. Chacun commence avec 5 pièces. À la fin
de chacun de vos tours, chaque région que vous occupez (avec votre peuple actif comme avec votre
peuple en déclin) rapporte **1 pièce**, plus les bonus des peuples et des pouvoirs. En cas
d'égalité, le joueur qui a le plus de jetons sur la carte l'emporte.

Une partie se joue à 2, 3, 4 ou 5 joueurs, en 10 manches à 2 et 3 joueurs, 9 à 4 joueurs et 8 à
5 joueurs. Dans une manche, chaque joueur joue un tour, toujours dans le même ordre (J1, J2, …).

## 2. Peuples, pouvoirs et marché

Un **peuple** (race) est associé à un **pouvoir** : la combinaison donne un nombre de jetons (ceux
du peuple plus ceux du pouvoir) et des capacités. Le **marché** montre 6 combinaisons, numérotées
de 0 à 5. Prendre la combinaison n coûte **n pièces** : vous posez 1 pièce sur chacune des
combinaisons placées avant elle. Les pièces déjà posées sur la combinaison que vous prenez sont à
vous. Le marché se complète ensuite depuis les pioches, dont l'ordre est caché.

Vous avez au plus un **peuple actif**, et en général un seul peuple **en déclin**.

## 3. Le tour de jeu

1. **Choix** (phase `pick`), si vous n'avez pas de peuple actif : prenez une combinaison du marché.
   Vous enchaînez aussitôt avec les conquêtes, tous vos jetons en main.
2. **Préparation** (automatique) : vos jetons sur la carte reviennent en main, sauf 1 par région.
3. **Conquêtes** (phase `conquest`) : conquérez des régions une par une avec les jetons en main.
   Avant toute conquête, vous pouvez aussi **abandonner** des régions (leurs jetons reviennent en
   main) ou **passer en déclin** (voir plus bas). L'action `endConquest` termine les conquêtes.
4. **Redéploiement** (phase `redeploy`) : répartissez les jetons en main sur vos régions (au moins
   1 par région). Vous pouvez aussi déplacer des jetons d'une région à l'autre (reprendre puis
   poser). Tous les jetons en main doivent être posés avant `endTurn`.
5. **Décompte** (à `endTurn`) : vous gagnez vos pièces, puis le joueur suivant joue.

**Goules en déclin** : si vous avez des Goules en déclin encore présentes, elles jouent d'abord une
étape complète (conquêtes, redéploiement, `endTurn`) avant votre peuple actif ; votre tour continue
ensuite avec le peuple actif.

## 4. Conquérir une région

- **Coût** en jetons : 2, plus 1 par jeton adverse présent, plus 1 si la région est une
  **Montagne**, plus 1 par tanière, forteresse ou campement. Certains peuples et pouvoirs réduisent
  ce coût ; il vaut toujours au moins 1. La liste des coups permis donne le coût exact.
- **Où** : votre première conquête avec un nouveau peuple doit viser une région marquée `entrée`
  (bord de carte, ou côte d'une mer en bordure). Ensuite, seulement des régions adjacentes à celles
  du peuple. Les mers et lacs sont infranchissables (sauf pouvoir Marins). Certaines capacités font
  exception (Semi-hommes, Volants, Souterrains…).
- **Imprenables** : régions avec un héros, un dragon ou un terrier.
- **L'occupant perd 1 jeton** définitivement ; les autres lui sont rendus. Un peuple en déclin perd
  tous ses jetons de la région (sauf les Goules). Les tribus oubliées disparaissent. Les Elfes
  actifs ne perdent aucun jeton.
- **Dernière conquête au dé** (`reinforce`) : s'il vous manque 1 à 3 jetons pour une région, vous
  pouvez tenter le dé de renfort, faces 0, 0, 0, 1, 2, 3. Si vos jetons en main plus le dé
  atteignent le coût, la région est prise avec tous vos jetons en main. Réussi ou non, vos
  conquêtes s'arrêtent là. Chances : il manque 1 jeton → 50 %, 2 → 33 %, 3 → 17 %.

Les jetons rendus à un joueur attaqué : s'il joue juste après vous, il les retrouve à son tour ;
sinon il les replace aussitôt sur ses régions (phase `defend`, qui peut donc vous concerner en
dehors de votre tour : `deploy` puis `endDefense`).

## 5. Le déclin

Au début d'un tour, avant toute conquête (et pas au tour où vous l'avez choisi), vous pouvez mettre
votre peuple actif **en déclin** (`decline`) : il ne garde qu'un jeton par région, perd son pouvoir
(sauf mention contraire), mais ses régions rapportent toujours 1 pièce chacune. Votre tour s'arrête
là, après le décompte. Au tour suivant, vous choisissez un nouveau peuple au marché.

Vous ne gardez qu'**un seul peuple en déclin** : le précédent disparaît de la carte (sauf pouvoir
Spirituels). Le bon moment pour décliner : quand votre peuple est trop étalé pour encore conquérir.

## 6. La carte

Terrains : Ferme, Forêt, Colline, Montagne, Marais, Mer, Lac. Particularités des régions :
`entrée` (première conquête possible), `bord`, `côte` (touche une mer ou un lac), `caverne`,
`source magique`, `mine` (des peuples et pouvoirs en tirent des bonus). Marques posées : tanière,
forteresse, campement (+1 défense chacun), héros, dragon, terrier (région imprenable). Une région
peut aussi être occupée par une **tribu oubliée** (neutre, à conquérir).

## 7. Peuples et pouvoirs

<!-- peuples et pouvoirs -->

## 8. L'API

Toutes les requêtes et réponses sont en JSON. Les adresses sont relatives au serveur du jeu.

### Créer une partie

```
POST /api/llm/games
{"players": 3, "opponents": "expert", "seat": 0, "name": "Claude", "seed": 12345}
```

Tous les champs sont facultatifs :

- `players` : de 2 à 5 (2 par défaut) ;
- `opponents` : un niveau pour tous les adversaires, ou une liste d'un niveau par adversaire, parmi
  `facile`, `normal`, `difficile`, `expert` (`expert` par défaut) ;
- `seat` : votre place, de 0 (premier à jouer) à `players − 1` (au hasard par défaut) ;
- `name` : votre nom dans la partie ;
- `seed` : un entier, pour rejouer la même carte et le même hasard.

La réponse est la vue de la partie (ci-dessous), à votre premier tour : les adversaires qui jouent
avant vous ont déjà joué.

### Voir la partie

```
GET /api/llm/games/{id}
GET /api/llm/games/{id}?format=text     (le texte seul, en Markdown)
```

### Jouer un coup

```
POST /api/llm/games/{id}/actions
{"type": "conquer", "region": 12}
```

Le corps est **une** action. Le serveur la joue, fait jouer les adversaires jusqu'à votre prochaine
décision, et renvoie la nouvelle vue. Un tour demande en général plusieurs actions (choix, plusieurs
conquêtes, redéploiement, `endTurn`) : une requête par action. Ajoutez `?format=text` pour recevoir
le texte seul.

Erreurs : `422` action refusée par les règles (le message dit pourquoi ; rien n'a changé), `409`
ce n'est pas votre tour ou une requête est déjà en cours, `404` partie inconnue, `400` requête mal
formée.

### La vue

- `id`, `seat` (votre place), `round`, `maxRounds`, `phase`, `yourTurn`, `gameOver` ;
- `events` : ce qui s'est passé depuis votre dernier coup, celui-ci compris (résultat des dés,
  actions des adversaires) ;
- `text` : toute la situation en Markdown (joueurs, pièces, marché, carte, coups permis) ;
- `legal` : les coups permis, chacun `{"action": …, "note": …}` ; renvoyez `action` telle quelle ;
- `ranking` : à la fin, le classement (`seat`, `name`, `coins`).

Les pièces de chacun sont données : un joueur attentif les reconstitue à partir des gains annoncés.
L'ordre des pioches et les dés à venir restent cachés.

Dans la carte, chaque ligne donne : `id · nom · terrain · particularités · voisines → occupant`.
L'occupant s'écrit `J2 Orques ×3` (joueur, peuple, nombre de jetons).

### Les actions

| Action | Phase | Effet |
| --- | --- | --- |
| `{"type":"pick","index":i}` | `pick` | prendre la combinaison i du marché (coûte i pièces) |
| `{"type":"skipPick"}` | `pick` | passer, seulement si le marché est vide |
| `{"type":"conquer","region":r}` | `conquest` | conquérir r avec les jetons en main |
| `{"type":"reinforce","region":r}` | `conquest` | dernière conquête au dé sur r |
| `{"type":"berserkRoll"}` | `conquest` | Berserks : dé lancé avant la conquête suivante |
| `{"type":"dragon","region":r}` | `conquest` | Dompteurs de dragon : conquête avec 1 jeton |
| `{"type":"sorcery","region":r}` | `conquest` | Sorciers : remplacer un jeton adverse seul |
| `{"type":"abandon","region":r}` | `conquest` | abandonner r, avant toute conquête |
| `{"type":"decline"}` | `conquest` | passer en déclin, avant toute conquête (finit le tour) |
| `{"type":"endConquest"}` | `conquest` | passer au redéploiement |
| `{"type":"deploy","region":r,"delta":d}` | `redeploy`, `defend` | poser d jetons de la main sur r (d < 0 : en reprendre ; 1 au moins reste) |
| `{"type":"place","tokens":{"3":2,"7":4}}` | `redeploy` | fixer d'un coup les jetons de chaque région du peuple (au moins 1 chacune) ; le reste va en main |
| `{"type":"autoDeploy"}` | `redeploy`, `defend` | répartir automatiquement les jetons en main |
| `{"type":"autoDeploy","all":true}` | `redeploy`, `defend` | tout reprendre (1 jeton reste par région) et tout répartir |
| `{"type":"camp","region":r,"delta":1}` | `redeploy` | Bivouaqueurs : poser (1) ou retirer (−1) un campement |
| `{"type":"fortress","region":r}` | `redeploy` | Fortifiés : une forteresse par tour |
| `{"type":"hero","region":r}` | `redeploy` | Héroïques : poser ou retirer un héros |
| `{"type":"endTurn"}` | `redeploy` | finir le tour ; options `"peace": j` (Diplomates, j = place de l'adversaire) et `"stout": true` (Robustes : déclin après le décompte) |
| `{"type":"endDefense"}` | `defend` | finir de replacer ses jetons rendus |

La liste `legal` (et la section « Coups permis » du texte) donne les coups simples permis à
l'instant, avec leur coût. `place` n'y figure pas (trop de combinaisons) mais reste toujours permis
au redéploiement.

### Exemple

```
curl -s -X POST http://SERVEUR/api/llm/games -d '{"players":2}' | jq -r .text
curl -s -X POST 'http://SERVEUR/api/llm/games/abcd2345/actions?format=text' -d '{"type":"pick","index":0}'
curl -s -X POST 'http://SERVEUR/api/llm/games/abcd2345/actions?format=text' -d '{"type":"conquer","region":4}'
curl -s -X POST 'http://SERVEUR/api/llm/games/abcd2345/actions?format=text' -d '{"type":"endConquest"}'
curl -s -X POST 'http://SERVEUR/api/llm/games/abcd2345/actions?format=text' -d '{"type":"autoDeploy"}'
curl -s -X POST 'http://SERVEUR/api/llm/games/abcd2345/actions?format=text' -d '{"type":"endTurn"}'
```

Conseil : lisez la section « Coups permis » à chaque réponse et choisissez-y votre coup ; en cas de
refus, le message d'erreur explique la règle en cause.
