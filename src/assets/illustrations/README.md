# Illustrations mobiles

Collection du jeu de base et des trois extensions (Même pas peur, Maauuudits,
Honneur aux Dames) : les 24 peuples, les 32 pouvoirs et les 7 terrains.
Chaque terrain possède une icône pour la légende et les fiches ainsi qu'une
texture pour la carte. Les pouvoirs Sylvestres, des Collines et des Marais
partagent l'icône de leur terrain. Souterrains conserve l'emoji de la carte ;
son ancienne illustration reste archivée ici. La collection contient 60 icônes
et 7 textures ; les extensions ajoutent 10 portraits et 12 icônes de pouvoirs.

Les illustrations sont générées avec l'outil intégré imagegen, une génération
par sujet. Les prompts exacts sont enregistrés dans `prompts.json`.
Les portraits des extensions reprennent les repères des scans officiels fournis
(coiffes, ailes, masques, fiole, etc.) dans le style simplifié de la collection.
Direction : silhouettes franches, contours foncés, aplats de couleur,
fond transparent, aucun texte ni décor fin. Les fichiers livrés sont des
WebP de 192 × 192 pixels avec transparence, pour un affichage de 20 à 42 px.
Les textures sont des WebP transparents de 512 × 512 pixels, exportés depuis
les générations originales. L'export conserve le canal alpha et les proportions.

Le marché, les fiches des peuples, la légende et les règles utilisent le
composant `ArtIcon`. Les noms et les règles restent visibles en texte.
Les pastilles des jetons et leurs animations (déchirure, retour en main)
utilisent le même portrait que le marché et les fiches, sélectionné par
l'identifiant du peuple. Les portraits en déclin sont désaturés ; la couleur
du joueur et le nombre de jetons restent visibles. Les symboles d'action
gardent leurs repères habituels. La carte applique les textures à toute la surface des
sept terrains. Chaque région reçoit une tuile carrée centrée sur elle, de la
taille de la plus grande région texturée de la carte : l'image n'est jamais
étirée, le grain est le même partout, et aucune répétition ne tombe à
l'intérieur d'une région (les sillons des fermes n'ont pas de coutures).
Les couleurs de base restent visibles ; les textures sont atténuées à 30 %
pour garder les coûts lisibles.

Ajouter un sujet : générer une illustration dans le même style, exporter
un WebP transparent de 192 px, puis ajouter son import et son identifiant
à `src/ui/art.ts`. Les registres sont exhaustifs : TypeScript signale un peuple,
un pouvoir ou un terrain sans illustration. Pour un terrain, ajouter aussi sa
texture à `TERRAIN_TEXTURES`.
Les étiquettes portent l'identifiant de leur peuple ou de leur pouvoir :
l'illustration n'est jamais déduite de l'emoji, que deux sujets peuvent
partager (Humains et Fermes). Aucun changement des règles ni des sauvegardes.
