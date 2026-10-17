# Illustrations mobiles

Collection complète : les 14 peuples, les 20 pouvoirs et les 7 terrains.
Chaque terrain possède une icône pour la légende et les fiches ainsi qu'une
texture pour la carte. Les pouvoirs Sylvestres, des Collines et des Marais
partagent l'icône de leur terrain. La collection contient 38 icônes et
7 textures ; le second lot ajoute 32 fichiers aux 13 du premier.

Les illustrations sont générées avec l'outil intégré imagegen, une génération
par sujet. Les prompts exacts sont enregistrés dans `prompts.json`.
Direction : silhouettes franches, contours foncés, aplats de couleur,
fond transparent, aucun texte ni décor fin. Les fichiers livrés sont des
WebP de 192 × 192 pixels avec transparence, pour un affichage de 20 à 42 px.
Les textures sont des WebP transparents de 512 × 512 pixels, exportés depuis
les générations originales. L'export conserve le canal alpha et les proportions.

Le marché, les fiches des peuples, la légende et les règles utilisent le
composant `ArtIcon`. Les noms et les règles restent visibles en texte.
Les petites pastilles des jetons et les symboles d'action gardent leurs
repères habituels. La carte applique les textures à toute la surface des
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
