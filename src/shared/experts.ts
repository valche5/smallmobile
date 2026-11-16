/**
 * L'IA Expert du jeu : l'Expert d'un labo de l'IA (`labN/`, voir labtools/README.md), avec son programme natif
 * (`target/release/labN-expert`, construit par `bun run kernel:build`) et son réseau (`labN/nets/<net>.bin`).
 */
export interface ExpertLab {
  id: string;
  /** Le réseau joué, nom du fichier sans `.bin`. */
  net: string;
  /** Itérations de recherche par décision. */
  iterations: number;
  /** Autres réglages de la recherche, passés au programme (`--search roll=1,mix=0.5`) ; défaut : ceux du labo. */
  search?: string;
  /** Le programme accepte `--threads N` (évaluations sur plusieurs fils, mêmes décisions) : le serveur lui
   * donne les cœurs de la machine moins un. */
  threads?: boolean;
}

/**
 * lab17 : la recherche de lab13 (fins simulées, light=1, place=1) et c130 élargi aux peuples, pouvoirs et coups
 * des extensions, puis 36 pas d'auto-jeu avec extensions (x1-c036). Avec les trois extensions, +35,2 ± 2,3
 * contre la Difficile et +39,9 ± 3,4 contre c130w ; au jeu de base, à peine moins fort que c130 de lab13
 * (−2,5 ± 3,2 contre c130w), qu'il remplace pour n'avoir qu'un Expert. 117 ms en moyenne par décision à
 * 4 joueurs, 170 ms à 5, sur le PC à 6 fils (lab17/journal.md).
 */
export const EXPERT_LAB: ExpertLab = {
  id: 'lab17',
  net: 'x1-c036',
  iterations: 100,
  search: 'roll=1,mix=0.5,light=1,place=1',
  threads: true,
};
