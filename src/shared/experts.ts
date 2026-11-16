/**
 * Les variantes de l'IA Expert : une par labo de l'IA (`labN/`, voir labtools/README.md), chacune avec
 * son programme natif (`target/release/labN-expert`, construit par `bun run kernel:build`) et son
 * réseau (`labN/nets/<net>.bin`). Défaut : `DEFAULT_EXPERT_LAB`.
 */
export interface ExpertLab {
  id: string;
  /** Nom affiché. */
  label: string;
  /** Le réseau joué, nom du fichier sans `.bin`. */
  net: string;
  /** Itérations de recherche par décision. */
  iterations: number;
  /** Autres réglages de la recherche, passés au programme (`--search roll=1,mix=0.5`) ; défaut : ceux du labo. */
  search?: string;
  /** Le programme accepte `--threads N` (évaluations sur plusieurs fils, mêmes décisions) : le serveur lui
   * donne les cœurs de la machine moins un. */
  threads?: boolean;
  /** Le réseau connaît les peuples et pouvoirs des extensions (lab17) ; les autres, le jeu de base seulement. */
  expansions?: boolean;
}

export const EXPERT_LABS: readonly ExpertLab[] = [
  // n=200 : ~0,1 à 0,2 s par décision sur mini (lab1/JOURNAL.md)
  { id: 'lab1', label: 'Labo 1', net: 'gen11', iterations: 200 },
  // n=800 : 179 ms en moyenne par décision sur mini (c040, même taille), fins de tour comprises, p95 624 ms
  // (lab2/journal.md)
  { id: 'lab2', label: 'Labo 2', net: 'c060', iterations: 800 },
  // c130, 128x7 : face au c060 de lab2 à n=800, +10,1 ± 1,6 à n=400 et +9,6 ± 1,6 à n=800 (920 donnes,
  // lab3/journal.md) ; n=400 : 63 ms par recherche sur un cœur du PC, contre 72 pour lab2 à n=800
  { id: 'lab3', label: 'Labo 3', net: 'c130', iterations: 400 },
  // c130 et fins simulées (lab7), sauf au redéploiement et à la défense (light=1 : le coup préféré du
  // réseau), jetons des fins simulées posés par le noyau (place=1) : +3,4 ± 1,6 contre c130 à n=400 (920
  // donnes ; sans place +4,7, différence −1,3 ± 2,1) ; évaluations sur les cœurs de la machine moins un :
  // sur le PC à 6 fils, 0,14 s en moyenne par décision à 4 joueurs, 0,21 s à 5 (p95 0,44 s), 4,5 fois moins
  // qu'avant (lab13/journal.md, D10 et D16)
  { id: 'lab13', label: 'Labo 13', net: 'c130', iterations: 100, search: 'roll=1,mix=0.5,light=1,place=1', threads: true },
  // les extensions : la recherche de lab13 et c130 élargi à leurs peuples, pouvoirs et coups, puis 36 pas
  // d'auto-jeu avec extensions (x1-c036) : avec les trois extensions, +35,2 ± 2,3 contre la Difficile (c130w
  // élargi seul : +4,2) et +39,9 ± 3,4 contre c130w ; au jeu de base −2,5 ± 3,2 contre c130w, d'où c130
  // (lab13) par défaut sans extension ; 117 ms en moyenne par décision à 4 joueurs, 170 ms à 5, sur le PC à
  // 6 fils (lab17/journal.md)
  {
    id: 'lab17',
    label: 'Labo 17',
    net: 'x1-c036',
    iterations: 100,
    search: 'roll=1,mix=0.5,light=1,place=1',
    threads: true,
    expansions: true,
  },
];

/** Celui proposé par défaut : le dernier du jeu de base seul (c130, le plus fort sans extension) ; les parties
 * avec extensions vont d'office à `EXPANSIONS_EXPERT_LAB` (`expertFor`). */
export const DEFAULT_EXPERT_LAB = [...EXPERT_LABS].reverse().find((l) => !l.expansions)!.id;

/** Le labo demandé s'il existe, sinon celui par défaut. */
export function expertLab(raw: unknown): string {
  return EXPERT_LABS.some((l) => l.id === raw) ? (raw as string) : DEFAULT_EXPERT_LAB;
}

/** Le dernier labo dont l'Expert connaît les extensions, s'il y en a un. */
export const EXPANSIONS_EXPERT_LAB = [...EXPERT_LABS].reverse().find((l) => l.expansions)?.id;

/** Le labo dont l'Expert joue une partie : celui de la salle ; avec des extensions, s'il ne les connaît pas,
 * le dernier qui les connaît (aucun : `undefined`, l'IA Difficile joue à sa place). */
export function expertFor(lab: string, expansions: readonly string[]): string | undefined {
  if (!expansions.length) return lab;
  return EXPERT_LABS.find((l) => l.id === lab)?.expansions ? lab : EXPANSIONS_EXPERT_LAB;
}
