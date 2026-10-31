/**
 * Les variantes de l'IA Expert : une par labo de l'IA (`labN/`, voir labtools/README.md), chacune avec
 * son programme natif (`target/release/labN-expert`, construit par `bun run kernel:build`) et son
 * réseau (`labN/nets/<net>.bin`). La dernière est proposée par défaut.
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
];

export const DEFAULT_EXPERT_LAB = EXPERT_LABS[EXPERT_LABS.length - 1]!.id;

/** Le labo demandé s'il existe, sinon celui par défaut. */
export function expertLab(raw: unknown): string {
  return EXPERT_LABS.some((l) => l.id === raw) ? (raw as string) : DEFAULT_EXPERT_LAB;
}
