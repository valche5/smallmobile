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
}

export const EXPERT_LABS: readonly ExpertLab[] = [
  // n=200 : ~0,1 à 0,2 s par décision sur mini (lab1/JOURNAL.md)
  { id: 'lab1', label: 'Labo 1', net: 'gen11', iterations: 200 },
  // lab2 revient ici avec le premier réseau de sa boucle (lab2/journal.md)
];

export const DEFAULT_EXPERT_LAB = EXPERT_LABS[EXPERT_LABS.length - 1]!.id;

/** Le labo demandé s'il existe, sinon celui par défaut. */
export function expertLab(raw: unknown): string {
  return EXPERT_LABS.some((l) => l.id === raw) ? (raw as string) : DEFAULT_EXPERT_LAB;
}
