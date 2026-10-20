export type Terrain = 'farmland' | 'forest' | 'hill' | 'mountain' | 'swamp' | 'sea' | 'lake';

export type RaceId =
  | 'amazons'
  | 'dwarves'
  | 'elves'
  | 'ghouls'
  | 'giants'
  | 'halflings'
  | 'humans'
  | 'orcs'
  | 'ratmen'
  | 'skeletons'
  | 'sorcerers'
  | 'tritons'
  | 'trolls'
  | 'wizards';

export type PowerId =
  | 'alchemist'
  | 'berserk'
  | 'bivouacking'
  | 'commando'
  | 'diplomat'
  | 'dragonMaster'
  | 'flying'
  | 'forest'
  | 'fortified'
  | 'heroic'
  | 'hill'
  | 'merchant'
  | 'mounted'
  | 'pillaging'
  | 'seafaring'
  | 'spirit'
  | 'stout'
  | 'swamp'
  | 'underworld'
  | 'wealthy';

/** `expert` : l'IA du labo (recherche guidée par un réseau), jouée par le serveur. */
export type AiLevel = 'facile' | 'normal' | 'difficile' | 'expert';

export interface RegionDef {
  id: number;
  name: string;
  terrain: Terrain;
  cavern: boolean;
  magic: boolean;
  mine: boolean;
  lostTribe: boolean;
  /** Touche le bord de la carte. */
  border: boolean;
  /** Peut servir de première conquête (bord de carte ou côte d'une mer en bordure). */
  entry: boolean;
  /** Adjacente à une mer ou un lac. */
  coastal: boolean;
  adj: number[];
  path: string;
  cx: number;
  cy: number;
}

export interface MapDef {
  id: string;
  name: string;
  seed: number;
  width: number;
  height: number;
  players: number;
  rounds: number;
  regions: RegionDef[];
}

export interface RegionState {
  /** Peuple qui occupe la région, 'lost' = tribu oubliée. */
  race: RaceId | 'lost' | null;
  tokens: number;
  lair: boolean;
  fortress: boolean;
  camps: number;
  hero: boolean;
  dragon: boolean;
  hole: boolean;
}

export interface RaceState {
  race: RaceId;
  power: PowerId;
  owner: number;
  declined: boolean;
  /** Jetons en main (pas encore posés sur la carte). */
  hand: number;
  holesPlaced: number;
  pickedRound: number;
}

export interface Player {
  id: number;
  name: string;
  color: string;
  kind: 'human' | 'ai';
  ai: AiLevel;
  coins: number;
  active: RaceId | null;
  /** Peuples en déclin, du plus récent au plus ancien. */
  declined: RaceId[];
  lastIncome: number;
}

export interface ComboSlot {
  race: RaceId;
  power: PowerId;
  coins: number;
}

export type Phase = 'pick' | 'conquest' | 'redeploy' | 'defend' | 'gameOver';

export interface TurnState {
  /** 'ghouls' : les Goules en déclin jouent avant le peuple actif. */
  stage: 'ghouls' | 'active';
  /** Le peuple qui agit actuellement. */
  army: RaceId | null;
  conquests: number;
  /** Régions non vides conquises par le peuple actif ce tour (Orques, Pillards, Squelettes). */
  nonEmpty: number;
  /** Joueurs dont le peuple actif a été attaqué ce tour (Diplomates). */
  attacked: number[];
  /** Joueurs déjà ensorcelés ce tour (Sorciers). */
  sorcery: number[];
  dragonUsed: boolean;
  fortressRegion: number | null;
  berserkDie: number | null;
  dieRolled: boolean;
  declined: boolean;
  picked: boolean;
  amazonBonus: number;
  /** Jetons ramassés en début de tour, par région (pour restaurer les Goules si elles passent en déclin). */
  readied: Record<number, number>;
}

export interface DefenseTask {
  player: number;
  race: RaceId;
}

export interface Peace {
  diplomat: number;
  target: number;
}

export type LogKind =
  | 'info'
  | 'turn'
  | 'pick'
  | 'conquer'
  | 'die'
  | 'decline'
  | 'abandon'
  | 'score'
  | 'defense'
  | 'power'
  | 'end';

export interface LogEntry {
  seq: number;
  round: number;
  player: number | null;
  kind: LogKind;
  text: string;
  /** Version courte pour l'affichage en direct, sans le nom du joueur (« prend « Bois de Fer » »). */
  short?: string;
  /** Conséquence principale, s'il y en a une (« Bob perd 1 jeton »). */
  detail?: string;
  region?: number;
  /** Joueur qui a subi l'action, pour les récapitulatifs. */
  victim?: number;
  /** Données structurées, pour raconter l'action sans analyser le texte (absentes des anciennes sauvegardes). */
  how?: 'conquer' | 'dragon' | 'sorcery' | 'die';
  /** Jetons posés (conquête), en main (turn), du peuple (pick). */
  tokens?: number;
  /** Jetons perdus par la victime (0 = Elfes ; tous = peuple en déclin). */
  lost?: number;
  /** Jetons rendus à la victime, à replacer. */
  returned?: number;
  victimRace?: RaceId | 'lost' | null;
  /** Peuple concerné (pick, decline, turn ; paix : le peuple protégé). */
  race?: RaceId;
  power?: PowerId;
  /** Déclin : anciens peuples qui disparaissent. */
  removed?: RaceId[];
  /** Décompte : pièces gagnées. */
  gain?: number;
  die?: { value: number; need: number; success: boolean | null };
  /** Jamais montré en direct (Amazones rendues, Squelettes levés, fin des Goules). */
  quiet?: boolean;
}

export interface DieEvent {
  seq: number;
  player: number;
  value: number;
  need: number;
  success: boolean | null;
  region: number | null;
  berserk: boolean;
}

export interface GameState {
  version: 1;
  map: MapDef;
  players: Player[];
  races: Partial<Record<RaceId, RaceState>>;
  regions: RegionState[];
  market: ComboSlot[];
  raceDeck: RaceId[];
  powerDeck: PowerId[];
  /** Cartes du fond des pioches dont l'ordre est public (rendues sous la pioche par un déclin). */
  raceTail: number;
  powerTail: number;
  round: number;
  maxRounds: number;
  current: number;
  phase: Phase;
  turn: TurnState;
  defense: DefenseTask[];
  peace: Peace[];
  log: LogEntry[];
  /** De quoi rejouer la partie à l'identique (graines et actions) ; gardé par le serveur seulement. */
  record?: GameRecord;
  seq: number;
  rng: number;
  lastDie: DieEvent | null;
  /** Dernière région touchée par une action (pour l'animation). */
  lastRegion: number | null;
}

/** Une partie enregistrée : `createGame` avec ces graines, puis ces actions, redonne la même partie. */
export interface GameRecord {
  mapSeed: number;
  seed: number;
  /** Début de la partie (ms depuis 1970). */
  startedAt: number;
  moves: RecordedMove[];
}

export interface RecordedMove {
  /** Le joueur qui agit. */
  p: number;
  a: Action;
  /** Instant de l'action (ms depuis 1970). */
  t: number;
  /** Première action d'une décision d'IA : ce qu'elle a coûté. */
  ai?: AiThought;
}

/** Une décision d'IA, pour mesurer son temps de calcul pendant les vraies parties. */
export interface AiThought {
  level: AiLevel;
  /** Durée de la décision, attente comprise (ms). */
  ms: number;
  /** Expert : actions jouées pour ce coup, itérations, chance de gagner estimée, temps de calcul seul (ms). */
  actions?: number;
  iterations?: number;
  value?: number;
  computeMs?: number;
  /** Expert : son réseau (nom du fichier sans `.bin`, par exemple `gen9`). */
  net?: string;
  /** L'Expert n'a pas répondu : l'IA Difficile a joué à sa place. */
  fallback?: boolean;
}

export type Action =
  | { type: 'pick'; index: number }
  | { type: 'conquer'; region: number }
  | { type: 'reinforce'; region: number }
  | { type: 'berserkRoll' }
  | { type: 'dragon'; region: number }
  | { type: 'sorcery'; region: number }
  | { type: 'abandon'; region: number }
  | { type: 'decline' }
  | { type: 'endConquest' }
  | { type: 'deploy'; region: number; delta: number }
  /** `all` : reprend d'abord tous les jetons des régions (un seul y reste) et replace tout. */
  | { type: 'autoDeploy'; all?: boolean }
  /** Fin de tour : fixe le nombre de jetons de chaque région du peuple (au moins 1) ; le reste va en main. */
  | { type: 'place'; tokens: Record<number, number> }
  | { type: 'camp'; region: number; delta: number }
  | { type: 'fortress'; region: number }
  | { type: 'hero'; region: number }
  | { type: 'endTurn'; peace?: number | null; stout?: boolean }
  | { type: 'endDefense' }
  | { type: 'skipPick' };
