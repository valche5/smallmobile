import type { AiLevel, PowerId, RaceId, Terrain } from './types';

export interface RaceDef {
  id: RaceId;
  name: string;
  tokens: number;
  max: number;
  icon: string;
  /** Étiquette du bandeau et de la ligne du haut : 18 caractères au plus. */
  tag: string;
  short: string;
  desc: string;
  /** Ce qui continue de s'appliquer quand le peuple est en déclin. */
  inDecline?: string;
  /** Ce que ce peuple peut vous faire, vu de l'adversaire (52 caractères au plus). */
  foe: string;
  kind: EffectKind;
}

/** Nature d'un effet : attaque, défense, revenu, spécial. */
export type EffectKind = 'attaque' | 'défense' | 'revenu' | 'spécial';

/** Ordre d'affichage des étiquettes, et libellé de chaque nature. */
export const KIND_ORDER: EffectKind[] = ['attaque', 'défense', 'spécial', 'revenu'];
export const KIND_LABEL: Record<EffectKind, string> = { attaque: 'Attaque', défense: 'Défense', spécial: 'Spécial', revenu: 'Revenu' };

export interface PowerDef {
  id: PowerId;
  name: string;
  tokens: number;
  icon: string;
  /** Étiquette du bandeau et de la ligne du haut : 18 caractères au plus. */
  tag: string;
  short: string;
  desc: string;
  /** Ce que ce pouvoir permet contre vous, vu de l'adversaire (52 caractères au plus). */
  foe: string;
  kind: EffectKind;
}

export const RACES: Record<RaceId, RaceDef> = {
  amazons: {
    id: 'amazons',
    name: 'Amazones',
    tokens: 6,
    max: 15,
    icon: '🏹',
    tag: '+4 pour attaquer',
    short: '+4 jetons pour conquérir',
    desc: "Reçoivent 4 jetons supplémentaires au début de chaque tour, utilisables uniquement pour conquérir : ils sont retirés au moment du redéploiement.",
    foe: 'Attaquent avec 4 jetons de plus chaque tour',
    kind: 'attaque',
  },
  dwarves: {
    id: 'dwarves',
    name: 'Nains',
    tokens: 3,
    max: 8,
    icon: '🪓',
    tag: '+1 💰 par Mine',
    short: '+1 💰 par Mine, même en déclin',
    desc: 'Chaque région avec une Mine rapporte 1 pièce de plus à la fin du tour, même lorsque les Nains sont en déclin.',
    inDecline: '+1 💰 par Mine, même en déclin',
    foe: '+1 💰 par Mine, même en déclin',
    kind: 'revenu',
  },
  elves: {
    id: 'elves',
    name: 'Elfes',
    tokens: 6,
    max: 11,
    icon: '🧝',
    tag: 'aucune perte',
    short: 'Ne perdent pas de jeton quand on les attaque',
    desc: "Quand une de leurs régions est conquise, les Elfes récupèrent tous leurs jetons au lieu d'en perdre un (tant qu'ils sont actifs).",
    foe: 'Ne perdent aucun jeton quand vous les attaquez',
    kind: 'défense',
  },
  ghouls: {
    id: 'ghouls',
    name: 'Goules',
    tokens: 5,
    max: 10,
    icon: '🧟',
    tag: 'actives en déclin',
    short: 'Continuent de conquérir en déclin',
    desc: "En déclin, les Goules gardent tous leurs jetons sur la carte et peuvent encore conquérir à chaque tour, avant votre peuple actif.",
    inDecline: 'Gardent leurs jetons et conquièrent encore à chaque tour',
    foe: 'Attaquent encore, même en déclin',
    kind: 'attaque',
  },
  giants: {
    id: 'giants',
    name: 'Géants',
    tokens: 6,
    max: 11,
    icon: '🗿',
    tag: '−1 près Montagnes',
    short: '−1 jeton près de leurs Montagnes',
    desc: "Conquérir une région adjacente à une Montagne qu'ils occupent leur coûte 1 jeton de moins (minimum 1).",
    foe: '−1 pour attaquer autour de leurs Montagnes',
    kind: 'attaque',
  },
  halflings: {
    id: 'halflings',
    name: 'Semi-hommes',
    tokens: 6,
    max: 11,
    icon: '🍄',
    tag: 'terriers protégés',
    short: 'Entrent partout, 2 terriers imprenables',
    desc: "Leur première conquête peut se faire n'importe où. Leurs 2 premières régions conquises reçoivent un terrier qui les rend imprenables (perdus en déclin).",
    foe: "Surgissent n'importe où · 2 terriers imprenables",
    kind: 'défense',
  },
  humans: {
    id: 'humans',
    name: 'Humains',
    tokens: 5,
    max: 10,
    icon: '🌾',
    tag: '+1 💰 par Ferme',
    short: '+1 💰 par Ferme',
    desc: 'Chaque Ferme occupée rapporte 1 pièce de plus à la fin du tour (tant que les Humains sont actifs).',
    foe: '+1 💰 par Ferme',
    kind: 'revenu',
  },
  orcs: {
    id: 'orcs',
    name: 'Orques',
    tokens: 5,
    max: 10,
    icon: '👹',
    tag: '+1 💰 par victime',
    short: '+1 💰 par région non vide conquise',
    desc: 'Chaque région non vide conquise ce tour rapporte 1 pièce de plus (tant que les Orques sont actifs).',
    foe: '+1 💰 pour chaque région occupée qu\'ils prennent',
    kind: 'attaque',
  },
  ratmen: {
    id: 'ratmen',
    name: 'Hommes-rats',
    tokens: 8,
    max: 13,
    icon: '🐀',
    tag: 'très nombreux',
    short: 'Aucun pouvoir… mais très nombreux',
    desc: 'Pas de pouvoir racial, mais 8 jetons de départ.',
    foe: 'Aucun pouvoir, mais très nombreux',
    kind: 'attaque',
  },
  skeletons: {
    id: 'skeletons',
    name: 'Squelettes',
    tokens: 6,
    max: 18,
    icon: '💀',
    tag: '+1 jeton/2 prises',
    short: '+1 jeton pour 2 régions non vides conquises',
    desc: 'Au redéploiement, reçoivent 1 nouveau jeton pour chaque paire de régions non vides conquises ce tour.',
    foe: 'Grossissent en prenant des régions occupées',
    kind: 'attaque',
  },
  sorcerers: {
    id: 'sorcerers',
    name: 'Sorciers',
    tokens: 5,
    max: 18,
    icon: '🔮',
    tag: 'volent jeton isolé',
    short: 'Ensorcellent les jetons isolés',
    desc: "Une fois par adversaire et par tour, peuvent remplacer le jeton unique d'un peuple actif adverse (dans une région adjacente) par un de leurs propres jetons, gratuitement.",
    foe: 'Prennent gratis vos régions gardées par 1 jeton',
    kind: 'attaque',
  },
  tritons: {
    id: 'tritons',
    name: 'Tritons',
    tokens: 6,
    max: 11,
    icon: '🔱',
    tag: '−1 sur les côtes',
    short: '−1 jeton sur les côtes',
    desc: 'Conquérir une région bordant une mer ou un lac leur coûte 1 jeton de moins (minimum 1).',
    foe: '−1 pour attaquer les côtes',
    kind: 'attaque',
  },
  trolls: {
    id: 'trolls',
    name: 'Trolls',
    tokens: 5,
    max: 10,
    icon: '🧌',
    tag: '+1 défense partout',
    short: 'Tanières : +1 défense, même en déclin',
    desc: "Chaque région occupée par les Trolls reçoit une tanière qui ajoute 1 à sa défense, y compris en déclin.",
    inDecline: 'Les tanières restent : +1 défense',
    foe: '+1 défense partout (tanières), même en déclin',
    kind: 'défense',
  },
  wizards: {
    id: 'wizards',
    name: 'Mages',
    tokens: 5,
    max: 10,
    icon: '🧙',
    tag: '+1 💰 par Source',
    short: '+1 💰 par Source magique',
    desc: 'Chaque Source magique occupée rapporte 1 pièce de plus à la fin du tour (tant que les Mages sont actifs).',
    foe: '+1 💰 par Source magique',
    kind: 'revenu',
  },
};

export const POWERS: Record<PowerId, PowerDef> = {
  alchemist: {
    id: 'alchemist',
    name: 'Alchimistes',
    tokens: 4,
    icon: '⚗️',
    tag: '+2 💰 par tour',
    short: '+2 💰 à chaque tour',
    desc: "Gagnent 2 pièces supplémentaires à la fin de chaque tour tant qu'ils sont actifs.",
    foe: '+2 💰 à chaque tour',
    kind: 'revenu',
  },
  berserk: {
    id: 'berserk',
    name: 'Berserks',
    tokens: 4,
    icon: '🎲',
    tag: 'dé avant attaque',
    short: 'Dé de renfort avant chaque conquête',
    desc: "Peuvent lancer le dé de renfort avant chaque conquête (et pas seulement la dernière). Le résultat réduit le coût de la conquête suivante.",
    foe: "Dé avant chaque attaque : jusqu'à −3 jetons",
    kind: 'attaque',
  },
  bivouacking: {
    id: 'bivouacking',
    name: 'Bivouaqueurs',
    tokens: 5,
    icon: '⛺',
    tag: '5 camps +1 défense',
    short: '5 campements (+1 défense chacun)',
    desc: 'Disposent de 5 campements à placer au redéploiement : chacun ajoute 1 à la défense de sa région. Perdus en déclin.',
    foe: '5 campements : +1 défense chacun',
    kind: 'défense',
  },
  commando: {
    id: 'commando',
    name: 'Commandos',
    tokens: 4,
    icon: '🗡️',
    tag: '−1 par attaque',
    short: 'Toute conquête coûte 1 jeton de moins',
    desc: 'Chaque conquête coûte 1 jeton de moins (minimum 1).',
    foe: 'Toutes leurs attaques coûtent 1 jeton de moins',
    kind: 'attaque',
  },
  diplomat: {
    id: 'diplomat',
    name: 'Diplomates',
    tokens: 5,
    icon: '🕊️',
    tag: 'paix forcée',
    short: 'Paix forcée avec un adversaire',
    desc: "En fin de tour, choisissez un adversaire dont vous n'avez pas attaqué le peuple actif : il ne pourra pas attaquer votre peuple actif pendant son prochain tour.",
    foe: 'Peuvent vous interdire de les attaquer 1 tour',
    kind: 'défense',
  },
  dragonMaster: {
    id: 'dragonMaster',
    name: 'Dompteurs de dragon',
    tokens: 5,
    icon: '🐉',
    tag: 'dragon : prise à 1',
    short: 'Conquête à 1 jeton + région imprenable',
    desc: "Une fois par tour, conquièrent une région avec 1 seul jeton quelle que soit sa défense. La région du dragon est imprenable jusqu'à ce qu'il soit déplacé.",
    foe: '1 seul jeton suffit (dragon), région imprenable',
    kind: 'attaque',
  },
  flying: {
    id: 'flying',
    name: 'Volants',
    tokens: 5,
    icon: '🦅',
    tag: 'attaquent partout',
    short: 'Attaquent sans être adjacents',
    desc: "Peuvent conquérir n'importe quelle région terrestre de la carte, même non adjacente.",
    foe: 'Attaquent partout : la distance ne protège pas',
    kind: 'attaque',
  },
  forest: {
    id: 'forest',
    name: 'Sylvestres',
    tokens: 4,
    icon: '🌲',
    tag: '+1 💰 par Forêt',
    short: '+1 💰 par Forêt',
    desc: 'Chaque Forêt occupée rapporte 1 pièce de plus (tant que le peuple est actif).',
    foe: '+1 💰 par Forêt',
    kind: 'revenu',
  },
  fortified: {
    id: 'fortified',
    name: 'Fortifiés',
    tokens: 3,
    icon: '🏰',
    tag: 'forteresse +1 déf',
    short: '1 forteresse par tour (+1 💰, +1 défense)',
    desc: "Une fois par tour, construisent une forteresse (6 max) : +1 défense, et +1 pièce par forteresse tant qu'ils sont actifs.",
    foe: 'Forteresses : +1 défense et +1 💰 chacune',
    kind: 'défense',
  },
  heroic: {
    id: 'heroic',
    name: 'Héroïques',
    tokens: 5,
    icon: '🦸',
    tag: 'héros imprenables',
    short: '2 héros rendent 2 régions imprenables',
    desc: "Au redéploiement, placez vos 2 héros : leurs régions sont imprenables. Perdus en déclin.",
    foe: '2 régions imprenables (héros)',
    kind: 'défense',
  },
  hill: {
    id: 'hill',
    name: 'des Collines',
    tokens: 4,
    icon: '🏞️',
    tag: '+1 💰 par Colline',
    short: '+1 💰 par Colline',
    desc: 'Chaque Colline occupée rapporte 1 pièce de plus (tant que le peuple est actif).',
    foe: '+1 💰 par Colline',
    kind: 'revenu',
  },
  merchant: {
    id: 'merchant',
    name: 'Marchands',
    tokens: 2,
    icon: '⚖️',
    tag: '+1 💰 par région',
    short: '+1 💰 par région',
    desc: 'Chaque région occupée par ce peuple rapporte 1 pièce de plus (tant que le peuple est actif).',
    foe: '+1 💰 par région',
    kind: 'revenu',
  },
  mounted: {
    id: 'mounted',
    name: 'Montés',
    tokens: 5,
    icon: '🐎',
    tag: '−1 Collines/Fermes',
    short: '−1 jeton sur Collines et Fermes',
    desc: 'Conquérir une Colline ou une Ferme coûte 1 jeton de moins (minimum 1).',
    foe: '−1 pour attaquer Collines et Fermes',
    kind: 'attaque',
  },
  pillaging: {
    id: 'pillaging',
    name: 'Pillards',
    tokens: 5,
    icon: '🔥',
    tag: '+1 💰 par victime',
    short: '+1 💰 par région non vide conquise',
    desc: 'Chaque région non vide conquise ce tour rapporte 1 pièce de plus.',
    foe: '+1 💰 pour chaque région occupée qu\'ils prennent',
    kind: 'attaque',
  },
  seafaring: {
    id: 'seafaring',
    name: 'Marins',
    tokens: 5,
    icon: '⛵',
    tag: 'prennent mers/lacs',
    short: 'Conquièrent mers et lacs',
    desc: 'Peuvent conquérir les mers et les lacs, qui rapportent des pièces comme les autres régions (même en déclin).',
    foe: 'Occupent mers et lacs',
    kind: 'spécial',
  },
  spirit: {
    id: 'spirit',
    name: 'Spirituels',
    tokens: 5,
    icon: '👻',
    tag: '2e déclin gardé',
    short: '2e peuple en déclin autorisé',
    desc: "Quand ce peuple passe en déclin, il ne compte pas dans la limite : vous pouvez garder un autre peuple en déclin en même temps, jusqu'à la fin de la partie.",
    foe: 'Gardent un 2e peuple en déclin',
    kind: 'spécial',
  },
  stout: {
    id: 'stout',
    name: 'Robustes',
    tokens: 4,
    icon: '🍺',
    tag: 'déclin fin de tour',
    short: 'Déclin possible après avoir conquis',
    desc: 'Peuvent passer en déclin à la fin d\'un tour normal, après avoir conquis et marqué leurs points.',
    foe: 'Peuvent attaquer puis décliner dans le même tour',
    kind: 'spécial',
  },
  swamp: {
    id: 'swamp',
    name: 'des Marais',
    tokens: 4,
    icon: '🐸',
    tag: '+1 💰 par Marais',
    short: '+1 💰 par Marais',
    desc: 'Chaque Marais occupé rapporte 1 pièce de plus (tant que le peuple est actif).',
    foe: '+1 💰 par Marais',
    kind: 'revenu',
  },
  underworld: {
    id: 'underworld',
    name: 'Souterrains',
    tokens: 5,
    icon: '🦇',
    tag: 'cavernes reliées',
    short: 'Cavernes reliées, −1 jeton',
    desc: 'Conquérir une Caverne coûte 1 jeton de moins, et toutes les Cavernes sont considérées comme adjacentes entre elles.',
    foe: 'Cavernes reliées entre elles, −1 pour les prendre',
    kind: 'attaque',
  },
  wealthy: {
    id: 'wealthy',
    name: 'Fortunés',
    tokens: 4,
    icon: '💎',
    tag: '+7 💰 une fois',
    short: '+7 💰 à la fin du 1er tour',
    desc: 'Gagnent 7 pièces une seule fois, à la fin du tour où ils ont été choisis.',
    foe: '+7 💰 au tour où ils sont choisis',
    kind: 'revenu',
  },
};

export const RACE_IDS = Object.keys(RACES) as RaceId[];
export const POWER_IDS = Object.keys(POWERS) as PowerId[];

export interface TagInfo {
  /** D'où vient l'étiquette : deux sujets peuvent partager le même emoji (Humains et Fermes). */
  race?: RaceId;
  power?: PowerId;
  icon: string;
  tag: string;
  kind: EffectKind;
}

/** Les 2 étiquettes d'une combinaison : attaque, puis défense, spécial, revenu (le peuple d'abord à égalité). */
export function tagsOf(race: RaceId, power: PowerId): TagInfo[] {
  const r = RACES[race];
  const p = POWERS[power];
  const tags = [
    { race, icon: r.icon, tag: r.tag, kind: r.kind },
    { power, icon: p.icon, tag: p.tag, kind: p.kind },
  ];
  return tags.sort((a, b) => KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind));
}

export interface TerrainDef {
  name: string;
  color: string;
  icon: string;
}

export const TERRAINS: Record<Terrain, TerrainDef> = {
  farmland: { name: 'Ferme', color: '#ecd77c', icon: '🌾' },
  forest: { name: 'Forêt', color: '#356b2c', icon: '🌲' },
  hill: { name: 'Colline', color: '#86a442', icon: '🏞️' },
  mountain: { name: 'Montagne', color: '#9b948e', icon: '🏔️' },
  swamp: { name: 'Marais', color: '#7b6345', icon: '🐸' },
  sea: { name: 'Mer', color: '#4b8cc6', icon: '🌊' },
  lake: { name: 'Lac', color: '#71b4df', icon: '💧' },
};

export const FEATURE_ICONS = {
  cavern: '🕳️',
  magic: '✨',
  mine: '⛏️',
  lost: '🛖',
  lair: '🪨',
  fortress: '🏰',
  camp: '⛺',
  hero: '🦸',
  dragon: '🐉',
  hole: '🏡',
};

/** Faces du dé de renfort. */
export const DIE_FACES = [0, 0, 0, 1, 2, 3];

export function dieChance(need: number): number {
  if (need <= 0) return 1;
  let hits = 0;
  for (const f of DIE_FACES) if (f >= need) hits++;
  return hits / DIE_FACES.length;
}

export const PLAYER_COLORS = [
  { id: 'rouge', name: 'Rouge', color: '#d62839' },
  { id: 'bleu', name: 'Bleu', color: '#1f6fe0' },
  { id: 'violet', name: 'Violet', color: '#8e3ccb' },
  { id: 'orange', name: 'Orange', color: '#ec7f00' },
  { id: 'noir', name: 'Noir', color: '#2b2d42' },
];

/** Noms donnés aux IA qui complètent la table. */
export const AI_NAMES = ['Gorbak', 'Ysolde', 'Thorvald', 'Mirabelle', 'Zéphyr'];

/** Les niveaux d'IA proposés, du plus faible au plus fort. */
export const AI_LEVELS: { id: AiLevel; label: string }[] = [
  { id: 'facile', label: 'Facile' },
  { id: 'normal', label: 'Normale' },
  { id: 'difficile', label: 'Difficile' },
  { id: 'expert', label: 'Expert' },
];

export function comboName(race: RaceId, power: PowerId): string {
  return `${RACES[race].name} ${POWERS[power].name}`;
}

/** Noms des régions : un nom commun selon le terrain, puis un complément (numéros tirés par le noyau). */
export const NOUNS: Record<Terrain, string[]> = {
  farmland: ['Champs', 'Prairie', 'Vergers', 'Pâturages', 'Plaine', 'Moissons', 'Bocage', 'Jardins'],
  forest: ['Forêt', 'Bois', 'Sylve', 'Futaie', 'Bosquet', 'Taillis', 'Ramée'],
  hill: ['Collines', 'Coteaux', 'Butte', 'Tertre', 'Vallons', 'Mamelons'],
  mountain: ['Pics', 'Mont', 'Cimes', 'Crêtes', 'Aiguilles', 'Massif'],
  swamp: ['Marais', 'Tourbière', 'Marécages', 'Fondrières', 'Bayou', 'Palus'],
  sea: ['Mer', 'Golfe', 'Baie', 'Détroit', 'Océan'],
  lake: ['Lac', 'Étang', 'Lagune'],
};

export const SUFFIXES = [
  'de Brume', 'des Corbeaux', 'du Roi Fou', "d'Ambre", 'des Lucioles', 'de Valombre', 'des Murmures',
  "de l'Aube", 'du Crépuscule', 'des Géants', "d'Argent", 'des Songes', 'de Fer', 'du Loup',
  'des Ombres', 'de Cendre', 'des Anciens', 'du Dragon', 'des Vents', "d'Opale", 'des Échos',
  'de Givre', 'du Hibou', 'des Sorcières', 'de Lune', 'du Soleil', 'de Rouille', 'du Tonnerre',
  'des Fées', 'de Sel', 'du Pendu', 'des Trois Rois', 'de Jade', 'de Grisval', 'des Mille Pas',
  'du Silence', 'de Pierrefeu', 'des Ronces', 'du Vieux Chêne', 'des Larmes', "de l'Ours",
  'du Sanglier', 'de Bruyère', 'des Cloches', 'du Nord', 'du Sud', "de l'Est", "de l'Ouest",
  'des Brumes Hautes', 'de Mirelune', 'du Héron', 'des Oubliés', "d'Écume", 'de Corail',
];
