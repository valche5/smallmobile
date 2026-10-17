import { POWERS, RACES, TERRAINS, type TagInfo } from '../../engine/data';
import type { PowerId, RaceId, Terrain } from '../../engine/types';
import { POWER_ART, RACE_ART, TERRAIN_ART } from '../art';

export type ArtSubject = { race: RaceId } | { power: PowerId } | { terrain: Terrain } | { icon: string };

/** Illustration décorative : le nom et les règles restent dans le texte voisin. */
export function ArtIcon(props: ArtSubject & { size?: number }) {
  let src: string | undefined;
  let icon: string;
  if ('race' in props) {
    src = RACE_ART[props.race];
    icon = RACES[props.race].icon;
  } else if ('power' in props) {
    src = POWER_ART[props.power];
    icon = POWERS[props.power].icon;
  } else if ('terrain' in props) {
    src = TERRAIN_ART[props.terrain];
    icon = TERRAINS[props.terrain].icon;
  } else {
    icon = props.icon;
  }
  const size = props.size ?? 24;
  return src ? (
    <img class="art-icon" src={src} width={size} height={size} alt="" aria-hidden="true" decoding="async" draggable={false} />
  ) : (
    <span class="art-icon emo" aria-hidden="true" style={{ width: size, height: size, fontSize: Math.min(size, 24) }}>{icon}</span>
  );
}

/** L'illustration d'une étiquette : celle de son peuple ou de son pouvoir, jamais devinée d'après l'emoji. */
export function TagArt({ t, size }: { t: TagInfo; size?: number }) {
  if (t.race) return <ArtIcon race={t.race} size={size} />;
  if (t.power) return <ArtIcon power={t.power} size={size} />;
  return <ArtIcon icon={t.icon} size={size} />;
}
