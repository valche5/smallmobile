import type { ComponentChildren } from 'preact';
import { useMemo } from 'preact/hooks';
import { TERRAINS } from '../../engine/data';
import { generateMap } from '../../engine/mapgen';

/** Aperçu miniature d'une carte, avec quelques chiffres. */
export function MapPreview({ size, seed, children }: { size: number; seed: number; children?: ComponentChildren }) {
  const map = useMemo(() => generateMap(size, seed), [size, seed]);
  const water = map.regions.filter((r) => r.terrain === 'sea' || r.terrain === 'lake').length;
  return (
    <div class="map-preview">
      <svg viewBox={`-8 -8 ${map.width + 16} ${map.height + 16}`} aria-label="Aperçu de la carte">
        <rect x={-8} y={-8} width={map.width + 16} height={map.height + 16} rx={20} fill="#3d2f22" />
        {map.regions.map((r) => (
          <path key={r.id} d={r.path} fill={TERRAINS[r.terrain].color} stroke="#3d2f22" stroke-width={5} stroke-linejoin="round" />
        ))}
      </svg>
      <div class="map-info">
        <b>{map.name}</b>
        <span class="muted small">
          {map.regions.length - water} terres, {water} étendues d'eau, {map.regions.filter((r) => r.lostTribe).length} tribus
          oubliées · {map.rounds} tours
        </span>
        {children}
      </div>
    </div>
  );
}
