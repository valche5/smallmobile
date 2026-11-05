import type { ComponentChildren } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { TERRAINS } from '../../engine/data';
import { cachedMap, fetchMap, MAP_H, MAP_W } from '../../engine/maps';
import type { MapDef } from '../../engine/types';

/** Aperçu miniature d'une carte, avec quelques chiffres (la carte est demandée au serveur). */
export function MapPreview({ size, seed, children }: { size: number; seed: number; children?: ComponentChildren }) {
  const [map, setMap] = useState<MapDef | null>(() => cachedMap(size, seed));
  useEffect(() => {
    let live = true;
    fetchMap(size, seed).then(
      (m) => live && setMap(m),
      (e) => console.warn(e),
    );
    return () => {
      live = false;
    };
  }, [size, seed]);
  const water = map?.regions.filter((r) => r.terrain === 'sea' || r.terrain === 'lake').length ?? 0;
  return (
    <div class="map-preview">
      <svg viewBox={`-8 -8 ${MAP_W + 16} ${MAP_H + 16}`} aria-label="Aperçu de la carte">
        <rect x={-8} y={-8} width={MAP_W + 16} height={MAP_H + 16} rx={20} fill="#3d2f22" />
        {map?.regions.map((r) => (
          <path key={r.id} d={r.path} fill={TERRAINS[r.terrain].color} stroke="#3d2f22" stroke-width={5} stroke-linejoin="round" />
        ))}
      </svg>
      <div class="map-info">
        {map ? (
          <>
            <b>{map.name}</b>
            <span class="muted small">
              {map.regions.length - water} terres, {water} étendues d'eau, {map.regions.filter((r) => r.lostTribe).length} tribus
              oubliées · {map.rounds} tours
            </span>
          </>
        ) : (
          <span class="muted small">Chargement de la carte…</span>
        )}
        {children}
      </div>
    </div>
  );
}
