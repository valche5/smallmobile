import { AI_LEVELS, EXPANSION_IDS, EXPANSIONS, expansionBlurb } from '../../engine/data';
import type { AiLevel, ExpansionId } from '../../engine/types';

/** La liste `value` avec l'extension `x` cochée ou non, dans l'ordre du noyau. */
export function toggleExpansion(value: readonly ExpansionId[], x: ExpansionId, on: boolean): ExpansionId[] {
  return EXPANSION_IDS.filter((y) => (y === x ? on : value.includes(y)));
}

/** Les trois extensions, une case chacune : leur nom, et en petit ce qu'elles apportent. `onToggle` absent : lecture seule. */
export function ExpansionChoice({ value, onToggle }: { value: readonly ExpansionId[]; onToggle?: (x: ExpansionId, on: boolean) => void }) {
  const toggle = (x: ExpansionId, on: boolean) => onToggle?.(x, on);
  return (
    <div class="expansions">
      {EXPANSION_IDS.map((x) => (
        <label key={x} class={`stout-row expansion ${onToggle ? '' : 'readonly'}`}>
          <input
            type="checkbox"
            checked={value.includes(x)}
            disabled={!onToggle}
            onChange={(e) => toggle(x, (e.target as HTMLInputElement).checked)}
          />
          <span>
            <b>{EXPANSIONS[x].name}</b> <span class="muted small">· {expansionBlurb(x)}</span>
          </span>
        </label>
      ))}
    </div>
  );
}

/** Les niveaux d'IA ; avec des extensions, l'Expert (jeu de base seulement) est grisé. */
export function LevelChoice({ level, expansions, onPick }: { level: AiLevel; expansions: readonly ExpansionId[]; onPick?: (l: AiLevel) => void }) {
  const noExpert = expansions.length > 0;
  return (
    <>
      <div class="seg">
        {AI_LEVELS.map((l) => {
          const off = noExpert && l.id === 'expert';
          return (
            <button
              key={l.id}
              class={`${level === l.id ? 'on' : ''} ${off ? 'off' : ''}`}
              disabled={off || !onPick}
              title={off ? 'Jeu de base seulement' : undefined}
              onClick={() => onPick?.(l.id)}
            >
              {l.label}
            </button>
          );
        })}
      </div>
      {noExpert && <p class="muted small">L'IA Expert ne connaît que le jeu de base : pas d'Expert avec une extension.</p>}
    </>
  );
}
