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

/** Les niveaux d'IA. */
export function LevelChoice({ level, onPick }: { level: AiLevel; onPick?: (l: AiLevel) => void }) {
  return (
    <div class="seg">
      {AI_LEVELS.map((l) => (
        <button key={l.id} class={level === l.id ? 'on' : ''} disabled={!onPick} onClick={() => onPick?.(l.id)}>
          {l.label}
        </button>
      ))}
    </div>
  );
}
