import type { GameState } from '../../engine/types';
import { LOG_ICONS } from '../components/common';
import { openRegion, setState } from '../store';

/** Journal complet : mes pertes en rouge, les entrées sans enjeu en gris. */
export function LogView({ g, me }: { g: GameState; me: number }) {
  const entries = [...g.log].reverse();
  let lastRound = -1;
  return (
    <div class="view scroll">
      <div class="view-head">
        <h2>Journal</h2>
        <p class="muted small">Touchez une ligne liée à une région pour la voir sur la carte.</p>
      </div>
      <ul class="log">
        {entries.map((l) => {
          const header = l.round !== lastRound;
          lastRound = l.round;
          const p = l.player !== null ? g.players[l.player] : null;
          const hit = l.victim === me && l.player !== me;
          // « quiet » : entrée sans enjeu (Amazones rendues, Squelettes levés…), jamais lue en direct
          const quiet = !hit && (l as { quiet?: boolean }).quiet === true;
          return [
            header ? (
              <li key={`h${l.seq}`} class="log-round">
                Tour {l.round}
              </li>
            ) : null,
            <li
              key={l.seq}
              class={`log-entry k-${l.kind} ${l.region !== undefined ? 'link' : ''} ${hit ? 'hit' : ''} ${quiet ? 'quiet' : ''}`}
              style={p ? { borderLeftColor: p.color } : undefined}
              onClick={() => {
                if (l.region === undefined) return;
                setState({ tab: 'map' });
                openRegion(l.region);
              }}
            >
              <span class="li-icon">{LOG_ICONS[l.kind] ?? '•'}</span>
              <span>{l.text}</span>
            </li>,
          ];
        })}
      </ul>
    </div>
  );
}
