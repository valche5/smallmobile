import { useState } from 'preact/hooks';
import { POWERS, RACES } from '../../engine/data';
import { declineBrief, previewEndTurn } from '../../engine/preview';
import { raceOf, scoreBreakdown } from '../../engine/rules';
import type { GameState } from '../../engine/types';
import { act } from '../store';
import { Consequences, Details, Lines, PlayerDot, Sheet } from './common';

interface Base {
  g: GameState;
  onClose: () => void;
}

/** Déclin (irréversible) : 4 lignes au plus, le décompte dans Détails. */
export function DeclineSheet({ g, onClose }: Base) {
  const race = g.players[g.current]!.active!;
  const rs = raceOf(g, race);
  const brief = declineBrief(g);
  return (
    <Sheet
      onClose={onClose}
     
      size="standard"
      title={
        <span class="ell">
          💤 Décliner les {RACES[race].icon} {RACES[race].name} {POWERS[rs.power].name}&nbsp;?
        </span>
      }
      footer={
        <div class="btn-row">
          <button class="btn ghost" onClick={onClose}>
            Continuer
          </button>
          <button class="btn danger" onClick={() => act({ type: 'decline' })}>
            💤 Décliner · +{brief.gain} 💰
          </button>
        </div>
      }
    >
      <Consequences lines={brief.lines} />
      <Details>
        <ScoreTable lines={brief.score.lines} total={brief.score.total} />
        <Lines lines={brief.more} />
      </Details>
    </Sheet>
  );
}

/** Ce tour se termine sur un choix (Diplomates ou Robustes) : une feuille de 4 lignes au lieu de la confirmation. */
export function needsEndTurnSheet(g: GameState): boolean {
  if (g.phase !== 'redeploy' || g.turn.stage !== 'active' || !g.turn.army) return false;
  const rs = raceOf(g, g.turn.army);
  if (rs.declined) return false;
  if (rs.power === 'stout') return true;
  return rs.power === 'diplomat' && peaceOptions(g).length > 0;
}

function peaceOptions(g: GameState) {
  return g.players.filter((q) => q.id !== g.current && q.active && !g.turn.attacked.includes(q.id));
}

/** Fin du tour des Diplomates (imposer la paix) ou des Robustes (décliner après le décompte). */
export function EndTurnSheet({ g, onClose }: Base) {
  const race = g.turn.army!;
  const rs = raceOf(g, race);
  const diplomat = !rs.declined && rs.power === 'diplomat';
  const [peace, setPeace] = useState<number | null>(null);
  const [stout, setStout] = useState(false);
  const income = scoreBreakdown(g, g.current).total;
  const icon = (pid: number) => {
    const a = g.players[pid]!.active;
    return a ? `${RACES[a].icon}${POWERS[raceOf(g, a).power].icon}` : '';
  };
  return (
    <Sheet
      onClose={onClose}
     
      size="standard"
      title={diplomat ? `${POWERS.diplomat.icon} Imposer la paix à…` : '✓ Fin du tour'}
      footer={
        <div class="btn-row">
          <button class="btn primary" onClick={() => act({ type: 'endTurn', peace: diplomat ? peace : undefined, stout: diplomat ? undefined : stout })}>
            ✓ Terminer · +{income} 💰
          </button>
        </div>
      }
    >
      {diplomat ? (
        <>
          <div class="chips peace-chips">
            <button class={`chip ${peace === null ? 'on' : ''}`} onClick={() => setPeace(null)}>
              Personne
            </button>
            {peaceOptions(g).map((q) => (
              <button key={q.id} class={`chip ${peace === q.id ? 'on' : ''}`} onClick={() => setPeace(q.id)}>
                <PlayerDot player={q} size={10} /> {q.name} <span class="emo">{icon(q.id)}</span>
              </button>
            ))}
          </div>
          <div class="sheet-line muted">
            {peace === null ? 'Il' : g.players[peace]!.name} ne pourra pas attaquer vos {RACES[race].name} à son prochain tour.
          </div>
        </>
      ) : (
        <>
          <label class="stout-row">
            <input type="checkbox" checked={stout} onChange={(e) => setStout((e.target as HTMLInputElement).checked)} />
            <span>
              {POWERS.stout.icon} Décliner après le décompte <span class="muted">(nouveau peuple au prochain tour)</span>
            </span>
          </label>
          {stout && <Consequences lines={previewEndTurn(g, { stout: true }).lines.filter((l) => l.icon === '🪦')} />}
        </>
      )}
    </Sheet>
  );
}

export function ScoreTable({ lines, total }: { lines: { label: string; value: number; icon: string }[]; total: number }) {
  return (
    <div class="cost-table score">
      <div class="ct-title">Pièces gagnées ce tour</div>
      {lines.length === 0 && <div class="ct-row muted">Aucune région occupée.</div>}
      {lines.map((l, i) => (
        <div key={i} class="ct-row">
          <span>
            <span class="emo">{l.icon}</span> {l.label}
          </span>
          <span>+{l.value}</span>
        </div>
      ))}
      <div class="ct-row total">
        <span>Total</span>
        <span>+{total} 💰</span>
      </div>
    </div>
  );
}
