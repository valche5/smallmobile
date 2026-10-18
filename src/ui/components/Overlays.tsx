import { useEffect, useState } from 'preact/hooks';
import { ranking } from '../../engine/rules';
import type { DieEvent, GameState } from '../../engine/types';
import { dismissOverlay, getState, quitGame, setState } from '../store';
import { PlayerDot, plural } from './common';

const PIPS: Record<number, number[][]> = {
  0: [],
  1: [[50, 50]],
  2: [
    [28, 28],
    [72, 72],
  ],
  3: [
    [26, 26],
    [50, 50],
    [74, 74],
  ],
};

export function DieOverlay({ ev, auto }: { ev: DieEvent; auto: boolean }) {
  const [rolled, setRolled] = useState(false);
  useEffect(() => {
    const t1 = setTimeout(() => setRolled(true), 650);
    const speed = getState().aiSpeed;
    const t2 = auto ? setTimeout(dismissOverlay, speed === 'rapide' ? 1100 : speed === 'lent' ? 2600 : 1700) : null;
    return () => {
      clearTimeout(t1);
      if (t2) clearTimeout(t2);
    };
  }, []);
  let verdict = '';
  if (ev.berserk) verdict = ev.value ? `La prochaine conquête coûtera ${plural(ev.value, 'jeton')} de moins.` : 'Pas de chance : aucun bonus.';
  else verdict = ev.success ? 'Réussite : la région est conquise !' : `Échec : il fallait ${ev.need} ou plus.`;
  return (
    <div class="overlay die-overlay" onClick={() => rolled && dismissOverlay()}>
      <div class={`die ${rolled ? 'done' : 'rolling'}`}>
        <svg viewBox="0 0 100 100" width="120" height="120">
          <rect x="4" y="4" width="92" height="92" rx="18" fill="#fffaf0" stroke="#2b2118" stroke-width="4" />
          {rolled && ev.value === 0 && <text x="50" y="62" text-anchor="middle" font-size="34">∅</text>}
          {(rolled ? PIPS[ev.value]! : PIPS[2]!).map(([x, y], i) => (
            <circle key={i} cx={x} cy={y} r="9" fill="#2b2118" />
          ))}
        </svg>
      </div>
      {rolled && (
        <div class={`die-result ${ev.berserk ? '' : ev.success ? 'good' : 'bad'}`}>
          <div class="big">🎲 {ev.value}</div>
          <div>{verdict}</div>
          {!auto && <div class="muted small">Touchez pour continuer</div>}
        </div>
      )}
    </div>
  );
}

/** Écran de fin : le vainqueur, ma place, le podium (les jetons en jeu seulement pour départager des ex æquo). */
export function GameOverOverlay({ g, me, onClose }: { g: GameState; me: number | null; onClose: () => void }) {
  const rank = ranking(g);
  const medals = ['🥇', '🥈', '🥉', '4.', '5.'];
  const best = rank[0]!;
  const mine = rank.findIndex((r) => r.pid === me);
  const name = (pid: number) => (pid === me ? 'Vous' : g.players[pid]!.name);
  return (
    <div class="overlay gameover">
      <div class="handoff-card">
        <h1>
          🏁 {best.pid === me ? "Vous l'emportez" : `${g.players[best.pid]!.name} l'emporte`} avec {best.coins}&nbsp;💰
        </h1>
        {mine > 0 && (
          <p class="lead">
            Vous : {rank[mine]!.coins}&nbsp;💰 · {mine + 1}e
          </p>
        )}
        <ol class="podium">
          {rank.map((r, i) => {
            const p = g.players[r.pid]!;
            const tied = rank.some((o) => o.pid !== r.pid && o.coins === r.coins);
            return (
              <li key={r.pid} class={r.pid === me ? 'me' : ''} style={{ borderLeftColor: p.color }}>
                <span class="medal">{medals[i]}</span>
                <span class="pname">
                  <PlayerDot player={p} /> {name(r.pid)}
                </span>
                <span class="score">
                  <span>
                    <b>{r.coins}</b> 💰{best.coins > r.coins && <span class="gap">(−{best.coins - r.coins})</span>}
                  </span>
                  {tied && <small class="tie">{plural(r.tokens, 'jeton')} en jeu</small>}
                </span>
              </li>
            );
          })}
        </ol>
        <div class="btn-row">
          <button class="btn ghost" onClick={onClose}>
            Voir la carte
          </button>
          <button
            class="btn primary"
            onClick={() => {
              quitGame();
              setState({ tab: 'map' });
            }}
          >
            Nouvelle partie
          </button>
        </div>
      </div>
    </div>
  );
}
