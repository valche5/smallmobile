import { useEffect, useState } from 'preact/hooks';
import { RACES } from '../../engine/data';
import { ranking } from '../../engine/rules';
import type { DieEvent, GameState } from '../../engine/types';
import { dismissOverlay, getState, quitGame, setState } from '../store';
import { PlayerDot, plural } from './common';
import { KIND_CLASS, playerTags } from './TopBar';
import { TagArt } from './ArtIcon';

/** Passe-et-joue : écran de passage minimal (5 lignes au plus, aucune liste). La relecture démarre à sa fermeture. */
export function HandoffOverlay({ g, player, sinceSeq, defense }: { g: GameState; player: number; sinceSeq: number; defense: boolean }) {
  const p = g.players[player]!;
  const hits = g.log.filter((l) => l.seq > sinceSeq && l.player !== player && l.kind === 'conquer' && l.victim === player);
  const attackers = [...new Set(hits.map((l) => l.player))];
  const by = attackers.length === 1 && attackers[0] !== null ? g.players[attackers[0]!]!.name : 'On';
  const task = defense ? g.defense[0] : null;
  const hand = task ? (g.races[task.race]?.hand ?? 0) : 0;
  const tags = playerTags(g, player);
  const parts: string[] = [];
  if (hits.length) parts.push(`${by} vous a pris ${plural(hits.length, 'région')}`);
  if (task) parts.push(`${plural(hand, 'jeton')} ${RACES[task.race].name} à replacer`);
  return (
    <div class="overlay handoff" style={{ '--pc': p.color } as Record<string, string>}>
      <div class="handoff-card">
        <div class="handoff-top">
          <PlayerDot player={p} size={14} />
          <span>
            Tour {g.round}/{g.maxRounds}
          </span>
        </div>
        <h1>{task && task.player !== g.current ? `${p.name}, défendez-vous !` : `À vous, ${p.name} !`}</h1>
        {tags.length > 0 ? (
          <p class="handoff-tags">
            {tags.map((t, i) => (
              <span key={t.icon}>
                {i > 0 && ' · '}
                <span class={`emo k-${KIND_CLASS[t.kind]}`}><TagArt t={t} size={16} /></span> {t.tag}
              </span>
            ))}
          </p>
        ) : (
          g.phase === 'pick' && <p class="handoff-tags">👑 Choisissez un peuple</p>
        )}
        {parts.length > 0 && <p class="lead hit-line">{parts.join(' · ')}.</p>}
        <button class="btn primary big-btn" onClick={dismissOverlay}>
          C'est parti&nbsp;!
        </button>
      </div>
    </div>
  );
}

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
