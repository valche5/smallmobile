import { useEffect, useRef, useState } from 'preact/hooks';
import { ranking } from '../../engine/rules';
import type { DieEvent, GameState } from '../../engine/types';
import { clearCoin, dismissOverlay, getState, quitGame, setState, type CoinPop } from '../store';
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

/**
 * Le lancer : le dé fait HALF_TURNS demi-tours sur lui-même en ROLL_MS, en ralentissant ; chaque fois qu'il passe
 * sur la tranche, une nouvelle face apparaît, la dernière étant le résultat. Le nombre de demi-tours est pair :
 * le dé finit à plat, à l'endroit.
 */
const HALF_TURNS = 12;
const ROLL_MS = 1500;

/** Les faces qui défilent : jamais deux fois la même de suite, la dernière différente du résultat. */
function rollFaces(result: number): number[] {
  const faces: number[] = [];
  let prev = -1;
  for (let i = 0; i < HALF_TURNS; i++) {
    const last = i === HALF_TURNS - 1;
    const options = [0, 1, 2, 3].filter((f) => f !== prev && !(last && f === result));
    prev = options[Math.floor(Math.random() * options.length)]!;
    faces.push(prev);
  }
  return faces;
}

export function DieOverlay({ ev, auto }: { ev: DieEvent; auto: boolean }) {
  const [step, setStep] = useState(0);
  // les faces qui défilent, puis le résultat, montré pendant le dernier demi-tour
  const [faces] = useState(() => [...rollFaces(ev.value), ev.value]);
  // le dé est arrêté : le résultat est annoncé
  const [rolled, setRolled] = useState(false);
  const dieRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const t0 = performance.now();
    let frame = 0;
    let shown = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / ROLL_MS);
      const e = 1 - Math.pow(1 - t, 3); // ralentit jusqu'à l'arrêt
      const a = e * HALF_TURNS * 180; // angle de la rotation, en degrés
      // la face change quand le dé est sur la tranche (90°, 270°…)
      const idx = Math.min(HALF_TURNS, Math.floor((a + 90) / 180));
      if (idx !== shown) {
        shown = idx;
        setStep(idx);
      }
      const el = dieRef.current;
      if (el && !reduced) {
        const grow = Math.min(1, t / 0.12); // le dé grossit en arrivant
        const lift = Math.sin(Math.PI * Math.min(1, t * 1.15)) * -18; // et se soulève pendant qu'il roule
        el.style.transform =
          t < 1 ? `translateY(${lift.toFixed(1)}px) perspective(500px) rotateX(${a.toFixed(1)}deg) rotateZ(${((a / (HALF_TURNS * 180)) * 360).toFixed(1)}deg) scale(${(0.5 + 0.5 * grow).toFixed(3)})` : '';
      }
      if (t < 1) frame = requestAnimationFrame(tick);
      else setRolled(true);
    };
    frame = requestAnimationFrame(tick);
    // page en arrière-plan : le navigateur suspend l'animation, le résultat s'affiche quand même
    const done = setTimeout(() => {
      setStep(HALF_TURNS);
      setRolled(true);
      if (dieRef.current) dieRef.current.style.transform = '';
    }, ROLL_MS + 100);
    let timer: ReturnType<typeof setTimeout> | null = null;
    if (auto) {
      const speed = getState().aiSpeed;
      timer = setTimeout(dismissOverlay, ROLL_MS + (speed === 'rapide' ? 600 : speed === 'lent' ? 1800 : 1100));
    }
    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(done);
      if (timer) clearTimeout(timer);
    };
  }, []);
  const face = rolled ? ev.value : faces[step]!;
  let verdict = '';
  if (ev.berserk) verdict = ev.value ? `La prochaine conquête coûtera ${plural(ev.value, 'jeton')} de moins.` : 'Pas de chance : aucun bonus.';
  else verdict = ev.success ? 'Réussite : la région est conquise !' : `Échec : il fallait ${ev.need} ou plus.`;
  return (
    <div class="overlay die-overlay" onClick={() => rolled && dismissOverlay()}>
      {/* rotation continue (tick), puis un petit rebond sur le résultat */}
      <div ref={dieRef} class={`die ${rolled ? 'done' : 'rolling'}`}>
        <svg class={rolled ? 'land' : ''} viewBox="0 0 100 100" width="120" height="120">
          <rect x="4" y="4" width="92" height="92" rx="18" fill="#fffaf0" stroke="#2b2118" stroke-width="4" />
          {PIPS[face]!.map(([x, y], i) => (
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

/**
 * Fin du tour d'un joueur : une pièce d'or de la taille du dé, au milieu de l'écran, avec les pièces gagnées.
 * Elle apparaît en tournant sur elle-même, reste un instant puis s'efface ; on peut continuer à jouer dessous.
 */
export function CoinOverlay({ coin, me }: { coin: CoinPop; me: boolean }) {
  useEffect(() => {
    const t = setTimeout(() => clearCoin(coin.key), coin.ms);
    return () => clearTimeout(t);
  }, []);
  const text = `+${coin.gain}`;
  return (
    <div class="coin-pop" style={{ '--coin-ms': `${coin.ms}ms`, '--pc': coin.color } as Record<string, string>} aria-live="polite">
      <svg class="coin" viewBox="0 0 100 100" width="120" height="120" aria-hidden="true">
        <defs>
          <radialGradient id="coin-face" cx="38%" cy="32%" r="75%">
            <stop offset="0%" stop-color="#fff2b0" />
            <stop offset="45%" stop-color="#f7c843" />
            <stop offset="100%" stop-color="#c98a12" />
          </radialGradient>
        </defs>
        <circle cx="50" cy="50" r="46" fill="url(#coin-face)" stroke="#8a5a00" stroke-width="3" />
        <circle cx="50" cy="50" r="37" fill="none" stroke="#a86f05" stroke-width="2" stroke-dasharray="3 3" opacity="0.7" />
        <text x="50" y={text.length > 3 ? 61 : 63} text-anchor="middle" class="coin-gain" font-size={text.length > 3 ? 30 : 36}>
          {text}
        </text>
      </svg>
      <div class="coin-who">
        <span class="pdot" style={{ background: coin.color }} />
        {me ? 'Votre tour' : coin.name} · {coin.gain} {coin.gain > 1 ? 'pièces' : 'pièce'}
      </div>
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
