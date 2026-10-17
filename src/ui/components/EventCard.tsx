import { POWERS, RACES } from '../../engine/data';
import type { GameState } from '../../engine/types';
import { nextFrame, skipPlayback, type Playback, type UIState } from '../store';

/** Ce que fait le joueur qui a la main, quand aucune action n'est à montrer. */
function waitingText(g: GameState, actor: number): string {
  if (g.phase === 'pick') return 'choisit un peuple…';
  if (g.phase === 'redeploy') return 'redéploie…';
  if (g.phase === 'defend') {
    const task = g.defense[0];
    return task && task.player === actor ? `replace ses ${RACES[task.race].name}…` : 'réfléchit…';
  }
  return 'réfléchit…';
}

/** Emojis de la combinaison active du joueur, dans l'état affiché. */
function comboIcons(g: GameState, pid: number): string {
  const race = g.players[pid]?.active;
  if (!race) return '';
  const power = g.races[race]?.power;
  return `${RACES[race].icon}${power ? POWERS[power].icon : ''}`;
}

/**
 * Carte d'événement : pendant le tour d'un autre joueur, une trame par action (toucher = trame suivante, ⏩ = tout passer),
 * puis l'attente (« réfléchit… »). Occupe la zone d'action (84 px).
 */
export function EventCard({ s, g, actor }: { s: UIState; g: GameState; actor: number }) {
  // « connexion perdue » et « adversaire hors ligne » sont rendus par la zone d'action avant d'arriver ici
  const pb = s.playback;
  if (pb) return <Frame g={g} pb={pb} />;

  const p = g.players[actor]!;
  // la dernière trame reste lisible, puis passe en attente ; rien d'un autre tour ne s'affiche
  const card = s.card && s.card.turn === `${g.round}:${g.current}` && s.card.actor === actor && s.card.story ? s.card : null;
  const wait = waitingText(g, actor);
  return (
    <div class={`actionbar eventcard ${s.idle ? 'idle' : ''}`} style={{ '--pc': p.color } as Record<string, string>} aria-live="polite">
      <div class="ec-body">
        <div class="ec-who">
          <span class="pdot" style={{ background: p.color }} />
          <span class="ec-name">{p.name}</span>
          {comboIcons(g, actor) && <span class="ec-combo">· {comboIcons(g, actor)}</span>}
          {card && s.idle && (
            <span class="ec-suffix">
              <span class="spinner" /> · {wait}
            </span>
          )}
        </div>
        {card ? (
          <>
            <div class="ec-main">{card.story!.main}</div>
            {card.story!.detail && <div class={`ec-detail ${card.story!.tone === 'hit' ? 'hit' : ''}`}>{card.story!.detail}</div>}
          </>
        ) : (
          <div class="ec-main">
            <span class="spinner" /> {wait}
          </div>
        )}
      </div>
    </div>
  );
}

/** Une trame en cours d'affichage. */
function Frame({ g, pb }: { g: GameState; pb: Playback }) {
  const who = pb.actor !== null ? g.players[pb.actor] : null;
  const color = who?.color ?? '#7b6d5b';
  const story = pb.story;
  const skip = (e: Event) => {
    e.stopPropagation();
    skipPlayback();
  };
  const tall = !!story?.extra;
  // groupe de redéploiement loin de moi : la carte précédente pâlit, « · redéploie… »
  const wait = pb.kind === 'wait' && who ? waitingText(g, pb.actor!) : null;
  let body;
  if (pb.kind === 'header') {
    const [main, rest] = (pb.header ?? '').split(' · ');
    body = (
      <>
        <div class="ec-who">
          <span class="ec-name">Pendant votre absence</span>
        </div>
        <div class="ec-main">{main}</div>
        {rest && <div class="ec-detail">{rest}</div>}
      </>
    );
  } else {
    body = (
      <>
        <div class="ec-who">
          {who && <span class="pdot" style={{ background: color }} />}
          {who && <span class="ec-name">{who.name}</span>}
          {who && comboIcons(g, pb.actor!) && <span class="ec-combo">· {comboIcons(g, pb.actor!)}</span>}
          {wait && story && (
            <span class="ec-suffix">
              <span class="spinner" /> · {wait}
            </span>
          )}
        </div>
        {story && pb.rolling && story.rolling ? (
          <div class="ec-main rolling">
            <span class="ec-die" aria-hidden="true">
              🎲
            </span>
            <span class="ec-ell">{story.rolling}</span>
          </div>
        ) : story ? (
          <>
            <div key={pb.id} class={`ec-main ${story.dieRoll ? 'verdict' : ''}`}>
              {story.main}
            </div>
            {story.detail && <div class={`ec-detail ${story.tone === 'hit' ? 'hit' : ''}`}>{story.detail}</div>}
            {story.extra && <div class="ec-detail">{story.extra}</div>}
          </>
        ) : (
          wait && (
            <div class="ec-main">
              <span class="spinner" /> {wait}
            </div>
          )
        )}
      </>
    );
  }
  return (
    <div
      class={`actionbar eventcard live ${tall ? 'tall' : ''} ${pb.kind === 'wait' ? 'wait' : ''} ${pb.kind === 'header' ? 'header' : ''}`}
      style={{ '--pc': color } as Record<string, string>}
      onClick={() => nextFrame()}
      aria-live="polite"
    >
      <div class="ec-inner">
        <div key={pb.id} class="ec-progress" style={{ animationDuration: `${pb.ms}ms` }} />
        <div key={`b${pb.id}`} class="ec-body">
          {body}
        </div>
        {pb.backlog > 0 && (
          <button class="ec-skip" onClick={skip} aria-label={`Passer les ${pb.backlog} actions en attente`}>
            ⏩ {pb.backlog}
          </button>
        )}
      </div>
    </div>
  );
}

/** Ancien nom, gardé pour la zone d'action. */
export const WaitingBar = EventCard;
