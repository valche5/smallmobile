import { useEffect, useRef, useState } from 'preact/hooks';
import { RACES, tagsOf, type EffectKind, type TagInfo } from '../../engine/data';
import { actingRace, currentActor, raceOf } from '../../engine/rules';
import type { GameState, RaceId } from '../../engine/types';
import { canAct, setState, viewerSeat, type UIState } from '../store';
import { TagArt } from './ArtIcon';

/** Classe CSS de chaque nature d'effet (couleurs --k-*). */
export const KIND_CLASS: Record<EffectKind, string> = { attaque: 'atk', défense: 'def', spécial: 'spe', revenu: 'inc' };

/**
 * Le joueur dont ce téléphone montre le point de vue : en ligne, mon siège ; seul contre l'IA, l'unique humain ;
 * en passe-et-joue, l'humain qui a la main, ou le dernier humain à l'avoir eue pendant qu'une IA joue (ou pendant une relecture).
 */
export function mySeat(s: UIState, g: GameState): number {
  const seat = viewerSeat(s);
  if (seat !== null) return seat;
  const actor = currentActor(g);
  if (actor !== null && canAct(s)) return actor;
  return s.lastHuman ?? actor ?? g.current;
}

/** Une étiquette : l'emoji souligné à la couleur de sa nature, puis le texte court. */
export function Tag({ t }: { t: TagInfo }) {
  return (
    <span class={`tagbox k-${KIND_CLASS[t.kind]}`}>
      <span class="emo"><TagArt t={t} size={16} /></span>
      <span class="tag">{t.tag}</span>
    </span>
  );
}

/** Les étiquettes du peuple actif d'un joueur (vide sans peuple actif). */
export function playerTags(g: GameState, pid: number): TagInfo[] {
  const race = g.players[pid]!.active;
  return race ? tagsOf(race, raceOf(g, race).power) : [];
}

export function TopBar({ g, s }: { g: GameState; s: UIState }) {
  const actor = currentActor(g);
  const me = mySeat(s, g);
  const p = g.players[me]!;
  const over = g.phase === 'gameOver';
  const myTurn = !over && actor === me;
  const last = g.round === g.maxRounds;
  const acting = myTurn && canAct(s) && g.phase !== 'pick' ? actingRace(g) : null;
  const n = g.players.length;
  // les adversaires dans l'ordre où ils joueront après moi
  const foes = Array.from({ length: n - 1 }, (_, i) => (me + 1 + i) % n);
  return (
    <header class="topbar">
      <div class="me-row">
        <div class={`round ${myTurn ? 'mine' : ''} ${last && !over ? 'last' : ''}`} aria-label={`Tour ${g.round} sur ${g.maxRounds}`}>
          <div class="round-num">
            {over ? '🏁' : g.round}
            {!over && <small>/{g.maxRounds}</small>}
          </div>
          <div class="round-label" style={myTurn ? ({ '--pc': p.color } as Record<string, string>) : undefined}>
            {over ? 'fin' : myTurn ? 'à vous' : last ? 'dernier' : 'tour'}
          </div>
        </div>
        <button class="who" title="Peuples en jeu" onClick={() => setState({ sheet: { kind: 'people', player: me } })}>
          <MyPeople g={g} pid={me} choosing={myTurn && g.phase === 'pick'} />
        </button>
        <div class="stats">
          <Counter key={`c${me}`} icon="💰" value={p.coins} label="Pièces" secret={!over} />
          {acting && <Counter key={`h${me}`} icon="🖐️" value={g.races[acting]?.hand ?? 0} label="Jetons en main" />}
        </div>
        <button class="menu-btn" aria-label="Menu" onClick={() => setState({ sheet: { kind: 'menu' } })}>
          ☰
        </button>
      </div>
      <div class={`foes ${foes.length > 1 ? 'rows' : 'one'}`}>
        {foes.map((pid) => (
          <Foe key={pid} g={g} pid={pid} me={me} now={actor === pid && !over} over={over} single={foes.length === 1} />
        ))}
      </div>
    </header>
  );
}

/** Mes 2 étiquettes, une par ligne ; sans peuple actif, ce qu'il me reste. */
function MyPeople({ g, pid, choosing }: { g: GameState; pid: number; choosing: boolean }) {
  const p = g.players[pid]!;
  const tags = playerTags(g, pid);
  const dot = <span class="pdot" style={{ background: p.color }} />;
  if (tags.length) {
    return (
      <>
        <div class="who-line">
          {dot}
          <Tag t={tags[0]!} />
        </div>
        <div class="who-line indent">
          <Tag t={tags[1]!} />
        </div>
      </>
    );
  }
  const old = p.declined[0];
  return (
    <>
      <div class="who-line">
        {dot}
        <span class="who-text">{choosing ? '👑 Choisissez un peuple' : 'Pas de peuple actif'}</span>
      </div>
      {old && (
        <div class="who-line indent">
          <span class="who-text">
            💤 <span class="emo">{RACES[old].icon}</span> {RACES[old].inDecline ? RACES[old].tag : 'en déclin'}
          </span>
        </div>
      )}
    </>
  );
}

/**
 * Compteur de la ligne du haut : chaque changement fait monter une bulle « +4 » ou « −3 ».
 * `secret` : le total reste caché (les pièces, jusqu'à la fin de la partie), seule la bulle passe.
 */
function Counter({ icon, value, label, secret }: { icon: string; value: number; label: string; secret?: boolean }) {
  const prev = useRef(value);
  const [bubble, setBubble] = useState<{ id: number; d: number } | null>(null);
  useEffect(() => {
    const d = value - prev.current;
    prev.current = value;
    if (!d) return;
    const id = Math.random();
    setBubble({ id, d });
    const t = setTimeout(() => setBubble((b) => (b?.id === id ? null : b)), 1000);
    return () => clearTimeout(t);
  }, [value]);
  return (
    <div class="stat" title={secret ? `${label} : cachées jusqu'à la fin` : label} aria-label={secret ? `${label} cachées` : `${label} : ${value}`}>
      <span class="emo">{icon}</span>
      {!secret && <b>{value}</b>}
      {bubble && (
        <span key={bubble.id} class={`bubble ${bubble.d > 0 ? 'up' : 'down'}`} aria-hidden="true">
          {bubble.d > 0 ? `+${bubble.d}` : `−${-bubble.d}`}
        </span>
      )}
    </div>
  );
}

/** Un adversaire : puce de 2 lignes s'il est seul, sinon une rangée. Toucher = sa fiche dans Peuples en jeu. */
function Foe({ g, pid, me, now, over, single }: { g: GameState; pid: number; me: number; now: boolean; over: boolean; single: boolean }) {
  const p = g.players[pid]!;
  const tags = playerTags(g, pid);
  const peace = g.peace.some((pc) => pc.diplomat === pid && pc.target === me);
  const flip = usePivot(p.active);
  const style = { '--pc': p.color } as Record<string, string>;
  const open = () => setState({ sheet: { kind: 'people', player: pid } });
  const label = `Peuples de ${p.name}`;
  const what = tags.length ? (
    <span class="foe-tags">
      <Tag t={tags[0]!} />
      <Tag t={tags[1]!} />
    </span>
  ) : (
    <NoPeople g={g} pid={pid} now={now} single={single} />
  );
  // en déclin : tous en puce simple ; seulement les Goules (qui attaquent encore) en rangée, s'il reste de la place
  const shown = single ? p.declined : p.declined.filter((r) => r === 'ghouls');
  const old = tags.length && shown.length ? <OldPeoples races={shown} /> : null;
  if (single) {
    return (
      <button class={`foe one ${now ? 'now' : ''} ${flip}`} style={style} onClick={open} aria-label={label}>
        <div class="foe-l1">
          <span class="pdot" style={{ background: p.color }} />
          <span class="foe-name">{p.name}</span>
          {over && <span class="foe-coins">· {p.coins} 💰</span>}
          <span class="foe-end">
            {now && <span class="foe-now">▶ joue</span>}
            {peace && <span class="emo foe-peace">🕊️</span>}
            <span class="chev">›</span>
          </span>
        </div>
        <div class="foe-l2">
          {what}
          {old}
        </div>
      </button>
    );
  }
  return (
    <button class={`foe row ${now ? 'now' : ''} ${flip}`} style={style} onClick={open} aria-label={label}>
      <span class="pdot" style={{ background: p.color }} />
      <span class="foe-name">{p.name}</span>
      <span class="foe-play">{now ? '▶' : ''}</span>
      {what}
      {old}
      <span class="foe-end">
        {peace && <span class="emo foe-peace">🕊️</span>}
        <span class="chev">›</span>
      </span>
    </button>
  );
}

/** Adversaire sans peuple actif : il choisit, il joue ses Goules en déclin, ou il n'a rien. */
function NoPeople({ g, pid, now, single }: { g: GameState; pid: number; now: boolean; single: boolean }) {
  const p = g.players[pid]!;
  if (now && g.phase === 'pick') return <span class="foe-none">👑 choisit un peuple…</span>;
  if (p.declined.includes('ghouls')) {
    const rest = single ? p.declined.filter((r) => r !== 'ghouls') : [];
    return (
      <span class="foe-tags">
        <span class="foe-zz">💤</span>
        <Tag t={{ race: 'ghouls', icon: RACES.ghouls.icon, tag: RACES.ghouls.tag, kind: RACES.ghouls.kind }} />
        {rest.length > 0 && <OldPeoples races={rest} />}
      </span>
    );
  }
  return (
    <span class="foe-none">
      sans peuple actif
      {p.declined.length > 0 && (
        <>
          {' '}💤<span class="emo foe-gray">{p.declined.map((r) => RACES[r].icon).join('')}</span>
        </>
      )}
    </span>
  );
}

/** Emojis gris des peuples en déclin, masqués entièrement s'ils ne tiennent pas (jamais coupés). */
function OldPeoples({ races }: { races: RaceId[] }) {
  return (
    <span class="foe-old">
      <span class="foe-old-in">
        💤<span class="emo foe-gray">{races.map((r) => RACES[r].icon).join('')}</span>
      </span>
    </span>
  );
}

/** Classe d'animation « pivot » quand le joueur prend un nouveau peuple (pas au premier affichage). */
function usePivot(race: RaceId | null): string {
  const prev = useRef(race);
  const [n, setN] = useState(0);
  useEffect(() => {
    if (race && race !== prev.current) setN((k) => k + 1);
    prev.current = race;
  }, [race]);
  // deux classes identiques en alternance : changer de classe relance l'animation
  return n === 0 ? '' : n % 2 ? 'pivot-a' : 'pivot-b';
}
