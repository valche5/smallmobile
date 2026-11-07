import { useEffect, useRef, useState } from 'preact/hooks';
import { RACES, tagsOf, type EffectKind, type TagInfo } from '../../engine/data';
import { actingRace, currentActor, raceOf, regionsOf } from '../../engine/rules';
import type { GameState, RaceId } from '../../engine/types';
import { canAct, setState, viewerSeat, type UIState } from '../store';
import { TagArt } from './ArtIcon';

/** Classe CSS de chaque nature d'effet (couleurs --k-*). */
export const KIND_CLASS: Record<EffectKind, string> = { attaque: 'atk', défense: 'def', spécial: 'spe', revenu: 'inc' };

/** Le joueur dont ce téléphone montre le point de vue : son siège (à défaut, le joueur qui a la main). */
export function mySeat(s: UIState, g: GameState): number {
  const seat = viewerSeat(s);
  if (seat !== null) return seat;
  return currentActor(g) ?? g.current;
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
  const single = n === 2;
  // tous les joueurs dans l'ordre du tour de jeu (le siège 0 joue en premier à chaque manche), moi à ma place ;
  // les adversaires qui se suivent partagent un même bandeau
  const groups: number[][] = [];
  for (let pid = 0; pid < n; pid++) {
    const prev = groups[groups.length - 1];
    if (pid !== me && prev && prev[0] !== me) prev.push(pid);
    else groups.push([pid]);
  }
  return (
    <header class="topbar">
      <div class="me-row">
        <div class={`round ${myTurn ? 'mine' : ''} ${last && !over ? 'last' : ''}`} aria-label={`Tour ${g.round} sur ${g.maxRounds}`}>
          <div class="round-num">
            {over ? (
              '🏁 Fin'
            ) : (
              <>
                Tour {g.round}
                <small>/{g.maxRounds}</small>
              </>
            )}
          </div>
        </div>
        {/* « à vous » : centré entre le compteur de tours et le menu */}
        {myTurn && (
          <div class="my-turn" style={{ '--pc': p.color } as Record<string, string>}>
            à vous
          </div>
        )}
        <div class="stats">
          {acting && (
            <Counter key={`h${me}`} icon="🖐️" value={g.races[acting]?.hand ?? 0} total={inPlay(g, acting)} label="Jetons en main" />
          )}
        </div>
        <button class="menu-btn" aria-label="Menu" onClick={() => setState({ sheet: { kind: 'menu' } })}>
          ☰
        </button>
      </div>
      {groups.map((grp) =>
        grp[0] === me ? (
          // mon peuple : la même puce que celle d'un adversaire seul (nom, puis les 2 étiquettes côte à côte)
          <div key={`me${me}`} class="foes one mine">
            <Foe g={g} pid={me} me={me} now={myTurn} over={over} single self />
          </div>
        ) : (
          <div key={`f${grp[0]}`} class={`foes ${single ? 'one' : 'rows'}`}>
            {grp.map((pid) => (
              <Foe key={pid} g={g} pid={pid} me={me} now={actor === pid && !over} over={over} single={single} />
            ))}
          </div>
        ),
      )}
    </header>
  );
}

/** Les jetons du peuple encore en jeu : ceux en main et ceux sur la carte (baisse avec les pertes). */
function inPlay(g: GameState, race: RaceId): number {
  return (g.races[race]?.hand ?? 0) + regionsOf(g, race).reduce((n, r) => n + g.regions[r]!.tokens, 0);
}

/**
 * Compteur de la ligne du haut : chaque changement fait monter une bulle « +4 » ou « −3 ».
 * `total` : affiché après la valeur (« 5/12 », jetons en main sur jetons en jeu).
 */
function Counter({ icon, value, total, label }: { icon: string; value: number; total?: number; label: string }) {
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
    <div class="stat" title={label} aria-label={total === undefined ? `${label} : ${value}` : `${label} : ${value} sur ${total} en jeu`}>
      <span class="emo">{icon}</span>
      <b>{value}</b>
      {total !== undefined && <small class="stat-total">/{total}</small>}
      {bubble && (
        <span key={bubble.id} class={`bubble ${bubble.d > 0 ? 'up' : 'down'}`} aria-hidden="true">
          {bubble.d > 0 ? `+${bubble.d}` : `−${-bubble.d}`}
        </span>
      )}
    </div>
  );
}

/**
 * Un adversaire : puce de 2 lignes s'il est seul, sinon une rangée. Toucher = sa fiche dans Peuples en jeu.
 * `self` : ma propre puce (toujours de 2 lignes), sans « ▶ joue » : le compteur de tours dit déjà « à vous ».
 */
function Foe({ g, pid, me, now, over, single, self = false }: { g: GameState; pid: number; me: number; now: boolean; over: boolean; single: boolean; self?: boolean }) {
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
    <NoPeople g={g} pid={pid} now={now} single={single} self={self} />
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
            {now && !self && <span class="foe-now">▶ joue</span>}
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
function NoPeople({ g, pid, now, single, self }: { g: GameState; pid: number; now: boolean; single: boolean; self: boolean }) {
  const p = g.players[pid]!;
  if (now && g.phase === 'pick') return <span class="foe-none">{self ? '👑 Choisissez un peuple' : '👑 choisit un peuple…'}</span>;
  if (p.declined.includes('ghouls')) {
    const rest = single ? p.declined.filter((r) => r !== 'ghouls') : [];
    return (
      <>
        <span class="foe-tags">
          <span class="foe-zz">💤</span>
          <Tag t={{ race: 'ghouls', icon: RACES.ghouls.icon, tag: RACES.ghouls.tag, kind: RACES.ghouls.kind }} />
        </span>
        {rest.length > 0 && <OldPeoples races={rest} />}
      </>
    );
  }
  // les peuples en déclin à droite de la ligne, comme quand il y a un peuple actif
  return (
    <>
      <span class="foe-none">sans peuple actif</span>
      {p.declined.length > 0 && <OldPeoples races={p.declined} />}
    </>
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
