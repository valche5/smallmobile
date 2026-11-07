import { useState } from 'preact/hooks';
import { KIND_LABEL, KIND_ORDER, POWERS, RACES, type EffectKind } from '../../engine/data';
import * as preview from '../../engine/preview';
import { currentActor, regionsOf, scoreBreakdown } from '../../engine/rules';
import type { GameState, PowerId, RaceId } from '../../engine/types';
import { plural, Sheet } from './common';
import { KIND_CLASS, playerTags } from './TopBar';
import { ScoreTable } from './TurnSheets';
import { ArtIcon, TagArt, type ArtSubject } from './ArtIcon';

interface Props {
  g: GameState;
  /** Joueur touché : il vient en premier, déplié. */
  player: number;
  /** Le joueur de ce téléphone (même règle que la ligne du haut). */
  me: number;
  onClose: () => void;
}

/** État variable de mon pouvoir (fourni par preview.ts ; absent tant qu'il n'existe pas). */
const powerStatus = (preview as unknown as { powerStatus?: (g: GameState, pid: number) => string | null }).powerStatus;

/** Peuples en jeu : chaque joueur, ses pouvoirs en clair, vus de l'adversaire ou de moi. */
export function PeopleSheet({ g, player, me, onClose }: Props) {
  const n = g.players.length;
  const order = Array.from({ length: n }, (_, i) => (player + i) % n);
  const [open, setOpen] = useState<number[]>([player]);
  const toggle = (pid: number) => setOpen((o) => (o.includes(pid) ? o.filter((x) => x !== pid) : [...o, pid]));
  return (
    <Sheet onClose={onClose} title="Peuples en jeu" accent={g.players[player]!.color} scrollKey={player}>
      <div class="people">
        {order.map((pid) => (
          <PlayerBlock key={pid} g={g} pid={pid} me={me} open={open.includes(pid)} onToggle={() => toggle(pid)} />
        ))}
      </div>
    </Sheet>
  );
}

function PlayerBlock({ g, pid, me, open, onToggle }: { g: GameState; pid: number; me: number; open: boolean; onToggle: () => void }) {
  const p = g.players[pid]!;
  const mine = pid === me;
  const now = g.phase !== 'gameOver' && currentActor(g) === pid;
  const name = mine ? 'Vous' : `${p.name}${p.kind === 'ai' ? ' (IA)' : ''}`;
  const tags = playerTags(g, pid);
  const regions = [p.active, ...p.declined].filter((r): r is RaceId => !!r).reduce((s, r) => s + regionsOf(g, r).length, 0);
  return (
    <section class={`pblock ${open ? 'open' : ''}`} style={{ '--pc': p.color } as Record<string, string>}>
      <button class="pblock-head" aria-expanded={open} onClick={onToggle}>
        <span class="pdot" style={{ background: p.color }} />
        <span class="pblock-line">
          <b>{name}</b>
          {g.phase === 'gameOver' && ` · ${p.coins} 💰`}
          {open ? (
            <>
              {' · '}
              {plural(regions, 'région')}
              {now && (mine ? ' · ▶ à vous' : ' · ▶ joue')}
            </>
          ) : tags.length ? (
            tags.map((t) => (
              <span key={t.icon} class="pblock-tag">
                {' · '}
                <span class="emo"><TagArt t={t} size={16} /></span> {t.tag}
              </span>
            ))
          ) : (
            <span class="muted"> · {now && g.phase === 'pick' ? '👑 choisit un peuple…' : 'sans peuple actif'}</span>
          )}
        </span>
        <span class="chev">{open ? '▾' : '▸'}</span>
      </button>
      {open && <PlayerDetail g={g} pid={pid} me={me} />}
    </section>
  );
}

interface Effect {
  art: ArtSubject;
  kind: EffectKind;
  icon: string;
  name: string;
  text: string;
}

/** Le peuple et le pouvoir actifs, dans l'ordre des étiquettes (attaque, défense, spécial, revenu). */
function effects(race: RaceId, power: PowerId, mine: boolean): Effect[] {
  const r = RACES[race];
  const pw = POWERS[power];
  return [
    { art: { race }, kind: r.kind, icon: r.icon, name: r.name, text: mine ? r.short : r.foe },
    { art: { power }, kind: pw.kind, icon: pw.icon, name: pw.name, text: mine ? pw.short : pw.foe },
  ].sort((a, b) => KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind));
}

function PlayerDetail({ g, pid, me }: { g: GameState; pid: number; me: number }) {
  const p = g.players[pid]!;
  const mine = pid === me;
  const race = p.active;
  const power = race ? g.races[race]!.power : null;
  const status = mine ? (powerStatus?.(g, pid) ?? null) : foeStatus(g, pid, me);
  const income = mine ? scoreBreakdown(g, pid) : null;
  return (
    <div class="pblock-body">
      {race && power ? (
        effects(race, power, mine).map((e) => (
          <div key={e.icon} class="peffect">
            <span class={`kind k-${KIND_CLASS[e.kind]}`}>{KIND_LABEL[e.kind]}</span>
            <span class="peffect-text">
              <ArtIcon {...e.art} size={28} /> <b>{e.name}</b> — {e.text}
            </span>
          </div>
        ))
      ) : (
        <div class="peffect muted">{p.declined.length ? 'Pas de peuple actif.' : 'Pas encore de peuple.'}</div>
      )}
      {p.declined.map((r) => (
        <div key={r} class="peffect declined">
          <span class="kind k-old">En déclin</span>
          <span class="peffect-text">
            <ArtIcon race={r} /> <b>{RACES[r].name}</b> — {RACES[r].inDecline ?? 'plus aucun effet'} · {plural(regionsOf(g, r).length, 'région')}
          </span>
        </div>
      ))}
      {status && <div class="pstatus">{status}</div>}
      {income && (
        <details class="pmore">
          <summary>Revenu de ce tour : +{income.total} 💰</summary>
          <ScoreTable lines={income.lines} total={income.total} />
        </details>
      )}
      {(race || p.declined.length > 0) && (
        <details class="pmore">
          <summary>Règle complète</summary>
          {race && power && (
            <>
              <p class="prule">
                <ArtIcon race={race} /> <b>{RACES[race].name}</b> : {RACES[race].desc}
              </p>
              <p class="prule">
                <ArtIcon power={power} /> <b>{POWERS[power].name}</b> : {POWERS[power].desc}
              </p>
            </>
          )}
          {p.declined
            .filter((r) => RACES[r].inDecline)
            .map((r) => (
              <p key={r} class="prule">
                <ArtIcon race={r} /> <b>{RACES[r].name}</b> : {RACES[r].desc}
              </p>
            ))}
        </details>
      )}
    </div>
  );
}

/** Nom de région coupé à 18 caractères. */
function cut(name: string): string {
  const c = [...name];
  return c.length > 18 ? `${c.slice(0, 17).join('')}…` : name;
}

/** Une seule ligne d'état pour un adversaire, si elle existe : paix, dragon, héros. */
function foeStatus(g: GameState, pid: number, me: number): string | null {
  if (g.peace.some((x) => x.diplomat === pid && x.target === me)) return '🕊️ Vous impose la paix ce tour';
  if (g.peace.some((x) => x.diplomat === me && x.target === pid)) return '🕊️ Ne peut pas vous attaquer (paix)';
  const race = g.players[pid]!.active;
  if (!race) return null;
  const own = regionsOf(g, race);
  const dragon = own.find((i) => g.regions[i]!.dragon);
  if (dragon !== undefined) return `🐉 Dragon sur « ${cut(g.map.regions[dragon]!.name)} »`;
  const heroes = own.filter((i) => g.regions[i]!.hero).map((i) => `« ${cut(g.map.regions[i]!.name)} »`);
  if (heroes.length) return `🦸 Héros sur ${heroes.join(' et ')}`;
  return null;
}
