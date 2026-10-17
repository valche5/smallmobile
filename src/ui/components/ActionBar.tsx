import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';
import { RACES } from '../../engine/data';
import { endTurnReminder, powerStatus, redeployNote } from '../../engine/preview';
import { actingRace, allTargets, canDecline, currentActor, fullDeployment, raceOf, regionsOf, scoreBreakdown } from '../../engine/rules';
import type { GameState } from '../../engine/types';
import { act, canUndo, onlineSend, setState, undoLast, type UIState } from '../store';
import { plural } from './common';
import { WaitingBar } from './EventCard';
import { needsEndTurnSheet } from './TurnSheets';

/** Durée de la confirmation d'une action irréversible, et délai avant d'accepter le 2e toucher. */
const CONFIRM_MS = 4000;
const SECOND_TAP_MS = 400;

/** Zone d'action (84 px) : une rangée d'indication, une rangée de boutons. */
export function ActionBar({ s, g, human }: { s: UIState; g: GameState; human: boolean }) {
  const online = s.online;
  if (online && online.status !== 'open' && g.phase !== 'gameOver') {
    return (
      <Bar hint={<><span class="spinner" /> ⏳ Connexion perdue, reconnexion…</>} />
    );
  }
  if (g.phase === 'gameOver') {
    // les dernières trames (décompte du dernier joueur, « Fin de la partie ») passent avant le podium
    if (s.playback) return <WaitingBar s={s} g={g} actor={currentActor(g) ?? g.current} />;
    return (
      <Bar hint="🏁 Partie terminée">
        <button class="btn primary main" onClick={() => setState({ podium: true })}>
          🏆 Podium
        </button>
      </Bar>
    );
  }
  const actor = currentActor(g)!;
  if (!human) {
    const seat = online?.room?.seats[actor];
    const away = !!seat && seat.kind === 'human' && !seat.online && !seat.replaced;
    if (away && !s.playback) return <AwayBar g={g} actor={actor} />;
    return <WaitingBar s={s} g={g} actor={actor} />;
  }
  return <MyBar key={`${g.current}:${g.phase}:${g.turn.stage}`} s={s} g={g} />;
}

/** Un adversaire a fermé son téléphone : la partie attend ; au bout de 20 s, on peut confier son tour à l'IA. */
function AwayBar({ g, actor }: { g: GameState; actor: number }) {
  const p = g.players[actor]!;
  const [patience, setPatience] = useState(false);
  useEffect(() => {
    setPatience(false);
    const t = setTimeout(() => setPatience(true), 20000);
    return () => clearTimeout(t);
  }, [actor]);
  return (
    <Bar hint={`⚠️ ${p.name} est hors ligne · la partie reprendra à son retour`}>
      {patience && (
        <button class="btn alt main" onClick={() => onlineSend({ t: 'replace', seat: actor })}>
          🤖 Faire jouer l'IA
        </button>
      )}
    </Bar>
  );
}

/** Gabarit de la zone d'action : indication (une ligne), puis les boutons, le principal à droite. */
function Bar(props: {
  hint?: preact.ComponentChildren;
  tone?: string;
  /** Couleur du joueur, pour le fond du flash « À vous ». */
  color?: string;
  children?: preact.ComponentChildren;
  rowRef?: preact.Ref<HTMLDivElement>;
  squeeze?: boolean;
}) {
  const { hint, tone, color, children, rowRef, squeeze } = props;
  return (
    <div class={`actionbar decide ${squeeze ? 'squeeze' : ''}`}>
      <div class={`hint-row ${tone ?? ''}`} role="status" style={color ? ({ '--pc': color } as Record<string, string>) : undefined}>
        {hint}
      </div>
      <div class="bar-row" ref={rowRef}>
        {children}
      </div>
    </div>
  );
}

interface Btn {
  icon: string;
  label: string;
  /** Ordre de repli en icône seule quand la rangée déborde (1 : le premier replié). */
  fold?: number;
  cls?: string;
  main?: boolean;
  disabled?: boolean;
  onClick: () => void;
}

/** Repli des boutons secondaires en icônes tant que la rangée déborde. */
function useFold(sig: string) {
  const row = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(() => window.innerWidth);
  const [st, setSt] = useState({ sig, width, level: 0 });
  const level = st.sig === sig && st.width === width ? st.level : 0;
  useEffect(() => {
    const on = () => setWidth(window.innerWidth);
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);
  useLayoutEffect(() => {
    const el = row.current;
    if (el && level < 3 && el.scrollWidth > el.clientWidth + 1) setSt({ sig, width, level: level + 1 });
  });
  return { row, level };
}

type ConfirmKind = 'end' | 'berserk' | 'skip';
type Confirm = { kind: ConfirmKind; at: number } | null;

/** Mon tour : indication de la phase, ↶ et 3 boutons au plus. */
function MyBar({ s, g }: { s: UIState; g: GameState }) {
  const [confirm, setConfirm] = useState<Confirm>(null);
  // la confirmation ne tombe que si ce qu'elle annonce change (pas à chaque accusé de réception du serveur)
  const rs = actingRace(g) ? raceOf(g, actingRace(g)!) : null;
  const what = `${g.current}|${g.phase}|${g.turn.stage}|${scoreBreakdown(g, g.current).total}|${rs?.hand ?? ''}|${g.turn.declined}|${g.turn.conquests}`;
  useEffect(() => setConfirm(null), [what]);
  useEffect(() => {
    if (!confirm) return;
    const t = setTimeout(() => setConfirm(null), CONFIRM_MS);
    return () => clearTimeout(t);
  }, [confirm]);
  const arm = (kind: ConfirmKind) => setConfirm({ kind, at: Date.now() });
  // le 2e toucher n'est pris qu'après un court délai : un double toucher accidentel ne confirme rien
  const second = (fn: () => void) => () => {
    if (confirm && Date.now() - confirm.at >= SECOND_TAP_MS) {
      setConfirm(null);
      fn();
    }
  };

  const plan = planBar(s, g, arm);
  let hint: preact.ComponentChildren = plan.hint;
  let tone = '';
  let buttons = plan.buttons;
  if (confirm) {
    tone = 'confirm';
    if (confirm.kind === 'berserk') {
      hint = '⚠️ Après le dé : plus de déclin ce tour';
      buttons = [
        { icon: '', label: 'Retour', cls: 'ghost', onClick: () => setConfirm(null) },
        { icon: '🎲', label: 'Lancer', cls: 'primary', main: true, onClick: second(() => act({ type: 'berserkRoll' })) },
      ];
    } else if (confirm.kind === 'skip') {
      hint = `⚠️ Aucune attaque : ${plural(rs?.hand ?? 0, 'jeton')} en main`;
      buttons = [
        { icon: '⚔️', label: 'Attaquer', cls: 'ghost', onClick: () => setConfirm(null) },
        { icon: '', label: 'Redéployer →', cls: 'primary', main: true, onClick: second(() => act({ type: 'endConquest' })) },
      ];
    } else {
      hint = endTurnReminder(g) ?? 'Terminer le tour ?';
      buttons = [
        { icon: '', label: 'Retour', cls: 'ghost', onClick: () => setConfirm(null) },
        { icon: '✓', label: `Terminer · +${scoreBreakdown(g, g.current).total} 💰`, cls: 'primary', main: true, onClick: second(() => act({ type: 'endTurn' })) },
      ];
    }
  } else if (s.flash) {
    tone = `flash ${s.flash.tone}`;
    hint = (
      <span key={s.flash.id} class="flash-text">
        <span class="emo">{s.flash.icon}</span> {s.flash.text}
      </span>
    );
  }
  const undo = !confirm && canUndo(s);
  const sig = `${undo}|${buttons.map((b) => `${b.icon}${b.label}`).join('|')}`;
  const { row, level } = useFold(sig);
  return (
    <Bar hint={hint} tone={tone} color={s.flash?.color} rowRef={row} squeeze={level >= 3}>
      {undo && (
        <button class="btn ghost icon-btn undo" aria-label="Annuler la dernière action" onClick={undoLast}>
          ↶
        </button>
      )}
      {buttons.map((b) => {
        const folded = !!b.fold && level >= b.fold;
        const text = b.icon ? `${b.icon} ${b.label}` : b.label;
        return (
          <button
            key={b.label}
            class={`btn ${b.cls ?? 'ghost'} ${b.main ? 'main' : ''} ${folded ? 'folded' : ''}`}
            disabled={b.disabled}
            aria-label={folded ? text : undefined}
            onClick={b.onClick}
          >
            {folded ? b.icon : text}
          </button>
        );
      })}
    </Bar>
  );
}

/** Indication et boutons de la phase en cours (hors confirmation). */
function planBar(s: UIState, g: GameState, arm: (kind: ConfirmKind) => void): { hint: preact.ComponentChildren; buttons: Btn[] } {
  if (g.phase === 'pick') {
    const buttons: Btn[] = [];
    if (!g.market.length) buttons.push({ icon: '', label: 'Passer', cls: 'primary', main: true, onClick: () => act({ type: 'skipPick' }) });
    else if (s.tab !== 'market')
      buttons.push({ icon: '👑', label: 'Voir le marché', cls: 'primary', main: true, onClick: () => setState({ tab: 'market', sheet: null }) });
    return { hint: '👑 Choisissez un peuple', buttons };
  }

  const race = actingRace(g)!;
  const rs = raceOf(g, race);

  if (g.phase === 'conquest') {
    const targets = allTargets(g);
    const n = targets.filter((t) => t.kind === 'normal' || t.dragon || t.sorcery).length;
    const die = targets.some((t) => t.kind === 'die');
    const berserkReady = !rs.declined && rs.power === 'berserk' && g.turn.berserkDie === null && rs.hand > 0 && !g.turn.dieRolled;
    // Berserks : une cible trop chère peut encore tomber grâce au dé (jusqu'à −3)
    const berserkHope = berserkReady && targets.some((t) => t.reachable && Math.max(1, t.cost - 3) <= rs.hand);
    const more = n > 0 || die || berserkHope;
    const ghouls = g.turn.stage === 'ghouls';
    const declinable = !canDecline(g);
    let hint: string | null;
    if (ghouls) hint = `${RACES.ghouls.icon} Vos Goules en déclin jouent d'abord`;
    else if (!regionsOf(g, race).length) hint = rs.power === 'flying' || race === 'halflings' ? '🧭 Entrez où vous voulez' : '🧭 Entrez par le bord de la carte';
    else if (!more) hint = 'Plus de conquête possible';
    else if (!n && die) hint = '🎲 Plus que le dé de renfort';
    else if (!n && berserkHope) hint = '🎲 Lancez le dé berserk pour continuer';
    else {
      // dit toujours qu'on attaque : les jetons en main au début du tour ne sont pas « à redéployer »
      const status = powerStatus(g, g.current);
      hint = status ? `⚔️ Attaquez · ${status}` : `⚔️ Attaquez une région · ${plural(rs.hand, 'jeton')}`;
    }
    // partir sans avoir rien tenté alors qu'une conquête est possible : on le fait confirmer
    const skipping = more && !ghouls && !g.turn.conquests && !g.turn.dieRolled;
    const buttons: Btn[] = [];
    if (declinable) buttons.push({ icon: '💤', label: 'Déclin', fold: 1, onClick: () => setState({ sheet: { kind: 'decline' } }) });
    if (berserkReady)
      buttons.push({ icon: '🎲', label: 'Dé', fold: 2, cls: 'alt', onClick: () => (declinable ? arm('berserk') : act({ type: 'berserkRoll' })) });
    buttons.push({
      icon: '',
      label: ghouls ? 'Fin des Goules →' : more ? 'Fin des attaques →' : 'Redéployer →',
      cls: more ? 'ghost' : 'primary',
      main: true,
      onClick: () => (skipping ? arm('skip') : act({ type: 'endConquest' })),
    });
    return { hint, buttons };
  }

  const own = regionsOf(g, race).length;
  const mustPlace = rs.hand > 0 && own > 0;
  const auto: Btn = { icon: '🪄', label: 'Auto', fold: 3, onClick: () => act({ type: 'autoDeploy' }) };

  if (g.phase === 'defend') {
    return {
      hint: mustPlace ? `🛡️ Replacez ${plural(rs.hand, 'jeton')} ${RACES[race].name}` : '🛡️ Tout est replacé',
      buttons: [...(mustPlace ? [auto] : []), { icon: '', label: 'Valider', cls: 'primary', main: true, disabled: mustPlace, onClick: () => act({ type: 'endDefense' }) }],
    };
  }

  // redéploiement : « Auto » reprend tous les jetons et les replace, donc utile même la main vide
  const target = own ? fullDeployment(g, race) : {};
  const placeAll: Btn = { ...auto, onClick: () => act({ type: 'autoDeploy', all: true }) };
  const canImprove = mustPlace || regionsOf(g, race).some((r) => g.regions[r]!.tokens !== target[r]);
  const place = mustPlace ? `🔁 ${plural(rs.hand, 'jeton')} à placer` : own ? '✓ Tout est placé' : '🔁 Aucune région où placer';
  if (g.turn.stage === 'ghouls') {
    return {
      hint: `${RACES.ghouls.icon} ${place}`,
      buttons: [...(canImprove ? [placeAll] : []), { icon: '', label: 'Peuple actif →', cls: 'primary', main: true, disabled: mustPlace, onClick: () => act({ type: 'endTurn' }) }],
    };
  }
  const note = redeployNote(g);
  const status = powerStatus(g, g.current);
  const head = note ? `${note} · ${mustPlace ? `🔁 ${rs.hand} à placer` : place}` : place;
  const income = scoreBreakdown(g, g.current).total;
  return {
    hint: status ? `${head} · ${status}` : head,
    buttons: [
      ...(canImprove ? [placeAll] : []),
      {
        icon: '',
        label: `Fin du tour · +${income} 💰`,
        cls: 'primary',
        main: true,
        disabled: mustPlace,
        onClick: () => (needsEndTurnSheet(g) ? setState({ sheet: { kind: 'endTurn' } }) : arm('end')),
      },
    ],
  };
}
