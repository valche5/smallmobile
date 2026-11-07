import { createContext, type ComponentChildren } from 'preact';
import { useContext, useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';
import { POWERS, RACES } from '../../engine/data';
import type { Line } from '../../engine/preview';
import type { CostPart } from '../../engine/rules';
import type { Player, PowerId, RaceId } from '../../engine/types';
import { ArtIcon } from './ArtIcon';

export function Lines({ lines }: { lines: Line[] }) {
  return (
    <ul class="lines">
      {lines.map((l, i) => (
        <li key={i} class={`tone-${l.tone ?? 'none'}`}>
          <span class="li-icon">{l.icon}</span>
          <span>{l.text}</span>
        </li>
      ))}
    </ul>
  );
}

export function CostTable({ parts, title, totalLabel }: { parts: CostPart[]; title: string; totalLabel: string }) {
  const total = parts.reduce((s, p) => s + p.value, 0);
  return (
    <div class="cost-table">
      <div class="ct-title">{title}</div>
      {parts.map((p, i) => (
        <div key={i} class={`ct-row ${p.value < 0 ? 'neg' : ''}`}>
          <span>{p.label}</span>
          <span>{p.value > 0 ? `+${p.value}` : p.value}</span>
        </div>
      ))}
      <div class="ct-row total">
        <span>{totalLabel}</span>
        <span>{total}</span>
      </div>
    </div>
  );
}

export function PlayerDot({ player, size = 12 }: { player: Player; size?: number }) {
  return <span class="pdot" style={{ background: player.color, width: `${size}px`, height: `${size}px` }} />;
}

export function PlayerName({ player }: { player: Player }) {
  return (
    <span class="pname">
      <PlayerDot player={player} />
      {player.name}
      {player.kind === 'ai' && <span class="ai-tag">IA</span>}
    </span>
  );
}

export function ComboLabel({ race, power, declined }: { race: RaceId; power?: PowerId; declined?: boolean }) {
  return (
    <span class={`combo-label ${declined ? 'declined' : ''}`}>
      <ArtIcon race={race} />
      {RACES[race].name}
      {power && !declined && (
        <>
          {' '}
          <ArtIcon power={power} />
          {POWERS[power].name}
        </>
      )}
      {declined && <span class="decl-tag">déclin</span>}
    </span>
  );
}

/**
 * Taille d'une feuille :
 * - 'peek' (190 px au plus) se pose au-dessus de la zone d'action, qui reste visible ;
 * - 'standard' (min(45dvh, 380px)) recouvre la zone d'action, au-dessus des onglets ;
 * - 'expanded' (min(62dvh, 580px)) : quand « Détails ▾ » est ouvert.
 * Pendant la lecture du tour adverse, toute feuille se pose au-dessus de la carte d'événement.
 */
export type SheetSize = 'peek' | 'standard' | 'expanded';

interface SheetProps {
  title: ComponentChildren;
  subtitle?: ComponentChildren;
  onClose: () => void;
  children: ComponentChildren;
  footer?: ComponentChildren;
  accent?: string;
  /** Change de valeur quand le contenu change complètement (remonte en haut). */
  scrollKey?: string | number;
  size?: SheetSize;
  /** Classe en plus sur la feuille. */
  class?: string;
}

const Expand = createContext<{ set: (open: boolean) => void; key: string | number | undefined }>({ set: () => {}, key: undefined });

/** La zone d'action affiche-t-elle la carte d'événement d'un adversaire ? */
function showsEvent(bar: Element): boolean {
  return bar.classList.contains('eventcard') || bar.classList.contains('live') || bar.classList.contains('ai');
}

/** Distance entre le bas de l'écran et le haut de l'élément sur lequel la feuille se pose. */
function anchorBottom(size: SheetSize): number {
  const bar = document.querySelector('.app > .actionbar');
  const tabs = document.querySelector('.app > .tabbar');
  let el = bar && (size === 'peek' || showsEvent(bar)) ? bar : tabs;
  // la carte d'un choix de peuple déborde vers le haut : la feuille se pose au-dessus de son vrai sommet
  if (el === bar && bar && showsEvent(bar)) el = bar.querySelector('.ec-inner') ?? bar;
  const top = el ? Math.min(el.getBoundingClientRect().top, el === tabs ? Infinity : bar!.getBoundingClientRect().top) : 0;
  return el ? Math.max(0, Math.round(window.innerHeight - top)) : 0;
}

/** Feuille glissante en bas de l'écran (glisser vers le bas pour fermer). */
export function Sheet({ title, subtitle, onClose, children, footer, accent, scrollKey, size = 'standard', class: cls }: SheetProps) {
  const ref = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  const eff: SheetSize = expanded ? 'expanded' : size;
  const [bottom, setBottom] = useState(0);

  useEffect(() => {
    bodyRef.current?.scrollTo(0, 0);
    setExpanded(false);
  }, [scrollKey]);
  const drag = useRef<{ y: number; dy: number } | null>(null);

  // la feuille suit la zone d'action ou les onglets (carte d'événement qui apparaît, rotation de l'écran…)
  useLayoutEffect(() => {
    const b = anchorBottom(eff);
    if (b !== bottom) setBottom(b);
  });
  useEffect(() => {
    const on = () => setBottom(anchorBottom(eff));
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, [eff]);

  const onDown = (e: PointerEvent) => {
    drag.current = { y: e.clientY, dy: 0 };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onMove = (e: PointerEvent) => {
    if (!drag.current || !ref.current) return;
    drag.current.dy = Math.max(0, e.clientY - drag.current.y);
    ref.current.style.transform = `translateY(${drag.current.dy}px)`;
  };
  const onUp = () => {
    if (!drag.current || !ref.current) return;
    const dy = drag.current.dy;
    drag.current = null;
    ref.current.style.transform = '';
    if (dy > 70) onClose();
  };

  const style: Record<string, string> = { '--sheet-bottom': `${bottom}px` };
  if (accent) style.borderTopColor = accent;
  return (
    <div class={`sheet ${eff} ${bottom ? 'raised' : ''} ${cls ?? ''}`} ref={ref} role="dialog" style={style}>
      <div class="sheet-head" onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        <div class="grab" />
        <div class="sheet-titles">
          <div class="sheet-title">{title}</div>
          {subtitle && <div class="sheet-sub">{subtitle}</div>}
        </div>
        <button class="close" aria-label="Fermer" onPointerDown={(e) => e.stopPropagation()} onClick={onClose}>
          ✕
        </button>
      </div>
      <div class="sheet-body" ref={bodyRef}>
        <Expand.Provider value={{ set: setExpanded, key: scrollKey }}>{children}</Expand.Provider>
      </div>
      {footer && <div class="sheet-foot">{footer}</div>}
    </div>
  );
}

/** « Détails ▾ » : le détail, fermé par défaut ; l'ouvrir agrandit la feuille. */
export function Details({ children, label = 'Détails' }: { children: ComponentChildren; label?: string }) {
  const ctx = useContext(Expand);
  return (
    <details key={ctx.key} class="more" onToggle={(e) => ctx.set((e.currentTarget as HTMLDetailsElement).open)}>
      <summary>{label}</summary>
      <div class="more-body">{children}</div>
    </details>
  );
}

/** Conséquences d'une décision : une ligne visuelle chacune, sans fond coloré. */
export function Consequences({ lines }: { lines: Line[] }) {
  if (!lines.length) return null;
  return (
    <ul class="conseq">
      {lines.map((l, i) => (
        <li key={i} class={`consequence tone-${l.tone ?? 'none'}`}>
          {l.icon && <span class="emo">{l.icon}</span>} {l.text}
        </li>
      ))}
    </ul>
  );
}

export function plural(n: number, word: string, pluralWord?: string): string {
  return `${n} ${n > 1 ? (pluralWord ?? `${word}s`) : word}`;
}

export const LOG_ICONS: Record<string, string> = {
  pick: '👑',
  conquer: '⚔️',
  die: '🎲',
  decline: '💤',
  abandon: '🏳️',
  score: '💰',
  defense: '🛡️',
  power: '✨',
  info: 'ℹ️',
  turn: '▶️',
  end: '🏁',
};
