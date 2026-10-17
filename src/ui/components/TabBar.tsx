import type { GameState } from '../../engine/types';
import { setState, type Tab } from '../store';

const TABS: { id: Tab; icon: string; label: string }[] = [
  { id: 'map', icon: '🗺️', label: 'Carte' },
  { id: 'market', icon: '👑', label: 'Marché' },
  { id: 'log', icon: '📜', label: 'Journal' },
];

/** Trois onglets ; une seule pastille : sur le Marché, quand je dois choisir un peuple. */
export function TabBar({ tab, g, human }: { tab: Tab; g: GameState; human: boolean }) {
  const pickNow = human && g.phase === 'pick';
  return (
    <nav class="tabbar">
      {TABS.map((t) => (
        <button key={t.id} class={`tab ${tab === t.id ? 'on' : ''}`} onClick={() => setState({ tab: t.id, sheet: null })}>
          <span class="tab-icon">
            {t.icon}
            {t.id === 'market' && pickNow && <span class="badge-dot" aria-label="À choisir" />}
          </span>
          <span class="tab-label">{t.label}</span>
        </button>
      ))}
    </nav>
  );
}
