import { useEffect, useMemo, useState } from 'preact/hooks';
import { ActionBar } from './components/ActionBar';
import { TabBar } from './components/TabBar';
import { mySeat, TopBar } from './components/TopBar';
import { DieOverlay, GameOverOverlay } from './components/Overlays';
import { PeopleSheet } from './components/PeopleSheet';
import { RegionSheet } from './components/RegionSheet';
import { LegendSheet, MenuSheet } from './components/MenuSheet';
import { DeclineSheet, EndTurnSheet } from './components/TurnSheets';
import { canAct, openRegion, setState, viewerSeat, type UIState } from './store';
import { LogView } from './views/LogView';
import { computeMarks } from './views/marks';
import { MapView } from './views/MapView';
import { MarketView } from './views/MarketView';
import { RulesView } from './views/RulesView';

const NO_COVER = { bottom: 0, right: 0 };
const WIDE = '(min-width: 760px)';

/** Sur grand écran, les feuilles s'affichent en panneau à droite au lieu d'en bas. */
function useWide(): boolean {
  const [wide, setWide] = useState(() => window.matchMedia(WIDE).matches);
  useEffect(() => {
    const mq = window.matchMedia(WIDE);
    const on = () => setWide(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return wide;
}

export function GameScreen({ s }: { s: UIState }) {
  const g = s.game!;
  // « human » : ce téléphone a la main
  const human = canAct(s);
  // le joueur dont on montre le point de vue (ligne du haut, Peuples en jeu, Journal)
  const me = mySeat(s, g);
  const [sheetH, setSheetH] = useState(0);
  const wide = useWide();
  const viewer = viewerSeat(s);
  const marks = useMemo(() => computeMarks(g, viewer, human), [g, viewer, human]);

  // Pendant le tour adverse, on ne garde ouvertes que les fenêtres d'information
  useEffect(() => {
    if (!human && s.sheet && ['decline', 'endTurn'].includes(s.sheet.kind)) setState({ sheet: null });
  }, [human]);

  // l'écran de fin s'ouvre une fois la dernière action affichée (la trame « Fin de la partie » d'abord)
  const busy = !!s.playback;
  useEffect(() => {
    if (g.phase === 'gameOver' && !busy) setState({ podium: true });
  }, [g.phase, busy]);

  const selected = s.sheet?.kind === 'region' ? s.sheet.region : null;
  const closeSheet = () => setState({ sheet: null });

  let sheet = null;
  if (s.sheet) {
    const common = { g, onClose: closeSheet, onHeight: setSheetH };
    switch (s.sheet.kind) {
      case 'region':
        sheet = <RegionSheet {...common} region={s.sheet.region} human={human} />;
        break;
      case 'decline':
        sheet = human && g.phase === 'conquest' && <DeclineSheet {...common} />;
        break;
      case 'endTurn':
        sheet = human && g.phase === 'redeploy' && <EndTurnSheet {...common} />;
        break;
      case 'legend':
        sheet = <LegendSheet onClose={closeSheet} onHeight={setSheetH} />;
        break;
      case 'people':
        sheet = <PeopleSheet key={s.sheet.player} {...common} player={s.sheet.player} me={me} />;
        break;
      case 'menu':
        sheet = <MenuSheet onClose={closeSheet} onHeight={setSheetH} />;
        break;
    }
  }

  let overlay = null;
  if (s.overlay?.kind === 'die') overlay = <DieOverlay key={s.overlay.event.seq} ev={s.overlay.event} auto={s.overlay.auto} />;
  else if (g.phase === 'gameOver' && s.podium)
    overlay = <GameOverOverlay g={g} me={viewerSeat(s)} onClose={() => setState({ podium: false })} />;

  return (
    <div class="app">
      <TopBar g={g} s={s} />
      <main class="main">
        <div class={`pane ${s.tab === 'map' ? '' : 'hidden'}`}>
          <MapView
            game={g}
            selected={selected}
            marks={marks}
            showCosts
            pulse={g.lastRegion}
            focus={selected !== null ? s.focus : null}
            follow={s.follow}
            cover={!sheetH ? NO_COVER : wide ? { bottom: 0, right: 450 } : { bottom: sheetH, right: 0 }}
            onTap={(r) => openRegion(r)}
            onLegend={() => setState({ sheet: { kind: 'legend' } })}
            onToggleCosts={() => undefined}
          />
        </div>
        {s.tab === 'market' && <MarketView g={g} human={human} />}
        {s.tab === 'log' && <LogView g={g} me={me} />}
        {s.tab === 'rules' && <RulesView />}
      </main>
      <ActionBar s={s} g={g} human={human} />
      <TabBar tab={s.tab} g={g} human={human} />
      {sheet}
      {overlay}
    </div>
  );
}
