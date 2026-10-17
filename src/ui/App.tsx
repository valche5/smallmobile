import { useEffect } from 'preact/hooks';
import { normalizeCode } from '../shared/protocol';
import { GameScreen } from './Game';
import { Lobby } from './Lobby';
import { Setup } from './Setup';
import { onlineResume, savedSession, useStore, wantsHome } from './store';

export function App() {
  const s = useStore();

  // Une partie en ligne était en cours sur ce téléphone : on s'y reconnecte directement
  useEffect(() => {
    const session = savedSession();
    const invited = normalizeCode(new URLSearchParams(location.search).get('partie') ?? '');
    if (session && !wantsHome() && (!invited || invited === session.code)) onlineResume();
  }, []);

  let screen;
  if (s.online) screen = s.game && s.online.map ? <GameScreen s={s} /> : <Lobby s={s} />;
  else screen = s.game ? <GameScreen s={s} /> : <Setup />;

  return (
    <>
      {screen}
      {s.toast && (
        <div key={s.toast.id} class={`toast ${s.toast.tone}`} role="status">
          {s.toast.text}
        </div>
      )}
    </>
  );
}
