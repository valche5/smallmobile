import { render } from 'preact';
import { initEngine } from './engine/engine';
import { App } from './ui/App';

// Le manifeste (installation sur l'écran d'accueil) est servi à part, hors du bundle.
const manifest = document.createElement('link');
manifest.rel = 'manifest';
manifest.href = 'manifest.webmanifest';
document.head.append(manifest);

// le noyau du jeu (règles, IA, cartes) est chargé avant tout le reste
initEngine(fetch('game-kernel.wasm').then((r) => r.arrayBuffer())).then(
  () => render(<App />, document.getElementById('app')!),
  (e) => {
    document.getElementById('app')!.textContent = `Le jeu n'a pas pu démarrer : ${e instanceof Error ? e.message : e}`;
  },
);
