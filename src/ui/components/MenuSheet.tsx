import { useState } from 'preact/hooks';
import { FEATURE_ICONS, TERRAINS } from '../../engine/data';
import { getState, onlineQuit, onlineSend, quitGame, setState, type AiSpeed } from '../store';
import { Sheet } from './common';
import { ArtIcon } from './ArtIcon';
import type { Terrain } from '../../engine/types';

interface Base {
  onClose: () => void;
  onHeight: (h: number) => void;
}

export function LegendSheet({ onClose, onHeight }: Base) {
  return (
    <Sheet onClose={onClose} onHeight={onHeight} title="Légende de la carte">
      <div class="legend-grid">
        {Object.entries(TERRAINS).map(([k, t]) => (
          <div key={k} class="legend-item">
            <span class="swatch" style={{ background: t.color }} />
            <ArtIcon terrain={k as Terrain} size={32} /> {t.name}
            {k === 'mountain' && <span class="muted small"> (+1 défense)</span>}
            {(k === 'sea' || k === 'lake') && <span class="muted small"> (infranchissable)</span>}
          </div>
        ))}
      </div>
      <div class="legend-grid">
        <div class="legend-item">{FEATURE_ICONS.cavern} Caverne</div>
        <div class="legend-item">{FEATURE_ICONS.magic} Source magique</div>
        <div class="legend-item">{FEATURE_ICONS.mine} Mine</div>
        <div class="legend-item">{FEATURE_ICONS.lost} Tribu oubliée</div>
        <div class="legend-item">{FEATURE_ICONS.fortress} Forteresse (+1)</div>
        <div class="legend-item">{FEATURE_ICONS.lair} Tanière (+1)</div>
        <div class="legend-item">{FEATURE_ICONS.camp} Campement (+1)</div>
        <div class="legend-item">{FEATURE_ICONS.hero} Héros (imprenable)</div>
        <div class="legend-item">{FEATURE_ICONS.dragon} Dragon (imprenable)</div>
        <div class="legend-item">{FEATURE_ICONS.hole} Terrier (imprenable)</div>
      </div>
      <div class="legend-list">
        <div>
          <span class="cost-demo c-normal">⚔3</span> Conquête possible : coût en jetons
        </div>
        <div>
          <span class="cost-demo c-die">🎲33%</span> Possible seulement avec le dé de renfort
        </div>
        <div>
          <span class="cost-demo c-special">🐉1</span> Pouvoir spécial (dragon, sorcellerie)
        </div>
        <div>
          <span class="cost-demo c-threat">⚔4</span> Redéploiement : ce qu'un ennemi paiera pour vous prendre la région
        </div>
        <div>
          <span class="cost-demo c-threat hot">⚔2</span> Prise facile (3 jetons ou moins, dragon, sorcellerie)
        </div>
        <div>
          <span class="lost-demo" /> Région qu'on vous a prise depuis votre dernier décompte
        </div>
        <div>
          <span class="token-demo" /> Jetons d'un joueur (pastille grise = peuple en déclin)
        </div>
      </div>
      <p class="muted small">Touchez une région pour la détailler.</p>
    </Sheet>
  );
}

export function MenuSheet({ onClose, onHeight }: Base) {
  const s = getState();
  const online = s.online;
  const speeds: { id: AiSpeed; label: string }[] = [
    { id: 'lent', label: '🐢 Lente' },
    { id: 'normal', label: '🚶 Normale' },
    { id: 'rapide', label: '⚡ Rapide' },
  ];
  const speed = online?.room?.speed ?? s.aiSpeed;
  const setSpeed = (id: AiSpeed) => (online ? onlineSend({ t: 'speed', speed: id }) : setState({ aiSpeed: id }));
  const [confirmQuit, setConfirmQuit] = useState(false);
  return (
    <Sheet onClose={onClose} onHeight={onHeight} title="Menu">
      {online && (
        <div class="power-block">
          <div class="option-title">Partie en ligne {online.code}</div>
          <div class="muted small">
            {online.status === 'open' ? '🟢 Connecté au serveur.' : '⚪ Connexion perdue, reconnexion en cours…'} La partie est conservée sur le
            serveur : vous pouvez fermer l'appli et la reprendre plus tard.
          </div>
          <ul class="seats compact">
            {online.room?.seats.map((seat, i) => (
              <li key={i} class="seat-line">
                <span class="pdot" style={{ background: seat.color }} />
                <span class="seat-name">
                  {seat.name}
                  {i === online.seat && <span class="you-tag dark">vous</span>}
                </span>
                <span class="small">
                  {seat.kind === 'ai' ? '🤖 IA' : seat.replaced ? "🤖 joué par l'IA" : seat.online ? '🟢' : '⚪ hors ligne'}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div class="power-block">
        <div class="option-title">
          Vitesse des adversaires <span class="muted small">· pour toute la table</span>
        </div>
        <div class="chips">
          {speeds.map((sp) => (
            <button key={sp.id} class={`chip ${speed === sp.id ? 'on' : ''}`} onClick={() => setSpeed(sp.id)}>
              {sp.label}
            </button>
          ))}
        </div>
      </div>
      <div class="power-block">
        <button class="btn ghost" onClick={() => setState({ sheet: { kind: 'legend' } })}>
          🗺️ Légende de la carte
        </button>
        <button class="btn ghost" onClick={() => setState({ tab: 'rules', sheet: null })}>
          📖 Règles du jeu
        </button>
      </div>
      <div class="power-block">
        <div class="option-title">Partie</div>
        {online ? (
          <>
            <button class="btn ghost" onClick={() => onlineQuit(false)}>
              Revenir à l'accueil (la partie vous attend)
            </button>
            {!confirmQuit ? (
              <button class="btn danger" onClick={() => setConfirmQuit(true)}>
                Abandonner ma place
              </button>
            ) : (
              <>
                <div class="muted small">Vos adversaires pourront confier vos tours à l'IA.</div>
                <div class="btn-row">
                  <button class="btn ghost" onClick={() => setConfirmQuit(false)}>
                    Non, rester
                  </button>
                  <button class="btn danger" onClick={() => onlineQuit(true)}>
                    Oui, abandonner
                  </button>
                </div>
              </>
            )}
          </>
        ) : (
          <>
            <div class="muted small">La partie est sauvegardée automatiquement sur cet appareil.</div>
            {!confirmQuit ? (
              <button class="btn danger" onClick={() => setConfirmQuit(true)}>
                Abandonner et revenir à l'accueil
              </button>
            ) : (
              <div class="btn-row">
                <button class="btn ghost" onClick={() => setConfirmQuit(false)}>
                  Non, continuer
                </button>
                <button class="btn danger" onClick={() => quitGame()}>
                  Oui, effacer la partie
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </Sheet>
  );
}
