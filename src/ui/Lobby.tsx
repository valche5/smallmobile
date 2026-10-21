import { useEffect, useRef, useState } from 'preact/hooks';
import { AI_LEVELS, PLAYER_COLORS } from '../engine/data';
import { PRESETS, presetMap } from '../engine/mapgen';
import { randomSeed } from '../engine/rng';
import { EXPERT_LABS } from '../shared/experts';
import { MapPreview } from './components/MapPreview';
import { QrCode } from './components/QrCode';
import { saveMe } from './me';
import { onlineQuit, onlineSend, toast, type UIState } from './store';

/** Adresse à partager : si l'hôte a ouvert « localhost », on donne l'adresse du serveur sur le réseau local. */
function useJoinUrl(code: string): string {
  const [base, setBase] = useState(location.origin);
  useEffect(() => {
    if (!['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) return;
    fetch('/api/info')
      .then((r) => r.json() as Promise<{ lan: string[] }>)
      .then((info) => info.lan[0] && setBase(info.lan[0]))
      .catch(() => {});
  }, []);
  return `${base}/?partie=${code}`;
}

export function Lobby({ s }: { s: UIState }) {
  const o = s.online!;
  if (!o.code || !o.room) {
    return (
      <div class="setup scroll">
        <div class="connecting">
          <span class="spinner big" />
          <p>{o.status === 'closed' ? 'Serveur injoignable, nouvelle tentative…' : 'Connexion au serveur…'}</p>
          <button class="btn ghost" onClick={() => onlineQuit(false)}>
            Annuler
          </button>
        </div>
      </div>
    );
  }
  return <Room s={s} />;
}

function Room({ s }: { s: UIState }) {
  const o = s.online!;
  const room = o.room!;
  const me = o.seat!;
  const host = me === 0;
  const url = useJoinUrl(room.code);
  const ready = room.seats.every((seat) => seat.kind === 'ai' || seat.taken);
  const mySeat = room.seats[me]!;
  const [name, setName] = useState(mySeat.name);
  const nameTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [confirmLeave, setConfirmLeave] = useState(false);

  const sendName = (v: string) => {
    setName(v);
    if (nameTimer.current) clearTimeout(nameTimer.current);
    nameTimer.current = setTimeout(() => {
      onlineSend({ t: 'lobby', name: v });
      saveMe({ name: v.trim(), color: mySeat.color });
    }, 500);
  };
  const share = async () => {
    const data = { title: 'SmallMobile', text: `Rejoins ma partie de SmallMobile (code ${room.code})`, url };
    try {
      if (navigator.share) await navigator.share(data);
      else {
        await navigator.clipboard.writeText(url);
        toast('Lien copié !', 'good');
      }
    } catch {
      /* partage annulé */
    }
  };

  return (
    <div class="setup scroll">
      <header class="lobby-head">
        <div class="muted">Partie en ligne · code</div>
        <div class="room-code" aria-label={`Code ${room.code.split('').join(' ')}`}>
          {room.code.split('').map((c, i) => (
            <span key={i}>{c}</span>
          ))}
        </div>
        {o.status !== 'open' && (
          <div class="small bad-text">
            <span class="spinner" /> Connexion perdue, reconnexion…
          </div>
        )}
      </header>

      {!ready && (
        <section class="card invite">
          <h2>📲 Invitez votre adversaire</h2>
          <p class="small">Sur l'autre téléphone : scannez ce code avec l'appareil photo (ou ouvrez le lien), puis entrez un nom.</p>
          <QrCode text={url} label={`QR code de la partie ${room.code}`} />
          <div class="join-link">{url}</div>
          <button class="btn primary wide" onClick={share}>
            📤 Envoyer le lien
          </button>
          <p class="muted small">
            Les deux téléphones doivent joindre ce serveur : même Wi-Fi si le serveur tourne sur un ordinateur de la maison.
          </p>
        </section>
      )}

      <section class="card">
        <h2>Joueurs</h2>
        <ul class="seats">
          {room.seats.map((seat, i) => (
            <li key={i} class={`seat-line ${i === me ? 'mine' : ''}`}>
              <span class="pdot big" style={{ background: seat.color }} />
              <span class="seat-name">
                {seat.kind === 'human' && !seat.taken ? <i class="muted">En attente d'un joueur…</i> : seat.name}
                {i === me && <span class="you-tag dark">vous</span>}
                {i === 0 && <span class="ai-tag">hôte</span>}
                {seat.kind === 'ai' && <span class="ai-tag">IA {seat.ai}</span>}
              </span>
              <span class="seat-status small">
                {seat.kind === 'ai' ? '🤖' : !seat.taken ? <span class="spinner" /> : seat.online ? '🟢 connecté' : '⚪ hors ligne'}
              </span>
            </li>
          ))}
        </ul>
        <div class="seat">
          <div class="seat-row">
            <input
              class="name-input"
              value={name}
              maxLength={14}
              aria-label="Votre nom"
              placeholder="Votre nom"
              onInput={(e) => sendName((e.target as HTMLInputElement).value)}
            />
          </div>
          <div class="colors" role="radiogroup" aria-label="Votre couleur">
            {PLAYER_COLORS.map((c) => (
              <button
                key={c.id}
                class={`color-dot ${mySeat.color === c.color ? 'on' : ''}`}
                style={{ background: c.color }}
                aria-label={c.name}
                onClick={() => {
                  onlineSend({ t: 'lobby', color: c.color });
                  saveMe({ name: name.trim(), color: c.color });
                }}
              />
            ))}
          </div>
        </div>
      </section>

      <section class="card">
        <h2>Carte</h2>
        {host ? (
          <>
            <div class="map-sizes">
              {[2, 3, 4, 5].map((n) => {
                const pr = PRESETS[n]!;
                return (
                  <button key={n} class={`size-opt ${room.size === n ? 'on' : ''}`} onClick={() => onlineSend({ t: 'lobby', size: n })}>
                    <b>{pr.size}</b>
                    <span>{presetMap(n).regions.length} régions</span>
                    <span>{presetMap(n).rounds} tours</span>
                    <span class="ai-count">{n === 2 ? 'Duel' : `+${n - 2} IA`}</span>
                  </button>
                );
              })}
            </div>
            {room.size > 2 && (
              <div class="extra-ai">
                <div class="small">Niveau des IA qui complètent la table :</div>
                <div class="seg">
                  {AI_LEVELS.map((l) => (
                    <button key={l.id} class={room.aiLevel === l.id ? 'on' : ''} onClick={() => onlineSend({ t: 'lobby', aiLevel: l.id })}>
                      {l.label}
                    </button>
                  ))}
                </div>
                {room.aiLevel === 'expert' && EXPERT_LABS.length > 1 && (
                  <div class="seg">
                    {EXPERT_LABS.map((l) => (
                      <button key={l.id} class={room.expertLab === l.id ? 'on' : ''} onClick={() => onlineSend({ t: 'lobby', expertLab: l.id })}>
                        {l.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <p class="small muted">L'hôte choisit la carte. Les changements s'affichent ici en direct.</p>
        )}
        <MapPreview size={room.size} seed={room.mapSeed}>
          {host && (
            <>
              <button class="btn ghost" onClick={() => onlineSend({ t: 'lobby', mapSeed: randomSeed() })}>
                🎲 Autre carte
              </button>
              {room.mapSeed !== PRESETS[room.size]!.seed && (
                <button class="btn ghost" onClick={() => onlineSend({ t: 'lobby', mapSeed: PRESETS[room.size]!.seed })}>
                  ↩ Carte classique
                </button>
              )}
            </>
          )}
        </MapPreview>
      </section>

      {host ? (
        <button class="btn primary big-btn wide" disabled={!ready || o.status !== 'open'} onClick={() => onlineSend({ t: 'start' })}>
          {ready ? '🚀 Lancer la partie' : "En attente de l'autre joueur…"}
        </button>
      ) : (
        <div class="card waiting-start">
          <span class="spinner" /> En attente du lancement par {room.seats[0]!.name}…
        </div>
      )}
      {!confirmLeave ? (
        <button class="btn ghost wide" onClick={() => setConfirmLeave(true)}>
          Quitter la partie
        </button>
      ) : (
        <div class="btn-row leave-row">
          <button class="btn ghost" onClick={() => setConfirmLeave(false)}>
            Rester
          </button>
          <button class="btn danger" onClick={() => onlineQuit(true)}>
            {host ? 'Fermer la partie' : 'Quitter'}
          </button>
        </div>
      )}
    </div>
  );
}
