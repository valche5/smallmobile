import { useState } from 'preact/hooks';
import { AI_NAMES, PLAYER_COLORS } from '../engine/data';
import { PRESETS, presetMap } from '../engine/mapgen';
import { randomSeed } from '../engine/rng';
import type { AiLevel } from '../engine/types';
import { normalizeCode } from '../shared/protocol';
import { MapPreview } from './components/MapPreview';
import { loadMe, saveMe, type Me } from './me';
import { forgetSession, hasSave, onlineCreate, onlineJoin, onlineResume, resumeGame, savedSession, startGame, toast } from './store';

interface Seat {
  name: string;
  color: number;
  kind: 'human' | 'ai';
  ai: AiLevel;
}

const LEVELS: { id: AiLevel; label: string }[] = [
  { id: 'facile', label: 'Facile' },
  { id: 'normal', label: 'Normale' },
  { id: 'difficile', label: 'Difficile' },
];

export function Setup() {
  const joinCode = normalizeCode(new URLSearchParams(location.search).get('partie') ?? '');
  const [mode, setMode] = useState<'online' | 'local'>('online');
  const [, refresh] = useState(0);
  const session = savedSession();
  const save = hasSave();

  return (
    <div class="setup scroll">
      <header class="hero">
        <div class="hero-emoji">🗺️⚔️🧌</div>
        <h1>SmallMobile</h1>
        <p>Un monde trop petit pour tous vos peuples. Conquérez, déclinez, recommencez.</p>
      </header>

      {session && !joinCode && (
        <div class="card resume">
          <div>
            <b>Partie en ligne {session.code}</b>
            <div class="muted small">Vous pouvez la reprendre là où vous l'avez laissée.</div>
          </div>
          <div class="resume-buttons">
            <button class="btn primary" onClick={onlineResume}>
              Reprendre
            </button>
            <button
              class="btn ghost small-btn"
              onClick={() => {
                forgetSession();
                refresh((n) => n + 1);
              }}
            >
              Oublier
            </button>
          </div>
        </div>
      )}

      <div class="mode-tabs" role="tablist">
        <button role="tab" aria-selected={mode === 'online'} class={mode === 'online' ? 'on' : ''} onClick={() => setMode('online')}>
          <span class="emo">📱📱</span>
          <b>En ligne</b>
          <span>chacun sur son téléphone</span>
        </button>
        <button role="tab" aria-selected={mode === 'local'} class={mode === 'local' ? 'on' : ''} onClick={() => setMode('local')}>
          <span class="emo">📱</span>
          <b>Sur ce téléphone</b>
          <span>contre l'IA ou en passe-et-joue</span>
        </button>
      </div>

      {mode === 'online' ? <OnlineSetup joinCode={joinCode} /> : <LocalSetup save={save} />}
    </div>
  );
}

// --- En ligne ------------------------------------------------------------------

function OnlineSetup({ joinCode }: { joinCode: string }) {
  const [me, setMe] = useState<Me>(loadMe);
  const [size, setSize] = useState(2);
  const [aiLevel, setAiLevel] = useState<AiLevel>('normal');
  const [customSeed, setCustomSeed] = useState<number | null>(null);
  const [code, setCode] = useState(joinCode);
  const seed = customSeed ?? PRESETS[size]!.seed;

  const update = (patch: Partial<Me>) => {
    const next = { ...me, ...patch };
    setMe(next);
    saveMe(next);
  };
  const name = () => me.name.trim();
  const needName = () => {
    if (name()) return false;
    toast("Indiquez d'abord votre nom.", 'bad');
    return true;
  };
  const join = (c: string) => {
    if (needName()) return;
    if (c.length !== 4) return toast('Le code de partie fait 4 lettres.', 'bad');
    onlineJoin(c, name(), me.color);
  };

  return (
    <>
      <section class="card">
        <h2>Vous</h2>
        <div class="seat">
          <input
            class="name-input"
            value={me.name}
            maxLength={14}
            placeholder="Votre nom"
            aria-label="Votre nom"
            onInput={(e) => update({ name: (e.target as HTMLInputElement).value })}
          />
          <div class="colors" role="radiogroup" aria-label="Votre couleur">
            {PLAYER_COLORS.map((c) => (
              <button
                key={c.id}
                class={`color-dot ${me.color === c.color ? 'on' : ''}`}
                style={{ background: c.color }}
                aria-label={c.name}
                onClick={() => update({ color: c.color })}
              />
            ))}
          </div>
          <p class="muted small">Si votre adversaire prend la même couleur, l'une des deux sera changée automatiquement.</p>
        </div>
      </section>

      {joinCode && (
        <section class="card highlight">
          <h2>🎟️ Invitation reçue</h2>
          <p class="small">
            Invitation à rejoindre la partie <b class="code-inline">{joinCode}</b>. Choisissez votre nom et votre couleur ci-dessus.
          </p>
          <button class="btn primary big-btn wide" onClick={() => join(joinCode)}>
            Rejoindre la partie
          </button>
        </section>
      )}

      <section class="card">
        <h2>Créer une partie</h2>
        <p class="muted small">Vous recevrez un code et un QR code à montrer à votre adversaire.</p>
        <div class="map-sizes">
          {[2, 3, 4, 5].map((n) => {
            const pr = PRESETS[n]!;
            return (
              <button
                key={n}
                class={`size-opt ${size === n ? 'on' : ''}`}
                onClick={() => {
                  setSize(n);
                  setCustomSeed(null);
                }}
              >
                <b>{pr.size}</b>
                <span>{presetMap(n).regions.length} régions</span>
                <span>{presetMap(n).rounds} tours</span>
                <span class="ai-count">{n === 2 ? 'Duel' : `+${n - 2} IA`}</span>
              </button>
            );
          })}
        </div>
        {size > 2 && (
          <div class="extra-ai">
            <div class="small">
              Carte pour {size} joueurs : {size - 2 === 1 ? 'une IA complète' : `${size - 2} IA complètent`} la table (
              {AI_NAMES.slice(0, size - 2).join(', ')}).
            </div>
            <div class="seg">
              {LEVELS.map((l) => (
                <button key={l.id} class={aiLevel === l.id ? 'on' : ''} onClick={() => setAiLevel(l.id)}>
                  {l.label}
                </button>
              ))}
            </div>
          </div>
        )}
        <MapPreview size={size} seed={seed}>
          <button class="btn ghost" onClick={() => setCustomSeed(randomSeed())}>
            🎲 Autre carte
          </button>
          {customSeed !== null && (
            <button class="btn ghost" onClick={() => setCustomSeed(null)}>
              ↩ Carte classique
            </button>
          )}
        </MapPreview>
        <button
          class="btn primary big-btn wide"
          onClick={() => !needName() && onlineCreate({ name: name(), color: me.color, size, aiLevel, mapSeed: seed })}
        >
          Créer la partie
        </button>
      </section>

      {!joinCode && (
        <section class="card">
          <h2>Rejoindre une partie</h2>
          <p class="muted small">Entrez le code à 4 lettres affiché sur le téléphone de votre adversaire, ou scannez son QR code.</p>
          <div class="seat-row">
            <input
              class="name-input code-input"
              value={code}
              placeholder="ABCD"
              aria-label="Code de la partie"
              autocapitalize="characters"
              autocomplete="off"
              spellcheck={false}
              onInput={(e) => setCode(normalizeCode((e.target as HTMLInputElement).value))}
              onKeyDown={(e) => e.key === 'Enter' && join(code)}
            />
            <button class="btn primary" onClick={() => join(code)}>
              Rejoindre
            </button>
          </div>
        </section>
      )}
      <p class="muted small center">
        Les deux téléphones se connectent au même serveur SmallMobile (sur le Wi-Fi de la maison, ou hébergé en ligne).
      </p>
    </>
  );
}

// --- Sur ce téléphone -----------------------------------------------------------

function LocalSetup({ save }: { save: ReturnType<typeof hasSave> }) {
  const [size, setSize] = useState(2);
  const [seats, setSeats] = useState<Seat[]>([
    { name: 'Joueur 1', color: 0, kind: 'human', ai: 'normal' },
    { name: 'Joueur 2', color: 1, kind: 'ai', ai: 'normal' },
  ]);
  const [aiLevel, setAiLevel] = useState<AiLevel>('normal');
  const [customSeed, setCustomSeed] = useState<number | null>(null);
  const seed = customSeed ?? PRESETS[size]!.seed;
  const extra = size - 2;

  const update = (i: number, patch: Partial<Seat>) => {
    setSeats((prev) => {
      const next = prev.map((s) => ({ ...s }));
      if (patch.color !== undefined) {
        const other = next.find((s, j) => j !== i && s.color === patch.color);
        if (other) other.color = next[i]!.color;
      }
      Object.assign(next[i]!, patch);
      return next;
    });
  };

  const start = () => {
    const used = new Set(seats.map((s) => s.color));
    const free = PLAYER_COLORS.map((_, i) => i).filter((i) => !used.has(i));
    const players = [
      ...seats.map((s, i) => ({
        name: s.name.trim() || `Joueur ${i + 1}`,
        color: PLAYER_COLORS[s.color]!.color,
        kind: s.kind,
        ai: s.ai,
      })),
      ...Array.from({ length: extra }, (_, k) => ({
        name: AI_NAMES[k]!,
        color: PLAYER_COLORS[free[k]!]!.color,
        kind: 'ai' as const,
        ai: aiLevel,
      })),
    ];
    startGame({ players, mapSeed: seed, seed: randomSeed() });
  };

  return (
    <>
      {save && (
        <div class="card resume">
          <div>
            <b>Partie en cours sur ce téléphone</b>
            <div class="muted small">
              {save.players.join(' · ')} — tour {save.round}/{save.maxRounds}
            </div>
          </div>
          <button class="btn primary" onClick={resumeGame}>
            Reprendre
          </button>
        </div>
      )}

      <section class="card">
        <h2>Joueurs</h2>
        {seats.map((s, i) => (
          <div key={i} class="seat">
            <div class="seat-row">
              <span class="pdot big" style={{ background: PLAYER_COLORS[s.color]!.color }} />
              <input
                class="name-input"
                value={s.name}
                maxLength={14}
                aria-label={`Nom du joueur ${i + 1}`}
                onInput={(e) => update(i, { name: (e.target as HTMLInputElement).value })}
              />
              <div class="seg">
                <button class={s.kind === 'human' ? 'on' : ''} onClick={() => update(i, { kind: 'human' })}>
                  👤 Humain
                </button>
                <button class={s.kind === 'ai' ? 'on' : ''} onClick={() => update(i, { kind: 'ai' })}>
                  🤖 IA
                </button>
              </div>
            </div>
            <div class="seat-row">
              <div class="colors" role="radiogroup" aria-label="Couleur">
                {PLAYER_COLORS.map((c, ci) => (
                  <button
                    key={c.id}
                    class={`color-dot ${s.color === ci ? 'on' : ''}`}
                    style={{ background: c.color }}
                    aria-label={c.name}
                    onClick={() => update(i, { color: ci })}
                  />
                ))}
              </div>
              {s.kind === 'ai' && (
                <select class="select" value={s.ai} onChange={(e) => update(i, { ai: (e.target as HTMLSelectElement).value as AiLevel })}>
                  {LEVELS.map((l) => (
                    <option key={l.id} value={l.id}>
                      IA {l.label.toLowerCase()}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </div>
        ))}
      </section>

      <section class="card">
        <h2>Carte</h2>
        <div class="map-sizes">
          {[2, 3, 4, 5].map((n) => {
            const pr = PRESETS[n]!;
            return (
              <button
                key={n}
                class={`size-opt ${size === n ? 'on' : ''}`}
                onClick={() => {
                  setSize(n);
                  setCustomSeed(null);
                }}
              >
                <b>{pr.size}</b>
                <span>{presetMap(n).regions.length} régions</span>
                <span>{presetMap(n).rounds} tours</span>
                <span class="ai-count">{n === 2 ? 'Duel' : `+${n - 2} IA`}</span>
              </button>
            );
          })}
        </div>
        {extra > 0 && (
          <div class="extra-ai">
            <div class="small">
              Cette carte est faite pour {size} joueurs : {extra === 1 ? 'une IA complète' : `${extra} IA complètent`} la table (
              {AI_NAMES.slice(0, extra).join(', ')}).
            </div>
            <div class="seg">
              {LEVELS.map((l) => (
                <button key={l.id} class={aiLevel === l.id ? 'on' : ''} onClick={() => setAiLevel(l.id)}>
                  {l.label}
                </button>
              ))}
            </div>
          </div>
        )}
        <MapPreview size={size} seed={seed}>
          <button class="btn ghost" onClick={() => setCustomSeed(randomSeed())}>
            🎲 Autre carte
          </button>
          {customSeed !== null && (
            <button class="btn ghost" onClick={() => setCustomSeed(null)}>
              ↩ Carte classique
            </button>
          )}
        </MapPreview>
      </section>

      <button class="btn primary big-btn wide" onClick={start}>
        Commencer la partie
      </button>
      <p class="muted small center">
        {seats.filter((s) => s.kind === 'human').length > 1
          ? 'Passe-et-joue : un écran vous invitera à passer le téléphone à chaque changement de joueur.'
          : 'Les IA jouent automatiquement ; vous pouvez régler leur vitesse pendant la partie.'}
      </p>
    </>
  );
}
