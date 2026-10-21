import { useState } from 'preact/hooks';
import { AI_LEVELS, AI_NAMES, PLAYER_COLORS } from '../engine/data';
import { PRESETS, presetMap } from '../engine/mapgen';
import { randomSeed } from '../engine/rng';
import type { AiLevel } from '../engine/types';
import { DEFAULT_EXPERT_LAB, EXPERT_LABS } from '../shared/experts';
import { normalizeCode } from '../shared/protocol';
import { MapPreview } from './components/MapPreview';
import { loadMe, saveMe, type Me } from './me';
import { forgetSession, onlineCreate, onlineJoin, onlineResume, savedSession, toast } from './store';

export function Setup() {
  const joinCode = normalizeCode(new URLSearchParams(location.search).get('partie') ?? '');
  const [, refresh] = useState(0);
  const session = savedSession();

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
            <b>Partie en cours ({session.code})</b>
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

      <OnlineSetup joinCode={joinCode} />
    </div>
  );
}

// --- Nouvelle partie (toujours sur le serveur : il fait jouer les IA et garde les parties) ---------

function OnlineSetup({ joinCode }: { joinCode: string }) {
  const [me, setMe] = useState<Me>(loadMe);
  const [size, setSize] = useState(2);
  const [opponent, setOpponent] = useState<'human' | 'ai'>('ai');
  const [aiLevel, setAiLevel] = useState<AiLevel>('normal');
  const [expertLab, setExpertLab] = useState(DEFAULT_EXPERT_LAB);
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
          {opponent === 'human' && (
            <p class="muted small">Si votre adversaire prend la même couleur, l'une des deux sera changée automatiquement.</p>
          )}
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
        <h2>Nouvelle partie</h2>
        <div class="seg wide-seg" role="radiogroup" aria-label="Adversaire">
          <button class={opponent === 'ai' ? 'on' : ''} onClick={() => setOpponent('ai')}>
            🤖 Contre l'IA
          </button>
          <button class={opponent === 'human' ? 'on' : ''} onClick={() => setOpponent('human')}>
            👤 Contre un ami
          </button>
        </div>
        {opponent === 'human' && <p class="muted small">Vous recevrez un code et un QR code à montrer à votre adversaire.</p>}
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
                <span class="ai-count">{opponent === 'ai' ? `${n - 1} IA` : n === 2 ? 'Duel' : `+${n - 2} IA`}</span>
              </button>
            );
          })}
        </div>
        {(opponent === 'ai' || size > 2) && (
          <div class="extra-ai">
            <div class="small">
              {opponent === 'ai'
                ? `Vous contre ${AI_NAMES.slice(0, size - 1).join(', ')}.`
                : `Carte pour ${size} joueurs : ${size - 2 === 1 ? 'une IA complète' : `${size - 2} IA complètent`} la table (${AI_NAMES.slice(0, size - 2).join(', ')}).`}
            </div>
            <div class="seg">
              {AI_LEVELS.map((l) => (
                <button key={l.id} class={aiLevel === l.id ? 'on' : ''} onClick={() => setAiLevel(l.id)}>
                  {l.label}
                </button>
              ))}
            </div>
            {aiLevel === 'expert' && EXPERT_LABS.length > 1 && (
              <div class="seg">
                {EXPERT_LABS.map((l) => (
                  <button key={l.id} class={expertLab === l.id ? 'on' : ''} onClick={() => setExpertLab(l.id)}>
                    {l.label}
                  </button>
                ))}
              </div>
            )}
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
          onClick={() => !needName() && onlineCreate({ name: name(), color: me.color, size, aiLevel, expertLab, mapSeed: seed, opponent })}
        >
          {opponent === 'ai' ? 'Jouer' : 'Créer la partie'}
        </button>
        {aiLevel === 'expert' && (opponent === 'ai' || size > 2) && (
          <p class="muted small">L'IA Expert vient du labo : plus forte que Difficile, elle réfléchit un peu plus longtemps.</p>
        )}
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
        Les parties se jouent sur le serveur SmallMobile (sur le Wi-Fi de la maison, ou hébergé en ligne), qui les garde pour améliorer l'IA.
      </p>
    </>
  );
}
