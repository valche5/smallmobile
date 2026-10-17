import { useState } from 'preact/hooks';
import type { ComponentChildren } from 'preact';
import { FEATURE_ICONS, POWER_IDS, POWERS, RACE_IDS, RACES, TERRAINS } from '../../engine/data';
import { setState } from '../store';
import { ArtIcon } from '../components/ArtIcon';
import type { Terrain } from '../../engine/types';

function Section({ title, children, open = false }: { title: string; children: ComponentChildren; open?: boolean }) {
  const [isOpen, setOpen] = useState(open);
  return (
    <section class={`rule ${isOpen ? 'open' : ''}`}>
      <button class="rule-head" onClick={() => setOpen(!isOpen)} aria-expanded={isOpen}>
        <span>{title}</span>
        <span class="chev">{isOpen ? '▾' : '▸'}</span>
      </button>
      {isOpen && <div class="rule-body">{children}</div>}
    </section>
  );
}

export function RulesView() {
  return (
    <div class="view scroll">
      <div class="view-head">
        <button class="btn ghost back-btn" onClick={() => setState({ tab: 'map' })}>
          ← Carte
        </button>
        <h2>Règles en bref</h2>
        <p class="muted small">Le monde est trop petit pour tout le monde : conquérez, puis laissez vos peuples s'éteindre au bon moment.</p>
      </div>
      <Section title="🎯 Le but" open>
        <p>
          Avoir le plus de <b>pièces</b> 💰 à la fin du dernier tour. À la fin de chacun de vos tours, chaque région que vous occupez
          rapporte 1 pièce, plus les bonus de vos peuples et pouvoirs. Chacun commence avec 5 pièces. Les totaux restent <b>cachés</b>,
          le vôtre compris, jusqu'au podium.
        </p>
      </Section>
      <Section title="🔄 Déroulement d'un tour">
        <ol>
          <li>
            <b>Sans peuple actif ?</b> Choisissez un peuple dans 👑 Marché. Le premier est gratuit ; prendre le n-ième coûte n−1 pièces
            (1 pièce posée sur chaque peuple sauté).
          </li>
          <li>
            <b>Préparation :</b> vos jetons sur la carte reviennent en main, sauf 1 par région.
          </li>
          <li>
            <b>Conquêtes :</b> touchez une région marquée ⚔ : le bouton indique son coût et ce qu'il vous restera en main.
          </li>
          <li>
            <b>Redéploiement :</b> répartissez vos jetons en main (au moins 1 par région). « Auto » reprend tous vos jetons et les répartit selon les menaces.
          </li>
          <li>
            <b>Décompte :</b> vous gagnez vos pièces. Le joueur suivant, s'il a été attaqué, retrouve ses jetons récupérés à son tour ;
            les autres joueurs attaqués les replacent aussitôt sur leurs régions.
          </li>
        </ol>
      </Section>
      <Section title="⚔️ Conquérir une région">
        <p>
          Coût = <b>2 jetons</b> + 1 par jeton ennemi présent + 1 si <b>Montagne</b> + 1 par tanière, forteresse ou campement.
          Certains pouvoirs réduisent ce coût (minimum 1).
        </p>
        <p>
          Votre <b>première conquête</b> doit se faire en bord de carte (ou sur la côte d'une mer en bordure). Ensuite, seulement des
          régions adjacentes aux vôtres. Mers et lacs sont infranchissables (sauf Marins).
        </p>
        <p>
          Le défenseur <b>perd 1 jeton</b> définitivement ; il replace les autres sur ses régions, ou les reprend en main s'il joue
          juste après vous. Un peuple en déclin les perd tous (sauf les Goules). Les tribus oubliées 🛖 disparaissent simplement.
        </p>
      </Section>
      <Section title="🎲 Le dé de renfort">
        <p>
          Quand il vous manque 1 à 3 jetons, vous pouvez tenter une <b>dernière conquête</b> au dé. Ses faces : 0, 0, 0, 1, 2, 3. Si vos
          jetons + le dé atteignent le coût, la région est prise. Dans tous les cas, vos conquêtes s'arrêtent.
        </p>
        <p>Chances : il manque 1 jeton → 50 %, 2 jetons → 33 %, 3 jetons → 17 %.</p>
      </Section>
      <Section title="💤 Le déclin">
        <p>
          Au début d'un tour (avant toute conquête), vous pouvez mettre votre peuple actif <b>en déclin</b> : il ne garde qu'un jeton par
          région, perd son pouvoir, mais ses régions rapportent toujours des pièces. Votre tour s'arrête là, et au tour suivant vous
          choisissez un nouveau peuple tout frais.
        </p>
        <p>
          Vous ne pouvez avoir qu'<b>un seul peuple en déclin</b> : l'ancien disparaît de la carte (sauf avec le pouvoir Spirituels).
          Le bon moment pour décliner : quand votre peuple est trop étalé pour conquérir.
        </p>
      </Section>
      <Section title="🗺️ Terrains et symboles">
        <ul class="plain">
          {Object.entries(TERRAINS).map(([id, t]) => (
            <li key={t.name}>
              <span class="swatch" style={{ background: t.color }} /> <ArtIcon terrain={id as Terrain} size={32} /> {t.name}
            </li>
          ))}
          <li>{FEATURE_ICONS.cavern} Caverne · {FEATURE_ICONS.magic} Source magique · {FEATURE_ICONS.mine} Mine</li>
          <li>{FEATURE_ICONS.lost} Tribu oubliée (1 jeton neutre)</li>
          <li>
            {FEATURE_ICONS.fortress} Forteresse · {FEATURE_ICONS.lair} Tanière · {FEATURE_ICONS.camp} Campement : +1 défense chacun
          </li>
          <li>
            {FEATURE_ICONS.hero} Héros · {FEATURE_ICONS.dragon} Dragon · {FEATURE_ICONS.hole} Terrier : région imprenable
          </li>
        </ul>
      </Section>
      <Section title={`🧝 Les ${RACE_IDS.length} peuples`}>
        <ul class="glossary">
          {RACE_IDS.map((id) => {
            const r = RACES[id];
            return (
              <li key={id}>
                <div class="g-head">
                  <ArtIcon race={id} size={32} /> <b>{r.name}</b> · {r.tag} <span class="muted small">{r.tokens} jetons</span>
                </div>
                <div class="small">{r.desc}</div>
              </li>
            );
          })}
        </ul>
      </Section>
      <Section title={`✨ Les ${POWER_IDS.length} pouvoirs`}>
        <ul class="glossary">
          {POWER_IDS.map((id) => {
            const p = POWERS[id];
            return (
              <li key={id}>
                <div class="g-head">
                  <ArtIcon power={id} size={32} /> <b>{p.name}</b> · {p.tag} <span class="muted small">+{p.tokens} jetons</span>
                </div>
                <div class="small">{p.desc}</div>
              </li>
            );
          })}
        </ul>
      </Section>
      <Section title="📱 Astuces d'interface">
        <ul>
          <li>Touchez une région : le bouton donne le coût, « Détails ▾ » la formule et les conséquences.</li>
          <li>En conquête, ⚔ indique le coût de chaque région ; 🎲 la chance au dé.</li>
          <li>En redéploiement, ⚔ sur vos régions indique ce que l'ennemi le plus dangereux paiera pour vous la prendre.</li>
          <li>Les étiquettes du haut résument vos pouvoirs et ceux des adversaires : touchez-les pour le détail.</li>
          <li>↶ annule la dernière action tant qu'aucun dé n'a été lancé.</li>
          <li>Glissez la feuille vers le bas pour la fermer. Pincez la carte pour zoomer.</li>
          <li>En ligne, votre téléphone vibre quand c'est votre tour ; en attendant, vous pouvez explorer la carte et les autres vues.</li>
          <li>Les parties sont sauvegardées automatiquement (sur le serveur en ligne, sur l'appareil sinon).</li>
        </ul>
      </Section>
    </div>
  );
}
