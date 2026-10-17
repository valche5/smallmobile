import { useState } from 'preact/hooks';
import { FEATURE_ICONS, POWERS, RACES, TERRAINS } from '../../engine/data';
import { ArtIcon } from './ArtIcon';
import {
  abandonBrief,
  attackBrief,
  attackDetails,
  conquestBlock,
  costLine,
  cut,
  foeThreatReason,
  foeThreatText,
  iconFormula,
  immuneKind,
  lostRegions,
  type AttackHow,
  type Line,
} from '../../engine/preview';
import {
  actingRace,
  analyzeTarget,
  campsOnBoard,
  canAbandon,
  currentActor,
  defenseParts,
  defenseValue,
  fortressesOnBoard,
  heroesOnBoard,
  isWater,
  raceOf,
} from '../../engine/rules';
import type { GameState } from '../../engine/types';
import { act, getState, setState, viewerSeat } from '../store';
import { Consequences, CostTable, Details, Lines, plural, Sheet, type SheetSize } from './common';

interface Props {
  g: GameState;
  region: number;
  human: boolean;
  onClose: () => void;
  onHeight: (h: number) => void;
}

interface Section {
  size: SheetSize;
  title?: preact.ComponentChildren;
  /** false : pas de sous-titre (panneau de redéploiement). */
  subtitle?: boolean;
  /** false : pas de ligne occupant. */
  occupant?: boolean;
  body: preact.ComponentChildren;
  footer?: preact.ComponentChildren;
}

/** Le joueur qui regarde : mon siège en ligne, sinon l'humain qui a la main ou le dernier qui l'a eue. */
function viewerOf(g: GameState, human: boolean): number | null {
  return viewerSeat() ?? (human ? currentActor(g) : getState().lastHuman);
}

/** Fiche d'une région : l'essentiel en quelques lignes, les chiffres sur les boutons, le détail à la demande. */
export function RegionSheet({ g, region, human, onClose, onHeight }: Props) {
  const def = g.map.regions[region]!;
  const r = g.regions[region]!;
  const race = actingRace(g);
  const viewer = viewerOf(g, human);
  const owner = r.race && r.race !== 'lost' ? g.players[raceOf(g, r.race).owner] : null;

  let section: Section;
  if (human && g.phase === 'conquest' && race) section = r.race === race ? ownConquestSection(g, region, viewer) : attackSection(g, region);
  else if (human && (g.phase === 'redeploy' || g.phase === 'defend') && race && r.race === race)
    section = { size: 'peek', subtitle: false, occupant: false, body: <DeploySection g={g} region={region} me={viewer} /> };
  else section = viewSection(g, region, viewer);

  const title = section.title ?? def.name;
  return (
    <Sheet
      onClose={onClose}
      onHeight={onHeight}
      scrollKey={region}
      size={section.size}
      class="region-sheet"
      accent={owner?.color}
      title={<span class="region-title"><ArtIcon terrain={def.terrain} size={32} /><span class="ell">{title}</span></span>}
      subtitle={section.subtitle === false ? undefined : <span class="ell">{subtitle(g, region, viewer, human)}</span>}
      footer={section.footer}
    >
      {section.occupant !== false && <Occupant g={g} region={region} viewer={viewer} />}
      {section.body}
    </Sheet>
  );
}

/** « 🌾 Ferme · ⛏️ Mine · 🧭 entrée · prise par Bob » : le terrain et tout ce que la carte y dessine. */
function subtitle(g: GameState, region: number, viewer: number | null, human: boolean): string {
  const def = g.map.regions[region]!;
  const terrain = TERRAINS[def.terrain];
  const parts = [`${terrain.icon} ${terrain.name}`];
  if (def.cavern) parts.push(`${FEATURE_ICONS.cavern} Caverne`);
  if (def.magic) parts.push(`${FEATURE_ICONS.magic} Source magique`);
  if (def.mine) parts.push(`${FEATURE_ICONS.mine} Mine`);
  const army = g.turn.army;
  if (human && g.phase === 'conquest' && army && def.entry && !g.regions.some((x) => x.race === army)) parts.push('🧭 entrée');
  if (viewer !== null && lostRegions(g, viewer).includes(region)) {
    const by = [...g.log].reverse().find((l) => l.kind === 'conquer' && l.region === region && l.victim === viewer && l.player !== viewer);
    if (by?.player !== undefined && by.player !== null) parts.push(`prise par ${g.players[by.player]!.name}`);
  }
  return parts.join(' · ');
}

/** « ● Bob · 🧝 Elfes 🕊️ Diplomates · 2 jetons · 🏰 › » : toucher ouvre Peuples en jeu. */
function Occupant({ g, region, viewer }: { g: GameState; region: number; viewer: number | null }) {
  const r = g.regions[region]!;
  if (!r.race) return <div class="occ plain">Vide</div>;
  if (r.race === 'lost')
    return (
      <div class="occ plain">
        <span class="emo">{FEATURE_ICONS.lost}</span> Tribu oubliée · {plural(r.tokens, 'jeton')}
      </div>
    );
  const rs = raceOf(g, r.race);
  const owner = g.players[rs.owner]!;
  const markers: string[] = [];
  if (r.fortress) markers.push(FEATURE_ICONS.fortress);
  if (r.camps) markers.push(`${FEATURE_ICONS.camp}${r.camps > 1 ? r.camps : ''}`);
  if (r.lair) markers.push(FEATURE_ICONS.lair);
  if (r.hero) markers.push(FEATURE_ICONS.hero);
  if (r.dragon) markers.push(FEATURE_ICONS.dragon);
  if (r.hole) markers.push(FEATURE_ICONS.hole);
  const combo = rs.declined
    ? `${RACES[r.race].icon} ${RACES[r.race].name} en déclin`
    : `${RACES[r.race].icon} ${RACES[r.race].name} ${POWERS[rs.power].icon} ${POWERS[rs.power].name}`;
  return (
    <button class="occ" style={{ borderLeftColor: owner.color }} onClick={() => setState({ sheet: { kind: 'people', player: owner.id } })}>
      <span class="occ-text">
        <span class="pdot" style={{ background: owner.color }} /> {owner.id === viewer ? 'Vous' : owner.name} · {combo}
      </span>
      {/* les jetons décident de l'attaque : ils restent visibles, c'est le nom qui se coupe */}
      <span class="occ-count">
        · {plural(r.tokens, 'jeton')}
        {markers.map((m) => ` · ${m}`).join('')}
      </span>
      <span class="occ-more">›</span>
    </button>
  );
}

function defenseDetails(g: GameState, region: number) {
  return (
    <Details>
      <div class="detail-line">
        Défense : {iconFormula(defenseParts(g, region))} = {defenseValue(g, region)}
      </div>
    </Details>
  );
}

/** Consultation (hors de mon tour, ou une région que ma phase ne concerne pas) : une seule ligne utile. */
function viewSection(g: GameState, region: number, viewer: number | null): Section {
  const r = g.regions[region]!;
  const def = g.map.regions[region]!;
  const mine = viewer !== null && !!r.race && r.race !== 'lost' && raceOf(g, r.race).owner === viewer;
  let line: preact.ComponentChildren;
  if (mine) line = <ThreatLine g={g} region={region} me={viewer} />;
  else if (immuneKind(r)) line = <div class="reason">⛔ Imprenable : {IMMUNE[immuneKind(r)!]}</div>;
  else if (isWater(def) && !r.race) line = <div class="reason">{def.terrain === 'sea' ? '⛔ Mer : seuls les Marins la prennent' : '⛔ Lac : seuls les Marins le prennent'}</div>;
  else line = <div class="reason">Défense {defenseValue(g, region)}</div>;
  return {
    size: 'peek',
    body: (
      <>
        {line}
        {defenseDetails(g, region)}
      </>
    ),
  };
}

const IMMUNE = { héros: 'un héros la garde', dragon: 'le dragon la garde', terrier: 'terrier des Semi-hommes' };

/** Fiche d'attaque : formule s'il y a un modificateur, 3 conséquences au plus, Détails, puis les boutons. */
function attackSection(g: GameState, region: number): Section {
  const info = analyzeTarget(g, region);
  const r = g.regions[region]!;
  const hand = raceOf(g, g.turn.army!).hand;
  const normal = info.kind === 'normal';
  const die = info.kind === 'die';
  const how: AttackHow | null = info.sorcery ? 'sorcery' : normal ? 'conquer' : info.dragon ? 'dragon' : die ? 'die' : null;
  if (!how) {
    // pas attaquable : une seule raison, en feuille basse
    return {
      size: 'peek',
      body: (
        <>
          <div class="reason">{conquestBlock(g, region)}</div>
          {defenseDetails(g, region)}
        </>
      ),
    };
  }
  const owner = r.race && r.race !== 'lost' ? g.players[raceOf(g, r.race).owner]!.name : '';
  const head: Line[] = [];
  if (info.sorcery) head.push({ icon: RACES.sorcerers.icon, text: `Gratuit : un Sorcier remplace le jeton de ${owner}` });
  if (info.dragon) {
    head.push({ icon: FEATURE_ICONS.dragon, text: '1 seul jeton, la région devient imprenable' });
    const old = g.regions.findIndex((x) => x.dragon);
    if (old >= 0 && old !== region && how !== 'dragon')
      head.push({ icon: '⚠️', text: `Le dragon quittera « ${cut(g.map.regions[old]!.name)} »`, tone: 'warn' });
  }
  if (die && !normal) head.push({ icon: '⛔', text: 'Dernière conquête · raté : jetons gardés', tone: 'warn' });
  const brief = attackBrief(g, region, how);
  const all = [...head, ...brief.lines];
  const shown = all.slice(0, 3);
  const more = [...all.slice(3), ...brief.more];

  let formula: string | null = null;
  if (normal) formula = costLine(info.parts, info.effective);
  else if (die) formula = `Coût ${info.cost} · il vous manque ${info.need}`;

  const pct = `${Math.round(info.chance * 100)} %`;
  // l'option de pouvoir à gauche ; libellés raccourcis quand deux ou trois boutons se partagent la rangée
  const count = [info.sorcery, info.dragon, normal, die].filter(Boolean).length;
  const pair = count >= 2;
  const tight = count >= 3;
  const buttons = [];
  if (info.sorcery)
    buttons.push(
      <button key="s" class="btn alt" aria-label="Ensorceler · gratuit" onClick={() => act({ type: 'sorcery', region })}>
        {RACES.sorcerers.icon} {tight ? 'Gratuit' : 'Ensorceler · gratuit'}
      </button>,
    );
  if (info.dragon)
    buttons.push(
      <button key="d" class="btn alt" aria-label="Dragon · 1 jeton" onClick={() => act({ type: 'dragon', region })}>
        {FEATURE_ICONS.dragon} {tight ? '1' : pair ? '1 jeton' : 'Dragon · 1 jeton'}
      </button>,
    );
  if (normal)
    buttons.push(
      <button key="c" class={`btn ${info.sorcery ? 'ghost' : 'primary'}`} onClick={() => act({ type: 'conquer', region })}>
        {pair ? `⚔️ ${tight ? '' : 'Conquérir · '}${info.effective}` : `⚔️ Conquérir · ${plural(info.effective, 'jeton')} · reste ${hand - info.effective}`}
      </button>,
    );
  if (die)
    buttons.push(
      <button key="r" class="btn risk" onClick={() => act({ type: 'reinforce', region })}>
        🎲 {pair ? `Dé · ${pct}` : `Tenter le dé · ${pct}`}
      </button>,
    );

  return {
    size: 'standard',
    body: (
      <>
        {formula && <div class="formula">{formula}</div>}
        <Consequences lines={shown} />
        <Details>
          <CostTable parts={info.parts} title="Coût de la conquête" totalLabel="Jetons à engager" />
          <Lines lines={[...attackDetails(g, region, how), ...more]} />
        </Details>
      </>
    ),
    footer: <div class="btn-row">{buttons}</div>,
  };
}

/** Ma région pendant la conquête : abandon direct (annulable avec ↶), une alerte au plus. */
function ownConquestSection(g: GameState, region: number, viewer: number | null): Section {
  const r = g.regions[region]!;
  const def = g.map.regions[region]!;
  const title = `${def.name} · à vous · ${plural(r.tokens, 'jeton')}`;
  if (canAbandon(g, region)) {
    return {
      size: 'peek',
      title,
      occupant: false,
      body: viewer !== null ? <ThreatLine g={g} region={region} me={viewer} /> : null,
    };
  }
  const brief = abandonBrief(g, region);
  return {
    size: 'peek',
    title,
    occupant: false,
    body: (
      <>
        {brief.alert && <div class="reason warn">{brief.alert}</div>}
        {brief.more.length > 0 && (
          <Details>
            <Lines lines={brief.more} />
          </Details>
        )}
      </>
    ),
    footer: (
      <div class="btn-row">
        <button class="btn danger" onClick={() => act({ type: 'abandon', region })}>
          🏳️ Abandonner · +{r.tokens} en main
        </button>
      </div>
    ),
  };
}

/** « ⚔ Bob la prend avec 4 (🗡️ −1) » ; toucher ajoute la raison (« Bob : 3 jetons à côté »). */
function ThreatLine({ g, region, me }: { g: GameState; region: number; me: number }) {
  const [why, setWhy] = useState(false);
  const t = foeThreatText(g, region, me);
  const reason = why ? foeThreatReason(g, region, me) : null;
  if (t.muted) return <div class="reason muted threat">{t.text}</div>;
  return (
    <button class="reason threat" style={t.color ? { borderLeftColor: t.color } : undefined} onClick={() => setWhy(!why)}>
      <span class="ell">
        {t.text}
        {reason && <span class="muted"> · {reason}</span>}
      </span>
    </button>
  );
}

/** Panneau de redéploiement ou de défense : stepper, menace ennemie, une rangée de pouvoir. */
function DeploySection({ g, region, me }: { g: GameState; region: number; me: number | null }) {
  const race = actingRace(g)!;
  const rs = raceOf(g, race);
  const r = g.regions[region]!;
  const power = g.phase === 'redeploy' && g.turn.stage === 'active' && !rs.declined ? rs.power : null;
  const camps = 5 - campsOnBoard(g, race);
  const heroes = heroesOnBoard(g, race);
  return (
    <>
      <div class="deploy-row">
        <button class="step" disabled={r.tokens <= 1} onClick={() => act({ type: 'deploy', region, delta: -1 })} aria-label="Retirer un jeton">
          −
        </button>
        <div class="step-value">{plural(r.tokens, 'jeton')}</div>
        <button class="step" disabled={rs.hand < 1} onClick={() => act({ type: 'deploy', region, delta: 1 })} aria-label="Ajouter un jeton">
          +
        </button>
        {rs.hand > 1 && (
          <button class="btn ghost all-here" onClick={() => act({ type: 'deploy', region, delta: rs.hand })}>
            Tout ici +{rs.hand}
          </button>
        )}
      </div>
      {me !== null && <ThreatLine g={g} region={region} me={me} />}
      {power === 'bivouacking' && (
        <div class="power-row">
          <span class="ell">{FEATURE_ICONS.camp} Campements</span>
          <span class="power-btns">
            <button class="btn ghost" disabled={r.camps < 1} onClick={() => act({ type: 'camp', region, delta: -1 })} aria-label="Retirer un campement">
              −
            </button>
            <b>{r.camps}</b>
            <button class="btn ghost" disabled={camps < 1} onClick={() => act({ type: 'camp', region, delta: 1 })} aria-label="Poser un campement">
              +
            </button>
          </span>
          <span class="muted">· {camps} en réserve</span>
        </div>
      )}
      {power === 'fortified' && (
        <div class="power-row">
          {g.turn.fortressRegion === region ? (
            <>
              <span>{FEATURE_ICONS.fortress} Bâtie ici ce tour</span>
              <button class="btn ghost" onClick={() => act({ type: 'fortress', region })}>
                Annuler
              </button>
            </>
          ) : r.fortress ? (
            <span class="muted">{FEATURE_ICONS.fortress} Forteresse déjà bâtie</span>
          ) : (
            <button class="btn alt" disabled={g.turn.fortressRegion !== null || fortressesOnBoard(g, race) >= 6} onClick={() => act({ type: 'fortress', region })}>
              {FEATURE_ICONS.fortress} Bâtir ici · +1 défense, +1 💰/tour
            </button>
          )}
        </div>
      )}
      {power === 'heroic' && (
        <div class="power-row">
          {r.hero ? (
            <button class="btn ghost" onClick={() => act({ type: 'hero', region })}>
              Retirer le héros
            </button>
          ) : (
            <button class="btn alt" disabled={heroes >= 2} onClick={() => act({ type: 'hero', region })}>
              {POWERS.heroic.icon} {heroes >= 2 ? 'Vos 2 héros sont placés' : `Héros ici · imprenable (${heroes + 1}/2)`}
            </button>
          )}
        </div>
      )}
    </>
  );
}
