import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { FEATURE_ICONS, RACES, TERRAINS } from '../../engine/data';
import { TERRAIN_TEXTURES } from '../art';
import { ownerOf } from '../../engine/rules';
import type { GameState, Terrain } from '../../engine/types';
import { canAct, useStore } from '../store';
import type { Mark, MarkKind } from './marks';

interface Props {
  game: GameState;
  selected: number | null;
  marks: Map<number, Mark>;
  /** Ignoré : les pastilles sont toujours affichées. */
  showCosts?: boolean;
  /** Ignoré : la pulsation de mes actions et la surbrillance des actions adverses viennent du store. */
  pulse?: number | null;
  focus: { region: number; n: number } | null;
  follow: { region: number; n: number } | null;
  /** Partie de la carte masquée par une feuille (en px, depuis le bas ou la droite de l'écran). */
  cover: { bottom: number; right: number };
  onTap: (region: number | null) => void;
  /** Ignorés : la légende passe par ☰, les coûts sont toujours affichés. */
  onLegend?: () => void;
  onToggleCosts?: () => void;
}

/** Après un toucher de la carte, la caméra de la lecture adverse reste immobile ce temps-là (ms). */
const HOLD_MS = 4000;

interface View {
  x: number;
  y: number;
  k: number;
}

/** Position sur un axe : centrée si la carte tient dans l'écran, sinon sans bord vide. */
function fitAxis(pos: number, mapSize: number, viewSize: number): number {
  if (mapSize <= viewSize) return (viewSize - mapSize) / 2;
  return Math.min(0, Math.max(viewSize - mapSize, pos));
}

const PATTERNS: Record<Terrain, string> = {
  farmland: `<pattern id="pat-farmland" width="20" height="20" patternUnits="userSpaceOnUse" patternTransform="rotate(32)"><line x1="0" y1="0" x2="0" y2="20" stroke="#7a5c12" stroke-width="2.4" stroke-opacity="0.22"/></pattern>`,
  forest: `<pattern id="pat-forest" width="38" height="34" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="6.5" fill="#1d4a18" fill-opacity="0.32"/><circle cx="28" cy="26" r="7.5" fill="#1d4a18" fill-opacity="0.32"/><rect x="9" y="15" width="2" height="5" fill="#3b2a12" fill-opacity="0.35"/><rect x="27" y="32" width="2" height="2" fill="#3b2a12" fill-opacity="0.35"/></pattern>`,
  hill: `<pattern id="pat-hill" width="44" height="30" patternUnits="userSpaceOnUse"><path d="M3 22 q9 -12 18 0" fill="none" stroke="#6b4e1c" stroke-width="2.2" stroke-opacity="0.35"/><path d="M24 9 q9 -12 18 0" fill="none" stroke="#6b4e1c" stroke-width="2.2" stroke-opacity="0.35"/></pattern>`,
  mountain: `<pattern id="pat-mountain" width="46" height="38" patternUnits="userSpaceOnUse"><path d="M3 32 l11 -20 l11 20 z" fill="#ffffff" fill-opacity="0.25" stroke="#3e3631" stroke-width="2" stroke-opacity="0.45" stroke-linejoin="round"/><path d="M10 19 l4 -7 l4 7" fill="#fff" fill-opacity="0.6"/><path d="M26 16 l8 -14 l8 14 z" fill="#ffffff" fill-opacity="0.18" stroke="#3e3631" stroke-width="2" stroke-opacity="0.35" stroke-linejoin="round"/></pattern>`,
  swamp: `<pattern id="pat-swamp" width="40" height="30" patternUnits="userSpaceOnUse"><path d="M3 12 h14 M22 26 h14" stroke="#20453a" stroke-width="2.2" stroke-opacity="0.35"/><path d="M8 12 v-6 M11 12 v-8 M14 12 v-5 M27 26 v-6 M30 26 v-8" stroke="#20453a" stroke-width="1.6" stroke-opacity="0.4"/></pattern>`,
  sea: `<pattern id="pat-sea" width="52" height="28" patternUnits="userSpaceOnUse"><path d="M2 14 q6.5 -7 13 0 t13 0" fill="none" stroke="#ffffff" stroke-width="2" stroke-opacity="0.35"/><path d="M28 27 q6.5 -7 13 0" fill="none" stroke="#ffffff" stroke-width="2" stroke-opacity="0.25"/></pattern>`,
  lake: `<pattern id="pat-lake" width="44" height="26" patternUnits="userSpaceOnUse"><path d="M2 13 q5.5 -6 11 0 t11 0" fill="none" stroke="#ffffff" stroke-width="2" stroke-opacity="0.4"/></pattern>`,
};

const DEFS = `
${Object.values(PATTERNS).join('')}
<filter id="gray"><feColorMatrix type="saturate" values="0.05"/></filter>
<filter id="shadow" x="-30%" y="-30%" width="160%" height="160%"><feDropShadow dx="0" dy="1.2" stdDeviation="1.2" flood-color="#000" flood-opacity="0.35"/></filter>
`;

/** Avale le prochain « click » (celui que le navigateur émet après un toucher sur la carte). */
/** Boîte englobante d'un tracé de région (polygone en coordonnées absolues). */
function pathBox(path: string): { x: number; y: number; w: number; h: number } {
  const n = (path.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number);
  const xs = n.filter((_, i) => i % 2 === 0);
  const ys = n.filter((_, i) => i % 2 === 1);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

function swallowNextClick() {
  const swallow = (ev: Event) => {
    ev.stopPropagation();
    ev.preventDefault();
  };
  window.addEventListener('click', swallow, { capture: true, once: true });
  setTimeout(() => window.removeEventListener('click', swallow, true), 400);
}

/** La dernière région touchée clignote à la couleur de son nouvel occupant. */
function pulseColor(g: GameState, region: number): string {
  const race = g.regions[region]!.race;
  const rs = race && race !== 'lost' ? g.races[race] : undefined;
  return rs ? g.players[rs.owner]!.color : '#ffe066';
}

export function MapView(props: Props) {
  const { game: g, selected, marks, focus, follow, cover, onTap } = props;
  const s = useStore();
  const covered = cover.bottom > 0 || cover.right > 0;
  const live = !!s.playback;
  const mine = canAct(s);
  // contour des régions perdues : masqué pendant mon redéploiement et ma défense
  const showLost = !(mine && (g.phase === 'redeploy' || g.phase === 'defend'));
  /** L'utilisateur a touché la carte il y a moins de HOLD_MS : la lecture ne déplace pas la vue. */
  const [held, setHeld] = useState(false);
  const holdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** « 📍 Suivre » touché : la caméra suit même avec une feuille ouverte, jusqu'au prochain toucher. */
  const forced = useRef(false);
  const sheetOpen = s.sheet !== null;
  const suspended = !forced.current && (held || sheetOpen);
  const map = g.map;
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement>(null);
  const view = useRef<View>({ x: 0, y: 0, k: 1 });
  const fitK = useRef(1);
  const size = useRef({ w: 1, h: 1 });
  const onTapRef = useRef(onTap);
  onTapRef.current = onTap;
  const anim = useRef<number | null>(null);
  /** Vue d'avant un recentrage automatique, rétablie quand la feuille se referme. */
  const restore = useRef<View | null>(null);

  const apply = () => {
    const v = view.current;
    gRef.current?.setAttribute('transform', `translate(${v.x.toFixed(2)},${v.y.toFixed(2)}) scale(${v.k.toFixed(5)})`);
    const zoom = v.k / fitK.current;
    // Les pastilles gardent une taille d'écran à peu près constante (elles grossissent un peu au zoom)
    const px = Math.min(1.45, 1 + (zoom - 1) * 0.22) * (size.current.w < 500 ? 1 : 1.15);
    svgRef.current?.style.setProperty('--s', String(px / v.k));
  };

  const clamp = () => {
    const v = view.current;
    const { w, h } = size.current;
    v.k = Math.max(fitK.current * 0.85, Math.min(fitK.current * 6, v.k));
    const keep = 60; // au moins 60 px de carte restent visibles
    v.x = Math.min(w - keep, Math.max(keep - map.width * v.k, v.x));
    v.y = Math.min(h - keep, Math.max(keep - map.height * v.k, v.y));
  };

  const animTarget = useRef<View | null>(null);

  /** Arrête l'animation en cours ; `finish` saute directement à sa position finale. */
  const stopAnim = (finish = false) => {
    if (anim.current) cancelAnimationFrame(anim.current);
    anim.current = null;
    if (finish && animTarget.current) {
      view.current = { ...animTarget.current };
      apply();
    }
    animTarget.current = null;
  };

  const fit = () => {
    restore.current = null;
    stopAnim();
    const { w, h } = size.current;
    const k = Math.min(w / map.width, h / map.height) * 0.97;
    fitK.current = Math.min(w / map.width, h / map.height) * 0.97;
    view.current = { k, x: (w - map.width * k) / 2, y: (h - map.height * k) / 2 };
    apply();
  };

  const animateTo = (target: View) => {
    stopAnim();
    animTarget.current = target;
    const from = { ...view.current };
    const t0 = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - t0) / 260);
      const e = 1 - Math.pow(1 - t, 3);
      view.current = {
        x: from.x + (target.x - from.x) * e,
        y: from.y + (target.y - from.y) * e,
        k: from.k + (target.k - from.k) * e,
      };
      apply();
      if (t < 1) anim.current = requestAnimationFrame(step);
      else {
        anim.current = null;
        animTarget.current = null;
      }
    };
    anim.current = requestAnimationFrame(step);
  };

  const zoomAt = (sx: number, sy: number, factor: number) => {
    restore.current = null;
    stopAnim();
    const v = view.current;
    const k = Math.max(fitK.current * 0.85, Math.min(fitK.current * 6, v.k * factor));
    const f = k / v.k;
    v.x = sx - (sx - v.x) * f;
    v.y = sy - (sy - v.y) * f;
    v.k = k;
    clamp();
    apply();
  };

  // Taille du conteneur
  useEffect(() => {
    const el = wrapRef.current!;
    let first = true;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      if (r.width < 10 || r.height < 10) return;
      const old = size.current;
      size.current = { w: r.width, h: r.height };
      if (first) {
        first = false;
        fit();
      } else {
        // garder le même centre
        const v = view.current;
        const cx = (old.w / 2 - v.x) / v.k;
        const cy = (old.h / 2 - v.y) / v.k;
        fitK.current = Math.min(r.width / map.width, r.height / map.height) * 0.97;
        v.x = r.width / 2 - cx * v.k;
        v.y = r.height / 2 - cy * v.k;
        clamp();
        apply();
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [map]);

  // Gestes : glisser, pincer, molette, toucher
  useEffect(() => {
    const svg = svgRef.current!;
    const pointers = new Map<number, { x: number; y: number }>();
    let start: { x: number; y: number; t: number; region: number | null } | null = null;
    let moved = false;
    let multi = false;

    const local = (e: PointerEvent | WheelEvent) => {
      const r = svg.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const down = (e: PointerEvent) => {
      touched();
      stopAnim(true);
      svg.setPointerCapture(e.pointerId);
      const p = local(e);
      pointers.set(e.pointerId, p);
      if (pointers.size === 1) {
        const el = (e.target as Element).closest('[data-region]');
        start = { ...p, t: performance.now(), region: el ? Number(el.getAttribute('data-region')) : null };
        moved = false;
        multi = false;
      } else multi = true;
    };
    const move = (e: PointerEvent) => {
      const prev = pointers.get(e.pointerId);
      if (!prev) return;
      const p = local(e);
      if (pointers.size === 1) {
        if (start && Math.hypot(p.x - start.x, p.y - start.y) > 9) moved = true;
        if (moved) {
          restore.current = null;
          view.current.x += p.x - prev.x;
          view.current.y += p.y - prev.y;
          clamp();
          apply();
        }
        pointers.set(e.pointerId, p);
      } else if (pointers.size === 2) {
        const [aId, bId] = [...pointers.keys()];
        const a0 = pointers.get(aId!)!;
        const b0 = pointers.get(bId!)!;
        pointers.set(e.pointerId, p);
        const a1 = pointers.get(aId!)!;
        const b1 = pointers.get(bId!)!;
        const d0 = Math.hypot(a0.x - b0.x, a0.y - b0.y);
        const d1 = Math.hypot(a1.x - b1.x, a1.y - b1.y);
        const m0 = { x: (a0.x + b0.x) / 2, y: (a0.y + b0.y) / 2 };
        const m1 = { x: (a1.x + b1.x) / 2, y: (a1.y + b1.y) / 2 };
        view.current.x += m1.x - m0.x;
        view.current.y += m1.y - m0.y;
        if (d0 > 0) zoomAt(m1.x, m1.y, d1 / d0);
        moved = true;
      }
    };
    const up = (e: PointerEvent) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.delete(e.pointerId);
      if (pointers.size === 0) {
        if (start && !moved && !multi && performance.now() - start.t < 700) {
          // le « click » qui suit ce toucher tomberait sur la feuille qui s'ouvre sous le doigt : on l'avale
          swallowNextClick();
          onTapRef.current(start.region);
        }
        start = null;
      }
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      touched();
      restore.current = null;
      const p = local(e);
      zoomAt(p.x, p.y, Math.exp(-e.deltaY * 0.0015));
    };
    svg.addEventListener('pointerdown', down);
    svg.addEventListener('pointermove', move);
    svg.addEventListener('pointerup', up);
    svg.addEventListener('pointercancel', up);
    svg.addEventListener('wheel', wheel, { passive: false });
    return () => {
      svg.removeEventListener('pointerdown', down);
      svg.removeEventListener('pointermove', move);
      svg.removeEventListener('pointerup', up);
      svg.removeEventListener('pointercancel', up);
      svg.removeEventListener('wheel', wheel);
    };
  }, []);

  /** Un toucher, un glisser, un pincement ou la molette suspendent la caméra de la lecture. */
  const touched = () => {
    forced.current = false;
    setHeld(true);
    if (holdTimer.current) clearTimeout(holdTimer.current);
    holdTimer.current = setTimeout(() => setHeld(false), HOLD_MS);
  };
  useEffect(() => () => void (holdTimer.current && clearTimeout(holdTimer.current)), []);

  /** Amène une région dans la partie visible de la carte (sans rien bouger si elle y est déjà), par panoramique seulement. */
  const bringIntoView = (region: number, remember: boolean, margin = 60) => {
    const def = map.regions[region];
    if (!def) return;
    const v = view.current;
    const { w, h } = size.current;
    const top = wrapRef.current?.getBoundingClientRect().top ?? 0;
    const visibleW = Math.max(120, w - cover.right);
    const visibleH = Math.max(120, Math.min(h, window.innerHeight - cover.bottom - top));
    const sx = def.cx * v.k + v.x;
    const sy = def.cy * v.k + v.y;
    const inX = sx > margin && sx < visibleW - margin;
    const inY = sy > margin && sy < visibleH - margin;
    if (inX && inY) return;
    // on ne bouge que l'axe nécessaire, sans jamais découvrir de vide autour de la carte
    // (avec une feuille ouverte, la vue d'avant sera rétablie à la fermeture)
    const target = {
      k: v.k,
      x: inX ? v.x : fitAxis(visibleW / 2 - def.cx * v.k, map.width * v.k, visibleW),
      y: inY ? v.y : fitAxis(visibleH * 0.45 - def.cy * v.k, map.height * v.k, visibleH),
    };
    if (Math.abs(target.x - v.x) < 2 && Math.abs(target.y - v.y) < 2) return;
    if (remember) restore.current ??= { ...v };
    animateTo(target);
  };

  // Région ouverte dans la feuille : on la recentre, puis on revient à la vue d'avant à la fermeture
  useEffect(() => {
    if (focus) bringIntoView(focus.region, true);
  }, [focus?.n, cover.bottom, cover.right]);

  // Action adverse en cours d'affichage : la caméra la suit, sauf si l'on vient de toucher la carte ou qu'une feuille est ouverte
  useEffect(() => {
    if (follow && !focus && !suspended) bringIntoView(follow.region, false, 40);
  }, [follow?.n]);

  // une feuille qui s'ouvre ou se ferme rend la main à la règle normale
  useEffect(() => {
    forced.current = false;
  }, [s.sheet?.kind]);

  const resumeFollow = () => {
    forced.current = true;
    setHeld(false);
    if (holdTimer.current) clearTimeout(holdTimer.current);
    const region = s.highlight?.region ?? follow?.region;
    if (region !== undefined) bringIntoView(region, false, 40);
  };

  useEffect(() => {
    if (!covered && restore.current) {
      animateTo(restore.current);
      restore.current = null;
    }
  }, [covered]);

  const base = useMemo(
    () =>
      map.regions.map((r) => (
        <g key={r.id}>
          <path d={r.path} fill={TERRAINS[r.terrain].color} />
          <path d={r.path} fill={`url(#${TERRAIN_TEXTURES[r.terrain] ? `texture-${r.id}` : `pat-${r.terrain}`})`} />
        </g>
      )),
    [map],
  );

  // Textures : une tuile carrée par région, centrée sur elle et assez grande pour la couvrir (aucun raccord
  // à l'intérieur), de la même taille pour toutes (même grain partout, image jamais étirée).
  const textures = useMemo(() => {
    const boxes = map.regions.filter((r) => TERRAIN_TEXTURES[r.terrain]).map((r) => ({ r, ...pathBox(r.path) }));
    const tile = Math.ceil(Math.max(0, ...boxes.map((b) => Math.max(b.w, b.h))));
    return boxes.map(({ r, x, y, w, h }) => (
      <pattern key={r.id} id={`texture-${r.id}`} x={x + w / 2 - tile / 2} y={y + h / 2 - tile / 2} width={tile} height={tile} patternUnits="userSpaceOnUse">
        <image class="terrain-texture" href={TERRAIN_TEXTURES[r.terrain]} width={tile} height={tile} />
      </pattern>
    ));
  }, [map]);

  const humanMarks = marks.size > 0;
  // surbrillance de la trame adverse ; mes propres actions gardent leur pulsation
  const hl = live && s.highlight && map.regions[s.highlight.region] ? s.highlight : null;
  const pulseOn = !live && s.ownPulse ? s.ownPulse : null;

  return (
    <div class="map-wrap" ref={wrapRef}>
      <svg ref={svgRef} class="map-svg" role="img" aria-label="Carte du monde">
        <defs dangerouslySetInnerHTML={{ __html: DEFS }} />
        <defs>{textures}</defs>
        <g ref={gRef}>
          <rect x={-6} y={-6} width={map.width + 12} height={map.height + 12} rx={14} fill="#3d2f22" />
          {base}
          {/* Zones de toucher + contours */}
          {map.regions.map((r) => {
            const m = marks.get(r.id);
            const cls = [
              'region',
              m ? `m-${m.kind}` : '',
              selected === r.id ? 'sel' : '',
              humanMarks && m?.kind === 'none' ? 'dim' : '',
              showLost && m?.lost ? 'lost' : '',
            ]
              .filter(Boolean)
              .join(' ');
            return <path key={r.id} d={r.path} class={cls} data-region={r.id} vector-effect="non-scaling-stroke" />;
          })}
          {selected !== null && map.regions[selected] && (
            <path d={map.regions[selected]!.path} class="sel-outline" vector-effect="non-scaling-stroke" />
          )}
          {showLost &&
            map.regions.map((r) =>
              marks.get(r.id)?.lost ? <path key={`l${r.id}`} d={r.path} class="lost-outline" vector-effect="non-scaling-stroke" /> : null,
            )}
          {pulseOn && map.regions[pulseOn.region] && (
            <path
              key={`p${pulseOn.key}`}
              d={map.regions[pulseOn.region]!.path}
              class="pulse"
              style={{ stroke: pulseColor(g, pulseOn.region) }}
              vector-effect="non-scaling-stroke"
            />
          )}
          {live &&
            s.blink?.regions.map((id) =>
              map.regions[id] ? (
                <path key={`k${s.blink!.key}-${id}`} d={map.regions[id]!.path} class="hl-blink" style={{ stroke: s.blink!.color }} vector-effect="non-scaling-stroke" />
              ) : null,
            )}
          {hl && hl.hit && <path key={`f${hl.key}`} d={map.regions[hl.region]!.path} class="hl-flash" />}
          {hl && (
            <path
              key={`h${hl.key}`}
              d={map.regions[hl.region]!.path}
              class={`hl ${hl.hit ? 'hit' : ''}`}
              data-hl={hl.region}
              style={{ stroke: hl.color }}
              vector-effect="non-scaling-stroke"
            />
          )}
          {map.regions.map((r) => (
            <RegionMarks key={r.id} g={g} id={r.id} mark={marks.get(r.id)} />
          ))}
          {hl && hl.loss > 0 && (
            <g key={`b${hl.key}`} class="mk" style={{ transform: `translate(${map.regions[hl.region]!.cx}px, ${map.regions[hl.region]!.cy}px) scale(var(--s))` }}>
              <text class="loss-bubble" x={0} y={-20}>
                −{hl.loss}
              </text>
            </g>
          )}
        </g>
      </svg>
      <div class="map-tools">
        <button class="tool" aria-label="Voir toute la carte" onClick={() => fit()}>
          ⤢
        </button>
        {live && suspended && (
          <button class="tool follow" aria-label="Suivre l'action" onClick={resumeFollow}>
            📍<span>Suivre</span>
          </button>
        )}
      </div>
    </div>
  );
}

function RegionMarks({ g, id, mark }: { g: GameState; id: number; mark?: Mark }) {
  const def = g.map.regions[id]!;
  const r = g.regions[id]!;
  const feats: string[] = [];
  if (def.cavern) feats.push(FEATURE_ICONS.cavern);
  if (def.magic) feats.push(FEATURE_ICONS.magic);
  if (def.mine) feats.push(FEATURE_ICONS.mine);
  const markers: string[] = [];
  if (r.fortress) markers.push(FEATURE_ICONS.fortress);
  if (r.lair) markers.push(FEATURE_ICONS.lair);
  if (r.hero) markers.push(FEATURE_ICONS.hero);
  if (r.dragon) markers.push(FEATURE_ICONS.dragon);
  if (r.hole) markers.push(FEATURE_ICONS.hole);
  for (let i = 0; i < r.camps; i++) markers.push(FEATURE_ICONS.camp);

  let badge = null;
  if (r.race === 'lost') {
    badge = (
      <g>
        <circle r={11} fill="#efe6d1" stroke="#8a7651" stroke-width={2} filter="url(#shadow)" />
        <text y={4} class="emo" font-size={11}>
          {FEATURE_ICONS.lost}
        </text>
      </g>
    );
  } else if (r.race) {
    const owner = ownerOf(g, id)!;
    const color = g.players[owner]!.color;
    const declined = g.races[r.race]!.declined;
    badge = (
      <g>
        <circle r={13} fill={declined ? '#77736c' : color} stroke={declined ? color : '#fff'} stroke-width={declined ? 3 : 2} filter="url(#shadow)" />
        <text y={4.5} class="emo" font-size={13} filter={declined ? 'url(#gray)' : undefined}>
          {RACES[r.race].icon}
        </text>
        <circle cx={12} cy={-10} r={7.5} fill="#fff" stroke={color} stroke-width={1.8} />
        <text x={12} y={-6.6} class="count">
          {r.tokens}
        </text>
      </g>
    );
  }

  const hasBadge = !!badge;
  return (
    <g class="mk" style={{ transform: `translate(${def.cx}px, ${def.cy}px) scale(var(--s))` }}>
      {badge}
      {feats.length > 0 && (
        <g transform={`translate(0 ${hasBadge ? 21 : 0})`}>
          <rect x={-feats.length * 6.5 - 2} y={-7.5} width={feats.length * 13 + 4} height={15} rx={7.5} fill="#fffaf0" fill-opacity={0.82} />
          {feats.map((f, i) => (
            <text key={i} x={(i - (feats.length - 1) / 2) * 13} y={3.8} class="emo" font-size={10}>
              {f}
            </text>
          ))}
        </g>
      )}
      {markers.length > 0 && (
        <g transform={`translate(0 ${hasBadge ? -23 : -14})`}>
          <rect x={-markers.length * 6.5 - 2} y={-7.5} width={markers.length * 13 + 4} height={15} rx={7.5} fill="#2b2118" fill-opacity={0.72} />
          {markers.map((f, i) => (
            <text key={i} x={(i - (markers.length - 1) / 2) * 13} y={3.8} class="emo" font-size={10}>
              {f}
            </text>
          ))}
        </g>
      )}
      {mark?.label && <CostPill label={mark.label} kind={mark.kind} ring={mark.ring} hot={mark.hot} beside={hasBadge} />}
    </g>
  );
}

/** Pastille de coût ; avec `ring`, c'est la menace d'un ennemi (anneau à sa couleur, fond orange si la prise est facile). */
function CostPill({ label, kind, ring, hot, beside }: { label: string; kind: MarkKind; ring?: string; hot?: boolean; beside: boolean }) {
  const w = 10 + 8 * [...label].length;
  const x = beside ? 15 : -w / 2;
  const y = beside ? 1 : -8.5;
  return (
    <g class={`cost cost-${kind} ${ring ? 'threat' : ''} ${hot ? 'hot' : ''}`}>
      <rect x={x} y={y} width={w} height={17} rx={8.5} style={ring ? { stroke: ring } : undefined} />
      <text x={x + w / 2} y={y + 12.5}>
        {label}
      </text>
    </g>
  );
}
