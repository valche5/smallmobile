//! Les parties d'imitation : c130 (ou tout joueur `az` de lab7) joue sans bruit, une recherche
//! complète à chaque décision, et chaque partie est découpée en plans (`plan::Start`). Un
//! enregistrement par plan, avec ses cibles :
//!
//! - l'entrée de l'encodeur au début du plan (`encode`), les voisines de chaque région ;
//! - chaque décision du plan : le coup joué, les entrées du pas (`tokens::dynamics`), les coups
//!   possibles avec la politique améliorée de la recherche ;
//! - ce que dit le réseau du joueur seul (c130), pour comparer : sa politique a priori à chaque
//!   décision, sa valeur au début du plan ;
//! - les valeurs, en joueurs relatifs (0 : celui qui agit) : résultat final (victoire, pièces gagnées
//!   d'ici la fin) ; valeur de la recherche au début du plan ; valeur de la recherche à la première
//!   décision du prochain tour du joueur, et pièces gagnées d'ici là (le résultat final si la partie
//!   finit avant) ; propriétaires des régions au début de ce prochain tour.
//!
//! Fichiers d'un fil, à côté les uns des autres (`r{graine}-{fil}`), lus par `lab8/py/data.py`. Forme
//! compacte, pour que toutes les parties tiennent en mémoire à l'apprentissage : la carte une fois
//! par partie ; des entrées de la position, seuls les octets non nuls, repérés par des masques de
//! bits (bit j de l'octet j / 8 en j % 8, comme `numpy.unpackbits(bitorder="little")`).
//!
//! ```text
//! .games  par partie, une ligne : donne, graine de carte, graine, joueurs, plans
//! .map    par partie : regions × MAP_COLS (u8) : les colonnes de la carte (`encode` 0 à 14)
//! .adj    par partie : regions × u64 (voisines)
//! .seg    par plan, SEG octets : partie (u32, rang dans le fichier), steps, node_nz, glob_nz (u16),
//!         players, regions, phase (u8), 3 octets nuls ; puis f16 × 5 : win, gain, here, next_win,
//!         next_gain, prior_win (la valeur du réseau seul)
//! .nmask  par plan : regions × u64 : les colonnes non nulles des régions hors carte (bit j : colonne
//!         MAP_COLS + j)                         .nval   par plan : leurs node_nz valeurs (u8)
//! .gmask  par plan : GLOBAL_MASK octets : les entrées non nulles de la partie
//!                                               .gval   par plan : leurs glob_nz valeurs (u8)
//! .owner  par plan : regions × u8
//! .step   par décision, STEP octets : key (u16), coups possibles (u16), dyn (u8 × DYN)
//! .legal  par coup possible : key (u16), probabilité de la recherche et a priori du réseau seul,
//!         × PROBABILITY (u16, u16)
//! .raw    (vérifications seulement, `Config::raw`) par plan : regions × NODE puis GLOBAL (u8), et
//!         les 30 valeurs en f32
//! ```

use std::fs::File;
use std::future::Future;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Game, Phase};
use smallmobile_lab7::az::{Az, Direct, Evals, Oracle};
use smallmobile_lab7::batch::Batch;
use smallmobile_lab7::gpu::Gpu;
use smallmobile_lab7::net::Eval;
use smallmobile_lab7::selfplay::owners;
use smallmobile_lab7::table::{Outcome, Table};

use crate::encode::{GLOBAL, NODE, REGIONS, encode};
use crate::know::Knowledge;
use crate::moves::{self, Key};
use crate::plan::Start;
use crate::rand::{Rng, mix};
use crate::tokens::{DYN, dynamics};

/// Une probabilité de la politique est enregistrée multipliée par ce nombre.
pub const PROBABILITY: f32 = 60_000.0;
pub const SEG: usize = 16 + 30 * 2;
pub const LEGAL: usize = 6;
pub const STEP: usize = 4 + DYN;
/// Colonnes des régions qui ne dépendent que de la carte (terrain à nombre de voisines).
pub const MAP_COLS: usize = 15;
pub const GLOBAL_MASK: usize = GLOBAL.div_ceil(8);
const _: () = assert!(NODE - MAP_COLS <= 64);

/// Un f32 en f16 (arrondi au plus proche, pair en cas d'égalité ; sous-normaux compris).
pub fn half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    // la mantisse avec son 1 implicite, et le nombre de bits à retirer
    let (m, shift) = if e <= 0 { (man | 0x80_0000, (14 - e) as u32) } else { (man, 13) };
    if shift > 24 {
        return sign;
    }
    let half_bit = 1u32 << (shift - 1);
    let mut r = m >> shift;
    let rest = m & ((1 << shift) - 1);
    if rest > half_bit || (rest == half_bit && r & 1 == 1) {
        r += 1;
    }
    // un report déborde proprement sur l'exposant
    sign | if e <= 0 { r as u16 } else { (((e as u32) << 10) + r) as u16 }
}

/// Une décision d'un plan.
struct Step {
    key: Key,
    dy: [u8; DYN],
    legal: Vec<(Key, u16, u16)>,
}

/// Un plan, avec ses cibles.
struct Segment {
    players: usize,
    regions: usize,
    phase: Phase,
    actor: usize,
    /// Le tour (`turn` de `play`) où il commence.
    turn: usize,
    node: Vec<u8>,
    global: Vec<u8>,
    adj: Vec<u64>,
    steps: Vec<Step>,
    coins: [i32; MAX_PLAYERS],
    here: [f32; MAX_PLAYERS],
    /// La valeur du réseau seul au début du plan (relative).
    prior: [f32; MAX_PLAYERS],
    next: Option<([f32; MAX_PLAYERS], [f32; MAX_PLAYERS])>,
    owners: Option<[u8; REGIONS]>,
    /// Résultat final : victoire, pièces gagnées depuis le début du plan.
    win: [f32; MAX_PLAYERS],
    gain: [f32; MAX_PLAYERS],
}

/// Les valeurs de joueurs réels vues du joueur `actor` (relatif 0).
fn relative<T: Copy + Default>(v: &[T; MAX_PLAYERS], actor: usize, n: usize) -> [T; MAX_PLAYERS] {
    let mut out = [T::default(); MAX_PLAYERS];
    for (r, x) in out.iter_mut().enumerate().take(n) {
        *x = v[(actor + r) % n];
    }
    out
}

fn coins(g: &Game) -> [i32; MAX_PLAYERS] {
    std::array::from_fn(|q| g.state.players.get(q).map_or(0, |p| p.coins))
}

/// Joue une partie et la découpe en plans.
async fn play(table: &Table, seed: u32, az: &Az, rng: &mut Rng, oracle: &impl Oracle) -> Vec<Segment> {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut evals = Evals::default();
    let n = table.players;
    let adj: Vec<u64> = table.map.hoods.iter().map(|h| h.adj.0).collect();
    let mut segments: Vec<Segment> = Vec::new();
    let mut open: Option<Start> = None;
    let mut turn = 0;
    while g.state.phase != Phase::GameOver {
        let legal = moves::legal(&g, &k);
        let mv = if legal.len() == 1 {
            legal[0].mv
        } else {
            let s = az.run(&g, &k, rng, az.cfg.iterations, false, &mut evals, oracle).await;
            let actor = g.state.actor().expect("un joueur doit agir");
            // la première décision du prochain tour des joueurs qui l'attendaient
            if actor == g.state.current {
                let now = coins(&g);
                for p in segments.iter_mut().filter(|p| p.actor == actor && p.turn < turn && p.next.is_none()) {
                    let gain = std::array::from_fn(|q| (now[q] - p.coins[q]) as f32);
                    p.next = Some((relative(&s.wins, actor, n), relative(&gain, actor, n)));
                }
            }
            if open.is_none() {
                open = Some(Start::of(&g));
                let e = encode(&g, &k);
                let prior = oracle.eval(&az.net, &e, &table.map).await.win;
                segments.push(Segment {
                    players: n,
                    regions: e.regions,
                    phase: g.state.phase,
                    actor,
                    turn,
                    node: e.node[..e.regions].iter().flatten().copied().collect(),
                    global: e.global.to_vec(),
                    adj: adj.clone(),
                    steps: Vec::new(),
                    coins: coins(&g),
                    here: relative(&s.wins, actor, n),
                    prior,
                    next: None,
                    owners: None,
                    win: [0.0; MAX_PLAYERS],
                    gain: [0.0; MAX_PLAYERS],
                });
            }
            let key = legal.iter().find(|c| c.mv == s.mv).expect("coup de la recherche").key;
            let probs = legal
                .iter()
                .map(|c| {
                    let p = s.policy.iter().find(|x| x.0 == c.key).map_or(0.0, |x| x.1);
                    let prior = s.details.iter().find(|x| x.0 == c.key).map_or(0.0, |x| x.1);
                    (c.key, (p * PROBABILITY).round() as u16, (prior * PROBABILITY).round() as u16)
                })
                .collect();
            segments.last_mut().expect("plan ouvert").steps.push(Step { key, dy: dynamics(&g, &k), legal: probs });
            s.mv
        };
        let current = g.state.current;
        k.play(&mut g, &mv).expect("coup possible");
        if let Some(start) = open
            && (g.state.phase == Phase::GameOver || !start.goes_on(&mv, &g))
        {
            open = None;
        }
        if g.state.current != current {
            turn += 1;
            let c = g.state.current;
            for p in segments.iter_mut().filter(|p| p.actor == c && p.owners.is_none()) {
                p.owners = Some(owners(&g, c));
            }
        }
    }
    let o = Outcome::of(&g);
    let end = coins(&g);
    let win: [f32; MAX_PLAYERS] = std::array::from_fn(|q| o.win[q] as f32);
    for p in segments.iter_mut() {
        let gain: [f32; MAX_PLAYERS] = std::array::from_fn(|q| (end[q] - p.coins[q]) as f32);
        p.win = relative(&win, p.actor, n);
        p.gain = relative(&gain, p.actor, n);
        if p.next.is_none() {
            p.next = Some((p.win, p.gain));
        }
        if p.owners.is_none() {
            p.owners = Some(owners(&g, p.actor));
        }
    }
    segments
}

/// Les fichiers d'un fil (`r{graine}-{fil}.*`).
pub struct Files {
    out: Vec<BufWriter<File>>,
    raw: Option<BufWriter<File>>,
    /// Parties déjà écrites.
    games: u32,
}

/// L'ordre des fichiers dans `Files::out`.
pub const EXTENSIONS: [&str; 11] =
    [".games", ".map", ".adj", ".seg", ".nmask", ".nval", ".gmask", ".gval", ".owner", ".step", ".legal"];
const GAMES: usize = 0;
const MAP: usize = 1;
const ADJ: usize = 2;
const SEGS: usize = 3;
const NMASK: usize = 4;
const NVAL: usize = 5;
const GMASK: usize = 6;
const GVAL: usize = 7;
const OWNER: usize = 8;
const STEPS: usize = 9;
const LEGALS: usize = 10;

impl Files {
    pub fn create(prefix: &Path, raw: bool) -> std::io::Result<Files> {
        let open = |ext: &str| -> std::io::Result<BufWriter<File>> {
            let mut name = prefix.as_os_str().to_owned();
            name.push(ext);
            Ok(BufWriter::new(File::create(name)?))
        };
        let out = EXTENSIONS.iter().map(|e| open(e)).collect::<std::io::Result<_>>()?;
        Ok(Files { out, raw: if raw { Some(open(".raw")?) } else { None }, games: 0 })
    }

    fn write(&mut self, game: &Game_, segments: &[Segment]) -> std::io::Result<()> {
        let o = &mut self.out;
        if let Some(first) = segments.first() {
            for row in first.node.chunks(NODE) {
                o[MAP].write_all(&row[..MAP_COLS])?;
            }
            for a in &first.adj {
                o[ADJ].write_all(&a.to_le_bytes())?;
            }
        }
        for p in segments {
            let (mut nmask, mut nval) = (Vec::with_capacity(8 * p.regions), Vec::new());
            for (row, first) in p.node.chunks(NODE).zip(segments[0].node.chunks(NODE)) {
                assert_eq!(row[..MAP_COLS], first[..MAP_COLS], "colonnes de la carte");
                let mut mask = 0u64;
                for (j, &x) in row[MAP_COLS..].iter().enumerate() {
                    if x != 0 {
                        mask |= 1 << j;
                        nval.push(x);
                    }
                }
                nmask.extend_from_slice(&mask.to_le_bytes());
            }
            let (mut gmask, mut gval) = ([0u8; GLOBAL_MASK], Vec::new());
            for (j, &x) in p.global.iter().enumerate() {
                if x != 0 {
                    gmask[j / 8] |= 1 << (j % 8);
                    gval.push(x);
                }
            }
            let (next_win, next_gain) = p.next.expect("valeur au prochain tour");
            let values = [&p.win, &p.gain, &p.here, &next_win, &next_gain, &p.prior];
            let mut b = Vec::with_capacity(SEG);
            b.extend_from_slice(&self.games.to_le_bytes());
            for count in [p.steps.len(), nval.len(), gval.len()] {
                b.extend_from_slice(&(count as u16).to_le_bytes());
            }
            b.extend_from_slice(&[p.players as u8, p.regions as u8, p.phase.index() as u8, 0, 0, 0]);
            for x in values.iter().flat_map(|v| v.iter()) {
                b.extend_from_slice(&half(*x).to_le_bytes());
            }
            debug_assert_eq!(b.len(), SEG);
            o[SEGS].write_all(&b)?;
            o[NMASK].write_all(&nmask)?;
            o[NVAL].write_all(&nval)?;
            o[GMASK].write_all(&gmask)?;
            o[GVAL].write_all(&gval)?;
            o[OWNER].write_all(&p.owners.expect("propriétaires")[..p.regions])?;
            for s in &p.steps {
                o[STEPS].write_all(&s.key.to_le_bytes())?;
                o[STEPS].write_all(&(s.legal.len() as u16).to_le_bytes())?;
                o[STEPS].write_all(&s.dy)?;
                for (key, p, prior) in &s.legal {
                    o[LEGALS].write_all(&key.to_le_bytes())?;
                    o[LEGALS].write_all(&p.to_le_bytes())?;
                    o[LEGALS].write_all(&prior.to_le_bytes())?;
                }
            }
            if let Some(raw) = &mut self.raw {
                raw.write_all(&p.node)?;
                raw.write_all(&p.global)?;
                for x in values.iter().flat_map(|v| v.iter()) {
                    raw.write_all(&x.to_le_bytes())?;
                }
            }
        }
        self.games += 1;
        writeln!(o[GAMES], "{} {} {} {} {}", game.deal, game.map_seed, game.seed, game.players, segments.len())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        for f in self.out.iter_mut().chain(self.raw.as_mut()) {
            f.flush()?;
        }
        Ok(())
    }
}

/// Une donne : de quoi rejouer la partie.
struct Game_ {
    deal: u64,
    map_seed: u64,
    seed: u32,
    players: usize,
}

pub struct Config {
    pub players: Vec<usize>,
    pub games: usize,
    pub threads: usize,
    pub seed: u64,
    /// Écrire aussi les entrées pleines (`.raw`), pour vérifier la forme compacte.
    pub raw: bool,
}

/// La partie `i` (donne, carte et hasard tirés de `cfg.seed` et `i`), découpée en plans.
async fn game(i: usize, cfg: &Config, az: &Az, oracle: &impl Oracle) -> (Game_, Vec<Segment>) {
    let players = cfg.players[i % cfg.players.len()];
    let deal = mix(cfg.seed, i as u64);
    let (map_seed, seed) = (deal % 1_000_000_007, mix(deal, 1) as u32);
    let table = Table::new(players, map_seed);
    let mut rng = Rng::new(mix(deal, 99));
    let segments = play(&table, seed, az, &mut rng, oracle).await;
    (Game_ { deal, map_seed, seed, players }, segments)
}

/// Parties menées par chaque fil en même temps, avec le GPU (comme le selfplay de lab7).
const GPU_GAMES: usize = 16;

type Running<'a> = Pin<Box<dyn Future<Output = (Game_, Vec<Segment>)> + 'a>>;

/// La partie de la place `slot` avance jusqu'à ses prochaines évaluations ; une partie finie
/// (`done`) laisse sa place à la suivante (`start`). Faux : plus de partie à commencer.
fn advance<'a>(
    slot: &mut Option<Running<'a>>,
    mut start: impl FnMut() -> Option<Running<'a>>,
    done: &mut impl FnMut(Game_, Vec<Segment>) -> std::io::Result<()>,
) -> std::io::Result<bool> {
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if slot.is_none() {
            let Some(started) = start() else { return Ok(false) };
            *slot = Some(started);
        }
        let Poll::Ready((g, segments)) = slot.as_mut().expect("une partie").as_mut().poll(&mut cx) else {
            return Ok(true);
        };
        *slot = None;
        done(g, segments)?;
    }
}

/// Les parties d'imitation, en parallèle, un jeu de fichiers par fil dans `dir`. Avec `gpu`, chaque
/// fil mène `GPU_GAMES` parties, chacune reprise dès que ses évaluations sont là ; sans, une partie à
/// la fois par fil, évaluée sur le processeur.
pub fn run(cfg: &Config, az: &Az, dir: &Path, gpu: Option<&Gpu>) -> std::io::Result<()> {
    let mut lanes = gpu.map(|gpu| gpu.lanes(cfg.threads)).unwrap_or_default();
    if let Some(lane) = lanes.first()
        && GPU_GAMES * smallmobile_lab7::az::CONSIDERED > lane.free()
    {
        return Err(std::io::Error::other("l'anneau du GPU ne tient pas les parties d'un fil"));
    }
    std::fs::create_dir_all(dir)?;
    let next = AtomicUsize::new(0);
    let (finished, planned) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let start = Instant::now();
    std::thread::scope(|scope| -> std::io::Result<()> {
        let mut handles = Vec::new();
        for t in 0..cfg.threads {
            let (next, finished, planned) = (&next, &finished, &planned);
            let lane = (!lanes.is_empty()).then(|| lanes.remove(0));
            handles.push(scope.spawn(move || -> std::io::Result<()> {
                let mut files = Files::create(&dir.join(format!("r{:x}-{t:02}", cfg.seed)), cfg.raw)?;
                let mut done = |g: Game_, segments: Vec<Segment>| -> std::io::Result<()> {
                    files.write(&g, &segments)?;
                    let count = finished.fetch_add(1, Ordering::Relaxed) + 1;
                    let plans = planned.fetch_add(segments.len(), Ordering::Relaxed) + segments.len();
                    if count % 50 == 0 || count == cfg.games {
                        let secs = start.elapsed().as_secs_f64();
                        eprintln!(
                            "{count}/{} parties, {plans} plans, {:.2} parties/s, {:.0} plans/s",
                            cfg.games,
                            count as f64 / secs,
                            plans as f64 / secs
                        );
                    }
                    Ok(())
                };
                match gpu {
                    Some(gpu) => {
                        let batch = Batch::new(&az.net, Some((gpu, lane.expect("les places du fil"), 0)));
                        let start_game = || {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            (i < cfg.games).then(|| Box::pin(game(i, cfg, az, &batch)) as Running)
                        };
                        let mut slots: Vec<Option<Running>> = (0..GPU_GAMES).map(|_| None).collect();
                        let mut ready: Vec<(usize, Option<Vec<Eval>>)> = (0..GPU_GAMES).map(|s| (s, None)).collect();
                        let mut waiting = std::collections::VecDeque::new();
                        loop {
                            while let Some((s, evals)) = ready.pop() {
                                if let Some(evals) = evals {
                                    batch.resume(evals);
                                }
                                if advance(&mut slots[s], start_game, &mut done)? {
                                    waiting
                                        .push_back((s, batch.send().expect("une partie qui attend a ses positions")));
                                }
                            }
                            let Some((s, sent)) = waiting.pop_front() else { break };
                            ready.push((s, Some(batch.receive(sent))));
                            while waiting.front().is_some_and(|(_, sent)| gpu.done(sent)) {
                                let (s, sent) = waiting.pop_front().expect("une attente");
                                ready.push((s, Some(batch.receive(sent))));
                            }
                        }
                    }
                    None => loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= cfg.games {
                            break;
                        }
                        let (g, segments) = smallmobile_lab7::az::now(game(i, cfg, az, &Direct));
                        done(g, segments)?;
                    },
                }
                drop(done);
                files.flush()
            }));
        }
        for h in handles {
            h.join().expect("fil des parties d'imitation")?;
        }
        Ok(())
    })
}
