//! Parties d'entraînement : une IA qui cherche joue contre elle-même, et chaque position cherchée à
//! fond devient un exemple pour le réseau (ce qu'elle voyait, la politique améliorée de la
//! recherche, puis le résultat final).
//!
//! Format des fichiers : des enregistrements de taille fixe (`RECORD` octets), lus en Python par
//! `numpy.fromfile` avec le type décrit dans `lab3/py/data.py` :
//!
//! ```text
//! u8 joueurs, u8 régions
//! f32      poids de la position (`weights` : surprise de la recherche)
//! f32 × 5  victoire de chaque joueur relatif (part, 0 à 1)
//! f32 × 5  pièces gagnées d'ici la fin par chaque joueur relatif
//! u8 × REGIONS  à qui appartient chaque région au début du prochain tour du joueur qui agit (à la
//!          fin de la partie si elle finit avant) : `NOBODY`, 1 + joueur relatif, `OFF_MAP`
//! u16 × KEYS  politique améliorée de chaque coup, × `PROBABILITY` (65535 : coup impossible)
//! u8 × REGIONS × 8  voisines (un bit par région, 64 bits par région)
//! u8 × REGIONS × NODE  entrées par région
//! u8 × GLOBAL  entrées de la partie
//! ```
//!
//! Fichiers `*.pos` (`SUFFIX`). À côté de chacun : `games_file` (positions de chaque partie) et
//! `moves_file` (de quoi rejouer chaque partie, `GameLog`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::state::Occupant;
use smallmobile_kernel::wire::{self, Writer};
use smallmobile_kernel::{Action, Game, Phase};

use crate::az::{Az, Evals, Oracle, Searched};
use crate::batch::Batch;
use crate::encode::{Encoded, GLOBAL, NODE, REGIONS, encode, relative};
use crate::gpu::Gpu;
use crate::know::Knowledge;
use crate::moves::{self, KEYS, Key};
use crate::net::Eval;
use crate::rand::{Rng, mix};
use crate::table::{Outcome, Table};

pub const RECORD: usize = 2 + 4 + 20 + 20 + REGIONS + KEYS * 2 + REGIONS * 8 + REGIONS * NODE + GLOBAL;
/// Le suffixe des fichiers de positions.
pub const SUFFIX: &str = ".pos";
pub const ILLEGAL: u16 = u16::MAX;
/// Une probabilité de la politique améliorée est enregistrée multipliée par ce nombre.
pub const PROBABILITY: f32 = 60_000.0;
/// Propriétaire d'une région : personne (vide ou peuple oublié) ; sinon 1 + joueur relatif.
pub const NOBODY: u8 = 0;
/// Propriétaire d'une région hors de la carte.
pub const OFF_MAP: u8 = u8::MAX;

/// Un exemple pour le réseau, cibles en joueurs relatifs (0 : celui qui agit).
pub struct Record<'a> {
    pub enc: &'a Encoded,
    pub map: &'a smallmobile_kernel::Map,
    pub weight: f32,
    pub win: [f32; MAX_PLAYERS],
    pub gain: [f32; MAX_PLAYERS],
    /// `owners`, au début du prochain tour du joueur qui agit.
    pub owners: [u8; REGIONS],
    pub policy: &'a [u16; KEYS],
}

impl Record<'_> {
    pub fn bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(RECORD);
        buf.extend_from_slice(&[self.enc.players as u8, self.enc.regions as u8]);
        buf.extend_from_slice(&self.weight.to_le_bytes());
        for x in self.win.iter().chain(&self.gain) {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&self.owners);
        for v in self.policy {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for i in 0..REGIONS {
            let mut bits = 0u64;
            if i < self.map.len() {
                for &a in &self.map.regions[i].adj {
                    bits |= 1 << a;
                }
            }
            buf.extend_from_slice(&bits.to_le_bytes());
        }
        for row in &self.enc.node {
            buf.extend_from_slice(row);
        }
        buf.extend_from_slice(&self.enc.global);
        debug_assert_eq!(buf.len(), RECORD);
        buf
    }
}

/// Une position cherchée à fond, gardée jusqu'à la fin de la partie (ses cibles) : son encodage
/// sans les lignes des régions absentes (toutes à zéro), sa politique sur ses seuls coups possibles
/// (~3,5 Ko au lieu de 5,8).
struct Pending {
    regions: u8,
    players: u8,
    node: Box<[[u8; NODE]]>,
    global: Box<[u8; GLOBAL]>,
    actor: usize,
    coins: [i32; MAX_PLAYERS],
    policy: Box<[(Key, u16)]>,
    owners: Option<[u8; REGIONS]>,
    /// Écart de la politique améliorée à l'a priori du réseau (`surprise`).
    surprise: f32,
}

impl Pending {
    fn new(e: &Encoded, actor: usize, coins: [i32; MAX_PLAYERS], policy: Box<[(Key, u16)]>, surprise: f32) -> Pending {
        Pending {
            regions: e.regions as u8,
            players: e.players as u8,
            node: e.node[..e.regions].into(),
            global: Box::new(e.global),
            actor,
            coins,
            policy,
            owners: None,
            surprise,
        }
    }

    /// L'encodage entier, tel que `encode` l'a donné.
    fn encoded(&self) -> Encoded {
        let mut e = Encoded {
            regions: self.regions as usize,
            players: self.players as usize,
            node: [[0; NODE]; REGIONS],
            global: *self.global,
        };
        e.node[..self.node.len()].copy_from_slice(&self.node);
        e
    }

    /// La politique sur toutes les clés (`ILLEGAL` : coup impossible).
    fn policy(&self) -> [u16; KEYS] {
        let mut policy = [ILLEGAL; KEYS];
        for &(key, p) in &self.policy {
            policy[key as usize] = p;
        }
        policy
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Part des coups cherchés à fond (et enregistrés) ; les autres le sont en `fast` itérations
    /// (« playout cap randomization » de KataGo : plus de parties pour le même calcul). Toutes les
    /// recherches de l'auto-apprentissage tirent leurs coups de racine avec le bruit de Gumbel.
    pub full_share: f64,
    pub fast: usize,
}

/// La recette des données : une recherche complète (enregistrée) pour 8 coups. La valeur apprend
/// un seul résultat par partie, quel que soit le nombre de ses positions : mieux vaut beaucoup de
/// parties que beaucoup de positions par partie (AlphaGo n'en gardait qu'une par partie pour sa
/// valeur).
pub const RECIPE: Options = Options { full_share: 0.125, fast: 50 };

/// Une partie jouée : son résultat, le nombre de positions écrites, les actions du noyau de chaque
/// coup (`moves`).
pub struct Played {
    pub outcome: Outcome,
    pub positions: usize,
    /// Les actions de toute la partie, à la suite ; la fin de celles de chaque coup.
    actions: Vec<Action>,
    ends: Vec<u32>,
}

impl Played {
    /// Les actions de chaque coup.
    pub fn moves(&self) -> impl Iterator<Item = &[Action]> {
        let starts = std::iter::once(0).chain(self.ends.iter().copied());
        starts.zip(&self.ends).map(|(a, &b)| &self.actions[a as usize..b as usize])
    }
}

/// Joue une partie ; écrit un enregistrement par position cherchée à fond (plus d'un coup
/// possible). Les évaluations du réseau viennent de `oracle` (`Direct` : une partie seule ; `Batch` :
/// avec les autres parties du fil, `run`).
pub async fn play(
    table: &Table,
    seed: u32,
    az: &Az,
    opts: Options,
    rng: &mut Rng,
    out: &mut impl Write,
    oracle: &impl Oracle,
) -> std::io::Result<Played> {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut pending: Vec<Pending> = Vec::new();
    let (mut actions, mut ends) = (Vec::new(), Vec::new());
    let mut evals = Evals::default();
    while g.state.phase != Phase::GameOver {
        let actor = g.state.actor().expect("un joueur doit agir");
        let legal = moves::legal(&g, &k);
        let mv = if legal.len() == 1 {
            legal[0].mv
        } else if rng.unit() >= opts.full_share {
            az.run(&g, &k, rng, opts.fast, true, &mut evals, oracle).await.mv
        } else {
            let s = az.run(&g, &k, rng, az.cfg.iterations, true, &mut evals, oracle).await;
            let policy = s.policy.iter().map(|&(key, p)| (key, (p * PROBABILITY).round() as u16)).collect();
            pending.push(Pending::new(&encode(&g, &k), actor, coins(&g), policy, surprise(&s)));
            SEARCHED.fetch_add(1, Ordering::Relaxed);
            s.mv
        };
        moves::actions_into(&g, &mv, &mut actions);
        ends.push(actions.len() as u32);
        let current = g.state.current;
        k.play(&mut g, &mv).expect("coup possible");
        // le tour passe : les positions de ce joueur qui attendaient son prochain tour l'ont
        if g.state.current != current {
            let c = g.state.current;
            for p in pending.iter_mut().filter(|p| p.actor == c && p.owners.is_none()) {
                p.owners = Some(owners(&g, c));
            }
        }
    }
    for p in pending.iter_mut().filter(|p| p.owners.is_none()) {
        p.owners = Some(owners(&g, p.actor));
    }
    let o = Outcome::of(&g);
    let n = o.players;
    let weights = weights(&pending.iter().map(|p| p.surprise).collect::<Vec<_>>());
    for (p, &weight) in pending.iter().zip(&weights) {
        let mut win = [0.0; MAX_PLAYERS];
        let mut gain = [0.0; MAX_PLAYERS];
        for r in 0..n {
            let q = (p.actor + r) % n;
            win[r] = o.win[q] as f32;
            gain[r] = (o.coins[q] - p.coins[q]) as f32;
        }
        let owners = p.owners.expect("propriétaires");
        let (enc, policy) = (p.encoded(), p.policy());
        let rec = Record { enc: &enc, map: &table.map, weight, win, gain, owners, policy: &policy };
        out.write_all(&rec.bytes())?;
    }
    Ok(Played { outcome: o, positions: pending.len(), actions, ends })
}

/// La surprise d'une recherche : divergence de Kullback-Leibler de sa politique améliorée (la
/// cible) à l'a priori du réseau.
fn surprise(s: &Searched) -> f32 {
    s.policy
        .iter()
        .filter(|&&(_, t)| t > 0.0)
        .map(|&(key, t)| {
            let prior = s.details.iter().find(|d| d.0 == key).map_or(0.0, |d| d.1);
            t * (t / prior.max(1e-9)).ln()
        })
        .sum::<f32>()
        .max(0.0)
}

/// Les poids des positions d'une partie, d'après leurs surprises (« policy surprise weighting » de
/// KataGo) : la moitié du poids total réparti également, l'autre moitié en proportion de la
/// surprise ; en moyenne 1 par position. L'entraînement tire les positions en proportion de leur
/// poids : celles où la recherche a appris quelque chose au réseau reviennent plus souvent.
pub fn weights(surprises: &[f32]) -> Vec<f32> {
    let mean = surprises.iter().sum::<f32>() / surprises.len().max(1) as f32;
    surprises.iter().map(|&x| if mean > 1e-6 { 0.5 + 0.5 * x / mean } else { 1.0 }).collect()
}

/// À qui appartient chaque région, vu du joueur `me` (`NOBODY`, 1 + joueur relatif, `OFF_MAP`).
pub fn owners(g: &Game, me: usize) -> [u8; REGIONS] {
    let s = &g.state;
    let n = s.player_count();
    let mut out = [OFF_MAP; REGIONS];
    for (o, r) in out.iter_mut().zip(&s.regions[..g.map.len()]) {
        *o = match r.occupant {
            Occupant::Race(race) => 1 + relative(s.race(race).owner, me, n) as u8,
            Occupant::Empty | Occupant::Lost => NOBODY,
        };
    }
    out
}

/// Les pièces de chaque joueur.
fn coins(g: &Game) -> [i32; MAX_PLAYERS] {
    let mut coins = [0; MAX_PLAYERS];
    for (c, p) in coins.iter_mut().zip(&g.state.players) {
        *c = p.coins;
    }
    coins
}

/// De quoi rejouer une partie : la donne et les actions du noyau de chaque coup. Une ligne par
/// partie dans `moves_file`, séparée par des tabulations : donne, joueurs, graine de la carte,
/// graine de la partie, pièces finales (séparées par des virgules), puis les coups séparés par des
/// espaces ; un coup est une suite d'actions séparées par « / », une action ses mots `wire`
/// séparés par des virgules.
#[derive(Clone, Debug, PartialEq)]
pub struct GameLog {
    pub deal: u64,
    pub players: usize,
    pub map_seed: u64,
    pub seed: u32,
    pub coins: Vec<i32>,
    /// Les mots de chaque action de chaque coup.
    pub moves: Vec<Vec<Vec<u32>>>,
}

impl GameLog {
    pub fn new(deal: u64, map_seed: u64, seed: u32, table: &Table, played: &Played) -> GameLog {
        let regions = table.map.len();
        let mut buffer = vec![0u32; 8 + regions];
        let moves = played
            .moves()
            .map(|actions| {
                actions
                    .iter()
                    .map(|a| {
                        let mut w = Writer::new(&mut buffer);
                        wire::encode_action(a, regions, &mut w);
                        w.written().to_vec()
                    })
                    .collect()
            })
            .collect();
        let coins = played.outcome.coins[..table.players].to_vec();
        GameLog { deal, players: table.players, map_seed, seed, coins, moves }
    }

    pub fn line(&self) -> String {
        let join = |xs: &[u32]| xs.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let coins: Vec<String> = self.coins.iter().map(i32::to_string).collect();
        let moves: Vec<String> =
            self.moves.iter().map(|m| m.iter().map(|a| join(a)).collect::<Vec<_>>().join("/")).collect();
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.deal,
            self.players,
            self.map_seed,
            self.seed,
            coins.join(","),
            moves.join(" ")
        )
    }

    pub fn parse(line: &str) -> Option<GameLog> {
        let mut f = line.split('\t');
        let (deal, players, map_seed, seed) =
            (f.next()?.parse().ok()?, f.next()?.parse().ok()?, f.next()?.parse().ok()?, f.next()?.parse().ok()?);
        let coins = f.next()?.split(',').map(|c| c.parse().ok()).collect::<Option<Vec<i32>>>()?;
        let words = |a: &str| a.split(',').map(|w| w.parse().ok()).collect::<Option<Vec<u32>>>();
        let moves = f.next()?.split(' ').map(|m| m.split('/').map(words).collect()).collect::<Option<Vec<_>>>()?;
        Some(GameLog { deal, players, map_seed, seed, coins, moves })
    }

    /// La table de la partie.
    pub fn table(&self) -> Table {
        Table::new(self.players, self.map_seed)
    }

    /// Rejoue la partie sur sa table : `visit` voit chaque position avant un coup, avec ce que sait
    /// le joueur qui agit. Renvoie la partie finie.
    pub fn replay<'m>(
        &self,
        table: &'m Table,
        mut visit: impl FnMut(&Game<'m>, &Knowledge),
    ) -> Result<Game<'m>, String> {
        let mut g = table.game(self.seed);
        let mut k = Knowledge::default();
        for mv in &self.moves {
            visit(&g, &k);
            let actions = self.actions(mv, table)?;
            // le coup de `moves` qui donne ces actions : la fin de tour reste suivie comme en jouant
            // (`Knowledge::redeploy`)
            let c = moves::legal(&g, &k)
                .into_iter()
                .find(|c| moves::actions(&g, &c.mv) == actions)
                .ok_or_else(|| format!("aucun coup ne donne {actions:?}"))?;
            k.play(&mut g, &c.mv).map_err(|e| format!("coup refusé {:?} : {e:?}", c.mv))?;
        }
        Ok(g)
    }

    /// Les actions d'un coup (`moves`) de cette partie.
    pub fn actions(&self, mv: &[Vec<u32>], table: &Table) -> Result<Vec<Action>, String> {
        mv.iter()
            .map(|words| wire::decode_action(words, table.map.len()).map_err(|e| format!("action illisible : {}", e.0)))
            .collect()
    }
}

pub struct Config {
    pub players: Vec<usize>,
    pub games: usize,
    pub threads: usize,
    pub seed: u64,
    pub opts: Options,
}

/// Le fichier qui accompagne un fichier de positions : le nombre de positions de chaque partie,
/// dans l'ordre (u32 petit-boutiste ; `GAMES` dans `lab3/py/data.py`).
pub fn games_file(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".games");
    name.into()
}

/// Le fichier qui accompagne un fichier de positions : une ligne `GameLog` par partie.
pub fn moves_file(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".moves");
    name.into()
}

/// Parties que mène chaque fil en même temps. Sur le processeur du fil, un groupe de 8 dont les
/// positions à évaluer sont évaluées ensemble, couche après couche (`Batch::submit` ; ses
/// évaluations gardées et les poids d'une couche tiennent dans le cache du cœur). Avec le GPU,
/// `GPU_GAMES` parties reprises une par une : chacune envoie les positions d'un tour de sa
/// recherche (~4 en moyenne, `az::CONSIDERED` au plus) dès qu'elle les a, et reprend dès qu'elles
/// sont évaluées ; le fil fait avancer les autres entre-temps. Peu de parties par fil : leurs arbres
/// tiennent dans les caches. Un fil garde tout ce qu'il touche (parties, évaluations gardées) :
/// rien de modifiable n'est partagé entre les cœurs, hormis le compteur des parties (une fois par
/// partie) et l'anneau du GPU.
const CONCURRENT: usize = 8;
const GPU_GAMES: usize = 24;

/// Une partie finie : ses positions (enregistrements) et sa ligne de journal (`GameLog`).
pub struct Finished {
    pub players: usize,
    pub positions: usize,
    pub records: Vec<u8>,
    pub line: String,
}

/// La partie `i` de l'auto-apprentissage `cfg` (donne, carte et hasard tirés de `cfg.seed` et `i`).
pub async fn game(i: usize, cfg: &Config, az: &Az, oracle: &impl Oracle) -> std::io::Result<Finished> {
    let players = cfg.players[i % cfg.players.len()];
    let deal = mix(cfg.seed, i as u64);
    let (map_seed, seed) = (deal % 1_000_000_007, mix(deal, 1) as u32);
    let table = Table::new(players, map_seed);
    let mut rng = Rng::new(mix(deal, 99));
    let mut records = Vec::new();
    let played = play(&table, seed, az, cfg.opts, &mut rng, &mut records, oracle).await?;
    let line = GameLog::new(deal, map_seed, seed, &table, &played).line();
    Ok(Finished { players, positions: played.positions, records, line })
}

/// Les positions retenues par toutes les parties, comptées dès leur recherche : une partie ne compte
/// les siennes qu'à sa fin, une à deux minutes après son début en selfplay GPU (`pace`).
pub static SEARCHED: AtomicUsize = AtomicUsize::new(0);

/// La période de `pace`.
const PACE: Duration = Duration::from_secs(10);

/// Le rythme du selfplay, toutes les `PACE` jusqu'à `stopped` : positions retenues par seconde sur la
/// dernière période, pas depuis le départ, pour lire le régime établi en une minute, quel que soit le
/// nombre de parties ; avec `profile`, la répartition du temps des fils de jeu sur la même période,
/// et les évaluations envoyées au GPU.
fn pace(stopped: &Receiver<()>, threads: usize, profile: bool) {
    use crate::batch::{SENT, SENT_LOTS, SENT_POSITIONS};
    let sent = || SENT.each_ref().map(|s| s.load(Ordering::Relaxed));
    let mut last_sent = sent();
    let now = || {
        (
            Instant::now(),
            SEARCHED.load(Ordering::Relaxed),
            crate::batch::TIMES.each_ref().map(|t| t.load(Ordering::Relaxed)),
        )
    };
    let mut last = now();
    while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(PACE) {
        let next = now();
        let secs = (next.0 - last.0).as_secs_f64();
        let spent = std::array::from_fn(|i| next.2[i] - last.2[i]);
        eprintln!("rythme : {:.0} positions/s sur les {secs:.0} dernières s", (next.1 - last.1) as f64 / secs);
        if profile {
            eprintln!("{}", split(spent, secs * threads as f64));
            let now_sent = sent();
            let (lots, positions) =
                (now_sent[SENT_LOTS] - last_sent[SENT_LOTS], now_sent[SENT_POSITIONS] - last_sent[SENT_POSITIONS]);
            eprintln!(
                "GPU : {:.0} évaluations/s, {:.0} lots/s de {:.0} positions en moyenne",
                positions as f64 / secs,
                lots as f64 / secs,
                positions as f64 / lots.max(1) as f64
            );
            last_sent = now_sent;
        }
        last = next;
    }
}

/// La répartition du temps des fils de jeu (`batch::TIMES` : `spent`, ns) sur `all` secondes de fil.
fn split(spent: [u64; 3], all: f64) -> String {
    use crate::batch::{TIME_DECODE, TIME_SUBMIT, TIME_WAIT};
    let share = |what: usize| 100.0 * spent[what] as f64 / 1e9 / all;
    let (submit, wait, decode) = (share(TIME_SUBMIT), share(TIME_WAIT), share(TIME_DECODE));
    format!(
        "fils de jeu : jeu et recherche {:.0} %, envoi {submit:.1} %, attente {wait:.0} %, décodage {decode:.1} %",
        100.0 - submit - wait - decode
    )
}

/// Une partie en cours (`game`).
type Running<'a> = Pin<Box<dyn Future<Output = std::io::Result<Finished>> + 'a>>;

/// La partie de la place `slot` avance jusqu'à ses prochaines évaluations ; une partie finie
/// (`record`) laisse sa place à la suivante (`start`). Faux : plus de partie à commencer.
fn advance<'a>(
    slot: &mut Option<(usize, Running<'a>)>,
    mut start: impl FnMut() -> Option<(usize, Running<'a>)>,
    record: &mut impl FnMut(usize, Finished) -> std::io::Result<()>,
) -> std::io::Result<bool> {
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if slot.is_none() {
            let Some(started) = start() else { return Ok(false) };
            *slot = Some(started);
        }
        let (i, running) = slot.as_mut().expect("une partie");
        let Poll::Ready(finished) = running.as_mut().poll(&mut cx) else { return Ok(true) };
        let i = *i;
        *slot = None;
        record(i, finished?)?;
    }
}

/// Des parties d'entraînement en parallèle, un fichier par fil dans `dir` (avec `games_file` et
/// `moves_file`). Chaque fil mène plusieurs parties (`CONCURRENT`, `GPU_GAMES`) : chacune avance
/// jusqu'à ses prochaines évaluations ; sur le processeur, le groupe est évalué d'un coup ; avec
/// `gpu`, chaque partie envoie les siennes et reprend dès qu'elles sont là. Sans GPU, chaque partie
/// est la même, au bit près, que jouée seule (`Direct`).
pub fn run(cfg: &Config, az: &Az, dir: &Path, gpu: Option<&Gpu>) -> std::io::Result<()> {
    // le GPU, une partie à la fois ; le processeur (ou les évaluations instantanées des profils,
    // `LAB3_INSTANT`), par groupes
    let stream = gpu.filter(|_| std::env::var_os("LAB3_INSTANT").is_none());
    let (mut groups, mut concurrent) = if stream.is_some() { (1, GPU_GAMES) } else { (1, CONCURRENT) };
    // essais et profils : `LAB3_GROUPS`, `LAB3_CONCURRENT`
    let env = |key: &str| std::env::var(key).ok().and_then(|v| v.parse().ok());
    groups = env("LAB3_GROUPS").unwrap_or(groups);
    concurrent = env("LAB3_CONCURRENT").unwrap_or(concurrent);
    // les places de chaque fil dans l'anneau du GPU : au plus un tour de recherche en vol par
    // partie, elles doivent tous les tenir
    let mut lanes = stream.map(|gpu| gpu.lanes(cfg.threads)).unwrap_or_default();
    if let Some(lane) = lanes.first()
        && concurrent * crate::az::CONSIDERED > lane.free()
    {
        return Err(std::io::Error::other(format!(
            "{concurrent} parties par fil : jusqu'à {} positions en vol, l'anneau du GPU en tient {} par fil",
            concurrent * crate::az::CONSIDERED,
            lane.free()
        )));
    }
    std::fs::create_dir_all(dir)?;
    let next = AtomicUsize::new(0);
    let positions = AtomicUsize::new(0);
    let start = Instant::now();
    let profile = gpu.is_some() && std::env::var_os("LAB3_VULKAN_PROFILE").is_some();
    std::thread::scope(|scope| -> std::io::Result<()> {
        // `pace` s'arrête quand `stop` disparaît, à la fin de cette fonction
        let (stop, stopped) = std::sync::mpsc::channel::<()>();
        scope.spawn(move || pace(&stopped, cfg.threads, profile));
        let mut handles = Vec::new();
        for t in 0..cfg.threads {
            let (next, positions) = (&next, &positions);
            let lane = (!lanes.is_empty()).then(|| lanes.remove(0));
            // nommé, pour les profils (`perf report --sort comm`)
            let thread = std::thread::Builder::new().name(format!("jeu-{t:02}"));
            handles.push(thread.spawn_scoped(scope, move || -> std::io::Result<()> {
                let path = dir.join(format!("s{:x}-{t:02}{SUFFIX}", cfg.seed));
                let mut out = BufWriter::new(File::create(&path)?);
                let mut games = BufWriter::new(File::create(games_file(&path))?);
                let mut logs = BufWriter::new(File::create(moves_file(&path))?);
                let mut record = |i: usize, finished: Finished| -> std::io::Result<()> {
                    out.write_all(&finished.records)?;
                    games.write_all(&(finished.positions as u32).to_le_bytes())?;
                    writeln!(logs, "{}", finished.line)?;
                    let count = finished.positions;
                    let total = positions.fetch_add(count, Ordering::Relaxed) + count;
                    let secs = start.elapsed().as_secs_f64();
                    eprintln!(
                        "partie {}/{} ({} j.) : {count} positions, {total} en tout, {:.0} positions/s",
                        i + 1,
                        cfg.games,
                        finished.players,
                        total as f64 / secs
                    );
                    Ok(())
                };
                // la partie suivante, jouée avec les évaluations de `batch`
                let start_game = |batch| {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    (i < cfg.games).then(|| (i, Box::pin(game(i, cfg, az, batch)) as Running))
                };
                if let Some(gpu) = stream {
                    let batch = Batch::new(&az.net, Some((gpu, lane.expect("les places du fil"))));
                    let mut slots: Vec<Option<(usize, Running)>> = (0..concurrent).map(|_| None).collect();
                    // les parties à reprendre, avec leurs évaluations ; celles qui attendent le GPU,
                    // dans l'ordre de leurs envois (celui où il les évalue)
                    let mut ready: Vec<(usize, Option<Vec<Eval>>)> = (0..concurrent).map(|s| (s, None)).collect();
                    let mut waiting = std::collections::VecDeque::new();
                    loop {
                        while let Some((s, evals)) = ready.pop() {
                            if let Some(evals) = evals {
                                batch.resume(evals);
                            }
                            if advance(&mut slots[s], || start_game(&batch), &mut record)? {
                                waiting.push_back((s, batch.send().expect("une partie qui attend a ses positions")));
                            }
                        }
                        // la plus vieille attente, puis celles déjà évaluées
                        let Some((s, sent)) = waiting.pop_front() else { break };
                        ready.push((s, Some(batch.receive(sent))));
                        while waiting.front().is_some_and(|(_, sent)| gpu.done(sent)) {
                            let (s, sent) = waiting.pop_front().expect("une attente");
                            ready.push((s, Some(batch.receive(sent))));
                        }
                    }
                } else {
                    let batches: Vec<Batch> = (0..groups).map(|_| Batch::new(&az.net, None)).collect();
                    let mut slots: Vec<Vec<Option<(usize, Running)>>> =
                        (0..groups).map(|_| (0..concurrent).map(|_| None).collect()).collect();
                    loop {
                        let mut live = false;
                        for (batch, slots) in batches.iter().zip(&mut slots) {
                            for slot in slots.iter_mut() {
                                advance(slot, || start_game(batch), &mut record)?;
                            }
                            if slots.iter().any(Option::is_some) {
                                live = true;
                                batch.submit();
                            }
                        }
                        if !live {
                            break;
                        }
                    }
                }
                drop(record);
                out.flush()?;
                logs.flush()?;
                games.flush()
            })?);
        }
        for h in handles {
            h.join().expect("fil d'auto-apprentissage")?;
        }
        drop(stop);
        if profile {
            let spent = crate::batch::TIMES.each_ref().map(|t| t.load(Ordering::Relaxed));
            eprintln!("{}", split(spent, start.elapsed().as_secs_f64() * cfg.threads as f64));
        }
        Ok(())
    })
}

/// Relit un enregistrement : la position encodée et les voisines de chaque région.
pub fn read_record(rec: &[u8]) -> (Encoded, Vec<Vec<u8>>) {
    assert_eq!(rec.len(), RECORD);
    let players = rec[0] as usize;
    let regions = rec[1] as usize;
    let mut at = 2 + 4 + 20 + 20 + REGIONS + KEYS * 2;
    let mut adj = Vec::with_capacity(regions);
    for i in 0..REGIONS {
        let bits = u64::from_le_bytes(rec[at..at + 8].try_into().expect("8 octets"));
        at += 8;
        if i < regions {
            adj.push((0..64u8).filter(|&b| bits >> b & 1 == 1).collect());
        }
    }
    let mut e = Encoded { regions, players, node: [[0; NODE]; REGIONS], global: [0; GLOBAL] };
    for row in e.node.iter_mut() {
        row.copy_from_slice(&rec[at..at + NODE]);
        at += NODE;
    }
    e.global.copy_from_slice(&rec[at..at + GLOBAL]);
    (e, adj)
}
