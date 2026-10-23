//! Recherche guidée par le réseau : Gumbel AlphaZero (Danihelka et al., *Policy improvement by
//! planning with Gumbel*, 2022), avec des nœuds de hasard.
//!
//! Chaque nœud est une position publique, telle que la voient les joueurs ; ses arêtes sont les
//! coups possibles. La suite d'un coup se range par résultat public de son hasard (dé, carte arrivée
//! au marché : `moves::chance`) : la recherche ne mêle pas deux résultats, et une décision ne dépend
//! jamais d'un hasard encore inconnu. Chaque itération tire un monde possible (ordre caché des
//! pioches, dés à venir : `determinize`), si bien que les résultats arrivent avec leur vraie
//! probabilité. Les coups forcés se jouent sans nœud.
//!
//! - À la racine, `CONSIDERED` coups sont tirés sans remise selon les logits du réseau plus un bruit
//!   de Gumbel (`explore` ; sans lui, les plus probables), puis départagés par halving séquentiel :
//!   les itérations se répartissent entre les coups retenus, et la moitié la moins bonne est écartée
//!   à chaque étape. Le coup joué est le dernier qui reste.
//! - Ailleurs, la sélection est déterministe : le coup qui rapproche le plus les visites de la
//!   politique améliorée (softmax des logits plus les valeurs des coups, `completed`).
//! - Une position nouvelle est jugée par le réseau (sa valeur), sans fin simulée.
//!
//! Chaque joueur maximise sa propre valeur. La politique améliorée de la racine est la cible de
//! politique de l'auto-apprentissage. Le réseau ne voit que ce que sait le joueur (`encode`).

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::{Game, Map, Phase};

use crate::encode::{Encoded, encode, encode_with};
use crate::know::{Knowledge, determinize};
use crate::moves::{self, Choice, Key, Move};
use crate::net::{Eval, Net};
use crate::outcome::Outcome;
use crate::rand::Rng;

/// Coups retenus à la racine : aussi les positions qu'une recherche fait évaluer ensemble, au plus
/// (un tour du halving, `Az::run`).
pub const CONSIDERED: usize = 16;
/// Les valeurs complétées des coups, ramenées entre 0 et 1 dans le nœud, sont multipliées par
/// `(VISIT_INIT + visites du coup le plus vu) × VALUE_SCALE` (réglages de mctx, la bibliothèque de
/// DeepMind) : plus la recherche avance, plus elles l'emportent sur les logits du réseau.
const VISIT_INIT: f32 = 50.0;
const VALUE_SCALE: f32 = 0.1;
/// Poids de la probabilité de victoire dans la valeur ; le reste vient de l'avance en pièces prévue
/// sur le meilleur rival (pièces actuelles + pièces que le réseau prévoit d'ici la fin), sigmoïde
/// d'échelle `SCALE`.
const WIN_WEIGHT: f32 = 0.5;
const SCALE: f32 = 8.0;

/// Les réglages de la recherche ; les autres sont fixes (constantes ci-dessus).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub iterations: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config { iterations: 200 }
    }
}

impl Config {
    /// Lit des réglages du type `n=200`.
    pub fn parse(text: &str) -> Option<Config> {
        let mut cfg = Config::default();
        for part in text.split(',').filter(|p| !p.is_empty()) {
            let (k, v) = part.split_once('=')?;
            match k {
                "n" => cfg.iterations = v.parse().ok()?,
                _ => return None,
            }
        }
        Some(cfg)
    }
}

/// Par joueur : la valeur (0 à 1), les pièces finales attendues (`COINS + joueur`), puis la part de
/// victoire attendue seule (`WINS + joueur`), sans l'avance en pièces que mêle la valeur.
type Value = [f32; VALUE];
const VALUE: usize = 3 * MAX_PLAYERS;
const COINS: usize = MAX_PLAYERS;
const WINS: usize = 2 * MAX_PLAYERS;
type PerPlayer = [f32; MAX_PLAYERS];

/// Pas de position : une arête jamais suivie, la fin d'une liste de `Chance`.
const NONE: u32 = u32::MAX;

/// Ce que la sélection lit de chaque coup d'un nœud, à chaque passage (`Tree::stats`, 16 octets) :
/// son logit, sa probabilité a priori (la softmax des logits du nœud, calculée une fois), ses
/// visites et la somme de leurs valeurs pour le joueur qui agit au nœud (seule valeur qui compte
/// pour ses coups).
#[derive(Clone, Copy)]
struct Stat {
    logit: f32,
    prior: f32,
    value: f32,
    visits: u32,
}

/// Le reste d'un coup, lu seulement quand il est joué (`Tree::edges`). Les positions atteintes se
/// rangent par résultat du hasard du coup et des coups forcés qui suivent (`moves::chance`,
/// empreinte `hash`) : la première dans l'arête, les autres (coups à hasard seulement) dans
/// `Tree::chances`, à partir de `more`. Aucune si la partie s'y termine.
struct Edge {
    hash: u64,
    child: u32,
    more: u32,
    key: Key,
    mv: Move,
}

/// Un autre résultat du hasard d'un coup : sa position, et le suivant (`NONE` : le dernier).
struct Chance {
    hash: u64,
    child: u32,
    next: u32,
}

/// Une position de l'arbre : ses coups (`Tree::stats` et `Tree::edges`, `first..first + len`), le
/// joueur qui agit, et la valeur que lui donne le réseau. Les visites et les valeurs de toute la position ne servent
/// qu'à la racine (`Az::run`).
#[derive(Clone, Copy)]
struct Node {
    first: u32,
    len: u32,
    /// La valeur du réseau pour le joueur qui agit.
    raw: f32,
    actor: u8,
}

/// L'arbre d'une recherche, et ses tampons, gardés d'une recherche à la suivante (dans `Evals`, un
/// par partie) : plus d'allocation une fois assez grands. Nœuds de 16 octets, coups en 16 octets
/// lus à chaque passage plus 24 lus seulement pour le coup joué, rangés à la suite : ceux de
/// plusieurs dizaines de parties tiennent dans le cache d'un cœur.
#[derive(Default)]
struct Tree {
    nodes: Vec<Node>,
    stats: Vec<Stat>,
    edges: Vec<Edge>,
    chances: Vec<Chance>,
    /// Le chemin d'une itération : (nœud, coup dans `edges`).
    path: Vec<(u32, u32)>,
    scratch: Scratch,
}

impl Tree {
    fn clear(&mut self) {
        self.nodes.clear();
        self.stats.clear();
        self.edges.clear();
        self.chances.clear();
    }

    fn stats(&self, n: Node) -> &[Stat] {
        &self.stats[n.first as usize..(n.first + n.len) as usize]
    }

    fn edges(&self, n: Node) -> &[Edge] {
        &self.edges[n.first as usize..(n.first + n.len) as usize]
    }

    /// La position atteinte par le coup `e` (indice dans `edges`) avec ce résultat du hasard.
    fn child(&self, e: usize, hash: u64) -> Option<u32> {
        let edge = &self.edges[e];
        if edge.child == NONE {
            return None;
        }
        if edge.hash == hash {
            return Some(edge.child);
        }
        let mut at = edge.more;
        while at != NONE {
            let c = &self.chances[at as usize];
            if c.hash == hash {
                return Some(c.child);
            }
            at = c.next;
        }
        None
    }

    fn add_child(&mut self, e: usize, hash: u64, child: u32) {
        let more = self.edges[e].more;
        let edge = &mut self.edges[e];
        if edge.child == NONE {
            (edge.child, edge.hash) = (child, hash);
        } else {
            edge.more = self.chances.len() as u32;
            self.chances.push(Chance { hash, child, next: more });
        }
    }
}

/// La valeur moyenne d'un coup pour le joueur qui agit ; `None` s'il n'a jamais été essayé.
fn q(e: &Stat) -> Option<f32> {
    (e.visits > 0).then(|| e.value / e.visits as f32)
}

impl Node {
    /// Les valeurs complétées des coups, transformées pour s'ajouter aux logits : un coup jamais
    /// essayé reçoit la valeur mêlée du nœud (valeur du réseau et valeurs des coups essayés,
    /// pondérées par leur probabilité a priori) ; puis toutes sont ramenées entre 0 et 1 et
    /// multipliées par `(VISIT_INIT + visites du coup le plus vu) × VALUE_SCALE`.
    /// Dans `s.values`.
    fn completed_into(self, edges: &[Stat], s: &mut Scratch) {
        let visits: u32 = edges.iter().map(|e| e.visits).sum();
        let raw = self.raw;
        let mixed = if visits == 0 {
            raw
        } else {
            let (mut mass, mut weighted) = (0.0f32, 0.0f32);
            for e in edges {
                if let Some(q) = q(e) {
                    mass += e.prior;
                    weighted += e.prior * q;
                }
            }
            (raw + visits as f32 * weighted / mass.max(f32::MIN_POSITIVE)) / (visits as f32 + 1.0)
        };
        let values = &mut s.values;
        values.clear();
        values.extend(edges.iter().map(|e| q(e).unwrap_or(mixed)));
        let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let most = edges.iter().map(|e| e.visits).max().unwrap_or(0);
        let scale = (VISIT_INIT + most as f32) * VALUE_SCALE;
        values.iter_mut().for_each(|v| *v = scale * (*v - lo) / (hi - lo).max(1e-8));
    }

    /// La politique améliorée : softmax des logits plus les valeurs complétées.
    fn improved(self, edges: &[Stat]) -> Vec<f32> {
        let mut s = Scratch::default();
        self.improved_into(edges, &mut s);
        s.values
    }

    /// `improved`, dans `s.values`.
    fn improved_into(self, edges: &[Stat], s: &mut Scratch) {
        self.completed_into(edges, s);
        s.values.iter_mut().zip(edges).for_each(|(c, e)| *c += e.logit);
        softmax_in_place(&mut s.values);
    }

    /// Le coup à suivre hors de la racine : celui dont la part des visites est la plus en retard
    /// sur la politique améliorée.
    fn select(self, edges: &[Stat], s: &mut Scratch) -> usize {
        let visits: u32 = edges.iter().map(|e| e.visits).sum();
        self.improved_into(edges, s);
        let improved = &s.values;
        let score = |i: usize| improved[i] - edges[i].visits as f32 / (1.0 + visits as f32);
        (0..edges.len()).fold(0, |best, i| if score(i) > score(best) { i } else { best })
    }
}

/// Les tampons des calculs d'un nœud (`Node::select`), gardés d'une simulation à la suivante : ces
/// calculs ne demandent plus d'allocation (une dizaine par nœud traversé auparavant).
#[derive(Default)]
struct Scratch {
    values: Vec<f32>,
}

/// La softmax de `xs`, à sa place : le maximum, les exponentielles, leur somme, puis chacune divisée
/// par la somme.
fn softmax_in_place(xs: &mut [f32]) {
    let top = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    xs.iter_mut().for_each(|x| *x = (*x - top).exp());
    let sum: f32 = xs.iter().sum();
    xs.iter_mut().for_each(|e| *e /= sum);
}

/// Le calendrier du halving séquentiel (`get_sequence_of_considered_visits` de mctx) : pour chaque
/// itération, le nombre de visites qu'a déjà le coup de la racine à suivre. `m` coups retenus,
/// `n` itérations ; à chaque étape, chaque coup encore retenu reçoit autant de visites, puis la
/// moitié est écartée (il en reste au moins 2).
fn considered_visits(m: usize, n: usize) -> Vec<u32> {
    if m <= 1 {
        return (0..n as u32).collect();
    }
    let log2 = (m as f64).log2().ceil() as usize;
    let mut out = Vec::with_capacity(n);
    let mut visits = vec![0u32; m];
    let mut considered = m;
    while out.len() < n {
        let extra = (n / (log2 * considered)).max(1);
        for _ in 0..extra {
            out.extend_from_slice(&visits[..considered]);
            visits[..considered].iter_mut().for_each(|v| *v += 1);
        }
        considered = (considered / 2).max(2);
    }
    out.truncate(n);
    out
}

/// Une simulation d'un tour (`Az::run`), descendue : son chemin (`Tree::path`, de `path.0` à
/// `path.1`), puis une partie finie ou une position nouvelle, atteinte par le coup `edge` avec ce
/// résultat du hasard, et l'empreinte de son évaluation.
struct Leaf {
    path: (usize, usize),
    end: End,
}

enum End {
    Over(Value),
    New { seat: Seat, legal: Vec<Choice>, edge: usize, hash: u64, fp: u128 },
}

/// Ce que lit `Az::judge` d'une position : le joueur qui agit, le nombre de joueurs, leurs pièces.
#[derive(Clone, Copy)]
struct Seat {
    me: usize,
    n: usize,
    coins: PerPlayer,
}

impl Seat {
    fn of(w: &Game) -> Seat {
        let s = &w.state;
        let mut coins = [0.0; MAX_PLAYERS];
        coins.iter_mut().zip(&s.players).for_each(|(c, p)| *c = p.coins as f32);
        Seat { me: s.actor().expect("un joueur doit agir"), n: s.player_count(), coins }
    }
}

/// Un tirage de la loi de Gumbel.
fn gumbel(rng: &mut Rng) -> f32 {
    -(-(rng.unit().max(1e-12)).ln()).ln() as f32
}

/// D'où viennent les évaluations du réseau : tout de suite (`Direct` : le jeu, l'arène), ou
/// groupées avec celles d'autres parties (la génération de parties du labo) ; la recherche attend
/// les siennes. L'évaluation attendue ne garde pas la position (`use<'s, Self>`) : la recherche en
/// demande plusieurs à la fois (`Az::run`).
pub trait Oracle {
    fn eval<'s>(&'s self, net: &Net, e: &Encoded, map: &Map) -> impl Future<Output = Eval> + use<'s, Self>;
}

/// Chaque évaluation tout de suite, seule.
pub struct Direct;

impl Oracle for Direct {
    fn eval<'s>(&'s self, net: &Net, e: &Encoded, map: &Map) -> impl Future<Output = Eval> + use<'s> {
        std::future::ready(net.eval(e, map))
    }
}

/// Toutes les évaluations demandées, dans l'ordre, une fois arrivées.
async fn all<F: Future>(mut futures: Vec<Pin<Box<F>>>) -> Vec<F::Output> {
    let mut done: Vec<Option<F::Output>> = futures.iter().map(|_| None).collect();
    std::future::poll_fn(|cx| {
        let mut ready = true;
        for (f, d) in futures.iter_mut().zip(&mut done) {
            if d.is_none() {
                match f.as_mut().poll(cx) {
                    Poll::Ready(out) => *d = Some(out),
                    Poll::Pending => ready = false,
                }
            }
        }
        if ready { Poll::Ready(()) } else { Poll::Pending }
    })
    .await;
    done.into_iter().map(|d| d.expect("arrivée")).collect()
}

/// Le résultat d'un calcul qui n'attend jamais (évaluations `Direct`).
pub fn now<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(out) => out,
        Poll::Pending => unreachable!("une évaluation directe n'attend pas"),
    }
}

/// Les évaluations du réseau déjà faites, gardées d'une recherche à la suivante : un `Evals` par
/// partie (ou par décision). Une même position revient, dans une recherche (coups joués dans un
/// autre ordre) comme d'une recherche à la suivante (la décision d'après repasse par les positions
/// que la précédente a explorées) ; le réseau répond toujours pareil à la même entrée sur la même
/// carte : reprendre sa réponse ne change rien à la recherche, au bit près. Restent celles de la
/// recherche en cours et de la précédente ; sur une autre carte ou avec un autre réseau, tout est
/// oublié. Chacune tient en quelques centaines d'octets (`Judged`, `fingerprint`), dans une seule
/// table, une seule allocation (ses logits) : celles de plusieurs parties tiennent ensemble dans le
/// cache d'un cœur.
#[derive(Default)]
pub struct Evals {
    /// la carte et le réseau des évaluations gardées (adresses)
    owner: (usize, usize),
    /// chacune avec la dernière recherche qui s'en est servie (`search`)
    known: HashMap<u128, Kept>,
    /// le numéro de la recherche en cours
    search: u32,
    /// l'arbre des recherches de la partie, réutilisé de l'une à la suivante
    tree: Tree,
}

struct Kept {
    search: u32,
    judged: Judged,
}

/// Ce que la recherche garde d'une évaluation : la valeur, et les logits des coups possibles dans
/// l'ordre de `moves::legal`.
struct Judged {
    win: [f32; 5],
    gain: [f32; 5],
    logits: Box<[f32]>,
}

impl Evals {
    /// Une recherche commence : les évaluations dont la précédente s'est servie restent, les plus
    /// vieilles s'en vont.
    fn next(&mut self, net: &Net, map: &Map) {
        let owner = (std::ptr::from_ref(map) as usize, std::ptr::from_ref(net) as usize);
        self.search += 1;
        if owner != self.owner {
            self.owner = owner;
            self.known.clear();
        } else {
            let search = self.search;
            self.known.retain(|_, k| k.search + 1 >= search);
        }
    }

    /// L'évaluation d'empreinte `fp` est-elle gardée ? Elle sert alors à la recherche en cours.
    fn touch(&mut self, fp: u128) -> bool {
        let search = self.search;
        self.known.get_mut(&fp).map(|k| k.search = search).is_some()
    }

    /// Garde une évaluation du réseau : sa valeur et les logits des coups possibles.
    fn keep(&mut self, fp: u128, eval: Eval, legal: &[Choice]) {
        let logits = legal.iter().map(|c| eval.logits[c.key as usize]).collect();
        let judged = Judged { win: eval.win, gain: eval.gain, logits };
        self.known.insert(fp, Kept { search: self.search, judged });
    }

    fn judged(&self, fp: u128) -> &Judged {
        &self.known[&fp].judged
    }
}

/// L'empreinte de 128 bits de ce que lit le réseau et des coups possibles : deux hachages à la
/// façon de wyhash (produits de 128 bits repliés), de graines différentes. Deux positions
/// différentes ont la même avec une chance de l'ordre de 2⁻¹²⁸.
fn fingerprint(e: &Encoded, legal: &[Choice]) -> u128 {
    const SECRET: [u64; 4] =
        [0xa076_1d64_78bd_642f, 0xe703_7ed1_a0b4_28db, 0x8ebc_6af0_9c88_c6e3, 0x5899_65cc_7537_4cc3];
    fn mum(a: u64, b: u64) -> u64 {
        let r = u128::from(a) * u128::from(b);
        (r as u64) ^ ((r >> 64) as u64)
    }
    let mut lanes = [SECRET[0], SECRET[1]];
    let mut pair = |w0: u64, w1: u64| {
        lanes[0] = mum(w0 ^ SECRET[2], w1 ^ lanes[0]);
        lanes[1] = mum(w1 ^ SECRET[3], w0 ^ lanes[1]);
    };
    let mut bytes = |bytes: &[u8]| {
        let (pairs, rest) = bytes.as_chunks::<16>();
        for p in pairs {
            let (w0, w1) = p.split_at(8);
            pair(u64::from_le_bytes(w0.try_into().unwrap()), u64::from_le_bytes(w1.try_into().unwrap()));
        }
        if !rest.is_empty() {
            let mut t = [0u8; 16];
            t[..rest.len()].copy_from_slice(rest);
            let (w0, w1) = t.split_at(8);
            pair(
                u64::from_le_bytes(w0.try_into().unwrap()),
                u64::from_le_bytes(w1.try_into().unwrap()) ^ rest.len() as u64,
            );
        }
    };
    bytes(&[e.regions as u8, e.players as u8]);
    e.node[..e.regions].iter().for_each(|row| bytes(row));
    bytes(&e.global);
    let keys: Vec<u8> = legal.iter().flat_map(|c| c.key.to_le_bytes()).collect();
    bytes(&keys);
    (u128::from(mum(lanes[0], SECRET[1])) << 64) | u128::from(mum(lanes[1], SECRET[0]))
}

/// Le résultat d'une recherche.
pub struct Searched {
    pub mv: Move,
    /// La politique améliorée de la racine, par coup : la cible de politique de l'auto-apprentissage.
    pub policy: Vec<(Key, f32)>,
    /// Les visites de chaque coup de la racine.
    pub visits: Vec<(Key, u32)>,
    /// Pièces finales attendues de chaque joueur (réel) selon la recherche ; NaN si inconnues.
    pub finals: [f32; MAX_PLAYERS],
    /// Part de victoire attendue de chaque joueur (réel) selon la recherche ; NaN si inconnue.
    pub wins: [f32; MAX_PLAYERS],
    /// Pour chaque coup de la racine : a priori du réseau et valeur moyenne pour le joueur qui agit.
    pub details: Vec<(Key, f32, f32)>,
    /// Valeur de la racine pour le joueur qui agit ; NaN pour un coup forcé, joué sans chercher.
    pub value: f32,
}

pub struct Az {
    pub cfg: Config,
    pub net: Arc<Net>,
    /// Nom affiché (le fichier de poids).
    pub label: String,
}

/// La valeur du réseau (joueurs relatifs) remise en joueurs réels.
fn absolute(win: &[f32; 5], me: usize, n: usize) -> PerPlayer {
    let mut v = [0.0; MAX_PLAYERS];
    for (r, &w) in win.iter().enumerate().take(n) {
        v[(me + r) % n] = w;
    }
    v
}

/// Mêle au résultat d'un coup celui de son hasard (`moves::chance`).
fn fold(hash: u64, chance: u64) -> u64 {
    (hash ^ chance).wrapping_mul(0x0100_0000_01b3)
}

impl Az {
    pub fn new(cfg: Config, net: Arc<Net>, label: String) -> Self {
        Az { cfg, net, label }
    }

    /// Le nom affiché : le réseau et les réglages.
    pub fn describe(&self) -> String {
        format!("az({}, n={})", self.label, self.cfg.iterations)
    }

    /// Une position nouvelle, évaluée (`judged`) : ses coups et leurs logits, et la valeur que lui
    /// donne le réseau. Ses coups vont à la suite de ceux de `tree`, le nœud à la fin de ses nœuds.
    fn expand(&self, tree: &mut Tree, seat: &Seat, legal: &[Choice], judged: &Judged) -> Value {
        let value = self.judge(seat, judged);
        let first = tree.edges.len() as u32;
        // les probabilités a priori : la softmax des logits, une fois pour toutes
        let mut priors = std::mem::take(&mut tree.scratch.values);
        priors.clear();
        priors.extend_from_slice(&judged.logits);
        softmax_in_place(&mut priors);
        tree.stats.extend(judged.logits.iter().zip(&priors).map(|(&logit, &prior)| Stat {
            logit,
            prior,
            value: 0.0,
            visits: 0,
        }));
        tree.scratch.values = priors;
        tree.edges.extend(legal.iter().map(|c| Edge { hash: 0, child: NONE, more: NONE, key: c.key, mv: c.mv }));
        let actor = seat.me;
        tree.nodes.push(Node { first, len: legal.len() as u32, raw: value[actor], actor: actor as u8 });
        value
    }

    /// La valeur que le réseau donne à une position (joueurs réels).
    fn judge(&self, seat: &Seat, eval: &Judged) -> Value {
        let Seat { me, n, coins } = *seat;
        let mut finals = [0.0f32; MAX_PLAYERS];
        for r in 0..n {
            let q = (me + r) % n;
            finals[q] = coins[q] + eval.gain[r];
        }
        self.blend(&absolute(&eval.win, me, n), &finals, n)
    }

    /// La valeur d'une partie finie.
    fn terminal(&self, g: &Game) -> Value {
        let o = Outcome::of(g);
        let (mut win, mut finals) = ([0.0; MAX_PLAYERS], [0.0; MAX_PLAYERS]);
        for q in 0..o.players {
            win[q] = o.win[q] as f32;
            finals[q] = o.coins[q] as f32;
        }
        self.blend(&win, &finals, o.players)
    }

    /// La valeur de chaque joueur : victoire et avance en pièces (finales ou prévues) mêlées.
    fn blend(&self, win: &PerPlayer, finals: &PerPlayer, n: usize) -> Value {
        let mut v = [0.0; VALUE];
        for q in 0..n {
            v[COINS + q] = finals[q];
            v[WINS + q] = win[q];
            let rival = (0..n).filter(|&o| o != q).map(|o| finals[o]).fold(f32::NEG_INFINITY, f32::max);
            let lead = 1.0 / (1.0 + (-(finals[q] - rival) / SCALE).exp());
            v[q] = WIN_WEIGHT * win[q] + (1.0 - WIN_WEIGHT) * lead;
        }
        v
    }

    /// Une recherche de `cfg.iterations` itérations. `explore` : bruit de Gumbel à la racine
    /// (auto-apprentissage) ; sans lui, la recherche ne dépend du hasard que par les mondes possibles.
    /// `evals` : les évaluations gardées de la partie (ou de la décision). Chaque évaluation tout de
    /// suite (`Direct`).
    pub fn search(&self, root: &Game, k: &Knowledge, rng: &mut Rng, explore: bool, evals: &mut Evals) -> Searched {
        self.search_with(root, k, rng, self.cfg.iterations, explore, evals)
    }

    pub fn search_with(
        &self,
        root: &Game,
        k: &Knowledge,
        rng: &mut Rng,
        iterations: usize,
        explore: bool,
        evals: &mut Evals,
    ) -> Searched {
        now(self.run(root, k, rng, iterations, explore, evals, &Direct))
    }

    /// La recherche, qui attend chaque évaluation du réseau de `oracle`.
    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        &self,
        root: &Game<'_>,
        k: &Knowledge,
        rng: &mut Rng,
        iterations: usize,
        explore: bool,
        evals: &mut Evals,
        oracle: &impl Oracle,
    ) -> Searched {
        let legal = moves::legal(root, k);
        if legal.len() == 1 {
            return Searched {
                mv: legal[0].mv,
                policy: vec![(legal[0].key, 1.0)],
                visits: vec![(legal[0].key, 0)],
                finals: [f32::NAN; MAX_PLAYERS],
                wins: [f32::NAN; MAX_PLAYERS],
                details: vec![],
                value: f32::NAN,
            };
        }
        evals.next(&self.net, root.map);
        // l'arbre de la partie, sorti de `evals` le temps de la recherche
        let mut tree = std::mem::take(&mut evals.tree);
        tree.clear();
        // la racine : ses visites et ses valeurs, pour le résultat
        let encoded = encode(root, k);
        let fp = fingerprint(&encoded, &legal);
        if !evals.touch(fp) {
            let eval = oracle.eval(&self.net, &encoded, root.map).await;
            evals.keep(fp, eval, &legal);
        }
        let mut root_total = self.expand(&mut tree, &Seat::of(root), &legal, evals.judged(fp));
        let mut root_visits = 1u32;
        let noise: Vec<f32> = legal.iter().map(|_| if explore { gumbel(rng) } else { 0.0 }).collect();
        let schedule = considered_visits(CONSIDERED.min(legal.len()), iterations);
        // le score d'un coup de la racine (`root_pick`, `root_picks`)
        let root_score = |root: Node, edges: &[Stat], s: &mut Scratch| {
            root.completed_into(edges, s);
            s.values.iter_mut().enumerate().for_each(|(i, c)| *c += noise[i] + edges[i].logit);
        };
        // le meilleur coup de la racine parmi ceux qui ont `visits` visites
        let root_pick = |root: Node, edges: &[Stat], visits: u32, s: &mut Scratch| -> usize {
            root_score(root, edges, s);
            let score = &s.values;
            (0..edges.len())
                .filter(|&i| edges[i].visits == visits)
                .max_by(|&a, &b| score[a].total_cmp(&score[b]))
                .expect("un coup retenu")
        };
        // les `count` meilleurs, du meilleur au moins bon (à score égal, le dernier, comme `root_pick`)
        let root_picks = |root: Node, edges: &[Stat], visits: u32, count: usize, s: &mut Scratch, out: &mut Vec<usize>| {
            root_score(root, edges, s);
            let score = &s.values;
            out.clear();
            out.extend((0..edges.len()).filter(|&i| edges[i].visits == visits));
            out.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(b.cmp(&a)));
            assert!(out.len() >= count, "des coups retenus");
            out.truncate(count);
        };
        let (mut picks, mut leaves, mut asked, mut futures) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // Un tour du halving séquentiel : les simulations d'une suite du calendrier à visites égales
        // suivent chacune un coup différent de la racine, donc des sous-arbres disjoints. Elles
        // descendent toutes, leurs positions nouvelles sont évaluées ensemble (un seul envoi au
        // GPU par partie et par tour), puis elles remontent leurs valeurs, dans l'ordre. Seuls les
        // choix de la racine diffèrent d'une recherche une simulation à la fois : ils sont faits au
        // début du tour, sans les valeurs des simulations du même tour.
        let mut at = 0;
        while at < schedule.len() {
            let considered = schedule[at];
            let count = schedule[at..].iter().take_while(|&&c| c == considered).count();
            at += count;
            let root_node = tree.nodes[0];
            let stats = &tree.stats[root_node.first as usize..(root_node.first + root_node.len) as usize];
            root_picks(root_node, stats, considered, count, &mut tree.scratch, &mut picks);
            tree.path.clear();
            leaves.clear();
            asked.clear();
            for &pick in &picks {
                let mut w = determinize(root, k, rng);
                let mut kw = *k;
                let start = tree.path.len();
                let mut node = 0u32;
                let end = loop {
                    let n = tree.nodes[node as usize];
                    let stats = &tree.stats[n.first as usize..(n.first + n.len) as usize];
                    let i = if node == 0 { pick } else { n.select(stats, &mut tree.scratch) };
                    let e = n.first as usize + i;
                    let mv = tree.edges[e].mv;
                    tree.path.push((node, e as u32));
                    // le coup, puis les coups forcés qui suivent, et le résultat de leur hasard
                    kw.play(&mut w, &mv).expect("coup possible");
                    let mut hash = fold(0, moves::chance(&w, &mv));
                    // les coups forcés qui suivent (sans calculer tous les coups d'une position qui
                    // n'en a pas : `moves::forced`)
                    let known = loop {
                        if w.state.phase == Phase::GameOver {
                            break None;
                        }
                        match moves::forced(&w, &kw) {
                            moves::Forced::One(only) => {
                                kw.play(&mut w, &only).expect("coup forcé");
                                hash = fold(hash, moves::chance(&w, &only));
                            }
                            moves::Forced::Several(legal) => break Some(legal),
                        }
                    };
                    let Some(known) = known else { break End::Over(self.terminal(&w)) };
                    if let Some(child) = tree.child(e, hash) {
                        node = child;
                        continue;
                    }
                    // une position nouvelle : les cibles des conquêtes une fois, pour ses coups et
                    // son encodage
                    let targets = (w.state.phase == Phase::Conquest).then(|| w.targets());
                    let legal = known.unwrap_or_else(|| moves::legal_with(&w, &kw, targets.as_deref()));
                    // une position nouvelle : son évaluation, gardée ou demandée (une fois par tour)
                    let encoded = encode_with(&w, &kw, targets.as_deref());
                    let fp = fingerprint(&encoded, &legal);
                    if !evals.touch(fp) && !asked.iter().any(|&(f, _)| f == fp) {
                        asked.push((fp, leaves.len()));
                        futures.push(Box::pin(oracle.eval(&self.net, &encoded, w.map)));
                    }
                    break End::New { seat: Seat::of(&w), legal, edge: e, hash, fp };
                };
                leaves.push(Leaf { path: (start, tree.path.len()), end });
            }
            for (eval, &(fp, leaf)) in all(std::mem::take(&mut futures)).await.into_iter().zip(&asked) {
                let End::New { legal, .. } = &leaves[leaf].end else { unreachable!("une position à évaluer") };
                evals.keep(fp, eval, legal);
            }
            for leaf in &leaves {
                let value = match &leaf.end {
                    End::Over(value) => *value,
                    End::New { seat, legal, edge, hash, fp } => {
                        let id = tree.nodes.len() as u32;
                        let value = self.expand(&mut tree, seat, legal, evals.judged(*fp));
                        tree.add_child(*edge, *hash, id);
                        value
                    }
                };
                for &(nd, e) in &tree.path[leaf.path.0..leaf.path.1] {
                    if nd == 0 {
                        root_visits += 1;
                        root_total.iter_mut().zip(value).for_each(|(t, v)| *t += v);
                    }
                    let actor = tree.nodes[nd as usize].actor as usize;
                    let stat = &mut tree.stats[e as usize];
                    stat.visits += 1;
                    stat.value += value[actor];
                }
            }
        }
        let root_node = tree.nodes[0];
        let (stats, edges) = (tree.stats(root_node), tree.edges(root_node));
        let most = stats.iter().map(|e| e.visits).max().unwrap_or(0);
        let chosen = root_pick(root_node, stats, most, &mut Scratch::default());
        let me = root_node.actor as usize;
        let policy = edges.iter().zip(root_node.improved(stats)).map(|(e, p)| (e.key, p)).collect();
        let visits = edges.iter().zip(stats).map(|(e, s)| (e.key, s.visits)).collect();
        let details = edges.iter().zip(stats).map(|(e, s)| (e.key, s.prior, q(s).unwrap_or(f32::NAN))).collect();
        let average =
            |at: usize| -> [f32; MAX_PLAYERS] { std::array::from_fn(|q| root_total[at + q] / root_visits as f32) };
        let (finals, wins) = (average(COINS), average(WINS));
        let value = root_total[me] / root_visits as f32;
        let mv = edges[chosen].mv;
        evals.tree = tree;
        Searched { mv, policy, visits, finals, wins, details, value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L'arbre tient dans le cache : nœuds de 16 octets, coups en 16 + 24 octets (`Tree`).
    #[test]
    fn the_tree_stays_small() {
        assert_eq!(std::mem::size_of::<Node>(), 16);
        assert_eq!(std::mem::size_of::<Stat>(), 16);
        assert!(std::mem::size_of::<Edge>() <= 24, "{} octets", std::mem::size_of::<Edge>());
    }

    /// Le calendrier de mctx : 4 coups, 10 itérations → chacun une fois, puis les 2 meilleurs.
    #[test]
    fn sequential_halving_follows_mctx() {
        assert_eq!(considered_visits(4, 10), vec![0, 0, 0, 0, 1, 1, 2, 2, 3, 3]);
        assert_eq!(considered_visits(1, 3), vec![0, 1, 2]);
        // 16 coups, 200 itérations : 3 visites chacun, puis 8 coups à 6, 4 à 12, 2 au reste
        let s = considered_visits(16, 200);
        assert_eq!(s.len(), 200);
        assert_eq!(&s[..16], &[0; 16]);
        assert_eq!(s.iter().filter(|&&v| v == 3).count(), 8);
    }
}
