//! Les évaluations d'un fil de calcul. Chaque partie que mène le fil dépose ses positions et attend
//! (`Oracle`). Sur le processeur, le fil évalue d'un coup celles du groupe de parties qui attendent
//! (`submit`), couche après couche (`Net::eval_batch`), puis chaque partie reprend avec les siennes.
//! Avec le GPU, le fil envoie les positions d'une partie dès qu'elle les a déposées (`send`), fait
//! avancer les autres, et la reprend quand ses évaluations sont là (`receive`, `resume`). Les
//! parties et leurs évaluations gardées appartiennent au fil : aucune donnée modifiable n'est
//! partagée entre les cœurs, seuls les poids du réseau, en lecture.

use std::cell::RefCell;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use smallmobile_kernel::Map;

use crate::az::Oracle;
use crate::encode::{Encoded, REGIONS};
use crate::gpu::{Gpu, Lane, Sent};
use crate::net::{Eval, Net, Position};

/// Le temps (ns) des fils de jeu, tous ensemble, pour leurs évaluations sur GPU : écriture des
/// positions dans l'anneau, attente des évaluations, lecture des sorties (`TIME_*` ; affiché par
/// `selfplay::run` avec `LAB6_VULKAN_PROFILE`).
pub static TIMES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
pub const TIME_SUBMIT: usize = 0;
pub const TIME_WAIT: usize = 1;
pub const TIME_DECODE: usize = 2;

/// Les calculs soumis au GPU (`SENT_LOTS`) et leurs positions (`SENT_POSITIONS`), depuis le départ
/// (`gpu::drive` ; affichés par `selfplay::pace`).
pub static SENT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
pub const SENT_LOTS: usize = 0;
pub const SENT_POSITIONS: usize = 1;

fn count(what: usize, since: std::time::Instant) {
    TIMES[what].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

pub struct Batch<'a> {
    net: &'a Net,
    /// Le GPU qui évalue les positions, et les places du fil dans son anneau ; sans lui, le
    /// processeur du fil.
    gpu: Option<(&'a Gpu, RefCell<Lane>, u8)>,
    /// Les positions déposées depuis le dernier envoi, dans l'ordre des tickets.
    waiting: RefCell<Vec<Request>>,
    /// Les évaluations du dernier envoi (ou de la partie reprise), par ticket, chacune prise par sa
    /// partie.
    done: RefCell<Vec<Option<Eval>>>,
    /// Profils (`LAB6_INSTANT=1`) : des évaluations instantanées, sans réseau (`instant`), pour
    /// mesurer le jeu et la recherche seuls.
    instant: bool,
}

/// Une position en attente : son encodage et les voisines de ses régions (copiées : la carte reste
/// à sa partie).
struct Request {
    e: Encoded,
    adj: Vec<u8>,
    starts: [u16; REGIONS + 1],
}

impl Position for Request {
    fn encoded(&self) -> &Encoded {
        &self.e
    }

    fn adj(&self, region: usize) -> &[u8] {
        &self.adj[self.starts[region] as usize..self.starts[region + 1] as usize]
    }
}

impl<'a> Batch<'a> {
    /// `gpu` : le GPU, les places du fil dans son anneau, et le rang de ce réseau sur ce GPU
    /// (`Gpu::start_nets`).
    pub fn new(net: &'a Net, gpu: Option<(&'a Gpu, Lane, u8)>) -> Self {
        let instant = std::env::var_os("LAB6_INSTANT").is_some();
        let gpu = gpu.map(|(gpu, lane, rank)| (gpu, RefCell::new(lane), rank));
        Batch { net, gpu, waiting: RefCell::default(), done: RefCell::default(), instant }
    }

    /// Évalue les positions en attente sur le processeur du fil (ou instantanément, `instant`) ;
    /// chaque partie trouvera les siennes en reprenant.
    pub fn submit(&self) {
        let waiting = std::mem::take(&mut *self.waiting.borrow_mut());
        *self.done.borrow_mut() = if self.instant {
            waiting.iter().map(|r| Some(instant(&r.e))).collect()
        } else {
            assert!(self.gpu.is_none(), "avec le GPU : `send`");
            self.net.eval_batch(&waiting).into_iter().map(Some).collect()
        };
    }

    /// Écrit au GPU les positions en attente (celles de la partie qui vient de s'arrêter) ; `None`
    /// s'il n'y en a pas.
    pub fn send(&self) -> Option<Sent> {
        let waiting = std::mem::take(&mut *self.waiting.borrow_mut());
        if waiting.is_empty() {
            return None;
        }
        let (gpu, lane, rank) = self.gpu.as_ref().expect("un GPU");
        let start = std::time::Instant::now();
        let sent = lane.borrow_mut().send(gpu, &waiting, *rank);
        count(TIME_SUBMIT, start);
        Some(sent)
    }

    /// Les évaluations de `sent`, attendues et lues.
    pub fn receive(&self, sent: Sent) -> Vec<Eval> {
        let (gpu, lane, _) = self.gpu.as_ref().expect("un GPU");
        let start = std::time::Instant::now();
        gpu.wait(&sent);
        count(TIME_WAIT, start);
        let start = std::time::Instant::now();
        let evals = lane.borrow_mut().take(gpu, sent);
        count(TIME_DECODE, start);
        evals
    }

    /// Ce groupe évalue-t-il ce réseau ? (l'arène : un groupe par réseau, `arena::run`)
    pub fn serves(&self, net: &Net) -> bool {
        std::ptr::eq(net, self.net)
    }

    /// Avant de reprendre la partie qui attendait `evals` (`receive`) : elle les trouvera.
    pub fn resume(&self, evals: Vec<Eval>) {
        *self.done.borrow_mut() = evals.into_iter().map(Some).collect();
    }
}

impl<'a> Oracle for Batch<'a> {
    fn eval<'s>(&'s self, net: &Net, e: &Encoded, map: &Map) -> impl Future<Output = Eval> + use<'s, 'a> {
        assert!(std::ptr::eq(net, self.net), "un groupe, un réseau");
        let (mut adj, mut starts) = (Vec::new(), [0u16; REGIONS + 1]);
        for i in 0..e.regions {
            adj.extend_from_slice(&map.regions[i].adj);
            starts[i + 1] = adj.len() as u16;
        }
        Waiting { batch: self, request: Some(Request { e: e.clone(), adj, starts }), ticket: 0 }
    }
}

/// Une évaluation attendue : la position est déposée au premier passage, l'évaluation prise après
/// `Batch::submit` (et `collect`).
struct Waiting<'b, 'a> {
    batch: &'b Batch<'a>,
    request: Option<Request>,
    ticket: usize,
}

impl Future for Waiting<'_, '_> {
    type Output = Eval;

    fn poll(mut self: Pin<&mut Self>, _: &mut Context) -> Poll<Eval> {
        if let Some(request) = self.request.take() {
            let mut waiting = self.batch.waiting.borrow_mut();
            self.ticket = waiting.len();
            waiting.push(request);
            return Poll::Pending;
        }
        match self.batch.done.borrow_mut().get_mut(self.ticket).and_then(Option::take) {
            Some(eval) => Poll::Ready(eval),
            None => Poll::Pending,
        }
    }
}

/// Une évaluation sans réseau (`LAB6_INSTANT`) : logits et valeurs tirés d'une empreinte de la partie,
/// assez variés pour que la recherche se comporte à peu près comme avec un réseau, et bon marché
/// (environ 1 µs).
fn instant(e: &Encoded) -> Eval {
    let mut h = e.regions as u64 ^ 0x9e37_79b9_7f4a_7c15;
    for c in e.global.as_chunks::<8>().0 {
        h = (h ^ u64::from_le_bytes(*c)).wrapping_mul(0x100_0000_01b3);
    }
    let mut next = || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        (h >> 40) as f32 / (1u64 << 24) as f32
    };
    let logits = (0..crate::moves::KEYS).map(|_| 2.0 * next()).collect();
    let v: [f32; 10] = std::array::from_fn(|_| next());
    Eval::from_heads(e.players, logits, &v)
}
