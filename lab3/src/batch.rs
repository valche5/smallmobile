//! Les évaluations groupées d'un fil de calcul. Chaque partie que mène le fil dépose sa position et
//! attend (`Oracle`) ; le fil envoie le groupe (`submit`) : son processeur l'évalue aussitôt,
//! couche après couche (`Net::eval_batch`), ou le GPU (`gpu::Gpu`) pendant que le fil fait avancer
//! un autre groupe ; puis chaque partie reprend avec la sienne (`collect`). Le groupe, les parties
//! et leurs évaluations gardées appartiennent au fil : aucune donnée modifiable n'est partagée
//! entre les cœurs, seuls les poids du réseau, en lecture.

use std::cell::RefCell;
use std::pin::Pin;
use std::task::{Context, Poll};

use smallmobile_kernel::Map;

use crate::az::Oracle;
use crate::encode::{Encoded, REGIONS};
use crate::gpu::{Gpu, Pending};
use crate::net::{Eval, Net, Position};

pub struct Batch<'a> {
    net: &'a Net,
    /// Le GPU qui évalue le groupe ; sans lui, le processeur du fil.
    gpu: Option<&'a Gpu>,
    /// Les positions déposées depuis le dernier envoi, dans l'ordre des tickets.
    waiting: RefCell<Vec<Request>>,
    /// Le groupe envoyé au GPU, pas encore repris.
    sent: RefCell<Option<Pending>>,
    /// Les évaluations du dernier envoi, par ticket, chacune prise par sa partie.
    done: RefCell<Vec<Option<Eval>>>,
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
    pub fn new(net: &'a Net, gpu: Option<&'a Gpu>) -> Self {
        Batch { net, gpu, waiting: RefCell::default(), sent: RefCell::default(), done: RefCell::default() }
    }

    /// Envoie les positions en attente : le processeur du fil les évalue aussitôt ; le GPU, pendant
    /// que le fil fait autre chose (`collect`).
    pub fn submit(&self) {
        let waiting = std::mem::take(&mut *self.waiting.borrow_mut());
        match self.gpu {
            Some(gpu) => {
                let players = waiting.iter().map(|r| r.e.players);
                *self.sent.borrow_mut() = Some(gpu.submit(&waiting, players));
            }
            None => *self.done.borrow_mut() = self.net.eval_batch(&waiting).into_iter().map(Some).collect(),
        }
    }

    /// Les évaluations du dernier envoi au GPU (attend qu'elles arrivent) ; chaque partie trouvera la
    /// sienne en reprenant. Rien à faire sans GPU.
    pub fn collect(&self) {
        if let Some(pending) = self.sent.borrow_mut().take() {
            *self.done.borrow_mut() = pending.wait().into_iter().map(Some).collect();
        }
    }
}

impl Oracle for Batch<'_> {
    fn eval(&self, net: &Net, e: &Encoded, map: &Map) -> impl Future<Output = Eval> {
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
