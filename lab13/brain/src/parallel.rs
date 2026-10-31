//! Les évaluations d'une recherche menée seule sur le processeur, à plusieurs fils (`Parallel`) :
//! les positions que la recherche demande en même temps (les feuilles d'un tour du halving, puis
//! chaque pas de leurs fins simulées) attendent (`Oracle`), puis `submit` les évalue ensemble,
//! réparties entre les fils, et la recherche reprend (`drive`). Chaque position reçoit les mêmes
//! opérations que seule (`Net::eval_batch`) : mêmes décisions au bit près qu'avec `Direct`. Pour le
//! programme `lab13-expert` sur une machine à plusieurs cœurs (mini en a 6 ; lab13/lab.md, D16).

use std::cell::RefCell;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;

use smallmobile_kernel::Map;

use crate::encode::{Encoded, REGIONS};
use crate::net::{Eval, Net, Position};
use crate::search::Oracle;

/// Une position en attente : son encodage et les voisines de ses régions (copiées : la carte reste
/// à sa partie, les fils n'y touchent pas).
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

/// Le travail d'un fil : les positions d'un envoi, et sa tranche.
type Job = (Arc<Vec<Request>>, usize, usize);

pub struct Parallel {
    net: Arc<Net>,
    /// Les positions déposées depuis le dernier envoi, dans l'ordre des tickets.
    waiting: RefCell<Vec<Request>>,
    /// Les évaluations du dernier envoi, par ticket, chacune prise par sa recherche.
    done: RefCell<Vec<Option<Eval>>>,
    /// Les autres fils (le fil qui appelle `submit` prend la première tranche).
    workers: Vec<Sender<Job>>,
    handles: Vec<JoinHandle<()>>,
    results: Receiver<(usize, Vec<Eval>)>,
}

impl Parallel {
    /// `threads` fils en tout, celui qui appelle compris (1 : tout sur place, sans autre fil).
    pub fn new(net: Arc<Net>, threads: usize) -> Parallel {
        let (results_tx, results) = channel();
        let (mut workers, mut handles) = (Vec::new(), Vec::new());
        for _ in 1..threads.max(1) {
            let (tx, rx) = channel::<Job>();
            let (net, results_tx) = (net.clone(), results_tx.clone());
            handles.push(std::thread::spawn(move || {
                for (requests, start, end) in rx {
                    let evals = net.eval_batch(&requests[start..end]);
                    if results_tx.send((start, evals)).is_err() {
                        break;
                    }
                }
            }));
            workers.push(tx);
        }
        Parallel { net, waiting: RefCell::default(), done: RefCell::default(), workers, handles, results }
    }

    /// Les fils, celui qui appelle compris.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Évalue les positions en attente, réparties entre les fils en tranches égales (une seule
    /// position : ici, sans passer par un autre fil) ; chaque recherche trouvera les siennes en
    /// reprenant.
    pub fn submit(&self) {
        let waiting = std::mem::take(&mut *self.waiting.borrow_mut());
        let n = waiting.len();
        let mut done: Vec<Option<Eval>> = (0..n).map(|_| None).collect();
        if n > 0 {
            let chunk = n.div_ceil(self.threads().min(n));
            let requests = Arc::new(waiting);
            let mut sent = 0;
            for (worker, start) in self.workers.iter().zip((chunk..n).step_by(chunk)) {
                worker.send((requests.clone(), start, (start + chunk).min(n))).expect("fil d'évaluation");
                sent += 1;
            }
            for (i, eval) in self.net.eval_batch(&requests[..chunk]).into_iter().enumerate() {
                done[i] = Some(eval);
            }
            for _ in 0..sent {
                let (start, evals) = self.results.recv().expect("fil d'évaluation");
                for (i, eval) in evals.into_iter().enumerate() {
                    done[start + i] = Some(eval);
                }
            }
        }
        *self.done.borrow_mut() = done;
    }

    /// Mène un calcul à son terme : à chaque attente, évalue les positions déposées (`submit`).
    pub fn drive<F: Future>(&self, f: F) -> F::Output {
        let mut f = std::pin::pin!(f);
        let mut cx = Context::from_waker(Waker::noop());
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(out) => break out,
                Poll::Pending => self.submit(),
            }
        }
    }
}

impl Drop for Parallel {
    fn drop(&mut self) {
        // plus d'envois : chaque fil sort de sa boucle
        self.workers.clear();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Oracle for Parallel {
    fn eval<'s>(&'s self, net: &Net, e: &Encoded, map: &Map) -> impl Future<Output = Eval> + use<'s> {
        assert!(std::ptr::eq(net, &*self.net), "un oracle, un réseau");
        let (mut adj, mut starts) = (Vec::new(), [0u16; REGIONS + 1]);
        for i in 0..e.regions {
            adj.extend_from_slice(&map.regions[i].adj);
            starts[i + 1] = adj.len() as u16;
        }
        Waiting { oracle: self, request: Some(Request { e: e.clone(), adj, starts }), ticket: 0 }
    }
}

/// Une évaluation attendue : la position est déposée au premier passage, l'évaluation prise après
/// `Parallel::submit`.
struct Waiting<'a> {
    oracle: &'a Parallel,
    request: Option<Request>,
    ticket: usize,
}

impl Future for Waiting<'_> {
    type Output = Eval;

    fn poll(mut self: Pin<&mut Self>, _: &mut Context) -> Poll<Eval> {
        if let Some(request) = self.request.take() {
            let mut waiting = self.oracle.waiting.borrow_mut();
            self.ticket = waiting.len();
            waiting.push(request);
            return Poll::Pending;
        }
        match self.oracle.done.borrow_mut().get_mut(self.ticket).and_then(Option::take) {
            Some(eval) => Poll::Ready(eval),
            None => Poll::Pending,
        }
    }
}
