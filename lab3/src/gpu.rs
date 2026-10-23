//! Le réseau sur GPU par Vulkan, pour la génération des parties (`lab3 selfplay --device vulkan`).
//! Un flot continu, sans copie ni fil entre le jeu et le GPU :
//!
//! - chaque fil de jeu a ses places dans l'anneau des entrées (`Lane`) ; il y écrit les positions
//!   d'une partie, puis leurs places à la suite dans la file d'ordre (`Lane::send`) ;
//! - le fil « gpu » (`drive`) soumet en un calcul les numéros de la file écrits à la suite
//!   (`MAX_BATCH` au plus), tant qu'il y a moins de `IN_FLIGHT` calculs en vol, et note la fin de
//!   chacun : pendant qu'un calcul tourne, le suivant grossit ;
//! - le fil de jeu lit les sorties sur place, dans l'anneau des sorties (`Lane::take`), et reprend
//!   ses places.
//!
//! Un fil n'attend jamais les places d'un autre : les siennes reviennent quand il a lu, et un
//! numéro de la file d'ordre dès que le GPU l'a calculé.
//!
//! Les nombres du GPU (bf16, sommes dans un autre ordre) diffèrent un peu de ceux du processeur :
//! les parties ne sont plus les mêmes au bit près que sans GPU.

use std::path::Path;

use crate::net::{Eval, Position};

/// Des positions envoyées (`Lane::send`) : leurs numéros dans la file d'ordre
/// (`first..first + cells.len()`), leurs places, le nombre de joueurs de chacune (pour lire ses
/// sorties).
pub struct Sent {
    first: u64,
    cells: Vec<u32>,
    players: Vec<u8>,
}

/// Les places d'un fil de jeu dans l'anneau, libres.
pub struct Lane {
    free: Vec<u32>,
}

/// Calculs en vol au plus, par défaut : celui qui tourne et le suivant, qui grossit en attendant
/// (`LAB3_IN_FLIGHT` pour essayer).
#[cfg(feature = "vulkan")]
const IN_FLIGHT: usize = 2;

#[cfg(feature = "vulkan")]
struct Ring {
    engine: crate::vulkan::Engine,
    /// Le prochain numéro à prendre dans la file d'ordre.
    head: std::sync::atomic::AtomicU64,
    /// Chaque numéro `t` de la file : `t + 1` quand sa place y est écrite.
    seq: Box<[std::sync::atomic::AtomicU64]>,
    /// Les positions de numéro plus petit sont évaluées.
    done: std::sync::atomic::AtomicU64,
    /// Réveille les fils de jeu qui attendent une évaluation (`done`).
    evaluated: (std::sync::Mutex<()>, std::sync::Condvar),
    /// Réveille le fil « gpu » qui attend des positions (`sleeping`).
    written: (std::sync::Mutex<()>, std::sync::Condvar),
    sleeping: std::sync::atomic::AtomicBool,
    stop: std::sync::atomic::AtomicBool,
}

pub struct Gpu {
    #[cfg(feature = "vulkan")]
    ring: std::sync::Arc<Ring>,
    #[cfg(not(feature = "vulkan"))]
    never: std::convert::Infallible,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(feature = "vulkan"))]
impl Gpu {
    pub fn start(net: &Path) -> Result<(Gpu, String), String> {
        let _ = net;
        Err("programme lab3 compilé sans Vulkan (cargo build --features vulkan)".into())
    }

    pub fn lanes(&self, _: usize) -> Vec<Lane> {
        match self.never {}
    }

    pub fn done(&self, _: &Sent) -> bool {
        match self.never {}
    }

    pub fn wait(&self, _: &Sent) {
        match self.never {}
    }
}

#[cfg(not(feature = "vulkan"))]
impl Lane {
    pub fn free(&self) -> usize {
        self.free.len()
    }

    pub fn send<P: Position>(&mut self, gpu: &Gpu, _: &[P]) -> Sent {
        match gpu.never {}
    }

    pub fn take(&mut self, gpu: &Gpu, _: Sent) -> Vec<Eval> {
        match gpu.never {}
    }
}

#[cfg(feature = "vulkan")]
impl Gpu {
    /// Ouvre le GPU avec les poids `net` et lance le fil « gpu » ; rend aussi l'appareil qui calcule.
    pub fn start(net: &Path) -> Result<(Gpu, String), String> {
        use crate::vulkan::{COMMANDS, Engine, MAX_BATCH, ORDER};
        use std::sync::atomic::{AtomicBool, AtomicU64};
        let engine = Engine::new(net, MAX_BATCH)?;
        let device = engine.describe();
        let ring = std::sync::Arc::new(Ring {
            engine,
            head: AtomicU64::new(0),
            seq: (0..ORDER).map(|_| AtomicU64::new(0)).collect(),
            done: AtomicU64::new(0),
            evaluated: Default::default(),
            written: Default::default(),
            sleeping: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let in_flight =
            std::env::var("LAB3_IN_FLIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(IN_FLIGHT).clamp(1, COMMANDS);
        let r = ring.clone();
        // nommé, pour les profils
        let thread = std::thread::Builder::new()
            .name("gpu".into())
            .spawn(move || drive(&r, in_flight))
            .map_err(|e| format!("fil du GPU : {e}"))?;
        Ok((Gpu { ring, thread: Some(thread) }, device))
    }

    /// L'anneau partagé entre `n` fils de jeu, chacun ses places (`RING / n`).
    pub fn lanes(&self, n: usize) -> Vec<Lane> {
        let per = (crate::vulkan::RING / n.max(1)) as u32;
        (0..n as u32).map(|l| Lane { free: (l * per..(l + 1) * per).rev().collect() }).collect()
    }

    /// Les évaluations de `sent` sont-elles là ?
    pub fn done(&self, sent: &Sent) -> bool {
        self.ring.done.load(std::sync::atomic::Ordering::Acquire) >= sent.first + sent.cells.len() as u64
    }

    /// Attend les évaluations de `sent`.
    pub fn wait(&self, sent: &Sent) {
        let r = &*self.ring;
        while !self.done(sent) {
            let held = r.evaluated.0.lock().expect("verrou");
            if !self.done(sent) {
                // le fil « gpu » réveille à chaque calcul fini ; le délai, par prudence
                let _ = r.evaluated.1.wait_timeout(held, std::time::Duration::from_millis(1));
            }
        }
    }
}

#[cfg(feature = "vulkan")]
impl Lane {
    /// Les places libres de ce fil.
    pub fn free(&self) -> usize {
        self.free.len()
    }

    /// Écrit `positions` à des places libres de ce fil, puis leurs places à la suite dans la file
    /// d'ordre ; le fil « gpu » les soumet dès qu'elles y sont.
    pub fn send<P: Position>(&mut self, gpu: &Gpu, positions: &[P]) -> Sent {
        use std::sync::atomic::Ordering;
        let (r, order) = (&*gpu.ring, crate::vulkan::ORDER as u64);
        assert!(positions.len() <= self.free.len(), "plus de place pour ce fil dans l'anneau du GPU");
        let cells: Vec<u32> = (0..positions.len()).map(|_| self.free.pop().expect("une place")).collect();
        for (&cell, p) in cells.iter().zip(positions) {
            r.engine.write(cell as usize, p);
        }
        let first = r.head.fetch_add(cells.len() as u64, Ordering::Relaxed);
        for (j, &cell) in cells.iter().enumerate() {
            let t = first + j as u64;
            // un numéro revient quand le GPU a calculé celui d'un tour de file avant : on
            // n'attend presque jamais
            while r.done.load(Ordering::Acquire) + order <= t {
                std::thread::yield_now();
            }
            r.engine.order((t % order) as usize, cell as usize);
        }
        // les écritures dans la mémoire du GPU (write-combining) avant leurs marques : sans cette
        // barrière, le fil « gpu » pourrait soumettre une position encore à moitié écrite
        std::sync::atomic::fence(Ordering::SeqCst);
        for j in 0..cells.len() as u64 {
            let t = first + j;
            r.seq[(t % order) as usize].store(t + 1, Ordering::Release);
        }
        if r.sleeping.load(Ordering::Acquire) {
            let _held = r.written.0.lock().expect("verrou");
            r.written.1.notify_one();
        }
        Sent { first, cells, players: positions.iter().map(|p| p.encoded().players as u8).collect() }
    }

    /// Les évaluations de `sent`, lues sur place (attend qu'elles soient là) ; ses places
    /// reviennent à ce fil.
    pub fn take(&mut self, gpu: &Gpu, sent: Sent) -> Vec<Eval> {
        gpu.wait(&sent);
        let r = &*gpu.ring;
        let read =
            |(&cell, &players): (&u32, &u8)| crate::vulkan::eval_of(players as usize, r.engine.output(cell as usize));
        let evals = sent.cells.iter().zip(&sent.players).map(read).collect();
        self.free.extend_from_slice(&sent.cells);
        evals
    }
}

/// Le fil « gpu » : soumet les positions rangées à la suite dans la file d'ordre, note la fin des
/// calculs, jusqu'à l'arrêt (`Gpu::drop`) une fois tout évalué.
#[cfg(feature = "vulkan")]
fn drive(r: &Ring, in_flight: usize) {
    use crate::batch::{SENT, SENT_LOTS, SENT_POSITIONS};
    use crate::vulkan::{MAX_BATCH, ORDER};
    use std::sync::atomic::Ordering;
    let order = ORDER as u64;
    let fail = |e: String| -> ! { panic!("GPU : {e}") };
    // le prochain numéro à soumettre ; les calculs en vol (numéro du calcul, fin, positions)
    let mut next = 0u64;
    let mut flight = std::collections::VecDeque::<(u64, u64, usize)>::new();
    loop {
        // les calculs finis, dans l'ordre
        let finished = r.engine.completed().unwrap_or_else(|e| fail(e));
        let mut woke = false;
        while let Some(&(value, end, count)) = flight.front()
            && value <= finished
        {
            r.engine.profile_lot(value, count);
            r.done.store(end, Ordering::Release);
            flight.pop_front();
            woke = true;
        }
        if woke {
            let _held = r.evaluated.0.lock().expect("verrou");
            r.evaluated.1.notify_all();
        }
        // les numéros écrits à la suite, en un calcul
        if flight.len() < in_flight {
            let mut end = next;
            while end - next < MAX_BATCH as u64 && r.seq[(end % order) as usize].load(Ordering::Acquire) == end + 1 {
                end += 1;
            }
            if end > next {
                let count = (end - next) as usize;
                let value = r.engine.submit((next % order) as usize, count).unwrap_or_else(|e| fail(e));
                SENT[SENT_LOTS].fetch_add(1, Ordering::Relaxed);
                SENT[SENT_POSITIONS].fetch_add(count as u64, Ordering::Relaxed);
                flight.push_back((value, end, count));
                next = end;
                continue;
            }
        }
        match flight.front() {
            // la fin du plus vieux calcul, ou de nouvelles positions dans 100 µs
            Some(&(value, ..)) => {
                r.engine.wait_for(value, 100_000).unwrap_or_else(|e| fail(e));
            }
            None => {
                if r.stop.load(Ordering::Acquire) && r.head.load(Ordering::Acquire) == next {
                    break;
                }
                // rien en vol : attend des positions (`Lane::send` réveille)
                let held = r.written.0.lock().expect("verrou");
                r.sleeping.store(true, Ordering::Release);
                if r.seq[(next % order) as usize].load(Ordering::Acquire) != next + 1 {
                    let _ = r.written.1.wait_timeout(held, std::time::Duration::from_millis(1));
                }
                r.sleeping.store(false, Ordering::Release);
            }
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        self.ring.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
