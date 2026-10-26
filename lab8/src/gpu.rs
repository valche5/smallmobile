//! Le réseau sur GPU par Vulkan, pour la génération des parties (`lab8 selfplay --device vulkan`).
//! Un flot continu, sans copie ni fil entre le jeu et le GPU :
//!
//! - chaque fil de jeu a ses places dans l'anneau des entrées (`Lane`) ; il y écrit les positions
//!   d'une partie, puis leurs places à la suite dans la file d'ordre (`Lane::send`) ;
//! - le fil « gpu » (`drive`) lance en un calcul les numéros de la file écrits à la suite
//!   (`MAX_BATCH` au plus), tant qu'il y a moins de `IN_FLIGHT` calculs en vol, et note la fin de
//!   chacun : pendant qu'un calcul tourne, le suivant grossit. Les calculs sont des étapes de
//!   commandes soumises d'avance, qui attendent chacune leur feu vert (`vulkan::STEPS`) ;
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
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
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
/// (`LAB8_IN_FLIGHT` pour essayer).
#[cfg(feature = "vulkan")]
const IN_FLIGHT: usize = 2;

/// Le moteur qui calcule le réseau : Vulkan (tout GPU), ou CUDA (NVIDIA, `--device cuda`). Même
/// anneau, même file d'ordre ; CUDA n'a pas de commandes soumises d'avance (tours et recyclage vides).
#[cfg(feature = "vulkan")]
enum Engine {
    Vulkan(Box<crate::vulkan::Engine>),
    #[cfg(feature = "cuda")]
    Cuda(Box<crate::cuda::Engine>),
}

#[cfg(feature = "vulkan")]
macro_rules! each {
    ($e:expr, $x:ident => $body:expr) => {
        match $e {
            Engine::Vulkan($x) => $body,
            #[cfg(feature = "cuda")]
            Engine::Cuda($x) => $body,
        }
    };
}

#[cfg(feature = "vulkan")]
impl Engine {
    fn describe(&self) -> String {
        each!(self, e => e.describe())
    }

    fn write<P: Position + ?Sized>(&self, cell: usize, p: &P, net: u8) {
        each!(self, e => e.write(cell, p, net))
    }

    fn order(&self, at: usize, cell: usize) {
        each!(self, e => e.order(at, cell))
    }

    fn output(&self, cell: usize) -> &[f32] {
        each!(self, e => e.output(cell))
    }

    fn launch(&self, slot: usize, first: usize, count: usize) -> Result<(), String> {
        each!(self, e => e.launch(slot, first, count))
    }

    fn finished(&self, slot: usize) -> Result<bool, String> {
        each!(self, e => e.finished(slot))
    }

    fn submit_round(&self, round: usize) -> Result<u64, String> {
        match self {
            Engine::Vulkan(e) => e.submit_round(round),
            #[cfg(feature = "cuda")]
            Engine::Cuda(_) => Ok(0),
        }
    }

    fn wait(&self, value: u64) -> Result<(), String> {
        match self {
            Engine::Vulkan(e) => e.wait(value),
            #[cfg(feature = "cuda")]
            Engine::Cuda(_) => Ok(()),
        }
    }

    fn recycle(&self, slot: usize) -> Result<(), String> {
        match self {
            Engine::Vulkan(e) => e.recycle(slot),
            #[cfg(feature = "cuda")]
            Engine::Cuda(_) => Ok(()),
        }
    }

    fn profile_step(&self, slot: usize, count: usize) -> Option<(u64, u64)> {
        match self {
            Engine::Vulkan(e) => e.profile_step(slot, count),
            #[cfg(feature = "cuda")]
            Engine::Cuda(_) => None,
        }
    }
}

/// Le moteur demandé (`--device`) : `vulkan` (ou `gpu`), `cuda`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    Vulkan,
    Cuda,
}

impl Device {
    /// `cpu` → `None` ; un nom inconnu → `Err`.
    pub fn parse(name: &str) -> Result<Option<Device>, String> {
        match name {
            "cpu" => Ok(None),
            "gpu" | "vulkan" => Ok(Some(Device::Vulkan)),
            "cuda" => Ok(Some(Device::Cuda)),
            _ => Err(format!("appareil inconnu : {name} (cpu, vulkan, cuda)")),
        }
    }
}

#[cfg(feature = "vulkan")]
struct Ring {
    engine: Engine,
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
    /// Profils (`LAB8_LATENCY`) : pour chaque numéro de la file, les instants (`now`) où sa position
    /// a été écrite, soumise, vue finie.
    stamps: Option<Box<[[std::sync::atomic::AtomicU64; 3]]>>,
}

/// Le trajet des positions (`LAB8_LATENCY`), totaux depuis le départ, ns : écrite → soumise,
/// soumise → vue finie (par position) ; puis, par envoi, écrite → reprise, et la fin du trajet :
/// le fil de jeu attendait déjà (vue finie → réveillé) ou revenait plus tard (vue finie → repris).
/// Affiché par `selfplay::pace` (`latency`).
/// Puis, par calcul (avec aussi `LAB8_VULKAN_PROFILE`, horodatages du GPU) : l'appel de soumission,
/// soumis → début sur le GPU, la durée sur le GPU, fin sur le GPU → vue par le fil « gpu », le trou
/// sur le GPU depuis la fin du calcul d'avant.
pub static LATENCY: [std::sync::atomic::AtomicU64; 15] = [const { std::sync::atomic::AtomicU64::new(0) }; 15];
const L_QUEUED: usize = 0;
const L_COMPUTED: usize = 1;
const L_POSITIONS: usize = 2;
const L_TRIP: usize = 3;
const L_SENT: usize = 4;
const L_WOKEN: usize = 5;
const L_WAITED: usize = 6;
const L_LATE: usize = 7;
const L_IDLE: usize = 8;
const L_CALCS: usize = 9;
const L_SUBMIT: usize = 10;
const L_TO_GPU: usize = 11;
const L_RUN: usize = 12;
const L_NOTICE: usize = 13;
const L_GAP: usize = 14;

/// L'instant présent pour `LATENCY` : ns de `CLOCK_MONOTONIC`, l'horloge où `vulkan::Engine`
/// ramène celle du GPU.
#[cfg(feature = "vulkan")]
fn now() -> u64 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, at: *mut Timespec) -> i32;
    }
    let mut at = Timespec { sec: 0, nsec: 0 };
    // SAFETY : `CLOCK_MONOTONIC` (1), structure de la taille attendue
    unsafe { clock_gettime(1, &mut at) };
    at.sec as u64 * 1_000_000_000 + at.nsec as u64
}

/// Le trajet moyen depuis `last` (totaux de `LATENCY`), en µs, ou `None` sans mesure.
pub fn latency(last: &mut [u64; 15]) -> Option<String> {
    use std::sync::atomic::Ordering;
    let all: [u64; 15] = LATENCY.each_ref().map(|l| l.load(Ordering::Relaxed));
    let d: [u64; 15] = std::array::from_fn(|i| all[i] - last[i]);
    *last = all;
    let per = |sum: usize, n: usize| d[sum] as f64 / 1e3 / d[n].max(1) as f64;
    (d[L_POSITIONS] > 0).then(|| {
        format!(
            "trajet (µs) : écrite → soumise {:.0}, soumise → vue finie {:.0} ; écrite → reprise {:.0} ; \
             le fil attendait déjà ({:.0} % des envois) : vue finie → réveillé {:.0} ; sinon vue finie → reprise {:.0}",
            per(L_QUEUED, L_POSITIONS),
            per(L_COMPUTED, L_POSITIONS),
            per(L_TRIP, L_SENT),
            100.0 * d[L_WAITED] as f64 / d[L_SENT].max(1) as f64,
            per(L_WOKEN, L_WAITED),
            per(L_LATE, L_IDLE),
        ) + &if d[L_CALCS] > 0 {
            format!(
                "\nfil gpu (µs par calcul) : appel de soumission {:.0}, soumis → début GPU {:.0}, calcul {:.0}, \
                 fin GPU → vue {:.0}, trou sur le GPU avant le calcul {:.0}",
                per(L_SUBMIT, L_CALCS),
                per(L_TO_GPU, L_CALCS),
                per(L_RUN, L_CALCS),
                per(L_NOTICE, L_CALCS),
                per(L_GAP, L_CALCS),
            )
        } else {
            String::new()
        }
    })
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
        Gpu::start_nets(&[net])
    }

    pub fn start_nets(nets: &[&Path]) -> Result<(Gpu, String), String> {
        let _ = nets;
        Err("programme lab8 compilé sans Vulkan (cargo build --features vulkan)".into())
    }

    pub fn start_on(_: Device, nets: &[&Path]) -> Result<(Gpu, String), String> {
        Gpu::start_nets(nets)
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

    pub fn send<P: Position>(&mut self, gpu: &Gpu, _: &[P], _: u8) -> Sent {
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
        Gpu::start_nets(&[net])
    }

    /// `start` avec plusieurs réseaux de même forme (l'arène : un par joueur `az`) ; une position
    /// dit le sien à l'envoi (`Lane::send`), par son rang dans `nets`.
    pub fn start_nets(nets: &[&Path]) -> Result<(Gpu, String), String> {
        Gpu::start_on(Device::Vulkan, nets)
    }

    /// `start_nets` sur le moteur `device`.
    pub fn start_on(device: Device, nets: &[&Path]) -> Result<(Gpu, String), String> {
        use crate::vulkan::{MAX_BATCH, ORDER, STEPS};
        use std::sync::atomic::{AtomicBool, AtomicU64};
        let engine = match device {
            Device::Vulkan => Engine::Vulkan(Box::new(crate::vulkan::Engine::new_nets(nets, MAX_BATCH)?)),
            #[cfg(feature = "cuda")]
            Device::Cuda => Engine::Cuda(Box::new(crate::cuda::Engine::new_nets(nets)?)),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda => return Err("programme lab8 compilé sans CUDA (cargo build --features cuda)".into()),
        };
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
            stamps: std::env::var_os("LAB8_LATENCY").map(|_| (0..ORDER).map(|_| Default::default()).collect()),
        });
        let in_flight =
            std::env::var("LAB8_IN_FLIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(IN_FLIGHT).clamp(1, STEPS);
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
        if let Some(stamps) = &r.stamps {
            self.measured_wait(sent, stamps);
            return;
        }
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
impl Gpu {
    /// `wait`, et le trajet de `sent` (`LATENCY`).
    fn measured_wait(&self, sent: &Sent, stamps: &[[std::sync::atomic::AtomicU64; 3]]) {
        use std::sync::atomic::Ordering;
        let r = &*self.ring;
        let called = now();
        let already = self.done(sent);
        while !self.done(sent) {
            let held = r.evaluated.0.lock().expect("verrou");
            if !self.done(sent) {
                let _ = r.evaluated.1.wait_timeout(held, std::time::Duration::from_millis(1));
            }
        }
        let woke = now();
        let order = crate::vulkan::ORDER as u64;
        let stamp = |t: u64, k: usize| stamps[(t % order) as usize][k].load(Ordering::Relaxed);
        let last = sent.first + sent.cells.len() as u64 - 1;
        let (written, finished) = (stamp(sent.first, 0), stamp(last, 2));
        let add = |what: usize, v: u64| LATENCY[what].fetch_add(v, Ordering::Relaxed);
        add(L_SENT, 1);
        add(L_TRIP, woke.saturating_sub(written));
        if already {
            add(L_IDLE, 1);
            add(L_LATE, called.saturating_sub(finished));
        } else {
            add(L_WAITED, 1);
            add(L_WOKEN, woke.saturating_sub(finished));
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
    /// d'ordre ; le fil « gpu » les soumet dès qu'elles y sont. `net` : le rang du réseau qui les
    /// évalue (`Gpu::start_nets`).
    pub fn send<P: Position>(&mut self, gpu: &Gpu, positions: &[P], net: u8) -> Sent {
        use std::sync::atomic::Ordering;
        let (r, order) = (&*gpu.ring, crate::vulkan::ORDER as u64);
        assert!(positions.len() <= self.free.len(), "plus de place pour ce fil dans l'anneau du GPU");
        let cells: Vec<u32> = (0..positions.len()).map(|_| self.free.pop().expect("une place")).collect();
        for (&cell, p) in cells.iter().zip(positions) {
            r.engine.write(cell as usize, p, net);
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
        if let Some(stamps) = &r.stamps {
            let at = now();
            for j in 0..cells.len() as u64 {
                stamps[((first + j) % order) as usize][0].store(at, Ordering::Relaxed);
            }
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

/// Le fil « gpu » : les commandes de `ROUNDS` × `STEPS` étapes sont soumises d'avance ; chaque
/// calcul est un feu vert (`Engine::launch`) pour les numéros de la file d'ordre écrits à la suite,
/// sa fin un événement que le fil surveille ; jusqu'à l'arrêt (`Gpu::drop`), une fois tout évalué.
/// Une soumission pour `STEPS` calculs : pas de tâche du noyau par calcul, dont le GPU payait le
/// changement (80 à 150 µs de trou avant chaque calcul soumis seul, 30 par étape, journal).
#[cfg(feature = "vulkan")]
fn drive(r: &Ring, in_flight: usize) {
    use crate::batch::{SENT, SENT_LOTS, SENT_POSITIONS};
    use crate::vulkan::{MAX_BATCH, ORDER, ROUNDS, STEP_SLOTS, STEPS};
    use std::sync::atomic::Ordering;
    let order = ORDER as u64;
    let fail = |e: String| -> ! { panic!("GPU : {e}") };
    let e = &r.engine;
    // la soumission en cours de chaque tour d'étapes ; les étapes qu'ils couvrent
    let mut rounds: Vec<u64> = (0..ROUNDS).map(|k| e.submit_round(k).unwrap_or_else(|e| fail(e))).collect();
    let mut covered = STEP_SLOTS as u64;
    // la prochaine étape, le prochain numéro de la file ; les étapes en vol (étape, fin, positions,
    // feu vert)
    let (mut step, mut next) = (0u64, 0u64);
    let mut flight = std::collections::VecDeque::<(u64, u64, usize, u64)>::new();
    let mut last_end = 0u64;
    loop {
        // les étapes finies, dans l'ordre ; un tour fini est soumis à nouveau, derrière les autres
        let mut woke = false;
        while let Some(&(k, end, count, launched)) = flight.front() {
            let slot = (k % STEP_SLOTS as u64) as usize;
            if !e.finished(slot).unwrap_or_else(|e| fail(e)) {
                break;
            }
            let ran = e.profile_step(slot, count);
            if let (Some(_), Some((began, ended))) = (&r.stamps, ran) {
                let add = |what: usize, v: u64| LATENCY[what].fetch_add(v, Ordering::Relaxed);
                add(L_CALCS, 1);
                add(L_TO_GPU, began.saturating_sub(launched));
                add(L_RUN, ended.saturating_sub(began));
                add(L_NOTICE, now().saturating_sub(ended));
                if last_end > 0 {
                    add(L_GAP, began.saturating_sub(last_end));
                }
                last_end = ended;
            }
            if let Some(stamps) = &r.stamps {
                let at = now();
                for t in end - count as u64..end {
                    let s = &stamps[(t % order) as usize];
                    LATENCY[L_COMPUTED].fetch_add(at.saturating_sub(s[1].load(Ordering::Relaxed)), Ordering::Relaxed);
                    s[2].store(at, Ordering::Relaxed);
                }
            }
            e.recycle(slot).unwrap_or_else(|e| fail(e));
            r.done.store(end, Ordering::Release);
            flight.pop_front();
            woke = true;
            if slot % STEPS == STEPS - 1 {
                let round = slot / STEPS;
                e.wait(rounds[round]).unwrap_or_else(|e| fail(e));
                rounds[round] = e.submit_round(round).unwrap_or_else(|e| fail(e));
                covered += STEPS as u64;
            }
        }
        if woke {
            let _held = r.evaluated.0.lock().expect("verrou");
            r.evaluated.1.notify_all();
        }
        // les numéros écrits à la suite, en une étape
        if flight.len() < in_flight && step < covered {
            let mut end = next;
            while end - next < MAX_BATCH as u64 && r.seq[(end % order) as usize].load(Ordering::Acquire) == end + 1 {
                end += 1;
            }
            if end > next {
                let count = (end - next) as usize;
                let slot = (step % STEP_SLOTS as u64) as usize;
                let calling = now();
                e.launch(slot, (next % order) as usize, count).unwrap_or_else(|e| fail(e));
                let launched = now();
                SENT[SENT_LOTS].fetch_add(1, Ordering::Relaxed);
                SENT[SENT_POSITIONS].fetch_add(count as u64, Ordering::Relaxed);
                if let Some(stamps) = &r.stamps {
                    LATENCY[L_SUBMIT].fetch_add(launched - calling, Ordering::Relaxed);
                    for t in next..end {
                        let s = &stamps[(t % order) as usize];
                        LATENCY[L_QUEUED]
                            .fetch_add(launched.saturating_sub(s[0].load(Ordering::Relaxed)), Ordering::Relaxed);
                        s[1].store(launched, Ordering::Relaxed);
                    }
                    LATENCY[L_POSITIONS].fetch_add(count as u64, Ordering::Relaxed);
                }
                flight.push_back((step, end, count, launched));
                step += 1;
                next = end;
                continue;
            }
        }
        if flight.is_empty() {
            if r.stop.load(Ordering::Acquire) && r.head.load(Ordering::Acquire) == next {
                // les étapes soumises sans calcul passent à vide, puis tout s'arrête
                while step < covered {
                    e.launch((step % STEP_SLOTS as u64) as usize, 0, 0).unwrap_or_else(|e| fail(e));
                    step += 1;
                }
                for &v in &rounds {
                    e.wait(v).unwrap_or_else(|e| fail(e));
                }
                break;
            }
            // rien en vol : attend des positions (`Lane::send` réveille)
            let held = r.written.0.lock().expect("verrou");
            r.sleeping.store(true, Ordering::Release);
            if r.seq[(next % order) as usize].load(Ordering::Acquire) != next + 1 {
                let _ = r.written.1.wait_timeout(held, std::time::Duration::from_millis(1));
            }
            r.sleeping.store(false, Ordering::Release);
        } else {
            // une étape en vol : on la surveille (une lecture en mémoire)
            std::thread::yield_now();
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
