//! Le plafond de la valeur (lab4/journal.md) : la part de chance d'une position. Des positions de
//! parties neuves, jouées par un agent jusqu'au début d'une manche tirée au hasard ; depuis chacune,
//! des suites jouées jusqu'au bout par le même agent, chacune dans un monde possible tiré à nouveau
//! (`know::determinize` : ordre caché des pioches, dés à venir). La dispersion des vainqueurs donne
//! ce qu'une valeur parfaite pour ces joueurs prédirait ; la valeur du réseau est relevée à côté.
//!
//! Les parties avancent sur le GPU comme celles de l'arène : plusieurs par fil, chacune attend
//! ses évaluations, envoyées aussitôt, pendant que les autres avancent (`drive`).

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use smallmobile_kernel::{Game, Phase};

use crate::az::{Az, act_with};
use crate::batch::Batch;
use crate::encode::encode;
use crate::gpu::Gpu;
use crate::know::{Knowledge, determinize};
use crate::rand::{Rng, mix};
use crate::table::{Outcome, Table};

/// Une position d'où partent les suites : la partie, ce qu'en sait le joueur qui agit, et la
/// victoire que lui prédit le réseau pour chaque joueur (réel).
#[derive(Clone, Copy)]
pub struct Position<'m> {
    pub game: Game<'m>,
    pub knowledge: Knowledge,
    pub net: [f32; 5],
}

pub struct Config {
    pub players: Vec<usize>,
    pub positions: usize,
    pub playouts: usize,
    pub seed: u64,
    pub threads: usize,
    pub concurrent: usize,
}

/// La position `i` : sa table (nombre de joueurs, carte), la graine de sa partie et la manche où
/// elle s'arrête (de la 2e à la dernière).
pub fn deal(cfg: &Config, i: usize) -> (Table, u32, u64) {
    let deal = mix(cfg.seed, i as u64);
    let n = cfg.players[i % cfg.players.len()];
    (Table::new(n, deal % 1_000_000_007), mix(deal, 1) as u32, deal)
}

/// Les positions, puis les vainqueurs de chaque suite (`playouts` par position).
pub fn run<'m>(az: &Az, gpu: &Gpu, cfg: &Config, tables: &'m [Table]) -> (Vec<Position<'m>>, Vec<Vec<Outcome>>) {
    let n = cfg.positions;
    let positions: Mutex<Vec<Option<Position<'m>>>> = Mutex::new(vec![None; n]);
    let outcomes: Mutex<Vec<Vec<Outcome>>> = Mutex::new(vec![Vec::new(); n]);
    let (next, done) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let (next2, done2) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let barrier = std::sync::Barrier::new(cfg.threads);
    std::thread::scope(|scope| {
        for lane in gpu.lanes(cfg.threads) {
            let (positions, outcomes, barrier) = (&positions, &outcomes, &barrier);
            let (next, done, next2, done2) = (&next, &done, &next2, &done2);
            scope.spawn(move || {
                let batch = Batch::new(&az.net, Some((gpu, lane, 0)));
                // 1. les positions : chaque partie jusqu'au début de sa manche
                let take = || {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    (i < n).then(|| {
                        let (_, seed, deal) = deal(cfg, i);
                        let g = tables[i].game(seed);
                        let round = 2 + (mix(deal, 3) % (g.state.max_rounds - 1) as u64) as i32;
                        let job: Job<'_, (Game<'m>, Knowledge)> =
                            Box::pin(until(az, g, Knowledge::default(), round, mix(deal, 4), &batch));
                        (i, job)
                    })
                };
                drive(&batch, gpu, cfg.concurrent, take, |i, (g, k)| {
                    let w = az.net.eval(&encode(&g, &k), g.map).win;
                    let (me, players) = (g.state.actor().expect("un joueur agit"), g.state.player_count());
                    let mut net = [0.0; 5];
                    for r in 0..players {
                        net[(me + r) % players] = w[r];
                    }
                    positions.lock().unwrap()[i] = Some(Position { game: g, knowledge: k, net });
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d.is_multiple_of(20) || d == n {
                        eprint!("\r{d}/{n} positions");
                    }
                });
                barrier.wait();
                // 2. les suites de chaque position, chacune dans un monde tiré à nouveau
                let all: Vec<Position<'m>> = positions.lock().unwrap().iter().map(|p| p.expect("position")).collect();
                let total = n * cfg.playouts;
                let take = || {
                    let t = next2.fetch_add(1, Ordering::Relaxed);
                    (t < total).then(|| {
                        let (i, j) = (t / cfg.playouts, t % cfg.playouts);
                        let p = all[i];
                        let world = mix(mix(cfg.seed, 1_000_003 + i as u64), j as u64);
                        let g = determinize(&p.game, &p.knowledge, &mut Rng::new(world));
                        let job: Job<'_, Outcome> = Box::pin(out(az, g, p.knowledge, mix(world, 5), &batch));
                        (i, job)
                    })
                };
                drive(&batch, gpu, cfg.concurrent, take, |i, o| {
                    outcomes.lock().unwrap()[i].push(o);
                    let d = done2.fetch_add(1, Ordering::Relaxed) + 1;
                    if d.is_multiple_of(200) || d == total {
                        eprint!("\r{d}/{total} suites");
                    }
                });
            });
        }
    });
    eprintln!();
    let positions = positions.into_inner().unwrap().into_iter().map(|p| p.expect("position")).collect();
    (positions, outcomes.into_inner().unwrap())
}

type Job<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// La partie jouée par `az` à toutes les places jusqu'au début de la manche `round`.
async fn until<'m>(az: &Az, mut g: Game<'m>, mut k: Knowledge, round: i32, seed: u64, batch: &Batch<'_>) -> (Game<'m>, Knowledge) {
    let mut rng = Rng::new(seed);
    while g.state.phase != Phase::GameOver && g.state.round < round {
        for a in act_with(az, &g, &k, &mut rng.fork(), batch).await {
            k.apply(&mut g, &a).expect("action de la recherche");
        }
    }
    (g, k)
}

/// La partie jouée par `az` à toutes les places jusqu'au bout.
async fn out(az: &Az, mut g: Game<'_>, mut k: Knowledge, seed: u64, batch: &Batch<'_>) -> Outcome {
    let mut rng = Rng::new(seed);
    while g.state.phase != Phase::GameOver {
        for a in act_with(az, &g, &k, &mut rng.fork(), batch).await {
            k.apply(&mut g, &a).expect("action de la recherche");
        }
    }
    Outcome::of(&g)
}

/// Mène les tâches de `take` sur le GPU, `concurrent` à la fois : chacune avance jusqu'à attendre
/// ses évaluations (envoyées au GPU par `batch`), et reprend dès qu'elles sont là. `finish` reçoit
/// le résultat de chacune, avec son numéro.
fn drive<'b, T>(
    batch: &'b Batch<'_>,
    gpu: &Gpu,
    concurrent: usize,
    take: impl Fn() -> Option<(usize, Job<'b, T>)>,
    mut finish: impl FnMut(usize, T),
) {
    let mut cx = Context::from_waker(Waker::noop());
    let mut slots: Vec<Option<(usize, Job<'b, T>)>> = (0..concurrent).map(|_| None).collect();
    let mut ready: Vec<(usize, Option<Vec<crate::net::Eval>>)> = (0..concurrent).map(|s| (s, None)).collect();
    let mut waiting = VecDeque::new();
    loop {
        while let Some((s, evals)) = ready.pop() {
            if let Some(evals) = evals {
                batch.resume(evals);
            }
            loop {
                if slots[s].is_none() {
                    let Some(job) = take() else { break };
                    slots[s] = Some(job);
                }
                let (i, job) = slots[s].as_mut().expect("une tâche");
                if let Poll::Ready(t) = job.as_mut().poll(&mut cx) {
                    let i = *i;
                    slots[s] = None;
                    finish(i, t);
                } else {
                    waiting.push_back((s, batch.send().expect("une tâche qui attend a ses positions")));
                    break;
                }
            }
        }
        let Some(i) = waiting.iter().position(|(_, sent)| gpu.done(sent)).or((!waiting.is_empty()).then_some(0)) else {
            break;
        };
        let (s, sent) = waiting.remove(i).expect("une attente");
        ready.push((s, Some(batch.receive(sent))));
    }
}
