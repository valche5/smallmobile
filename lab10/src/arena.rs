//! Comparer des joueurs sur beaucoup de parties.
//!
//! Format « duplicate » : chaque donne (nombre de joueurs, carte, pioches, dés) est jouée autant de
//! fois qu'il y a de places, la table (`Lineup`) tournant d'une place à chaque fois : chaque joueur
//! occupe chaque place une fois. La chance de la donne et de la place s'annule ainsi en grande
//! partie. Le cas courant : le joueur testé (`hero`) face au reste de la table (`field`).
//!
//! Aucun joueur ne sait qui occupe les autres places. Le résultat de chaque place de chaque partie
//! peut être écrit (`write`) pour `labtools/score.py`, qui calcule le score et ses intervalles par
//! donne.
//!
//! Une arène peut s'arrêter dès que son résultat est clair (`Sequential`) : elle garde alors les
//! premières donnes, les mêmes que dans l'arène complète.
//!
//! Sur GPU (`run` avec `gpu`), comme le selfplay : chaque fil mène plusieurs parties, un joueur
//! `az` attend ses évaluations du GPU (un seul moteur pour tous les réseaux, chaque position avec
//! le sien) pendant que le fil fait avancer les autres parties ; les IA du jeu jouent directement.

use std::io::Write;
use std::pin::Pin;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use smallmobile_kernel::Phase;

use crate::agent::Agent;
use crate::batch::Batch;
use crate::gpu::Gpu;
use crate::know::Knowledge;
use crate::moves;
use crate::net::Net;
use crate::rand::{Rng, mix};
use crate::table::{Outcome, Table};

/// Garde-fou : aucune vraie partie n'approche ce nombre de coups.
const MAX_MOVES: usize = 20_000;

/// Une partie jouée : le résultat, le nombre de décisions, et pour chaque place la durée (s) de
/// chacune de ses décisions où il y avait un choix (toute fin de tour ou défense comprise).
pub struct Played {
    pub outcome: Outcome,
    pub moves: usize,
    pub times: Vec<Vec<f32>>,
}

/// Joue une partie jusqu'au bout.
pub fn play_game(table: &Table, seed: u32, seats: &[&dyn Agent], rng_seed: u64) -> Played {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut rng = Rng::new(rng_seed);
    let mut times = vec![Vec::new(); seats.len()];
    let mut count = 0;
    while g.state.phase != Phase::GameOver {
        assert!(count < MAX_MOVES, "partie sans fin");
        let pid = g.state.actor().expect("un joueur doit agir");
        let unit = matches!(g.state.phase, Phase::Redeploy | Phase::Defend);
        let choice = unit || moves::legal(&g, &k).len() > 1;
        let start = Instant::now();
        let actions = seats[pid].act(&g, &k, &mut rng.fork());
        if choice {
            times[pid].push(start.elapsed().as_secs_f32());
        }
        for a in &actions {
            if let Err(e) = k.apply(&mut g, a) {
                panic!("action refusée ({}) : {a:?} → {e:?}", seats[pid].name());
            }
        }
        count += 1;
    }
    Played { outcome: Outcome::of(&g), moves: count, times }
}

/// `play_game`, les joueurs `az` attendant leurs évaluations du groupe de leur réseau (`batches`).
pub async fn play_game_on(
    table: &Table,
    seed: u32,
    seats: &[&dyn Agent],
    rng_seed: u64,
    batches: &[Batch<'_>],
) -> Played {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut rng = Rng::new(rng_seed);
    let mut times = vec![Vec::new(); seats.len()];
    let mut count = 0;
    while g.state.phase != Phase::GameOver {
        assert!(count < MAX_MOVES, "partie sans fin");
        let pid = g.state.actor().expect("un joueur doit agir");
        let unit = matches!(g.state.phase, Phase::Redeploy | Phase::Defend);
        let choice = unit || moves::legal(&g, &k).len() > 1;
        // la durée comprend le travail des autres parties du fil pendant les attentes
        let start = Instant::now();
        let actions = match seats[pid].az() {
            Some(az) => {
                let batch = batches.iter().find(|b| b.serves(&az.net)).expect("le groupe du réseau de ce joueur");
                crate::az::act_with(az, &g, &k, &mut rng.fork(), batch).await
            }
            None => seats[pid].act(&g, &k, &mut rng.fork()),
        };
        if choice {
            times[pid].push(start.elapsed().as_secs_f32());
        }
        for a in &actions {
            if let Err(e) = k.apply(&mut g, a) {
                panic!("action refusée ({}) : {a:?} → {e:?}", seats[pid].name());
            }
        }
        count += 1;
    }
    Played { outcome: Outcome::of(&g), moves: count, times }
}

/// Parties que mène chaque fil de l'arène sur GPU (`LAB10_CONCURRENT` pour essayer), comme le
/// selfplay (`selfplay::GPU_GAMES`).
const GPU_GAMES: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
pub struct Tally {
    pub games: usize,
    pub win: f64,
    pub win2: f64,
    pub margin: f64,
    pub rank: f64,
}

impl Tally {
    fn add(&mut self, r: &SeatResult) {
        self.games += 1;
        self.win += r.win;
        self.win2 += r.win * r.win;
        self.margin += r.margin as f64;
        self.rank += r.rank as f64;
    }

    fn merge(&mut self, o: &Tally) {
        self.games += o.games;
        self.win += o.win;
        self.win2 += o.win2;
        self.margin += o.margin;
        self.rank += o.rank;
    }

    pub fn win_rate(&self) -> f64 {
        self.win / self.games.max(1) as f64
    }

    /// Demi-largeur de l'intervalle de confiance à 95 % du taux de victoire, les parties supposées
    /// indépendantes (`labtools/score.py` tient compte des donnes).
    pub fn ci95(&self) -> f64 {
        let n = self.games.max(2) as f64;
        let mean = self.win / n;
        let var = (self.win2 / n - mean * mean).max(0.0) * n / (n - 1.0);
        1.96 * (var / n).sqrt()
    }
}

pub struct Config {
    pub deals: usize,
    pub threads: usize,
    pub seed: u64,
    /// Arrêt dès que le résultat d'un rôle est clair ; sans lui, toutes les donnes.
    pub stop: Option<Sequential>,
}

/// Arrêt anticipé : test séquentiel de Wald sur le score d'un rôle (`score`), dans son
/// approximation normale, comme les bancs d'essai des moteurs d'échecs. H0 « score 0 » contre H1
/// « score `delta` points », 5 % d'erreur de chaque côté. Le score est regardé toutes les `look`
/// donnes par table, sur les premières donnes (équilibrées entre les tailles), à partir du
/// deuxième regard ; sans conclusion, l'arène va jusqu'au bout.
#[derive(Clone, Copy, Debug)]
pub struct Sequential {
    pub role: usize,
    pub delta: f64,
    pub look: usize,
}

/// ln(0,95 / 0,05) : les bornes du rapport de vraisemblance pour 5 % d'erreur de chaque côté.
const BOUND: f64 = 2.944;

/// Le score d'un rôle en points au-dessus de l'attendu (1/n), tailles à poids égaux, et son
/// écart-type, d'après la valeur de chaque donne : `(joueurs, écart moyen des places du rôle à 1/n,
/// en points)`, comme `labtools/score.py`. `None` si une taille a moins de deux donnes.
pub fn score(deals: &[(usize, f64)]) -> Option<(f64, f64)> {
    let mut sizes = [(0usize, 0.0f64, 0.0f64); 6];
    for &(n, x) in deals {
        let s = &mut sizes[n];
        *s = (s.0 + 1, s.1 + x, s.2 + x * x);
    }
    let used: Vec<_> = sizes.iter().filter(|s| s.0 > 0).collect();
    if used.is_empty() || used.iter().any(|s| s.0 < 2) {
        return None;
    }
    let k = used.len() as f64;
    let (mut score, mut var) = (0.0, 0.0);
    for &&(count, sum, squares) in &used {
        let c = count as f64;
        let mean = sum / c;
        score += mean / k;
        var += (squares - c * mean * mean).max(0.0) / (c - 1.0) / c;
    }
    Some((score, var.sqrt() / k))
}

/// La conclusion du test séquentiel pour un score et son écart-type : `Some(true)` si le rôle gagne
/// `delta` points plutôt que 0, `Some(false)` dans le cas contraire, `None` pour continuer.
pub fn verdict(score: f64, se: f64, delta: f64) -> Option<bool> {
    let llr = delta * (score - delta / 2.0) / (se * se);
    (llr.abs() >= BOUND).then_some(llr > 0.0)
}

/// La valeur de chaque donne pour un rôle : `(joueurs, écart moyen de ses places à 1/n, en points)`.
pub fn deal_values(results: &[SeatResult], role: usize) -> Vec<(usize, f64)> {
    let mut sums: std::collections::BTreeMap<usize, (usize, f64, usize)> = Default::default();
    for r in results.iter().filter(|r| r.role == role) {
        let e = sums.entry(r.deal).or_insert((r.players, 0.0, 0));
        e.1 += 100.0 * (r.win - 1.0 / r.players as f64);
        e.2 += 1;
    }
    sums.into_values().map(|(n, sum, seats)| (n, sum / seats as f64)).collect()
}

/// Les donnes finies, pour le test séquentiel.
struct Progress {
    players: Vec<usize>,
    /// Parties de chaque donne encore à jouer.
    left: Vec<usize>,
    /// Somme des écarts à 1/n des places du rôle (points), et nombre de ces places.
    value: Vec<f64>,
    seats: Vec<usize>,
    /// Les donnes `0..complete` sont finies.
    complete: usize,
    /// Prochain regard (nombre de donnes), et écart entre deux regards.
    look: usize,
    step: usize,
    verdict: Option<bool>,
}

impl Progress {
    fn new(players: Vec<usize>, step: usize) -> Progress {
        let deals = players.len();
        Progress {
            left: players.clone(),
            players,
            value: vec![0.0; deals],
            seats: vec![0; deals],
            complete: 0,
            look: 2 * step,
            step,
            verdict: None,
        }
    }

    /// Compte une partie de la donne `d` ; si le test conclut, renvoie le nombre de donnes à garder.
    fn add(&mut self, d: usize, value: f64, seats: usize, delta: f64) -> Option<usize> {
        self.value[d] += value;
        self.seats[d] += seats;
        self.left[d] -= 1;
        while self.complete < self.left.len() && self.left[self.complete] == 0 {
            self.complete += 1;
        }
        while self.verdict.is_none() && self.look <= self.complete {
            let deals: Vec<(usize, f64)> = (0..self.look)
                .filter(|&d| self.seats[d] > 0)
                .map(|d| (self.players[d], self.value[d] / self.seats[d] as f64))
                .collect();
            self.verdict = score(&deals).and_then(|(s, se)| verdict(s, se, delta));
            if self.verdict.is_some() {
                return Some(self.look);
            }
            self.look += self.step;
        }
        None
    }
}

/// Une table avant rotation : le joueur de chaque place, et son rôle. Les places d'un même rôle
/// sont comptées ensemble (le joueur testé face à plusieurs copies du même adversaire).
pub struct Lineup<'a> {
    pub seats: Vec<&'a dyn Agent>,
    pub roles: Vec<usize>,
}

impl<'a> Lineup<'a> {
    /// Le joueur testé (rôle 0) à la première place, `field` (rôle 1) aux autres.
    pub fn hero(players: usize, hero: &'a dyn Agent, field: &'a dyn Agent) -> Lineup<'a> {
        let mut seats = vec![field; players];
        seats[0] = hero;
        Lineup { seats, roles: (0..players).map(|i| usize::from(i > 0)).collect() }
    }
}

/// Le résultat d'une place dans une partie.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeatResult {
    pub deal: usize,
    pub players: usize,
    /// Rotation : la place `p` de la table avant rotation joue à la place `(p + shift) % players`.
    pub shift: usize,
    pub seat: usize,
    pub role: usize,
    pub win: f64,
    pub coins: i32,
    pub margin: i32,
    pub rank: usize,
    /// Décisions avec plus d'un coup possible, et leur durée totale (s).
    pub decisions: usize,
    pub seconds: f64,
}

pub struct Run {
    /// Par donne, rotation et place.
    pub results: Vec<SeatResult>,
    /// Durée de chaque décision (s), par rôle.
    pub times: Vec<Vec<f32>>,
    /// Donnes jouées : `cfg.deals`, ou moins si le test séquentiel a conclu (`verdict`).
    pub deals: usize,
    pub verdict: Option<bool>,
}

/// Joue `cfg.deals` donnes ; la donne `d` se joue à la table `lineups[d % lineups.len()]`, une fois
/// par rotation. Avec `gpu` (le GPU, et les réseaux des joueurs `az` dans l'ordre de
/// `Gpu::start_nets`), sur GPU.
pub fn run(cfg: &Config, lineups: &[Lineup], gpu: Option<(&Gpu, &[&Net])>) -> Run {
    let roles = lineups.iter().flat_map(|l| l.roles.iter()).max().map_or(0, |r| r + 1);
    let players: Vec<usize> = (0..cfg.deals).map(|d| lineups[d % lineups.len()].seats.len()).collect();
    let jobs: Vec<(usize, usize)> =
        players.iter().enumerate().flat_map(|(d, &n)| (0..n).map(move |shift| (d, shift))).collect();
    let next = AtomicUsize::new(0);
    // les donnes à jouer : toutes, ou les premières si le test séquentiel conclut
    let limit = AtomicUsize::new(cfg.deals);
    let step = cfg.stop.map_or(1, |s| s.look.max(1) * lineups.len());
    let progress = Mutex::new(Progress::new(players, step));
    let start = Instant::now();
    let done = AtomicUsize::new(0);
    // la partie suivante à jouer : sa donne, sa rotation, et les joueurs de chaque place
    let take = || -> Option<(usize, usize, Vec<&dyn Agent>)> {
        let j = next.fetch_add(1, Ordering::Relaxed);
        let &(d, shift) = jobs.get(j)?;
        if d >= limit.load(Ordering::Relaxed) {
            return None;
        }
        let lineup = &lineups[d % lineups.len()];
        let n = lineup.seats.len();
        // la place `seat` de la table reçoit la place `(seat + n − shift) % n` d'avant la rotation
        Some((d, shift, (0..n).map(|seat| lineup.seats[(seat + n - shift) % n]).collect()))
    };
    let deal_of = |d: usize, shift: usize| {
        let deal = mix(cfg.seed, d as u64);
        let n = lineups[d % lineups.len()].seats.len();
        (Table::new(n, deal % 1_000_000_007), mix(deal, 1) as u32, mix(deal, 2 + shift as u64))
    };
    // par fil : les résultats, et la durée des décisions de chaque place (donne, rôle, durées)
    type Part = (Vec<SeatResult>, Vec<(usize, usize, Vec<f32>)>);
    let finish = |d: usize, shift: usize, played: Played, part: &mut Part| {
        let (results, times) = part;
        let lineup = &lineups[d % lineups.len()];
        let n = lineup.seats.len();
        let o = &played.outcome;
        let first = results.len();
        for (seat, seat_times) in played.times.into_iter().enumerate() {
            let role = lineup.roles[(seat + n - shift) % n];
            results.push(SeatResult {
                deal: d,
                players: n,
                shift,
                seat,
                role,
                win: o.win[seat],
                coins: o.coins[seat],
                margin: o.margin(seat),
                rank: o.rank[seat],
                decisions: seat_times.len(),
                seconds: seat_times.iter().map(|&t| t as f64).sum(),
            });
            times.push((d, role, seat_times));
        }
        if let Some(stop) = cfg.stop {
            let mine = results[first..].iter().filter(|r| r.role == stop.role);
            let value = mine.clone().map(|r| 100.0 * (r.win - 1.0 / n as f64)).sum();
            if let Some(kept) = progress.lock().unwrap().add(d, value, mine.count(), stop.delta) {
                limit.store(kept, Ordering::Relaxed);
            }
        }
        let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
        if finished.is_multiple_of(50) || finished == jobs.len() {
            eprint!("\r{finished}/{} parties, {:.0} s", jobs.len(), start.elapsed().as_secs_f64());
        }
    };
    // les places de chaque fil dans l'anneau du GPU, une part par réseau
    let mut lanes: Vec<Vec<crate::gpu::Lane>> = (0..cfg.threads).map(|_| Vec::new()).collect();
    if let Some((gpu, nets)) = gpu {
        for (i, lane) in gpu.lanes(cfg.threads * nets.len()).into_iter().enumerate() {
            lanes[i / nets.len()].push(lane);
        }
    }
    let concurrent = std::env::var("LAB10_CONCURRENT").ok().and_then(|v| v.parse().ok()).unwrap_or(GPU_GAMES);
    // chaque partie a au plus un tour de recherche en vol (`az::CONSIDERED` positions, fins simulées
    // comprises) : celles d'un fil doivent tenir dans ses places de l'anneau, pour chaque réseau
    if let Some(lane) = lanes.iter().flatten().next() {
        assert!(
            concurrent * crate::az::CONSIDERED <= lane.free(),
            "{concurrent} parties par fil : jusqu'à {} positions en vol, l'anneau du GPU en tient {} par fil et par réseau (LAB10_CONCURRENT)",
            concurrent * crate::az::CONSIDERED,
            lane.free()
        );
    }
    let parts: Vec<Part> = std::thread::scope(|scope| {
        let threads: Vec<_> = lanes
            .into_iter()
            .map(|lanes| {
                let (take, deal_of, finish) = (&take, &deal_of, &finish);
                scope.spawn(move || {
                    let mut part: Part = Default::default();
                    match gpu {
                        None => {
                            while let Some((d, shift, seats)) = take() {
                                let (table, seed, rng) = deal_of(d, shift);
                                finish(d, shift, play_game(&table, seed, &seats, rng), &mut part);
                            }
                        }
                        Some((gpu, nets)) => {
                            let batches: Vec<Batch> = nets
                                .iter()
                                .zip(lanes)
                                .enumerate()
                                .map(|(rank, (&net, lane))| Batch::new(net, Some((gpu, lane, rank as u8))))
                                .collect();
                            on_gpu(&batches, gpu, concurrent, take, deal_of, |d, shift, played| {
                                finish(d, shift, played, &mut part)
                            });
                        }
                    }
                    part
                })
            })
            .collect();
        threads.into_iter().map(|t| t.join().expect("fil de l'arène")).collect()
    });
    eprintln!();
    let deals = limit.into_inner();
    let verdict = progress.into_inner().unwrap().verdict;
    let mut run = Run { results: Vec::new(), times: vec![Vec::new(); roles], deals, verdict };
    for (results, times) in parts {
        // les parties commencées après la conclusion ne comptent pas
        run.results.extend(results.into_iter().filter(|r| r.deal < deals));
        for (_, role, t) in times.into_iter().filter(|(d, _, _)| *d < deals) {
            run.times[role].extend(t);
        }
    }
    run.results.sort_by_key(|r| (r.deal, r.shift, r.seat));
    run
}

/// Une partie de l'arène en cours sur GPU : sa donne, sa rotation, et ce qu'elle rendra.
type Running<'a> = Pin<Box<dyn Future<Output = (usize, usize, Played)> + 'a>>;

/// Les parties d'un fil de l'arène sur GPU : `concurrent` à la fois ; chacune avance jusqu'à ce
/// qu'un joueur `az` attende ses évaluations (envoyées aussitôt au GPU, par le groupe de son
/// réseau, `batches`),
/// et reprend dès qu'elles sont là. `take` donne la partie suivante, `finish` reçoit les finies.
fn on_gpu<'b, 's: 'b, 'n: 'b>(
    batches: &'b [Batch<'n>],
    gpu: &Gpu,
    concurrent: usize,
    take: &(dyn Fn() -> Option<(usize, usize, Vec<&'s dyn Agent>)> + Sync),
    deal_of: &(dyn Fn(usize, usize) -> (Table, u32, u64) + Sync),
    mut finish: impl FnMut(usize, usize, Played),
) {
    use crate::gpu::Sent;
    use crate::net::Eval;
    let mut cx = Context::from_waker(Waker::noop());
    let mut slots: Vec<Option<Running<'b>>> = (0..concurrent).map(|_| None).collect();
    // les parties à reprendre (avec les évaluations attendues, et le groupe qui les a) ; celles qui
    // attendent un GPU, avec leur groupe
    type Resumed = (usize, Option<(usize, Vec<Eval>)>);
    let mut ready: Vec<Resumed> = (0..concurrent).map(|s| (s, None)).collect();
    let mut waiting: Vec<(usize, usize, Sent)> = Vec::new();
    loop {
        while let Some((s, evals)) = ready.pop() {
            if let Some((b, evals)) = evals {
                batches[b].resume(evals);
            }
            loop {
                if slots[s].is_none() {
                    let Some((d, shift, seats)) = take() else { break };
                    let (table, seed, rng) = deal_of(d, shift);
                    slots[s] = Some(Box::pin(async move {
                        let played = play_game_on(&table, seed, &seats, rng, batches).await;
                        (d, shift, played)
                    }));
                }
                let running = slots[s].as_mut().expect("une partie");
                let Poll::Ready((d, shift, played)) = running.as_mut().poll(&mut cx) else {
                    // la partie attend : ses positions sont dans le groupe du réseau qui cherche
                    let (b, sent) = batches
                        .iter()
                        .enumerate()
                        .find_map(|(b, batch)| batch.send().map(|sent| (b, sent)))
                        .expect("une partie qui attend a ses positions");
                    waiting.push((s, b, sent));
                    break;
                };
                slots[s] = None;
                finish(d, shift, played);
            }
        }
        if waiting.is_empty() {
            break;
        }
        // la première attente déjà servie, sinon la plus vieille
        let i = waiting.iter().position(|(_, _, sent)| gpu.done(sent)).unwrap_or(0);
        let (s, b, sent) = waiting.remove(i);
        ready.push((s, Some((b, batches[b].receive(sent)))));
    }
}

/// Les résultats d'un rôle, par nombre de joueurs (indice = nombre de joueurs).
pub type Report = [Tally; 6];

pub fn report(run: &Run, role: usize) -> Report {
    let mut out = Report::default();
    for r in run.results.iter().filter(|r| r.role == role) {
        out[r.players].add(r);
    }
    out
}

/// Le tableau des résultats d'un rôle, lisible.
pub fn describe(report: &Report, hero: &str, field: &str) -> String {
    let mut out = format!("{hero} contre {field}\n");
    out += "joueurs  parties  victoires (attendu à force égale)  avance moy.  place moy.\n";
    let mut total = Tally::default();
    let mut excess = 0.0;
    for (n, t) in report.iter().enumerate().filter(|(_, t)| t.games > 0) {
        out += &format!(
            "{n:>7}  {:>7}  {:>5.1} % ± {:>4.1} ({:>4.1} %)        {:>+6.1}        {:>4.2}\n",
            t.games,
            100.0 * t.win_rate(),
            100.0 * t.ci95(),
            100.0 / n as f64,
            t.margin / t.games as f64,
            1.0 + t.rank / t.games as f64,
        );
        total.merge(t);
        excess += t.win - t.games as f64 / n as f64;
    }
    out += &format!(
        "  total  {:>7}  {:>5.1} % ± {:>4.1}   écart à l'attendu : {:+.1} points\n",
        total.games,
        100.0 * total.win_rate(),
        100.0 * total.ci95(),
        100.0 * excess / total.games.max(1) as f64,
    );
    out
}

/// Durée des décisions d'un rôle : moyenne et 95e centile.
pub fn describe_times(times: &[f32]) -> String {
    if times.is_empty() {
        return "aucune décision".into();
    }
    let mut sorted = times.to_vec();
    sorted.sort_by(f32::total_cmp);
    let mean = sorted.iter().map(|&t| t as f64).sum::<f64>() / sorted.len() as f64;
    let p95 = sorted[(sorted.len() - 1) * 95 / 100];
    format!("{} décisions, {:.1} ms en moyenne, p95 {:.1} ms", sorted.len(), 1e3 * mean, 1e3 * p95 as f64)
}

/// Écrit le résultat de chaque place de chaque partie, pour `labtools/score.py` : un en-tête commenté
/// (graine, rôles), puis une ligne par place, séparée par des tabulations.
pub fn write(out: &mut impl Write, cfg: &Config, roles: &[String], run: &Run) -> std::io::Result<()> {
    writeln!(out, "# graine {}, {} donnes", cfg.seed, run.deals)?;
    for (r, name) in roles.iter().enumerate() {
        writeln!(out, "# rôle {r} : {name}")?;
    }
    writeln!(out, "deal\tplayers\tshift\tseat\trole\twin\tcoins\tmargin\trank\tdecisions\tseconds")?;
    for r in &run.results {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}",
            r.deal, r.players, r.shift, r.seat, r.role, r.win, r.coins, r.margin, r.rank, r.decisions, r.seconds
        )?;
    }
    Ok(())
}
