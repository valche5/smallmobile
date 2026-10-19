//! Comparer des joueurs sur beaucoup de parties.
//!
//! Format « duplicate » : chaque donne (nombre de joueurs, carte, pioches, dés) est jouée autant de
//! fois qu'il y a de places, la table (`Lineup`) tournant d'une place à chaque fois : chaque joueur
//! occupe chaque place une fois. La chance de la donne et de la place s'annule ainsi en grande
//! partie. Le cas courant : le joueur testé (`hero`) face au reste de la table (`field`).
//!
//! Chaque joueur connaît le type des autres places (`Agent::model`), comme dans le jeu où l'on voit
//! quelles places sont des IA. Le résultat de chaque place de chaque partie peut être écrit
//! (`write`) pour `lab/py/score.py`, qui calcule le score et ses intervalles par donne.

use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use smallmobile_brain::opponents::Model;
use smallmobile_kernel::Phase;

use crate::agent::Agent;
use crate::know::{Knowledge, Returns};
use crate::moves;
use crate::rand::{Rng, mix};
use crate::table::{Outcome, Table};

/// Garde-fou : aucune vraie partie n'approche ce nombre de coups.
const MAX_MOVES: usize = 20_000;

/// Une partie jouée : le résultat, le nombre de coups, et pour chaque place la durée (s) de chacune
/// de ses décisions où il y avait plus d'un coup possible.
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
    let models: Vec<Model> = seats.iter().map(|a| a.model()).collect();
    let mut times = vec![Vec::new(); seats.len()];
    let mut count = 0;
    while g.state.phase != Phase::GameOver {
        assert!(count < MAX_MOVES, "partie sans fin");
        let pid = g.state.actor().expect("un joueur doit agir");
        let choice = moves::legal(&g).len() > 1;
        let start = Instant::now();
        let m = seats[pid].act_at(&g, &k, &models, &mut rng.fork());
        if choice {
            times[pid].push(start.elapsed().as_secs_f32());
        }
        let mut returns = Returns::default();
        if let Err(e) = moves::play(&mut g, &m, &mut returns) {
            panic!("coup refusé ({}) : {m:?} → {e:?}", seats[pid].name());
        }
        k.observe(returns.0, &g.state);
        count += 1;
    }
    Played { outcome: Outcome::of(&g), moves: count, times }
}

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
    /// indépendantes (`lab/py/score.py` tient compte des donnes).
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
}

/// Joue `cfg.deals` donnes ; la donne `d` se joue à la table `lineups[d % lineups.len()]`, une fois
/// par rotation.
pub fn run(cfg: &Config, lineups: &[Lineup]) -> Run {
    let roles = lineups.iter().flat_map(|l| l.roles.iter()).max().map_or(0, |r| r + 1);
    let jobs: Vec<(usize, usize)> =
        (0..cfg.deals).flat_map(|d| (0..lineups[d % lineups.len()].seats.len()).map(move |shift| (d, shift))).collect();
    let next = AtomicUsize::new(0);
    let all = Mutex::new(Run { results: Vec::new(), times: vec![Vec::new(); roles] });
    let start = Instant::now();
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..cfg.threads {
            scope.spawn(|| {
                let mut mine = Run { results: Vec::new(), times: vec![Vec::new(); roles] };
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(d, shift)) = jobs.get(j) else { break };
                    let lineup = &lineups[d % lineups.len()];
                    let n = lineup.seats.len();
                    // la place `seat` de la table reçoit la place `from(seat)` d'avant la rotation
                    let from = |seat: usize| (seat + n - shift) % n;
                    let seats: Vec<&dyn Agent> = (0..n).map(|seat| lineup.seats[from(seat)]).collect();
                    let deal = mix(cfg.seed, d as u64);
                    let table = Table::new(n, deal % 1_000_000_007);
                    let played = play_game(&table, mix(deal, 1) as u32, &seats, mix(deal, 2 + shift as u64));
                    let o = &played.outcome;
                    for (seat, times) in played.times.iter().enumerate() {
                        let role = lineup.roles[from(seat)];
                        mine.results.push(SeatResult {
                            deal: d,
                            players: n,
                            shift,
                            seat,
                            role,
                            win: o.win[seat],
                            coins: o.coins[seat],
                            margin: o.margin(seat),
                            rank: o.rank[seat],
                            decisions: times.len(),
                            seconds: times.iter().map(|&t| t as f64).sum(),
                        });
                        mine.times[role].extend(times);
                    }
                    let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if finished.is_multiple_of(50) || finished == jobs.len() {
                        eprint!("\r{finished}/{} parties, {:.0} s", jobs.len(), start.elapsed().as_secs_f64());
                    }
                }
                let mut all = all.lock().unwrap();
                all.results.extend(mine.results);
                for (a, b) in all.times.iter_mut().zip(mine.times) {
                    a.extend(b);
                }
            });
        }
    });
    eprintln!();
    let mut run = all.into_inner().unwrap();
    run.results.sort_by_key(|r| (r.deal, r.shift, r.seat));
    run
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

/// Écrit le résultat de chaque place de chaque partie, pour `lab/py/score.py` : un en-tête commenté
/// (graine, rôles), puis une ligne par place, séparée par des tabulations.
pub fn write(out: &mut impl Write, cfg: &Config, roles: &[String], run: &Run) -> std::io::Result<()> {
    writeln!(out, "# graine {}, {} donnes", cfg.seed, cfg.deals)?;
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
