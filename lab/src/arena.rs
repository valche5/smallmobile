//! Comparer deux joueurs sur beaucoup de parties.
//!
//! Format « duplicate » : chaque donne (nombre de joueurs, carte, pioches, dés) est jouée autant de
//! fois qu'il y a de places, le joueur testé (`hero`) occupant chaque place à tour de rôle face au
//! reste de la table (`field`). La chance de la donne et de la place s'annule ainsi en grande partie.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use smallmobile_kernel::Phase;

use crate::agent::Agent;
use crate::know::{Knowledge, Returns};
use crate::moves;
use crate::rand::{Rng, mix};
use crate::table::{Outcome, Table};

/// Garde-fou : aucune vraie partie n'approche ce nombre de coups.
const MAX_MOVES: usize = 20_000;

/// Joue une partie jusqu'au bout ; renvoie le résultat et le nombre de coups joués.
pub fn play_game(table: &Table, seed: u32, seats: &[&dyn Agent], rng_seed: u64) -> (Outcome, usize) {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut rng = Rng::new(rng_seed);
    let mut count = 0;
    while g.state.phase != Phase::GameOver {
        assert!(count < MAX_MOVES, "partie sans fin");
        let pid = g.state.actor().expect("un joueur doit agir");
        let m = seats[pid].act(&g, &k, &mut rng.fork());
        let mut returns = Returns::default();
        if let Err(e) = moves::play(&mut g, &m, &mut returns) {
            panic!("coup refusé ({}) : {m:?} → {e:?}", seats[pid].name());
        }
        k.observe(returns.0, &g.state);
        count += 1;
    }
    (Outcome::of(&g), count)
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
    fn add(&mut self, o: &Outcome, seat: usize) {
        self.games += 1;
        self.win += o.win[seat];
        self.win2 += o.win[seat] * o.win[seat];
        self.margin += o.margin(seat) as f64;
        self.rank += o.rank[seat] as f64;
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

    /// Demi-largeur de l'intervalle de confiance à 95 % du taux de victoire.
    pub fn ci95(&self) -> f64 {
        let n = self.games.max(2) as f64;
        let mean = self.win / n;
        let var = (self.win2 / n - mean * mean).max(0.0) * n / (n - 1.0);
        1.96 * (var / n).sqrt()
    }
}

pub struct Config {
    /// Nombres de joueurs, joués à tour de rôle d'une donne à l'autre.
    pub players: Vec<usize>,
    pub deals: usize,
    pub threads: usize,
    pub seed: u64,
}

/// Résultats du joueur testé, par nombre de joueurs (indice = nombre de joueurs).
pub type Report = [Tally; 6];

pub fn run(cfg: &Config, hero: &dyn Agent, field: &dyn Agent) -> Report {
    let jobs: Vec<(usize, usize)> = (0..cfg.deals)
        .flat_map(|d| {
            let n = cfg.players[d % cfg.players.len()];
            (0..n).map(move |seat| (d, seat))
        })
        .collect();
    let next = AtomicUsize::new(0);
    let report = Mutex::new(Report::default());
    let start = Instant::now();
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..cfg.threads {
            scope.spawn(|| {
                let mut mine = Report::default();
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(d, seat)) = jobs.get(j) else { break };
                    let n = cfg.players[d % cfg.players.len()];
                    let deal = mix(cfg.seed, d as u64);
                    let table = Table::new(n, deal % 1_000_000_007);
                    let mut seats: Vec<&dyn Agent> = vec![field; n];
                    seats[seat] = hero;
                    let (o, _) = play_game(&table, mix(deal, 1) as u32, &seats, mix(deal, 2 + seat as u64));
                    mine[n].add(&o, seat);
                    let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if finished.is_multiple_of(50) || finished == jobs.len() {
                        eprint!("\r{finished}/{} parties, {:.0} s", jobs.len(), start.elapsed().as_secs_f64());
                    }
                }
                let mut all = report.lock().unwrap();
                for (a, b) in all.iter_mut().zip(mine.iter()) {
                    a.merge(b);
                }
            });
        }
    });
    eprintln!();
    report.into_inner().unwrap()
}

/// Le tableau des résultats, lisible.
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
