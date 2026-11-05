//! Vitesse des parties au hasard :
//! `cargo run --release -p smallmobile-lab14 --example random_speed -- [parties] [fils]`.
//!
//! Chaque partie a sa graine, qui choisit sa carte parmi celles du jeu (`maps`) et mène la partie ;
//! un coup au hasard parmi les coups possibles (`moves::legal_into`), à chaque décision. L'empreinte des
//! résultats (pièces finales et nombre de coups de chaque partie) ne dépend ni des fils ni de la
//! vitesse : elle doit rester la même quand on optimise.

use std::time::Instant;

use smallmobile_kernel::{Phase, maps};
use smallmobile_lab14::know::Knowledge;
use smallmobile_lab14::moves;
use smallmobile_lab14::rand::Rng;
use smallmobile_lab14::table::Table;

/// Une partie au hasard jusqu'au bout ; rend le nombre de coups et l'empreinte du résultat.
fn play(table: &Table, seed: u32, rng: &mut Rng) -> (u64, u64) {
    let mut g = table.game(seed);
    let mut k = Knowledge::default();
    let mut count = 0u64;
    let mut legal = Vec::new();
    while g.state.phase != Phase::GameOver {
        moves::legal_into(&g, &k, None, &mut legal);
        let mv = legal[rng.below(legal.len())].mv;
        k.play(&mut g, &mv).expect("coup possible");
        count += 1;
    }
    let mut h = count;
    for p in g.state.players.iter() {
        h = h.wrapping_mul(0x0100_0000_01b3) ^ p.coins as u64;
    }
    (count, h.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: u64 = args.get(1).map_or(20_000, |s| s.parse().unwrap());
    let threads: u64 = args.get(2).map_or(1, |s| s.parse().unwrap());
    for players in 2..=5 {
        let pool: Vec<Table> = (0..maps::COUNT as u64).map(|i| Table::from_pool(players, i)).collect();
        let start = Instant::now();
        let (moves, print) = std::thread::scope(|sc| {
            let pool = &pool;
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    sc.spawn(move || {
                        let (mut moves, mut print) = (0u64, 0u64);
                        for i in (t..games).step_by(threads as usize) {
                            let mut rng = Rng::new(i);
                            let seed = rng.next_u32();
                            let (n, h) = play(&pool[seed as usize % pool.len()], seed, &mut rng);
                            moves += n;
                            print = print.wrapping_add(h);
                        }
                        (moves, print)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).fold((0u64, 0u64), |a, b| (a.0 + b.0, a.1.wrapping_add(b.1)))
        });
        let wall = start.elapsed().as_secs_f64();
        println!(
            "{players} joueurs, {threads} fil(s) : {:>8.0} parties/s, {:>10.0} coups/s, {:.1} µs/partie ; empreinte {print:016x}",
            games as f64 / wall,
            moves as f64 / wall,
            1e6 * wall * threads as f64 / games as f64,
        );
    }
}
