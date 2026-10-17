//! Le laboratoire en ligne de commande.
//!
//! ```text
//! lab arena --hero hard --field normal [--players 2,3,4,5] [--deals 100] [--threads N] [--seed 1]
//! lab bench --agent random [--players 2] [--games 20]
//! ```

use std::collections::HashMap;
use std::time::Instant;

use smallmobile_lab::agent::{self, Agent};
use smallmobile_lab::arena::{self, Config, play_game};
use smallmobile_lab::table::Table;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { usage() };
    let opts = options(&args[1..]);
    match cmd.as_str() {
        "arena" => arena_cmd(&opts),
        "bench" => bench_cmd(&opts),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage : lab arena --hero A --field B [--players 2,3,4,5] [--deals N] [--threads N] [--seed S]");
    eprintln!("        lab bench --agent A [--players N] [--games N]");
    std::process::exit(2)
}

fn options(args: &[String]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut it = args.iter();
    while let Some(k) = it.next() {
        let Some(k) = k.strip_prefix("--") else { usage() };
        let Some(v) = it.next() else { usage() };
        out.insert(k.to_string(), v.clone());
    }
    out
}

fn get<T: std::str::FromStr>(opts: &HashMap<String, String>, key: &str, default: T) -> T {
    opts.get(key).map(|v| v.parse().unwrap_or_else(|_| usage())).unwrap_or(default)
}

fn agent(opts: &HashMap<String, String>, key: &str) -> Box<dyn Agent> {
    let name = opts.get(key).unwrap_or_else(|| usage());
    agent::by_name(name).unwrap_or_else(|| {
        eprintln!("joueur inconnu : {name}");
        std::process::exit(2)
    })
}

fn players(opts: &HashMap<String, String>, default: &str) -> Vec<usize> {
    let text = opts.get("players").map(String::as_str).unwrap_or(default);
    text.split(',').map(|x| x.parse().unwrap_or_else(|_| usage())).collect()
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

fn arena_cmd(opts: &HashMap<String, String>) {
    let hero = agent(opts, "hero");
    let field = agent(opts, "field");
    let cfg = Config {
        players: players(opts, "2,3,4,5"),
        deals: get(opts, "deals", 100),
        threads: get(opts, "threads", threads()),
        seed: get(opts, "seed", 1),
    };
    let start = Instant::now();
    let report = arena::run(&cfg, hero.as_ref(), field.as_ref());
    print!("{}", arena::describe(&report, &hero.name(), &field.name()));
    println!("durée : {:.1} s", start.elapsed().as_secs_f64());
}

fn bench_cmd(opts: &HashMap<String, String>) {
    let a = agent(opts, "agent");
    let games: usize = get(opts, "games", 20);
    for n in players(opts, "2,3,4,5") {
        let start = Instant::now();
        let mut moves = 0;
        for i in 0..games {
            let table = Table::new(n, 1000 + i as u64);
            let seats: Vec<&dyn Agent> = vec![a.as_ref(); n];
            moves += play_game(&table, i as u32 + 1, &seats, i as u64).1;
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{} joueurs : {:.2} ms par partie, {:.0} coups par partie, {:.1} µs par coup (1 cœur)",
            n,
            1000.0 * secs / games as f64,
            moves as f64 / games as f64,
            1e6 * secs / moves as f64
        );
    }
}
