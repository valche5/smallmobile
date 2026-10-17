//! Le laboratoire en ligne de commande.
//!
//! ```text
//! lab arena --hero hard --field normal [--players 2,3,4,5] [--deals 100] [--threads N] [--seed 1]
//! lab bench --agent random [--players 2] [--games 20]
//! lab selfplay --agent mcts:n=200 --games 1000 --out lab/data/gen0 [--players 2,3,4,5] [--sample-rounds 2]
//! lab check-net --net poids.bin --data parties.bin [--count 5]   (comparaison avec lab/py/check.py)
//! lab analyze --game partie.smk --agent az:net=poids.bin,n=20000   (partie exportée par lab/tools/export-room.ts)
//! ```

use std::collections::HashMap;
use std::time::Instant;

use smallmobile_lab::agent::{self, Agent};
use smallmobile_lab::arena::{self, Config, play_game};
use smallmobile_lab::selfplay;
use smallmobile_lab::table::Table;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { usage() };
    let opts = options(&args[1..]);
    match cmd.as_str() {
        "arena" => arena_cmd(&opts),
        "bench" => bench_cmd(&opts),
        "selfplay" => selfplay_cmd(&opts),
        "check-net" => check_net_cmd(&opts),
        "analyze" => analyze_cmd(&opts),
        "agree" => agree_cmd(&opts),
        "trace" => trace_cmd(&opts),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage : lab arena --hero A --field B [--players 2,3,4,5] [--deals N] [--threads N] [--seed S]");
    eprintln!("        lab bench --agent A [--players N] [--games N]");
    eprintln!("        lab selfplay --agent A --games N --out DIR [--players 2,3,4,5] [--sample-rounds R] [--seed S]");
    eprintln!("            [--full-share 0.25 --fast 50]   (part des recherches complètes, itérations des autres)");
    eprintln!("        lab analyze --game partie.smk --agent az:net=poids.bin,n=20000");
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

fn selfplay_cmd(opts: &HashMap<String, String>) {
    let name = opts.get("agent").unwrap_or_else(|| usage());
    let searcher = agent::searcher_by_name(name).unwrap_or_else(|| {
        eprintln!("IA de recherche inconnue : {name}");
        std::process::exit(2)
    });
    let cfg = selfplay::Config {
        players: players(opts, "2,3,4,5"),
        games: get(opts, "games", 100),
        threads: get(opts, "threads", threads()),
        seed: get(opts, "seed", 1),
        opts: selfplay::Options {
            sample_rounds: get(opts, "sample-rounds", 2),
            full_share: get(opts, "full-share", 1.0),
            fast: get(opts, "fast", 50),
        },
    };
    let out = opts.get("out").unwrap_or_else(|| usage());
    let start = Instant::now();
    if let Err(e) = selfplay::run(&cfg, searcher.as_ref(), std::path::Path::new(out)) {
        eprintln!("échec : {e}");
        std::process::exit(1)
    }
    println!("{} parties en {:.0} s", cfg.games, start.elapsed().as_secs_f64());
}

/// Évalue les premières positions d'un fichier de parties : mêmes nombres attendus qu'en Python.
fn check_net_cmd(opts: &HashMap<String, String>) {
    let net = smallmobile_lab::net::Net::load(std::path::Path::new(opts.get("net").unwrap_or_else(|| usage())))
        .unwrap_or_else(|e| panic!("réseau illisible : {e}"));
    let data = std::fs::read(opts.get("data").unwrap_or_else(|| usage())).expect("fichier de parties");
    let count: usize = get(opts, "count", 5);
    let start = Instant::now();
    let mut evals = 0;
    let path = opts.get("data").unwrap();
    let size = if path.ends_with(".v2.bin") { selfplay::RECORD } else { selfplay::RECORD_V1 };
    for rec in data.chunks_exact(size).take(count) {
        let (e, adj) = selfplay::read_record(rec);
        let ev = net.eval_with(&e, |i| &adj[i]);
        evals += 1;
        let legal: Vec<String> = (0..smallmobile_lab::moves::KEYS)
            .filter(|&k| ev.logits[k].is_finite())
            .take(8)
            .map(|k| format!("{k}:{:.4}", ev.logits[k]))
            .collect();
        println!("win {:?} gain {:?} logits {}", ev.win, ev.gain, legal.join(" "));
    }
    eprintln!("{:.1} µs par évaluation", 1e6 * start.elapsed().as_secs_f64() / evals.max(1) as f64);
}

/// Le meilleur coup d'une vraie partie, avec ce que la recherche pense de chaque coup.
fn analyze_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab::moves;
    let words: Vec<u32> = std::fs::read(opts.get("game").unwrap_or_else(|| usage()))
        .expect("fichier de partie")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (map, state) =
        smallmobile_kernel::wire::decode_game(&words).unwrap_or_else(|e| panic!("partie illisible : {e:?}"));
    let g = smallmobile_kernel::Game::new(&map, state);
    let Some(me) = g.state.actor() else {
        println!("La partie est finie.");
        return;
    };
    let name = opts.get("agent").unwrap_or_else(|| usage());
    let Some(az) = name.strip_prefix("az:").and(agent::searcher_by_name(name)) else {
        eprintln!("l'analyse demande une IA az:…");
        std::process::exit(2)
    };
    // l'ordre des cartes rendues sous la pioche n'est pas sauvegardé : on le suppose inconnu
    let k = smallmobile_lab::know::Knowledge::default();
    let mut rng = smallmobile_lab::rand::Rng::new(get(opts, "seed", 1));
    let start = Instant::now();
    let s = az.search(&g, &k, &mut rng, az.iterations(), false);
    let legal = moves::legal(&g);
    println!(
        "Joueur {} à jouer, manche {}/{} ({:.0} s de recherche)",
        me + 1,
        g.state.round,
        g.state.max_rounds,
        start.elapsed().as_secs_f64()
    );
    println!("Chance de victoire estimée : {:.1} %", 100.0 * s.value);
    let mut rows: Vec<_> =
        s.details.iter().zip(&s.visits).map(|(&(key, prior, q), &(_, n))| (n, key, prior, q)).collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    println!("{:>8}  {:>7}  {:>8}  coup", "visites", "victoire", "a priori");
    for (n, key, prior, q) in rows.iter().take(get(opts, "top", 12)) {
        let mv = legal.iter().find(|c| c.key == *key).map(|c| moves::describe(&g, &c.mv)).unwrap_or_default();
        println!("{n:>8}  {:>7.1} %  {:>7.1} %  {mv}", 100.0 * q, 100.0 * prior);
    }
}

/// Deux recherches indépendantes sur les mêmes positions choisissent-elles le même coup ?
/// (positions prises dans des parties de l'IA Normal, là où il y a plus d'un coup possible)
fn agree_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab::{know::Knowledge, moves, rand::Rng};
    let name = opts.get("agent").unwrap_or_else(|| usage());
    let searcher = agent::searcher_by_name(name).unwrap_or_else(|| usage());
    let positions: usize = get(opts, "positions", 60);
    let normal = agent::by_name("normal").unwrap();
    let mut rng = Rng::new(get(opts, "seed", 5));
    let (mut same, mut total) = (0, 0);
    let mut game = 0u64;
    while total < positions {
        game += 1;
        let table = Table::new(2 + game as usize % 4, 7000 + game);
        let mut g = table.game(game as u32);
        let k = Knowledge::default();
        while g.state.phase != smallmobile_kernel::Phase::GameOver && total < positions {
            let legal = moves::legal(&g);
            if legal.len() > 1 && rng.below(8) == 0 {
                let n = searcher.iterations();
                let a = searcher.search(&g, &k, &mut rng.fork(), n, false).mv;
                let b = searcher.search(&g, &k, &mut rng.fork(), n, false).mv;
                same += usize::from(a == b);
                total += 1;
            }
            let m = normal.act(&g, &k, &mut rng);
            moves::play(&mut g, &m, &mut smallmobile_kernel::NoJournal).unwrap();
        }
    }
    println!("{}: même coup {same}/{total} ({:.0} %)", searcher.name(), 100.0 * same as f64 / total as f64);
}

/// Une partie du joueur testé (place 0) contre le reste de la table, coup par coup.
fn trace_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab::{
        know::{Knowledge, Returns},
        moves,
        rand::Rng,
    };
    let hero = agent(opts, "hero");
    let field = agent(opts, "field");
    let n: usize = get(opts, "players", 2);
    let table = Table::new(n, get(opts, "map", 3));
    let mut g = table.game(get(opts, "seed", 1));
    let mut k = Knowledge::default();
    let mut rng = Rng::new(1);
    while g.state.phase != smallmobile_kernel::Phase::GameOver {
        let pid = g.state.actor().unwrap();
        let a: &dyn Agent = if pid == 0 { hero.as_ref() } else { field.as_ref() };
        let m = a.act(&g, &k, &mut rng.fork());
        let coins: Vec<i32> = g.state.players.iter().map(|p| p.coins).collect();
        println!("m{} j{} {:?} {:?} : {}", g.state.round, pid + 1, g.state.phase, coins, moves::describe(&g, &m));
        let mut ret = Returns::default();
        moves::play(&mut g, &m, &mut ret).unwrap();
        k.observe(ret.0, &g.state);
    }
    let coins: Vec<i32> = g.state.players.iter().map(|p| p.coins).collect();
    println!("fin : {coins:?}");
}
