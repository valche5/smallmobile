//! Le laboratoire en ligne de commande.
//!
//! ```text
//! lab arena --hero hard --field normal [--players 2,3,4,5] [--deals 100] [--threads N] [--seed 1] [--out r.tsv]
//!     [--stop 3]   (arrêt dès que le score du joueur testé est clairement 0 ou 3 points : test séquentiel)
//! lab arena --table 'az:net=poids.bin,rp=4|hard|normal' [--deals 100] [--out r.tsv]   (table fixe)
//! lab bench --agent random [--players 2] [--games 20]
//! lab selfplay --agent mcts:n=200 --games 1000 --out lab/data/gen0 [--players 2,3,4,5] [--sample-rounds 2]
//! lab check-net --net poids.bin --data parties.bin [--count 5]   (comparaison avec lab/py/check.py)
//! lab label --games 20000 --out lab/data/label1 [--share 0.1] [--rollouts 32] [--random 0.1] [--agent az:net=poids.bin,n=50]
//!     [--judge az:net=poids.bin]   (fins de partie jouées par le réseau plutôt que par l'IA Normal)
//! lab judge --games lab/data/p2 --out lab/data/p2-judged [--share 0.1] [--rollouts 32] [--judge az:net=poids.bin]
//!     (positions de parties déjà jouées, fichiers .moves de lab selfplay)
//! lab noise --net poids.bin [--positions 2000] [--rollouts 32] [--judge az:…]   (bruit des cibles, erreur du réseau)
//! lab analyze --game partie.smk --agent az:net=poids.bin,n=20000   (partie exportée par lab/tools/export-room.ts)
//! ```

use std::collections::HashMap;
use std::time::Instant;

use smallmobile_lab::agent::{self, Agent};
use smallmobile_lab::arena::{self, Config, Lineup, Sequential, play_game};
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
        "label" => label_cmd(&opts),
        "judge" => judge_cmd(&opts),
        "noise" => noise_cmd(&opts),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!(
        "usage : lab arena --hero A --field B [--players 2,3,4,5] [--deals N] [--threads N] [--seed S] [--out F] \
         [--stop D]"
    );
    eprintln!("        lab arena --table 'A|B|B' [--deals N] [--threads N] [--seed S] [--out F]");
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
    agent_named(opts.get(key).unwrap_or_else(|| usage()))
}

fn agent_named(name: &str) -> Box<dyn Agent> {
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

/// `--hero A --field B` : A face à des B, à chaque nombre de joueurs de `--players` ; `--table 'A|B|B'` :
/// une table fixe, chaque joueur passant à chaque place (les places d'un même joueur sont comptées
/// ensemble). `--out` : le résultat de chaque place de chaque partie, pour `lab/py/score.py`.
fn arena_cmd(opts: &HashMap<String, String>) {
    // --stop D : le joueur testé gagne-t-il D points ou rien ? Regard toutes les 23 donnes par taille
    let stop = opts.get("stop").map(|d| Sequential { role: 0, delta: d.parse().unwrap_or_else(|_| usage()), look: 23 });
    if stop.is_some() && opts.contains_key("table") {
        eprintln!("--stop : seulement avec --hero et --field");
        std::process::exit(2)
    }
    let cfg = Config {
        deals: get(opts, "deals", 100),
        threads: get(opts, "threads", threads()),
        seed: get(opts, "seed", 1),
        stop,
    };
    let (agents, specs): (Vec<Box<dyn Agent>>, Option<Vec<usize>>) = match opts.get("table") {
        Some(table) => {
            let mut distinct: Vec<&str> = Vec::new();
            let mut roles = Vec::new();
            for spec in table.split('|') {
                let role = distinct.iter().position(|&d| d == spec).unwrap_or_else(|| {
                    distinct.push(spec);
                    distinct.len() - 1
                });
                roles.push(role);
            }
            (distinct.into_iter().map(agent_named).collect(), Some(roles))
        }
        None => (vec![agent(opts, "hero"), agent(opts, "field")], None),
    };
    let lineups: Vec<Lineup> = match &specs {
        Some(roles) => {
            vec![Lineup { seats: roles.iter().map(|&r| agents[r].as_ref()).collect(), roles: roles.clone() }]
        }
        None => players(opts, "2,3,4,5")
            .into_iter()
            .map(|n| Lineup::hero(n, agents[0].as_ref(), agents[1].as_ref()))
            .collect(),
    };
    let names: Vec<String> = agents.iter().map(|a| a.name()).collect();
    let start = Instant::now();
    let run = arena::run(&cfg, &lineups);
    // face à un champ homogène, seul le joueur testé compte (le champ en est le miroir)
    let shown = if specs.is_some() { names.len() } else { 1 };
    for (role, name) in names.iter().enumerate().take(shown) {
        let others: Vec<&str> = names.iter().enumerate().filter(|&(r, _)| r != role).map(|(_, n)| n.as_str()).collect();
        print!("{}", arena::describe(&arena::report(&run, role), name, &others.join(" + ")));
        println!("temps par décision : {}", arena::describe_times(&run.times[role]));
    }
    if let Some(stop) = cfg.stop {
        let shown = arena::score(&arena::deal_values(&run.results, 0))
            .map_or(String::new(), |(s, se)| format!(" ; score {s:+.1} ± {:.1} points", 1.96 * se));
        let delta = stop.delta;
        match run.verdict {
            Some(true) => println!(
                "arrêt après {} donnes sur {} : gain de {delta} points plutôt que 0{shown}",
                run.deals, cfg.deals
            ),
            Some(false) => {
                println!("arrêt après {} donnes sur {} : 0 plutôt que {delta} points{shown}", run.deals, cfg.deals)
            }
            None => println!("test séquentiel sans conclusion après {} donnes{shown}", run.deals),
        }
    }
    if let Some(path) = opts.get("out") {
        let written =
            std::fs::File::create(path).and_then(|f| arena::write(&mut std::io::BufWriter::new(f), &cfg, &names, &run));
        if let Err(e) = written {
            eprintln!("échec de l'écriture de {path} : {e}");
            std::process::exit(1)
        }
    }
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
            moves += play_game(&table, i as u32 + 1, &seats, i as u64).moves;
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
    let size = if path.ends_with(".v3.bin") {
        selfplay::RECORD
    } else if path.ends_with(".v2.bin") {
        selfplay::RECORD_V2
    } else {
        selfplay::RECORD_V1
    };
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

fn label_cmd(opts: &HashMap<String, String>) {
    let cfg = smallmobile_lab::label::Config {
        players: players(opts, "2,3,4,5"),
        games: get(opts, "games", 1000),
        threads: get(opts, "threads", threads()),
        seed: get(opts, "seed", 1),
        share: get(opts, "share", 0.1),
        rollouts: get(opts, "rollouts", 32),
        random: get(opts, "random", 0.1),
        judge: judge_option(opts),
    };
    let out = opts.get("out").unwrap_or_else(|| usage());
    let player = opts.contains_key("agent").then(|| agent(opts, "agent"));
    let start = Instant::now();
    if let Err(e) = smallmobile_lab::label::run(&cfg, player.as_deref(), std::path::Path::new(out)) {
        eprintln!("échec : {e}");
        std::process::exit(1)
    }
    println!("{} parties en {:.0} s", cfg.games, start.elapsed().as_secs_f64());
}

/// `--judge az:…` : les fins de partie qui jugent sont jouées par ce réseau ; l'IA Normal sinon.
fn judge_option(opts: &HashMap<String, String>) -> smallmobile_lab::label::Judge {
    use smallmobile_lab::label::Judge;
    match opts.get("judge") {
        None => Judge::Normal,
        Some(name) => Judge::Net(Box::new(agent::az_named(name).unwrap_or_else(|| {
            eprintln!("juge inconnu : {name} (az:net=poids.bin,…)");
            std::process::exit(2)
        }))),
    }
}

/// Juge à nouveau les positions de parties déjà jouées : les fichiers `.moves` de `--games`.
fn judge_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab::selfplay::GameLog;
    fn moves_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{} : {e}", dir.display())) {
            let path = entry.expect("dossier lisible").path();
            if path.is_dir() {
                moves_files(&path, out);
            } else if path.to_string_lossy().ends_with(".moves") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    moves_files(std::path::Path::new(opts.get("games").unwrap_or_else(|| usage())), &mut files);
    files.sort();
    let mut logs = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{} : {e}", path.display()));
        logs.extend(
            text.lines().map(|l| GameLog::parse(l).unwrap_or_else(|| panic!("ligne illisible : {}", path.display()))),
        );
    }
    let out = opts.get("out").unwrap_or_else(|| usage());
    let start = Instant::now();
    let judged = smallmobile_lab::label::rejudge(
        &logs,
        get(opts, "share", 0.1),
        get(opts, "rollouts", 32),
        &judge_option(opts),
        get(opts, "seed", 1),
        get(opts, "threads", threads()),
        std::path::Path::new(out),
    );
    if let Err(e) = judged {
        eprintln!("échec : {e}");
        std::process::exit(1)
    }
    println!("{} parties ({} fichiers) rejugées en {:.0} s", logs.len(), files.len(), start.elapsed().as_secs_f64());
}

/// Bruit des positions jugées et erreur du réseau : chaque position (parties de l'IA Normal, 10 % de
/// coups au hasard) est jugée deux fois par `rollouts` fins simulées indépendantes. L'écart entre les
/// deux jugements donne le bruit ; le réseau est comparé à leur moyenne, bruit retiré. Par nombre de
/// joueurs et par manches restantes, avec la dispersion des cibles (ce qu'il y a à expliquer).
fn noise_cmd(opts: &HashMap<String, String>) {
    use smallmobile_kernel::{Level, Phase};
    use smallmobile_lab::agent::KernelAi;
    use smallmobile_lab::encode::encode;
    use smallmobile_lab::know::{Knowledge, Returns};
    use smallmobile_lab::label::judge;
    use smallmobile_lab::moves;
    use smallmobile_lab::rand::{Rng, mix};
    let net = smallmobile_lab::net::Net::load(std::path::Path::new(opts.get("net").unwrap_or_else(|| usage())))
        .unwrap_or_else(|e| panic!("réseau illisible : {e}"));
    let positions: usize = get(opts, "positions", 2000);
    let rollouts: usize = get(opts, "rollouts", 32);
    let threads: usize = get(opts, "threads", threads());
    let seed: u64 = get(opts, "seed", 9);
    let by = judge_option(opts);
    /// Une position : joueurs, manches restantes, puis pour l'avance du joueur qui agit sur son
    /// meilleur rival : jugement A, jugement B, réseau.
    struct Sample {
        players: usize,
        left: i32,
        lead: [f64; 3],
        coins: Vec<[f64; 3]>,
    }
    let samples = std::sync::Mutex::new(Vec::<Sample>::new());
    std::thread::scope(|scope| {
        for t in 0..threads {
            let (net, samples, by) = (&net, &samples, &by);
            scope.spawn(move || {
                let normal = KernelAi::new(Level::Normal);
                let mut rng = Rng::new(mix(seed, t as u64));
                let mut local = Vec::new();
                let mut game = 0;
                while local.len() < positions / threads {
                    game += 1;
                    let table = Table::new(2 + (t + game) % 4, mix(seed, (t * 100_000 + game) as u64) % 1_000_000_007);
                    let mut g = table.game(game as u32);
                    let mut k = Knowledge::default();
                    while g.state.phase != Phase::GameOver && local.len() < positions / threads {
                        let legal = moves::legal(&g);
                        if legal.len() > 1 && rng.unit() < 0.05 {
                            let (_, fa) = judge(&g, &k, rollouts, by, &mut rng);
                            let (_, fb) = judge(&g, &k, rollouts, by, &mut rng);
                            let s = &g.state;
                            let n = s.player_count();
                            let me = s.actor().expect("un joueur doit agir");
                            let ev = net.eval(&encode(&g, k.race_tail), g.map);
                            let mut pred = [0.0f32; 5];
                            for r in 0..n {
                                let q = (me + r) % n;
                                pred[q] = s.players[q].coins as f32 + ev.gain[r];
                            }
                            let lead = |f: &[f32; 5]| {
                                let rival = (0..n).filter(|&q| q != me).map(|q| f[q]).fold(f32::MIN, f32::max);
                                (f[me] - rival) as f64
                            };
                            local.push(Sample {
                                players: n,
                                left: s.max_rounds - s.round,
                                lead: [lead(&fa), lead(&fb), lead(&pred)],
                                coins: (0..n).map(|q| [fa[q] as f64, fb[q] as f64, pred[q] as f64]).collect(),
                            });
                        }
                        let mv = if rng.unit() < 0.1 {
                            legal[rng.below(legal.len())].mv
                        } else {
                            normal.act(&g, &k, &mut rng)
                        };
                        let mut ret = Returns::default();
                        moves::play(&mut g, &mv, &mut ret).expect("coup possible");
                        k.observe(ret.0, &g.state);
                    }
                }
                samples.lock().unwrap().extend(local);
            });
        }
    });
    let samples = samples.into_inner().unwrap();
    // pour des triplets (A, B, réseau) : dispersion de la cible, bruit d'un jugement, erreur du réseau
    // (bruit de la moyenne de deux jugements retiré)
    fn stats<'a>(cells: impl Iterator<Item = &'a [f64; 3]>) -> (usize, f64, f64, f64) {
        let v: Vec<&[f64; 3]> = cells.collect();
        let m = v.len().max(1) as f64;
        let mean = v.iter().map(|c| (c[0] + c[1]) / 2.0).sum::<f64>() / m;
        let d2 = v.iter().map(|c| (c[0] - c[1]).powi(2)).sum::<f64>() / m;
        let spread =
            (v.iter().map(|c| ((c[0] + c[1]) / 2.0 - mean).powi(2)).sum::<f64>() / m - d2 / 4.0).max(0.0).sqrt();
        let e2 = v.iter().map(|c| (c[2] - (c[0] + c[1]) / 2.0).powi(2)).sum::<f64>() / m;
        (v.len(), spread, (d2 / 2.0).sqrt(), (e2 - d2 / 4.0).max(0.0).sqrt())
    }
    println!(
        "{} positions, {rollouts} fins par jugement ; avance du joueur qui agit sur son meilleur rival :",
        samples.len()
    );
    println!("{:<22} {:>6} {:>10} {:>8} {:>8}", "", "pos.", "dispersion", "bruit", "réseau");
    let row = |name: &str, f: &dyn Fn(&Sample) -> bool| {
        let (c, sp, no, er) = stats(samples.iter().filter(|s| f(s)).map(|s| &s.lead));
        println!("{name:<22} {c:>6} {sp:>10.2} {no:>8.2} {er:>8.2}");
    };
    row("toutes", &|_| true);
    for n in 2..=5 {
        row(&format!("{n} joueurs"), &|s| s.players == n);
    }
    for (lo, hi) in [(0, 0), (1, 2), (3, 5), (6, 20)] {
        row(&format!("{lo} à {hi} manches restantes"), &|s| s.left >= lo && s.left <= hi);
    }
    let (c, sp, no, er) = stats(samples.iter().flat_map(|s| &s.coins));
    println!("pièces finales, par joueur : {c} ; dispersion {sp:.2}, bruit {no:.2}, réseau {er:.2}");
}
