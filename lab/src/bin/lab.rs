//! Le laboratoire en ligne de commande.
//!
//! ```text
//! lab arena --hero hard --field normal [--players 2,3,4,5] [--deals 100] [--threads N] [--seed 1] [--out r.tsv]
//!     [--stop 3]   (arrêt dès que le score du joueur testé est clairement 0 ou 3 points : test séquentiel)
//! lab arena --table 'az:net=poids.bin,rp=4|hard|normal' [--deals 100] [--out r.tsv]   (table fixe)
//! lab bench --agent random [--players 2] [--games 20]
//! lab speed --agent az:net=poids.bin,n=200 [--positions 12] [--seed 1]   (temps et empreinte des recherches)
//! lab selfplay --agent mcts:n=200 --games 1000 --out lab/data/gen0 [--players 2,3,4,5] [--sample-rounds 2]
//!     [--field 'easy|hard|az:net=gen9.bin,rp=2' --mixed 0.5]   (tables variées)
//! lab check-net --net poids.bin --data parties.bin [--count 5]   (comparaison avec lab/py/check.py)
//! lab label --games 20000 --out lab/data/label1 [--share 0.1] [--rollouts 32] [--random 0.1] [--agent az:net=poids.bin,n=50]
//!     [--judge az:net=poids.bin]   (fins de partie jouées par le réseau plutôt que par l'IA Normal)
//! lab judge --games lab/data/p2 --out lab/data/p2-judged [--share 0.1] [--rollouts 32] [--judge az:net=poids.bin]
//!     (positions de parties déjà jouées, fichiers .moves de lab selfplay)
//! lab reencode --games lab/data/gen10 --data lab/data/gen10-judged --out lab/data/gen10-judged.v4
//!     (positions *.v3.bin passées à l'encodage d'aujourd'hui, *.v4.bin, d'après les parties .moves)
//! lab audit --games lab/data/gen10/c00   (ce que l'entrée du réseau laisse de côté, dans des parties jouées)
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
        "speed" => speed_cmd(&opts),
        "selfplay" => selfplay_cmd(&opts),
        "check-net" => check_net_cmd(&opts),
        "analyze" => analyze_cmd(&opts),
        "agree" => agree_cmd(&opts),
        "trace" => trace_cmd(&opts),
        "label" => label_cmd(&opts),
        "judge" => judge_cmd(&opts),
        "noise" => noise_cmd(&opts),
        "audit" => audit_cmd(&opts),
        "reencode" => reencode_cmd(&opts),
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
    eprintln!("        lab speed --agent az:… [--players 2,3,4,5] [--positions 12] [--seed S]");
    eprintln!("        lab selfplay --agent A --games N --out DIR [--players 2,3,4,5] [--sample-rounds R] [--seed S]");
    eprintln!("            [--full-share 0.25 --fast 50]   (part des recherches complètes, itérations des autres)");
    eprintln!("            [--field 'easy|hard|az:…' --mixed 0.5]   (tables variées : adversaires, part des parties)");
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

/// Des recherches sur des positions fixes (parties de l'IA Normale, `--positions` par nombre de
/// joueurs), chacune avec sa graine : leur temps sur un cœur, et deux empreintes de leurs résultats,
/// les décisions (coup et visites de la racine) et les valeurs (au bit près). Une optimisation qui
/// ne change pas la recherche garde les deux.
fn speed_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab::know::{Knowledge, Returns};
    use smallmobile_lab::moves;
    use smallmobile_lab::rand::{Rng, mix};

    let name = opts.get("agent").unwrap_or_else(|| usage());
    let az = agent::az_named(name).unwrap_or_else(|| {
        eprintln!("recherche guidée par le réseau attendue (az:…) : {name}");
        std::process::exit(2)
    });
    let per: usize = get(opts, "positions", 12);
    let seed: u64 = get(opts, "seed", 1);
    let normal = agent_named("normal");
    let fnv = |h: u64, bytes: &[u8]| bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    let (mut decisions, mut values) = (0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64);
    let (mut count, mut secs) = (0, 0.0);
    for n in players(opts, "2,3,4,5") {
        let table = Table::new(n, mix(seed, n as u64));
        let mut g = table.game(mix(seed, 10 + n as u64) as u32);
        let mut k = Knowledge::default();
        let mut rng = Rng::new(mix(seed, 20 + n as u64));
        // une position sur 7 parmi celles où le joueur a le choix
        let mut positions = Vec::new();
        let mut choices = 0;
        while g.state.phase != smallmobile_kernel::Phase::GameOver && positions.len() < per {
            if moves::legal(&g).len() > 1 {
                if choices % 7 == 3 {
                    positions.push((g, k));
                }
                choices += 1;
            }
            let m = normal.act(&g, &k, &mut rng.fork());
            let mut returns = Returns::default();
            moves::play(&mut g, &m, &mut returns).expect("coup de l'IA Normale");
            k.observe(returns.0, &g.state);
        }
        let start = Instant::now();
        for (i, (g, k)) in positions.iter().enumerate() {
            let s = az.search(g, k, &mut Rng::new(mix(seed, (100 * n + i) as u64)));
            decisions = fnv(decisions, format!("{:?}{:?}", s.mv, s.visits).as_bytes());
            for v in [s.value].iter().chain(&s.finals).chain(&s.wins) {
                values = fnv(values, &v.to_bits().to_le_bytes());
            }
        }
        let t = start.elapsed().as_secs_f64();
        println!(
            "{n} joueurs : {} recherches, {:.1} ms par recherche",
            positions.len(),
            1000.0 * t / positions.len() as f64
        );
        count += positions.len();
        secs += t;
    }
    println!("total : {count} recherches, {:.1} ms par recherche (1 cœur)", 1000.0 * secs / count as f64);
    println!("empreintes : décisions {decisions:016x}, valeurs {values:016x}");
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
    // tables variées : les adversaires possibles, séparés par « | » (leur numéro dans les .moves)
    let others: Vec<Box<dyn Agent>> = opts.get("field").map_or(Vec::new(), |f| f.split('|').map(agent_named).collect());
    for (i, a) in others.iter().enumerate() {
        eprintln!("adversaire {i} : {}", a.name());
    }
    let field = selfplay::Field { agents: others.iter().map(|a| a.as_ref()).collect(), share: get(opts, "mixed", 0.0) };
    let out = opts.get("out").unwrap_or_else(|| usage());
    let start = Instant::now();
    if let Err(e) = selfplay::run_with(&cfg, searcher.as_ref(), &field, std::path::Path::new(out)) {
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
    let size = if path.ends_with(selfplay::SUFFIX) {
        selfplay::RECORD
    } else if path.ends_with(".v3.bin") {
        selfplay::RECORD_V3
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
/// Les parties enregistrées (fichiers `.moves` de `lab selfplay`) du dossier `--games`, et le nombre
/// de fichiers lus.
fn game_logs(opts: &HashMap<String, String>) -> (Vec<smallmobile_lab::selfplay::GameLog>, usize) {
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
    (logs, files.len())
}

/// Ce que l'entrée du réseau (`encode`) laisse de côté, dans des parties déjà jouées : à chaque
/// décision (plusieurs coups possibles), la fréquence de chaque information publique absente de
/// l'entrée, par nombre de joueurs ; puis, dans chaque partie, les positions publiquement
/// différentes qui donnent au réseau exactement la même entrée, et ce qui les distingue.
fn audit_cmd(opts: &HashMap<String, String>) {
    use smallmobile_kernel::State;
    use smallmobile_lab::encode::encode;
    use smallmobile_lab::moves;

    const GAPS: [&str; 8] = [
        "paix entre deux adversaires",
        "peuples connus au fond de la pioche",
        "  dont bientôt au marché (≤ 2 cachés)",
        "pouvoirs connus au fond de la pioche",
        "  dont bientôt au marché (≤ 2 cachés)",
        "plusieurs défenses en attente",
        "peuple actif d'un adversaire choisi il y a 3 manches ou plus",
        "fin de tour avec paix à imposer (clé : place absolue)",
    ];
    // ce qu'un joueur sait : sans l'ordre caché des pioches, les dés à venir, ni le type des places
    fn public(s: &State, race_tail: usize, power_tail: usize) -> State {
        let mut p = *s;
        let hidden = p.race_deck.len() - race_tail.min(p.race_deck.len());
        p.race_deck.as_mut_slice()[..hidden].sort_unstable();
        let hidden = p.power_deck.len() - power_tail.min(p.power_deck.len());
        p.power_deck.as_mut_slice()[..hidden].sort_unstable();
        (p.rng, p.seq, p.last_die, p.last_region) = (0, 0, None, None);
        for q in p.players.as_mut_slice() {
            q.level = smallmobile_kernel::Level::Normal;
        }
        p
    }
    fn differences(a: &State, b: &State) -> Vec<&'static str> {
        let (t, u) = (&a.turn, &b.turn);
        [
            ("joueurs", a.players != b.players),
            ("peuples", a.races != b.races),
            ("régions", a.regions != b.regions),
            ("marché", a.market != b.market),
            ("pioche des peuples", a.race_deck != b.race_deck || a.race_tail != b.race_tail),
            ("pioche des pouvoirs", a.power_deck != b.power_deck || a.power_tail != b.power_tail),
            ("manche, joueur, phase", (a.round, a.current, a.phase) != (b.round, b.current, b.phase)),
            ("défenses", a.defense != b.defense),
            ("paix", a.peace != b.peace),
            ("tour : peuple, étape", (t.army, t.stage) != (u.army, u.stage)),
            (
                "tour : conquêtes",
                (t.conquests, t.non_empty, t.attacked, t.sorcery) != (u.conquests, u.non_empty, u.attacked, u.sorcery),
            ),
            (
                "tour : dés, dragon, forteresse",
                (t.berserk_die, t.die_rolled, t.dragon_used, t.fortress_region)
                    != (u.berserk_die, u.die_rolled, u.dragon_used, u.fortress_region),
            ),
            (
                "tour : choix, déclin, Amazones",
                (t.picked, t.declined, t.amazon_bonus) != (u.picked, u.declined, u.amazon_bonus),
            ),
            ("tour : jetons ramassés", t.readied != u.readied),
        ]
        .into_iter()
        .filter_map(|(name, differ)| differ.then_some(name))
        .collect()
    }

    let (logs, _) = game_logs(opts);
    let mut counts = [[0u64; GAPS.len() + 1]; 6];
    let mut same_input = 0u64;
    let mut kinds: HashMap<String, u64> = HashMap::new();
    for log in &logs {
        let table = log.table();
        // dans une partie : entrée du réseau → état public déjà vu avec cette entrée
        let mut seen: HashMap<Vec<u8>, State> = HashMap::new();
        log.replay(&table, |g, k| {
            if moves::legal(g).len() < 2 {
                return;
            }
            let s = &g.state;
            let n = s.player_count();
            let me = s.actor().expect("un joueur doit agir");
            let soon = |deck: usize, tail: usize| tail > 0 && deck - tail.min(deck) <= 2;
            let old = s
                .players
                .iter()
                .enumerate()
                .any(|(q, p)| q != me && p.active.is_some_and(|race| s.round - s.race(race).picked_round >= 3));
            let peace_keys =
                moves::legal(g).iter().any(|c| matches!(c.mv, moves::Move::Finish(f) if f.peace.is_some()));
            let gaps = [
                s.peace.iter().any(|p| p.diplomat != me && p.target != me),
                k.race_tail > 0,
                soon(s.race_deck.len(), k.race_tail),
                k.power_tail > 0,
                soon(s.power_deck.len(), k.power_tail),
                s.defense.len() > 1,
                old,
                peace_keys,
            ];
            counts[n][0] += 1;
            for (c, gap) in counts[n][1..].iter_mut().zip(gaps) {
                *c += gap as u64;
            }
            let e = encode(g, k.race_tail);
            let mut input = vec![e.regions as u8, e.players as u8];
            e.node[..e.regions].iter().for_each(|row| input.extend_from_slice(row));
            input.extend_from_slice(&e.global);
            let p = public(s, k.race_tail, k.power_tail);
            match seen.get(&input) {
                Some(q) if *q != p => {
                    same_input += 1;
                    *kinds.entry(differences(q, &p).join(", ")).or_default() += 1;
                }
                Some(_) => {}
                None => {
                    seen.insert(input, p);
                }
            }
        })
        .unwrap_or_else(|e| panic!("partie {} : {e}", log.deal));
    }
    let total: u64 = counts.iter().map(|c| c[0]).sum();
    println!("{} parties, {total} décisions", logs.len());
    println!(
        "{:<62} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "décisions où le réseau ne voit pas…", "2 j.", "3 j.", "4 j.", "5 j.", "toutes"
    );
    for (i, gap) in GAPS.iter().enumerate() {
        let share = |c: &[u64; GAPS.len() + 1]| format!("{:.1} %", 100.0 * c[i + 1] as f64 / c[0].max(1) as f64);
        let all: u64 = counts.iter().map(|c| c[i + 1]).sum();
        let cols: Vec<String> = counts[2..6].iter().map(share).collect();
        println!(
            "{gap:<62} {:>8} {:>8} {:>8} {:>8} {:>7.1} %",
            cols[0],
            cols[1],
            cols[2],
            cols[3],
            100.0 * all as f64 / total as f64
        );
    }
    println!("\npositions différentes, même entrée que la précédente (dans une partie) : {same_input}");
    let mut kinds: Vec<_> = kinds.into_iter().collect();
    kinds.sort_by_key(|k| std::cmp::Reverse(k.1));
    for (what, count) in kinds.iter().take(15) {
        println!("{count:>8}  {what}");
    }
}

fn reencode_cmd(opts: &HashMap<String, String>) {
    let (logs, _) = game_logs(opts);
    let data = opts.get("data").unwrap_or_else(|| usage());
    let out = opts.get("out").unwrap_or_else(|| usage());
    let start = Instant::now();
    let threads = get(opts, "threads", threads());
    match smallmobile_lab::reencode::run(&logs, std::path::Path::new(data), std::path::Path::new(out), threads) {
        Ok((files, positions)) => println!(
            "{positions} positions ({files} fichiers, {} parties) réencodées en {:.0} s",
            logs.len(),
            start.elapsed().as_secs_f64()
        ),
        Err(e) => {
            eprintln!("échec : {e}");
            std::process::exit(1)
        }
    }
}

fn judge_cmd(opts: &HashMap<String, String>) {
    let (logs, files) = game_logs(opts);
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
    println!("{} parties ({files} fichiers) rejugées en {:.0} s", logs.len(), start.elapsed().as_secs_f64());
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
