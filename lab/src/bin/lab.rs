//! Le laboratoire en ligne de commande.
//!
//! ```text
//! lab arena --hero hard --field normal [--players 2,3,4,5] [--deals 100] [--threads N] [--seed 1] [--out r.tsv]
//!     [--stop 3]   (arrêt dès que le score du joueur testé est clairement 0 ou 3 points : test séquentiel)
//! lab arena --table 'az:net=poids.bin|hard|normal' [--deals 100] [--out r.tsv]   (table fixe)
//! lab speed --agent az:net=poids.bin,n=200 [--positions 12] [--seed 1]   (temps et empreinte des recherches)
//! lab selfplay --agent az:net=poids.bin,noise=0.25 --games 1000 --out lab/data/gen13 [--players 2,3,4,5]
//!     [--sample-rounds 2] [--full-share 0.5 --fast 50]
//!     [--field 'easy|normal:aggressive|az:net=gen11.bin,n=20' --mixed 0.5 [--observe 1]]   (tables variées ;
//!     --observe : coups des autres joueurs enregistrés aussi, pour la tête « coup réel »)
//! lab check-net --net poids.bin --data parties.v6.bin [--count 5]   (comparaison avec lab/py/check.py)
//! lab judge --games lab/data/gen13 --out lab/data/gen13-judged --judge az:net=poids.bin [--share 0.1] [--rollouts 32]
//!     (positions de parties déjà jouées, fichiers .moves de lab selfplay, rejugées par le réseau)
//! lab analyze --game partie.smk --agent az:net=poids.bin,n=20000   (partie exportée par lab/tools/export-room.ts)
//! ```

use std::collections::HashMap;
use std::time::Instant;

use smallmobile_lab::agent::{self, Agent};
use smallmobile_lab::arena::{self, Config, Lineup, Sequential};
use smallmobile_lab::selfplay;
use smallmobile_lab::table::Table;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { usage() };
    let opts = options(&args[1..]);
    match cmd.as_str() {
        "arena" => arena_cmd(&opts),
        "speed" => speed_cmd(&opts),
        "selfplay" => selfplay_cmd(&opts),
        "check-net" => check_net_cmd(&opts),
        "analyze" => analyze_cmd(&opts),
        "judge" => judge_cmd(&opts),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!(
        "usage : lab arena --hero A --field B [--players 2,3,4,5] [--deals N] [--threads N] [--seed S] [--out F] \
         [--stop D]"
    );
    eprintln!("        lab arena --table 'A|B|B' [--deals N] [--threads N] [--seed S] [--out F]");
    eprintln!("        lab speed --agent az:… [--players 2,3,4,5] [--positions 12] [--seed S]");
    eprintln!("        lab selfplay --agent az:… --games N --out DIR [--players 2,3,4,5] [--sample-rounds R] [--seed S]");
    eprintln!("            [--full-share 0.25 --fast 50]   (part des recherches complètes, itérations des autres)");
    eprintln!("            [--field 'easy|hard|az:…' --mixed 0.5]   (tables variées : adversaires, part des parties)");
    eprintln!("        lab check-net --net poids.bin --data parties.v6.bin [--count N]");
    eprintln!("        lab judge --games DIR --out DIR --judge az:… [--share 0.1] [--rollouts 32] [--seed S]");
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

/// Des recherches sur des positions fixes (parties de l'IA Normale, `--positions` par nombre de
/// joueurs), chacune avec sa graine : leur temps sur un cœur, et deux empreintes de leurs résultats,
/// les décisions (coup et visites de la racine) et les valeurs (au bit près). Une optimisation qui
/// ne change pas la recherche garde les deux.
fn speed_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab::know::Knowledge;
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
            k.play(&mut g, &m).expect("coup de l'IA Normale");
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
    let az = agent::az_named(name).unwrap_or_else(|| {
        eprintln!("recherche guidée par le réseau attendue (az:…) : {name}");
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
            observe: get(opts, "observe", 0) == 1,
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
    if let Err(e) = selfplay::run(&cfg, &az, &field, std::path::Path::new(out)) {
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
    for rec in data.as_chunks::<{ selfplay::RECORD }>().0.iter().take(count) {
        let (e, adj) = selfplay::read_record(rec);
        let ev = net.eval_with(&e, |i| &adj[i]);
        evals += 1;
        let legal: Vec<String> = (0..smallmobile_lab::moves::KEYS)
            .filter(|&k| ev.logits[k].is_finite())
            .take(8)
            .map(|k| format!("{k}:{:.4}", ev.logits[k]))
            .collect();
        let behaviour = ev.behaviour.as_ref().map_or(String::new(), |b| {
            let keys = (0..smallmobile_lab::moves::KEYS).filter(|&k| b[k].is_finite()).take(8);
            format!(" behaviour {}", keys.map(|k| format!("{k}:{:.4}", b[k])).collect::<Vec<_>>().join(" "))
        });
        println!("win {:?} gain {:?} logits {}{behaviour}", ev.win, ev.gain, legal.join(" "));
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
    let Some(az) = agent::az_named(name) else {
        eprintln!("l'analyse demande une IA az:…");
        std::process::exit(2)
    };
    // l'ordre des cartes rendues sous la pioche n'est pas sauvegardé : on le suppose inconnu
    let k = smallmobile_lab::know::Knowledge::default();
    let mut rng = smallmobile_lab::rand::Rng::new(get(opts, "seed", 1));
    let start = Instant::now();
    let s = az.search(&g, &k, &mut rng);
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

fn judge_cmd(opts: &HashMap<String, String>) {
    let (logs, files) = game_logs(opts);
    let out = opts.get("out").unwrap_or_else(|| usage());
    let start = Instant::now();
    let name = opts.get("judge").unwrap_or_else(|| usage());
    let az = agent::az_named(name).unwrap_or_else(|| {
        eprintln!("juge inconnu : {name} (az:net=poids.bin)");
        std::process::exit(2)
    });
    let judged = smallmobile_lab::judge::rejudge(
        &logs,
        get(opts, "share", 0.1),
        get(opts, "rollouts", 32),
        &az,
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
