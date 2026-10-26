//! Le programme du labo 8.
//!
//! lab8 record --games N --out DIR [--agent az:net=lab7/nets/c130.bin,n=400] [--players 2,3,4,5]
//!     [--seed S] [--threads N] [--device vulkan] [--raw true]   (parties d'imitation découpées en plans,
//!     `record` ; `--raw` : aussi les entrées pleines, pour vérifier la forme compacte)
//! lab8 arena --hero A --field B [--players 2,3,4,5] [--deals N] [--seed S] [--out F] [--device vulkan]
//!     (l'arène de lab7 ; joueurs de lab7, et `plan:net=…[,t=…]` : le décodeur de lab8 sans recherche)
//! lab8 fixture --out DIR [--games N] [--seed S] [--every K]   (un plan sur K de parties au hasard, pour
//!     comparer le réseau Rust à PyTorch : `lab8/py/check.py`)
//! lab8 check-net --net poids.bin --fixture DIR   (le réseau Rust face aux sorties de PyTorch)
//! lab8 speed --net poids.bin [--positions N] [--seed S]   (encodeur et pas du décodeur, un cœur)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use smallmobile_kernel::Phase;
use smallmobile_lab7::agent::{self, Agent};
use smallmobile_lab7::arena::{self, Config, Lineup};
use smallmobile_lab7::gpu::{Device, Gpu};
use smallmobile_lab7::table::Table;
use smallmobile_lab8::agent::Plan;
use smallmobile_lab8::encode::{GLOBAL, NODE, REGIONS, encode};
use smallmobile_lab8::know::Knowledge;
use smallmobile_lab8::moves;
use smallmobile_lab8::net::Net;
use smallmobile_lab8::plan::Start;
use smallmobile_lab8::rand::{Rng, mix};
use smallmobile_lab8::record;
use smallmobile_lab8::tokens::{distances, dynamics};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { usage() };
    let opts = options(&args[1..]);
    match cmd.as_str() {
        "record" => record_cmd(&opts),
        "arena" => arena_cmd(&opts),
        "fixture" => fixture_cmd(&opts),
        "check-net" => check_net_cmd(&opts),
        "speed" => speed_cmd(&opts),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage : voir l'en-tête de lab8/src/bin/lab.rs");
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

fn players(opts: &HashMap<String, String>) -> Vec<usize> {
    let text = opts.get("players").map_or("2,3,4,5", String::as_str);
    text.split(',').map(|x| x.parse().unwrap_or_else(|_| usage())).collect()
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// `--device vulkan` : le GPU ; rien : le processeur.
fn device(opts: &HashMap<String, String>) -> Option<Device> {
    Device::parse(opts.get("device").map_or("cpu", String::as_str)).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    })
}

/// Un joueur : le décodeur de lab8 (`plan:…`), ou un joueur de lab7.
fn agent_named(name: &str) -> Box<dyn Agent> {
    if name.starts_with("plan:") {
        return Box::new(Plan::named(name).unwrap_or_else(|| usage()));
    }
    agent::by_name(name).unwrap_or_else(|| {
        eprintln!("joueur inconnu : {name}");
        std::process::exit(2)
    })
}

/// Le GPU pour les réseaux de c130 (`net=` des joueurs `az`), s'il est demandé.
fn start_gpu(device: Option<Device>, paths: &[PathBuf]) -> Option<Gpu> {
    let device = device?;
    if paths.is_empty() {
        return None;
    }
    let paths: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let (gpu, name) = Gpu::start_on(device, &paths).unwrap_or_else(|e| {
        eprintln!("GPU : {e}");
        std::process::exit(2)
    });
    eprintln!("réseaux de lab7 évalués par {name}");
    Some(gpu)
}

fn net_path(spec: &str) -> PathBuf {
    let net = spec.split(',').find_map(|p| p.strip_prefix("az:").unwrap_or(p).strip_prefix("net="));
    PathBuf::from(net.unwrap_or_else(|| usage()))
}

fn record_cmd(opts: &HashMap<String, String>) {
    let spec = opts.get("agent").map_or("az:net=lab7/nets/c130.bin,n=400", String::as_str);
    let player = agent_named(spec);
    let Some(az) = player.az() else {
        eprintln!("--agent : un joueur az de lab7");
        std::process::exit(2)
    };
    let cfg = record::Config {
        players: players(opts),
        games: get(opts, "games", 100),
        threads: get(opts, "threads", threads()),
        seed: get(opts, "seed", 81_000_000),
        raw: get(opts, "raw", false),
    };
    let dir = PathBuf::from(opts.get("out").unwrap_or_else(|| usage()));
    let gpu = start_gpu(device(opts), &[net_path(spec)]);
    let start = Instant::now();
    record::run(&cfg, az, &dir, gpu.as_ref()).unwrap_or_else(|e| {
        eprintln!("échec : {e}");
        std::process::exit(1)
    });
    eprintln!("{} parties en {:.0} s", cfg.games, start.elapsed().as_secs_f64());
}

fn arena_cmd(opts: &HashMap<String, String>) {
    let cfg = Config {
        deals: get(opts, "deals", 100),
        threads: get(opts, "threads", threads()),
        seed: get(opts, "seed", 1),
        stop: None,
    };
    let specs: Vec<String> =
        ["hero", "field"].iter().map(|k| opts.get(*k).cloned().unwrap_or_else(|| usage())).collect();
    let agents: Vec<Box<dyn Agent>> = specs.iter().map(|s| agent_named(s)).collect();
    let device = device(opts);
    let (mut paths, mut nets) = (Vec::new(), Vec::new());
    for (a, spec) in agents.iter().zip(&specs) {
        if let Some(az) = a.az().filter(|_| device.is_some()) {
            paths.push(net_path(spec));
            nets.push(&*az.net);
        }
    }
    let gpu = start_gpu(device, &paths);
    let lineups: Vec<Lineup> =
        players(opts).into_iter().map(|n| Lineup::hero(n, agents[0].as_ref(), agents[1].as_ref())).collect();
    let out = opts.get("out").map(|path| {
        std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("échec de la création de {path} : {e}");
            std::process::exit(1)
        })
    });
    let start = Instant::now();
    let run = arena::run(&cfg, &lineups, gpu.as_ref().map(|g| (g, &nets[..])));
    let names: Vec<String> = agents.iter().map(|a| a.name()).collect();
    print!("{}", arena::describe(&arena::report(&run, 0), &names[0], &names[1]));
    println!("temps par décision : {}", arena::describe_times(&run.times[0]));
    println!("durée : {:.1} s", start.elapsed().as_secs_f64());
    if let Some(mut f) = out {
        arena::write(&mut f, &cfg, &names, &run).expect("écriture des résultats");
    }
}

/// Les plans de parties au hasard (`fixture.in`, lu par `lab8/py/check.py`) : par plan, joueurs,
/// régions (u8), node (48 × 77), glob (820), distances (48 × 48), pas (u8), puis par pas le coup
/// précédent (u16, 65535 au premier pas) et ses entrées (u8 × DYN).
fn fixture_cmd(opts: &HashMap<String, String>) {
    use std::io::Write;
    let dir = PathBuf::from(opts.get("out").unwrap_or_else(|| usage()));
    std::fs::create_dir_all(&dir).expect("dossier");
    let games: usize = get(opts, "games", 4);
    let seed: u64 = get(opts, "seed", 8);
    let every: usize = get(opts, "every", 1);
    let mut out = Vec::new();
    let (mut count, mut seen) = (0, 0);
    let mut rng = Rng::new(seed);
    for i in 0..games {
        let n = 2 + i % 4;
        let table = Table::new(n, mix(seed, i as u64) % 1_000_000_007);
        let dist = distances(&table.map);
        let mut g = table.game(i as u32);
        let mut k = Knowledge::default();
        // le plan en cours : son début, et ses octets (gardés un plan sur `every`)
        let mut open: Option<(Start, Vec<u8>)> = None;
        let mut last = u16::MAX;
        while g.state.phase != Phase::GameOver {
            let legal = moves::legal(&g, &k);
            let c = legal[rng.below(legal.len())];
            if legal.len() > 1 {
                let (_, plan) = open.get_or_insert_with(|| {
                    let e = encode(&g, &k);
                    let mut plan = vec![n as u8, e.regions as u8];
                    plan.extend(e.node.iter().flatten());
                    plan.extend_from_slice(&e.global);
                    plan.extend_from_slice(&dist);
                    plan.push(0);
                    last = u16::MAX;
                    (Start::of(&g), plan)
                });
                plan.extend_from_slice(&last.to_le_bytes());
                plan.extend_from_slice(&dynamics(&g, &k));
                plan[STEPS_AT] += 1;
                last = c.key;
            }
            k.play(&mut g, &c.mv).expect("coup possible");
            if let Some((start, plan)) = &open
                && (g.state.phase == Phase::GameOver || !start.goes_on(&c.mv, &g) || plan[STEPS_AT] == 40)
            {
                if seen % every == 0 {
                    out.extend_from_slice(plan);
                    count += 1;
                }
                seen += 1;
                open = None;
            }
        }
    }
    std::fs::File::create(dir.join("fixture.in")).and_then(|mut f| f.write_all(&out)).expect("écriture");
    eprintln!("{count} plans");
}

/// La place du nombre de pas dans un plan de `fixture.in`.
const STEPS_AT: usize = 2 + REGIONS * NODE + GLOBAL + REGIONS * REGIONS;

/// Le réseau Rust face aux sorties de PyTorch (`fixture.out`, écrit par `lab8/py/check.py`) : écarts
/// maximaux des têtes et des logits des coups.
fn check_net_cmd(opts: &HashMap<String, String>) {
    let net = Net::load(Path::new(opts.get("net").unwrap_or_else(|| usage()))).expect("poids");
    let dir = PathBuf::from(opts.get("fixture").unwrap_or_else(|| usage()));
    let (heads, logits, plans) = smallmobile_lab8::fixture::check(&net, &dir).expect("fixture");
    println!("{plans} plans : écart maximal des têtes {heads:.2e}, des logits {logits:.2e}");
}

fn speed_cmd(opts: &HashMap<String, String>) {
    let net = Net::load(Path::new(opts.get("net").unwrap_or_else(|| usage()))).expect("poids");
    let count: usize = get(opts, "positions", 200);
    let mut rng = Rng::new(get(opts, "seed", 5));
    let (mut encodes, mut steps) = (0.0, 0.0);
    let (mut n_enc, mut n_steps) = (0, 0);
    let mut i = 0;
    while n_enc < count {
        let n = 2 + i % 4;
        let table = Table::new(n, mix(77, i as u64) % 1_000_000_007);
        let dist = distances(&table.map);
        let mut g = table.game(i as u32);
        let mut k = Knowledge::default();
        while g.state.phase != Phase::GameOver && n_enc < count {
            let legal = moves::legal(&g, &k);
            if legal.len() > 1 && rng.below(4) == 0 {
                let t = Instant::now();
                let mut w = smallmobile_lab8::plan::Writer::new(&net, &g, &k, &dist);
                encodes += t.elapsed().as_secs_f64();
                n_enc += 1;
                let t = Instant::now();
                for _ in 0..8 {
                    let l = w.logits(&g, &k, &legal);
                    w.played(legal[l.len() / 2].key);
                }
                steps += t.elapsed().as_secs_f64();
                n_steps += 8;
            }
            let c = legal[rng.below(legal.len())];
            k.play(&mut g, &c.mv).expect("coup possible");
        }
        i += 1;
    }
    println!(
        "{} paramètres ; encodeur {:.2} ms par position, pas du décodeur {:.3} ms (un cœur)",
        net.parameters(),
        1e3 * encodes / n_enc as f64,
        1e3 * steps / n_steps as f64
    );
}
