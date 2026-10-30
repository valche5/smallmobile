//! Le laboratoire en ligne de commande.
//!
//! ```text
//! lab14 arena --hero hard --field normal [--players 2,3,4,5] [--deals 100] [--threads N] [--seed 1] [--out r.tsv]
//!     [--stop 3]   (arrêt dès que le score du joueur testé est clairement 0 ou 3 points : test séquentiel)
//!     [--device vulkan]   (les joueurs `az` sur GPU, un par réseau, plusieurs parties par fil)
//! lab14 arena --table 'az:net=poids.bin|hard|normal' [--deals 100] [--out r.tsv]   (table fixe)
//! lab14 speed --agent az:net=poids.bin,n=200 [--positions 12] [--seed 1]   (temps et empreinte des recherches)
//!     (réglages de `az` : `n`, `win`, `roll`, `mix`, `market=late`, `crn`, `explore`, `history` ; `search::Config`)
//! lab14 selfplay --agent az:net=poids.bin --games 1000 --out lab14/data/gen1 [--players 2,3,4,5] [--pos true]
//!     [--device cpu|vulkan|cuda]   (CPU par défaut ; Vulkan inclus à la compilation, `gpu` est un alias de `vulkan`)
//!     [--noisy-rounds R] [--fast-roll 1] [--luck 1]   (bruit de Gumbel pour jouer seulement aux R premières
//!     manches ; fins simulées aussi dans les recherches rapides ; part du hasard de chaque résultat, `luck`)
//!     (auto-jeu, recette `selfplay::RECIPE`)
//! lab14 check-net --net poids.bin --data parties.pos [--count 5]   (comparaison avec lab14/py/check.py)
//! lab14 bench-net --net poids.bin --data parties.pos [--count 5000]   (vitesse et écarts du calcul bf16)
//! lab14 bench-vulkan --net poids.bin [--regions 40] [--count N] [--rounds 20] [--stream LOTS]   (durée d'un lot sur
//!     GPU selon sa taille, ou de `rounds` lots de N positions : une capture RGP, `MESA_VK_TRACE=rgp` ; en flux :
//!     LOTS lots soumis à la suite ; Vulkan inclus à la compilation par défaut)
//! lab14 check-gpu --net poids.bin --device vulkan|cuda   (le GPU comparé au processeur, positions de l'IA Normale)
//! lab14 bench-gpu --net poids.bin --device vulkan|cuda [--seconds 3]   (débit du moteur seul selon la taille
//!     des lots, deux lots en vol ; `LAB14_CUDA_PHASES=1` : la durée de chaque étape d'un calcul CUDA)
//! lab14 analyze --game partie.smk --agent az:net=poids.bin,n=20000   (partie exportée par labtools/export-room.ts)
//! lab14 consistency --agent az:… --games DIR [--positions 1500] [--every 61] [--repeat 4] [--explore 1]
//!     [--seed S] [--threads N] [--out F]   (la même recherche répétée sur des positions de parties
//!     enregistrées : le plafond de ses cibles, lab14/lab.md D5)
//! ```

use std::collections::HashMap;
use std::time::Instant;

use smallmobile_lab14::agent::{self, Agent};
use smallmobile_lab14::arena::{self, Config, Lineup, Sequential};
use smallmobile_lab14::gpu::{Device, Gpu};
use smallmobile_lab14::selfplay;
use smallmobile_lab14::table::Table;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { usage() };
    let opts = options(&args[1..]);
    match cmd.as_str() {
        "arena" => arena_cmd(&opts),
        "speed" => speed_cmd(&opts),
        "selfplay" => selfplay_cmd(&opts),
        "check-net" => check_net_cmd(&opts),
        "bench-net" => bench_net_cmd(&opts),
        "bench-vulkan" => bench_vulkan_cmd(&opts),
        "check-gpu" => check_gpu_cmd(&opts),
        "bench-gpu" => bench_gpu_cmd(&opts),
        "analyze" => analyze_cmd(&opts),
        "consistency" => consistency_cmd(&opts),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!(
        "usage : lab14 arena --hero A --field B [--players 2,3,4,5] [--deals N] [--threads N] [--seed S] [--out F] \
         [--stop D] [--device cpu|vulkan|cuda]"
    );
    eprintln!("        lab14 arena --table 'A|B|B' [--deals N] [--threads N] [--seed S] [--out F]");
    eprintln!("        lab14 speed --agent az:… [--players 2,3,4,5] [--positions 12] [--seed S]");
    eprintln!(
        "        lab14 selfplay --agent az:… --games N --out DIR [--players 2,3,4,5] [--seed S] [--device cpu|vulkan|cuda] [--pos true]"
    );
    eprintln!("        lab14 check-net --net poids.bin --data parties.pos [--count N]");
    eprintln!("        lab14 bench-net --net poids.bin --data parties.pos [--count N]");
    eprintln!("        lab14 bench-vulkan --net poids.bin [--regions R] [--count N] [--rounds R]");
    eprintln!("        lab14 check-gpu --net poids.bin --device vulkan|cuda");
    eprintln!("        lab14 bench-gpu --net poids.bin --device vulkan|cuda [--seconds 3]");
    eprintln!("        lab14 analyze --game partie.smk --agent az:net=poids.bin,n=20000");
    eprintln!(
        "        lab14 consistency --agent az:… --games DIR [--positions N] [--every K] [--repeat R] [--explore 0|1] [--seed S] [--threads N] [--out F]"
    );
    std::process::exit(2)
}

/// `--device cpu|vulkan|gpu|cuda` (le processeur par défaut).
fn device(opts: &HashMap<String, String>) -> Option<Device> {
    Device::parse(opts.get("device").map_or("cpu", String::as_str)).unwrap_or_else(|e| {
        eprintln!("{e}");
        usage()
    })
}

/// Le réseau sur GPU (`--device vulkan|cuda`) comparé au processeur, sur des positions de parties
/// de l'IA Normale à 2, 3, 4 et 5 joueurs : écarts les plus grands (victoire, pièces, logits). Le
/// moteur porte deux fois le réseau, et les positions vont à l'un ou l'autre, mêlées dans les
/// calculs (l'arène : un réseau par joueur).
fn check_gpu_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab14::encode::encode;
    use smallmobile_lab14::know::Knowledge;
    use smallmobile_lab14::rand::{Rng, mix};
    let net_path = std::path::Path::new(opts.get("net").unwrap_or_else(|| usage()));
    let device = device(opts).unwrap_or_else(|| usage());
    let cpu = smallmobile_lab14::net::Net::load(net_path).unwrap_or_else(|e| {
        eprintln!("réseau illisible : {e}");
        std::process::exit(2)
    });
    let (gpu, name) = Gpu::start_on(device, &[net_path, net_path]).unwrap_or_else(|e| {
        eprintln!("GPU : {e}");
        std::process::exit(2)
    });
    println!("{name}");
    let normal = agent_named("normal");
    let tables: Vec<Table> = (2..=5).map(|n| Table::new(n, mix(7, n as u64))).collect();
    let mut positions = Vec::new();
    for (t, table) in tables.iter().enumerate() {
        let (mut g, mut k) = (table.game(t as u32 + 1), Knowledge::default());
        let mut rng = Rng::new(mix(8, t as u64));
        let mut step = 0;
        while g.state.phase != smallmobile_kernel::Phase::GameOver && step < 2000 {
            if step % 5 == 0 {
                positions.push((encode(&g, &k), t));
            }
            for a in normal.act(&g, &k, &mut rng.fork()) {
                k.apply(&mut g, &a).expect("action de l'IA Normale");
            }
            step += 1;
        }
    }
    let items: Vec<_> = positions.iter().map(|(e, t)| (e, &tables[*t].map)).collect();
    let mut lane = gpu.lanes(1).pop().expect("une file");
    let mut far = Vec::new();
    for chunk in items.chunks(lane.free().min(512) / 8) {
        // huit envois en vol, réseaux alternés : un même calcul en reçoit des deux
        let sent: Vec<_> =
            chunk.chunks(chunk.len().div_ceil(8)).enumerate().map(|(i, c)| lane.send(&gpu, c, (i % 2) as u8)).collect();
        for s in sent {
            far.extend(lane.take(&gpu, s));
        }
    }
    let (mut win, mut gain, mut logit) = (0.0f32, 0.0f32, 0.0f32);
    for ((e, t), g) in positions.iter().zip(&far) {
        let c = cpu.eval(e, &tables[*t].map);
        for q in 0..e.players {
            win = win.max((c.win[q] - g.win[q]).abs());
            gain = gain.max((c.gain[q] - g.gain[q]).abs());
        }
        for (a, b) in c.logits.iter().zip(&g.logits) {
            if a.is_finite() || b.is_finite() {
                logit = logit.max((a - b).abs());
            }
        }
    }
    println!("{} positions : écart maximal victoire {win:.4}, pièces {gain:.3}, logits {logit:.3}", positions.len());
}

/// Le débit du moteur GPU seul (`--device`), en positions par seconde, selon la taille des lots : des
/// positions de parties de l'IA Normale (comme `check-gpu`), envoyées par lots, deux en vol, pendant
/// `--seconds` par taille.
fn bench_gpu_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab14::encode::encode;
    use smallmobile_lab14::know::Knowledge;
    use smallmobile_lab14::rand::{Rng, mix};
    let net_path = std::path::Path::new(opts.get("net").unwrap_or_else(|| usage()));
    let device = device(opts).unwrap_or_else(|| usage());
    let seconds: f64 = get(opts, "seconds", 3.0);
    let (gpu, name) = Gpu::start_on(device, &[net_path]).unwrap_or_else(|e| {
        eprintln!("GPU : {e}");
        std::process::exit(2)
    });
    println!("{name}");
    let normal = agent_named("normal");
    let tables: Vec<Table> = (2..=5).map(|n| Table::new(n, mix(7, n as u64))).collect();
    let mut positions = Vec::new();
    for (t, table) in tables.iter().enumerate() {
        let (mut g, mut k) = (table.game(t as u32 + 1), Knowledge::default());
        let mut rng = Rng::new(mix(8, t as u64));
        while g.state.phase != smallmobile_kernel::Phase::GameOver {
            positions.push((encode(&g, &k), t));
            for a in normal.act(&g, &k, &mut rng.fork()) {
                k.apply(&mut g, &a).expect("action de l'IA Normale");
            }
        }
    }
    let items: Vec<_> = positions.iter().map(|(e, t)| (e, &tables[*t].map)).collect();
    let regions = positions.iter().map(|(e, _)| e.regions).sum::<usize>() as f64 / positions.len() as f64;
    println!("{} positions de l'IA Normale, {regions:.1} régions en moyenne", positions.len());
    let mut lane = gpu.lanes(1).pop().expect("une file");
    for count in [64usize, 256, 512, 1024, 2048] {
        let count = count.min(lane.free() / 2);
        let lot = |i: usize| -> Vec<_> { (0..count).map(|j| items[(i * count + j) % items.len()]).collect() };
        let start = Instant::now();
        let (mut done, mut i) = (0usize, 0usize);
        let mut sent = std::collections::VecDeque::new();
        while start.elapsed().as_secs_f64() < seconds || !sent.is_empty() {
            if sent.len() < 2 && start.elapsed().as_secs_f64() < seconds {
                sent.push_back(lane.send(&gpu, &lot(i), 0));
                i += 1;
                continue;
            }
            let s = sent.pop_front().expect("un lot en vol");
            done += lane.take(&gpu, s).len();
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "lots de {count:5} : {:9.0} positions/s ({:.2} ms par lot)",
            done as f64 / secs,
            1e3 * secs * count as f64 / done as f64
        );
    }
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
/// ensemble). `--out` : le résultat de chaque place de chaque partie, pour `labtools/score.py`.
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
    // le nom de chaque joueur distinct, dans l'ordre de `agents`
    let named: Vec<String>;
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
            named = distinct.iter().map(|d| d.to_string()).collect();
            (distinct.into_iter().map(agent_named).collect(), Some(roles))
        }
        None => {
            named = ["hero", "field"].iter().map(|k| opts.get(*k).cloned().unwrap_or_default()).collect();
            (vec![agent(opts, "hero"), agent(opts, "field")], None)
        }
    };
    // --device vulkan|cuda : chaque réseau des joueurs `az` sur un GPU, le sien
    let device = device(opts);
    let on_gpu = device.is_some();
    // les réseaux des joueurs `az`, sur un seul GPU (un moteur pour tous, `Gpu::start_nets`)
    let (mut paths, mut nets) = (Vec::new(), Vec::new());
    for (a, name) in agents.iter().zip(&named) {
        if let Some(az) = a.az().filter(|_| on_gpu) {
            let net = name.split(',').find_map(|p| p.strip_prefix("az:").unwrap_or(p).strip_prefix("net="));
            paths.push(std::path::PathBuf::from(net.unwrap_or_else(|| usage())));
            nets.push(&*az.net);
        }
    }
    let gpu = (!paths.is_empty()).then(|| {
        let paths: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();
        let (gpu, name) = Gpu::start_on(device.expect("un GPU"), &paths).unwrap_or_else(|e| {
            eprintln!("GPU : {e}");
            std::process::exit(2)
        });
        eprintln!("réseaux évalués par {name}");
        gpu
    });
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
    // le fichier de sortie avant l'arène : un chemin faux échoue tout de suite, pas après le calcul
    // (et `exit` après le calcul laisserait le pilote attendre les commandes Vulkan soumises d'avance)
    let out = opts.get("out").map(|path| {
        std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("échec de la création de {path} : {e}");
            std::process::exit(1)
        })
    });
    let start = Instant::now();
    let run = arena::run(&cfg, &lineups, gpu.as_ref().map(|g| (g, &nets[..])));
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
    if let Some(f) = out {
        let written = arena::write(&mut std::io::BufWriter::new(f), &cfg, &names, &run);
        if let Err(e) = written {
            eprintln!("échec de l'écriture de {} : {e}", opts["out"]);
            drop(gpu);
            std::process::exit(1)
        }
    }
    rollouts();
    println!("durée : {:.1} s", start.elapsed().as_secs_f64());
}

/// Le coût des fins simulées jouées depuis le départ (`roll`), s'il y en a eu.
fn rollouts() {
    use smallmobile_lab14::az::{ROLLOUT_COUNT, ROLLOUT_EVALS, ROLLOUTS};
    let count = ROLLOUTS[ROLLOUT_COUNT].load(std::sync::atomic::Ordering::Relaxed);
    let evals = ROLLOUTS[ROLLOUT_EVALS].load(std::sync::atomic::Ordering::Relaxed);
    if count > 0 {
        println!("fins simulées : {count}, {:.1} évaluations chacune en moyenne", evals as f64 / count as f64);
    }
}

/// Des recherches sur des positions fixes (parties de l'IA Normale, `--positions` par nombre de
/// joueurs), chacune avec sa graine : leur temps sur un cœur, et deux empreintes de leurs résultats,
/// les décisions (coup et visites de la racine) et les valeurs (au bit près). Une optimisation qui
/// ne change pas la recherche garde les deux.
fn speed_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab14::know::Knowledge;
    use smallmobile_lab14::moves;
    use smallmobile_lab14::rand::{Rng, mix};

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
    // l'ancrage : écart (KL, en nats) de la politique de la racine à l'a priori du réseau, et part
    // des décisions où le coup joué n'est pas le coup préféré du réseau
    let (mut kl_sum, mut disagreed) = (0.0f64, 0usize);
    for n in players(opts, "2,3,4,5") {
        let table = Table::new(n, mix(seed, n as u64));
        let mut g = table.game(mix(seed, 10 + n as u64) as u32);
        let mut k = Knowledge::default();
        let mut rng = Rng::new(mix(seed, 20 + n as u64));
        // une position sur 7 parmi celles où le joueur a le choix ; pour une fin de tour, celle de
        // son premier choix (pouvoir ou jeton)
        let mut positions = Vec::new();
        let mut choices = 0;
        while g.state.phase != smallmobile_kernel::Phase::GameOver && positions.len() < per {
            let (mut p, mut kp) = (g, k);
            if moves::legal(&p, &kp).iter().any(|c| c.mv == moves::Move::Lift) {
                kp.play(&mut p, &moves::Move::Lift).expect("jetons relevés");
            }
            if moves::legal(&p, &kp).len() > 1 {
                if choices % 7 == 3 {
                    positions.push((p, kp));
                }
                choices += 1;
            }
            for a in normal.act(&g, &k, &mut rng.fork()) {
                k.apply(&mut g, &a).expect("action de l'IA Normale");
            }
        }
        let start = Instant::now();
        for (i, (g, k)) in positions.iter().enumerate() {
            let s = az.search(g, k, &mut Rng::new(mix(seed, (100 * n + i) as u64)), false, &mut Default::default());
            decisions = fnv(decisions, format!("{:?}{:?}", s.mv, s.visits).as_bytes());
            kl_sum += s
                .policy
                .iter()
                .zip(&s.details)
                .filter(|((_, p), _)| *p > 0.0)
                .map(|((_, p), (_, prior, _))| (*p as f64) * (*p as f64 / (*prior as f64).max(1e-12)).ln())
                .sum::<f64>();
            let preferred = s.details.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|d| d.0);
            let chosen = moves::legal(g, k).iter().find(|c| c.mv == s.mv).map(|c| c.key);
            disagreed += (preferred != chosen) as usize;
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
    println!(
        "ancrage : KL(politique de la racine ‖ a priori) {:.3} nat par recherche ; coup joué ≠ coup préféré du réseau : {disagreed} sur {count} ({:.0} %)",
        kl_sum / count as f64,
        100.0 * disagreed as f64 / count as f64
    );
    rollouts();
    println!("empreintes : décisions {decisions:016x}, valeurs {values:016x}");
}

/// Le plafond des cibles d'une recherche (lab14/lab.md, D5). Des positions de parties enregistrées par
/// le selfplay (`--games DIR` : ses `.moves`), où le joueur a le choix : dans la partie i, celles dont le
/// rang parmi ces positions vaut i modulo `--every`, jusqu'à `--positions`. Sur chacune, `--repeat`
/// recherches de `--agent`, chacune sa graine, avec le bruit de Gumbel à la racine comme les
/// recherches enregistrées du selfplay (`--explore 0` : sans, comme l'arène et le jeu). Pour chaque
/// recherche r, sa cible t (la politique améliorée) comparée :
/// - aux autres recherches de la même position : même coup préféré ;
/// - à l'a priori p du réseau : même coup préféré, et entropie croisée −Σ t log p (la perte de
///   politique de `train.py`) ;
/// - à la moyenne m des autres cibles, mêlée à 5 % de l'a priori (jamais nulle) : −Σ t log m, ce
///   qu'atteindrait un réseau qui saurait la cible moyenne (un peu pessimiste : m est bruitée) ;
/// - à elle-même : son entropie, le plancher de toute prédiction.
///
/// `--out` : une ligne par recherche (position, joueurs, recherche, coup préféré, coup préféré de
/// l'a priori, les trois entropies).
fn consistency_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab14::az::{Evals, Searched};
    use smallmobile_lab14::know::Knowledge;
    use smallmobile_lab14::moves;
    use smallmobile_lab14::rand::{Rng, mix};
    use smallmobile_lab14::selfplay::GameLog;

    let name = opts.get("agent").unwrap_or_else(|| usage());
    let az = agent::az_named(name).unwrap_or_else(|| {
        eprintln!("recherche guidée par le réseau attendue (az:…) : {name}");
        std::process::exit(2)
    });
    let dir = opts.get("games").unwrap_or_else(|| usage());
    let wanted: usize = get(opts, "positions", 1500);
    let every: usize = get(opts, "every", 61);
    let repeat: usize = get(opts, "repeat", 4);
    let explore = get(opts, "explore", 1u8) == 1;
    let seed: u64 = get(opts, "seed", 1);
    let threads: usize = get(opts, "threads", threads());
    if repeat < 2 {
        eprintln!("--repeat : au moins 2");
        std::process::exit(2)
    }
    // les parties, dans l'ordre des fichiers
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| {
            eprintln!("{dir} : {e}");
            std::process::exit(1)
        })
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "moves"))
        .collect();
    files.sort();
    let logs: Vec<GameLog> = files
        .iter()
        .flat_map(|f| std::fs::read_to_string(f).unwrap_or_default().lines().map(String::from).collect::<Vec<_>>())
        .map(|line| GameLog::parse(&line).unwrap_or_else(|| panic!("ligne illisible dans {dir}")))
        .collect();
    let tables: Vec<Table> = logs.iter().map(GameLog::table).collect();
    let mut positions = Vec::new();
    for (i, (log, table)) in logs.iter().zip(&tables).enumerate() {
        if positions.len() >= wanted {
            break;
        }
        let mut choices = 0;
        log.replay(table, |g, k| {
            if positions.len() < wanted && moves::legal(g, k).len() > 1 {
                if choices % every == i % every {
                    positions.push((*g, *k));
                }
                choices += 1;
            }
        })
        .unwrap_or_else(|e| panic!("partie {} illisible : {e}", log.deal));
    }
    eprintln!("{} positions de {} parties ; {repeat} recherches chacune, {threads} fils", positions.len(), logs.len());

    // une recherche : sa cible et l'a priori, par clé de coup
    type Run = (Vec<(moves::Key, f32)>, Vec<(moves::Key, f32)>);
    let search = |g: &smallmobile_kernel::Game, k: &Knowledge, s: u64| -> Run {
        let r: Searched = az.search(g, k, &mut Rng::new(s), explore, &mut Evals::default());
        let prior = r.details.iter().map(|d| (d.0, d.1)).collect();
        (r.policy, prior)
    };
    let start = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut runs: Vec<Option<Vec<Run>>> = vec![None; positions.len()];
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((g, k)) = positions.get(i) else { break };
                        mine.push((i, (0..repeat).map(|r| search(g, k, mix(seed, (i * repeat + r) as u64))).collect()));
                    }
                    mine
                })
            })
            .collect();
        for w in workers {
            for (i, r) in w.join().expect("fil") {
                runs[i] = Some(r);
            }
        }
    });
    let secs = start.elapsed().as_secs_f64();

    let best = |xs: &[(moves::Key, f32)]| xs.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|x| x.0);
    let p_of = |xs: &[(moves::Key, f32)], key| xs.iter().find(|x| x.0 == key).map_or(0.0, |x| x.1);
    let mut out = opts.get("out").map(|path| {
        std::io::BufWriter::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("échec de la création de {path} : {e}");
            std::process::exit(1)
        }))
    });
    // par nombre de joueurs (index 0 : tous) : recherches, paires d'accord, accords avec l'a priori,
    // entropies croisées à l'a priori et à la moyenne des autres, entropie
    let mut sums = [[0.0f64; 7]; 6];
    for (i, ((g, _), r)) in positions.iter().zip(&runs).enumerate() {
        let r = r.as_ref().expect("recherches");
        let n = g.state.player_count();
        let keys: Vec<_> = r.iter().map(|(t, _)| best(t)).collect();
        let pairs = (0..repeat).flat_map(|a| (a + 1..repeat).map(move |b| (a, b)));
        let agree = pairs.clone().filter(|&(a, b)| keys[a] == keys[b]).count() as f64 / pairs.count() as f64;
        for (run, (t, prior)) in r.iter().enumerate() {
            let ce = |q: &dyn Fn(moves::Key) -> f32| -> f64 {
                t.iter().filter(|x| x.1 > 0.0).map(|&(key, p)| -(p as f64) * (q(key).max(1e-12) as f64).ln()).sum()
            };
            let others = |key| {
                let m = r.iter().enumerate().filter(|&(o, _)| o != run).map(|(_, (u, _))| p_of(u, key)).sum::<f32>()
                    / (repeat - 1) as f32;
                0.95 * m + 0.05 * p_of(prior, key)
            };
            let (ce_prior, ce_others, entropy) = (ce(&|key| p_of(prior, key)), ce(&others), ce(&|key| p_of(t, key)));
            let same_prior = (best(prior) == keys[run]) as u8 as f64;
            for at in [0, n] {
                let s = &mut sums[at];
                s[0] += 1.0;
                s[1] += agree;
                s[2] += same_prior;
                s[3] += ce_prior;
                s[4] += ce_others;
                s[5] += entropy;
            }
            if let Some(f) = out.as_mut() {
                use std::io::Write;
                writeln!(
                    f,
                    "{i}\t{n}\t{run}\t{:?}\t{:?}\t{ce_prior:.4}\t{ce_others:.4}\t{entropy:.4}",
                    keys[run],
                    best(prior)
                )
                .expect("écriture");
            }
        }
    }
    println!("{} ({}) : {:.0} s", az.describe(), if explore { "bruit de Gumbel" } else { "sans bruit" }, secs);
    println!(
        "joueurs  recherches  même coup entre recherches  coup de l'a priori  CE a priori  CE moyenne des autres  entropie"
    );
    for (at, s) in sums.iter().enumerate().filter(|(_, s)| s[0] > 0.0) {
        let c = s[0];
        println!(
            "{:>7}  {:>10}  {:>26.1} %  {:>16.1} %  {:>11.3}  {:>21.3}  {:>8.3}",
            if at == 0 { "tous".to_string() } else { at.to_string() },
            c,
            100.0 * s[1] / c,
            100.0 * s[2] / c,
            s[3] / c,
            s[4] / c,
            s[5] / c
        );
    }
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
            noisy_rounds: get(opts, "noisy-rounds", selfplay::RECIPE.noisy_rounds),
            fast_roll: get(opts, "fast-roll", 0u8) == 1,
            luck: get(opts, "luck", 0u8) == 1,
            ..selfplay::RECIPE
        },
        pos: get(opts, "pos", false),
    };
    let out = opts.get("out").unwrap_or_else(|| usage());
    // Le réseau sur GPU (Vulkan ou CUDA) : les mêmes poids que la recherche.
    let gpu = device(opts).map(|device| {
        let net = name
            .split(',')
            .find_map(|p| p.strip_prefix("az:").unwrap_or(p).strip_prefix("net="))
            .unwrap_or_else(|| usage());
        let (gpu, name) = Gpu::start_on(device, &[std::path::Path::new(net)]).unwrap_or_else(|e| {
            eprintln!("GPU : {e}");
            std::process::exit(2)
        });
        eprintln!("réseau évalué par {name}");
        gpu
    });
    let start = Instant::now();
    if let Err(e) = selfplay::run(&cfg, &az, std::path::Path::new(out), gpu.as_ref()) {
        eprintln!("échec : {e}");
        std::process::exit(1)
    }
    println!("{} parties en {:.0} s", cfg.games, start.elapsed().as_secs_f64());
}

/// La durée d'un lot sur le GPU par Vulkan, selon le nombre de positions.
#[cfg(feature = "vulkan")]
fn bench_vulkan_cmd(opts: &HashMap<String, String>) {
    // SAFETY : avant tout autre fil
    unsafe { std::env::set_var("LAB14_VULKAN_PROFILE", "1") };
    let net = opts.get("net").unwrap_or_else(|| usage());
    let regions = get(opts, "regions", 40);
    let rounds = get(opts, "rounds", 20);
    let counts: Vec<usize> = match opts.get("count") {
        Some(n) => vec![n.parse().unwrap_or_else(|_| usage())],
        None => vec![1, 16, 64, 128, 256, 512, 1024, 2048],
    };
    let engine =
        smallmobile_lab14::vulkan::Engine::new(std::path::Path::new(net), smallmobile_lab14::vulkan::MAX_BATCH)
            .unwrap_or_else(|e| {
                eprintln!("GPU : {e}");
                std::process::exit(2)
            });
    println!("{}, {regions} régions par position", engine.describe());
    if let Some(lots) = opts.get("stream") {
        // en flux (`--stream LOTS`) : lots de `--count` positions, à la suite
        let lots: usize = lots.parse().unwrap_or_else(|_| usage());
        let count = get(opts, "count", 256);
        let (wall, gpu) = engine.bench_stream(count, regions, lots).unwrap_or_else(|e| panic!("{e}"));
        println!(
            "{lots} lots de {count} en flux : {wall:.2} ms, GPU {gpu:.2} ms ({:.0} %), {:.0} positions/s",
            100.0 * gpu / wall,
            (lots * count) as f64 / (wall / 1e3)
        );
        return;
    }
    for count in counts {
        if let Some(phases) = engine.bench_phases(count, regions, rounds).unwrap_or_else(|e| panic!("{e}")) {
            // shader instrumenté (`LAB14_VULKAN_PHASES`) : la durée de chaque étape d'une position
            let all: f64 = phases.iter().sum();
            println!("{count:5} positions, une position : {all:.1} µs");
            for (name, us) in smallmobile_lab14::vulkan::PHASES.iter().zip(&phases) {
                println!("    {name:<20} {us:7.1} µs  {:4.1} %", 100.0 * us / all);
            }
            continue;
        }
        let ms = engine.bench(count, regions, rounds).unwrap_or_else(|e| panic!("{e}"));
        println!("{count:5} positions : {ms:7.3} ms, {:8.0} positions/s", count as f64 / (ms / 1e3));
    }
}

#[cfg(not(feature = "vulkan"))]
fn bench_vulkan_cmd(_: &HashMap<String, String>) {
    eprintln!("programme lab14 compilé sans Vulkan (cargo build --features vulkan)");
    std::process::exit(2)
}

/// Évalue les premières positions d'un fichier de parties en `f32` : mêmes nombres attendus qu'en
/// Python.
fn check_net_cmd(opts: &HashMap<String, String>) {
    let net = smallmobile_lab14::net::Net::load(std::path::Path::new(opts.get("net").unwrap_or_else(|| usage())))
        .unwrap_or_else(|e| panic!("réseau illisible : {e}"));
    let data = std::fs::read(opts.get("data").unwrap_or_else(|| usage())).expect("fichier de parties");
    let count: usize = get(opts, "count", 5);
    let start = Instant::now();
    let mut evals = 0;
    for rec in data.as_chunks::<{ selfplay::RECORD }>().0.iter().take(count) {
        let (e, adj) = selfplay::read_record(rec);
        let ev = net.eval_f32(&e, |i| &adj[i]);
        evals += 1;
        let legal: Vec<String> = (0..smallmobile_lab14::moves::KEYS)
            .filter(|&k| ev.logits[k].is_finite())
            .take(8)
            .map(|k| format!("{k}:{:.4}", ev.logits[k]))
            .collect();
        println!("win {:?} gain {:?} logits {}", ev.win, ev.gain, legal.join(" "));
    }
    eprintln!("{:.1} µs par évaluation", 1e6 * start.elapsed().as_secs_f64() / evals.max(1) as f64);
}

/// Le calcul en bf16 face au calcul `f32` sur les premières positions d'un fichier : temps par
/// évaluation (un cœur) et plus grands écarts des sorties.
fn bench_net_cmd(opts: &HashMap<String, String>) {
    use std::hint::black_box;
    let net = smallmobile_lab14::net::Net::load(std::path::Path::new(opts.get("net").unwrap_or_else(|| usage())))
        .unwrap_or_else(|e| panic!("réseau illisible : {e}"));
    let data = std::fs::read(opts.get("data").unwrap_or_else(|| usage())).expect("fichier de parties");
    let count: usize = get(opts, "count", 5000);
    let positions: Vec<_> =
        data.as_chunks::<{ selfplay::RECORD }>().0.iter().take(count).map(|r| selfplay::read_record(r)).collect();
    let time = |eval: &dyn Fn(&smallmobile_lab14::encode::Encoded, &[Vec<u8>])| {
        let start = Instant::now();
        for (e, adj) in &positions {
            eval(e, adj);
        }
        1e6 * start.elapsed().as_secs_f64() / positions.len().max(1) as f64
    };
    let f32_us = time(&|e, adj| {
        black_box(net.eval_f32(e, |i| &adj[i]));
    });
    println!("{} positions ; f32 : {f32_us:.1} µs par évaluation", positions.len());
    if net.eval_bf16(&positions[0].0, |i| &positions[0].1[i]).is_none() {
        println!("bf16 : indisponible sur ce processeur (AVX-512 BF16)");
        return;
    }
    let bf16_us = time(&|e, adj| {
        black_box(net.eval_bf16(e, |i| &adj[i]));
    });
    let mut worst = [0.0f32; 3];
    let gap = |a: &[f32], b: &[f32]| {
        a.iter().zip(b).filter(|(a, _)| a.is_finite()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
    };
    for (e, adj) in &positions {
        let exact = net.eval_f32(e, |i| &adj[i]);
        let low = net.eval_bf16(e, |i| &adj[i]).expect("bf16");
        worst[0] = worst[0].max(gap(&low.win, &exact.win));
        worst[1] = worst[1].max(gap(&low.gain, &exact.gain));
        worst[2] = worst[2].max(gap(&low.logits, &exact.logits));
    }
    println!(
        "bf16 : {bf16_us:.1} µs par évaluation ({:.2} fois plus rapide) ; plus grands écarts : victoire {:.4}, \
         pièces {:.3}, logits {:.4}",
        f32_us / bf16_us,
        worst[0],
        worst[1],
        worst[2]
    );
}

/// Le meilleur coup d'une vraie partie, avec ce que la recherche pense de chaque coup.
fn analyze_cmd(opts: &HashMap<String, String>) {
    use smallmobile_lab14::moves;
    let words: Vec<u32> = std::fs::read(opts.get("game").unwrap_or_else(|| usage()))
        .expect("fichier de partie")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
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
    let mut k = smallmobile_lab14::know::Knowledge::default();
    let mut g = g;
    // une fin de tour : son premier choix, jetons relevés
    if moves::legal(&g, &k).iter().any(|c| c.mv == moves::Move::Lift) {
        k.play(&mut g, &moves::Move::Lift).expect("jetons relevés");
    }
    let mut rng = smallmobile_lab14::rand::Rng::new(get(opts, "seed", 1));
    let start = Instant::now();
    let s = az.search(&g, &k, &mut rng, false, &mut Default::default());
    let legal = moves::legal(&g, &k);
    println!(
        "Joueur {} à jouer, manche {}/{} ({:.0} s de recherche)",
        me + 1,
        g.state.round,
        g.state.max_rounds,
        start.elapsed().as_secs_f64()
    );
    println!("Chance de victoire estimée : {:.1} % ; coup choisi : {}", 100.0 * s.value, moves::describe(&g, &s.mv));
    let mut rows: Vec<_> = s
        .details
        .iter()
        .zip(&s.visits)
        .zip(&s.policy)
        .map(|((&(key, prior, q), &(_, n)), &(_, p))| (n, key, prior, q, p))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    println!("{:>8}  {:>7}  {:>8}  {:>8}  coup", "visites", "valeur", "a priori", "amélioré");
    for (n, key, prior, q, p) in rows.iter().take(get(opts, "top", 12)) {
        let mv = legal.iter().find(|c| c.key == *key).map(|c| moves::describe(&g, &c.mv)).unwrap_or_default();
        println!("{n:>8}  {:>7.1} %  {:>7.1} %  {:>7.1} %  {mv}", 100.0 * q, 100.0 * prior, 100.0 * p);
    }
}
